# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 336_saoftrl_novel_adaptive_algorithmic\ntickers: [SPY, IWM, EFA, IEF, TLT, GLD, DBC]\nbuy_and_hold_ticker: SPY\ndata_start: 2022-01-03\nin_sample:\n  start: 2023-04-03\n  end: 2024-12-31\nout_of_sample:\n  start: 2025-01-01\n  end: 2026-10-30\n\nlookback_sessions: [63, 126, 252]\nfixed_benchmark_sessions: 126\nmax_asset_weight: 0.35\nrisk_aversion: 1.0\nscore_weeks: 12\nlearning_rate_candidates: [5.0, 10.0, 20.0, 40.0]\ntransaction_cost_bps: 5.0  # charged on each dollar bought or sold\nsessions_per_year: 252\n'
_orig_open = _b.open
def _open(f, *a, **k):
    try:
        if _os.path.basename(str(f)) == 'config.yaml' and not _os.path.exists(f):
            return _io.StringIO(_CONFIG_YAML)
    except TypeError:
        pass
    return _orig_open(f, *a, **k)
_b.open = _open
_io.open = _open
# --- end shim ---
import json
import math
import re
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq, minimize
from scipy.special import softmax
from sklearn.covariance import LedoitWolf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"


def load_prices(tickers, start, end):
    columns = pd.MultiIndex.from_product([["open", "close"], tickers])

    def validate(frame):
        frame = frame.reindex(columns=columns).sort_index().loc[start:end]
        frame.index = pd.DatetimeIndex(frame.index).as_unit("ns")
        prices = frame.to_numpy(dtype=float)
        if frame.empty or frame.index.has_duplicates or not np.all(np.isfinite(prices) & (prices > 0)):
            raise ValueError("missing, invalid, or misaligned ETF bars")
        if frame.index[0] > start + pd.Timedelta(days=7) or frame.index[-1] < end - pd.Timedelta(days=7):
            raise ValueError("history does not cover the configured dates")
        return frame

    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"yahoo_{'-'.join(tickers)}_{start.date()}_{end.date()}.csv"
    if path.is_file():
        try:
            return validate(pd.read_csv(path, index_col=0, parse_dates=True, header=[0, 1]))
        except ValueError as error:
            print(f"Refreshing unusable price cache {path.name}: {error}")

    import yfinance as yf

    with TemporaryDirectory(dir=CACHE, prefix=".yfinance_") as metadata_cache:
        yf.set_tz_cache_location(metadata_cache)
        raw = yf.download(
            tickers, start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
            auto_adjust=True, group_by="column", progress=False, timeout=10,
            ignore_tz=True, threads=4,
        )
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no ETF price history")
    frame = validate(raw.rename(columns={"Open": "open", "Close": "close"}, level=0))
    with NamedTemporaryFile(mode="w", dir=CACHE, suffix=".csv", delete=False) as temp:
        temporary_path = Path(temp.name)
        try:
            frame.to_csv(temp)
            temp.flush()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return frame


def mean_variance(sample, risk_aversion, cap):
    # Scale mean and covariance equally to keep the optimizer well conditioned.
    covariance = LedoitWolf().fit(sample).covariance_ * 252
    mean = sample.mean(axis=0) * 252
    risk = risk_aversion * covariance
    equal = np.full(len(mean), 1.0 / len(mean))
    result = minimize(
        lambda w: 0.5 * w @ risk @ w - mean @ w,
        equal, jac=lambda w: risk @ w - mean, method="SLSQP",
        bounds=[(0.0, cap)] * len(mean),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
        options={"ftol": 1e-11, "maxiter": 150},
    )
    if not result.success or abs(result.x.sum() - 1) > 1e-8:
        raise RuntimeError(f"Mean–variance solver failed: {result.message}")
    return result.x


def weekly_expert_targets(closes, config):
    windows = config["lookback_sessions"]
    returns = closes.pct_change(fill_method=None).to_numpy()
    weeks = closes.index.to_period("W-FRI")
    signals = np.flatnonzero(weeks[:-1] != weeks[1:])
    signals = signals[signals >= max(windows)]
    executions = closes.index[signals + 1]
    return {
        window: pd.DataFrame(
            [mean_variance(returns[t - window + 1:t + 1], config["risk_aversion"], config["max_asset_weight"])
             for t in signals], index=executions, columns=closes.columns,
        )
        for window in windows
    }


def simulate(prices, targets, cost_rate):
    """Start in cash, execute at the open, and mark net NAV at the close."""
    opens = prices["open"].to_numpy()
    closes = prices["close"].to_numpy()
    targets_by_day = dict(zip(prices.index.get_indexer(targets.index), targets.to_numpy(), strict=True))
    shares = np.zeros(opens.shape[1])
    cash = previous_close = 1.0
    daily = []
    for t, (opening, closing) in enumerate(zip(opens, closes)):
        holdings = shares * opening
        pretrade_nav = cash + holdings.sum()
        turnover = 0.0
        trades = 0
        if t in targets_by_day:
            target = targets_by_day[t]
            # Solve for investable NAV after fees on the actual dollars traded.
            def shares_budget(nav):
                return nav + cost_rate * np.abs(target * nav - holdings).sum() - pretrade_nav

            investable = brentq(shares_budget, 0.0, pretrade_nav, xtol=1e-14)
            target_values = target * investable
            changes = np.abs(target_values - holdings) / pretrade_nav
            turnover = changes.sum()
            trades = np.count_nonzero(changes > 1e-10)
            shares = target_values / opening
            cash = investable - target_values.sum()
        end_nav = cash + shares @ closing
        daily.append((end_nav / previous_close - 1, turnover, trades, pretrade_nav))
        previous_close = end_nav
    return pd.DataFrame(daily, index=prices.index, columns=["return", "turnover", "n_trades", "pretrade_nav"])


def performance(daily, annualization):
    returns = daily["return"]
    equity = np.r_[1.0, (1.0 + returns).cumprod().to_numpy()]
    volatility = returns.std(ddof=1)
    values = {
        "cagr": equity[-1] ** (annualization / len(returns)) - 1,
        "ann_vol": volatility * math.sqrt(annualization),
        "sharpe": returns.mean() / volatility * math.sqrt(annualization) if volatility > 0 else None,
        "max_drawdown": (equity / np.maximum.accumulate(equity) - 1).min(),
        "total_return": equity[-1] - 1,
        "turnover": daily["turnover"].sum(),
    }
    return {key: float(value) if value is not None and np.isfinite(value) else None
            for key, value in values.items()} | {"n_trades": int(daily["n_trades"].sum())}


def print_metrics(reports):
    print(f"{'Portfolio':<25}{'Period':<7}{'CAGR':>9}{'Ann. vol':>10}{'Sharpe':>9}"
          f"{'Max DD':>9}{'Return':>9}{'Turnover':>11}{'Trades':>8}")
    for name, report in reports.items():
        for period, row in report.items():
            percentages = [f"{row[key]:.2%}" if row[key] is not None else "n/a"
                           for key in ("cagr", "ann_vol", "max_drawdown", "total_return")]
            cagr, vol, drawdown, total = percentages
            sharpe = f"{row['sharpe']:.2f}" if row["sharpe"] is not None else "n/a"
            turnover = f"{row['turnover']:.2f}x"
            print(f"{name:<25}{period.upper():<7}{cagr:>9}{vol:>10}{sharpe:>9}"
                  f"{drawdown:>9}{total:>9}{turnover:>11}{row['n_trades']:>8}")


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as settings:
        config = yaml.safe_load(settings)
    start = pd.Timestamp(config["data_start"])
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    tickers = config["tickers"]
    windows = config["lookback_sessions"]
    fixed = config["fixed_benchmark_sessions"]
    cap = config["max_asset_weight"]
    cost = config["transaction_cost_bps"] / 10_000
    rates = sorted(set(config["learning_rate_candidates"]))
    if not (start < is_start <= is_end and oos_start == is_end + pd.Timedelta(days=1) <= end):
        raise ValueError("Set a warm-up start followed by contiguous IS and OOS ranges")
    if (not tickers or len(set(tickers)) != len(tickers) or config["buy_and_hold_ticker"] not in tickers
            or any(not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]*", ticker) for ticker in tickers)
            or not 1 / len(tickers) <= cap <= 1 or not 0 <= cost < 1
            or fixed not in windows or min(windows) < 2 or len(set(windows)) != len(windows)
            or not math.isfinite(config["risk_aversion"]) or config["risk_aversion"] <= 0
            or config["sessions_per_year"] <= 0 or config["score_weeks"] <= 0
            or not rates or any(not np.isfinite(rate) or rate <= 0 for rate in rates)):
        raise ValueError("Check the ETF universe, estimation parameters, costs, and annualization")

    prices = load_prices(tickers, start, end)
    live_prices = prices.loc[is_start:end]
    annualization = config["sessions_per_year"]
    experts = weekly_expert_targets(prices["close"], config)
    executions = experts[fixed].index
    if not any((executions >= is_start) & (executions <= is_end)) or live_prices.loc[oos_start:end].empty:
        raise ValueError("The configured sample ranges need IS rebalances and OOS price bars")

    open_navs = pd.DataFrame({window: simulate(prices, targets, cost)["pretrade_nav"].loc[executions]
                              for window, targets in experts.items()})
    weekly_log_returns = np.log(open_navs).diff()
    # The return ending at this execution's open is still unknown at signal time.
    scores = weekly_log_returns.rolling(config["score_weeks"], min_periods=1).sum().shift(1).fillna(0.0)
    live_scores = scores.loc[is_start:end]
    live_targets = np.stack([targets.loc[is_start:end].to_numpy() for targets in experts.values()], axis=1)

    def blend(weights, index, expert_targets):
        return pd.DataFrame(np.einsum("we,wea->wa", weights, expert_targets), index=index, columns=tickers)

    is_scores = live_scores.loc[:is_end]
    is_prices = live_prices.loc[:is_end]
    is_targets = live_targets[:len(is_scores)]

    def is_sharpe(rate):
        weights = softmax(rate * is_scores.to_numpy(), axis=1)
        daily = simulate(is_prices, blend(weights, is_scores.index, is_targets), cost)
        sharpe = performance(daily, annualization)["sharpe"]
        return sharpe if sharpe is not None else -math.inf

    rate = max(rates, key=is_sharpe)
    adaptive = blend(softmax(rate * live_scores.to_numpy(), axis=1), live_scores.index, live_targets)
    equal_mix = blend(np.full(live_scores.shape, 1 / len(experts)), live_scores.index, live_targets)
    hold_ticker = config["buy_and_hold_ticker"]
    hold = pd.DataFrame(0.0, index=live_scores.index[:1], columns=tickers)
    hold[hold_ticker] = 1.0

    def report(targets):
        daily = simulate(live_prices, targets, cost)
        return {
            "is": performance(daily.loc[:is_end], annualization),
            "oos": performance(daily.loc[oos_start:], annualization),
            "full": performance(daily, annualization),
        }

    strategy = report(adaptive)
    benchmark = report(hold)
    fixed_report = report(experts[fixed].loc[is_start:end])
    equal_report = report(equal_mix)
    data_range = f"{live_prices.index[0].date()} to {live_prices.index[-1].date()}"
    print(f"Source: Yahoo Finance | Trading dates: {data_range} | IS-selected learning rate: {rate:g}")
    print_metrics({"SAOFTRL approximation": strategy, f"{hold_ticker} buy and hold": benchmark,
                   f"Fixed {fixed}-session expert": fixed_report, "Equal-weight experts": equal_report})
    results = {
        "strategy": config["strategy"], "data_range": data_range, **strategy,
        "benchmark": {"name": f"{hold_ticker} buy and hold", **benchmark,
                      f"fixed_{fixed}_session": fixed_report, "equal_mix": equal_report},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
