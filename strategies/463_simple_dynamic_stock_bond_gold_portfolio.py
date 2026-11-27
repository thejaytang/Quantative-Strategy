# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 463_simple_dynamic_stock_bond_gold_portfolio\n\nuniverse:\n  risky_weights:\n    SPY: 0.50\n    AGG: 0.30\n    GLD: 0.20\n  cash_proxy: BIL\n\ndates:\n  data_start: 2007-06-01\n  start: 2007-07-01\n  oos_start: 2018-01-01\n  end_exclusive: 2026-01-01\n\nsignal:\n  lookback_sessions: 11\n  target_annual_vol: 0.07\n  execution_lag_sessions: 1\n\nexecution:\n  cost_bps: 2.5\n  slippage_bps: 1.0\n  cash_buffer: 0.05\n'
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
"""Research backtest for the paper's 50/30/20 volatility-control portfolio."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
TRADE_TOLERANCE = 1e-10


def validate_prices(prices, symbols, start, end):
    prices = prices.reindex(columns=symbols).sort_index()
    prices.index = prices.index.as_unit("s")
    prices.columns.name = None
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("Missing or duplicate quote dates.")
    starts_in_range = start <= prices.index[0] < start + pd.Timedelta(days=7)
    ends_in_range = end - pd.Timedelta(days=7) <= prices.index[-1] < end
    if not (starts_in_range and ends_in_range):
        raise ValueError("Yahoo Finance quotes do not cover the requested date range.")
    invalid = ~np.isfinite(prices) | (prices <= 0)
    if invalid.any().any():
        bad_dates = prices.index[invalid.any(axis=1)].strftime("%Y-%m-%d")
        raise ValueError(f"Missing or invalid closes on {', '.join(bad_dates[:5])}.")
    return prices


def load_prices(symbols, start, end):
    CACHE.mkdir(exist_ok=True)
    request = json.dumps(["yfinance_auto_adjust_close", symbols, str(start), str(end)])
    key = hashlib.sha256(request.encode()).hexdigest()[:16]
    path = CACHE / f"adjusted_close_{key}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as cached:
            fetched_at = cached.readline().strip().removeprefix("# fetched_at=")
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(cached, index_col="Date", parse_dates=["Date"])
        return validate_prices(prices, symbols, start, end), fetched_at

    with TemporaryDirectory(dir=CACHE, prefix="yfinance_") as quote_cache:
        yf.set_tz_cache_location(quote_cache)
        quotes = yf.download(
            symbols, start=start.date().isoformat(), end=end.date().isoformat(),
            interval="1d", auto_adjust=True, progress=False, timeout=15,
        )
    if quotes.empty or "Close" not in quotes.columns.get_level_values(0):
        raise ValueError("Yahoo Finance returned no adjusted close data.")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prices = quotes["Close"]
    prices.index = prices.index.tz_localize(None).normalize().rename("Date")
    prices = validate_prices(prices, symbols, start, end)

    # Keep prices and their fetch timestamp together, including on concurrent runs.
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE,
                            delete_on_close=False) as pending:
        pending.write(f"# fetched_at={fetched_at}\n")
        prices.to_csv(pending)
        pending.close()
        Path(pending.name).replace(path)
    return prices, fetched_at


def fill_sessions(dates, frequency, lag, first):
    # At the first session of a new period, the previous close is the period end.
    periods = dates.to_period(frequency)
    fills = np.flatnonzero(periods[1:] != periods[:-1]) + lag
    return np.r_[first, fills[(fills > first) & (fills < len(dates))]]


def execution(drifted, target, fee):
    # Post-trade NAV = pre-trade NAV minus fees on all ETF buy and sell notionals.
    # Solve in pre-trade NAV units so that the final weights sum to one after fees.
    if fee == 0:
        return 1.0, np.abs(target - drifted)
    lower, upper = 0.0, 1.0
    for _ in range(48):
        nav = (lower + upper) / 2
        traded = np.abs(nav * target - drifted).sum()
        if nav + fee * traded > 1:
            upper = nav
        else:
            lower = nav
    nav = (lower + upper) / 2
    return nav, np.abs(nav * target - drifted)


def simulate(returns, targets, fee, cash_buffer):
    # Start in uninvested USD. Subsequent closes mark existing positions first.
    # Size at the close when the lagged target is known (previous close relative
    # to the fill), after reserving cash_buffer of that day's wealth. Fill at the
    # *next* close with predetermined shares so trade size does not use the
    # fill-day close. The cash buffer is chosen so fills do not go to negative
    # cash; a shortfall raises rather than resizing shares. Leftover cash earns zero.
    if not 0 <= cash_buffer < 1:
        raise ValueError("cash_buffer must satisfy 0 <= cash_buffer < 1.")
    weights = np.zeros(returns.shape[1])
    pending = None
    rows = []
    for day, asset_returns in zip(returns.index, returns.to_numpy(), strict=True):
        marked = weights * (1 + asset_returns)
        gross = 1 - weights.sum() + marked.sum()
        turnover = 0.0
        trades = 0
        if pending is not None:
            sized_ratio, sized_target, traded = pending
            # Exact shares fixed at the previous close: value today is k * a_i * (1+r_i).
            # Do not renormalize to spend all wealth; leftover is cash.
            asset_value = sized_ratio * sized_target * (1 + asset_returns)
            # Predetermined shares only. Fees and turnover use those shares at this close;
            # a shortfall does not resize the fill.
            fill_traded = np.abs(asset_value - marked)
            post_wealth = gross - fee * float(fill_traded.sum())
            if post_wealth <= 0:
                raise ValueError("Non-positive wealth on a fill close.")
            leftover = post_wealth - float(asset_value.sum())
            if leftover < -1e-12:
                raise ValueError(f"Negative cash on fill {day}: {leftover}")
            weights = asset_value / post_wealth
            turnover = float(fill_traded.sum()) / 2
            trades = int(np.count_nonzero(fill_traded > TRADE_TOLERANCE))
            gross = post_wealth
            pending = None
        else:
            if gross <= 0:
                raise ValueError("Non-positive marked wealth.")
            weights = marked / gross
        target = targets.get(day)
        if target is not None:
            array_target = np.asarray(target, dtype=float)
            sized_ratio, traded = execution(weights, array_target, fee)
            sized_ratio = sized_ratio * (1.0 - cash_buffer)
            traded = np.abs(sized_ratio * array_target - weights)
            pending = (sized_ratio, array_target, traded)
        rows.append((gross - 1, turnover, trades))
    return pd.DataFrame(rows, index=returns.index, columns=["return", "turnover", "trades"])


def metrics(portfolio, cash_returns):
    daily = portfolio["return"]
    excess = daily - cash_returns.loc[daily.index]
    annualize = np.sqrt(SESSIONS_PER_YEAR)
    wealth = np.r_[1.0, (1 + daily).cumprod().to_numpy()]
    volatility = daily.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    return {
        "cagr": float(wealth[-1] ** (SESSIONS_PER_YEAR / len(daily)) - 1),
        "ann_vol": float(volatility * annualize),
        "sharpe": float(annualize * daily.mean() / volatility) if volatility > 0 else None,
        "sharpe_excess": (float(annualize * excess.mean() / excess_volatility)
                          if excess_volatility > 0 else None),
        "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
        "total_return": float(wealth[-1] - 1),
        "turnover": float(portfolio["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(portfolio["trades"].sum()),
    }


def summarize(portfolio, cash_returns, split):
    before_split = portfolio.index < split
    if min(before_split.sum(), (~before_split).sum()) < 2:
        raise ValueError("IS and OOS must each contain at least two portfolio dates.")
    return {
        "is": metrics(portfolio.loc[before_split], cash_returns),
        "oos": metrics(portfolio.loc[~before_split], cash_returns),
        "full": metrics(portfolio, cash_returns),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    risky = pd.Series(universe["risky_weights"], dtype=float)
    cash = universe["cash_proxy"]
    if cash in risky or not np.isfinite(risky).all() or (risky <= 0).any() or not np.isclose(
        risky.sum(), 1.0, rtol=0, atol=1e-12
    ):
        raise ValueError("Risky weights must be positive, sum to one and exclude the cash proxy.")
    risky /= risky.sum()
    symbols = [*risky.index, cash]

    dates = {key: pd.Timestamp(value) for key, value in config["dates"].items()}
    data_start, start, split, end = (dates[key] for key in (
        "data_start", "start", "oos_start", "end_exclusive",
    ))
    if not data_start < start < split < end:
        raise ValueError("Expected data_start < start < oos_start < end_exclusive.")
    signal = config["signal"]
    lag = signal["execution_lag_sessions"]
    lookback = signal["lookback_sessions"]
    target_vol = signal["target_annual_vol"]
    if type(lag) is not int or lag < 1 or type(lookback) is not int or lookback < 2:
        raise ValueError("Execution lag must be at least one session and lookback at least two.")
    if not np.isfinite(target_vol) or target_vol <= 0:
        raise ValueError("Target volatility must be finite and positive.")
    cost_bps = config["execution"]["cost_bps"]
    slippage_bps = config["execution"]["slippage_bps"]
    cash_buffer = config["execution"]["cash_buffer"]
    if not (cost_bps >= 0 and slippage_bps >= 0 and cost_bps + slippage_bps < 10_000):
        raise ValueError("Costs and slippage must be nonnegative with a total below 10,000 bps.")
    if not 0 <= cash_buffer < 1:
        raise ValueError("cash_buffer must satisfy 0 <= cash_buffer < 1.")
    fee = (cost_bps + slippage_bps) / 10_000

    prices, fetched_at = load_prices(symbols, data_start, end)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    first = returns.index.searchsorted(start)
    basket_return = returns[risky.index] @ risky
    vol = basket_return.rolling(lookback).std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)
    scale = (target_vol / vol).clip(upper=1).shift(lag)
    monthly = fill_sessions(returns.index, "M", lag, first)
    if scale.iloc[monthly].isna().any():
        raise ValueError("Insufficient price history before the first lagged rebalance.")
    strategy_targets = {
        day: np.r_[risky.to_numpy() * weight, 1 - weight]
        for day, weight in scale.iloc[monthly].items()
    }

    annual = fill_sessions(returns.index, "Y", lag, first)
    fixed = np.r_[risky.to_numpy(), 0.0]
    benchmark_targets = dict.fromkeys(returns.index[annual], fixed)
    evaluation = returns.iloc[first:]
    portfolio = simulate(evaluation, strategy_targets, fee, cash_buffer)
    benchmark = simulate(evaluation, benchmark_targets, fee, cash_buffer)
    portfolio_metrics = summarize(portfolio, evaluation[cash], split)
    benchmark_metrics = summarize(benchmark, evaluation[cash], split)

    results = {
        "strategy": config["strategy"],
        "data_range": {"start": evaluation.index[0].date().isoformat(),
                       "end": evaluation.index[-1].date().isoformat()},
        **portfolio_metrics,
        "benchmark": {"name": "Fixed-weight annual rebalance", **benchmark_metrics},
    }
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return")
    print("PARAM_COMBOS_TRIED: 1")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(portfolio_metrics[period], allow_nan=False)}")
        print(f"BENCHMARK {period.upper()}: {json.dumps(benchmark_metrics[period], allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
