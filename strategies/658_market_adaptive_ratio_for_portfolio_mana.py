# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 658_market_adaptive_ratio_for_portfolio_mana\nuniverse: [SPY, QQQ, IWM, BND, IEF, TLT]\nmarket_proxy: SPY\ncash_proxy: BIL\n\ndates:\n  download_start: '2008-06-02'\n  start: '2010-01-01'\n  is_end: '2014-12-31'\n  end: '2025-12-31'\n\nexecution:\n  lag_sessions: 1\n  commission_bps: 2.0\n  slippage_bps: 3.0\n\nmodel:\n  market_ma_sessions: 200\n  return_lookback: 120\n  rho_alpha: 5.0\n  rho_bounds: [0.5, 1.5]\n  regime_bandwidth: 0.10\n  regime_mix: 0.50\n  covariance_shrinkage: 0.20\n  weight_units: 20\n  max_asset_weight: 0.50\n\nrobustness:\n  - {name: lookback_100, return_lookback: 100}\n  - {name: lookback_140, return_lookback: 140}\n  - {name: alpha_4, rho_alpha: 4.0}\n  - {name: alpha_6, rho_alpha: 6.0}\n  - {name: fixed_rho_1, rho_alpha: 0.0}\n  - {name: double_friction, cost_multiple: 2.0}\n"
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
"""C236 research backtest for Market-Adaptive Ratio for Portfolio Management.

Paper: https://arxiv.org/abs/2312.13719
Implementation references, with independently written code:
- PeDiot RRL (no license found): https://github.com/PeDiot/rrl-trading-system
  https://raw.githubusercontent.com/PeDiot/rrl-trading-system/master/rrl_trading/model/rrl.py
- PyPortfolioOpt (MIT): https://github.com/PyPortfolio/PyPortfolioOpt
  https://raw.githubusercontent.com/PyPortfolio/PyPortfolioOpt/main/pypfopt/efficient_frontier/efficient_frontier.py
  https://raw.githubusercontent.com/PyPortfolio/PyPortfolioOpt/main/LICENSE
  https://pypi.org/project/pyportfolioopt/
- QuantConnect examples (method comparison):
  https://www.quantconnect.com/forum/discussion/992/strategy-help-combining-market-timing-and-momentum-within-a-diversified-portfolio/p1
  https://www.quantconnect.com/forum/discussion/17955/bootcamp-lesson-3-02-setting-up-tactical-asset-allocation/
"""

import gzip
import hashlib
import json
import math
import signal
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.special import expit


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
PRICE_FIELDS = ("Open", "Close")
ACTION_FIELDS = ("Dividends", "Capital Gains", "Stock Splits")


def load_quotes(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_key = hashlib.sha256(repr((tickers, str(start), str(end))).encode()).hexdigest()[:16]
    snapshot = cache_dir / f"yahoo_daily_v1_{cache_key}.csv.gz"
    price_columns = pd.MultiIndex.from_product([PRICE_FIELDS, tickers])
    action_columns = pd.MultiIndex.from_product([ACTION_FIELDS, tickers])
    if snapshot.exists():
        with gzip.open(snapshot, "rt") as source:
            fetched_at = source.readline().strip()
            datetime.fromisoformat(fetched_at)
            quotes = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True,
                                 float_precision="round_trip")
    else:
        with TemporaryDirectory(dir=cache_dir, prefix=".yahoo-") as client_cache:
            yf.set_tz_cache_location(client_cache)
            quotes = yf.download(
                tickers, start=start, end=end + pd.Timedelta(days=1), interval="1d",
                actions=True, auto_adjust=False, repair=False, ignore_tz=True,
                progress=False, timeout=12,
            )
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    prices = quotes.reindex(columns=price_columns).sort_index()
    if (prices.empty or prices.index.has_duplicates
            or prices.index[0] < start or prices.index[0] > start + pd.Timedelta(days=7)
            or prices.index[-1] > end or prices.index[-1] < end - pd.Timedelta(days=7)):
        raise RuntimeError("Yahoo daily data has incomplete or invalid date coverage")
    invalid = ~np.isfinite(prices) | (prices <= 0)
    if invalid.to_numpy().any():
        first_bad = invalid.index[invalid.any(axis=1)][0]
        raise RuntimeError(f"Missing or invalid opening/closing prices on {first_bad.date()}")
    actions = quotes.reindex(index=prices.index, columns=action_columns).fillna(0.0)
    if not np.isfinite(actions.to_numpy()).all() or (actions < 0).to_numpy().any():
        raise RuntimeError("Invalid Yahoo corporate action")
    quotes = pd.concat([prices, actions], axis=1)

    if not snapshot.exists():
        with NamedTemporaryFile(dir=cache_dir, prefix=".quotes-", delete=False) as pending:
            pending_path = Path(pending.name)
        try:
            with gzip.open(pending_path, "wt") as destination:
                destination.write(fetched_at + "\n")
                quotes.to_csv(destination)
            pending_path.replace(snapshot)
        finally:
            pending_path.unlink(missing_ok=True)
    return quotes, fetched_at


def total_return_prices(quotes):
    # Yahoo unadjusted OHLC already uses consistent split units. For dividends,
    # apply only actions on the ex-date and scale using the previous close.
    previous_close = quotes["Close"].shift(1)
    distribution = quotes["Dividends"] + quotes["Capital Gains"]
    if (distribution.iloc[1:] >= previous_close.iloc[1:]).to_numpy().any():
        raise RuntimeError("Cash distribution is at least as large as the previous close")
    multipliers = 1.0 / (1.0 - distribution.div(previous_close))
    multipliers.iloc[0] = 1.0
    scale = multipliers.cumprod()
    return quotes["Open"] * scale, quotes["Close"] * scale


def monthly_schedule(index, start, lag):
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("lag_sessions must be a positive integer")
    month = index.to_period("M")
    first_sessions = np.flatnonzero(month[1:] != month[:-1]) + 1
    fill_rows = first_sessions + lag - 1
    return [(int(fill - lag), int(fill)) for fill in fill_rows
            if fill < len(index) and index[fill] >= start]


def feasible_weights(asset_count, units, cap):
    if not isinstance(units, int) or units < 1 or not 0 < cap <= 1:
        raise ValueError("Invalid weight_units or max_asset_weight")
    if math.comb(units + asset_count - 1, asset_count - 1) > 500_000:
        raise ValueError("Weight grid is too large for this exact search")
    max_units = math.floor(cap * units + 1e-12)
    if asset_count * max_units < units:
        raise ValueError("No fully invested portfolio fits the requested grid and cap")
    # Separators among units enumerate every integer composition of the budget.
    separators = np.array(list(combinations(range(units + asset_count - 1), asset_count - 1)))
    boundaries = np.column_stack((np.full(len(separators), -1), separators,
                                  np.full(len(separators), units + asset_count - 1)))
    allocations = np.diff(boundaries, axis=1) - 1
    return allocations[(allocations <= max_units).all(axis=1)] / units


def choose_weights(window, state_history, current_state, grid, model):
    # Compare each return to the market state known before that return began.
    distance = (state_history - current_state) / model["regime_bandwidth"]
    similarity = -0.5 * np.square(distance)
    kernel = np.exp(similarity - similarity.max())
    mix = model["regime_mix"]
    probability = mix * kernel / kernel.sum() + (1.0 - mix) / len(window)
    mean = probability @ window
    centered = window - mean
    covariance = (centered.T * probability) @ centered / (1.0 - probability @ probability)
    shrinkage = model["covariance_shrinkage"]
    covariance = ((1.0 - shrinkage) * covariance
                  + shrinkage * np.diag(np.diag(covariance)))

    portfolio_mean = SESSIONS_PER_YEAR * np.einsum("ij,j->i", grid, mean)
    variance = SESSIONS_PER_YEAR * np.einsum("ij,jk,ik->i", grid, covariance, grid)
    positive = portfolio_mean > 0
    if not positive.any():
        # With negative expected excess returns, a larger denominator helps the signed ratio.
        return grid[variance.argmin()]

    rho = np.clip(2.0 * expit(model["rho_alpha"] * current_state), *model["rho_bounds"])
    score = np.full(len(grid), -np.inf)
    score[positive] = (rho * np.log(portfolio_mean[positive])
                       - np.log(np.maximum(variance[positive], 1e-16)) / (2.0 * rho))
    return grid[score.argmax()]


def build_targets(closes, universe, market, risk_free, schedule, model):
    ma_sessions = model["market_ma_sessions"]
    lookback = model["return_lookback"]
    lower_rho, upper_rho = model["rho_bounds"]
    if (ma_sessions < 2 or lookback < 2 or model["rho_alpha"] < 0
            or not 0 < lower_rho <= 1 <= upper_rho < 2
            or model["regime_bandwidth"] <= 0 or not 0 <= model["regime_mix"] < 1
            or not 0 <= model["covariance_shrinkage"] <= 1):
        raise ValueError("Invalid adaptive-ratio model parameters")
    grid = feasible_weights(len(universe), model["weight_units"], model["max_asset_weight"])
    returns = closes.pct_change(fill_method=None)
    excess = returns[universe].sub(returns[risk_free], axis=0).to_numpy()
    market_close = closes[market]
    state = (market_close / market_close.rolling(ma_sessions).mean() - 1.0).to_numpy()
    targets = {}
    for signal_row, fill_row in schedule:
        first = signal_row - lookback + 1
        if first < ma_sessions:
            raise RuntimeError(f"Insufficient warm-up for {closes.index[signal_row].date()}")
        targets[closes.index[fill_row]] = choose_weights(
            excess[first:signal_row + 1], state[first - 1:signal_row], state[signal_row],
            grid, model,
        )
    return targets


def rebalance(opening_values, cash, target, friction):
    opening_total = opening_values.sum() + cash
    # Solve for the affordable post-cost value. Target positions use only open prices.
    low, high = 0.0, opening_total
    for _ in range(48):
        middle = (low + high) / 2.0
        traded = np.abs(middle * target - opening_values).sum()
        if middle + friction * traded <= opening_total:
            low = middle
        else:
            high = middle
    new_values = low * target
    changes = np.abs(new_values - opening_values)
    gross_traded = changes.sum()
    remaining = opening_total - new_values.sum() - friction * gross_traded
    if remaining < -1e-12 * opening_total:
        raise RuntimeError("The rebalance requires negative cash")
    return new_values, max(0.0, remaining), gross_traded / opening_total, int(
        np.count_nonzero(changes > 1e-12 * opening_total)
    )


def simulate(opens, closes, targets, start, execution, cost_multiple=1.0):
    commission = execution["commission_bps"]
    slippage = execution["slippage_bps"]
    friction = (commission + slippage) * cost_multiple / 10_000.0
    if min(commission, slippage, cost_multiple) < 0 or not 0 <= friction < 1:
        raise ValueError("Invalid commission, slippage, or cost multiplier")
    index = opens.index
    first_row = index.searchsorted(start)
    if not 0 < first_row < len(index):
        raise RuntimeError("The scoring start must have at least one earlier session")
    overnight = (opens / closes.shift(1)).to_numpy()
    intraday = (closes / opens).to_numpy()
    positions = np.zeros(len(opens.columns))
    cash, previous_total = 1.0, 1.0
    rows = []
    for row in range(first_row, len(index)):
        opening_values = positions * overnight[row]
        weights = targets.get(index[row])
        turnover, trades = 0.0, 0
        if weights is not None:
            opening_values, cash, turnover, trades = rebalance(opening_values, cash, weights, friction)
        positions = opening_values * intraday[row]
        closing_total = positions.sum() + cash
        rows.append((closing_total / previous_total - 1.0, turnover, trades))
        previous_total = closing_total
    return pd.DataFrame(rows, index=index[first_row:], columns=["return", "turnover", "n_trades"])


def metrics(daily_results, cash_returns):
    returns = daily_results["return"].to_numpy()
    excess = returns - cash_returns.loc[daily_results.index].to_numpy()
    growth = np.r_[1.0, np.cumprod(1.0 + returns)]
    years = len(returns) / SESSIONS_PER_YEAR
    scale = math.sqrt(SESSIONS_PER_YEAR)
    standard_dev = returns.std(ddof=1)
    excess_dev = excess.std(ddof=1)
    return {
        "cagr": float(growth[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(standard_dev * scale),
        "sharpe": float(returns.mean() / standard_dev * scale) if standard_dev > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_dev * scale) if excess_dev > 0 else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1.0).min()),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(results, cash_returns, is_end):
    return {
        "is": metrics(results.loc[results.index <= is_end], cash_returns),
        "oos": metrics(results.loc[results.index > is_end], cash_returns),
        "full": metrics(results, cash_returns),
    }


def encode(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=False)


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe, market, risk_free = config["universe"], config["market_proxy"], config["cash_proxy"]
    if len(universe) < 2 or len(set(universe)) != len(universe) or risk_free in universe:
        raise ValueError("Provide distinct allocation assets and a separate cash proxy")
    tickers = sorted(set(universe) | {market, risk_free})
    dates = {key: pd.Timestamp(value) for key, value in config["dates"].items()}
    start, is_end, end = dates["start"], dates["is_end"], dates["end"]
    if not dates["download_start"] < start < is_end < end:
        raise ValueError("Download, scoring and split dates must be in chronological order")
    execution, model = config["execution"], config["model"]
    lag = execution["lag_sessions"]
    quotes, fetched_at = load_quotes(tickers, dates["download_start"], end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: Yahoo split-normalized open/close, forward ex-date total-return scaling")

    is_opens, is_closes = total_return_prices(quotes.loc[:is_end])
    is_schedule = monthly_schedule(is_closes.index, start, lag)
    is_cash = is_closes[risk_free].pct_change(fill_method=None)
    is_targets = build_targets(is_closes, universe, market, risk_free, is_schedule, model)
    is_default = simulate(is_opens[universe], is_closes[universe], is_targets, start, execution)
    print(f"ROBUSTNESS (IS only) default: {encode(metrics(is_default, is_cash))}")
    trials = 1
    for variant in config["robustness"]:
        overrides = {key: value for key, value in variant.items() if key not in {"name", "cost_multiple"}}
        if overrides.keys() - model.keys():
            raise ValueError(f"Unknown model override for {variant['name']}")
        targets = build_targets(is_closes, universe, market, risk_free, is_schedule, model | overrides)
        alternative = simulate(is_opens[universe], is_closes[universe], targets, start, execution,
                               variant.get("cost_multiple", 1.0))
        trials += 1
        print(f"ROBUSTNESS (IS only) {variant['name']}: {encode(metrics(alternative, is_cash))}")
    print(f"PARAM_COMBOS_TRIED: {trials}")

    opens, closes = total_return_prices(quotes)
    cash_returns = closes[risk_free].pct_change(fill_method=None)
    schedule = monthly_schedule(closes.index, start, lag)
    targets = build_targets(closes, universe, market, risk_free, schedule, model)
    official = simulate(opens[universe], closes[universe], targets, start, execution)
    pd.testing.assert_frame_equal(opens.loc[:is_end], is_opens, check_exact=True)
    pd.testing.assert_frame_equal(closes.loc[:is_end], is_closes, check_exact=True)
    full_is_dates = [date for date in targets if date <= is_end]
    if full_is_dates != list(is_targets):
        raise RuntimeError("Full run changed the in-sample fill schedule")
    for date in full_is_dates:
        np.testing.assert_array_equal(targets[date], is_targets[date])
    pd.testing.assert_frame_equal(official.loc[:is_end], is_default, check_exact=True)

    equal_weights = np.full(len(universe), 1.0 / len(universe))
    benchmark = simulate(opens[universe], closes[universe], dict.fromkeys(targets, equal_weights),
                         start, execution)
    report = period_metrics(official, cash_returns, is_end)
    comparison = period_metrics(benchmark, cash_returns, is_end)
    if min(report["is"]["n_trades"], report["oos"]["n_trades"]) < 1:
        raise RuntimeError("The official strategy must have simulated executions in both periods")
    first_signal, first_fill = schedule[0]
    print(f"FIRST_EXECUTION: signal {closes.index[first_signal].date()} close, "
          f"fill {closes.index[first_fill].date()} open")
    print("LOOKAHEAD_SELF_CHECK: month-end close signals, later open sizing and fills, nonnegative cash")
    print("LOOKAHEAD_SELF_CHECK: only market bars and effective ex-date actions, no delayed releases")
    print("LOOKAHEAD_SELF_CHECK: consistent split units and dividend factors from previous closes and ex-date actions")
    print("LOOKAHEAD_SELF_CHECK: standalone IS prices, targets and daily results match the full-run prefix")
    for period in ("is", "oos", "full"):
        print(f"STRATEGY {period.upper()}: {encode(report[period])}")
        print(f"BENCHMARK {period.upper()}: {encode(comparison[period])}")
    data_range = f"{official.index[0].date()} to {official.index[-1].date()}"
    payload = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **report,
        "benchmark": {"name": "Equal-weight ETF basket, monthly", **comparison["full"],
                      "is": comparison["is"], "oos": comparison["oos"]},
    }
    print(f"已回测 | 待审 | {data_range} | IS end {is_end.date()}")
    print("RESULTS_JSON: " + encode(payload))


def stop_on_timeout(_signum, _frame):
    raise SystemExit("Backtest exceeded the 14-minute execution limit")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_on_timeout)
    signal.alarm(14 * 60)
    main()
