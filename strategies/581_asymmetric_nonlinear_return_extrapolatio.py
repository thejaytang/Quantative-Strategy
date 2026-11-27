# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 581_asymmetric_nonlinear_return_extrapolatio\nuniverse:\n  equity: SPY\n  treasury: IEF\n  cash_proxy: BIL\ndates:\n  data_start: 2007-06-01\n  start: 2008-01-02\n  is_end: 2016-12-30\n  end: 2025-12-31\nexecution:\n  lag_sessions: 1\n  distribution_cash_lag_sessions: 45\n  cost_bps: 2.0\n  slippage_bps: 3.0\nsignal:\n  lookback: 60\n  half_life: 20.0\n  annual_drift_spread: 0.06\n  strength: 0.15\n  saturation: 3.0\n  asymmetry: 0.5\n  risk_aversion: 3.0\n  annual_spread_variance_floor: 0.01\nbenchmark_equity_weight: 0.60\nrobustness:\n  - {saturation: 2.5}\n  - {saturation: 3.5}\n  - {asymmetry: 0.4}\n  - {asymmetry: 0.6}\n  - {linear: true}\n  - {cost_bps: 6.0, slippage_bps: 9.0}\n'
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
"""C185 research backtest, implemented independently.

Yan, Ye, Zong and Chen (2026), equations (1), (7), and the myopic term of (12).
https://arxiv.org/abs/2606.10805
https://arxiv.org/html/2606.10805v1

Related implementations consulted for ideas, with no code or data copied:
Atmaz's linear extrapolation replication (CC0 1.0): https://doi.org/10.7910/DVN/A9979J
Sepp and Lucic's EWMA trend system (main branch MIT): https://github.com/ArturSepp/TrendFollowingSystems
https://pypi.org/project/trendfollowing/
QuantConnect tactical allocation tutorial index:
https://www.quantconnect.com/forum/discussion/12929/boot-camp-momentum-based-tactical-allocation/
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
BAR_FIELDS = ["Open", "Close", "Distribution", "Stock Splits"]


def load_history(symbol, start, end, cache_dir):
    path = cache_dir / f"yfinance_{symbol}_{start.date()}_{end.date()}.csv"
    cached = path.exists()
    if cached:
        frame = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
        fetched_at = frame.pop("fetched_at").iloc[0]
    else:
        history = yf.Ticker(symbol).history(
            start=str(start.date()),
            end=str((end + pd.Timedelta(days=1)).date()),
            interval="1d",
            auto_adjust=False,
            actions=True,
            timeout=10,
        )
        if history.empty:
            raise ValueError(f"No Yahoo Finance data for {symbol}")
        frame = history[["Open", "Close", "Stock Splits"]].copy()
        frame["Distribution"] = history["Dividends"] + history.get("Capital Gains", 0.0)
        frame.index = frame.index.tz_localize(None).rename("Date")
        fetched_at = datetime.now(UTC).isoformat(timespec="seconds")

    frame = frame[BAR_FIELDS]
    if (not frame.index.is_monotonic_increasing or not frame.index.is_unique
            or frame.index[0] != start or frame.index[-1] != end):
        raise ValueError(f"Incomplete or unordered history for {symbol}: {start.date()} to {end.date()}")
    if (not np.isfinite(frame.to_numpy()).all() or (frame[["Open", "Close"]] <= 0).any().any()
            or (frame[["Distribution", "Stock Splits"]] < 0).any().any()):
        raise ValueError(f"Invalid price or distribution data for {symbol}")

    if not cached:
        with NamedTemporaryFile(mode="w", dir=cache_dir, suffix=".tmp", delete=False) as out:
            frame.assign(fetched_at=fetched_at).to_csv(out)
        Path(out.name).replace(path)
    return frame, datetime.fromisoformat(fetched_at)


def load_market(symbols, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    # Keep transient yfinance state in the strategy directory, apart from the price snapshot.
    with TemporaryDirectory(dir=cache_dir) as yf_cache:
        yf.set_tz_cache_location(yf_cache)
        yf.config.debug.hide_exceptions = False
        snapshots = {symbol: load_history(symbol, start, end, cache_dir) for symbol in symbols}

    sessions = snapshots[symbols[0]][0].index
    for symbol, (frame, _) in snapshots.items():
        if not frame.index.equals(sessions):
            raise ValueError(f"Session mismatch for {symbol}; check all three Yahoo histories")
    bars = pd.concat({symbol: frame for symbol, (frame, _) in snapshots.items()}, axis=1)
    first_fetch = min(timestamp for _, timestamp in snapshots.values())
    return bars, first_fetch.isoformat(timespec="seconds")


def signal_state(returns, equity, treasury, signal):
    lookback = signal["lookback"]
    time_weights = np.exp2(-np.arange(lookback - 1, -1, -1) / signal["half_life"])
    time_weights /= time_weights.sum()

    def weighted_mean(series):
        return series.rolling(lookback).apply(lambda window: window @ time_weights, raw=True)

    treasury_returns = returns[treasury]
    spread_returns = returns[equity] - treasury_returns
    spread_mean = weighted_mean(spread_returns)
    treasury_mean = weighted_mean(treasury_returns)
    spread_variance = weighted_mean(spread_returns.pow(2)) - spread_mean.pow(2)
    treasury_spread_cov = weighted_mean(treasury_returns * spread_returns) - treasury_mean * spread_mean
    return SESSIONS_PER_YEAR * pd.DataFrame({
        "sentiment": weighted_mean(np.log1p(returns[equity])),
        "spread_variance": spread_variance,
        "treasury_spread_cov": treasury_spread_cov,
    })


def equity_targets(state, signal, linear=False):
    if linear:
        extrapolation = signal["saturation"] * (1 + signal["asymmetry"]) * state["sentiment"]
    else:
        z = np.tanh(signal["saturation"] * state["sentiment"])
        extrapolation = z + signal["asymmetry"] * z * (1 - z)
    drift_spread = signal["annual_drift_spread"] + signal["strength"] * extrapolation
    spread_variance = state["spread_variance"].clip(lower=signal["annual_spread_variance_floor"])
    return ((drift_spread / signal["risk_aversion"] - state["treasury_spread_cov"])
            / spread_variance).clip(0.0, 1.0)


def monthly_schedule(close_targets, lag, start):
    month = pd.Series(close_targets.index.to_period("M"), index=close_targets.index)
    first_session = month.ne(month.shift())
    delayed = close_targets.shift(lag)
    if delayed.loc[first_session & (delayed.index >= start)].isna().any():
        raise ValueError("The pre-sample history is too short for the configured signal and execution lag")
    return delayed.where(first_session)


def rebalance(positions, cash, opening, equity_weight, sell_factor, buy_factor):
    open_values = positions * opening
    deployable = open_values.sum() + cash
    target_values = deployable * np.array([equity_weight, 1 - equity_weight])
    change = target_values - open_values
    change[np.abs(change) <= deployable * 1e-12] = 0.0

    sells = np.maximum(-change, 0.0)
    cash += sells.sum() * sell_factor
    buys = np.maximum(change, 0.0)
    requested_cash = buys.sum() * buy_factor
    spent = min(cash, requested_cash)
    if requested_cash > 0:
        buys *= spent / requested_cash
    cash -= spent

    positions = (open_values - sells + buys) / opening
    turnover = (sells.sum() + buys.sum()) / deployable
    trades = np.count_nonzero(sells) + np.count_nonzero(buys)
    return positions, cash, turnover, trades


def simulate(opening, closing, distributions, schedule, execution):
    commission = execution["cost_bps"] / 10_000
    slippage = execution["slippage_bps"] / 10_000
    sell_factor = (1 - slippage) * (1 - commission)
    buy_factor = (1 + slippage) * (1 + commission)
    distribution_lag = execution["distribution_cash_lag_sessions"]
    future_cash = np.zeros(len(opening) + distribution_lag)
    pending_distributions = 0.0

    positions = np.zeros(opening.shape[1])
    cash = previous_value = 1.0
    daily_results = []
    inputs = zip(opening.to_numpy(), closing.to_numpy(), distributions.to_numpy(), schedule.to_numpy(), strict=True)
    for i, (open_prices, close_prices, distribution, target) in enumerate(inputs):
        cash += future_cash[i]
        pending_distributions -= future_cash[i]
        # Entitlement belongs to the previous close's shares. Recognize it after execution.
        earned = float(positions @ distribution)

        turnover = trades = 0
        if np.isfinite(target):
            positions, cash, turnover, trades = rebalance(
                positions, cash, open_prices, target, sell_factor, buy_factor,
            )

        future_cash[i + distribution_lag] += earned
        pending_distributions += earned
        value = float(positions @ close_prices + cash + pending_distributions)
        if cash < 0 or (positions < 0).any() or not np.isfinite(value) or value <= 0:
            raise ArithmeticError(f"Cash, position or value constraint failed on {opening.index[i].date()}")
        daily_results.append((value / previous_value - 1, turnover, trades))
        previous_value = value

    return pd.DataFrame(daily_results, index=opening.index, columns=["return", "turnover", "n_trades"])


def metrics(daily_results, cash_proxy):
    returns = daily_results["return"]
    excess = returns - cash_proxy.loc[returns.index]
    annualization = np.sqrt(SESSIONS_PER_YEAR)

    def sharpe(series):
        std = series.std(ddof=1)
        return float(annualization * series.mean() / std) if std > 0 else None

    path = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    years = len(returns) / SESSIONS_PER_YEAR
    return {
        "cagr": float(path[-1] ** (1 / years) - 1),
        "ann_vol": float(annualization * returns.std(ddof=1)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(excess),
        "max_drawdown": float(np.min(path / np.maximum.accumulate(path) - 1)),
        "total_return": float(path[-1] - 1),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def display(label, value):
    print(f"{label}: {json.dumps(value, allow_nan=False)}")


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    equity, treasury, cash_symbol = (config["universe"][role] for role in ("equity", "treasury", "cash_proxy"))
    symbols = [equity, treasury, cash_symbol]
    if len(set(symbols)) != 3 or not all(s.isascii() and s.isalpha() and s.isupper() for s in symbols):
        raise ValueError("Configure distinct uppercase U.S. ETF tickers")
    data_start, start, is_end, end = (pd.Timestamp(config["dates"][key])
                                      for key in ("data_start", "start", "is_end", "end"))
    execution = config["execution"]
    lag = execution["lag_sessions"]
    if not data_start < start < is_end < end or not isinstance(lag, int) or lag < 1:
        raise ValueError("Configure ordered dates and an integer execution lag of at least one session")
    distribution_lag = execution["distribution_cash_lag_sessions"]
    if not isinstance(distribution_lag, int) or distribution_lag < 1:
        raise ValueError("Distribution cash must lag its ex-date by at least one session")

    bars, fetched_at = load_market(symbols, data_start, end)
    opening = bars.xs("Open", axis=1, level=1)
    closing = bars.xs("Close", axis=1, level=1)
    distributions = bars.xs("Distribution", axis=1, level=1)
    split_counts = bars.xs("Stock Splits", axis=1, level=1).ne(0).sum()
    total_returns = (closing + distributions) / closing.shift(1) - 1
    cash_proxy = total_returns[cash_symbol]
    assets = [equity, treasury]
    signal = config["signal"]
    benchmark_weight = config["benchmark_equity_weight"]
    if not 0 <= benchmark_weight <= 1:
        raise ValueError("Benchmark equity weight must be between zero and one")
    if start not in closing.index or is_end not in closing.index:
        raise ValueError("The strategy start and IS end must be trading sessions in the data")

    def run(close_targets, last, friction):
        if not all(0 <= friction[key] < 10_000 for key in ("cost_bps", "slippage_bps")):
            raise ValueError("Cost and slippage must be in [0, 10000) basis points")
        dates = slice(start, last)
        schedule = monthly_schedule(close_targets, lag, start).loc[dates]
        return simulate(opening.loc[dates, assets], closing.loc[dates, assets],
                        distributions.loc[dates, assets], schedule, friction)

    # The robustness cases are evaluated on a physically truncated input history.
    is_history = total_returns.loc[:is_end]
    is_state = signal_state(is_history, equity, treasury, signal)
    variants = []
    for variant in config["robustness"]:
        variant_signal = signal | {key: variant[key] for key in ("saturation", "asymmetry") if key in variant}
        friction = execution | {key: variant[key] for key in ("cost_bps", "slippage_bps") if key in variant}
        close_targets = equity_targets(is_state, variant_signal, linear=variant.get("linear", False))
        variants.append({"variant": variant, "is": metrics(run(close_targets, is_end, friction), cash_proxy)})

    state = signal_state(total_returns, equity, treasury, signal)
    daily_results = run(equity_targets(state, signal), end, execution)
    benchmark_targets = pd.Series(benchmark_weight, index=state.index)
    benchmark = run(benchmark_targets, end, execution)
    is_days = daily_results.index <= is_end
    output = {
        "strategy": config["strategy"],
        "data_range": {"start": str(daily_results.index[0].date()), "end": str(daily_results.index[-1].date())},
        "is": metrics(daily_results.loc[is_days], cash_proxy),
        "oos": metrics(daily_results.loc[~is_days], cash_proxy),
        "full": metrics(daily_results, cash_proxy),
        "benchmark": metrics(benchmark, cash_proxy),
    }
    if any(output[period]["n_trades"] == 0 for period in ("is", "oos")):
        raise ValueError("The default strategy must have fills in both periods")

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=False, split-adjusted OHLC and explicit ETF distributions")
    print(f"Date Range: {start.date()} to {end.date()}")
    print(f"IS_END: {is_end.date()} | OOS_START: {daily_results.index[~is_days][0].date()}")
    display("SPLIT_EVENTS", {symbol: int(count) for symbol, count in split_counts.items()})
    print(f"PARAM_COMBOS_TRIED: {1 + len(variants)}")
    print("ROBUSTNESS (IS only)")
    for result in variants:
        display("ROBUSTNESS_IS", result)
    print("STATUS: 已回测 / pending review")
    for period in ("is", "oos", "full"):
        display(period.upper(), output[period])
    print(f"BENCHMARK: {benchmark_weight:.0%} {equity} / {1 - benchmark_weight:.0%} {treasury}, monthly")
    display("BENCHMARK_IS", metrics(benchmark.loc[is_days], cash_proxy))
    display("BENCHMARK_OOS", metrics(benchmark.loc[~is_days], cash_proxy))
    display("BENCHMARK_FULL", output["benchmark"])
    display("RESULTS_JSON", output)


if __name__ == "__main__":
    main()
