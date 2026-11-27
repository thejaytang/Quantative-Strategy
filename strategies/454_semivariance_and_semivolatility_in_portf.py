# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 454_semivariance_and_semivolatility_in_portf\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nrisk_free: BIL\ndata_start: 2008-03-01\ndata_end: 2026-11-06\nis_end: 2017-12-31\nlookback_sessions: 504\nrebalance_sessions: 21\nexecution_lag_sessions: 1\ntransaction_cost_bps: 50\nslippage_bps: 5\nsessions_per_year: 252\n'
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
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_ADJUSTMENT = "yfinance auto_adjust=True total-return adjusted close"
WEIGHT_TOLERANCE = 1e-10


def validate_prices(prices, symbols, start, end):
    prices = prices.sort_index().reindex(columns=symbols).loc[start:end]
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("Empty price panel or duplicate price dates.")
    if (prices.index[0] - start).days > 7 or (end - prices.index[-1]).days > 7:
        raise ValueError("The price panel does not cover the requested date range.")
    valid = np.isfinite(prices) & prices.gt(0)
    if not valid.all().all():
        first_bad_date = valid.index[~valid.all(axis=1)][0]
        missing = valid.columns[~valid.loc[first_bad_date]].tolist()
        raise ValueError(f"Missing or invalid prices on {first_bad_date.date()} for {missing}.")
    return prices


def load_prices(symbols, start, end):
    key = json.dumps([PRICE_ADJUSTMENT, sorted(symbols), str(start.date()), str(end.date())])
    cache_file = CACHE / f"close_{hashlib.sha256(key.encode()).hexdigest()[:16]}.csv"
    if cache_file.exists():
        with cache_file.open(encoding="utf-8") as handle:
            prefix, _, fetched_at = handle.readline().strip().partition("=")
            if prefix != "# fetched_at" or datetime.fromisoformat(fetched_at).utcoffset() is None:
                raise ValueError(f"Invalid source timestamp in {cache_file.name}.")
            prices = pd.read_csv(handle, index_col="Date", parse_dates=["Date"])
        return validate_prices(prices, symbols, start, end), fetched_at

    yf_cache = CACHE / "yfinance"
    yf_cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(yf_cache))
    downloaded = yf.download(
        symbols,
        start=start.date().isoformat(),
        end=(end + pd.Timedelta(days=1)).date().isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=4,
        timeout=15,
    )
    if downloaded.empty or "Close" not in downloaded:
        raise RuntimeError("yfinance returned no adjusted closing prices.")
    prices = validate_prices(downloaded["Close"], symbols, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as handle:
        temp_path = Path(handle.name)
        try:
            handle.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(handle, index_label="Date")
            handle.close()
            temp_path.replace(cache_file)
        finally:
            temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def minimum_semivariance(window, equal_weights):
    # Percent units improve numerical conditioning without changing the minimizer.
    scenarios = 100.0 * window

    def objective(weights):
        downside = np.minimum(scenarios @ weights, 0.0)
        risk = np.mean(downside**2)
        gradient = (2.0 / len(scenarios)) * (scenarios.T @ downside)
        return risk, gradient

    result = minimize(
        objective,
        equal_weights,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(equal_weights),
        constraints={
            "type": "eq",
            "fun": lambda weights: weights.sum() - 1.0,
            "jac": lambda weights: np.ones_like(weights),
        },
        options={"ftol": 1e-10, "maxiter": 300},
    )
    weights = result.x
    if (not result.success or not np.isfinite(weights).all()
            or weights.min() < -WEIGHT_TOLERANCE or abs(weights.sum() - 1.0) > 1e-8):
        return equal_weights, False
    weights = np.where(weights > WEIGHT_TOLERANCE, weights, 0.0)
    return weights / weights.sum(), True


def schedule_targets(returns, lookback, cadence, lag):
    equal_weights = np.full(returns.shape[1], 1.0 / returns.shape[1])
    strategy = {}
    benchmark = {}
    fallbacks = 0
    for signal in range(lookback - 1, len(returns) - lag, cadence):
        window = returns[signal + 1 - lookback:signal + 1]
        target, success = minimum_semivariance(window, equal_weights)
        execution = signal + lag
        strategy[execution] = target
        benchmark[execution] = equal_weights
        fallbacks += int(not success)
    if not strategy:
        raise ValueError("Insufficient price history for the lookback and execution lag.")
    return strategy, benchmark, fallbacks


def simulate(returns, targets, friction):
    first = min(targets)
    weights = np.zeros(returns.shape[1])
    rows = []
    for day in range(first, len(returns)):
        marked = weights * (1.0 + returns[day])
        gross = 1.0 + weights @ returns[day]
        weights = marked / gross
        turnover, trades = 0.0, 0
        target = targets.get(day)
        if target is not None:
            # Solve for post-cost wealth using the dollar turnover of the fill.
            def budget(net):
                return net + friction * np.abs(net * target - weights).sum() - 1.0

            net = brentq(budget, 0.0, 1.0) if friction else 1.0
            trade_sizes = np.abs(net * target - weights)
            turnover = trade_sizes.sum()
            trades = np.count_nonzero(trade_sizes > WEIGHT_TOLERANCE)
            gross *= net
            weights = target
        rows.append((gross - 1.0, turnover, trades))
    return rows


def metrics(period, annualization):
    returns = period["return"].to_numpy()
    excess = returns - period["risk_free"].to_numpy()
    wealth = np.r_[1.0, np.cumprod(1.0 + returns)]

    def sharpe(series):
        volatility = series.std(ddof=1)
        return float(series.mean() / volatility * np.sqrt(annualization)) if volatility > 0 else None

    return {
        "cagr": float(wealth[-1] ** (annualization / len(returns)) - 1.0),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(annualization)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(excess),
        "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0)),
        "total_return": float(wealth[-1] - 1.0),
        "turnover": float(period["turnover"].sum()),
        "n_trades": int(period["n_trades"].sum()),
    }


def summarize(daily, is_end, annualization):
    segments = {"is": daily.loc[:is_end], "oos": daily.loc[daily.index > is_end], "full": daily}
    if any(len(period) < 2 for period in segments.values()):
        raise ValueError("The IS/OOS split must leave at least two evaluated sessions in each period.")
    return {label: metrics(period, annualization) for label, period in segments.items()}


def print_metrics(name, summary):
    for period, values in summary.items():
        formatted = []
        for metric, value in values.items():
            if value is None:
                display = "n/a"
            elif metric == "n_trades":
                display = str(value)
            elif metric.startswith("sharpe"):
                display = f"{value:.3f}"
            else:
                display = f"{value:.2%}"
            formatted.append(f"{metric}={display}")
        print(f"{name} {period.upper()} " + "  ".join(formatted))


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    risk_free = config["risk_free"]
    symbols = [*universe, risk_free]
    start, end, is_end = (pd.Timestamp(config[key]) for key in ("data_start", "data_end", "is_end"))
    lookback = config["lookback_sessions"]
    cadence = config["rebalance_sessions"]
    lag = config["execution_lag_sessions"]
    annualization = config["sessions_per_year"]
    cost, slippage = (float(config[key]) for key in ("transaction_cost_bps", "slippage_bps"))
    if not start < is_end < end or len(universe) < 2 or len(symbols) != len(set(symbols)):
        raise ValueError("Use chronological dates and distinct ETFs for the portfolio and cash reference.")
    if not all(type(value) is int and value >= 1 for value in (lookback, cadence, lag, annualization)):
        raise ValueError("Lookback, rebalance cadence, lag and annualization must be positive integers.")
    if not (cost >= 0 and slippage >= 0 and cost + slippage < 10_000):
        raise ValueError("Friction must be nonnegative and total less than 10,000 bps.")
    friction = (cost + slippage) / 10_000.0

    prices, fetched_at = load_prices(symbols, start, end)
    daily_returns = prices.pct_change(fill_method=None).iloc[1:]
    asset_returns = daily_returns[universe].to_numpy()
    strategy_targets, benchmark_targets, fallbacks = schedule_targets(asset_returns, lookback, cadence, lag)
    first_day = min(strategy_targets)
    summaries = {}
    for label, targets in (("MSV0", strategy_targets), ("EW", benchmark_targets)):
        daily = pd.DataFrame(
            simulate(asset_returns, targets, friction),
            index=daily_returns.index[first_day:],
            columns=["return", "turnover", "n_trades"],
        )
        daily["risk_free"] = daily_returns[risk_free]
        summaries[label] = summarize(daily, is_end, annualization)

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"OPTIMIZER_FALLBACKS: {fallbacks}")
    for label, summary in summaries.items():
        print_metrics(label, summary)
    result = {
        "strategy": config["strategy"],
        "data_range": {
            "start": str(daily_returns.index[first_day].date()),
            "end": str(daily_returns.index[-1].date()),
        },
        **summaries["MSV0"],
        "benchmark": summaries["EW"],
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
