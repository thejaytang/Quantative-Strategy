# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 520_beyond_de_prado_and_cotton_hierarchical\nuniverse: [SPY, IWM, EFA, EEM, IEF, TLT, LQD, GLD, VNQ]\nrisk_free_proxy: BIL\n\ndates:\n  price_start: 2007-07-02\n  backtest_start: 2009-01-02\n  is_end: 2017-12-29\n  backtest_end: 2025-12-31\n\nmodel:\n  lookback_sessions: 252\n  gamma: 0.50\n  risk_aversion: 5.0\n\nexecution:\n  lag_sessions: 1\n  cost_bps: 2.0\n  slippage_bps: 3.0\n\nrobustness:\n  - label: gamma_0.30\n    model: {gamma: 0.30}\n  - label: gamma_0.70\n    model: {gamma: 0.70}\n  - label: lookback_189\n    model: {lookback_sessions: 189}\n  - label: lookback_315\n    model: {lookback_sessions: 315}\n  - label: double_friction\n    execution: {cost_bps: 4.0, slippage_bps: 6.0}\n'
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
"""Monthly public-ETF backtest of the constrained CRISP covariance operator."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
TIME_LIMIT_SECONDS = 840
WEIGHT_TOLERANCE = 1e-10


def check_prices(prices, tickers, start, end):
    index = prices.index
    if prices.empty or list(prices.columns) != tickers:
        raise ValueError("Adjusted close data are missing one or more requested series")
    if not isinstance(index, pd.DatetimeIndex) or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Price index must contain unique sessions in date order")
    if index[0] != start or index[-1] != end:
        raise ValueError(f"Expected price sessions from {start.date()} to {end.date()}")
    values = prices.to_numpy(dtype=float)
    invalid = ~np.isfinite(values) | (values <= 0)
    if invalid.any():
        row, col = np.argwhere(invalid)[0]
        raise ValueError(f"Missing or invalid adjusted close for {tickers[col]} on {index[row].date()}")


def load_prices(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    key = hashlib.sha256(",".join(tickers).encode()).hexdigest()[:16]
    path = cache_dir / f"yahoo_adjusted_close_{start.date()}_{end.date()}_{key}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as source:
            header = source.readline().strip()
            fetched_at = header.removeprefix("# fetched_at=")
            if header == fetched_at:
                raise ValueError(f"Missing fetch timestamp in {path.name}")
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(source, index_col="Date", parse_dates=["Date"])
        check_prices(prices, tickers, start, end)
        return prices, fetched_at

    with TemporaryDirectory(prefix="yf_", dir=cache_dir) as provider_cache:
        yf.set_tz_cache_location(provider_cache)
        history = yf.download(
            tickers, start=start, end=end + pd.Timedelta(days=1), interval="1d",
            auto_adjust=True, ignore_tz=True, multi_level_index=True,
            progress=False, threads=True, timeout=15,
        )
    if history is None or history.empty or "Close" not in history:
        raise RuntimeError("yfinance returned no adjusted closes; rerun when Yahoo Finance is reachable")
    prices = history["Close"].reindex(columns=tickers).rename_axis(index="Date", columns=None)
    check_prices(prices, tickers, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", dir=cache_dir, encoding="utf-8", delete=False) as temp:
        temporary_path = Path(temp.name)
        try:
            temp.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(temp)
            temp.flush()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def crisp_weights(window, gamma, risk_aversion):
    # Section 9.2, equation 53, with a fixed risk-aversion multiplier.
    signal_mean = window.mean(axis=0)
    covariance = np.cov(window, rowvar=False, ddof=1)
    penalty = risk_aversion * (gamma * covariance + (1 - gamma) * np.diag(covariance.diagonal()))
    scale = max(np.abs(penalty).max(), np.abs(signal_mean).max())
    size = window.shape[1]
    equal_weights = np.full(size, 1.0 / size)
    if scale == 0.0:
        return equal_weights
    penalty /= scale
    signal_mean /= scale

    def objective(weights):
        return 0.5 * weights @ penalty @ weights - signal_mean @ weights

    def gradient(weights):
        return penalty @ weights - signal_mean

    result = minimize(
        objective, equal_weights, jac=gradient, method="SLSQP",
        bounds=[(0.0, 1.0)] * size,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1,
                     "jac": lambda weights: np.ones_like(weights)},
        options={"ftol": 1e-12, "maxiter": 200},
    )
    weights = result.x
    if not result.success or not np.isfinite(weights).all() or abs(weights.sum() - 1) > 1e-8:
        raise RuntimeError(f"CRISP quadratic program failed ({result.message})")
    weights[weights < WEIGHT_TOLERANCE] = 0.0
    return weights / weights.sum()


def execution_targets(returns, start, model, lag):
    lookback = model["lookback_sessions"]
    gamma = model["gamma"]
    risk_aversion = model["risk_aversion"]
    if type(lookback) is not int or lookback < 2 or not 0 <= gamma <= 1:
        raise ValueError("Require at least two lookback sessions and gamma in [0, 1]")
    if not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("Risk aversion must be positive and finite")
    if type(lag) is not int or lag < 1:
        raise ValueError("Execution lag must be at least one session")

    dates = returns.index
    months = dates.to_period("M")
    first_sessions = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])
    targets = {}
    values = returns.to_numpy()
    for row in first_sessions:
        if dates[row] < start or row + lag >= len(dates):
            continue
        if row + 1 < lookback:
            raise ValueError(f"Insufficient warm-up history at the {dates[row].date()} signal")
        window = values[row + 1 - lookback:row + 1]
        targets[dates[row + lag]] = crisp_weights(window, gamma, risk_aversion)
    if not targets:
        raise ValueError("No monthly allocations can be executed within the requested dates")
    return targets


def simulate(returns, targets, execution):
    charges = np.array([execution["cost_bps"], execution["slippage_bps"]], dtype=float)
    rate = charges.sum() / 10_000
    if not np.isfinite(charges).all() or (charges < 0).any() or rate >= 1:
        raise ValueError("Costs and slippage must be finite, nonnegative, and total less than 10,000 bps")

    weights = np.zeros(returns.shape[1])
    net_returns = np.zeros(len(returns))
    turnover = np.zeros(len(returns))
    trades = np.zeros(len(returns), dtype=int)
    for row, (date, asset_returns) in enumerate(zip(returns.index, returns.to_numpy())):
        gross_move = weights @ asset_returns
        weights = weights * (1 + asset_returns) / (1 + gross_move)
        net_returns[row] = gross_move

        if date in targets:
            target = targets[date]

            # Express trading amounts in units of value before the close rebalance.
            def value_balance(remaining):
                return remaining + rate * np.abs(remaining * target - weights).sum() - 1

            remaining = brentq(value_balance, 0.0, 1.0, xtol=1e-14) if rate else 1.0
            changes = np.abs(remaining * target - weights)
            turnover[row] = changes.sum()
            trades[row] = np.count_nonzero(changes > WEIGHT_TOLERANCE)
            net_returns[row] = (1 + gross_move) * remaining - 1
            weights = target
    return pd.DataFrame({"net": net_returns, "turnover": turnover, "trades": trades}, index=returns.index)


def metrics(daily_results, reference):
    net = daily_results["net"]
    if len(net) < 2:
        raise ValueError("Each reporting period requires at least two sessions")
    scale = np.sqrt(SESSIONS_PER_YEAR)
    years = len(net) / SESSIONS_PER_YEAR
    excess = net - reference.loc[net.index]
    vol = net.std(ddof=1) * scale
    excess_vol = excess.std(ddof=1) * scale
    growth = np.r_[1.0, np.cumprod(1 + net.to_numpy())]
    return {
        "cagr": float(growth[-1] ** (1 / years) - 1),
        "ann_vol": float(vol),
        "sharpe": float(SESSIONS_PER_YEAR * net.mean() / vol) if vol > 0 else None,
        "sharpe_excess": float(SESSIONS_PER_YEAR * excess.mean() / excess_vol) if excess_vol > 0 else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["trades"].sum()),
    }


def report_periods(daily_results, reference, is_end):
    return {
        "is": metrics(daily_results.loc[:is_end], reference),
        "oos": metrics(daily_results.loc[daily_results.index > is_end], reference),
        "full": metrics(daily_results, reference),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    proxy = config["risk_free_proxy"]
    if len(universe) < 2 or len(set(universe)) != len(universe) or proxy in universe:
        raise ValueError("Use at least two distinct ETFs and a separate bill proxy")
    price_start, start, is_end, end = (
        pd.Timestamp(config["dates"][key]) for key in ("price_start", "backtest_start", "is_end", "backtest_end")
    )
    if not price_start < start < is_end < end:
        raise ValueError("Dates must satisfy price_start < backtest_start < is_end < backtest_end")
    prices, fetched_at = load_prices([*universe, proxy], price_start, end)
    if start not in prices.index or is_end not in prices.index:
        raise ValueError("The backtest start and IS end must be observed sessions")
    returns = prices.pct_change(fill_method=None).iloc[1:]
    asset_returns = returns[universe]
    sample = asset_returns.loc[start:]
    reference = returns[proxy]
    model = config["model"]
    execution = config["execution"]

    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True adjusted close, total-return basis")
    print("ROBUSTNESS: in-sample only, fixed one-at-a-time alternatives")
    is_returns = asset_returns.loc[:is_end]
    attempts = 0
    for case in config["robustness"]:
        case_execution = {**execution, **case.get("execution", {})}
        attempts += 1
        targets = execution_targets(is_returns, start, {**model, **case.get("model", {})}, case_execution["lag_sessions"])
        daily_results = simulate(is_returns.loc[start:], targets, case_execution)
        print(f"ROBUSTNESS_IS {case['label']}: {json.dumps(metrics(daily_results, reference), allow_nan=False)}")

    attempts += 1
    targets = execution_targets(asset_returns, start, model, execution["lag_sessions"])
    official = report_periods(simulate(sample, targets, execution), reference, is_end)
    equal_weights = np.full(len(universe), 1.0 / len(universe))
    benchmark_targets = dict.fromkeys(targets, equal_weights)
    benchmark = report_periods(simulate(sample, benchmark_targets, execution), reference, is_end)

    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(official[period], allow_nan=False)}")
        print(f"BENCHMARK EQUAL_WEIGHT {period.upper()}: {json.dumps(benchmark[period], allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {attempts}")
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": sample.index[0].date().isoformat(), "end": sample.index[-1].date().isoformat()},
        **official,
        "benchmark": {"name": "equal_weight_same_etfs_monthly", **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


def stop_at_deadline(_signum, _frame):
    raise SystemExit(f"Backtest reached its {TIME_LIMIT_SECONDS}-second time limit")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_at_deadline)
    signal.alarm(TIME_LIMIT_SECONDS)
    try:
        main()
    finally:
        signal.alarm(0)
