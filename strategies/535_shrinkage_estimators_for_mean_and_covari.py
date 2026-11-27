# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "535_ shrinkage_estimators_for_mean_and_covari"\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nrisk_free: BIL\nbenchmark: equal_weight\n\ndata_start: 2012-09-20\nbacktest_start: 2014-01-01\nis_end: 2024-06-06\nbacktest_end: 2026-10-30\n\nlookback_days: 260\nrebalance_frequency: monthly\nexecution_lag_days: 1\ncost_bps: 2.0\nslippage_bps: 3.0\n\nrobustness:\n  lookback_days: [234, 252, 286]\n  high_cost_bps: 4.0\n  high_slippage_bps: 6.0\n'
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
"""Run a long-only COV2 minimum-variance backtest on US sector ETFs."""

import hashlib
import json
import signal
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, minimize


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252


def load_config():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    for key in ("data_start", "backtest_start", "is_end", "backtest_end"):
        config[key] = pd.Timestamp(config[key])
    start, first, split, end = (
        config[key] for key in ("data_start", "backtest_start", "is_end", "backtest_end")
    )
    if not start < first < split < end:
        raise ValueError("Require data_start < backtest_start < is_end < backtest_end.")
    universe = config["universe"]
    if len(universe) < 2 or len(set(universe)) != len(universe) or config["risk_free"] in universe:
        raise ValueError("Use at least two distinct sector ETFs and a separate cash proxy.")
    if config["rebalance_frequency"] != "monthly" or config["benchmark"] != "equal_weight":
        raise ValueError("Expected monthly rebalancing and the equal_weight benchmark.")
    lag = config["execution_lag_days"]
    if type(lag) is not int or lag < 1:
        raise ValueError("execution_lag_days must be a positive integer.")
    lookbacks = [config["lookback_days"], *config["robustness"]["lookback_days"]]
    if any(type(days) is not int or days < 2 for days in lookbacks):
        raise ValueError("All lookbacks must be integers of at least two sessions.")
    if len(set(lookbacks)) != len(lookbacks):
        raise ValueError("Use distinct lookbacks for the default and robustness cases.")
    cost, slippage = config["cost_bps"], config["slippage_bps"]
    stressed_cost = config["robustness"]["high_cost_bps"]
    stressed_slippage = config["robustness"]["high_slippage_bps"]
    if not (0 <= cost <= stressed_cost and 0 <= slippage <= stressed_slippage
            and cost + slippage < stressed_cost + stressed_slippage < 10_000):
        raise ValueError("Use nonnegative costs and a higher friction case below 10,000 bps.")
    return config


def check_prices(prices, tickers, start, end):
    if list(prices.columns) != tickers or prices.empty:
        raise ValueError("Yahoo price panel is empty or has unexpected columns.")
    if prices.index.hasnans or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Yahoo price dates must be unique and increasing.")
    if not (start <= prices.index[0] <= start + pd.Timedelta(days=7)
            and end - pd.Timedelta(days=7) <= prices.index[-1] <= end):
        raise ValueError("Yahoo price panel does not span the requested dates.")
    values = prices.to_numpy(dtype=float)
    bad_rows, bad_cols = np.where(~np.isfinite(values) | (values <= 0))
    if bad_rows.size:
        examples = [f"{tickers[col]} {prices.index[row].date()}"
                    for row, col in zip(bad_rows[:5], bad_cols[:5])]
        raise ValueError(f"Missing or invalid adjusted close: {', '.join(examples)}")


def load_prices(config):
    tickers = [*config["universe"], config["risk_free"]]
    start, end = config["data_start"], config["backtest_end"]
    cache_key = json.dumps(["yfinance daily auto_adjust close", tickers, str(start.date()), str(end.date())])
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / f"adjusted_closes_{digest}.csv"
    if cache_file.is_file():
        with cache_file.open(encoding="utf-8") as source:
            fetched_at = source.readline().removeprefix("# fetched_at=").strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(source, index_col="Date", parse_dates=["Date"])
        check_prices(prices, tickers, start, end)
        return prices, fetched_at

    with tempfile.TemporaryDirectory(dir=CACHE) as working_cache:
        yf.set_tz_cache_location(working_cache)
        for attempt in range(2):
            raw = yf.download(
                tickers, start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
                interval="1d", auto_adjust=True, ignore_tz=True, keepna=True,
                group_by="column", multi_level_index=True, progress=False, timeout=15,
            )
            try:
                if raw is None or raw.empty or "Close" not in raw.columns.get_level_values(0):
                    raise ValueError("Yahoo returned no adjusted closes.")
                prices = raw["Close"].reindex(columns=tickers).sort_index()
                check_prices(prices, tickers, start, end)
            except ValueError as error:
                if attempt == 1:
                    raise RuntimeError(f"Public Yahoo download failed: {error}") from error
            else:
                break

        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        temporary_csv = Path(working_cache) / "prices.csv"
        with temporary_csv.open("w", encoding="utf-8") as destination:
            destination.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(destination, index_label="Date", date_format="%Y-%m-%d")
        temporary_csv.replace(cache_file)
    return prices, fetched_at


def cov2(returns):
    """C149 §3.2, equations (6)–(8), with a two-parameter target."""
    sessions, assets = returns.shape
    centered = returns - returns.mean(axis=0)
    sample = centered.T @ centered / sessions
    off_diagonal = ~np.eye(assets, dtype=bool)
    sample_off_sum = sample[off_diagonal].sum()
    target = np.full_like(sample, sample_off_sum / (assets * (assets - 1)))
    np.fill_diagonal(target, np.trace(sample) / assets)

    squared = centered ** 2
    pi_terms = squared.T @ squared / sessions - sample ** 2
    off_products = centered.sum(axis=1) ** 2 - squared.sum(axis=1)
    rho_off = (np.mean(off_products ** 2) - sample_off_sum ** 2) / (assets * (assets - 1))
    rho = np.trace(pi_terms) + rho_off
    gamma = np.sum((sample - target) ** 2)
    if gamma <= np.finfo(float).eps * np.sum(sample ** 2):
        return sample
    intensity = np.clip((pi_terms.sum() - rho) / (sessions * gamma), 0.0, 1.0)
    return (1.0 - intensity) * sample + intensity * target


def minimum_variance_weights(covariance):
    assets = covariance.shape[0]
    equal = np.full(assets, 1.0 / assets)
    typical_variance = np.trace(covariance) / assets
    if typical_variance == 0:
        return equal
    scaled = covariance / typical_variance
    result = minimize(
        lambda weights: 0.5 * weights @ scaled @ weights,
        equal,
        jac=lambda weights: scaled @ weights,
        bounds=[(0.0, 1.0)] * assets,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0,
                     "jac": lambda weights: np.ones_like(weights)},
        method="SLSQP", options={"ftol": 1e-13, "maxiter": 200},
    )
    weights = result.x
    gradient = scaled @ weights
    if (not result.success or abs(weights.sum() - 1.0) > 1e-9
            or weights.min() < -1e-9 or gradient @ weights - gradient.min() > 2e-6):
        raise RuntimeError(f"Minimum variance solve failed: {result.message}")
    weights = weights.clip(min=0.0)
    return weights / weights.sum()


def build_targets(asset_returns, start, lookback, lag):
    months = asset_returns.index.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])
    values = asset_returns.to_numpy()
    targets = {}
    for signal_row in month_ends:
        execution_row = signal_row + lag
        if execution_row >= len(months) or asset_returns.index[execution_row] < start:
            continue
        first_row = signal_row + 1 - lookback
        if first_row < 0:
            raise ValueError(f"Insufficient warmup for lookback_days={lookback}.")
        try:
            weights = minimum_variance_weights(cov2(values[first_row:signal_row + 1]))
        except RuntimeError as error:
            raise RuntimeError(f"Signal {asset_returns.index[signal_row].date()}: {error}") from error
        targets[execution_row] = weights
    if not targets:
        raise ValueError("No scheduled rebalance in the evaluation period.")
    return targets


def simulate(asset_returns, targets, friction_bps):
    first_execution = min(targets)
    calendar = asset_returns.index[first_execution:]
    values = asset_returns.to_numpy()
    weights = np.zeros(values.shape[1])
    daily_results = np.zeros((len(calendar), 3))
    fee = friction_bps / 10_000.0
    for output_row, row in enumerate(range(first_execution, len(values))):
        drifted = weights * (1.0 + values[row])
        gross_growth = 1.0 + weights @ values[row]
        weights = drifted / gross_growth
        retained = 1.0
        if row in targets:
            target = targets[row]
            # Express both pre-fill and post-fill exposures per dollar just before the fill.
            if fee:
                retained = brentq(
                    lambda fraction: fraction + fee * np.abs(fraction * target - weights).sum() - 1.0,
                    0.0, 1.0,
                )
            changes = np.abs(retained * target - weights)
            daily_results[output_row, 1] = changes.sum()
            daily_results[output_row, 2] = np.count_nonzero(changes > 1e-9)
            weights = target
        daily_results[output_row, 0] = gross_growth * retained - 1.0

    return pd.DataFrame(daily_results, index=calendar, columns=["return", "turnover", "n_trades"])


def metrics(daily_results, cash_returns):
    returns = daily_results["return"].to_numpy()
    if len(returns) < 2:
        raise ValueError("A metrics period must contain at least two sessions.")
    excess = returns - cash_returns.loc[daily_results.index].to_numpy()
    growth = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    years = len(returns) / SESSIONS_PER_YEAR
    annual_scale = np.sqrt(SESSIONS_PER_YEAR)
    vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    return {
        "cagr": float(growth[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(vol * annual_scale),
        "sharpe": float(annual_scale * returns.mean() / vol) if vol else None,
        "sharpe_excess": float(annual_scale * excess.mean() / excess_vol) if excess_vol else None,
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1.0)),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, cash_returns, is_end):
    return {
        "is": metrics(daily_results.loc[:is_end], cash_returns),
        "oos": metrics(daily_results.loc[daily_results.index > is_end], cash_returns),
        "full": metrics(daily_results, cash_returns),
    }


def main():
    config = load_config()
    prices, fetched_at = load_prices(config)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    asset_returns = returns[config["universe"]]
    cash_returns = returns[config["risk_free"]]
    start, is_end = config["backtest_start"], config["is_end"]
    lag, lookback = config["execution_lag_days"], config["lookback_days"]
    cost, slippage = config["cost_bps"], config["slippage_bps"]
    is_asset_returns = asset_returns.loc[:is_end]

    # Freeze the official specification and run all alternatives on the IS slice.
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True dividend- and split-adjusted close total-return")
    print(f"RISK_FREE_PROXY: {config['risk_free']} adjusted daily total return")
    print("ROBUSTNESS (IS only)")
    robustness = config["robustness"]
    cases = [(days, cost, slippage) for days in robustness["lookback_days"]]
    cases.append((lookback, robustness["high_cost_bps"], robustness["high_slippage_bps"]))
    for days, case_cost, case_slippage in cases:
        targets = build_targets(is_asset_returns, start, days, lag)
        run = simulate(is_asset_returns, targets, case_cost + case_slippage)
        summary = metrics(run, cash_returns.loc[:is_end])
        print(f"  lookback_days={days} cost_bps={case_cost:g} slippage_bps={case_slippage:g} "
              f"{json.dumps(summary, allow_nan=False)}")
    targets = build_targets(asset_returns, start, lookback, lag)
    friction = cost + slippage
    daily_results = simulate(asset_returns, targets, friction)
    equal = np.full(len(config["universe"]), 1.0 / len(config["universe"]))
    benchmark_results = simulate(asset_returns, dict.fromkeys(targets, equal), friction)
    official = period_metrics(daily_results, cash_returns, is_end)
    benchmark = period_metrics(benchmark_results, cash_returns, is_end)
    print(f"PARAM_COMBOS_TRIED: {1 + len(cases)}")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(official[period], allow_nan=False)}")
        print(f"BENCHMARK equal_weight {period.upper()}: {json.dumps(benchmark[period], allow_nan=False)}")

    result = {
        "strategy": config["strategy"],
        "data_range": {
            "start": str(daily_results.index[0].date()),
            "end": str(daily_results.index[-1].date()),
        },
        **official,
        "benchmark": {"name": config["benchmark"], **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


def stop_on_timeout(_signum, _frame):
    raise SystemExit("Backtest stopped at the 14-minute time limit.")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_on_timeout)
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)
