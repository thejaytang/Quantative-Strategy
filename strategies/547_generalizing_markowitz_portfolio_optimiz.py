# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 547_generalizing_markowitz_portfolio_optimiz\nuniverse: [XLK, XLF, XLE, XLV]\nrisk_free_ticker: BIL\n\nhistory_start: 2008-01-01\nstart: 2010-01-04\nis_end: 2017-12-29\nend: 2025-12-31\nsignal_lag_sessions: 1\n\nlookback_sessions: 252\nrisk_aversion: 8.0\ncovariance_shrinkage: 0.10\nturnover_penalty: 0.15\ncost_bps: 2.0\nslippage_bps: 3.0\n\nrobustness:\n  lookback_sessions: [210, 294]\n  turnover_penalty: [0.10, 0.20]\n  friction_multiplier: 2.0\n'
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
"""Monthly long-only sector allocation with a quadratic turnover penalty."""

import json
import signal
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, minimize


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
TIME_LIMIT_SECONDS = 13 * 60


def adjusted_closes(tickers, first_date, last_date):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_name = f"yfinance_adjusted_close_{'_'.join(tickers)}_{first_date}_{last_date}.csv"
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        with cache_path.open() as source:
            fetched_at = source.readline().strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(source, index_col=0, parse_dates=True)
    else:
        end_exclusive = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).date()
        with TemporaryDirectory(dir=CACHE_DIR) as download_cache:
            yf.set_tz_cache_location(download_cache)
            downloaded = yf.download(
                tickers, start=str(first_date), end=str(end_exclusive), interval="1d",
                auto_adjust=True, ignore_tz=True, progress=False, threads=False, timeout=15,
            )
        if downloaded is None or downloaded.empty or "Close" not in downloaded.columns:
            raise RuntimeError("yfinance did not return daily adjusted closes.")
        prices = downloaded["Close"]
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    prices = prices.reindex(columns=tickers).sort_index().loc[str(first_date):str(last_date)]
    values = prices.to_numpy(dtype=float)
    if prices.empty or prices.index.has_duplicates or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Expected a complete panel of positive adjusted closes for all configured ETFs.")
    if (prices.index[0] - pd.Timestamp(first_date) > pd.Timedelta(days=7)
            or pd.Timestamp(last_date) - prices.index[-1] > pd.Timedelta(days=7)):
        raise ValueError("Adjusted close coverage does not span the configured price dates.")
    if not cache_path.exists():
        with NamedTemporaryFile(mode="w", dir=CACHE_DIR, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                temporary.write(f"{fetched_at}\n")
                prices.to_csv(temporary, index_label="Date")
                temporary.close()
                temporary_path.replace(cache_path)
            finally:
                temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def optimal_weights(sample, positions, settings):
    expected_return = sample.mean(axis=0) * SESSIONS_PER_YEAR
    covariance = np.cov(sample, rowvar=False) * SESSIONS_PER_YEAR
    identity = np.eye(len(positions))
    shrinkage = settings["covariance_shrinkage"]
    shrunk_covariance = (1 - shrinkage) * covariance + shrinkage * covariance.diagonal().mean() * identity
    penalty = settings["turnover_penalty"]
    quadratic = settings["risk_aversion"] * shrunk_covariance + penalty * identity
    adjusted_return = expected_return + penalty * positions

    # Proposition 8 gives the budget-constrained stationary point.
    ones = np.ones(len(positions))
    return_direction, budget_direction = np.linalg.solve(
        quadratic, np.column_stack((adjusted_return, ones)),
    ).T
    unconstrained = return_direction + (1 - return_direction.sum()) / budget_direction.sum() * budget_direction
    if unconstrained.min() >= 0:
        return unconstrained

    # Project in the metric induced by the objective's Hessian.
    starting_weights = unconstrained.clip(min=0)
    projection = minimize(
        lambda weights: 0.5 * weights @ quadratic @ weights - adjusted_return @ weights,
        starting_weights / starting_weights.sum(),
        jac=lambda weights: quadratic @ weights - adjusted_return,
        bounds=[(0.0, 1.0)] * len(positions),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1, "jac": lambda weights: ones},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 100},
    )
    if not projection.success or abs(projection.x.sum() - 1) > 1e-8:
        raise RuntimeError(f"Long-only quadratic optimization failed: {projection.message}")
    weights = projection.x.clip(min=0)
    return weights / weights.sum()


def simulate(returns, start, settings, *, equal_weight=False):
    first_session = returns.index.searchsorted(start)
    lag = settings["signal_lag_sessions"]
    lookback = settings["lookback_sessions"]
    if first_session < lookback - 1 or first_session + lag >= len(returns):
        raise ValueError("Insufficient sessions for the configured estimation window and execution lag.")
    friction = (settings["cost_bps"] + settings["slippage_bps"]) / 10_000
    values = returns.to_numpy()
    months = returns.index.to_period("M")
    equal_target = np.full(returns.shape[1], 1 / returns.shape[1])
    positions = np.zeros(returns.shape[1])
    scheduled = {}
    records = []

    for session in range(first_session, len(returns)):
        gross_factor = 1 + positions @ values[session]
        positions = positions * (1 + values[session]) / gross_factor
        turnover = 0.0
        trades = 0

        if session in scheduled:
            target = scheduled.pop(session)
            # Solve for the fraction of value left after paying costs on the executed dollars.
            retained = brentq(
                lambda fraction: fraction + friction * np.abs(fraction * target - positions).sum() - 1,
                0.0, 1.0, xtol=1e-14,
            )
            traded = np.abs(retained * target - positions)
            turnover = float(traded.sum())
            trades = int(np.count_nonzero(traded > 1e-10))
            gross_factor *= retained
            positions = target

        records.append((gross_factor - 1, turnover, trades))
        is_signal_session = session == first_session or months[session] != months[session - 1]
        if is_signal_session and session + lag < len(returns):
            sample = values[session - lookback + 1:session + 1]
            scheduled[session + lag] = equal_target if equal_weight else optimal_weights(sample, positions, settings)

    return pd.DataFrame(records, index=returns.index[first_session:], columns=["return", "turnover", "n_trades"])


def metrics(daily_results, risk_free_returns):
    returns = daily_results["return"]
    excess = returns - risk_free_returns.loc[returns.index]
    growth = (1 + returns).cumprod().to_numpy()
    running_peak = np.maximum.accumulate(np.r_[1.0, growth])[1:]
    annual_scale = np.sqrt(SESSIONS_PER_YEAR)
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(volatility * annual_scale),
        "sharpe": float(returns.mean() / volatility * annual_scale) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_volatility * annual_scale) if excess_volatility > 0 else None,
        "max_drawdown": float((growth / running_peak - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def robustness_cases(settings):
    robustness = settings["robustness"]
    for name in ("lookback_sessions", "turnover_penalty"):
        for value in robustness[name]:
            yield f"{name}={value}", {**settings, name: value}
    multiplier = robustness["friction_multiplier"]
    yield f"cost_and_slippage={multiplier:g}x", {
        **settings,
        "cost_bps": settings["cost_bps"] * multiplier,
        "slippage_bps": settings["slippage_bps"] * multiplier,
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        settings = yaml.safe_load(source)
    start = pd.Timestamp(settings["start"])
    split = pd.Timestamp(settings["is_end"])
    end = pd.Timestamp(settings["end"])
    if not pd.Timestamp(settings["history_start"]) < start < split < end:
        raise ValueError("Expected history_start < start < is_end < end.")
    lag = settings["signal_lag_sessions"]
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("Execution lag must be at least one full trading session.")
    universe = settings["universe"]
    risk_free_ticker = settings["risk_free_ticker"]
    tickers = [*universe, risk_free_ticker]
    if len(universe) < 2 or len(set(tickers)) != len(tickers) or any(not ticker.isalpha() for ticker in tickers):
        raise ValueError("Use distinct ETF symbols for the sector universe and cash proxy.")
    cases = list(robustness_cases(settings))
    for _, case in [("default", settings), *cases]:
        if (case["lookback_sessions"] < 2 or case["risk_aversion"] <= 0
                or not 0 < case["covariance_shrinkage"] <= 1 or case["turnover_penalty"] < 0):
            raise ValueError("Invalid estimation or quadratic penalty settings.")
        if min(case["cost_bps"], case["slippage_bps"]) < 0 or case["cost_bps"] + case["slippage_bps"] >= 10_000:
            raise ValueError("Combined costs must be nonnegative and below 10,000 bps.")

    prices, fetched_at = adjusted_closes(tickers, settings["history_start"], settings["end"])
    returns = prices.pct_change().iloc[1:]
    sector_returns = returns[universe]
    risk_free_returns = returns[risk_free_ticker]
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, dividend- and split-adjusted total-return closes")

    daily_results = simulate(sector_returns, start, settings)
    benchmark = simulate(sector_returns, start, settings, equal_weight=True)
    is_results = daily_results.loc[:split]
    oos_results = daily_results.loc[daily_results.index > split]
    if min(len(is_results), len(oos_results)) < 2:
        raise ValueError("Both sample periods need at least two realized sessions.")
    data_range = {"start": str(daily_results.index[0].date()), "end": str(daily_results.index[-1].date())}
    results = {
        "strategy": settings["strategy"],
        "data_range": data_range,
        "is": metrics(is_results, risk_free_returns),
        "oos": metrics(oos_results, risk_free_returns),
        "full": metrics(daily_results, risk_free_returns),
        "benchmark": metrics(benchmark, risk_free_returns),
    }
    print(f"Date Range: {data_range['start']} to {data_range['end']}")
    print(f"IS through {is_results.index[-1].date()}, OOS from {oos_results.index[0].date()}")
    print(f"SHARPE_EXCESS_PROXY: {risk_free_ticker} daily total return")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(results[period], allow_nan=False)}")
    print(f"BENCHMARK (monthly equal weight, full): {json.dumps(results['benchmark'], allow_nan=False)}")

    print("ROBUSTNESS (IS only, no parameter selection)")
    for name, case in cases:
        case_results = simulate(sector_returns.loc[:split], start, case)
        print(f"  {name}: {json.dumps(metrics(case_results, risk_free_returns), allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {1 + len(cases)}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


def time_limit_reached(_signum, _frame):
    raise SystemExit(f"Backtest exceeded the {TIME_LIMIT_SECONDS}-second time limit.")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, time_limit_reached)
    signal.alarm(TIME_LIMIT_SECONDS)
    try:
        main()
    finally:
        signal.alarm(0)
