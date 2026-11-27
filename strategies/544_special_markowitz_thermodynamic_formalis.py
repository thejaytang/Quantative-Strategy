# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 544_special_markowitz_thermodynamic_formalis\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nrisk_free: BIL\ndata_start: '2007-08-01'\nbacktest_start: '2009-01-01'\nis_end: '2017-12-31'\nend: '2025-12-31'\n\n# Month-end close signal and execution at the close of a later session.\nexecution_lag_sessions: 1\n# Both rates apply to the gross buy and sell amounts.\ncost_bps: 2.0\nslippage_bps: 3.0\n\nmodel:\n  lookback_sessions: 252\n  reference_diagonal_weight: 0.75\n  bulk_persistence: 0.25\n  risk_aversion: 4.0\n  max_weight: 0.35\n\n# These overrides run through is_end only.\nrobustness:\n  shorter_lookback: {lookback_sessions: 210}\n  longer_lookback: {lookback_sessions: 294}\n  lower_bulk_persistence: {bulk_persistence: 0.15}\n  higher_bulk_persistence: {bulk_persistence: 0.35}\n  empirical_moments: {bulk_persistence: 1.0}\n"
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
"""Rolling Special Markowitz research backtest on US sector ETFs."""

import hashlib
import json
import signal
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, minimize


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
TRADE_THRESHOLD = 1e-8


@dataclass(frozen=True)
class Model:
    lookback_sessions: int
    reference_diagonal_weight: float
    bulk_persistence: float
    risk_aversion: float
    max_weight: float

    def validate(self, n_assets):
        if (
            not isinstance(self.lookback_sessions, int)
            or self.lookback_sessions <= n_assets
            or not 0 <= self.reference_diagonal_weight <= 1
            or not 0 < self.bulk_persistence <= 1
            or not (np.isfinite(self.risk_aversion) and self.risk_aversion > 0)
            or not 1 / n_assets <= self.max_weight <= 1
        ):
            raise ValueError(f"Invalid model settings: {self}")


def validate_prices(prices, tickers, start, end):
    if (
        prices.empty
        or list(prices.columns) != tickers
        or prices.index.hasnans
        or prices.index.has_duplicates
        or not prices.index.is_monotonic_increasing
    ):
        raise ValueError("Incomplete or malformed yfinance price panel")
    if not (
        start <= prices.index[0] <= start + pd.Timedelta(days=7)
        and end - pd.Timedelta(days=7) <= prices.index[-1] <= end
    ):
        raise ValueError("Price history does not cover the requested date range")
    invalid = ~np.isfinite(prices) | (prices <= 0)
    if invalid.any().any():
        raise ValueError(f"Missing or invalid adjusted closes: {invalid.sum()[invalid.any()].to_dict()}")


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    end_exclusive = end + pd.Timedelta(days=1)
    request = [tickers, start.date().isoformat(), end_exclusive.date().isoformat(), "auto_adjust"]
    cache_key = hashlib.sha256(json.dumps(request).encode()).hexdigest()[:16]
    cache_path = CACHE / f"adjusted_close_{cache_key}.csv"
    if cache_path.exists():
        with cache_path.open() as saved:
            fetched_at = saved.readline().strip().removeprefix("# fetched_at=")
            prices = pd.read_csv(saved, index_col="Date", parse_dates=["Date"])
        prices.index = prices.index.as_unit("ns")
        if pd.Timestamp(fetched_at).tzinfo is None:
            raise ValueError(f"Missing UTC retrieval timestamp in {cache_path.name}")
        validate_prices(prices, tickers, start, end)
        return prices, fetched_at

    with TemporaryDirectory(dir=CACHE) as staging:
        staging = Path(staging)
        yf.set_tz_cache_location(str(staging / "yfinance"))
        downloaded = yf.download(
            tickers,
            start=request[1],
            end=request[2],
            auto_adjust=True,
            interval="1d",
            group_by="column",
            ignore_tz=True,
            keepna=True,
            progress=False,
            timeout=12,
        )
        fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
        if downloaded.empty or "Close" not in downloaded.columns.get_level_values(0):
            raise RuntimeError("yfinance returned no adjusted closes")
        prices = downloaded["Close"].reindex(columns=tickers).sort_index().rename_axis("Date", axis=0)
        prices.columns.name = None
        prices.index = prices.index.as_unit("ns")
        validate_prices(prices, tickers, start, end)
        # Commit closes and their retrieval timestamp together.
        pending_path = staging / cache_path.name
        with pending_path.open("w", newline="") as pending:
            pending.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(pending)
        pending_path.replace(cache_path)
    return prices, fetched_at


def regularized_moments(window, model, equal_weights):
    empirical_mean = window.mean(axis=0)
    empirical_cov = np.cov(window, rowvar=False, ddof=1)
    reference_mean = equal_weights @ empirical_mean
    diagonal = model.reference_diagonal_weight
    reference_cov = (1 - diagonal) * empirical_cov + diagonal * np.diag(np.diag(empirical_cov))

    ref_eigenvalues, ref_modes = np.linalg.eigh(reference_cov)
    if ref_eigenvalues[0] <= 0:
        raise ValueError("The reference covariance must be positive definite")
    square_root = (ref_modes * np.sqrt(ref_eigenvalues)) @ ref_modes.T
    inverse_root = (ref_modes / np.sqrt(ref_eigenvalues)) @ ref_modes.T
    relative_cov = inverse_root @ empirical_cov @ inverse_root
    eigenvalues, modes = np.linalg.eigh((relative_cov + relative_cov.T) / 2)

    # Use the upper-spike overlap from the spiked covariance model as a heuristic.
    aspect = window.shape[1] / (window.shape[0] - 1)
    above_edge = eigenvalues > (1 + np.sqrt(aspect)) ** 2
    shifted = eigenvalues[above_edge] - 1 - aspect
    spike = (shifted + np.sqrt(np.maximum(shifted ** 2 - 4 * aspect, 0))) / 2
    overlap = np.zeros_like(eigenvalues)
    overlap[above_edge] = np.clip((1 - aspect / spike ** 2) / (1 + aspect / spike), 0, 1)
    persistence = model.bulk_persistence + (1 - model.bulk_persistence) * overlap

    # Apply the same persistence to the return and covariance deviations (Eq. 30).
    basis = square_root @ modes
    return_deviation = modes.T @ inverse_root @ (empirical_mean - reference_mean)
    adjusted_deviation = basis @ (persistence * return_deviation)
    adjusted_eigenvalues = 1 + persistence * (eigenvalues - 1)
    if np.any(adjusted_eigenvalues <= 0):
        raise ValueError("The regularized covariance must be positive definite")
    adjusted_cov = (basis * adjusted_eigenvalues) @ basis.T
    return adjusted_deviation, adjusted_cov


def target_weights(window, model, equal_weights):
    return_deviation, covariance = regularized_moments(window, model, equal_weights)
    reward = SESSIONS_PER_YEAR * return_deviation
    penalty = (SESSIONS_PER_YEAR * model.risk_aversion) * covariance

    def objective(weights):
        active = weights - equal_weights
        return 0.5 * (active @ penalty @ active) - reward @ active

    def gradient(weights):
        return penalty @ (weights - equal_weights) - reward

    solution = minimize(
        objective,
        equal_weights,
        method="SLSQP",
        jac=gradient,
        bounds=[(0, model.max_weight)] * len(equal_weights),
        constraints={
            "type": "eq",
            "fun": lambda weights: weights.sum() - 1,
            "jac": lambda weights: np.ones_like(weights),
        },
        options={"ftol": 1e-12, "maxiter": 200},
    )
    if not solution.success or not np.isfinite(solution.x).all():
        raise RuntimeError(f"Weight optimization failed: {solution.message}")
    weights = np.clip(solution.x, 0, model.max_weight)
    return weights / weights.sum()


def execution_schedule(index, first, lag):
    months = index.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])
    return {day + lag: day for day in month_ends if first <= day + lag < len(index)}


def simulate(returns, first, schedule, friction, equal_weights, model=None):
    values = returns.to_numpy()
    weights = np.zeros_like(equal_weights)
    rows = []
    for day in range(first, len(returns)):
        ending = weights * (1 + values[day])
        before_trade = (1 - weights.sum()) + ending.sum()
        weights = ending / before_trade
        gross_traded = 0.0
        trades = 0
        after_cost = 1.0
        if day in schedule:
            target = equal_weights
            if model is not None:
                last = schedule[day] + 1
                window = values[last - model.lookback_sessions:last]
                target = target_weights(window, model, equal_weights)

            # The new target is expressed relative to value after both costs.
            def budget(fraction):
                return fraction + friction * np.abs(fraction * target - weights).sum() - 1

            after_cost = brentq(budget, 0, 1, xtol=1e-14)
            changes = np.abs(after_cost * target - weights)
            gross_traded = changes.sum()
            trades = np.count_nonzero(changes > TRADE_THRESHOLD)
            weights = target
        rows.append((before_trade * after_cost - 1, gross_traded, trades))
    return pd.DataFrame(rows, index=returns.index[first:], columns=["return", "turnover", "n_trades"])


def metrics(daily_results, risk_free):
    daily = daily_results["return"].to_numpy()
    excess = daily - risk_free.loc[daily_results.index].to_numpy()
    if len(daily) < 2 or not np.isfinite(excess).all():
        raise ValueError("Each period needs at least two observations with a BIL return")

    def sharpe(values):
        volatility = values.std(ddof=1)
        return float(np.sqrt(SESSIONS_PER_YEAR) * values.mean() / volatility) if volatility > 0 else None

    growth = np.log1p(daily).cumsum()
    peak = np.maximum.accumulate(np.maximum(growth, 0))
    return {
        "cagr": float(np.expm1(growth[-1] * SESSIONS_PER_YEAR / len(daily))),
        "ann_vol": float(daily.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": sharpe(daily),
        "sharpe_excess": sharpe(excess),
        "max_drawdown": float(np.expm1(growth - peak).min()),
        "total_return": float(np.expm1(growth[-1])),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, risk_free, is_end):
    return {
        "is": metrics(daily_results.loc[daily_results.index <= is_end], risk_free),
        "oos": metrics(daily_results.loc[daily_results.index > is_end], risk_free),
        "full": metrics(daily_results, risk_free),
    }


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    universe = cfg["universe"]
    risk_ticker = cfg["risk_free"]
    tickers = [*universe, risk_ticker]
    if risk_ticker != "BIL" or len(universe) < 2 or len(set(tickers)) != len(tickers):
        raise ValueError("Use distinct sector symbols and BIL as the risk-free series")

    data_start, backtest_start, is_end, end = (
        pd.Timestamp(cfg[key]) for key in ("data_start", "backtest_start", "is_end", "end")
    )
    lag = cfg["execution_lag_sessions"]
    cost, slippage = cfg["cost_bps"], cfg["slippage_bps"]
    if not data_start < backtest_start <= is_end < end:
        raise ValueError("Dates must satisfy data_start < backtest_start <= is_end < end")
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("execution_lag_sessions must be a positive integer")
    if not (cost >= 0 and slippage >= 0 and cost + slippage < 5000):
        raise ValueError("Invalid cost or slippage rates")
    friction = (cost + slippage) / 10000

    official = Model(**cfg["model"])
    variants = {label: replace(official, **overrides) for label, overrides in cfg["robustness"].items()}
    if len({official, *variants.values()}) != len(variants) + 1:
        raise ValueError("Each robustness setting must be distinct")
    models = (official, *variants.values())
    for model in models:
        model.validate(len(universe))
    equal_weights = np.full(len(universe), 1 / len(universe))

    prices, fetched_at = load_prices(tickers, data_start, end)
    returns = prices.pct_change(fill_method=None)
    sector_returns = returns[universe]
    risk_free = returns[risk_ticker]
    is_returns = sector_returns.loc[:is_end]
    first = int(returns.index.searchsorted(backtest_start))
    is_schedule = execution_schedule(is_returns.index, first, lag)
    max_lookback = max(model.lookback_sessions for model in models)
    if not is_schedule or min(is_schedule.values()) < max_lookback:
        raise ValueError("Insufficient warm-up before the first in-sample rebalance")

    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return closes", flush=True)
    print("ROBUSTNESS (IS only)", flush=True)
    for label, model in variants.items():
        daily_results = simulate(is_returns, first, is_schedule, friction, equal_weights, model)
        print(f"  {label} {json.dumps(cfg['robustness'][label])} "
              f"{json.dumps(metrics(daily_results, risk_free), allow_nan=False)}", flush=True)

    schedule = execution_schedule(sector_returns.index, first, lag)
    daily_results = simulate(sector_returns, first, schedule, friction, equal_weights, official)
    equal_results = simulate(sector_returns, first, schedule, friction, equal_weights)
    official_periods = period_metrics(daily_results, risk_free, is_end)
    benchmark = {"name": "Monthly equal weight of the same sector ETFs",
                 **period_metrics(equal_results, risk_free, is_end)}
    data_range = {
        "start": daily_results.index[0].date().isoformat(),
        "end": daily_results.index[-1].date().isoformat(),
    }
    print(f"PARAM_COMBOS_TRIED: {1 + len(variants)}")
    for period, values in official_periods.items():
        print(f"{period.upper()} STRATEGY: {json.dumps(values, allow_nan=False)}")
        print(f"{period.upper()} BENCHMARK: {json.dumps(benchmark[period], allow_nan=False)}")
    result = {"strategy": cfg["strategy"], "data_range": data_range, **official_periods,
              "benchmark": benchmark}
    print(f"RESULTS_JSON: {json.dumps(result, allow_nan=False)}")


def stop_on_timeout(_signum, _frame):
    raise SystemExit("Backtest reached its 14-minute runtime limit")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_on_timeout)
    signal.alarm(840)
    main()
