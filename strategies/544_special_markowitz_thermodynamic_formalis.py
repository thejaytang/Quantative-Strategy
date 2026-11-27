# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 544_special_markowitz_thermodynamic_formalis\n# Nine Select Sector SPDR ETFs in the Philadelphia Stock Exchange's January 2007 filing.\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nrisk_free: BIL\ndata_start: '2007-08-01'\nbacktest_start: '2009-01-01'\nis_end: '2017-12-31'\nend: '2025-12-31'\n\n# Month-end close signal. A one-session lag fills at the next close.\n# The strategy and benchmark lock share changes using the close before each fill.\nexecution_lag_sessions: 1\n# Both portfolios reserve 5% of prior-close value when sizing shares.\n# At the fill close, settle fixed quantities and fees together. Stop on a cash shortfall.\n# The residual can differ from 5% at execution and earns zero interest.\ncash_buffer_fraction: 0.05\n# Both rates apply to the gross buy and sell amounts.\ncost_bps: 2.0\nslippage_bps: 3.0\n\nmodel:\n  lookback_sessions: 252\n  reference_diagonal_weight: 0.75\n  bulk_persistence: 0.25\n  risk_aversion: 4.0\n  max_weight: 0.35\n\n# These overrides run through is_end only and do not select the main model.\nrobustness:\n  shorter_lookback: {lookback_sessions: 210}\n  longer_lookback: {lookback_sessions: 294}\n  lower_bulk_persistence: {bulk_persistence: 0.15}\n  higher_bulk_persistence: {bulk_persistence: 0.35}\n  empirical_moments: {bulk_persistence: 1.0}\n"
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
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from scipy.optimize import minimize


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


def nyse_sessions(start, end):
    if not 2007 <= start.year <= end.year <= 2025:
        raise ValueError("The NYSE closure calendar covers 2007–2025")

    holidays = AbstractHolidayCalendar(rules=[
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]).holidays(start, end)
    # Full-day closures for Ford, Hurricane Sandy, Bush, and Carter.
    special_closures = pd.to_datetime([
        "2007-01-02", "2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09",
    ])
    return pd.date_range(start, end, freq="B").difference(holidays.union(special_closures))


def validate_prices(prices, tickers, sessions):
    if (
        prices.empty
        or list(prices.columns) != tickers
        or prices.index.hasnans
        or prices.index.has_duplicates
        or not prices.index.is_monotonic_increasing
    ):
        raise ValueError("Incomplete or malformed yfinance price panel")
    if not prices.index.equals(sessions):
        missing = sessions.difference(prices.index).strftime("%Y-%m-%d").tolist()
        unexpected = prices.index.difference(sessions).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"NYSE session mismatch: missing={missing}, unexpected={unexpected}")
    invalid = ~np.isfinite(prices) | (prices <= 0)
    if invalid.any().any():
        raise ValueError(f"Missing or invalid adjusted closes: {invalid.sum()[invalid.any()].to_dict()}")


def load_prices(tickers, start, end, sessions):
    CACHE.mkdir(exist_ok=True)
    end_exclusive = end + pd.Timedelta(days=1)
    request = [tickers, start.date().isoformat(), end_exclusive.date().isoformat(), "auto_adjust"]
    cache_key = hashlib.sha256(json.dumps(request).encode()).hexdigest()[:16]
    cache_path = CACHE / f"adjusted_close_{cache_key}.csv"
    if cache_path.exists():
        contents = cache_path.read_bytes()
        with BytesIO(contents) as saved:
            fetched_at = saved.readline().decode().strip().removeprefix("# fetched_at=")
            prices = pd.read_csv(saved, index_col="Date", parse_dates=["Date"])
        prices.index = prices.index.as_unit("ns")
        if pd.Timestamp(fetched_at).tzinfo is None:
            raise ValueError(f"Missing UTC retrieval timestamp in {cache_path.name}")
        validate_prices(prices, tickers, sessions)
        return prices, fetched_at, hashlib.sha256(contents).hexdigest()

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
        validate_prices(prices, tickers, sessions)
        # Commit closes and their retrieval timestamp together.
        pending_path = staging / cache_path.name
        with pending_path.open("w", newline="") as pending:
            pending.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(pending)
        digest = hashlib.sha256(pending_path.read_bytes()).hexdigest()
        pending_path.replace(cache_path)
    return prices, fetched_at, digest


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
    """Map fill sessions to month-end signal sessions on the market calendar."""
    if type(lag) is not int or lag < 1:
        raise ValueError("execution_lag_sessions must be a positive integer")
    months = index.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])
    return {day + lag: day for day in month_ends if first <= day + lag < len(index)}


def size_position_changes(positions, target, prior_close, previous_value, cash_buffer):
    """Lock the next fill's share changes using information at this close."""
    target_positions = previous_value * (1 - cash_buffer) * target / prior_close
    changes = target_positions - positions
    changes[np.abs(changes) * prior_close <= TRADE_THRESHOLD * previous_value] = 0
    changes.setflags(write=False)
    return changes


def settle_position_changes(positions, cash, planned_changes, fill_close, friction, fill_date):
    """Return cash and turnover for the fixed changes, or stop before applying them."""
    before_trade = cash + positions @ fill_close
    notionals = planned_changes * fill_close
    gross_traded = np.abs(notionals).sum()
    cash_after_fill = cash - notionals.sum() - gross_traded * friction
    if not np.isfinite(cash_after_fill):
        raise RuntimeError(f"{fill_date}: nonfinite cash after fixed-size fills")
    if cash_after_fill < 0:
        raise RuntimeError(f"{fill_date}: insufficient cash for fixed-size fills "
                           f"(shortfall/value: {-cash_after_fill / before_trade:.2%})")
    return cash_after_fill, gross_traded / before_trade


def simulate(prices, first, schedule, friction, cash_buffer, equal_weights, model=None):
    closes = prices.to_numpy()
    positions = np.zeros_like(equal_weights)
    cash = previous_value = 1.0
    planned_changes = None
    rows = []
    # Include the prior close so the first backtest session can fill a pre-sized order.
    for day in range(max(first - 1, 0), len(prices)):
        close = closes[day]
        turnover = 0.0
        trades = 0
        if planned_changes is not None:
            cash, turnover = settle_position_changes(
                positions, cash, planned_changes, close, friction, prices.index[day].date(),
            )
            positions += planned_changes
            trades = np.count_nonzero(planned_changes)

        value = cash + positions @ close
        if day >= first:
            rows.append((value / previous_value - 1, turnover, trades))
        previous_value = value

        # This session's close can size the next session's fill only.
        planned_changes = None
        signal_day = schedule.get(day + 1)
        if signal_day is not None:
            target = equal_weights
            if model is not None:
                first_close = signal_day - model.lookback_sessions
                signal_closes = closes[first_close:signal_day + 1]
                signal_returns = signal_closes[1:] / signal_closes[:-1] - 1
                target = target_weights(signal_returns, model, equal_weights)
            planned_changes = size_position_changes(positions, target, close, value, cash_buffer)

    return pd.DataFrame(rows, index=prices.index[first:], columns=["return", "turnover", "n_trades"])


def metrics(daily_results, risk_free):
    if daily_results.empty:
        return dict.fromkeys(("cagr", "ann_vol", "sharpe", "sharpe_excess", "max_drawdown",
                              "total_return", "turnover")) | {"n_trades": 0}

    daily = daily_results["return"].to_numpy()
    excess = daily - risk_free.loc[daily_results.index].to_numpy()
    if not np.isfinite(excess).all():
        raise ValueError("Every observation needs finite strategy and BIL returns")

    def sharpe(values):
        if len(values) < 2:
            return None
        volatility = values.std(ddof=1)
        return float(np.sqrt(SESSIONS_PER_YEAR) * values.mean() / volatility) if volatility > 0 else None

    growth = np.log1p(daily).cumsum()
    peak = np.maximum.accumulate(np.maximum(growth, 0))
    return {
        "cagr": float(np.expm1(growth[-1] * SESSIONS_PER_YEAR / len(daily))),
        "ann_vol": float(daily.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)) if len(daily) > 1 else None,
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
    cash_buffer = cfg["cash_buffer_fraction"]
    cost, slippage = cfg["cost_bps"], cfg["slippage_bps"]
    if not data_start < backtest_start <= is_end < end:
        raise ValueError("Dates must satisfy data_start < backtest_start <= is_end < end")
    if not 0 < cash_buffer < 1:
        raise ValueError("cash_buffer_fraction must be between zero and one")
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

    sessions = nyse_sessions(data_start, end)
    first = int(sessions.searchsorted(backtest_start))
    if first == 0:
        raise ValueError("The backtest needs a prior close before its first session")
    if first == len(sessions):
        raise ValueError("The backtest dates contain no trading sessions")
    schedule = execution_schedule(sessions, first, lag)
    prices, fetched_at, digest = load_prices(tickers, data_start, end, sessions)
    sector_prices = prices[universe]
    risk_free = prices[risk_ticker].pct_change(fill_method=None)
    is_stop = sessions.searchsorted(is_end, side="right")
    is_prices = sector_prices.iloc[:is_stop]
    is_schedule = {day: signal_day for day, signal_day in schedule.items() if day < is_stop}
    active_models = models if is_schedule else (official,)
    max_lookback = max(model.lookback_sessions for model in active_models)
    if schedule and min(schedule.values()) < max_lookback:
        raise ValueError("Insufficient warm-up before the first scheduled rebalance")

    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print(f"DATA_SHA256: {digest}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return closes", flush=True)
    print("ROBUSTNESS (IS only)", flush=True)
    for label, model in variants.items():
        daily_results = simulate(is_prices, first, is_schedule, friction, cash_buffer, equal_weights, model)
        print(f"  {label} {json.dumps(cfg['robustness'][label])} "
              f"{json.dumps(metrics(daily_results, risk_free), allow_nan=False)}", flush=True)

    daily_results = simulate(sector_prices, first, schedule, friction, cash_buffer, equal_weights, official)
    equal_results = simulate(sector_prices, first, schedule, friction, cash_buffer, equal_weights)
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
    raise SystemExit("Backtest reached its 9-minute runtime limit")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_on_timeout)
    signal.alarm(540)
    main()
