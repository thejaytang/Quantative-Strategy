# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 653_risk_budgeting_portfolios_from_simulatio\nuniverse: [SPY, IWM, EFA, TLT, DBC, GLD]\nrisk_free_proxy: BIL\ndata_start: 2008-01-01\nbacktest_start: 2011-01-03\nis_end: 2018-12-31\nbacktest_end: 2025-12-31\nrebalance_lag_sessions: 1\ncost_bps: 5\nslippage_bps: 5\ndistribution_cash_lag_sessions: 30\ninitial_capital: 100000\nmodel:\n  lookback: 500\n  horizon: 5\n  scenarios: 1000\n  confidence: 0.95\n  ewma_decay: 0.97\n  variance_seed_days: 20\n  seed: 653\nrobustness:\n  lookbacks: [400, 600]\n  confidence_levels: [0.925, 0.975]\n  friction_multiplier: 2\n'
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
"""Monthly simulated Expected Shortfall risk budgeting on cross-asset ETFs.

Paper: https://arxiv.org/abs/2302.01196
Research references for the independent Python implementation:
SamplingRB.jl (GPL-3.0-or-later): https://github.com/bfpc/SamplingRB.jl
Paper-era redirect: https://github.com/bfpc/RiskBudgeting.jl
Julia ES solver: https://github.com/bfpc/SamplingRB.jl/blob/main/src/cvar_cp.jl
Julia example: https://github.com/bfpc/SamplingRB.jl/blob/main/examples/calculate.jl
skfolio (BSD-3-Clause): https://github.com/skfolio/skfolio
Python solver: https://github.com/skfolio/skfolio/blob/main/src/skfolio/optimization/convex/_risk_budgeting.py
CVaR tutorial: https://skfolio.org/auto_examples/risk_budgeting/plot_2_risk_budgeting_CVaR.html
Python distribution: https://pypi.org/project/skfolio/
Related project: https://github.com/dcajasn/Riskfolio-Lib (BSD-3-Clause)
LEAN variance comparison (Apache-2.0):
https://github.com/QuantConnect/Lean/blob/master/Algorithm.Framework/Portfolio/RiskParityPortfolioOptimizer.py
"""

import hashlib
import json
import tempfile
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
PRICE_FIELDS = ("Open", "Close", "Distribution", "Stock Splits")
PRICE_ADJUSTMENT = (
    "Yahoo Finance split-adjusted OHLC (auto_adjust=False); ex-date cash "
    "distributions for total-return signals and receivables for performance"
)


@dataclass(frozen=True)
class Model:
    lookback: int
    horizon: int
    scenarios: int
    confidence: float
    ewma_decay: float
    variance_seed_days: int
    seed: int


def load_prices(tickers, start, end):
    request_key = json.dumps(["split_ohlc_actions_v1", tickers, str(start), str(end)])
    key = hashlib.sha256(request_key.encode()).hexdigest()[:16]
    path = CACHE / f"yahoo_{key}.csv"
    CACHE.mkdir(exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            fetched_at = stream.readline().strip()
            prices = pd.read_csv(stream, header=[0, 1], index_col=0, parse_dates=True)
        datetime.fromisoformat(fetched_at)
    else:
        series = {}
        with tempfile.TemporaryDirectory(dir=CACHE) as yf_cache:
            yf.set_tz_cache_location(yf_cache)
            for ticker in tickers:
                history = yf.Ticker(ticker).history(
                    start=str(start.date()),
                    end=str((end + pd.Timedelta(days=1)).date()),
                    auto_adjust=False,
                    actions=True,
                    timeout=15,
                )
                if history.empty:
                    raise RuntimeError(f"Yahoo Finance returned no history for {ticker}")
                history.index = history.index.tz_localize(None).normalize()
                gains = history.get("Capital Gains", 0.0)
                history["Distribution"] = history["Dividends"] + gains
                series[ticker] = history.loc[:, list(PRICE_FIELDS)]
        prices = pd.concat(series, axis=1).sort_index()
        check_prices(prices, tickers, start, end)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=CACHE, suffix=".partial", delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(fetched_at + "\n")
                prices.to_csv(stream)
                stream.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)

    check_prices(prices, tickers, start, end)
    return prices, fetched_at


def check_prices(prices, tickers, start, end):
    required = pd.MultiIndex.from_product([tickers, PRICE_FIELDS])
    if prices.empty or not prices.columns.equals(required):
        raise ValueError("Yahoo Finance data is empty or has unexpected columns")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Expected unique, ascending trading-session dates")
    if not (start <= prices.index[0] <= start + pd.Timedelta(days=5)):
        raise ValueError(f"Incomplete data at the beginning: {prices.index[0].date()}")
    if not (end - pd.Timedelta(days=5) <= prices.index[-1] <= end):
        raise ValueError(f"Incomplete data at the end: {prices.index[-1].date()}")
    if not np.isfinite(prices.to_numpy()).all():
        missing = prices.index[prices.isna().any(axis=1)]
        raise ValueError(f"Incomplete synchronized Yahoo Finance bars: {missing[:10].tolist()}")
    ohlc = prices.loc[:, (slice(None), ["Open", "Close"])]
    actions = prices.loc[:, (slice(None), ["Distribution", "Stock Splits"])]
    if (ohlc <= 0).any().any() or (actions < 0).any().any():
        raise ValueError("Invalid Yahoo Finance prices or corporate actions")


def scenario_losses(log_returns, model, signal_date):
    """Resample synchronous blocks of causally standardized returns."""
    seed = np.random.SeedSequence([model.seed, signal_date.toordinal()])
    rng = np.random.default_rng(seed)
    history = np.ascontiguousarray(log_returns[-model.lookback :])
    variance = np.mean(history[: model.variance_seed_days] ** 2, axis=0)
    residuals = np.empty_like(history[model.variance_seed_days :])
    for i, observation in enumerate(history[model.variance_seed_days :]):
        residuals[i] = observation / np.sqrt(np.maximum(variance, 1e-12))
        variance = model.ewma_decay * variance + (1 - model.ewma_decay) * observation**2

    # A zero conditional log drift keeps the allocation focused on risk.
    residuals -= residuals.mean(axis=0)
    cumulative = np.vstack([np.zeros(history.shape[1]), np.cumsum(residuals, axis=0)])
    blocks = cumulative[model.horizon :] - cumulative[: -model.horizon]
    starts = (rng.random(model.scenarios) * len(blocks)).astype(int)
    future_log_returns = blocks[starts] * np.sqrt(variance)
    return -np.expm1(future_log_returns)


def tail_gradient(losses, exposures, size):
    """One valid ES subgradient, including fractional mass at the VaR boundary."""
    portfolio_losses = losses @ exposures
    count = int(np.ceil(size))
    tail = np.argpartition(portfolio_losses, -count)[-count:]
    boundary = tail[np.argmin(portfolio_losses[tail])]
    return (losses[tail].sum(axis=0) - (count - size) * losses[boundary]) / size


def risk_budget_weights(losses, confidence):
    """Minimize empirical ES minus the equally weighted log-exposure penalty."""
    losses = losses * 100.0  # Express losses in percentage points for the solver.
    n_assets = losses.shape[1]
    budget = np.full(n_assets, 1.0 / n_assets)
    size = round((1 - confidence) * len(losses), 10)
    cuts = [tail_gradient(losses, vertex, size) for vertex in np.eye(n_assets)]
    standalone_es = np.diag(cuts)
    if np.any(standalone_es <= 0):
        raise RuntimeError("Need positive standalone ES to form positive risk budgets")
    exposures = budget / standalone_es
    cuts.append(tail_gradient(losses, exposures, size))
    exposure_bounds = [(1e-8, 100.0)] * n_assets + [(None, None)]
    gap = parity_error = float("inf")

    def objective(point):
        return point[-1] - budget @ np.log(point[:-1])

    def gradient(point):
        return np.r_[-budget / point[:-1], 1.0]

    for _ in range(80):
        planes = np.column_stack((-np.asarray(cuts), np.ones(len(cuts))))
        initial = np.r_[exposures, np.max(-planes[:, :-1] @ exposures)]
        result = minimize(
            objective,
            initial,
            jac=gradient,
            bounds=exposure_bounds,
            constraints=[{
                "type": "ineq",
                "fun": lambda point: planes @ point,
                "jac": lambda point: planes,
            }],
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 150},
        )
        exposures = result.x[:-1]
        new_cut = tail_gradient(losses, exposures, size)
        primal = new_cut @ exposures - budget @ np.log(exposures)
        # Any convex combination of cuts with positive entries gives an
        # independent lower bound, even if the numerical solver stalls.
        multipliers = np.maximum(result.multipliers, 0.0)
        if multipliers.sum() > 0:
            stress = multipliers @ np.asarray(cuts) / multipliers.sum()
            if np.all(stress > 0):
                dual = 1.0 + budget @ np.log(stress / budget)
                gap = max(0.0, primal - dual)
                contributions = exposures * stress
                parity_error = np.max(np.abs(contributions / contributions.sum() - budget))
                if gap <= 1e-6 and parity_error <= 5e-4:
                    return exposures / exposures.sum(), parity_error, gap
        cuts.append(new_cut)

    raise RuntimeError(f"ES solution missed the tolerances (gap={gap:g}, budget error={parity_error:g})")


def monthly_signals(dates, first, last, lag):
    months = dates.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])
    return {
        signal + lag: signal
        for signal in month_ends
        if first <= signal + lag <= last
    }


def compute_targets(log_returns, dates, schedule, model):
    targets = {}
    max_parity_error = max_gap = 0.0
    for fill, signal in schedule.items():
        if not 0 <= signal < fill or signal < model.lookback:
            raise ValueError("Signal timing or rolling price history is insufficient")
        lookback = log_returns[signal - model.lookback + 1 : signal + 1]
        losses = scenario_losses(lookback, model, dates[signal])
        try:
            weights, parity_error, gap = risk_budget_weights(losses, model.confidence)
        except RuntimeError as exc:
            raise RuntimeError(f"Signal {dates[signal].date()}: {exc}") from exc
        targets[fill] = weights
        max_parity_error = max(max_parity_error, parity_error)
        max_gap = max(max_gap, gap)
    return targets, {"max_budget_error": max_parity_error, "max_objective_gap": max_gap}


def rebalance_at_open(positions, cash, opening, weights, friction):
    """Size fractional units from observed opening prices and net sale proceeds."""
    old_values = positions * opening
    opening_value = cash + old_values.sum()
    # Find the largest affordable allocation with the requested weight ratios.
    lower, upper = 0.0, opening_value
    for _ in range(48):
        allocation = (lower + upper) / 2
        trade_value = np.abs(allocation * weights - old_values).sum()
        if allocation + friction * trade_value <= opening_value:
            lower = allocation
        else:
            upper = allocation

    new_values = lower * weights
    trades = new_values - old_values
    sold = -trades[trades < 0].sum()
    bought = trades[trades > 0].sum()
    cash = cash + sold * (1 - friction) - bought * (1 + friction)
    if cash < -1e-12 * opening_value:
        raise RuntimeError("Rebalance generated negative cash")
    return (
        new_values / opening,
        max(0.0, cash),
        (sold + bought) / opening_value,
        int(np.count_nonzero(np.abs(trades) > 1e-12 * opening_value)),
    )


def simulate(opens, closes, distributions, dates, first, last, targets, config, friction):
    cash = previous_value = float(config["initial_capital"])
    positions = np.zeros(opens.shape[1])
    pending = deque()
    receivables = 0.0
    cash_lag = config["distribution_cash_lag_sessions"]
    records = []

    for day in range(first, last + 1):
        # Only positions carried into the ex-date earn the distribution.
        distribution = float(distributions[day] @ positions)
        if distribution > 0:
            pending.append((day + cash_lag, distribution))
            receivables += distribution
        while pending and pending[0][0] <= day:
            _, payment = pending.popleft()
            receivables -= payment
            cash += payment

        turnover = trades = 0
        if day in targets:
            positions, cash, turnover, trades = rebalance_at_open(
                positions, cash, opens[day], targets[day], friction
            )
        closing_value = cash + receivables + positions @ closes[day]
        records.append((closing_value / previous_value - 1, turnover, trades))
        previous_value = closing_value

    return pd.DataFrame(
        records,
        index=dates[first : last + 1],
        columns=["return", "turnover", "n_trades"],
    )


def metrics(daily_results, cash_returns):
    returns = daily_results["return"]
    excess = returns - cash_returns.loc[returns.index]
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    growth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(volatility * annualizer),
        "sharpe": float(annualizer * returns.mean() / volatility) if volatility else None,
        "sharpe_excess": (
            float(annualizer * excess.mean() / excess_volatility) if excess_volatility else None
        ),
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1)),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, cash_returns, split_date):
    return {
        "is": metrics(daily_results.loc[:split_date], cash_returns),
        "oos": metrics(daily_results.loc[daily_results.index > split_date], cash_returns),
        "full": metrics(daily_results, cash_returns),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    model = Model(**config["model"])
    universe = config["universe"]
    proxy = config["risk_free_proxy"]
    data_start = pd.Timestamp(config["data_start"])
    backtest_start = pd.Timestamp(config["backtest_start"])
    split_date = pd.Timestamp(config["is_end"])
    backtest_end = pd.Timestamp(config["backtest_end"])
    lag = config["rebalance_lag_sessions"]
    cases = config["robustness"]
    friction = (config["cost_bps"] + config["slippage_bps"]) / 10_000
    cost_multiple = cases["friction_multiplier"]
    if not (
        data_start < backtest_start < split_date < backtest_end
        and lag >= 1 and config["distribution_cash_lag_sessions"] >= 1
        and config["initial_capital"] > 0 and len(set(universe)) == len(universe) >= 2
        and proxy not in universe and min(config["cost_bps"], config["slippage_bps"]) >= 0
        and cost_multiple > 1 and friction * cost_multiple < 1
    ):
        raise ValueError("Invalid sample, universe, execution, or friction settings")

    alternatives = [
        (f"lookback_{lookback}", replace(model, lookback=lookback))
        for lookback in cases["lookbacks"]
    ] + [
        (f"confidence_{confidence}", replace(model, confidence=confidence))
        for confidence in cases["confidence_levels"]
    ]
    for specification in (model, *(case for _, case in alternatives)):
        if not (
            0 < specification.confidence < 1 and 0 < specification.ewma_decay < 1
            and specification.scenarios >= 10 and specification.horizon >= 1
            and specification.variance_seed_days >= 2
            and specification.lookback >= specification.variance_seed_days + specification.horizon
        ):
            raise ValueError(f"Invalid scenario model: {specification}")

    prices, fetched_at = load_prices([*universe, proxy], data_start, backtest_end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"EXCESS_SHARPE_PROXY: {proxy} daily total return")
    dates = prices.index
    first = int(dates.searchsorted(backtest_start))
    last = len(dates) - 1
    is_last = int(dates.searchsorted(split_date, side="right")) - 1
    if first >= is_last or is_last >= last - 1:
        raise ValueError("Both periods must have at least two trading sessions")

    close = prices.xs("Close", axis=1, level=1)
    distribution = prices.xs("Distribution", axis=1, level=1)
    gross = (close + distribution) / close.shift()
    log_returns = np.log(gross[universe].to_numpy())
    cash_returns = gross[proxy] - 1
    opens = prices.xs("Open", axis=1, level=1)[universe].to_numpy()
    closes = close[universe].to_numpy()
    distributions = distribution[universe].to_numpy()

    schedule = monthly_signals(dates, first, last, lag)
    is_schedule = {fill: signal for fill, signal in schedule.items() if fill <= is_last}
    if not is_schedule or len(is_schedule) == len(schedule):
        raise ValueError("The monthly schedule must generate fills in both periods")
    print("ROBUSTNESS (IS only)")
    is_targets, is_diagnostics = compute_targets(log_returns, dates, is_schedule, model)
    for name, specification in alternatives:
        trial_targets, _ = compute_targets(log_returns, dates, is_schedule, specification)
        trial_results = simulate(
            opens, closes, distributions, dates, first, is_last, trial_targets, config, friction
        )
        print(f"ROBUSTNESS_IS {name}: " + json.dumps(metrics(trial_results, cash_returns)))
    stress_results = simulate(
        opens, closes, distributions, dates, first, is_last, is_targets, config,
        friction * cost_multiple,
    )
    print(f"ROBUSTNESS_IS friction_x{cost_multiple}: "
          + json.dumps(metrics(stress_results, cash_returns)))
    print(f"PARAM_COMBOS_TRIED: {len(alternatives) + 2}")

    oos_schedule = {fill: signal for fill, signal in schedule.items() if fill > is_last}
    oos_targets, oos_diagnostics = compute_targets(log_returns, dates, oos_schedule, model)
    print("ES_SOLVER_IS: " + json.dumps(is_diagnostics))
    print("ES_SOLVER_OOS: " + json.dumps(oos_diagnostics))

    daily_results = simulate(
        opens, closes, distributions, dates, first, last, is_targets | oos_targets, config, friction
    )
    equal_weights = np.full(len(universe), 1.0 / len(universe))
    benchmark_results = simulate(
        opens, closes, distributions, dates, first, last,
        dict.fromkeys(schedule, equal_weights), config, friction,
    )
    stats = period_metrics(daily_results, cash_returns, split_date)
    benchmark = period_metrics(benchmark_results, cash_returns, split_date)
    if any(stats[period]["n_trades"] <= 0 for period in ("is", "oos")):
        raise RuntimeError("The official specification must generate trades in both periods")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: " + json.dumps(stats[period], allow_nan=False))
        print(f"BENCHMARK_EQUAL_WEIGHT_{period.upper()}: "
              + json.dumps(benchmark[period], allow_nan=False))
    result = {
        "strategy": config["strategy"],
        "data_range": {
            "price_start": str(dates[0].date()),
            "start": str(dates[first].date()),
            "is_end": str(dates[is_last].date()),
            "oos_start": str(dates[is_last + 1].date()),
            "end": str(dates[last].date()),
        },
        **stats,
        "benchmark": {"name": "monthly_equal_weight", **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
