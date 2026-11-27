# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: '450_ a_declining_cvar_glidepath_framework_for'\nuniverse: [SPY, AGG, BIL]\nrisk_free_ticker: BIL\nbenchmark_weights:\n  SPY: 0.60\n  AGG: 0.40\n\nprice_start: 2007-06-01\nbacktest_start: 2013-01-01\nis_end: 2019-12-31\nbacktest_end: 2025-12-31\nexecution_lag_sessions: 1\ncommission_bps: 2.0\nslippage_bps: 3.0\n\nglidepath_start: 1985-12-01\nglidepath_transition: 2018-12-01\nglidepath_target: 2025-12-01\ninitial_cvar_limit: 0.06\nterminal_cvar_limit: 0.03\ncvar_confidence: 0.90\nlookback_months_candidates: [40, 60]\nhit_and_run_burn_in: 20\nrandom_seed: 450\n"
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
"""Historical research simulation of a monthly CVaR glidepath."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, linprog


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252


def load_prices(tickers, start, end):
    cache = ROOT / "data_cache"
    cache.mkdir(exist_ok=True)
    identity = json.dumps([tickers, str(start.date()), str(end.date())])
    key = hashlib.sha256(identity.encode()).hexdigest()[:16]
    path = cache / f"yfinance_adjusted_close_{key}.csv"
    if path.exists():
        with path.open(newline="") as source:
            label, timestamp = next(csv.reader(source))
            if label != "fetched_at":
                raise ValueError(f"Invalid download timestamp in {path}")
            datetime.fromisoformat(timestamp)
            prices = pd.read_csv(source, index_col="date", parse_dates=["date"])
    else:
        with TemporaryDirectory(dir=cache) as yf_cache:
            yf.set_tz_cache_location(yf_cache)
            data = yf.download(
                tickers, start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
                auto_adjust=True, ignore_tz=True, threads=False, progress=False, timeout=12,
            )
        if data.empty or "Close" not in data:
            raise RuntimeError("Yahoo Finance returned no adjusted close data")
        prices = data["Close"].reindex(columns=tickers).sort_index()
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    prices = prices.reindex(columns=tickers)
    if (prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing
            or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any()):
        raise ValueError("Incomplete or invalid daily adjusted close data for the configured ETFs")
    seven_days = pd.Timedelta(days=7)
    if not (start <= prices.index[0] <= start + seven_days
            and end - seven_days <= prices.index[-1] <= end):
        raise ValueError("The ETF price history does not cover the configured dates")

    if not path.exists():
        with NamedTemporaryFile(mode="w", newline="", dir=cache, delete=False) as target:
            temporary_path = Path(target.name)
            csv.writer(target).writerow(["fetched_at", timestamp])
            prices.to_csv(target, index_label="date")
        temporary_path.replace(path)
    return prices, timestamp


def month_end_schedule(prices, start, lag):
    monthly = prices.groupby(prices.index.to_period("M")).tail(1)
    schedule = []
    for signal in monthly.index[:-1]:
        fill_position = prices.index.get_loc(signal) + lag
        if fill_position < len(prices) and prices.index[fill_position] >= start:
            schedule.append((signal, prices.index[fill_position]))
    return monthly.pct_change().dropna(), schedule


def cvar_limit(month, config):
    transition = pd.Period(config["glidepath_transition"], freq="M")
    target = pd.Period(config["glidepath_target"], freq="M")
    progress = np.clip((month.ordinal - transition.ordinal) / (target.ordinal - transition.ordinal), 0, 1)
    initial = config["initial_cvar_limit"]
    return initial + (config["terminal_cvar_limit"] - initial) * progress


def sample_allocation(scenarios, limit, confidence, cash_index, steps, rng):
    """Draw one allocation on the long-only simplex after hit-and-run burn-in."""
    n_scenarios, n_assets = scenarios.shape
    tail_size = max(1.0, round((1 - confidence) * n_scenarios, 12))
    n_tail = int(np.ceil(tail_size))
    tail_weights = np.ones(n_tail)
    tail_weights[-1] = tail_size - (n_tail - 1)
    losses = -scenarios

    def cvar(weights):
        ranked = np.sort(losses @ weights)[::-1]
        return float(ranked[:n_tail] @ tail_weights / tail_size)

    weights = np.full(n_assets, 1.0 / n_assets)
    if cvar(weights) > limit:
        weights = np.eye(n_assets)[cash_index]
    if cvar(weights) > limit:
        # Minimize threshold + average excess loss over the long-only simplex.
        objective = np.r_[np.zeros(n_assets), 1.0, np.full(n_scenarios, 1 / tail_size)]
        fit = linprog(
            objective, A_ub=np.c_[losses, -np.ones(n_scenarios), -np.eye(n_scenarios)],
            b_ub=np.zeros(n_scenarios), A_eq=[np.r_[np.ones(n_assets), np.zeros(n_scenarios + 1)]],
            b_eq=[1.0], bounds=[(0, None)] * n_assets + [(None, None)] + [(0, None)] * n_scenarios,
            method="highs",
        )
        if not fit.success:
            raise RuntimeError(f"Unable to compute minimum CVaR: {fit.message}")
        weights = fit.x[:n_assets]
        if cvar(weights) > limit:
            return np.zeros(n_assets)

    for _ in range(steps):
        direction = rng.normal(size=n_assets)
        direction -= direction.mean()
        positive = direction > 0
        negative = direction < 0
        low = np.max(-weights[positive] / direction[positive])
        high = np.min(-weights[negative] / direction[negative])

        def margin(distance):
            return cvar(weights + distance * direction) - limit

        if low < 0 and margin(low) > 0:
            low = brentq(margin, low, 0)
        if high > 0 and margin(high) > 0:
            high = brentq(margin, 0, high)
        candidate = np.maximum(weights + rng.uniform(low, high) * direction, 0)
        candidate /= candidate.sum()
        if cvar(candidate) <= limit:
            weights = candidate
    return weights


def strategy_orders(monthly_returns, schedule, lookback, config):
    orders = {}
    cash_index = config["universe"].index(config["risk_free_ticker"])
    for signal, fill in schedule:
        scenarios = monthly_returns.loc[:signal].tail(lookback).to_numpy()
        if len(scenarios) != lookback:
            raise ValueError(f"Insufficient monthly history for the {signal.date()} signal")
        month = signal.to_period("M") + 1
        rng = np.random.default_rng([config["random_seed"], month.ordinal])
        orders[fill] = sample_allocation(
            scenarios, cvar_limit(month, config), config["cvar_confidence"], cash_index,
            config["hit_and_run_burn_in"] + 1, rng,
        )
    return orders


def simulate(returns, orders, cost_rate):
    n_assets = returns.shape[1]
    weights = np.zeros(n_assets)
    rows = []
    for date, price_changes in zip(returns.index, returns.to_numpy()):
        # Existing holdings earn the full close-to-close return before any close fill.
        positions = weights * (1 + price_changes)
        gross = 1 + weights @ price_changes
        weights = positions / gross
        turnover = 0.0
        trades = 0
        if date in orders:
            target = orders[date]

            def funding(net):
                return net + cost_rate * np.abs(net * target - weights).sum() - 1

            net = brentq(funding, 0.0, 1.0)
            traded = np.abs(net * target - weights)
            turnover = traded.sum()
            trades = np.count_nonzero(traded > 1e-10)
            gross *= net
            weights = target
        rows.append((gross - 1, turnover, trades))
    return pd.DataFrame(rows, index=returns.index, columns=["return", "turnover", "n_trades"])


def period_metrics(daily_results, risk_free):
    if len(daily_results) < 2:
        raise ValueError("Each period needs at least two return observations")
    returns = daily_results["return"]
    excess = returns - risk_free.loc[returns.index]
    years = len(returns) / SESSIONS_PER_YEAR
    wealth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]

    def sharpe(series):
        volatility = series.std(ddof=1)
        return float(np.sqrt(SESSIONS_PER_YEAR) * series.mean() / volatility) if volatility > 0 else None

    return {
        "cagr": float(wealth[-1] ** (1 / years) - 1),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(excess),
        "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
        "total_return": float(wealth[-1] - 1),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def main():
    with (ROOT / "config.yaml").open() as source:
        config = yaml.safe_load(source)
    tickers = config["universe"]
    price_start, test_start, is_end, test_end = (
        pd.Timestamp(config[key]) for key in ("price_start", "backtest_start", "is_end", "backtest_end")
    )
    if not price_start < test_start < is_end < test_end:
        raise ValueError("Check the configured sample dates")
    entry, transition, target = (
        pd.Period(config[key], freq="M") for key in ("glidepath_start", "glidepath_transition", "glidepath_target")
    )
    if not entry <= test_start.to_period("M") or not entry <= transition < target:
        raise ValueError("Check the configured glidepath dates")
    if not 0 <= config["terminal_cvar_limit"] < config["initial_cvar_limit"]:
        raise ValueError("The monthly CVaR limit must decline to a nonnegative value")
    lag = config["execution_lag_sessions"]
    commission = config["commission_bps"]
    slippage = config["slippage_bps"]
    cost_rate = (commission + slippage) / 10_000
    if not isinstance(lag, int) or lag < 1 or not 0 <= cost_rate < 1 or commission < 0 or slippage < 0:
        raise ValueError("Check the execution lag and fee assumptions")
    if not 0 < config["cvar_confidence"] < 1 or config["hit_and_run_burn_in"] < 0:
        raise ValueError("Check the CVaR sampling settings")
    if len(set(tickers)) != len(tickers) or len(tickers) < 2 or config["risk_free_ticker"] not in tickers:
        raise ValueError("Check the ETF universe and risk-free proxy")
    weights = pd.Series(config["benchmark_weights"], dtype=float)
    if (not weights.index.isin(tickers).all() or not np.isfinite(weights).all()
            or (weights < 0).any() or not np.isclose(weights.sum(), 1)):
        raise ValueError("Benchmark weights must be long-only, on universe ETFs, and sum to one")
    benchmark_weights = weights.reindex(tickers, fill_value=0).to_numpy()

    prices, timestamp = load_prices(tickers, price_start, test_end)
    returns = prices.pct_change().loc[test_start:]
    risk_free = returns[config["risk_free_ticker"]]
    print(f"DATA_FETCHED_AT: {timestamp}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return adjusted close (splits and distributions)")

    is_monthly, is_schedule = month_end_schedule(prices.loc[:is_end], test_start, lag)
    scores = []
    for lookback in config["lookback_months_candidates"]:
        if lookback < 2:
            raise ValueError("The lookback must contain at least two monthly returns")
        orders = strategy_orders(is_monthly, is_schedule, lookback, config)
        daily_results = simulate(returns.loc[:is_end], orders, cost_rate)
        score = period_metrics(daily_results, risk_free)["sharpe_excess"]
        scores.append((lookback, score))
    eligible = [(lookback, score) for lookback, score in scores if score is not None]
    if not eligible:
        raise ValueError("No in-sample candidate has a defined excess Sharpe ratio")
    selected, _ = max(eligible, key=lambda pair: (pair[1], -pair[0]))
    print(f"PARAM_COMBOS_TRIED: {len(scores)}")
    print(f"IS_SELECTED_LOOKBACK_MONTHS: {selected}")

    monthly, schedule = month_end_schedule(prices, test_start, lag)
    orders = strategy_orders(monthly, schedule, selected, config)
    print(f"INFEASIBLE_CVAR_CASH_REBALANCES: {sum(not target.any() for target in orders.values())}")
    strategy_daily_results = simulate(returns, orders, cost_rate)
    benchmark_daily_results = simulate(returns, {fill: benchmark_weights for _, fill in schedule}, cost_rate)
    periods = {"is": returns.index <= is_end, "oos": returns.index > is_end, "full": slice(None)}
    benchmark_metrics = {period: period_metrics(benchmark_daily_results.loc[mask], risk_free) for period, mask in periods.items()}
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": str(returns.index[0].date()), "end": str(returns.index[-1].date())},
    }
    print(f"Benchmark weights {config['benchmark_weights']}")
    for period, mask in periods.items():
        result[period] = period_metrics(strategy_daily_results.loc[mask], risk_free)
        print(f"{period.upper()} strategy {json.dumps(result[period], allow_nan=False)}")
        print(f"{period.upper()} benchmark {json.dumps(benchmark_metrics[period], allow_nan=False)}")
    result["benchmark"] = benchmark_metrics["full"]
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
