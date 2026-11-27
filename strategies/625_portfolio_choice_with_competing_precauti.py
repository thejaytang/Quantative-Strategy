# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 625_portfolio_choice_with_competing_precauti\n\nuniverse:\n  equity: SPY\n  defensive: SHY\n  cash_proxy: BIL\n\ndates:\n  warmup_start: 2007-06-01\n  start: 2008-01-01\n  out_of_sample_start: 2016-01-01\n  end: 2025-12-31\n  goal_deadline: 2026-01-01\n\nmodel:\n  covariance_lookback: 126\n  equity_excess_drift: 0.05\n  defensive_excess_drift: 0.0\n  reserve: 0.60\n  goal_median: 1.65\n  goal_log_dispersion: 0.50\n  annual_shock_intensity: 0.20\n  accumulation_priority: 0.50\n  weight_grid_size: 11\n  quadrature_nodes: 24\n\nexecution:\n  lag_sessions: 1\n  cost_bps: 2\n  slippage_bps: 3\n  benchmark_equity_weight: 0.60\n\nrobustness:\n  reserves: [0.55, 0.65]\n  annual_shock_intensities: [0.15, 0.25]\n  cost_and_slippage_multiplier: 2\n'
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
#!/usr/bin/env python3
"""Two-goal allocation by prospective scenario integration.

Paper (CC BY 4.0): https://arxiv.org/abs/2606.03158
Single-goal candidate comparison (no source license stated, ideas only):
https://github.com/matimacazaga/goal_based_wealth_management/blob/master/main.py
CPPI buffer and gap risk (MIT, ideas only):
https://cpoli.github.io/financialkit/_modules/financialkit/portfolio/systems/growth.html#cppi
https://pypi.org/project/financialkit/
"""

import json
import signal
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.special import ndtr, ndtri, roots_legendre


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
PRICE_FIELDS = ["Open", "Close", "Dividends", "Capital Gains"]
FETCH_PREFIX = "# DATA_FETCHED_AT: "


def valid_prices(prices, start, end):
    if prices.empty or not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("Expected a nonempty, sorted series of unique session dates")
    if prices.index[0] < start or prices.index[-1] > end:
        raise ValueError("Price dates fall outside the requested window")
    if prices.index[0] > start + pd.Timedelta(days=7) or prices.index[-1] < end - pd.Timedelta(days=7):
        raise ValueError("Price series does not cover the requested window")
    if (not np.isfinite(prices.to_numpy()).all()
            or (prices[["Open", "Close"]] <= 0).any().any()
            or (prices[["Dividends", "Capital Gains"]] < 0).any().any()):
        raise ValueError("Invalid price or cash distribution")
    return prices


def load_prices(ticker, start, end):
    path = CACHE / f"yfinance_split_{ticker}_{start.date()}_{end.date()}.csv"
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            header = stream.readline().strip()
            if not header.startswith(FETCH_PREFIX):
                raise ValueError(f"Missing download time in {path.name}")
            fetched_at = datetime.fromisoformat(header.removeprefix(FETCH_PREFIX))
            prices = pd.read_csv(stream, index_col="Date", parse_dates=["Date"])[PRICE_FIELDS]
        return valid_prices(prices, start, end), fetched_at

    history = yf.Ticker(ticker).history(
        start=start.date().isoformat(), end=(end + pd.Timedelta(days=1)).date().isoformat(),
        interval="1d", auto_adjust=False, actions=True, repair=False, timeout=15,
    )
    fetched_at = datetime.now(timezone.utc)
    if not {"Open", "Close", "Dividends"}.issubset(history.columns):
        raise RuntimeError(f"Yahoo Finance did not return daily prices and dividends for {ticker}")
    # Yahoo omits the Capital Gains field when an ETF has no such events.
    prices = history.assign(**{"Capital Gains": history.get("Capital Gains", 0.0)})[PRICE_FIELDS]
    prices.index = prices.index.tz_localize(None).rename("Date")
    valid_prices(prices, start, end)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(FETCH_PREFIX + fetched_at.isoformat() + "\n")
            prices.to_csv(stream)
            stream.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return prices, fetched_at


@dataclass(frozen=True)
class Market:
    dates: pd.DatetimeIndex
    overnight: np.ndarray
    intraday: np.ndarray
    cash_proxy_return: np.ndarray
    discount_base: np.ndarray
    covariance: np.ndarray


def prepare_market(histories, universe, lookback):
    equity, defensive, proxy = (universe[key] for key in ("equity", "defensive", "cash_proxy"))
    sessions = histories[equity].index
    overnight, intraday, total_gross = {}, {}, {}
    for ticker in (equity, defensive, proxy):
        prices = histories[ticker]
        if not sessions.equals(prices.index):
            raise ValueError(f"Session dates for {ticker} differ from {equity}")
        distributions = prices["Dividends"] + prices["Capital Gains"]
        overnight[ticker] = (prices["Open"] + distributions) / prices["Close"].shift(1)
        intraday[ticker] = prices["Close"] / prices["Open"]
        total_gross[ticker] = overnight[ticker] * intraday[ticker]

    excess_equity = np.log(total_gross[equity] / total_gross[proxy])
    excess_defensive = np.log(total_gross[defensive] / total_gross[proxy])
    cross = excess_equity.rolling(lookback).cov(excess_defensive)
    covariance = np.empty((len(sessions), 2, 2))
    covariance[:, 0, 0] = excess_equity.rolling(lookback).var()
    covariance[:, 1, 1] = excess_defensive.rolling(lookback).var()
    covariance[:, 0, 1] = covariance[:, 1, 0] = cross
    covariance *= SESSIONS_PER_YEAR
    return Market(
        sessions, np.column_stack([overnight[equity], overnight[defensive]]),
        np.column_stack([intraday[equity], intraday[defensive]]),
        (total_gross[proxy] - 1).to_numpy(),
        total_gross[proxy].fillna(1.0).cumprod().to_numpy(), covariance,
    )


def split_normal_samples(log_threshold, mean, deviation, unit_nodes):
    """Integrate separately on either side of the mandatory payment threshold."""
    miss = ndtr((log_threshold - mean) / deviation)
    miss_nodes = miss[..., None] * unit_nodes
    hit_nodes = miss[..., None] + (1 - miss[..., None]) * unit_nodes
    # Intervals of zero probability still need finite evaluation points.
    epsilon = np.finfo(float).eps
    below = mean[..., None] + deviation[..., None] * ndtri(np.clip(miss_nodes, epsilon, 1 - epsilon))
    above = mean[..., None] + deviation[..., None] * ndtri(np.clip(hit_nodes, epsilon, 1 - epsilon))
    return miss, below, above


def equity_choice(wealth, years_left, covariance, model, unit_nodes, probability_weights):
    """Compare prospective constant-weight policies from the signal close."""
    reserve = model["reserve"]
    if wealth <= reserve or years_left <= 0:
        return 0.0

    cap = 1 - reserve / wealth
    equity_grid = np.linspace(0, cap, model["weight_grid_size"])
    mix = np.column_stack([equity_grid, 1 - equity_grid])
    variance = np.einsum("ij,jk,ik->i", mix, covariance, mix)
    if not np.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError("Invalid rolling portfolio variance on the signal date")
    drift = mix @ [model["equity_excess_drift"], model["defensive_excess_drift"]] - variance / 2
    log_wealth = np.log(wealth)
    log_reserve = np.log(reserve)
    log_goal = np.log(model["goal_median"])
    goal_dispersion = model["goal_log_dispersion"]
    goal_priority = model["accumulation_priority"]
    reserve_priority = 1 - goal_priority

    intensity = model["annual_shock_intensity"]
    arrival_probability = -np.expm1(-intensity * years_left)
    arrival_time = -np.log1p(-arrival_probability * unit_nodes) / intensity
    time_after_arrival = years_left - arrival_time
    before_mean = log_wealth + drift[:, None] * arrival_time
    before_deviation = np.sqrt(variance[:, None] * arrival_time)
    miss, below, above = split_normal_samples(log_reserve, before_mean, before_deviation, unit_nodes)

    # Log wealth after paying the emergency, evaluated only on the affordable branch.
    distance = np.maximum(above - log_reserve, np.finfo(float).eps)
    after_payment = above + np.log(-np.expm1(-distance))
    after_growth = drift[:, None, None] * time_after_arrival[None, :, None]
    goal_deviation = np.sqrt(goal_dispersion**2 + variance[:, None] * time_after_arrival)[..., None]
    goal_after_miss = ndtr((below + after_growth - log_goal) / goal_deviation)
    goal_after_hit = ndtr((after_payment + after_growth - log_goal) / goal_deviation)
    value_on_arrival = (
        miss * goal_priority * (goal_after_miss @ probability_weights)
        + (1 - miss) * (reserve_priority + goal_priority * (goal_after_hit @ probability_weights))
    ) @ probability_weights

    # If the emergency remains pending, compare paying the fixed goal with reserving funds.
    deadline_mean = log_wealth + drift * years_left
    deadline_deviation = np.sqrt(variance * years_left)
    miss, below, above = split_normal_samples(log_reserve, deadline_mean, deadline_deviation, unit_nodes)
    goal_affordable = ndtr((above - log_goal) / goal_dispersion)
    distance = np.maximum(above - log_reserve, np.finfo(float).eps)
    log_surplus = above + np.log(-np.expm1(-distance))
    both_affordable = ndtr((log_surplus - log_goal) / goal_dispersion)
    keep_or_pay = (
        reserve_priority + goal_priority * both_affordable
        + max(0.0, goal_priority - reserve_priority) * (goal_affordable - both_affordable)
    )
    goal_without_reserve = goal_priority * ndtr((below - log_goal) / goal_dispersion)
    value_at_deadline = (
        miss * (goal_without_reserve @ probability_weights)
        + (1 - miss) * (keep_or_pay @ probability_weights)
    )
    objective = arrival_probability * value_on_arrival + (1 - arrival_probability) * value_at_deadline
    # Tie-break conservatively if candidate scores differ only by rounding.
    return float(equity_grid[np.flatnonzero(objective >= objective.max() - 1e-12)[0]])


def fill_at_open(positions, cash, equity_weight, friction):
    """Find the largest post-fee target affordable at the observed open."""
    available = float(positions.sum() + cash)
    weights = np.array([equity_weight, 1 - equity_weight])
    low, high = 0.0, available
    for _ in range(54):
        midpoint = (low + high) / 2
        target = weights * midpoint
        fee = friction * np.abs(target - positions).sum()
        if target.sum() + fee <= available:
            low = midpoint
        else:
            high = midpoint
    target = weights * low
    volume = np.abs(target - positions)
    cash = float(available - (target.sum() + friction * volume.sum()))
    if cash < 0 or (target < 0).any():
        raise RuntimeError("Opening fill exceeds available funds")
    return target, cash, float(volume.sum() / available), int(np.count_nonzero(volume > available * 1e-12))


def simulate(market, dates, model, execution, benchmark=False):
    first = market.dates.searchsorted(pd.Timestamp(dates["start"]))
    deadline = pd.Timestamp(dates["goal_deadline"]).to_period("M")
    lag = execution["lag_sessions"]
    if not isinstance(lag, int) or not 1 <= lag <= first < len(market.dates):
        raise ValueError("Expected a positive signal lag and sufficient prior sessions")
    friction = (execution["cost_bps"] + execution["slippage_bps"]) / 10_000
    if min(execution["cost_bps"], execution["slippage_bps"]) < 0 or friction >= 1:
        raise ValueError("Execution costs and slippage must be nonnegative and total less than 100%")

    nodes, probability_weights = roots_legendre(model["quadrature_nodes"])
    unit_nodes, probability_weights = (1 + nodes) / 2, probability_weights / 2
    month = market.dates.to_period("M")
    close_values = np.ones(len(market.dates))
    base_at_start = market.discount_base[first - 1]
    positions, cash = np.zeros(2), 1.0
    rows = []
    for i in range(first, len(market.dates)):
        positions *= market.overnight[i]
        turnover, trades, target_equity = 0.0, 0, np.nan
        if i == first or month[i] != month[i - 1]:
            if benchmark:
                target_equity = execution["benchmark_equity_weight"]
            else:
                signal_day = i - lag
                wealth = close_values[signal_day]
                if signal_day >= first:
                    wealth *= base_at_start / market.discount_base[signal_day]
                years_left = (deadline.ordinal - month[i].ordinal) / 12
                target_equity = equity_choice(
                    wealth, years_left, market.covariance[signal_day], model, unit_nodes, probability_weights,
                )
            positions, cash, turnover, trades = fill_at_open(positions, cash, target_equity, friction)
        positions *= market.intraday[i]
        close_values[i] = positions.sum() + cash
        rows.append((close_values[i] / close_values[i - 1] - 1, market.cash_proxy_return[i],
                     turnover, trades, target_equity, cash))
    return pd.DataFrame(rows, index=market.dates[first:],
                        columns=["daily_return", "cash_proxy_return", "turnover", "trades", "equity_target", "cash"])


def metrics(daily_results):
    returns = daily_results["daily_return"].to_numpy()
    if len(returns) < 2:
        raise ValueError("Each evaluation period needs at least two sessions")
    excess = returns - daily_results["cash_proxy_return"].to_numpy()
    scale = np.sqrt(SESSIONS_PER_YEAR)
    growth = np.r_[1.0, np.cumprod(1 + returns)]
    deviation = returns.std(ddof=1)
    excess_deviation = excess.std(ddof=1)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(deviation * scale),
        "sharpe": float(returns.mean() / deviation * scale) if deviation else None,
        "sharpe_excess": float(excess.mean() / excess_deviation * scale) if excess_deviation else None,
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1)),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["trades"].sum()),
    }


def periods(results, split):
    within = results.index < split
    return {"is": metrics(results.loc[within]), "oos": metrics(results.loc[~within]), "full": metrics(results)}


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    dates, universe = config["dates"], config["universe"]
    model, execution = config["model"], config["execution"]
    first_data, split, last_data = map(pd.Timestamp, (
        dates["warmup_start"], dates["out_of_sample_start"], dates["end"],
    ))
    start = pd.Timestamp(dates["start"])
    if not first_data < start < split <= last_data:
        raise ValueError("Expected warm-up, strategy start, split, and end in chronological order")
    if (min(model[key] for key in ("reserve", "goal_median", "goal_log_dispersion", "annual_shock_intensity")) <= 0
            or not 0 <= model["accumulation_priority"] <= 1
            or min(model["covariance_lookback"], model["weight_grid_size"], model["quadrature_nodes"]) < 2):
        raise ValueError("Invalid reserve, goal distribution, priority, or model resolution")
    CACHE.mkdir(exist_ok=True)
    histories, fetched_times = {}, []
    with TemporaryDirectory(dir=CACHE) as session_cache:
        yf.set_tz_cache_location(session_cache)
        for ticker in universe.values():
            histories[ticker], fetched_at = load_prices(ticker, first_data, last_data)
            fetched_times.append(fetched_at)
    print(f"DATA_FETCHED_AT: {max(fetched_times).isoformat()}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance split-adjusted open/close; ex-date distributions reinvested at the open",
          flush=True)

    within_histories = {ticker: prices.loc[prices.index < split] for ticker, prices in histories.items()}
    within_market = prepare_market(within_histories, universe, model["covariance_lookback"])
    settings = config["robustness"]
    variants = []
    for parameter, values in (("reserve", settings["reserves"]),
                              ("annual_shock_intensity", settings["annual_shock_intensities"])):
        variants.extend((f"{parameter}={value}", model | {parameter: value}, execution) for value in values)
    multiplier = settings["cost_and_slippage_multiplier"]
    higher_cost = {key: execution[key] * multiplier for key in ("cost_bps", "slippage_bps")}
    variants.append((f"cost_and_slippage_x{multiplier}", model, execution | higher_cost))

    print("ROBUSTNESS (IS ONLY)", flush=True)
    trials = 0
    for label, trial_model, trial_execution in variants:
        result = simulate(within_market, dates, trial_model, trial_execution)
        trials += 1
        print(f"  {label}: {json.dumps(metrics(result), allow_nan=False)}", flush=True)

    market = prepare_market(histories, universe, model["covariance_lookback"])
    results = simulate(market, dates, model, execution)
    trials += 1
    official = periods(results, split)
    if min(official["is"]["n_trades"], official["oos"]["n_trades"]) <= 0:
        raise RuntimeError("Official configuration must produce fills in both periods")
    benchmark = periods(simulate(market, dates, model, execution, benchmark=True), split)
    bench_equity = execution["benchmark_equity_weight"]
    bench_name = f"Monthly {bench_equity:.0%} {universe['equity']} / {1 - bench_equity:.0%} {universe['defensive']}"
    for period, values in official.items():
        print(f"{period.upper()}: {json.dumps(values, allow_nan=False)}")
        print(f"BENCHMARK {period.upper()}: {json.dumps(benchmark[period], allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {trials}")
    print("已回测 · 待审查")
    output = {
        "strategy": config["strategy"],
        "data_range": {"start": results.index[0].date().isoformat(), "end": results.index[-1].date().isoformat()},
        **official,
        "benchmark": {"name": bench_name, **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(output, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    main()
