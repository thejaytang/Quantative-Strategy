# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 708_doubly_robust_mean_cvar_portfolio\nuniverse: [SPY, XLK, XLF, AGG, GLD]\nrisk_free: BIL\ndownload_start: '2007-07-02'\nstart: '2013-01-02'\nis_end: '2019-12-31'\nend: '2025-12-31'\nlookback: 1260\ntail_levels: [0.95, 0.96, 0.97, 0.98, 0.99]\nmean_confidence: 0.95\nexecution_lag_sessions: 1\ncost_bps: 2\nslippage_bps: 3\nrobustness:\n  - {label: mean_confidence_90, mean_confidence: 0.90}\n  - {label: mean_confidence_99, mean_confidence: 0.99}\n  - {label: lookback_1008, lookback: 1008}\n  - {label: doubled_friction, cost_bps: 4, slippage_bps: 6}\n"
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
"""Monthly ETF research implementation of Doubly Robust Mean-CVaR Portfolio.

Paper (Nakagawa, Abe, and Kuroki, 2023): https://arxiv.org/abs/2309.11693
CVaR modeling reference, PyPortfolioOpt (MIT):
https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/pypfopt/efficient_frontier/efficient_cvar.py
https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/LICENSE
https://pypi.org/project/pyportfolioopt/
Ellipsoidal worst-case mean reference, Daniel P. Palomar:
https://portfoliooptimizationbook.com/python-code/python-robust-portfolios.html
Author publication list: https://sites.google.com/view/keinakagawa/research
The optimization and simulation code was written independently.
"""

import hashlib
import json
import signal
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, linprog
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
PRICE_FIELDS = ("Open", "Close", "Distribution")


def load_prices(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    identity = json.dumps([tickers, str(start.date()), str(end.date())])
    fingerprint = hashlib.sha256(identity.encode()).hexdigest()[:16]
    path = cache_dir / f"yfinance_split_ohlc_actions_{fingerprint}.csv"
    columns = pd.MultiIndex.from_product([PRICE_FIELDS, tickers])

    if path.exists():
        with path.open() as source:
            fetched_at = source.readline().strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True)
    else:
        with TemporaryDirectory(dir=cache_dir) as yf_cache:
            yf.set_tz_cache_location(yf_cache)
            downloaded = yf.download(
                tickers, start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=False, actions=True, ignore_tz=True,
                progress=False, threads=False, timeout=15,
            )
        if downloaded.empty or not {"Open", "Close", "Dividends"}.issubset(downloaded.columns.levels[0]):
            raise RuntimeError("Yahoo Finance returned no complete price and action data")

        distributions = downloaded["Dividends"].reindex(columns=tickers)
        if "Capital Gains" in downloaded.columns.levels[0]:
            gains = downloaded["Capital Gains"].reindex(columns=tickers).fillna(0.0)
            distributions = distributions + gains
        prices = pd.concat({
            "Open": downloaded["Open"], "Close": downloaded["Close"],
            "Distribution": distributions,
        }, axis=1)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    prices = prices.reindex(columns=columns)
    values = prices.to_numpy()
    if (prices.empty or not prices.index.is_monotonic_increasing or not prices.index.is_unique
            or prices.index[0] != start or prices.index[-1] != end
            or not np.isfinite(values).all() or (prices[["Open", "Close"]] <= 0).any().any()
            or (prices["Distribution"] < 0).any().any()):
        raise RuntimeError("Incomplete or invalid split-adjusted Yahoo Finance history")
    if not path.exists():
        with NamedTemporaryFile(mode="w", dir=cache_dir, delete=False) as pending:
            pending.write(fetched_at + "\n")
            prices.to_csv(pending, float_format="%.17g")
        Path(pending.name).replace(path)
    return prices, fetched_at


def market_factors(prices):
    # The ex-date amount and prior close are available before the opening fill.
    # Reinvestment is modeled at the theoretical ex-distribution prior close.
    previous_ex_price = (prices["Close"].shift(1) - prices["Distribution"]).iloc[1:]
    if (previous_ex_price <= 0).any().any():
        raise RuntimeError("Invalid ex-distribution prior price")
    open_prices = prices["Open"].iloc[1:]
    close_prices = prices["Close"].iloc[1:]
    return open_prices / previous_ex_price, close_prices / open_prices, close_prices / previous_ex_price - 1.0


def tail_supports(returns, levels, weights):
    """Return the active affine supports of all empirical CVaRs at weights."""
    worst_first = np.argsort(returns @ weights, kind="stable")
    losses = -returns[worst_first]
    cumulative = np.vstack([np.zeros(returns.shape[1]), np.cumsum(losses, axis=0)])
    tail_size = len(returns) * (1.0 - levels)
    whole = np.floor(tail_size).astype(int)
    remainder = (tail_size - whole)[:, None]
    partial = losses[np.minimum(whole, len(returns) - 1)]
    return (cumulative[whole] + remainder * partial) / tail_size[:, None]


def solve_epigraph(linear, support_groups, tolerance):
    """Minimize a linear term plus a sum of convex epigraphs over the simplex.

    Each group returns the supporting slopes and offsets active at the current
    weights. The master LP supplies a lower bound and evaluating the active
    supports at that feasible point supplies an upper bound.
    """
    assets = len(linear)
    groups = len(support_groups)
    weights = np.full(assets, 1.0 / assets)
    unit_slacks = np.eye(groups)
    objective = np.r_[linear, np.ones(groups)]
    rows, limits = [], []
    bounds = [(0, 1)] * assets + [(None, None)] * groups
    equality = np.r_[np.ones(assets), np.zeros(groups)][None, :]
    best_value = np.inf
    solution = None

    for _ in range(400):
        value = float(linear @ weights)
        for group, active_supports in enumerate(support_groups):
            slopes, offsets = active_supports(weights)
            support_values = slopes @ weights + offsets
            value += float(support_values.max())
            for slope, offset, support_value in zip(slopes, offsets, support_values):
                if solution is None or support_value > solution.x[assets + group] + tolerance / (2 * groups):
                    rows.append(np.r_[slope, -unit_slacks[group]])
                    limits.append(-offset)
        if value < best_value:
            best_weights, best_value = weights, value
        if solution is not None and best_value - solution.fun <= tolerance:
            return best_weights, best_value

        solution = linprog(
            objective, A_ub=rows, b_ub=limits, A_eq=equality, b_eq=[1.0], bounds=bounds,
            method="highs", options={"primal_feasibility_tolerance": 1e-10,
                                     "dual_feasibility_tolerance": 1e-10, "time_limit": 8.0},
        )
        if not solution.success:
            raise RuntimeError(f"CVaR epigraph LP failed: {solution.message}")
        weights = np.maximum(solution.x[:assets], 0)
        weights /= weights.sum()
    raise RuntimeError("CVaR epigraph solver exceeded its iteration limit")


def calibrate_window(returns, levels):
    assets = returns.shape[1]
    if not np.isfinite(returns).all():
        raise ValueError("Rolling scenarios contain missing or non-finite returns")

    reference = []
    for level in levels:
        one_level = np.array([level])

        def cvar_support(weights):
            return tail_supports(returns, one_level, weights), np.zeros(1)

        _, minimum = solve_epigraph(np.zeros(assets), [cvar_support], tolerance=1e-9)
        reference.append(minimum)

    reference = np.array(reference)
    if (np.abs(reference) <= 1e-8).any():
        raise RuntimeError("A minimum CVaR is too close to zero to normalize regret")
    mean = returns.mean(axis=0)
    covariance_of_mean = np.cov(returns, rowvar=False, ddof=1) / len(returns)
    return reference, mean, covariance_of_mean


def robust_weights(returns, levels, confidence, calibration):
    reference, mean, covariance_of_mean = calibration
    scale = np.abs(reference)
    radius = np.sqrt(chi2.ppf(confidence, df=len(mean)))

    def regret_supports(weights):
        return tail_supports(returns, levels, weights) / scale[:, None], -reference / scale

    def mean_penalty_support(weights):
        projected = covariance_of_mean @ weights
        spread = np.sqrt(max(float(weights @ projected), 0.0))
        slope = radius * projected / spread if spread > 0 else np.zeros_like(weights)
        return slope[None, :], np.zeros(1)

    return solve_epigraph(-mean, [regret_supports, mean_penalty_support], tolerance=1e-7)[0]


def monthly_execution_dates(sessions, lag, start, end):
    months = sessions.to_period("M")
    # At the first opening of a new month its preceding session is the signal date.
    for signal_idx in np.flatnonzero(months[1:] != months[:-1]):
        fill_idx = signal_idx + lag
        if fill_idx < len(sessions) and start <= sessions[fill_idx] <= end:
            yield sessions[signal_idx], sessions[fill_idx]


def execute_at_open(open_positions, cash, weights, cost_rate, slippage_rate):
    open_value = cash + open_positions.sum()
    rate = cost_rate + slippage_rate
    open_weights = open_positions / open_value

    def budget_gap(fraction):
        return fraction + rate * np.abs(weights * fraction - open_weights).sum() - 1.0

    # The scalar budget is strictly increasing for rate < 1. Its root funds both
    # the target positions and the charges on the actual bought and sold amounts.
    fraction = brentq(budget_gap, 0.0, 1.0, xtol=1e-14)
    invested = open_value * max(0.0, fraction - 2e-14)
    positions = weights * invested
    amounts = np.abs(positions - open_positions)
    gross = amounts.sum()
    remaining = open_value - positions.sum() - gross * cost_rate - gross * slippage_rate
    if remaining < 0 or (positions < 0).any():
        raise RuntimeError("Opening simulation exceeded its available budget")
    return positions, remaining, gross / open_value, int((amounts > 1e-9 * open_value).sum())


def simulate(overnight, intraday, targets, settings):
    if not targets:
        raise RuntimeError("The evaluation period contains no scheduled fills")
    cost_rate = settings["cost_bps"] / 10_000
    slippage_rate = settings["slippage_bps"] / 10_000
    cash = 1.0
    positions = np.zeros(overnight.shape[1])
    daily = []

    for day, gap, session in zip(overnight.index, overnight.to_numpy(), intraday.to_numpy()):
        previous_value = cash + positions.sum()
        positions *= gap
        turnover = trades = 0
        if day in targets:
            positions, cash, turnover, trades = execute_at_open(
                positions, cash, targets[day], cost_rate, slippage_rate,
            )
        positions *= session
        daily.append(((cash + positions.sum()) / previous_value - 1.0, turnover, trades))

    return pd.DataFrame(daily, index=overnight.index, columns=["return", "gross_turnover", "n_trades"])


def metrics(daily_results, risk_free):
    daily_returns = daily_results["return"]
    path = (1 + daily_returns).cumprod()
    length = len(path)
    volatility = daily_returns.std(ddof=1)
    excess = daily_returns - risk_free.loc[daily_returns.index]
    excess_volatility = excess.std(ddof=1)
    annualization = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(path.iloc[-1] ** (SESSIONS_PER_YEAR / length) - 1.0),
        "ann_vol": float(volatility * annualization),
        "sharpe": float(daily_returns.mean() / volatility * annualization) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_volatility * annualization) if excess_volatility > 0 else None,
        "max_drawdown": float((path / path.cummax().clip(lower=1.0) - 1.0).min()),
        "total_return": float(path.iloc[-1] - 1.0),
        "turnover": float(daily_results["gross_turnover"].mean() * SESSIONS_PER_YEAR / 2),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def validate_settings(config, trials):
    levels = np.asarray(config["tail_levels"], dtype=float)
    tickers = config["universe"] + [config["risk_free"]]
    if len(tickers) < 3 or len(set(tickers)) != len(tickers):
        raise ValueError("Use distinct ETFs and a separate risk-free proxy")
    if not (len(levels) >= 2 and np.all((levels > 0) & (levels < 1)) and np.all(np.diff(levels) > 0)):
        raise ValueError("Tail confidence levels must be distinct, increasing, and inside (0, 1)")
    lag = config["execution_lag_sessions"]
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("Execution lag must be a positive number of sessions")
    if not config["download_start"] < config["start"] < config["is_end"] < config["end"]:
        raise ValueError("The data warmup, evaluation, IS and OOS dates must be chronological")
    for trial in [config, *trials]:
        if not isinstance(trial["lookback"], int) or trial["lookback"] < 2:
            raise ValueError("Lookback must contain at least two returns")
        if not 0 < trial["mean_confidence"] < 1:
            raise ValueError("Mean confidence must be inside (0, 1)")
        cost_bps, slippage_bps = trial["cost_bps"], trial["slippage_bps"]
        if not (cost_bps >= 0 and slippage_bps >= 0 and cost_bps + slippage_bps < 10_000):
            raise ValueError("Cost and slippage must be nonnegative and total less than 10,000 bps")
    return levels


def print_json(label, output):
    print(f"{label}: {json.dumps(output, allow_nan=False)}", flush=True)


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    variant_fields = {"label", "lookback", "mean_confidence", "cost_bps", "slippage_bps"}
    if any("label" not in trial or set(trial) - variant_fields for trial in config["robustness"]):
        raise ValueError("Each robustness variant needs a label and supported parameter overrides")
    trials = [config | trial for trial in config["robustness"]]
    levels = validate_settings(config, trials)
    tickers = config["universe"]
    prices, fetched_at = load_prices(tickers + [config["risk_free"]], config["download_start"], config["end"])
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance split-adjusted Open/Close with causal ex-date distribution total return", flush=True)
    overnight, intraday, daily_returns = market_factors(prices)
    risk_free = daily_returns[config["risk_free"]]
    daily_returns = daily_returns[tickers]
    start, is_end, end = map(pd.Timestamp, (config["start"], config["is_end"], config["end"]))
    lag = config["execution_lag_sessions"]

    @cache
    def window_for(signal_date, lookback):
        window = daily_returns.loc[:signal_date].tail(lookback).to_numpy()
        if len(window) != lookback:
            raise ValueError(f"Insufficient warmup for the signal on {signal_date.date()}")
        return window, calibrate_window(window, levels)

    @cache
    def weights_for(signal_date, lookback, confidence):
        window, calibration = window_for(signal_date, lookback)
        return robust_weights(window, levels, confidence, calibration)

    def targets_through(settings, last_day):
        dates = monthly_execution_dates(prices.loc[:last_day].index, lag, start, last_day)
        return {fill: weights_for(observed, settings["lookback"], settings["mean_confidence"])
                for observed, fill in dates}

    combos_tried = 1
    official_is = targets_through(config, is_end)
    print("ROBUSTNESS (IS only)", flush=True)
    for settings in trials:
        combos_tried += 1
        targets = targets_through(settings, is_end)
        daily_results = simulate(overnight.loc[start:is_end, tickers], intraday.loc[start:is_end, tickers], targets, settings)
        print_json(settings["label"], metrics(daily_results, risk_free))

    official_targets = targets_through(config, end)
    if any(not np.array_equal(official_targets[day], weights) for day, weights in official_is.items()):
        raise RuntimeError("Official IS targets changed when OOS sessions were appended")
    test_overnight = overnight.loc[start:end, tickers]
    test_intraday = intraday.loc[start:end, tickers]
    daily_results = simulate(test_overnight, test_intraday, official_targets, config)
    equal_weights = np.full(len(tickers), 1.0 / len(tickers))
    benchmark_results = simulate(
        test_overnight, test_intraday, dict.fromkeys(official_targets, equal_weights), config,
    )

    periods = {
        "is": daily_results.index <= is_end, "oos": daily_results.index > is_end,
        "full": np.ones(len(daily_results), dtype=bool),
    }
    strategy_summary = {name: metrics(daily_results.loc[mask], risk_free) for name, mask in periods.items()}
    benchmark_summary = {name: metrics(benchmark_results.loc[mask], risk_free) for name, mask in periods.items()}
    if not all(strategy_summary[period]["n_trades"] > 0 for period in ("is", "oos")):
        raise RuntimeError("The default strategy must produce fills in both evaluation periods")

    print(f"PARAM_COMBOS_TRIED: {combos_tried}", flush=True)
    print("LOOK_AHEAD_CHECK: month-end close signal, later open fill, ex-date events, rolling IS/OOS separation", flush=True)
    for name in periods:
        print_json(name.upper(), strategy_summary[name])
        print_json(f"BENCHMARK EQUAL WEIGHT {name.upper()}", benchmark_summary[name])
    print_json("RESULTS_JSON", {
        "strategy": config["strategy"],
        "data_range": {"start": str(daily_results.index[0].date()), "end": str(daily_results.index[-1].date()),
                       "is_end": str(is_end.date())},
        **strategy_summary,
        "benchmark": {"name": "monthly_equal_weight", **benchmark_summary["full"], **benchmark_summary},
    })


if __name__ == "__main__":
    signal.alarm(14 * 60)
    main()
