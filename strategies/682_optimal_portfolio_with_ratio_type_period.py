# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 682_optimal_portfolio_with_ratio_type_period\nuniverse: [SPY, EFA, EEM, IEF, TLT, GLD, VNQ, BIL]\ncash_proxy: BIL\ndates:\n  data_start: 2008-01-01\n  start: 2010-01-04\n  oos_start: 2018-01-01\n  end: 2025-12-31\nmodel:\n  lookback_sessions: 252\n  evaluation_sessions: 21\n  alpha: 0.5\n  gamma: 0.8\n  annual_discount: 0.30\n  max_risky_weight: 0.40\nexecution:\n  lag_sessions: 1\n  cost_bps: 2\n  slippage_bps: 3\nrobustness:\n  - name: alpha_lower\n    model: {alpha: 0.4}\n  - name: alpha_higher\n    model: {alpha: 0.6}\n  - name: gamma_one\n    model: {gamma: 1.0}\n  - name: longer_lookback\n    model: {lookback_sessions: 315}\n  - name: higher_friction\n    execution: {cost_bps: 15, slippage_bps: 10}\n'
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
"""C257 | Long-only periodic ratio-utility portfolio research.

Paper: Wang, Yan and Yu (2023), https://arxiv.org/abs/2311.12517
Related research examined before implementation (independent implementation):
- CVX Group finite-outcome Kelly notebook (GPL-3.0):
  https://github.com/cvxgrp/kelly_code/blob/master/finite_outcome.ipynb
- QuantConnect LEAN rolling portfolio construction (Apache-2.0):
  https://github.com/QuantConnect/Lean/blob/master/Algorithm.Framework/Portfolio/MeanVarianceOptimizationPortfolioConstructionModel.py
- skfolio scenario and periodicity documentation (BSD-3-Clause):
  https://github.com/skfolio/skfolio/blob/main/docs/user_guide/data_preparation.rst
  https://pypi.org/project/skfolio/
"""

import gzip
import hashlib
import json
import signal
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, minimize


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
ADJUSTMENT = "yfinance split-adjusted OHLC and ex-date distributions, reinvested at the ex-date open"


@dataclass(frozen=True)
class Model:
    lookback_sessions: int
    evaluation_sessions: int
    alpha: float
    gamma: float
    annual_discount: float
    max_risky_weight: float

    def __post_init__(self):
        if not 1 <= self.evaluation_sessions < self.lookback_sessions:
            raise ValueError("The evaluation horizon must be shorter than the lookback")
        if not (0 < self.alpha < 1 and 0 < self.gamma <= 1 and self.annual_discount > 0):
            raise ValueError("Use 0 < alpha < 1, 0 < gamma <= 1, and positive discounting")
        if not 0 < self.max_risky_weight <= 1:
            raise ValueError("Use a positive risky-asset limit no greater than one")


@dataclass(frozen=True)
class Execution:
    lag_sessions: int
    cost_bps: float
    slippage_bps: float

    def __post_init__(self):
        if type(self.lag_sessions) is not int or self.lag_sessions < 1:
            raise ValueError("Execution must lag the signal by at least one session")
        if not (0 <= self.cost_bps and 0 <= self.slippage_bps and self.friction < 1):
            raise ValueError("Use nonnegative costs and slippage totaling less than 10,000 bp")

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def check_bars(bars, tickers, start, end):
    expected = pd.MultiIndex.from_product([tickers, ["Open", "Close", "Distribution"]])
    if bars.empty or not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError("Missing or unordered Yahoo Finance sessions")
    if not bars.columns.equals(expected):
        raise ValueError("Unexpected OHLC or distribution fields")
    if not (start <= bars.index[0] <= start + pd.Timedelta(days=7)
            and end - pd.Timedelta(days=7) <= bars.index[-1] <= end):
        raise ValueError("The Yahoo Finance bars do not cover the requested date range")
    if not np.isfinite(bars.to_numpy()).all():
        missing = bars.columns[bars.isna().any()].get_level_values(0).unique().tolist()
        raise ValueError(f"Incomplete Yahoo Finance data for {missing}")
    for field in ("Open", "Close"):
        if (bars.xs(field, axis=1, level=1) <= 0).any().any():
            raise ValueError(f"Nonpositive Yahoo Finance {field} price")
    if (bars.xs("Distribution", axis=1, level=1) < 0).any().any():
        raise ValueError("Negative Yahoo Finance distribution")


def load_bars(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    key = json.dumps(["yahoo-split-adjusted-v1", tickers, start.date().isoformat(), end.date().isoformat()])
    path = CACHE / f"etf-bars-{hashlib.sha256(key.encode()).hexdigest()[:16]}.csv.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as source:
            fetched_at = json.loads(source.readline())["fetched_at"]
            bars = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True)
        check_bars(bars, tickers, start, end)
        return bars, fetched_at

    with TemporaryDirectory(prefix="yfinance-", dir=CACHE) as session_dir:
        yf.set_tz_cache_location(session_dir)
        for attempt in range(2):
            try:
                raw = yf.download(tickers, start=start.date().isoformat(),
                                  end=(end + pd.Timedelta(days=1)).date().isoformat(),
                                  interval="1d", auto_adjust=False, actions=True, repair=False,
                                  group_by="ticker", threads=4, progress=False, timeout=15)
                by_ticker = {}
                for ticker in tickers:
                    fields = raw[ticker]
                    distributions = fields.reindex(columns=["Dividends", "Capital Gains"]).fillna(0)
                    by_ticker[ticker] = fields[["Open", "Close"]].assign(Distribution=distributions.sum(axis=1))
                bars = pd.concat(by_ticker, axis=1)
                check_bars(bars, tickers, start, end)
                break
            except (KeyError, TypeError, ValueError) as exc:
                if attempt == 1:
                    raise RuntimeError(f"Unable to load complete Yahoo Finance data: {exc}") from exc
                print(f"Retrying Yahoo Finance after: {exc}", flush=True)

    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    with NamedTemporaryFile(dir=CACHE, suffix=".gz", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        with gzip.open(temp_path, "wt", encoding="utf-8") as target:
            target.write(json.dumps({"fetched_at": fetched_at}) + "\n")
            bars.to_csv(target)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    return bars, fetched_at


def market_moves(bars):
    opens = bars.xs("Open", axis=1, level=1)
    closes = bars.xs("Close", axis=1, level=1)
    distributions = bars.xs("Distribution", axis=1, level=1)
    # Prior exposure gets the distribution at the opening mark on its ex-date.
    overnight = (opens + distributions) / closes.shift(1)
    intraday = closes / opens
    return overnight, intraday


def ratio_weights(scenarios, model, cash_column):
    """Solve A = d max_w E[G(w)^alpha + A G(w)^beta] on the capped simplex."""
    alpha = model.alpha
    beta = alpha * (1 - model.gamma)
    discount = np.exp(-model.annual_discount * model.evaluation_sessions / SESSIONS_PER_YEAR)
    # Jensen bounds the continuation term over every feasible weight vector.
    if discount * scenarios.mean(axis=0).max() ** beta >= 1:
        raise RuntimeError("The empirical continuation operator lacks the required contraction bound")

    excess = scenarios - 1
    caps = np.full(scenarios.shape[1], model.max_risky_weight)
    caps[cash_column] = 1.0
    weights = np.zeros_like(caps)
    weights[cash_column] = 1.0
    budget = {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)}
    continuation = discount / (1 - discount)

    def objective(candidate):
        gross = 1 + excess @ candidate
        current = gross ** alpha
        future = gross ** beta
        value = ((current - 1) + continuation * (future - 1)).mean()
        gradient = excess.T @ ((alpha * current + continuation * beta * future) / gross)
        return -value, -gradient / len(gross)

    # Update via the convergent fractional-programming form of the fixed point.
    for _ in range(20):
        solution = minimize(objective, weights, method="SLSQP", jac=True,
                            bounds=list(zip(np.zeros_like(caps), caps)), constraints=budget,
                            options={"ftol": 1e-12, "maxiter": 150})
        if not solution.success:
            raise RuntimeError(f"Utility optimization failed: {solution.message}")
        weights = np.clip(solution.x, 0, caps)
        weights /= weights.sum()
        if (weights > caps + 1e-10).any():
            raise RuntimeError("The optimizer violated the weight cap")
        gross = scenarios @ weights
        next_value = discount * (gross ** alpha).mean() / (1 - discount * (gross ** beta).mean())
        if abs(next_value - continuation) < 1e-9 * max(1, continuation):
            return weights
        continuation = next_value
    raise RuntimeError("The periodic utility fixed point did not converge")


def rebalance(opening_values, cash, weights, friction):
    """Size simulated fills from open marks and fund buys with sales and free cash."""
    opening_value = float(opening_values.sum()) + cash

    def budget(post_trade_value):
        traded_value = np.abs(weights * post_trade_value - opening_values).sum()
        return post_trade_value + friction * traded_value - opening_value

    investable = brentq(budget, 0, opening_value, xtol=opening_value * 1e-14)
    changes = weights * investable - opening_values
    changes[np.abs(changes) <= opening_value * 1e-10] = 0
    sales = np.maximum(-changes, 0)
    buys = np.maximum(changes, 0)
    available = cash + float(sales.sum()) * (1 - friction)
    purchase_cost = float(buys.sum()) * (1 + friction)
    if purchase_cost > available:
        buys *= available / purchase_cost
        purchase_cost = available
    positions = opening_values - sales + buys
    remaining_cash = available - purchase_cost
    turnover = (float(sales.sum()) + float(buys.sum())) / opening_value
    n_trades = np.count_nonzero(sales) + np.count_nonzero(buys)
    return positions, remaining_cash, turnover, n_trades


def simulate(bars, start, end, cash_proxy, model, execution, *, equal_weight=False):
    bars = bars.loc[:end]
    overnight, intraday = market_moves(bars)
    close_growth = overnight * intraday
    gross_scenarios = np.exp(np.log(close_growth).rolling(model.evaluation_sessions).sum()).to_numpy()
    cash_column = overnight.columns.get_loc(cash_proxy)
    cash_proxy_returns = close_growth.iloc[:, cash_column] - 1
    nights, days = overnight.to_numpy(), intraday.to_numpy()
    first = bars.index.searchsorted(start)
    if first - execution.lag_sessions < model.lookback_sessions - 1 or first >= len(bars):
        raise ValueError("Insufficient history to start at the configured date and lag")

    cash = 1.0
    positions = np.zeros(len(overnight.columns))
    previous_value = cash
    daily_results = []
    for step in range(first, len(bars)):
        positions = positions * nights[step]
        turnover, n_trades = 0.0, 0
        if (step - first) % model.evaluation_sessions == 0:
            signal_step = step - execution.lag_sessions
            first_scenario = signal_step - model.lookback_sessions + model.evaluation_sessions + 1
            if equal_weight:
                weights = np.full(len(positions), 1 / len(positions))
            else:
                weights = ratio_weights(gross_scenarios[first_scenario:signal_step + 1], model, cash_column)
            positions, cash, turnover, n_trades = rebalance(positions, cash, weights, execution.friction)
        positions = positions * days[step]
        closing_value = cash + float(positions.sum())
        daily_results.append((closing_value / previous_value - 1, turnover, n_trades, cash))
        previous_value = closing_value

    results = pd.DataFrame(daily_results, index=bars.index[first:],
                           columns=["return", "turnover", "n_trades", "cash"])
    results["cash_proxy_return"] = cash_proxy_returns.loc[results.index]
    if (results["cash"] < 0).any() or not np.isfinite(results.to_numpy()).all():
        raise RuntimeError("Invalid balances or returns in the simulation")
    return results


def metrics(daily_results):
    if len(daily_results) < 2:
        raise ValueError("At least two return observations are required per period")
    returns = daily_results["return"].to_numpy()
    excess = returns - daily_results["cash_proxy_return"].to_numpy()
    growth = np.concatenate(([1.0], np.cumprod(1 + returns)))
    vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    scale = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(vol * scale),
        "sharpe": float(returns.mean() / vol * scale) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * scale) if excess_vol > 0 else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, oos_start):
    return {"is": metrics(daily_results.loc[daily_results.index < oos_start]),
            "oos": metrics(daily_results.loc[oos_start:]), "full": metrics(daily_results)}


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    dates = {name: pd.Timestamp(value) for name, value in config["dates"].items()}
    data_start, start, oos_start, end = (dates[key] for key in ("data_start", "start", "oos_start", "end"))
    if not data_start < start < oos_start < end:
        raise ValueError("Require warm-up, IS and OOS in chronological order")
    tickers, cash_proxy = config["universe"], config["cash_proxy"]
    if len(set(tickers)) != len(tickers) or len(tickers) < 2 or cash_proxy not in tickers:
        raise ValueError("Require a unique ETF universe that contains the cash proxy")
    model, execution = Model(**config["model"]), Execution(**config["execution"])
    bars, fetched_at = load_bars(tickers, data_start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print(f"PRICE_ADJUSTMENT: {ADJUSTMENT}", flush=True)

    is_end = oos_start - pd.Timedelta(days=1)
    is_bars = bars.loc[:is_end]
    base_is = simulate(is_bars, start, is_end, cash_proxy, model, execution)
    print("ROBUSTNESS (IS ONLY)", flush=True)
    print("BASE IS: " + json.dumps(metrics(base_is), allow_nan=False), flush=True)
    tried = {(model, execution)}
    for case in config["robustness"]:
        case_model = replace(model, **case.get("model", {}))
        case_execution = replace(execution, **case.get("execution", {}))
        if (case_model, case_execution) in tried:
            raise ValueError(f"Repeated parameter combination: {case['name']}")
        variant = simulate(is_bars, start, is_end, cash_proxy, case_model, case_execution)
        tried.add((case_model, case_execution))
        print(f"{case['name']} IS: " + json.dumps(metrics(variant), allow_nan=False), flush=True)
    print(f"PARAM_COMBOS_TRIED: {len(tried)}")

    official = simulate(bars, start, end, cash_proxy, model, execution)
    pd.testing.assert_frame_equal(base_is, official.loc[:is_end], check_exact=True, check_freq=False)
    benchmark = simulate(bars, start, end, cash_proxy, model, execution, equal_weight=True)
    official_periods = period_metrics(official, oos_start)
    benchmark_periods = period_metrics(benchmark, oos_start)
    if any(official_periods[key]["n_trades"] == 0 for key in ("is", "oos")):
        raise RuntimeError("No ETF trades in one or more required periods")

    print("LOOK_AHEAD_SELF_CHECK: completed close windows, lagged open sizing, ex-date distributions, "
          "no future economic releases, and identical IS results with or without OOS bars")
    print("RUN_STATUS: 已回测")
    print("REVIEW_STATUS: 待审，确有交易")
    for period in ("is", "oos", "full"):
        print(f"STRATEGY {period.upper()}: " + json.dumps(official_periods[period], allow_nan=False))
        print(f"BENCHMARK 1/N {period.upper()}: " + json.dumps(benchmark_periods[period], allow_nan=False))
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": official.index[0].date().isoformat(),
                       "end": official.index[-1].date().isoformat()},
        **official_periods,
        "benchmark": {"name": "1/N same ETF universe", **benchmark_periods,
                      **benchmark_periods["full"]},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False), flush=True)


def stop_on_timeout(_signum, _frame):
    raise TimeoutError("The backtest reached its time limit")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_on_timeout)
    signal.alarm(840)
    try:
        main()
    finally:
        signal.alarm(0)
