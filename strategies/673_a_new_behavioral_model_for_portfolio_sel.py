# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 673_a_new_behavioral_model_for_portfolio_sel\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nrisk_free_proxy: BIL\nbenchmark: equal_weight_sectors\ndata_start: 2008-01-01\nbacktest_start: 2010-01-04\noos_start: 2018-01-01\nend: 2025-12-31\n\nmodel:\n  lookback_sessions: 500\n  rebalance_sessions: 20\n  lag_sessions: 1\n  lambda_gain: 0.30\n  lambda_loss: 0.69\n  cost_bps: 3\n  slippage_bps: 2\n\nrobustness:\n  # Each row is an IS-only variant of the official model.\n  - {label: gain_0.29, lambda_gain: 0.29}\n  - {label: gain_0.31, lambda_gain: 0.31}\n  - {label: loss_0.68, lambda_loss: 0.68}\n  - {label: loss_0.70, lambda_loss: 0.70}\n  - {label: higher_friction, cost_bps: 9, slippage_bps: 6}\n'
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
"""Half-Full/Half-Empty sector allocation, research only.

Paper and author materials:
https://arxiv.org/abs/2312.10749
https://arxiv.org/pdf/2312.10749
https://www.francescocesarone.com/papers
https://www.francescocesarone.com/data-sets

Related implementations, consulted for ideas. This code was written independently.
skfolio (BSD-3-Clause):
https://raw.githubusercontent.com/skfolio/skfolio/main/src/skfolio/optimization/convex/_mean_risk.py
https://skfolio.org/auto_examples/online_learning/plot_3_online_portfolio_optimization_evaluation.html
https://pypi.org/project/skfolio/
https://raw.githubusercontent.com/skfolio/skfolio/main/LICENSE
QuantConnect Lean (Apache-2.0):
https://raw.githubusercontent.com/QuantConnect/Lean/master/Algorithm.Framework/Portfolio/MeanVarianceOptimizationPortfolioConstructionModel.py
https://raw.githubusercontent.com/QuantConnect/Lean/master/LICENSE
Data interface: https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
PRICE_ADJUSTMENT = (
    "yfinance auto_adjust=False, split-adjusted Open/Close with forward "
    "total returns and distribution reinvestment at ex-date open"
)


@dataclass(frozen=True)
class Model:
    lookback_sessions: int
    rebalance_sessions: int
    lag_sessions: int
    lambda_gain: float
    lambda_loss: float
    cost_bps: float
    slippage_bps: float

    def __post_init__(self):
        if type(self.lookback_sessions) is not int or self.lookback_sessions < 2:
            raise ValueError("lookback_sessions must be an integer of at least 2")
        for sessions in (self.rebalance_sessions, self.lag_sessions):
            if type(sessions) is not int or sessions < 1:
                raise ValueError("Rebalance spacing and signal lag must be positive session counts")
        if not (0 <= self.lambda_gain <= 1 and 0 <= self.lambda_loss <= 1):
            raise ValueError("HF/HE coefficients must be in [0, 1]")
        if not (0 <= self.cost_bps and 0 <= self.slippage_bps and self.friction < 1):
            raise ValueError("Cost and slippage must be nonnegative and total under 10000 bps")

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def load_bars(symbols, start, end):
    cache = ROOT / "data_cache"
    cache.mkdir(exist_ok=True)
    key = json.dumps([symbols, str(start.date()), str(end.date()), "split_ohlc_actions_v1"])
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    path = cache / f"yfinance_{digest}.csv.gz"
    from_cache = path.is_file()
    if from_cache:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            fetched_at = stream.readline().strip()
            bars = pd.read_csv(
                stream, header=[0, 1], index_col=0, parse_dates=True, dtype=object,
            ).astype(float)
        if datetime.fromisoformat(fetched_at).tzinfo is None:
            raise ValueError("Expected a timezone in the cached fetch timestamp")
    else:
        with TemporaryDirectory(dir=cache, prefix=".yfinance-") as client_cache:
            yf.set_tz_cache_location(client_cache)
            bars = yf.download(
                symbols, start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d", auto_adjust=False, actions=True, repair=False,
                keepna=True, ignore_tz=True, progress=False, threads=False, timeout=12,
            )
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    bars.index = pd.DatetimeIndex(bars.index).as_unit("ns")
    bars = bars.sort_index()
    if bars.empty or not bars.index.is_unique or bars.index.hasnans:
        raise ValueError("Expected nonempty yfinance data with one row per session")
    if (
        not ((bars.index >= start) & (bars.index <= end)).all()
        or bars.index[0] > start + pd.Timedelta(days=4)
        or bars.index[-1] < end - pd.Timedelta(days=4)
    ):
        raise ValueError("Price panel does not cover the requested dates")
    fields = ("Open", "Close", "Dividends", "Stock Splits")
    columns = pd.MultiIndex.from_product([fields, symbols], names=["Price", "Ticker"])
    capital_columns = pd.MultiIndex.from_product(
        [["Capital Gains"], symbols], names=columns.names,
    )
    bars = pd.concat([bars.reindex(columns=columns),
                      bars.reindex(columns=capital_columns, fill_value=0.0)], axis=1)
    values = bars.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all() or (values < 0).any()
        or (bars["Open"] <= 0).any().any() or (bars["Close"] <= 0).any().any()
    ):
        raise ValueError("Incomplete or invalid yfinance price/action panel")

    if not from_cache:
        with NamedTemporaryFile(dir=cache, prefix=".snapshot-", delete=False) as temp:
            pending = Path(temp.name)
        try:
            with gzip.open(pending, "wt", encoding="utf-8") as stream:
                stream.write(f"{fetched_at}\n")
                bars.to_csv(stream)
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
    return bars, fetched_at


def price_factors(bars):
    opens, closes = bars["Open"], bars["Close"]
    distributions = bars["Dividends"] + bars["Capital Gains"]
    # Only the positions carried from yesterday receive today's ex-date distribution.
    return (opens + distributions) / closes.shift(), closes / opens


def hfhe_score_and_gradient(weights, scenarios, lambda_gain, lambda_loss):
    """Equation (14), with a choice of derivative at each kink."""
    portfolio_returns = scenarios @ weights
    gains = np.maximum(portfolio_returns, 0.0)
    losses = np.minimum(portfolio_returns, 0.0)
    centered_gain = gains - gains.mean()
    centered_loss = losses - losses.mean()
    gain_coefficient = 2 * lambda_gain - 1
    loss_coefficient = 2 * lambda_loss - 1
    score = (
        portfolio_returns.mean() + gain_coefficient * np.abs(centered_gain).mean()
        + loss_coefficient * np.abs(centered_loss).mean()
    )

    gain_sign = np.sign(centered_gain)
    loss_sign = np.sign(centered_loss)
    scenario_derivatives = (
        1 + gain_coefficient * (gain_sign - gain_sign.mean()) * (portfolio_returns > 0)
        + loss_coefficient * (loss_sign - loss_sign.mean()) * (portfolio_returns < 0)
    )
    gradient = (scenario_derivatives @ scenarios) / len(scenarios)
    return score, gradient


def optimize_weights(returns, model):
    # A positive change of units leaves the maximizer unchanged.
    scenarios = 100.0 * returns
    assets = scenarios.shape[1]
    equal = np.full(assets, 1.0 / assets)
    starts = np.vstack([equal, np.eye(assets)])
    ones = np.ones(assets)

    def objective(weights):
        score, gradient = hfhe_score_and_gradient(
            weights, scenarios, model.lambda_gain, model.lambda_loss,
        )
        return -score, -gradient

    best = equal
    best_value = objective(best)[0]
    budget = {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: ones}
    for start in starts:
        solution = minimize(
            objective, start, jac=True, method="SLSQP", bounds=[(0, 1)] * assets,
            constraints=[budget], options={"maxiter": 100, "ftol": 1e-9},
        )
        # A feasible early iterate can still improve a nonsmooth objective.
        for candidate in (start, solution.x):
            if (not np.isfinite(candidate).all() or candidate.min() < -1e-8
                    or abs(candidate.sum() - 1.0) > 1e-8):
                continue
            candidate = np.where(candidate > 1e-10, candidate, 0.0)
            candidate /= candidate.sum()
            candidate_value = objective(candidate)[0]
            if candidate_value < best_value - 1e-12:
                best, best_value = candidate, candidate_value
    return best


def scheduled_weights(close_returns, first_fill, model):
    if first_fill - model.lag_sessions < model.lookback_sessions:
        raise ValueError("Insufficient pre-start observations for lag and rolling window")
    targets = {}
    values = close_returns.to_numpy(dtype=float)
    for fill_row in range(first_fill, len(close_returns), model.rebalance_sessions):
        signal_row = fill_row - model.lag_sessions
        window = values[signal_row - model.lookback_sessions + 1:signal_row + 1]
        targets[fill_row] = (signal_row, optimize_weights(window, model))
    return targets


def execute_at_open(open_positions, cash, weights, friction):
    opening_value = open_positions.sum() + cash
    # Solve post-cost invested value + friction × gross traded value <= opening value.
    # The budget function is increasing since the weights sum to one and friction < 1.
    lower, upper = 0.0, opening_value
    for _ in range(54):
        middle = (lower + upper) / 2
        proposed = middle * weights
        cost = friction * np.abs(proposed - open_positions).sum()
        if proposed.sum() + cost <= opening_value:
            lower = middle
        else:
            upper = middle
    positions = lower * weights
    traded = np.abs(positions - open_positions)
    cash = opening_value - (positions.sum() + friction * traded.sum())
    if cash < 0:
        raise ArithmeticError("Opening allocation exceeded cash budget")
    n_trades = np.count_nonzero(traded > 1e-10 * opening_value)
    return positions, cash, traded.sum() / opening_value, n_trades


def simulate(gap, intraday, first_fill, targets, model):
    gap_values = gap.to_numpy(dtype=float)
    intraday_values = intraday.to_numpy(dtype=float)
    positions = np.zeros(gap.shape[1])
    cash = previous_value = 1.0
    daily_results = []
    for row in range(first_fill, len(gap)):
        positions *= gap_values[row]
        turnover = n_trades = 0
        if row in targets:
            signal_row, weights = targets[row]
            if row - signal_row != model.lag_sessions:
                raise ValueError("Signal and fill rows disagree with the configured lag")
            positions, cash, turnover, n_trades = execute_at_open(
                positions, cash, weights, model.friction,
            )
        positions *= intraday_values[row]
        closing_value = positions.sum() + cash
        daily_results.append((closing_value / previous_value - 1, turnover, n_trades))
        previous_value = closing_value
    return pd.DataFrame(
        daily_results, index=gap.index[first_fill:], columns=["return", "turnover", "n_trades"],
    )


def summarize(daily_results, proxy):
    returns = daily_results["return"]
    excess = returns - proxy.loc[returns.index]
    growth = np.r_[1.0, (1 + returns.to_numpy()).cumprod()]
    volatility = returns.std()
    excess_volatility = excess.std()
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(volatility * annualizer),
        "sharpe": float(annualizer * returns.mean() / volatility) if volatility > 0 else None,
        "sharpe_excess": (float(annualizer * excess.mean() / excess_volatility)
                          if excess_volatility > 0 else None),
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, proxy, oos_start):
    in_sample = daily_results.index < oos_start
    return {
        "is": summarize(daily_results.loc[in_sample], proxy),
        "oos": summarize(daily_results.loc[~in_sample], proxy),
        "full": summarize(daily_results, proxy),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    model = Model(**cfg["model"])
    assets = cfg["universe"]
    proxy_symbol = cfg["risk_free_proxy"]
    symbols = [*assets, proxy_symbol]
    if len(assets) < 2 or len(set(symbols)) != len(symbols):
        raise ValueError("Expected distinct assets and one separate BIL return proxy")
    if cfg["benchmark"] != "equal_weight_sectors" or proxy_symbol != "BIL":
        raise ValueError("This experiment supports an equal-weight benchmark and BIL proxy")
    data_start, start, oos_start, end = (
        pd.Timestamp(cfg[key]) for key in ("data_start", "backtest_start", "oos_start", "end")
    )
    if not data_start < start < oos_start <= end:
        raise ValueError("Expected ordered sample dates including pre-start history")

    bars, fetched_at = load_bars(symbols, data_start, end)
    all_gap, all_intraday = price_factors(bars)
    gap, intraday = all_gap[assets], all_intraday[assets]
    first_fill = int(bars.index.searchsorted(start))
    split_row = int(bars.index.searchsorted(oos_start))
    if split_row - first_fill < 2 or len(bars) - split_row < 2:
        raise ValueError("Both reporting periods need at least two sessions")
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}", flush=True)

    # Pass only IS data into the variant evaluations.
    gap_is = gap.iloc[:split_row]
    intraday_is = intraday.iloc[:split_row]
    returns_is = gap_is * intraday_is - 1
    proxy_is = all_gap[proxy_symbol].iloc[:split_row] * all_intraday[proxy_symbol].iloc[:split_row] - 1
    variants_run = 0
    print("ROBUSTNESS (IS ONLY)", flush=True)
    for case in cfg["robustness"]:
        label = case["label"]
        changes = {key: value for key, value in case.items() if key != "label"}
        variant = replace(model, **changes)
        targets = scheduled_weights(returns_is, first_fill, variant)
        daily_results = simulate(gap_is, intraday_is, first_fill, targets, variant)
        print(label, json.dumps(summarize(daily_results, proxy_is), allow_nan=False), flush=True)
        variants_run += 1

    targets = scheduled_weights(gap * intraday - 1, first_fill, model)
    daily_results = simulate(gap, intraday, first_fill, targets, model)
    if daily_results["n_trades"].sum() == 0:
        raise RuntimeError("The default strategy produced no simulated trades")
    equal = np.full(len(assets), 1.0 / len(assets))
    benchmark_targets = {row: (source, equal) for row, (source, _) in targets.items()}
    benchmark_results = simulate(gap, intraday, first_fill, benchmark_targets, model)
    proxy = all_gap[proxy_symbol] * all_intraday[proxy_symbol] - 1
    official = period_metrics(daily_results, proxy, oos_start)
    reference = period_metrics(benchmark_results, proxy, oos_start)

    first_signal = targets[first_fill][0]
    print(f"FIRST_SIGNAL: {bars.index[first_signal].date()} close")
    print(f"FIRST_FILL: {bars.index[first_fill].date()} open")
    print("LOOK_AHEAD_SELF_CHECK: window ends at signal close; lag enforced; "
          "open-based sizing; ex-date distributions and split-adjusted ratios; "
          "IS-only variants; fixed official parameters")
    for name, metrics in official.items():
        print(f"{name.upper()}: {json.dumps(metrics, allow_nan=False)}")
    print(f"BENCHMARK ({cfg['benchmark']}): {json.dumps(reference, allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {1 + variants_run}")
    result = {
        "strategy": cfg["strategy"],
        "data_range": [str(daily_results.index[0].date()), str(daily_results.index[-1].date())],
        **official,
        "benchmark": {"name": cfg["benchmark"], **reference["full"], **reference},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


def abort_on_timeout(_signum, _frame):
    raise TimeoutError("Research run exceeded 14 minutes")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, abort_on_timeout)
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)
