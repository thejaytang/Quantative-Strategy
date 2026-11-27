# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 716_behavioral_probability_weighting_and_por\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nbenchmark: DIA\nrisk_free: BIL\n\ndata_start: '2007-06-01'\nstart: '2012-01-03'\nis_end: '2017-12-29'\nend: '2025-12-31'\nlag_sessions: 1\ncost_bps: 2.0\nslippage_bps: 3.0\n\nmodel:\n  lookback_sessions: 1008\n  confidence: 0.99\n  distortion_blend: 0.50\n  tail_penalty: 0.25\n  max_weight: 0.35\n\n# Seven fixed-Prelec trials preceded discovery of the overlapping strategy 713.\nprior_trials: 7\nrobustness:\n  - {name: distortion_blend_0.35, distortion_blend: 0.35}\n  - {name: distortion_blend_0.65, distortion_blend: 0.65}\n  - {name: tail_penalty_0.20, tail_penalty: 0.20}\n  - {name: tail_penalty_0.30, tail_penalty: 0.30}\n  - {name: uniform_scenario_probabilities, distortion_blend: 0.0}\n  - {name: triple_execution_costs, execution_cost_multiplier: 3.0}\n"
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
"""Rolling behavioral mean–CVaR allocation for C278.

Paper: https://arxiv.org/abs/2507.04208
Author page: https://ayush0345.github.io/Website/
NIG distribution: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norminvgauss.html
CVaR formulation: https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf
Research references, no code copied:
- PyPortfolioOpt EfficientCVaR (MIT):
  https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/pypfopt/efficient_frontier/efficient_cvar.py
  https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/LICENSE
  https://pypi.org/project/pyportfolioopt/
- Riskfolio-Lib ordered CVaR weights (BSD-3-Clause):
  https://github.com/dcajasn/Riskfolio-Lib/blob/master/riskfolio/src/OwaWeights.py
  https://github.com/dcajasn/Riskfolio-Lib/blob/master/LICENSE.txt
  https://pypi.org/project/riskfolio-lib/
- Louis Szeto's QuantConnect mean–CVaR example (formulation only):
  https://www.quantconnect.com/forum/discussion/11712/mean-cvar-portfolio-construction-model/
"""

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import linprog, minimize
from scipy.sparse import csc_matrix, eye, hstack
from scipy.stats import norminvgauss


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
PRICE_FIELDS = ["Open", "Close", "Distributions"]
SESSION_YEAR = 252


@dataclass(frozen=True)
class Model:
    lookback_sessions: int
    confidence: float
    distortion_blend: float
    tail_penalty: float
    max_weight: float


@dataclass(frozen=True)
class DailyMarket:
    open_relative: pd.DataFrame
    close_relative: pd.DataFrame
    distribution_relative: pd.DataFrame
    total_returns: pd.DataFrame


def load_prices(symbol, start, end):
    path = CACHE_DIR / f"yahoo_daily_v1_{symbol}_{start.date()}_{end.date()}.csv"
    if path.exists():
        with path.open() as source:
            fetched_at = datetime.fromisoformat(source.readline().strip())
            prices = pd.read_csv(source, index_col="Date", parse_dates=["Date"])
    else:
        history = yf.Ticker(symbol).history(
            start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
            interval="1d", auto_adjust=False, actions=True, repair=False, timeout=12,
        )
        prices = history[["Open", "Close"]].copy()
        prices["Distributions"] = history["Dividends"] + history.get("Capital Gains", 0.0)
        prices.index = prices.index.tz_localize(None).normalize()
        prices.index.name = "Date"
        fetched_at = datetime.now(timezone.utc)

    if (prices.empty or prices.columns.to_list() != PRICE_FIELDS
            or not prices.index.is_unique or not prices.index.is_monotonic_increasing
            or not np.isfinite(prices.to_numpy()).all()
            or (prices[["Open", "Close"]] <= 0).any().any()
            or (prices["Distributions"] < 0).any()
            or not start <= prices.index[0] <= start + pd.Timedelta(days=7)
            or not end - pd.Timedelta(days=7) <= prices.index[-1] <= end):
        raise ValueError(f"Invalid or incomplete Yahoo daily data for {symbol}")

    if not path.exists():
        with NamedTemporaryFile(mode="w", dir=CACHE_DIR, suffix=".tmp", delete=False) as target:
            temporary = Path(target.name)
            try:
                target.write(f"{fetched_at.isoformat()}\n")
                prices.to_csv(target)
                target.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    return prices, fetched_at


def load_market(symbols, start, end):
    CACHE_DIR.mkdir(exist_ok=True)
    yf.config.debug.hide_exceptions = False
    with TemporaryDirectory(prefix=".yfinance-", dir=CACHE_DIR) as client_cache:
        yf.set_tz_cache_location(client_cache)
        downloaded = {symbol: load_prices(symbol, start, end) for symbol in symbols}

    calendar = downloaded[symbols[0]][0].index
    for symbol, (prices, _) in downloaded.items():
        if not prices.index.equals(calendar):
            raise ValueError(f"Yahoo session calendar for {symbol} does not match {symbols[0]}")

    panels = {
        field: pd.DataFrame({symbol: downloaded[symbol][0][field] for symbol in symbols})
        for field in PRICE_FIELDS
    }
    opens, closes, distributions = (panels[field] for field in PRICE_FIELDS)
    previous_closes = closes.shift(1)
    market = DailyMarket(
        open_relative=opens / previous_closes,
        close_relative=closes / opens,
        distribution_relative=distributions / previous_closes,
        total_returns=(closes + distributions) / previous_closes - 1.0,
    )
    return market, max(fetched for _, fetched in downloaded.values())


def nig_parameters(shape, mean, std):
    log_a, skew_ratio = shape
    a = np.exp(log_a)
    unit_shape = np.sqrt(1.0 - skew_ratio**2)
    scale = std * np.sqrt(a * unit_shape**3)
    loc = mean - scale * skew_ratio / unit_shape
    return a, a * skew_ratio, loc, scale


def fit_nig(returns):
    mean, std = returns.mean(), returns.std()
    if not np.isfinite(std) or std <= 0.0:
        raise RuntimeError("NIG needs a nonconstant return series")
    standardized = (returns - mean) / std

    # Match the window's mean and variance exactly, and estimate the two shape parameters.
    def negative_log_likelihood(shape):
        return -norminvgauss.logpdf(standardized, *nig_parameters(shape, 0.0, 1.0)).mean()

    result = minimize(negative_log_likelihood, x0=[0.0, 0.0], method="L-BFGS-B",
                      bounds=[(np.log(0.20), np.log(60.0)), (-0.9, 0.9)],
                      options={"maxiter": 100, "ftol": 1e-10})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"NIG fit failed: {result.message}")
    return nig_parameters(result.x, mean, std)


def implied_probabilities(anchor_returns, pilot_returns):
    fits = (fit_nig(anchor_returns), fit_nig(pilot_returns))
    # Exponential tail rates give finite integration bounds. The grid is denser near the center.
    lower = min(loc - 26.0 * scale / (a + b) for a, b, loc, scale in fits)
    upper = max(loc + 26.0 * scale / (a - b) for a, b, loc, scale in fits)
    center = (anchor_returns.mean() + pilot_returns.mean()) / 2.0
    spread = min(anchor_returns.std(), pilot_returns.std())
    grid = center + spread * np.sinh(np.linspace(
        np.arcsinh((lower - center) / spread), np.arcsinh((upper - center) / spread), 4097,
    ))
    cdfs = []
    for params in fits:
        cdf = cumulative_trapezoid(norminvgauss.pdf(grid, *params), grid, initial=0.0)
        if abs(cdf[-1] - 1.0) > 2e-5 or not np.isfinite(cdf).all():
            raise RuntimeError("NIG CDF integration failed its probability mass check")
        cdfs.append(cdf / cdf[-1])
    anchor_cdf, pilot_cdf = cdfs

    _, group, counts = np.unique(anchor_returns, return_inverse=True, return_counts=True)
    empirical_levels = np.cumsum(counts)[:-1] / len(anchor_returns)
    quantiles = np.interp(empirical_levels, anchor_cdf, grid)
    distorted_levels = np.interp(quantiles, grid, pilot_cdf)
    group_mass = np.diff(distorted_levels, prepend=0.0, append=1.0)
    return (group_mass / counts)[group]


def solve_weights(returns, probabilities, confidence, max_weight, tail_penalty, mean_weight):
    scenarios = 100.0 * returns  # Percentage points improve the LP's scaling.
    n_scenarios, n_assets = scenarios.shape

    # Variables are asset weights, the loss threshold, and one excess loss per scenario.
    excess_loss_constraints = hstack(
        [-csc_matrix(scenarios), csc_matrix(-np.ones((n_scenarios, 1))),
         -eye(n_scenarios, format="csc")],
        format="csc",
    )
    objective = np.r_[
        -mean_weight * (probabilities @ scenarios), tail_penalty,
        tail_penalty * probabilities / (1.0 - confidence),
    ]
    full_allocation = csc_matrix(np.r_[np.ones(n_assets), np.zeros(n_scenarios + 1)][None, :])
    solution = linprog(
        objective, A_ub=excess_loss_constraints, b_ub=np.zeros(n_scenarios),
        A_eq=full_allocation, b_eq=[1.0],
        bounds=[(0.0, max_weight)] * n_assets + [(None, None)] + [(0.0, None)] * n_scenarios,
        method="highs", options={"time_limit": 4.0},
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    weights = np.clip(solution.x[:n_assets], 0.0, max_weight)
    return weights / weights.sum()


def strategy_targets(total_returns, universe, benchmark, sessions, lag, model):
    first = sessions.start
    months = total_returns.index[sessions].to_period("M")
    monthly_opens = np.flatnonzero(months[1:] != months[:-1]) + first + 1
    history_through_end = total_returns.iloc[:sessions.stop]
    asset_returns = history_through_end[universe].to_numpy()
    benchmark_returns = history_through_end[benchmark].to_numpy()
    uniform = np.full(model.lookback_sessions, 1.0 / model.lookback_sessions)
    targets = {}
    for fill in np.r_[first, monthly_opens]:
        signal = fill - lag
        history = slice(signal + 1 - model.lookback_sessions, signal + 1)
        if history.start < 1:
            raise ValueError(f"Insufficient history at {total_returns.index[signal].date()}")
        returns = asset_returns[history]
        try:
            probabilities = uniform
            if model.distortion_blend > 0.0:
                pilot = solve_weights(returns, uniform, model.confidence, model.max_weight,
                                      tail_penalty=1.0, mean_weight=0.0)
                implied = implied_probabilities(benchmark_returns[history], returns @ pilot)
                mix = model.distortion_blend
                probabilities = (1.0 - mix) * uniform + mix * implied
            targets[fill] = solve_weights(returns, probabilities, model.confidence, model.max_weight,
                                          tail_penalty=model.tail_penalty, mean_weight=1.0)
        except RuntimeError as error:
            raise RuntimeError(f"Optimization failed at {total_returns.index[signal].date()}") from error
    return targets


def fill_at_open(carried, cash, weights, expense_rate):
    available = cash + carried.sum()
    low, high = 0.0, available
    # Find the largest allocation whose purchases plus both sides' fees fit the open budget.
    for _ in range(48):
        base = (low + high) / 2.0
        target = base * weights
        traded = np.abs(target - carried).sum()
        if target.sum() + expense_rate * traded <= available:
            low = base
        else:
            high = base

    target = low * weights
    traded_by_asset = np.abs(target - carried)
    traded = traded_by_asset.sum()
    cash = available - (target.sum() + expense_rate * traded)
    if cash < 0.0:
        raise RuntimeError("Opening fill exceeded available cash")
    return target, cash, traded / available, np.count_nonzero(traded_by_asset > available * 1e-10)


def simulate(market, symbols, sessions, targets, expense_rate):
    overnight, intraday, payouts = (
        frame.iloc[sessions][symbols].to_numpy()
        for frame in (market.open_relative, market.close_relative, market.distribution_relative)
    )
    positions = np.zeros(len(symbols))
    cash, previous_value = 1.0, 1.0
    daily_results = []
    for row, session in enumerate(range(sessions.start, sessions.stop)):
        carried = positions * overnight[row]
        trade_count, turnover = 0, 0.0
        if session in targets:
            carried, cash, turnover, trade_count = fill_at_open(
                carried, cash, targets[session], expense_rate,
            )
        # Ex-date proceeds accrue to the pre-open positions and compound in that ETF
        # at the close as a total-return convention. They cannot finance today's fill.
        positions = carried * intraday[row] + positions * payouts[row]
        value = cash + positions.sum()
        daily_results.append((value / previous_value - 1.0, turnover, trade_count))
        previous_value = value

    return pd.DataFrame(daily_results, index=market.total_returns.index[sessions],
                        columns=["return", "turnover", "n_trades"])


def metrics(daily_results, risk_free):
    returns = daily_results["return"]
    wealth = np.r_[1.0, np.cumprod(1.0 + returns.to_numpy())]
    excess = returns - risk_free.loc[returns.index]
    annual_scale = np.sqrt(SESSION_YEAR)
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    years = len(returns) / SESSION_YEAR
    return {
        "cagr": float(wealth[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(volatility * annual_scale),
        "sharpe": float(returns.mean() * annual_scale / volatility) if volatility > 0.0 else None,
        "sharpe_excess": float(excess.mean() * annual_scale / excess_volatility)
        if excess_volatility > 0.0 else None,
        "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0)),
        "total_return": float(wealth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, risk_free, is_end):
    return {
        "is": metrics(daily_results.loc[:is_end], risk_free),
        "oos": metrics(daily_results.loc[daily_results.index > is_end], risk_free),
        "full": metrics(daily_results, risk_free),
    }


def validate_model(model, n_assets):
    if (not isinstance(model.lookback_sessions, int) or model.lookback_sessions < 2
            or not 0.0 < model.confidence < 1.0 or not 0.0 <= model.distortion_blend <= 1.0
            or not np.isfinite(model.tail_penalty) or model.tail_penalty <= 0.0
            or not 1.0 / n_assets <= model.max_weight <= 1.0):
        raise ValueError(f"Invalid allocation parameters: {model}")


def main():
    with (ROOT / "config.yaml").open() as source:
        config = yaml.safe_load(source)
    universe, benchmark = config["universe"], config["benchmark"]
    risk_free_symbol = config["risk_free"]
    symbols = list(dict.fromkeys([benchmark, *universe, risk_free_symbol]))
    if not universe or len(set(universe)) != len(universe):
        raise ValueError("Universe must contain distinct symbols")
    data_start, start, is_end, end = (
        pd.Timestamp(config[key]) for key in ("data_start", "start", "is_end", "end")
    )
    if not data_start < start <= is_end < end:
        raise ValueError("Dates must provide warm-up and an out-of-sample period")
    lag = config["lag_sessions"]
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("lag_sessions must be a positive integer")
    cost, slippage = config["cost_bps"], config["slippage_bps"]
    if not (cost >= 0 and slippage >= 0 and cost + slippage < 10_000):
        raise ValueError("Invalid cost or slippage")
    expense_rate = (cost + slippage) / 10_000.0
    model = Model(**config["model"])
    validate_model(model, len(universe))

    market, fetched_at = load_market(symbols, data_start, end)
    date_index = market.total_returns.index
    first = date_index.searchsorted(start)
    is_stop = date_index.searchsorted(is_end, side="right")
    last = date_index.searchsorted(end, side="right")
    if is_stop - first < 2 or last - is_stop < 2:
        raise ValueError("Each evaluation period needs at least two sessions")
    full_sessions = slice(first, last)
    is_sessions = slice(first, is_stop)
    risk_free = market.total_returns[risk_free_symbol]

    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=False split-adjusted Open/Close; "
          "ex-date distributions recognized at the close for prior positions", flush=True)
    print(f"IS: {date_index[first].date()} to {date_index[is_stop - 1].date()}")
    print(f"OOS: {date_index[is_stop].date()} to {date_index[last - 1].date()}")
    print(f"ROBUSTNESS (IS only through {date_index[is_stop - 1].date()})", flush=True)

    tested = {(model, expense_rate)}
    for variant in config["robustness"]:
        overrides = dict(variant)
        name = overrides.pop("name")
        variant_rate = expense_rate * overrides.pop("execution_cost_multiplier", 1.0)
        variant_model = replace(model, **overrides)
        validate_model(variant_model, len(universe))
        if not 0.0 <= variant_rate < 1.0 or (variant_model, variant_rate) in tested:
            raise ValueError(f"Invalid or duplicate robustness configuration: {name}")
        targets = strategy_targets(market.total_returns, universe, benchmark, is_sessions, lag, variant_model)
        daily_results = simulate(market, universe, is_sessions, targets, variant_rate)
        print(f"  {name}: {json.dumps(metrics(daily_results, risk_free), allow_nan=False)}", flush=True)
        tested.add((variant_model, variant_rate))

    targets = strategy_targets(market.total_returns, universe, benchmark, full_sessions, lag, model)
    official = simulate(market, universe, full_sessions, targets, expense_rate)
    reference = simulate(market, [benchmark], full_sessions, {first: np.ones(1)}, expense_rate)
    official_stats = period_metrics(official, risk_free, is_end)
    benchmark_stats = period_metrics(reference, risk_free, is_end)
    if any(official_stats[period]["n_trades"] <= 0 for period in ("is", "oos")):
        raise RuntimeError("Default strategy must generate fills in both periods")

    print(f"CURRENT_SPEC_TRIALS: {len(tested)}")
    print(f"PARAM_COMBOS_TRIED: {config['prior_trials'] + len(tested)}")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()} STRATEGY: {json.dumps(official_stats[period], allow_nan=False)}")
        print(f"{period.upper()} {benchmark}: {json.dumps(benchmark_stats[period], allow_nan=False)}")
    result = {
        "strategy": config["strategy"],
        "data_range": [str(official.index[0].date()), str(official.index[-1].date())],
        **official_stats,
        "benchmark": {"ticker": benchmark, **benchmark_stats},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
