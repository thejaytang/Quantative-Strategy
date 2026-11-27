"""623_dynamic_causal_portfolio_choice_hedging
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/623_dynamic_causal_portfolio_choice_hedging.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 25a9179813de3addbbaee65a8c08579eb6e27d08 on 2026-11-27 19:10 Europe/Oslo, exit 0, output sha256 1624565b1ee6e1745599c71ee819dbf757356e5e58d20feca9cfae938f11ad03
Data source: Yahoo Finance public daily ETF open and close prices and corporate actions via yfinance, using date-keyed CSV caches. Allocated ETFs: SPY, EFA, TLT, GLD, DBC, VNQ. Cash proxy: BIL. Driver return spreads: IEF–SHY, TIP–IEF, HYG–IEF, UUP–BIL. The FRED reference in the README explains data vintages; the configured inputs are entirely ETF data.
Data fetched: 2026-11-27T18:10:36.840402+00:00
Sample period: 2011-01-03 through 2025-12-31, inclusive, as reported for the simulated sessions. The requested warm-up begins 2009-01-01.   data_range (from output): {'start': '2011-01-03', 'end': '2025-12-31'}
IS/OOS split: IS 2011-01-03 through 2018-12-31; OOS 2019-01-02 through 2025-12-31. This is a retrospective reporting split of one continuous default simulation. Positions carry over at the boundary and fixed hyperparameters are used with rolling estimates. The six robustness variants run on inputs truncated at the IS cutoff.
Costs and slippage: Default one-way commission is 3 bps and modelled slippage is 5 bps, for 8 bps on the absolute opening notional of each buy or sell. Both are deducted as cash costs in the opening budget, with fills priced at the open. One IS robustness variant uses 6 bps commission and 10 bps slippage.
Signal lag: At each month-end signal close t, the target uses data through that close. The simulated fill is at the open of the next trading session, which is the first trading session of the new month. The default lag is one trading session.
Price adjustment: yfinance auto_adjust=False, split-adjusted OHLC, explicit ex-date cash distributions
Survivorship bias: The six allocated ETFs are a fixed, retrospectively chosen universe of funds that still existed when documented. The README identifies survivorship bias and hindsight in universe selection.
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.0485245508985781, ann_vol=0.09366761414069785, sharpe=0.5528583704807224, sharpe_excess=0.523525007622871, max_drawdown=-0.14466871399509973, total_return=0.45982966920543844, turnover=3.8268964800840037, n_trades=401
    oos: cagr=0.14103847200677433, ann_vol=0.12075933080970108, sharpe=1.1532982211128138, sharpe_excess=0.9440078451523798, max_drawdown=-0.18901173312889308, total_return=1.5129999791989177, turnover=3.4046852301263284, n_trades=332
    full: cagr=0.09071878655877863, ann_vol=0.10718480038380405, sharpe=0.8639827452351844, sharpe_excess=0.7403345118519027, max_drawdown=-0.18901173312889175, total_return=2.6685519283472643, turnover=3.6298944122352474, n_trades=733
    benchmark  full: cagr=0.06846691933036975, ann_vol=0.10343232506671601, sharpe=0.6922028660544317, sharpe_excess=0.5638997069433328, max_drawdown=-0.22748835919864085, total_return=1.694654929090468, turnover=0.3898288071026507, n_trades=1080
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 623_dynamic_causal_portfolio_choice_hedging\nuniverse: [SPY, EFA, TLT, GLD, DBC, VNQ]\ncash_proxy: BIL\ndrivers:\n  term: [IEF, SHY]\n  inflation: [TIP, IEF]\n  credit: [HYG, IEF]\n  dollar: [UUP, BIL]\ndates:\n  warmup_start: 2009-01-01\n  start: 2011-01-03\n  is_end: 2018-12-31\n  end: 2025-12-31\nexecution:\n  signal_lag_sessions: 1\n  commission_bps: 3.0\n  slippage_bps: 5.0\nmodel:\n  lookback_sessions: 252\n  state_sessions: 63\n  forecast_sessions: 21\n  ridge_penalty: 0.10\n  max_state_persistence: 0.995\n  state_clip_sigma: 3.0\n  residual_variance_ridge: 0.000001\n  risk_aversion: 4.0\n  premium_weight: 0.25\n  max_weight: 0.40\n  hedge_strength: 1.0\nrobustness:\n  - name: lookback_210\n    model: {lookback_sessions: 210}\n  - name: lookback_294\n    model: {lookback_sessions: 294}\n  - name: hedge_0_5\n    model: {hedge_strength: 0.5}\n  - name: hedge_1_5\n    model: {hedge_strength: 1.5}\n  - name: static_only\n    model: {hedge_strength: 0.0}\n  - name: double_cost\n    execution: {commission_bps: 6.0, slippage_bps: 10.0}\n'
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
"""C211: state-dependent factor exposures and a forecast rotation hedge.

Paper: https://arxiv.org/abs/2607.06702
Author's replication package (MIT, methodological reference):
https://github.com/AlejandroRodriguezDominguez/Dynamic-Causal-Portfolio-Choice-Manifold-Hedging-and-the-Geometry-of-Information/tree/v2
https://doi.org/10.5281/zenodo.21226184
Riskfolio-Lib Tutorial 2 (BSD-3-Clause, methodological reference):
https://github.com/dcajasn/Riskfolio-Lib/blob/master/examples/Tutorial%202%20-%20Portfolio%20Optimization%20using%20Risk%20Factors%20and%20Stepwise%20Regression.ipynb
https://pypi.org/project/riskfolio-lib/
All implementation code was written for this strategy.
"""

import json
import signal
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
PRICE_COLUMNS = ["Open", "Close", "Distribution", "Stock Splits"]


def read_prices(symbol, start, end):
    cache_file = CACHE / f"yfinance_daily_v1_{symbol}_{start.date()}_{end.date()}.csv"
    if cache_file.exists():
        with cache_file.open(encoding="utf-8") as source:
            fetched_at = datetime.fromisoformat(source.readline().strip())
            prices = pd.read_csv(source, index_col="Date", parse_dates=["Date"])
    else:
        history = yf.Ticker(symbol).history(
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=True,
            repair=False,
            keepna=True,
            timeout=15,
        )
        if history.empty:
            raise RuntimeError(f"Yahoo Finance returned no daily data for {symbol}")
        prices = history[["Open", "Close", "Stock Splits"]].copy()
        prices["Distribution"] = history["Dividends"] + history.get("Capital Gains", 0)
        prices.index = prices.index.tz_localize(None).normalize().rename("Date")
        prices = prices.loc[start:end].sort_index()
        fetched_at = datetime.now(timezone.utc)
        with NamedTemporaryFile(mode="w", dir=CACHE, encoding="utf-8", delete=False) as output:
            buffer_path = Path(output.name)
            try:
                output.write(f"{fetched_at.isoformat()}\n")
                prices.to_csv(output)
                output.close()
                buffer_path.replace(cache_file)
            finally:
                buffer_path.unlink(missing_ok=True)

    if prices.empty or prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError(f"Invalid session index for {symbol}")
    return prices[PRICE_COLUMNS], fetched_at


def load_market(config, start, end):
    CACHE.mkdir(exist_ok=True)
    symbols = dict.fromkeys(config["universe"] + [config["cash_proxy"]])
    for pair in config["drivers"].values():
        symbols.update(dict.fromkeys(pair))

    with TemporaryDirectory(prefix="yfinance_", dir=CACHE) as provider_cache:
        yf.set_tz_cache_location(provider_cache)
        fetched = {symbol: read_prices(symbol, start, end) for symbol in symbols}
    calendar = fetched[config["universe"][0]][0].index
    if end - calendar[-1] > pd.Timedelta(days=7):
        raise ValueError("Price data end before the requested sample")
    for symbol, (prices, _) in fetched.items():
        if not prices.index.equals(calendar):
            raise ValueError(f"Sessions for {symbol} differ from the reference ETF")
        if not np.isfinite(prices.to_numpy()).all():
            raise ValueError(f"Missing or invalid market values for {symbol}")
        if (prices[["Open", "Close"]] <= 0).any().any() or (prices["Distribution"] < 0).any():
            raise ValueError(f"Invalid price or distribution for {symbol}")
        split = prices["Stock Splits"].ne(0)
        split_gap = (prices["Open"] / prices["Close"].shift(1) - 1)[split]
        if (split_gap.abs() > 0.25).any():
            raise ValueError(f"Check split normalization for {symbol}")

    market = {
        field: pd.DataFrame({symbol: prices[field] for symbol, (prices, _) in fetched.items()})
        for field in PRICE_COLUMNS
    }
    return market, max(fetched_at for _, fetched_at in fetched.values())


def close_total_returns(market):
    close = market["Close"]
    return (close + market["Distribution"]) / close.shift(1) - 1


def make_features(total_returns, config):
    spreads = pd.DataFrame({
        driver: total_returns[long] - total_returns[short]
        for driver, (long, short) in config["drivers"].items()
    })
    state = spreads.rolling(config["model"]["state_sessions"]).mean()
    prior_state = state.shift(1)
    innovations = spreads - prior_state
    excess = total_returns[config["universe"]].sub(total_returns[config["cash_proxy"]], axis=0)
    return state, prior_state, innovations, excess


def estimate_geometry(features, cutoff, model):
    state, prior_state, innovations, excess = features
    window = slice(cutoff + 1 - model["lookback_sessions"], cutoff + 1)
    lagged = prior_state.iloc[window].to_numpy()
    realized = state.iloc[window].to_numpy()
    shocks = innovations.iloc[window].to_numpy()
    returns = excess.iloc[window].to_numpy()
    if len(returns) != model["lookback_sessions"] or not np.isfinite(
        np.column_stack([lagged, realized, shocks, returns])
    ).all():
        raise ValueError("Insufficient history for the rolling factor model")

    state_mean = lagged.mean(axis=0)
    state_scale = lagged.std(axis=0, ddof=1)
    shock_scale = shocks.std(axis=0, ddof=1)
    if (state_scale <= 0).any() or (shock_scale <= 0).any():
        raise ValueError("A driver has zero variation in the estimation window")

    states = (lagged - state_mean) / state_scale
    next_states = (realized - state_mean) / state_scale
    shocks = (shocks - shocks.mean(axis=0)) / shock_scale
    interactions = (shocks[:, :, None] * states[:, None, :]).reshape(len(shocks), -1)
    design = np.column_stack([states, shocks, interactions])
    design -= design.mean(axis=0)
    centered_returns = returns - returns.mean(axis=0)
    penalty = len(returns) * model["ridge_penalty"] * np.eye(design.shape[1])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ centered_returns)
    residuals = centered_returns - design @ coefficients

    # Estimate a separate AR(1) intercept and slope for each observable state.
    next_mean = next_states.mean(axis=0)
    persistence = (states * (next_states - next_mean)).sum(axis=0) / (states ** 2).sum(axis=0)
    persistence = np.clip(persistence, 0, model["max_state_persistence"])
    current_state = next_states[-1]
    forecast_state = current_state.copy()
    for _ in range(model["forecast_sessions"]):
        forecast_state = next_mean + persistence * forecast_state

    drivers = shocks.shape[1]
    main_loadings = coefficients[drivers:2 * drivers].T
    state_loadings = coefficients[2 * drivers:].T.reshape(-1, drivers, drivers)
    clip = model["state_clip_sigma"]
    current_loadings = main_loadings + state_loadings @ current_state.clip(-clip, clip)
    future_loadings = main_loadings + state_loadings @ forecast_state.clip(-clip, clip)
    factor_cov = np.atleast_2d(np.cov(shocks, rowvar=False, ddof=1)) * SESSIONS_PER_YEAR
    residual_var = residuals.var(axis=0, ddof=1) * SESSIONS_PER_YEAR
    residual_cov = np.diag(residual_var + model["residual_variance_ridge"])
    cov = current_loadings @ factor_cov @ current_loadings.T + residual_cov
    premium = returns.mean(axis=0) * SESSIONS_PER_YEAR * model["premium_weight"]
    return current_loadings, future_loadings, factor_cov, cov, premium


def mean_variance(cov, premium, seed, max_weight):
    def objective(weights):
        gradient = cov @ weights - premium
        value = 0.5 * weights @ (gradient - premium)
        return value, gradient

    solution = minimize(
        objective, seed, jac=True, method="SLSQP",
        bounds=[(0, max_weight)] * len(seed),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
        options={"ftol": 1e-12, "maxiter": 150},
    )
    weights = solution.x
    if not solution.success or abs(weights.sum() - 1) > 1e-9:
        raise RuntimeError(f"Allocation failed: {solution.message}")
    return weights


def allocate(features, cutoff, model):
    current, future, factor_cov, cov, premium = estimate_geometry(features, cutoff, model)
    equal = np.full(len(premium), 1 / len(premium))
    max_weight = model["max_weight"]
    mv_premium = premium / model["risk_aversion"]
    static = mean_variance(cov, mv_premium, equal, max_weight)
    strength = model["hedge_strength"]
    if strength == 0:
        return static

    span, _, _ = np.linalg.svd(current, full_matrices=False)
    span = span[:, :np.linalg.matrix_rank(current)]
    change = future - current
    rotation = change - span @ (span.T @ change)
    forecast = current + rotation
    reference_exposure = current.T @ static
    hedge_risk = forecast @ factor_cov @ forecast.T
    hedge_premium = forecast @ factor_cov @ reference_exposure
    return mean_variance(cov + strength * hedge_risk, mv_premium + strength * hedge_premium,
                         static, max_weight)


def build_targets(features, calendar, start, model, lag):
    first_sessions = np.flatnonzero(~calendar.to_period("M").duplicated())
    targets = {}
    for fill_idx in first_sessions[calendar[first_sessions] >= start]:
        cutoff = fill_idx - lag
        if cutoff < 0:
            raise ValueError("Signal cutoff precedes available price history")
        targets[calendar[fill_idx]] = allocate(features, cutoff, model)
    return targets


def opening_fill(positions, cash, open_prices, weights, fee_rate):
    marked = positions * open_prices
    before_cost = cash + marked.sum()
    if (weights < 0).any() or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-9):
        raise ValueError("Target must be long-only and fully allocated")
    weights = weights / weights.sum()

    # The target capital plus fees must fit the value marked at the current open.
    low, high = 0.0, before_cost
    for _ in range(52):
        middle = (low + high) / 2
        changed = np.abs(middle * weights - marked).sum()
        if middle + fee_rate * changed <= before_cost:
            low = middle
        else:
            high = middle

    target_values = low * weights
    traded = np.abs(target_values - marked)
    cash_left = before_cost - (target_values.sum() + fee_rate * traded.sum())
    if cash_left < -1e-12 * before_cost:
        raise RuntimeError("Opening fill exceeds available funds")
    trades = int((traded > 1e-10 * before_cost).sum())
    return target_values / open_prices, max(0.0, cash_left), traded.sum() / before_cost, trades


def simulate(market, targets, universe, start, execution):
    open_prices = market["Open"].loc[start:, universe]
    close_prices = market["Close"].loc[start:, universe].to_numpy()
    distributions = market["Distribution"].loc[start:, universe].to_numpy()
    fee_rate = (execution["commission_bps"] + execution["slippage_bps"]) / 10_000
    dates = open_prices.index
    positions = np.zeros(len(universe))
    cash = previous_value = 1.0
    daily_rows = []
    for i, prices in enumerate(open_prices.to_numpy()):
        # Ex-date entitlement belongs to the units carried from the previous close.
        distribution_cash = positions @ distributions[i]
        target = targets.get(dates[i])
        turnover, trades = 0.0, 0
        if target is not None:
            positions, cash, turnover, trades = opening_fill(positions, cash, prices, target, fee_rate)
        cash += distribution_cash
        closing_value = cash + positions @ close_prices[i]
        daily_rows.append((closing_value / previous_value - 1, turnover, trades))
        previous_value = closing_value
    return pd.DataFrame(daily_rows, index=dates, columns=["return", "turnover", "n_trades"])


def metrics(daily_results, cash_returns):
    returns = daily_results["return"].to_numpy()
    excess = returns - cash_returns.loc[daily_results.index].to_numpy()
    n_days = len(returns)
    if n_days < 2 or not np.isfinite(excess).all():
        raise ValueError("Missing observations in a performance period")
    growth = np.r_[1.0, np.cumprod(1 + returns)]
    return_std, excess_std = returns.std(ddof=1), excess.std(ddof=1)
    root_year = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / n_days) - 1),
        "ann_vol": float(return_std * root_year),
        "sharpe": float(root_year * returns.mean() / return_std) if return_std else None,
        "sharpe_excess": float(root_year * excess.mean() / excess_std) if excess_std else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, cash_returns, split):
    return {
        "is": metrics(daily_results.loc[:split], cash_returns),
        "oos": metrics(daily_results.loc[daily_results.index > split], cash_returns),
        "full": metrics(daily_results, cash_returns),
    }


def validate_trial(model, execution, asset_count):
    if not 0 < model["max_weight"] <= 1 or asset_count * model["max_weight"] < 1:
        raise ValueError("Maximum asset weight makes the budget infeasible")
    if model["ridge_penalty"] <= 0 or model["risk_aversion"] <= 0 or model["hedge_strength"] < 0:
        raise ValueError("Invalid model penalty or risk preference")
    if not 0 <= model["max_state_persistence"] < 1 or model["state_clip_sigma"] <= 0:
        raise ValueError("Invalid state forecast bounds")
    if model["residual_variance_ridge"] <= 0:
        raise ValueError("Residual variance regularization must be positive")
    for key, minimum in (("lookback_sessions", 2), ("state_sessions", 1), ("forecast_sessions", 1)):
        if not isinstance(model[key], int) or model[key] < minimum:
            raise ValueError(f"Invalid model window: {key}")
    lag = execution["signal_lag_sessions"]
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("Execution needs at least one session of signal lag")
    costs = [execution["commission_bps"], execution["slippage_bps"]]
    if min(costs) < 0 or sum(costs) >= 10_000:
        raise ValueError("Invalid open execution costs")


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    dates = {key: pd.Timestamp(value) for key, value in config["dates"].items()}
    start, split, end = dates["start"], dates["is_end"], dates["end"]
    if not dates["warmup_start"] < start < split < end:
        raise ValueError("Check warm-up, in-sample and out-of-sample date boundaries")
    universe, model, execution = config["universe"], config["model"], config["execution"]
    validate_trial(model, execution, len(universe))
    market, fetched_at = load_market(config, dates["warmup_start"], end)
    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=False, split-adjusted OHLC, explicit ex-date cash distributions")

    is_market = {field: values.loc[:split] for field, values in market.items()}
    is_returns = close_total_returns(is_market)
    cash_proxy = config["cash_proxy"]
    trials = 0
    print("ROBUSTNESS (IS only; no parameter selection)")
    for variant in config["robustness"]:
        trial_model = model | variant.get("model", {})
        trial_execution = execution | variant.get("execution", {})
        validate_trial(trial_model, trial_execution, len(universe))
        trial_config = config | {"model": trial_model}
        targets = build_targets(make_features(is_returns, trial_config), is_returns.index, start,
                                trial_model, trial_execution["signal_lag_sessions"])
        daily = simulate(is_market, targets, universe, start, trial_execution)
        print(f"ROBUSTNESS_IS {variant['name']}: {json.dumps(metrics(daily, is_returns[cash_proxy]))}")
        trials += 1

    total_returns = close_total_returns(market)
    calendar = total_returns.index
    targets = build_targets(make_features(total_returns, config), calendar, start, model,
                            execution["signal_lag_sessions"])
    daily = simulate(market, targets, universe, start, execution)
    trials += 1
    benchmark_weights = np.full(len(universe), 1 / len(universe))
    benchmark_daily = simulate(market, dict.fromkeys(targets, benchmark_weights), universe, start, execution)
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": daily.index[0].date().isoformat(), "end": daily.index[-1].date().isoformat()},
        **period_metrics(daily, total_returns[cash_proxy], split),
        "benchmark": period_metrics(benchmark_daily, total_returns[cash_proxy], split),
    }
    if min(results[period]["n_trades"] for period in ("is", "oos")) <= 0:
        raise RuntimeError("Expected actual simulated fills in both performance periods")

    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(results[period])}")
        print(f"{period.upper()} BENCHMARK (monthly equal weight, same ETF universe): "
              f"{json.dumps(results['benchmark'][period])}")
    print("LOOK_AHEAD_CHECK: past-only close signal using published end-of-day market quotes; "
          "lagged open sizing; split-consistent prices and distributions; IS-only robustness inputs")
    print(f"PARAM_COMBOS_TRIED: {trials}")
    print(f"RESULTS_JSON: {json.dumps(results, allow_nan=False)}")


if __name__ == "__main__":
    signal.alarm(14 * 60)
    main()
