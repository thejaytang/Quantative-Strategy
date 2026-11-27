"""649_dynamic_factor_models_with_forward_looki
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/649_dynamic_factor_models_with_forward_looki.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 9532cd9f815cc2d43ceb143b2f1416acf34e9864 on 2026-11-27 19:49 Europe/Oslo, exit 0, output sha256 ab79c5a554a1d06db991d5a398e3873a0eee89a12f788ba162d6e63441c91947
Data source: Yahoo Finance via yfinance, daily bars. The portfolio uses SPY, DBC, LQD, VNQ and TLT opens and closes. BIL daily close returns are used for excess-return statistics. The factors are the 10-year Treasury yield index (^TNX) close and the 10-year minus 5-year spread (^TNX − ^FVX), computed from paired index closes.
Data fetched: 2026-11-27T18:49:28.275828+00:00
Sample period: The run reports 2010-01-04 through 2025-12-31 inclusive. The history request starts at 2005-01-01 for warmup. The ETF calendar starts at common close coverage.   data_range (from output): {'start': '2010-01-04', 'end': '2025-12-31'}
IS/OOS split: IS trading sessions run from 2010-01-04 through 2017-12-29. OOS sessions run from 2018-01-02 through 2025-12-31. The configured cutoff is 2017-12-31, and results are sliced from one continuous portfolio simulation. OOS model estimates roll forward monthly with fixed model settings. The six robustness variants run through the IS cutoff.
Costs and slippage: The default models 2 bps of fees and 3 bps of slippage per side, charged together as 5 bps of each buy or sell notional. The sizing solver reserves these costs. Sell proceeds are reduced and purchase cash outlay is increased by the combined charge. Slippage is represented by that cash charge while fills use the opening price.
Signal lag: The signal uses ETF closes on the first trading session of each month. The target becomes eligible at the next ETF session’s open. The latest eligible paired Treasury index closes are dated on or before the preceding ETF session, with backward lookup limited to seven calendar days before that cutoff.
Price adjustment: yfinance auto_adjust=True total-return ETF opens and closes; raw Treasury yield-index closes
Survivorship bias: The five ETFs form a fixed universe and were listed before the evaluation start according to the README. The retrospective selection uses surviving ETFs and excludes delisted funds. Common close coverage determines the history start.
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.110251819131014, ann_vol=0.09570261740516442, sharpe=1.1409881522102114, sharpe_excess=1.1339783941161912, max_drawdown=-0.13729136871359815, total_return=1.30585086651754, turnover=4.8723778344627195, n_trades=336
    oos: cagr=0.03713738334546712, ann_vol=0.12888756975088866, sharpe=0.34762636697604254, sharpe_excess=0.1591353938204793, max_drawdown=-0.3216039584104794, total_return=0.33775350191582776, turnover=4.349545353247394, n_trades=326
    full: cagr=0.07309023582883323, ann_vol=0.113510162731495, sharpe=0.678495519282865, sharpe_excess=0.5689183854769118, max_drawdown=-0.32160395841047973, total_return=2.0846600715794823, turnover=4.611091522404066, n_trades=662
    benchmark: cagr=0.06550049018230153, ann_vol=0.09396512066110572, sharpe=0.7223781228356871, sharpe_excess=0.5899568071936522, max_drawdown=-0.22328088684857883, total_return=1.7541256019560016, turnover=0.36724378426335663, n_trades=960
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 649_dynamic_factor_models_with_forward_looki\nuniverse: [SPY, DBC, LQD, VNQ, TLT]\nexcess_reference: BIL\nyield_indices:\n  long: '^TNX'\n  short: '^FVX'\ndates:\n  warmup_start: 2005-01-01\n  start: 2010-01-04\n  is_end: 2017-12-31\n  end: 2025-12-31\nexecution:\n  initial_cash: 100000\n  cost_bps: 2.0\n  slippage_bps: 3.0\n  signal_lag_sessions: 1\n  yield_lag_sessions: 1\n  max_yield_age_days: 7\nmodel:\n  lookback_months: 36\n  daily_view_window: 252\n  view_horizon_months: 3\n  view_sessions_per_month: 21\n  view_noise: 0.05\n  ar_max_slope: 0.99\n  covariance_shrinkage: 0.15\n  risk_aversion: 8.0\n  max_weight: 0.45\nrobustness:\n  - {name: lookback_30_months, model: {lookback_months: 30}}\n  - {name: lookback_42_months, model: {lookback_months: 42}}\n  - {name: view_noise_0.025, model: {view_noise: 0.025}}\n  - {name: view_noise_0.10, model: {view_noise: 0.10}}\n  - {name: no_views, model: {view_noise: null}}\n  - {name: double_friction, execution: {cost_bps: 4.0, slippage_bps: 6.0}}\n"
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
"""Discrete dynamic factor allocation for paper C230.

Paper: https://arxiv.org/abs/2509.11528
Author research page: https://anasabdelhakmi.github.io/
PyPortfolioOpt, MIT (method reference only):
  https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/pypfopt/black_litterman.py
  https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/cookbook/4-Black-Litterman-Allocation.ipynb
  https://pypi.org/project/pyportfolioopt/
  https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/LICENSE
QuantConnect LEAN, Apache-2.0 (method reference only):
  https://github.com/QuantConnect/Lean/blob/master/Algorithm.Framework/Portfolio/BlackLittermanOptimizationPortfolioConstructionModel.py
  https://github.com/QuantConnect/Lean/blob/master/LICENSE
"""

import gzip
import hashlib
import json
from datetime import datetime, timezone
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


def download_prices(tickers, fields, start, end, adjusted):
    """Save the source retrieval time with each adjusted or raw price snapshot."""
    request = json.dumps([tickers, fields, start, end, adjusted])
    key = hashlib.sha256(request.encode()).hexdigest()[:16]
    path = CACHE / f"yahoo_{key}.csv.gz"
    columns = pd.MultiIndex.from_product([fields, tickers], names=["Price", "Ticker"])

    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as source:
            fetched_at = datetime.fromisoformat(source.readline().strip())
            prices = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True)
        return prices.loc[:, columns], fetched_at

    for _ in range(2):
        prices = yf.download(
            tickers, start=start, end=end, interval="1d", auto_adjust=adjusted,
            progress=False, threads=False, timeout=12,
        ).reindex(columns=columns)
        prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
        prices = prices.sort_index()
        if not prices.empty and prices.index.is_unique and prices.notna().any().all():
            last_dates = prices.apply(pd.Series.last_valid_index)
            if (pd.Timestamp(end) - last_dates).max() <= pd.Timedelta(days=7):
                break
    else:
        raise RuntimeError(f"Incomplete Yahoo Finance download for {tickers}")

    fetched_at = datetime.now(timezone.utc)
    with NamedTemporaryFile(dir=CACHE, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8") as destination:
            destination.write(fetched_at.isoformat() + "\n")
            prices.to_csv(destination)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def prepare_market_data(config, start, end):
    CACHE.mkdir(exist_ok=True)
    universe = config["universe"]
    reference = config["excess_reference"]
    long_yield, short_yield = (config["yield_indices"][key] for key in ("long", "short"))
    warmup = str(config["dates"]["warmup_start"])
    end_exclusive = str((end + pd.Timedelta(days=1)).date())
    with TemporaryDirectory(dir=CACHE, prefix="yf_session_") as session_cache:
        yf.set_tz_cache_location(session_cache)
        etfs, etfs_fetched = download_prices(universe + [reference], ["Open", "Close"],
                                             warmup, end_exclusive, adjusted=True)
        yields, yields_fetched = download_prices([long_yield, short_yield], ["Close"],
                                                 warmup, end_exclusive, adjusted=False)

    first_common = max(etfs["Close"][symbol].first_valid_index() for symbol in universe)
    sessions = etfs["Close"][universe[0]].dropna().loc[first_common:end].index
    investable = etfs.loc[sessions, pd.IndexSlice[:, universe]]
    prices = investable.to_numpy()
    if not np.isfinite(prices).all() or (prices <= 0).any() or sessions[0] >= start:
        raise ValueError("Invalid or incomplete common ETF price history")

    # Index closes are observations on their own market dates. Keep only paired prints.
    paired = yields["Close"].dropna()
    daily_factors = pd.DataFrame({
        "ten_year": paired[long_yield],
        "ten_minus_five_year": paired[long_yield] - paired[short_yield],
    })
    reference_returns = etfs["Close"][reference].pct_change(fill_method=None)
    reference_returns = reference_returns.reindex(sessions).loc[start:end]
    if not np.isfinite(reference_returns).all():
        raise ValueError(f"Missing {reference} returns in the evaluation period")
    return investable, daily_factors, reference_returns, max(etfs_fetched, yields_fetched)


def fit_ar1(states, max_slope):
    """Fit independent AR(1) conditional means while retaining joint shocks."""
    earlier, later = states[:-1], states[1:]
    centered = earlier - earlier.mean(axis=0)
    variation = np.sum(centered**2, axis=0)
    slopes = np.divide(np.sum(centered * (later - later.mean(axis=0)), axis=0),
                       variation, out=np.zeros(states.shape[1]), where=variation > 0)
    slopes = np.clip(slopes, 0.0, max_slope)
    offset = later.mean(axis=0) - slopes * earlier.mean(axis=0)
    return offset, slopes


def predict_return_moments(monthly_states, realized_returns, daily_states, model):
    """Condition next-month returns on a noisy factor forecast several months ahead."""
    x = monthly_states[:-1]
    design = np.column_stack((np.ones(len(x)), x))
    coefficients = np.linalg.lstsq(design, realized_returns, rcond=None)[0]
    return_errors = realized_returns - design @ coefficients
    offset, slope = fit_ar1(monthly_states, model["ar_max_slope"])
    factor_errors = monthly_states[1:] - (offset + x * slope)

    joint_errors = np.column_stack((factor_errors, return_errors))
    joint_cov = np.cov(joint_errors, rowvar=False)
    shrink = model["covariance_shrinkage"]
    joint_cov = (1 - shrink) * joint_cov + shrink * np.diag(joint_cov.diagonal())
    factor_count = x.shape[1]
    factor_cov = joint_cov[:factor_count, :factor_count]
    return_cov = joint_cov[factor_count:, factor_count:]
    last_state = monthly_states[-1]
    expected_returns = coefficients[0] + last_state @ coefficients[1:]
    noise = model["view_noise"]
    if noise is None:
        return expected_returns, return_cov

    horizon = model["view_horizon_months"]
    prior_factor_mean = last_state.copy()
    prior_factor_cov = np.zeros_like(factor_cov)
    for _ in range(horizon):
        prior_factor_mean = offset + slope * prior_factor_mean
        prior_factor_cov = slope[:, None] * prior_factor_cov * slope + factor_cov

    view_offset, view_slope = fit_ar1(daily_states, model["ar_max_slope"])
    view = daily_states[-1].copy()
    for _ in range(horizon * model["view_sessions_per_month"]):
        view = view_offset + view_slope * view

    # Only the first month's shock is shared with next month's return. Propagate
    # its effect to the view horizon through the factor transition matrix.
    cross_cov = joint_cov[factor_count:, :factor_count] * slope ** (horizon - 1)
    innovation_cov = (1 + noise) * prior_factor_cov
    gain = np.linalg.solve(innovation_cov, cross_cov.T).T
    posterior_returns = expected_returns + gain @ (view - prior_factor_mean)
    posterior_cov = return_cov - gain @ cross_cov.T
    return posterior_returns, (posterior_cov + posterior_cov.T) / 2


def optimize_weights(expected_returns, covariance, model):
    size = len(expected_returns)
    cap = model["max_weight"]
    penalty = model["risk_aversion"] * covariance

    def objective(weights):
        return 100 * (0.5 * weights @ penalty @ weights - weights @ expected_returns)

    def gradient(weights):
        return 100 * (penalty @ weights - expected_returns)

    result = minimize(
        objective, np.full(size, 1 / size), jac=gradient, method="SLSQP",
        bounds=[(0, cap)] * size,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1,
                     "jac": lambda weights: np.ones_like(weights)},
        options={"ftol": 1e-12, "maxiter": 100},
    )
    if not result.success or not np.isclose(result.x.sum(), 1, atol=1e-9, rtol=0):
        raise RuntimeError(f"Portfolio optimizer failed: {result.message}")
    weights = np.clip(result.x, 0, cap)
    return weights / weights.sum()


def monthly_targets(closes, daily_factors, start, execution, model):
    sessions = closes.index
    months = sessions.to_period("M")
    signal_rows = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])
    signal_dates = sessions[signal_rows]
    factor_lag = execution["yield_lag_sessions"]
    fill_lag = execution["signal_lag_sessions"]
    has_cutoff = signal_rows >= factor_lag
    cutoff_dates = sessions[signal_rows[has_cutoff] - factor_lag]
    source_rows = np.full(len(signal_rows), -1, dtype=int)
    source_rows[has_cutoff] = daily_factors.index.get_indexer(
        cutoff_dates, method="pad", tolerance=pd.Timedelta(days=execution["max_yield_age_days"]),
    )
    monthly_closes = closes.to_numpy()[signal_rows]
    factors = daily_factors.to_numpy()
    targets = {}
    for i, signal_date in enumerate(signal_dates):
        fill_row = signal_rows[i] + fill_lag
        if signal_date < start or fill_row >= len(sessions):
            continue
        begin = i - model["lookback_months"]
        daily_end = source_rows[i] + 1
        daily_begin = daily_end - model["daily_view_window"]
        if begin < 0 or daily_begin < 0 or (source_rows[begin:i + 1] < 0).any():
            raise ValueError(f"Insufficient lagged factor history at {signal_date.date()}")

        states = factors[source_rows[begin:i + 1]]
        price_history = monthly_closes[begin:i + 1]
        returns = price_history[1:] / price_history[:-1] - 1
        mean, covariance = predict_return_moments(
            states, returns, factors[daily_begin:daily_end], model,
        )
        fill_date = sessions[fill_row]
        assert daily_factors.index[daily_end - 1] < signal_date < fill_date
        targets[fill_date] = optimize_weights(mean, covariance, model)
    return targets


def rebalance_at_open(positions, cash, opens, weights, friction):
    marked = positions * opens
    opening_value = cash + marked.sum()

    # Solve for the target allocation after reserving fees at these opening prices.
    investable = brentq(
        lambda value: value + friction * np.abs(value * weights - marked).sum() - opening_value,
        0, opening_value,
    )
    changes = investable * weights - marked
    changes[np.abs(changes) <= 1e-12 * opening_value] = 0
    sales = np.maximum(-changes, 0)
    purchases = np.maximum(changes, 0)
    available = cash + sales.sum() * (1 - friction)
    spend = purchases.sum() * (1 + friction)
    if spend > available:
        purchases *= available / spend
        spend = available
    next_cash = available - spend
    next_positions = (marked - sales + purchases) / opens
    assert next_cash >= 0 and (next_positions >= 0).all()
    turnover = (sales.sum() + purchases.sum()) / opening_value
    trades = np.count_nonzero(sales + purchases)
    return next_positions, next_cash, turnover, trades


def simulate(prices, targets, execution):
    positions = np.zeros(prices["Open"].shape[1])
    cash = previous_value = execution["initial_cash"]
    friction = (execution["cost_bps"] + execution["slippage_bps"]) / 10_000
    rows = []
    for date, opens, closes in zip(prices.index, prices["Open"].to_numpy(),
                                   prices["Close"].to_numpy()):
        turnover = trades = 0
        if date in targets:
            positions, cash, turnover, trades = rebalance_at_open(
                positions, cash, opens, targets[date], friction,
            )
        value = cash + positions @ closes
        rows.append((value / previous_value - 1, turnover, trades))
        previous_value = value
    return pd.DataFrame(rows, index=prices.index, columns=["return", "turnover", "trades"])


def period_metrics(daily_results, reference_returns):
    returns = daily_results["return"].to_numpy()
    excess = returns - reference_returns.loc[daily_results.index].to_numpy()
    growth = np.r_[1.0, np.cumprod(1 + returns)]
    years = len(returns) / SESSIONS_PER_YEAR
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    annualize = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(growth[-1] ** (1 / years) - 1),
        "ann_vol": float(volatility * annualize),
        "sharpe": float(returns.mean() * annualize / volatility) if volatility > 0 else None,
        "sharpe_excess": (float(excess.mean() * annualize / excess_volatility)
                          if excess_volatility > 0 else None),
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1)),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["trades"].sum()),
    }


def run_strategy(prices, daily_factors, start, end, model, execution):
    if execution["signal_lag_sessions"] < 1 or execution["yield_lag_sessions"] < 1:
        raise ValueError("Signal and yield lags must be at least one session")
    if min(execution["cost_bps"], execution["slippage_bps"]) < 0 or (
        execution["cost_bps"] + execution["slippage_bps"] >= 10_000
    ) or execution["initial_cash"] <= 0:
        raise ValueError("Invalid execution costs or initial cash")
    if model["max_weight"] * prices["Close"].shape[1] < 1:
        raise ValueError("The position cap does not allow full investment")
    history = prices.loc[:end]
    targets = monthly_targets(history["Close"], daily_factors.loc[:end], start, execution, model)
    daily_results = simulate(history.loc[start:], targets, execution)
    if not daily_results["trades"].any():
        raise RuntimeError("Evaluation produced no fills")
    return daily_results, targets


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    dates = config["dates"]
    start, is_end, end = (pd.Timestamp(dates[key]) for key in ("start", "is_end", "end"))
    if not pd.Timestamp(dates["warmup_start"]) < start < is_end < end:
        raise ValueError("Expected warmup start < evaluation start < IS end < OOS end")
    model, execution = config["model"], config["execution"]
    prices, factors, reference, fetched_at = prepare_market_data(config, start, end)
    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True total-return ETF opens and closes; "
          "raw Treasury yield-index closes")

    official, targets = run_strategy(prices, factors, start, end, model, execution)
    size = len(config["universe"])
    equal_weight = np.full(size, 1 / size)
    benchmark = simulate(prices.loc[start:end], dict.fromkeys(targets, equal_weight), execution)
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": str(official.index[0].date()),
                       "end": str(official.index[-1].date())},
    }
    for label, mask in (("is", official.index <= is_end), ("oos", official.index > is_end),
                        ("full", slice(None))):
        results[label] = period_metrics(official.loc[mask], reference)
        benchmark_metrics = period_metrics(benchmark.loc[mask], reference)
        print(f"{label.upper()} strategy {json.dumps(results[label], allow_nan=False)}")
        print(f"{label.upper()} benchmark equal_weight {json.dumps(benchmark_metrics, allow_nan=False)}")
        if label == "full":
            results["benchmark"] = benchmark_metrics

    print("ROBUSTNESS: IS only; default specification fixed")
    trials = 1
    for case in config["robustness"]:
        daily_results, _ = run_strategy(
            prices, factors, start, is_end,
            model | case.get("model", {}), execution | case.get("execution", {}),
        )
        trials += 1
        metrics = period_metrics(daily_results, reference.loc[:is_end])
        print(f"ROBUSTNESS {case['name']} {json.dumps(metrics, allow_nan=False)}")

    print(f"PARAM_COMBOS_TRIED: {trials}")
    print("STATUS: 已回测 | pending review")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
