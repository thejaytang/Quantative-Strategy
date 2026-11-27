# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 586_bayesian_portfolio_optimization_by_predi\nuniverse: [SPY, EFA, EEM, VNQ, TLT, IEF, GLD, DBC]\nrisk_free_proxy: BIL\n\ndates:\n  data_start: 2008-01-01\n  start: 2014-01-01\n  is_end: 2018-12-31\n  end: 2025-12-31\n\nmodel:\n  short_mean_months: 12\n  long_lookback_months: 36\n  ar_orders: [1, 2, 3]\n  min_filter_updates: 12\n  state_discount: 0.97\n  residual_discount: 0.95\n  prior_intercept_sd: 0.005\n  prior_coefficient_sd: 0.25\n  risk_aversion: 6.0\n  max_weight: 0.45\n\nexecution:\n  lag_sessions: 1\n  cost_bps: 5.0\n  slippage_bps: 5.0\n  initial_value: 100000.0\n\nrobustness:\n  state_discounts: [0.95, 0.99]\n  higher_cost_bps: 20.0\n  higher_slippage_bps: 15.0\n'
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
"""Research backtest for Bayesian Portfolio Optimization by Predictive Synthesis.

Paper, Kato, Baba, Kaibuchi and Inokuchi (2025): https://arxiv.org/abs/2510.07180
Methods consulted, with independently written Python code:
- McAlinn et al., MATLAB replication (no explicit reuse license in repository):
  https://github.com/jasa-acs/Multivariate-Bayesian-Predictive-Synthesis-in-Macroeconomic-Forecasting
- McAlinn and West, author's replication page:
  https://www2.stat.duke.edu/~mw/mwsoftware/BPS/index.html
- Rilling, dynbps (Apache-2.0): https://pypi.org/project/dynbps/
  https://github.com/josephrilling/dynbps
"""

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
CACHE = ROOT / "data_cache"
BAR_FIELDS = ("Open", "Close", "Dividends", "Capital Gains", "Stock Splits")
SESSIONS_PER_YEAR = 252


def load_bars(tickers, start, end):
    request_key = f"yf-actions-v1|{','.join(tickers)}|{start.date()}|{end.date()}"
    cache_key = hashlib.sha256(request_key.encode()).hexdigest()[:16]
    cache_path = CACHE / f"daily_{cache_key}.csv.gz"
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as source:
            fetched_at = source.readline().strip()
            bars = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True,
                               float_precision="round_trip")
    else:
        CACHE.mkdir(exist_ok=True)
        # Keep the library's transient session data in the workspace and out of the bar cache.
        with TemporaryDirectory(dir=CACHE) as session_dir:
            yf.set_tz_cache_location(session_dir)
            bars = yf.download(
                tickers, start=start.strftime("%Y-%m-%d"),
                end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=False, actions=True, repair=False,
                progress=False, threads=False, timeout=15,
            )
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if bars.empty:
            raise RuntimeError("Yahoo Finance returned no daily bars")
        bars = bars.sort_index()
        for ticker in tickers:
            for field in BAR_FIELDS:
                if field == "Capital Gains" and (field, ticker) not in bars:
                    bars[field, ticker] = 0.0  # Yahoo omits the field for some instruments.
        bars = bars.loc[:, pd.MultiIndex.from_product([BAR_FIELDS, tickers])]

    if (bars.empty or not bars.index.is_unique or not bars.index.is_monotonic_increasing
            or bars.index[0] < start or bars.index[-1] > end
            or (bars.index[0] - start).days > 7 or (end - bars.index[-1]).days > 7):
        raise RuntimeError("Yahoo Finance session coverage does not match the configured dates")
    prices = bars[["Open", "Close"]]
    if not np.isfinite(bars.to_numpy()).all() or (prices <= 0).any().any():
        raise RuntimeError("Yahoo Finance returned missing or invalid bars on the common calendar")
    if not cache_path.exists():
        with NamedTemporaryFile(dir=CACHE, suffix=".csv.gz", delete=False) as target:
            temp_path = Path(target.name)
        try:
            with gzip.open(temp_path, "wt", encoding="utf-8") as target:
                target.write(fetched_at + "\n")
                bars.to_csv(target)
            temp_path.replace(cache_path)
        finally:
            temp_path.unlink(missing_ok=True)
    return bars, fetched_at


def total_return_prices(bars):
    """Use Yahoo split-normalized prices and reinvest distributions at the ex-date open."""
    opening = bars["Open"]
    closing = bars["Close"]
    distributions = bars["Dividends"] + bars["Capital Gains"]
    if (distributions < 0.0).any().any():
        raise RuntimeError("Yahoo Finance returned a negative distribution")
    factor = (1.0 + distributions / opening).cumprod()
    return opening * factor, closing * factor


def expert_forecasts(history, model):
    """Predict per-asset monthly means and their estimation variances."""
    short = history[-model["short_mean_months"]:]
    long = history[-model["long_lookback_months"]:]
    means = [short.mean(axis=0), long.mean(axis=0)]
    variances = [short.var(axis=0, ddof=1) / len(short), long.var(axis=0, ddof=1) / len(long)]
    for order in model["ar_orders"]:
        ar_mean = np.zeros(long.shape[1])
        ar_variance = np.zeros_like(ar_mean)
        for asset, series in enumerate(long.T):
            windows = np.lib.stride_tricks.sliding_window_view(series, order + 1)
            design = np.column_stack((np.ones(len(windows)), windows[:, -2::-1]))
            future = np.r_[1.0, series[-order:][::-1]]
            inverse = np.linalg.pinv(design)
            coefs = inverse @ windows[:, -1]
            errors = windows[:, -1] - design @ coefs
            residual_variance = errors @ errors / (len(windows) - order - 1)
            leverage = future @ inverse
            ar_mean[asset] = future @ coefs
            ar_variance[asset] = residual_variance * (leverage @ leverage)
        means.append(ar_mean)
        variances.append(ar_variance)
    return np.column_stack(means), np.column_stack(variances)


class PredictiveSynthesis:
    """Forward Gaussian moment filter for a multivariate latent-expert model."""

    def __init__(self, seed_history, model):
        self.model = model
        self.assets = seed_history.shape[1]
        self.experts = 2 + len(model["ar_orders"])
        asset_rows = np.arange(self.assets)[:, None]
        self.design_columns = asset_rows * (self.experts + 1) + np.arange(self.experts + 1)
        per_asset_mean = np.r_[0.0, np.full(self.experts, 1.0 / self.experts)]
        per_asset_sd = np.r_[model["prior_intercept_sd"],
                            np.full(self.experts, model["prior_coefficient_sd"])]
        self.state_mean = np.tile(per_asset_mean, self.assets)
        self.state_cov = np.diag(np.tile(per_asset_sd**2, self.assets))
        sample_cov = np.cov(seed_history, rowvar=False, ddof=1)
        self.residual_cov = sample_cov + 1e-8 * np.eye(self.assets)

    def moments(self, forecasts, variances):
        prior_cov = self.state_cov / self.model["state_discount"]
        design = np.zeros((self.assets, len(self.state_mean)))
        design[np.arange(self.assets)[:, None], self.design_columns] = np.column_stack(
            (np.ones(self.assets), forecasts)
        )
        coef_columns = self.design_columns[:, 1:]
        coef_second_moments = (self.state_mean**2 + np.diag(prior_cov))[coef_columns]
        latent_variance = np.sum(variances * coef_second_moments, axis=1)
        cross_cov = design @ prior_cov
        prediction_cov = cross_cov @ design.T + self.residual_cov + np.diag(latent_variance)
        return design @ self.state_mean, prediction_cov, prior_cov, cross_cov

    def update(self, outcome, prediction):
        mean, covariance, prior_cov, cross_cov = prediction
        innovation = outcome - mean
        solved = np.linalg.solve(covariance, np.column_stack((cross_cov, self.residual_cov, innovation)))
        state_size = len(self.state_mean)
        state_solve = solved[:, :state_size]
        residual_solve = solved[:, state_size:-1]
        innovation_solve = solved[:, -1]

        self.state_mean += cross_cov.T @ innovation_solve
        state_cov = prior_cov - cross_cov.T @ state_solve
        self.state_cov = (state_cov + state_cov.T) * 0.5

        noise_mean = self.residual_cov @ innovation_solve
        noise_cov = self.residual_cov - self.residual_cov @ residual_solve
        noise_second_moment = noise_cov + np.outer(noise_mean, noise_mean)
        retention = self.model["residual_discount"]
        updated = retention * self.residual_cov + (1.0 - retention) * noise_second_moment
        self.residual_cov = (updated + updated.T) * 0.5


def select_weights(mean, covariance, model):
    """Maximize expected quadratic utility on the capped long-only simplex."""
    penalty = model["risk_aversion"] * (covariance + np.outer(mean, mean))
    equal = np.full(len(mean), 1.0 / len(mean))
    result = minimize(
        lambda w: 100.0 * (0.5 * w @ penalty @ w - w @ mean),
        equal,
        jac=lambda w: 100.0 * (penalty @ w - mean),
        bounds=[(0.0, model["max_weight"])] * len(mean),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0,
                     "jac": lambda w: np.ones_like(w)},
        method="SLSQP",
        options={"ftol": 1e-11, "maxiter": 150},
    )
    if not result.success or abs(result.x.sum() - 1.0) > 1e-8:
        raise RuntimeError(f"Quadratic utility optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, model["max_weight"])
    return weights / weights.sum()


def month_end_closes(closes):
    # Only a month with a later session can produce a fill. A truncated final month cannot fill.
    return closes.groupby(closes.index.to_period("M")).tail(1)


def allocation_targets(monthly_closes, model):
    monthly_returns = monthly_closes.pct_change().iloc[1:]
    values = monthly_returns.to_numpy()
    window = model["long_lookback_months"]
    if len(values) < window + model["min_filter_updates"]:
        raise RuntimeError("Too little history for the experts and filter warm-up")
    synthesis = PredictiveSynthesis(values[:window], model)
    pending_prediction = None
    targets = {}
    for i in range(window - 1, len(values)):
        if pending_prediction is not None:
            synthesis.update(values[i], pending_prediction)
        forecasts = expert_forecasts(values[i - window + 1:i + 1], model)
        pending_prediction = synthesis.moments(*forecasts)
        if i - (window - 1) >= model["min_filter_updates"]:
            targets[monthly_returns.index[i]] = select_weights(*pending_prediction[:2], model)
    return targets


def rebalance(positions, cash, prices, weights, friction):
    previous_values = positions * prices
    open_value = cash + previous_values.sum()
    # Solve invested value plus both-sided friction equals the opening budget.
    lower, upper = 0.0, open_value
    for _ in range(48):
        middle = (lower + upper) * 0.5
        amount = np.abs(middle * weights - previous_values).sum()
        if middle + friction * amount <= open_value:
            lower = middle
        else:
            upper = middle
    target_values = lower * (1.0 - 1e-12) * weights
    changes = np.abs(target_values - previous_values)
    traded = changes.sum()
    cash = open_value - (target_values.sum() + traded * friction)
    if cash < 0.0:
        raise RuntimeError("Opening rebalance exceeded the cash available")
    return target_values / prices, cash, traded / open_value, np.count_nonzero(changes > 1e-10 * open_value)


def simulate(opening, closing, start, targets, execution):
    sessions = closing.index
    lag = execution["lag_sessions"]
    start_index = sessions.searchsorted(start)
    schedule = {}
    for month_end, weights in targets.items():
        fill_index = sessions.get_loc(month_end) + lag
        if start_index <= fill_index < len(sessions):
            schedule[fill_index] = (month_end, weights)
    if not schedule:
        raise RuntimeError("No opening allocation in the evaluation period")

    friction = (execution["cost_bps"] + execution["slippage_bps"]) / 10000.0
    open_prices, close_prices = opening.to_numpy(), closing.to_numpy()
    positions = np.zeros(closing.shape[1])
    cash = previous_value = execution["initial_value"]
    daily_rows = []
    for i in range(start_index, len(sessions)):
        turn = trades = 0
        signal_date = pd.NaT
        if i in schedule:
            signal_date, weights = schedule[i]
            positions, cash, turn, trades = rebalance(positions, cash, open_prices[i], weights, friction)
        close_value = cash + positions @ close_prices[i]
        daily_rows.append((close_value / previous_value - 1.0, turn, trades, cash, signal_date))
        previous_value = close_value
    return pd.DataFrame(daily_rows, index=sessions[start_index:],
                        columns=["return", "turnover", "n_trades", "cash", "signal_date"])


def period_metrics(daily_results, bill_returns):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise RuntimeError("Each evaluation period needs at least two sessions")
    wealth = np.r_[1.0, np.cumprod(1.0 + returns)]
    excess = returns - bill_returns.loc[returns.index]
    vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    scale = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(wealth[-1]**(SESSIONS_PER_YEAR / len(returns)) - 1.0),
        "ann_vol": float(vol * scale),
        "sharpe": float(scale * returns.mean() / vol) if vol > 0.0 else None,
        "sharpe_excess": float(scale * excess.mean() / excess_vol) if excess_vol > 0.0 else None,
        "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0)),
        "total_return": float(wealth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def summarize(daily_results, bill_returns, is_end):
    return {
        "is": period_metrics(daily_results.loc[:is_end], bill_returns),
        "oos": period_metrics(daily_results.loc[daily_results.index > is_end], bill_returns),
        "full": period_metrics(daily_results, bill_returns),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    universe, bills = config["universe"], config["risk_free_proxy"]
    dates = {key: pd.Timestamp(value) for key, value in config["dates"].items()}
    start, is_end = dates["start"], dates["is_end"]
    model, execution, robustness = config["model"], config["execution"], config["robustness"]
    if not dates["data_start"] < start <= is_end < dates["end"]:
        raise ValueError("Use chronological warm-up, IS and OOS dates")
    if len(set(universe)) != len(universe) or len(universe) < 2 or bills in universe or bills != "BIL":
        raise ValueError("Specify distinct ETFs and use BIL as the separate risk-free proxy")
    if (not 1.0 / len(universe) < model["max_weight"] <= 1.0
            or not 0.0 < model["state_discount"] <= 1.0
            or not 0.0 < model["residual_discount"] < 1.0):
        raise ValueError("Invalid weight limit or state discount")
    if not isinstance(execution["lag_sessions"], int) or execution["lag_sessions"] < 1:
        raise ValueError("The execution lag must be at least one full session index")
    window = model["long_lookback_months"]
    if (not 2 <= model["short_mean_months"] <= window or model["min_filter_updates"] < 0
            or any(not 1 <= order or 2 * order + 1 >= window for order in model["ar_orders"])):
        raise ValueError("Insufficient observations for the configured experts")
    if (min(execution["cost_bps"], execution["slippage_bps"], robustness["higher_cost_bps"],
            robustness["higher_slippage_bps"]) < 0.0 or execution["initial_value"] <= 0.0
            or max(execution["cost_bps"] + execution["slippage_bps"],
                   robustness["higher_cost_bps"] + robustness["higher_slippage_bps"]) >= 10000.0):
        raise ValueError("Invalid execution costs or initial value")

    bars, fetched_at = load_bars([*universe, bills], dates["data_start"], dates["end"])
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print("PRICE_ADJUSTMENT: Yahoo split-normalized OHLC plus forward ex-date open distribution reinvestment")

    costs = {**execution, "cost_bps": robustness["higher_cost_bps"],
             "slippage_bps": robustness["higher_slippage_bps"]}
    trials = [("default", model, execution)]
    trials.extend((f"state_discount={discount:g}", {**model, "state_discount": discount}, execution)
                  for discount in robustness["state_discounts"])
    trials.append(("higher_friction", model, costs))
    is_open, is_close = total_return_prices(bars.loc[:is_end])
    is_bill_returns = is_close[bills].pct_change()
    is_monthly = month_end_closes(is_close[universe])
    is_default = None
    print("ROBUSTNESS: IS only, fixed default")
    for label, trial_model, trial_execution in trials:
        trial_results = simulate(is_open[universe], is_close[universe], start,
                                 allocation_targets(is_monthly, trial_model), trial_execution)
        if label == "default":
            is_default = trial_results
        print(f"ROBUSTNESS {label}: {json.dumps(period_metrics(trial_results, is_bill_returns), allow_nan=False)}")

    open_all, close_all = total_return_prices(bars)
    opening, closing = open_all[universe], close_all[universe]
    bill_returns = close_all[bills].pct_change()
    monthly = month_end_closes(closing)
    targets = allocation_targets(monthly, model)
    daily_results = simulate(opening, closing, start, targets, execution)
    pd.testing.assert_frame_equal(is_default, daily_results.loc[:is_end], check_exact=True)
    equal = np.full(len(universe), 1.0 / len(universe))
    benchmark_results = simulate(opening, closing, start, dict.fromkeys(targets, equal), execution)
    metrics = summarize(daily_results, bill_returns, is_end)
    benchmark = summarize(benchmark_results, bill_returns, is_end)
    if metrics["is"]["n_trades"] < 1 or metrics["oos"]["n_trades"] < 1:
        raise RuntimeError("Both periods need simulated fills")

    fills = daily_results.dropna(subset=["signal_date"])
    observed_lag = closing.index.get_indexer(fills.index) - closing.index.get_indexer(fills["signal_date"])
    if not np.all(observed_lag == execution["lag_sessions"]):
        raise RuntimeError("Opening fills failed the signal lag check")
    print(f"EXECUTION_AUDIT: first_signal={fills['signal_date'].iloc[0].date()}, "
          f"first_fill={fills.index[0].date()}, lag_sessions={observed_lag.min()}, "
          f"min_cash={daily_results['cash'].min():.12g}, "
          f"benchmark_min_cash={benchmark_results['cash'].min():.12g}, IS_prefix_match=true")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(metrics[period], allow_nan=False)}")
        print(f"BENCHMARK_{period.upper()}: {json.dumps(benchmark[period], allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {len(trials)}")
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": str(daily_results.index[0].date()), "end": str(daily_results.index[-1].date())},
        **metrics,
        "benchmark": {"name": "monthly_equal_weight", **benchmark["full"], **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    main()
