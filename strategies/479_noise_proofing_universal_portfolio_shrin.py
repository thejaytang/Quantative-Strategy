"""479_noise_proofing_universal_portfolio_shrin
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/479_noise_proofing_universal_portfolio_shrin.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit cc50485fb68601d5f9b01f48262e3a778be06982 on 2026-11-27 09:02 Europe/Oslo, exit 0, output sha256 caf3c20dfbb1531a89a4a42e6becd49737d1cac216539d7d6c3b606536418421
Data source: Yahoo Finance daily ETF prices via yfinance for SPY, EFA, EEM, IWM, IEF, TLT, GLD and VNQ, with BIL as the cash-return proxy.
Data fetched: 2026-11-27T08:02:47+00:00
Sample period: The requested price history is 2008-01-02 through 2025-12-31, with data before 2015-01-02 used for warm-up. The logged evaluation runs from 2015-01-02 through 2025-12-31, inclusive.   data_range (from output): {'start': '2015-01-02', 'end': '2025-12-31'}
IS/OOS split: IS runs from 2015-01-02 through 2019-12-31 and OOS runs from 2020-01-02 through 2025-12-31. A continuous simulation carries positions across the split. The Average Oracle spectrum and average ridge mixture weights are frozen after the IS signal cutoff, while monthly covariance inputs and mean excess returns continue to update.
Costs and slippage: The model applies 2 bps of transaction costs plus 3 bps of constant slippage to each simulated buy or sell amount, including initial entry. At rebalance, it solves for remaining NAV after charging 5 bps on the sum of absolute dollar changes from drifted holdings to target holdings funded from that remaining NAV. The equal-weight benchmark uses the same friction and rebalance dates.
Signal lag: The signal uses adjusted closes through the last observed trading session of the month. With the configured one-session lag, the simulation first applies the next session's close-to-close return to existing holdings, then rebalances at that session's close. New weights first earn returns over the following close-to-close interval.
Price adjustment: yfinance auto_adjust=True adjusted close, total-return proxy
Survivorship bias: The fixed universe was selected from surviving ETFs with complete histories available at the time of selection. The README identifies survivorship bias and retrospective universe-selection bias.
Parameter combinations tried: 20
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.03394770265202629, ann_vol=0.09820080769288893, sharpe=0.38915459329774493, sharpe_excess=0.2992436344013061, max_drawdown=-0.16350196783080195, total_return=0.1813478490391729, turnover=2.480221739046408, n_trades=442
    oos: cagr=0.11275808619969463, ann_vol=0.12374806413739745, sharpe=0.9254267383033832, sharpe_excess=0.7141827784384404, max_drawdown=-0.2446341652984081, total_return=0.8952566686466776, turnover=2.082225625498262, n_trades=448
    full: cagr=0.07619600855498909, ann_vol=0.1128540938682332, sharpe=0.707248188896376, sharpe_excess=0.5453872798914859, max_drawdown=-0.24463416529840853, total_return=1.238957388882898, turnover=2.2632375961575413, n_trades=890
    benchmark: cagr=0.07119442249344154, ann_vol=0.11244533678500519, sharpe=0.6680816941170526, sharpe_excess=0.505585856977364, max_drawdown=-0.2513117716822164, total_return=1.1273560360340231, turnover=0.3965065359919078, n_trades=1056
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 479_ noise_proofing_universal_portfolio_shrin\nuniverse: [SPY, EFA, EEM, IWM, IEF, TLT, GLD, VNQ]\nrisk_free_ticker: BIL\ndata_start: 2008-01-02\nstart_date: 2015-01-02\nis_end: 2019-12-31\nend_date: 2025-12-31\ncost_bps: 2.0\nslippage_bps: 3.0\nsignal_lag_sessions: 1\nlookback_months: 36\nvalidation_months: 36\noracle_test_months: 6\noracle_half_life_months: 24\nridge_grid:\n  min: 1.0e-8\n  max: 0.1\n  points: 20\nsessions_per_year: 252\n'
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
"""Long-only ETF implementation of the averaged UPSA and Average Oracle estimators."""

import hashlib
import json
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.linalg import solve_triangular
from scipy.optimize import brentq, minimize, nnls


ROOT = Path(__file__).resolve().parent
PRICE_BASIS = "yfinance auto_adjust=True adjusted close, total-return proxy"


def validate_config(config):
    dates = [pd.Timestamp(config[key]) for key in ("data_start", "start_date", "is_end", "end_date")]
    if dates != sorted(set(dates)):
        raise ValueError("Require data_start < start_date < is_end < end_date.")
    tickers = config["universe"]
    if len(set(tickers)) != len(tickers) or len(tickers) < 2 or config["risk_free_ticker"] in tickers:
        raise ValueError("Require at least two distinct ETFs and a separate cash proxy.")
    if config["signal_lag_sessions"] < 1:
        raise ValueError("Execution must lag the signal by at least one session.")
    if min(config["lookback_months"], config["oracle_test_months"], config["validation_months"]) < 2:
        raise ValueError("Covariance and validation windows must have at least two months.")
    if config["oracle_half_life_months"] <= 0 or config["sessions_per_year"] <= 0:
        raise ValueError("Half-life and sessions per year must be positive.")
    grid = config["ridge_grid"]
    if not 0 < grid["min"] < grid["max"] or grid["points"] < 2:
        raise ValueError("Require a positive ascending ridge grid with at least two points.")
    costs = [config["cost_bps"], config["slippage_bps"]]
    if min(costs) < 0 or not 0 <= sum(costs) < 10000:
        raise ValueError("Costs must be nonnegative and total less than 10,000 bps.")


def validate_prices(prices, tickers, start, end):
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.empty:
        raise ValueError("No daily price history returned by Yahoo Finance.")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and sorted.")
    closure = pd.Timedelta(days=7)
    if prices.index[0] > start + closure or prices.index[-1] < end - closure:
        raise ValueError("Yahoo Finance returned incomplete date coverage.")
    prices = prices.reindex(columns=tickers)
    missing = ~np.isfinite(prices) | (prices <= 0)
    if missing.any().any():
        bad_tickers = ", ".join(missing.columns[missing.any()])
        raise ValueError(f"Incomplete or invalid daily closes for {bad_tickers}.")
    return prices


def load_prices(config):
    tickers = [*config["universe"], config["risk_free_ticker"]]
    start = pd.Timestamp(config["data_start"])
    end = pd.Timestamp(config["end_date"])
    start_text = start.date().isoformat()
    end_exclusive = (end + pd.Timedelta(days=1)).date().isoformat()
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_key = json.dumps([sorted(tickers), start_text, end_exclusive, PRICE_BASIS])
    cache_path = cache_dir / f"prices_{hashlib.sha256(cache_key.encode()).hexdigest()[:16]}.csv"
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as handle:
            fetched_at = handle.readline().strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(handle, index_col="date", parse_dates=["date"])
        return validate_prices(prices, tickers, start, end), fetched_at

    yf.set_tz_cache_location(str(cache_dir / "yfinance"))
    for attempt in range(2):
        market_data = yf.download(
            tickers, start=start_text, end=end_exclusive, auto_adjust=True, keepna=True,
            progress=False, timeout=10, multi_level_index=True,
        )
        try:
            prices = validate_prices(market_data["Close"], tickers, start, end)
        except (KeyError, ValueError):
            if attempt == 1:
                raise
        else:
            break

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_dir, delete=False) as handle:
        temp_path = Path(handle.name)
        try:
            handle.write(fetched_at + "\n")
            prices.to_csv(handle, index_label="date")
            handle.flush()
            os.replace(temp_path, cache_path)
        finally:
            temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def correlation_and_volatility(sample):
    covariance = np.cov(sample, rowvar=False, ddof=1)
    volatility = np.sqrt(np.diag(covariance))
    if np.any(volatility <= 0):
        raise ValueError("A monthly estimation window contains a zero-volatility ETF.")
    return covariance / np.outer(volatility, volatility), volatility


def oracle_projection(calibration, later):
    calibration_corr, _ = correlation_and_volatility(calibration)
    later_corr, _ = correlation_and_volatility(later)
    _, directions = np.linalg.eigh(calibration_corr)
    return np.diag(directions.T @ later_corr @ directions).clip(min=0)


def ridge_portfolios(sample, excess_mean, oracle, ridge_grid):
    correlation, volatility = correlation_and_volatility(sample)
    _, directions = np.linalg.eigh(correlation)
    filtered_corr = (directions * oracle) @ directions.T
    diagonal = np.sqrt(filtered_corr.diagonal())
    rescale = volatility / diagonal
    filtered_cov = filtered_corr * np.outer(rescale, rescale)

    # A constant excess mean gives the minimum-risk portfolio if all estimates are nonpositive.
    expected = excess_mean if excess_mean.max() > 0 else np.ones_like(excess_mean)
    identity = np.eye(len(expected))
    basis = []
    for ridge in ridge_grid:
        factor = np.linalg.cholesky(filtered_cov + ridge * identity)
        response = solve_triangular(factor, expected, lower=True)
        allocation, _ = nnls(factor.T, response)
        basis.append(allocation / allocation.sum())
    return np.asarray(basis).T


def fit_mixture(validation):
    samples = np.asarray(validation)
    means = samples.mean(axis=0)
    covariance = np.cov(samples, rowvar=False, ddof=1)
    scale = max(np.abs(means).max(), np.abs(covariance).max(), 1e-12)
    means = means / scale
    covariance = covariance / scale
    initial = np.full(len(means), 1 / len(means))
    fitted = minimize(
        lambda weights: 0.5 * weights @ covariance @ weights - means @ weights,
        initial, jac=lambda weights: covariance @ weights - means,
        bounds=[(0, 1)] * len(means),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1,
                     "jac": lambda weights: np.ones_like(weights)},
        method="SLSQP", options={"ftol": 1e-10},
    )
    if not fitted.success:
        raise RuntimeError(f"Ridge mixture fit failed: {fitted.message}")
    weights = fitted.x.clip(min=0)
    return weights / weights.sum()


def make_targets(prices, config, ridge_grid):
    tickers = config["universe"]
    cash_ticker = config["risk_free_ticker"]
    start = pd.Timestamp(config["start_date"])
    split = pd.Timestamp(config["is_end"])
    lag = config["signal_lag_sessions"]
    lookback = config["lookback_months"]
    test_months = config["oracle_test_months"]
    decay = 0.5 ** (1 / config["oracle_half_life_months"])
    validation = deque(maxlen=config["validation_months"])

    # Keep the actual last session date of each month for execution scheduling.
    monthly = prices.groupby(prices.index.to_period("M")).tail(1).pct_change().iloc[1:]
    asset_returns = monthly[tickers].to_numpy()
    excess_returns = monthly[tickers].sub(monthly[cash_ticker], axis=0).to_numpy()
    signal_rows = prices.index.get_indexer(monthly.index)
    targets = {}
    oracle = None
    mixture = np.zeros(len(ridge_grid))
    mixture_count = 0
    previous_basis = None
    previous_execution = None

    for month, (signal_date, row) in enumerate(zip(monthly.index, signal_rows)):
        execution_row = row + lag
        if execution_row >= len(prices):
            break
        execution_date = prices.index[execution_row]

        if signal_date <= split and month >= lookback + test_months:
            test_start = month - test_months
            projection = oracle_projection(
                asset_returns[test_start - lookback:test_start], asset_returns[test_start:month],
            )
            oracle = projection if oracle is None else decay * oracle + (1 - decay) * projection

        if oracle is not None:
            if signal_date <= split and previous_basis is not None and previous_execution < row:
                realized = prices.iloc[row] / prices.iloc[previous_execution] - 1
                validation.append((realized[tickers].to_numpy() - realized[cash_ticker]) @ previous_basis)
                if len(validation) == validation.maxlen:
                    mixture_count += 1
                    fitted = fit_mixture(validation)
                    mixture += (fitted - mixture) / mixture_count

            sample_start = month - lookback + 1
            basis = ridge_portfolios(
                asset_returns[sample_start:month + 1],
                excess_returns[sample_start:month + 1].mean(axis=0), oracle, ridge_grid,
            )
            previous_basis = basis
            previous_execution = execution_row

        if execution_date >= start:
            if mixture_count == 0:
                raise ValueError("Insufficient pre-sample history for the first scheduled execution.")
            targets[execution_date] = basis @ mixture

    return targets


def simulate(asset_returns, targets, friction):
    weights = np.zeros(asset_returns.shape[1])
    daily_results = []
    for date, daily in asset_returns.iterrows():
        moved = weights * (1 + daily.to_numpy())
        gross_growth = 1 + float(weights @ daily.to_numpy())
        weights = moved / gross_growth
        post_cost = 1.0
        turnover = 0.0
        trades = 0
        if date in targets:
            target = targets[date]
            # Finance fees from the same capital used to reach the target proportions.
            post_cost = brentq(
                lambda remaining: remaining + friction * np.abs(remaining * target - weights).sum() - 1,
                0, 1,
            )
            changes = np.abs(post_cost * target - weights)
            turnover = changes.sum()
            trades = np.count_nonzero(changes > 1e-10)
            weights = target

        daily_results.append((gross_growth * post_cost - 1, turnover, trades))

    return pd.DataFrame(daily_results, index=asset_returns.index, columns=["return", "turnover", "n_trades"])


def metrics(daily_results, cash_returns, sessions):
    if len(daily_results) < 2:
        raise ValueError("Each reporting window needs at least two daily observations.")
    returns = daily_results["return"]
    excess = returns - cash_returns.loc[returns.index]
    growth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    drawdowns = growth / np.maximum.accumulate(growth) - 1
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    return {
        "cagr": float(growth[-1] ** (sessions / len(returns)) - 1),
        "ann_vol": float(volatility * np.sqrt(sessions)),
        "sharpe": float(returns.mean() / volatility * np.sqrt(sessions)) if volatility else None,
        "sharpe_excess": float(excess.mean() / excess_volatility * np.sqrt(sessions)) if excess_volatility else None,
        "max_drawdown": float(drawdowns.min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * sessions),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    grid = config["ridge_grid"]
    ridge_grid = np.geomspace(grid["min"], grid["max"], grid["points"])
    prices, fetched_at = load_prices(config)
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print(f"PRICE_ADJUSTMENT: {PRICE_BASIS}", flush=True)
    print(f"PARAM_COMBOS_TRIED: {len(ridge_grid)}", flush=True)

    start = pd.Timestamp(config["start_date"])
    split = pd.Timestamp(config["is_end"])
    end = pd.Timestamp(config["end_date"])
    returns = prices.pct_change().iloc[1:].loc[start:end]
    cash_returns = returns[config["risk_free_ticker"]]
    asset_returns = returns[config["universe"]]
    friction = (config["cost_bps"] + config["slippage_bps"]) / 10000
    targets = make_targets(prices, config, ridge_grid)
    if not targets:
        raise ValueError("No scheduled executions in the evaluation period.")
    daily_results = simulate(asset_returns, targets, friction)
    equal = np.full(asset_returns.shape[1], 1 / asset_returns.shape[1])
    benchmark = simulate(asset_returns, dict.fromkeys(targets, equal), friction)
    sessions = config["sessions_per_year"]

    result = {
        "strategy": config["strategy"],
        "data_range": {"start": str(returns.index[0].date()), "end": str(returns.index[-1].date())},
        "is": metrics(daily_results.loc[daily_results.index <= split], cash_returns, sessions),
        "oos": metrics(daily_results.loc[daily_results.index > split], cash_returns, sessions),
        "full": metrics(daily_results, cash_returns, sessions),
        "benchmark": {"name": "Monthly equal weight", **metrics(benchmark, cash_returns, sessions)},
    }
    for period in ("is", "oos", "full", "benchmark"):
        print(f"{period.upper()}: {json.dumps(result[period], allow_nan=False)}")
    print(f"RESULTS_JSON: {json.dumps(result, allow_nan=False)}")


if __name__ == "__main__":
    main()
