# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 470_entropic_value_at_risk_parity_for_temper\nuniverse: [VTI, EFA, VWO, VNQ, AGG, TIP, GLD]\nrisk_free_symbol: BIL\ndata_start: 2005-04-01\nbacktest_start: 2008-01-02\nis_end: 2016-12-31\nend_date: 2026-03-31\nlookback_months: 12\ntail_probability: 0.05\nsignal_lag_sessions: 1\ncost_bps: 5\nslippage_bps: 5\n'
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
"""Monthly equal contributions to empirical EVaR deviation."""

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
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252


def price_cache_path(symbols, start, end):
    cache_key = json.dumps([sorted(symbols), str(start.date()), str(end.date())])
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"adjusted_close_{digest}.csv"


def load_prices(symbols, start, end, cache_file):
    CACHE_DIR.mkdir(exist_ok=True)
    if cache_file.exists():
        with cache_file.open(encoding="utf-8") as source:
            fetched_at = source.readline().strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(source, index_col="Date", parse_dates=["Date"])
        return prices[symbols], fetched_at

    with TemporaryDirectory(prefix="yf_", dir=CACHE_DIR) as temp_dir:
        # Keep yfinance's transient session files inside the strategy folder.
        yf.set_tz_cache_location(temp_dir)
        prices = yf.download(
            symbols, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=True, ignore_tz=True,
            progress=False, threads=False, timeout=20,
        )["Close"].reindex(columns=symbols).sort_index()

    missing = prices.columns[prices.notna().sum() == 0].tolist()
    if missing:
        raise RuntimeError(f"Yahoo Finance returned no adjusted closes for {missing}")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return prices, fetched_at


def cache_prices(prices, fetched_at, cache_file):
    if cache_file.exists():
        return
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE_DIR, delete=False) as output:
        temp_path = Path(output.name)
        try:
            output.write(f"{fetched_at}\n")
            prices.to_csv(output, index_label="Date")
            output.flush()
            temp_path.replace(cache_file)
        finally:
            temp_path.unlink(missing_ok=True)


def evar_tilt(losses, tail_probability):
    """Return weights for the entropy-constrained worst-case mean loss."""
    gaps = losses - losses.max()
    worst = gaps == 0
    if worst.mean() >= tail_probability:
        return worst.astype(float) / worst.sum()

    gaps /= -gaps.min()
    entropy_limit = -np.log(tail_probability)

    def tilted(inverse_temperature):
        logits = inverse_temperature * gaps
        values = np.exp(logits)
        normalizer = values.sum()
        probabilities = values / normalizer
        divergence = probabilities @ logits - np.log(normalizer / len(losses))
        return probabilities, divergence

    def entropy_gap(inverse_temperature):
        return tilted(inverse_temperature)[1] - entropy_limit

    upper = 1.0
    while entropy_gap(upper) <= 0:
        upper *= 2
    optimum = brentq(entropy_gap, 0, upper)
    return tilted(optimum)[0]


def equal_risk_weights(window, tail_probability):
    losses = window.mean(axis=0) - window
    scale = losses.std(axis=0)
    if (scale <= 0).any():
        raise ValueError("EVaR risk budgeting requires nonconstant return series")
    scenarios = losses / scale
    budget = 1.0 / window.shape[1]

    def marginal_risk(scaled_weights):
        probabilities = evar_tilt(scenarios @ scaled_weights, tail_probability)
        return probabilities @ scenarios

    def objective(scaled_weights):
        marginal = marginal_risk(scaled_weights)
        risk = scaled_weights @ marginal
        # At the minimum, each unnormalized Euler contribution equals its budget.
        value = risk - budget * np.log(scaled_weights).sum()
        gradient = marginal - budget / scaled_weights
        return value, gradient

    result = minimize(
        objective, np.full(window.shape[1], budget), jac=True, method="L-BFGS-B",
        bounds=[(1e-10, None)] * window.shape[1],
        options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 400},
    )
    contributions = result.x * marginal_risk(result.x)
    if not result.success or not np.allclose(contributions, budget, rtol=1e-4, atol=1e-8):
        raise RuntimeError(f"EVaR risk budgeting did not converge: {result.message}")
    weights = result.x / scale
    return weights / weights.sum()


def monthly_targets(returns, backtest_start, lookback_months, lag, tail_probability):
    dates = returns.index
    months = dates.to_period("M")
    new_months = np.flatnonzero(months[1:] != months[:-1]) + 1
    targets = {}
    for first_session in new_months:
        fill_row = first_session + lag - 1
        if fill_row >= len(dates) or dates[fill_row] < backtest_start:
            continue

        signal_date = dates[first_session - 1]
        window_start = signal_date - pd.DateOffset(months=lookback_months)
        if dates[0] > window_start:
            continue
        window = returns.loc[(dates > window_start) & (dates <= signal_date)]
        try:
            targets[dates[fill_row]] = equal_risk_weights(window.to_numpy(), tail_probability)
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(f"Signal on {signal_date.date()}: {error}") from error
    if not targets:
        raise ValueError("No monthly signals with enough estimation history before the end date")
    return targets


def simulate(returns, targets, fee_rate):
    daily_results = []
    weights = np.zeros(returns.shape[1])
    for date, asset_returns in zip(returns.index, returns.to_numpy()):
        growth = 1.0 + weights @ asset_returns
        weights = weights * (1.0 + asset_returns) / growth
        turnover = 0.0
        trades = 0
        target = targets.get(date)
        if target is not None:
            # Price movement through this close applies to the previous weights.
            turnover = 0.5 * (np.abs(target - weights).sum() + abs(1.0 - weights.sum()))

            def fee_balance(net_fraction):
                gross_traded = np.abs(net_fraction * target - weights).sum()
                return net_fraction + fee_rate * gross_traded - 1.0

            net_fraction = brentq(fee_balance, 0.0, 1.0)
            trades = np.count_nonzero(np.abs(net_fraction * target - weights) > 1e-10)
            growth *= net_fraction
            weights = target
        daily_results.append((growth - 1.0, turnover, trades))
    return pd.DataFrame(daily_results, index=returns.index,
                        columns=["return", "turnover", "n_trades"])


def metrics(daily_results, risk_free):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise ValueError("Each reporting period needs at least two sessions")
    excess = returns - risk_free.reindex(returns.index)
    cumulative = (1.0 + returns).cumprod()
    volatility = float(returns.std())
    excess_volatility = float(excess.std())
    annualize = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(cumulative.iloc[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(volatility * annualize),
        "sharpe": float(returns.mean() / volatility * annualize) if volatility > 0 else None,
        "sharpe_excess": (float(excess.mean() / excess_volatility * annualize)
                          if excess_volatility > 0 else None),
        "max_drawdown": float((cumulative / cumulative.cummax().clip(lower=1.0) - 1).min()),
        "total_return": float(cumulative.iloc[-1] - 1),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    risk_free_symbol = config["risk_free_symbol"]
    data_start = pd.Timestamp(config["data_start"])
    backtest_start = pd.Timestamp(config["backtest_start"])
    is_end = pd.Timestamp(config["is_end"])
    end = pd.Timestamp(config["end_date"])
    lookback = config["lookback_months"]
    lag = config["signal_lag_sessions"]
    tail_probability = config["tail_probability"]
    cost = config["cost_bps"]
    slippage = config["slippage_bps"]
    fee_rate = (cost + slippage) / 10_000

    if not data_start < backtest_start <= is_end < end:
        raise ValueError("Dates must allow estimation history and distinct IS and OOS periods")
    if len(set(universe)) != len(universe) or not universe or risk_free_symbol in universe:
        raise ValueError("Choose a nonempty, unique universe separate from the cash proxy")
    if not isinstance(lookback, int) or lookback < 1 or not isinstance(lag, int) or lag < 1:
        raise ValueError("Lookback months and signal lag sessions must be positive integers")
    if not 0 < tail_probability < 1 or not 0 <= fee_rate < 1 or min(cost, slippage) < 0:
        raise ValueError("Invalid tail probability or fee rates")

    symbols = [*universe, risk_free_symbol]
    cache_file = price_cache_path(symbols, data_start, end)
    prices, fetched_at = load_prices(symbols, data_start, end, cache_file)
    if (not prices.index.is_monotonic_increasing or not prices.index.is_unique
            or not prices.index[0] <= data_start + pd.Timedelta(days=7)
            or not prices.index[-1] >= end - pd.Timedelta(days=7)):
        raise ValueError("Adjusted closes do not cover the requested date range")
    arrays = (prices[universe].to_numpy(), prices.loc[backtest_start:, risk_free_symbol].to_numpy())
    for values in arrays:
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Missing or invalid adjusted closes in the required sample")
    cache_prices(prices, fetched_at, cache_file)

    asset_returns = prices[universe].pct_change(fill_method=None).iloc[1:]
    targets = monthly_targets(asset_returns, backtest_start, lookback, lag, tail_probability)
    evaluated = asset_returns.loc[min(targets):]
    daily_results = simulate(evaluated, targets, fee_rate)
    equal_weight = np.full(len(universe), 1.0 / len(universe))
    benchmark_results = simulate(evaluated, dict.fromkeys(targets, equal_weight), fee_rate)

    risk_free = prices.loc[evaluated.index, risk_free_symbol].pct_change(fill_method=None)
    risk_free.iloc[0] = 0.0
    periods = {
        "is": daily_results.loc[daily_results.index <= is_end],
        "oos": daily_results.loc[daily_results.index > is_end],
        "full": daily_results,
    }
    results = {
        "strategy": config["strategy"],
        "data_range": f"{evaluated.index[0].date()} to {evaluated.index[-1].date()}",
        **{period: metrics(series, risk_free) for period, series in periods.items()},
        "benchmark": {"name": "Monthly equal weight", **metrics(benchmark_results, risk_free)},
    }

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return adjusted closes")
    print("PARAM_COMBOS_TRIED: 1")
    for period, series in periods.items():
        print(f"{period.upper()} ({series.index[0].date()} to {series.index[-1].date()}): "
              + json.dumps(results[period], allow_nan=False))
    print("BENCHMARK: " + json.dumps(results["benchmark"], allow_nan=False))
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
