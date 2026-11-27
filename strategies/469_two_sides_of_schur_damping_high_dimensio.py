# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 469_two_sides_of_schur_damping_high_dimensio\nuniverse:\n  - SPY\n  - IWM\n  - EFA\n  - EEM\n  - VNQ\n  - XLU\n  - IEF\n  - TLT\n  - TIP\n  - LQD\n  - HYG\n  - EMB\n  - GLD\n  - DBC\n  - UUP\nrisk_free_ticker: BIL\nprice_start: '2009-01-01'\nbacktest_start: '2010-01-01'\nis_end: '2017-12-31'\nbacktest_end: '2025-12-31'\nrebalance: monthly\nlag_sessions: 1\nlookback_sessions: 45\ngamma: 0.86\ncost_bps: 2.0\nslippage_bps: 3.0\n"
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
"""Rolling multi-asset allocation using Cotton's damped Schur blocks."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import brentq
from scipy.spatial.distance import squareform
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_BASIS = "yfinance auto_adjust=True dividend- and split-adjusted close (total return)"
SESSIONS_PER_YEAR = 252
TRADE_TOLERANCE = 1e-10


def load_config():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for key in ("price_start", "backtest_start", "is_end", "backtest_end"):
        config[key] = pd.Timestamp(config[key])
    if not config["price_start"] < config["backtest_start"] <= config["is_end"] < config["backtest_end"]:
        raise ValueError("Expected price_start < backtest_start <= is_end < backtest_end")
    universe = config["universe"]
    if len(universe) < 2 or len(set(universe)) != len(universe) or config["risk_free_ticker"] in universe:
        raise ValueError("Use distinct ETFs and a separate cash-return proxy")
    if config["rebalance"] != "monthly":
        raise ValueError("Expected monthly rebalancing")
    for key, minimum in (("lag_sessions", 1), ("lookback_sessions", 3)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer of at least {minimum}")
    if not 0 <= config["gamma"] <= 1:
        raise ValueError("gamma must be in [0, 1]")
    fees = np.array([config["cost_bps"], config["slippage_bps"]], dtype=float)
    if not np.isfinite(fees).all() or (fees < 0).any() or fees.sum() >= 10_000:
        raise ValueError("Cost and slippage must be finite, nonnegative and total under 10,000 bps")
    return config


def validate_prices(prices, tickers, start, end):
    prices = prices.reindex(columns=tickers).sort_index()
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("Price panel is empty or contains duplicate sessions")
    if prices.index[0] < start or prices.index[-1] > end:
        raise ValueError("Price panel contains dates outside the request")
    if prices.index[0] > start + pd.Timedelta(days=7) or prices.index[-1] < end - pd.Timedelta(days=7):
        raise ValueError("Price panel does not span the requested window")
    invalid = ~np.isfinite(prices) | (prices <= 0)
    if invalid.any().any():
        counts = invalid.sum()
        raise ValueError(f"Missing or invalid adjusted closes: {counts[counts > 0].to_dict()}")
    return prices


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    request = [PRICE_BASIS, tickers, str(start.date()), str(end.date())]
    cache_key = sha256(json.dumps(request).encode("utf-8")).hexdigest()[:20]
    path = CACHE / f"prices_{cache_key}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as cached:
            fetched_at = cached.readline().strip().removeprefix("# fetched_at=")
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(cached, index_col=0, parse_dates=True)
        return validate_prices(prices, tickers, start, end), fetched_at

    with TemporaryDirectory(prefix=".yfinance-", dir=CACHE) as yf_cache:
        yf.set_tz_cache_location(yf_cache)
        panel = yf.download(
            tickers,
            start=str(start.date()),
            end=str((end + pd.Timedelta(days=1)).date()),
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=15,
        )
    if panel is None or panel.empty:
        raise RuntimeError("Yahoo Finance returned no adjusted prices")
    prices = validate_prices(panel["Close"], tickers, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as pending:
        pending_path = Path(pending.name)
        try:
            pending.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(pending, index_label="date")
            pending.flush()
            pending_path.replace(path)
        finally:
            pending_path.unlink(missing_ok=True)
    return prices, fetched_at


def schur_direction(covariance, budget, gamma):
    """Return a direction scaled to the inverse variance of its unit-budget form."""
    if len(budget) == 1:
        return budget / covariance[0, 0]

    cut = len(budget) // 2
    left = covariance[:cut, :cut]
    right = covariance[cut:, cut:]
    cross = covariance[:cut, cut:]
    left_budget, right_budget = budget[:cut], budget[cut:]

    # Cotton (2411.05807), equations (10) and (11), in generalized budget coordinates.
    left_hedge = np.linalg.solve(right, cross.T).T
    right_hedge = np.linalg.solve(left, cross).T
    left_direction = schur_direction(
        left - gamma * (left_hedge @ cross.T),
        left_budget - gamma * (left_hedge @ right_budget),
        gamma,
    )
    right_direction = schur_direction(
        right - gamma * (right_hedge @ cross),
        right_budget - gamma * (right_hedge @ left_budget),
        gamma,
    )
    direction = np.concatenate([left_direction, right_direction])
    variance = float(direction @ covariance @ direction)
    if variance == 0.0:
        return direction
    return direction * (float(budget @ direction) / variance)


def allocation(window, gamma):
    covariance = np.cov(window, rowvar=False, ddof=1)
    variances = np.diag(covariance)
    if (variances <= 0).any():
        raise ValueError("A rolling estimation window contains a constant ETF return")
    scales = np.sqrt(variances)
    correlations = covariance / np.outer(scales, scales)
    distances = np.sqrt((1.0 - np.clip(correlations, -1.0, 1.0)) / 2.0)
    np.fill_diagonal(distances, 0.0)
    order = leaves_list(linkage(squareform(distances, checks=False), method="single"))

    # A machine-scale eigenvalue floor stabilizes block solves for singular inputs.
    floor = np.finfo(float).eps * len(variances) * variances.max()
    ridge = max(0.0, floor - np.linalg.eigvalsh(covariance)[0])
    covariance = covariance + ridge * np.eye(len(variances))
    direction = schur_direction(covariance[np.ix_(order, order)], np.ones(len(order)), gamma)
    budget = direction.sum()
    if not np.isfinite(direction).all() or budget <= 0:
        raise ValueError("Invalid Schur allocation")
    long_weights = np.maximum(direction / budget, 0.0)
    weights = np.empty_like(long_weights)
    weights[order] = long_weights / long_weights.sum()
    return weights


def scheduled_targets(returns, config):
    sessions = returns.index
    month_ends = pd.Series(np.arange(len(sessions)), index=sessions).groupby(sessions.to_period("M")).last()
    lookback = config["lookback_sessions"]
    lag = config["lag_sessions"]
    targets = {}
    for signal in month_ends:
        execution = signal + lag
        if signal + 1 < lookback or execution >= len(sessions):
            continue
        date = sessions[execution]
        if date >= config["backtest_start"]:
            window = returns.iloc[signal - lookback + 1 : signal + 1]
            targets[date] = allocation(window.to_numpy(), config["gamma"])
    if not targets:
        raise ValueError("No fills in the requested backtest window")
    return targets


def simulate(asset_returns, cash_returns, targets, fee_rate):
    daily_results = []
    weights = np.zeros(asset_returns.shape[1])
    for date, returns, cash_return in zip(asset_returns.index, asset_returns.to_numpy(), cash_returns):
        marked = weights * (1.0 + returns)
        growth = marked.sum() + (1.0 - weights.sum()) * (1.0 + cash_return)
        weights = marked / growth
        turnover = 0.0
        trades = 0
        if date in targets:
            target = targets[date]
            # Size the fills against the post-fee value so cash and positions balance.
            remaining = brentq(
                lambda fraction: fraction + fee_rate * np.abs(fraction * target - weights).sum() - 1.0,
                0.0,
                1.0,
            )
            changes = np.abs(remaining * target - weights)
            turnover = float(changes.sum())
            trades = int(np.count_nonzero(changes > TRADE_TOLERANCE))
            growth *= remaining
            weights = target
        daily_results.append((growth - 1.0, turnover, trades))
    return pd.DataFrame(daily_results, index=asset_returns.index, columns=["return", "turnover", "n_trades"])


def annual_sharpe(daily_returns):
    daily_std = daily_returns.std(ddof=1)
    return float(np.sqrt(SESSIONS_PER_YEAR) * daily_returns.mean() / daily_std) if daily_std > 0 else None


def metrics(daily_results, cash_returns):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise ValueError("Each reporting period requires at least two sessions")
    growth = np.r_[1.0, (1.0 + returns).cumprod().to_numpy()]
    years = len(returns) / SESSIONS_PER_YEAR
    return {
        "cagr": float(growth[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": annual_sharpe(returns),
        "sharpe_excess": annual_sharpe(returns - cash_returns.loc[returns.index]),
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1.0).min()),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, cash_returns, split):
    return {
        "is": metrics(daily_results.loc[daily_results.index <= split], cash_returns),
        "oos": metrics(daily_results.loc[daily_results.index > split], cash_returns),
        "full": metrics(daily_results, cash_returns),
    }


def main():
    config = load_config()
    universe = config["universe"]
    cash_ticker = config["risk_free_ticker"]
    prices, fetched_at = load_prices(universe + [cash_ticker], config["price_start"], config["backtest_end"])
    returns = prices.pct_change(fill_method=None).iloc[1:]
    targets = scheduled_targets(returns[universe], config)
    report_returns = returns.loc[config["backtest_start"] :]
    asset_returns, cash_returns = report_returns[universe], report_returns[cash_ticker]
    fee_rate = (config["cost_bps"] + config["slippage_bps"]) / 10_000
    daily_results = simulate(asset_returns, cash_returns, targets, fee_rate)
    equal_weights = np.full(len(universe), 1.0 / len(universe))
    benchmark_targets = dict.fromkeys(targets, equal_weights)
    benchmark_results = simulate(asset_returns, cash_returns, benchmark_targets, fee_rate)

    strategy_metrics = period_metrics(daily_results, cash_returns, config["is_end"])
    benchmark_metrics = period_metrics(benchmark_results, cash_returns, config["is_end"])
    date_range = {"start": str(report_returns.index[0].date()), "end": str(report_returns.index[-1].date())}
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_BASIS}")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"Date Range: {date_range['start']} to {date_range['end']}")
    print(f"Risk-free proxy: {cash_ticker}")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(strategy_metrics[period], allow_nan=False)}")
        print(f"BENCHMARK {period.upper()}: {json.dumps(benchmark_metrics[period], allow_nan=False)}")
    result = {
        "strategy": config["strategy"],
        "data_range": date_range,
        **strategy_metrics,
        "benchmark": {"name": "Monthly equal-weight ETF basket", **benchmark_metrics["full"]},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
