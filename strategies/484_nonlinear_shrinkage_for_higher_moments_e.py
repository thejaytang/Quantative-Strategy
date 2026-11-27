# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 484_nonlinear_shrinkage_for_higher_moments_e\nuniverse:\n  - AAPL\n  - MSFT\n  - AMZN\n  - GOOGL\n  - NVDA\n  - IBM\n  - ORCL\n  - CSCO\n  - ADBE\n  - QCOM\n  - TXN\n  - INTC\n  - JPM\n  - BAC\n  - WFC\n  - GS\n  - MS\n  - AXP\n  - USB\n  - PNC\n  - JNJ\n  - MRK\n  - PFE\n  - ABT\n  - UNH\n  - CVS\n  - HUM\n  - BMY\n  - PG\n  - KO\n  - PEP\n  - COST\n  - WMT\n  - CL\n  - KMB\n  - MCD\n  - SBUX\n  - HD\n  - LOW\n  - TGT\n  - DIS\n  - XOM\n  - CVX\n  - COP\n  - SLB\n  - CAT\n  - DE\n  - UNP\n  - UPS\n  - DUK\nrisk_free_ticker: BIL\ndownload_start: '2008-12-01'\nbacktest_start: '2010-01-01'\noos_start: '2018-01-01'\nend: '2025-12-31'\nlookback_sessions: 252\nfourth_moment_penalty: 1.0\nexecution_lag_sessions: 1\ncost_bps: 5.0\nslippage_bps: 5.0\n"
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
import gzip
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import monotonic

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, minimize
from sklearn.covariance import ledoit_wolf
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
PRICE_ADJUSTMENT = "yfinance auto_adjust total-return adjusted daily close"
SESSIONS_PER_YEAR = 252
WEIGHT_TOLERANCE = 1e-10


def load_config():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    date_keys = ("download_start", "backtest_start", "oos_start", "end")
    for key in date_keys:
        config[key] = pd.Timestamp(config[key])
    download_start, start, split, end = (config[key] for key in date_keys)
    if not download_start < start < split <= end:
        raise ValueError("Dates must satisfy download_start < backtest_start < oos_start <= end.")
    universe = config["universe"]
    if len(universe) < 2 or len(universe) != len(set(universe)):
        raise ValueError("The stock universe must contain at least two distinct tickers.")
    if not all(re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z])?", ticker) for ticker in universe):
        raise ValueError("Use U.S. equity tickers such as MSFT or BRK-B.")
    if config["risk_free_ticker"] != "BIL" or "BIL" in universe:
        raise ValueError("Use BIL outside the stock universe as the cash proxy.")
    for key, minimum in (("lookback_sessions", 2), ("execution_lag_sessions", 1)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer of at least {minimum}.")
    for key in ("cost_bps", "slippage_bps", "fourth_moment_penalty"):
        if not np.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    if config["cost_bps"] + config["slippage_bps"] >= 10_000:
        raise ValueError("Cost and slippage together must be under 10,000 bps.")
    return config


def download_closes(ticker, start, end, deadline):
    for attempt in range(2):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("The Yahoo Finance download exceeded its time budget.")
        try:
            history = yf.Ticker(ticker).history(
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d",
                auto_adjust=True,
                timeout=min(8, remaining),
            )
            closes = history["Close"].dropna().rename(ticker)
            if closes.empty:
                raise ValueError("No adjusted daily closes returned.")
            sessions = pd.DatetimeIndex(closes.index).tz_localize(None).normalize()
            closes.index = sessions.as_unit("ns").rename(None)
            return closes
        except Exception as error:
            if attempt == 1:
                raise RuntimeError(f"Yahoo Finance download failed for {ticker}.") from error


def check_prices(prices, tickers, start, end):
    if prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Price data require a nonempty, ordered set of unique sessions.")
    if list(prices.columns) != tickers:
        raise ValueError("Price columns do not match the configured tickers.")
    if not (
        start <= prices.index[0] <= start + pd.Timedelta(days=7)
        and end - pd.Timedelta(days=7) <= prices.index[-1] <= end
    ):
        raise ValueError("Price data do not cover the requested dates.")
    valid = np.isfinite(prices) & (prices > 0)
    if not valid.to_numpy().all():
        row, column = np.argwhere(~valid.to_numpy())[0]
        raise ValueError(f"Missing or invalid close for {tickers[column]} on {prices.index[row].date()}.")


def load_prices(tickers, start, end):
    cache_key = ["yfinance-auto-adjust-v1", tickers, str(start.date()), str(end.date())]
    digest = hashlib.sha256(json.dumps(cache_key).encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"closes_{digest}.json.gz"
    if cache_file.exists():
        with gzip.open(cache_file, "rt", encoding="utf-8") as source:
            cached = json.load(source)
        sessions = pd.to_datetime(cached["sessions"]).as_unit("ns")
        prices = pd.DataFrame(cached["closes"], index=sessions, columns=cached["tickers"])
        check_prices(prices, tickers, start, end)
        return prices, cached["fetched_at"]

    CACHE_DIR.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=CACHE_DIR) as request_cache:
        yf.set_tz_cache_location(request_cache)
        hide_exceptions = yf.config.debug.hide_exceptions
        yf.config.debug.hide_exceptions = False
        try:
            deadline = monotonic() + 180
            prices = pd.concat(
                [download_closes(ticker, start, end, deadline) for ticker in tickers], axis=1
            ).sort_index()
        finally:
            yf.config.debug.hide_exceptions = hide_exceptions
            yf.set_tz_cache_location(str(CACHE_DIR))
    check_prices(prices, tickers, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    snapshot = {
        "fetched_at": fetched_at,
        "tickers": tickers,
        "sessions": prices.index.strftime("%Y-%m-%d").tolist(),
        "closes": prices.to_numpy().tolist(),
    }
    with NamedTemporaryFile(dir=CACHE_DIR, delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        with gzip.open(temp_path, "wt", encoding="utf-8") as destination:
            json.dump(snapshot, destination, separators=(",", ":"), allow_nan=False)
        temp_path.replace(cache_file)
    finally:
        temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def minimize_moments(window, equal_weights, penalty):
    centered = window - window.mean(axis=0)
    scale = np.sqrt(np.mean((centered @ equal_weights) ** 2))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The trailing variance of the equal-weight basket must be positive.")
    scaled = centered / scale
    covariance, _ = ledoit_wolf(scaled, assume_centered=True)

    def objective(weights):
        cov_product = covariance @ weights
        projected = scaled @ weights
        cubes = projected ** 3
        moment = np.mean(projected * cubes)
        gradient = 2 * cov_product + 4 * penalty * (scaled.T @ cubes) / len(scaled)
        return weights @ cov_product + penalty * moment, gradient

    solution = minimize(
        objective,
        equal_weights,
        jac=True,
        method="SLSQP",
        bounds=[(0, 1)] * len(equal_weights),
        constraints={
            "type": "eq",
            "fun": lambda weights: weights.sum() - 1,
            "jac": lambda weights: np.ones_like(weights),
        },
        options={"ftol": 1e-11, "maxiter": 250},
    )
    weights = solution.x
    if not solution.success or not np.isfinite(weights).all():
        raise RuntimeError(f"Portfolio optimization failed: {solution.message}")
    if abs(weights.sum() - 1) > 1e-8 or weights.min() < -WEIGHT_TOLERANCE:
        raise RuntimeError("Portfolio optimization produced infeasible weights.")
    weights = np.where(weights > WEIGHT_TOLERANCE, weights, 0.0)
    return weights / weights.sum()


def build_targets(asset_returns, config):
    dates = asset_returns.index
    row_numbers = pd.Series(np.arange(len(dates)), index=dates)
    month_ends = row_numbers.groupby(dates.to_period("M")).last()
    lookback = config["lookback_sessions"]
    lag = config["execution_lag_sessions"]
    equal_weights = np.full(asset_returns.shape[1], 1 / asset_returns.shape[1])
    strategy_targets, benchmark_targets = {}, {}
    with threadpool_limits(limits=1, user_api="blas"):
        for signal_row in month_ends:
            window_start = signal_row - lookback + 1
            fill_row = signal_row + lag
            if window_start < 0 or fill_row >= len(dates):
                continue
            window = asset_returns.iloc[window_start:signal_row + 1].to_numpy()
            try:
                target = minimize_moments(window, equal_weights, config["fourth_moment_penalty"])
            except (ValueError, RuntimeError) as error:
                raise RuntimeError(f"Signal on {dates[signal_row].date()} failed.") from error
            strategy_targets[dates[fill_row]] = target
            benchmark_targets[dates[fill_row]] = equal_weights
    return strategy_targets, benchmark_targets


def simulate(asset_returns, targets, cost_rate):
    weights = np.zeros(asset_returns.shape[1])
    rows = []
    for date, asset_gross in zip(asset_returns.index, 1 + asset_returns.to_numpy()):
        gross = 1 + weights @ (asset_gross - 1)
        weights = weights * asset_gross / gross
        net_fraction, traded, n_trades = 1.0, 0.0, 0

        if date in targets:
            target = targets[date]
            # Solve for the value remaining after fees, relative to the value before the fill.
            if cost_rate > 0:
                net_fraction = brentq(
                    lambda fraction: fraction + cost_rate * np.abs(fraction * target - weights).sum() - 1,
                    0.0, 1.0,
                )
            changes = np.abs(net_fraction * target - weights)
            traded = changes.sum()
            n_trades = np.count_nonzero(changes > WEIGHT_TOLERANCE)
            weights = target

        rows.append((gross * net_fraction - 1, traded, n_trades))
    return pd.DataFrame(rows, index=asset_returns.index, columns=["return", "turnover", "n_trades"])


def metrics(daily_results, cash_returns):
    daily = daily_results["return"].to_numpy()
    if len(daily) < 2:
        raise ValueError("Each reported period needs at least two sessions.")
    excess = daily - cash_returns.loc[daily_results.index].to_numpy()
    growth = np.r_[1.0, np.cumprod(1 + daily)]

    def sharpe(returns):
        volatility = returns.std(ddof=1)
        return float(np.sqrt(SESSIONS_PER_YEAR) * returns.mean() / volatility) if volatility > 0 else None

    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(daily)) - 1),
        "ann_vol": float(daily.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": sharpe(daily),
        "sharpe_excess": sharpe(excess),
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1)),
        "total_return": float(growth[-1] - 1),
        "turnover": float(SESSIONS_PER_YEAR * daily_results["turnover"].mean()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def report(label, daily_results, cash_returns, split):
    periods = {
        "is": daily_results.loc[daily_results.index < split],
        "oos": daily_results.loc[daily_results.index >= split],
        "full": daily_results,
    }
    output = {name: metrics(period, cash_returns) for name, period in periods.items()}
    for period, values in output.items():
        print(f"{label} {period.upper()}: {json.dumps(values, allow_nan=False)}")
    return output


def main():
    config = load_config()
    universe = config["universe"]
    cash_ticker = config["risk_free_ticker"]
    prices, fetched_at = load_prices([*universe, cash_ticker], config["download_start"], config["end"])
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    daily_returns = prices.pct_change(fill_method=None).iloc[1:]
    asset_returns = daily_returns[universe]
    cash_returns = daily_returns[cash_ticker]
    strategy_targets, benchmark_targets = build_targets(asset_returns, config)
    start, split = config["backtest_start"], config["oos_start"]
    if not strategy_targets or min(strategy_targets) >= split:
        raise ValueError("There must be enough history for an execution before the OOS split.")
    print("PARAM_COMBOS_TRIED: 1")
    cost_rate = (config["cost_bps"] + config["slippage_bps"]) / 10_000
    strategy_results = simulate(asset_returns, strategy_targets, cost_rate).loc[start:]
    benchmark_results = simulate(asset_returns, benchmark_targets, cost_rate).loc[start:]
    strategy_report = report("STRATEGY", strategy_results, cash_returns, split)
    benchmark_report = report("BENCHMARK_EQUAL_WEIGHT", benchmark_results, cash_returns, split)
    output = {
        "strategy": config["strategy"],
        "data_range": f"{strategy_results.index[0].date()} to {strategy_results.index[-1].date()}",
        **strategy_report,
        "benchmark": benchmark_report["full"],
    }
    print("RESULTS_JSON: " + json.dumps(output, allow_nan=False))


if __name__ == "__main__":
    main()
