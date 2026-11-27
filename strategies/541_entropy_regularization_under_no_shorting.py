# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "541_entropy_regularization_under_no_shorting"\nuniverse: [SPY, TLT, GLD, VNQ, IEF]\nrisk_free_ticker: BIL\n\ndata_start: 2008-01-01\nsample_start: 2010-01-01\nis_end: 2018-12-31\nsample_end: 2025-12-31\n\nrebalance_period: M\nsignal_lag_sessions: 1\nannualization_sessions: 252\n\ndefault:\n  lookback_sessions: 252\n  risk_aversion: 5.0\n  entropy_strength: 0.02\n  cost_bps: 5.0\n  slippage_bps: 5.0\n\nrobustness:\n  entropy_lower: {entropy_strength: 0.01}\n  entropy_higher: {entropy_strength: 0.03}\n  risk_aversion_lower: {risk_aversion: 4.0}\n  risk_aversion_higher: {risk_aversion: 6.0}\n  longer_lookback: {lookback_sessions: 315}\n  higher_friction: {cost_bps: 10.0, slippage_bps: 10.0}\n'
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
import json
import math
import signal
from dataclasses import asdict, dataclass, replace
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
PRICE_ADJUSTMENT = "yfinance auto_adjust=True, daily dividend- and split-adjusted closes"


@dataclass(frozen=True)
class Parameters:
    lookback_sessions: int
    risk_aversion: float
    entropy_strength: float
    cost_bps: float
    slippage_bps: float

    def __post_init__(self):
        if not isinstance(self.lookback_sessions, int) or self.lookback_sessions < 2:
            raise ValueError("lookback_sessions must be an integer of at least 2.")
        values = (self.risk_aversion, self.entropy_strength, self.cost_bps, self.slippage_bps)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("Risk aversion, entropy strength, costs and slippage must be finite and nonnegative.")
        if self.entropy_strength == 0:
            raise ValueError("entropy_strength must be positive.")
        if self.friction >= 1:
            raise ValueError("Combined cost and slippage must be below 10,000 bps.")

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def check_prices(prices, tickers, start, end):
    if prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Expected nonempty adjusted prices with unique, sorted dates.")
    missing = set(tickers) - set(prices.columns)
    if missing:
        raise ValueError(f"Missing adjusted closes for {', '.join(sorted(missing))}.")
    prices = prices.loc[:, tickers]
    # Allow weekends and exchange holidays at the request boundaries.
    first, last = prices.index[[0, -1]]
    if not (0 <= (first - start).days <= 7 and 0 <= (end - last).days <= 7):
        raise ValueError(f"Adjusted close coverage is incomplete: {first.date()} to {last.date()}.")
    invalid = ~np.isfinite(prices) | (prices <= 0)
    if invalid.any().any():
        date = invalid.index[invalid.any(axis=1)][0]
        symbols = ", ".join(invalid.columns[invalid.loc[date]])
        raise ValueError(f"Missing or invalid adjusted close on {date.date()}: {symbols}.")
    return prices


def download_prices(tickers, start, end):
    for attempt in range(2):
        try:
            raw = yf.download(
                tickers,
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=True,
                interval="1d",
                threads=False,
                progress=False,
                timeout=8,
            )
            if raw is None or raw.empty:
                raise ValueError("yfinance returned no daily prices.")
            return check_prices(raw["Close"], tickers, start, end)
        except (KeyError, ValueError) as error:
            if attempt == 1:
                raise RuntimeError(f"Could not load adjusted closes from yfinance: {error}") from error


def load_prices(tickers, start, end):
    key = "_".join(sorted(tickers))
    path = CACHE / f"adjusted_close_{key}_{start.date()}_{end.date()}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as source:
            fetched_at = datetime.fromisoformat(source.readline().strip()).isoformat()
            prices = pd.read_csv(
                source, index_col="Date", parse_dates=["Date"], converters=dict.fromkeys(tickers, float)
            )
        return check_prices(prices, tickers, start, end), fetched_at

    CACHE.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=CACHE, prefix="yfinance-") as yf_cache:
        yf.set_tz_cache_location(yf_cache)
        prices = download_prices(tickers, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Replace the complete snapshot atomically, including its fetch time.
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as destination:
        temporary_path = Path(destination.name)
        try:
            destination.write(fetched_at + "\n")
            prices.to_csv(destination, index_label="Date")
            destination.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def rebalance_schedule(dates, start, period, lag):
    periods = dates.to_period(period)
    period_ends = np.flatnonzero(periods[:-1] != periods[1:])
    schedule = {}
    for signal_row in period_ends:
        fill_row = signal_row + lag
        if fill_row < len(dates) and dates[fill_row] >= start:
            schedule[fill_row] = signal_row
    if not schedule:
        raise ValueError("No scheduled fills fall within the sample.")
    return schedule


def optimal_weights(window, parameters, annual_sessions):
    n_assets = window.shape[1]
    mean = window.mean(axis=0) * annual_sessions
    risk = np.cov(window, rowvar=False) * (annual_sessions * parameters.risk_aversion)
    entropy_strength = parameters.entropy_strength
    ones = np.ones(n_assets)
    equal = ones / n_assets

    def objective(weights):
        log_ratio = np.log(weights * n_assets)
        quadratic_gradient = risk @ weights
        value = weights @ (0.5 * quadratic_gradient - mean + entropy_strength * log_ratio)
        gradient = quadratic_gradient - mean + entropy_strength * (log_ratio + 1.0)
        return value, gradient

    fit = minimize(
        objective,
        equal,
        jac=True,
        method="SLSQP",
        bounds=[(1e-12, 1.0)] * n_assets,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0, "jac": lambda weights: ones},
        options={"ftol": 1e-12, "maxiter": 250},
    )
    weights = fit.x
    if not fit.success or not np.isfinite(weights).all() or weights.min() < 0 or abs(weights.sum() - 1) > 1e-9:
        raise RuntimeError(f"Weight optimization failed: {fit.message}")
    return weights / weights.sum()


def fill_at_close(weights, target, friction):
    def balance(retained):
        traded = np.abs(retained * target - weights).sum()
        return retained + friction * traded - 1.0

    # Targets are fractions of the post-cost value. Solve for that value first.
    retained = brentq(balance, 0.0, 1.0, xtol=1e-14) if friction else 1.0
    trades = np.abs(retained * target - weights)
    return retained, float(trades.sum()), int(np.count_nonzero(trades > 1e-8))


def simulate(asset_returns, start, targets, friction):
    first_row = asset_returns.index.searchsorted(start)
    weights = np.zeros(asset_returns.shape[1])
    rows = []
    for row, returns in enumerate(asset_returns.to_numpy()[first_row:], start=first_row):
        gross = 1.0 + weights @ returns
        weights = weights * (1.0 + returns) / gross
        traded = n_trades = 0
        if row in targets:
            target = targets[row]
            retained, traded, n_trades = fill_at_close(weights, target, friction)
            gross *= retained
            weights = target
        rows.append((gross - 1.0, traded, n_trades))
    return pd.DataFrame(rows, index=asset_returns.index[first_row:], columns=["return", "traded", "n_trades"])


def run_strategy(asset_returns, start, schedule, parameters, annual_sessions):
    returns = asset_returns.to_numpy()
    targets = {}
    for fill_row, signal_row in schedule.items():
        window_start = signal_row + 1 - parameters.lookback_sessions
        if window_start < 0:
            raise ValueError(f"Insufficient history at signal close {asset_returns.index[signal_row].date()}.")
        window = returns[window_start : signal_row + 1]
        try:
            targets[fill_row] = optimal_weights(window, parameters, annual_sessions)
        except RuntimeError as error:
            raise RuntimeError(f"Signal close {asset_returns.index[signal_row].date()}: {error}") from error
    return simulate(asset_returns, start, targets, parameters.friction)


def metrics(daily_results, risk_free_returns, annual_sessions):
    returns = daily_results["return"].to_numpy()
    if len(returns) < 2:
        raise ValueError("Each reported period must have at least two daily returns.")
    excess = returns - risk_free_returns.loc[daily_results.index].to_numpy()
    annual_scale = math.sqrt(annual_sessions)
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    growth = np.r_[1.0, np.cumprod(1.0 + returns)]
    return {
        "cagr": float(growth[-1] ** (annual_sessions / len(returns)) - 1.0),
        "ann_vol": float(volatility * annual_scale),
        "sharpe": float(returns.mean() / volatility * annual_scale) if volatility else None,
        "sharpe_excess": float(excess.mean() / excess_volatility * annual_scale) if excess_volatility else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1.0).min()),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["traded"].mean() * annual_sessions),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    data_start, start, is_end, end = (
        pd.Timestamp(config[key]) for key in ("data_start", "sample_start", "is_end", "sample_end")
    )
    if not data_start < start < is_end < end:
        raise ValueError("Expected data_start < sample_start < is_end < sample_end.")
    universe = config["universe"]
    risk_free_ticker = config["risk_free_ticker"]
    tickers = [*universe, risk_free_ticker]
    if len(universe) < 2 or len(set(tickers)) != len(tickers) or any(not ticker.isalnum() for ticker in tickers):
        raise ValueError("Set at least two distinct ETF symbols and a separate risk-free proxy.")
    lag = config["signal_lag_sessions"]
    annual_sessions = config["annualization_sessions"]
    if not isinstance(lag, int) or lag < 1 or not isinstance(annual_sessions, int) or annual_sessions < 1:
        raise ValueError("Execution lag and annualization must be positive session counts.")
    period = config["rebalance_period"]
    official = Parameters(**config["default"])
    cases = {"default": official}
    for label, adjustments in config["robustness"].items():
        case = replace(official, **adjustments)
        if label in cases or case in cases.values():
            raise ValueError(f"Repeated parameter combination or label: {label}.")
        cases[label] = case

    prices, fetched_at = load_prices(tickers, data_start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    daily_returns = prices.pct_change(fill_method=None).iloc[1:]
    asset_returns = daily_returns[universe]
    risk_free_returns = daily_returns[risk_free_ticker]

    is_returns = asset_returns.loc[:is_end]
    is_risk_free = risk_free_returns.loc[:is_end]
    is_schedule = rebalance_schedule(is_returns.index, start, period, lag)
    print("ROBUSTNESS (IS only)")
    for label, case in cases.items():
        daily_results = run_strategy(is_returns, start, is_schedule, case, annual_sessions)
        result = {"case": label, **asdict(case), **metrics(daily_results, is_risk_free, annual_sessions)}
        print(json.dumps(result, allow_nan=False))
    print(f"PARAM_COMBOS_TRIED: {len(cases)}")

    schedule = rebalance_schedule(asset_returns.index, start, period, lag)
    daily_results = run_strategy(asset_returns, start, schedule, official, annual_sessions)
    equal = np.full(len(universe), 1.0 / len(universe))
    benchmark = simulate(asset_returns, start, dict.fromkeys(schedule, equal), official.friction)
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": str(daily_results.index[0].date()), "end": str(daily_results.index[-1].date())},
        "is": metrics(daily_results.loc[:is_end], risk_free_returns, annual_sessions),
        "oos": metrics(daily_results.loc[daily_results.index > is_end], risk_free_returns, annual_sessions),
        "full": metrics(daily_results, risk_free_returns, annual_sessions),
        "benchmark": metrics(benchmark, risk_free_returns, annual_sessions),
    }
    for label in ("is", "oos", "full"):
        print(f"{label.upper()}: {json.dumps(results[label], allow_nan=False)}")
    print(f"BENCHMARK (equal weight, same universe and schedule): {json.dumps(results['benchmark'], allow_nan=False)}")
    print(f"RESULTS_JSON: {json.dumps(results, allow_nan=False)}")


if __name__ == "__main__":
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)
