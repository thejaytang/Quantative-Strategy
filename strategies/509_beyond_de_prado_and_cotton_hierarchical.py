# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 509_beyond_de_prado_and_cotton_hierarchical\nuniverse: [SPY, EFA, EEM, IEF, TLT, LQD, GLD, VNQ, DBC]\nrisk_free: BIL\ndata_start: '2007-06-01'\nsample_start: '2008-10-01'\nis_end: '2017-12-31'\nsample_end: '2025-12-31'\nexecution_lag_sessions: 1\ncost_bps: 5\nslippage_bps: 5\ngamma: 0.5\nlookback_sessions: 252\nmax_weight: 0.30\nrobustness:\n  gamma: [0.3, 0.7]\n  lookback_sessions: [189, 315]\n  friction_multiplier: 2\n"
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
"""Monthly long-only CRISP shrinkage research on a fixed ETF universe."""

import hashlib
import json
import math
import re
import signal
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
FETCH_PREFIX = "# DATA_FETCHED_AT: "


@dataclass(frozen=True)
class Specification:
    gamma: float
    lookback: int
    cost_bps: float
    slippage_bps: float

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def check_prices(close, symbols, start, end):
    if (
        close.empty or not isinstance(close.index, pd.DatetimeIndex)
        or close.index.hasnans or not close.index.is_unique or not close.index.is_monotonic_increasing
    ):
        raise ValueError("Expected sorted, distinct daily observations from Yahoo Finance.")
    if close.index[0] < start or close.index[-1] > end:
        raise ValueError("Price dates fall outside the requested range.")
    edge_tolerance = pd.Timedelta(days=7)
    if close.index[0] > start + edge_tolerance or close.index[-1] < end - edge_tolerance:
        raise ValueError("Prices do not cover the requested date range.")
    close = close.reindex(columns=symbols)
    prices = close.to_numpy(dtype=float)
    incomplete = ~(np.isfinite(prices) & (prices > 0)).all(axis=0)
    if incomplete.any():
        missing = ", ".join(np.asarray(symbols)[incomplete])
        raise ValueError(f"Missing or invalid adjusted closes for {missing}.")
    return close


def load_prices(symbols, start, end):
    CACHE_DIR.mkdir(exist_ok=True)
    key = json.dumps([symbols, str(start.date()), str(end.date()), "yfinance-adjusted-close-v1"])
    cache_path = CACHE_DIR / f"prices-{hashlib.sha256(key.encode()).hexdigest()[:16]}.csv"
    if cache_path.exists():
        try:
            with cache_path.open(encoding="utf-8") as source:
                header = source.readline().strip()
                if not header.startswith(FETCH_PREFIX):
                    raise ValueError("Missing fetch timestamp.")
                fetched_at = header.removeprefix(FETCH_PREFIX)
                if datetime.fromisoformat(fetched_at).tzinfo is None:
                    raise ValueError("Fetch timestamp must have a timezone.")
                close = pd.read_csv(source, index_col="Date", parse_dates=["Date"])
            return check_prices(close, symbols, start, end), fetched_at
        except (OSError, ValueError) as error:
            print(f"Ignoring price cache: {error}", file=sys.stderr)

    with TemporaryDirectory(dir=CACHE_DIR) as scratch:
        yf.set_tz_cache_location(scratch)
        for attempt in range(2):
            downloaded = yf.download(
                symbols,
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d",
                auto_adjust=True,
                ignore_tz=True,
                threads=False,
                progress=False,
                timeout=8,
            )
            try:
                if downloaded is None or downloaded.empty or "Close" not in downloaded:
                    raise ValueError("Yahoo Finance returned no adjusted close data.")
                close = check_prices(downloaded["Close"], symbols, start, end)
                break
            except ValueError as error:
                if attempt == 1:
                    raise RuntimeError(f"Unable to retrieve complete Yahoo Finance prices: {error}") from error
                print(f"Retrying price fetch: {error}", file=sys.stderr)

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=CACHE_DIR, delete=False
    ) as destination:
        temporary_path = Path(destination.name)
        destination.write(f"{FETCH_PREFIX}{fetched_at}\n")
        close.to_csv(destination, index_label="Date")
    temporary_path.replace(cache_path)
    return close, fetched_at


def crisp_weights(history, gamma, max_weight):
    # Section 9.2, Eq. (53), with a budget and long-only per-ETF bounds.
    expected = history.mean(axis=0) * SESSIONS_PER_YEAR
    covariance = np.cov(history, rowvar=False, ddof=1) * SESSIONS_PER_YEAR
    shrunk = gamma * covariance + (1 - gamma) * np.diag(np.diag(covariance))
    # Positive rescaling improves solver tolerances without changing the minimizer.
    scale = max(np.abs(expected).max(), np.abs(shrunk).max(), 1e-12)
    expected /= scale
    shrunk /= scale
    initial = np.full(len(expected), 1 / len(expected))
    result = minimize(
        lambda weights: 0.5 * (weights @ shrunk @ weights) - expected @ weights,
        initial,
        jac=lambda weights: shrunk @ weights - expected,
        bounds=[(0.0, max_weight)] * len(expected),
        constraints={
            "type": "eq",
            "fun": lambda weights: weights.sum() - 1,
            "jac": lambda weights: np.ones_like(weights),
        },
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 150},
    )
    weights = result.x
    if (
        not result.success
        or not np.isfinite(weights).all()
        or abs(weights.sum() - 1) > 1e-8
        or weights.min() < -1e-8
        or weights.max() > max_weight + 1e-8
    ):
        raise RuntimeError(f"CRISP allocation failed: {result.message}")
    return weights


def rebalance_schedule(dates, sample_start, lag):
    months = dates.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])
    return [
        (dates[row + lag], row)
        for row in month_ends
        if row + lag < len(dates) and dates[row + lag] >= sample_start
    ]


def trade_at_close(pretrade, target, friction):
    # Solve the cost against actual purchases and sales of the post-cost target.
    remaining = 1.0
    for _ in range(20):
        orders = np.abs(remaining * target - pretrade)
        turnover = float(orders.sum())
        after_cost = 1.0 - friction * turnover
        if abs(after_cost - remaining) < 1e-13:
            return after_cost, turnover, int(np.count_nonzero(orders > 1e-8))
        remaining = after_cost
    raise RuntimeError("Trading friction calculation did not converge.")


def simulate(asset_returns, targets, friction):
    weights = np.zeros(asset_returns.shape[1])
    rows = []
    for date, daily_return in zip(asset_returns.index, asset_returns.to_numpy()):
        growth = 1.0 + float(weights @ daily_return)
        weights = weights * (1 + daily_return) / growth
        turnover = 0.0
        trades = 0
        if date in targets:
            target = targets[date]
            remaining, turnover, trades = trade_at_close(weights, target, friction)
            growth *= remaining
            weights = target
        rows.append((growth - 1.0, turnover, trades))
    return pd.DataFrame(rows, columns=["return", "turnover", "n_trades"], index=asset_returns.index)


def run_strategy(asset_returns, schedule, sample_start, spec, max_weight):
    targets = {}
    for fill_date, signal_row in schedule:
        first_row = signal_row - spec.lookback + 1
        history = asset_returns.iloc[first_row : signal_row + 1].to_numpy()
        targets[fill_date] = crisp_weights(history, spec.gamma, max_weight)
    return simulate(asset_returns.loc[sample_start:], targets, spec.friction)


def annual_sharpe(returns):
    vol = returns.std(ddof=1)
    return float(returns.mean() * math.sqrt(SESSIONS_PER_YEAR) / vol) if vol > 0 else None


def metrics(daily_results, risk_free_returns):
    returns = daily_results["return"]
    factors = np.r_[1.0, (1.0 + returns.to_numpy()).cumprod()]
    years = len(returns) / SESSIONS_PER_YEAR
    return {
        "cagr": float(factors[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(returns.std(ddof=1) * math.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": annual_sharpe(returns),
        "sharpe_excess": annual_sharpe(returns - risk_free_returns.loc[returns.index]),
        "max_drawdown": float((factors / np.maximum.accumulate(factors) - 1.0).min()),
        "total_return": float(factors[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, risk_free_returns, is_end):
    is_mask = daily_results.index <= is_end
    return {
        "is": metrics(daily_results.loc[is_mask], risk_free_returns),
        "oos": metrics(daily_results.loc[~is_mask], risk_free_returns),
        "full": metrics(daily_results, risk_free_returns),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    primary = Specification(
        config["gamma"], config["lookback_sessions"], config["cost_bps"], config["slippage_bps"]
    )
    robustness = config["robustness"]
    multiplier = robustness["friction_multiplier"]
    variants = [
        ("gamma", replace(primary, gamma=value)) for value in robustness["gamma"]
    ] + [
        ("lookback", replace(primary, lookback=value)) for value in robustness["lookback_sessions"]
    ] + [
        ("higher_friction", replace(
            primary, cost_bps=primary.cost_bps * multiplier,
            slippage_bps=primary.slippage_bps * multiplier,
        )),
    ]
    # Count distinct strategy specifications, including the default.
    seen = {primary}
    checks = []
    for label, spec in variants:
        if spec not in seen:
            checks.append((label, spec))
            seen.add(spec)

    universe = config["universe"]
    risk_free = config["risk_free"]
    symbols = [*universe, risk_free]
    if (
        len(universe) < 2 or len(set(symbols)) != len(symbols) or risk_free != "BIL"
        or not all(re.fullmatch(r"[A-Z]{1,5}", symbol) for symbol in symbols)
    ):
        raise ValueError("Use distinct US ETF symbols and BIL as the cash-return proxy.")
    data_start, sample_start, is_end, sample_end = (
        pd.Timestamp(config[key]) for key in ("data_start", "sample_start", "is_end", "sample_end")
    )
    lag = config["execution_lag_sessions"]
    max_weight = config["max_weight"]
    if not data_start < sample_start <= is_end < sample_end:
        raise ValueError("Invalid data, sample, or split dates.")
    if not isinstance(lag, int) or lag < 1 or not 1 / len(universe) <= max_weight <= 1:
        raise ValueError("The lag must be positive and the upper weight limit feasible.")
    for spec in seen:
        if not (
            0 <= spec.gamma <= 1
            and isinstance(spec.lookback, int) and spec.lookback >= 2
            and spec.cost_bps >= 0 and spec.slippage_bps >= 0 and spec.friction < 0.1
        ):
            raise ValueError(f"Invalid shrinkage, lookback, or friction: {spec}")

    close, fetched_at = load_prices(symbols, data_start, sample_end)
    returns = close.pct_change(fill_method=None)
    schedule = rebalance_schedule(returns.index, sample_start, lag)
    is_asset_returns = returns.loc[:is_end, universe]
    is_schedule = rebalance_schedule(is_asset_returns.index, sample_start, lag)
    if not is_schedule or schedule[0][1] < max(spec.lookback for spec in seen):
        raise ValueError("Provide sufficient pre-sample history and an in-sample rebalance.")
    if min(len(returns.loc[sample_start:is_end]), len(returns.loc[returns.index > is_end])) < 2:
        raise ValueError("Each reporting period requires at least two daily returns.")
    risk_free_returns = returns[risk_free]

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, dividend- and split-adjusted close")
    print(f"PARAM_COMBOS_TRIED: {len(seen)}")
    print("ROBUSTNESS (IS only)")
    for label, spec in [("DEFAULT", primary), *checks]:
        check = run_strategy(is_asset_returns, is_schedule, sample_start, spec, max_weight)
        summary = metrics(check, risk_free_returns.loc[:is_end])
        print(f"  {label} {spec}", json.dumps(summary, allow_nan=False))

    results = run_strategy(returns[universe], schedule, sample_start, primary, max_weight)
    equal = np.full(len(universe), 1 / len(universe))
    benchmark_results = simulate(
        returns.loc[sample_start:, universe],
        {date: equal for date, _ in schedule},
        primary.friction,
    )

    official = period_metrics(results, risk_free_returns, is_end)
    benchmark = period_metrics(benchmark_results, risk_free_returns, is_end)
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(official[period], allow_nan=False)}")
    print("BENCHMARK: equal-weight monthly, same universe, lag and friction")
    for period in ("is", "oos", "full"):
        print(f"  {period.upper()}: {json.dumps(benchmark[period], allow_nan=False)}")
    output = {
        "strategy": config["strategy"],
        "data_range": {
            "start": results.index[0].date().isoformat(),
            "end": results.index[-1].date().isoformat(),
        },
        **official,
        "benchmark": {"name": "monthly_equal_weight", **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(output, allow_nan=False))


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, lambda *_: sys.exit("Stopped at the 14-minute runtime limit."))
    signal.alarm(14 * 60)
    main()
