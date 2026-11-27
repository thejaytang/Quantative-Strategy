# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 551_squeezed_covariance_matrix_estimation_an\n\n# Fixed research basket of large U.S. companies with long price histories.\nuniverse:\n  - AAPL\n  - ABT\n  - ACN\n  - ADBE\n  - AMGN\n  - AMZN\n  - AXP\n  - BAC\n  - BLK\n  - BMY\n  - CAT\n  - CL\n  - CMCSA\n  - COP\n  - COST\n  - CRM\n  - CSCO\n  - CVX\n  - DE\n  - DHR\n  - DIS\n  - GILD\n  - GS\n  - HD\n  - HON\n  - IBM\n  - INTU\n  - JNJ\n  - JPM\n  - KO\n  - LMT\n  - LOW\n  - MA\n  - MCD\n  - MDT\n  - MRK\n  - MS\n  - MSFT\n  - NKE\n  - NVDA\n  - ORCL\n  - PEP\n  - PG\n  - QCOM\n  - TXN\n  - UNH\n  - UPS\n  - V\n  - WMT\n  - XOM\nbenchmark: SPY\nrisk_free_proxy: BIL\n\ndata_start: 2008-06-01\nbacktest_start: 2010-01-01\nis_end: 2016-12-31\nbacktest_end: 2024-12-31\n\nexecution_lag_sessions: 1\nlookback_sessions: 252\ncost_bps: 10\nslippage_bps: 5\nestimator: atomic_iq\nnoise_gate: 0.5\nchannel_boundary: 1.5\nchannel_weights: [0.956, 0.870, 0.979]  # body, wing, tail\nminimum_correlation_eigenvalue: 0.05\n\nrobustness:\n  lower_boundary:\n    channel_boundary: 1.25\n  higher_boundary:\n    channel_boundary: 1.75\n  lower_eigenvalue_floor:\n    minimum_correlation_eigenvalue: 0.025\n  higher_eigenvalue_floor:\n    minimum_correlation_eigenvalue: 0.075\n  higher_cost:\n    cost_bps: 20\n  sample_correlation:\n    estimator: sample\n'
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
"""Research backtest for a squeezed covariance long-only tangency portfolio."""

import gzip
import hashlib
import json
import signal
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.linalg import solve_triangular
from scipy.optimize import brentq, nnls


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
PRICE_ADJUSTMENT = "yfinance auto_adjust=True, split- and dividend-adjusted close"


@dataclass(frozen=True)
class Model:
    execution_lag_sessions: int
    lookback_sessions: int
    cost_bps: float
    slippage_bps: float
    estimator: str
    noise_gate: float
    channel_boundary: float
    channel_weights: tuple[float, float, float]
    minimum_correlation_eigenvalue: float

    def __post_init__(self):
        if type(self.execution_lag_sessions) is not int or self.execution_lag_sessions < 1:
            raise ValueError("execution_lag_sessions must be a positive integer")
        if type(self.lookback_sessions) is not int or self.lookback_sessions < 2:
            raise ValueError("lookback_sessions must be an integer of at least two")
        if not (0 <= self.noise_gate < self.channel_boundary < np.inf):
            raise ValueError("Require 0 <= noise_gate < channel_boundary")
        if len(self.channel_weights) != 3 or not all(0 <= w <= 1 for w in self.channel_weights):
            raise ValueError("Require three channel weights in [0, 1]")
        if not 0 < self.minimum_correlation_eigenvalue <= 1:
            raise ValueError("The correlation eigenvalue floor must be in (0, 1]")
        if not (self.cost_bps >= 0 and self.slippage_bps >= 0 and self.rate < 1):
            raise ValueError("Costs and slippage must be nonnegative and total under 10000 bps")
        if self.estimator not in {"atomic_iq", "sample"}:
            raise ValueError(f"Unknown estimator {self.estimator!r}")

    @property
    def rate(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def validate_prices(prices, symbols, start, end):
    if prices.empty or not set(symbols).issubset(prices.columns):
        raise ValueError("Missing required columns or dates in the price panel")
    dates = prices.index
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Price dates must be unique and increasing")
    if dates[0] < start or dates[-1] > end:
        raise ValueError("Prices fall outside the requested dates")
    if (dates[0] - start).days > 7 or (end - dates[-1]).days > 7:
        raise ValueError("Price history does not span the requested dates")
    values = prices.loc[:, symbols].to_numpy()
    bad = ~np.isfinite(values) | (values <= 0)
    if bad.any():
        row, col = np.argwhere(bad)[0]
        raise ValueError(f"Missing or invalid adjusted close for {symbols[col]} on {dates[row].date()}")


def adjusted_closes(symbols, start, end):
    raw = yf.download(
        symbols,
        start=start.date().isoformat(),
        end=(end + pd.Timedelta(days=1)).date().isoformat(),
        interval="1d",
        auto_adjust=True,
        ignore_tz=True,
        repair=False,
        progress=False,
        # Serial requests avoid concurrent writes to yfinance's new SQLite cache.
        threads=False,
        timeout=10,
    )
    if raw is None or raw.empty or "Close" not in raw:
        raise ValueError(f"yfinance returned no adjusted closes for {', '.join(symbols)}")
    closes = raw["Close"].reindex(columns=symbols)
    missing = closes.columns[closes.isna().all()]
    if len(missing):
        raise ValueError(f"yfinance returned no adjusted closes for {', '.join(missing)}")
    return closes


def load_prices(universe, benchmark, risk_free, start, end):
    symbols = [benchmark, risk_free, *universe]
    cache_key = json.dumps([PRICE_ADJUSTMENT, start.isoformat(), end.isoformat(), symbols])
    fingerprint = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    path = CACHE_DIR / f"adjusted_closes_{fingerprint}.csv.gz"
    if path.is_file():
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            fetched_at = stream.readline().strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(stream, index_col="date", parse_dates=["date"])
        prices.index = prices.index.astype("datetime64[ns]")
        validate_prices(prices, symbols, start, end)
        return prices, fetched_at

    CACHE_DIR.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=".yfinance-", dir=CACHE_DIR) as session_cache:
        yf.set_tz_cache_location(session_cache)
        reference = adjusted_closes([benchmark, risk_free], start, end)
        reference = reference.loc[reference[benchmark].notna()]
        validate_prices(reference, [benchmark, risk_free], start, end)

        batches = [reference]
        remaining = iter(universe)
        while batch := list(islice(remaining, 10)):
            stocks = adjusted_closes(batch, start, end).reindex(reference.index)
            validate_prices(stocks, batch, start, end)
            batches.append(stocks)
    prices = pd.concat(batches, axis=1).rename_axis(columns=None)
    prices.index = prices.index.astype("datetime64[ns]").rename("date")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with NamedTemporaryFile(dir=CACHE_DIR, suffix=".csv.gz", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        with gzip.open(temp_path, "wt", encoding="utf-8") as stream:
            stream.write(f"{fetched_at}\n")
            prices.to_csv(stream, index_label="date")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def atomic_correlation(window, vols, model):
    n = len(vols)
    left, right = np.triu_indices(n)
    off_diagonal = left != right
    pair_scale = np.maximum(vols[left], vols[right])
    first = window[:, left]
    second = window[:, right]
    informative = np.minimum(np.abs(first), np.abs(second)) > model.noise_gate * pair_scale
    first_tail = np.abs(first) > model.channel_boundary * pair_scale
    second_tail = np.abs(second) > model.channel_boundary * pair_scale
    direction = np.sign(first) * np.sign(second)

    identity = np.eye(n)
    squeezed = (1.0 - sum(model.channel_weights)) * identity
    for weight, channel in zip(
        model.channel_weights,
        (~(first_tail | second_tail), first_tail ^ second_tail, first_tail & second_tail),
    ):
        events = informative & channel
        counts = events.sum(axis=0)
        # Each atom contributes one unit to each involved diagonal.
        mass = np.bincount(left, weights=counts, minlength=n)
        mass += np.bincount(right[off_diagonal], weights=counts[off_diagonal], minlength=n)
        signed_counts = (direction * events).sum(axis=0)[off_diagonal]
        rows = left[off_diagonal]
        cols = right[off_diagonal]
        scale = np.sqrt(mass[rows] * mass[cols])
        correlations = np.divide(signed_counts, scale, out=np.zeros_like(scale), where=scale > 0)
        matrix = identity.copy()
        matrix[rows, cols] = matrix[cols, rows] = correlations
        squeezed += weight * matrix
    return squeezed


def estimate_covariance(window, model):
    vols = window.std(axis=0, ddof=1)
    if not np.isfinite(vols).all() or (vols <= 0).any():
        raise ValueError("Every asset needs finite, positive volatility in the estimation window")
    if model.estimator == "sample":
        correlation = np.eye(1) if len(vols) == 1 else np.corrcoef(window, rowvar=False)
    else:
        correlation = atomic_correlation(window, vols, model)

    # Paper equation (20). Mixing with identity preserves the unit diagonal.
    smallest = np.linalg.eigvalsh(correlation)[0]
    floor = model.minimum_correlation_eigenvalue
    if smallest < floor:
        phi = (floor - smallest) / (1 - smallest)
        correlation = (1 - phi) * correlation + phi * np.eye(len(vols))
    return correlation * np.outer(vols, vols)


def tangency_weights(excess_mean, covariance):
    if excess_mean.max() <= 0:
        # With all means nonpositive, the triangle inequality puts an optimum at a vertex.
        weights = np.zeros_like(excess_mean)
        weights[np.argmax(excess_mean / np.sqrt(np.diag(covariance)))] = 1.0
        return weights

    # The nonnegative quadratic problem gives the tangency direction up to scale.
    root = np.linalg.cholesky(covariance)
    rhs = solve_triangular(root, excess_mean, lower=True)
    direction, _ = nnls(root.T, rhs)
    if direction.sum() <= 0:
        raise RuntimeError("Tangency optimization returned no positive weights")
    return direction / direction.sum()


def make_targets(stock_returns, risk_free_returns, start, model):
    dates = stock_returns.index
    months = dates.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])
    earliest_execution = dates.searchsorted(start)
    samples = stock_returns.to_numpy()
    risk_free = risk_free_returns.to_numpy()
    targets = {}
    for close in month_ends:
        execution = close + model.execution_lag_sessions
        if close < model.lookback_sessions or not earliest_execution <= execution < len(dates):
            continue
        window = slice(close - model.lookback_sessions + 1, close + 1)
        sample = samples[window]
        excess_mean = sample.mean(axis=0) - risk_free[window].mean()
        targets[dates[execution]] = tangency_weights(excess_mean, estimate_covariance(sample, model))
    if not targets:
        raise ValueError("No eligible executions in the requested period")
    return targets


def simulate(asset_returns, targets, rate):
    weights = np.zeros(asset_returns.shape[1])
    daily_results = []
    for day, returns in zip(asset_returns.index, asset_returns.to_numpy()):
        gross = 1.0 + weights @ returns
        weights = weights * (1.0 + returns) / gross
        multiplier, turnover, trades = 1.0, 0.0, 0
        if day in targets:
            target = targets[day]

            def cost_balance(fraction):
                return fraction + rate * np.abs(fraction * target - weights).sum() - 1.0

            multiplier = brentq(cost_balance, 0.0, 1.0)
            traded = np.abs(multiplier * target - weights)
            turnover = traded.sum()
            trades = int(np.count_nonzero(traded > 1e-8))
            weights = target
        daily_results.append((gross * multiplier - 1.0, turnover, trades))
    return pd.DataFrame(daily_results, index=asset_returns.index, columns=["returns", "turnover", "n_trades"])


def metrics(daily_results, risk_free_returns):
    gains = daily_results["returns"]
    if len(gains) < 2:
        raise ValueError("Each reported period requires at least two sessions")
    excess = gains - risk_free_returns.loc[gains.index]
    volatility = gains.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    years = len(gains) / SESSIONS_PER_YEAR
    cumulative = np.r_[1.0, (1.0 + gains).cumprod()]
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(cumulative[-1] ** (1 / years) - 1),
        "ann_vol": float(volatility * annualizer),
        "sharpe": float(gains.mean() / volatility * annualizer) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_volatility * annualizer) if excess_volatility > 0 else None,
        "max_drawdown": float(np.min(cumulative / np.maximum.accumulate(cumulative) - 1)),
        "total_return": float(cumulative[-1] - 1),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, risk_free_returns, is_end):
    return {
        "is": metrics(daily_results.loc[:is_end], risk_free_returns),
        "oos": metrics(daily_results.loc[daily_results.index > is_end], risk_free_returns),
        "full": metrics(daily_results, risk_free_returns),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    model = Model(**{key: config[key] for key in Model.__dataclass_fields__})
    cases = {name: replace(model, **changes) for name, changes in config["robustness"].items()}
    universe = config["universe"]
    benchmark = config["benchmark"]
    risk_free = config["risk_free_proxy"]
    symbols = [*universe, benchmark, risk_free]
    if not universe or len(set(symbols)) != len(symbols):
        raise ValueError("The universe, benchmark and risk-free symbols must be distinct")
    data_start, start, is_end, end = (
        pd.Timestamp(config[key]) for key in ("data_start", "backtest_start", "is_end", "backtest_end")
    )
    if not data_start < start < is_end < end:
        raise ValueError("Require data_start < backtest_start < is_end < backtest_end")

    prices, fetched_at = load_prices(universe, benchmark, risk_free, data_start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    returns = prices.pct_change(fill_method=None)
    stock_returns = returns[universe]
    risk_free_returns = returns[risk_free]
    evaluation = stock_returns.loc[start:]

    # Evaluate alternatives with inputs bounded by the IS end date.
    is_stocks = stock_returns.loc[:is_end]
    is_risk_free = risk_free_returns.loc[:is_end]
    robustness = {}
    combos_tried = 1
    for name, case in cases.items():
        targets = make_targets(is_stocks, is_risk_free, start, case)
        daily_results = simulate(is_stocks.loc[start:], targets, case.rate)
        robustness[name] = metrics(daily_results, is_risk_free)
        combos_tried += 1

    targets = make_targets(stock_returns, risk_free_returns, start, model)
    official = period_metrics(simulate(evaluation, targets, model.rate), risk_free_returns, is_end)
    benchmark_targets = {min(targets): np.ones(1)}
    benchmark_daily = simulate(returns.loc[start:, [benchmark]], benchmark_targets, model.rate)
    comparison = period_metrics(benchmark_daily, risk_free_returns, is_end)

    print(f"PARAM_COMBOS_TRIED: {combos_tried}")
    print("ROBUSTNESS (IS ONLY)")
    print("default:", json.dumps(official["is"], allow_nan=False))
    for name, values in robustness.items():
        print(f"{name}:", json.dumps(values, allow_nan=False))
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}:", json.dumps(official[period], allow_nan=False))
        print(f"BENCHMARK {benchmark} {period.upper()}:", json.dumps(comparison[period], allow_nan=False))

    result = {
        "strategy": config["strategy"],
        "data_range": {
            "start": evaluation.index[0].date().isoformat(),
            "end": evaluation.index[-1].date().isoformat(),
        },
        **official,
        "benchmark": {"name": benchmark, **comparison},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


def time_limit(_signum, _frame):
    raise SystemExit("Backtest exceeded its 14-minute runtime limit")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, time_limit)
    signal.alarm(14 * 60)
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from error
    finally:
        signal.alarm(0)
