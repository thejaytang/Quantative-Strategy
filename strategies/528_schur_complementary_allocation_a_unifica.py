# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: '528_ schur_complementary_allocation_a_unifica'\nuniverse: [SPY, TLT, GLD, VNQ, DBC]\ncash_proxy: BIL\ndata_start: 2007-06-01\nbacktest_start: 2009-01-01\nis_end: 2017-12-31\nbacktest_end: 2025-12-31\nlag_sessions: 1\n\nparameters:\n  lookback: 252\n  gamma: 0.50\n  covariance_shrinkage: 0.03\n  cost_bps: 5\n  slippage_bps: 5\n\nrobustness:\n  no_schur_coupling: {gamma: 0.00}\n  lower_coupling: {gamma: 0.25}\n  higher_coupling: {gamma: 0.75}\n  shorter_window: {lookback: 189}\n  longer_window: {lookback: 315}\n  higher_friction: {cost_bps: 10, slippage_bps: 10}\n"
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
"""Monthly long-only Schur allocation research backtest."""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import brentq
from scipy.spatial.distance import squareform
import yaml
import yfinance as yf


FOLDER = Path(__file__).resolve().parent
CACHE_DIR = FOLDER / "data_cache"
SESSIONS_PER_YEAR = 252
PRICE_ADJUSTMENT = "yfinance auto_adjust=True, split- and distribution-adjusted close (total-return proxy)"


@dataclass(frozen=True)
class Parameters:
    lookback: int
    gamma: float
    covariance_shrinkage: float
    cost_bps: float
    slippage_bps: float

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000

    def validate(self):
        if type(self.lookback) is not int or self.lookback < 2:
            raise ValueError("lookback must be an integer >= 2")
        if not 0 <= self.gamma <= 1 or not 0 < self.covariance_shrinkage <= 1:
            raise ValueError("gamma must be in [0, 1] and covariance_shrinkage in (0, 1]")
        if not (self.cost_bps >= 0 and self.slippage_bps >= 0 and self.friction < 1):
            raise ValueError("Costs and slippage must be nonnegative and sum to less than 10000 bps")


def validate_prices(prices, tickers, start, end):
    if prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Expected a nonempty panel with unique, ascending session dates")
    if list(prices.columns) != tickers:
        raise ValueError("Price columns do not match the requested tickers")
    if not (start <= prices.index[0] <= start + pd.Timedelta(days=7)
            and end - pd.Timedelta(days=7) <= prices.index[-1] <= end):
        raise ValueError("Downloaded prices do not cover the requested dates")
    values = prices.to_numpy(dtype=float)
    invalid = ~np.isfinite(values) | (values <= 0)
    if invalid.any():
        row, col = np.argwhere(invalid)[0]
        raise ValueError(f"Missing or invalid close for {tickers[col]} on {prices.index[row].date()}")


def load_prices(tickers, start, end):
    CACHE_DIR.mkdir(exist_ok=True)
    request = ["yfinance", PRICE_ADJUSTMENT, tickers, str(start.date()), str(end.date())]
    key = hashlib.sha256(json.dumps(request).encode()).hexdigest()[:16]
    snapshot = CACHE_DIR / f"closes_{key}.csv"
    if snapshot.exists():
        with snapshot.open(encoding="utf-8") as source:
            marker, _, timestamp = source.readline().strip().partition(" ")
            if marker != "#":
                raise ValueError(f"Missing fetch timestamp in {snapshot.name}")
            datetime.fromisoformat(timestamp)
            prices = pd.read_csv(source, index_col="Date", parse_dates=["Date"])
            prices.index = pd.DatetimeIndex(prices.index).as_unit("ns")
        validate_prices(prices, tickers, start, end)
        return prices, timestamp

    with TemporaryDirectory(dir=CACHE_DIR) as request_cache:
        yf.set_tz_cache_location(request_cache)
        for attempt in range(2):
            try:
                raw = yf.download(
                    tickers, start=start.date().isoformat(),
                    end=(end + pd.Timedelta(days=1)).date().isoformat(),
                    interval="1d", auto_adjust=True, progress=False, timeout=10,
                    multi_level_index=True,
                )
                prices = raw["Close"].reindex(columns=tickers)
                prices.index = pd.DatetimeIndex(prices.index, name="Date").tz_localize(None).as_unit("ns")
                validate_prices(prices, tickers, start, end)
                break
            except Exception as error:
                if attempt == 1:
                    raise RuntimeError(f"Could not fetch a complete yfinance panel: {error}") from error

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE_DIR, delete=False) as output:
        pending = Path(output.name)
        try:
            output.write(f"# {timestamp}\n")
            prices.to_csv(output)
            output.close()
            pending.replace(snapshot)
        finally:
            pending.unlink(missing_ok=True)
    return prices, timestamp


def condition_pair(block, cross, sibling, companion, sibling_companion, gamma):
    """Damp the Schur complement and the companion vector together."""
    cross_solution = np.linalg.solve(sibling, cross.T)
    new_block = block - gamma * (cross @ cross_solution)
    new_companion = companion - gamma * (cross_solution.T @ sibling_companion)
    return 0.5 * (new_block + new_block.T), new_companion


def recursive_direction(covariance, companion, gamma):
    if len(companion) == 1:
        return companion / covariance.diagonal()
    middle = len(companion) // 2
    left, right = slice(None, middle), slice(middle, None)
    cross = covariance[left, right]
    left_pair = condition_pair(
        covariance[left, left], cross, covariance[right, right],
        companion[left], companion[right], gamma,
    )
    right_pair = condition_pair(
        covariance[right, right], cross.T, covariance[left, left],
        companion[right], companion[left], gamma,
    )
    return np.concatenate([
        child_direction(*left_pair, gamma),
        child_direction(*right_pair, gamma),
    ])


def child_direction(block, companion, gamma):
    """Scale the recursively allocated child by inverse variance under b' w = 1."""
    if not np.any(companion):
        return np.zeros_like(companion)
    diagonal = block.diagonal()
    if len(companion) == 1:
        return companion / diagonal

    # At gamma=0 the reference is inverse variance. At gamma=1 the residual
    # variance and companion give the child's unconstrained minimum variance.
    precision = np.linalg.solve(block, np.eye(len(companion)))
    residual_variance = 1 / precision.diagonal()
    residual_companion = residual_variance * (precision @ companion)
    reference = ((1 - gamma) * companion + gamma * residual_companion) / (
        (1 - gamma) * diagonal + gamma * residual_variance
    )
    direction = recursive_direction(block, companion, gamma)
    alignment = float(companion @ direction)
    if abs(alignment) <= 1e-12 * np.linalg.norm(companion) * np.linalg.norm(direction):
        direction = precision @ companion
        alignment = float(companion @ direction)
    inverse_variance = float(companion @ reference) ** 2 / float(reference @ block @ reference)
    return direction * (inverse_variance / alignment)


def schur_weights(history, params):
    sample = np.cov(history, rowvar=False)
    variance = sample.diagonal()
    if (variance <= 0).any() or not np.isfinite(sample).all():
        raise ValueError("The rolling window must contain finite returns and positive asset variances")
    volatility = np.sqrt(variance)
    correlation = sample / np.outer(volatility, volatility)
    distance = np.sqrt(np.clip((1 - correlation) / 2, 0, 1))
    np.fill_diagonal(distance, 0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="single"))

    shrunk = (1 - params.covariance_shrinkage) * sample + params.covariance_shrinkage * np.diag(variance)
    covariance = shrunk[np.ix_(order, order)]
    direction = recursive_direction(covariance, np.ones(len(order)), params.gamma)
    positive = np.maximum(direction, 0)
    total = positive.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Schur recursion did not yield a finite positive allocation")
    weights = np.empty(len(order))
    weights[order] = positive / total
    return weights


def simulate(asset_returns, start, lag, params, *, equal_weight=False):
    sessions = asset_returns.index
    values = asset_returns.to_numpy()
    start_pos = sessions.searchsorted(start)
    months = sessions.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])
    targets = {}
    for signal_pos in month_ends:
        fill_pos = signal_pos + lag
        if not start_pos <= fill_pos < len(sessions):
            continue
        if equal_weight:
            targets[fill_pos] = np.full(values.shape[1], 1 / values.shape[1])
        else:
            first = signal_pos - params.lookback + 1
            if first < 1:
                raise ValueError(f"Insufficient warm-up history on {sessions[signal_pos].date()}")
            targets[fill_pos] = schur_weights(values[first:signal_pos + 1], params)

    if not targets:
        raise ValueError("The evaluation interval contains no scheduled rebalances")
    weights = np.zeros(values.shape[1])
    daily_results = np.zeros((len(sessions) - start_pos, 3))
    rate = params.friction
    for pos in range(start_pos, len(sessions)):
        asset_move = values[pos]
        gross = 1 + float(weights @ asset_move)
        weights = weights * (1 + asset_move) / gross
        traded_fraction = 0.0
        trades = 0
        if pos in targets:
            target = targets[pos]

            def fee_difference(fee):
                return fee - rate * np.abs((1 - fee) * target - weights).sum()

            fee = brentq(fee_difference, 0.0, 1.0) if rate else 0.0
            changes = np.abs((1 - fee) * target - weights)
            traded_fraction = float(changes.sum())
            trades = int(np.count_nonzero(changes > 1e-10))
            gross *= 1 - fee
            weights = target
        daily_results[pos - start_pos] = (gross - 1, traded_fraction, trades)

    return pd.DataFrame(daily_results, index=sessions[start_pos:], columns=["return", "traded", "n_trades"])


def metrics(daily_results, cash_proxy_returns):
    returns = daily_results["return"].to_numpy()
    if len(returns) < 2:
        raise ValueError("At least two sessions are required to report period metrics")
    excess = returns - cash_proxy_returns.loc[daily_results.index].to_numpy()
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    growth = np.r_[1.0, np.cumprod(1 + returns)]
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(volatility * annualizer),
        "sharpe": float(returns.mean() / volatility * annualizer) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_volatility * annualizer) if excess_volatility > 0 else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["traded"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, cash_proxy_returns, is_end):
    return {
        "is": metrics(daily_results.loc[:is_end], cash_proxy_returns),
        "oos": metrics(daily_results.loc[daily_results.index > is_end], cash_proxy_returns),
        "full": metrics(daily_results, cash_proxy_returns),
    }


def main():
    with (FOLDER / "config.yaml").open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    params = Parameters(**config["parameters"])
    cases = {name: replace(params, **overrides) for name, overrides in config["robustness"].items()}
    for spec in (params, *cases.values()):
        spec.validate()
    if len({params, *cases.values()}) != len(cases) + 1:
        raise ValueError("Each robustness case must specify a distinct parameter combination")

    start, first, split, end = (
        pd.Timestamp(config[key]) for key in ("data_start", "backtest_start", "is_end", "backtest_end")
    )
    lag = config["lag_sessions"]
    universe = config["universe"]
    cash_proxy = config["cash_proxy"]
    if not start < first <= split < end or type(lag) is not int or lag < 1:
        raise ValueError("Set ascending dates and a positive integer execution lag")
    if len(universe) < 2 or len(set(universe)) != len(universe) or cash_proxy in universe:
        raise ValueError("Choose at least two distinct assets and a separate cash proxy")

    prices, timestamp = load_prices([*universe, cash_proxy], start, end)
    returns = prices.pct_change(fill_method=None)
    asset_returns = returns[universe]
    cash_proxy_returns = returns[cash_proxy]
    print(f"DATA_FETCHED_AT: {timestamp}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")

    is_asset_returns = asset_returns.loc[:split]
    is_cash_proxy_returns = cash_proxy_returns.loc[:split]
    robustness = {}
    for name, spec in cases.items():
        daily_results = simulate(is_asset_returns, first, lag, spec)
        robustness[name] = {"parameters": asdict(spec), **metrics(daily_results, is_cash_proxy_returns)}

    official = simulate(asset_returns, first, lag, params)
    benchmark = simulate(asset_returns, first, lag, params, equal_weight=True)
    official_periods = period_metrics(official, cash_proxy_returns, split)
    benchmark_periods = period_metrics(benchmark, cash_proxy_returns, split)

    print("ROBUSTNESS IS ONLY")
    print("default", json.dumps({"parameters": asdict(params), **official_periods["is"]}))
    for name, result in robustness.items():
        print(name, json.dumps(result, allow_nan=False))
    print(f"PARAM_COMBOS_TRIED: {1 + len(robustness)}")
    for period in ("is", "oos", "full"):
        print(f"STRATEGY {period.upper()}: {json.dumps(official_periods[period])}")
        print(f"BENCHMARK equal_weight_monthly {period.upper()}: {json.dumps(benchmark_periods[period])}")
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": str(official.index[0].date()), "end": str(official.index[-1].date())},
        **official_periods,
        "benchmark": {"name": "equal_weight_monthly", **benchmark_periods},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    main()
