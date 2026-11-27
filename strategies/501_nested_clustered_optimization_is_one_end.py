# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 501_nested_clustered_optimization_is_one_end\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nrisk_free_ticker: BIL\ndata_start: 2007-06-01\nstart: 2009-01-01\nis_end: 2017-12-31\nend: 2025-12-31\nexecution_lag_sessions: 1\n\nallocation:\n  lookback_sessions: 252\n  n_clusters: 3\n  gamma: 0.5\n  covariance_ridge: 0.000001\n\ncost_bps: 2.0\nslippage_bps: 3.0\n\n# Individual changes to the default, evaluated only in sample.\nrobustness:\n  - {gamma: 0.0}\n  - {gamma: 0.25}\n  - {gamma: 0.75}\n  - {lookback_sessions: 189}\n  - {lookback_sessions: 315}\n  - {n_clusters: 4}\n  - {cost_bps: 4.0, slippage_bps: 6.0}\n'
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
"""Monthly long-only Schur bridge for a fixed universe of sector ETFs."""

import hashlib
import json
import os
import signal
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.linalg import solve_triangular
from scipy.optimize import brentq, nnls
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class Parameters:
    lookback_sessions: int
    n_clusters: int
    gamma: float
    covariance_ridge: float
    cost_bps: float
    slippage_bps: float

    def __post_init__(self):
        if self.lookback_sessions < 2 or not 0 <= self.gamma <= 1:
            raise ValueError("Require a lookback of at least two sessions and gamma in [0, 1].")
        if not np.isfinite(self.covariance_ridge) or self.covariance_ridge <= 0:
            raise ValueError("The covariance ridge must be positive and finite.")
        if min(self.cost_bps, self.slippage_bps) < 0 or not 0 <= self.fee_rate < 1:
            raise ValueError("Fees and slippage must be nonnegative and total under 10,000 bps.")

    @property
    def fee_rate(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def check_prices(prices, tickers, data_start, end):
    if list(prices.columns) != tickers or prices.empty:
        raise ValueError("Adjusted closes are missing for the configured universe.")
    dates = prices.index
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Adjusted closes must have unique, increasing session dates.")
    values = prices.to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        bad = prices.columns[(~np.isfinite(values) | (values <= 0)).any(axis=0)]
        raise ValueError(f"Missing or invalid adjusted closes for {', '.join(bad)}.")
    # Configured boundaries can fall on non-session dates.
    coverage = [dates[0] - data_start, end - dates[-1]]
    if any(not pd.Timedelta(0) <= gap <= pd.Timedelta(days=6) for gap in coverage):
        raise ValueError("Adjusted closes do not cover the configured date range.")


def load_prices(tickers, data_start, end):
    CACHE_DIR.mkdir(exist_ok=True)
    request = f"yfinance:1d:auto_adjust:{data_start.date()}:{end.date()}:{','.join(tickers)}"
    digest = hashlib.sha256(request.encode()).hexdigest()[:16]
    path = CACHE_DIR / f"adjusted_close_{digest}.csv"
    if path.exists():
        try:
            with path.open(encoding="utf-8") as source:
                fetched_at = source.readline().strip().removeprefix("# fetched_at=")
                datetime.fromisoformat(fetched_at)
                prices = pd.read_csv(
                    source, index_col="date", parse_dates=["date"], float_precision="round_trip"
                )
            check_prices(prices, tickers, data_start, end)
            return prices, fetched_at
        except (OSError, ValueError, TypeError):
            pass

    # Keep the library's temporary state inside this project.
    with tempfile.TemporaryDirectory(dir=CACHE_DIR) as download_dir:
        yf.set_tz_cache_location(download_dir)
        try:
            raw = yf.download(
                tickers,
                start=data_start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d",
                auto_adjust=True,
                group_by="column",
                ignore_tz=True,
                keepna=True,
                threads=False,
                progress=False,
                timeout=10,
            )
        finally:
            yf.set_tz_cache_location(str(CACHE_DIR))
    if raw is None or raw.empty or "Close" not in raw.columns.get_level_values(0):
        raise RuntimeError("yfinance returned no adjusted closes.")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prices = raw["Close"].reindex(columns=tickers).sort_index()
    check_prices(prices, tickers, data_start, end)
    with tempfile.NamedTemporaryFile(dir=CACHE_DIR, mode="w", encoding="utf-8", delete=False) as sink:
        pending = Path(sink.name)
        try:
            sink.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(sink, index_label="date")
            sink.flush()
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)
    return prices, fetched_at


def long_only_weights(covariance, budget_direction):
    # NNLS solves min_{x >= 0} (x' Q x / 2 - b' x) using Q = L L'.
    # Its normalized solution has the shape of the generalized minimum-variance direction.
    cholesky = np.linalg.cholesky(covariance)
    target = solve_triangular(cholesky, budget_direction, lower=True)
    direction, _ = nnls(cholesky.T, target)
    if direction.sum() == 0:
        # A nonpositive budget vector cannot give b' x = 1 with x >= 0.
        target = solve_triangular(cholesky, np.ones(len(direction)), lower=True)
        direction, _ = nnls(cholesky.T, target)
    return direction / direction.sum()


def bridge_weights(window, parameters):
    sample_covariance = np.cov(window, rowvar=False)
    standard_deviations = np.sqrt(sample_covariance.diagonal())
    if (standard_deviations <= 0).any():
        raise ValueError("A zero-variance series makes correlation clustering undefined.")
    correlation = sample_covariance / np.outer(standard_deviations, standard_deviations)
    distances = np.sqrt((1 - np.clip(correlation, -1, 1)) / 2)
    np.fill_diagonal(distances, 0)
    tree = linkage(squareform(distances, checks=False), method="average")
    labels = cut_tree(tree, n_clusters=[parameters.n_clusters]).ravel()
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    # Select the member nearest the others on average as each cluster's knot.
    knots = [group[np.argmin(distances[np.ix_(group, group)].sum(axis=1))] for group in groups]

    ridge = parameters.covariance_ridge * sample_covariance.diagonal().mean()
    covariance = sample_covariance + ridge * np.eye(window.shape[1])
    cluster_weights = np.zeros((window.shape[1], len(groups)))
    for column, group in enumerate(groups):
        conditioned = covariance[np.ix_(group, group)]
        budget_direction = np.ones(len(group))
        other_knots = knots[:column] + knots[column + 1 :]
        if parameters.gamma and other_knots:
            cross = covariance[np.ix_(group, other_knots)]
            knot_covariance = covariance[np.ix_(other_knots, other_knots)]
            regression = np.linalg.solve(knot_covariance, cross.T).T
            # Equation (8) damps both the conditional covariance and budget vector.
            conditioned -= parameters.gamma * (regression @ cross.T)
            budget_direction -= parameters.gamma * regression.sum(axis=1)
        cluster_weights[group, column] = long_only_weights(conditioned, budget_direction)

    outer_covariance = cluster_weights.T @ covariance @ cluster_weights
    outer_weights = long_only_weights(outer_covariance, np.ones(len(groups)))
    return cluster_weights @ outer_weights


def rebalance_schedule(dates, start, lag):
    months = dates.to_period("M")
    # On a new month's first session the previous month's last close is known.
    signals = np.flatnonzero(months[:-1] != months[1:])
    return {
        dates[signal_position + lag]: signal_position
        for signal_position in signals
        if signal_position + lag < len(dates) and dates[signal_position + lag] >= start
    }


def strategy_targets(returns, schedule, parameters):
    targets = {}
    for fill_date, signal_position in schedule.items():
        stop = signal_position + 1
        start = stop - parameters.lookback_sessions
        if start < 0:
            raise ValueError(f"Insufficient warm-up history for the rebalance on {fill_date.date()}.")
        targets[fill_date] = bridge_weights(returns.iloc[start:stop].to_numpy(), parameters)
    return targets


def simulate(returns, targets, fee_rate):
    positions = np.zeros(returns.shape[1])
    returns_net = np.zeros(len(returns))
    turnover = np.zeros(len(returns))
    n_trades = np.zeros(len(returns), dtype=int)
    for row, (date, asset_returns) in enumerate(zip(returns.index, returns.to_numpy())):
        # The old positions earn the move through this close.
        gross = 1 + positions @ asset_returns
        positions = positions * (1 + asset_returns) / gross
        remaining = 1.0
        if date in targets:
            target = targets[date]
            # Charge on actual buys and sells, including the first investment from cash.
            remaining = brentq(
                lambda fraction: fraction + fee_rate * np.abs(fraction * target - positions).sum() - 1,
                0.0,
                1.0,
            )
            changes = np.abs(remaining * target - positions)
            turnover[row] = changes.sum()
            n_trades[row] = np.count_nonzero(changes > 1e-10)
            positions = target
        returns_net[row] = gross * remaining - 1
    return pd.DataFrame(
        {"return": returns_net, "turnover": turnover, "n_trades": n_trades}, index=returns.index
    )


def metrics(daily_results, bill_returns):
    daily = daily_results["return"]
    if len(daily) < 2:
        raise ValueError("Each reported period needs at least two return observations.")
    annualization = SESSIONS_PER_YEAR / len(daily)
    growth = np.r_[1.0, (1 + daily).cumprod().to_numpy()]
    sigma = daily.std()
    excess = daily - bill_returns.loc[daily.index]
    excess_sigma = excess.std()
    return {
        "cagr": float(growth[-1] ** annualization - 1),
        "ann_vol": float(sigma * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": float(daily.mean() / sigma * np.sqrt(SESSIONS_PER_YEAR)) if sigma > 0 else None,
        "sharpe_excess": (
            float(excess.mean() / excess_sigma * np.sqrt(SESSIONS_PER_YEAR))
            if excess_sigma > 0
            else None
        ),
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].sum() * annualization),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, bill_returns, is_end):
    within_is = daily_results.index <= is_end
    periods = {"is": daily_results.loc[within_is], "oos": daily_results.loc[~within_is], "full": daily_results}
    return {label: metrics(period, bill_returns) for label, period in periods.items()}


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    tickers = config["universe"]
    bill = config["risk_free_ticker"]
    data_start, start, is_end, end = (
        pd.Timestamp(config[key]) for key in ("data_start", "start", "is_end", "end")
    )
    lag = config["execution_lag_sessions"]
    if not data_start < start <= is_end < end or not isinstance(lag, int) or lag < 1:
        raise ValueError("Check the configured period boundaries and positive execution lag.")
    if len(set([*tickers, bill])) != len(tickers) + 1:
        raise ValueError("The ETF universe and its cash proxy must have distinct tickers.")

    base = Parameters(**config["allocation"], cost_bps=config["cost_bps"], slippage_bps=config["slippage_bps"])
    alternatives = [(changes, replace(base, **changes)) for changes in config["robustness"]]
    cases = {base, *(parameters for _, parameters in alternatives)}
    if any(not 1 <= case.n_clusters <= len(tickers) for case in cases) or len(tickers) < 2:
        raise ValueError("Check the universe size and configured cluster counts.")
    if len(cases) != 1 + len(alternatives):
        raise ValueError("Robustness specifications must be distinct from the default and each other.")

    prices, fetched_at = load_prices([*tickers, bill], data_start, end)
    asset_returns = prices.pct_change(fill_method=None).iloc[1:]
    sector_returns = asset_returns[tickers]
    bill_returns = asset_returns[bill]
    evaluation_returns = sector_returns.loc[start:end]
    schedule = rebalance_schedule(sector_returns.index, start, lag)
    if not schedule or next(iter(schedule)) > is_end:
        raise ValueError("The sample contains no executable in-sample rebalance.")

    targets = strategy_targets(sector_returns, schedule, base)
    strategy_results = simulate(evaluation_returns, targets, base.fee_rate)
    equal_weights = np.full(len(tickers), 1 / len(tickers))
    benchmark_results = simulate(
        evaluation_returns, dict.fromkeys(schedule, equal_weights), base.fee_rate
    )
    official = period_metrics(strategy_results, bill_returns, is_end)
    benchmark = period_metrics(benchmark_results, bill_returns, is_end)
    data_range = {
        "start": evaluation_returns.index[0].date().isoformat(),
        "end": evaluation_returns.index[-1].date().isoformat(),
    }
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return")
    for label, values in official.items():
        print(f"{label.upper()}: {json.dumps(values, allow_nan=False)}")
    print("BENCHMARK: monthly equal-weight sector ETFs with the same lag, costs and slippage")
    for label, values in benchmark.items():
        print(f"BENCHMARK_{label.upper()}: {json.dumps(values, allow_nan=False)}")

    print("ROBUSTNESS (IS only)")
    in_sample = sector_returns.loc[:is_end]
    is_evaluation = evaluation_returns.loc[:is_end]
    is_schedule = rebalance_schedule(in_sample.index, start, lag)
    print(f"DEFAULT: {json.dumps(official['is'], allow_nan=False)}")
    for changes, parameters in alternatives:
        alternative_targets = strategy_targets(in_sample, is_schedule, parameters)
        alternative = simulate(is_evaluation, alternative_targets, parameters.fee_rate)
        values = metrics(alternative, bill_returns)
        print(f"{json.dumps(changes)}: {json.dumps(values, allow_nan=False)}")

    print(f"PARAM_COMBOS_TRIED: {len(cases)}")
    result = {"strategy": config["strategy"], "data_range": data_range, **official, "benchmark": benchmark}
    print(f"RESULTS_JSON: {json.dumps(result, allow_nan=False)}")


if __name__ == "__main__":
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)
