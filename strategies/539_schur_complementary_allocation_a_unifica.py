"""539_schur_complementary_allocation_a_unifica
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/539_schur_complementary_allocation_a_unifica.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c0a8f76240a0431b019ab49c34040a95a765860a on 2026-11-27 10:04 Europe/Oslo, exit 0, output sha256 fad5bba1fe0650074f770ca83e091d3d9ac4982f8a9ef373e62b7cefe284851b
Data source: Yahoo Finance daily USD adjusted closing prices via yfinance for SPY, TLT, GLD, VNQ and DBC. BIL is the risk-free proxy used for ex-post excess-return metrics. The CSV cache retains the original download timestamp, logged as 2026-11-27T08:17:33+00:00 for this run.
Data fetched: 2026-11-27T09:04:37+00:00
Sample period: The requested price history runs from 2009-01-01 through 2025-12-31, including warm-up. The backtest covers 2011-01-03 through 2025-12-31, inclusive, using the first and last evaluated trading sessions.   data_range (from output): {'start': '2011-01-03', 'end': '2025-12-31'}
IS/OOS split: Chronological split. IS is 2011-01-03 through 2018-12-31 and OOS is 2019-01-02 through 2025-12-31, inclusive. The default strategy runs continuously across the split. The five sensitivity variants use only price rows through the IS cutoff and are evaluated only in IS.
Costs and slippage: The default strategy and equal-weight benchmark charge 2 bps commission plus 3 bps modelled slippage, for 5 bps per dollar bought or sold, including initial entry. The simulator solves for the fee on each rebalance using the sum of absolute trades from drifted holdings to the after-fee target and deducts it from equity at that close. The ending portfolio is marked to market. The IS double-friction variant charges 4 bps commission plus 6 bps slippage.
Signal lag: The signal uses daily returns through the last trading session's close in each month. The configured fill is at the next trading session's close. Existing weights receive the fill day's close-to-close return before the new weights are applied.
Price adjustment: yfinance auto_adjust total-return close (splits and distributions)
Survivorship bias: The fixed ETF universe was selected retrospectively from funds with long available histories. The README discloses survivorship and hindsight selection bias.
Parameter combinations tried: 6
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.054328853646103736, ann_vol=0.06851935234651858, sharpe=0.8064698009569631, sharpe_excess=0.7670714047858982, max_drawdown=-0.11275027785745961, total_return=0.5256117879089592, turnover=1.0249390647074772, n_trades=453
    oos: cagr=0.08029830943484217, ann_vol=0.10245097923998639, sharpe=0.8053075896210831, sharpe_excess=0.5583153322880957, max_drawdown=-0.20836135266000444, total_return=0.7150367698031854, turnover=1.0829350740837453, n_trades=407
    full: cagr=0.06636748316598862, ann_vol=0.08602577117757094, sharpe=0.7901301040462668, sharpe_excess=0.636143471931454, max_drawdown=-0.2083613526600051, total_return=1.6164803127090392, turnover=1.0519997689763614, n_trades=860
    benchmark  full: cagr=0.06899007941273982, ann_vol=0.09755671836785411, sharpe=0.7328217598069419, sharpe_excess=0.5967059974334596, max_drawdown=-0.20960059384440122, total_return=1.7144717312237256, turnover=0.4152100666008557, n_trades=900
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 539_schur_complementary_allocation_a_unifica\nuniverse: [SPY, TLT, GLD, VNQ, DBC]\nrisk_free_ticker: BIL\n\ndates:\n  download_start: 2009-01-01\n  start: 2011-01-01\n  is_end: 2018-12-31\n  end: 2025-12-31\n\nrebalance: monthly\nexecution_lag_sessions: 1\nshrinkage: 0.03\nleaf_size: 2\nbudget_vector_floor: 0.10\n\nallocation:\n  lookback_sessions: 252\n  gamma: 0.50\n  cost_bps: 2.0\n  slippage_bps: 3.0\n\nrobustness:\n  lower_gamma: {gamma: 0.35}\n  higher_gamma: {gamma: 0.65}\n  shorter_window: {lookback_sessions: 210}\n  longer_window: {lookback_sessions: 294}\n  double_friction: {cost_bps: 4.0, slippage_bps: 6.0}\n'
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
"""Monthly long-only ETF allocation using recursive Schur adjustments."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import brentq, minimize
from scipy.spatial.distance import squareform
import yaml
import yfinance as yf


HERE = Path(__file__).resolve().parent
CACHE = HERE / "data_cache"
ANNUAL_SESSIONS = 252
PRICE_ADJUSTMENT = "yfinance auto_adjust total-return close (splits and distributions)"


@dataclass(frozen=True)
class Allocation:
    lookback_sessions: int
    gamma: float
    cost_bps: float
    slippage_bps: float

    def __post_init__(self):
        if type(self.lookback_sessions) is not int or self.lookback_sessions < 2:
            raise ValueError("lookback_sessions must be an integer of at least two")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between zero and one")
        if not (self.cost_bps >= 0 and self.slippage_bps >= 0 and self.friction < 0.5):
            raise ValueError("Invalid transaction cost or slippage")

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def check_prices(prices, tickers, start, end):
    if prices.empty or list(prices.columns) != tickers:
        raise ValueError("Missing price history or unexpected ticker columns")
    if prices.index.hasnans or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Expected unique, ordered session dates")
    if prices.index[0] < start or prices.index[-1] > end:
        raise ValueError("Prices extend beyond the requested date range")
    if prices.index[0] - start > pd.Timedelta(days=7) or end - prices.index[-1] > pd.Timedelta(days=7):
        raise ValueError("Price history does not cover the requested date range")
    if (prices.index.to_series().diff() > pd.Timedelta(days=7)).any():
        raise ValueError("Price history has a gap of more than seven calendar days")
    values = prices.to_numpy(dtype=float)
    bad = ~np.isfinite(values) | (values <= 0)
    if bad.any():
        row, col = np.argwhere(bad)[0]
        raise ValueError(f"Missing or invalid {tickers[col]} close on {prices.index[row].date()}")


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    key = sha256(f"yf-adjusted-close-v1|{tickers}|{start.date()}|{end.date()}".encode()).hexdigest()
    path = CACHE / f"{key[:20]}.csv"
    if path.exists():
        try:
            with path.open(encoding="utf-8") as source:
                prefix, fetched_at = source.readline().strip().split("=", 1)
                if prefix != "# fetched_at" or datetime.fromisoformat(fetched_at).tzinfo is None:
                    raise ValueError("Invalid cache timestamp")
                prices = pd.read_csv(source, index_col="Date", parse_dates=["Date"])
            check_prices(prices, tickers, start, end)
            return prices, fetched_at
        except ValueError as exc:
            print(f"Refreshing invalid price cache ({exc})", flush=True)

    with TemporaryDirectory(prefix="yfinance-", dir=CACHE) as download_cache:
        yf.set_tz_cache_location(download_cache)
        bars = yf.download(
            tickers, start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
            interval="1d", auto_adjust=True, progress=False, threads=False, timeout=15,
            keepna=True, ignore_tz=True, multi_level_index=True,
        )
    if bars is None or bars.empty or "Close" not in bars:
        raise RuntimeError("yfinance returned no adjusted closes")
    prices = bars["Close"].reindex(columns=tickers).sort_index().rename_axis("Date")
    check_prices(prices, tickers, start, end)
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")

    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as target:
        temporary = Path(target.name)
        try:
            target.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(target)
            target.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return prices, fetched_at


def minimum_variance(cov):
    """Return simplex minimum-variance weights and their variance."""
    if len(cov) == 1:
        return np.ones(1), float(cov[0, 0])

    scaled = cov / np.max(np.diag(cov))
    initial = np.full(len(cov), 1 / len(cov))
    fit = minimize(
        lambda w: 0.5 * (w @ scaled @ w), initial,
        jac=lambda w: scaled @ w, method="SLSQP", bounds=[(0, 1)] * len(cov),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
        options={"ftol": 1e-12, "maxiter": 100},
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Minimum-variance fit failed: {fit.message}")
    weights = np.maximum(fit.x, 0)
    weights /= weights.sum()
    return weights, float(weights @ cov @ weights)


def schur_children(cov, cut, gamma, budget_floor):
    left = cov[:cut, :cut]
    right = cov[cut:, cut:]
    if gamma == 0:
        return left, right

    cross = cov[:cut, cut:]
    left_projector = np.linalg.solve(right, cross.T).T
    right_projector = np.linalg.solve(left, cross).T
    left_loading = left_projector.sum(axis=1)
    right_loading = right_projector.sum(axis=1)
    max_loading = max(left_loading.max(), right_loading.max())
    # Use the same local gamma for both siblings and keep both budget vectors positive.
    if gamma * max_loading > 1 - budget_floor:
        gamma = (1 - budget_floor) / max_loading

    left_budget = 1 - gamma * left_loading
    right_budget = 1 - gamma * right_loading
    left_schur = left - gamma * (left_projector @ cross.T)
    right_schur = right - gamma * (right_projector @ cross)
    # Table 1's precision-domain transform equals the covariance-domain transform.
    left_adjusted = (left_schur + left_schur.T) / (2 * np.outer(left_budget, left_budget))
    right_adjusted = (right_schur + right_schur.T) / (2 * np.outer(right_budget, right_budget))
    return left_adjusted, right_adjusted


def bisect_weights(cov, gamma, leaf_size, budget_floor, minimum=None):
    if len(cov) <= leaf_size:
        weights, _ = minimum if minimum is not None else minimum_variance(cov)
        return weights

    left, right = schur_children(cov, len(cov) // 2, gamma, budget_floor)
    left_minimum = minimum_variance(left)
    right_minimum = minimum_variance(right)
    left_variance = left_minimum[1]
    right_variance = right_minimum[1]
    left_share = right_variance / (left_variance + right_variance)
    left_weights = bisect_weights(left, gamma, leaf_size, budget_floor, left_minimum)
    right_weights = bisect_weights(right, gamma, leaf_size, budget_floor, right_minimum)
    return np.concatenate((left_share * left_weights, (1 - left_share) * right_weights))


def schur_weights(window, gamma, shrinkage, leaf_size, budget_floor):
    sample_cov = np.cov(window, rowvar=False, ddof=1)
    variances = np.diag(sample_cov)
    if (variances <= 0).any():
        raise ValueError("Expected positive sample variance for every ETF")
    cov = (1 - shrinkage) * sample_cov + shrinkage * np.diag(variances)
    std = np.sqrt(variances)
    corr = np.clip(cov / np.outer(std, std), -1, 1)
    distance = np.sqrt((1 - corr) / 2)
    np.fill_diagonal(distance, 0)
    order = leaves_list(linkage(squareform(distance), method="single", optimal_ordering=True))
    ordered = bisect_weights(cov[np.ix_(order, order)], gamma, leaf_size, budget_floor)
    weights = np.empty_like(ordered)
    weights[order] = ordered
    return weights


def rebalance_schedule(index, first, last, lag):
    months = index.to_period("M")
    signal_days = np.flatnonzero(months[1:] != months[:-1])
    return {
        day + lag: day
        for day in signal_days
        if first <= day + lag < last
    }


def schur_targets(returns, schedule, spec, config):
    targets = {}
    for fill_day, signal_day in schedule.items():
        start = signal_day + 1 - spec.lookback_sessions
        if start < 1:
            raise ValueError("The download period does not cover the full initial lookback")
        window = returns[start:signal_day + 1]
        targets[fill_day] = schur_weights(
            window, spec.gamma, config["shrinkage"], config["leaf_size"], config["budget_vector_floor"],
        )
    return targets


def simulate(returns, first, last, targets, friction):
    weights = np.zeros(returns.shape[1])
    records = []
    for day in range(first, last):
        marked = weights * (1 + returns[day])
        gross = 1 + float(weights @ returns[day])
        weights = marked / gross
        fee = turnover = 0.0
        trades = 0
        if day in targets:
            target = targets[day]
            # Targets are weights after fees. Solve for self-financing traded notional.
            if friction:
                fee = brentq(
                    lambda fee: fee - friction * np.abs((1 - fee) * target - weights).sum(),
                    0, 2 * friction, xtol=1e-15,
                )
            changes = np.abs((1 - fee) * target - weights)
            turnover = float(changes.sum())
            trades = int(np.count_nonzero(changes > 1e-8))
            weights = target
        records.append((gross * (1 - fee) - 1, turnover, trades))
    return pd.DataFrame(records, columns=["return", "turnover", "n_trades"])


def metrics(daily_results, risk_free):
    returns = daily_results["return"].to_numpy()
    if len(returns) < 2 or len(returns) != len(risk_free):
        raise ValueError("Each period needs at least two aligned daily observations")
    excess = returns - risk_free
    annual_scale = np.sqrt(ANNUAL_SESSIONS)
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    growth_curve = np.concatenate(([1.0], np.cumprod(1 + returns)))
    growth = growth_curve[-1]
    return {
        "cagr": float(growth ** (ANNUAL_SESSIONS / len(returns)) - 1),
        "ann_vol": float(volatility * annual_scale),
        "sharpe": float(annual_scale * returns.mean() / volatility) if volatility else None,
        "sharpe_excess": float(annual_scale * excess.mean() / excess_volatility) if excess_volatility else None,
        "max_drawdown": float((growth_curve / np.maximum.accumulate(growth_curve) - 1).min()),
        "total_return": float(growth - 1),
        "turnover": float(daily_results["turnover"].mean() * ANNUAL_SESSIONS),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, risk_free, split):
    return {
        "is": metrics(daily_results.iloc[:split], risk_free[:split]),
        "oos": metrics(daily_results.iloc[split:], risk_free[split:]),
        "full": metrics(daily_results, risk_free),
    }


def main():
    config = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    reference = config["risk_free_ticker"]
    if len(universe) < 2 or len(set(universe)) != len(universe) or reference in universe:
        raise ValueError("Use a distinct risk-free ticker and at least two different assets")
    if config["rebalance"] != "monthly":
        raise ValueError("Expected a monthly rebalance")
    lag = config["execution_lag_sessions"]
    if type(lag) is not int or lag < 1:
        raise ValueError("execution_lag_sessions must be a positive integer")
    if not (0 < config["shrinkage"] <= 1 and 0 < config["budget_vector_floor"] < 1):
        raise ValueError("Invalid shrinkage or budget vector floor")
    if type(config["leaf_size"]) is not int or not 1 <= config["leaf_size"] < len(universe):
        raise ValueError("leaf_size must allow at least one split")

    dates = config["dates"]
    download_start, start, is_end, end = (pd.Timestamp(dates[k]) for k in ("download_start", "start", "is_end", "end"))
    if not download_start < start < is_end < end:
        raise ValueError("Expected download_start < start < is_end < end")
    official = Allocation(**config["allocation"])
    variants = {name: replace(official, **overrides) for name, overrides in config["robustness"].items()}
    if len(set(variants.values()) | {official}) != len(variants) + 1:
        raise ValueError("Each strategy parameter combination must be distinct")

    prices, fetched_at = load_prices([*universe, reference], download_start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}", flush=True)
    index = prices.index
    first = int(index.searchsorted(start))
    split = int(index.searchsorted(is_end, side="right"))
    last = len(index)
    if first < 1 or split - first < 2 or last - split < 2:
        raise ValueError("Insufficient data for the requested IS and OOS split")
    is_schedule = rebalance_schedule(index[:split], first, split, lag)
    full_schedule = rebalance_schedule(index, first, last, lag)
    if not is_schedule or len(is_schedule) == len(full_schedule):
        raise ValueError("Both IS and OOS periods must have at least one scheduled fill")

    # Slice before computing the sensitivity returns and targets.
    is_returns = prices[universe].iloc[:split].pct_change(fill_method=None).to_numpy()
    risk_free_is = prices[reference].iloc[:split].pct_change(fill_method=None).to_numpy()[first:]
    print("ROBUSTNESS (IS only)", flush=True)
    for name, spec in variants.items():
        targets = schur_targets(is_returns, is_schedule, spec, config)
        daily_results = simulate(is_returns, first, split, targets, spec.friction)
        print(f"  {name}: {json.dumps(metrics(daily_results, risk_free_is), allow_nan=False)}", flush=True)

    returns = prices[universe].pct_change(fill_method=None).to_numpy()
    risk_free = prices[reference].pct_change(fill_method=None).to_numpy()[first:]
    targets = schur_targets(returns, full_schedule, official, config)
    daily_results = simulate(returns, first, last, targets, official.friction)
    equal_weight = np.full(len(universe), 1 / len(universe))
    benchmark_targets = dict.fromkeys(full_schedule, equal_weight)
    benchmark_daily = simulate(returns, first, last, benchmark_targets, official.friction)
    periods = period_metrics(daily_results, risk_free, split - first)
    benchmark = period_metrics(benchmark_daily, risk_free, split - first)
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(periods[period], allow_nan=False)}")
        print(f"BENCHMARK {period.upper()} (monthly equal weight): {json.dumps(benchmark[period], allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {1 + len(variants)}")
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": str(index[first].date()), "end": str(index[-1].date())},
        **periods,
        "benchmark": {"name": "monthly_equal_weight", **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(12 * 60)
    try:
        main()
    finally:
        signal.alarm(0)