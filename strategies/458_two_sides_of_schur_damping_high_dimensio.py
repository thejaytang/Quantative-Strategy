# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 458_two_sides_of_schur_damping_high_dimensio\nuniverse: [SPY, IWM, EFA, EEM, VNQ, IEF, TLT, LQD, HYG, GLD, DBC]\ncash_proxy: BIL\ndownload_start: 2009-01-01\nbacktest_start: 2010-01-01\ntuning_end: 2018-11-30\noos_start: 2019-01-01\nend: 2025-12-31\nlookback_sessions: 60\ngamma_candidates: [0.25, 0.50, 0.75]\nsignal_lag_sessions: 1\ncost_bps: 2.0\nslippage_bps: 3.0\n'
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
"""Research backtest for Schur-damped hierarchical ETF allocation."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import brentq
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_ADJUSTMENT = "yfinance auto_adjust=True, split- and distribution-adjusted total-return close"
SESSIONS_PER_YEAR = 252
TRADE_TOLERANCE = 1e-10


def load_config():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for key in ("download_start", "backtest_start", "tuning_end", "oos_start", "end"):
        config[key] = pd.Timestamp(config[key])
    if not (config["download_start"] < config["backtest_start"] < config["tuning_end"]
            < config["oos_start"] < config["end"]):
        raise ValueError("Expected ordered download, backtest, tuning, OOS and end dates.")
    universe = config["universe"]
    if len(universe) < 3 or len(set(universe)) != len(universe) or config["cash_proxy"] in universe:
        raise ValueError("Expected at least three unique portfolio ETFs and a separate cash proxy.")
    if config["cash_proxy"] != "BIL":
        raise ValueError("This backtest uses BIL as its cash proxy.")
    for key, minimum in (("signal_lag_sessions", 1), ("lookback_sessions", 3)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer of at least {minimum}.")
    gammas = config["gamma_candidates"]
    if not gammas or len(set(gammas)) != len(gammas) or not all(0 < gamma < 1 for gamma in gammas):
        raise ValueError("Damping candidates must be unique and strictly between zero and one.")
    costs = [config["cost_bps"], config["slippage_bps"]]
    if not all(np.isfinite(cost) and cost >= 0 for cost in costs) or sum(costs) >= 10_000:
        raise ValueError("Costs and slippage must be nonnegative and total less than 10,000 bps.")
    return config


def check_prices(prices, tickers, start, end):
    prices = prices.reindex(columns=tickers)
    if (prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing
            or not start <= prices.index[0] <= start + pd.Timedelta(days=7)
            or not end - pd.Timedelta(days=7) <= prices.index[-1] <= end):
        raise ValueError("Adjusted prices have incomplete date coverage or invalid session dates.")
    invalid = (~np.isfinite(prices) | prices.le(0)).sum()
    if invalid.any():
        raise ValueError(f"Missing or invalid closes by ticker: {invalid[invalid > 0].to_dict()}")
    return prices


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    cache_key = hashlib.sha256(repr((tickers, start, end, PRICE_ADJUSTMENT)).encode()).hexdigest()[:16]
    path = CACHE / f"adjusted_closes_{cache_key}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            fetched_at = datetime.fromisoformat(handle.readline().strip()).isoformat()
            prices = pd.read_csv(handle, index_col="date", parse_dates=["date"])
        return check_prices(prices, tickers, start, end), fetched_at

    download_args = dict(start=start.date().isoformat(),
                         end=(end + pd.Timedelta(days=1)).date().isoformat(),
                         auto_adjust=True, progress=False, timeout=10)
    with TemporaryDirectory(dir=CACHE, prefix=".yfinance_") as metadata:
        yf.set_tz_cache_location(metadata)
        prices = yf.download(tickers, threads=6, **download_args)["Close"].reindex(columns=tickers)
        for ticker in tickers:
            if prices[ticker].isna().all():
                retry = yf.download(ticker, threads=False, **download_args)["Close"]
                prices = prices.combine_first(retry)
    prices = check_prices(prices.sort_index(), tickers, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat()
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as handle:
        staging = Path(handle.name)
        try:
            handle.write(fetched_at + "\n")
            prices.to_csv(handle, index_label="date")
            handle.close()
            staging.replace(path)
        finally:
            staging.unlink(missing_ok=True)
    return prices, fetched_at


def ordered_covariance(window):
    covariance = np.cov(window, rowvar=False, ddof=1)
    variances = np.diag(covariance)
    if (variances <= 0).any():
        raise ValueError("Each ETF needs positive variance in the estimation window.")
    correlation = covariance / np.sqrt(np.outer(variances, variances))
    distance = np.sqrt(np.clip((1 - correlation) / 2, 0, 1))
    np.fill_diagonal(distance, 0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="single"))

    # Scale out the covariance units. The diagonal ridge stabilizes the Schur solves.
    covariance /= variances.mean()
    covariance += np.eye(len(covariance)) * 1e-8
    return order, covariance[np.ix_(order, order)]


def schur_direction(covariance, budget, gamma):
    """Return the recursive direction in asset units for a generalized budget."""
    if len(budget) == 1:
        return budget / covariance[0, 0]

    cut = len(budget) // 2
    left, right = covariance[:cut, :cut], covariance[cut:, cut:]
    cross = covariance[:cut, cut:]
    left_budget, right_budget = budget[:cut], budget[cut:]
    pieces = []
    for own, other, coupling, own_budget, other_budget in (
        (left, right, cross, left_budget, right_budget),
        (right, left, cross.T, right_budget, left_budget),
    ):
        hedge = np.linalg.solve(other, coupling.T).T
        residual = own - gamma * (hedge @ coupling.T)
        adjusted_budget = own_budget - gamma * (hedge @ other_budget)
        local_direction = np.linalg.solve(residual, adjusted_budget)
        local_precision = adjusted_budget @ local_direction
        if local_precision == 0:
            pieces.append(local_direction)
            continue
        # Passing the budget directly avoids dividing the covariance by small budget entries.
        recursive = schur_direction(residual, adjusted_budget, gamma)
        recursive_budget = adjusted_budget @ recursive
        if recursive_budget <= 1e-10 * local_precision:
            pieces.append(local_direction)
        else:
            pieces.append(recursive * (local_precision / recursive_budget))
    return np.concatenate(pieces)


def hrp_weights(covariance):
    if len(covariance) == 1:
        return np.ones(1)
    cut = len(covariance) // 2
    blocks = (covariance[:cut, :cut], covariance[cut:, cut:])
    risks = []
    for block in blocks:
        inverse_variance = 1 / np.diag(block)
        weights = inverse_variance / inverse_variance.sum()
        risks.append(weights @ block @ weights)
    mass_left = risks[1] / sum(risks)
    return np.r_[mass_left * hrp_weights(blocks[0]), (1 - mass_left) * hrp_weights(blocks[1])]


def build_targets(returns, config, gamma=None):
    sessions = returns.index
    months = sessions.to_period("M")
    lag = config["signal_lag_sessions"]
    lookback = config["lookback_sessions"]
    targets = {}
    for signal in np.flatnonzero(months[:-1] != months[1:]):
        execution = signal + lag
        if execution >= len(sessions) or sessions[execution] < config["backtest_start"]:
            continue
        if signal < lookback:
            raise ValueError(f"Insufficient return history at {sessions[signal].date()}.")
        window = returns.iloc[signal - lookback + 1:signal + 1].to_numpy()
        order, covariance = ordered_covariance(window)
        if gamma is None:
            ordered_weights = hrp_weights(covariance)
        else:
            direction = schur_direction(covariance, np.ones(len(order)), gamma)
            ordered_weights = np.maximum(direction, 0)
        mass = ordered_weights.sum()
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError(f"Invalid target weights at {sessions[signal].date()}.")
        weights = np.empty(len(order))
        weights[order] = ordered_weights / mass
        targets[sessions[execution]] = weights
    return targets


def simulate(returns, targets, config):
    rate = (config["cost_bps"] + config["slippage_bps"]) / 10_000
    weights = np.zeros(returns.shape[1])
    records = []
    for row in returns.itertuples(name=None):
        session, *daily_returns = row
        asset_values = weights * (1 + np.asarray(daily_returns))
        gross = 1 + weights @ daily_returns
        weights = asset_values / gross
        traded, n_trades, net_fraction = 0.0, 0, 1.0
        if session in targets:
            target = targets[session]
            # Target notionals are fractions of post-cost NAV. Costs fund themselves from NAV.
            net_fraction = brentq(
                lambda nav: nav + rate * np.abs(nav * target - weights).sum() - 1, 0, 1
            )
            changes = np.abs(net_fraction * target - weights)
            traded, n_trades = changes.sum(), np.count_nonzero(changes > TRADE_TOLERANCE)
            weights = target
        records.append((gross * net_fraction - 1, traded, n_trades))
    return pd.DataFrame(records, index=returns.index, columns=["return", "turnover", "n_trades"])


def metrics(history, cash_returns):
    daily = history["return"].to_numpy()
    if len(daily) < 2:
        raise ValueError("Each reported period needs at least two sessions.")
    excess = daily - cash_returns.loc[history.index].to_numpy()
    equity = np.r_[1.0, np.cumprod(1 + daily)]

    def sharpe(series):
        risk = series.std(ddof=1)
        return float(series.mean() * np.sqrt(SESSIONS_PER_YEAR) / risk) if risk > 0 else None

    return {
        "cagr": float(equity[-1] ** (SESSIONS_PER_YEAR / len(daily)) - 1),
        "ann_vol": float(daily.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": sharpe(daily),
        "sharpe_excess": sharpe(excess),
        "max_drawdown": float((equity / np.maximum.accumulate(equity) - 1).min()),
        "total_return": float(equity[-1] - 1),
        "turnover": float(history["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(history["n_trades"].sum()),
    }


def report(history, cash_returns, oos_start, label):
    periods = {
        "is": history.loc[history.index < oos_start],
        "oos": history.loc[history.index >= oos_start],
        "full": history,
    }
    reports = {name: metrics(period, cash_returns) for name, period in periods.items()}
    for name, values in reports.items():
        print(f"{label} {name.upper()}: {json.dumps(values, allow_nan=False)}")
    return reports


def main():
    config = load_config()
    universe, cash_proxy = config["universe"], config["cash_proxy"]
    prices, fetched_at = load_prices([*universe, cash_proxy], config["download_start"], config["end"])
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    daily = prices.pct_change(fill_method=None)
    asset_returns = daily[universe]
    evaluation_returns = asset_returns.loc[config["backtest_start"]:]
    tuning_data = asset_returns.loc[:config["tuning_end"]]
    tuning_returns = evaluation_returns.loc[:config["tuning_end"]]
    scores = {}
    for gamma in config["gamma_candidates"]:
        targets = build_targets(tuning_data, config, gamma)
        scores[gamma] = simulate(tuning_returns, targets, config)["return"].std(ddof=1)
    if not all(np.isfinite(score) for score in scores.values()):
        raise ValueError("Each damping candidate needs enough tuning history to estimate volatility.")
    selected = min(scores, key=lambda gamma: (scores[gamma], gamma))
    print(f"PARAM_COMBOS_TRIED: {len(scores)}")
    print(f"SELECTED_GAMMA: {selected:g}")

    selected_targets = build_targets(asset_returns, config, selected)
    if not any(config["tuning_end"] < date < config["oos_start"] for date in selected_targets):
        raise ValueError("The selected strategy needs a rebalance between tuning and OOS reporting.")
    strategy = simulate(evaluation_returns, selected_targets, config)
    benchmark = simulate(evaluation_returns, build_targets(asset_returns, config), config)
    cash_returns = daily[cash_proxy]
    strategy_report = report(strategy, cash_returns, config["oos_start"], "SCHUR")
    benchmark_report = report(benchmark, cash_returns, config["oos_start"], "BENCHMARK_HRP")
    result = {
        "strategy": config["strategy"],
        "data_range": f"{strategy.index[0].date()} to {strategy.index[-1].date()}",
        **strategy_report,
        "benchmark": benchmark_report["full"],
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
