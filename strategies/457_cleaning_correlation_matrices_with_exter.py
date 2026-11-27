# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "457_ cleaning_correlation_matrices_with_exter"\nuniverse:\n  equity: [SPY, IWM, EFA, EEM, EWJ, EWU]\n  bonds: [IEF, TLT, IEI, TIP]\n  currency: [FXE, FXY, FXB, FXA, FXC]\nrisk_free: BIL\ndates:\n  data_start: 2008-01-01\n  evaluation_start: 2012-01-03\n  is_end: 2018-12-31\n  end: 2025-12-31\nexecution:\n  signal_lag_sessions: 1\n  cost_bps: 2.0\n  slippage_bps: 3.0\nestimator:\n  volatility_alpha_daily: 0.01\n  correlation_alpha_daily: 0.0013333333333333333\n  warmup_weeks: 150\n'
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
"""Research backtest for the grouped Maximum Variance correlation estimator."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252


def adjusted_closes(tickers, start, end):
    query = json.dumps(["yfinance", "auto_adjust", tickers, str(start), str(end)])
    key = hashlib.sha256(query.encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"close_{key}.csv"
    cache_hit = cache_file.is_file()
    if cache_hit:
        with cache_file.open(encoding="utf-8") as cached:
            fetched_at = datetime.fromisoformat(cached.readline().strip()).isoformat()
            close = pd.read_csv(cached, index_col=0)
    else:
        CACHE_DIR.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=CACHE_DIR) as download_cache:
            yf.set_tz_cache_location(download_cache)
            prices = yf.download(
                tickers,
                start=str(start.date()),
                end=str((end + pd.Timedelta(days=1)).date()),
                interval="1d",
                auto_adjust=True,
                progress=False,
                timeout=15,
            )
            fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        close = prices["Close"]

    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    close = close.sort_index().loc[start:end].reindex(columns=tickers)
    if close.empty or close.index.has_duplicates:
        raise ValueError("The adjusted close panel is empty or has duplicate dates.")
    one_week = pd.Timedelta(days=7)
    if close.index[0] > start + one_week or close.index[-1] < end - one_week:
        raise ValueError("The adjusted close panel does not cover the requested dates.")
    bad = (~np.isfinite(close) | (close <= 0)).sum()
    if bad.any():
        raise ValueError(f"Missing or invalid adjusted closes by ticker: {bad[bad > 0].to_dict()}")

    if not cache_hit:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE_DIR, delete=False) as tmp:
            pending = Path(tmp.name)
            tmp.write(f"{fetched_at}\n")
            close.to_csv(tmp, index_label="date")
        pending.replace(cache_file)
    return close, fetched_at


def to_correlation(matrix):
    scale = np.sqrt(np.diag(matrix))
    return matrix / np.outer(scale, scale)


def maxvar_correlation(correlation, groups):
    # Disjoint groups make the leading group eigenvectors an orthonormal basis.
    n_assets = len(correlation)
    basis = np.zeros((n_assets, len(groups)))
    for column, group in enumerate(groups):
        _, eigenvectors = np.linalg.eigh(correlation[np.ix_(group, group)])
        basis[group, column] = eigenvectors[:, -1]

    # P (P.T C P) P.T equals Z M Z.T, with every constrained component retained.
    projected = basis.T @ correlation @ basis
    residual = (np.trace(correlation) - np.trace(projected)) / (n_assets - len(groups))
    completed = basis @ projected @ basis.T + residual * (np.eye(n_assets) - basis @ basis.T)
    return to_correlation(completed)


def portfolio_weights(correlation, volatility, expected_sharpe):
    direction = np.linalg.solve(correlation, expected_sharpe) / volatility
    return direction / np.abs(direction).sum()


def target_weights(returns, groups, expected_sharpe, start, execution, estimator):
    lag = execution["signal_lag_sessions"]
    daily_variance = returns.pow(2).ewm(alpha=estimator["volatility_alpha_daily"], adjust=False).mean()
    daily_vol = np.sqrt(daily_variance)
    prior_vol = daily_vol.shift()
    standardized = returns.div(prior_vol.where(prior_vol > 0))
    weekly_alpha = 1 - (1 - estimator["correlation_alpha_daily"]) ** 5
    weeks = returns.index.to_period("W-FRI")
    moment = np.zeros((returns.shape[1], returns.shape[1]))
    cleaned_targets, raw_targets = {}, {}
    n_weeks = 0

    for positions in returns.groupby(weeks, sort=True).indices.values():
        signal_pos = positions[0]
        fill_pos = signal_pos + lag
        # At the first close of this week, the previous week's moment is complete.
        if fill_pos < len(returns) and n_weeks >= estimator["warmup_weeks"]:
            fill_date = returns.index[fill_pos]
            if fill_date >= start:
                raw = to_correlation(moment)
                cleaned = maxvar_correlation(raw, groups)
                vol = daily_vol.iloc[signal_pos].to_numpy()
                cleaned_targets[fill_date] = portfolio_weights(cleaned, vol, expected_sharpe)
                raw_targets[fill_date] = portfolio_weights(raw, vol, expected_sharpe)

        block = standardized.iloc[positions].to_numpy()
        if np.isfinite(block).all():
            weekly = block.sum(axis=0)
            moment = (1 - weekly_alpha) * moment + weekly_alpha * np.outer(weekly, weekly)
            n_weeks += 1

    if not cleaned_targets:
        raise ValueError("No rebalances fall inside the evaluation dates after warmup and execution lag.")
    return cleaned_targets, raw_targets


def simulate(returns, close_targets, cost_rate):
    weights = np.zeros(returns.shape[1])
    daily = []
    for date, asset_returns in zip(returns.index, returns.to_numpy(), strict=True):
        marked_positions = weights * (1 + asset_returns)
        gross_nav = 1 + weights @ asset_returns
        if gross_nav <= 0:
            raise ValueError(f"Portfolio equity became nonpositive on {date.date()}.")
        weights = marked_positions / gross_nav
        turnover = 0.0
        n_trades = 0
        target = close_targets.get(date)
        if target is not None:
            # Solve for post-fee equity so the actual orders reach the target weights.
            def fee_balance(nav_fraction):
                traded = np.abs(nav_fraction * target - weights).sum()
                return nav_fraction + cost_rate * traded - 1

            nav_fraction = brentq(fee_balance, 0.0, 1.0)
            orders = np.abs(nav_fraction * target - weights)
            turnover = orders.sum()
            n_trades = np.count_nonzero(orders > 1e-10)
            gross_nav *= nav_fraction
            weights = target
        daily.append((gross_nav - 1, turnover, n_trades))

    return pd.DataFrame(daily, index=returns.index, columns=["return", "turnover", "n_trades"])


def metrics(daily, risk_free):
    returns = daily["return"]
    excess = returns - risk_free.loc[returns.index]
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    annual_scale = np.sqrt(SESSIONS_PER_YEAR)
    vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    total = wealth.iloc[-1]
    return {
        "cagr": float(total ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(vol * annual_scale),
        "sharpe": float(returns.mean() / vol * annual_scale) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * annual_scale) if excess_vol > 0 else None,
        "max_drawdown": float((wealth / peak - 1).min()),
        "total_return": float(total - 1),
        "turnover": float(daily["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily["n_trades"].sum()),
    }


def period_metrics(daily, risk_free, split):
    periods = {"is": daily.loc[:split], "oos": daily.loc[daily.index > split], "full": daily}
    if any(len(period) < 2 for period in periods.values()):
        raise ValueError("Both evaluation periods need at least two sessions.")
    return {name: metrics(period, risk_free) for name, period in periods.items()}


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    universe = config["universe"]
    group_names = ("equity", "bonds", "currency")
    tickers = [ticker for name in group_names for ticker in universe[name]]
    groups = [
        np.array([tickers.index(ticker) for ticker in universe[name]], dtype=int)
        for name in group_names
    ]
    cash_proxy = config["risk_free"]
    if (
        any(len(group) == 0 for group in groups)
        or len(tickers) <= len(groups)
        or len(set(tickers + [cash_proxy])) != len(tickers) + 1
        or cash_proxy != "BIL"
    ):
        raise ValueError("Use three disjoint nonempty asset groups, with BIL reserved for excess returns.")
    expected_sharpe = np.ones(len(tickers))
    expected_sharpe[groups[2]] = 0

    dates = config["dates"]
    data_start = pd.Timestamp(dates["data_start"])
    start = pd.Timestamp(dates["evaluation_start"])
    split = pd.Timestamp(dates["is_end"])
    end = pd.Timestamp(dates["end"])
    if not data_start < start <= split < end:
        raise ValueError("Check the order of the warmup, evaluation and split dates.")
    execution = config["execution"]
    lag = execution["signal_lag_sessions"]
    fees_bps = [execution["cost_bps"], execution["slippage_bps"]]
    cost_rate = sum(fees_bps) / 10_000
    if type(lag) is not int or lag < 1 or min(fees_bps) < 0 or not 0 <= cost_rate < 1:
        raise ValueError("Execution requires a positive session lag and nonnegative costs and slippage.")
    estimator = config["estimator"]
    if estimator["warmup_weeks"] < len(tickers) or not all(
        0 < estimator[key] < 1 for key in ("volatility_alpha_daily", "correlation_alpha_daily")
    ):
        raise ValueError("Use enough warmup weeks for the raw matrix and EMA coefficients between 0 and 1.")

    close, fetched_at = adjusted_closes(tickers + [cash_proxy], data_start, end)
    daily_returns = close.pct_change(fill_method=None).iloc[1:]
    risky_returns = daily_returns[tickers]
    cleaned, raw = target_weights(risky_returns, groups, expected_sharpe, start, execution, estimator)
    evaluation_returns = risky_returns.loc[min(cleaned):]
    strategy_daily = simulate(evaluation_returns, cleaned, cost_rate)
    benchmark_daily = simulate(evaluation_returns, raw, cost_rate)
    stats = period_metrics(strategy_daily, daily_returns[cash_proxy], split)
    benchmark_stats = period_metrics(benchmark_daily, daily_returns[cash_proxy], split)
    data_range = [date.date().isoformat() for date in evaluation_returns.index[[0, -1]]]

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, split- and distribution-adjusted close total returns")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"{config['strategy']} | {data_range[0]} to {data_range[1]}")
    for name, portfolio_stats in (("MaxVar", stats), ("Raw correlation benchmark", benchmark_stats)):
        for period, values in portfolio_stats.items():
            print(f"{name} {period.upper()}: {json.dumps(values, allow_nan=False)}")
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **stats,
        "benchmark": benchmark_stats["full"],
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
