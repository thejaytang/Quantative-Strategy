# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 481_optimal_portfolio_construction_a_reinfor\n\nuniverse:\n  cyclical: [XLB, XLY, XLF, XLRE]\n  sensitive: [XLC, XLE, XLI, XLK]\n  defensive: [XLP, XLV, XLU]\nrisk_free_ticker: BIL\n\ndates:\n  data_start: '2018-06-19'\n  sample_start: '2019-08-01'\n  is_end: '2021-12-31'\n  sample_end: '2025-08-13'\n\nlookback_days: [63, 126, 252]\nsignal_lag_sessions: 1\ncost_bps: 5\nslippage_bps: 2\n"
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
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq
from sklearn.covariance import LedoitWolf


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252


def load_prices(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    query = f"daily_auto_adjust|{start.date()}|{end.date()}|{','.join(tickers)}"
    cache_path = cache_dir / f"prices_{hashlib.sha256(query.encode()).hexdigest()[:16]}.csv"

    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as file:
            fetched_at = datetime.fromisoformat(file.readline().strip())
            prices = pd.read_csv(file, index_col=0, parse_dates=True, converters=dict.fromkeys(tickers, float))
        return check_prices(prices, tickers), fetched_at

    with TemporaryDirectory(prefix="yfinance_", dir=cache_dir) as session_dir:
        yf.set_tz_cache_location(session_dir)
        raw = yf.download(
            tickers, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True, ignore_tz=True, progress=False, timeout=15,
        )
    if raw is None or raw.empty or "Close" not in raw:
        raise RuntimeError("Yahoo Finance returned no adjusted closing prices")
    prices = check_prices(raw["Close"], tickers)
    fetched_at = datetime.now(timezone.utc)

    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_dir, delete=False) as file:
        temp_path = Path(file.name)
        try:
            print(fetched_at.isoformat(), file=file)
            prices.to_csv(file, index_label="date")
            file.flush()
            temp_path.replace(cache_path)
        finally:
            temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def check_prices(prices, tickers):
    prices = prices.reindex(columns=tickers).sort_index()
    values = prices.to_numpy()
    if len(prices) < 2 or prices.index.has_duplicates:
        raise ValueError("Expected unique daily prices on multiple sessions")
    invalid = ~(np.isfinite(values) & (values > 0))
    if invalid.any():
        row, col = np.argwhere(invalid)[0]
        raise ValueError(f"Missing or invalid price for {tickers[col]} on {prices.index[row].date()}")
    return prices


def equal_risk_weights(covariance):
    size = len(covariance)
    covariance = covariance / np.diag(covariance).mean()
    diagonal = np.diag(covariance)
    budget = 1.0 / size
    weights = np.sqrt(budget / diagonal)
    # Coordinate updates minimize x' C x / 2 - sum(budget * log(x)).
    for _ in range(1000):
        for i in range(size):
            cross = covariance[i] @ weights - diagonal[i] * weights[i]
            weights[i] = (math.sqrt(cross**2 + 4 * diagonal[i] * budget) - cross) / (2 * diagonal[i])
        contributions = weights * (covariance @ weights)
        if np.allclose(contributions, budget, rtol=1e-9, atol=0):
            return weights / weights.sum()
    raise RuntimeError("Equal risk contribution solver did not converge")


def hierarchical_weights(observations, groups):
    covariance = LedoitWolf().fit(observations).covariance_
    within = np.zeros((covariance.shape[0], len(groups)))
    for col, members in enumerate(groups):
        within[members, col] = equal_risk_weights(covariance[np.ix_(members, members)])
    group_covariance = within.T @ covariance @ within
    return within @ equal_risk_weights(group_covariance)


def monthly_targets(returns, sample_start, lag, groups, lookback=None):
    # A new month makes the previous session's close the month-end signal.
    months = returns.index.to_period("M")
    first_rows = np.flatnonzero(months[1:] != months[:-1]) + 1
    targets = {}
    for first_row in first_rows:
        signal_row = first_row - 1
        fill_row = signal_row + lag
        if fill_row >= len(returns) or returns.index[fill_row] < sample_start:
            continue
        if lookback is None:
            weights = np.full(returns.shape[1], 1.0 / returns.shape[1])
        else:
            start_row = signal_row + 1 - lookback
            if start_row < 0:
                raise ValueError(f"Insufficient warm-up for {lookback} daily returns")
            observations = returns.iloc[start_row:signal_row + 1].to_numpy()
            weights = hierarchical_weights(observations, groups)
        targets[returns.index[fill_row]] = weights
    return targets


def simulate(returns, targets, fee_rate):
    weights = np.zeros(returns.shape[1])
    rows = []
    for date, asset_returns in zip(returns.index, returns.to_numpy()):
        # The incoming close-to-close return accrues to the previous close's weights.
        marked = weights * (1.0 + asset_returns)
        growth = 1.0 + weights @ asset_returns
        weights = marked / growth
        fraction, turnover, trades = 1.0, 0.0, 0
        if date in targets:
            target = targets[date]

            def funding_gap(fraction):
                traded = np.abs(fraction * target - weights).sum()
                return fraction + fee_rate * traded - 1.0

            fraction = brentq(funding_gap, 0.0, 1.0)
            changes = np.abs(fraction * target - weights)
            turnover = float(changes.sum())
            trades = int(np.count_nonzero(changes > 1e-10))
            weights = target
        rows.append((growth * fraction - 1.0, turnover, trades))
    return pd.DataFrame(rows, index=returns.index, columns=["return", "turnover", "n_trades"])


def annual_sharpe(returns):
    volatility = returns.std(ddof=1)
    if not np.isfinite(volatility) or volatility == 0:
        return None
    return float(math.sqrt(SESSIONS_PER_YEAR) * returns.mean() / volatility)


def metrics(daily_results, cash_proxy):
    returns = daily_results["return"]
    value_path = np.r_[1.0, (1.0 + returns.to_numpy()).cumprod()]
    drawdown = value_path / np.maximum.accumulate(value_path) - 1.0
    return {
        "cagr": float(value_path[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1.0),
        "ann_vol": float(returns.std(ddof=1) * math.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": annual_sharpe(returns),
        "sharpe_excess": annual_sharpe(returns - cash_proxy.loc[returns.index]),
        "max_drawdown": float(drawdown.min()),
        "total_return": float(value_path[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, cash_proxy, is_end):
    in_sample = daily_results.index <= is_end
    return {
        "is": metrics(daily_results.loc[in_sample], cash_proxy),
        "oos": metrics(daily_results.loc[~in_sample], cash_proxy),
        "full": metrics(daily_results, cash_proxy),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    dates = config["dates"]
    data_start, sample_start, is_end, sample_end = (
        pd.Timestamp(dates[key]) for key in ("data_start", "sample_start", "is_end", "sample_end")
    )
    if not data_start < sample_start <= is_end < sample_end:
        raise ValueError("Expected data_start < sample_start <= is_end < sample_end")

    groups, tickers = [], []
    for members in config["universe"].values():
        groups.append(list(range(len(tickers), len(tickers) + len(members))))
        tickers.extend(members)
    cash_ticker = config["risk_free_ticker"]
    if not groups or not all(groups) or len(set(tickers + [cash_ticker])) != len(tickers) + 1:
        raise ValueError("Expected nonempty groups with unique tickers and a distinct cash proxy")
    lag = config["signal_lag_sessions"]
    windows = config["lookback_days"]
    costs = [config["cost_bps"], config["slippage_bps"]]
    if type(lag) is not int or lag < 1:
        raise ValueError("signal_lag_sessions must be a positive integer")
    if not windows or any(type(days) is not int or days < 2 for days in windows):
        raise ValueError("lookback_days must contain integers of at least 2")
    if len(set(windows)) != len(windows):
        raise ValueError("lookback_days must be distinct")
    if any(not np.isfinite(cost) or cost < 0 for cost in costs) or sum(costs) >= 10_000:
        raise ValueError("Expected nonnegative costs totaling less than 10000 bps")
    fee_rate = sum(costs) / 10_000

    prices, fetched_at = load_prices(tickers + [cash_ticker], data_start, sample_end)
    all_returns = prices.pct_change(fill_method=None).iloc[1:]
    sector_returns = all_returns[tickers]
    cash_proxy = all_returns[cash_ticker]
    sample_returns = sector_returns.loc[sample_start:sample_end]
    is_history = sector_returns.loc[:is_end]
    is_returns = sample_returns.loc[:is_end]
    if min(len(is_returns), len(sample_returns) - len(is_returns)) < 2:
        raise ValueError("Both evaluation periods need at least two sessions")

    scores = {}
    for window in windows:
        targets = monthly_targets(is_history, sample_start, lag, groups, window)
        if not targets:
            raise ValueError("Expected at least one in-sample rebalance")
        is_daily = simulate(is_returns, targets, fee_rate)
        score = annual_sharpe(is_daily["return"] - cash_proxy.loc[is_returns.index])
        if score is None:
            raise ValueError(f"In-sample excess Sharpe is undefined for {window} days")
        scores[window] = score
    selected = max(scores, key=lambda window: (scores[window], -window))

    strategy_targets = monthly_targets(sector_returns, sample_start, lag, groups, selected)
    benchmark_targets = monthly_targets(sector_returns, sample_start, lag, groups)
    strategy_daily = simulate(sample_returns, strategy_targets, fee_rate)
    benchmark_daily = simulate(sample_returns, benchmark_targets, fee_rate)
    strategy_report = period_metrics(strategy_daily, cash_proxy, is_end)
    benchmark_report = period_metrics(benchmark_daily, cash_proxy, is_end)
    data_range = f"{sample_returns.index[0].date()} to {sample_returns.index[-1].date()}"

    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return close (splits and distributions)")
    print(f"PARAM_COMBOS_TRIED: {len(scores)}")
    print(f"SELECTED_LOOKBACK_DAYS: {selected}")
    print(f"BENCHMARK: monthly equal-weight sector ETFs | SHARPE_EXCESS_PROXY: {cash_ticker}")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()} STRATEGY: {json.dumps(strategy_report[period], allow_nan=False)}")
        print(f"{period.upper()} BENCHMARK: {json.dumps(benchmark_report[period], allow_nan=False)}")
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **strategy_report,
        "benchmark": {"name": "monthly_equal_weight_sector_etfs", **benchmark_report["full"], **benchmark_report},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
