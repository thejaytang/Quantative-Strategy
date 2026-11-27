# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "455_ hierarchical_risk_parity_for_portfolio_a"\nuniverse: [ECH, COLO, EPU]\nrisk_free_ticker: BIL\nhistory_start: 2018-06-01\nstart: 2019-07-01\nis_end: 2022-06-30\noos_start: 2022-07-01\nend: 2025-06-30\nlookback_sessions: [63, 126, 252]\nlinkage: single\nsignal_lag_sessions: 1\ncost_bps: 5\nslippage_bps: 5\n'
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
"""Walk-forward hierarchical risk parity on three NUAM country proxies."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import brentq
from scipy.spatial.distance import squareform
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252


def adjusted_closes(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    exclusive_end = end + pd.Timedelta(days=1)
    cache_key = f"adjusted-close|{','.join(tickers)}|{start.date()}|{exclusive_end.date()}"
    cache_path = cache_dir / f"closes_{sha256(cache_key.encode()).hexdigest()[:16]}.csv"
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as source:
            fetched_at = source.readline().strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(source, index_col="Date", parse_dates=["Date"])
            return prices.loc[:, tickers], fetched_at

    with TemporaryDirectory(dir=cache_dir) as session_cache:
        yf.set_tz_cache_location(session_cache)
        quotes = yf.download(
            tickers, start=start.date().isoformat(), end=exclusive_end.date().isoformat(),
            auto_adjust=True, progress=False, threads=False, timeout=10,
        )
    if quotes.empty or "Close" not in quotes:
        raise RuntimeError("Yahoo Finance returned no adjusted closing prices")
    prices = quotes["Close"].reindex(columns=tickers).dropna(how="all").sort_index()
    if prices.empty or not np.isfinite(prices).all().all() or (prices <= 0).any().any():
        missing = prices.columns[prices.isna().any()].tolist()
        raise RuntimeError(f"Incomplete or invalid Yahoo Finance closes. Missing data: {missing}")
    if prices.index.has_duplicates:
        raise RuntimeError("Yahoo Finance returned duplicate session dates")

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_dir, delete=False) as destination:
        temp_path = Path(destination.name)
        try:
            destination.write(f"{fetched_at}\n")
            prices.to_csv(destination, index_label="Date")
            destination.flush()
            temp_path.replace(cache_path)
        finally:
            temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def cluster_variance(covariance, members):
    block = covariance[np.ix_(members, members)]
    inverse_variances = 1.0 / block.diagonal()
    weights = inverse_variances / inverse_variances.sum()
    return float(weights @ block @ weights)


def hrp_weights(log_returns, linkage_method):
    covariance = np.cov(log_returns, rowvar=False, ddof=1)
    variances = covariance.diagonal()
    if (variances <= 0).any():
        raise ValueError("HRP requires positive rolling variance for every asset")
    std_dev = np.sqrt(variances)
    correlation = covariance / np.outer(std_dev, std_dev)
    distance = np.sqrt((1.0 - correlation.clip(-1.0, 1.0)) / 2.0)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance), method=linkage_method)
    weights = np.ones(len(variances))
    clusters = [leaves_list(tree)]
    while clusters:
        members = clusters.pop()
        if len(members) < 2:
            continue
        midpoint = len(members) // 2
        left, right = members[:midpoint], members[midpoint:]
        left_variance = cluster_variance(covariance, left)
        right_variance = cluster_variance(covariance, right)
        left_fraction = right_variance / (left_variance + right_variance)
        weights[left] *= left_fraction
        weights[right] *= 1.0 - left_fraction
        clusters.extend((left, right))
    return weights


def simulate(prices, equity_tickers, config, lookback=None):
    """Mark the old holdings at each close before applying that close's fills."""
    start = pd.Timestamp(config["start"])
    sessions = prices.index
    equity_prices = prices[equity_tickers]
    simple_returns = equity_prices.pct_change().to_numpy()
    log_returns = np.log(equity_prices).diff().to_numpy()
    equal_weights = np.full(len(equity_tickers), 1.0 / len(equity_tickers))
    lag = config["signal_lag_sessions"]
    trade_rate = (config["cost_bps"] + config["slippage_bps"]) / 10_000

    # The first session of each month can be identified at that day's close.
    targets = {}
    for signal in np.flatnonzero(~sessions.to_period("M").duplicated()):
        execution = signal + lag
        if execution >= len(sessions) or sessions[execution] < start:
            continue
        if lookback is None:
            targets[execution] = equal_weights
        else:
            if signal < lookback:
                raise ValueError(f"Insufficient warm-up history for {sessions[signal].date()}")
            window = log_returns[signal - lookback + 1:signal + 1]
            targets[execution] = hrp_weights(window, config["linkage"])

    if not targets:
        raise ValueError("No scheduled rebalance within the simulation period")

    weights = np.zeros(len(equity_tickers))
    cash = 1.0
    rows = []
    first_session = sessions.searchsorted(start)
    for session in range(first_session, len(sessions)):
        holdings = weights * (1.0 + simple_returns[session])
        equity_before_fills = holdings.sum() + cash
        held = holdings / equity_before_fills
        target = targets.get(session)
        turnover, trades, equity_fraction = 0.0, 0, 1.0
        if target is None:
            weights = held
            cash /= equity_before_fills
        else:
            # Solve for post-cost equity so buys, sells and fees use the same cash balance.
            def cash_balance(fraction):
                traded = np.abs(fraction * target - held).sum()
                return fraction + trade_rate * traded - 1.0

            equity_fraction = brentq(cash_balance, 0.0, 1.0)
            changes = np.abs(equity_fraction * target - held)
            turnover = float(changes.sum())
            trades = int(np.count_nonzero(changes > 1e-10))
            weights = target
            cash = 0.0
        rows.append((equity_before_fills * equity_fraction - 1.0, turnover, trades))

    performance = pd.DataFrame(rows, index=sessions[first_session:], columns=["return", "turnover", "trades"])
    performance["cash_proxy"] = prices[config["risk_free_ticker"]].pct_change()
    return performance


def metrics(performance):
    if len(performance) < 2:
        raise ValueError("Each reporting period requires at least two sessions")
    daily = performance["return"].to_numpy()
    excess = daily - performance["cash_proxy"].to_numpy()
    wealth = np.r_[1.0, np.cumprod(1.0 + daily)]
    annualization = np.sqrt(SESSIONS_PER_YEAR)
    daily_vol = daily.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    return {
        "cagr": float(wealth[-1] ** (SESSIONS_PER_YEAR / len(daily)) - 1.0),
        "ann_vol": float(daily_vol * annualization),
        "sharpe": float(daily.mean() * annualization / daily_vol) if daily_vol else None,
        "sharpe_excess": float(excess.mean() * annualization / excess_vol) if excess_vol else None,
        "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1.0).min()),
        "total_return": float(wealth[-1] - 1.0),
        "turnover": float(performance["turnover"].sum()),
        "n_trades": int(performance["trades"].sum()),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    history_start, start, is_end, oos_start, end = (
        pd.Timestamp(config[key]) for key in ("history_start", "start", "is_end", "oos_start", "end")
    )
    if not history_start < start <= is_end < oos_start <= end:
        raise ValueError("Expected history_start < start <= is_end < oos_start <= end")
    if oos_start != is_end + pd.Timedelta(days=1):
        raise ValueError("IS and OOS calendar dates must be contiguous")
    if type(config["signal_lag_sessions"]) is not int or config["signal_lag_sessions"] < 1:
        raise ValueError("signal_lag_sessions must be an integer of at least one")
    for key in ("cost_bps", "slippage_bps"):
        if not 0 <= config[key] < 5_000:
            raise ValueError(f"Invalid basis points for {key}")
    assets = config["universe"]
    tickers = [*assets, config["risk_free_ticker"]]
    windows = config["lookback_sessions"]
    if len(assets) < 2 or len(tickers) != len(set(tickers)):
        raise ValueError("Use at least two distinct assets and a separate cash proxy")
    if not windows or any(type(window) is not int or window < 2 for window in windows):
        raise ValueError("lookback_sessions must contain integers of at least two")
    if len(windows) != len(set(windows)):
        raise ValueError("lookback_sessions must contain distinct windows")

    prices, fetched_at = adjusted_closes(tickers, history_start, end)
    if prices.index[0] > start or prices.index[-1] < oos_start:
        raise ValueError("Downloaded prices do not cover both reporting periods")
    if end - prices.index[-1] > pd.Timedelta(days=7):
        raise ValueError(f"Downloaded prices end too early: {prices.index[-1].date()}")

    # Candidate trials see only warm-up and IS data. Lower IS volatility wins.
    trials = []
    for window in windows:
        trial = simulate(prices.loc[:is_end], assets, config, lookback=window)
        trials.append((metrics(trial)["ann_vol"], window))
    _, selected_window = min(trials)

    strategy = simulate(prices, assets, config, lookback=selected_window)
    benchmark = simulate(prices, assets, config)
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": str(strategy.index[0].date()), "end": str(strategy.index[-1].date())},
        "is": metrics(strategy.loc[:is_end]),
        "oos": metrics(strategy.loc[oos_start:]),
        "full": metrics(strategy),
        "benchmark": metrics(benchmark),
    }

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return close in USD")
    print(f"PARAM_COMBOS_TRIED: {len(trials)}")
    print(f"SELECTED_LOOKBACK_SESSIONS: {selected_window}")
    print(f"IS: {json.dumps(results['is'], allow_nan=False)}")
    print(f"OOS: {json.dumps(results['oos'], allow_nan=False)}")
    print(f"FULL: {json.dumps(results['full'], allow_nan=False)}")
    print(f"BENCHMARK (equal weight, full period): {json.dumps(results['benchmark'], allow_nan=False)}")
    print(f"RESULTS_JSON: {json.dumps(results, allow_nan=False)}")


if __name__ == "__main__":
    main()
