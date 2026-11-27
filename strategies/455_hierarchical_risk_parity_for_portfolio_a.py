# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "455_hierarchical_risk_parity_for_portfolio_a"\n# Fixed regional ETF proxies. COLO includes the same fund\'s GXG history.\nuniverse: [ECH, COLO, EPU]\nrisk_free_ticker: BIL\nhistory_start: 2018-06-01\nstart: 2019-07-01\nis_end: 2022-06-30\noos_start: 2022-07-01\nend: 2025-06-30\nlookback_sessions: [63, 126, 252]\n# Keep 63 sessions for all IS signals. Use the lowest IS volatility to select the OOS window.\ninitial_lookback_sessions: 63\nlinkage: single\n# Count sessions from the first session\'s close each month to the fill close.\n# Quantities are locked at the prior session\'s close.\nsignal_lag_sessions: 1\ncost_bps: 5\nslippage_bps: 5\n# Use 97% of the prior-close portfolio value to fix quantities before fees.\n# Budget uses same-fill sale proceeds. Cash shortfalls stop the simulation.\ncash_buffer_bps: 300\n'
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
"""Hierarchical risk parity on three NUAM country proxies."""

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
# Shared full-day closures for this sample's NYSE Arca and Cboe BZX listings.
US_EQUITY_HOLIDAYS = AbstractHolidayCalendar(rules=[
    Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
    USMartinLutherKingJr,
    USPresidentsDay,
    GoodFriday,
    USMemorialDay,
    Holiday("Juneteenth", month=6, day=19, observance=nearest_workday, start_date="2022-01-01"),
    Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
    USLaborDay,
    USThanksgivingDay,
    Holiday("Christmas", month=12, day=25, observance=nearest_workday),
])
US_EQUITY_SPECIAL_CLOSURES = pd.DatetimeIndex(["2018-12-05", "2025-01-09"])


def validate_closes(prices, tickers, start, end):
    dates = prices.index
    if (
        not isinstance(dates, pd.DatetimeIndex) or dates.tz is not None
        or dates.hasnans or not dates.equals(dates.normalize())
        or not dates.is_monotonic_increasing or not dates.is_unique
    ):
        raise ValueError("Expected unique, chronological US session dates in Yahoo Finance closes")
    if len(prices.columns) != len(tickers) or set(prices.columns) != set(tickers):
        raise ValueError("Yahoo Finance closes must contain exactly one column per requested ticker")
    prices = prices.loc[start:end, tickers].astype(float)
    if not 2018 <= start.year <= end.year <= 2025:
        raise ValueError("The NYSE Arca and Cboe BZX sample calendar covers 2018–2025")
    closures = US_EQUITY_HOLIDAYS.holidays(start, end).union(US_EQUITY_SPECIAL_CLOSURES)
    expected = pd.bdate_range(start, end).difference(closures)
    if not prices.index.equals(expected):
        missing = expected.difference(prices.index).strftime("%Y-%m-%d").tolist()
        extra = prices.index.difference(expected).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"US equity session mismatch. Missing: {missing}. Unexpected: {extra}")
    values = prices.to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Yahoo Finance closes must be complete, finite and strictly positive")
    prices.index = expected.rename("Date")
    prices.columns.name = None
    return prices


def adjusted_closes(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    exclusive_end = end + pd.Timedelta(days=1)
    cache_key = f"adjusted-close|{','.join(tickers)}|{start.date()}|{exclusive_end.date()}"
    cache_path = cache_dir / f"closes_{sha256(cache_key.encode()).hexdigest()[:16]}.csv"
    try:
        snapshot = cache_path.read_bytes()
        downloaded = False
    except FileNotFoundError:
        with TemporaryDirectory(dir=cache_dir) as session_cache:
            yf.set_tz_cache_location(session_cache)
            quotes = yf.download(
                tickers, start=start.date().isoformat(), end=exclusive_end.date().isoformat(),
                interval="1d", auto_adjust=True, keepna=True, ignore_tz=True,
                progress=False, threads=False, timeout=10,
            )
        if quotes.empty or "Close" not in quotes:
            raise RuntimeError("Yahoo Finance returned no adjusted closing prices")
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        snapshot = f"{fetched_at}\n{quotes['Close'].to_csv(index_label='Date')}".encode("utf-8")
        downloaded = True

    # Always simulate from the same CSV bytes that are used to compute the data hash.
    header, _, csv_data = snapshot.partition(b"\n")
    fetched_at = header.decode("utf-8").strip()
    if datetime.fromisoformat(fetched_at).utcoffset() is None:
        raise ValueError("Price cache fetch timestamp requires a UTC offset")
    prices = pd.read_csv(BytesIO(csv_data), index_col="Date", parse_dates=["Date"])
    prices = validate_closes(prices, tickers, start, end)
    if downloaded:
        with NamedTemporaryFile(dir=cache_dir, delete=False) as destination:
            temp_path = Path(destination.name)
            try:
                destination.write(snapshot)
                destination.flush()
                temp_path.replace(cache_path)
            finally:
                temp_path.unlink(missing_ok=True)
    return prices, fetched_at, sha256(snapshot).hexdigest()


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


def rebalance_targets(equity_prices, config, lookback, oos_lookback):
    start = pd.Timestamp(config["start"])
    oos_start = pd.Timestamp(config["oos_start"])
    sessions = equity_prices.index
    log_returns = np.log(equity_prices).diff().to_numpy()
    n_assets = len(equity_prices.columns)
    equal_weights = np.full(n_assets, 1.0 / n_assets)
    lag = config["signal_lag_sessions"]

    # The first session of each month can be identified at that day's close.
    targets = {}
    for signal in np.flatnonzero(~sessions.to_period("M").duplicated()):
        execution = signal + lag
        if execution >= len(sessions) or sessions[execution] < start:
            continue
        window_length = lookback
        if oos_lookback is not None and sessions[signal] >= oos_start:
            window_length = oos_lookback
        if window_length is None:
            targets[execution] = equal_weights
        else:
            if signal < window_length:
                raise ValueError(f"Insufficient warm-up history for {sessions[signal].date()}")
            window = log_returns[signal - window_length + 1:signal + 1]
            targets[execution] = hrp_weights(window, config["linkage"])
    return targets


def trade_cash_flows(changes, prices, trade_rate):
    amounts = changes * prices
    traded = float(np.abs(amounts).sum())
    fees = trade_rate * traded
    return float(amounts.sum()) + fees, fees, traded


def simulate(prices, equity_tickers, config, lookback=None, oos_lookback=None):
    """Size at the prior close and apply the locked quantities at the fill close."""
    prices = prices.loc[pd.Timestamp(config["history_start"]):pd.Timestamp(config["end"])]
    equity_prices = prices[equity_tickers]
    sessions = equity_prices.index
    closes = equity_prices.to_numpy()
    targets = rebalance_targets(equity_prices, config, lookback, oos_lookback)
    trade_rate = (config["cost_bps"] + config["slippage_bps"]) / 10_000
    invested_fraction = 1.0 - config["cash_buffer_bps"] / 10_000
    first_session = sessions.searchsorted(pd.Timestamp(config["start"]))

    positions = np.zeros(len(equity_tickers))
    cash = 1.0
    previous_value = cash
    previous_close = closes[max(0, first_session - 1)]
    planned_changes = None
    daily_results = []
    for session in range(max(0, first_session - 1), len(sessions)):
        closing_prices = closes[session]
        value_before_fill = float(positions @ closing_prices + cash)
        price_profit = float(positions @ (closing_prices - previous_close))
        fees, turnover, trades = 0.0, 0.0, 0
        if planned_changes is not None:
            spent, fees, traded = trade_cash_flows(planned_changes, closing_prices, trade_rate)
            cash_after_fill = cash - spent
            if not np.isfinite(cash_after_fill):
                raise RuntimeError(f"Nonfinite cash at the simulated fill on {sessions[session].date()}")
            if cash_after_fill < 0:
                raise RuntimeError(
                    f"Predetermined quantities exceed cash on {sessions[session].date()} "
                    f"by {-cash_after_fill:.8g} per unit of starting capital"
                )
            cash = cash_after_fill
            positions += planned_changes
            turnover = traded / value_before_fill
            trades = int(np.count_nonzero(planned_changes))

        closing_value = value_before_fill - fees
        if session >= first_session:
            daily_return = (price_profit - fees) / previous_value
            daily_results.append((daily_return, turnover, trades, cash / closing_value))
        previous_value = closing_value
        previous_close = closing_prices

        # Shares for the next session are fixed here, including the cash reserve.
        next_target = targets.get(session + 1)
        planned_changes = None
        if next_target is not None:
            target_positions = previous_value * invested_fraction * next_target / closing_prices
            if not np.isfinite(target_positions).all() or (target_positions < 0).any():
                raise RuntimeError(f"Invalid long-only quantities at sizing on {sessions[session].date()}")
            planned_changes = target_positions - positions
            # Ignore machine-precision differences before locking the quantities.
            unchanged = np.isclose(target_positions, positions, rtol=8 * np.finfo(float).eps, atol=0.0)
            planned_changes[unchanged] = 0.0
            estimated_spend, _, _ = trade_cash_flows(planned_changes, closing_prices, trade_rate)
            if not np.isfinite(estimated_spend):
                raise RuntimeError(f"Nonfinite estimated cash use at sizing on {sessions[session].date()}")
            if estimated_spend > cash:
                raise RuntimeError(
                    f"Predetermined quantities exceed cash at sizing on {sessions[session].date()} "
                    f"by {estimated_spend - cash:.8g} per unit of starting capital"
                )
            planned_changes.setflags(write=False)

    daily = pd.DataFrame(
        daily_results, index=sessions[first_session:], columns=["return", "turnover", "trades", "cash_fraction"],
    )
    daily["cash_proxy"] = prices[config["risk_free_ticker"]].pct_change(fill_method=None)
    return daily


def metrics(performance):
    if performance.empty:
        raise ValueError("Each reporting period requires at least one session")
    daily = performance["return"].to_numpy()
    excess = daily - performance["cash_proxy"].to_numpy()
    wealth = np.r_[1.0, np.cumprod(1.0 + daily)]
    annualization = np.sqrt(SESSIONS_PER_YEAR)
    daily_vol = daily.std(ddof=1) if len(daily) > 1 else None
    excess_vol = excess.std(ddof=1) if len(excess) > 1 and np.isfinite(excess).all() else None
    return {
        "cagr": float(wealth[-1] ** (SESSIONS_PER_YEAR / len(daily)) - 1.0),
        "ann_vol": float(daily_vol * annualization) if daily_vol is not None else None,
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
    if not 0 < config["cash_buffer_bps"] < 10_000:
        raise ValueError("cash_buffer_bps must be positive and less than 10,000")
    assets = config["universe"]
    tickers = [*assets, config["risk_free_ticker"]]
    windows = config["lookback_sessions"]
    initial_window = config["initial_lookback_sessions"]
    if len(assets) < 2 or len(tickers) != len(set(tickers)):
        raise ValueError("Use at least two distinct assets and a separate cash proxy")
    if not windows or any(type(window) is not int or window < 2 for window in windows):
        raise ValueError("lookback_sessions must contain integers of at least two")
    if len(windows) != len(set(windows)):
        raise ValueError("lookback_sessions must contain distinct windows")
    if type(initial_window) is not int or initial_window not in windows:
        raise ValueError("initial_lookback_sessions must be one of the candidate windows")

    prices, fetched_at, data_hash = adjusted_closes(tickers, history_start, end)

    # Candidate trials see only warm-up and IS data. Lower IS volatility wins.
    trial_vols = {}
    for window in windows:
        trial = simulate(prices.loc[:is_end], assets, config, lookback=window)
        trial_vols[window] = metrics(trial)["ann_vol"]
    if any(vol is None for vol in trial_vols.values()):
        raise ValueError("Window selection requires at least two IS sessions")
    selected_window = min(trial_vols, key=lambda window: (trial_vols[window], window))

    # The IS starting positions must come from the predeclared window.
    strategy = simulate(prices, assets, config, lookback=initial_window, oos_lookback=selected_window)
    benchmark = simulate(prices, assets, config)
    minimum_fill_cash = {}
    for name, daily in (("hrp", strategy), ("benchmark", benchmark)):
        fill_cash = daily.loc[daily["trades"] > 0, "cash_fraction"]
        minimum_fill_cash[name] = None if fill_cash.empty else float(fill_cash.min())
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": str(strategy.index[0].date()), "end": str(strategy.index[-1].date())},
        "is": metrics(strategy.loc[:is_end]),
        "oos": metrics(strategy.loc[oos_start:]),
        "full": metrics(strategy),
        "benchmark": metrics(benchmark),
    }

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"DATA_SHA256: {data_hash}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return close in USD")
    print(f"PARAM_COMBOS_TRIED: {len(trial_vols)}")
    print(f"IS_CANDIDATE_ANN_VOL: {json.dumps(trial_vols, allow_nan=False)}")
    print(f"INITIAL_LOOKBACK_SESSIONS: {initial_window}")
    print(f"SELECTED_LOOKBACK_SESSIONS: {selected_window}")
    print(f"MIN_POST_FILL_CASH_FRACTION: {json.dumps(minimum_fill_cash, allow_nan=False)}")
    print(f"IS: {json.dumps(results['is'], allow_nan=False)}")
    print(f"OOS: {json.dumps(results['oos'], allow_nan=False)}")
    print(f"FULL: {json.dumps(results['full'], allow_nan=False)}")
    print(f"BENCHMARK (equal weight, IS): {json.dumps(metrics(benchmark.loc[:is_end]), allow_nan=False)}")
    print(f"BENCHMARK (equal weight, OOS): {json.dumps(metrics(benchmark.loc[oos_start:]), allow_nan=False)}")
    print(f"BENCHMARK (equal weight, full period): {json.dumps(results['benchmark'], allow_nan=False)}")
    print(f"RESULTS_JSON: {json.dumps(results, allow_nan=False)}")


if __name__ == "__main__":
    main()
