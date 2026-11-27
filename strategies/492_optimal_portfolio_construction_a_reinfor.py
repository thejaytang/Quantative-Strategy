# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 492_optimal_portfolio_construction_a_reinfor\nuniverse:\n  cyclical: [XLB, XLE, XLF, XLI, XLRE, XLY]\n  defensive: [XLP, XLU, XLV]\n  technology_and_communication: [XLK, XLC]\nrisk_free: BIL\n\ndownload_start: '2010-01-01'\nstart: '2012-01-01'\nis_end: '2019-12-31'\nend: '2025-08-13'\n\ncovariance_sessions: 126\neligibility_sessions: 150\nsignal_lag_sessions: 1\ncost_bps: 5\nslippage_bps: 2\ncash_buffer_bps: 200  # Reserve 2% of the previous-close value when sizing.\n\nrobustness:\n  covariance_sessions: [105, 147]\n  higher_cost_bps: 23\n"
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
"""Monthly, two-stage risk budgeting with U.S. sector ETFs."""

import hashlib
import json
import signal
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from sklearn.covariance import LedoitWolf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252


class NyseHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def nyse_sessions(start, end):
    holidays = NyseHolidays().holidays(start, end)
    # Exceptional closures for the research window: Hurricane Sandy, Bush and Carter.
    special_closures = pd.to_datetime(["2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])
    return pd.bdate_range(start, end).difference(holidays.union(special_closures))


def adjusted_prices(symbols, start, end):
    sessions = nyse_sessions(start, end)
    CACHE.mkdir(exist_ok=True)
    query = "|".join(["yfinance-adjusted-close-v1", start.date().isoformat(), end.date().isoformat(), *symbols])
    key = hashlib.sha256(query.encode()).hexdigest()[:16]
    path = CACHE / f"prices-{key}.csv"

    if path.exists():
        with path.open() as cached:
            fetched_at = cached.readline().strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(cached, index_col=0, parse_dates=True)
        return check_prices(prices, symbols, sessions), fetched_at

    exclusive_end = str((end + pd.Timedelta(days=1)).date())
    with TemporaryDirectory(dir=CACHE) as provider_cache:
        yf.set_tz_cache_location(provider_cache)
        raw = yf.download(
            symbols,
            start=str(start.date()),
            end=exclusive_end,
            auto_adjust=True,
            progress=False,
            threads=4,
            timeout=12,
        )
    if raw is None or raw.empty or "Close" not in raw:
        raise RuntimeError("Yahoo Finance returned no adjusted closes.")
    prices = check_prices(raw["Close"], symbols, sessions)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", dir=CACHE, delete=False) as target:
        print(fetched_at, file=target)
        prices.to_csv(target)
    Path(target.name).replace(path)
    return prices, fetched_at


def check_prices(prices, symbols, sessions):
    prices = prices.reindex(columns=symbols).sort_index()
    if not prices.index.is_unique:
        raise ValueError("Expected unique daily price history.")
    if sessions.empty:
        raise ValueError("The requested dates contain no NYSE sessions.")
    missing = sessions.difference(prices.dropna(how="all").index)
    unexpected = prices.index.difference(sessions)
    if len(missing) or len(unexpected):
        raise ValueError(f"NYSE session mismatch: missing {missing[:5].date.tolist()}, "
                         f"unexpected {unexpected[:5].date.tolist()}.")
    for ticker, series in prices.items():
        first = series.first_valid_index()
        if first is None:
            raise ValueError(f"No Yahoo Finance prices for {ticker}.")
        live = series.loc[first:].to_numpy()
        if not np.isfinite(live).all() or (live <= 0).any():
            raise ValueError(f"Missing or invalid price for {ticker} after its first observation.")
    if prices[symbols[-1]].isna().any():
        raise ValueError("BIL prices must cover every session in the data.")
    return prices.reindex(sessions.rename(prices.index.name))


def equal_risk_weights(covariance):
    """Solve x_i (C x)_i = 1/n by positive coordinate minimization."""
    scale = np.mean(np.diag(covariance))
    if scale <= 0:
        raise ValueError("The risk model requires positive average variance.")
    covariance = covariance / scale
    variances = np.diag(covariance)
    if (variances <= 0).any():
        raise ValueError("The risk model requires positive variances.")
    budget = 1.0 / len(variances)
    scaled_weights = np.sqrt(budget / variances)
    for _ in range(1000):
        for i, variance in enumerate(variances):
            cross = covariance[i] @ scaled_weights - variance * scaled_weights[i]
            root = np.sqrt(cross**2 + 4 * variance * budget)
            if cross >= 0:
                scaled_weights[i] = 2 * budget / (root + cross)
            else:
                scaled_weights[i] = (root - cross) / (2 * variance)
        contributions = scaled_weights * (covariance @ scaled_weights)
        if np.max(np.abs(contributions - budget)) < 1e-10:
            return scaled_weights / scaled_weights.sum()
    raise RuntimeError("Equal-risk solver did not converge.")


def monthly_targets(returns, groups, start, config, lookback):
    strategy_targets = {}
    benchmark_targets = {}
    months = returns.index.to_period("M")
    history_length = config["eligibility_sessions"]
    lag = config["signal_lag_sessions"]
    for signal_row in np.flatnonzero(months[:-1] != months[1:]):
        fill_row = signal_row + lag
        if fill_row >= len(returns) or returns.index[fill_row] < start:
            continue
        past = returns.iloc[max(0, signal_row + 1 - history_length) : signal_row + 1]
        if len(past) < history_length:
            continue
        eligible = past.columns[past.notna().all()]
        if eligible.empty:
            continue
        covariance = LedoitWolf(store_precision=False).fit(past[eligible].tail(lookback)).covariance_
        eligible_groups = [eligible.get_indexer(eligible.intersection(group)) for group in groups]
        active_groups = [members for members in eligible_groups if members.size]
        sleeve_mix = np.zeros((len(eligible), len(active_groups)))
        for sleeve, members in enumerate(active_groups):
            sleeve_mix[members, sleeve] = equal_risk_weights(covariance[np.ix_(members, members)])
        sleeve_cov = sleeve_mix.T @ covariance @ sleeve_mix
        weights = np.zeros(returns.shape[1])
        equal_weights = np.zeros_like(weights)
        indexes = returns.columns.get_indexer(eligible)
        weights[indexes] = sleeve_mix @ equal_risk_weights(sleeve_cov)
        equal_weights[indexes] = 1.0 / len(eligible)
        fill_day = returns.index[fill_row]
        strategy_targets[fill_day] = weights
        benchmark_targets[fill_day] = equal_weights
    return strategy_targets, benchmark_targets


def size_positions(target, previous_close, previous_value, cash_buffer):
    active = target > 0
    if (previous_close[active] <= 0).any():
        raise ValueError("Missing previous close for an eligible ETF.")
    positions = np.zeros_like(target)
    positions[active] = (1 - cash_buffer) * previous_value * target[active] / previous_close[active]
    return positions


def simulate(prices, targets, cost_bps, slippage_bps, cash_buffer):
    """Hold adjusted units and cash. The first row is the pre-sample close."""
    positions = np.zeros(prices.shape[1])
    cash = previous_value = 1.0
    fee_rate = (cost_bps + slippage_bps) / 10_000
    # Before the first valid quote an ETF has zero units and zero target weight.
    closes = prices.fillna(0).to_numpy()
    records = []
    for row, day in enumerate(prices.index[1:], start=1):
        target = targets.get(day)
        if target is not None:
            next_positions = size_positions(target, closes[row - 1], previous_value, cash_buffer)
            changes = next_positions - positions

        close = closes[row]
        turnover, trades = 0.0, 0
        if target is not None:
            traded_values = changes * close
            gross_traded = np.abs(traded_values).sum()
            turnover = gross_traded / (cash + positions @ close)
            remaining_cash = cash - (traded_values.sum() + fee_rate * gross_traded)
            if remaining_cash < 0:
                raise RuntimeError(f"{day.date()}: fixed quantities would leave negative cash ({remaining_cash:.8g}).")
            trades = int(np.count_nonzero(changes))
            positions = next_positions
            cash = remaining_cash

        value = cash + positions @ close
        records.append((value / previous_value - 1, turnover, trades))
        previous_value = value
    return pd.DataFrame(records, index=prices.index[1:], columns=["return", "turnover", "n_trades"])


def metrics(daily_results, cash_returns):
    daily = daily_results["return"].to_numpy()
    excess = daily - cash_returns.loc[daily_results.index].to_numpy()
    if not np.isfinite(excess).all():
        raise ValueError("Every simulated session requires a finite BIL return.")
    growth = np.r_[1.0, np.cumprod(1 + daily)]
    scale = np.sqrt(SESSIONS_PER_YEAR)
    vol = daily.std(ddof=1) if len(daily) >= 2 else None
    excess_vol = excess.std(ddof=1) if len(daily) >= 2 else None
    years = len(daily) / SESSIONS_PER_YEAR
    return {
        "cagr": float(growth[-1] ** (1 / years) - 1) if years else None,
        "ann_vol": float(vol * scale) if vol is not None else None,
        "sharpe": float(scale * daily.mean() / vol) if vol else None,
        "sharpe_excess": float(scale * excess.mean() / excess_vol) if excess_vol else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()) if years else None,
        "total_return": float(growth[-1] - 1) if years else None,
        "turnover": float(daily_results["turnover"].sum() / years) if years else None,
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, cash_returns, is_end):
    is_mask = daily_results.index <= is_end
    return {
        "is": metrics(daily_results.loc[is_mask], cash_returns),
        "oos": metrics(daily_results.loc[~is_mask], cash_returns),
        "full": metrics(daily_results, cash_returns),
    }


def main():
    signal.alarm(9 * 60)
    with (ROOT / "config.yaml").open() as source:
        config = yaml.safe_load(source)
    date_keys = ("download_start", "start", "is_end", "end")
    download_start, start, is_end, end = (pd.Timestamp(config[key]) for key in date_keys)
    if not download_start < start <= is_end < end:
        raise ValueError("Invalid warm-up, start, split or end dates.")

    groups = list(config["universe"].values())
    tickers = [ticker for group in groups for ticker in group]
    cash_ticker = config["risk_free"]
    if any(not (ticker.isascii() and ticker.isalpha() and ticker.isupper()) for ticker in tickers):
        raise ValueError("Use U.S. ETF symbols consisting of uppercase letters only.")
    if not tickers or len(tickers) != len(set(tickers)) or cash_ticker != "BIL" or cash_ticker in tickers:
        raise ValueError("Configure unique sector ETFs and use BIL separately.")
    lag = config["signal_lag_sessions"]
    lookback = config["covariance_sessions"]
    cost = config["cost_bps"]
    slip = config["slippage_bps"]
    cash_buffer = config["cash_buffer_bps"] / 10_000
    robust = config["robustness"]
    cases = dict.fromkeys([
        (lookback, cost),
        *((window, cost) for window in robust["covariance_sessions"]),
        (lookback, robust["higher_cost_bps"]),
    ])
    if type(lag) is not int or lag < 1:
        raise ValueError("Execution lag must be a positive number of sessions.")
    if not 0 < cash_buffer < 1:
        raise ValueError("The cash buffer must be between 0 and 100 percent.")
    history_length = config["eligibility_sessions"]
    if type(history_length) is not int or history_length < 2:
        raise ValueError("Eligibility requires at least two complete return observations.")
    for window, cost_case in cases:
        if type(window) is not int or not 2 <= window <= history_length:
            raise ValueError("Each covariance window must fit inside the eligibility window.")
        if not (0 <= cost_case and 0 <= slip and cost_case + slip < 10_000):
            raise ValueError("Invalid transaction cost or slippage in bps.")

    prices, fetched_at = adjusted_prices([*tickers, cash_ticker], download_start, end)
    first_sample_row = prices.index.searchsorted(start)
    if not 0 < first_sample_row < len(prices):
        raise ValueError("The sample needs a session and its preceding warm-up close.")
    sample_prices = prices[tickers].iloc[first_sample_row - 1:]
    all_returns = prices.pct_change(fill_method=None)
    returns = all_returns[tickers]
    cash_returns = all_returns[cash_ticker]
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return")

    print("ROBUSTNESS (IS only)")
    is_history = returns.loc[:is_end]
    is_prices = sample_prices.loc[:is_end]
    for window, cost_case in cases:
        if (window, cost_case) == (lookback, cost):
            continue
        targets, _ = monthly_targets(is_history, groups, start, config, window)
        daily = simulate(is_prices, targets, cost_case, slip, cash_buffer)
        stats = metrics(daily, cash_returns)
        print(f"  covariance_sessions={window}, cost_bps={cost_case}, slippage_bps={slip} "
              + json.dumps(stats, allow_nan=False))
    print(f"PARAM_COMBOS_TRIED: {len(cases)}")

    targets, benchmark_targets = monthly_targets(returns, groups, start, config, lookback)
    strategy_daily = simulate(sample_prices, targets, cost, slip, cash_buffer)
    benchmark_daily = simulate(sample_prices, benchmark_targets, cost, slip, cash_buffer)
    strategy_stats = period_metrics(strategy_daily, cash_returns, is_end)
    benchmark = period_metrics(benchmark_daily, cash_returns, is_end)
    for period, stats in strategy_stats.items():
        print(f"{period.upper()}: " + json.dumps(stats, allow_nan=False))
    print("BENCHMARK: Monthly equal-weight sector ETFs, with the same eligibility, lag, costs and cash buffer")
    for period, stats in benchmark.items():
        print(f"BENCHMARK_{period.upper()}: " + json.dumps(stats, allow_nan=False))

    results = {
        "strategy": config["strategy"],
        "data_range": {
            "start": strategy_daily.index[0].date().isoformat(),
            "end": strategy_daily.index[-1].date().isoformat(),
        },
        **strategy_stats,
        "benchmark": benchmark,
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))
    signal.alarm(0)


if __name__ == "__main__":
    main()
