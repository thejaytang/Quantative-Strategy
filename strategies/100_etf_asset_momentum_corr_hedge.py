# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 100_etf_asset_momentum_corr_hedge\ncategory: momentum/reversal\n\nuniverse: [SPY, IWM, EFA, EEM, IYR, QQQ, LQD, IEF, TIP, GLD, USO, DBC, FXE]\ncash_proxy: BIL\ndownload_start: 2007-06-01\nextra_market_closures: [2012-10-29, 2012-10-30, 2018-12-05, 2025-01-09]\nin_sample:\n  start: 2008-07-01\n  end: 2023-02-28\nout_of_sample:\n  start: 2023-03-01\n  end: 2026-11-24\n\nmomentum_sessions: [63, 126, 189, 252]\ncorrelation_sessions:\n  short: 20\n  long: 250\nlong_count: 4\nshort_count: 4\nshort_gross_weight: 0.30\nlong_cash_buffer: 0.05  # Reserve this fraction of prior-close portfolio value before next-open fills.\n\ncosts:\n  commission_bps: 2             # Per dollar bought or sold, including covers.\n  slippage_bps: 3               # Adverse opening-price slippage per dollar traded.\n  short_borrow_bps_annual: 100  # Charged on marked short notional.\n  short_rebate_haircut_bps_annual: 25  # Reduces the BIL proxy rate on short proceeds.\n  trading_bps_sensitivity: [0, 15]  # Combined commission and slippage.\n  short_borrow_bps_annual_sensitivity: [300]\n'
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
"""C01 cross-asset ETF momentum backtest."""

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
TRADING_DAYS = 252
BPS = 10_000
FETCH_HEADER = "# DATA_FETCHED_AT: "
ANNUALIZED_ROUNDOFF = np.finfo(float).eps * np.sqrt(TRADING_DAYS)


class NYSECalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def load_prices(symbols, start, end, extra_closures):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    columns = pd.MultiIndex.from_product([["Open", "Close"], symbols])
    cache_key = hashlib.sha256(f"adjusted:{symbols}:{start}:{end}".encode()).hexdigest()[:16]
    cache = cache_dir / f"yahoo_timestamped_{cache_key}.csv"
    expected = pd.date_range(start, end, freq=CustomBusinessDay(calendar=NYSECalendar(), holidays=extra_closures))

    def validate(prices):
        prices = prices.reindex(columns=columns).sort_index().loc[start:end]
        if prices.index.has_duplicates:
            raise ValueError("Duplicate Yahoo Finance dates")
        missing = expected.difference(prices.index)
        unexpected = prices.index.difference(expected)
        if len(missing) or len(unexpected):
            raise ValueError(f"Yahoo Finance session mismatch. Missing {list(missing.date)}, "
                             f"unexpected {list(unexpected.date)}")
        valid = np.isfinite(prices) & prices.gt(0)
        gaps = valid.columns[~valid.all(axis=0)]
        if len(gaps):
            raise ValueError(f"Missing or invalid adjusted bars for {list(gaps)}")
        return prices

    if cache.exists():
        with cache.open(encoding="utf-8") as handle:
            header = handle.readline().strip()
            if not header.startswith(FETCH_HEADER):
                raise ValueError(f"Missing fetch timestamp in {cache.name}")
            fetched_at = datetime.fromisoformat(header.removeprefix(FETCH_HEADER))
            prices = pd.read_csv(handle, header=[0, 1], index_col=0, parse_dates=True)
        return validate(prices), fetched_at

    # Keep yfinance's anonymous session and timezone caches in the working folder.
    with TemporaryDirectory(dir=cache_dir) as yf_cache:
        yf.set_tz_cache_location(yf_cache)
        for threads in (5, 2):
            raw = yf.download(
                symbols, start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date(),
                auto_adjust=True, progress=False, threads=threads, timeout=10,
            )
            try:
                prices = validate(raw)
                fetched_at = datetime.now(timezone.utc).replace(microsecond=0)
                break
            except ValueError as error:
                failure = error
        else:
            raise RuntimeError(f"Public Yahoo Finance download failed: {failure}") from failure

    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_dir, delete=False) as handle:
        temporary_path = Path(handle.name)
        try:
            handle.write(FETCH_HEADER + fetched_at.isoformat() + "\n")
            prices.to_csv(handle, index_label="Date")
            handle.flush()
            temporary_path.replace(cache)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def monthly_targets(closes, inception, config):
    windows = config["momentum_sessions"]
    fast = config["correlation_sessions"]["short"]
    slow = config["correlation_sessions"]["long"]
    long_count, short_count = config["long_count"], config["short_count"]
    short_gross = config["short_gross_weight"]
    ranks = [closes.pct_change(window, fill_method=None).rank(axis=1, pct=True) for window in windows]
    scores = sum(ranks) / len(ranks)
    daily_returns = closes.pct_change(fill_method=None)
    pairs = np.triu_indices(closes.shape[1], k=1)
    months = closes.index.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])

    hedged, long_only, equal_weight = {}, {}, {}
    for day in month_ends:
        if day < max(max(windows), slow, fast):
            continue
        execution = max(closes.index[day + 1], inception)
        rank_order = np.argsort(-scores.iloc[day].to_numpy(), kind="stable")
        long = np.zeros(closes.shape[1])
        long[rank_order[:long_count]] = 1 / long_count
        target = long.copy()
        fast_corr = daily_returns.iloc[day + 1 - fast:day + 1].corr().to_numpy()[pairs]
        slow_corr = daily_returns.iloc[day + 1 - slow:day + 1].corr().to_numpy()[pairs]
        if not (np.isfinite(fast_corr).all() and np.isfinite(slow_corr).all()):
            raise ValueError(f"Undefined correlation at {closes.index[day].date()}")
        if fast_corr.mean() > slow_corr.mean():
            target[rank_order[-short_count:]] = -short_gross / short_count
        hedged[execution] = target
        long_only[execution] = long
        equal_weight[execution] = np.full(closes.shape[1], 1 / closes.shape[1])

    if inception not in hedged:
        raise ValueError("Increase download history to allow a complete month-end signal before the sample")
    return hedged, long_only, equal_weight


def simulate(opens, closes, targets, prior_close, cash_buffer, trade_bps, borrow_bps, rebate_haircut_bps):
    """Size at the prior close, fill fixed units at the open, and mark at the close."""
    positions = np.zeros(opens.shape[1] - 1)
    cash = previous_value = 1.0
    previous_date = prior_close.name
    previous_prices = prior_close.to_numpy()
    trade_rate = trade_bps / BPS
    short_daily_rate = (borrow_bps + rebate_haircut_bps) / (BPS * 365)
    daily_results = []

    for date, opening, closing in zip(opens.index, opens.to_numpy(), closes.to_numpy()):
        target = targets.get(date)
        if target is not None:
            sizing_weights = np.where(target > 0, target * (1 - cash_buffer), target)
            target_units = previous_value * sizing_weights / previous_prices[:-1]
            order_units = target_units - positions

        days_elapsed = (date - previous_date).days
        short_notional = -(np.minimum(positions, 0) @ previous_prices[:-1])
        cash *= opening[-1] / previous_prices[-1]
        cash -= short_notional * short_daily_rate * days_elapsed
        if cash < 0:
            raise ValueError(f"Negative cash before trading on {date.date()}")
        opening_value = cash + positions @ opening[:-1]
        if opening_value <= 0:
            raise ValueError(f"Portfolio capital exhausted at the opening on {date.date()}")

        turnover = 0.0
        n_trades = 0
        if target is not None:
            filled_amounts = order_units * opening[:-1]
            traded = np.abs(filled_amounts).sum()
            cash -= filled_amounts.sum() + traded * trade_rate
            if cash < 0:
                raise ValueError(f"Negative cash after predetermined fills on {date.date()}: {cash:.6g}")
            positions = target_units
            turnover = traded / opening_value
            n_trades = np.count_nonzero(order_units)

        cash *= closing[-1] / opening[-1]
        closing_value = cash + positions @ closing[:-1]
        if closing_value <= 0:
            raise ValueError(f"Portfolio capital exhausted at the close on {date.date()}")
        daily_results.append((closing_value / previous_value - 1, turnover, n_trades))
        previous_date, previous_prices, previous_value = date, closing, closing_value
    return pd.DataFrame(daily_results, index=opens.index, columns=["return", "turnover", "n_trades"])


def metrics(history, cash_returns, start, end, initial_date):
    period = history.loc[start:end]
    if period.empty:
        raise ValueError(f"No returns for {start.date()} through {end.date()}")
    daily = period["return"]
    excess = daily - cash_returns.loc[daily.index]
    equity = np.r_[1.0, (1 + daily).cumprod().to_numpy()]
    start_index = history.index.get_loc(period.index[0])
    base_date = history.index[start_index - 1] if start_index else initial_date
    years = (period.index[-1] - base_date).days / 365.25
    vol = daily.std() * np.sqrt(TRADING_DAYS) if len(daily) > 1 else np.nan
    excess_vol = excess.std() * np.sqrt(TRADING_DAYS) if len(excess) > 1 else np.nan
    values = dict(
        cagr=equity[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        ann_vol=vol,
        sharpe=daily.mean() * TRADING_DAYS / vol if vol > ANNUALIZED_ROUNDOFF else np.nan,
        sharpe_excess=excess.mean() * TRADING_DAYS / excess_vol if excess_vol > ANNUALIZED_ROUNDOFF else np.nan,
        max_drawdown=(equity / np.maximum.accumulate(equity) - 1).min(),
        total_return=equity[-1] - 1,
        turnover=period["turnover"].sum(),
    )
    result = {name: float(value) if math.isfinite(value) else None for name, value in values.items()}
    result["n_trades"] = int(period["n_trades"].sum())
    return result


def print_table(reports):
    columns = [
        ("cagr", "CAGR", ".2%"), ("ann_vol", "Ann. vol", ".2%"), ("sharpe", "Sharpe", ".2f"),
        ("sharpe_excess", "Excess Sharpe", ".2f"), ("max_drawdown", "Max DD", ".2%"),
        ("total_return", "Total return", ".2%"), ("turnover", "Turnover", ".2f"), ("n_trades", "Trades", "d"),
    ]
    print(f"{'Portfolio / assumptions':<40} {'Period':<6} " + " ".join(f"{header:>13}" for _, header, _ in columns))
    for name, report in reports.items():
        for period, row in report.items():
            cells = []
            for key, _, fmt in columns:
                value = row[key]
                cells.append("n/a" if value is None else format(value, fmt))
            print(f"{name:<40} {period.upper():<6} " + " ".join(f"{cell:>13}" for cell in cells))


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Define nonoverlapping IS and OOS periods in chronological order")
    periods = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    universe = config["universe"]
    symbols = [*universe, config["cash_proxy"]]
    if len(set(symbols)) != len(symbols):
        raise ValueError("Risk tickers and the cash proxy must be distinct")
    if not (0 < config["long_count"] and 0 < config["short_count"]
            and config["long_count"] + config["short_count"] <= len(universe)):
        raise ValueError("Long and short counts must form disjoint nonempty baskets")

    cash_buffer = config["long_cash_buffer"]
    if not 0 < cash_buffer < 1:
        raise ValueError("The long cash buffer must be between zero and one")
    prices, fetched_at = load_prices(symbols, str(config["download_start"]), str(oos_end.date()),
                                     config["extra_market_closures"])
    evaluation = prices.loc[is_start:oos_end]
    if evaluation.empty:
        raise ValueError("The configured evaluation period has no common price data")
    inception = evaluation.index[0]
    hedged, long_only, equal_weight = monthly_targets(prices["Close"][universe], inception, config)
    costs = config["costs"]
    trade = costs["commission_bps"] + costs["slippage_bps"]
    borrow = costs["short_borrow_bps_annual"]
    haircut = costs["short_rebate_haircut_bps_annual"]
    cases = {
        f"Selective hedge ({trade:g} bps)": (hedged, trade, borrow, haircut),
        f"Long-only momentum ({trade:g} bps)": (long_only, trade, 0, 0),
        f"Equal weight ({trade:g} bps)": (equal_weight, trade, 0, 0),
        "Selective hedge (gross, before costs)": (hedged, 0, 0, 0),
    }
    for rate in costs["trading_bps_sensitivity"]:
        cases[f"Selective hedge ({rate:g} bps trading)"] = (hedged, rate, borrow, haircut)
    for rate in costs["short_borrow_bps_annual_sensitivity"]:
        cases[f"Selective hedge ({rate:g} bps annual borrow)"] = (hedged, trade, rate, haircut)

    opens, closes = evaluation["Open"][symbols], evaluation["Close"][symbols]
    prior_close = prices["Close"][symbols].iloc[prices.index.get_loc(inception) - 1]
    cash_returns = prices["Close"][symbols[-1]].pct_change(fill_method=None)
    reports = {}
    for name, (targets, trading_rate, borrow_rate, rebate_rate) in cases.items():
        history = simulate(opens, closes, targets, prior_close, cash_buffer, trading_rate, borrow_rate, rebate_rate)
        reports[name] = {
            period: metrics(history, cash_returns, start, end, prior_close.name)
            for period, (start, end) in periods.items()
        }

    data_range = f"{inception.date()} to {evaluation.index[-1].date()}"
    print(f"{config['strategy']} | {data_range}")
    print(f"IS {is_start.date()} to {is_end.date()} | OOS {oos_start.date()} to {oos_end.date()}")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}")
    print("PRICE_ADJUSTMENT: Yahoo Finance daily Open and Close, yfinance auto_adjust=True (splits and distributions)")
    print(f"Excess Sharpe uses daily returns minus {symbols[-1]} close-to-close returns.")
    print("Turnover sums dollars traded divided by pre-trade portfolio value. Trades count nonzero ETF fills.")
    print_table(reports)
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **reports[f"Selective hedge ({trade:g} bps)"],
        "benchmark": {"name": "equal_weight", **reports[f"Equal weight ({trade:g} bps)"]},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
