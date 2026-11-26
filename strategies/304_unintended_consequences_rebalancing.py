# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 304_unintended_consequences_rebalancing\ncategory: momentum/reversal\nuniverse:\n  equity: SPY\n  bonds: IEF\n\ntarget_equity_weight: 0.60\ndrift_band: 0.01\nequity_tilt: 0.20\nexclude_month_start_sessions: 2\nexclude_month_end_sessions: 2\ntransaction_cost_bps: 2.0  # Per dollar bought or sold at the open.\nannualization_sessions: 252\n\nwarmup_start: 2006-10-01\nin_sample:\n  start: 2007-01-01\n  end: 2025-03-31\nout_of_sample:\n  start: 2025-04-01\n  end: 2026-10-30\n'
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
import json
import math
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
from pandas.tseries.offsets import CustomBusinessDay


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
DAY = pd.Timedelta(days=1)


class NyseRegularHolidays(AbstractHolidayCalendar):
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
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def load_prices(tickers, start, end):
    cache_path = CACHE / f"yahoo_adjusted_{'_'.join(tickers)}_{start.date()}_{end.date()}.csv"
    columns = pd.MultiIndex.from_product([["Open", "Close"], tickers])
    cached = cache_path.is_file()
    if cached:
        prices = pd.read_csv(cache_path, header=[0, 1], index_col=0, parse_dates=True)
    else:
        CACHE.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix=".yfinance-", dir=CACHE) as yahoo_cache:
            yf.set_tz_cache_location(yahoo_cache)
            prices = yf.download(tickers, start=str(start.date()), end=str((end + DAY).date()),
                                 interval="1d", auto_adjust=True, ignore_tz=True,
                                 threads=False, progress=False, timeout=15)

    prices = prices.reindex(columns=columns).sort_index().loc[start:end]
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("Yahoo Finance supplied no sessions or duplicate dates.")
    if prices.index[0] > start + 7 * DAY or prices.index[-1] < end - 7 * DAY:
        raise ValueError(f"Incomplete price coverage for {start.date()} through {end.date()}.")
    invalid = ~np.isfinite(prices) | prices.le(0)
    if invalid.any().any():
        date = invalid.index[invalid.any(axis=1)][0]
        fields = ", ".join(f"{ticker} {field}" for field, ticker in invalid.columns[invalid.loc[date]])
        raise ValueError(f"Missing or invalid adjusted price on {date.date()} for {fields}.")

    if not cached:
        with NamedTemporaryFile(mode="w", dir=CACHE, suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            try:
                prices.to_csv(handle, index_label="Date")
                handle.close()
                temporary_path.replace(cache_path)
            finally:
                temporary_path.unlink(missing_ok=True)
    return prices


def month_edge_mask(dates, exclude_first, exclude_last):
    # At each close the month's elapsed sessions are known. Rank the last sessions
    # by the regular NYSE schedule so surprise closures cannot rewrite earlier signals.
    observed = pd.Series(dates, index=dates)
    sessions_elapsed = observed.groupby(dates.to_period("M")).cumcount()
    months = dates.to_period("M")
    scheduled = pd.date_range(months[0].start_time, months[-1].end_time.normalize(),
                              freq=CustomBusinessDay(calendar=NyseRegularHolidays()))
    scheduled_dates = pd.Series(scheduled, index=scheduled)
    remaining = scheduled_dates.groupby(scheduled.to_period("M")).cumcount(ascending=False)
    remaining = remaining.reindex(dates)
    if remaining.isna().any():
        raise ValueError(f"Unexpected NYSE session date {remaining.index[remaining.isna()][0].date()}.")
    return (sessions_elapsed < exclude_first) | (remaining < exclude_last)


def threshold_crossings(close_returns, target, band):
    crossings = np.zeros(len(close_returns), dtype=int)
    weight = target
    for i, (equity_return, bond_return) in enumerate(close_returns.to_numpy()):
        equity_value = weight * (1 + equity_return)
        bond_value = (1 - weight) * (1 + bond_return)
        weight = equity_value / (equity_value + bond_value)
        if weight > target + band:
            crossings[i] = 1
        elif weight < target - band:
            crossings[i] = -1
        if crossings[i]:
            weight = target
    return pd.Series(crossings, index=close_returns.index)


def simulate(open_growth, equity_target, fee_rate):
    # Rows initially represent entry opens. Start from cash, then carry the
    # drifted weights forward to the next open before trading to new targets.
    equity_growth, bond_growth = open_growth.to_numpy().T
    gross_growth = equity_target * equity_growth + (1 - equity_target) * bond_growth
    prior_weight = (equity_target * equity_growth / gross_growth).shift(1)
    delta = equity_target - prior_weight

    # With one buy and one sell, solve for turnover when the post-trade
    # target weights are fractions of NAV after both fees have been paid.
    turnover = 2 * delta.abs() / (1 + fee_rate * np.sign(delta) * (2 * equity_target - 1))
    turnover.iloc[0] = 1 / (1 + fee_rate)
    net_return = (1 - fee_rate * turnover) * gross_growth - 1
    return pd.DataFrame({"return": net_return, "turnover": turnover})


def metrics(daily, annualization):
    returns = daily["return"]
    equity = (1 + returns).cumprod()
    vol = returns.std(ddof=1)
    values = {
        "cagr": equity.iloc[-1] ** (annualization / len(returns)) - 1,
        "ann_vol": vol * math.sqrt(annualization),
        "sharpe": returns.mean() / vol * math.sqrt(annualization) if vol > 0 else None,
        "max_drawdown": (equity / equity.cummax().clip(lower=1) - 1).min(),
        "total_return": equity.iloc[-1] - 1,
        "turnover": daily["turnover"].sum(),
    }
    return {key: float(value) if value is not None and math.isfinite(value) else None
            for key, value in values.items()}


def print_results(strategy, periods, performance, labels, entry_start):
    data_range = f"{entry_start.date()} to {periods['full'][-1].date()}"
    print(f"{strategy} | Opening marks: {data_range} | Returns dated by ending open")
    for period in ("is", "oos"):
        dates = periods[period]
        print(f"{period.upper()} return dates: {dates[0].date()} to {dates[-1].date()}")
    print(f"{'Portfolio':<22} {'Period':<6} {'CAGR':>9} {'Ann.Vol':>9} {'Sharpe':>8} "
          f"{'Max.DD':>9} {'Return':>9} {'Turnover':>9}")
    formats = {"cagr": ".2%", "ann_vol": ".2%", "sharpe": ".2f", "max_drawdown": ".2%",
               "total_return": ".2%", "turnover": ".2f"}
    for portfolio, segments in performance.items():
        for period, values in segments.items():
            cagr, vol, sharpe, drawdown, total, turnover = (
                format(values[key], spec) if values[key] is not None else "n/a"
                for key, spec in formats.items()
            )
            print(f"{labels[portfolio]:<22} {period.upper():<6} {cagr:>9} {vol:>9} {sharpe:>8} "
                  f"{drawdown:>9} {total:>9} {turnover:>9}")

    threshold = performance["threshold"]
    for period, values in threshold.items():
        print(f"{period.upper()} crossings: {values['n_crossings']} | "
              f"Calendar skips: {values['n_skipped_crossings']} | Tilts: {values['n_tilt_sessions']}")

    result = {"strategy": strategy, "data_range": data_range, **threshold,
              "benchmark": {key: performance[key] for key in labels if key != "threshold"}}
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


def main():
    with (ROOT / "config.yaml").open() as config_file:
        config = yaml.safe_load(config_file)

    tickers = [config["universe"][asset] for asset in ("equity", "bonds")]
    target, band, tilt = (config[key] for key in ("target_equity_weight", "drift_band", "equity_tilt"))
    exclude_first = config["exclude_month_start_sessions"]
    exclude_last = config["exclude_month_end_sessions"]
    fee_rate = config["transaction_cost_bps"] / 10_000
    annualization = config["annualization_sessions"]
    warmup_start = pd.Timestamp(config["warmup_start"])
    is_start, is_end = (pd.Timestamp(config["in_sample"][edge]) for edge in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][edge]) for edge in ("start", "end"))
    if not (
        warmup_start.to_period("M") < is_start.to_period("M")
        and is_start < is_end < oos_start <= oos_end
    ):
        raise ValueError("Provide a warm-up month and ordered, non-overlapping sample dates.")
    max_shift = min(target, 1 - target)
    if len(set(tickers)) != 2 or not (0 < band < max_shift and 0 < tilt <= max_shift):
        raise ValueError("Use two distinct ETFs and positive band and tilt values within the weight bounds.")
    if not (0 <= fee_rate < 1 and annualization > 0 and exclude_first >= 0 and exclude_last >= 0):
        raise ValueError("Invalid transaction cost, annualization or calendar exclusions.")

    prices = load_prices(tickers, warmup_start, oos_end)
    blocked = month_edge_mask(prices.index, exclude_first, exclude_last)
    close_returns = prices["Close"].pct_change().iloc[1:]
    crossings = threshold_crossings(close_returns, target, band).reindex(prices.index, fill_value=0)
    eligible_crossings = crossings.mask(blocked, 0)
    relative_return = close_returns[tickers[0]] - close_returns[tickers[1]]
    reversal = np.sign(relative_return).reindex(prices.index, fill_value=0).mask(blocked, 0)
    # Close-t observations first become positions at the next session's open.
    targets = {
        "threshold": target - tilt * eligible_crossings.shift(1, fill_value=0),
        "constant_weight": pd.Series(target, index=prices.index),
        "one_day_reversal": target - tilt * reversal.shift(1, fill_value=0),
    }

    opens = prices["Open"].loc[is_start:oos_end]
    entry_dates, exit_dates = opens.index[:-1], opens.index[1:]
    periods = {"is": exit_dates[exit_dates <= is_end], "oos": exit_dates[exit_dates >= oos_start],
               "full": exit_dates}
    if any(dates.empty for dates in periods.values()):
        raise ValueError("Each period requires at least one completed open-to-open return.")
    open_growth = opens.shift(-1).div(opens).loc[entry_dates]

    performance = {}
    for name, weights in targets.items():
        daily = simulate(open_growth, weights.loc[entry_dates], fee_rate).set_axis(exit_dates)
        performance[name] = {period: metrics(daily.loc[dates], annualization)
                             for period, dates in periods.items()}

    realized_crossings = crossings.shift(1, fill_value=0).loc[entry_dates].set_axis(exit_dates)
    realized_tilts = eligible_crossings.shift(1, fill_value=0).loc[entry_dates].set_axis(exit_dates)
    for period, dates in periods.items():
        n_crossings = int(realized_crossings.loc[dates].ne(0).sum())
        n_tilts = int(realized_tilts.loc[dates].ne(0).sum())
        performance["threshold"][period].update(n_crossings=n_crossings,
                                                n_skipped_crossings=n_crossings - n_tilts,
                                                n_tilt_sessions=n_tilts)

    labels = {"threshold": "Threshold reversal",
              "constant_weight": f"{100 * target:g}/{100 * (1 - target):g} constant",
              "one_day_reversal": "One-day reversal"}
    print_results(config["strategy"], periods, performance, labels, entry_dates[0])


if __name__ == "__main__":
    main()
