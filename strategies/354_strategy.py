"""354_strategy
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/354_strategy.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit da9754ee51e8d74f175ab50e14ef28c47c764ad1 on 2026-11-27 12:03 Europe/Oslo, exit 0, output sha256 a7ad4eefde1ceca30eabca038ca160c06acdf0b13cf9054a1b257dce144d6306
Data source: Free Yahoo Finance daily ETF prices via yfinance. GLD is the strategy asset and buy-and-hold benchmark. BIL daily returns are used only for excess Sharpe. The script reads a validated cache when available and otherwise downloads prices. The run reports a data fetch timestamp of 2026-11-27T05:41:37+00:00.
Data fetched: 2026-11-27T11:03:59+00:00
Sample period: 2007-06-01 through 2026-11-25 inclusive, on New York exchange dates. One preceding session's close is loaded to calculate the first sample day's return.   data_range (from output): {'start': '2007-06-01', 'end': '2026-11-25'}
IS/OOS split: IS runs from 2007-06-01 through 2025-01-28. OOS runs from 2025-01-29 through 2026-11-25. The source publication date is documented as 2025-01-28. The split slices a continuous portfolio path with fixed parameters. IS is retrospective, and OOS is a historical simulation.
Costs and slippage: The model applies a fee of 1 bp and fixed slippage of 2 bps per side on executed notional. Combined friction f = 0.0003 multiplies gross daily wealth by 1/(1+f) on an entry and by 1-f on an exit. The GLD buy-and-hold benchmark incurs the same friction on its initial buy.
Signal lag: The signal at session t's close is determined from the exchange calendar and filled at the next session's close. The strategy usually signals Wednesday for Thursday's buy and Thursday for Friday's sell. It skips weeks without both Thursday and Friday sessions. Close-to-close returns accrue to the position held before the day's fill.
Price adjustment: yfinance auto_adjust=True, adjusted close total-return basis
Survivorship bias: GLD and BIL were selected retrospectively and survived throughout the study period. The single-asset design avoids historical constituent backfilling. Asset-selection and survivorship bias remain.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.01786254925491959, ann_vol=0.07042201445664334, sharpe=0.2866284820676395, sharpe_excess=0.12239962412021266, max_drawdown=-0.19438860373030287, total_return=0.36646116397209605, turnover=88.29315573708453, n_trades=1730
    oos: cagr=-0.05854899601158403, ann_vol=0.111744818628293, sharpe=-0.4825618692258482, sharpe_excess=-0.8267343604533957, max_drawdown=-0.12993934698103926, total_return=-0.10428376365357384, turnover=84.72831554235637, n_trades=172
    full: cagr=0.010438970679190307, ann_vol=0.07526493659475457, sharpe=0.1758253313432972, sharpe_excess=-0.010891555037427154, max_drawdown=-0.21994732751992763, total_return=0.22396145090664255, turnover=87.95877023757906, n_trades=1902
    benchmark  full: cagr=0.0813681527102732, ann_vol=0.1749200557364572, sharpe=0.5349418444930027, sharpe_excess=0.45481140413551085, max_drawdown=-0.44032139611969257, total_return=3.582859635658579, turnover=0.046868981550691184, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "354_ strategy"\nuniverse:\n  asset: GLD\n  risk_free: BIL\ndates: # Inclusive sample endpoints, in New York exchange dates\n  start: "2007-06-01"\n  end: "2026-11-25"\n  oos_start: "2025-01-29"\nexecution:\n  cost_bps: 1.0 # Per side, on executed notional\n  slippage_bps: 2.0 # Per side, on executed notional\n  signal_lag_sessions: 1 # Minimum sessions from the calendar signal to the entry close\n  cash_buffer: 0.10 # Fraction reserved at the sizing close, before any fill-price move\n'
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
"""Calendar-only GLD Friday seasonality research backtest."""

import json
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

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


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
TIMESTAMP_PREFIX = "# DATA_FETCHED_AT: "
FIRST_POST_PUBLICATION_SIGNAL = pd.Timestamp("2025-01-29")
JUNETEENTH = Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday)
# New York dates when the announcements were certainly public by the start of the day.
SPECIAL_CLOSURES = {
    "2012-10-29": "2012-10-29",
    "2012-10-30": "2012-10-30",
    "2018-12-05": "2018-12-02",
    "2025-01-09": "2024-12-31",
}


class NyseHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def nyse_calendar(start, end):
    holidays = NyseHolidays().holidays(start, end)
    regular_sessions = pd.date_range(start, end, freq="B").difference(holidays).rename("Date")
    known_from = {pd.Timestamp(day): pd.Timestamp(public) for day, public in SPECIAL_CLOSURES.items()}
    for day in JUNETEENTH.dates(start, end):
        known_from[day] = pd.Timestamp("2021-12-28")
    return regular_sessions, known_from


def validate_prices(prices, tickers, sessions):
    if list(prices.columns) != tickers:
        raise ValueError(f"Expected price columns {tickers}, got {list(prices.columns)}")
    if not prices.index.equals(sessions):
        missing = sessions.difference(prices.index).strftime("%Y-%m-%d").tolist()
        extra = prices.index.difference(sessions).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Unexpected session coverage (missing={missing[:10]}, extra={extra[:10]})")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Adjusted-close data contain a missing or invalid price")


def adjusted_closes(tickers, sessions):
    CACHE.mkdir(exist_ok=True)
    start, end = sessions[0], sessions[-1]
    stem = f"{'_'.join(tickers)}_{start.date()}_{end.date()}_adjusted_close"
    path = CACHE / f"{stem}.csv"
    if path.exists():
        try:
            payload = path.read_bytes()
            with BytesIO(payload) as cached:
                header = cached.readline().decode("utf-8").strip()
                if not header.startswith(TIMESTAMP_PREFIX):
                    raise ValueError("Missing cache timestamp")
                fetched_at = header.removeprefix(TIMESTAMP_PREFIX)
                if datetime.fromisoformat(fetched_at).utcoffset() is None:
                    raise ValueError("Cache timestamp must have a time zone")
                prices = pd.read_csv(cached, index_col="Date", parse_dates=["Date"])
            validate_prices(prices, tickers, sessions)
            return prices, fetched_at, sha256(payload).hexdigest()
        except (OSError, ValueError) as exc:
            print(f"Refreshing price cache: {exc}")

    yf.set_tz_cache_location(str(CACHE / "yfinance"))
    bars = yf.download(
        tickers,
        start=start.date().isoformat(),
        end=(end + pd.Timedelta(days=1)).date().isoformat(),
        interval="1d",
        auto_adjust=True,
        ignore_tz=True,
        progress=False,
        threads=False,
        timeout=15,
    )
    if bars.empty:
        raise RuntimeError(f"Yahoo Finance returned no daily bars for {', '.join(tickers)}")
    prices = bars["Close"].reindex(columns=tickers).sort_index().rename_axis("Date")
    validate_prices(prices, tickers, sessions)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with TemporaryDirectory(dir=CACHE) as directory:
        temp_path = Path(directory) / path.name
        with temp_path.open("w", encoding="utf-8") as temporary:
            temporary.write(f"{TIMESTAMP_PREFIX}{fetched_at}\n")
            prices.to_csv(temporary)
        digest = sha256(temp_path.read_bytes()).hexdigest()
        temp_path.replace(path)
    return prices, fetched_at, digest


def friday_orders(regular_sessions, known_from, realized_sessions, sample, oos_start, lag):
    orders = pd.Series(0, index=sample, dtype=int)
    planned_entries = {}
    exit_due = None
    next_day = pd.Timedelta(days=1)
    calendar_updates = pd.DatetimeIndex(sorted({regular_sessions[0], *known_from.values()}))
    calendars = [
        regular_sessions.difference([day for day, public in known_from.items() if public <= update])
        for update in calendar_updates
    ]
    for session_index, session in enumerate(realized_sessions):
        if session > sample[-1]:
            break
        known_sessions = calendars[calendar_updates.searchsorted(session, side="right") - 1]
        if exit_due is not None and session >= exit_due:
            # If Friday closes unexpectedly, carry to the next actual session.
            orders.at[session] = -1
            exit_due = None
        elif exit_due is None and session in planned_entries:
            signal_index = planned_entries.pop(session)
            friday = session + next_day
            if session_index - signal_index >= lag and friday in known_sessions:
                orders.at[session] = 1
                exit_due = friday

        # At this close, plan the pair using only the schedule known as of this date.
        entry_index = int(known_sessions.searchsorted(session, side="right")) + lag - 1
        if entry_index + 1 >= len(known_sessions):
            continue
        entry, friday = known_sessions[entry_index : entry_index + 2]
        if entry.day_of_week != 3 or friday != entry + next_day or not sample[0] <= entry <= sample[-1]:
            continue
        if friday >= oos_start and session < FIRST_POST_PUBLICATION_SIGNAL:
            continue
        planned_entries[entry] = session_index
    return orders


def portfolio(asset_prices, orders, friction, cash_buffer):
    cash = previous_value = 1.0
    position = 0.0
    daily_results = []
    for (day, side), sizing_price in zip(orders.items(), asset_prices.iloc[:-1], strict=True):
        quantity = 0.0
        if side == 1:
            quantity = cash * (1 - cash_buffer) / (sizing_price * (1 + friction))
        elif side == -1:
            quantity = -position

        # The fill close is used only to settle the fixed quantity and mark value.
        fill_price = asset_prices.at[day]
        fill_amount = quantity * fill_price
        traded_amount = abs(fill_amount)
        value_before_fill = cash + position * fill_price
        cash -= fill_amount + traded_amount * friction
        if cash < 0:
            raise RuntimeError(f"{day.date()}: fixed fill quantity would leave negative cash")
        position += quantity
        value = cash + position * fill_price
        daily_results.append((value / previous_value - 1.0, traded_amount / value_before_fill, quantity))
        previous_value = value
    return pd.DataFrame(daily_results, index=orders.index, columns=["return", "turnover", "quantity"])


def sharpe(returns):
    vol = returns.std(ddof=1)
    return float(returns.mean() / vol * np.sqrt(SESSIONS_PER_YEAR)) if vol > 0 else None


def metrics(path, risk_free):
    returns = path["return"]
    wealth = (1.0 + returns).cumprod()
    years = len(returns) / SESSIONS_PER_YEAR
    return {
        "cagr": float(wealth.iloc[-1] ** (1 / years) - 1.0),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(returns - risk_free.loc[returns.index]),
        "max_drawdown": float((wealth / wealth.cummax().clip(lower=1.0) - 1.0).min()),
        "total_return": float(wealth.iloc[-1] - 1.0),
        "turnover": float(path["turnover"].sum() / years),
        "n_trades": int(path["quantity"].ne(0).sum()),
    }


def summarize(path, risk_free, oos_start):
    is_session = path.index < oos_start
    return {
        "is": metrics(path.loc[is_session], risk_free),
        "oos": metrics(path.loc[~is_session], risk_free),
        "full": metrics(path, risk_free),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    asset = config["universe"]["asset"]
    cash_proxy = config["universe"]["risk_free"]
    if (asset, cash_proxy) != ("GLD", "BIL"):
        raise ValueError("C018 uses GLD as its only asset and BIL as the excess-return reference")

    dates = config["dates"]
    start, end, oos_start = (pd.Timestamp(dates[name]) for name in ("start", "end", "oos_start"))
    today = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
    if not pd.Timestamp("2007-05-29") <= start < oos_start <= end < today:
        raise ValueError("Require 2007-05-29 <= start < oos_start <= end before today's NY date")
    if oos_start < FIRST_POST_PUBLICATION_SIGNAL:
        raise ValueError("OOS must start after publication of the source on 2025-01-28")

    execution = config["execution"]
    lag = execution["signal_lag_sessions"]
    if type(lag) is not int or lag < 1:
        raise ValueError("signal_lag_sessions must be a positive integer")
    cost_bps = float(execution["cost_bps"])
    slippage_bps = float(execution["slippage_bps"])
    friction = (cost_bps + slippage_bps) / 10_000
    if not (cost_bps >= 0 and slippage_bps >= 0 and friction < 1):
        raise ValueError("Cost and slippage must be nonnegative and sum to less than 10,000 bps")
    cash_buffer = float(execution["cash_buffer"])
    if not 0 < cash_buffer < 1:
        raise ValueError("cash_buffer must be a fraction between 0 and 1")

    regular_sessions, known_from = nyse_calendar(pd.Timestamp("2007-05-25"), end + pd.Timedelta(days=7))
    sessions = regular_sessions.difference(list(known_from))
    sample = sessions[(sessions >= start) & (sessions <= end)]
    if (sample < oos_start).sum() < 2 or (sample >= oos_start).sum() < 2:
        raise ValueError("IS and OOS each need at least two trading sessions")
    first_index = sessions.get_loc(sample[0])
    last_index = sessions.get_loc(sample[-1])

    orders = friday_orders(regular_sessions, known_from, sessions, sample, oos_start, lag)
    benchmark_orders = pd.Series(0, index=sample, dtype=int)
    benchmark_orders.iloc[0] = 1
    prices, fetched_at, data_digest = adjusted_closes([asset, cash_proxy], sessions[first_index - 1 : last_index + 1])
    risk_free = prices[cash_proxy].pct_change(fill_method=None).loc[sample]
    strategy_periods = summarize(portfolio(prices[asset], orders, friction, cash_buffer), risk_free, oos_start)
    benchmark_periods = summarize(portfolio(prices[asset], benchmark_orders, friction, cash_buffer), risk_free, oos_start)
    print("PARAM_COMBOS_TRIED: 1")
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"DATA_SHA256: {data_digest}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, adjusted close total-return basis")
    print(f"BENCHMARK: {asset} buy-and-hold with the same entry sizing and friction")
    print(f"EXCESS_SHARPE_REFERENCE: {cash_proxy} adjusted-close daily total returns")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(strategy_periods[period], allow_nan=False)}")
        print(f"BENCHMARK_{period.upper()}: {json.dumps(benchmark_periods[period], allow_nan=False)}")
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": sample[0].date().isoformat(), "end": sample[-1].date().isoformat()},
        **strategy_periods,
        "benchmark": benchmark_periods,
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
