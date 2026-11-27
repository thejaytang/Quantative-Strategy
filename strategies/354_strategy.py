# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "354_ strategy"\nuniverse:\n  asset: GLD\n  risk_free: BIL\ndates: # Inclusive sample endpoints, in New York exchange dates\n  start: "2007-06-01"\n  end: "2026-11-25"\n  oos_start: "2025-01-29"\nexecution:\n  cost_bps: 1.0 # Per side, on executed notional\n  slippage_bps: 2.0 # Per side, on executed notional\n  signal_lag_sessions: 1\n'
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
    # Non-recurring closures in the supported sample since BIL's inception.
    closures = pd.to_datetime(["2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])
    holidays = NyseHolidays().holidays(start, end).union(closures)
    return pd.date_range(start, end, freq="B").difference(holidays).rename("Date")


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
            with path.open(encoding="utf-8") as cached:
                header = cached.readline().strip()
                if not header.startswith(TIMESTAMP_PREFIX):
                    raise ValueError("Missing cache timestamp")
                fetched_at = header.removeprefix(TIMESTAMP_PREFIX)
                if datetime.fromisoformat(fetched_at).utcoffset() is None:
                    raise ValueError("Cache timestamp must have a time zone")
                prices = pd.read_csv(cached, index_col="Date", parse_dates=["Date"])
            validate_prices(prices, tickers, sessions)
            return prices, fetched_at
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
        temp_path.replace(path)
    return prices, fetched_at


def execute_after_lag(sessions, fill_indices, sides, lag):
    # Assign signals to earlier closes using only the exchange calendar.
    signals = pd.Series(0, index=sessions, dtype=int)
    signals.iloc[fill_indices - lag] = sides
    return signals.shift(lag, fill_value=0)


def friday_orders(sessions, first_session, lag):
    candidate_entries = sessions[:-1]
    next_is_friday = sessions[1:] == candidate_entries + pd.Timedelta(days=1)
    entries = np.flatnonzero(
        (candidate_entries >= first_session) & (candidate_entries.day_of_week == 3) & next_is_friday
    )
    fill_indices = np.concatenate((entries, entries + 1))
    sides = np.concatenate((np.ones(len(entries), dtype=int), -np.ones(len(entries), dtype=int)))
    return execute_after_lag(sessions, fill_indices, sides, lag)


def portfolio(asset_returns, orders, friction):
    # Close-to-close returns accrue to inventory held before today's fill.
    position = orders.cumsum().shift(1, fill_value=0)
    gross_factor = 1.0 + asset_returns * position
    friction_factor = np.select([orders.eq(1), orders.eq(-1)], [1 / (1 + friction), 1 - friction], default=1.0)
    traded_fraction = orders.abs().where(orders.ne(1), 1 / (1 + friction))
    return pd.DataFrame({"return": gross_factor * friction_factor - 1.0, "turnover": traded_fraction, "order": orders})


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
        "n_trades": int(path["order"].ne(0).sum()),
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
    if oos_start < pd.Timestamp("2025-01-29"):
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

    calendar_start = start - pd.offsets.BDay(2 * lag + 5)
    sessions = nyse_sessions(calendar_start, end + pd.Timedelta(days=7))
    sample = sessions[(sessions >= start) & (sessions <= end)]
    if (sample < oos_start).sum() < 2 or (sample >= oos_start).sum() < 2:
        raise ValueError("IS and OOS each need at least two trading sessions")
    first_index = sessions.get_loc(sample[0])
    last_index = sessions.get_loc(sample[-1])

    orders = friday_orders(sessions, sample[0], lag).loc[sample]
    benchmark_orders = execute_after_lag(sessions, np.array([first_index]), [1], lag).loc[sample]
    prices, fetched_at = adjusted_closes([asset, cash_proxy], sessions[first_index - 1 : last_index + 1])
    daily = prices.pct_change(fill_method=None).loc[sample]

    strategy_periods = summarize(portfolio(daily[asset], orders, friction), daily[cash_proxy], oos_start)
    benchmark_periods = summarize(
        portfolio(daily[asset], benchmark_orders, friction), daily[cash_proxy], oos_start
    )
    print("PARAM_COMBOS_TRIED: 1")
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, adjusted close total-return basis")
    print(f"BENCHMARK: {asset} buy-and-hold with the same entry friction")
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
