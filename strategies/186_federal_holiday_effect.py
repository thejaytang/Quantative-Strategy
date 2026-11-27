# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 186_federal_holiday_effect\nuniverse:\n  small_cap: IWM\n  large_cap: SPY\n  cash_proxy: BIL\n\n# Inclusive calendar dates. The OOS start follows the paper's online publication.\nin_sample:\n  start: 2008-01-01\n  end: 2025-09-30\nout_of_sample:\n  start: 2025-10-01\n  end: 2026-11-24\n\nholidays:\n  - mlk_day\n  - presidents_day\n  - memorial_day\n  - juneteenth\n  - independence_day\n  - labor_day\n  - thanksgiving\n\n# Preserve the original 2 bp all-in assumption, with a separate slippage allowance.\ncost_bps_per_side: 1.0\nslippage_bps_per_side: 1.0\ncontrol_lookback_weeks: 4\nsessions_per_year: 252\n"
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
"""Long-only, open-to-close US equity holiday backtest."""

import json
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from pandas.tseries.holiday import (
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
from scipy.stats import ttest_1samp


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
DAY = pd.Timedelta(days=1)
OSLO = ZoneInfo("Europe/Oslo")
HOLIDAY_RULES = {
    "new_year": Holiday("New Year", month=1, day=1, observance=sunday_to_monday),
    "mlk_day": USMartinLutherKingJr,
    "presidents_day": USPresidentsDay,
    "good_friday": GoodFriday,
    "memorial_day": USMemorialDay,
    "juneteenth": Holiday(
        "Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday
    ),
    "independence_day": Holiday(
        "Independence Day", month=7, day=4, observance=nearest_workday
    ),
    "labor_day": USLaborDay,
    "thanksgiving": USThanksgivingDay,
    "christmas": Holiday("Christmas", month=12, day=25, observance=nearest_workday),
}
ONE_OFF_CLOSURES = pd.DatetimeIndex([
    "2007-01-02",  # Ford national day of mourning, in the padded calendar
    "2012-10-29", "2012-10-30",  # Hurricane Sandy
    "2018-12-05", "2025-01-09",  # Bush and Carter national days of mourning
])


def exchange_sessions(start, end, selected_holidays):
    """Build the scheduled calendar and lag its signals on actual sessions."""
    first = pd.Timestamp(start.year - 1, 1, 1)
    last = pd.Timestamp(end.year + 1, 12, 31)
    holidays = {key: rule.dates(first, last) for key, rule in HOLIDAY_RULES.items()}
    closed = pd.DatetimeIndex([date for dates in holidays.values() for date in dates])
    closed = closed[closed.day_of_week < 5]  # A Saturday New Year has no weekday observance.
    planned = pd.bdate_range(first, last).difference(closed)
    actual = planned.difference(ONE_OFF_CLOSURES)

    selected = pd.DatetimeIndex([date for key in selected_holidays for date in holidays[key]])
    selected = selected.intersection(closed)
    positions = planned.searchsorted(selected)
    positions = positions[(positions > 0) & (positions < len(planned))]
    before, after = planned[positions - 1], planned[positions]

    # A signal issued at a scheduled close carries the intended next session date.
    # Lag on actual sessions and discard a signal if a one-off closure made it stale.
    next_planned = planned.to_series().shift(-1)
    target_at_open = next_planned.reindex(actual).shift(1)
    on_schedule = target_at_open.eq(actual.to_series())
    sessions = pd.DataFrame(index=actual)
    sessions["pre"] = actual.isin(before) & on_schedule
    sessions["post"] = actual.isin(after) & on_schedule

    july_3 = actual[(actual.month == 7) & (actual.day == 3)]
    early_july_3 = july_3[
        july_3.day_of_week.isin([0, 1, 3]) | ((july_3.day_of_week == 2) & (july_3.year >= 2013))
    ]
    early_july_5 = actual[
        (actual.month == 7) & (actual.day == 5) & (actual.day_of_week == 4) & (actual.year < 2013)
    ]
    christmas_eves = actual[(actual.month == 12) & (actual.day == 24)]
    early = early_july_3.union(early_july_5).union(christmas_eves).union(holidays["thanksgiving"] + DAY)
    sessions["early_close"] = actual.isin(early)

    closure_weeks = closed.union(ONE_OFF_CLOSURES).to_period("W-SUN")
    sessions["control"] = ~(
        actual.to_period("W-SUN").isin(closure_weeks)
        | sessions[["pre", "post", "early_close"]].any(axis=1)
    )
    return sessions.loc[start:end]


def check_prices(prices, ticker, expected):
    if not prices.index.equals(expected):
        missing = expected.difference(prices.index).strftime("%Y-%m-%d")[:5].tolist()
        extra = prices.index.difference(expected).strftime("%Y-%m-%d")[:5].tolist()
        raise ValueError(f"{ticker} session mismatch. Missing dates {missing}, extra dates {extra}.")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{ticker} has a missing or invalid open/close.")
    return prices


def load_prices(ticker, start, end, expected):
    path = CACHE / f"{ticker}_{start.date()}_{end.date()}_adjusted.csv"
    columns = ["Open", "Close"]
    if path.is_file():
        try:
            cached = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
            prices = check_prices(cached[columns], ticker, expected)
            fetched_at = datetime.fromtimestamp(path.stat().st_mtime, OSLO)
            return prices, fetched_at, "cache"
        except (KeyError, ValueError, OSError):
            pass  # Replace an invalid cache only after validating the fresh download.

    history = yf.Ticker(ticker).history(
        start=start.date().isoformat(), end=(end + DAY).date().isoformat(),
        interval="1d", auto_adjust=True, actions=False, timeout=15,
    )
    if history.empty:
        raise RuntimeError(f"Yahoo Finance returned no daily prices for {ticker}.")
    prices = history[columns].copy()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    prices = check_prices(prices.sort_index(), ticker, expected)
    with NamedTemporaryFile(mode="w", dir=CACHE, suffix=".tmp", delete=False) as tmp:
        temporary_path = Path(tmp.name)
        try:
            prices.to_csv(tmp, index_label="Date")
            tmp.flush()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    fetched_at = datetime.fromtimestamp(path.stat().st_mtime, OSLO)
    return prices, fetched_at, "download"


def rotation_returns(equity, cash, active, cost):
    """BIL overnight, with full switches at both ends of an active session."""
    overnight = cash["Open"] / cash["Close"].shift()
    overnight.iloc[0] = 1.0  # Each evaluation starts flat at its first open.
    cash_intraday = cash["Close"] / cash["Open"]
    equity_intraday = equity["Close"] / equity["Open"]
    growth = overnight * cash_intraday.where(~active, equity_intraday)

    # Each switch sells one ETF and buys another. Start and end flat.
    buy, sell = 1 / (1 + cost), 1 - cost
    growth.loc[active] *= (buy * sell) ** 2
    growth.iloc[0] *= 1 / sell if active.iloc[0] else buy
    growth.iloc[-1] *= 1 / buy if active.iloc[-1] else sell
    return growth - 1


def hold_returns(prices, cost=0.0):
    previous_mark = prices["Close"].shift()
    previous_mark.iloc[0] = prices["Open"].iloc[0]
    growth = prices["Close"] / previous_mark
    growth.iloc[0] /= 1 + cost
    growth.iloc[-1] *= 1 - cost
    return growth - 1


def rotation_trades(active):
    # Include opening and liquidating the portfolio. At an active boundary the
    # portfolio switches directly between USD and equity, skipping the BIL leg.
    return 2 + 2 * int(active.sum()) - int(active.iloc[0]) - int(active.iloc[-1])


def metrics(returns, cash_returns, trades, sessions_per_year):
    nav = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    excess = returns - cash_returns
    scale = np.sqrt(sessions_per_year)
    volatility, excess_volatility = returns.std(ddof=1), excess.std(ddof=1)
    sharpe_excess = excess.mean() / excess_volatility * scale if excess_volatility > 0 else np.nan
    results = {
        "cagr": nav[-1] ** (sessions_per_year / len(returns)) - 1,
        "ann_vol": volatility * scale,
        "sharpe": sharpe_excess,
        "sharpe_excess": sharpe_excess,
        "sharpe_zero": returns.mean() / volatility * scale if volatility > 0 else np.nan,
        "max_drawdown": (nav / np.maximum.accumulate(nav) - 1).min(),
        "total_return": nav[-1] - 1,
        "turnover": trades * sessions_per_year / len(returns),
    }
    return {key: float(value) if np.isfinite(value) else None for key, value in results.items()} | {
        "n_trades": int(trades), "n_sessions": len(returns),
    }


def event_comparisons(period, sessions, small, large, cash, lookback):
    small_return = small["Close"] / small["Open"] - 1
    large_return = large["Close"] / large["Open"] - 1
    cash_return = cash["Close"] / cash["Open"] - 1
    control = small_return.loc[sessions["control"]]
    matched = pd.Series(np.nan, index=sessions.index)
    for weeks in range(1, lookback + 1):
        matched = matched.fillna(control.shift(weeks, freq="7D"))

    active = sessions["pre"] | sessions["post"]
    early = sessions["early_close"]
    groups = {
        "All events": active,
        "Pre-holiday": sessions["pre"], "Post-holiday": sessions["post"],
        "Regular hours": active & ~early, "Early close": active & early,
    }
    rows = []
    for label, mask in groups.items():
        cash_excess = (small_return - cash_return).loc[mask]
        t_stat, p_value = np.nan, np.nan
        if len(cash_excess) > 1 and cash_excess.std(ddof=1) > 0:
            t_stat, p_value = ttest_1samp(cash_excess, popmean=0.0)
        rows.append({
            "Period": period, "Event group": label, "Events": len(cash_excess),
            "Small - cash": cash_excess.mean(), "t-stat": t_stat, "p-value": p_value,
            "Matched": int(matched.loc[mask].count()),
            "Small cap": small_return.loc[mask].mean(), "Large cap": large_return.loc[mask].mean(),
            "Small - large": (small_return - large_return).loc[mask].mean(),
            "Weekday control": matched.loc[mask].mean(),
            "Small - control": (small_return - matched).loc[mask].mean(),
        })
    return rows


def print_table(rows, percentages):
    data = pd.DataFrame(rows)
    formatters = {column: "{:.2%}".format for column in percentages}
    if "p-value" in data:
        formatters["p-value"] = "{:.4g}".format
    print(data.to_string(index=False, na_rep="—", formatters=formatters, float_format="{:.3f}".format))


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    in_sample, out_of_sample = (
        tuple(pd.Timestamp(config[key][edge]) for edge in ("start", "end"))
        for key in ("in_sample", "out_of_sample")
    )
    start, end = in_sample[0], out_of_sample[1]
    if start.year < 2008 or not start <= in_sample[1] < out_of_sample[0] <= end:
        raise ValueError("Use ordered, nonoverlapping IS/OOS dates starting in 2008 or later.")
    fee_bps = float(config["cost_bps_per_side"])
    slippage_bps = float(config["slippage_bps_per_side"])
    cost = (fee_bps + slippage_bps) / 10_000
    annual = int(config["sessions_per_year"])
    lookback = int(config["control_lookback_weeks"])
    if not (fee_bps >= 0 and slippage_bps >= 0 and cost < 1) or annual < 1 or lookback < 1:
        raise ValueError("Invalid trading cost, annualization, or control lookback.")
    sessions = exchange_sessions(start, end, config["holidays"])
    if any(sessions.loc[left:right].empty for left, right in (in_sample, out_of_sample)):
        raise ValueError("Both evaluation periods need at least one trading session.")

    CACHE.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(CACHE / "yfinance"))
    symbols = [config["universe"][key] for key in ("small_cap", "large_cap", "cash_proxy")]
    inputs = [load_prices(ticker, start, end, sessions.index) for ticker in symbols]
    small_prices, large_prices, cash_prices = (prices for prices, _, _ in inputs)

    event_masks = {
        "holidays": sessions["pre"] | sessions["post"],
        "pre-holiday only": sessions["pre"],
        "post-holiday only": sessions["post"],
        "holidays (regular only)": (sessions["pre"] | sessions["post"]) & ~sessions["early_close"],
    }

    period_metrics, benchmark, performance_rows, event_rows = {}, {}, [], []
    for period, (left, right) in {"is": in_sample, "oos": out_of_sample, "full": (start, end)}.items():
        window = sessions.loc[left:right]
        small, large, cash = (prices.loc[window.index] for prices in (small_prices, large_prices, cash_prices))
        cash_returns = hold_returns(cash)
        rotations = {}
        for label, mask in event_masks.items():
            active = mask.loc[window.index]
            rotations[label] = metrics(rotation_returns(small, cash, active, cost),
                                       cash_returns, rotation_trades(active), annual)

        active = event_masks["holidays"].loc[window.index]
        hold_metrics = metrics(hold_returns(large, cost), cash_returns, 2, annual)
        period_metrics[period] = {**rotations["holidays"], "n_events": int(active.sum())}
        benchmark[period] = hold_metrics

        portfolios = [(f"{symbols[0]} {label}", row) for label, row in rotations.items()]
        portfolios.extend([
            (f"{symbols[1]} buy & hold", hold_metrics),
            (f"{symbols[1]} holidays", metrics(rotation_returns(large, cash, active, cost),
                                              cash_returns, rotation_trades(active), annual)),
            (f"{symbols[2]} buy & hold", metrics(hold_returns(cash, cost), cash_returns, 2, annual)),
        ])
        for label, row in portfolios:
            display_metrics = {key: value for key, value in row.items() if key != "sharpe"}
            performance_rows.append({"Period": period.upper(), "Portfolio": label, **display_metrics})
        event_rows.extend(event_comparisons(period.upper(), window, small, large, cash, lookback))

    data_range = f"{sessions.index[0].date()} to {sessions.index[-1].date()}"
    print(f"{config['strategy']} | {data_range} | {fee_bps:g} bp fee + {slippage_bps:g} bp slippage per ETF side")
    print(f"PARAM_COMBOS_TRIED: {len(event_masks)}")
    print(f"DATA_FETCHED_AT: {max(fetched_at for _, fetched_at, _ in inputs).isoformat(timespec='seconds')}")
    print("PRICE_ADJUSTMENT: Yahoo Finance dividend- and split-adjusted daily Open and Close (auto_adjust=True)")
    for symbol, (_, fetched_at, source) in zip(symbols, inputs, strict=True):
        print(f"Yahoo Finance {symbol} | {source} | {fetched_at.isoformat(timespec='seconds')}")
    print(f"\nNet portfolio performance | sharpe_excess uses gross {symbols[2]} daily returns | sharpe_zero uses zero")
    print_table(performance_rows, ["cagr", "ann_vol", "max_drawdown", "total_return", "turnover"])
    print(f"\nGross open-to-close event means | two-sided t-test on daily {symbols[0]} minus {symbols[2]}")
    print_table(event_rows, ["Small - cash", "Small cap", "Large cap", "Small - large", "Weekday control", "Small - control"])
    event_counts = " | ".join(f"{period.upper()} {row['n_events']}" for period, row in period_metrics.items())
    print(f"\nEvent counts | {event_counts}")
    print("Event p-values are descriptive and unadjusted; nearby holiday sessions may be dependent.")
    results = {
        "strategy": config["strategy"], "data_range": data_range, **period_metrics,
        "benchmark": {"ticker": symbols[1], **benchmark["full"], "is": benchmark["is"], "oos": benchmark["oos"]},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
