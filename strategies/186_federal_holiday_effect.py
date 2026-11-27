"""186_federal_holiday_effect
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/186_federal_holiday_effect.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit e0984d15ab251762c76186215f63849cd5effddf on 2026-11-27 11:34 Europe/Oslo, exit 0, output sha256 6aac571044f81542c4ee12e8c86aa60bb55379f4dd84860c38cc617b72594285
Data source: Public Yahoo Finance daily ETF Open and Close via yfinance for IWM (small cap), SPY (large cap), and BIL (cash proxy). The loader uses a valid CSV cache or downloads from Yahoo Finance on a cache miss or validation failure. The recorded run used the cache for all three ETFs. Its reported 2026-11-25T12:18:09+01:00 timestamp comes from the CSV modification times..
Data fetched: 2026-11-27T11:34:59+01:00
Sample period: Observed trading sessions span 2008-01-02 through 2026-11-24, with 4,755 sessions in the full window. data_range (from output): 2008-01-02 to 2026-11-24
IS/OOS split: Fixed, nonoverlapping calendar-date windows with inclusive endpoints: IS 2008-01-01 through 2025-09-30; OOS 2025-10-01 through 2026-11-24. The README gives 2025-09-16 as the paper's online publication date. The same rules run in IS, OOS, and the full window, with no parameter fitting documented..
Costs and slippage: 1 bp fee plus 1 bp fixed adverse slippage per ETF buy or sell (2 bp per side). With c = 0.0002, a buy multiplies portfolio growth by 1/(1+c) and a sell by 1-c. An interior event session has two ETF switches, each with one buy and one sell. Each evaluation starts and ends flat, so the first open and last close carry only the applicable entry or liquidation legs. The reported buy-and-hold portfolios pay entry and exit costs. Slippage is modeled as a flat allowance..
Signal lag: At session t's close, the rule uses the published holiday schedule to identify the intended next session. It simulates a switch into IWM at that session's open if the session is eligible and the intended date still matches. The exit is predetermined at entry and simulated at that session's close, including scheduled early closes. An unscheduled closure that causes the intended open to be missed invalidates the signal..
Price adjustment: Yahoo Finance dividend- and split-adjusted daily Open and Close (auto_adjust=True)
Survivorship bias: Uses historical prices for the fixed IWM, SPY, and BIL ETF universe, with no reconstruction from current stock constituent lists. The ETFs themselves are surviving products selected retrospectively..
Parameter combinations tried: 4
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.00082811685337425, ann_vol=0.0447347907971129, sharpe=-0.2578922709758068, sharpe_excess=-0.2578922709758068, max_drawdown=-0.17880211904795118, total_return=-0.01457165416472117, turnover=18.846220093059678, n_trades=557
    oos: cagr=0.06946553620470897, ann_vol=0.019581321263556162, sharpe=1.6003011320098892, sharpe_excess=1.6003011320098892, max_drawdown=-0.010117061912433112, total_return=0.08035120635586779, turnover=19.11281388394728, n_trades=37
    full: cagr=0.003356979677578531, ann_vol=0.043628642002345175, sharpe=-0.2039764296266909, sharpe_excess=-0.2039764296266909, max_drawdown=-0.17880211904795118, total_return=0.06527920182523772, turnover=18.80949298411327, n_trades=592
    benchmark: cagr=0.10779267158197237, ann_vol=0.19759623936324217, sharpe=0.5488870347264164, sharpe_excess=0.5488870347264164, max_drawdown=-0.5240353511420894, total_return=5.9006392443005, turnover=0.052988513565903374, n_trades=2
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 186_federal_holiday_effect\nuniverse:\n  small_cap: IWM\n  large_cap: SPY\n  cash_proxy: BIL\n\n# Inclusive calendar dates. The OOS start follows the paper\'s online publication.\nin_sample:\n  start: 2008-01-01\n  end: 2025-09-30\nout_of_sample:\n  start: 2025-10-01\n  end: 2026-11-24\n\nholidays:\n  - mlk_day\n  - presidents_day\n  - memorial_day\n  - juneteenth\n  - independence_day\n  - labor_day\n  - thanksgiving\n\n# Preserve the original 2 bp all-in assumption, with a separate slippage allowance.\ncost_bps_per_side: 1.0\nslippage_bps_per_side: 1.0\n# Fixed reserve as a share of portfolio value at the open before trading.\ncash_buffer_bps: 1.0\ncontrol_lookback_weeks: 4\nsessions_per_year: 252\n\n# Original CSV write time recorded in the archived run, tied to the bundled bytes.\nbundled_snapshot:\n  fetched_at: "2026-11-25T12:18:09+01:00"\n  sha256:\n    IWM: a77dda543a5faaee891980fef372ce2c1f80998429e07682c1a553cd817ceee0\n    SPY: 0ba108006930c6d1347bc73fff4ed488037597ec6d6ffaaf597fe531a975224c\n    BIL: abf3978caea0949ac79478b2cddf9077cdfba051f70a32560df97fa77899e2fe\n'
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
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from io import BytesIO
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


@dataclass
class PriceSnapshot:
    prices: pd.DataFrame
    fetched_at: datetime | None
    source: str
    digest: str


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


def load_prices(ticker, start, end, expected, bundled_snapshot):
    path = CACHE / f"{ticker}_{start.date()}_{end.date()}_adjusted.csv"
    columns = ["Open", "Close"]
    if path.is_file():
        try:
            contents = path.read_bytes()
            digest = sha256(contents).hexdigest()
            cached = pd.read_csv(BytesIO(contents), index_col="Date", parse_dates=["Date"])
            prices = check_prices(cached[columns], ticker, expected)
            fetched_at = None
            source = "cache"
            if digest == bundled_snapshot.get("sha256", {}).get(ticker):
                fetched_at = datetime.fromisoformat(bundled_snapshot["fetched_at"])
                source = "bundled cache (archived timestamp)"
            elif "FetchedAt" in cached:
                timestamp, = cached["FetchedAt"].unique()
                fetched_at = datetime.fromisoformat(timestamp)
                if fetched_at.tzinfo is None:
                    raise ValueError(f"{ticker} cache timestamp needs a timezone.")
            return PriceSnapshot(prices, fetched_at, source, digest)
        except (KeyError, TypeError, ValueError, OSError):
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
    fetched_at = datetime.now(OSLO).replace(microsecond=0)
    with NamedTemporaryFile(mode="w", dir=CACHE, suffix=".tmp", delete=False) as tmp:
        temporary_path = Path(tmp.name)
        try:
            prices.assign(FetchedAt=fetched_at.isoformat()).to_csv(tmp, index_label="Date")
            tmp.flush()
            digest = sha256(temporary_path.read_bytes()).hexdigest()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return PriceSnapshot(prices, fetched_at, "download", digest)


def simulate_positions(prices, open_targets, close_exits, cost, cash_buffer):
    """Size purchases at the open and sell existing shares at designated closes."""
    opens = np.column_stack([fund["Open"] for fund in prices])
    closes = np.column_stack([fund["Close"] for fund in prices])
    cash = previous_value = 1.0
    position = None
    shares = 0.0
    rows = []

    for i, (target, exit_at_close) in enumerate(zip(open_targets, close_exits, strict=True)):
        trades = 0
        turnover = 0.0
        if position != target:
            sale = shares * opens[i, position] if position is not None else 0.0
            opening_value = cash + sale
            cash += sale * (1 - cost)

            # A simultaneous opening swap can use its net sale proceeds. Leave a
            # cash reserve and cap the purchase at the amount actually available.
            reserve = opening_value * cash_buffer
            unit_cost = opens[i, target] * (1 + cost)
            requested = (opening_value - reserve) / unit_cost
            affordable = max(cash - reserve, 0.0) / unit_cost
            shares = np.nextafter(min(requested, affordable), 0.0)
            cash -= shares * unit_cost
            if cash < 0:
                raise ArithmeticError("Opening purchase exceeded available cash.")
            position = target if shares > 0 else None
            traded = sale + shares * opens[i, target]
            trades += int(traded > 0)
            turnover += traded / (2 * opening_value)

        invested_at_open = position is not None
        close_value = cash
        if invested_at_open:
            sale = shares * closes[i, position]
            close_value += sale
            if exit_at_close or i == len(opens) - 1:
                # The exit quantity is the position already established at the open.
                # Proceeds stay in cash until at least the next session's open.
                cash += sale * (1 - cost)
                turnover += sale / (2 * close_value)
                trades += 1
                position, shares = None, 0.0
                close_value = cash

        rows.append((close_value / previous_value - 1, turnover, trades, invested_at_open))
        previous_value = close_value

    return pd.DataFrame(rows, index=prices[0].index,
                        columns=["return", "turnover", "trades", "invested_at_open"])


def rotation_results(equity, treasury, active, cost, cash_buffer):
    return simulate_positions([equity, treasury], np.where(active, 0, 1), active, cost, cash_buffer)


def hold_results(prices, cost, cash_buffer):
    return simulate_positions([prices], np.zeros(len(prices), dtype=int),
                              np.zeros(len(prices), dtype=bool), cost, cash_buffer)


def cash_reference_returns(prices):
    previous_mark = prices["Close"].shift()
    previous_mark.iloc[0] = prices["Open"].iloc[0]
    return prices["Close"] / previous_mark - 1


def metrics(daily_results, cash_returns, sessions_per_year):
    returns = daily_results["return"]
    values = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    excess = returns - cash_returns
    scale = np.sqrt(sessions_per_year)
    volatility, excess_volatility = returns.std(ddof=1), excess.std(ddof=1)
    sharpe_excess = excess.mean() / excess_volatility * scale if excess_volatility > 0 else np.nan
    results = {
        "cagr": values[-1] ** (sessions_per_year / len(returns)) - 1,
        "ann_vol": volatility * scale,
        "sharpe": sharpe_excess,
        "sharpe_excess": sharpe_excess,
        "sharpe_zero": returns.mean() / volatility * scale if volatility > 0 else np.nan,
        "max_drawdown": (values / np.maximum.accumulate(values) - 1).min(),
        "total_return": values[-1] - 1,
        "turnover": daily_results["turnover"].mean() * sessions_per_year,
    }
    return {key: float(value) if np.isfinite(value) else None for key, value in results.items()} | {
        "n_trades": int(daily_results["trades"].sum()), "n_sessions": len(returns),
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
    buffer_bps = float(config["cash_buffer_bps"])
    cash_buffer = buffer_bps / 10_000
    annual = int(config["sessions_per_year"])
    lookback = int(config["control_lookback_weeks"])
    if not (fee_bps >= 0 and slippage_bps >= 0 and cost < 1) or annual < 1 or lookback < 1:
        raise ValueError("Invalid trading cost, annualization, or control lookback.")
    if not 0 < cash_buffer < 1:
        raise ValueError("Cash buffer must be between 0 and 10,000 bps, exclusive.")
    sessions = exchange_sessions(start, end, config["holidays"])
    if any(sessions.loc[left:right].empty for left, right in (in_sample, out_of_sample)):
        raise ValueError("Both evaluation periods need at least one trading session.")

    CACHE.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(CACHE / "yfinance"))
    symbols = [config["universe"][key] for key in ("small_cap", "large_cap", "cash_proxy")]
    bundled_snapshot = config.get("bundled_snapshot", {})
    inputs = [load_prices(ticker, start, end, sessions.index, bundled_snapshot) for ticker in symbols]
    small_prices, large_prices, cash_prices = (item.prices for item in inputs)

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
        cash_returns = cash_reference_returns(cash)
        rotation_days = {
            label: rotation_results(small, cash, mask.loc[window.index], cost, cash_buffer)
            for label, mask in event_masks.items()
        }
        rotations = {label: metrics(days, cash_returns, annual) for label, days in rotation_days.items()}
        active = event_masks["holidays"].loc[window.index]
        events = int((active & rotation_days["holidays"]["invested_at_open"]).sum())
        hold_metrics = metrics(hold_results(large, cost, cash_buffer), cash_returns, annual)
        period_metrics[period] = {**rotations["holidays"], "n_events": events}
        benchmark[period] = hold_metrics

        portfolios = [(f"{symbols[0]} {label}", row) for label, row in rotations.items()]
        portfolios.extend([
            (f"{symbols[1]} buy & hold", hold_metrics),
            (f"{symbols[1]} holidays", metrics(rotation_results(large, cash, active, cost, cash_buffer),
                                              cash_returns, annual)),
            (f"{symbols[2]} buy & hold", metrics(hold_results(cash, cost, cash_buffer), cash_returns, annual)),
        ])
        for label, row in portfolios:
            display_metrics = {key: value for key, value in row.items() if key != "sharpe"}
            performance_rows.append({"Period": period.upper(), "Portfolio": label, **display_metrics})
        event_rows.extend(event_comparisons(period.upper(), window, small, large, cash, lookback))

    data_range = f"{sessions.index[0].date()} to {sessions.index[-1].date()}"
    print(f"{config['strategy']} | {data_range} | {fee_bps:g} bp fee + {slippage_bps:g} bp slippage per ETF side")
    print(f"Opening cash reserve | {buffer_bps:g} bp of pre-trade portfolio value")
    print(f"PARAM_COMBOS_TRIED: {len(event_masks)}")
    timestamps = [item.fetched_at.isoformat(timespec="seconds") if item.fetched_at else "unknown" for item in inputs]
    latest = (max(item.fetched_at for item in inputs).isoformat(timespec="seconds")
              if all(item.fetched_at for item in inputs) else "unknown")
    print(f"DATA_FETCHED_AT: {latest}")
    print("PRICE_ADJUSTMENT: Yahoo Finance dividend- and split-adjusted daily Open and Close (auto_adjust=True)")
    for symbol, item, timestamp in zip(symbols, inputs, timestamps, strict=True):
        print(f"Yahoo Finance {symbol} | {item.source} | {timestamp} | SHA-256 {item.digest}")
    print(f"\nNet portfolio performance | sharpe_excess uses gross {symbols[2]} daily returns | sharpe_zero uses zero")
    print_table(performance_rows, ["cagr", "ann_vol", "max_drawdown", "total_return", "turnover"])
    print(f"\nGross open-to-close event means | two-sided t-test on daily {symbols[0]} minus {symbols[2]}")
    print_table(event_rows, ["Small - cash", "Small cap", "Large cap", "Small - large", "Weekday control", "Small - control"])
    event_counts = " | ".join(f"{period.upper()} {row['n_events']}" for period, row in period_metrics.items())
    print(f"\nEvent counts | {event_counts}")
    print("Event p-values are descriptive and unadjusted. Nearby holiday sessions may be dependent.")
    results = {
        "strategy": config["strategy"], "data_range": data_range, **period_metrics,
        "benchmark": {"ticker": symbols[1], **benchmark["full"], "is": benchmark["is"], "oos": benchmark["oos"]},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
