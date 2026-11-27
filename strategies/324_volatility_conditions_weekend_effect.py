"""324_volatility_conditions_weekend_effect
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/324_volatility_conditions_weekend_effect.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c71364ad99535bbf59be998b004cff431ee83048 on 2026-11-27 11:54 Europe/Oslo, exit 0, output sha256 6f6b9fe6a3c88e27433dfe9da7944934a14ae00dde6bb552d7fb4340a3b44ecf
Data source: Yahoo Finance adjusted daily closes via yfinance for U.S. equity ETFs SPY, IWM and QUAL, plus the VIX volatility index (^VIX) for the signal.
Data fetched: 2026-11-27T10:54:12+00:00
Sample period: Reported trading sessions 2016-01-04 through 2025-12-31. The configured data range starts 2015-01-01, with 2015 sessions used for warm-up.   data_range (from output): 2016-01-04 to 2025-12-31
IS/OOS split: Configured IS 2016-01-01 through 2023-12-31 and OOS 2024-01-01 through 2025-12-31, inclusive. Reported sessions are IS 2016-01-04 through 2023-12-29 and OOS 2024-01-02 through 2025-12-31. The strategy keeps each entry and exit within the same split and carries prior price history into the rolling rank.
Costs and slippage: 5 bps per side on absolute spread notional at entry and exit, with exit notional estimated from adjusted close price ratios. Short borrow costs 50 bps per year on initial short notional, prorated by calendar days held divided by 365. The SPY buy-and-hold benchmark pays one 5 bps buy cost at the start of each report period. The backtest has no explicit slippage model.
Signal lag: The ^VIX close at session t is ranked over the 252 sessions ending at t, using average ranks for ties. The rank is shifted by one trading session. If the next session is an eligible Friday and the rank is at or below 50%, the strategy enters at that Friday's adjusted close and exits at the immediately following eligible Monday's adjusted close.
Price adjustment: ETF fills: Yahoo Close (no dividend adjustment); distributions in cash; BIL: Adj Close
Survivorship bias: The universe is a fixed list of three ETF proxies. The README identifies selection bias from choosing ETFs that still trade today, and the implementation has no point-in-time universe or delisting treatment.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.022138521551471557, ann_vol=0.00828200682622171, sharpe=-2.698725253914078, sharpe_excess=-4.193843371886932, max_drawdown=-0.16408254355079377, total_return=-0.16367163737120038, turnover=363.6354280226407, n_trades=382
    oos: cagr=-0.02621953327310811, ann_vol=0.008867282753250005, sharpe=-2.997570026976813, sharpe_excess=-7.895909936502528, max_drawdown=-0.05164916195945701, total_return=-0.05164916195945701, turnover=72.3235364727716, n_trades=76
    full: cagr=-0.022931216357406248, ann_vol=0.008400994800869368, sharpe=-2.761030526304509, sharpe_excess=-4.94360006335486, max_drawdown=-0.20692029460749028, total_return=-0.2068672964239031, turnover=435.95896449541226, n_trades=458
    benchmark: cagr=0.13405726191909184, ann_vol=0.16161902971036532, sharpe=0.8607123876388185, sharpe_excess=0.735528618510076, max_drawdown=-0.3065070808035426, total_return=2.514289289648169, turnover=0.9367195204427511, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 324_volatility_conditions_weekend_effect\ncategory: seasonality\n\nuniverse:\n  market: SPY\n  small_cap: IWM\n  quality: QUAL\n  volatility: '^VIX'\n  cash_proxy: BIL\n\ndates:\n  data_start: '2015-01-01'\n  in_sample:\n    start: '2016-01-01'\n    end: '2023-12-31'\n  out_of_sample:\n    start: '2024-01-01'\n    end: '2025-12-31'\n\nsignal:\n  vix_rank_sessions: 252\n  max_vix_percentile: 0.50\n\nportfolio:\n  size_spread_gross: 0.50\n  quality_spread_gross: 0.50\n  cash_buffer_fraction: 0.05\n\ncosts:\n  commission_bps_per_side: 2.5\n  slippage_bps_per_side: 2.5\n  short_borrow_bps_per_year: 50.0\n"
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
"""C115 volatility-conditioned U.S. equity ETF weekend spreads."""

import json
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


ROOT = Path(__file__).resolve().parent
DAY = pd.Timedelta(days=1)
SESSIONS_PER_YEAR = 252
PARAM_COMBOS_TRIED = 1
PRICE_ADJUSTMENT = "ETF fills: Yahoo Close (no dividend adjustment); distributions in cash; BIL: Adj Close"
MOURNING_CLOSURES = [
    (pd.Timestamp("2018-12-05"), pd.Timestamp("2018-12-01")),
    (pd.Timestamp("2025-01-09"), pd.Timestamp("2024-12-30")),
]


class NyseHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def trading_calendar(start, end):
    scheduled = pd.bdate_range(start, end).difference(NyseHolidays().holidays(start, end))
    sessions = scheduled.difference([closed for closed, _ in MOURNING_CLOSURES])
    black_fridays = USThanksgivingDay.dates(start - DAY, end) + DAY
    july_eves = (scheduled.month == 7) & (scheduled.day == 3)
    christmas_eves = (scheduled.month == 12) & (scheduled.day == 24)
    regular = scheduled[~(scheduled.isin(black_fridays) | july_eves | christmas_eves)]
    return sessions, regular


def regular_as_of(dates, decision_dates, regular_sessions):
    eligible = dates.isin(regular_sessions)
    for closed, announced in MOURNING_CLOSURES:
        eligible &= ~((dates == closed) & (decision_dates >= announced))
    return eligible


def validate_quotes(frame, sessions, equity_tickers, vix, cash_proxy):
    columns = [("Close", ticker) for ticker in [*equity_tickers, vix]] + [("Adj Close", cash_proxy)]
    columns += [(field, ticker) for field in ("Dividends", "Capital Gains", "Stock Splits")
                for ticker in equity_tickers]
    quotes = frame.reindex(index=sessions, columns=pd.MultiIndex.from_tuples(columns)).astype(float)
    invalid = ~np.isfinite(quotes)
    price_columns = quotes.columns.get_level_values(0).isin(["Close", "Adj Close"])
    invalid.loc[:, price_columns] |= quotes.loc[:, price_columns].le(0)
    invalid.loc[:, ~price_columns] |= quotes.loc[:, ~price_columns].lt(0)
    if invalid.any().any():
        first_day = invalid.index[invalid.any(axis=1)][0]
        fields = ", ".join(f"{ticker} {field}" for field, ticker in invalid.columns[invalid.loc[first_day]])
        raise ValueError(f"Missing or invalid quotes on {first_day.date()} for {fields}")
    if quotes["Stock Splits"].ne(0).any().any():
        raise ValueError("ETF fills require a split-free sample for fixed share quantities")
    return quotes


def load_quotes(equity_tickers, vix, cash_proxy, start, end, sessions):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    tickers = [*equity_tickers, vix, cash_proxy]
    cache_key = "_".join(tickers).replace("^", "index-")
    cache_path = cache_dir / f"yahoo_closes_actions_{cache_key}_{start.date()}_{end.date()}.csv"
    date_prefix = "# DATA_FETCHED_AT: "
    if cache_path.is_file():
        try:
            with cache_path.open(encoding="utf-8") as handle:
                header = handle.readline()
                if not header.startswith(date_prefix):
                    raise ValueError("Missing download timestamp")
                fetched_at = pd.Timestamp(header.removeprefix(date_prefix).strip())
                if fetched_at.tzinfo is None:
                    raise ValueError("Download timestamp must include its time zone")
                cached = pd.read_csv(handle, header=[0, 1], index_col=0, parse_dates=True)
            return validate_quotes(cached, sessions, equity_tickers, vix, cash_proxy), fetched_at.isoformat()
        except (OSError, ValueError) as error:
            print(f"Refreshing invalid price cache: {error}")

    with TemporaryDirectory(dir=cache_dir, prefix="yahoo_session_") as download_cache:
        yf.set_tz_cache_location(download_cache)
        for attempt in range(2):
            try:
                quotes = yf.download(
                    tickers, start=start.date().isoformat(), end=(end + DAY).date().isoformat(),
                    auto_adjust=False, actions=True, ignore_tz=True, progress=False,
                    threads=False, timeout=12,
                )
                if quotes is None or quotes.empty:
                    raise ValueError("Yahoo returned no quotes")
                quotes = validate_quotes(quotes, sessions, equity_tickers, vix, cash_proxy)
                break
            except (KeyError, ValueError, OSError) as error:
                if attempt == 1:
                    raise RuntimeError(f"Could not download complete Yahoo prices and actions: {error}") from error
                print(f"Retrying Yahoo download: {error}")

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_dir, suffix=".tmp", delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(date_prefix + fetched_at + "\n")
        quotes.to_csv(handle)
    temporary_path.replace(cache_path)
    return quotes, fetched_at


def spread_trades(periods, regular_sessions, quiet):
    weekend_frames = []
    for start, end in periods:
        regular = regular_sessions[(regular_sessions >= start) & (regular_sessions <= end)]
        fridays = regular[regular.day_of_week == 4]
        mondays = fridays + 3 * DAY
        eligible = regular_as_of(fridays, fridays, regular) & regular_as_of(mondays, fridays, regular)
        weekends = pd.DataFrame({"entry": fridays[eligible], "exit": mondays[eligible]})
        weekends["quiet"] = quiet.reindex(weekends["entry"]).to_numpy()
        # Plan the midweek comparison with the calendar known on Friday.
        weekends["mid_entry"] = weekends["entry"] + 5 * DAY
        weekends["mid_exit"] = weekends["entry"] + 6 * DAY
        mid_entry = pd.DatetimeIndex(weekends["mid_entry"])
        mid_exit = pd.DatetimeIndex(weekends["mid_exit"])
        decisions = pd.DatetimeIndex(weekends["entry"])
        weekends["match"] = regular_as_of(mid_entry, decisions, regular) & regular_as_of(mid_exit, decisions, regular)
        weekends["mid_open"] = regular_as_of(mid_entry, mid_entry, regular) & regular_as_of(mid_exit, mid_entry, regular)
        weekend_frames.append(weekends)

    weekends = pd.concat(weekend_frames, ignore_index=True)
    gated = weekends.loc[weekends["quiet"]]
    matched = gated.loc[gated["match"]]
    return {
        "strategy": gated[["entry", "exit"]],
        "inverse_weekend": gated[["entry", "exit"]],
        "always_on_weekend": weekends[["entry", "exit"]],
        "matched_weekend": matched[["entry", "exit"]],
        "matched_midweek": matched.loc[matched["mid_open"], ["mid_entry", "mid_exit"]].rename(
            columns={"mid_entry": "entry", "mid_exit": "exit"}
        ),
    }


def size_positions(cash, weights, prior_closes, cash_buffer):
    return cash * (1 - cash_buffer) * weights / prior_closes


def trade_at_close(cash, position_changes, closing_prices, trading_rate):
    fills = position_changes * closing_prices
    notional = np.abs(fills).sum()
    return cash - fills.sum() - notional * trading_rate, notional


def require_cash(cash, day):
    if not np.isfinite(cash) or cash < 0:
        raise ValueError(f"Insufficient free cash on {day.date()} for the fixed positions")


def simulate_spreads(closes, distributions, trades, weights, trading_rate, borrow_rate, cash_buffer):
    prices = closes.loc[:, weights.index]
    payouts = distributions.loc[:, weights.index]
    daily_results = pd.DataFrame(0.0, index=prices.index, columns=["return", "turnover", "trades"])
    daily_results[["gross_round_trip_return", "net_round_trip_return"]] = np.nan
    cash = 1.0
    last_exit = prices.index[0]
    for entry, exit_day in trades.itertuples(index=False, name=None):
        if not last_exit < entry < exit_day:
            raise ValueError("Spread fills must be ordered, nonoverlapping and preceded by a sizing session")
        entry_row = prices.index.get_loc(entry)
        exit_row = prices.index.get_loc(exit_day)
        starting_cash = cash
        positions = size_positions(starting_cash, weights, prices.iloc[entry_row - 1], cash_buffer)

        entry_prices = prices.iloc[entry_row]
        entry_values = positions * entry_prices
        short_notional = -entry_values.clip(upper=0).sum()
        cash, entry_notional = trade_at_close(cash, positions, entry_prices, trading_rate)
        require_cash(cash - short_notional, entry)
        value_at_close = cash + entry_values.sum()
        daily_results.loc[entry, ["return", "turnover", "trades"]] = [
            value_at_close / starting_cash - 1, entry_notional / starting_cash, 1,
        ]

        distribution_pnl = 0.0
        previous_day = entry
        for day, closing_prices in prices.iloc[entry_row + 1:exit_row + 1].iterrows():
            # Positions carried into an ex-date receive or owe the distribution.
            payout = positions @ payouts.loc[day]
            distribution_pnl += payout
            cash += payout - short_notional * borrow_rate * (day - previous_day).days / 365
            marked_positions = positions * closing_prices
            require_cash(cash + marked_positions.clip(upper=0).sum(), day)
            previous_value = value_at_close
            value_at_close = cash + marked_positions.sum()
            if day == exit_day:
                cash, exit_notional = trade_at_close(cash, -positions, closing_prices, trading_rate)
                require_cash(cash, day)
                daily_results.loc[day, ["turnover", "trades"]] = [exit_notional / value_at_close, 1]
                value_at_close = cash
                gross_pnl = positions @ (closing_prices - entry_prices) + distribution_pnl
                daily_results.loc[day, ["gross_round_trip_return", "net_round_trip_return"]] = [
                    gross_pnl / starting_cash, cash / starting_cash - 1,
                ]
            daily_results.loc[day, "return"] = value_at_close / previous_value - 1
            previous_day = day
        last_exit = exit_day
    return daily_results


def buy_and_hold(prices, payouts, start, end, trading_rate, cash_buffer):
    marks = prices.loc[start:end]
    entry_day = marks.index[0]
    entry_row = prices.index.get_loc(entry_day)
    if entry_row == 0:
        raise ValueError("Benchmark entry requires a preceding sizing session")
    prior_close = prices.iloc[entry_row - 1]
    positions = size_positions(1.0, 1.0, prior_close, cash_buffer)
    cash, notional = trade_at_close(1.0, positions, marks.iloc[0], trading_rate)
    require_cash(cash, entry_day)
    # Distributions stay in cash. The opening day has no entitled position.
    distributions = payouts.loc[marks.index].copy()
    distributions.iloc[0] = 0
    values = cash + positions * (marks + distributions.cumsum())
    daily_results = pd.DataFrame(0.0, index=marks.index, columns=["return", "turnover", "trades"])
    daily_results["return"] = values.pct_change(fill_method=None)
    daily_results.iloc[0] = [values.iloc[0] - 1, notional, 1]
    return daily_results


def finite(value):
    return float(value) if np.isfinite(value) else None


def sharpe(returns):
    daily_vol = returns.std(ddof=1)
    return finite(returns.mean() / daily_vol * np.sqrt(SESSIONS_PER_YEAR)) if daily_vol > 0 else None


def metrics(daily_results, cash_returns):
    returns = daily_results["return"]
    excess_returns = returns - cash_returns.loc[returns.index]
    # Each period begins at its first close, with no preceding BIL exposure.
    excess_returns.iloc[0] = returns.iloc[0]
    cumulative_value = (1 + returns).cumprod()
    peak = cumulative_value.cummax().clip(lower=1)
    years = (returns.index[-1] - returns.index[0]).days / 365.2425
    volatility = returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)
    result = {
        "start": returns.index[0].date().isoformat(),
        "end": returns.index[-1].date().isoformat(),
        "cagr": finite(cumulative_value.iloc[-1] ** (1 / years) - 1) if years > 0 else None,
        "ann_vol": finite(volatility),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(excess_returns),
        "max_drawdown": finite((cumulative_value / peak - 1).min()),
        "total_return": finite(cumulative_value.iloc[-1] - 1),
        "n_trades": int(daily_results["trades"].sum()),
        "turnover": finite(daily_results["turnover"].sum()),
    }
    if "gross_round_trip_return" in daily_results:
        result["n_round_trips"] = int(daily_results["gross_round_trip_return"].count())
        result["mean_gross_trade_bps"] = finite(daily_results["gross_round_trip_return"].mean() * 10_000)
        result["mean_net_trade_bps"] = finite(daily_results["net_round_trip_return"].mean() * 10_000)
    return result


def print_metrics(reports):
    fields = {
        "cagr": "CAGR", "ann_vol": "Ann. vol", "sharpe": "Sharpe", "sharpe_excess": "Excess Sharpe",
        "max_drawdown": "Max DD",
        "total_return": "Total", "n_trades": "Trades", "turnover": "Turnover",
        "n_round_trips": "Round trips", "mean_gross_trade_bps": "Gross bp/round", "mean_net_trade_bps": "Net bp/round",
    }
    formats = [".2%", ".2%", ".2f", ".2f", ".2%", ".2%", "d", ".2f", "d", ".2f", ".2f"]
    rows = {}
    for name, periods in reports.items():
        for period, summary in periods.items():
            rows[(name, period.upper())] = {
                label: format(summary[field], spec) if summary.get(field) is not None else "–"
                for (field, label), spec in zip(fields.items(), formats)
            }
    table = pd.DataFrame.from_dict(rows, orient="index")
    print(table.to_string())


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    dates = config["dates"]
    data_start = pd.Timestamp(dates["data_start"])
    is_start, is_end = (pd.Timestamp(dates["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(dates["out_of_sample"][key]) for key in ("start", "end"))
    if not data_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Use ordered, nonoverlapping date ranges with a warm-up")
    periods = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    sessions, regular = trading_calendar(data_start, oos_end)
    lookback = int(config["signal"]["vix_rank_sessions"])
    threshold = float(config["signal"]["max_vix_percentile"])
    if not 1 <= lookback <= (sessions < is_start).sum() or not 0 < threshold <= 1:
        raise ValueError("Check VIX rank settings and warm-up length")

    universe = config["universe"]
    market, small_cap, quality, vix, cash_proxy = (
        universe[key] for key in ("market", "small_cap", "quality", "volatility", "cash_proxy")
    )
    equity_tickers = [market, small_cap, quality]
    if len({*equity_tickers, vix, cash_proxy}) != len(equity_tickers) + 2:
        raise ValueError("Universe entries must be distinct tickers")
    size_gross = float(config["portfolio"]["size_spread_gross"])
    quality_gross = float(config["portfolio"]["quality_spread_gross"])
    cash_buffer = float(config["portfolio"]["cash_buffer_fraction"])
    commission_bps = float(config["costs"]["commission_bps_per_side"])
    slippage_bps = float(config["costs"]["slippage_bps_per_side"])
    trading_rate = (commission_bps + slippage_bps) / 10_000
    borrow_rate = float(config["costs"]["short_borrow_bps_per_year"]) / 10_000
    gross = size_gross + quality_gross
    if not (size_gross >= 0 and quality_gross >= 0 and 0 < gross <= 1 and 0 < cash_buffer < 1
            and commission_bps >= 0 and slippage_bps >= 0 and 0 <= trading_rate < 1
            and np.isfinite(borrow_rate) and borrow_rate >= 0):
        raise ValueError("Gross allocations and cost settings are invalid")
    weights = pd.Series({market: gross / 2, small_cap: -size_gross / 2, quality: -quality_gross / 2})

    quotes, fetched_at = load_quotes(equity_tickers, vix, cash_proxy, data_start, oos_end, sessions)
    closes = quotes["Close"]
    distributions = quotes["Dividends"] + quotes["Capital Gains"]
    cash_returns = quotes["Adj Close"][cash_proxy].pct_change(fill_method=None)
    previous_rank = closes[vix].rolling(lookback).rank(method="average", pct=True).shift(1)
    trades = spread_trades([periods["is"], periods["oos"]], regular, previous_rank.le(threshold))
    reports = {}
    for name, schedule in trades.items():
        signed_weights = -weights if name == "inverse_weekend" else weights
        daily_results = simulate_spreads(closes, distributions, schedule, signed_weights,
                                         trading_rate, borrow_rate, cash_buffer)
        reports[name] = {
            key: metrics(daily_results.loc[start:end], cash_returns) for key, (start, end) in periods.items()
        }
    benchmark_name = f"{market} buy-and-hold"
    reports[benchmark_name] = {
        key: metrics(
            buy_and_hold(closes[market], distributions[market], start, end, trading_rate, cash_buffer),
            cash_returns,
        )
        for key, (start, end) in periods.items()
    }
    primary = reports[benchmark_name]
    benchmark = {"name": benchmark_name, **primary["full"], **primary}
    benchmark.update({name: reports[name] for name in trades if name != "strategy"})
    result = {
        "strategy": config["strategy"],
        "data_range": f"{primary['full']['start']} to {primary['full']['end']}",
        **reports["strategy"],
        "benchmark": benchmark,
    }

    print(f"{config['strategy']} | {result['data_range']} | Yahoo Finance daily prices and distributions")
    print(f"PARAM_COMBOS_TRIED: {PARAM_COMBOS_TRIED}")
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"Trading: {commission_bps:g} bps commission + {slippage_bps:g} bps slippage per side | "
          f"Short borrow: {borrow_rate * 10_000:g} bps/year | Sizing cash buffer: {cash_buffer:.0%}")
    print(f"Excess Sharpe cash proxy: {cash_proxy} | Trades: basket entries and exits, or benchmark buy")
    print_metrics(reports)
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
