"""324_volatility_conditions_weekend_effect
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/324_volatility_conditions_weekend_effect.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 3140bbcffa77faae04956fc55adbc5be3b33f896 on 2026-11-26 16:06 Europe/Oslo, exit 0, output sha256 c5cc1df8764f97ff69a8a33143e06d45a52201b83892d210508f806186348b00
Data source: Yahoo Finance adjusted daily closes via yfinance for U.S. equity ETFs SPY, IWM and QUAL, plus the VIX volatility index (^VIX) for the signal.
Data fetched: 2026-11-26 16:06 Europe/Oslo (run time)
Sample period: Reported trading sessions 2016-01-04 through 2025-12-31. The configured data range starts 2015-01-01, with 2015 sessions used for warm-up.   data_range (from output): 2016-01-04 to 2025-12-31
IS/OOS split: Configured IS 2016-01-01 through 2023-12-31 and OOS 2024-01-01 through 2025-12-31, inclusive. Reported sessions are IS 2016-01-04 through 2023-12-29 and OOS 2024-01-02 through 2025-12-31. The strategy keeps each entry and exit within the same split and carries prior price history into the rolling rank.
Costs and slippage: 5 bps per side on absolute spread notional at entry and exit, with exit notional estimated from adjusted close price ratios. Short borrow costs 50 bps per year on initial short notional, prorated by calendar days held divided by 365. The SPY buy-and-hold benchmark pays one 5 bps buy cost at the start of each report period. The backtest has no explicit slippage model.
Signal lag: The ^VIX close at session t is ranked over the 252 sessions ending at t, using average ranks for ties. The rank is shifted by one trading session. If the next session is an eligible Friday and the rank is at or below 50%, the strategy enters at that Friday's adjusted close and exits at the immediately following eligible Monday's adjusted close.
Price adjustment: The Yahoo download sets auto_adjust=True and reads Close. The cache stores the resulting adjusted closes. Spread P&L and estimated exit notional both use ratios of adjusted closes.
Survivorship bias: The universe is a fixed list of three ETF proxies. The README identifies selection bias from choosing ETFs that still trade today, and the implementation has no point-in-time universe or delisting treatment.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.02327086560398106, ann_vol=0.008669456151541412, sharpe=-2.711365337179041, max_drawdown=-0.17181053432777016, total_return=-0.17137230775531942, turnover=382.27445856708147, n_trades=191
    oos: cagr=-0.027600941870314366, ann_vol=0.009296937143096134, sharpe=-3.011595179771147, max_drawdown=-0.054332463297737665, total_return=-0.054332463297737665, turnover=76.1173092447298, n_trades=38
    full: cagr=-0.024112281949736802, ann_vol=0.00879715308399614, sharpe=-2.773981843436604, max_drawdown=-0.21644929394821422, total_return=-0.2163936914316924, turnover=458.39176781181123, n_trades=229
    benchmark: cagr=0.14886308711635587, ann_vol=0.18003735870183768, sharpe=0.8623737650767168, max_drawdown=-0.33717239100465857, total_return=3.000530654041155, turnover=0.9995002498750625, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 324_volatility_conditions_weekend_effect\ncategory: seasonality\n\nuniverse:\n  market: SPY\n  small_cap: IWM\n  quality: QUAL\n  volatility: '^VIX'\n\ndates:\n  data_start: '2015-01-01'\n  in_sample:\n    start: '2016-01-01'\n    end: '2023-12-31'\n  out_of_sample:\n    start: '2024-01-01'\n    end: '2025-12-31'\n\nsignal:\n  vix_rank_sessions: 252\n  max_vix_percentile: 0.50\n\nportfolio:\n  size_spread_gross: 0.50\n  quality_spread_gross: 0.50\n\ncosts:\n  trading_bps_per_side: 5.0\n  short_borrow_bps_per_year: 50.0\n"
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
MOURNING_CLOSURES = pd.to_datetime(["2018-12-05", "2025-01-09"])


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
    closed = NyseHolidays().holidays(start, end).union(MOURNING_CLOSURES)
    sessions = pd.bdate_range(start, end).difference(closed)
    black_fridays = USThanksgivingDay.dates(start - DAY, end) + DAY
    july_eves = (sessions.month == 7) & (sessions.day == 3)
    christmas_eves = (sessions.month == 12) & (sessions.day == 24)
    regular = sessions[~(sessions.isin(black_fridays) | july_eves | christmas_eves)]
    return sessions, regular


def validated_closes(frame, sessions, tickers):
    closes = frame.reindex(index=sessions, columns=tickers).astype(float)
    invalid = ~np.isfinite(closes) | closes.le(0)
    if invalid.any().any():
        first_day = invalid.index[invalid.any(axis=1)][0]
        symbols = ", ".join(invalid.columns[invalid.loc[first_day]])
        raise ValueError(f"Missing or invalid close on {first_day.date()} for {symbols}")
    return closes


def load_closes(tickers, start, end, sessions):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_key = "_".join(sorted(tickers)).replace("^", "index-")
    cache_path = cache_dir / f"yahoo_adjusted_{cache_key}_{start.date()}_{end.date()}.csv"
    if cache_path.is_file():
        try:
            cached = pd.read_csv(cache_path, index_col="Date", parse_dates=["Date"])
            return validated_closes(cached, sessions, tickers)
        except (OSError, ValueError) as error:
            print(f"Refreshing invalid price cache: {error}")

    with TemporaryDirectory(dir=cache_dir, prefix="yahoo_session_") as download_cache:
        yf.set_tz_cache_location(download_cache)
        for attempt in range(2):
            try:
                quotes = yf.download(
                    tickers, start=start.date().isoformat(), end=(end + DAY).date().isoformat(),
                    auto_adjust=True, progress=False, threads=False, timeout=12,
                )
                closes = validated_closes(quotes["Close"], sessions, tickers)
                break
            except (KeyError, ValueError, OSError) as error:
                if attempt == 1:
                    raise RuntimeError(f"Could not download complete Yahoo closes: {error}") from error
                print(f"Retrying Yahoo close download: {error}")

    with NamedTemporaryFile(mode="w", dir=cache_dir, suffix=".tmp", delete=False) as handle:
        temporary_path = Path(handle.name)
        closes.to_csv(handle, index_label="Date")
    temporary_path.replace(cache_path)
    return closes


def spread_trades(periods, regular_sessions, quiet):
    weekend_frames = []
    for start, end in periods:
        regular = regular_sessions[(regular_sessions >= start) & (regular_sessions <= end)]
        fridays = regular[regular.day_of_week == 4]
        mondays = fridays + 3 * DAY
        eligible = mondays.isin(regular)
        weekends = pd.DataFrame({"entry": fridays[eligible], "exit": mondays[eligible]})
        weekends["quiet"] = quiet.reindex(weekends["entry"]).to_numpy()
        # Match to the following week's Wednesday and Thursday within the same split.
        weekends["mid_entry"] = weekends["entry"] + 5 * DAY
        weekends["mid_exit"] = weekends["entry"] + 6 * DAY
        weekends["match"] = weekends["mid_entry"].isin(regular) & weekends["mid_exit"].isin(regular)
        weekend_frames.append(weekends)

    weekends = pd.concat(weekend_frames, ignore_index=True)
    gated = weekends.loc[weekends["quiet"]]
    matched = gated.loc[gated["match"]]
    return {
        "strategy": gated[["entry", "exit"]],
        "inverse_weekend": gated[["entry", "exit"]],
        "always_on_weekend": weekends[["entry", "exit"]],
        "matched_weekend": matched[["entry", "exit"]],
        "matched_midweek": matched[["mid_entry", "mid_exit"]].set_axis(["entry", "exit"], axis=1),
    }


def simulate_spreads(closes, trades, weights, trading_rate, borrow_rate):
    entry = pd.DatetimeIndex(trades["entry"])
    exit_ = pd.DatetimeIndex(trades["exit"])
    entry_prices = closes.loc[entry, weights.index].to_numpy()
    exit_prices = closes.loc[exit_, weights.index].to_numpy()
    price_ratio = exit_prices / entry_prices
    gross_pnl = (price_ratio - 1) @ weights.to_numpy()
    entry_turnover = weights.abs().sum()
    exit_turnover = price_ratio @ weights.abs().to_numpy()
    entry_fee = entry_turnover * trading_rate
    exit_fee = exit_turnover * trading_rate
    short_notional = -weights.clip(upper=0).sum()
    borrow_fee = short_notional * borrow_rate * (exit_ - entry).days.to_numpy() / 365
    net_pnl = gross_pnl - entry_fee - exit_fee - borrow_fee
    if (net_pnl <= -1).any():
        raise ValueError("Spread loss exhausted portfolio equity")

    daily = pd.DataFrame(0.0, index=closes.index, columns=["return", "turnover", "trades"])
    daily.loc[entry, "return"] = -entry_fee
    daily.loc[entry, "turnover"] = entry_turnover
    # Dollar P&L is sized off pre-entry equity. Exit returns are relative to post-entry equity.
    daily.loc[exit_, "return"] = (gross_pnl - exit_fee - borrow_fee) / (1 - entry_fee)
    daily.loc[exit_, "turnover"] = exit_turnover / (1 - entry_fee)
    daily.loc[exit_, "trades"] = 1
    daily.loc[exit_, "gross_trade_return"] = gross_pnl
    daily.loc[exit_, "net_trade_return"] = net_pnl
    return daily


def buy_and_hold(prices, trading_rate):
    daily = pd.DataFrame(0.0, index=prices.index, columns=["return", "turnover", "trades"])
    daily["return"] = prices.pct_change(fill_method=None)
    bought = 1 / (1 + trading_rate)
    daily.iloc[0] = [bought - 1, bought, 1]
    return daily


def finite(value):
    return float(value) if np.isfinite(value) else None


def metrics(daily):
    returns = daily["return"]
    equity = (1 + returns).cumprod()
    peak = equity.cummax().clip(lower=1)
    years = (returns.index[-1] - returns.index[0]).days / 365.2425
    volatility = returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)
    result = {
        "start": returns.index[0].date().isoformat(),
        "end": returns.index[-1].date().isoformat(),
        "cagr": finite(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else None,
        "ann_vol": finite(volatility),
        "sharpe": finite(returns.mean() * SESSIONS_PER_YEAR / volatility) if volatility > 0 else None,
        "max_drawdown": finite((equity / peak - 1).min()),
        "total_return": finite(equity.iloc[-1] - 1),
        "n_trades": int(daily["trades"].sum()),
        "turnover": finite(daily["turnover"].sum()),
    }
    if "gross_trade_return" in daily:
        result["mean_gross_trade_bps"] = finite(daily["gross_trade_return"].mean() * 10_000)
        result["mean_net_trade_bps"] = finite(daily["net_trade_return"].mean() * 10_000)
    return result


def print_metrics(reports):
    fields = {
        "cagr": "CAGR", "ann_vol": "Ann. vol", "sharpe": "Sharpe", "max_drawdown": "Max DD",
        "total_return": "Total", "n_trades": "Trades", "turnover": "Turnover",
        "mean_gross_trade_bps": "Gross bp/trade", "mean_net_trade_bps": "Net bp/trade",
    }
    formats = [".2%", ".2%", ".2f", ".2%", ".2%", "d", ".2f", ".2f", ".2f"]
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
    market, small_cap, quality, vix = (universe[key] for key in ("market", "small_cap", "quality", "volatility"))
    tickers = [market, small_cap, quality, vix]
    if len(set(tickers)) != len(tickers):
        raise ValueError("Universe entries must be distinct tickers")
    size_gross = float(config["portfolio"]["size_spread_gross"])
    quality_gross = float(config["portfolio"]["quality_spread_gross"])
    trading_rate = float(config["costs"]["trading_bps_per_side"]) / 10_000
    borrow_rate = float(config["costs"]["short_borrow_bps_per_year"]) / 10_000
    gross = size_gross + quality_gross
    if not (size_gross >= 0 and quality_gross >= 0 and np.isfinite(gross) and gross > 0
            and 0 <= trading_rate < 1 and gross * trading_rate < 1
            and np.isfinite(borrow_rate) and borrow_rate >= 0):
        raise ValueError("Gross allocations and cost settings are invalid")
    weights = pd.Series({market: gross / 2, small_cap: -size_gross / 2, quality: -quality_gross / 2})

    closes = load_closes(tickers, data_start, oos_end, sessions)
    previous_rank = closes[vix].rolling(lookback).rank(method="average", pct=True).shift(1)
    trades = spread_trades([periods["is"], periods["oos"]], regular, previous_rank.le(threshold))
    closes = closes.loc[is_start:oos_end]
    reports = {}
    for name, schedule in trades.items():
        signed_weights = -weights if name == "inverse_weekend" else weights
        daily = simulate_spreads(closes, schedule, signed_weights, trading_rate, borrow_rate)
        reports[name] = {key: metrics(daily.loc[start:end]) for key, (start, end) in periods.items()}
    reports[f"{market} buy-and-hold"] = {
        key: metrics(buy_and_hold(closes.loc[start:end, market], trading_rate))
        for key, (start, end) in periods.items()
    }
    primary = reports[f"{market} buy-and-hold"]
    benchmark = {"name": f"{market} buy-and-hold", **primary["full"], **primary}
    benchmark.update({name: reports[name] for name in trades if name != "strategy"})
    result = {
        "strategy": config["strategy"],
        "data_range": f"{closes.index[0].date()} to {closes.index[-1].date()}",
        **reports["strategy"],
        "benchmark": benchmark,
    }

    print(f"{config['strategy']} | {result['data_range']} | Yahoo Finance adjusted daily closes")
    print(f"Trading: {trading_rate * 10_000:g} bps per side | Short borrow: "
          f"{borrow_rate * 10_000:g} bps/year | Turnover: sum of fractions of equity traded")
    print_metrics(reports)
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
