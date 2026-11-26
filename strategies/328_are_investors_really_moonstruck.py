"""328_are_investors_really_moonstruck
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/328_are_investors_really_moonstruck.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit b9dcd3d64e5973b6eae2ce2f8a86efeffe63a9df on 2026-11-26 15:35 Europe/Oslo, exit 0, output sha256 81f30ea0751a4478497e5c2694910278bda0ec6c8463232ee02cce7755f6443a
Data source: Yahoo Finance 的日开盘价和收盘价，经 yfinance 获取或从本地行情缓存读取。股票 ETF 为 SPY、EFA、EEM，防御 ETF 为 BIL，基准为 SPY（config.yaml:3–6，README.md:13，strategy.py:129–155、314–319）。新月和满月的 UTC 时刻用代码中的 Meeus 公式计算（strategy.py:40–43、83–119、320–321）。
Data fetched: 2026-11-26 15:35 Europe/Oslo (run time)
Sample period: 配置覆盖 2008-01-01 至 2026-11-24，含两个端点（config.yaml:7–12，strategy.py:296–309）。日志记录的实际交易日范围为 2008-01-02 至 2026-11-24（logs/run_20261125_144038.log:1）。   data_range (from output): 2008-01-02 to 2026-11-24
IS/OOS split: 样本内 2008-01-01 至 2019-12-31，样本外 2020-01-01 至 2026-11-24，端点均包含（config.yaml:7–12，strategy.py:296–310）。持仓跨分界连续计算，再按日期分段报告（strategy.py:263–268、328–330，README.md:16）。README 记载两个区间均未调参，参考论文已研究部分样本外年份（README.md:21）。
Costs and slippage: 买卖各按成交额收取 5 bps，首次建仓收买入费，切换配置时由开盘净值同时支付卖出费和买入费，未切换时不交易（config.yaml:14，strategy.py:209–232、301–302，README.md:16）。滑点未单独建模，成交按开盘价模拟，README 将固定费用视为点差和滑点的近似（strategy.py:209–228，README.md:23）。
Signal lag: 月相日历信号可预先计算，并按前一计划交易日收盘已知处理。信号在计划交易日索引上后移一位，再映射到实际交易日，按对应开盘价模拟换仓。计划窗口保留临时休市日，实际成交仅发生在开市日（strategy.py:34–37、118–126、209–228、306–309、320–328，README.md:14–15）。
Price adjustment: 下载显式使用 `auto_adjust=True`，模拟使用复权后的 `Open` 和 `Close`。有效缓存直接复用相同字段，代码校验日期、列及价格值，未独立重算复权（strategy.py:129–163，README.md:13）。
Survivorship bias: ETF 的历史行情反映基金随指数调整的持仓，模拟没有用当今指数成分股回填历史。标的固定为目前仍存续的四只 ETF，仍有标的选择偏差（config.yaml:3–6，strategy.py:314–319，README.md:23）。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.01571143093041072, ann_vol=0.11170074776271348, sharpe=0.1948883480695728, max_drawdown=-0.2071171136072949, total_return=0.2054896379416622, turnover=596.7016491754124, n_trades=299
    oos: cagr=-0.009367582417288967, ann_vol=0.09017503295375442, sharpe=-0.05947096566098231, max_drawdown=-0.1955069887344505, total_return=-0.06270920177955341, turnover=339.8300849575213, n_trades=170
    full: cagr=0.006493183456579565, ann_vol=0.10436002390081434, sharpe=0.11378870437459518, max_drawdown=-0.2728503578998869, total_return=0.1298943449928207, turnover=936.5317341329336, n_trades=469
    benchmark  full: cagr=0.10779183277232152, ann_vol=0.19761401047590313, sharpe=0.6169873246863916, max_drawdown=-0.5242305002338081, total_return=5.900540652386718, turnover=0.9995002498750625, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 328_are_investors_really_moonstruck\ncategory: seasonality\nuniverse:\n  equity: [SPY, EFA, EEM]\n  defensive: BIL\n  benchmark: SPY\nin_sample:\n  start: '2008-01-01'\n  end: '2019-12-31'\nout_of_sample:\n  start: '2020-01-01'\n  end: '2026-11-24'\nwindow_sessions_each_side: 2\ncost_bps_per_side: 5.0\nannualization_sessions: 252\nplacebos:\n  count: 200\n  seed: 328\n"
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
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from pandas.tseries import holiday


ROOT = Path(__file__).resolve().parent


class USMarketHolidays(holiday.AbstractHolidayCalendar):
    rules = [
        holiday.Holiday("New Year", month=1, day=1, observance=holiday.sunday_to_monday),
        holiday.USMartinLutherKingJr,
        holiday.USPresidentsDay,
        holiday.GoodFriday,
        holiday.USMemorialDay,
        holiday.Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                        observance=holiday.nearest_workday),
        holiday.Holiday("Independence Day", month=7, day=4, observance=holiday.nearest_workday),
        holiday.USLaborDay,
        holiday.USThanksgivingDay,
        holiday.Holiday("Christmas", month=12, day=25, observance=holiday.nearest_workday),
    ]


# These dates are removed from expected prices only. Scheduled windows do not anticipate them.
EXCEPTIONAL_CLOSURES = pd.DatetimeIndex([
    "2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09",
])


# Jean Meeus, Astronomical Algorithms, 2nd ed., ch. 49. Each term is
# (lunar anomaly, solar anomaly, lunar latitude, node, eccentricity power,
#  new-moon amplitude in days, full-moon amplitude in days).
LUNAR_TERMS = np.array([
    (1, 0, 0, 0, 0, -0.40720, -0.40614),
    (0, 1, 0, 0, 1, 0.17241, 0.17302),
    (2, 0, 0, 0, 0, 0.01608, 0.01614),
    (0, 0, 2, 0, 0, 0.01039, 0.01043),
    (1, -1, 0, 0, 1, 0.00739, 0.00734),
    (1, 1, 0, 0, 1, -0.00514, -0.00515),
    (0, 2, 0, 0, 2, 0.00208, 0.00209),
    (1, 0, -2, 0, 0, -0.00111, -0.00111),
    (1, 0, 2, 0, 0, -0.00057, -0.00057),
    (2, 1, 0, 0, 1, 0.00056, 0.00056),
    (3, 0, 0, 0, 0, -0.00042, -0.00042),
    (0, 1, 2, 0, 1, 0.00042, 0.00042),
    (0, 1, -2, 0, 1, 0.00038, 0.00038),
    (2, -1, 0, 0, 1, -0.00024, -0.00024),
    (0, 0, 0, 1, 0, -0.00017, -0.00017),
    (1, 2, 0, 0, 0, -0.00007, -0.00007),
    (2, 0, -2, 0, 0, 0.00004, 0.00004),
    (0, 3, 0, 0, 0, 0.00004, 0.00004),
    (1, 1, -2, 0, 0, 0.00003, 0.00003),
    (2, 0, 2, 0, 0, 0.00003, 0.00003),
    (1, 1, 2, 0, 0, -0.00003, -0.00003),
    (1, -1, 2, 0, 0, 0.00003, 0.00003),
    (1, -1, -2, 0, 0, -0.00002, -0.00002),
    (3, 1, 0, 0, 0, -0.00002, -0.00002),
    (4, 0, 0, 0, 0, 0.00002, 0.00002),
])

# Initial longitude (degrees), degrees per lunation, amplitude (days).
PLANETARY_TERMS = np.array([
    (299.70, 0.107408, 0.000325), (251.88, 0.016321, 0.000165),
    (251.83, 26.651886, 0.000164), (349.42, 36.412478, 0.000126),
    (84.66, 18.206239, 0.000110), (141.74, 53.303771, 0.000062),
    (207.17, 2.453732, 0.000060), (154.84, 7.306860, 0.000056),
    (34.52, 27.261239, 0.000047), (207.19, 0.121824, 0.000042),
    (291.34, 1.844379, 0.000040), (161.72, 24.198154, 0.000037),
    (239.56, 25.513099, 0.000035), (331.55, 3.592518, 0.000023),
])


def moon_times_utc(start, end, full=False):
    """Calculate geocentric primary phase times, then convert TT to UTC."""
    epoch = pd.Timestamp("2000-01-06 14:20:37.824")  # JDE 2451550.09766 in TT
    synodic_days = 29.530588861
    first = int(np.floor((start - epoch) / pd.Timedelta(days=synodic_days))) - 1
    last = int(np.ceil((end - epoch) / pd.Timedelta(days=synodic_days))) + 1
    lunations = np.arange(first, last + 1, dtype=float) + (0.5 if full else 0.0)
    centuries = lunations / 1236.85
    mean_days = (synodic_days * lunations + 0.00015437 * centuries**2
                 - 0.000000150 * centuries**3 + 0.00000000073 * centuries**4)
    eccentricity = 1 - 0.002516 * centuries - 0.0000074 * centuries**2
    angles = np.column_stack([
        201.5643 + 385.81693528 * lunations + 0.0107582 * centuries**2
        + 0.00001238 * centuries**3 - 0.000000058 * centuries**4,
        2.5534 + 29.10535670 * lunations - 0.0000014 * centuries**2 - 0.00000011 * centuries**3,
        160.7108 + 390.67050284 * lunations - 0.0016118 * centuries**2
        - 0.00000227 * centuries**3 + 0.000000011 * centuries**4,
        124.7746 - 1.56375588 * lunations + 0.0020672 * centuries**2 + 0.00000215 * centuries**3,
    ])
    amplitude = LUNAR_TERMS[:, 6 if full else 5]
    lunar = (np.sin(np.deg2rad(angles @ LUNAR_TERMS[:, :4].T))
             * eccentricity[:, None] ** LUNAR_TERMS[:, 4]) @ amplitude
    base, rate, amplitude = PLANETARY_TERMS.T
    planets = base + lunations[:, None] * rate
    planets[:, 0] -= 0.009173 * centuries**2
    planetary = np.sin(np.deg2rad(planets)) @ amplitude
    tt = epoch + pd.to_timedelta(mean_days + lunar + planetary, unit="D")

    # TT minus UTC was 65.184 seconds in 2007. Later leap seconds add one second each.
    leap_dates = pd.DatetimeIndex(["2009-01-01", "2012-07-01", "2015-07-01", "2017-01-01"])
    leap_instants_tt = leap_dates + pd.to_timedelta(66.184 + np.arange(len(leap_dates)), unit="s")
    tt_minus_utc = 65.184 + leap_instants_tt.searchsorted(tt, side="right")
    return tt - pd.to_timedelta(tt_minus_utc, unit="s")


def lunar_fills(calendar, sessions, radius, full=False):
    phase_dates = moon_times_utc(calendar[0], calendar[-1], full).normalize()
    centers = calendar.searchsorted(phase_dates)  # Roll non-session UTC dates forward.
    centers = centers[(phase_dates >= calendar[0]) & (centers < len(calendar))]
    signal_days = centers[:, None] + np.arange(-radius - 1, radius)
    signal_days = signal_days[(signal_days >= 0) & (signal_days < len(calendar))]
    signals = pd.Series(False, index=calendar)
    signals.iloc[signal_days] = True
    return signals.shift(1, fill_value=False).loc[sessions].to_numpy()


def load_prices(tickers, sessions):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    start = sessions[0].date().isoformat()
    end = (sessions[-1] + pd.Timedelta(days=1)).date().isoformat()
    cache = cache_dir / f"yahoo_adjusted_{start}_{end}_{'_'.join(tickers)}.csv"
    columns = [f"{field}_{ticker}" for field in ("Open", "Close") for ticker in tickers]

    def valid(prices):
        return (prices.index.equals(sessions) and list(prices.columns) == columns
                and np.isfinite(prices.to_numpy()).all() and (prices.to_numpy() > 0).all())

    if cache.exists():
        try:
            prices = pd.read_csv(cache, index_col=0, parse_dates=True,
                                 dtype=dict.fromkeys(columns, float))
            if valid(prices):
                return prices.to_numpy().reshape(-1, 2, len(tickers))
        except (OSError, ValueError):
            pass  # Replace an unreadable or corrupt cache with a validated download.

    yf.set_tz_cache_location(str(cache_dir / "yfinance"))
    for _ in range(2):
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, threads=True,
                          progress=False, timeout=15, multi_level_index=True)
        prices = raw.reindex(columns=pd.MultiIndex.from_product([("Open", "Close"), tickers]))
        prices.columns = columns
        if valid(prices):
            partial = cache.with_suffix(f".{os.getpid()}.tmp")
            try:
                prices.to_csv(partial, index_label="Date")
                partial.replace(cache)
            finally:
                partial.unlink(missing_ok=True)
            return prices.to_numpy().reshape(-1, 2, len(tickers))
    missing = sessions.difference(prices.dropna().index)
    raise RuntimeError(f"Incomplete Yahoo prices for {tickers}. Missing sessions: {list(missing[:10].date)}")


def shifted_schedules(sessions, new_moon, is_oos, count, seed):
    """Shift entire equity blocks, matching dates by year, weekday and split."""
    edges = np.flatnonzero(np.diff(new_moon, prepend=False, append=False))
    blocks = list(zip(edges[::2], edges[1::2]))
    calendar_keys = list(zip(sessions.year, sessions.day_of_week, is_oos))
    possible_starts = {}
    for length in {stop - start for start, stop in blocks}:
        for start in range(len(sessions) - length + 1):
            key = tuple(calendar_keys[start:start + length])
            possible_starts.setdefault(key, []).append(start)

    pools = []
    for start, stop in blocks:
        key = tuple(calendar_keys[start:stop])
        starts = [start] if start == 0 or stop == len(sessions) else possible_starts[key]
        pools.append((stop - start, np.asarray(starts)))
    pools.sort(key=lambda block: len(block[1]))

    rng = np.random.default_rng(seed)
    accepted = []
    seen = {new_moon.tobytes()}
    for _ in range(100 * count):
        active = np.zeros_like(new_moon)
        for length, starts in pools:
            for start in rng.permutation(starts):
                # Keep at least one defensive session between adjacent windows.
                if not active[max(0, start - 1):start + length + 1].any():
                    active[start:start + length] = True
                    break
            else:
                break
        else:
            signature = active.tobytes()
            if signature not in seen:
                seen.add(signature)
                accepted.append(active)
            if len(accepted) == count:
                return np.column_stack(accepted)
    raise RuntimeError(f"Generated only {len(accepted)} unique exposure-matched schedules out of {count}.")


def backtest(prices, sleeves, weights, fee):
    """Trade at the open and carry drifting positions while the sleeve is unchanged."""
    opens, closes = prices[:, 0], prices[:, 1]
    overnight = opens[1:] / closes[:-1]
    intraday = closes / opens
    net_returns = np.empty(sleeves.shape)
    turnover = np.zeros(sleeves.shape)
    holdings = np.zeros((sleeves.shape[1], opens.shape[1]))
    previous = np.full(sleeves.shape[1], -1)
    for day, target in enumerate(sleeves):
        if day:
            holdings *= overnight[day - 1]
        switch = target != previous
        sold = holdings[switch].sum(axis=1)
        opening_nav = sold if day else np.ones(sold.size)
        # The sleeves are disjoint. Finance both legs: new holdings + fee on both legs = NAV.
        purchased = (opening_nav - fee * sold) / (1 + fee)
        turnover[day, switch] = (sold + purchased) / opening_nav
        holdings[switch] = purchased[:, None] * weights[target[switch]]
        holdings *= intraday[day]
        closing_nav = holdings.sum(axis=1)
        net_returns[day] = closing_nav - 1
        holdings /= closing_nav[:, None]
        previous = target
    return net_returns, turnover


def finite(value):
    return float(value) if np.isfinite(value) else None


def metrics(daily, turnover, equity_days, annualization):
    nav = np.cumprod(np.r_[1.0, 1 + daily])
    vol = daily.std(ddof=1) if len(daily) > 1 else np.nan
    return {
        "cagr": finite(nav[-1] ** (annualization / len(daily)) - 1),
        "ann_vol": finite(vol * np.sqrt(annualization)),
        "sharpe": finite(daily.mean() / vol * np.sqrt(annualization)) if vol > 0 else None,
        "max_drawdown": finite(np.min(nav / np.maximum.accumulate(nav) - 1)),
        "total_return": finite(nav[-1] - 1),
        "n_trades": int(np.count_nonzero(turnover)),
        "turnover": finite(turnover.sum()),
        "n_sessions": len(daily),
        "equity_sessions": int(equity_days.sum()),
    }


def report(config, sessions, masks, sleeves, daily, turnover):
    annualization = config["annualization_sessions"]
    benchmark_name = f"{config['universe']['benchmark']} buy & hold"
    ranges = f"{sessions[0].date()} to {sessions[-1].date()}"
    strategy = {}
    benchmark = {"name": benchmark_name, "full_moon": {}, "calendar_placebos": {}}
    rows = []
    for period, mask in masks.items():
        returns = daily[mask]
        for col, (name, output) in enumerate([
            ("New moon", strategy), ("Full moon", benchmark["full_moon"]), (benchmark_name, benchmark),
        ]):
            values = metrics(returns[:, col], turnover[mask, col], sleeves[mask, col] != 0, annualization)
            output[period] = values
            rows.append({"Period": period.upper(), "Portfolio": name, **values})

        placebo_cagrs = np.prod(1 + returns[:, 3:], axis=0) ** (annualization / len(returns)) - 1
        low, median, high = np.quantile(placebo_cagrs, [0.05, 0.50, 0.95])
        upper_tail = (1 + np.count_nonzero(placebo_cagrs >= strategy[period]["cagr"])) / (len(placebo_cagrs) + 1)
        benchmark["calendar_placebos"][period] = {
            "count": len(placebo_cagrs), "cagr_p05": finite(low), "cagr_median": finite(median),
            "cagr_p95": finite(high), "new_moon_upper_tail_rank": finite(upper_tail),
            "equity_sessions_each": strategy[period]["equity_sessions"],
        }

    print(f"{config['strategy']} | {ranges} | {config['cost_bps_per_side']:g} bps per side")
    table = pd.DataFrame(rows).drop(columns=["turnover", "n_sessions"])
    percents = {column: "{:.2%}".format for column in ("cagr", "ann_vol", "max_drawdown", "total_return")}
    print(table.to_string(index=False, formatters={**percents, "sharpe": "{:.2f}".format}))
    print("\nMatched calendar placebos (annualized return)")
    for period, values in benchmark["calendar_placebos"].items():
        print(f"{period.upper():4}  median {values['cagr_median']:7.2%}  "
              f"5th–95th pct. {values['cagr_p05']:7.2%} to {values['cagr_p95']:7.2%}  "
              f"new-moon upper-tail rank {values['new_moon_upper_tail_rank']:.3f}  (n={values['count']})")
    result = {"strategy": config["strategy"], "data_range": ranges, **strategy, "benchmark": benchmark}
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not (pd.Timestamp("2008-01-01") <= start <= is_end < oos_start <= end
            and oos_start - is_end == pd.Timedelta(days=1)):
        raise ValueError("Use contiguous, ordered IS and OOS ranges starting in 2008 or later.")
    radius = config["window_sessions_each_side"]
    fee = config["cost_bps_per_side"] / 10_000
    count = config["placebos"]["count"]
    if radius < 0 or not 0 <= fee < 1 or count < 1 or config["annualization_sessions"] <= 0:
        raise ValueError("Invalid event window, trading cost, placebo count or annualization.")
    business_day = pd.offsets.CustomBusinessDay(calendar=USMarketHolidays())
    pad = (radius + 3) * business_day
    scheduled = pd.date_range(start - pad, end + pad, freq=business_day, name="Date")
    sessions = scheduled[(scheduled >= start) & (scheduled <= end)].difference(EXCEPTIONAL_CLOSURES)
    is_oos = sessions >= oos_start
    if not is_oos.any() or is_oos.all():
        raise ValueError("Each sample needs at least one trading session.")

    universe = config["universe"]
    equities = universe["equity"]
    tickers = [*equities, universe["defensive"]]
    if not equities or len(set(tickers)) != len(tickers) or universe["benchmark"] not in equities:
        raise ValueError("Use distinct equity and defensive tickers and an equity ticker as benchmark.")
    prices = load_prices(tickers, sessions)
    new_moon = lunar_fills(scheduled, sessions, radius)
    full_moon = lunar_fills(scheduled, sessions, radius, full=True)
    placebos = shifted_schedules(sessions, new_moon, is_oos, count, config["placebos"]["seed"])
    sleeves = np.column_stack([new_moon, full_moon, np.full(len(sessions), 2), placebos])
    weights = np.zeros((3, len(tickers)))
    weights[0, -1] = 1
    weights[1, :-1] = 1 / len(equities)
    weights[2, equities.index(universe["benchmark"])] = 1
    daily, turnover = backtest(prices, sleeves, weights, fee)
    report(config, sessions, {"is": ~is_oos, "oos": is_oos, "full": np.ones(len(sessions), dtype=bool)},
           sleeves, daily, turnover)


if __name__ == "__main__":
    main()
