"""179_crude_oil_news_sentiment
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/179_crude_oil_news_sentiment.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 10f82382008748cd9ff2f56e2b8ef160caa66e38 on 2026-11-27 11:24 Europe/Oslo, exit 0, output sha256 8998c356a317833c2b0c1473183f710ce0c666dc8d06b0cc63bf0983d4762418
Data source: 美国能源信息署（EIA）公开的 Past Weeks in Petroleum 周度石油市场标题及例行数据公告，发布日期从归档版次链接提取（https://www.eia.gov/petroleum/weekly/includes/archive.php）。Yahoo Finance 经 yfinance 提供 USO 和 BIL 的日频开盘价与收盘价。
Data fetched: EIA=2026-11-27T11:24:13+01:00, USO=2026-11-27T11:24:14+01:00, BIL=2026-11-27T11:24:14+01:00
Sample period: 配置日历区间为 2010-01-01 至 2025-10-31。日志记录的实际回测交易日区间为 2010-01-04 至 2025-10-31。 data_range (from output): 2010-01-04 to 2025-10-31
IS/OOS split: 按日历日期切分，IS 为 2010-01-01 至 2017-12-31，OOS 为 2018-01-01 至 2025-10-31，端点均包含。先计算全样本的同一持仓路径，再分段汇报，OOS 承接原有持仓。README 将 OOS 描述为回顾性留出。
Costs and slippage: 每笔买或卖按成交名义金额计 5 bps（f=0.0005）。首日开盘建仓乘以 1/(1+f)，后续开盘换仓乘以 (1−f)/(1+f)，沿用持仓无换仓费。隔夜收益归原持仓，日内收益归开盘后的持仓。未建模滑点。
Signal lag: 主信号按 EIA 标题的发布日期汇总，最早在严格晚于发布日期的首个交易日开盘生效。The main signal is first used at the open of the first trading session strictly after the EIA publication date (not on the publication date). 趋势对照仅在同一组开盘时点评估，使用执行日前最近一次收盘及再往前五个交易日的收盘，因此使用 t 日收盘的信号最早在 t+1 交易日开盘执行。
Price adjustment: Yahoo Finance auto_adjust=True, split- and distribution-adjusted Open and Close in units relative to the first warm-up close
Survivorship bias: 固定使用 USO 与 BIL 两只贯穿样本的基金。基金选择仍可能受事后可得性影响。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.02819742247864021, ann_vol=0.054295041468959866, sharpe=-0.4995555940559343, sharpe_excess=-0.5094987592755904, max_drawdown=-0.23259753375993697, total_return=-0.20425805100651184, turnover=6.592047251744572, n_trades=53
    oos: cagr=-0.03329981124739423, ann_vol=0.13642023475046533, sharpe=-0.1734391488174532, sharpe_excess=-0.3495132397307309, max_drawdown=-0.5062112009948347, total_return=-0.23260448498815056, turnover=9.624500502137458, n_trades=76
    full: cagr=-0.030724431650478, ann_vol=0.10340190645130357, sharpe=-0.2457468465858118, sharpe_excess=-0.3633118703087918, max_drawdown=-0.595311988036344, total_return=-0.3893511972356116, turnover=8.091904872451071, n_trades=129
    benchmark: cagr=-0.08951406083401903, ann_vol=0.3515875143212233, sharpe=-0.08793854182967842, sharpe_excess=-0.12245160909651197, max_drawdown=-0.9519782476409938, total_return=-0.7728613793898408, turnover=0.06317403688314015, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 179_crude_oil_news_sentiment\n\nuniverse:\n  oil: USO\n  defensive: BIL\n\nin_sample:\n  start: 2010-01-01\n  end: 2017-12-31\nout_of_sample:\n  start: 2018-01-01\n  end: 2025-10-31\n\neia_archive_url: https://www.eia.gov/petroleum/weekly/includes/archive.php\ndata_cache_dir: data_cache\noriginal_yahoo_snapshot:\n  fetched_date: "2026-11-25"\n  sha256:\n    USO: a5881e506f1cad751445bb284f52179d607f6ca6640c697b07741b02cb6d69c3\n    BIL: 90e0b7d9a7eaad219fc26c1db0cbefe75f6a1b73a79802a4d6b0984983b73fd5\n\nheadline_window_tokens: 4\nmax_holding_sessions: 5\ntrend_lookback_sessions: 5\ntransaction_cost_bps: 5\nslippage_bps: 2\ncash_buffer_bps: 10\nsessions_per_year: 252\n'
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
"""Backtest the C84 EIA headline proxy on USO and BIL."""

import json
import re
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urljoin, urlsplit
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from bs4 import BeautifulSoup
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)


ROOT = Path(__file__).resolve().parent
OSLO = ZoneInfo("Europe/Oslo")
ARCHIVE_EDITION = re.compile(
    r"/petroleum/weekly/archive/(?:20\d{2}/)?(\d{6})/(?:includes/)?"
    r"(analysis_print\.php|twipprint\.html|hometables_print\.php)"
)
EIA_MIGRATION = pd.Timestamp("2024-01-08")
PRICE_COLUMNS = ["Open", "Close"]
CLAUSE_BOUNDARY = re.compile(r"[,;:!?–—]|\b(?:as|while|whereas|but|amid|despite)\b")
NOUN_POLARITY = {
    "demand": 1,
    "production": -1,
    "supply": -1,
    "supplies": -1,
    "inventory": -1,
    "inventories": -1,
}
RISING = frozenset("""
    rise rises rising rose risen
    increase increases increased increasing
    grow grows growing grew grown growth
    build builds building built buildup buildups
""".split())
FALLING = frozenset("""
    fall falls falling fell fallen
    decrease decreases decreased decreasing
    drop drops dropped dropping
    decline declines declined declining
    draw draws drawing drew drawn drawdown drawdowns
""".split())


def cache_csv(frame, path):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                            suffix=".tmp", delete=False) as temporary:
        temp_path = Path(temporary.name)
        try:
            frame.to_csv(temporary, index=False, date_format="%Y-%m-%d")
            temporary.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)


def parse_archive(html, archive_url):
    soup = BeautifulSoup(html, "html.parser")
    archive_host = urlsplit(archive_url).hostname
    releases = []
    # Some edition links sit in cells without an enclosing row.
    for link in soup.select('div[id^="tabs-past-"] a[href]'):
        title = " ".join(link.get_text(" ", strip=True).split())
        source_url = urljoin(archive_url, link["href"])
        edition_url = urlsplit(source_url)
        edition = ARCHIVE_EDITION.fullmatch(edition_url.path)
        if title and edition and edition_url.hostname == archive_host:
            # Use the archived edition's date when the index's visible date disagrees.
            releases.append({
                "release_date": datetime.strptime(edition[1], "%y%m%d"),
                "title": title,
                "data_notice": edition[2] == "hometables_print.php",
                "source_url": source_url,
            })
    if not releases:
        raise RuntimeError("No dated headlines found in the EIA archive.")
    return (pd.DataFrame(releases)
            .drop_duplicates(subset=["release_date", "title", "data_notice"])
            .sort_values(["release_date", "source_url"], ignore_index=True))


def load_headlines(url, cache_dir):
    path = cache_dir / "eia_weekly_headlines.csv"
    if path.is_file():
        cached = pd.read_csv(path, parse_dates=["release_date"])
        if {"source_url", "fetched_at"}.issubset(cached.columns):
            return cached, snapshot_time(cached, path)
    with urlopen(url, timeout=30) as response:
        html = response.read()
    fetched_at = datetime.now(OSLO).isoformat(timespec="seconds")
    headlines = parse_archive(html, url).assign(fetched_at=fetched_at)
    cache_csv(headlines, path)
    return headlines, fetched_at


def snapshot_time(frame, path):
    dates = frame["fetched_at"].unique()
    if len(dates) != 1 or not isinstance(dates[0], str):
        raise RuntimeError(f"Missing or inconsistent fetch timestamp in {path.name}.")
    return dates[0]


def score_title(title, window):
    score = 0
    for clause in CLAUSE_BOUNDARY.split(title.lower()):
        tokens = re.findall(r"[a-z]+|\d+", clause)
        directions = {
            index: 1 if token in RISING else -1
            for index, token in enumerate(tokens) if token in RISING or token in FALLING
        }
        for index, token in enumerate(tokens):
            if token not in NOUN_POLARITY:
                continue
            nearby = [(abs(index - other), sign) for other, sign in directions.items()
                      if abs(index - other) <= window]
            if not nearby:
                continue
            closest = min(distance for distance, _ in nearby)
            signs = {sign for distance, sign in nearby if distance == closest}
            if len(signs) == 1:
                score += NOUN_POLARITY[token] * signs.pop()
    return score


class EquityHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def exchange_sessions(start, end):
    if start.year < 2009 or end.year > 2025:
        raise ValueError("The historical exchange calendar covers 2009–2025.")
    # Sandy and the national days of mourning for George H. W. Bush and Jimmy Carter.
    special_closures = pd.to_datetime(["2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])
    closures = EquityHolidays().holidays(start, end).union(special_closures)
    return pd.bdate_range(start, end).difference(closures)


def load_prices(ticker, start, end, cache_dir, sessions, original_snapshot):
    path = cache_dir / f"yahoo_{ticker}_{start.date()}_{end.date()}.csv"
    cached = path.is_file()
    if cached:
        prices = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
        if "fetched_at" in prices:
            fetched_at = snapshot_time(prices, path)
        elif sha256(path.read_bytes()).hexdigest() == original_snapshot["sha256"].get(ticker):
            fetched_at = original_snapshot["fetched_date"]
        else:
            fetched_at = "unknown (legacy cache without fetch metadata)"
    else:
        prices = yf.Ticker(ticker).history(
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d",
            auto_adjust=True,
            actions=False,
            timeout=20,
        )
        if prices.empty:
            raise RuntimeError(f"Yahoo Finance returned no daily prices for {ticker}.")
        prices.index = prices.index.tz_localize(None).normalize().rename("Date")
        fetched_at = datetime.now(OSLO).isoformat(timespec="seconds")

    prices = prices[PRICE_COLUMNS].sort_index()
    if (prices.empty or prices.index.has_duplicates
            or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any()):
        raise RuntimeError(f"Missing, duplicate or nonpositive daily prices for {ticker}.")
    if not prices.index.equals(sessions):
        missing = sessions.difference(prices.index).strftime("%Y-%m-%d").tolist()
        extra = prices.index.difference(sessions).strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"Exchange session mismatch for {ticker}. Missing: {missing[:10]} "
                           f"Extra: {extra[:10]}")
    if not cached:
        cache_csv(prices.reset_index().assign(fetched_at=fetched_at), path)
    # Any later adjustment factor common to both prices cancels in these units.
    return prices / prices["Close"].iloc[0], fetched_at


def event_positions(n_sessions, event_opens, entry_signal, hold):
    in_oil = np.zeros(n_sessions, dtype=bool)
    next_opens = np.r_[event_opens[1:], n_sessions]
    for start, next_open, enter in zip(event_opens, next_opens, entry_signal):
        if enter:
            in_oil[start:min(next_open, start + hold)] = True
    return in_oil


def backtest(opens, closes, prior_closes, in_oil, fee, slippage, cash_buffer):
    cash = previous_value = 1.0
    asset = None
    units = 0.0
    buy_multiplier = (1 + slippage) * (1 + fee)
    daily_results = []
    for open_row, close_row, prior_row, wants_oil in zip(
            opens.to_numpy(), closes.to_numpy(), prior_closes.to_numpy(), in_oil, strict=True):
        desired_asset = int(wants_oil)  # 0 = BIL, 1 = USO
        trades = 0
        turnover = 0.0
        scaled_down = False
        if desired_asset != asset:
            requested_units = ((1 - cash_buffer) * previous_value
                               / (prior_row[desired_asset] * buy_multiplier))
            open_value = cash
            sale_value = 0.0
            if asset is not None:
                open_value += units * open_row[asset]
                sale_value = units * open_row[asset] * (1 - slippage)
                cash += sale_value * (1 - fee)
                trades += 1
            affordable_units = ((1 - cash_buffer) * cash
                                / (open_row[desired_asset] * buy_multiplier))
            units = min(requested_units, affordable_units)
            scaled_down = units < requested_units
            purchase_value = units * open_row[desired_asset] * (1 + slippage)
            cash -= purchase_value * (1 + fee)
            if cash < 0:
                raise RuntimeError("Simulated fill exceeded available cash.")
            turnover = (sale_value + purchase_value) / open_value
            trades += 1
            asset = desired_asset
        value = cash + units * close_row[asset]
        daily_results.append((value / previous_value - 1, turnover, trades,
                              wants_oil, cash / value, scaled_down))
        previous_value = value
    return pd.DataFrame(
        daily_results, index=opens.index,
        columns=["return", "turnover", "trades", "in_oil", "cash_weight", "scaled_down"],
    )


def metrics(portfolio, cash_returns, sessions_per_year):
    daily = portfolio["return"]
    excess = daily - cash_returns
    cumulative_value = (1 + daily).cumprod()
    volatility = daily.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    values = {
        "cagr": cumulative_value.iloc[-1] ** (sessions_per_year / len(daily)) - 1,
        "ann_vol": volatility * np.sqrt(sessions_per_year),
        "sharpe": daily.mean() / volatility * np.sqrt(sessions_per_year) if volatility > 0 else None,
        "sharpe_excess": (excess.mean() / excess_volatility * np.sqrt(sessions_per_year)
                          if excess_volatility > 0 else None),
        "max_drawdown": (cumulative_value / cumulative_value.cummax().clip(lower=1) - 1).min(),
        "total_return": cumulative_value.iloc[-1] - 1,
        "turnover": portfolio["turnover"].mean() * sessions_per_year,
        "n_trades": int(portfolio["trades"].sum()),
        "oil_exposure": portfolio["in_oil"].mean(),
    }
    for name, value in values.items():
        if name != "n_trades":
            values[name] = float(value) if value is not None and np.isfinite(value) else None
    return values


def print_metrics(reports):
    rows = [
        {"Portfolio": name, "Period": period.upper(), **stats}
        for name, splits in reports.items() for period, stats in splits.items()
    ]
    table = pd.DataFrame(rows)
    percent = ("cagr", "ann_vol", "max_drawdown", "total_return", "turnover", "oil_exposure")
    formatters = {column: "{:.2%}".format for column in percent}
    formatters["sharpe"] = "{:.2f}".format
    formatters["sharpe_excess"] = "{:.2f}".format
    print(table.to_string(index=False, formatters=formatters, na_rep="n/a"))


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    periods = {
        label: (pd.Timestamp(config[key]["start"]), pd.Timestamp(config[key]["end"]))
        for label, key in (("is", "in_sample"), ("oos", "out_of_sample"))
    }
    start, is_end = periods["is"]
    oos_start, end = periods["oos"]
    if not start <= is_end < oos_start <= end:
        raise ValueError("IS must end before OOS starts, and both ranges must be ordered.")
    periods["full"] = (start, end)

    window = int(config["headline_window_tokens"])
    hold = int(config["max_holding_sessions"])
    trend_lookback = int(config["trend_lookback_sessions"])
    sessions_per_year = int(config["sessions_per_year"])
    fee = float(config["transaction_cost_bps"]) / 10_000
    slippage = float(config["slippage_bps"]) / 10_000
    cash_buffer = float(config["cash_buffer_bps"]) / 10_000
    if min(window, hold, trend_lookback, sessions_per_year) < 1:
        raise ValueError("Windows and sessions per year must be positive.")
    if not (0 <= fee < 1 and 0 <= slippage < 1 and 0 < cash_buffer < 1):
        raise ValueError("Fees and slippage must be in [0, 10,000) bps and buffer in (0, 10,000) bps.")

    cache_dir = ROOT / config["data_cache_dir"]
    cache_dir.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir / "yfinance"))
    headlines, eia_fetched_at = load_headlines(config["eia_archive_url"], cache_dir)
    headlines = headlines.loc[headlines["release_date"].between(start, end)].copy()
    if (headlines.empty or headlines["release_date"].min() > start + pd.Timedelta(days=14)
            or headlines["release_date"].max() < end - pd.Timedelta(days=14)):
        raise RuntimeError("The configured dates extend beyond the available EIA archive.")
    headlines["score"] = [
        0 if notice else score_title(title, window)
        for notice, title in zip(headlines["data_notice"], headlines["title"])
    ]
    releases = headlines.groupby("release_date")["score"].sum()

    oil_ticker = config["universe"]["oil"]
    defensive_ticker = config["universe"]["defensive"]
    warmup_start = start - pd.Timedelta(days=max(30, 3 * trend_lookback))
    price_sessions = exchange_sessions(warmup_start, end)
    snapshot = config["original_yahoo_snapshot"]
    oil, oil_fetched_at = load_prices(oil_ticker, warmup_start, end, cache_dir,
                                      price_sessions, snapshot)
    defensive, defensive_fetched_at = load_prices(defensive_ticker, warmup_start, end, cache_dir,
                                                  price_sessions, snapshot)
    prior_trend = oil["Close"].pct_change(trend_lookback).shift(1)
    closes = pd.DataFrame({defensive_ticker: defensive["Close"], oil_ticker: oil["Close"]})
    prior_closes = closes.shift(1).loc[start:end]
    closes = closes.loc[start:end]
    opens = pd.DataFrame({defensive_ticker: defensive["Open"], oil_ticker: oil["Open"]}).loc[start:end]
    sessions = closes.index
    if any(not len(sessions[(sessions >= first) & (sessions <= last)])
           for first, last in periods.values()):
        raise RuntimeError("At least one reporting period has no trading sessions.")

    event_opens = sessions.searchsorted(releases.index, side="right")
    eligible = event_opens < len(sessions)
    scores = pd.Series(releases.to_numpy()[eligible], index=event_opens[eligible])
    # Releases are in date order, so the latest release wins when opens coincide.
    superseded = scores.index.duplicated(keep="last")
    scores = scores.loc[~superseded]
    event_opens = scores.index.to_numpy()
    trend_on_release = prior_trend.reindex(sessions[event_opens])
    if trend_on_release.isna().any():
        raise RuntimeError("Insufficient prior prices for the same-release trend benchmark.")
    allocations = {
        "EIA positive": event_positions(len(sessions), event_opens, scores > 0, hold),
        f"{oil_ticker} buy & hold": np.ones(len(sessions), dtype=bool),
        f"{defensive_ticker} buy & hold": np.zeros(len(sessions), dtype=bool),
        "Prior-week trend": event_positions(len(sessions), event_opens, trend_on_release > 0, hold),
        "EIA negative (long oil)": event_positions(len(sessions), event_opens, scores < 0, hold),
    }

    print(f"{config['strategy']} | {oil_ticker}/{defensive_ticker} | "
          f"{config['transaction_cost_bps']:g} bps fee + {config['slippage_bps']:g} bps slippage per side")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"DATA_FETCHED_AT: EIA={eia_fetched_at}, {oil_ticker}={oil_fetched_at}, "
          f"{defensive_ticker}={defensive_fetched_at}")
    print("PRICE_ADJUSTMENT: Yahoo Finance auto_adjust=True, split- and distribution-adjusted "
          "Open and Close in units relative to the first warm-up close")
    print(f"EIA releases superseded before the same open: {superseded.sum()}")
    print("EIA archive counts by release date")
    print(f"{'Period':<16}{'Titles':>8}{'Notices':>9}{'Positive':>10}{'Negative':>10}{'Neutral':>9}")
    count_periods = {
        **periods,
        "pre_migration": (start, min(end, EIA_MIGRATION - pd.Timedelta(days=1))),
        "post_migration": (max(start, EIA_MIGRATION), end),
    }
    for label, (first, last) in count_periods.items():
        titles = headlines.loc[headlines["release_date"].between(first, last)]
        scores_in_period = releases.loc[first:last]
        print(f"{label.upper():<16}{len(titles):>8}{titles['data_notice'].sum():>9}"
              f"{(scores_in_period > 0).sum():>10}{(scores_in_period < 0).sum():>10}"
              f"{(scores_in_period == 0).sum():>9}")

    cash_returns = (closes[defensive_ticker] / prior_closes[defensive_ticker] - 1)
    cash_returns.iloc[0] = closes[defensive_ticker].iloc[0] / opens[defensive_ticker].iloc[0] - 1
    reports = {}
    for name, in_oil in allocations.items():
        portfolio = backtest(opens, closes, prior_closes, in_oil, fee, slippage, cash_buffer)
        reports[name] = {
            label: metrics(portfolio.loc[first:last], cash_returns.loc[first:last], sessions_per_year)
            for label, (first, last) in periods.items()
        }
        if name == "EIA positive":
            print(f"Opening fills scaled down: {portfolio['scaled_down'].sum()}, "
                  f"minimum closing cash weight: {portfolio['cash_weight'].min():.4%}")
    print("\nCost-adjusted performance (trade count includes buys and sells)")
    print_metrics(reports)

    strategy = reports["EIA positive"]
    benchmark = reports[f"{oil_ticker} buy & hold"]
    result = {
        "strategy": config["strategy"],
        "data_range": f"{sessions[0].date()} to {sessions[-1].date()}",
        **strategy,
        "benchmark": {"ticker": oil_ticker, **benchmark["full"],
                      "is": benchmark["is"], "oos": benchmark["oos"],
                      "cash_proxy": {"ticker": defensive_ticker,
                                     **reports[f"{defensive_ticker} buy & hold"]}},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
