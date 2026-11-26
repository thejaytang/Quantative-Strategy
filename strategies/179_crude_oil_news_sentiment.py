"""179_crude_oil_news_sentiment
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/179_crude_oil_news_sentiment.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 2b23c5bbf706aea933c0cba5f5805efb450ee206 on 2026-11-26 14:26 Europe/Oslo, exit 0, output sha256 1b70f32c8299146a50139a2accccf0d7b05862099e9988dc81497b8bf46b8b2d
Data source: 美国能源信息署（EIA）公开的 Past Weeks in Petroleum 周度石油市场标题及例行数据公告，发布日期从归档版次链接提取。Yahoo Finance 经 yfinance 提供 USO 和 BIL 的日频开盘价与收盘价。（config.yaml:3–5、14；strategy.py:55–84、109–125、235–239；README.md:9）
Data fetched: 2026-11-26 14:26 Europe/Oslo (run time)
Sample period: 配置日历区间为 2010-01-01 至 2025-10-31。日志记录的实际回测交易日区间为 2010-01-04 至 2025-10-31。（config.yaml:7–12；strategy.py:237–245、287–289；logs/run_20261125_120733.log:22）   data_range (from output): 2010-01-04 to 2025-10-31
IS/OOS split: 按日历日期切分，IS 为 2010-01-01 至 2017-12-31，OOS 为 2018-01-01 至 2025-10-31，端点均包含。先计算全样本的同一持仓路径，再分段汇报，OOS 承接原有持仓。README 将 OOS 描述为回顾性留出。（config.yaml:7–12；strategy.py:203–211、275–281；README.md:15、19）
Costs and slippage: 每笔买或卖按成交名义金额计 5 bps（f=0.0005）。首日开盘建仓乘以 1/(1+f)，后续开盘换仓乘以 (1−f)/(1+f)，沿用持仓无换仓费。隔夜收益归原持仓，日内收益归开盘后的持仓。未建模滑点。（config.yaml:20；strategy.py:149–165、217；README.md:15、23；logs/run_20261125_120733.log:1）
Signal lag: 主信号按 EIA 标题的发布日期汇总，最早在严格晚于发布日期的首个交易日开盘生效。趋势对照仅在同一组开盘时点评估，使用执行日前最近一次收盘及再往前五个交易日的收盘，因此使用 t 日收盘的信号最早在 t+1 交易日开盘执行。（config.yaml:19；strategy.py:229–233、242–261；README.md:13–14）
Price adjustment: yfinance 下载时设置 auto_adjust=True 和 actions=False，保留复权 Open 与 Close。缓存命中时直接读取对应价格 CSV，代码未重新复权。（strategy.py:109–125、135–137；README.md:15）
Survivorship bias: 固定使用 USO 与 BIL 两只贯穿样本的基金。基金选择仍可能受事后可得性影响。（config.yaml:3–5；strategy.py:235–241；README.md:23）
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.04037611994920132, ann_vol=0.0491779696598379, sharpe=-0.8132646690425923, max_drawdown=-0.28921239773751417, total_return=-0.28051547581475855, n_trades=45
    oos: cagr=-0.03891400097621467, ann_vol=0.1370356456210015, sharpe=-0.21434493771707433, max_drawdown=-0.508513414896028, total_return=-0.26676265242027675, n_trades=72
    full: cagr=-0.039653231117312226, ann_vol=0.10250725171295638, sharpe=-0.3389139440305118, max_drawdown=-0.6423721382042386, total_return=-0.4724470758617526, n_trades=117
    benchmark: cagr=-0.08969906133865335, ann_vol=0.3532998025682416, sharpe=-0.08630966182464592, max_drawdown=-0.9528239193254501, total_return=-0.7735897414071253, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 179_crude_oil_news_sentiment\n\nuniverse:\n  oil: USO\n  defensive: BIL\n\nin_sample:\n  start: 2010-01-01\n  end: 2017-12-31\nout_of_sample:\n  start: 2018-01-01\n  end: 2025-10-31\n\neia_archive_url: https://www.eia.gov/petroleum/weekly/includes/archive.php\ndata_cache_dir: data_cache\n\nheadline_window_tokens: 4\nmax_holding_sessions: 5\ntrend_lookback_sessions: 5\ntransaction_cost_bps: 5\nsessions_per_year: 252\n'
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
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent
ARCHIVE_DATE = re.compile(r"/archive/(?:20\d{2}/)?(\d{6})/")
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


def parse_archive(html):
    soup = BeautifulSoup(html, "html.parser")
    releases = []
    for row in soup.select('div[id^="tabs-past-"] tr'):
        cells = row.find_all("td", recursive=False)
        if len(cells) != 3:
            continue
        for link in cells[-1].find_all("a", href=True):
            title = link.get_text(" ", strip=True)
            date_match = ARCHIVE_DATE.search(link["href"])
            if title and date_match:
                # Edition URLs also disambiguate misprinted release dates in the index.
                releases.append({
                    "release_date": datetime.strptime(date_match[1], "%y%m%d"),
                    "title": title,
                    "data_notice": link["href"].endswith("/hometables_print.php"),
                })
    if not releases:
        raise RuntimeError("No dated headlines found in the EIA archive.")
    return pd.DataFrame(releases).drop_duplicates().sort_values("release_date")


def load_headlines(url, cache_dir):
    path = cache_dir / "eia_weekly_headlines.csv"
    if path.is_file():
        return pd.read_csv(path, parse_dates=["release_date"])
    with urlopen(url, timeout=30) as response:
        headlines = parse_archive(response.read())
    cache_csv(headlines, path)
    return headlines


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


def load_prices(ticker, start, end, cache_dir):
    path = cache_dir / f"yahoo_{ticker}_{start.date()}_{end.date()}.csv"
    if path.is_file():
        prices = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
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
        prices = prices[["Open", "Close"]]
        prices.index = prices.index.tz_localize(None).normalize().rename("Date")

    prices = prices.sort_index()
    if (prices.empty or prices.index.has_duplicates
            or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any()):
        raise RuntimeError(f"Missing, duplicate or nonpositive daily prices for {ticker}.")
    if (prices.index[0] > start + pd.Timedelta(days=7)
            or prices.index[-1] < end - pd.Timedelta(days=7)):
        raise RuntimeError(f"Yahoo Finance history for {ticker} does not cover {start.date()} "
                           f"through {end.date()}.")
    if not path.is_file():
        cache_csv(prices.reset_index(), path)
    return prices


def event_positions(n_sessions, event_opens, entry_signal, hold):
    in_oil = np.zeros(n_sessions, dtype=bool)
    next_opens = np.r_[event_opens[1:], n_sessions]
    for start, next_open, enter in zip(event_opens, next_opens, entry_signal):
        if enter:
            in_oil[start:min(next_open, start + hold)] = True
    return in_oil


def backtest(oil, defensive, in_oil, fee):
    previously_in_oil = np.r_[False, in_oil[:-1]]
    overnight = np.where(
        previously_in_oil, oil["Open"] / oil["Close"].shift(),
        defensive["Open"] / defensive["Close"].shift()
    )
    overnight[0] = 1.0
    intraday = np.where(in_oil, oil["Close"] / oil["Open"],
                        defensive["Close"] / defensive["Open"])
    switched = np.r_[False, in_oil[1:] != in_oil[:-1]]
    cost_multiplier = (1 - fee * switched) / (1 + fee * switched)
    cost_multiplier[0] = 1 / (1 + fee)
    trades = 2 * switched.astype(int)
    trades[0] = 1
    return pd.DataFrame({
        "return": overnight * cost_multiplier * intraday - 1,
        "trades": trades,
        "in_oil": in_oil,
    }, index=oil.index)


def metrics(portfolio, sessions_per_year):
    daily = portfolio["return"]
    equity = (1 + daily).cumprod()
    volatility = daily.std(ddof=1)
    values = {
        "cagr": equity.iloc[-1] ** (sessions_per_year / len(daily)) - 1,
        "ann_vol": volatility * np.sqrt(sessions_per_year),
        "sharpe": daily.mean() / volatility * np.sqrt(sessions_per_year) if volatility > 0 else None,
        "max_drawdown": (equity / equity.cummax().clip(lower=1) - 1).min(),
        "total_return": equity.iloc[-1] - 1,
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
    percent = ("cagr", "ann_vol", "max_drawdown", "total_return", "oil_exposure")
    formatters = {column: "{:.2%}".format for column in percent}
    formatters["sharpe"] = "{:.2f}".format
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
    if min(window, hold, trend_lookback, sessions_per_year) < 1 or not 0 <= fee < 1:
        raise ValueError("Windows must be positive and costs must be in [0, 10,000) bps.")

    cache_dir = ROOT / config["data_cache_dir"]
    cache_dir.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir / "yfinance"))
    headlines = load_headlines(config["eia_archive_url"], cache_dir)
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
    oil = load_prices(oil_ticker, warmup_start, end, cache_dir)
    defensive = load_prices(defensive_ticker, warmup_start, end, cache_dir)
    if not oil.index.equals(defensive.index):
        raise RuntimeError("The ETFs have different session coverage.")
    prior_trend = oil["Close"].pct_change(trend_lookback).shift(1)
    oil = oil.loc[start:end]
    defensive = defensive.loc[start:end]
    sessions = oil.index
    if any(not len(sessions[(sessions >= first) & (sessions <= last)])
           for first, last in periods.values()):
        raise RuntimeError("At least one reporting period has no trading sessions.")

    opens = sessions.searchsorted(releases.index, side="right")
    eligible = opens < len(sessions)
    opens = opens[eligible]
    scores = releases.to_numpy()[eligible]
    trend_on_release = prior_trend.reindex(sessions[opens])
    if trend_on_release.isna().any():
        raise RuntimeError("Insufficient prior prices for the same-release trend benchmark.")
    allocations = {
        "EIA positive": event_positions(len(sessions), opens, scores > 0, hold),
        f"{oil_ticker} buy & hold": np.ones(len(sessions), dtype=bool),
        "Prior-week trend": event_positions(len(sessions), opens, trend_on_release > 0, hold),
        "EIA negative (long oil)": event_positions(len(sessions), opens, scores < 0, hold),
    }

    print(f"{config['strategy']} | {oil_ticker}/{defensive_ticker} | "
          f"{config['transaction_cost_bps']:g} bps per side")
    print("EIA archive counts by release date")
    print(f"{'Period':<8}{'Titles':>8}{'Notices':>9}{'Positive':>10}{'Negative':>10}{'Neutral':>9}")
    for label, (first, last) in periods.items():
        titles = headlines.loc[headlines["release_date"].between(first, last)]
        scores_in_period = releases.loc[first:last]
        print(f"{label.upper():<8}{len(titles):>8}{titles['data_notice'].sum():>9}"
              f"{(scores_in_period > 0).sum():>10}{(scores_in_period < 0).sum():>10}"
              f"{(scores_in_period == 0).sum():>9}")

    reports = {}
    for name, in_oil in allocations.items():
        portfolio = backtest(oil, defensive, in_oil, fee)
        reports[name] = {
            label: metrics(portfolio.loc[first:last], sessions_per_year)
            for label, (first, last) in periods.items()
        }
    print("\nCost-adjusted performance (trade count includes buys and sells)")
    print_metrics(reports)

    strategy = reports["EIA positive"]
    benchmark = reports[f"{oil_ticker} buy & hold"]
    result = {
        "strategy": config["strategy"],
        "data_range": f"{sessions[0].date()} to {sessions[-1].date()}",
        **strategy,
        "benchmark": {"ticker": oil_ticker, **benchmark["full"],
                      "is": benchmark["is"], "oos": benchmark["oos"]},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
