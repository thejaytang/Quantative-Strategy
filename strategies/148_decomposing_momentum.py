"""148_decomposing_momentum
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/148_decomposing_momentum.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 2b23c5bbf706aea933c0cba5f5805efb450ee206 on 2026-11-26 14:26 Europe/Oslo, exit 0, output sha256 a5b5ff3d6bc62b8f1e6e785aa48a7a96dc6b0319b998810ba876d538b06e8618
Data source: Yahoo Finance 免费日线，经 yfinance 读取复权开盘价和收盘价，优先使用同一请求的本地缓存。固定 US24 股票池为 AAPL、MSFT、AMZN、GOOGL、META、NVDA、JPM、BAC、GS、MS、XOM、CVX、COP、SLB、WMT、COST、PG、KO、PEP、JNJ、PFE、MRK、HD、LOW，基准为 SPY，现金替代为 BIL。
Data fetched: 2026-11-26 14:26 Europe/Oslo (run time)
Sample period: 请求的数据区间为 2005-12-01 至 2025-12-31，2005-12-01 起为预热数据。日志中的实际评估区间为 2007-01-03 至 2025-12-31。 data_range (from output): 2007-01-03 to 2025-12-31
IS/OOS split: 固定日历切分，样本内 2007-01-01 至 2016-12-31，样本外 2017-01-01 至 2025-12-31。各组合连续模拟，样本外承接已有持仓，按日期切分日收益后计算指标。README 记载日期预先固定，未拟合参数或依据样本外结果筛选。
Costs and slippage: 基准费率为每个买入或卖出方向按成交金额收取 5 bps，另测 0 和 15 bps。按开盘价计算调仓前净值和各标的净买卖金额，从组合资金扣费，SPY 初始买入也计费。成交使用复权开盘价，未单独建模滑点、买卖价差或市场冲击。
Signal lag: 月末交易日 t 收盘后生成目标。股票排序使用截至 t 往前第 21 个交易日的 231 个复权收盘价。合格股票不足时，以 t 日 BIL 收盘价是否存在决定配置 BIL 或现金。目标最早在下一交易日开盘成交。现有持仓或目标资产缺开盘价时整篮子顺延，直至开盘价齐备或被下一月目标替换。
Price adjustment: 下载设置为 auto_adjust=True，复权 Close 用于形成期排名和日终估值，复权 Open 用于模拟成交。收盘价缺失时估值取当日开盘价，再缺失则沿用最近可用估值。
Survivorship bias: 固定 US24 由事后已知的存续股票组成，有幸存者偏差及股票池选择偏差，时间上的样本外切分无法消除。形成窗口还要求股票的收盘报价齐全。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.1415516409645261, ann_vol=0.2560315509766984, sharpe=0.6451512424879071, max_drawdown=-0.6215985005743061, total_return=2.7540436042754752, turnover=57.130399335021416, n_trades=875
    oos: cagr=0.28914342239072655, ann_vol=0.2543400123344971, sharpe=1.1263023796500553, max_drawdown=-0.31591396161312346, total_return=8.774283096333589, turnover=43.28566725478996, n_trades=766
    full: cagr=0.20916162811949102, ann_vol=0.25523430287511967, sharpe=0.8720364930091413, max_drawdown=-0.6215985005743061, total_return=35.69308494416898, turnover=100.41606658981138, n_trades=1641
    benchmark full: cagr=0.10646986469321074, ann_vol=0.1975891865355508, sharpe=0.6110029502087301, max_drawdown=-0.551894372777572, total_return=5.814865764647665, turnover=0.9995002498750625, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 148_decomposing_momentum\nuniverse: [AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, BAC, GS, MS, XOM, CVX, COP, SLB, WMT, COST, PG, KO, PEP, JNJ, PFE, MRK, HD, LOW]\nbenchmark: SPY\ncash_etf: BIL\ndownload_start: 2005-12-01\nin_sample:\n  start: 2007-01-01\n  end: 2016-12-31\nout_of_sample:\n  start: 2017-01-01\n  end: 2025-12-31\nformation_sessions: 231\nskip_sessions: 21\ntop_n: 6\nmin_eligible: 16\n# Per dollar bought or sold. Sensitivities are fixed before the test.\ntransaction_cost_bps: 5\ncost_sensitivity_bps: [0, 15]\n'
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
"""Backtest monthly formation-start-to-high momentum on the frozen US24 basket."""

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SIGNALS = ("start_to_high", "high_to_end", "standard_momentum")
SESSIONS_PER_YEAR = 252


def load_prices(tickers, start, end):
    """Cache adjusted daily open and close prices for the exact request."""
    CACHE.mkdir(exist_ok=True)
    universe_key = hashlib.sha256(",".join(tickers).encode()).hexdigest()[:12]
    cache_file = CACHE / f"yahoo_adjusted_{start:%Y%m%d}_{end:%Y%m%d}_{universe_key}.csv.gz"
    if cache_file.exists():
        prices = pd.read_csv(cache_file, header=[0, 1], index_col=0, parse_dates=True)
    else:
        yf.set_tz_cache_location(str(CACHE / "yfinance"))
        # Start with one symbol so yfinance initializes its SQLite cache serially.
        download_args = dict(start=start, end=end + pd.Timedelta(days=1),
                             auto_adjust=True, progress=False, timeout=8)
        first = yf.download(tickers[:1], threads=False, **download_args)
        rest = yf.download(tickers[1:], threads=4, **download_args)
        raw = pd.concat([first, rest], axis=1)
        columns = pd.MultiIndex.from_product([["Open", "Close"], tickers])
        prices = raw.reindex(columns=columns)
        for ticker in missing_symbols(prices, tickers):
            retry = yf.download([ticker], threads=False, **download_args).reindex(
                columns=[("Open", ticker), ("Close", ticker)]
            )
            prices = prices.combine_first(retry)
        prices = prices.reindex(columns=columns).sort_index()
        missing = missing_symbols(prices, tickers)
        if missing:
            raise RuntimeError(f"Yahoo returned no usable prices for {', '.join(missing)}")
        with NamedTemporaryFile(dir=CACHE, prefix=".prices-", delete=False) as temporary:
            temp_path = Path(temporary.name)
        try:
            prices.to_csv(temp_path, compression="gzip")
            temp_path.replace(cache_file)
        finally:
            temp_path.unlink(missing_ok=True)

    prices = prices.where(np.isfinite(prices) & prices.gt(0)).loc[start:end]
    missing = missing_symbols(prices, tickers)
    if missing:
        raise RuntimeError(f"Missing usable adjusted prices for {', '.join(missing)}")
    return prices["Open"], prices["Close"]


def missing_symbols(prices, tickers):
    has_price = (np.isfinite(prices) & prices.gt(0)).any()
    return [ticker for ticker in tickers
            if not (has_price.get(("Open", ticker), False)
                    and has_price.get(("Close", ticker), False))]


def monthly_targets(closes, config):
    """Build target baskets known at each month-end, keyed by the next session."""
    schedules = {signal: {} for signal in SIGNALS}
    months = closes.index.to_period("M")
    stocks = closes[sorted(config["universe"])]
    cash_etf = config["cash_etf"]

    for signal_row in np.flatnonzero(months[:-1] != months[1:]):
        end = signal_row - config["skip_sessions"]
        start = end - config["formation_sessions"] + 1
        fill_date = closes.index[signal_row + 1]
        fallback = (cash_etf,) if pd.notna(closes[cash_etf].iat[signal_row]) else ()
        window = stocks.iloc[start:end + 1].dropna(axis=1) if start >= 0 else None
        if window is None or window.shape[1] < config["min_eligible"]:
            for schedule in schedules.values():
                schedule[fill_date] = fallback
            continue

        first, high, last = window.iloc[0], window.max(), window.iloc[-1]
        scores = (np.log(high / first), np.log(last / high), np.log(last / first))
        for signal, score in zip(SIGNALS, scores):
            # Stable sorting of average ranks breaks allocation ties by ticker.
            ranks = score.rank(method="average", ascending=False)
            ordered = ranks.sort_values(kind="stable").index
            schedules[signal][fill_date] = tuple(ordered[:config["top_n"]])
    return schedules


def simulate(opens, closes, dates, schedule, cost_bps):
    """Value close-to-close, rebalance at the open and let weights drift between fills."""
    fee = cost_bps / 10_000
    columns = {ticker: i for i, ticker in enumerate(opens.columns)}
    opening_prices = opens.loc[dates].to_numpy()
    marks = closes.combine_first(opens).ffill().loc[dates].to_numpy()
    missing_close = closes.loc[dates].isna().to_numpy()
    units = np.zeros(len(columns))
    previous_nav = cash = 1.0
    earlier = [date for date in schedule if date < dates[0]]
    pending = schedule[max(earlier)] if earlier else None
    observations = []

    for i, date in enumerate(dates):
        pending = schedule.get(date, pending)
        turnover = trades = delayed = 0
        if pending is not None:
            selected = [columns[ticker] for ticker in pending]
            target_weights = np.zeros(len(columns))
            if selected:
                target_weights[selected] = 1 / len(selected)

            relevant = (units > 0) | (target_weights > 0)
            opening = opening_prices[i]
            if np.isfinite(opening[relevant]).all():
                safe_open = np.where(relevant, opening, 1.0)
                values = units * safe_open
                capital = cash + values.sum()
                old_weights = values / capital
                # Fees are paid from current capital. Solve for the investable NAV fraction.
                fraction = 1.0
                if fee:
                    fraction = brentq(
                        lambda x: x + fee * np.abs(x * target_weights - old_weights).sum() - 1,
                        0.0, 1.0,
                    )
                new_weights = fraction * target_weights
                changes = np.abs(new_weights - old_weights)
                turnover = changes.sum()
                trades = np.count_nonzero(changes > 1e-12)
                units = capital * new_weights / safe_open
                cash = 0.0 if selected else capital * fraction
                pending = None
            else:
                delayed = 1

        held = units > 0
        nav = cash + units[held] @ marks[i, held]
        observations.append((nav / previous_nav - 1, turnover, trades,
                             int(missing_close[i, held].any()), delayed))
        previous_nav = nav

    return pd.DataFrame(observations, index=dates, columns=[
        "return", "turnover", "n_trades", "missing_close_sessions", "delayed_fill_sessions",
    ])


def metrics(frame):
    returns = frame["return"].to_numpy()
    equity = np.r_[1.0, np.cumprod(1 + returns)]
    daily_vol = returns.std(ddof=1) if len(returns) > 1 else np.nan
    result = {
        "cagr": equity[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1,
        "ann_vol": daily_vol * np.sqrt(SESSIONS_PER_YEAR),
        "sharpe": returns.mean() / daily_vol * np.sqrt(SESSIONS_PER_YEAR) if daily_vol > 0 else np.nan,
        "max_drawdown": (equity / np.maximum.accumulate(equity) - 1).min(),
        "total_return": equity[-1] - 1,
        "turnover": frame["turnover"].sum(),
    }
    return {
        **{key: float(value) if np.isfinite(value) else None for key, value in result.items()},
        **{key: int(frame[key].sum()) for key in
           ("n_trades", "missing_close_sessions", "delayed_fill_sessions")},
    }


def print_results(result, reports, names):
    print(f"{result['strategy']} | {result['data_range']}")
    print(f"{'Portfolio':<21} {'bp/side':>7} {'Period':<6} {'CAGR':>8} {'Ann vol':>8} "
          f"{'Sharpe':>8} {'Max DD':>8} {'Return':>9} {'Turnover':>9} {'Trades':>7}")
    print("-" * 101)
    metric_keys = ("cagr", "ann_vol", "sharpe", "max_drawdown", "total_return", "turnover")
    formats = (".2%", ".2%", ".2f", ".2%", ".2%", ".2f")
    for cost, portfolios in reports.items():
        for portfolio, periods in portfolios.items():
            for period, row in periods.items():
                numbers = [format(row[key], fmt) if row[key] is not None else "—"
                           for key, fmt in zip(metric_keys, formats)]
                print(f"{names[portfolio]:<21} {cost:>7g} {period.upper():<6} "
                      f"{numbers[0]:>8} {numbers[1]:>8} {numbers[2]:>8} {numbers[3]:>8} "
                      f"{numbers[4]:>9} {numbers[5]:>8}x {row['n_trades']:>7}")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False, separators=(",", ":")))


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    download_start = pd.Timestamp(config["download_start"])
    if not download_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Use chronological download, in-sample and out-of-sample dates.")
    if not 1 <= config["top_n"] <= config["min_eligible"] <= len(set(config["universe"])):
        raise ValueError("Expected 1 <= top_n <= min_eligible <= universe size.")
    if config["formation_sessions"] < 2 or config["skip_sessions"] < 0:
        raise ValueError("Expected at least two formation closes and a nonnegative skip.")
    cost = config["transaction_cost_bps"]
    costs = dict.fromkeys([cost, *config["cost_sensitivity_bps"]])
    if any(not 0 <= value < 10_000 for value in costs):
        raise ValueError("Transaction costs must be at least 0 and below 10,000 bps.")

    benchmark = config["benchmark"]
    tickers = sorted(set(config["universe"]) | {benchmark, config["cash_etf"]})
    opens, closes = load_prices(tickers, download_start, oos_end)
    calendar = closes[benchmark].dropna().index
    if (calendar[0] > download_start + pd.Timedelta(days=7)
            or calendar[-1] < oos_end - pd.Timedelta(days=7)):
        raise RuntimeError("Benchmark prices do not cover the requested date range.")
    dates = calendar[(calendar >= is_start) & (calendar <= oos_end)]
    ranges = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    if any(not ((dates >= start) & (dates <= end)).any() for start, end in ranges.values()):
        raise RuntimeError("One or more evaluation periods have no trading sessions.")
    opens, closes = opens.loc[calendar], closes.loc[calendar]
    schedules = monthly_targets(closes, config)
    schedules["benchmark"] = {dates[0]: (benchmark,)}
    reports = {}
    for value in costs:
        reports[value] = {}
        for portfolio, schedule in schedules.items():
            frame = simulate(opens, closes, dates, schedule, value)
            reports[value][portfolio] = {period: metrics(frame.loc[start:end])
                                         for period, (start, end) in ranges.items()}

    primary = reports[cost]
    result = {
        "strategy": config["strategy"],
        "data_range": f"{dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}",
        **primary["start_to_high"],
        "benchmark": {"name": benchmark, **primary["benchmark"]},
        "comparisons": {name: primary[name] for name in SIGNALS[1:]},
        "transaction_cost_bps": cost,
        "cost_sensitivity": {f"{value:g}": reports[value] for value in costs if value != cost},
    }
    names = dict(zip(schedules, ("Start-to-high", "High-to-end", "Standard momentum", f"{benchmark} buy-and-hold")))
    print_results(result, reports, names)


if __name__ == "__main__":
    main()
