"""100_etf_asset_momentum_corr_hedge
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/100_etf_asset_momentum_corr_hedge.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit e45e81ae3a3f162e8a4d3aa6588411ed39316f30 on 2026-11-26 13:10 Europe/Oslo, exit 0, output sha256 2609c110c5b77e013237670c949bfffe62636743c5021969eee4a182d1d8decf
Data source: Yahoo Finance（yfinance）ETF 日线开盘价和收盘价，优先读取本地行情缓存。风险资产为 SPY、IWM、EFA、EEM、IYR、QQQ、LQD、IEF、TIP、GLD、USO、DBC、FXE，现金收益代理为 BIL。（config.yaml:4-5，strategy.py:21-26，strategy.py:46-55，strategy.py:205-206，README.md:11-12）
Data fetched: 2026-11-26 13:10 Europe/Oslo (run time)
Sample period: 日志记录的实际回测区间为 2008-07-01 至 2026-11-24。行情请求起始日为 2007-06-01，用于信号预热。（logs/run_20261125_094846.log:1，config.yaml:6-12，strategy.py:213-218，README.md:16）   data_range (from output): 2008-07-01 to 2026-11-24
IS/OOS split: 样本内为 2008-07-01 至 2023-02-28，样本外为 2023-03-01 至 2026-11-24。每个情景先连续模拟持仓，再按区间切分收益。README 说明分界取自原文资料区间终点，样本外起始日早于原文发表，仍属历史回测。（config.yaml:7-12，strategy.py:162-163，strategy.py:200-204，strategy.py:234-238，README.md:16-17，logs/run_20261125_094846.log:2）
Costs and slippage: 基准交易费为每美元买入或卖出名义金额 5 bps，含空头回补，在调仓时从开盘净值扣减，目标权重按扣费后的净值确定。现金余额（含卖空所得）按 BIL 计息。上个收盘的空头名义金额按实际经过的日历天数和 365 天年基准计提年化 100 bps 借券费及 25 bps 空头所得利率折减。另运行零费用毛收益、交易费 0 和 15 bps、借券费年化 300 bps 情景。未建模滑点，成交采用复权开盘价，未单独模拟买卖价差或市场冲击。（config.yaml:22-27，strategy.py:113-153，strategy.py:219-232，README.md:12，README.md:24）
Signal lag: 月末交易日 t 的复权收盘价用于计算动量与相关性。目标仓位在下一条交易日行情的开盘执行，且开盘执行日不早于回测首日。旧仓承担隔夜跳空，新仓自执行日开盘起承担价格变化。（strategy.py:76-106，strategy.py:121-154，README.md:11）
Price adjustment: yfinance 下载显式使用 auto_adjust=True，缓存保存复权 Open 和 Close。风险资产的复权收盘价用于信号，复权开盘价用于模拟成交，开盘与收盘的复权价格比用于持仓及 BIL 现金代理的计价。（strategy.py:21-26，strategy.py:53-55，strategy.py:65-73，strategy.py:121-154，strategy.py:213-218，strategy.py:234-237，README.md:11-12）
Survivorship bias: 全样本使用固定的 13 只风险 ETF，沿用原文事后选定的资产池。存在潜在存续和选样偏差。加载器从所有标的均有有效开盘和收盘价的第一个日期起使用共同历史，并拒绝其后的无效行情或缺失值。（config.yaml:4-5，strategy.py:28-44，strategy.py:205-218，README.md:22）
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.07162566073221677, ann_vol=0.1398082749807784, sharpe=0.5654442567755426, max_drawdown=-0.2981251493417135, total_return=1.7571963339601178, turnover=112.23451252250881
    oos: cagr=0.08868327950935284, ann_vol=0.14072745523855276, sharpe=0.6761679956461814, max_drawdown=-0.1661751588663628, total_return=0.37374853010960996, turnover=30.884621625801135
    full: cagr=0.07506872233162931, ann_vol=0.13998053340366046, sharpe=0.5880769053765496, max_drawdown=-0.2981251493417135, total_return=2.7876944110013215, turnover=143.11913414830997
    benchmark  full: cagr=0.05318178530090534, ann_vol=0.12751991314141278, sharpe=0.47085320407409675, max_drawdown=-0.3927906774656982, total_return=1.5943255850281126, turnover=7.602589200374794
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 100_etf_asset_momentum_corr_hedge\ncategory: momentum/reversal\n\nuniverse: [SPY, IWM, EFA, EEM, IYR, QQQ, LQD, IEF, TIP, GLD, USO, DBC, FXE]\ncash_proxy: BIL\ndownload_start: 2007-06-01\nin_sample:\n  start: 2008-07-01\n  end: 2023-02-28\nout_of_sample:\n  start: 2023-03-01\n  end: 2026-11-24\n\nmomentum_sessions: [63, 126, 189, 252]\ncorrelation_sessions:\n  short: 20\n  long: 250\nlong_count: 4\nshort_count: 4\nshort_gross_weight: 0.30\n\ncosts:\n  trading_bps: 5                # Per dollar bought or sold, including covers.\n  short_borrow_bps_annual: 100  # Charged on marked short notional.\n  short_rebate_haircut_bps_annual: 25  # Reduces the BIL proxy rate on short proceeds.\n  trading_bps_sensitivity: [0, 15]\n  short_borrow_bps_annual_sensitivity: [300]\n'
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
"""C01 cross-asset ETF momentum backtest."""

import hashlib
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
TRADING_DAYS = 252
BPS = 10_000


def load_prices(symbols, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    columns = pd.MultiIndex.from_product([["Open", "Close"], symbols])
    cache_key = hashlib.sha256(f"adjusted:{symbols}:{start}:{end}".encode()).hexdigest()[:16]
    cache = cache_dir / f"yahoo_{cache_key}.csv"

    def validate(prices):
        prices = prices.reindex(columns=columns).sort_index().loc[start:end]
        if prices.index.has_duplicates:
            raise ValueError("Duplicate Yahoo Finance dates")
        valid = np.isfinite(prices) & prices.gt(0)
        complete_dates = valid.index[valid.all(axis=1)]
        if complete_dates.empty:
            missing = valid.columns[~valid.any(axis=0)].get_level_values(1).unique()
            raise ValueError(f"No common adjusted open and close history. Missing symbols {list(missing)}")
        if (pd.Timestamp(end) - prices.index[-1]).days > 4:
            raise ValueError(f"Yahoo Finance history ends on {prices.index[-1].date()}, requested {end}")
        first = complete_dates[0]
        common_history = valid.loc[first:]
        gaps = common_history.columns[~common_history.all(axis=0)]
        if len(gaps):
            raise ValueError(f"Missing or invalid bars after {first.date()} for {list(gaps)}")
        return prices.loc[first:]

    if cache.exists():
        return validate(pd.read_csv(cache, header=[0, 1], index_col=0, parse_dates=True))

    # Keep yfinance's anonymous session and timezone caches in the working folder.
    with TemporaryDirectory(dir=cache_dir) as yf_cache:
        yf.set_tz_cache_location(yf_cache)
        for threads in (5, 2):
            raw = yf.download(
                symbols, start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date(),
                auto_adjust=True, progress=False, threads=threads, timeout=10,
            )
            try:
                prices = validate(raw)
                break
            except ValueError as error:
                failure = error
        else:
            raise RuntimeError(f"Public Yahoo Finance download failed: {failure}") from failure

    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_dir, delete=False) as handle:
        temporary_path = Path(handle.name)
        try:
            prices.to_csv(handle, index_label="Date")
            handle.flush()
            temporary_path.replace(cache)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices


def monthly_targets(closes, inception, config):
    windows = config["momentum_sessions"]
    fast = config["correlation_sessions"]["short"]
    slow = config["correlation_sessions"]["long"]
    long_count, short_count = config["long_count"], config["short_count"]
    short_gross = config["short_gross_weight"]
    ranks = [closes.pct_change(window, fill_method=None).rank(axis=1, pct=True) for window in windows]
    scores = sum(ranks) / len(ranks)
    daily_returns = closes.pct_change(fill_method=None)
    pairs = np.triu_indices(closes.shape[1], k=1)
    months = closes.index.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])

    hedged, long_only, equal_weight = {}, {}, {}
    for day in month_ends:
        if day < max(max(windows), slow, fast):
            continue
        execution = max(closes.index[day + 1], inception)
        rank_order = np.argsort(-scores.iloc[day].to_numpy(), kind="stable")
        long = np.zeros(closes.shape[1])
        long[rank_order[:long_count]] = 1 / long_count
        target = long.copy()
        fast_corr = daily_returns.iloc[day + 1 - fast:day + 1].corr().to_numpy()[pairs]
        slow_corr = daily_returns.iloc[day + 1 - slow:day + 1].corr().to_numpy()[pairs]
        if not (np.isfinite(fast_corr).all() and np.isfinite(slow_corr).all()):
            raise ValueError(f"Undefined correlation at {closes.index[day].date()}")
        if fast_corr.mean() > slow_corr.mean():
            target[rank_order[-short_count:]] = -short_gross / short_count
        hedged[execution] = target
        long_only[execution] = long
        equal_weight[execution] = np.full(closes.shape[1], 1 / closes.shape[1])

    if inception not in hedged:
        raise ValueError("Increase download history to allow a complete month-end signal before the sample")
    return hedged, long_only, equal_weight


def simulate(opens, closes, targets, trade_bps, borrow_bps, rebate_haircut_bps):
    """Mark at the close, keeping the old positions through each opening gap."""
    positions = np.zeros(opens.shape[1] - 1)
    cash = previous_nav = 1.0
    trade_rate = trade_bps / BPS
    short_daily_rate = (borrow_bps + rebate_haircut_bps) / (BPS * 365)
    results = []

    open_values = opens.to_numpy()
    close_values = closes.to_numpy()
    for day, date in enumerate(opens.index):
        opening = open_values[day]
        closing = close_values[day]
        if day:
            previous_close = close_values[day - 1]
            short_notional = -np.minimum(positions, 0).sum()
            days_elapsed = (date - opens.index[day - 1]).days
            cash *= opening[-1] / previous_close[-1]
            cash -= short_notional * short_daily_rate * days_elapsed
            positions *= opening[:-1] / previous_close[:-1]
        open_nav = cash + positions.sum()
        if open_nav <= 0:
            raise ValueError(f"Portfolio capital exhausted at the opening on {date.date()}")

        turnover = 0.0
        if date in targets:
            target = targets[date]
            # The target weights apply to NAV after the cost of reaching them.
            invested_nav = open_nav
            if trade_rate:
                invested_nav = brentq(
                    lambda nav: nav + trade_rate * np.abs(nav * target - positions).sum() - open_nav,
                    0, open_nav, xtol=1e-14,
                )
            new_positions = invested_nav * target
            turnover = np.abs(new_positions - positions).sum() / open_nav
            positions = new_positions
            cash = invested_nav - positions.sum()

        positions *= closing[:-1] / opening[:-1]
        cash *= closing[-1] / opening[-1]
        nav = cash + positions.sum()
        if nav <= 0:
            raise ValueError(f"Portfolio capital exhausted at the close on {date.date()}")
        results.append((nav / previous_nav - 1, turnover))
        previous_nav = nav
    return pd.DataFrame(results, index=opens.index, columns=["return", "turnover"])


def metrics(history, start, end):
    period = history.loc[start:end]
    if period.empty:
        raise ValueError(f"No returns for {start.date()} through {end.date()}")
    daily = period["return"]
    equity = np.r_[1.0, (1 + daily).cumprod().to_numpy()]
    start_index = history.index.get_loc(period.index[0])
    base_date = history.index[max(0, start_index - 1)]
    years = (period.index[-1] - base_date).days / 365.25
    vol = daily.std() * np.sqrt(TRADING_DAYS) if len(daily) > 1 else np.nan
    values = dict(
        cagr=equity[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        ann_vol=vol,
        sharpe=daily.mean() * TRADING_DAYS / vol if vol > 0 else np.nan,
        max_drawdown=(equity / np.maximum.accumulate(equity) - 1).min(),
        total_return=equity[-1] - 1,
        turnover=period["turnover"].sum(),
    )
    return {name: float(value) if math.isfinite(value) else None for name, value in values.items()}


def print_table(reports):
    keys = ["cagr", "ann_vol", "sharpe", "max_drawdown", "total_return", "turnover"]
    headers = ["CAGR", "Ann. vol", "Sharpe", "Max DD", "Total return", "Turnover"]
    print(f"{'Portfolio / assumptions':<40} {'Period':<6} " + " ".join(f"{h:>13}" for h in headers))
    for name, report in reports.items():
        for period, row in report.items():
            cells = []
            for key in keys:
                value = row[key]
                fmt = ".2f" if key in {"sharpe", "turnover"} else ".2%"
                cells.append("n/a" if value is None else format(value, fmt))
            print(f"{name:<40} {period.upper():<6} " + " ".join(f"{cell:>13}" for cell in cells))


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Define nonoverlapping IS and OOS periods in chronological order")
    periods = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    universe = config["universe"]
    symbols = [*universe, config["cash_proxy"]]
    if len(set(symbols)) != len(symbols):
        raise ValueError("Risk tickers and the cash proxy must be distinct")
    if not (0 < config["long_count"] and 0 < config["short_count"]
            and config["long_count"] + config["short_count"] <= len(universe)):
        raise ValueError("Long and short counts must form disjoint nonempty baskets")

    prices = load_prices(symbols, str(config["download_start"]), str(oos_end.date()))
    evaluation = prices.loc[is_start:oos_end]
    if evaluation.empty:
        raise ValueError("The configured evaluation period has no common price data")
    inception = evaluation.index[0]
    hedged, long_only, equal_weight = monthly_targets(prices["Close"][universe], inception, config)
    costs = config["costs"]
    trade = costs["trading_bps"]
    borrow = costs["short_borrow_bps_annual"]
    haircut = costs["short_rebate_haircut_bps_annual"]
    cases = {
        f"Selective hedge ({trade:g} bps)": (hedged, trade, borrow, haircut),
        f"Long-only momentum ({trade:g} bps)": (long_only, trade, 0, 0),
        f"Equal weight ({trade:g} bps)": (equal_weight, trade, 0, 0),
        "Selective hedge (gross, before costs)": (hedged, 0, 0, 0),
    }
    for rate in costs["trading_bps_sensitivity"]:
        cases[f"Selective hedge ({rate:g} bps trading)"] = (hedged, rate, borrow, haircut)
    for rate in costs["short_borrow_bps_annual_sensitivity"]:
        cases[f"Selective hedge ({rate:g} bps annual borrow)"] = (hedged, trade, rate, haircut)

    opens, closes = evaluation["Open"][symbols], evaluation["Close"][symbols]
    reports = {}
    for name, (targets, trading_rate, borrow_rate, rebate_rate) in cases.items():
        history = simulate(opens, closes, targets, trading_rate, borrow_rate, rebate_rate)
        reports[name] = {period: metrics(history, start, end) for period, (start, end) in periods.items()}

    data_range = f"{inception.date()} to {evaluation.index[-1].date()}"
    print(f"{config['strategy']} | {data_range}")
    print(f"IS {is_start.date()} to {is_end.date()} | OOS {oos_start.date()} to {oos_end.date()}")
    print("Turnover is cumulative dollars traded divided by pre-trade NAV at each rebalance.")
    print_table(reports)
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **reports[f"Selective hedge ({trade:g} bps)"],
        "benchmark": {"name": "equal_weight", **reports[f"Equal weight ({trade:g} bps)"]},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
