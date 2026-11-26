"""101_correlated_stress_reversal
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/101_correlated_stress_reversal.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 3140bbcffa77faae04956fc55adbc5be3b33f896 on 2026-11-26 16:06 Europe/Oslo, exit 0, output sha256 3ff8bfda66ce7393fd97b31555c635fbe5403aba43679813b5e3a34b5aadbf26
Data source: Yahoo Finance 免费 ETF 日线开盘价和收盘价，经 yfinance 获取。风险 ETF 为 SPY、USO、GLD，国债确认信号用 IEF，现金代理用 BIL，买入并持有基准用 SPY。相同标的和日期区间优先读取本地 CSV。config.yaml:3-7，strategy.py:15、20、23-43、64-75、176-183，README.md:9-11、19。
Data fetched: 2026-11-26 16:06 Europe/Oslo (run time)
Sample period: 报告覆盖 2008-01-02 至 2026-11-24，共 4755 个交易日。预热数据的请求起点是 2007-12-14。config.yaml:13-19，strategy.py:192-204、210-212，logs/run_20261125_094917.log:1、5。   data_range (from output): 2008-01-02 to 2026-11-24
IS/OOS split: 固定日期切分，端点包含在内。IS 为 2008-01-02 至 2025-04-25（4357 个交易日），OOS 为 2025-04-26 至 2026-11-24（398 个交易日，按区间内实际交易日统计）。策略和两条基准均先连续模拟，再按日期切片，跨切分点延续持仓与信号。README 将 OOS 定义为原文发布后的历史检验，并记载门槛未依据 IS 或 OOS 结果调整。config.yaml:14-19，strategy.py:131-135、165-174、199-222，README.md:17-19，logs/run_20261125_094917.log:3-4。
Costs and slippage: 每笔买入或卖出按成交额计 5 bps，含 BIL 换仓。卖出所得乘以 (1 − 0.0005)，买入所需价格乘以 (1 + 0.0005)，每次卖出并买入的换仓净值乘以两者之比。回测首日开盘只有买入，无信号继续持有 BIL 时不换仓。SPY 买入并持有基准仅在首日开盘扣买入成本，期末持仓按收盘价估值。按开盘和收盘价模拟成交，未单独建模滑点、价差、冲击或竞价执行偏差。config.yaml:21，strategy.py:78-107、185，README.md:11、19、24。
Signal lag: 交易日 t 收盘后，用当日收盘到收盘收益计算信号。下一交易日开盘等权买入有信号的风险 ETF，当天收盘卖出并买回 BIL。无信号时持有 BIL，隔夜也持有 BIL。单资产跌幅基准使用相同的一交易日信号滞后。strategy.py:78-94、199-208、218-220，README.md:9-11、19。
Price adjustment: 下载时使用 yfinance 的 auto_adjust=True 和 actions=False，使用经拆分和分红复权的 Open 与 Close；缓存命中时直接读取相同格式的价格列。历史价格及复权数据可能由供应商修订。strategy.py:20、27-43、52-58，README.md:9、25。
Survivorship bias: 回顾性选定的存续 ETF 构成固定标的池，存在存续标的及代理品种选择偏差。回测在 ETF 层面运行，未用当前指数成分股回填历史。config.yaml:3-7，strategy.py:176-183、192，README.md:26。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.03748985026400087, ann_vol=0.08717121970428113, sharpe=-0.39472496522180744, max_drawdown=-0.5623019878028662, total_return=-0.4834846719443592, n_trades=3343
    oos: cagr=0.016265630944831466, ann_vol=0.04278142761364916, sharpe=0.39855338057705875, max_drawdown=-0.0686548858113063, total_return=0.025810139982630043, n_trades=252
    full: cagr=-0.0331016237951367, ann_vol=0.08435818757229768, sharpe=-0.3568289041254561, max_drawdown=-0.5623019878028662, total_return=-0.4701533390240683, n_trades=3595
    benchmark SPY full: cagr=0.10779182407269272, ann_vol=0.19761391693340563, sharpe=0.6169874843141896, max_drawdown=-0.5242304608665198, total_return=5.900539629857733, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 101_correlated_stress_reversal\n\nuniverse:\n  risk_tickers: [SPY, USO, GLD]\n  treasury_ticker: IEF\n  cash_ticker: BIL\n  benchmark_ticker: SPY\n\nsignal:\n  risk_drop_pct: 1.0\n  treasury_rise_pct: 1.0\n\ndata_start: 2007-12-14  # Warm-up history before the first reported session.\nin_sample:\n  start: 2008-01-02\n  end: 2025-04-25\nout_of_sample:\n  start: 2025-04-26\n  end: 2026-11-24\n\ntransaction_cost_bps_per_side: 5.0\nsessions_per_year: 252\n'
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
"""Backtest next-session cross-asset stress reversals using public ETF prices."""

import json
import math
from pathlib import Path
import re
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_COLUMNS = ["Open", "Close"]


def get_prices(ticker, start, end):
    if not re.fullmatch(r"[A-Z0-9^._=-]+", ticker):
        raise ValueError(f"Invalid Yahoo Finance ticker: {ticker!r}")

    path = CACHE / f"yahoo_adjusted_{ticker}_{start.date()}_{end.date()}.csv"
    if path.exists():
        prices = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    else:
        history = yf.Ticker(ticker).history(
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True,
            actions=False,
            timeout=15,
        )
        if history.empty:
            raise RuntimeError(f"Yahoo Finance returned no history for {ticker}")
        prices = history[PRICE_COLUMNS].copy()
        prices.index = prices.index.tz_localize(None).normalize()

    prices = prices[PRICE_COLUMNS].sort_index().loc[start:end]
    if prices.empty or prices.index.has_duplicates:
        raise ValueError(f"Missing or duplicate trading dates for {ticker}")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"Invalid adjusted opening or closing prices for {ticker}")
    if (prices.index[0] - start).days > 7 or (end - prices.index[-1]).days > 7:
        raise ValueError(f"{ticker} history does not cover {start.date()} to {end.date()}")

    if not path.exists():
        with NamedTemporaryFile(mode="w", dir=CACHE, suffix=".tmp", delete=False) as tmp:
            temporary = Path(tmp.name)
            try:
                prices.to_csv(tmp, index_label="Date")
                tmp.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    return prices


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="yahoo_", dir=CACHE) as metadata_cache:
        yf.set_tz_cache_location(metadata_cache)
        histories = {ticker: get_prices(ticker, start, end) for ticker in tickers}
    opens = pd.DataFrame({ticker: data["Open"] for ticker, data in histories.items()})
    closes = pd.DataFrame({ticker: data["Close"] for ticker, data in histories.items()})
    missing = closes.isna().sum()
    if missing.any():
        detail = ", ".join(f"{ticker} ({count})" for ticker, count in missing.items() if count)
        raise ValueError(f"Missing sessions in the shared ETF calendar: {detail}")
    return opens, closes


def reversal_returns(opens, closes, selected, cash_ticker, cost):
    """Hold the cash ETF overnight and rotate into the selected basket during the session."""
    n_selected = selected.sum(axis=1)
    risk_day = n_selected.gt(0)
    intraday = closes / opens
    risk_gross = intraday[selected.columns].where(selected, 0).sum(axis=1) / n_selected.where(risk_day)
    session_gross = risk_gross.where(risk_day, intraday[cash_ticker])

    overnight_gross = opens[cash_ticker] / closes[cash_ticker].shift()
    overnight_gross.iloc[0] = 1.0  # The backtest starts in cash at the first open.

    # A sale leaves (1 - cost) of proceeds. A purchase spends price * (1 + cost).
    switch_factor = (1 - cost) / (1 + cost)
    entry_factor = pd.Series(np.where(risk_day, switch_factor, 1.0), index=opens.index)
    entry_factor.iloc[0] = 1 / (1 + cost)
    exit_factor = np.where(risk_day, switch_factor, 1.0)
    returns = overnight_gross * entry_factor * session_gross * exit_factor - 1

    # Count each asset-side fill, including BIL. The first open has only buys.
    trades = (2 * (n_selected + 1)).where(risk_day, 0)
    trades.iloc[0] += -1 if risk_day.iloc[0] else 1
    return pd.DataFrame({"returns": returns, "trades": trades})


def buy_and_hold(opens, closes, ticker, cost):
    gross = closes[ticker] / closes[ticker].shift()
    gross.iloc[0] = closes[ticker].iloc[0] / opens[ticker].iloc[0] / (1 + cost)
    trades = pd.Series(0, index=opens.index)
    trades.iloc[0] = 1
    return pd.DataFrame({"returns": gross - 1, "trades": trades})


def finite_or_none(value):
    return float(value) if math.isfinite(value) else None


def metrics(daily, sessions_per_year):
    returns = daily["returns"]
    wealth = (1 + returns).cumprod()
    terminal_wealth = wealth.iloc[-1]
    volatility = returns.std(ddof=1)
    sharpe = returns.mean() / volatility * math.sqrt(sessions_per_year) if volatility > 0 else math.nan
    return {
        "cagr": finite_or_none(terminal_wealth ** (sessions_per_year / len(returns)) - 1),
        "ann_vol": finite_or_none(volatility * math.sqrt(sessions_per_year)),
        "sharpe": finite_or_none(sharpe),
        "max_drawdown": finite_or_none((wealth / wealth.cummax().clip(lower=1.0) - 1).min()),
        "total_return": finite_or_none(terminal_wealth - 1),
        "n_trades": int(daily["trades"].sum()),
        "n_sessions": len(returns),
    }


def report(daily, periods, sessions_per_year):
    return {
        name: metrics(daily.loc[start:end], sessions_per_year)
        for name, (start, end) in periods.items()
    }


def print_results(results):
    print(f"{results['strategy']}  |  {results['data_range']}")
    print(f"{'Portfolio':<23} {'Period':<6} {'CAGR':>9} {'Ann. vol':>9} {'Sharpe':>8} "
          f"{'Max DD':>9} {'Total':>10} {'Trades':>8} {'Sessions':>9}")
    portfolios = {
        "Correlated stress": results,
        results["benchmark"]["name"]: results["benchmark"],
        "Individual drop": results["benchmark"]["unconfirmed_drop"],
    }
    for label, portfolio in portfolios.items():
        for period in ("is", "oos", "full"):
            stats = portfolio[period]
            displayed = []
            for key in ("cagr", "ann_vol", "sharpe", "max_drawdown", "total_return"):
                value = stats[key]
                fmt = ".2f" if key == "sharpe" else ".2%"
                displayed.append(format(value, fmt) if value is not None else "n/a")
            cagr, volatility, sharpe, drawdown, total = displayed
            print(f"{label:<23} {period.upper():<6} {cagr:>9} {volatility:>9} {sharpe:>8} "
                  f"{drawdown:>9} {total:>10} {stats['n_trades']:>8} {stats['n_sessions']:>9}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    periods = {
        key: tuple(pd.Timestamp(cfg[section][bound]) for bound in ("start", "end"))
        for key, section in (("is", "in_sample"), ("oos", "out_of_sample"))
    }
    is_start, is_end = periods["is"]
    oos_start, oos_end = periods["oos"]
    data_start = pd.Timestamp(cfg["data_start"])
    if not data_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Expected warm-up, in-sample, and out-of-sample dates in order")
    periods["full"] = (is_start, oos_end)

    universe = cfg["universe"]
    risk = universe["risk_tickers"]
    treasury = universe["treasury_ticker"]
    cash = universe["cash_ticker"]
    benchmark = universe["benchmark_ticker"]
    if not risk or len(set([*risk, treasury, cash])) != len(risk) + 2:
        raise ValueError("The risk, Treasury, and cash roles require distinct tickers")
    tickers = list(dict.fromkeys([*risk, treasury, cash, benchmark]))

    cost = float(cfg["transaction_cost_bps_per_side"]) / 10_000
    risk_drop = float(cfg["signal"]["risk_drop_pct"]) / 100
    treasury_rise = float(cfg["signal"]["treasury_rise_pct"]) / 100
    sessions_per_year = int(cfg["sessions_per_year"])
    if not (0 <= cost < 1 and 0 < risk_drop < 1 and 0 < treasury_rise < 1 and sessions_per_year > 0):
        raise ValueError("Invalid threshold, transaction cost, or annualization parameter")

    opens, closes = load_prices(tickers, data_start, oos_end)
    if (closes.index < is_start).sum() < 2:
        raise ValueError("At least two warm-up sessions are required before in-sample starts")
    for name, (start, end) in periods.items():
        if closes.loc[start:end].empty:
            raise ValueError(f"No trading sessions in {name}")

    close_returns = closes.pct_change(fill_method=None)
    drops = close_returns[risk].le(-risk_drop)
    confirmed = drops.sum(axis=1).ge(2) | close_returns[treasury].ge(treasury_rise)
    confirmed_drops = drops.where(confirmed, False, axis=0)
    trade_dates = closes.loc[is_start:oos_end].index
    opens, closes = opens.loc[trade_dates], closes.loc[trade_dates]

    def run_reversal(signal):
        next_session = signal.shift(1, fill_value=False).loc[trade_dates]
        return report(reversal_returns(opens, closes, next_session, cash, cost), periods, sessions_per_year)

    results = {
        "strategy": cfg["strategy"],
        "data_range": f"{trade_dates[0].date()} to {trade_dates[-1].date()}",
        **run_reversal(confirmed_drops),
        "benchmark": {
            "name": f"{benchmark} buy and hold",
            "ticker": benchmark,
            **report(buy_and_hold(opens, closes, benchmark, cost), periods, sessions_per_year),
            "unconfirmed_drop": {
                "name": "Individual risk ETF drop",
                **run_reversal(drops),
            },
        },
    }
    print_results(results)


if __name__ == "__main__":
    main()
