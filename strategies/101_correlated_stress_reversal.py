"""101_correlated_stress_reversal
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/101_correlated_stress_reversal.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 10f82382008748cd9ff2f56e2b8ef160caa66e38 on 2026-11-27 11:24 Europe/Oslo, exit 0, output sha256 2733f15a96324bfae750b9cc79b1049d1d673d30d25d891bad53bdf70f7976e5
Data source: Yahoo Finance 免费 ETF 日线开盘价和收盘价，经 yfinance 获取。风险 ETF 为 SPY、USO、GLD，国债确认信号用 IEF，现金代理用 BIL，买入并持有基准用 SPY。相同标的和日期区间优先读取本地 CSV。
Data fetched: 2026-11-27T10:24:04+00:00
Sample period: 报告覆盖 2008-01-02 至 2026-11-24，共 4755 个交易日。预热数据的请求起点是 2007-12-14。 data_range (from output): 2008-01-02 to 2026-11-24
IS/OOS split: 固定日期切分，端点包含在内。IS 为 2008-01-02 至 2025-04-25（4357 个交易日），OOS 为 2025-04-26 至 2026-11-24（398 个交易日，按区间内实际交易日统计）。策略和两条基准均先连续模拟，再按日期切片，跨切分点延续持仓与信号。README 将 OOS 定义为原文发布后的历史检验，并记载门槛未依据 IS 或 OOS 结果调整。
Costs and slippage: 每笔买入或卖出按成交额计 5 bps，含 BIL 换仓。卖出所得乘以 (1 − 0.0005)，买入所需价格乘以 (1 + 0.0005)，每次卖出并买入的换仓净值乘以两者之比。回测首日开盘只有买入，无信号继续持有 BIL 时不换仓。SPY 买入并持有基准仅在首日开盘扣买入成本，期末持仓按收盘价估值。按开盘和收盘价模拟成交，未单独建模滑点、价差、冲击或竞价执行偏差。
Signal lag: 交易日 t 收盘后，用当日收盘到收盘收益计算信号。下一交易日开盘等权买入有信号的风险 ETF，当天收盘卖出并买回 BIL。无信号时持有 BIL，隔夜也持有 BIL。单资产跌幅基准使用相同的一交易日信号滞后。
Price adjustment: Yahoo Finance split- and distribution-adjusted Open and Close (auto_adjust=True)
Survivorship bias: 回顾性选定的存续 ETF 构成固定标的池，存在存续标的及代理品种选择偏差。回测在 ETF 层面运行，未用当前指数成分股回填历史。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.044528587016328225, ann_vol=0.0857479007908623, sharpe=-0.48834818230165716, sharpe_excess=-0.6146061527227347, max_drawdown=-0.6082373491249764, total_return=-0.5450430799480642, turnover=112.22885620697551, n_trades=3151
    oos: cagr=0.008477393370199593, ann_vol=0.04252360518143846, sharpe=0.21976932803356766, sharpe_excess=-0.6664614166510161, max_drawdown=-0.06919454302681083, total_return=0.013421739915091102, turnover=90.90517449918836, n_trades=236
    full: cagr=-0.040200837249976584, ann_vol=0.08299967147512655, sharpe=-0.4528647975895032, sharpe_excess=-0.6104310628273976, max_drawdown=-0.6082373491249764, total_return=-0.5389367664945559, turnover=110.44403489894202, n_trades=3387
    benchmark SPY full: cagr=0.10773575444165018, ann_vol=0.19744346912752808, sharpe=0.6170935244879814, sharpe_excess=0.5491259906030055, max_drawdown=-0.5237536887204206, total_return=5.893952348759535, turnover=0.052917389885498885, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 101_correlated_stress_reversal\n\nuniverse:\n  risk_tickers: [SPY, USO, GLD]\n  treasury_ticker: IEF\n  cash_ticker: BIL\n  benchmark_ticker: SPY\n\nsignal:\n  risk_drop_pct: 1.0\n  treasury_rise_pct: 1.0\n\ndata_start: 2007-12-14  # Warm-up history before the first reported session.\nin_sample:\n  start: 2008-01-02\n  end: 2025-04-25\nout_of_sample:\n  start: 2025-04-26\n  end: 2026-11-24\n\ntransaction_cost_bps_per_side: 5.0\nslippage_bps_per_side: 1.0\ncash_buffer_bps: 10.0\nsessions_per_year: 252\nparam_combos_tried: 1\n'
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

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
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
import yaml
import yfinance as yf
from yfinance.exceptions import YFException


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_COLUMNS = ["Open", "Close"]
PRICE_ADJUSTMENT = "Yahoo Finance split- and distribution-adjusted Open and Close (auto_adjust=True)"
LEGACY_SNAPSHOT_BY = "2026-11-25"


def get_prices(ticker, start, end):
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
    return prices


def expected_sessions(start, end):
    calendar = AbstractHolidayCalendar(rules=[
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ])
    # Hurricane Sandy and the Bush and Carter days of mourning.
    special_closures = pd.DatetimeIndex(["2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])
    return pd.bdate_range(start, end).difference(calendar.holidays(start, end)).difference(special_closures)


def load_prices(tickers, start, end):
    for ticker in tickers:
        if not re.fullmatch(r"[A-Z]{1,5}", ticker):
            raise ValueError(f"Expected a U.S. ETF ticker: {ticker!r}")
    try:
        with TemporaryDirectory(prefix="yahoo_", dir=ROOT) as metadata_cache:
            yf.set_tz_cache_location(metadata_cache)
            histories = {ticker: get_prices(ticker, start, end) for ticker in tickers}
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        source = "Yahoo Finance via yfinance (download)"
    except (OSError, RuntimeError, YFException) as error:
        paths = {
            ticker: CACHE / f"yahoo_adjusted_{ticker}_{start.date()}_{end.date()}.csv"
            for ticker in tickers
        }
        if not all(path.is_file() for path in paths.values()):
            raise
        print(f"Yahoo Finance download failed ({error}). Using the bundled snapshot.", file=sys.stderr)
        histories = {
            ticker: pd.read_csv(path, index_col="Date", parse_dates=["Date"])
            for ticker, path in paths.items()
        }
        fetched_at = f"on or before {LEGACY_SNAPSHOT_BY} (original timestamp unavailable)"
        source = "Yahoo Finance via yfinance (bundled snapshot)"

    sessions = expected_sessions(start, end)
    for ticker, prices in histories.items():
        missing = sessions.difference(prices.index)
        unexpected = prices.index.difference(sessions)
        if prices.index.has_duplicates or len(missing) or len(unexpected):
            affected = missing.union(unexpected)[:5].strftime("%Y-%m-%d").tolist()
            raise ValueError(
                f"Invalid NYSE session coverage for {ticker}: {len(missing)} missing, "
                f"{len(unexpected)} unexpected, duplicates={prices.index.has_duplicates}. "
                f"First affected dates: {affected}"
            )
        values = prices[PRICE_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Invalid adjusted opening or closing prices for {ticker}")

    opens = pd.DataFrame({ticker: data["Open"] for ticker, data in histories.items()})
    closes = pd.DataFrame({ticker: data["Close"] for ticker, data in histories.items()})
    return opens.loc[sessions], closes.loc[sessions], fetched_at, source


def fill_buys(cash, requested_shares, fill_prices, fee, cash_buffer):
    """Apply a common downward cap to requested shares using available opening cash."""
    notional = float(np.sum(requested_shares * fill_prices))
    spending_limit = cash * (1 - cash_buffer) / (1 + fee)
    if notional > spending_limit:
        requested_shares = requested_shares * (spending_limit / notional)
        notional = float(np.sum(requested_shares * fill_prices))
    remaining_cash = cash - notional * (1 + fee)
    if remaining_cash < 0:
        raise ArithmeticError("Opening fill would make cash negative")
    return requested_shares, remaining_cash, notional


def reversal_returns(opens, closes, prior_closes, selected, cash_ticker, fee, slippage, cash_buffer):
    """Rotate into the risk basket at the open and liquidate it into cash at the close."""
    tickers = [*selected.columns, cash_ticker]
    buy_factor = (1 + slippage) * (1 + fee)
    sell_factor = (1 - slippage) * (1 - fee)
    cash = previous_value = 1.0
    cash_etf_shares = 0.0
    rows = []
    for opening, closing, sizing_prices, risk_mask in zip(
        opens[tickers].to_numpy(), closes[tickers].to_numpy(), prior_closes[tickers].to_numpy(),
        selected.to_numpy(dtype=bool), strict=True,
    ):
        turnover_notional = 0.0
        trades = 0
        if risk_mask.any():
            # Estimate sale proceeds at the previous close before forming the purchase quantities.
            estimated_cash = cash + cash_etf_shares * sizing_prices[-1] * sell_factor
            per_etf = estimated_cash * (1 - cash_buffer) / risk_mask.sum()
            requested = per_etf / (sizing_prices[:-1][risk_mask] * buy_factor)

            open_sale = cash_etf_shares * opening[-1] * (1 - slippage)
            cash += open_sale * (1 - fee)
            trades = int(cash_etf_shares > 0)
            cash_etf_shares = 0.0
            risk_shares, cash, buy_notional = fill_buys(
                cash, requested, opening[:-1][risk_mask] * (1 + slippage), fee, cash_buffer,
            )
            # The opening fill fixes the closing sale quantities.
            close_sale = float(risk_shares @ closing[:-1][risk_mask]) * (1 - slippage)
            cash += close_sale * (1 - fee)
            turnover_notional = open_sale + buy_notional + close_sale
            trades += 2 * int(np.count_nonzero(risk_shares))
        elif cash_etf_shares == 0:
            requested = cash * (1 - cash_buffer) / (sizing_prices[-1] * buy_factor)
            cash_etf_shares, cash, turnover_notional = fill_buys(
                cash, requested, opening[-1] * (1 + slippage), fee, cash_buffer,
            )
            trades = int(cash_etf_shares > 0)

        value = cash + cash_etf_shares * closing[-1]
        rows.append((value / previous_value - 1, trades, turnover_notional / previous_value))
        previous_value = value
    return pd.DataFrame(rows, index=opens.index, columns=["returns", "trades", "turnover"])


def buy_and_hold(opens, closes, ticker, fee, slippage, cash_buffer):
    fill_price = opens[ticker].iloc[0] * (1 + slippage)
    requested = (1 - cash_buffer) / (fill_price * (1 + fee))
    shares, cash, notional = fill_buys(1.0, requested, fill_price, fee, cash_buffer)
    value = cash + shares * closes[ticker]
    returns = value / value.shift(fill_value=1.0) - 1
    daily_results = pd.DataFrame({"returns": returns, "trades": 0, "turnover": 0.0})
    daily_results.loc[value.index[0], ["trades", "turnover"]] = (int(shares > 0), notional)
    return daily_results


def finite_or_none(value):
    return float(value) if math.isfinite(value) else None


def metrics(daily, cash_returns, sessions_per_year):
    returns = daily["returns"]
    excess_returns = returns - cash_returns
    wealth = (1 + returns).cumprod()
    terminal_wealth = wealth.iloc[-1]
    volatility = returns.std(ddof=1)
    excess_volatility = excess_returns.std(ddof=1)
    annualization = math.sqrt(sessions_per_year)
    return {
        "cagr": finite_or_none(terminal_wealth ** (sessions_per_year / len(returns)) - 1),
        "ann_vol": finite_or_none(volatility * annualization),
        "sharpe": finite_or_none(returns.mean() / volatility * annualization) if volatility > 0 else None,
        "sharpe_excess": (
            finite_or_none(excess_returns.mean() / excess_volatility * annualization)
            if excess_volatility > 0 else None
        ),
        "max_drawdown": finite_or_none((wealth / wealth.cummax().clip(lower=1.0) - 1).min()),
        "total_return": finite_or_none(terminal_wealth - 1),
        "turnover": finite_or_none(daily["turnover"].mean() * sessions_per_year),
        "n_trades": int(daily["trades"].sum()),
        "n_sessions": len(returns),
    }


def report(daily, cash_returns, periods, sessions_per_year):
    return {
        name: metrics(daily.loc[start:end], cash_returns.loc[start:end], sessions_per_year)
        for name, (start, end) in periods.items()
    }


def print_results(results):
    print(f"{results['strategy']}  |  {results['data_range']}")
    print(f"{'Portfolio':<23} {'Period':<6} {'CAGR':>9} {'Ann. vol':>9} {'Sharpe':>8} "
          f"{'Ex.Sharpe':>9} {'Max DD':>9} {'Total':>10} {'Turnover':>10} {'Trades':>8} {'Sessions':>9}")
    portfolios = {
        "Correlated stress": results,
        results["benchmark"]["name"]: results["benchmark"],
        results["benchmark"]["cash"]["name"]: results["benchmark"]["cash"],
        "Individual drop": results["benchmark"]["unconfirmed_drop"],
    }
    for label, portfolio in portfolios.items():
        for period in ("is", "oos", "full"):
            stats = portfolio[period]
            displayed = []
            for key in ("cagr", "ann_vol", "sharpe", "sharpe_excess", "max_drawdown", "total_return", "turnover"):
                value = stats[key]
                fmt = ".2f" if key in ("sharpe", "sharpe_excess") else ".2%"
                displayed.append(format(value, fmt) if value is not None else "n/a")
            cagr, volatility, sharpe, sharpe_excess, drawdown, total, turnover = displayed
            print(f"{label:<23} {period.upper():<6} {cagr:>9} {volatility:>9} {sharpe:>8} "
                  f"{sharpe_excess:>9} {drawdown:>9} {total:>10} {turnover:>10} "
                  f"{stats['n_trades']:>8} {stats['n_sessions']:>9}")
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

    fee = float(cfg["transaction_cost_bps_per_side"]) / 10_000
    slippage = float(cfg["slippage_bps_per_side"]) / 10_000
    cash_buffer = float(cfg["cash_buffer_bps"]) / 10_000
    execution = (fee, slippage, cash_buffer)
    risk_drop = float(cfg["signal"]["risk_drop_pct"]) / 100
    treasury_rise = float(cfg["signal"]["treasury_rise_pct"]) / 100
    sessions_per_year = int(cfg["sessions_per_year"])
    param_combos_tried = int(cfg["param_combos_tried"])
    if not (0 <= fee < 1 and 0 <= slippage < 1 and 0 < cash_buffer < 1):
        raise ValueError("Invalid fee, slippage, or cash buffer")
    if not (0 < risk_drop < 1 and 0 < treasury_rise < 1 and sessions_per_year > 0 and param_combos_tried > 0):
        raise ValueError("Invalid signal threshold, annualization, or parameter count")

    opens, closes, fetched_at, data_source = load_prices(tickers, data_start, oos_end)
    prices_csv = pd.concat({"Open": opens, "Close": closes}, axis=1).to_csv()
    price_data_hash = sha256(prices_csv.encode("utf-8")).hexdigest()
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
    prior_closes = closes.shift(1).loc[trade_dates]
    opens, closes = opens.loc[trade_dates], closes.loc[trade_dates]
    cash_daily = buy_and_hold(opens, closes, cash, *execution)
    cash_returns = cash_daily["returns"]

    def run_reversal(signal):
        next_session = signal.shift(1, fill_value=False).loc[trade_dates]
        daily = reversal_returns(opens, closes, prior_closes, next_session, cash, *execution)
        return report(daily, cash_returns, periods, sessions_per_year)

    benchmark_daily = buy_and_hold(opens, closes, benchmark, *execution)
    results = {
        "strategy": cfg["strategy"],
        "data_range": f"{trade_dates[0].date()} to {trade_dates[-1].date()}",
        **run_reversal(confirmed_drops),
        "benchmark": {
            "name": f"{benchmark} buy and hold",
            "ticker": benchmark,
            **report(benchmark_daily, cash_returns, periods, sessions_per_year),
            "cash": {
                "name": f"{cash} buy and hold",
                "ticker": cash,
                **report(cash_daily, cash_returns, periods, sessions_per_year),
            },
            "unconfirmed_drop": {
                "name": "Individual risk ETF drop",
                **run_reversal(drops),
            },
        },
    }
    print(f"PARAM_COMBOS_TRIED: {param_combos_tried}")
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"DATA_SOURCE: {data_source}")
    print(f"DATA_SHA256: {price_data_hash}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print_results(results)


if __name__ == "__main__":
    main()
