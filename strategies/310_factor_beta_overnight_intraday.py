"""310_factor_beta_overnight_intraday
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/310_factor_beta_overnight_intraday.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 2b23c5bbf706aea933c0cba5f5805efb450ee206 on 2026-11-26 14:27 Europe/Oslo, exit 0, output sha256 33a6179fd4fbb1da9c83469c9ab66fa4976954ed318e5be363c9bd06b3c13fda
Data source: Yahoo Finance 公开日频 ETF 开盘价和收盘价，通过 yfinance 读取。标的为 QUAL（质量代理）、SPY（市场对冲和基准）、IWF（成长风格对照）。优先读取 data_cache/ 中的 CSV，缺失时请求 Yahoo Finance（config.yaml:2-5；strategy.py:17-39,196-199,218,235-246；README.md:7,20）。
Data fetched: 2026-11-26 14:27 Europe/Oslo (run time)
Sample period: 配置的数据请求区间为 2014-01-01 至 2025-12-31（含边界），2014 年仅用于初始化对冲。日志中的实际评估区间为 2015-01-02 至 2025-12-31，共 2,766 个交易日（config.yaml:7-14；strategy.py:32-44,218-226；README.md:18；logs/run_20261125_135059.log:3）。   data_range (from output): 2015-01-02 to 2025-12-31
IS/OOS split: IS 配置区间为 2015-01-01 至 2023-05-31，日志首个交易日为 2015-01-02，共 2,117 个交易日。OOS 为 2023-06-01 至 2025-12-31，共 649 个交易日。每个组合先模拟一条连续路径，再按日期切片统计，切分日不额外平仓。OOS 继续使用截至上月最后一个交易日收盘的数据更新月度 beta（config.yaml:7-14；strategy.py:72-95,218-229,244-249；README.md:14,18；logs/run_20261125_135059.log:1-2）。
Costs and slippage: 按每条腿每次成交的绝对名义金额收取单边 1.0 bp，适用于开仓、平仓和调仓。费用从净值扣除，目标头寸按扣费后净值确定。未建模滑点，模拟成交直接使用复权开盘价或收盘价，未单独模拟价差、冲击或借券费用（config.yaml:18；strategy.py:116-147,205-208,244-248；README.md:14,30）。
Signal lag: 月末交易日 t 收盘后，用最近 126 个完整交易日估计 beta。新权重从下一交易日 t+1 起使用，并固定至该月结束。主隔夜组合最早在 t+1 收盘按新权重建仓，在下一交易日开盘平仓。日内对照从 t+1 开盘建仓、当日收盘平仓，持续持有对照在 t+1 开盘调仓（config.yaml:16；strategy.py:72-95,98-111,116-122,231-241；README.md:11-14）。
Price adjustment: yfinance 请求 interval=1d、auto_adjust=True、actions=False，选取复权 Open 和 Close。缓存文件名以 _adjusted.csv 结尾，缓存命中时直接使用其中的 Open 和 Close。以复权价格近似股息对多空头寸的影响，未单独模拟股息现金流（strategy.py:15-17,26-44,58-69,119-147；README.md:20,30）。
Survivorship bias: 使用固定的 QUAL、SPY、IWF 历史 ETF 行情，无成分股回建。选择仍存续的 ETF 可能引入标的选择偏差和存续偏差（config.yaml:2-5；strategy.py:26-44,192-199,218；README.md:32）。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.019001402782930277, ann_vol=0.019050522262511787, sharpe=-0.9975645691377443, max_drawdown=-0.16384139016309407, total_return=-0.148846610356149, turnover=503.9604052387037, n_trades=8466
    oos: cagr=-0.05651798714439615, ann_vol=0.010493774815245405, sharpe=-5.538168988391903, max_drawdown=-0.14050112792227643, total_return=-0.13914688002883024, turnover=503.69170795113473, n_trades=2594
    full: cagr=-0.027935906863817195, ann_vol=0.0174538031514519, sharpe=-1.614616855678312, max_drawdown=-0.27784281184084847, total_return=-0.26728194895105784, turnover=503.8973594904635, n_trades=11060
    benchmark  full: cagr=0.13405871472984554, ann_vol=0.17790037879375842, sharpe=0.7964989491760844, max_drawdown=-0.3371727370591743, total_return=2.978231913649049, turnover=0.18152847462261917, n_trades=2
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 310_factor_beta_overnight_intraday\nuniverse:\n  quality: QUAL\n  market: SPY\n  growth: IWF\n\n# Date bounds are inclusive. The earlier data is used only to initialize the hedge.\ndata_start: '2014-01-01'\nin_sample:\n  start: '2015-01-01'\n  end: '2023-05-31'\nout_of_sample:\n  start: '2023-06-01'\n  end: '2025-12-31'\n\nbeta_lookback_sessions: 126\ngross_exposure: 1.0\ntransaction_cost_bps_per_fill: 1.0\nsessions_per_year: 252\n"
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
"""Backtest the C101 US ETF proxy for session-specific profitability exposure."""

import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_COLUMNS = ["Open", "Close"]


def load_prices(symbols, start, end):
    CACHE.mkdir(exist_ok=True)
    prices = {}
    # Keep yfinance's auxiliary cache local and separate from the price CSVs.
    with TemporaryDirectory(dir=CACHE) as yahoo_cache:
        yf.set_tz_cache_location(yahoo_cache)
        for symbol in symbols:
            path = CACHE / f"{symbol}_{start.date()}_{end.date()}_adjusted.csv"
            cached = path.exists()
            if cached:
                frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
            else:
                frame = yf.Ticker(symbol).history(
                    start=start.date().isoformat(),
                    end=(end + pd.Timedelta(days=1)).date().isoformat(),
                    interval="1d",
                    auto_adjust=True,
                    actions=False,
                    timeout=15,
                )
                if frame.empty:
                    raise ValueError(f"Yahoo Finance returned no daily prices for {symbol}")
                frame.index = frame.index.tz_localize(None)

            frame = frame[PRICE_COLUMNS].loc[start:end]
            values = frame.to_numpy()
            coverage_margin = pd.Timedelta(days=7)
            if (
                frame.empty
                or not frame.index.is_unique
                or not frame.index.is_monotonic_increasing
                or frame.index[0] > start + coverage_margin
                or frame.index[-1] < end - coverage_margin
                or not np.isfinite(values).all()
                or (values <= 0).any()
            ):
                raise ValueError(f"Invalid or incomplete adjusted daily prices for {symbol}")

            if not cached:
                temp = Path(yahoo_cache) / path.name
                frame.to_csv(temp, index_label="date")
                temp.replace(path)
            prices[symbol] = frame

    calendar = prices[symbols[0]].index
    for symbol, frame in prices.items():
        if not frame.index.equals(calendar):
            raise ValueError(f"Daily price dates do not match for {symbol} and {symbols[0]}")
    panel = pd.concat(prices, axis=1)
    return panel.xs("Open", axis=1, level=1), panel.xs("Close", axis=1, level=1)


def month_starts(index):
    month = index.to_period("M")
    return np.r_[True, month[1:] != month[:-1]]


def hedge_betas(opens, closes, market, long_legs, lookback):
    overnight = opens.div(closes.shift(1)).sub(1)
    intraday = closes.div(opens).sub(1)
    # Pool two observations per date, ordered by when each return is known.
    sessions = pd.concat({"overnight": overnight, "intraday": intraday})
    dates_and_sessions = pd.MultiIndex.from_product(
        [closes.index, ["overnight", "intraday"]]
    )
    sessions = sessions.swaplevel().reindex(dates_and_sessions)
    market_returns = sessions[market]
    window = 2 * lookback
    variance = market_returns.rolling(window).var()
    beta = sessions[long_legs].rolling(window).cov(market_returns).div(
        variance.where(variance > 0), axis=0
    )
    # At the first trading day's open only the prior month's closing estimate is usable.
    previous_close = beta.xs("intraday", level=1).shift(1)
    updates = previous_close.loc[month_starts(closes.index)]
    return updates.reindex(closes.index, method="ffill").clip(lower=0)


def simulate(opens, closes, targets, mode, fee_rate):
    open_targets = targets.copy()
    close_targets = targets.copy()
    if mode == "overnight":
        open_targets[:] = 0.0
    elif mode == "intraday":
        close_targets[:] = 0.0
    else:
        rebalance = (
            month_starts(opens.index) if mode == "continuous" else np.arange(len(targets)) == 0
        )
        open_targets[~rebalance] = np.nan
        close_targets[:] = np.nan
    close_targets[-1] = 0.0

    cash = equity = 1.0
    shares = np.zeros(opens.shape[1])
    daily = []
    for morning, evening, at_open, at_close in zip(
        opens.to_numpy(), closes.to_numpy(), open_targets, close_targets
    ):
        start_equity = equity
        notional = 0.0
        trades = 0
        for price, target in ((morning, at_open), (evening, at_close)):
            holdings = shares * price
            equity = cash + holdings.sum()
            if equity <= 0:
                raise ValueError(f"Portfolio equity was exhausted in the {mode} backtest")
            if np.isnan(target).all():
                continue

            if fee_rate:
                # Size the target on post-cost equity so gross exposure respects the cap.
                current_weights = holdings / equity

                def balance(remaining):
                    traded_fraction = np.abs(target * remaining - current_weights).sum()
                    return remaining + fee_rate * traded_fraction - 1

                if balance(0.0) >= 0:
                    raise ValueError(f"Trading costs exhausted equity in the {mode} backtest")
                equity *= brentq(balance, 0.0, 1.0)
            new_holdings = target * equity
            traded = np.abs(new_holdings - holdings)
            notional += traded.sum()
            trades += np.count_nonzero(traded)
            shares = new_holdings / price
            cash = equity - new_holdings.sum()
        daily.append((equity / start_equity - 1, notional / start_equity, trades))
    return pd.DataFrame(daily, index=opens.index, columns=["return", "turnover", "n_trades"])


def metrics(frame, sessions_per_year):
    returns = frame["return"]
    wealth = (1 + returns).cumprod()
    volatility = returns.std() * np.sqrt(sessions_per_year)
    result = {
        "cagr": wealth.iloc[-1] ** (sessions_per_year / len(frame)) - 1,
        "ann_vol": volatility,
        "sharpe": returns.mean() * sessions_per_year / volatility if volatility > 0 else np.nan,
        "max_drawdown": (wealth / wealth.cummax().clip(lower=1.0) - 1).min(),
        "total_return": wealth.iloc[-1] - 1,
        "turnover": frame["turnover"].mean() * sessions_per_year,
    }
    return {
        **{key: float(value) if np.isfinite(value) else None for key, value in result.items()},
        "n_trades": int(frame["n_trades"].sum()),
        "n_sessions": len(frame),
    }


def print_table(reports, periods):
    columns = {
        "cagr": ("CAGR", ".2%"),
        "ann_vol": ("Ann. vol", ".2%"),
        "sharpe": ("Sharpe", ".2f"),
        "max_drawdown": ("Max DD", ".2%"),
        "total_return": ("Total", ".2%"),
        "turnover": ("Ann. TO", ".2f"),
        "n_trades": ("Fills", ",d"),
    }
    for period, dates in periods.items():
        print(f"{period.upper():<4} {dates[0].date()} to {dates[-1].date()} ({len(dates)} sessions)")
    print(f"{'Portfolio':<23} {'Period':<6}" + "".join(f"{title:>10}" for title, _ in columns.values()))
    for name, report in reports.items():
        for period, values in report.items():
            cells = [
                format(values[key], spec) if values[key] is not None else "n/a"
                for key, (_, spec) in columns.items()
            ]
            print(f"{name:<23} {period.upper():<6}" + "".join(f"{cell:>10}" for cell in cells))


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)

    quality, market, growth = (config["universe"][leg] for leg in ("quality", "market", "growth"))
    symbols = [quality, market, growth]
    if len(set(symbols)) != 3 or not all(re.fullmatch(r"[A-Z0-9.^=-]+", s) for s in symbols):
        raise ValueError("Configure three distinct Yahoo ticker symbols")
    data_start = pd.Timestamp(config["data_start"])
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not (data_start < is_start <= is_end < oos_start <= oos_end):
        raise ValueError("Ensure that the data warmup, IS and OOS ranges are ordered and disjoint")
    lookback = config["beta_lookback_sessions"]
    gross = float(config["gross_exposure"])
    fee_rate = float(config["transaction_cost_bps_per_fill"]) / 10_000
    sessions_per_year = config["sessions_per_year"]
    if not (
        isinstance(lookback, int)
        and lookback >= 2
        and 0 < gross <= 1
        and 0 <= fee_rate < 1
        and sessions_per_year > 0
    ):
        raise ValueError("Check lookback, gross exposure, trading cost and annualization settings")

    opens, closes = load_prices(symbols, data_start, oos_end)
    betas = hedge_betas(opens, closes, market, [quality, growth], lookback).loc[is_start:oos_end]
    if betas.empty or not np.isfinite(betas.to_numpy()).all():
        raise ValueError("A valid hedge is needed for every evaluation date; check warmup and prices")
    opens, closes = opens.loc[betas.index], closes.loc[betas.index]
    periods = {
        "is": betas.loc[is_start:is_end].index,
        "oos": betas.loc[oos_start:oos_end].index,
        "full": betas.index,
    }
    if not all(len(dates) for dates in periods.values()):
        raise ValueError("Each sample range must contain trading sessions")

    def spread_targets(asset):
        beta = betas[asset].to_numpy()
        return np.column_stack((np.ones(len(beta)), -beta)) * (gross / (1 + beta))[:, None]

    quality_targets = spread_targets(quality)
    benchmark_name = f"{market} buy-and-hold"
    books = {
        f"{quality} overnight": ([quality, market], quality_targets, "overnight"),
        f"{quality} intraday": ([quality, market], quality_targets, "intraday"),
        f"{quality} continuous": ([quality, market], quality_targets, "continuous"),
        f"{growth} overnight": ([growth, market], spread_targets(growth), "overnight"),
        benchmark_name: ([market], np.ones((len(betas), 1)), "buy_hold"),
    }
    reports = {}
    for name, (legs, targets, mode) in books.items():
        frame = simulate(opens[legs], closes[legs], targets, mode, fee_rate)
        reports[name] = {
            period: metrics(frame.loc[dates], sessions_per_year) for period, dates in periods.items()
        }

    strategy_name = f"{quality} overnight"
    results = {
        "strategy": config["strategy"],
        "data_range": f"{betas.index[0].date()} to {betas.index[-1].date()}",
        **reports[strategy_name],
        "benchmark": {
            "name": benchmark_name,
            **reports[benchmark_name],
            "comparisons": {
                name: reports[name] for name in (f"{quality} intraday", f"{quality} continuous", f"{growth} overnight")
            },
        },
    }
    print_table(reports, periods)
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()