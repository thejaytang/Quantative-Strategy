"""310_factor_beta_overnight_intraday
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/310_factor_beta_overnight_intraday.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c71364ad99535bbf59be998b004cff431ee83048 on 2026-11-27 11:54 Europe/Oslo, exit 0, output sha256 5554d9315b377f7813686841564b6dc7110b73f1c832c5fd4f003c05a1a728b0
Data source: Yahoo Finance 公开日频 ETF 开盘价和收盘价，通过 yfinance 读取。标的为 QUAL（质量代理）、SPY（市场对冲和基准）、IWF（成长风格对照）。优先读取 data_cache/ 中的 CSV，缺失时请求 Yahoo Finance（config.yaml:2-5；strategy.py:17-39,196-199,218,235-246；README.md:7,20）。
Data fetched: QUAL=2026-11-27T10:54:03+00:00, SPY=2026-11-27T10:54:03+00:00, IWF=2026-11-27T10:54:03+00:00, BIL=2026-11-27T10:54:03+00:00
Sample period: 配置的数据请求区间为 2014-01-01 至 2025-12-31（含边界），2014 年仅用于初始化对冲。日志中的实际评估区间为 2015-01-02 至 2025-12-31，共 2,766 个交易日（config.yaml:7-14；strategy.py:32-44,218-226；README.md:18；logs/run_20261125_135059.log:3）。   data_range (from output): 2015-01-02 to 2025-12-31
IS/OOS split: IS 配置区间为 2015-01-01 至 2023-05-31，日志首个交易日为 2015-01-02，共 2,117 个交易日。OOS 为 2023-06-01 至 2025-12-31，共 649 个交易日。每个组合先模拟一条连续路径，再按日期切片统计，切分日不额外平仓。OOS 继续使用截至上月最后一个交易日收盘的数据更新月度 beta（config.yaml:7-14；strategy.py:72-95,218-229,244-249；README.md:14,18；logs/run_20261125_135059.log:1-2）。
Costs and slippage: 按每条腿每次成交的绝对名义金额收取单边 1.0 bp，适用于开仓、平仓和调仓。费用从净值扣除，目标头寸按扣费后净值确定。未建模滑点，模拟成交直接使用复权开盘价或收盘价，未单独模拟价差、冲击或借券费用（config.yaml:18；strategy.py:116-147,205-208,244-248；README.md:14,30）。
Signal lag: 月末交易日 t 收盘后，用最近 126 个完整交易日估计 beta。新权重从下一交易日 t+1 起使用，并固定至该月结束。主隔夜组合最早在 t+1 收盘按新权重建仓，在下一交易日开盘平仓。日内对照从 t+1 开盘建仓、当日收盘平仓，持续持有对照在 t+1 开盘调仓（config.yaml:16；strategy.py:72-95,98-111,116-122,231-241；README.md:11-14）。
Price adjustment: Yahoo Finance yfinance auto_adjust=True (split/dividend-adjusted OHLC)
Survivorship bias: 使用固定的 QUAL、SPY、IWF 历史 ETF 行情，无成分股回建。选择仍存续的 ETF 可能引入标的选择偏差和存续偏差（config.yaml:2-5；strategy.py:26-44,192-199,218；README.md:32）。
Parameter combinations tried: 2
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.05722784595279118, ann_vol=0.016332520089129325, sharpe=-3.5996724393929456, sharpe_excess=-4.136229717989697, max_drawdown=-0.3918087261515014, total_return=-0.39046821231113715, turnover=428.4374749606773, n_trades=8466
    oos: cagr=-0.08796980308291413, ann_vol=0.009076381840579583, sharpe=-10.138867894911591, sharpe_excess=-14.891490001257427, max_drawdown=-0.21183797968667584, total_return=-0.21112559314118962, turnover=428.3027721177002, n_trades=2594
    full: cagr=-0.06453275687716464, ann_vol=0.014974422649961317, sharpe=-4.446864496020203, sharpe_excess=-5.572561725558017, max_drawdown=-0.5191647041034225, total_return=-0.5191559725253567, turnover=428.40586897908224, n_trades=11060
    benchmark  full: cagr=0.1218133588081105, ann_vol=0.16109554983624247, sharpe=0.7944044338064401, sharpe_excess=0.680863311890333, max_drawdown=-0.3072370422517515, total_return=2.5313283036005454, turnover=0.16406179141094573, n_trades=2
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 310_factor_beta_overnight_intraday\nuniverse:\n  quality: QUAL\n  market: SPY\n  growth: IWF\ncash_proxy: BIL\n\n# Date bounds are inclusive. The earlier data is used only to initialize the hedge.\ndata_start: '2014-01-01'\nin_sample:\n  start: '2015-01-01'\n  end: '2023-05-31'\nout_of_sample:\n  start: '2023-06-01'\n  end: '2025-12-31'\n\nbeta_lookback_sessions: 126\ngross_exposure: 1.0\n# Extra room for the price move between open sizing and close settlement.\ncash_buffer: 0.15\ntransaction_cost_bps_per_fill: 1.0\nslippage_bps_per_fill: 1.0\nsessions_per_year: 252\n# The 10% buffer attempt stopped on a gross exposure guard before reporting metrics.\nparam_combos_tried: 2\n"
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

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

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


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_COLUMNS = ["Open", "Close"]
PRICE_ADJUSTMENT = "Yahoo Finance yfinance auto_adjust=True (split/dividend-adjusted OHLC)"


class NyseHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def expected_sessions(start, end):
    # Full-day NYSE Group closures in the study interval. Early closes remain sessions.
    special_closures = pd.DatetimeIndex(["2018-12-05", "2025-01-09"])
    closures = NyseHolidays().holidays(start, end).union(special_closures)
    return pd.bdate_range(start, end).difference(closures)


def load_prices(symbols, start, end):
    CACHE.mkdir(exist_ok=True)
    sessions = expected_sessions(start, end)
    prices = {}
    fetched_at = {}
    # Keep yfinance's auxiliary cache local and separate from the price CSVs.
    with TemporaryDirectory(dir=CACHE) as yahoo_cache:
        yf.set_tz_cache_location(yahoo_cache)
        for symbol in symbols:
            path = CACHE / f"{symbol}_{start.date()}_{end.date()}_adjusted.csv"
            cached = path.exists()
            if cached:
                frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
                stamp = frame.get("fetched_at_utc", pd.Series(dtype="object")).dropna()
                if stamp.empty:
                    modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                    fetched_at[symbol] = modified.isoformat(timespec="seconds") + " (legacy CSV mtime estimate)"
                else:
                    fetched_at[symbol] = stamp.iloc[0]
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
                fetched_at[symbol] = datetime.now(timezone.utc).isoformat(timespec="seconds")

            frame = frame[PRICE_COLUMNS].loc[start:end]
            values = frame.to_numpy()
            if not frame.index.equals(sessions):
                missing = sessions.difference(frame.index).strftime("%Y-%m-%d")[:5].tolist()
                extra = frame.index.difference(sessions).strftime("%Y-%m-%d")[:5].tolist()
                raise ValueError(
                    f"{symbol} dates must match the ordered NYSE sessions "
                    f"(first missing dates {missing}, first extra dates {extra})"
                )
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"Invalid adjusted daily prices for {symbol}")

            if not cached:
                temp = Path(yahoo_cache) / path.name
                snapshot = frame.assign(fetched_at_utc="")
                snapshot.at[frame.index[0], "fetched_at_utc"] = fetched_at[symbol]
                snapshot.to_csv(temp, index_label="date")
                temp.replace(path)
            prices[symbol] = frame

    panel = pd.concat(prices, axis=1)
    return panel.xs("Open", axis=1, level=1), panel.xs("Close", axis=1, level=1), fetched_at


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


def fill_positions(positions, next_positions, prices, cash, fee_rate, slippage_rate, context):
    changes = next_positions - positions
    # Buy above the reference quote and sell below it, including when covering a short.
    fill_values = changes * prices * (1 + np.sign(changes) * slippage_rate)
    notional = np.abs(fill_values).sum()
    cash_after = cash - fill_values.sum() - notional * fee_rate
    marked_positions = next_positions * prices
    equity_after = cash_after + marked_positions.sum()
    if cash_after < 0:
        raise ValueError(f"Predetermined fills would make cash negative ({context}: {cash_after})")
    if np.abs(marked_positions).sum() > equity_after:
        raise ValueError(f"Predetermined fills would exceed 100% gross exposure ({context})")
    return cash_after, notional, np.count_nonzero(changes)


def simulate(opens, closes, targets, mode, cash_buffer, fee_rate, slippage_rate):
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
    positions = np.zeros(opens.shape[1])
    daily_results = []
    for date, morning, evening, at_open, at_close in zip(
        opens.index, opens.to_numpy(), closes.to_numpy(), open_targets, close_targets
    ):
        start_equity = equity
        opening_equity = cash + np.dot(positions, morning)
        if opening_equity <= 0:
            raise ValueError(f"Nonpositive opening equity in {mode} on {date.date()}")
        notional = 0.0
        trades = 0

        if not np.isnan(at_open).all():
            opening_positions = at_open * (opening_equity * (1 - cash_buffer)) / morning
            cash, notional, trades = fill_positions(
                positions, opening_positions, morning, cash, fee_rate, slippage_rate,
                f"{mode} {date.date()} open",
            )
            positions = opening_positions
            opening_equity = cash + np.dot(positions, morning)

        # Set the closing quantities using information available just after the open.
        if not np.isnan(at_close).all():
            closing_positions = at_close * (opening_equity * (1 - cash_buffer)) / morning
            cash, close_notional, close_trades = fill_positions(
                positions, closing_positions, evening, cash, fee_rate, slippage_rate,
                f"{mode} {date.date()} close",
            )
            positions = closing_positions
            notional += close_notional
            trades += close_trades

        equity = cash + np.dot(positions, evening)
        if equity <= 0:
            raise ValueError(f"Nonpositive closing equity in {mode} on {date.date()}")
        daily_results.append((equity / start_equity - 1, notional / start_equity, trades))
    return pd.DataFrame(daily_results, index=opens.index, columns=["return", "turnover", "n_trades"])


def metrics(frame, cash_returns, sessions_per_year):
    returns = frame["return"]
    wealth = (1 + returns).cumprod()
    vol_scale = np.sqrt(sessions_per_year)
    volatility = returns.std() * vol_scale
    excess = returns - cash_returns
    excess_volatility = excess.std() * vol_scale
    result = {
        "cagr": wealth.iloc[-1] ** (sessions_per_year / len(frame)) - 1,
        "ann_vol": volatility,
        "sharpe": returns.mean() * sessions_per_year / volatility if volatility > 0 else np.nan,
        "sharpe_excess": (
            excess.mean() * sessions_per_year / excess_volatility if excess_volatility > 0 else np.nan
        ),
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
        "sharpe_excess": ("Ex. SR", ".2f"),
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
    cash_proxy = config["cash_proxy"]
    symbols = [quality, market, growth, cash_proxy]
    if len(set(symbols)) != 4 or not all(re.fullmatch(r"[A-Z]{1,5}", s) for s in symbols):
        raise ValueError("Configure four distinct U.S. ETF ticker symbols")
    data_start = pd.Timestamp(config["data_start"])
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not (data_start < is_start <= is_end < oos_start <= oos_end):
        raise ValueError("Ensure that the data warmup, IS and OOS ranges are ordered and disjoint")
    lookback = config["beta_lookback_sessions"]
    gross = float(config["gross_exposure"])
    cash_buffer = float(config["cash_buffer"])
    fee_rate = float(config["transaction_cost_bps_per_fill"]) / 10_000
    slippage_rate = float(config["slippage_bps_per_fill"]) / 10_000
    sessions_per_year = config["sessions_per_year"]
    if not (
        isinstance(lookback, int)
        and lookback >= 2
        and 0 < gross <= 1
        and 0 < cash_buffer < 1
        and 0 <= fee_rate < 1
        and 0 <= slippage_rate < 1
        and sessions_per_year > 0
    ):
        raise ValueError("Check lookback, exposure, cash buffer, costs and annualization settings")

    opens, closes, fetched_at = load_prices(symbols, data_start, oos_end)
    cash_returns = closes[cash_proxy].pct_change(fill_method=None)
    hedge_symbols = [quality, market, growth]
    betas = hedge_betas(
        opens[hedge_symbols], closes[hedge_symbols], market, [quality, growth], lookback
    ).loc[is_start:oos_end]
    if betas.empty or not np.isfinite(betas.to_numpy()).all():
        raise ValueError("A valid hedge is needed for every evaluation date; check warmup and prices")
    opens, closes = opens.loc[betas.index], closes.loc[betas.index]
    cash_returns = cash_returns.loc[betas.index]
    if not np.isfinite(cash_returns.to_numpy()).all():
        raise ValueError("A valid cash proxy return is needed for every evaluation date")
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
    portfolios = {
        f"{quality} overnight": ([quality, market], quality_targets, "overnight"),
        f"{quality} intraday": ([quality, market], quality_targets, "intraday"),
        f"{quality} continuous": ([quality, market], quality_targets, "continuous"),
        f"{growth} overnight": ([growth, market], spread_targets(growth), "overnight"),
        benchmark_name: ([market], np.ones((len(betas), 1)), "buy_hold"),
    }
    reports = {}
    for name, (legs, targets, mode) in portfolios.items():
        frame = simulate(opens[legs], closes[legs], targets, mode, cash_buffer, fee_rate, slippage_rate)
        reports[name] = {
            period: metrics(frame.loc[dates], cash_returns.loc[dates], sessions_per_year)
            for period, dates in periods.items()
        }

    strategy_name = f"{quality} overnight"
    results = {
        "strategy": config["strategy"],
        "data_range": f"{betas.index[0].date()} to {betas.index[-1].date()}",
        **reports[strategy_name],
        "benchmark": {
            "name": benchmark_name,
            "cash_proxy": cash_proxy,
            **reports[benchmark_name],
            "comparisons": {
                name: reports[name] for name in (f"{quality} intraday", f"{quality} continuous", f"{growth} overnight")
            },
        },
    }
    print(f"PARAM_COMBOS_TRIED: {config['param_combos_tried']}")
    print("DATA_FETCHED_AT: " + ", ".join(f"{symbol}={stamp}" for symbol, stamp in fetched_at.items()))
    print("PRICE_ADJUSTMENT: " + PRICE_ADJUSTMENT)
    print_table(reports, periods)
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
