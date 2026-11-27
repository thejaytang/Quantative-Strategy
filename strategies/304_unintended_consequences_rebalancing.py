"""304_unintended_consequences_rebalancing
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/304_unintended_consequences_rebalancing.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 2b23c5bbf706aea933c0cba5f5805efb450ee206 on 2026-11-26 14:27 Europe/Oslo, exit 0, output sha256 8167cfeb0dc91155e2f0fdf054f20ebfd863fbeb0a3abf02c599e494f8786bc8
Data source: Yahoo Finance 公开 ETF 日频开盘价和收盘价，经 yfinance 下载或从 `data_cache/` 缓存读取。股票为 SPY，债券为 IEF。
Data fetched: 2026-11-26 14:27 Europe/Oslo (run time)
Sample period: 行情请求区间为 2006-10-01 至 2026-10-30，2006 年 10 月起用于预热。该次运行使用的开盘估值区间为 2007-01-03 至 2026-10-30，首个收益日期为 2007-01-04。 data_range (from output): 2007-01-03 to 2026-10-30
IS/OOS split: 配置样本内为 2007-01-01 至 2025-03-31，样本外为 2025-04-01 至 2026-10-30。实际样本内收益自 2007-01-04 开始。收益按区间结束的开盘日归属，首个样本外区间从 2025-03-31 开盘持有至 2025-04-01 开盘，组合状态跨边界延续。README 将样本外描述为固定规则的历史留出检验，并记载两段均未调参。
Costs and slippage: 三条组合均在每次开盘按买入金额和卖出金额各收取 2 bps，涵盖首次从现金建仓及后续按漂移后权重调仓。换手率为买卖金额之和除以成交前净值，目标权重以扣费后净值为基准，净增长因子为开盘至开盘的毛增长因子乘以 (1 − 费率 × 换手率)。未建模滑点，成交以复权开盘价模拟。
Signal lag: 阈值策略和单日反转基准在交易日 t 收盘后根据截至当日的收盘收益生成信号，并按 t 日做月界过滤。目标权重向后移一个交易日，在 t+1 开盘成交并持有至 t+2 开盘。恒定权重基准在各期起始开盘使用固定目标。
Price adjustment: 下载使用 `auto_adjust=True`。缓存文件名以 `yahoo_adjusted_` 开头，缓存路径保留同一组开盘价和收盘价列。信号使用复权收盘收益，成交估值使用复权开盘价。README 记载 ETF 收益包含分配和产品费用的影响，并提示供应商可能修订复权行情。
Survivorship bias: 固定使用 SPY 和 IEF 两只历史 ETF，无个股成分股回填。事后选择现存 ETF 仍有产品选择偏差。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.08504310807922866, ann_vol=0.11354916586810568, sharpe=0.7757690140581055, max_drawdown=-0.24235691337852994, total_return=3.4208012606708316, turnover=207.02584815266658
    oos: cagr=0.1505930788854566, ann_vol=0.1135745115977659, sharpe=1.292024235799741, max_drawdown=-0.08136397241218807, total_return=0.24870384519681665, turnover=11.699755593567938
    full: cagr=0.09014626117652913, ann_vol=0.11354422903840392, sharpe=0.8171238877819083, max_drawdown=-0.24235691337852994, total_return=4.5202715330506145, turnover=218.72560374623453
    benchmark constant_weight full: cagr=0.08284328286703246, ann_vol=0.10892604333333947, sharpe=0.7853282475611681, max_drawdown=-0.3158588672605598, total_return=3.832585554561005, turnover=22.985372166014624
    benchmark one_day_reversal full: cagr=0.08573170677514574, ann_vol=0.1232627341416053, sharpe=0.7291028160612114, max_drawdown=-0.2594910579297136, total_return=4.094236482162026, turnover=1762.3105949832534
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 304_unintended_consequences_rebalancing\ncategory: momentum/reversal\nuniverse:\n  equity: SPY\n  bonds: IEF\n\ntarget_equity_weight: 0.60\ndrift_band: 0.01\nequity_tilt: 0.20\nexclude_month_start_sessions: 2\nexclude_month_end_sessions: 2\ntransaction_cost_bps: 2.0  # Per dollar bought or sold at the open.\nannualization_sessions: 252\n\nwarmup_start: 2006-10-01\nin_sample:\n  start: 2007-01-01\n  end: 2025-03-31\nout_of_sample:\n  start: 2025-04-01\n  end: 2026-10-30\n'
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
import math
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
from pandas.tseries.offsets import CustomBusinessDay


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
DAY = pd.Timedelta(days=1)


class NyseRegularHolidays(AbstractHolidayCalendar):
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
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def load_prices(tickers, start, end):
    cache_path = CACHE / f"yahoo_adjusted_{'_'.join(tickers)}_{start.date()}_{end.date()}.csv"
    columns = pd.MultiIndex.from_product([["Open", "Close"], tickers])
    cached = cache_path.is_file()
    if cached:
        prices = pd.read_csv(cache_path, header=[0, 1], index_col=0, parse_dates=True)
    else:
        CACHE.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix=".yfinance-", dir=CACHE) as yahoo_cache:
            yf.set_tz_cache_location(yahoo_cache)
            prices = yf.download(tickers, start=str(start.date()), end=str((end + DAY).date()),
                                 interval="1d", auto_adjust=True, ignore_tz=True,
                                 threads=False, progress=False, timeout=15)

    prices = prices.reindex(columns=columns).sort_index().loc[start:end]
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("Yahoo Finance supplied no sessions or duplicate dates.")
    if prices.index[0] > start + 7 * DAY or prices.index[-1] < end - 7 * DAY:
        raise ValueError(f"Incomplete price coverage for {start.date()} through {end.date()}.")
    invalid = ~np.isfinite(prices) | prices.le(0)
    if invalid.any().any():
        date = invalid.index[invalid.any(axis=1)][0]
        fields = ", ".join(f"{ticker} {field}" for field, ticker in invalid.columns[invalid.loc[date]])
        raise ValueError(f"Missing or invalid adjusted price on {date.date()} for {fields}.")

    if not cached:
        with NamedTemporaryFile(mode="w", dir=CACHE, suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            try:
                prices.to_csv(handle, index_label="Date")
                handle.close()
                temporary_path.replace(cache_path)
            finally:
                temporary_path.unlink(missing_ok=True)
    return prices


def month_edge_mask(dates, exclude_first, exclude_last):
    # At each close the month's elapsed sessions are known. Rank the last sessions
    # by the regular NYSE schedule so surprise closures cannot rewrite earlier signals.
    observed = pd.Series(dates, index=dates)
    sessions_elapsed = observed.groupby(dates.to_period("M")).cumcount()
    months = dates.to_period("M")
    scheduled = pd.date_range(months[0].start_time, months[-1].end_time.normalize(),
                              freq=CustomBusinessDay(calendar=NyseRegularHolidays()))
    scheduled_dates = pd.Series(scheduled, index=scheduled)
    remaining = scheduled_dates.groupby(scheduled.to_period("M")).cumcount(ascending=False)
    remaining = remaining.reindex(dates)
    if remaining.isna().any():
        raise ValueError(f"Unexpected NYSE session date {remaining.index[remaining.isna()][0].date()}.")
    return (sessions_elapsed < exclude_first) | (remaining < exclude_last)


def threshold_crossings(close_returns, target, band):
    crossings = np.zeros(len(close_returns), dtype=int)
    weight = target
    for i, (equity_return, bond_return) in enumerate(close_returns.to_numpy()):
        equity_value = weight * (1 + equity_return)
        bond_value = (1 - weight) * (1 + bond_return)
        weight = equity_value / (equity_value + bond_value)
        if weight > target + band:
            crossings[i] = 1
        elif weight < target - band:
            crossings[i] = -1
        if crossings[i]:
            weight = target
    return pd.Series(crossings, index=close_returns.index)


def simulate(open_growth, equity_target, fee_rate):
    # Rows initially represent entry opens. Start from cash, then carry the
    # drifted weights forward to the next open before trading to new targets.
    equity_growth, bond_growth = open_growth.to_numpy().T
    gross_growth = equity_target * equity_growth + (1 - equity_target) * bond_growth
    prior_weight = (equity_target * equity_growth / gross_growth).shift(1)
    delta = equity_target - prior_weight

    # With one buy and one sell, solve for turnover when the post-trade
    # target weights are fractions of NAV after both fees have been paid.
    turnover = 2 * delta.abs() / (1 + fee_rate * np.sign(delta) * (2 * equity_target - 1))
    turnover.iloc[0] = 1 / (1 + fee_rate)
    net_return = (1 - fee_rate * turnover) * gross_growth - 1
    return pd.DataFrame({"return": net_return, "turnover": turnover})


def metrics(daily, annualization):
    returns = daily["return"]
    equity = (1 + returns).cumprod()
    vol = returns.std(ddof=1)
    values = {
        "cagr": equity.iloc[-1] ** (annualization / len(returns)) - 1,
        "ann_vol": vol * math.sqrt(annualization),
        "sharpe": returns.mean() / vol * math.sqrt(annualization) if vol > 0 else None,
        "max_drawdown": (equity / equity.cummax().clip(lower=1) - 1).min(),
        "total_return": equity.iloc[-1] - 1,
        "turnover": daily["turnover"].sum(),
    }
    return {key: float(value) if value is not None and math.isfinite(value) else None
            for key, value in values.items()}


def print_results(strategy, periods, performance, labels, entry_start):
    data_range = f"{entry_start.date()} to {periods['full'][-1].date()}"
    print(f"{strategy} | Opening marks: {data_range} | Returns dated by ending open")
    for period in ("is", "oos"):
        dates = periods[period]
        print(f"{period.upper()} return dates: {dates[0].date()} to {dates[-1].date()}")
    print(f"{'Portfolio':<22} {'Period':<6} {'CAGR':>9} {'Ann.Vol':>9} {'Sharpe':>8} "
          f"{'Max.DD':>9} {'Return':>9} {'Turnover':>9}")
    formats = {"cagr": ".2%", "ann_vol": ".2%", "sharpe": ".2f", "max_drawdown": ".2%",
               "total_return": ".2%", "turnover": ".2f"}
    for portfolio, segments in performance.items():
        for period, values in segments.items():
            cagr, vol, sharpe, drawdown, total, turnover = (
                format(values[key], spec) if values[key] is not None else "n/a"
                for key, spec in formats.items()
            )
            print(f"{labels[portfolio]:<22} {period.upper():<6} {cagr:>9} {vol:>9} {sharpe:>8} "
                  f"{drawdown:>9} {total:>9} {turnover:>9}")

    threshold = performance["threshold"]
    for period, values in threshold.items():
        print(f"{period.upper()} crossings: {values['n_crossings']} | "
              f"Calendar skips: {values['n_skipped_crossings']} | Tilts: {values['n_tilt_sessions']}")

    result = {"strategy": strategy, "data_range": data_range, **threshold,
              "benchmark": {key: performance[key] for key in labels if key != "threshold"}}
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


def main():
    with (ROOT / "config.yaml").open() as config_file:
        config = yaml.safe_load(config_file)

    tickers = [config["universe"][asset] for asset in ("equity", "bonds")]
    target, band, tilt = (config[key] for key in ("target_equity_weight", "drift_band", "equity_tilt"))
    exclude_first = config["exclude_month_start_sessions"]
    exclude_last = config["exclude_month_end_sessions"]
    fee_rate = config["transaction_cost_bps"] / 10_000
    annualization = config["annualization_sessions"]
    warmup_start = pd.Timestamp(config["warmup_start"])
    is_start, is_end = (pd.Timestamp(config["in_sample"][edge]) for edge in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][edge]) for edge in ("start", "end"))
    if not (
        warmup_start.to_period("M") < is_start.to_period("M")
        and is_start < is_end < oos_start <= oos_end
    ):
        raise ValueError("Provide a warm-up month and ordered, non-overlapping sample dates.")
    max_shift = min(target, 1 - target)
    if len(set(tickers)) != 2 or not (0 < band < max_shift and 0 < tilt <= max_shift):
        raise ValueError("Use two distinct ETFs and positive band and tilt values within the weight bounds.")
    if not (0 <= fee_rate < 1 and annualization > 0 and exclude_first >= 0 and exclude_last >= 0):
        raise ValueError("Invalid transaction cost, annualization or calendar exclusions.")

    prices = load_prices(tickers, warmup_start, oos_end)
    blocked = month_edge_mask(prices.index, exclude_first, exclude_last)
    close_returns = prices["Close"].pct_change().iloc[1:]
    crossings = threshold_crossings(close_returns, target, band).reindex(prices.index, fill_value=0)
    eligible_crossings = crossings.mask(blocked, 0)
    relative_return = close_returns[tickers[0]] - close_returns[tickers[1]]
    reversal = np.sign(relative_return).reindex(prices.index, fill_value=0).mask(blocked, 0)
    # Close-t observations first become positions at the next session's open.
    targets = {
        "threshold": target - tilt * eligible_crossings.shift(1, fill_value=0),
        "constant_weight": pd.Series(target, index=prices.index),
        "one_day_reversal": target - tilt * reversal.shift(1, fill_value=0),
    }

    opens = prices["Open"].loc[is_start:oos_end]
    entry_dates, exit_dates = opens.index[:-1], opens.index[1:]
    periods = {"is": exit_dates[exit_dates <= is_end], "oos": exit_dates[exit_dates >= oos_start],
               "full": exit_dates}
    if any(dates.empty for dates in periods.values()):
        raise ValueError("Each period requires at least one completed open-to-open return.")
    open_growth = opens.shift(-1).div(opens).loc[entry_dates]

    performance = {}
    for name, weights in targets.items():
        daily = simulate(open_growth, weights.loc[entry_dates], fee_rate).set_axis(exit_dates)
        performance[name] = {period: metrics(daily.loc[dates], annualization)
                             for period, dates in periods.items()}

    realized_crossings = crossings.shift(1, fill_value=0).loc[entry_dates].set_axis(exit_dates)
    realized_tilts = eligible_crossings.shift(1, fill_value=0).loc[entry_dates].set_axis(exit_dates)
    for period, dates in periods.items():
        n_crossings = int(realized_crossings.loc[dates].ne(0).sum())
        n_tilts = int(realized_tilts.loc[dates].ne(0).sum())
        performance["threshold"][period].update(n_crossings=n_crossings,
                                                n_skipped_crossings=n_crossings - n_tilts,
                                                n_tilt_sessions=n_tilts)

    labels = {"threshold": "Threshold reversal",
              "constant_weight": f"{100 * target:g}/{100 * (1 - target):g} constant",
              "one_day_reversal": "One-day reversal"}
    print_results(config["strategy"], periods, performance, labels, entry_dates[0])


if __name__ == "__main__":
    main()
