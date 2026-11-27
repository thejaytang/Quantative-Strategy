"""315_leveraged_funds_robust_replication
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/315_leveraged_funds_robust_replication.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 10f82382008748cd9ff2f56e2b8ef160caa66e38 on 2026-11-27 11:24 Europe/Oslo, exit 0, output sha256 d8aa6aea31e90d4d3e367f5837f1ff90d66038d0dd0b1e0f1ffd5d6b34c4a48d
Data source: 通过 yfinance 获取 Yahoo Finance 的日频 ETF 复权收盘价（QLD、QQQ、SPY、SSO、TQQQ、UPRO），并通过 FRED CSV 获取联邦基金有效利率 DFF。两类数据均优先读取对应的本地 CSV 缓存。（config.yaml:11-17；strategy.py:27-29；strategy.py:64-77；strategy.py:94-105；strategy.py:224-235；README.md:7；README.md:11；README.md:21）
Data fetched: Yahoo Finance 2026-11-27T10:24:19+00:00; FRED 2026-11-27T10:24:21+00:00
Sample period: 价格数据请求自 2011-01-01 至 2025-12-31，2011 年用于滚动预热。实际报告区间为 2012-01-03 至 2025-12-31。（config.yaml:3-9；strategy.py:64-74；strategy.py:214-217；strategy.py:239-246；strategy.py:278-291；README.md:19；logs/run_20261125_140620.log:1）   data_range (from output): 2012-01-03 to 2025-12-31
IS/OOS split: 样本内 2012-01-01 至 2018-12-31，样本外 2019-01-01 至 2025-12-31。按日期切分同一次连续模拟的收益，样本外延续此前的持仓与信号历史。README 记载参数在首次回测前固定，两个区间均未用于调参。（config.yaml:4-9；strategy.py:239-260；strategy.py:271-275；README.md:19；logs/run_20261125_140620.log:2）
Costs and slippage: 每条腿按净单向成交额收取 5 bp，计入入场、再平衡和出场。空头借券费按期初空头权重名义额以年化 30 bp 和实际日历天数/360 计提。主策略的合成复制收益扣除 (L−1) 倍的融资成本，利率为期初已知 DFF 加年化 100 bp，同样按实际日历天数/360 计提。现金及空头抵押品利息按零处理。未建模滑点或市场冲击。（config.yaml:25-29；strategy.py:127-137；strategy.py:157-179；strategy.py:235-237；strategy.py:254-260；README.md:11；README.md:15；README.md:25）
Signal lag: 每周最后一个计划交易日收盘 t，用截至该收盘的价格计算分数。在下一交易日收盘 t+1 模拟执行符合阈值的最低分组合，若原组合仍在持有期内则跳过。执行后的目标权重参与 t+1 至 t+2 的收盘到收盘收益。目标持有期为五个收盘到收盘区间，期间每日收盘再平衡。（config.yaml:19-23；strategy.py:123-154；strategy.py:157-179；strategy.py:226-231；strategy.py:254-260；README.md:13）
Price adjustment: Yahoo Finance dividend- and split-adjusted daily close; fractional adjusted units
Survivorship bias: 固定选择 SSO/SPY、UPRO/SPY、QLD/QQQ、TQQQ/QQQ 四组。价格校验要求各标的覆盖全部计划交易日。README 明示可能存在基金层面的存续和样本选择偏差，研究范围限于所选标的。（config.yaml:11-16；strategy.py:78-88；strategy.py:224-233；README.md:25）
Parameter combinations tried: 2 (one main specification, one zero-financing control)
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.007594995415349137, ann_vol=0.002639441130891839, sharpe=-2.887121205240521, sharpe_excess=-2.9212514211567138, max_drawdown=-0.051854089216154864, total_return=-0.051854089216154864, turnover=12.708795117049679, n_trades=569
    oos: cagr=-0.014934051904091583, ann_vol=0.003428407760174544, sharpe=-4.386981661311098, sharpe_excess=-9.806465754664393, max_drawdown=-0.09975468211786631, total_return=-0.09975468211786631, turnover=20.155210221376894, n_trades=888
    full: cagr=-0.011271333106851134, ann_vol=0.0030679454519118815, sharpe=-3.6931472194596076, sharpe_excess=-6.3655704418292505, max_drawdown=-0.14643608314775347, total_return=-0.14643608314775347, turnover=16.432002669213286, n_trades=1457
    benchmark  full: cagr=0.14791185059635037, ann_vol=0.16577436665061196, sharpe=0.9154259186549065, sharpe_excess=0.8293999692060713, max_drawdown=-0.3356242879788358, total_return=5.867800849711584, turnover=0.07052655169163688, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 315_leveraged_funds_robust_replication\n\ndata_start: '2011-01-01'\nin_sample:\n  start: '2012-01-01'\n  end: '2018-12-31'\nout_of_sample:\n  start: '2019-01-01'\n  end: '2025-12-31'\n\npairs:\n  - {fund: SSO, hedge: SPY, leverage: 2}\n  - {fund: UPRO, hedge: SPY, leverage: 3}\n  - {fund: QLD, hedge: QQQ, leverage: 2}\n  - {fund: TQQQ, hedge: QQQ, leverage: 3}\nbenchmark: SPY\ncash_proxy: BIL\nfunding:\n  target_range: [DFEDTARL, DFEDTARU]\n  availability_lag_days: 1\n\nsignal:\n  cumulative_sessions: 20\n  tracking_error_sessions: 60\n  entry_below: -2.0\n  hold_sessions: 5\n\ncosts:\n  synthetic_funding_spread_bps_annual: 100\n  commission_bps_one_way: 1\n  slippage_bps_one_way: 4\n  short_borrow_bps_annual: 30\n  financing_day_count: 360\n  cash_buffer_fraction: 0.01\n\nannualization_sessions: 252\n"
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
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import requests
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


ROOT = Path(__file__).resolve().parent
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
BPS = 10_000
PRICE_ADJUSTMENT = "Yahoo Finance dividend- and split-adjusted daily close; fractional adjusted units"


class USEquityHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
        Holiday("Carter mourning day", year=2025, month=1, day=9),
        Holiday("Bush mourning day", year=2018, month=12, day=5),
        Holiday("Hurricane Sandy day one", year=2012, month=10, day=29),
        Holiday("Hurricane Sandy day two", year=2012, month=10, day=30),
    ]


def load_prices(tickers, start, end, sessions):
    with TemporaryDirectory(dir=ROOT, prefix="yahoo-") as metadata_cache:
        yf.set_tz_cache_location(metadata_cache)
        for _ in range(2):
            download = yf.download(
                tickers, start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
                auto_adjust=True, progress=False, threads=False, timeout=10, multi_level_index=True,
            )
            if download is None or download.empty or "Close" not in download:
                continue
            prices = download["Close"].sort_index().reindex(columns=tickers)
            values = prices.to_numpy(dtype=float)
            if prices.index.equals(sessions) and np.isfinite(values).all() and (values > 0).all():
                return prices, datetime.now(timezone.utc).isoformat(timespec="seconds")
    raise RuntimeError(f"Yahoo Finance did not return complete daily adjusted closes for {tickers}.")


def load_policy_rate(funding, sessions):
    lower, upper = funding["target_range"]
    # Use the announced target range, allowing a full calendar-date lag on its effective dates.
    observation_dates = sessions - pd.Timedelta(days=funding["availability_lag_days"])
    response = requests.get(
        FRED_CSV, params={"id": f"{lower},{upper}", "cosd": str(observation_dates[0].date()),
                          "coed": str(observation_dates[-1].date())}, timeout=(5, 25),
    )
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text), index_col=0, parse_dates=True)
    limits = frame.reindex(index=observation_dates, columns=[lower, upper])
    if not np.isfinite(limits.to_numpy(dtype=float)).all() or (limits[lower] > limits[upper]).any():
        raise RuntimeError("FRED is missing a valid federal funds target range for a required date.")
    rates = limits.mean(axis=1).set_axis(sessions) / 100
    return rates, datetime.now(timezone.utc).isoformat(timespec="seconds")


def compounded_return(returns, window):
    return np.expm1(np.log1p(returns).rolling(window).sum())


def tracking_scores(returns, pairs, funding_cost, signal):
    scores = {}
    for pair in pairs:
        fund, hedge, leverage = pair["fund"], pair["hedge"], pair["leverage"]
        actual = returns[fund]
        replica = leverage * returns[hedge] - (leverage - 1) * funding_cost
        if (replica <= -1).any():
            raise RuntimeError(f"The synthetic {fund} replica lost all its equity within one session.")
        window = signal["cumulative_sessions"]
        gap = compounded_return(actual, window) - compounded_return(replica, window)
        error = (actual - replica).rolling(signal["tracking_error_sessions"]).std()
        scores[fund] = gap.div(error.where(error > 0))
    return pd.DataFrame(scores)


def trade_targets(scores, weekly_closes, dates, pair_weights, signal):
    qualifiers = scores.loc[weekly_closes].where(lambda score: score < signal["entry_below"])
    winners = qualifiers.dropna(how="all").idxmin(axis=1)
    executions = winners.reindex(scores.index).shift(1).reindex(dates)
    targets = pd.DataFrame(0.0, index=dates, columns=pair_weights.columns)
    entries = pd.Series(0, index=dates, dtype=int)
    next_available = 1  # The initial close is only used to initialize the simulation.
    for idx, fund in enumerate(executions):
        if idx >= next_available and pd.notna(fund):
            next_available = idx + signal["hold_sessions"]
            targets.iloc[idx:next_available] = pair_weights.loc[fund].to_numpy()
            entries.iloc[idx] = 1
    return targets, entries


def settle_fills(positions, desired, cash, close, friction):
    quantities = desired - positions
    trade_values = quantities * close
    # Short proceeds and liabilities offset in free cash. Opening or covering consumes fees.
    free_cash = cash + np.minimum(positions, 0) @ close
    long_cost = (np.maximum(desired, 0) - np.maximum(positions, 0)) * close
    required = long_cost + friction * np.abs(trade_values)
    cash_limited = required.sum() > free_cash
    if cash_limited:
        # Within each simulation a ticker has a fixed side. Reduce only additions.
        adds = np.abs(desired) > np.abs(positions)
        available = free_cash - required[~adds].sum()
        if available < 0 or not adds.any():
            raise RuntimeError("Cash and short collateral are insufficient even before adding exposure.")
        room = max(0.0, available - 1e-12 * max(1.0, cash))
        quantities[adds] *= min(1.0, room / required[adds].sum())
        trade_values = quantities * close

    traded = np.abs(trade_values).sum()
    cash -= trade_values.sum() + friction * traded
    return positions + quantities, cash, traded, np.count_nonzero(quantities), cash_limited


def run_execution(prices, day_fraction, targets, entries, costs, *, rebalance=True):
    friction = (costs["commission_bps_one_way"] + costs["slippage_bps_one_way"]) / BPS
    borrow = costs["short_borrow_bps_annual"] / BPS
    invested_fraction = 1 - costs["cash_buffer_fraction"]
    positions = np.zeros(prices.shape[1])
    cash = equity = 1.0
    output = []
    closes = prices.to_numpy()
    for idx, (previous, close, elapsed, target) in enumerate(zip(
        closes[:-1], closes[1:], day_fraction.to_numpy(), targets.to_numpy(), strict=True
    )):
        cash += (np.minimum(positions, 0) @ previous) * borrow * elapsed
        traded, trades, cash_limited = 0.0, 0, False
        if rebalance or idx == 0:
            desired = (invested_fraction * equity) * target / previous
            positions, cash, traded, trades, cash_limited = settle_fills(
                positions, desired, cash, close, friction
            )
        free_cash = cash + np.minimum(positions, 0) @ close
        if free_cash < 0:
            raise RuntimeError(f"Negative cash after short collateral on {targets.index[idx].date()}.")
        ending_equity = cash + positions @ close
        if not np.isfinite(ending_equity) or ending_equity <= 0:
            raise RuntimeError(f"Nonpositive or invalid equity on {targets.index[idx].date()}.")
        output.append((ending_equity / equity - 1, traded / equity, trades, cash_limited,
                       np.count_nonzero(positions)))
        equity = ending_equity

    daily_results = pd.DataFrame(output, index=targets.index,
                                 columns=["return", "turnover", "n_trades", "cash_limited", "open_legs"])
    daily_results["entries"] = entries
    # Allocate the initial benchmark purchase and its cost to the first reported interval.
    initial, first = daily_results.index[:2]
    daily_results.at[first, "return"] = (
        (1 + daily_results.at[initial, "return"]) * (1 + daily_results.at[first, "return"]) - 1
    )
    for column in ("turnover", "n_trades", "entries"):
        daily_results.at[first, column] += daily_results.at[initial, column]
    daily_results.at[first, "cash_limited"] |= daily_results.at[initial, "cash_limited"]
    return daily_results.iloc[1:]


def summarize(period, cash_returns, sessions_per_year):
    if period.empty:
        raise ValueError("No observations in a configured reporting period.")
    returns = period["return"]
    excess_returns = returns - cash_returns.loc[returns.index]
    annualization = np.sqrt(sessions_per_year)
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax().clip(lower=1) - 1
    wealth = equity.iloc[-1]
    volatility = returns.std()
    excess_volatility = excess_returns.std()
    values = {
        "cagr": wealth ** (sessions_per_year / len(period)) - 1,
        "ann_vol": volatility * annualization,
        "sharpe": returns.mean() / volatility * annualization if volatility > 0 else None,
        "sharpe_excess": (excess_returns.mean() / excess_volatility * annualization
                          if excess_volatility > 0 else None),
        "max_drawdown": drawdown.min(),
        "total_return": wealth - 1,
        "turnover": period["turnover"].mean() * sessions_per_year,
    }
    metrics = {key: float(value) if value is not None and np.isfinite(value) else None
               for key, value in values.items()}
    metrics["n_trades"] = int(period["n_trades"].sum())
    return metrics


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    start = pd.Timestamp(config["data_start"])
    in_sample, out_of_sample = config["in_sample"], config["out_of_sample"]
    is_start, is_end = (pd.Timestamp(in_sample[key]) for key in ("start", "end"))
    oos_start, end = (pd.Timestamp(out_of_sample[key]) for key in ("start", "end"))
    if not start < is_start <= is_end < oos_start <= end:
        raise ValueError("Configure warm-up, in-sample and out-of-sample dates in chronological order.")
    costs = config["costs"]
    commission, slippage = costs["commission_bps_one_way"], costs["slippage_bps_one_way"]
    if (min(commission, slippage, costs["short_borrow_bps_annual"]) < 0 or commission + slippage >= BPS
            or costs["financing_day_count"] <= 0 or not 0 < costs["cash_buffer_fraction"] < 1):
        raise ValueError("Invalid transaction costs, financing day count, or cash buffer.")
    funding = config["funding"]
    if funding["availability_lag_days"] < 1:
        raise ValueError("The policy target must be lagged by at least one calendar day.")

    pairs, benchmark = config["pairs"], config["benchmark"]
    cash_proxy = config["cash_proxy"]
    tickers = sorted({benchmark, *(pair[key] for pair in pairs for key in ("fund", "hedge"))})
    equity_day = pd.offsets.CustomBusinessDay(calendar=USEquityHolidays())
    # Extend the scheduled calendar through the week containing the end date.
    schedule = pd.date_range(start, end + pd.Timedelta(days=7), freq=equity_day)
    week = schedule.to_period("W-FRI")
    week_ends = schedule[~week.duplicated(keep="last")]
    sessions = schedule[schedule <= end]
    prices, yahoo_fetched_at = load_prices(sorted(set(tickers) | {cash_proxy}), start, end, sessions)
    rates, fred_fetched_at = load_policy_rate(funding, sessions)
    all_returns = prices.pct_change(fill_method=None)
    returns, cash_returns = all_returns[tickers], all_returns[cash_proxy]
    day_fraction = sessions.to_series().diff().dt.days / costs["financing_day_count"]
    funding_cost = (rates.shift(1) + costs["synthetic_funding_spread_bps_annual"] / BPS) * day_fraction

    first = sessions.searchsorted(is_start)
    if first < 2 or first == len(sessions):
        raise ValueError("The simulation needs two sessions before the first in-sample return.")
    dates = sessions[first - 1:]
    execution_prices = prices.loc[sessions[first - 2]:, tickers]
    execution_days = day_fraction.loc[dates]

    pair_weights = pd.DataFrame(0.0, index=[pair["fund"] for pair in pairs], columns=tickers)
    for pair in pairs:
        fund, hedge, leverage = pair["fund"], pair["hedge"], pair["leverage"]
        pair_weights.at[fund, fund] = 1 / (1 + leverage)
        pair_weights.at[fund, hedge] = -leverage / (1 + leverage)

    signal = config["signal"]
    weekly_closes = sessions.intersection(week_ends)
    results = {}
    signal_variants = {"Financed signal": funding_cost, "Unfinanced signal": 0.0}
    for label, daily_cost in signal_variants.items():
        scores = tracking_scores(returns, pairs, daily_cost, signal)
        targets, entries = trade_targets(scores, weekly_closes, dates, pair_weights, signal)
        results[label] = run_execution(execution_prices, execution_days, targets, entries, costs)

    entries = pd.Series(0, index=dates, dtype=int)
    entries.iloc[0] = 1
    always_held = pd.DataFrame([pair_weights.mean().to_numpy()] * len(dates), index=dates, columns=tickers)
    buy_hold = pd.DataFrame(0.0, index=dates, columns=tickers)
    buy_hold[benchmark] = 1.0
    benchmark_label = f"{benchmark} buy and hold"
    results["Always-held hedged"] = run_execution(execution_prices, execution_days, always_held, entries, costs)
    results[benchmark_label] = run_execution(
        execution_prices, execution_days, buy_hold, entries, costs, rebalance=False
    )

    table, summary = [], {}
    for label, daily in results.items():
        periods = {"is": daily.loc[is_start:is_end], "oos": daily.loc[oos_start:end], "full": daily}
        summary[label] = {key: summarize(period, cash_returns, config["annualization_sessions"])
                          for key, period in periods.items()}
        for key, metrics in summary[label].items():
            table.append({"Portfolio": label, "Period": key.upper(), **metrics})
    data_range = f"{dates[1].date()} to {dates[-1].date()}"
    results_json = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **summary["Financed signal"],
        "benchmark": {
            "name": benchmark_label,
            **summary[benchmark_label],
            "unfinanced_signal": summary["Unfinanced signal"],
            "always_held_hedged": summary["Always-held hedged"],
        },
    }
    print(f"{config['strategy']} | {data_range}")
    print(f"IS {is_start.date()} to {is_end.date()} | OOS {oos_start.date()} to {end.date()}")
    print(f"PARAM_COMBOS_TRIED: {len(signal_variants)} (one main specification, one zero-financing control)")
    print(f"DATA_FETCHED_AT: Yahoo Finance {yahoo_fetched_at}; FRED {fred_fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"CASH_PROXY: {cash_proxy} adjusted close-to-close return for excess Sharpe")
    display = pd.DataFrame(table)
    for key in ("cagr", "ann_vol", "max_drawdown", "total_return", "turnover"):
        display[key] = display[key].map(lambda value: "—" if pd.isna(value) else f"{value:.2%}")
    print(display.to_string(index=False, float_format=lambda value: f"{value:.3f}", na_rep="—"))
    main_results = results["Financed signal"]
    print(f"FINANCED_ENTRY_OR_RENEWAL: IS {main_results.loc[is_start:is_end, 'entries'].sum()}, "
          f"OOS {main_results.loc[oos_start:end, 'entries'].sum()}, FULL {main_results['entries'].sum()}")
    for label, daily in results.items():
        print(f"EXECUTION: {label} | cash_limited_sessions={daily['cash_limited'].sum()} "
              f"| end_open_legs={daily['open_legs'].iloc[-1]}")
    print("RESULTS_JSON: " + json.dumps(results_json, allow_nan=False))


if __name__ == "__main__":
    main()
