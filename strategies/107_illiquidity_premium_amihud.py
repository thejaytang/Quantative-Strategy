"""107_illiquidity_premium_amihud
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/107_illiquidity_premium_amihud.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit da9754ee51e8d74f175ab50e14ef28c47c764ad1 on 2026-11-27 12:03 Europe/Oslo, exit 0, output sha256 736d75bcc4c9db7bc1399c97e03e499f638495ebc083a6ed3e4331cd91dafbec
Data source: 免费 Yahoo Finance（yfinance）日线 Open、Close、Adj Close、Volume，优先读取 data_cache/ 中的行情 CSV。固定 US24 美股篮子：AAPL、MSFT、AMZN、GOOGL、META、NVDA、JPM、BAC、GS、MS、XOM、CVX、COP、SLB、WMT、COST、PG、KO、PEP、JNJ、PFE、MRK、HD、LOW。（README.md:3,18；config.yaml:2-4；strategy.py:17,21-45,163-181）
Data fetched: 2026-11-27T12:03:33+01:00
Sample period: 日志实际日收益区间为 2016-01-04 至 2025-12-31。配置从 2015-09-01 请求预热数据，样本内起点之前的数据仅用于信号预热。（logs/run_20261125_095950.log:1,31；config.yaml:6-13；strategy.py:181,190-200,207-220）   data_range (from output): 2016-01-04 to 2025-12-31
IS/OOS split: IS 为 2016-01-01 至 2020-12-31，OOS 为 2021-01-01 至 2025-12-31，均含端点。模拟在 IS 起点以现金启动，可使用此前形成的月末信号，OOS 延续 IS 持仓。README 记载无拟合、参数搜索或按 OOS 表现修改规则。（config.yaml:6-17；README.md:15-16；strategy.py:62-80,95-105,196-205；logs/run_20261125_095950.log:2）
Costs and slippage: 基准成本为单边 5 bps，敏感性情景为 0 和 15 bps。开盘调仓时按实际买入与卖出金额的绝对值之和计费，费用从组合净值扣除，目标权重按扣费后净值计算。未单独建模滑点、买卖价差或开盘冲击。（config.yaml:20-22；strategy.py:95-124,176-179,193-200；README.md:17,24）
Signal lag: 使用月末最后一个交易日 t 的收盘数据形成排序，按下一交易日 t+1 的复权开盘价模拟调仓。隔夜收益归原持仓，开盘至收盘收益归调仓后持仓。（strategy.py:62-80,107-124,184-194；README.md:9,17）
Price adjustment: Yahoo split-adjusted Open/Close/Volume and dividends (provider spin-off factors retained); Adj Close for Amihud and BIL returns only; equity distributions accrued separately
Survivorship bias: 事后选定的固定 US24 存续大盘股篮子缺少退市股和小盘股，存在幸存者偏差及成分选择偏差。IS 与 OOS 使用同一篮子。（config.yaml:2-4；README.md:22；strategy.py:163-168,181-201）
Parameter combinations tried: 3
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.1324717611277788, ann_vol=0.24235964161154347, sharpe=0.6353494888675097, sharpe_excess=0.5945469854821412, max_drawdown=-0.4427229243011289, total_return=0.8617549308470001, turnover=9.816605803358502, n_trades=382
    oos: cagr=0.18375116759955046, ann_vol=0.1885532124419667, sharpe=0.9890520296965588, sharpe_excess=0.8259556400906399, max_drawdown=-0.18480581879442914, total_return=1.3165865052565433, turnover=7.580965895568957, n_trades=378
    full: cagr=0.15778681683311424, ann_vol=0.21713128693470618, sharpe=0.7839042842398569, sharpe_excess=0.6904660052994563, max_drawdown=-0.4427229243011289, total_return=3.312916348895012, turnover=17.397571698927457, n_trades=760
    benchmark  full: cagr=0.17473831923645045, ann_vol=0.1597028760977052, sharpe=1.0888460489461547, sharpe_excess=0.9619420784111493, max_drawdown=-0.31006223877029426, total_return=3.9859289891052656, turnover=5.931660810140919, n_trades=2880
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 107_illiquidity_premium_amihud\nuniverse:\n  [AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, BAC, GS, MS, XOM, CVX,\n   COP, SLB, WMT, COST, PG, KO, PEP, JNJ, PFE, MRK, HD, LOW]\ncash_proxy: BIL\n\n# Inclusive dates. Pre-IS data are used only for signal and return warmup.\ndata_start: 2015-09-01\nin_sample:\n  start: 2016-01-01\n  end: 2020-12-31\nout_of_sample:\n  start: 2021-01-01\n  end: 2025-12-31\n\nlookback_sessions: 63\nmin_observations: 50\nquartile_fraction: 0.25\nannualization_sessions: 252\n\nexecution:\n  cash_buffer_fraction: 0.10\n  distribution_delay_days: 60\n\n# One-way basis points, with the primary scenario first.\ncost_scenarios:\n  - {fee_bps: 1, slippage_bps: 4}\n  - {fee_bps: 0, slippage_bps: 0}\n  - {fee_bps: 1, slippage_bps: 14}\n'
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
"""C08 Amihud illiquidity sort on the fixed US24 stock basket."""

from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
FIELDS = ["Open", "Close", "Adj Close", "Volume", "Dividends"]
PORTFOLIOS = ("High illiquidity", "Low illiquidity", "US24 equal weight")
PRICE_ADJUSTMENT = (
    "Yahoo split-adjusted Open/Close/Volume and dividends (provider spin-off factors retained); "
    "Adj Close for Amihud and BIL returns only; equity distributions accrued separately"
)


def market_sessions(start, end):
    # NYSE full-session closures for the configured 2015–2025 research sample.
    if start.year < 2015 or end.year > 2025:
        raise ValueError("Extend the NYSE holiday rules before changing the calendar range")
    holidays = AbstractHolidayCalendar(rules=[
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]).holidays(start, end)
    mourning_days = pd.to_datetime(["2018-12-05", "2025-01-09"])
    return pd.bdate_range(start, end).difference(holidays.union(mourning_days))


def validate_sessions(actual, expected):
    if not actual.is_unique or not actual.is_monotonic_increasing:
        raise RuntimeError("Yahoo Finance session dates are duplicated or out of order")
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    if len(missing) or len(extra):
        raise RuntimeError(f"NYSE session mismatch: missing={missing.date}, unexpected={extra.date}")


def load_prices(tickers, cash_proxy, start, end, sessions):
    fields_by_ticker = {ticker: FIELDS for ticker in tickers}
    fields_by_ticker[cash_proxy] = ["Adj Close"]
    prices = {}
    with TemporaryDirectory(dir=ROOT, prefix=".yfinance-") as cache:
        yf.set_tz_cache_location(cache)
        for _ in range(2):
            pending = [ticker for ticker in fields_by_ticker if ticker not in prices]
            if not pending:
                break
            downloaded = yf.download(
                pending, start=start.date(), end=(end + pd.Timedelta(days=1)).date(),
                auto_adjust=False, actions=True, ignore_tz=True, group_by="column",
                progress=False, threads=8, timeout=15, multi_level_index=True,
            )
            for ticker in pending:
                if ticker not in downloaded.columns.get_level_values(1):
                    continue
                bars = downloaded.xs(ticker, axis=1, level=1).reindex(columns=fields_by_ticker[ticker])
                if bars.notna().any().all():
                    prices[ticker] = bars.dropna(how="all")

    missing = fields_by_ticker.keys() - prices.keys()
    if missing:
        raise RuntimeError(f"Yahoo Finance returned incomplete data for {', '.join(sorted(missing))}")
    for bars in prices.values():
        if not bars.index.is_unique or not bars.index.is_monotonic_increasing:
            raise RuntimeError("Yahoo Finance session dates are duplicated or out of order")
    panel = pd.concat(prices, axis=1).swaplevel(axis=1).sort_index()
    validate_sessions(panel.index, sessions)
    fetched_at = datetime.now(ZoneInfo("Europe/Oslo")).isoformat(timespec="seconds")
    return panel, fetched_at


def portfolio_targets(scores, signal_closes, fraction):
    targets = {name: {} for name in PORTFOLIOS}
    months = scores.index.to_period("M")
    # Each target is keyed by the session when its prior-close signal will fill.
    for next_session in np.flatnonzero(months[1:] != months[:-1]) + 1:
        close = signal_closes.iloc[next_session - 1]
        ranked = scores.iloc[next_session - 1].where(close.notna()).dropna()
        ranked = ranked.sort_values(kind="stable")
        size = math.floor(len(ranked) * fraction)
        groups = (
            ranked.index[-size:] if size else ranked.index[:0],
            ranked.index[:size],
            close.dropna().index,
        )
        for name, chosen in zip(PORTFOLIOS, groups):
            weights = pd.Series(0.0, index=scores.columns)
            if len(chosen):
                weights.loc[chosen] = 1.0 / len(chosen)
            targets[name][next_session] = weights.to_numpy()
    return targets


def require_prices(prices, active, date, tickers, field):
    invalid = active & (~np.isfinite(prices) | (prices <= 0))
    if invalid.any():
        raise RuntimeError(f"Missing or invalid {field} on {date.date()} for {', '.join(tickers[invalid])}")


def size_at_prior_close(positions, cash, previous_close, weights, cash_buffer, date, tickers):
    held = positions > 0
    selected = weights > 0
    require_prices(previous_close, held | selected, date, tickers, "sizing close")
    sizing_value = cash + positions[held] @ previous_close[held]
    desired_positions = np.zeros_like(positions)
    desired_positions[selected] = (
        (1.0 - cash_buffer) * sizing_value * weights[selected] / previous_close[selected]
    )
    return desired_positions


def fill_at_open(positions, desired_positions, cash, unpaid_dividends, opening_prices, fees, slippage, date, tickers):
    changes = desired_positions - positions
    traded = changes != 0
    held = positions > 0
    require_prices(opening_prices, held | traded, date, tickers, "fill open")
    opening_value = cash + unpaid_dividends + positions[held] @ opening_prices[held]
    fill_prices = opening_prices[traded] * (1.0 + np.sign(changes[traded]) * slippage)
    traded_values = changes[traded] * fill_prices
    gross_traded = np.abs(traded_values).sum()
    cash_after_fill = cash - traded_values.sum() - fees * gross_traded
    if cash_after_fill < 0:
        raise RuntimeError(f"Predetermined shares would leave negative cash on {date.date()}: "
                           f"{cash_after_fill / opening_value:.2%} of opening value")
    return cash_after_fill, gross_traded / opening_value, np.count_nonzero(traded)


def simulate(open_prices, close_prices, dividends, targets, start, end, costs, execution):
    fees = costs["fee_bps"] / 10_000.0
    slippage = costs["slippage_bps"] / 10_000.0
    cash_buffer = execution["cash_buffer_fraction"]
    distribution_delay = pd.Timedelta(days=execution["distribution_delay_days"])
    dates = close_prices.index
    tickers = close_prices.columns
    first = dates.searchsorted(start, side="left")
    stop = dates.searchsorted(end, side="right")
    opens, closes, distributions = (frame.to_numpy() for frame in (open_prices, close_prices, dividends))
    positions = np.zeros(len(tickers))
    cash = previous_value = 1.0
    scheduled_distributions = deque()
    daily_returns = np.empty(stop - first)
    daily_turnover = np.zeros(stop - first)
    daily_trades = np.zeros(stop - first, dtype=int)

    for row in range(first, stop):
        date = dates[row]
        desired_positions = None
        if row in targets:
            desired_positions = size_at_prior_close(
                positions, cash, closes[row - 1], targets[row], cash_buffer, dates[row - 1], tickers,
            )

        # The pre-open positions own today's distributions. The historical feed
        # supplies ex-dates; the delay models availability of the cash separately.
        held = positions > 0
        per_share = distributions[row, held]
        if not (np.isfinite(per_share) & (per_share >= 0)).all():
            raise RuntimeError(f"Missing or invalid distribution data on {date.date()}")
        distribution = positions[held] @ per_share
        if distribution:
            scheduled_distributions.append((date + distribution_delay, distribution))
        while scheduled_distributions and scheduled_distributions[0][0] <= date:
            _, paid = scheduled_distributions.popleft()
            cash += paid
        unpaid_dividends = sum(amount for _, amount in scheduled_distributions)

        if desired_positions is not None:
            cash, daily_turnover[row - first], daily_trades[row - first] = fill_at_open(
                positions, desired_positions, cash, unpaid_dividends, opens[row], fees, slippage, date, tickers,
            )
            positions = desired_positions

        held = positions > 0
        require_prices(closes[row], held, date, tickers, "valuation close")
        closing_value = cash + unpaid_dividends + positions[held] @ closes[row, held]
        daily_returns[row - first] = closing_value / previous_value - 1.0
        previous_value = closing_value

    return pd.DataFrame(
        {"return": daily_returns, "turnover": daily_turnover, "n_trades": daily_trades},
        index=dates[first:stop],
    )


def metrics(daily, cash_returns, sessions_per_year):
    returns = daily["return"]
    n = len(returns)
    if not n:
        raise ValueError("A configured reporting period has no trading sessions")
    excess = returns - cash_returns.loc[returns.index]
    growth = np.r_[1.0, (1.0 + returns).cumprod().to_numpy()]
    volatility = returns.std(ddof=1) * math.sqrt(sessions_per_year)
    excess_vol = excess.std(ddof=1) * math.sqrt(sessions_per_year)
    values = {
        "cagr": growth[-1] ** (sessions_per_year / n) - 1.0,
        "ann_vol": volatility,
        "sharpe": returns.mean() * sessions_per_year / volatility if volatility > 0 else np.nan,
        "sharpe_excess": excess.mean() * sessions_per_year / excess_vol if excess_vol > 0 else np.nan,
        "max_drawdown": (growth / np.maximum.accumulate(growth) - 1.0).min(),
        "total_return": growth[-1] - 1.0,
        "turnover": daily["turnover"].sum(),
    }
    return {
        **{key: float(value) if np.isfinite(value) else None for key, value in values.items()},
        "n_trades": int(daily["n_trades"].sum()),
    }


def print_metrics(results):
    fields = (
        ("cagr", "CAGR", ".2%"), ("ann_vol", "Ann vol", ".2%"),
        ("sharpe", "Sharpe", ".2f"), ("sharpe_excess", "Ex. Sharpe", ".2f"),
        ("max_drawdown", "Max DD", ".2%"), ("total_return", "Total", ".2%"),
        ("turnover", "Turnover", ".2f"), ("n_trades", "Trades", "d"),
    )
    headings = " ".join(f"{label:>10}" for _, label, _ in fields)
    print(f"{'Portfolio':<20} {'Cost bps':>8} {'Period':<6} {headings}")
    for cost, portfolios in results.items():
        for name, periods in portfolios.items():
            for period, values in periods.items():
                cells = [format(values[key], spec) if values[key] is not None else "n/a"
                         for key, _, spec in fields]
                row = " ".join(f"{cell:>10}" for cell in cells)
                print(f"{name:<20} {cost:>8g} {period.upper():<6} {row}")


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    tickers = config["universe"]
    cash_proxy = config["cash_proxy"]
    data_start = pd.Timestamp(config["data_start"])
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not data_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Configure warmup data and chronologically ordered IS and OOS ranges")
    lookback = config["lookback_sessions"]
    minimum = config["min_observations"]
    fraction = config["quartile_fraction"]
    if not 2 <= minimum <= lookback or not 0 < fraction <= 0.5:
        raise ValueError("Invalid Amihud lookback, observation minimum, or quartile fraction")
    execution = config["execution"]
    if not 0 < execution["cash_buffer_fraction"] < 1 or execution["distribution_delay_days"] < 0:
        raise ValueError("Invalid cash buffer or distribution delay")
    costs = {}
    for scenario in config["cost_scenarios"]:
        fee, slippage = scenario["fee_bps"], scenario["slippage_bps"]
        total = fee + slippage
        if not 0 <= fee <= total < 10_000:
            raise ValueError("Fees and slippage must be nonnegative and total less than 10,000 bps")
        if total in costs:
            raise ValueError("Each cost scenario must have a distinct total")
        costs[total] = scenario
    if not costs:
        raise ValueError("Configure at least one cost scenario")
    base_cost = next(iter(costs))

    sessions = market_sessions(data_start, oos_end)
    panel, fetched_at = load_prices(tickers, cash_proxy, data_start, oos_end, sessions)
    open_, close, adj_close, volume = (
        panel[field].reindex(columns=tickers).where(lambda values: np.isfinite(values) & (values > 0))
        for field in FIELDS[:4]
    )
    valid_sessions = (open_.notna() & close.notna() & adj_close.notna() & volume.notna()).any(axis=1)
    if not valid_sessions.all():
        raise RuntimeError(f"Missing equity bars on NYSE sessions: {sessions[~valid_sessions].date}")
    cash_close = panel["Adj Close"][cash_proxy]
    if not (np.isfinite(cash_close) & (cash_close > 0)).all():
        raise RuntimeError(f"{cash_proxy} lacks a valid adjusted close on a required session")
    cash_returns = cash_close.pct_change(fill_method=None)

    daily_impact = adj_close.pct_change(fill_method=None).abs() * 1_000_000 / (close * volume)
    scores = daily_impact.replace([np.inf, -np.inf], np.nan).rolling(lookback, min_periods=minimum).mean()
    targets = portfolio_targets(scores, close.where(adj_close.notna()), fraction)
    dividends = panel["Dividends"].reindex(columns=tickers)
    results = {}
    for rate, scenario in costs.items():
        results[rate] = {}
        for name in PORTFOLIOS:
            daily = simulate(
                open_, close, dividends, targets[name],
                is_start, oos_end, scenario, execution,
            )
            periods = {"is": daily.loc[is_start:is_end], "oos": daily.loc[oos_start:oos_end], "full": daily}
            results[rate][name] = {
                period: metrics(sample, cash_returns, config["annualization_sessions"])
                for period, sample in periods.items()
            }

    full_dates = sessions[(sessions >= is_start) & (sessions <= oos_end)]
    data_range = f"{full_dates[0].date()} to {full_dates[-1].date()}"
    high, low, benchmark = PORTFOLIOS
    base = results[base_cost]
    output = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **base[high],
        "benchmark": {"name": benchmark, **base[benchmark]},
        "low_illiquidity": {"name": low, **base[low]},
        "cash_proxy": cash_proxy,
        "transaction_cost_bps": base_cost,
        **costs[base_cost],
        "cost_sensitivity": {str(rate): results[rate] for rate in costs if rate != base_cost},
    }
    print(f"{config['strategy']} | Daily returns {data_range}")
    print(f"IS {is_start.date()} to {is_end.date()} | OOS {oos_start.date()} to {oos_end.date()}")
    print(f"PARAM_COMBOS_TRIED: {len(costs)}")
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"CASH_PROXY: {cash_proxy} adjusted-close daily returns")
    print_metrics(results)
    print("RESULTS_JSON: " + json.dumps(output, allow_nan=False))


if __name__ == "__main__":
    main()
