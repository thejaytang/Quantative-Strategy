"""338_optimal_portfolio_using_factor
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/338_optimal_portfolio_using_factor.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 10f82382008748cd9ff2f56e2b8ef160caa66e38 on 2026-11-27 11:24 Europe/Oslo, exit 0, output sha256 c7e271c48b93237fc2cdfaabcc1a18ab9bb6940fb2ca31315fc8edf90ba01302
Data source: Yahoo Finance 免费公开历史行情接口的日频 ETF 开盘价和收盘价。配置资产为 XLB、XLE、XLF、XLI、XLK、XLP、XLU、XLV、XLY，SPY 仅用作市场因子收益。优先读取同一数据源的 CSV 缓存。（strategy.py:25-45, strategy.py:51-58, strategy.py:132-138, strategy.py:218-219, strategy.py:242, config.yaml:3-4, README.md:23）
Data fetched: 2026-11-27T11:24:24+01:00
Sample period: 请求行情区间为 2006-01-01 至 2025-12-31，含边界；2008 年之前的数据用于滚动估计预热。日志所报实际回测区间为 2008-01-02 至 2025-12-31。（config.yaml:5-12, strategy.py:25-45, strategy.py:236-250, strategy.py:252-254, README.md:19, logs/run_20261125_151042.log:11）   data_range (from output): 2008-01-02 to 2025-12-31
IS/OOS split: IS 为 2008-01-01 至 2018-12-31，OOS 为 2019-01-01 至 2025-12-31，两段均含日期边界。连续模拟后按日期切分统计，进入 OOS 时延续持仓。OOS 沿用固定参数，每月用截至信号日的 252 个日收益更新协方差。（config.yaml:7-16, strategy.py:121-139, strategy.py:223-225, strategy.py:236-248, README.md:17-19）
Costs and slippage: 每笔买入和卖出均按成交额收取 10 bps，首次建仓也计费。调仓开盘先让原持仓经历隔夜价格变动，再按买卖成交额绝对值之和扣费，并以扣费后的净值确定目标持仓金额。未建模滑点、开盘价差、市场冲击或税负。（config.yaml:18, strategy.py:143-171, strategy.py:226, README.md:17, README.md:31）
Signal lag: 每月最后一个交易日 t 收盘后，根据截至该日调整收盘价的滚动收益生成权重，次月首个交易日开盘执行。原持仓承担隔夜收益，新持仓承担执行日开盘至收盘的收益。（strategy.py:121-139, strategy.py:143-171, README.md:5-7）
Price adjustment: Yahoo split-adjusted quote open/close for sizing, fills and marking; separate cash distributions; Yahoo dividend-and-split-adjusted close for signals and BIL.
Survivorship bias: 全期使用同一组九只存续行业 ETF，存在 ETF 选择及存续偏差。代码按固定代码清单要求全期价格对齐、有效，未构造逐时点可选 ETF 池。（config.yaml:3, strategy.py:51-67, strategy.py:217-219, strategy.py:236-238, README.md:31）
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.07822280186171748, ann_vol=0.1496919081420537, sharpe=0.5780374350912254, sharpe_excess=0.5503630163800561, max_drawdown=-0.43495788574009164, total_return=1.2877272420505261, turnover=0.9873920175625962, n_trades=701
    oos: cagr=0.11938202342023807, ann_vol=0.1598357581574513, sharpe=0.7858836550282611, sharpe_excess=0.6275462330219864, max_drawdown=-0.30381759259543095, total_return=1.1982185231953606, turnover=1.083964949568538, n_trades=517
    full: cagr=0.09403463821519598, ann_vol=0.15370063003691237, sharpe=0.6617808751579951, sharpe_excess=0.5812683928970208, max_drawdown=-0.43495788574009164, total_return=4.028924399494145, turnover=1.0249209114311009, n_trades=1218
    benchmark: cagr=0.09753242275559804, ann_vol=0.18861861539738306, sharpe=0.5880639007728449, sharpe_excess=0.5226818940906788, max_drawdown=-0.5127092738135058, total_return=4.325860100895052, turnover=0.3439729759003887, n_trades=1944
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 338_optimal_portfolio_using_factor\n\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nmarket_factor: SPY\ndownload_start: 2006-01-01\n\nin_sample:\n  start: 2008-01-01\n  end: 2018-12-31\nout_of_sample:\n  start: 2019-01-01\n  end: 2025-12-31\n\nlookback_sessions: 252\ngraphical_lasso_alpha: 0.08\nmax_weight: 0.25\nrebalance: monthly\ntransaction_cost_bps: 10  # Per dollar bought or sold, including initial purchases.\nslippage_bps: 5           # Adverse price move on each opening fill.\ncash_buffer: 0.01        # Fraction of prior-close investable value withheld when sizing.\ndistribution_delay_days: 30  # Calendar days after ex-date before cash is spendable.\nsessions_per_year: 252\n'
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
"""Monthly sector allocation using a market factor and sparse residual precision."""

import json
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)
from scipy.optimize import minimize
from sklearn.covariance import graphical_lasso
from sklearn.exceptions import ConvergenceWarning


ROOT = Path(__file__).resolve().parent
FACTOR_GMV = "Factor graphical-lasso GMV"
SAMPLE_GMV = "Sample covariance GMV"
EQUAL_WEIGHT = "Equal weight"
CASH_PROXY = "BIL"
PARAM_COMBOS_TRIED = 1
PRICE_FIELDS = {"Open": "open", "Close": "close", "Adj Close": "adjusted_close"}
NYSE_SPECIAL_CLOSURES = pd.to_datetime([
    "2007-01-02", "2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09",
])


class NyseHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def nyse_sessions(start, end):
    if start.year < 2006 or end.year > 2025:
        raise ValueError("The historical NYSE session calendar covers 2006 through 2025.")
    holidays = NyseHolidays().holidays(start, end).union(NYSE_SPECIAL_CLOSURES)
    return pd.bdate_range(start, end).difference(holidays)


def download_prices(symbol, sessions):
    for _ in range(2):
        try:
            history = yf.Ticker(symbol).history(
                start=sessions[0].date().isoformat(),
                end=(sessions[-1] + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d", auto_adjust=False, actions=True, repair=False, timeout=15,
            )
            if not history.empty:
                break
            last_error = ValueError("Yahoo Finance returned no rows.")
        except Exception as error:
            last_error = error
    else:
        raise RuntimeError(f"Yahoo Finance download failed for {symbol}") from last_error

    index = history.index.tz_convert("America/New_York").tz_localize(None).normalize()
    if not index.equals(sessions):
        missing = sessions.difference(index).strftime("%Y-%m-%d").tolist()
        unexpected = index.difference(sessions).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{symbol}: unexpected session sequence; missing={missing}, extra={unexpected}")

    prices = history[list(PRICE_FIELDS)].rename(columns=PRICE_FIELDS)
    prices.index = index
    values = prices.to_numpy(dtype=float)
    distributions = history["Dividends"].to_numpy() + history["Capital Gains"].to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{symbol}: expected positive finite prices on all sessions.")
    if not np.isfinite(distributions).all() or (distributions < 0).any():
        raise ValueError(f"{symbol}: expected finite, nonnegative cash distributions.")
    prices["distribution"] = distributions
    return prices


def load_prices(symbols, sessions, cash_sessions):
    with TemporaryDirectory(prefix=".yahoo-", dir=ROOT) as temp_dir:
        yf.set_tz_cache_location(temp_dir)
        with ThreadPoolExecutor(max_workers=4) as pool:
            downloads = [pool.submit(download_prices, symbol, sessions) for symbol in symbols]
            cash_download = pool.submit(download_prices, CASH_PROXY, cash_sessions)
            history = [download.result() for download in downloads]
            cash_returns = cash_download.result()["adjusted_close"].pct_change(fill_method=None)
    prices = pd.concat(history, axis=1, keys=symbols).swaplevel(axis=1)
    return prices, cash_returns


def factor_covariance(returns, market, alpha):
    assets = returns - returns.mean(axis=0)
    factor = market - market.mean()
    factor_ss = factor @ factor
    if factor_ss <= 0:
        raise ValueError("The market factor must have positive variance.")
    beta = (factor @ assets) / factor_ss
    residuals = assets - np.outer(factor, beta)
    scales = residuals.std(axis=0, ddof=1)
    if (scales <= 1e-12).any():
        raise ValueError("Each portfolio ETF needs nonzero residual variance.")
    standardized = residuals / scales
    correlation = standardized.T @ standardized / (len(market) - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        residual_cov, _ = graphical_lasso(correlation, alpha=alpha, max_iter=500, tol=1e-7, enet_tol=1e-8)
    return residual_cov * np.outer(scales, scales) + np.outer(beta, beta) * factor.var(ddof=1)


def minimum_variance(covariance, max_weight):
    covariance = covariance / np.mean(np.diag(covariance))
    n_assets = len(covariance)
    equal = np.full(n_assets, 1 / n_assets)
    result = minimize(
        fun=lambda w: 0.5 * w @ covariance @ w,
        x0=equal,
        jac=lambda w: covariance @ w,
        bounds=[(0, max_weight)] * n_assets,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 300},
    )
    weights = result.x
    if (not result.success or not np.isfinite(weights).all() or abs(weights.sum() - 1) > 1e-9
            or (weights < -1e-9).any() or (weights > max_weight + 1e-9).any()):
        raise RuntimeError(f"Minimum-variance optimization failed: {result.message}")
    return np.where(weights > 1e-12, weights, 0.0)


def monthly_targets(closes, universe, market_factor, simulation_start, lookback, alpha, max_weight):
    daily_returns = closes.pct_change(fill_method=None)
    sessions = daily_returns.index
    months = sessions.to_period("M")
    targets = {name: {} for name in (FACTOR_GMV, SAMPLE_GMV, EQUAL_WEIGHT)}
    for position in np.flatnonzero(months[:-1] != months[1:]):
        execution = sessions[position + 1]
        if execution < simulation_start:
            continue
        if position < lookback:
            raise ValueError(f"Need {lookback} trailing daily returns by {sessions[position].date()}.")
        window = daily_returns.iloc[position - lookback + 1:position + 1]
        assets = window[universe].to_numpy()
        market = window[market_factor].to_numpy()
        factor_cov = factor_covariance(assets, market, alpha)
        sample_cov = np.cov(assets, rowvar=False, ddof=1)
        targets[FACTOR_GMV][execution] = minimum_variance(factor_cov, max_weight)
        targets[SAMPLE_GMV][execution] = minimum_variance(sample_cov, max_weight)
        targets[EQUAL_WEIGHT][execution] = np.full(len(universe), 1 / len(universe))
    return targets


def fill_rebalance(positions, proposed, open_prices, cash, cost_rate, slippage_rate):
    changes = proposed - positions
    sells = np.maximum(-changes, 0.0)
    buys = np.maximum(changes, 0.0)
    sale_value = sells @ (open_prices * (1 - slippage_rate))
    buy_value = buys @ (open_prices * (1 + slippage_rate))
    cash_after_sales = cash + sale_value * (1 - cost_rate)
    buy_cost = buy_value * (1 + cost_rate)
    if buy_cost > cash_after_sales:
        scale = np.nextafter(cash_after_sales / buy_cost, 0.0)
        buys *= scale
        buy_value *= scale
        buy_cost *= scale
    cash = cash_after_sales - buy_cost
    updated = positions - sells + buys
    n_trades = np.count_nonzero(updated != positions)
    return updated, cash, sale_value + buy_value, n_trades


def simulate(opens, closes, distributions, targets, simulation_start, cost_rate,
             slippage_rate, cash_buffer, distribution_delay):
    sessions = opens.loc[simulation_start:].index
    if sessions.empty or sessions[0] not in targets or sessions[0] == closes.index[0]:
        raise ValueError("Need a prior month-end signal for the first simulated session.")
    positions = np.zeros(len(opens.columns))
    cash = previous_value = investable_value = 1.0
    last_close = closes.loc[:sessions[0]].iloc[-2].to_numpy()
    pending_distributions = deque()
    records = []
    for session in sessions:
        distribution = positions @ distributions.loc[session].to_numpy()
        if distribution > 0:
            pending_distributions.append((session + distribution_delay, distribution))
        while pending_distributions and pending_distributions[0][0] <= session:
            cash += pending_distributions.popleft()[1]
        receivable = sum(amount for _, amount in pending_distributions)

        turnover = 0.0
        n_trades = 0
        target = targets.get(session)
        if target is not None:
            proposed = target * (investable_value * (1 - cash_buffer)) / last_close
            open_prices = opens.loc[session].to_numpy()
            open_value = cash + receivable + positions @ open_prices
            positions, cash, traded_value, n_trades = fill_rebalance(
                positions, proposed, open_prices, cash, cost_rate, slippage_rate,
            )
            turnover = traded_value / open_value
            if cash < 0 or (positions < 0).any():
                raise RuntimeError(f"Negative cash or position on {session.date()}.")

        last_close = closes.loc[session].to_numpy()
        investable_value = cash + positions @ last_close
        closing_value = investable_value + receivable
        records.append((closing_value / previous_value - 1, turnover, n_trades, cash))
        previous_value = closing_value
    return pd.DataFrame(records, index=sessions, columns=["return", "turnover", "n_trades", "cash"])


def metrics(results, cash_returns, sessions_per_year):
    returns = results["return"]
    excess = returns - cash_returns.loc[returns.index]
    growth = (1 + returns).cumprod()
    monthly = (1 + returns).groupby(returns.index.to_period("M")).prod() - 1
    daily_vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    years = len(returns) / sessions_per_year
    values = {
        "cagr": growth.iloc[-1] ** (1 / years) - 1,
        "ann_vol": daily_vol * np.sqrt(sessions_per_year),
        "sharpe": returns.mean() / daily_vol * np.sqrt(sessions_per_year) if daily_vol > 0 else np.nan,
        "sharpe_excess": excess.mean() / excess_vol * np.sqrt(sessions_per_year) if excess_vol > 0 else np.nan,
        "max_drawdown": (growth / growth.clip(lower=1).cummax() - 1).min(),
        "total_return": growth.iloc[-1] - 1,
        "turnover": results["turnover"].sum() / years,
        "mean_monthly_return": monthly.mean(),
        "monthly_variance": monthly.var(ddof=1),
    }
    stats = {key: float(value) if np.isfinite(value) else None for key, value in values.items()}
    stats["n_trades"] = int(results["n_trades"].sum())
    return stats


def print_table(summary):
    columns = {
        "cagr": "CAGR",
        "ann_vol": "Ann. vol",
        "sharpe": "Sharpe",
        "sharpe_excess": "Sharpe vs BIL",
        "max_drawdown": "Max DD",
        "total_return": "Total return",
        "turnover": "Ann. turnover",
        "n_trades": "Trades",
        "mean_monthly_return": "Mean mo. ret",
        "monthly_variance": "Mo. variance",
    }
    formats = {"sharpe": ".3f", "sharpe_excess": ".3f", "n_trades": "d", "monthly_variance": ".3e"}
    print(f"{'Portfolio':<28} {'Period':<6}" + "".join(f" {label:>14}" for label in columns.values()))
    for portfolio, periods in summary.items():
        for period, stats in periods.items():
            row = []
            for key in columns:
                form = formats.get(key, ".2%")
                row.append(format(stats[key], form) if stats[key] is not None else "n/a")
            print(f"{portfolio:<28} {period.upper():<6}" + "".join(f" {value:>14}" for value in row))


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    universe, market_factor = config["universe"], config["market_factor"]
    symbols = [*universe, market_factor]
    download_start = pd.Timestamp(config["download_start"])
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    lookback = int(config["lookback_sessions"])
    alpha = float(config["graphical_lasso_alpha"])
    max_weight = float(config["max_weight"])
    cost_rate = float(config["transaction_cost_bps"]) / 10_000
    slippage_rate = float(config["slippage_bps"]) / 10_000
    cash_buffer = float(config["cash_buffer"])
    distribution_delay = pd.Timedelta(days=int(config["distribution_delay_days"]))
    sessions_per_year = int(config["sessions_per_year"])
    if not download_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Expected chronological price history, IS and OOS dates.")
    if len(universe) < 2 or len(set([*symbols, CASH_PROXY])) != len(symbols) + 1:
        raise ValueError("Use distinct sector ETFs and a separate market factor.")
    if not (lookback >= 3 and 0 < alpha and 1 / len(universe) <= max_weight <= 1
            and 0 <= cost_rate < 1 and 0 <= slippage_rate < 1 and 0 < cash_buffer < 1
            and distribution_delay.days > 0 and sessions_per_year > 0 and config["rebalance"] == "monthly"):
        raise ValueError("Invalid model, allocation, execution or annualization settings.")

    sessions = nyse_sessions(download_start, oos_end)
    first_scored = sessions.searchsorted(is_start)
    if not 0 < first_scored < len(sessions):
        raise ValueError("The scoring period needs a session preceded by a price observation.")
    cash_sessions = sessions[first_scored - 1:]
    prices, cash_returns = load_prices(symbols, sessions, cash_sessions)
    fetched_at = datetime.now(ZoneInfo("Europe/Oslo")).isoformat(timespec="seconds")
    simulation_start = is_start.replace(day=1)
    targets = monthly_targets(
        prices["adjusted_close"], universe, market_factor, simulation_start, lookback, alpha, max_weight,
    )
    ranges = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    summary = {}
    for portfolio, schedule in targets.items():
        results = simulate(
            prices["open"][universe], prices["close"][universe], prices["distribution"][universe],
            schedule, simulation_start, cost_rate, slippage_rate, cash_buffer, distribution_delay,
        )
        summary[portfolio] = {}
        for period, (start, end) in ranges.items():
            segment = results.loc[start:end]
            if segment.empty:
                raise ValueError(f"No trading sessions in the {period} date range.")
            summary[portfolio][period] = metrics(segment, cash_returns, sessions_per_year)

    scored = results.loc[is_start:oos_end].index
    benchmark = summary[EQUAL_WEIGHT]
    report = {
        "strategy": config["strategy"],
        "data_range": f"{scored[0].date()} to {scored[-1].date()}",
        **summary[FACTOR_GMV],
        "benchmark": {"name": EQUAL_WEIGHT, **benchmark["full"], **benchmark},
    }
    print(f"PARAM_COMBOS_TRIED: {PARAM_COMBOS_TRIED}")
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: Yahoo split-adjusted quote open/close for sizing, fills and marking; "
          "separate cash distributions; Yahoo dividend-and-split-adjusted close for signals and BIL.")
    print_table(summary)
    print("RESULTS_JSON: " + json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
