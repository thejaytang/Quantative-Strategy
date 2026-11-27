"""113_kalman_garch_pairs
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/113_kalman_garch_pairs.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c71364ad99535bbf59be998b004cff431ee83048 on 2026-11-27 11:53 Europe/Oslo, exit 0, output sha256 ba3b6afdfc40a0972f1390b7a2dc63a1aead2869f8cc98ec14816135258f0d54
Data source: Yahoo Finance 公开 Chart API 的日频复权收盘价，经 requests 获取并优先读取本地 CSV 缓存。固定配对为 XOM/CVX、KO/PEP、LMT/NOC，基准为 SPY。（config.yaml:2-3；strategy.py:31-54,210-220；README.md:17）
Data fetched: 2026-11-27T11:53:59+01:00
Sample period: 请求区间为 2015-01-01 至 2025-12-31。日志中 OLS 初始化区间为 2015-01-02 至 2015-07-02，全期绩效统计区间为 2015-07-07 至 2025-12-31。（config.yaml:4-9；strategy.py:194-195,219-220；logs/run_20261125_100500.log:1-2）   data_range (from output): 2015-07-07 to 2025-12-31
IS/OOS split: 配置的 IS 为 2015-01-01 至 2019-12-31，OOS 为 2020-01-01 至 2025-12-31。日志中的实际统计区间分别为 2015-07-07 至 2019-12-31 和 2020-01-02 至 2025-12-31。IS 的前 126 个交易日用于 OLS 初始化并从绩效统计中排除。模型状态与持仓跨越分界连续更新，在同一模拟路径上按日期分段统计。（config.yaml:4-11；strategy.py:75-108,226-259；README.md:22；logs/run_20261125_100500.log:1-2）
Costs and slippage: 每买入或卖出 1 美元名义金额收取 2 bps。先按收盘价计算既有持仓的收益及漂移后权重，再将目标权重与漂移后权重之差的绝对值按各标的求和，以 0.0002 乘以该换手比例从调仓前净值扣费，包含对冲调整。SPY 在首个统计日收盘买入时同样计费。未建模滑点，以标记收盘价模拟成交。（config.yaml:18；strategy.py:136-155,240-250；README.md:21,26；logs/run_20261125_100500.log:3）
Signal lag: 第 t 个交易日收盘价用于计算当日信号和目标权重，目标在下一交易日 t+1 收盘执行。t 至 t+1 的收益归既有持仓，新仓位从 t+1 收盘至下一收盘的区间开始承担价格收益。（strategy.py:95-108,113-133,136-155,240-247；README.md:5）
Price adjustment: Yahoo Finance Adj Close (splits and distributions); adjusted share units
Survivorship bias: 固定配对来自事后可见且仍在交易的股票，README 明示存在幸存者偏差和配对选择偏差。配置固定三组配对，代码遍历这些标的。（config.yaml:2；strategy.py:210-220；README.md:11,26）
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.002276350440157948, ann_vol=0.019086715524809495, sharpe=-0.10986534743348635, sharpe_excess=-0.6220292341174757, max_drawdown=-0.036262563567896366, total_return=-0.01017599401489111, turnover=33.50623200600576, n_trades=1310
    oos: cagr=-0.017843220055616338, ann_vol=0.024107705382642855, sharpe=-0.7347368338551516, sharpe_excess=-1.803578001338001, max_drawdown=-0.11773382992752524, total_return=-0.10213916508844734, turnover=78.30570029671046, n_trades=2216
    full: cagr=-0.011201668030926482, ann_vol=0.02209744698347704, sharpe=-0.4987142532321675, sharpe_excess=-1.352436579865977, max_drawdown=-0.12637887791161795, total_return=-0.1112757915707121, turnover=111.8119323027162, n_trades=3526
    benchmark  full: cagr=0.13877536009436953, ann_vol=0.17983871832468032, sharpe=0.8129404690330289, sharpe_excess=0.7059976465029972, max_drawdown=-0.3365316062948239, total_return=2.8997065104211583, turnover=0.9964257029373915, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 113_kalman_garch_pairs\npairs: [[XOM, CVX], [KO, PEP], [LMT, NOC]]  # Each pair is [dependent, predictor].\nbenchmark: SPY\ncash_proxy: BIL\nin_sample:\n  start: 2015-01-01\n  end: 2019-12-31\nout_of_sample:\n  start: 2020-01-01\n  end: 2025-12-31\nparameters:\n  initial_window: 126\n  process_variance_ratio: 0.0001\n  ewma_decay: 0.94\n  entry_z: 2.0\n  exit_z: 0.25\n  max_holding_sessions: 15\n  gross_exposure: 1.0\ncash_buffer: 0.15  # Reserve 15% of prior-close value for price moves and execution costs.\nbenchmark_cash_buffer: 0.01  # Reserve 1% for the single SPY entry.\ntransaction_cost_bps: 2.0  # Commission per dollar of executed notional, on each leg.\nslippage_bps: 2.0  # Buy above the reference close and sell below it by this amount.\nsessions_per_year: 252\n'
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
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent


def validate_prices(prices, start, end):
    prices = prices.sort_index().loc[start:end]
    if (
        prices.empty
        or prices.index.has_duplicates
        or not np.isfinite(prices.to_numpy()).all()
        or (prices <= 0).any()
        or (prices.index[0] - start).days > 7
        or (end - prices.index[-1]).days > 7
    ):
        raise ValueError(f"Missing, invalid or incomplete adjusted closes for {prices.name}")
    return prices


def load_prices(ticker, start, end):
    exclusive_end = end + pd.Timedelta(days=1)
    history = yf.Ticker(ticker).history(
        start=f"{start:%Y-%m-%d}", end=f"{exclusive_end:%Y-%m-%d}",
        interval="1d", auto_adjust=False, actions=False, timeout=10,
    )
    if history.empty or "Adj Close" not in history:
        raise ValueError(f"Yahoo Finance returned no adjusted closes for {ticker}")
    prices = history["Adj Close"].rename(ticker)
    prices.index = prices.index.tz_localize(None).normalize()
    return validate_prices(prices, start, end)


def kalman_signals(log_prices, params):
    """Return filtered hedge ratios and two scores of the same pre-update innovation."""
    y, x = log_prices.T
    window = params["initial_window"]
    center = x[:window].mean()
    scale = x[:window].std()
    if scale <= 0:
        raise ValueError("Predictor has zero variance in the initialization window")
    design = np.column_stack((np.ones(len(x)), (x - center) / scale))
    initial_design = design[:window]
    state = np.linalg.lstsq(initial_design, y[:window], rcond=None)[0]
    residuals = y[:window] - initial_design @ state
    observation_var = residuals @ residuals / (window - initial_design.shape[1])
    if observation_var <= 0:
        raise ValueError("Initial OLS residual variance must be positive")

    covariance = observation_var * np.linalg.inv(initial_design.T @ initial_design)
    process_cov = np.eye(2) * (params["process_variance_ratio"] * observation_var)
    decay = params["ewma_decay"]
    ewma_var = observation_var
    fixed_std = math.sqrt(observation_var)
    hedge, ewma_z, fixed_z = np.full((3, len(x)), np.nan)

    for day in range(window, len(x)):
        row = design[day]
        innovation = y[day] - row @ state
        ewma_z[day] = innovation / math.sqrt(ewma_var)
        fixed_z[day] = innovation / fixed_std

        covariance += process_cov
        projected = covariance @ row
        innovation_var = row @ projected + observation_var
        gain = projected / innovation_var
        state += gain * innovation
        covariance -= np.outer(projected, projected) / innovation_var
        hedge[day] = state[1] / scale
        ewma_var = decay * ewma_var + (1 - decay) * innovation**2

    return hedge, ewma_z, fixed_z


def pair_targets(hedge, scores, params, sleeve):
    """Compute close-time decisions before shifting them to execution dates."""
    targets = np.zeros((len(scores), 2))
    direction = 0
    entry_day = 0
    for day in range(params["initial_window"], len(scores)):
        beta, score = hedge[day], scores[day]
        if not (np.isfinite(beta) and beta > 0 and np.isfinite(score)):
            direction = 0
            continue
        if direction:
            age = day - entry_day
            if abs(score) <= params["exit_z"] or age >= params["max_holding_sessions"]:
                direction = 0
        elif abs(score) > params["entry_z"]:
            direction = -1 if score > 0 else 1
            entry_day = day

        dependent_weight = direction * sleeve / (1 + beta)
        targets[day] = dependent_weight, -beta * dependent_weight
    return targets


def fill_at_close(changes, close, cash, cost_rate, slippage_rate):
    """Price share changes fixed at the previous close."""
    execution_prices = close * (1 + np.sign(changes) * slippage_rate)
    signed_notional = changes * execution_prices
    notional = np.abs(signed_notional).sum()
    return cash - signed_notional.sum() - cost_rate * notional, notional


def simulate(closes, targets, cost_rate, slippage_rate, cash_buffer):
    """Size at the signal close and fill at the following close."""
    positions = np.zeros(closes.shape[1])
    cash = previous_value = 1.0
    returns = np.zeros(len(targets))
    turnover = np.zeros(len(targets))
    trades = np.zeros(len(targets), dtype=int)
    for day, (signal_close, fill_close, target) in enumerate(
        zip(closes[:-1], closes[1:], targets, strict=True)
    ):
        if target is not None:
            planned_positions = target * (previous_value * (1 - cash_buffer)) / signal_close
            changes = planned_positions - positions
        value_before_trade = cash + positions @ fill_close
        if value_before_trade <= 0:
            raise ValueError("Portfolio equity exhausted before rebalancing")
        if target is not None:
            cash, notional = fill_at_close(changes, fill_close, cash, cost_rate, slippage_rate)
            positions = planned_positions
            trades[day] = np.count_nonzero(changes)
            turnover[day] = notional / value_before_trade
        # Cash covering twice the short value implies equity covers gross exposure.
        required_cash = 2 * (np.maximum(-positions, 0) @ fill_close)
        if cash < required_cash:
            raise ValueError(
                f"Cash or short collateral insufficient at scored session {day + 1}: "
                f"cash={cash:.8f}, required={required_cash:.8f}"
            )
        closing_value = cash + positions @ fill_close
        returns[day] = closing_value / previous_value - 1
        previous_value = closing_value
    return returns, turnover, trades


def metrics(returns, cash_returns, turnover, trades, sessions_per_year):
    cumulative_value = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(np.r_[1.0, cumulative_value])[1:]
    scale = math.sqrt(sessions_per_year)

    def sharpe(daily_returns):
        std = daily_returns.std(ddof=1) if len(daily_returns) > 1 else np.nan
        return daily_returns.mean() / std * scale if std > 0 else np.nan

    values = {
        "cagr": cumulative_value[-1] ** (sessions_per_year / len(returns)) - 1,
        "ann_vol": returns.std(ddof=1) * scale if len(returns) > 1 else np.nan,
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(returns - cash_returns),
        "max_drawdown": (cumulative_value / peaks - 1).min(),
        "total_return": cumulative_value[-1] - 1,
        "turnover": turnover.sum(),
    }
    result = {name: float(value) if np.isfinite(value) else None for name, value in values.items()}
    return {**result, "n_trades": int(trades.sum())}


def print_report(ranges, portfolios, cost_bps, slippage_bps, cash_proxy, calibration_range):
    print(f"OLS initialization (excluded from performance): {calibration_range}")
    print("Performance dates: " + " | ".join(f"{name.upper()} {dates}" for name, dates in ranges.items()))
    print(f"Commission: {cost_bps:g} bps of fill notional. Adverse slippage: {slippage_bps:g} bps per fill.")
    print(f"Sharpe uses a zero cash rate. Excess Sharpe uses {cash_proxy} adjusted close-to-close returns.")
    print(f"{'Portfolio':<24} {'Period':<6} {'CAGR':>9} {'Ann. vol':>9} {'Sharpe':>8} "
          f"{'Ex. Sharpe':>10} {'Max DD':>9} {'Total':>9} {'Turnover':>10} {'Trades':>8}")
    formats = (".2%", ".2%", ".2f", ".2f", ".2%", ".2%", ".2f", "d")
    widths = (9, 9, 8, 10, 9, 9, 10, 8)
    for label, periods in portfolios.items():
        for period, values in periods.items():
            display = [format(value, fmt) if value is not None else "n/a"
                       for value, fmt in zip(values.values(), formats, strict=True)]
            line = " ".join(value.rjust(width) for value, width in zip(display, widths, strict=True))
            print(f"{label:<24} {period.upper():<6} {line}")


def main():
    with (ROOT / "config.yaml").open() as source:
        config = yaml.safe_load(source)
    train_start, train_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    test_start, test_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not train_start <= train_end < test_start <= test_end or (test_start - train_end).days != 1:
        raise ValueError("IS and OOS must be consecutive, ordered calendar ranges")
    params = config["parameters"]
    window = params["initial_window"]
    cost_bps = config["transaction_cost_bps"]
    slippage_bps = config["slippage_bps"]
    cash_buffer = config["cash_buffer"]
    benchmark_cash_buffer = config["benchmark_cash_buffer"]
    if (
        window < 3 or not 0 < params["ewma_decay"] < 1
        or not 0 <= params["exit_z"] < params["entry_z"]
        or params["max_holding_sessions"] < 1 or params["process_variance_ratio"] < 0
        or not 0 < params["gross_exposure"] <= 1 or not 0 <= cost_bps < 10_000
        or not 0 <= slippage_bps < 10_000
        or not (0 < cash_buffer < 1 and 0 < benchmark_cash_buffer < 1)
        or config["sessions_per_year"] <= 0
    ):
        raise ValueError("Invalid strategy parameters or execution assumptions")

    pairs = config["pairs"]
    symbols = [ticker for pair in pairs for ticker in pair]
    if not pairs or any(len(pair) != 2 for pair in pairs) or len(set(symbols)) != len(symbols):
        raise ValueError("Configure at least one pair with two distinct, non-overlapping tickers")
    benchmark = config["benchmark"]
    cash_proxy = config["cash_proxy"]
    with TemporaryDirectory(dir=ROOT) as cache:
        yf.set_tz_cache_location(cache)
        series = {ticker: load_prices(ticker, train_start, test_end)
                  for ticker in dict.fromkeys([*symbols, benchmark, cash_proxy])}
    fetched_at = datetime.now(ZoneInfo("Europe/Oslo")).isoformat(timespec="seconds")

    index = series[benchmark].index
    for ticker, ticker_prices in series.items():
        if not ticker_prices.index.equals(index):
            raise ValueError(f"Trading dates for {ticker} and {benchmark} do not match")
    first_execution = window + 1
    if (index <= train_end).sum() <= first_execution or not (index >= test_start).any():
        raise ValueError("Need initialization and subsequent trading in IS, and observations in OOS")
    prices = pd.DataFrame(series)
    log_prices = np.log(prices[symbols].to_numpy())
    ewma_targets = np.zeros_like(log_prices)
    fixed_targets = np.zeros_like(log_prices)
    sleeve = params["gross_exposure"] / len(pairs)
    for leg in range(0, len(symbols), 2):
        columns = slice(leg, leg + 2)
        hedge, ewma_z, fixed_z = kalman_signals(log_prices[:, columns], params)
        ewma_targets[:, columns] = pair_targets(hedge, ewma_z, params, sleeve)
        fixed_targets[:, columns] = pair_targets(hedge, fixed_z, params, sleeve)

    # The first executable close is one session after the first post-OLS signal.
    scored_dates = index[first_execution:]
    execution_costs = {"cost_rate": cost_bps / 10_000, "slippage_rate": slippage_bps / 10_000}
    score_prices = prices.iloc[window:]
    stock_closes = score_prices[symbols].to_numpy()
    cash_returns = score_prices[cash_proxy].pct_change(fill_method=None).iloc[1:].to_numpy()
    benchmark_name = f"{benchmark} buy and hold"
    daily_results = {
        "Kalman + EWMA": simulate(
            stock_closes, ewma_targets[window:-1], cash_buffer=cash_buffer, **execution_costs
        ),
        "Kalman + fixed variance": simulate(
            stock_closes, fixed_targets[window:-1], cash_buffer=cash_buffer, **execution_costs
        ),
        benchmark_name: simulate(
            score_prices[[benchmark]].to_numpy(), [np.ones(1)] + [None] * (len(scored_dates) - 1),
            cash_buffer=benchmark_cash_buffer, **execution_costs,
        ),
    }
    masks = {"is": scored_dates <= train_end, "oos": scored_dates >= test_start,
             "full": np.ones(len(scored_dates), dtype=bool)}
    ranges = {period: f"{scored_dates[mask][0]:%Y-%m-%d} to {scored_dates[mask][-1]:%Y-%m-%d}"
              for period, mask in masks.items()}
    portfolios = {
        name: {period: metrics(returns[mask], cash_returns[mask], turnover[mask], trades[mask],
                               config["sessions_per_year"])
               for period, mask in masks.items()}
        for name, (returns, turnover, trades) in daily_results.items()
    }
    calibration_range = f"{index[0]:%Y-%m-%d} to {index[window - 1]:%Y-%m-%d}"
    print("PARAM_COMBOS_TRIED: 1")
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: Yahoo Finance Adj Close (splits and distributions); adjusted share units")
    print_report(ranges, portfolios, cost_bps, slippage_bps, cash_proxy, calibration_range)
    results = {
        "strategy": config["strategy"],
        "data_range": ranges["full"],
        **portfolios["Kalman + EWMA"],
        "benchmark": {"name": benchmark_name, **portfolios[benchmark_name]},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
