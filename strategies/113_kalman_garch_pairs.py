"""113_kalman_garch_pairs
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/113_kalman_garch_pairs.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit ad594150a0142009ae8f6ffab0995de08b2f8304 on 2026-11-26 13:49 Europe/Oslo, exit 0, output sha256 e8eaa2da358586a14efb23af6aa43e9378f3d06d90a969018f9d9982547394c9
Data source: Yahoo Finance 公开 Chart API 的日频复权收盘价，经 requests 获取并优先读取本地 CSV 缓存。固定配对为 XOM/CVX、KO/PEP、LMT/NOC，基准为 SPY。（config.yaml:2-3；strategy.py:31-54,210-220；README.md:17）
Data fetched: 2026-11-26 13:49 Europe/Oslo (run time)
Sample period: 请求区间为 2015-01-01 至 2025-12-31。日志中 OLS 初始化区间为 2015-01-02 至 2015-07-02，全期绩效统计区间为 2015-07-07 至 2025-12-31。（config.yaml:4-9；strategy.py:194-195,219-220；logs/run_20261125_100500.log:1-2）   data_range (from output): 2015-07-07 to 2025-12-31
IS/OOS split: 配置的 IS 为 2015-01-01 至 2019-12-31，OOS 为 2020-01-01 至 2025-12-31。日志中的实际统计区间分别为 2015-07-07 至 2019-12-31 和 2020-01-02 至 2025-12-31。IS 的前 126 个交易日用于 OLS 初始化并从绩效统计中排除。模型状态与持仓跨越分界连续更新，在同一模拟路径上按日期分段统计。（config.yaml:4-11；strategy.py:75-108,226-259；README.md:22；logs/run_20261125_100500.log:1-2）
Costs and slippage: 每买入或卖出 1 美元名义金额收取 2 bps。先按收盘价计算既有持仓的收益及漂移后权重，再将目标权重与漂移后权重之差的绝对值按各标的求和，以 0.0002 乘以该换手比例从调仓前净值扣费，包含对冲调整。SPY 在首个统计日收盘买入时同样计费。未建模滑点，以标记收盘价模拟成交。（config.yaml:18；strategy.py:136-155,240-250；README.md:21,26；logs/run_20261125_100500.log:3）
Signal lag: 第 t 个交易日收盘价用于计算当日信号和目标权重，目标在下一交易日 t+1 收盘执行。t 至 t+1 的收益归既有持仓，新仓位从 t+1 收盘至下一收盘的区间开始承担价格收益。（strategy.py:95-108,113-133,136-155,240-247；README.md:5）
Price adjustment: 使用 Yahoo Finance 的 adjclose 字段或缓存中的 adjusted_close 列。按交易所时区归属交易日，使用复权收盘价计算对数价信号与收盘到收盘收益。README 说明复权价格将股息和拆股计入收益。（strategy.py:31-36,49-54,229-242；README.md:26）
Survivorship bias: 固定配对来自事后可见且仍在交易的股票，README 明示存在幸存者偏差和配对选择偏差。配置固定三组配对，代码遍历这些标的。（config.yaml:2；strategy.py:210-220；README.md:11,26）
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.0008517215722365634, ann_vol=0.022512580416404086, sharpe=-0.02660539657033425, max_drawdown=-0.040609774891884665, total_return=-0.0038169332768013264, turnover=39.46340529825778
    oos: cagr=-0.019753971352626354, ann_vol=0.02866370713095161, sharpe=-0.6816769614834477, max_drawdown=-0.1322857295870491, total_return=-0.11254143108156456, turnover=91.93034382947565
    full: cagr=-0.011697193932705185, ann_vol=0.026206923161372273, sharpe=-0.4358413397896265, max_drawdown=-0.13845164107731966, total_return=-0.11592880122505067, turnover=131.39374912773343
    benchmark  full: cagr=0.13908683887802442, ann_vol=0.1802285335832455, sharpe=0.8130916128895719, max_drawdown=-0.3372105646036555, total_return=2.9108911977420693, turnover=1.0
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 113_kalman_garch_pairs\npairs: [[XOM, CVX], [KO, PEP], [LMT, NOC]]  # Each pair is [dependent, predictor].\nbenchmark: SPY\nin_sample:\n  start: 2015-01-01\n  end: 2019-12-31\nout_of_sample:\n  start: 2020-01-01\n  end: 2025-12-31\nparameters:\n  initial_window: 126\n  process_variance_ratio: 0.0001\n  ewma_decay: 0.94\n  entry_z: 2.0\n  exit_z: 0.25\n  max_holding_sessions: 15\n  gross_exposure: 1.0\ntransaction_cost_bps: 2.0  # Per dollar bought or sold, including hedge rebalancing.\nsessions_per_year: 252\n'
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
from tempfile import NamedTemporaryFile
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import yaml


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"


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
        raise ValueError("Missing, invalid or incomplete adjusted closes")
    return prices


def load_prices(session, ticker, start, end):
    symbol = quote(ticker, safe="")
    cache_path = CACHE / f"yahoo_{symbol}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, index_col="date", parse_dates=["date"])
        return validate_prices(cached["adjusted_close"], start, end).rename(ticker)

    query = {
        "period1": int(start.tz_localize("UTC").timestamp()),
        "period2": int((end + pd.Timedelta(days=1)).tz_localize("UTC").timestamp()),
        "interval": "1d",
    }
    errors = []
    for host in ("query1", "query2"):
        try:
            url = f"https://{host}.finance.yahoo.com/v8/finance/chart/{symbol}"
            response = session.get(url, params=query, timeout=(5, 15))
            response.raise_for_status()
            chart = response.json()["chart"]["result"][0]
            dates = pd.to_datetime(chart["timestamp"], unit="s", utc=True)
            dates = dates.tz_convert(chart["meta"]["exchangeTimezoneName"]).normalize().tz_localize(None)
            dates.name = "date"
            prices = pd.Series(chart["indicators"]["adjclose"][0]["adjclose"], index=dates, dtype=float)
            prices = validate_prices(prices, start, end)
            break
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            errors.append(f"{host}: {exc}")
    else:
        raise RuntimeError(f"Yahoo Finance download failed for {ticker}: {' | '.join(errors)}")

    with NamedTemporaryFile(dir=CACHE, suffix=".tmp", delete=False, mode="w") as output:
        temp_path = Path(output.name)
        try:
            prices.to_csv(output, header=["adjusted_close"])
            output.close()
            temp_path.replace(cache_path)
        finally:
            temp_path.unlink(missing_ok=True)
    return prices.rename(ticker)


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


def simulate(asset_returns, targets, cost_rate):
    """Mark existing holdings to the close, then rebalance and charge costs."""
    weights = np.zeros(asset_returns.shape[1])
    returns = np.zeros(len(asset_returns))
    turnover = np.zeros(len(asset_returns))
    for day, (moves, target) in enumerate(zip(asset_returns, targets, strict=True)):
        value_before_trade = 1 + weights @ moves
        if value_before_trade <= 0:
            raise ValueError("Portfolio equity exhausted before rebalancing")
        drifted = weights * (1 + moves) / value_before_trade
        if target is None:  # Buy-and-hold benchmark after its initial purchase.
            returns[day] = value_before_trade - 1
            weights = drifted
            continue
        turnover[day] = np.abs(target - drifted).sum()
        after_cost = 1 - cost_rate * turnover[day]
        if after_cost <= 0:
            raise ValueError("Portfolio equity exhausted by transaction costs")
        returns[day] = value_before_trade * after_cost - 1
        weights = target / after_cost
    return returns, turnover


def metrics(returns, turnover, sessions_per_year):
    nav = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(np.r_[1.0, nav])[1:]
    std = returns.std(ddof=1) if len(returns) > 1 else np.nan
    scale = math.sqrt(sessions_per_year)
    values = {
        "cagr": nav[-1] ** (sessions_per_year / len(returns)) - 1,
        "ann_vol": std * scale,
        "sharpe": returns.mean() / std * scale if std > 0 else np.nan,
        "max_drawdown": (nav / peaks - 1).min(),
        "total_return": nav[-1] - 1,
        "turnover": turnover.sum(),
    }
    return {name: float(value) if np.isfinite(value) else None for name, value in values.items()}


def print_report(ranges, portfolios, cost_bps, calibration_range):
    print(f"OLS initialization (excluded from performance): {calibration_range}")
    print("Performance dates: " + " | ".join(f"{name.upper()} {dates}" for name, dates in ranges.items()))
    print(f"Net of {cost_bps:g} bps per dollar traded. Sharpe uses a zero risk-free rate.")
    print(f"{'Portfolio':<24} {'Period':<6} {'CAGR':>9} {'Ann. vol':>9} {'Sharpe':>8} "
          f"{'Max DD':>9} {'Total':>9} {'Turnover':>10}")
    formats = (".2%", ".2%", ".2f", ".2%", ".2%", ".2f")
    widths = (9, 9, 8, 9, 9, 10)
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
    if (
        window < 3 or not 0 < params["ewma_decay"] < 1
        or not 0 <= params["exit_z"] < params["entry_z"]
        or params["max_holding_sessions"] < 1 or params["process_variance_ratio"] < 0
        or not 0 < params["gross_exposure"] <= 1 or not 0 <= cost_bps < 10_000
        or config["sessions_per_year"] <= 0
    ):
        raise ValueError("Invalid strategy parameters or transaction costs")

    pairs = config["pairs"]
    symbols = [ticker for pair in pairs for ticker in pair]
    if not pairs or any(len(pair) != 2 for pair in pairs) or len(set(symbols)) != len(symbols):
        raise ValueError("Configure at least one pair with two distinct, non-overlapping tickers")
    benchmark = config["benchmark"]
    CACHE.mkdir(exist_ok=True)
    with requests.Session() as session:
        session.trust_env = False
        session.headers["User-Agent"] = "Mozilla/5.0"
        series = {ticker: load_prices(session, ticker, train_start, test_end)
                  for ticker in dict.fromkeys([*symbols, benchmark])}

    index = series[benchmark].index
    for ticker in symbols:
        if not series[ticker].index.equals(index):
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
    price_returns = prices.pct_change().iloc[first_execution:]
    cost_rate = cost_bps / 10_000
    stock_returns = price_returns[symbols].to_numpy()
    portfolio_returns = {
        "Kalman + EWMA": simulate(stock_returns, ewma_targets[window:-1], cost_rate),
        "Kalman + fixed variance": simulate(stock_returns, fixed_targets[window:-1], cost_rate),
        f"{benchmark} buy and hold": simulate(
            price_returns[[benchmark]].to_numpy(), [np.ones(1)] + [None] * (len(scored_dates) - 1), cost_rate
        ),
    }
    masks = {"is": scored_dates <= train_end, "oos": scored_dates >= test_start,
             "full": np.ones(len(scored_dates), dtype=bool)}
    ranges = {period: f"{scored_dates[mask][0]:%Y-%m-%d} to {scored_dates[mask][-1]:%Y-%m-%d}"
              for period, mask in masks.items()}
    portfolios = {
        name: {period: metrics(returns[mask], turnover[mask], config["sessions_per_year"])
               for period, mask in masks.items()}
        for name, (returns, turnover) in portfolio_returns.items()
    }
    calibration_range = f"{index[0]:%Y-%m-%d} to {index[window - 1]:%Y-%m-%d}"
    print_report(ranges, portfolios, cost_bps, calibration_range)
    results = {
        "strategy": config["strategy"],
        "data_range": ranges["full"],
        **portfolios["Kalman + EWMA"],
        "benchmark": {"name": f"{benchmark} buy and hold", **portfolios[f"{benchmark} buy and hold"]},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()