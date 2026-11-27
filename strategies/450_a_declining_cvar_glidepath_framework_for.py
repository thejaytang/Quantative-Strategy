"""450_a_declining_cvar_glidepath_framework_for
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/450_a_declining_cvar_glidepath_framework_for.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 6df04a4d89c55b7a1b162dbdcff0dafc1e2bd682 on 2026-11-27 08:11 Europe/Oslo, exit 0, output sha256 0604d77c4da2e197bae0a3f034f344c82c2c14a4f8de2f2e08099ea862d14377
Data source: Yahoo Finance 免费 ETF 日线收盘价，经 yfinance 获取。标的为 SPY、AGG、BIL，BIL 日总回报用作无风险收益代理。程序优先读取 data_cache/ 中的价格 CSV，缺失时下载。（config.yaml:2-3；strategy.py:21-44；strategy.py:234-236；README.md:17）
Data fetched: 2026-11-27T07:11:27+00:00
Sample period: 行情请求区间为 2007-06-01 至 2025-12-31，早期行情用于初始滚动窗口。实际回测收益区间为 2013-01-02 至 2025-12-31。（config.yaml:8-11；strategy.py:37-40；strategy.py:234-235；README.md:18；logs/run_20261127_074903.log:13）   data_range (from output): {'start': '2013-01-02', 'end': '2025-12-31'}
IS/OOS split: 样本内为 2013-01-02 至 2019-12-31，样本外为 2020-01-02 至 2025-12-31，均按实际交易日计。用样本内扣费后相对 BIL 的超额夏普选择滚动窗口，平手取较短窗口。该次运行选中 40 个月。样本外固定所选窗口，每月更新截至信号日的收益情景，并承接样本内期末持仓。README 将样本外区间描述为回溯式留出样本。（README.md:13；README.md:18-19；strategy.py:135-147；strategy.py:234-261；logs/run_20261127_074903.log:4）
Costs and slippage: 按每笔买入或卖出的成交金额计 2 bps 佣金和 3 bps 固定滑点，合计 5 bps。成交时以调仓前净值为基准，对扣费后的目标头寸与漂移后原头寸之间的绝对差额求和，从组合净值扣费。建仓和后续调仓均计费，策略与基准采用同一费率。固定滑点已建模，README 说明未计入随成交规模变化的市场冲击。（config.yaml:13-14；strategy.py:151-175；strategy.py:218-223；strategy.py:256-260；README.md:21；README.md:25）
Signal lag: 每月最后一个交易日收盘后生成信号，于下一交易日收盘模拟成交。成交日收盘前的收盘到收盘收益归原持仓，新持仓在成交后生效。（config.yaml:12；strategy.py:64-71；strategy.py:135-148；strategy.py:151-175；README.md:20）
Price adjustment: yfinance auto_adjust total-return adjusted close (splits and distributions)
Survivorship bias: 固定使用事后选定且存续至今的 SPY、AGG、BIL，README 明示存在存续偏差和资产选择偏差。（config.yaml:2；README.md:25）
Parameter combinations tried: 2
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.054930731879154315, ann_vol=0.04451402330839231, sharpe=1.2237041520544454, sharpe_excess=1.0896049680858813, max_drawdown=-0.054054877275687585, total_return=0.4533937644677768, turnover=9.811002440415205, n_trades=252
    oos: cagr=0.031104204313283423, ann_vol=0.07285853487071847, sharpe=0.45700911842597103, sharpe_excess=0.0980560849573388, max_drawdown=-0.1807288264948893, total_return=0.2011692188191052, turnover=7.588871321169314, n_trades=216
    full: cagr=0.04387519298450271, ann_vol=0.05928753078861906, sharpe=0.7540691369308014, sharpe_excess=0.4956268725128241, max_drawdown=-0.18072882649488908, total_return=0.7457718527023236, turnover=8.786239832518323, n_trades=468
    benchmark: cagr=0.09598659366551754, ann_vol=0.10384395290022047, sharpe=0.9348004938612747, sharpe_excess=0.7868358453702305, max_drawdown=-0.21456333570961805, total_return=2.2848867362003387, turnover=0.2671247740612504, n_trades=312
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: '450_ a_declining_cvar_glidepath_framework_for'\nuniverse: [SPY, AGG, BIL]\nrisk_free_ticker: BIL\nbenchmark_weights:\n  SPY: 0.60\n  AGG: 0.40\n\nprice_start: 2007-06-01\nbacktest_start: 2013-01-01\nis_end: 2019-12-31\nbacktest_end: 2025-12-31\nexecution_lag_sessions: 1\ncommission_bps: 2.0\nslippage_bps: 3.0\n\nglidepath_start: 1985-12-01\nglidepath_transition: 2018-12-01\nglidepath_target: 2025-12-01\ninitial_cvar_limit: 0.06\nterminal_cvar_limit: 0.03\ncvar_confidence: 0.90\nlookback_months_candidates: [40, 60]\nhit_and_run_burn_in: 20\nrandom_seed: 450\n"
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
"""Historical research simulation of a monthly CVaR glidepath."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, linprog


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252


def load_prices(tickers, start, end):
    cache = ROOT / "data_cache"
    cache.mkdir(exist_ok=True)
    identity = json.dumps([tickers, str(start.date()), str(end.date())])
    key = hashlib.sha256(identity.encode()).hexdigest()[:16]
    path = cache / f"yfinance_adjusted_close_{key}.csv"
    if path.exists():
        with path.open(newline="") as source:
            label, timestamp = next(csv.reader(source))
            if label != "fetched_at":
                raise ValueError(f"Invalid download timestamp in {path}")
            datetime.fromisoformat(timestamp)
            prices = pd.read_csv(source, index_col="date", parse_dates=["date"])
    else:
        with TemporaryDirectory(dir=cache) as yf_cache:
            yf.set_tz_cache_location(yf_cache)
            data = yf.download(
                tickers, start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
                auto_adjust=True, ignore_tz=True, threads=False, progress=False, timeout=12,
            )
        if data.empty or "Close" not in data:
            raise RuntimeError("Yahoo Finance returned no adjusted close data")
        prices = data["Close"].reindex(columns=tickers).sort_index()
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    prices = prices.reindex(columns=tickers)
    if (prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing
            or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any()):
        raise ValueError("Incomplete or invalid daily adjusted close data for the configured ETFs")
    seven_days = pd.Timedelta(days=7)
    if not (start <= prices.index[0] <= start + seven_days
            and end - seven_days <= prices.index[-1] <= end):
        raise ValueError("The ETF price history does not cover the configured dates")

    if not path.exists():
        with NamedTemporaryFile(mode="w", newline="", dir=cache, delete=False) as target:
            temporary_path = Path(target.name)
            csv.writer(target).writerow(["fetched_at", timestamp])
            prices.to_csv(target, index_label="date")
        temporary_path.replace(path)
    return prices, timestamp


def month_end_schedule(prices, start, lag):
    monthly = prices.groupby(prices.index.to_period("M")).tail(1)
    schedule = []
    for signal in monthly.index[:-1]:
        fill_position = prices.index.get_loc(signal) + lag
        if fill_position < len(prices) and prices.index[fill_position] >= start:
            schedule.append((signal, prices.index[fill_position]))
    return monthly.pct_change().dropna(), schedule


def cvar_limit(month, config):
    transition = pd.Period(config["glidepath_transition"], freq="M")
    target = pd.Period(config["glidepath_target"], freq="M")
    progress = np.clip((month.ordinal - transition.ordinal) / (target.ordinal - transition.ordinal), 0, 1)
    initial = config["initial_cvar_limit"]
    return initial + (config["terminal_cvar_limit"] - initial) * progress


def sample_allocation(scenarios, limit, confidence, cash_index, steps, rng):
    """Draw one allocation on the long-only simplex after hit-and-run burn-in."""
    n_scenarios, n_assets = scenarios.shape
    tail_size = max(1.0, round((1 - confidence) * n_scenarios, 12))
    n_tail = int(np.ceil(tail_size))
    tail_weights = np.ones(n_tail)
    tail_weights[-1] = tail_size - (n_tail - 1)
    losses = -scenarios

    def cvar(weights):
        ranked = np.sort(losses @ weights)[::-1]
        return float(ranked[:n_tail] @ tail_weights / tail_size)

    weights = np.full(n_assets, 1.0 / n_assets)
    if cvar(weights) > limit:
        weights = np.eye(n_assets)[cash_index]
    if cvar(weights) > limit:
        # Minimize threshold + average excess loss over the long-only simplex.
        objective = np.r_[np.zeros(n_assets), 1.0, np.full(n_scenarios, 1 / tail_size)]
        fit = linprog(
            objective, A_ub=np.c_[losses, -np.ones(n_scenarios), -np.eye(n_scenarios)],
            b_ub=np.zeros(n_scenarios), A_eq=[np.r_[np.ones(n_assets), np.zeros(n_scenarios + 1)]],
            b_eq=[1.0], bounds=[(0, None)] * n_assets + [(None, None)] + [(0, None)] * n_scenarios,
            method="highs",
        )
        if not fit.success:
            raise RuntimeError(f"Unable to compute minimum CVaR: {fit.message}")
        weights = fit.x[:n_assets]
        if cvar(weights) > limit:
            return np.zeros(n_assets)

    for _ in range(steps):
        direction = rng.normal(size=n_assets)
        direction -= direction.mean()
        positive = direction > 0
        negative = direction < 0
        low = np.max(-weights[positive] / direction[positive])
        high = np.min(-weights[negative] / direction[negative])

        def margin(distance):
            return cvar(weights + distance * direction) - limit

        if low < 0 and margin(low) > 0:
            low = brentq(margin, low, 0)
        if high > 0 and margin(high) > 0:
            high = brentq(margin, 0, high)
        candidate = np.maximum(weights + rng.uniform(low, high) * direction, 0)
        candidate /= candidate.sum()
        if cvar(candidate) <= limit:
            weights = candidate
    return weights


def strategy_orders(monthly_returns, schedule, lookback, config):
    orders = {}
    cash_index = config["universe"].index(config["risk_free_ticker"])
    for signal, fill in schedule:
        scenarios = monthly_returns.loc[:signal].tail(lookback).to_numpy()
        if len(scenarios) != lookback:
            raise ValueError(f"Insufficient monthly history for the {signal.date()} signal")
        month = signal.to_period("M") + 1
        rng = np.random.default_rng([config["random_seed"], month.ordinal])
        orders[fill] = sample_allocation(
            scenarios, cvar_limit(month, config), config["cvar_confidence"], cash_index,
            config["hit_and_run_burn_in"] + 1, rng,
        )
    return orders


def simulate(returns, orders, cost_rate):
    n_assets = returns.shape[1]
    weights = np.zeros(n_assets)
    rows = []
    for date, price_changes in zip(returns.index, returns.to_numpy()):
        # Existing holdings earn the full close-to-close return before any close fill.
        positions = weights * (1 + price_changes)
        gross = 1 + weights @ price_changes
        weights = positions / gross
        turnover = 0.0
        trades = 0
        if date in orders:
            target = orders[date]

            def funding(net):
                return net + cost_rate * np.abs(net * target - weights).sum() - 1

            net = brentq(funding, 0.0, 1.0)
            traded = np.abs(net * target - weights)
            turnover = traded.sum()
            trades = np.count_nonzero(traded > 1e-10)
            gross *= net
            weights = target
        rows.append((gross - 1, turnover, trades))
    return pd.DataFrame(rows, index=returns.index, columns=["return", "turnover", "n_trades"])


def period_metrics(daily_results, risk_free):
    if len(daily_results) < 2:
        raise ValueError("Each period needs at least two return observations")
    returns = daily_results["return"]
    excess = returns - risk_free.loc[returns.index]
    years = len(returns) / SESSIONS_PER_YEAR
    wealth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]

    def sharpe(series):
        volatility = series.std(ddof=1)
        return float(np.sqrt(SESSIONS_PER_YEAR) * series.mean() / volatility) if volatility > 0 else None

    return {
        "cagr": float(wealth[-1] ** (1 / years) - 1),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(excess),
        "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
        "total_return": float(wealth[-1] - 1),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def main():
    with (ROOT / "config.yaml").open() as source:
        config = yaml.safe_load(source)
    tickers = config["universe"]
    price_start, test_start, is_end, test_end = (
        pd.Timestamp(config[key]) for key in ("price_start", "backtest_start", "is_end", "backtest_end")
    )
    if not price_start < test_start < is_end < test_end:
        raise ValueError("Check the configured sample dates")
    entry, transition, target = (
        pd.Period(config[key], freq="M") for key in ("glidepath_start", "glidepath_transition", "glidepath_target")
    )
    if not entry <= test_start.to_period("M") or not entry <= transition < target:
        raise ValueError("Check the configured glidepath dates")
    if not 0 <= config["terminal_cvar_limit"] < config["initial_cvar_limit"]:
        raise ValueError("The monthly CVaR limit must decline to a nonnegative value")
    lag = config["execution_lag_sessions"]
    commission = config["commission_bps"]
    slippage = config["slippage_bps"]
    cost_rate = (commission + slippage) / 10_000
    if not isinstance(lag, int) or lag < 1 or not 0 <= cost_rate < 1 or commission < 0 or slippage < 0:
        raise ValueError("Check the execution lag and fee assumptions")
    if not 0 < config["cvar_confidence"] < 1 or config["hit_and_run_burn_in"] < 0:
        raise ValueError("Check the CVaR sampling settings")
    if len(set(tickers)) != len(tickers) or len(tickers) < 2 or config["risk_free_ticker"] not in tickers:
        raise ValueError("Check the ETF universe and risk-free proxy")
    weights = pd.Series(config["benchmark_weights"], dtype=float)
    if (not weights.index.isin(tickers).all() or not np.isfinite(weights).all()
            or (weights < 0).any() or not np.isclose(weights.sum(), 1)):
        raise ValueError("Benchmark weights must be long-only, on universe ETFs, and sum to one")
    benchmark_weights = weights.reindex(tickers, fill_value=0).to_numpy()

    prices, timestamp = load_prices(tickers, price_start, test_end)
    returns = prices.pct_change().loc[test_start:]
    risk_free = returns[config["risk_free_ticker"]]
    print(f"DATA_FETCHED_AT: {timestamp}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return adjusted close (splits and distributions)")

    is_monthly, is_schedule = month_end_schedule(prices.loc[:is_end], test_start, lag)
    scores = []
    for lookback in config["lookback_months_candidates"]:
        if lookback < 2:
            raise ValueError("The lookback must contain at least two monthly returns")
        orders = strategy_orders(is_monthly, is_schedule, lookback, config)
        daily_results = simulate(returns.loc[:is_end], orders, cost_rate)
        score = period_metrics(daily_results, risk_free)["sharpe_excess"]
        scores.append((lookback, score))
    eligible = [(lookback, score) for lookback, score in scores if score is not None]
    if not eligible:
        raise ValueError("No in-sample candidate has a defined excess Sharpe ratio")
    selected, _ = max(eligible, key=lambda pair: (pair[1], -pair[0]))
    print(f"PARAM_COMBOS_TRIED: {len(scores)}")
    print(f"IS_SELECTED_LOOKBACK_MONTHS: {selected}")

    monthly, schedule = month_end_schedule(prices, test_start, lag)
    orders = strategy_orders(monthly, schedule, selected, config)
    print(f"INFEASIBLE_CVAR_CASH_REBALANCES: {sum(not target.any() for target in orders.values())}")
    strategy_daily_results = simulate(returns, orders, cost_rate)
    benchmark_daily_results = simulate(returns, {fill: benchmark_weights for _, fill in schedule}, cost_rate)
    periods = {"is": returns.index <= is_end, "oos": returns.index > is_end, "full": slice(None)}
    benchmark_metrics = {period: period_metrics(benchmark_daily_results.loc[mask], risk_free) for period, mask in periods.items()}
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": str(returns.index[0].date()), "end": str(returns.index[-1].date())},
    }
    print(f"Benchmark weights {config['benchmark_weights']}")
    for period, mask in periods.items():
        result[period] = period_metrics(strategy_daily_results.loc[mask], risk_free)
        print(f"{period.upper()} strategy {json.dumps(result[period], allow_nan=False)}")
        print(f"{period.upper()} benchmark {json.dumps(benchmark_metrics[period], allow_nan=False)}")
    result["benchmark"] = benchmark_metrics["full"]
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()