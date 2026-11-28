"""522_asset_allocation_using_a_markov_process
WARNING: known defects (see DEEP_REVIEW_FINDINGS.md); numbers invalid.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/522_asset_allocation_using_a_markov_process.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 8cbf3ed748f7cfff4bfd2cf26cd63111c21d942b on 2026-11-27 10:20 Europe/Oslo, exit 0, output sha256 2a0d3723b02b8385d6179ec0c117797b75978fce7a5fe2428c32b9d91a13e506
Data source: 公开 Yahoo Finance 行情，经 yfinance 获取日线。风险资产为 SPY、EFA、TLT、GLD、VNQ，BIL 为现金基准。
Data fetched: 2026-11-27T09:20:23+00:00
Sample period: 行情覆盖 2007-12-03 至 2025-12-31，初始估计使用 2008-01 至 2015-12 的 96 个完整自然月，实际回测覆盖 2016-01-04 至 2025-12-31。 data_range (from output): {'start': '2016-01-04', 'end': '2025-12-31'}
IS/OOS split: 样本内为 2016-01-04 至 2020-12-31，样本外为 2021-01-04 至 2025-12-31。稳健性变体仅在样本内运行，正式回测沿用默认参数，模型按月扩展历史重新估计，持仓跨分界连续。
Costs and slippage: 每笔买入或卖出按实际买卖名义金额计 2 bps 手续费及 5 bps 固定比例滑点，合计 7 bps，于换仓收盘从组合净值扣除，首次建仓也计费。
Signal lag: 月末最后一个交易日收盘后，以截至该日的历史数据计算信号，默认于下一交易日收盘模拟成交。成交日的收盘到收盘收益先按旧权重计入，再换仓扣费，新权重从成交收盘后开始承担收益。
Price adjustment: yfinance auto_adjust=True daily total-return adjusted close
Survivorship bias: 风险资产池为事后固定选取的持续存续代表性基金，因此存在资产选择与幸存者偏差。
Parameter combinations tried: 6
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.14685712340990054, ann_vol=0.1073450501053131, sharpe=1.330605162898823, sharpe_excess=1.2386946625076418, max_drawdown=-0.1714272591093443, total_return=0.9829439295244229, turnover=2.775717042848573, n_trades=299
    oos: cagr=0.06961751912275171, ann_vol=0.12905070974510402, sharpe=0.5858841016609875, sharpe_excess=0.3485537503164322, max_drawdown=-0.273116761834426, total_return=0.3981784596905116, turnover=3.5690539607006357, n_trades=291
    full: cagr=0.10762564315972512, ann_vol=0.11867315591232751, sharpe=0.9208046948635887, sharpe_excess=0.7502313811827466, max_drawdown=-0.27311676183442557, total_return=1.7725094890351114, turnover=3.171754366597315, n_trades=590
    benchmark: cagr=0.09012054940737335, ann_vol=0.1116825771194709, sharpe=0.8287206032321855, sharpe_excess=0.6475542054218475, max_drawdown=-0.2535336318288488, total_return=1.3651190906385082, turnover=0.39183687445769433, n_trades=600
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: '522_ asset_allocation_using_a_markov_process'\nuniverse: [SPY, EFA, TLT, GLD, VNQ]\ncash_proxy: BIL\ndata_start: 2007-12-03\ndata_end: 2025-12-31\nbacktest_start: 2016-01-04\nis_end: 2020-12-31\nlag_trading_days: 1\ncost_bps: 2.0\nslippage_bps: 5.0\nmodel:\n  states: 4\n  minimum_history_months: 96\n  covariance_shrinkage: 0.10\nrobustness:\n  states: [3, 5]\n  covariance_shrinkage: [0.05, 0.15]\n  friction_multiplier: 2.0\n"
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
import hashlib
import json
import signal
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.optimize import brentq, nnls


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
PRICE_ADJUSTMENT = "yfinance auto_adjust=True daily total-return adjusted close"


def load_settings():
    settings = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for field in ("data_start", "data_end", "backtest_start", "is_end"):
        settings[field] = pd.Timestamp(settings[field]).normalize()
    if not settings["data_start"] < settings["backtest_start"] < settings["is_end"] < settings["data_end"]:
        raise ValueError("Expected data_start < backtest_start < is_end < data_end")
    lag = settings["lag_trading_days"]
    if type(lag) is not int or lag < 1:
        raise ValueError("lag_trading_days must be an integer of at least one")
    universe = settings["universe"]
    if len(set(universe)) != len(universe) or len(universe) < 2 or settings["cash_proxy"] in universe:
        raise ValueError("Use at least two distinct risky assets and a separate cash proxy")
    return settings


def check_prices(prices, symbols, start, end):
    if prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Expected a nonempty, strictly increasing daily price index")
    if (prices.index[0] < start or prices.index[-1] > end
            or prices.index[0] - start > pd.Timedelta(days=6)
            or end - prices.index[-1] > pd.Timedelta(days=6)):
        raise ValueError("Price history does not cover the requested date range")
    prices = prices.reindex(columns=symbols)
    invalid = ~np.isfinite(prices) | (prices <= 0)
    if invalid.any().any():
        counts = invalid.sum()
        raise ValueError(f"Missing or invalid adjusted closes per ticker: {counts[counts > 0].to_dict()}")
    return prices


def load_prices(settings):
    symbols = [*settings["universe"], settings["cash_proxy"]]
    start, end = settings["data_start"], settings["data_end"]
    request = f"yfinance:auto_adjust:1d:{start.date()}:{end.date()}:{','.join(symbols)}"
    key = hashlib.sha256(request.encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"adjusted_closes_{key}.csv"
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as cached:
            fetched_at = datetime.fromisoformat(cached.readline().strip()).isoformat()
            prices = pd.read_csv(cached, index_col="Date", parse_dates=["Date"])
        return check_prices(prices, symbols, start, end), fetched_at

    CACHE_DIR.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(CACHE_DIR / "yfinance"))
    for attempt in range(2):
        raw = yf.download(
            symbols, start=start.date().isoformat(), end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=True, ignore_tz=True, threads=6, progress=False, timeout=15,
        )
        try:
            if raw is None or "Close" not in raw:
                raise ValueError("yfinance returned no adjusted closes")
            prices = check_prices(raw["Close"].rename_axis("Date"), symbols, start, end)
            break
        except ValueError:
            if attempt == 1:
                raise
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE_DIR, delete=False) as temporary:
        try:
            temporary.write(f"{fetched_at}\n")
            prices.to_csv(temporary, index_label="Date")
            temporary.flush()
            Path(temporary.name).replace(cache_path)
        finally:
            Path(temporary.name).unlink(missing_ok=True)
    return prices, fetched_at


def regularized_covariance(returns, shrinkage):
    covariance = np.cov(returns, rowvar=False)
    average_variance = np.trace(covariance) / len(covariance)
    if average_variance <= 0:
        raise ValueError("Cannot estimate a frontier from a zero-variance return sample")
    return (1.0 - shrinkage) * covariance + shrinkage * average_variance * np.eye(len(covariance))


def frontier_coefficients(returns, shrinkage):
    mean = returns.mean(axis=0)
    covariance = regularized_covariance(returns, shrinkage)
    ones = np.ones(len(mean))
    inverse_times_ones = np.linalg.solve(covariance, ones)
    a = ones @ inverse_times_ones
    minimum_variance_return = mean @ inverse_times_ones / a
    centered_mean = mean - minimum_variance_return
    # Equivalent to (A*C - B^2) / A^2, avoiding cancellation.
    curvature_squared = centered_mean @ np.linalg.solve(covariance, centered_mean) / a
    return np.array([minimum_variance_return, 1.0 / np.sqrt(a), np.sqrt(max(curvature_squared, 0.0))])


def tangency_weights(returns, shrinkage):
    risky = returns[:, :-1]
    covariance = regularized_covariance(risky, shrinkage)
    excess_mean = (risky - returns[:, -1:]).mean(axis=0)
    if excess_mean.max() <= 0:
        # For all nonpositive excess means, the best fully invested Sharpe is at a single asset.
        weights = np.zeros(len(excess_mean))
        weights[np.argmax(excess_mean / np.sqrt(covariance.diagonal()))] = 1.0
        return weights

    # Solve the convex problem over nonnegative directions, then normalize the best direction.
    cholesky = np.linalg.cholesky(covariance)
    direction, _ = nnls(cholesky.T, np.linalg.solve(cholesky, excess_mean))
    return direction / direction.sum()


def markov_weights(coefficients, month_returns, states, shrinkage):
    features = np.asarray(coefficients)
    scales = features.std(axis=0)
    standardized = (features - features.mean(axis=0)) / np.where(scales > 0, scales, 1.0)
    labels = cut_tree(linkage(standardized, method="ward"), n_clusters=states).ravel()
    counts = np.zeros((states, states), dtype=int)
    np.add.at(counts, (labels[:-1], labels[1:]), 1)
    outgoing = counts[labels[-1]]
    if not outgoing.sum():
        outgoing = counts.sum(axis=0)
    probabilities = outgoing / outgoing.sum()

    target = np.zeros(month_returns[0].shape[1] - 1)
    for state in np.flatnonzero(probabilities):
        state_returns = np.concatenate([block for label, block in zip(labels, month_returns) if label == state])
        target += probabilities[state] * tangency_weights(state_returns, shrinkage)
    return target / target.sum()


def build_targets(daily_returns, settings, states, shrinkage):
    dates = daily_returns.index
    lag = settings["lag_trading_days"]
    minimum_history = settings["model"]["minimum_history_months"]
    if type(states) is not int or type(minimum_history) is not int or not 2 <= states < minimum_history:
        raise ValueError("Require 2 <= states < minimum_history_months, both integers")
    if not 0 < shrinkage <= 1:
        raise ValueError("covariance_shrinkage must be in (0, 1]")

    targets, coefficients, month_returns = {}, [], []
    for number, (_, month) in enumerate(daily_returns.groupby(dates.to_period("M"))):
        if number == 0:
            continue  # The first price month can be incomplete.
        execution_index = dates.get_loc(month.index[-1]) + lag
        if execution_index >= len(dates):
            break  # There is no eligible fill before the evaluation cutoff.
        block = month.to_numpy()
        if len(block) <= block.shape[1] - 1:
            raise ValueError(f"Too few daily returns to estimate the frontier in {month.index[-1]:%Y-%m}")
        coefficients.append(frontier_coefficients(block[:, :-1], shrinkage))
        month_returns.append(block)
        execution_date = dates[execution_index]
        if len(coefficients) >= minimum_history and execution_date >= settings["backtest_start"]:
            targets[execution_date] = markov_weights(coefficients, month_returns, states, shrinkage)
    if not targets:
        raise ValueError("No scheduled fills in the evaluation period")
    return targets


def simulate(daily_returns, targets, cost_bps, slippage_bps):
    friction = (cost_bps + slippage_bps) / 10_000
    if not (cost_bps >= 0 and slippage_bps >= 0 and friction < 1):
        raise ValueError("Cost and slippage must be nonnegative and sum to less than 10000 bps")
    daily_returns = daily_returns.loc[min(targets):]
    weights = np.zeros(daily_returns.shape[1] - 1)
    rows = []
    for date, returns in zip(daily_returns.index, daily_returns.to_numpy()):
        risky_returns, cash_return = returns[:-1], returns[-1]
        growth = 1.0 + weights @ risky_returns
        drifted = weights * (1.0 + risky_returns) / growth
        traded, trades, retained = 0.0, 0, 1.0
        weights = drifted
        if date in targets:
            weights = targets[date]
            # Post-cost target values and costs on actual traded value must agree.
            retained = brentq(lambda scale: scale + friction * np.abs(scale * weights - drifted).sum() - 1.0,
                              0.0, 1.0)
            trade_sizes = np.abs(retained * weights - drifted)
            traded = trade_sizes.sum()
            trades = np.count_nonzero(trade_sizes > 1e-8)
        rows.append((growth * retained - 1.0, cash_return, traded, trades))
    return pd.DataFrame(rows, index=daily_returns.index, columns=["return", "cash_return", "turnover", "n_trades"])


def metrics(daily_results):
    if len(daily_results) < 2:
        raise ValueError("At least two sessions are required for period metrics")
    returns = daily_results["return"].to_numpy()
    excess = returns - daily_results["cash_return"].to_numpy()
    vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    equity = np.r_[1.0, np.cumprod(1.0 + returns)]
    years = len(returns) / SESSIONS_PER_YEAR
    return {
        "cagr": float(equity[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(vol * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": float(returns.mean() / vol * np.sqrt(SESSIONS_PER_YEAR)) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * np.sqrt(SESSIONS_PER_YEAR)) if excess_vol > 0 else None,
        "max_drawdown": float(np.min(equity / np.maximum.accumulate(equity) - 1.0)),
        "total_return": float(equity[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, is_end):
    return {
        "is": metrics(daily_results.loc[:is_end]),
        "oos": metrics(daily_results.loc[daily_results.index > is_end]),
        "full": metrics(daily_results),
    }


def main():
    settings = load_settings()
    prices, fetched_at = load_prices(settings)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"PRICE_DATA_RANGE: {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}")

    daily_returns = prices.pct_change(fill_method=None)
    is_end = settings["is_end"]
    is_returns = daily_returns.loc[:is_end]
    states = settings["model"]["states"]
    shrinkage = settings["model"]["covariance_shrinkage"]
    cost, slippage = settings["cost_bps"], settings["slippage_bps"]
    robustness = settings["robustness"]
    multiplier = robustness["friction_multiplier"]
    default = (states, shrinkage, cost, slippage)
    variants = list(dict.fromkeys([
        default,
        *((value, shrinkage, cost, slippage) for value in robustness["states"]),
        *((states, value, cost, slippage) for value in robustness["covariance_shrinkage"]),
        (states, shrinkage, cost * multiplier, slippage * multiplier),
    ]))
    print("ROBUSTNESS (IS only, fixed configurations)")
    target_sets = {}
    for case_states, case_shrinkage, case_cost, case_slippage in variants:
        model = (case_states, case_shrinkage)
        if model not in target_sets:
            target_sets[model] = build_targets(is_returns, settings, *model)
        case = simulate(is_returns, target_sets[model], case_cost, case_slippage)
        print(f"IS states={case_states} shrinkage={case_shrinkage:g} cost_bps={case_cost:g} "
              f"slippage_bps={case_slippage:g}: {json.dumps(metrics(case), allow_nan=False)}")

    targets = build_targets(daily_returns, settings, states, shrinkage)
    strategy_results = simulate(daily_returns, targets, cost, slippage)
    n_assets = len(settings["universe"])
    equal_weights = np.full(n_assets, 1.0 / n_assets)
    benchmark_targets = dict.fromkeys(targets, equal_weights)
    benchmark_results = simulate(daily_returns, benchmark_targets, cost, slippage)
    strategy_periods = period_metrics(strategy_results, is_end)
    benchmark_periods = period_metrics(benchmark_results, is_end)

    start_date, end_date = strategy_results.index[[0, -1]]
    oos_start = strategy_results.index[strategy_results.index > is_end][0]
    print(f"Date Range: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d} "
          f"(IS through {is_end:%Y-%m-%d}, OOS from {oos_start:%Y-%m-%d})")
    for period in ("is", "oos", "full"):
        print(f"STRATEGY {period.upper()}: {json.dumps(strategy_periods[period], allow_nan=False)}")
        print(f"EQUAL WEIGHT {period.upper()}: {json.dumps(benchmark_periods[period], allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {len(variants)}")
    result = {
        "strategy": settings["strategy"],
        "data_range": {"start": start_date.date().isoformat(), "end": end_date.date().isoformat()},
        **strategy_periods,
        "benchmark": {"name": "Monthly equal weight", **benchmark_periods["full"], **benchmark_periods},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)