"""537_hierarchical_minimum_variance_portfolios
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/537_hierarchical_minimum_variance_portfolios.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c0a8f76240a0431b019ab49c34040a95a765860a on 2026-11-27 10:04 Europe/Oslo, exit 0, output sha256 79906d170fa3ca7aa3ad49d9d535eff8a93897d92f50957aa908c8e6b21dfdbf
Data source: Yahoo Finance，经 yfinance 获取日频 ETF 复权收盘价。资产池为 SPY、XLB、XLE、XLF、XLI、XLK、XLP、XLU、XLV、XLY。BIL 的日总回报只用于计算超额夏普比率。。
Data fetched: 2026-11-27T09:04:26+00:00
Sample period: 行情请求区间为 2008-01-01 至 2025-12-31（含），2008–2009 年用于预热。日志记录的实际回测区间为 2010-01-04 至 2025-12-31（含）。。 data_range (from output): {'start': '2010-01-04', 'end': '2025-12-31'}
IS/OOS split: 样本内为 2010-01-04 至 2018-12-31（含），样本外为 2019 年首个交易日至 2025-12-31（含）。正式配置连续模拟一次，随后按 2018-12-31 切分统计，持仓跨切分日延续。样本外为历史留出。。
Costs and slippage: 每笔买入和卖出按实际成交名义金额收取 5 bps 成本和 3 bps 固定滑点，合计 8 bps，含首次建仓。执行日先按收盘价更新旧持仓价值，再从组合扣除摩擦。成交金额按扣费后资本对应的目标持仓与交易前持仓的绝对差额计算。样本内 2 倍摩擦情形合计 16 bps。。
Signal lag: 每月最后一个交易日收盘后，用截至该收盘的日收益计算目标权重，于下一交易日收盘执行。信号日收盘至执行日收盘的收益按原持仓计，新权重从执行日收盘后的下一个收盘到收盘区间生效。。
Price adjustment: yfinance auto_adjust=True daily total-return close (splits and distributions)
Survivorship bias: 从当前仍存续的 ETF 中回溯选定固定资产池，存在幸存者偏差及事后选样风险。。
Parameter combinations tried: 4
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.1113720440024859, ann_vol=0.11283740515524897, sharpe=0.9925044802391011, sharpe_excess=0.9706624099394938, max_drawdown=-0.12062257845735058, total_return=1.5823035073279645, turnover=1.699560287908907, n_trades=466
    oos: cagr=0.08604005756322164, ann_vol=0.15973381940878645, sharpe=0.5969295966661868, sharpe_excess=0.4385160648856859, max_drawdown=-0.3295919234549559, total_return=0.7797210082319923, turnover=1.6695575586806808, n_trades=391
    full: cagr=0.10022054842146422, ann_vol=0.13534549450451006, sharpe=0.7736712410276406, sharpe_excess=0.6817861556657228, max_drawdown=-0.3295919234549557, total_return=3.5957798016227374, turnover=1.6864378218448715, n_trades=857
    benchmark: cagr=0.12614125466936388, ann_vol=0.1658711755626719, sharpe=0.7995870488747571, sharpe_excess=0.7245703824924257, max_drawdown=-0.36416724151352875, total_return=5.665734754938514, turnover=0.3234073934868083, n_trades=1920
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 537_hierarchical_minimum_variance_portfolios\nuniverse: [SPY, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nrisk_free_ticker: BIL\ndata_start: 2008-01-01\nsample_start: 2010-01-04\nis_end: 2018-12-31\nsample_end: 2025-12-31\nlookback_days: 252\nrelative_diagonal_ridge: 0.000001\nexecution_lag_sessions: 1\ncost_bps: 5.0\nslippage_bps: 3.0\nrobustness:\n  is_lookback_days: [210, 294]\n  friction_multiplier: 2.0\n'
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
"""Monthly long-only minimum variance with hierarchical Schur reduction."""

import hashlib
import json
import signal
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.cluster.hierarchy import linkage
from scipy.optimize import brentq
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
WEIGHT_TOLERANCE = 1e-10
PRICE_ADJUSTMENT = "yfinance auto_adjust=True daily total-return close (splits and distributions)"


def checked_prices(prices, tickers, start, end):
    prices = prices.reindex(columns=tickers).dropna(how="all").sort_index()
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("Yahoo price dates are empty or duplicated")
    if prices.index[0] > start + pd.Timedelta(days=7) or prices.index[-1] < end - pd.Timedelta(days=7):
        raise ValueError("Yahoo prices do not cover the requested date range")
    valid = np.isfinite(prices) & (prices > 0)
    if not valid.to_numpy().all():
        missing = valid.columns[~valid.all()].tolist()
        raise ValueError(f"Missing or invalid adjusted closes for {missing}")
    return prices.rename_axis("Date")


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    request_key = json.dumps([PRICE_ADJUSTMENT, sorted(tickers), str(start), str(end)])
    key = hashlib.sha256(request_key.encode()).hexdigest()[:20]
    path = CACHE / f"adjusted_closes_{key}.csv"
    marker = "DATA_FETCHED_AT: "
    if path.exists():
        with path.open(encoding="utf-8") as source:
            header = source.readline().strip()
            if not header.startswith(marker):
                raise ValueError(f"Missing source timestamp in {path.name}")
            fetched_at = header[len(marker):]
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(source, index_col="Date", parse_dates=["Date"])
        return checked_prices(prices, tickers, start, end), fetched_at

    with TemporaryDirectory(dir=CACHE) as yahoo_cache:
        yf.set_tz_cache_location(yahoo_cache)
        download = yf.download(
            tickers,
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d",
            auto_adjust=True,
            ignore_tz=True,
            multi_level_index=True,
            progress=False,
            timeout=10,
        )
    if download is None or download.empty:
        raise RuntimeError("Yahoo returned no adjusted prices")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prices = checked_prices(download["Close"], tickers, start, end)

    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as output:
        pending = Path(output.name)
        try:
            output.write(f"{marker}{fetched_at}\n")
            prices.to_csv(output)
            output.close()
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
    return prices, fetched_at


def elimination_levels(covariance):
    """Return the level at which each cluster representative is eliminated."""
    variances = covariance.diagonal()
    scales = np.sqrt(variances)
    correlation = covariance / np.outer(scales, scales)
    distance = np.sqrt(np.clip((1.0 - correlation) / 2.0, 0.0, 1.0))
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance), method="average")

    size = len(variances)
    representatives = list(range(size))
    heights = [0] * size
    levels = np.empty(size, dtype=int)
    for left, right in tree[:, :2].astype(int):
        first, second = representatives[left], representatives[right]
        if (variances[first], first) > (variances[second], second):
            first, second = second, first
        height = 1 + max(heights[left], heights[right])
        levels[second] = height
        representatives.append(first)
        heights.append(height)
    levels[representatives[-1]] = heights[-1] + 1
    return levels


def schur_solve(covariance, gamma, block_sizes):
    """Reduce covariance and gamma together, then reconstruct interior entries."""
    if len(block_sizes) == 1:
        return np.linalg.solve(covariance, gamma)

    n_interior = block_sizes[0]
    interior = covariance[:n_interior, :n_interior]
    cross = covariance[:n_interior, n_interior:]
    responses = np.linalg.solve(interior, np.column_stack((gamma[:n_interior], cross)))
    gamma_response, cross_response = responses[:, 0], responses[:, 1:]
    reduced = covariance[n_interior:, n_interior:] - cross.T @ cross_response
    reduced_gamma = gamma[n_interior:] - cross.T @ gamma_response
    junction = schur_solve(reduced, reduced_gamma, block_sizes[1:])
    return np.concatenate((gamma_response - cross_response @ junction, junction))


def minimum_variance_weights(window, relative_ridge):
    covariance = np.cov(window, rowvar=False, ddof=1)
    variances = covariance.diagonal()
    if np.any(variances <= 0):
        raise ValueError("An asset has zero variance within an estimation window")
    levels = elimination_levels(covariance)
    order = np.argsort(levels, kind="stable")
    size = len(variances)
    covariance = covariance / variances.mean() + relative_ridge * np.eye(size)

    # Each active set solves the paper's equality-constrained problem on its free assets.
    weights = np.full(size, 1.0 / size)
    free = np.ones(size, dtype=bool)
    for _ in range(4 * size**2):
        indices = order[free[order]]
        block_sizes = np.unique(levels[indices], return_counts=True)[1]
        raw = schur_solve(covariance[np.ix_(indices, indices)], np.ones(len(indices)), block_sizes)
        trial = np.zeros(size)
        trial[indices] = raw / raw.sum()

        negative = np.flatnonzero(trial < -WEIGHT_TOLERANCE)
        if len(negative):
            # Move only to the first zero bound, keeping the iterate feasible.
            fractions = weights[negative] / (weights[negative] - trial[negative])
            first_hit = np.argmin(fractions)
            weights += fractions[first_hit] * (trial - weights)
            weights[negative[first_hit]] = 0.0
            free[negative[first_hit]] = False
            continue

        trial = np.maximum(trial, 0.0)
        weights = trial / trial.sum()
        gradient = covariance @ weights
        reduced_gradient = np.where(free, np.inf, gradient - weights @ gradient)
        entrant = np.argmin(reduced_gradient)
        if reduced_gradient[entrant] >= -WEIGHT_TOLERANCE:
            return weights
        free[entrant] = True

    raise RuntimeError("The long-only minimum-variance solver did not converge")


def monthly_schedule(index, start, lag):
    months = index.to_period("M")
    first_sessions = np.flatnonzero(months[1:] != months[:-1]) + 1
    signal_rows = first_sessions - 1
    execution_rows = signal_rows + lag
    within_data = execution_rows < len(index)
    schedule = {
        index[execution]: int(as_of)
        for execution, as_of in zip(execution_rows[within_data], signal_rows[within_data])
        if index[execution] >= start
    }
    if not schedule:
        raise ValueError("No monthly execution in the requested sample")
    return schedule


def build_targets(returns, schedule, lookback, relative_ridge):
    values = returns.to_numpy()
    targets = {}
    for execution, as_of in schedule.items():
        first = as_of + 1 - lookback
        if first < 0:
            raise ValueError(f"Insufficient warm-up for {lookback} returns before {execution.date()}")
        targets[execution] = minimum_variance_weights(values[first:as_of + 1], relative_ridge)
    return targets


def simulate(returns, targets, friction):
    """Mark current weights to the close, then execute and deduct fees."""
    sample = returns.loc[min(targets):]
    weights = np.zeros(sample.shape[1])
    net_returns, turnover, trades = [], [], []
    for date, asset_returns in zip(sample.index, sample.to_numpy()):
        gross_multiplier = 1.0 + weights @ asset_returns
        drifted = weights * (1.0 + asset_returns) / gross_multiplier
        remaining = 1.0
        notional = np.zeros_like(weights)
        if date in targets:
            target = targets[date]
            if friction:
                # Fees depend on actual buys and sells, sized against post-fee capital.
                def fee_residual(remaining):
                    return remaining + friction * np.abs(remaining * target - drifted).sum() - 1.0

                remaining = brentq(fee_residual, 0.0, 1.0, xtol=1e-14)
            notional = np.abs(remaining * target - drifted)
            weights = target
        else:
            weights = drifted
        net_returns.append(gross_multiplier * remaining - 1.0)
        turnover.append(notional.sum())
        trades.append(np.count_nonzero(notional > WEIGHT_TOLERANCE))
    return pd.DataFrame(
        {"return": net_returns, "turnover": turnover, "n_trades": trades}, index=sample.index
    )


def metrics(daily_results, risk_free):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise ValueError("At least two sessions are needed for each metrics period")
    excess = returns - risk_free.loc[returns.index]
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    volatility, excess_volatility = returns.std(), excess.std()
    growth = np.concatenate(([1.0], np.cumprod(1.0 + returns.to_numpy())))
    years = len(returns) / SESSIONS_PER_YEAR
    return {
        "cagr": float(growth[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(volatility * annualizer),
        "sharpe": float(returns.mean() / volatility * annualizer) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_volatility * annualizer) if excess_volatility > 0 else None,
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1.0)),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, is_end, risk_free):
    return {
        "is": metrics(daily_results.loc[:is_end], risk_free),
        "oos": metrics(daily_results.loc[daily_results.index > is_end], risk_free),
        "full": metrics(daily_results, risk_free),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    data_start, sample_start, is_end, sample_end = (
        pd.Timestamp(config[key]) for key in ("data_start", "sample_start", "is_end", "sample_end")
    )
    universe = config["universe"]
    risk_free_ticker = config["risk_free_ticker"]
    lookback = config["lookback_days"]
    lag = config["execution_lag_sessions"]
    ridge = float(config["relative_diagonal_ridge"])
    cost_bps, slippage_bps = float(config["cost_bps"]), float(config["slippage_bps"])
    friction = (cost_bps + slippage_bps) / 10_000
    sensitivity = config["robustness"]["is_lookback_days"]
    multiplier = float(config["robustness"]["friction_multiplier"])

    if not data_start < sample_start < is_end < sample_end:
        raise ValueError("Expected data_start < sample_start < is_end < sample_end")
    if len(universe) < 2 or len(set(universe)) != len(universe) or risk_free_ticker != "BIL" or "BIL" in universe:
        raise ValueError("Use distinct assets and BIL as the separate risk-free proxy")
    if type(lag) is not int or lag < 1:
        raise ValueError("Execution lag must be at least one session")
    if any(type(days) is not int or days < 2 for days in [lookback, *sensitivity]):
        raise ValueError("Lookbacks must be integer counts of at least two sessions")
    if len(sensitivity) < 2 or len(set([lookback, *sensitivity])) != len(sensitivity) + 1:
        raise ValueError("Specify at least two distinct alternate lookbacks")
    if not (np.isfinite(ridge) and ridge > 0):
        raise ValueError("Relative diagonal ridge must be positive and finite")
    if not (cost_bps >= 0 and slippage_bps >= 0 and multiplier > 1 and 0 < friction * multiplier < 1):
        raise ValueError("Use nonnegative costs and slippage with a valid higher-friction case")

    prices, fetched_at = load_prices([*universe, risk_free_ticker], data_start, sample_end)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    asset_returns = returns[universe]
    risk_free = returns[risk_free_ticker]
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")

    print("ROBUSTNESS (IS only, fixed cases)")
    is_returns = asset_returns.loc[:is_end]
    is_schedule = monthly_schedule(is_returns.index, sample_start, lag)
    for days in sensitivity:
        targets = build_targets(is_returns, is_schedule, days, ridge)
        stress = metrics(simulate(is_returns, targets, friction), risk_free)
        print(f"IS_LOOKBACK_{days}: {json.dumps(stress, allow_nan=False)}")
    is_targets = build_targets(is_returns, is_schedule, lookback, ridge)
    stress = metrics(simulate(is_returns, is_targets, friction * multiplier), risk_free)
    print(f"IS_FRICTION_{multiplier:g}X: {json.dumps(stress, allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {len(sensitivity) + 2}")

    schedule = monthly_schedule(asset_returns.index, sample_start, lag)
    targets = build_targets(asset_returns, schedule, lookback, ridge)
    daily_results = simulate(asset_returns, targets, friction)
    official = period_metrics(daily_results, is_end, risk_free)
    equal_weights = np.full(len(universe), 1.0 / len(universe))
    benchmark_targets = dict.fromkeys(schedule, equal_weights)
    benchmark_results = simulate(asset_returns, benchmark_targets, friction)
    benchmark = period_metrics(benchmark_results, is_end, risk_free)

    for period, stats in official.items():
        print(f"{period.upper()}: {json.dumps(stats, allow_nan=False)}")
    for period, stats in benchmark.items():
        print(f"BENCHMARK_{period.upper()} (monthly equal weight): {json.dumps(stats, allow_nan=False)}")

    result = {
        "strategy": config["strategy"],
        "data_range": {
            "start": daily_results.index[0].date().isoformat(),
            "end": daily_results.index[-1].date().isoformat(),
        },
        **official,
        "benchmark": benchmark["full"],
    }
    print(f"RESULTS_JSON: {json.dumps(result, allow_nan=False)}")


if __name__ == "__main__":
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)