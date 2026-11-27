"""494_wasserstein_penalisation_as_an_endogenou
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/494_wasserstein_penalisation_as_an_endogenou.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit d2f4ee010416fa7342bf8a9f3404c2adc27e3f7d on 2026-11-27 09:18 Europe/Oslo, exit 0, output sha256 bd006c59bb4d89c767d9479487da142c8051e40473881e7b81ba250fdb7e8eb6
Data source: Yahoo Finance（yfinance）日频调整收盘价。配置的组合为九只美国 Select Sector SPDR ETF：XLB、XLE、XLF、XLI、XLK、XLP、XLU、XLV、XLY。BIL 的日收益仅用于计算超额夏普比率。（config.yaml:2-3；README.md:15；strategy.py:69-80、175-186、225-228）
Data fetched: 2026-11-27T08:18:37.121411+00:00
Sample period: 配置自 2008-07-01 起下载行情用于预热，评价区间设为 2010-01-01 至 2025-12-31。日志记录的实际评价区间为 2010-01-04 至 2025-12-31。（config.yaml:4-8；README.md:17；strategy.py:225-229、256-261；logs/run_20261127_084116.log:19）   data_range (from output): {'start': '2010-01-04', 'end': '2025-12-31'}
IS/OOS split: 样本内从评价期首个交易日至 2017-12-31（含），样本外为 2018-01-01 至 2025-12-31，均按实际交易日计。正式模型以固定参数滚动估计，并在分界处延续同一次模拟的持仓。六组单参数变体仅在样本内运行。（config.yaml:4-12、17-23；README.md:17、23；strategy.py:229-243、246-248、263-266）
Costs and slippage: 交易成本 5 bp，固定滑点费用 5 bp。每次调仓按执行日收盘时的买入与卖出金额分别扣除合计 10 bp，首次建仓同样计费。模拟以收盘价成交，滑点通过扣减组合净值计入。（config.yaml:13-16；strategy.py:150-172、213-219；README.md:19）
Signal lag: 使用截至月末最后一个交易日收盘 t 的数据计算目标权重，于下一交易日收盘 t+1 执行。旧持仓计入截至 t+1 收盘的收益，新持仓从 t+1 收盘至再下一交易日收盘的区间开始贡献收益。（config.yaml:13-14；strategy.py:129-147、150-172；README.md:19）
Price adjustment: yfinance auto_adjust total-return adjusted close
Survivorship bias: 固定使用九只早期行业 ETF，存在事后选择和存续偏差风险。行业分类、成分调整及后来增设的行业 ETF 会影响代表性。（config.yaml:2；strategy.py:198-206、225-228；README.md:15、27）
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.12566735348053282, ann_vol=0.13045785079606306, sharpe=0.9729468784197771, sharpe_excess=0.9681872677405006, max_drawdown=-0.16126163576153585, total_return=1.5743556354858965, turnover=1.3116130203998737, n_trades=817
    oos: cagr=0.10183296367771555, ann_vol=0.1716739306243289, sharpe=0.6511371829737314, sharpe_excess=0.5095345032396296, max_drawdown=-0.3302553340803467, total_return=1.1681556590601505, turnover=1.4689327242925239, n_trades=821
    full: cagr=0.11369232235830018, ann_vol=0.15243670395615222, sharpe=0.7830114386936448, sharpe_excess=0.7013912723831803, max_drawdown=-0.33025533408034713, total_return=4.581603739512112, turnover=1.3902337769923487, n_trades=1638
    benchmark: cagr=0.12455456780964957, ann_vol=0.16569857321703862, sharpe=0.7917386012723643, sharpe_excess=0.7166443424684629, max_drawdown=-0.36716111915496197, total_return=5.517336032309704, turnover=0.34399504283592286, n_trades=1728
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "494_wasserstein_penalisation_as_an_endogenou"\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nrisk_free_ticker: BIL\ndates:\n  price_start: 2008-07-01\n  evaluation_start: 2010-01-01\n  is_end: 2017-12-31\n  end: 2025-12-31\nmodel:\n  lookback_sessions: 252\n  risk_aversion: 10.0\n  delta: 1.0e-5\nexecution:\n  lag_sessions: 1\n  cost_bps: 5.0\n  slippage_bps: 5.0\nrobustness:\n  - {delta: 5.0e-6}\n  - {delta: 2.0e-5}\n  - {risk_aversion: 7.5}\n  - {risk_aversion: 12.5}\n  - {lookback_sessions: 60}\n  - {delta: 0.0}\n'
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
"""Research backtest of Chen's L2-penalised long-only allocation."""

import hashlib
import json
import re
import signal
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, minimize
from sklearn.covariance import LedoitWolf


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
TRADE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class Model:
    lookback_sessions: int
    risk_aversion: float
    delta: float

    def __post_init__(self):
        if type(self.lookback_sessions) is not int or self.lookback_sessions < 2:
            raise ValueError("lookback_sessions must be an integer of at least two")
        if not np.isfinite(self.risk_aversion) or self.risk_aversion <= 0:
            raise ValueError("risk_aversion must be finite and positive")
        if not np.isfinite(self.delta) or self.delta < 0:
            raise ValueError("delta must be finite and nonnegative")


def check_prices(prices, tickers, start, end):
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("Expected a nonempty Yahoo Finance panel with unique sessions")
    buffer = pd.Timedelta(days=7)
    if not start <= prices.index[0] <= start + buffer or not end - buffer <= prices.index[-1] <= end:
        raise ValueError("Yahoo Finance panel does not cover the requested date range")
    problems = (~np.isfinite(prices)) | (prices <= 0)
    if problems.any().any():
        rows, cols = np.where(problems.to_numpy())
        examples = ", ".join(f"{prices.index[row]:%Y-%m-%d} {tickers[col]}"
                             for row, col in zip(rows[:5], cols[:5], strict=True))
        raise ValueError(f"Missing or invalid adjusted closes: {examples}")


def load_prices(tickers, start, end):
    cache = ROOT / "data_cache"
    cache.mkdir(exist_ok=True)
    request = f"auto_adjust|{start:%Y-%m-%d}|{end:%Y-%m-%d}|{','.join(tickers)}"
    key = hashlib.sha256(request.encode()).hexdigest()[:20]
    path = cache / f"yfinance_close_{key}.csv"

    if path.exists():
        with path.open(encoding="utf-8") as handle:
            fetched_at = datetime.fromisoformat(handle.readline().strip())
            prices = pd.read_csv(handle, index_col="Date", parse_dates=["Date"])
        prices = prices.reindex(columns=tickers).sort_index()
        check_prices(prices, tickers, start, end)
        return prices, fetched_at

    with TemporaryDirectory(dir=cache) as request_cache:
        yf.set_tz_cache_location(request_cache)
        raw = yf.download(
            tickers, start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True, ignore_tz=True, progress=False, threads=4, timeout=12,
        )
    if raw is None or raw.empty or "Close" not in raw.columns.get_level_values(0):
        raise RuntimeError("Yahoo Finance returned no adjusted closes")
    prices = raw["Close"].reindex(columns=tickers).sort_index()
    prices.index.name = "Date"
    check_prices(prices, tickers, start, end)
    fetched_at = datetime.now(timezone.utc)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache, delete=False) as handle:
        temporary_path = Path(handle.name)
        try:
            handle.write(fetched_at.isoformat() + "\n")
            prices.to_csv(handle)
            handle.flush()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def fit_weights(window, model):
    mean = window.mean(axis=0)
    mean = mean - mean.mean()  # Subtract a constant from the simplex objective.
    variance = model.risk_aversion * LedoitWolf().fit(window).covariance_
    penalty = np.sqrt(model.delta)
    scale = max(np.max(np.abs(mean)), np.max(np.abs(variance)), penalty, 1e-12)
    mean, variance, penalty = mean / scale, variance / scale, penalty / scale

    def loss(weights):
        variance_gradient = variance @ weights
        norm = np.linalg.norm(weights)
        value = weights @ (variance_gradient - mean) + penalty * norm
        gradient = 2 * variance_gradient - mean + penalty * weights / norm
        return value, gradient

    n_assets = len(mean)
    initial = np.full(n_assets, 1 / n_assets)
    ones = np.ones(n_assets)
    result = minimize(
        loss, initial, jac=True, method="SLSQP", bounds=[(0, 1)] * n_assets,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: ones},
        options={"ftol": 1e-12, "maxiter": 200},
    )
    weights = result.x
    if not np.isfinite(weights).all() or weights.min() < -1e-8 or abs(weights.sum() - 1) > 1e-8:
        raise RuntimeError(f"Infeasible optimisation result: {result.message}")
    weights = np.maximum(weights, 0)
    weights /= weights.sum()
    # The linearisation gives an upper bound on the objective gap over the simplex.
    _, gradient = loss(weights)
    if weights @ gradient - gradient.min() > 2e-5:
        raise RuntimeError(f"Optimisation did not reach the gap tolerance: {result.message}")
    return weights


def rebalance_indices(index, start, lag):
    month = index.to_period("M")
    signal_indices = np.flatnonzero(month[:-1] != month[1:])
    return [(i, i + lag) for i in signal_indices if start <= i + lag < len(index)]


def build_targets(returns, schedule, model):
    targets = {}
    values = returns.to_numpy()
    for signal_index, fill_index in schedule:
        start = signal_index + 1 - model.lookback_sessions
        if start < 0:
            raise ValueError(f"Insufficient warm-up history for {returns.index[signal_index]:%Y-%m-%d}")
        try:
            weights = fit_weights(values[start:signal_index + 1], model)
        except RuntimeError as error:
            raise RuntimeError(f"{returns.index[signal_index]:%Y-%m-%d}: {error}") from error
        targets[returns.index[fill_index]] = weights
    return targets


def simulate(returns, targets, fee_rate):
    positions = np.zeros(returns.shape[1])
    daily_results = []
    for date, movement in zip(returns.index, returns.to_numpy(), strict=True):
        gross_growth = 1 + positions @ movement
        positions = positions * (1 + movement) / gross_growth
        fraction = 1.0
        turnover = 0.0
        trades = 0
        if date in targets:
            target = targets[date]
            if fee_rate:
                # Charge both buy and sell legs on their dollar amounts at this close.
                fraction = brentq(
                    lambda value: value + fee_rate * np.abs(value * target - positions).sum() - 1,
                    0.0, 1.0, xtol=1e-14,
                )
            changes = np.abs(fraction * target - positions)
            turnover = changes.sum()
            trades = int(np.count_nonzero(changes > TRADE_TOLERANCE))
            positions = target
        daily_results.append((gross_growth * fraction - 1, turnover, trades))
    return pd.DataFrame(daily_results, index=returns.index, columns=["return", "turnover", "trades"])


def metrics(daily_results, cash_returns):
    returns = daily_results["return"].to_numpy()
    excess = returns - cash_returns.loc[daily_results.index].to_numpy()
    growth = np.cumprod(np.r_[1.0, 1 + returns])
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(volatility * annualizer),
        "sharpe": float(returns.mean() * annualizer / volatility) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() * annualizer / excess_volatility) if excess_volatility > 0 else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["trades"].sum()),
    }


def median_breadth(targets):
    return float(np.median([1 / (weights @ weights) for weights in targets.values()]))


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    universe = config["universe"]
    cash_ticker = config["risk_free_ticker"]
    if (len(set(universe)) != len(universe) or len(universe) < 2
            or not all(re.fullmatch("[A-Z]{1,5}", ticker) for ticker in universe)
            or cash_ticker != "BIL" or cash_ticker in universe):
        raise ValueError("Use distinct US equity ETF tickers for the universe and BIL as the cash proxy")
    dates = config["dates"]
    start, evaluation_start, split, end = (
        pd.Timestamp(dates[key]) for key in ("price_start", "evaluation_start", "is_end", "end")
    )
    if not start < evaluation_start <= split < end:
        raise ValueError("Expected price_start < evaluation_start <= is_end < end")
    lag = config["execution"]["lag_sessions"]
    if type(lag) is not int or lag < 1:
        raise ValueError("lag_sessions must be a positive integer")
    costs = [float(config["execution"][key]) for key in ("cost_bps", "slippage_bps")]
    fee_rate = sum(costs) / 10_000
    if not all(cost >= 0 for cost in costs) or not fee_rate < 1:
        raise ValueError("Cost and slippage must be nonnegative and sum to fewer than 10,000 bps")
    official = Model(**config["model"])
    variants = [replace(official, **overrides) for overrides in config["robustness"]]
    if len(set([official, *variants])) != 1 + len(variants):
        raise ValueError("The robustness list must contain distinct alternatives to the official model")

    prices, fetched_at = load_prices([*universe, cash_ticker], start, end)
    daily_returns = prices.pct_change(fill_method=None).iloc[1:]
    cash_returns = daily_returns[cash_ticker]
    asset_returns = daily_returns[universe]
    evaluation = asset_returns.loc[evaluation_start:]
    is_history = asset_returns.loc[:split]
    is_evaluation = evaluation.loc[:split]
    first_evaluation = asset_returns.index.searchsorted(evaluation_start)
    is_schedule = rebalance_indices(is_history.index, first_evaluation, lag)
    if len(is_evaluation) < 2 or len(evaluation) - len(is_evaluation) < 2 or not is_schedule:
        raise ValueError("Need at least one executable IS signal and two return sessions per period")

    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return adjusted close")
    print("ROBUSTNESS: in-sample only, one parameter changed at a time")
    for overrides, model in zip(config["robustness"], variants, strict=True):
        targets = build_targets(is_history, is_schedule, model)
        daily_results = simulate(is_evaluation, targets, fee_rate)
        summary = metrics(daily_results, cash_returns)
        summary["median_effective_breadth"] = median_breadth(targets)
        print(f"ROBUSTNESS_IS {json.dumps(overrides)}: {json.dumps(summary, allow_nan=False)}")
    schedule = rebalance_indices(asset_returns.index, first_evaluation, lag)
    targets = build_targets(asset_returns, schedule, official)
    daily_results = simulate(evaluation, targets, fee_rate)
    equal_weights = np.full(len(universe), 1 / len(universe))
    comparison = simulate(evaluation, dict.fromkeys(targets, equal_weights), fee_rate)
    is_targets = {date: weights for date, weights in targets.items() if date <= split}
    print(f"PARAM_COMBOS_TRIED: {1 + len(variants)}")
    print("BENCHMARK: same sector ETFs, equal-weight monthly with the same execution and costs")
    print(f"OFFICIAL_IS_MEDIAN_EFFECTIVE_BREADTH: {median_breadth(is_targets):.6f}")

    results = {
        "strategy": config["strategy"],
        "data_range": {
            "start": evaluation.index[0].date().isoformat(),
            "end": evaluation.index[-1].date().isoformat(),
        },
    }
    in_sample = evaluation.index <= split
    for period, selected in (("is", in_sample), ("oos", ~in_sample), ("full", slice(None))):
        results[period] = metrics(daily_results.loc[selected], cash_returns)
        benchmark = metrics(comparison.loc[selected], cash_returns)
        print(f"{period.upper()}: {json.dumps(results[period], allow_nan=False)}")
        print(f"BENCHMARK_{period.upper()}: {json.dumps(benchmark, allow_nan=False)}")
    results["benchmark"] = benchmark
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)