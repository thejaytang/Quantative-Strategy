"""532_robust_enhanced_index_tracking_portfolio
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/532_robust_enhanced_index_tracking_portfolio.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit fb92b5054b1596d6a39ea87072bbd5a288327fb7 on 2026-11-27 09:42 Europe/Oslo, exit 0, output sha256 c7c9ec7a10bccd4d05611bca1c9c4a945cf84838739f3a2a37842a2ba69bf71d
Data source: 通过 yfinance 下载 Yahoo Finance 日频收盘价，命中有效缓存时读取本地 CSV。固定股票池为 AAPL、NVDA、MSFT、AVGO、ORCL、JNJ、ABBV、LLY、UNH、ABT、BRK-B、JPM、V、BAC、MA、AMZN、TSLA、HD、MCD、BKNG、WMT、COST、KO、PG、PM；SPY 为基准，BIL 日收益用于计算超额收益指标。
Data fetched: 2026-11-27T08:42:08+00:00
Sample period: 请求价格区间为 2009-11-01 至 2024-12-31，记录的价格区间为 2009-11-02 至 2024-12-31。配置评估起点为 2015-01-01，实际评估区间为 2015-01-02 至 2024-12-31，之前的价格用于预热。 data_range (from output): {'start': '2015-01-02', 'end': '2024-12-31'}
IS/OOS split: IS 为评估起点至 2019-12-31（含），OOS 为此日之后至 2024-12-31 的交易日。默认组连续运行一次后按日期切分，OOS 继承期初持仓并继续滚动估计；五个变体仅运行 IS。README 记载论文在策略编写前已公开该 OOS 时段的实证结果。
Costs and slippage: 默认每笔买入或卖出按成交金额扣 5 bps 成本及 5 bps 固定滑点，合计 10 bps；滑点通过比例费用建模。每次调仓先按当日收盘价计入旧仓收益，再对买卖双边金额之和（含从现金建仓）扣除费用，求得扣费后可投资金额。SPY 基准首次建仓使用相同费率。IS 高摩擦变体各扣 10 bps，合计 20 bps。
Signal lag: 在月末交易日收盘 t，以截至该收盘的 60 个月滚动月度收益生成股票目标权重。按配置在下一交易日 t+1 的收盘模拟成交；旧仓获得截至 t+1 收盘的价格变动，新仓自其后的收盘到收盘区间起计入价格收益。
Price adjustment: yfinance auto_adjust=True adjusted-close total-return proxy
Survivorship bias: 沿用 README 所述论文事后选定的 25 只股票，使用固定名单，缺少历史时点成分和退市股票，存在幸存者偏差与成分选择偏差。上市前缺值保留；每次仅纳入滚动窗口月收益完整的股票。
Parameter combinations tried: 6
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.21063817181840716, ann_vol=0.15072600590943344, sharpe=1.3441363891019988, sharpe_excess=1.2847807367720698, max_drawdown=-0.19471995339526282, total_return=1.5966473489993103, turnover=1.2210377879040515, n_trades=1371
    oos: cagr=0.2755154931593671, ann_vol=0.21181254184440457, sharpe=1.255784042476981, sharpe_excess=1.1461922852302215, max_drawdown=-0.30493886900900724, total_return=2.3696897367434517, turnover=1.081810324767444, n_trades=1388
    full: cagr=0.24265350953695464, ann_vol=0.1837987852562555, sharpe=1.2747285536143536, sharpe_excess=1.187326651625742, max_drawdown=-0.3049388690090076, total_return=7.74989592186507, turnover=1.1514240563357476, n_trades=2759
    benchmark SPY full: cagr=0.13014544073156809, ann_vol=0.17615749375574374, sharpe=0.7831374747702787, sharpe_excess=0.6920570539949203, max_drawdown=-0.3371727370591737, total_return=2.3923446695528723, turnover=0.10005892358833536, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 532_robust_enhanced_index_tracking_portfolio\nuniverse:\n  - AAPL\n  - NVDA\n  - MSFT\n  - AVGO\n  - ORCL\n  - JNJ\n  - ABBV\n  - LLY\n  - UNH\n  - ABT\n  - BRK-B\n  - JPM\n  - V\n  - BAC\n  - MA\n  - AMZN\n  - TSLA\n  - HD\n  - MCD\n  - BKNG\n  - WMT\n  - COST\n  - KO\n  - PG\n  - PM\nbenchmark: SPY\nrisk_free: BIL\ndates:\n  data_start: '2009-11-01'\n  start: '2015-01-01'\n  is_end: '2019-12-31'\n  end: '2024-12-31'\nexecution_lag_sessions: 1\nmodel:\n  lookback_months: 60\n  kappa: 5.0\n  gamma1: 0.22\n  gamma2: 0.025\n  alpha: -0.04\n  worst_return_floor: -0.01\n  max_weight: 0.15\n  cost_bps: 5.0\n  slippage_bps: 5.0\nrobustness:\n  kappa_4: {kappa: 4.0}\n  kappa_6: {kappa: 6.0}\n  gamma1_018: {gamma1: 0.18}\n  gamma1_026: {gamma1: 0.26}\n  higher_friction: {cost_bps: 10.0, slippage_bps: 10.0}\n"
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
"""Monthly robust enhanced index tracking with public adjusted close prices."""

import hashlib
import json
import math
import signal
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, minimize
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
PRICE_ADJUSTMENT = "yfinance auto_adjust=True adjusted-close total-return proxy"


@dataclass(frozen=True)
class Model:
    lookback_months: int
    kappa: float
    gamma1: float
    gamma2: float
    alpha: float
    worst_return_floor: float
    max_weight: float
    cost_bps: float
    slippage_bps: float

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000

    def validate(self, n_stocks):
        if not all(math.isfinite(value) for value in vars(self).values()):
            raise ValueError("Model parameters must be finite")
        if not isinstance(self.lookback_months, int) or self.lookback_months < 2:
            raise ValueError("lookback_months must be an integer of at least two")
        if self.kappa <= 0 or min(self.gamma1, self.gamma2) < 0:
            raise ValueError("kappa must be positive and uncertainty radii nonnegative")
        if not 1 / n_stocks <= self.max_weight <= 1:
            raise ValueError("max_weight must allow full investment across the universe")
        if min(self.cost_bps, self.slippage_bps) < 0 or self.friction >= 1:
            raise ValueError("Costs and slippage must be nonnegative and total below 10,000 bps")


class Rebalance(NamedTuple):
    weights: np.ndarray
    floor_miss: bool


def check_prices(prices, tickers, start, end):
    if list(prices.columns) != tickers or not prices.index.is_unique:
        raise ValueError("Price panel must contain the requested tickers and unique dates")
    if prices.empty or not prices.index.is_monotonic_increasing:
        raise ValueError("Price panel must contain dates in ascending order")
    one_week = pd.Timedelta(days=7)
    if not (start <= prices.index[0] <= start + one_week
            and end - one_week <= prices.index[-1] <= end):
        raise ValueError("Price panel does not cover the requested date range")
    absent = prices.columns[prices.isna().all()]
    if len(absent):
        raise ValueError(f"Missing ticker history: {', '.join(absent)}")
    if prices[tickers[-2:]].isna().any().any():
        raise ValueError("Missing daily closes for the benchmark or risk-free proxy")
    values = prices.to_numpy()
    if np.isinf(values).any() or (values <= 0).any():
        raise ValueError("Adjusted closes must be positive and finite where present")


def load_prices(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    request = f"v1|{start.date()}|{end.date()}|{','.join(tickers)}|{PRICE_ADJUSTMENT}"
    key = hashlib.sha256(request.encode()).hexdigest()[:16]
    cache_path = cache_dir / f"adjusted_close_{key}.csv"

    if cache_path.exists():
        try:
            with cache_path.open() as source:
                fetched_at = source.readline().strip()
                datetime.fromisoformat(fetched_at)
                prices = pd.read_csv(source, index_col="Date", parse_dates=["Date"])
            check_prices(prices, tickers, start, end)
            return prices, fetched_at
        except (OSError, ValueError) as error:
            print(f"Refreshing price cache after read error: {error}", flush=True)

    with tempfile.TemporaryDirectory(dir=cache_dir) as session_dir:
        yf.set_tz_cache_location(session_dir)
        for attempt in range(2):
            raw = yf.download(
                tickers, start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d", auto_adjust=True, ignore_tz=True,
                multi_level_index=True, progress=False, threads=4, timeout=12,
            )
            try:
                prices = raw["Close"].reindex(columns=tickers).sort_index()
                check_prices(prices, tickers, start, end)
                break
            except (KeyError, ValueError):
                if attempt == 1:
                    raise

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with tempfile.NamedTemporaryFile(mode="w", dir=cache_dir, delete=False) as output:
        temp_path = Path(output.name)
        try:
            output.write(f"{fetched_at}\n")
            prices.to_csv(output, index_label="Date")
            output.close()
            temp_path.replace(cache_path)
        finally:
            temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def solve_m_tsv(window, model):
    """Solve the convex objective in Eq. (4.8) with the configured weight cap."""
    mean = window.mean(axis=0)
    covariance = np.cov(window, rowvar=False)
    stock_cov = covariance[:-1, :-1]
    cross_cov = covariance[:-1, -1]
    stock_mean_loss = -mean[:-1]
    n_stocks = len(stock_mean_loss)
    mean_radius = math.sqrt(model.gamma1)
    upper_stock_cov = stock_cov + model.gamma2 * np.eye(n_stocks)
    upper_benchmark_var = covariance[-1, -1] + model.gamma2

    def worst_loss(weights):
        cov_times_weights = stock_cov @ weights
        volatility = math.sqrt(max(weights @ cov_times_weights, 0.0))
        gradient = stock_mean_loss
        if volatility > 0:
            gradient = gradient + mean_radius * cov_times_weights / volatility
        return stock_mean_loss @ weights + mean_radius * volatility, gradient

    def objective(weights):
        loss, loss_gradient = worst_loss(weights)
        shortfall = max(loss + mean[-1] - model.alpha, 0.0)
        tracking_gradient_half = upper_stock_cov @ weights - cross_cov
        tracking_variance = weights @ (tracking_gradient_half - cross_cov) + upper_benchmark_var
        value = loss + model.kappa * (shortfall ** 2 + tracking_variance)
        gradient = loss_gradient + 2 * model.kappa * (shortfall * loss_gradient + tracking_gradient_half)
        return 100 * value, 100 * gradient

    def floor_gap(weights):
        return 100 * (-model.worst_return_floor - worst_loss(weights)[0])

    def scaled_loss(weights):
        loss, gradient = worst_loss(weights)
        return 100 * loss, 100 * gradient

    full_investment = {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)}
    floor = {"type": "ineq", "fun": floor_gap, "jac": lambda w: -100 * worst_loss(w)[1]}
    bounds = [(0, model.max_weight)] * n_stocks
    options = {"ftol": 1e-11, "maxiter": 250}
    initial = np.full(n_stocks, 1 / n_stocks)

    if floor_gap(initial) < 0:
        feasible = minimize(scaled_loss, initial, method="SLSQP", jac=True,
                            bounds=bounds, constraints=[full_investment], options=options)
        if not feasible.success:
            raise RuntimeError(f"Worst-case return feasibility solve failed: {feasible.message}")
        initial = feasible.x
        if floor_gap(initial) < -1e-7:
            return Rebalance(initial, True)

    solution = minimize(objective, initial, method="SLSQP", jac=True, bounds=bounds,
                        constraints=[full_investment, floor], options=options)
    if not solution.success or abs(solution.x.sum() - 1) > 1e-8 or floor_gap(solution.x) < -1e-7:
        raise RuntimeError(f"M-TSV solve failed: {solution.message}")
    return Rebalance(solution.x, False)


def target_schedule(prices, start, lag, model):
    # prices includes the benchmark as the last column.
    months = prices.index.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])
    monthly_returns = prices.iloc[month_ends].pct_change(fill_method=None).to_numpy()
    first_session = prices.index.searchsorted(start)
    targets = {}
    for month, signal_session in enumerate(month_ends):
        execution = signal_session + lag
        if execution < first_session or execution >= len(prices):
            continue
        if month < model.lookback_months:
            raise ValueError("Insufficient month-end history before the evaluation period")
        window = monthly_returns[month - model.lookback_months + 1:month + 1]
        eligible = np.isfinite(window).all(axis=0)
        if not eligible[-1] or eligible[:-1].sum() * model.max_weight < 1 - 1e-10:
            raise ValueError(f"Insufficient eligible assets at {prices.index[signal_session].date()}")
        try:
            target = solve_m_tsv(window[:, eligible], model)
        except RuntimeError as error:
            raise RuntimeError(f"Signal on {prices.index[signal_session].date()}: {error}") from error
        weights = np.zeros(prices.shape[1] - 1)
        weights[eligible[:-1]] = target.weights
        targets[execution] = Rebalance(weights, target.floor_miss)
    return targets


def simulate(prices, start, targets, friction):
    first = prices.index.searchsorted(start)
    closes = prices.to_numpy()
    price_relatives = prices.pct_change(fill_method=None).to_numpy() + 1
    positions = np.zeros(prices.shape[1])
    daily_results = []
    for session in range(first, len(prices)):
        active = positions > 0
        changes = price_relatives[session, active]
        if not np.isfinite(changes).all():
            missing = prices.columns[active][~np.isfinite(changes)]
            raise ValueError(f"Missing close for open positions on {prices.index[session].date()}: {list(missing)}")
        marked = positions.copy()
        marked[active] *= changes
        growth = 1 + (marked - positions).sum()
        positions = marked / growth

        turnover = 0.0
        trades = 0
        floor_miss = False
        if session in targets:
            weights, floor_miss = targets[session]
            if not np.isfinite(closes[session, weights > 0]).all():
                raise ValueError(f"Missing close for a target on {prices.index[session].date()}")

            # Fee is charged on actual dollars bought and sold, including entry from cash.
            def turnover_at(scale):
                return np.abs(scale * weights - positions).sum()

            after_cost = brentq(lambda scale: scale + friction * turnover_at(scale) - 1, 0, 1)
            turnover = turnover_at(after_cost)
            trades = np.count_nonzero(np.abs(after_cost * weights - positions) > 1e-7)
            growth *= after_cost
            positions = weights.copy()
        daily_results.append((growth - 1, turnover, trades, floor_miss))

    return pd.DataFrame(daily_results, index=prices.index[first:],
                        columns=["return", "turnover", "n_trades", "floor_miss"])


def metrics(results, risk_free):
    returns = results["return"]
    excess = returns - risk_free.loc[returns.index]
    years = len(returns) / SESSIONS_PER_YEAR
    compounded = np.r_[1.0, (1 + returns.to_numpy()).cumprod()]

    def sharpe(series):
        std = series.std(ddof=1)
        return float(math.sqrt(SESSIONS_PER_YEAR) * series.mean() / std) if std > 0 else None

    def annual_vol(series):
        return float(math.sqrt(SESSIONS_PER_YEAR) * series.std(ddof=1)) if len(series) > 1 else None

    return {
        "cagr": float(compounded[-1] ** (1 / years) - 1),
        "ann_vol": annual_vol(returns),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(excess),
        "max_drawdown": float((compounded / np.maximum.accumulate(compounded) - 1).min()),
        "total_return": float(compounded[-1] - 1),
        "turnover": float(results["turnover"].sum() / years),
        "n_trades": int(results["n_trades"].sum()),
    }


def periods(results, is_end):
    return {"is": results.loc[:is_end], "oos": results.loc[results.index > is_end], "full": results}


def main():
    with (ROOT / "config.yaml").open() as source:
        config = yaml.safe_load(source)
    stocks = config["universe"]
    benchmark, risk_free_ticker = config["benchmark"], config["risk_free"]
    tickers = [*stocks, benchmark, risk_free_ticker]
    if not stocks or len(set(tickers)) != len(tickers):
        raise ValueError("Stocks, benchmark and risk-free proxy must be distinct")
    dates = config["dates"]
    data_start, start, is_end, end = (pd.Timestamp(dates[key]) for key in ("data_start", "start", "is_end", "end"))
    if not data_start < start <= is_end < end < pd.Timestamp.now(tz="UTC").tz_localize(None).normalize():
        raise ValueError("Require ordered dates with a completed sample end date")
    lag = config["execution_lag_sessions"]
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("execution_lag_sessions must be an integer of at least one")

    official = Model(**config["model"])
    variants = {name: replace(official, **changes) for name, changes in config["robustness"].items()}
    combinations = {official, *variants.values()}
    for model in combinations:
        model.validate(len(stocks))

    prices, fetched_at = load_prices(tickers, data_start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"PRICE_DATA_RANGE: {prices.index[0].date()} to {prices.index[-1].date()}")
    first_session = prices.index.searchsorted(start)
    risk_free = prices[risk_free_ticker].pct_change(fill_method=None).iloc[first_session:].copy()
    risk_free.iloc[0] = 0.0
    if risk_free.index[0] > is_end or risk_free.index[-1] <= is_end:
        raise ValueError("Both evaluation periods must contain sessions")

    model_prices = prices[[*stocks, benchmark]]
    official_targets = target_schedule(model_prices, start, lag, official)
    official_results = simulate(prices[stocks], start, official_targets, official.friction)
    benchmark_results = simulate(prices[[benchmark]], start, {first_session: Rebalance(np.ones(1), False)},
                                 official.friction)
    official_periods = periods(official_results, is_end)
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": str(official_results.index[0].date()),
                       "end": str(official_results.index[-1].date())},
        **{name: metrics(sample, risk_free) for name, sample in official_periods.items()},
        "benchmark": {"ticker": benchmark, **{
            name: metrics(sample, risk_free)
            for name, sample in periods(benchmark_results, is_end).items()
        }},
    }

    for name in ("is", "oos", "full"):
        print(f"{name.upper()}: {json.dumps(result[name], allow_nan=False)}")
        print(f"BENCHMARK {benchmark} {name.upper()}: {json.dumps(result['benchmark'][name], allow_nan=False)}")
        print(f"RETURN_FLOOR_MISSES {name.upper()}: {int(official_periods[name]['floor_miss'].sum())}")

    print("ROBUSTNESS (IS only)")
    print(f"  default: {json.dumps(result['is'], allow_nan=False)}")
    is_model_prices = model_prices.loc[:is_end]
    is_stock_prices = prices.loc[:is_end, stocks]
    is_risk_free = risk_free.loc[:is_end]
    for name, model in variants.items():
        targets = target_schedule(is_model_prices, start, lag, model)
        sample = simulate(is_stock_prices, start, targets, model.friction)
        print(f"  {name}: {json.dumps(metrics(sample, is_risk_free), allow_nan=False)}"
              f"  floor_misses={int(sample['floor_miss'].sum())}")

    print(f"PARAM_COMBOS_TRIED: {len(combinations)}")
    print(f"RESULTS_JSON: {json.dumps(result, allow_nan=False)}")


if __name__ == "__main__":
    signal.alarm(840)
    with threadpool_limits(limits=1, user_api="blas"):
        main()