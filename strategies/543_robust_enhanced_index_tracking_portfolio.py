"""543_robust_enhanced_index_tracking_portfolio
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/543_robust_enhanced_index_tracking_portfolio.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 69a306c0280e31031e1c1674f053aa22730544ac on 2026-11-27 09:47 Europe/Oslo, exit 0, output sha256 d287f1aca9885e121346b373776d5f09f8b9036ef374346644784b890e71f4cc
Data source: 免费 Yahoo Finance 日线，由 yfinance 获取或从本地价格缓存读取。股票池为 AAPL、NVDA、MSFT、AVGO、ORCL、JNJ、ABBV、LLY、UNH、ABT、BRK-B、JPM、V、BAC、MA、AMZN、TSLA、HD、MCD、BKNG、WMT、COST、KO、PG、PM。SPY 为跟踪和买入持有基准，BIL 的日收益仅用于计算超额 Sharpe。（config.yaml:2-7；strategy.py:78-105、310-326、335-336；README.md:15）
Data fetched: 2026-11-27T08:47:50+00:00
Sample period: 日线请求区间为 2009-12-01 至 2024-12-31（含），配置的评估起点为 2015-01-01。日志记录的实际评估交易日为 2015-01-02 至 2024-12-31。2009 年末至 2014 年末的价格用于初始 60 个月滚动估计。（config.yaml:8-15；strategy.py:81-82、267-270、337-340；README.md:16；logs/run_20261127_092024.log:22）   data_range (from output): {'start': '2015-01-02', 'end': '2024-12-31'}
IS/OOS split: 样本内为 2015-01-01 至 2019-12-31，样本外为 2020-01-01 至 2024-12-31，均按实际交易日计。默认配置连续模拟全期，按 2019-12-31 切分结果。样本外沿用默认配置参数，以每月已知收益更新滚动估计，7 个稳健性变体仅运行至样本内截止日。（config.yaml:8-15、24-32；README.md:16、27；strategy.py:201-224、315-318、328-349）
Costs and slippage: 默认在每次模拟成交时，买卖两侧各按实际成交金额扣除 5 bp 成本和 5 bp 固定滑点，合计 10 bp。代码用扣费后目标头寸与漂移后头寸之差的绝对金额加总，并联立求解扣费后的组合价值。首次建仓同样收费，SPY 基准在策略首次成交的收盘建仓并扣费一次。仅样本内的高摩擦变体使用 15 bp 成本和 10 bp 固定滑点，合计 25 bp。（config.yaml:22-32；strategy.py:52-54、232-264、328-336；README.md:21、27）
Signal lag: 在月末最后一个交易日 t 收盘后，用截至 t 的滚动月收益计算目标权重。默认在下一交易日 t+1 的收盘模拟成交。成交日的收盘到收盘收益由原持仓承担，新权重在 t+1 收盘调仓后生效。（config.yaml:13-15；strategy.py:201-224、232-263、307-309；README.md:21）
Price adjustment: yfinance auto_adjust=True, dividend- and split-adjusted daily Close
Survivorship bias: 25 只股票来自论文事后选定的五个行业的大盘股，为固定名单，存在幸存者偏差和事后股票池选择偏差。股票仅在信号日已有完整的 60 个月滚动收益后才可获权重。该筛选只处理历史价格覆盖，无法重建历史时点股票池。（config.yaml:2-5、15；strategy.py:68-75、212-224；README.md:9、17）
Parameter combinations tried: 8
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.21074029101027647, ann_vol=0.15029756603590094, sharpe=1.348097852843359, sharpe_excess=1.2888643278274334, max_drawdown=-0.1935592527122909, total_return=1.597740950757772, turnover=1.2230471165179937, n_trades=1372
    oos: cagr=0.27224583453874085, ann_vol=0.21086310513124484, sharpe=1.2483044498491502, sharpe_excess=1.1382364366202002, max_drawdown=-0.3049388378533605, total_return=2.3267890185821645, turnover=1.085551144800468, n_trades=1390
    full: cagr=0.24111211900700025, ann_vol=0.1830754448863819, sharpe=1.2722546674515127, sharpe_excess=1.1846384222263628, max_drawdown=-0.3049388378533606, total_return=7.642136068102127, turnover=1.1542991306592307, n_trades=2762
    benchmark: cagr=0.13014547134358656, ann_vol=0.1761576124971584, sharpe=0.7831372216289496, sharpe_excess=0.6921821417622633, max_drawdown=-0.33717263786002827, total_return=2.3923455869719112, turnover=0.10005892358833536, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 543_robust_enhanced_index_tracking_portfolio\n# Cai and Song, section 5. The benchmark and cash proxy are separate from the universe.\nuniverse:\n  [AAPL, NVDA, MSFT, AVGO, ORCL, JNJ, ABBV, LLY, UNH, ABT, BRK-B, JPM, V,\n   BAC, MA, AMZN, TSLA, HD, MCD, BKNG, WMT, COST, KO, PG, PM]\nbenchmark: SPY\nexcess_sharpe_reference: BIL\ndates:\n  data_start: 2009-12-01\n  evaluation_start: 2015-01-01\n  is_end: 2019-12-31\n  end: 2024-12-31\nrebalance_lag_sessions: 1\nmodel:\n  lookback_months: 60\n  kappa: 5.0\n  mean_radius: 0.22\n  covariance_radius: 0.025\n  target_monthly_excess: 0.04\n  worst_monthly_return_floor: -0.01\n  max_weight: 0.10\n  cost_bps: 5.0\n  slippage_bps: 5.0\n# Each case overrides only the listed baseline parameters and runs through is_end.\nrobustness:\n  kappa_4: {kappa: 4.0}\n  kappa_6: {kappa: 6.0}\n  mean_radius_0.18: {mean_radius: 0.18}\n  mean_radius_0.26: {mean_radius: 0.26}\n  covariance_radius_0.020: {covariance_radius: 0.020}\n  covariance_radius_0.030: {covariance_radius: 0.030}\n  higher_friction: {cost_bps: 15.0, slippage_bps: 10.0}\n'
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
"""Monthly distributionally robust mean–target semivariance index tracking."""

import signal

if __name__ == "__main__":
    signal.alarm(840)

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq, minimize
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
FEASIBILITY_TOLERANCE = 1e-8
TRADE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class Model:
    lookback_months: int
    kappa: float
    mean_radius: float
    covariance_radius: float
    target_monthly_excess: float
    worst_monthly_return_floor: float
    max_weight: float
    cost_bps: float
    slippage_bps: float

    def __post_init__(self):
        if (
            type(self.lookback_months) is not int or self.lookback_months < 2
            or not np.isfinite(list(vars(self).values())).all()
            or self.kappa <= 0
            or min(self.mean_radius, self.covariance_radius, self.cost_bps, self.slippage_bps) < 0
            or not 0 < self.max_weight <= 1
            or self.friction >= 1
        ):
            raise ValueError("Invalid model parameters in config.yaml")

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def validate_prices(prices, tickers, benchmark, reference, start, end):
    if list(prices.columns) != tickers or prices.empty:
        raise RuntimeError("Adjusted close data is missing or the ticker columns differ from the request")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise RuntimeError("Adjusted close dates must be unique and ordered")
    first_session, last_session = prices.index[[0, -1]]
    if (
        not start <= first_session <= start + pd.Timedelta(days=7)
        or not end - pd.Timedelta(days=7) <= last_session <= end
    ):
        raise RuntimeError("Adjusted closes do not span the requested dates")
    for ticker in tickers:
        series = prices[ticker]
        first_close = series.first_valid_index()
        if first_close is None or (ticker in (benchmark, reference) and first_close != first_session):
            raise RuntimeError(f"Missing adjusted close history for {ticker}")
        observed = series.loc[first_close:].to_numpy()
        if not np.isfinite(observed).all() or (observed <= 0).any():
            raise RuntimeError(f"Missing or invalid adjusted closes after {ticker}'s first observation")


def load_prices(tickers, benchmark, reference, start, end):
    import yfinance as yf

    start_date = start.date().isoformat()
    end_exclusive = (end + pd.Timedelta(days=1)).date().isoformat()
    key = sha256(json.dumps([tickers, start_date, end_exclusive]).encode()).hexdigest()[:16]
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"yfinance_adjusted_close_{key}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            fetched_at = handle.readline().strip().removeprefix("# fetched_at=")
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(handle, index_col="date", parse_dates=["date"])
        validate_prices(prices, tickers, benchmark, reference, start, end)
        return prices, fetched_at

    with TemporaryDirectory(dir=CACHE, prefix=".yfinance-") as transient_cache:
        yf.set_tz_cache_location(transient_cache)
        for attempt in range(2):
            raw = yf.download(
                tickers, start=start_date, end=end_exclusive, interval="1d",
                auto_adjust=True, ignore_tz=True, multi_level_index=True,
                progress=False, threads=False, timeout=8,
            )
            try:
                if raw is None or raw.empty or "Close" not in raw.columns.get_level_values(0):
                    raise RuntimeError("Yahoo Finance returned no adjusted daily closes")
                prices = raw["Close"].reindex(columns=tickers).sort_index()
                validate_prices(prices, tickers, benchmark, reference, start, end)
                break
            except RuntimeError:
                if attempt == 1:
                    raise

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as handle:
        temporary_path = Path(handle.name)
        try:
            handle.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(handle, index_label="date")
            handle.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def optimize_weights(monthly_returns, model):
    """Solve paper equation (4.8) after substituting its tight risk bounds."""
    n_assets = monthly_returns.shape[1] - 1
    if n_assets * model.max_weight < 1:
        raise ValueError("The eligible universe is too small for the configured weight cap")
    means = -monthly_returns.mean(axis=0)
    covariance = np.cov(monthly_returns, rowvar=False, ddof=1)
    covariance += 1e-10 * np.eye(n_assets + 1)
    asset_covariance = covariance[:-1, :-1]
    robust_covariance = covariance + model.covariance_radius * np.eye(n_assets + 1)
    mean_penalty = np.sqrt(model.mean_radius)
    max_worst_loss = -model.worst_monthly_return_floor

    def worst_loss(weights):
        covariance_product = asset_covariance @ weights
        volatility = np.sqrt(weights @ covariance_product)
        loss = means[:-1] @ weights + mean_penalty * volatility
        extra_gradient = mean_penalty * covariance_product / volatility if volatility else 0.0
        return loss, means[:-1] + extra_gradient

    def objective(weights):
        loss, loss_gradient = worst_loss(weights)
        shortfall = max(loss - means[-1] + model.target_monthly_excess, 0.0)
        active = np.append(weights, -1.0)
        covariance_product = robust_covariance @ active
        semivariance = shortfall**2 + active @ covariance_product
        gradient = loss_gradient + 2 * model.kappa * (
            shortfall * loss_gradient + covariance_product[:-1]
        )
        return 100 * (loss + model.kappa * semivariance), 100 * gradient

    ones = np.ones(n_assets)
    budget = {
        "type": "eq",
        "fun": lambda weights: weights.sum() - 1,
        "jac": lambda weights: ones,
    }
    floor = {
        "type": "ineq",
        "fun": lambda weights: 100 * (max_worst_loss - worst_loss(weights)[0]),
        "jac": lambda weights: -100 * worst_loss(weights)[1],
    }

    def solve(function, initial, constraints):
        result = minimize(
            function, initial, method="SLSQP", jac=True, constraints=constraints,
            bounds=[(0.0, model.max_weight)] * n_assets,
            options={"ftol": 1e-11, "maxiter": 150},
        )
        if (
            not result.success or not np.isfinite(result.x).all()
            or abs(result.x.sum() - 1) > FEASIBILITY_TOLERANCE
        ):
            raise RuntimeError(f"Weight optimization failed: {result.message}")
        return result.x

    equal_weights = ones / n_assets
    unconstrained_floor = solve(objective, equal_weights, [budget])
    if worst_loss(unconstrained_floor)[0] <= max_worst_loss:
        return unconstrained_floor, False

    def loss_objective(weights):
        loss, gradient = worst_loss(weights)
        return 100 * loss, 100 * gradient

    best_floor = solve(loss_objective, equal_weights, [budget])
    if worst_loss(best_floor)[0] >= max_worst_loss - FEASIBILITY_TOLERANCE:
        # Use the best attainable worst-case return at or beyond the feasibility boundary.
        return best_floor, bool(worst_loss(best_floor)[0] > max_worst_loss + FEASIBILITY_TOLERANCE)

    weights = solve(objective, best_floor, [budget, floor])
    if worst_loss(weights)[0] > max_worst_loss + FEASIBILITY_TOLERANCE:
        raise RuntimeError("Optimized weights violate the worst-case return floor")
    return weights, False


def monthly_targets(prices, start, lag, model):
    month = prices.index.to_period("M")
    month_end_rows = np.flatnonzero(month[:-1] != month[1:])
    monthly_returns = prices.iloc[month_end_rows].pct_change(fill_method=None)
    n_assets = prices.shape[1] - 1
    targets = {}
    relaxed_floors = []
    for row, signal_row in enumerate(month_end_rows):
        execution_row = signal_row + lag
        if execution_row >= len(prices) or prices.index[execution_row] < start:
            continue
        window = monthly_returns.iloc[max(0, row + 1 - model.lookback_months):row + 1]
        if len(window) != model.lookback_months or window.iloc[:, -1].isna().any():
            raise ValueError("Insufficient benchmark history for the first scheduled signal")
        present = window.iloc[:, :-1].notna().all(axis=0).to_numpy()
        eligible = window.iloc[:, np.append(present, True)]
        execution_date = prices.index[execution_row]
        try:
            selected_weights, relaxed = optimize_weights(eligible.to_numpy(), model)
        except (ValueError, RuntimeError) as error:
            raise RuntimeError(f"{prices.index[signal_row].date()} signal: {error}") from error
        weights = np.zeros(n_assets)
        weights[present] = selected_weights
        targets[execution_date] = weights
        if relaxed:
            relaxed_floors.append(execution_date)
    if not targets:
        raise ValueError("No monthly signals reach execution in the requested evaluation period")
    return targets, relaxed_floors


def simulate(returns, targets, friction):
    """Mark prior weights to the close, then apply any scheduled rebalance."""
    weights = np.zeros(returns.shape[1])
    daily = []
    for date, changes in zip(returns.index, returns.to_numpy(), strict=True):
        available = np.isfinite(changes)
        if np.any((weights > 0) & ~available):
            raise RuntimeError(f"Missing daily price change for a position on {date.date()}")
        marked_weights = weights * (1 + np.where(available, changes, 0.0))
        gross_growth = 1 + (marked_weights - weights).sum()
        drifted_weights = marked_weights / gross_growth

        remaining = 1.0
        traded_fraction = 0.0
        trade_count = 0
        target = targets.get(date)
        if target is not None:
            if np.any((target > 0) & ~available):
                raise RuntimeError(f"Missing execution close or preceding close on {date.date()}")

            def cash_balance(scale):
                traded = scale * target - drifted_weights
                return scale + friction * np.abs(traded).sum() - 1

            remaining = brentq(cash_balance, 0, 1, xtol=1e-14)
            traded = np.abs(remaining * target - drifted_weights)
            traded_fraction = traded.sum()
            trade_count = int(np.count_nonzero(traded > TRADE_TOLERANCE))
            weights = target
        else:
            weights = drifted_weights
        daily.append((gross_growth * remaining - 1, traded_fraction, trade_count))
    return pd.DataFrame(daily, index=returns.index, columns=["return", "turnover", "n_trades"])


def run_model(prices, start, lag, model):
    targets, relaxed_floors = monthly_targets(prices, start, lag, model)
    daily_returns = prices.iloc[:, :-1].pct_change(fill_method=None).loc[start:]
    return simulate(daily_returns, targets, model.friction), min(targets), relaxed_floors


def metrics(daily_results, reference):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise ValueError("Metrics require at least two daily returns in each period")
    excess = returns - reference.reindex(returns.index)
    if not np.isfinite(excess).all():
        raise ValueError("Missing returns for the excess-Sharpe reference")
    wealth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    years = len(returns) / SESSIONS_PER_YEAR
    volatility = float(returns.std())
    excess_volatility = float(excess.std())
    return {
        "cagr": float(wealth[-1] ** (1 / years) - 1),
        "ann_vol": float(volatility * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": float(returns.mean() * np.sqrt(SESSIONS_PER_YEAR) / volatility) if volatility else None,
        "sharpe_excess": (
            float(excess.mean() * np.sqrt(SESSIONS_PER_YEAR) / excess_volatility)
            if excess_volatility else None
        ),
        "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
        "total_return": float(wealth[-1] - 1),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    dates = config["dates"]
    data_start = pd.Timestamp(dates["data_start"])
    start = pd.Timestamp(dates["evaluation_start"])
    is_end = pd.Timestamp(dates["is_end"])
    end = pd.Timestamp(dates["end"])
    lag = config["rebalance_lag_sessions"]
    if not data_start < start <= is_end < end or type(lag) is not int or lag < 1:
        raise ValueError("Check chronology and set rebalance_lag_sessions to a positive integer")
    benchmark = config["benchmark"]
    reference_ticker = config["excess_sharpe_reference"]
    tickers = [*config["universe"], benchmark, reference_ticker]
    if len(tickers) < 3 or len(tickers) != len(set(tickers)):
        raise ValueError("Use a nonempty universe and distinct benchmark and reference tickers")
    model = Model(**config["model"])
    robustness = [(label, replace(model, **overrides)) for label, overrides in config["robustness"].items()]
    if len({model, *(variant for _, variant in robustness)}) != 1 + len(robustness):
        raise ValueError("Every robustness case must specify a distinct parameter combination")
    prices, fetched_at = load_prices(tickers, benchmark, reference_ticker, data_start, end)
    reference = prices[reference_ticker].pct_change(fill_method=None)
    model_prices = prices.drop(columns=reference_ticker)

    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, dividend- and split-adjusted daily Close")
    print(f"BENCHMARK: {benchmark} buy-and-hold")
    print(f"EXCESS_SHARPE_REFERENCE: {reference_ticker} daily adjusted return")

    robustness_results = []
    with threadpool_limits(limits=1, user_api="blas"):
        for label, variant in robustness:
            sample, _, relaxed = run_model(model_prices.loc[:is_end], start, lag, variant)
            robustness_results.append((label, metrics(sample, reference.loc[:is_end]), len(relaxed)))
        daily_results, first_execution, relaxed = run_model(model_prices, start, lag, model)

    benchmark_returns = prices[[benchmark]].pct_change(fill_method=None).loc[daily_results.index]
    benchmark_results = simulate(benchmark_returns, {first_execution: np.ones(1)}, model.friction)
    sessions = daily_results.index
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": sessions[0].date().isoformat(), "end": sessions[-1].date().isoformat()},
    }
    benchmark_metrics = {}
    for name, period in (
        ("is", sessions[sessions <= is_end]),
        ("oos", sessions[sessions > is_end]),
        ("full", sessions),
    ):
        result[name] = metrics(daily_results.loc[period], reference)
        benchmark_metrics[name] = metrics(benchmark_results.loc[period], reference)
        print(f"{name.upper()}: {json.dumps(result[name], allow_nan=False)}")
        print(f"BENCHMARK_{name.upper()}: {json.dumps(benchmark_metrics[name], allow_nan=False)}")
    result["benchmark"] = benchmark_metrics["full"]

    relaxed_is = sum(date <= is_end for date in relaxed)
    print(f"RETURN_FLOOR_INFEASIBLE_REBALANCES: IS={relaxed_is} OOS={len(relaxed) - relaxed_is}")
    print("ROBUSTNESS (IS ONLY)")
    print(f"default: {json.dumps(result['is'], allow_nan=False)}; floor_infeasible={relaxed_is}")
    for label, row, relaxed_count in robustness_results:
        print(f"{label}: {json.dumps(row, allow_nan=False)}; floor_infeasible={relaxed_count}")
    print(f"PARAM_COMBOS_TRIED: {1 + len(robustness_results)}")
    print(f"RESULTS_JSON: {json.dumps(result, allow_nan=False)}")


if __name__ == "__main__":
    main()