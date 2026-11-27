"""719_an_analytic_solution_for_asset_allocatio
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/719_an_analytic_solution_for_asset_allocatio.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 9532cd9f815cc2d43ceb143b2f1416acf34e9864 on 2026-11-27 20:03 Europe/Oslo, exit 0, output sha256 4a207f472806f3713dc210c0e99d7e1c3d6a28a25764de85ec6b3ed1b2688128
Data source: 通过 yfinance 获取 Yahoo Finance 免费日频开盘价、收盘价及公司行动（股息、资本利得分配、拆股）。风险标的为 SPY、TLT、GLD、IWM、EEM，储备标的为 BIL，买入持有基准为 SPY。读取本地行情快照，缺少快照时下载。
Data fetched: 2026-11-27T19:03:10+00:00
Sample period: 实际预热数据始于 2009-01-02，配置的取数起点为 2009-01-01。实际评估期为 2011-01-03 至 2025-12-31（含首尾）。   data_range (from output): {'start': '2011-01-03', 'end': '2025-12-31', 'warmup_start': '2009-01-02', 'is_end': '2018-12-31', 'oos_start': '2019-01-02'}
IS/OOS split: IS 为 2011-01-03 至 2018-12-31，OOS 为 2019-01-02 至 2025-12-31，均含首尾。正式配置连续模拟后按日期切分，OOS 承接 IS 的仓位，继续使用截至各信号日的滚动窗口。敏感性变体只运行于 IS 行情切片，正式配置预先指定。
Costs and slippage: 每笔买卖按单边 5 bps 成本及 5 bps 固定滑点建模，合计 10 bps。按各标的开盘成交名义金额的绝对值计费并汇总，在开盘资金方程中扣除，仓位用扣费后资金计算，成交价格仍取开盘价。SPY 基准首次买入按同一费率。IS 三倍费率变体将两项各提高至 15 bps。
Signal lag: 在月末最后一个交易日 t 的收盘后，以截至 t（含）的收盘总收益估计目标权重，按配置滞后 1 个交易日，于 t+1 开盘按开盘价模拟成交。首笔信号为 2010-12-31 收盘，成交为 2011-01-03 开盘。
Price adjustment: yfinance split-adjusted OHLC (auto_adjust=False); forward total-return units with ex-date distributions reinvested at the open
Survivorship bias: 使用固定名单。README 说明名单事后选自仍有完整历史的基金，存在存续偏差和选择偏差。
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.05920567795585607, ann_vol=0.09667631370362369, sharpe=0.6434911832237671, sharpe_excess=0.6153502331404792, max_drawdown=-0.17808286901660308, total_return=0.582872334962155, turnover=5.047334669523483, n_trades=318
    oos: cagr=0.11967390914486242, ann_vol=0.11070370665574232, sharpe=1.0767503202274464, sharpe_excess=0.8490829982151086, max_drawdown=-0.1514764280130928, total_return=1.202224942206319, turnover=4.281090375456174, n_trades=247
    full: cagr=0.08700243276489816, ann_vol=0.10345992648625028, sharpe=0.8583189103058212, sharpe_excess=0.7306550203408048, max_drawdown=-0.17808286901660308, total_return=2.4858409363820413, turnover=4.689808169640539, n_trades=565
    benchmark: cagr=0.13911926160884414, ann_vol=0.1716410140978022, sharpe=0.8451202831328385, sharpe_excess=0.7677281286390707, max_drawdown=-0.3372273754868048, total_return=6.026549094261065, turnover=0.06674131806680801, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "719_ an_analytic_solution_for_asset_allocatio"\nrisky_symbols: [SPY, TLT, GLD, IWM, EEM]\nreserve_symbol: BIL\nbenchmark_symbol: SPY\n\ndata_start: \'2009-01-01\'\nbacktest_start: \'2011-01-03\'\nis_end: \'2018-12-31\'\nbacktest_end: \'2025-12-31\'\nsessions_per_year: 252\n\nlookback_sessions: 252\nrisk_aversion: 15.0\ncovariance_diagonal_shrinkage: 0.10\nsignal_lag_sessions: 1\ncost_bps: 5.0\nslippage_bps: 5.0\n\nrobustness:\n  lookback_sessions: [189, 315]\n  risk_aversion: [12.0, 18.0]\n  cost_multiplier: 3.0\n'
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
"""Cross-asset allocation by elliptical Laplace exponential utility.

Paper: Graham L. Giller (2024), https://arxiv.org/abs/2411.08967
Related research and implementations consulted, with no source code copied:
  https://medium.gillerinvestments.com/optimal-asset-allocation-with-a-multivariate-laplace-distribution-cbb207b05d81
  https://web.stanford.edu/~boyd/papers/exp_util_gm_ret_portfolio.html
  https://github.com/cvxgrp/exp_util_gm_portfolio_opt (no license listed)
  https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/pypfopt/efficient_frontier/efficient_frontier.py (MIT)
  https://pypi.org/project/pyportfolioopt/ (MIT)
  https://paperswithbacktest.com/strategies/an-analytic-solution-for-asset-allocation-with-a-multivariate-laplace-distribution
"""

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, minimize


ROOT = Path(__file__).resolve().parent
BPS = 10_000
PRICE_FIELDS = ("Open", "Close", "Dividends", "Capital Gains", "Stock Splits")
ADJUSTMENT = (
    "yfinance split-adjusted OHLC (auto_adjust=False); forward total-return units "
    "with ex-date distributions reinvested at the open"
)


@dataclass(frozen=True)
class Trial:
    lookback: int
    risk_aversion: float
    shrinkage: float
    cost_bps: float
    slippage_bps: float
    gaussian: bool = False

    @property
    def fill_rate(self):
        return (self.cost_bps + self.slippage_bps) / BPS


def load_prices(config, symbols):
    cache = ROOT / "data_cache"
    cache.mkdir(exist_ok=True)
    start = pd.Timestamp(config["data_start"])
    end = pd.Timestamp(config["backtest_end"])
    key = json.dumps([start.date().isoformat(), end.date().isoformat(), symbols])
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    snapshot = cache / f"yahoo_ohlc_actions_v1_{digest}.csv"
    expected = pd.MultiIndex.from_product([PRICE_FIELDS, symbols])
    if snapshot.exists():
        with snapshot.open() as source:
            fetched_at = source.readline().strip().removeprefix("# fetched_at=")
            if datetime.fromisoformat(fetched_at).tzinfo is None:
                raise ValueError("Cache timestamp must include a time zone.")
            prices = pd.read_csv(
                source, header=[0, 1], index_col=0, parse_dates=True,
            ).reindex(columns=expected)
    else:
        # Keep Yahoo's transient library cache separate from the price snapshot.
        with TemporaryDirectory(prefix="yf_", dir=cache) as vendor_cache:
            yf.set_tz_cache_location(vendor_cache)
            downloaded = yf.download(
                list(symbols), start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d", auto_adjust=False, actions=True, repair=False,
                ignore_tz=True, progress=False, threads=False, timeout=15,
            )
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        prices = downloaded.reindex(columns=expected).copy()
        for symbol in symbols:
            if ("Capital Gains", symbol) not in downloaded.columns:
                prices["Capital Gains", symbol] = 0.0

    if prices.empty or prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Yahoo price history is empty or has invalid session dates.")
    if prices.index[0] < start or prices.index[-1] > end:
        raise ValueError("Price history extends outside the configured dates.")
    if (prices.index[0] - start).days > 7 or (end - prices.index[-1]).days > 7:
        raise ValueError("Price history does not cover the configured dates.")
    if not np.isfinite(prices.to_numpy(dtype=float)).all():
        missing = prices.isna().any(axis=1)
        raise ValueError(f"Incomplete Yahoo price or action history on {list(prices.index[missing][:5])}")
    if (prices[["Open", "Close"]] <= 0).any().any():
        raise ValueError("Yahoo returned a nonpositive open or close.")
    if (prices[["Dividends", "Capital Gains", "Stock Splits"]] < 0).any().any():
        raise ValueError("Yahoo returned a negative distribution or split ratio.")

    if not snapshot.exists():
        with NamedTemporaryFile(mode="w", dir=cache, delete=False) as output:
            temporary = Path(output.name)
            try:
                output.write(f"# fetched_at={fetched_at}\n")
                prices.to_csv(output, index_label="Date")
                output.flush()
                temporary.replace(snapshot)
            finally:
                temporary.unlink(missing_ok=True)
    return prices, fetched_at


def total_return_prices(prices):
    opens = prices["Open"].to_numpy(dtype=float)
    closes = prices["Close"].to_numpy(dtype=float)
    distributions = (prices["Dividends"] + prices["Capital Gains"]).to_numpy(dtype=float)
    # Previous positions receive the distribution before any ex-date open fills.
    # Neither today's close nor any later distribution can change today's opening value.
    factors = np.cumprod(1 + distributions / opens, axis=0)
    return opens * factors, closes * factors


def risky_weights(window, trial):
    mean = window.mean(axis=0)
    covariance = np.atleast_2d(np.cov(window, rowvar=False, ddof=1))
    diagonal = np.diag(np.diag(covariance))
    covariance = (1 - trial.shrinkage) * covariance + trial.shrinkage * diagonal
    size = len(mean)
    risk_aversion = trial.risk_aversion
    # Giller's scatter is 2V/(n+1). The log-MGF term uses q = lambda^2 w'Vw/(n+1).
    shape = (size + 1) / 2
    scaled_covariance = (risk_aversion**2 / (size + 1)) * covariance
    domain_max = 1 - 1e-6

    def objective(weights):
        product = scaled_covariance @ weights
        q = weights @ product
        # SLSQP can probe outside the finite-MGF region. Continue the penalty
        # with its tangent there, and independently constrain feasible weights.
        bounded_q = q if trial.gaussian else min(q, domain_max)
        slope = 1.0 if trial.gaussian else 1 / (1 - bounded_q)
        penalty = q if trial.gaussian else -np.log1p(-bounded_q) + (q - bounded_q) * slope
        value = -mean @ weights + (shape / risk_aversion) * penalty
        gradient = -mean + (2 * shape / risk_aversion) * slope * product
        return BPS * value, BPS * gradient

    constraints = [
        {"type": "ineq", "fun": lambda w: 1 - w.sum(), "jac": lambda w: -np.ones(size)},
    ]
    # The maximum of this convex quadratic over the nonnegative budget simplex
    # is attained at a vertex, so most windows need no additional constraint.
    if not trial.gaussian and np.diag(scaled_covariance).max() >= domain_max:
        constraints.append({
            "type": "ineq",
            "fun": lambda w: domain_max - w @ scaled_covariance @ w,
            "jac": lambda w: -2 * scaled_covariance @ w,
        })
    result = minimize(
        objective, np.zeros(size), jac=True, method="SLSQP",
        bounds=[(0.0, 1.0)] * size, constraints=constraints,
        options={"ftol": 1e-11, "maxiter": 200},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, 1.0)
    if weights.sum() > 1 + 1e-8:
        raise RuntimeError("Risky weights exceed the budget.")
    weights /= max(1.0, weights.sum())
    if not trial.gaussian and weights @ scaled_covariance @ weights >= 1:
        raise RuntimeError("Portfolio lies outside the Laplace moment domain.")
    return weights


def fill_at_open(positions, cash, opening, target, rate):
    current = positions * opening
    value = cash + current.sum()
    current_weights = current / value

    def spending(fraction):
        return fraction + rate * np.abs(fraction * target - current_weights).sum() - 1

    fraction = brentq(spending, 0.0, 1.0, xtol=1e-14)
    # A tiny buffer keeps root-finder roundoff on the funded side of the budget.
    fraction = max(0.0, fraction - 1e-12)
    cash = -value * spending(fraction)
    if cash < 0:
        raise RuntimeError("Opening fills require negative cash.")
    desired = fraction * target
    turnover = np.abs(desired - current_weights)
    trade_count = int(np.count_nonzero(turnover > 1e-10))
    return value * desired / opening, cash, turnover.sum(), trade_count


def simulate(prices, config, trial, benchmark=False):
    sessions = prices.index
    first = sessions.searchsorted(pd.Timestamp(config["backtest_start"]))
    if first == 0 or first >= len(sessions):
        raise ValueError("Backtest requires sessions before and after its start boundary.")
    opens, closes = total_return_prices(prices)
    returns = np.empty_like(closes)
    returns[0] = np.nan
    returns[1:] = closes[1:] / closes[:-1] - 1
    reserve_returns = returns[:, -1].copy()
    reserve_returns[first] = closes[first, -1] / opens[first, -1] - 1
    excess = returns[:, :-1] - returns[:, -1:]
    months = sessions.to_period("M")
    lag = config["signal_lag_sessions"]
    benchmark_target = np.zeros(opens.shape[1])
    if benchmark:
        benchmark_target[config["risky_symbols"].index(config["benchmark_symbol"])] = 1.0
    positions = np.zeros(opens.shape[1])
    cash = previous_value = 1.0
    results = []
    first_signal = first_fill = None
    for i in range(first, len(sessions)):
        turnover = trades = 0
        target = None
        signal = i - lag
        if benchmark and i == first:
            target = benchmark_target
        if not benchmark and signal >= 0 and months[signal] != months[signal + 1]:
            window_start = signal - trial.lookback + 1
            if window_start <= 0 or sessions[signal] >= sessions[i]:
                raise ValueError("Insufficient trailing history or a noncausal signal date.")
            weights = risky_weights(excess[window_start:signal + 1], trial)
            target = np.append(weights, max(0.0, 1 - weights.sum()))
            if first_signal is None:
                first_signal, first_fill = sessions[signal], sessions[i]

        if target is not None:
            positions, cash, turnover, trades = fill_at_open(
                positions, cash, opens[i], target, trial.fill_rate,
            )
        value = positions @ closes[i] + cash
        results.append((value / previous_value - 1, reserve_returns[i], turnover, trades, cash / value))
        previous_value = value

    daily_results = pd.DataFrame(
        results, index=sessions[first:],
        columns=["return", "reserve_return", "turnover", "n_trades", "cash_weight"],
    )
    return daily_results, first_signal, first_fill


def metrics(daily_results, annualization):
    if len(daily_results) < 2:
        raise ValueError("Each reported period needs at least two sessions.")
    returns = daily_results["return"].to_numpy()
    excess = returns - daily_results["reserve_return"].to_numpy()
    path = np.r_[1.0, (1 + returns).cumprod()]
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    return {
        "cagr": float(path[-1] ** (annualization / len(returns)) - 1),
        "ann_vol": float(volatility * np.sqrt(annualization)),
        "sharpe": float(returns.mean() / volatility * np.sqrt(annualization)) if volatility else None,
        "sharpe_excess": (
            float(excess.mean() / excess_volatility * np.sqrt(annualization))
            if excess_volatility else None
        ),
        "max_drawdown": float((path / np.maximum.accumulate(path) - 1).min()),
        "total_return": float(path[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * annualization),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def robustness_cases(config, official):
    sensitivity = config["robustness"]
    for lookback in sensitivity["lookback_sessions"]:
        yield f"lookback_sessions={lookback}", replace(official, lookback=lookback)
    for aversion in sensitivity["risk_aversion"]:
        yield f"risk_aversion={aversion:g}", replace(official, risk_aversion=aversion)
    yield "gaussian_objective", replace(official, gaussian=True)
    multiplier = sensitivity["cost_multiplier"]
    yield f"cost_and_slippage_x{multiplier:g}", replace(
        official, cost_bps=official.cost_bps * multiplier,
        slippage_bps=official.slippage_bps * multiplier,
    )


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    official = Trial(
        config["lookback_sessions"], config["risk_aversion"],
        config["covariance_diagonal_shrinkage"], config["cost_bps"], config["slippage_bps"],
    )
    variants = list(robustness_cases(config, official))
    for trial in {official, *(trial for _, trial in variants)}:
        if trial.lookback < 2 or not 0 < trial.risk_aversion < np.inf:
            raise ValueError("Lookback and risk aversion must be positive.")
        if not 0 <= trial.shrinkage <= 1 or not 0 <= trial.fill_rate < 1:
            raise ValueError("Invalid covariance shrinkage or execution charge.")
        if min(trial.cost_bps, trial.slippage_bps) < 0:
            raise ValueError("Execution charges must be nonnegative.")
    if not isinstance(config["signal_lag_sessions"], int) or config["signal_lag_sessions"] < 1:
        raise ValueError("Signal lag must be at least one session.")
    symbols = (*config["risky_symbols"], config["reserve_symbol"])
    if len(symbols) < 2 or len(set(symbols)) != len(symbols):
        raise ValueError("Risky symbols and reserve must be distinct.")
    if config["benchmark_symbol"] not in config["risky_symbols"]:
        raise ValueError("Benchmark must be one of the risky symbols.")
    is_end = pd.Timestamp(config["is_end"])
    if not pd.Timestamp(config["backtest_start"]) <= is_end < pd.Timestamp(config["backtest_end"]):
        raise ValueError("Invalid IS/OOS boundary.")
    annualization = config["sessions_per_year"]
    if annualization <= 0:
        raise ValueError("Sessions per year must be positive.")

    prices, fetched_at = load_prices(config, symbols)
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print(f"PRICE_ADJUSTMENT: {ADJUSTMENT}", flush=True)
    print("ROBUSTNESS (IS only; fixed official specification)", flush=True)
    tried = set()
    is_prices = prices.loc[:is_end]
    for label, trial in variants:
        if trial == official or trial in tried:
            continue
        sensitivity, _, _ = simulate(is_prices, config, trial)
        tried.add(trial)
        print(f"  {label}: {json.dumps(metrics(sensitivity, annualization), allow_nan=False)}")

    daily_results, first_signal, first_fill = simulate(prices, config, official)
    tried.add(official)
    reference, _, _ = simulate(prices, config, official, benchmark=True)
    is_results = daily_results.loc[:is_end]
    oos_results = daily_results.loc[daily_results.index > is_end]
    if not is_results["n_trades"].sum() or not oos_results["n_trades"].sum():
        raise RuntimeError("Official backtest requires fills in both periods.")
    periods = {
        "is": metrics(is_results, annualization),
        "oos": metrics(oos_results, annualization),
        "full": metrics(daily_results, annualization),
        "benchmark": metrics(reference, annualization),
    }
    print(f"PARAM_COMBOS_TRIED: {len(tried)}")
    print(f"FIRST_REBALANCE: signal_close={first_signal.date()} fill_open={first_fill.date()}")
    print(f"MIN_CASH_WEIGHT: {daily_results['cash_weight'].min():.3g}")
    print(
        "LOOK_AHEAD_CHECK: trailing month-end close signal; configured later-open sizing; "
        "ex-date distributions at the open and no delayed macro fields; "
        "split-adjusted price units and forward distribution factors; IS-only sensitivity"
    )
    for label, summary in periods.items():
        print(f"{label.upper()}: {json.dumps(summary, allow_nan=False)}")
    print("STATUS: 已回测")
    result = {
        "strategy": config["strategy"],
        "data_range": {
            "start": daily_results.index[0].date().isoformat(),
            "end": daily_results.index[-1].date().isoformat(),
            "warmup_start": prices.index[0].date().isoformat(),
            "is_end": is_results.index[-1].date().isoformat(),
            "oos_start": oos_results.index[0].date().isoformat(),
        },
        **periods,
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
