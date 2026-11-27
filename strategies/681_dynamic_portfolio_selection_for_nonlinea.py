"""681_dynamic_portfolio_selection_for_nonlinea
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/681_dynamic_portfolio_selection_for_nonlinea.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 9532cd9f815cc2d43ceb143b2f1416acf34e9864 on 2026-11-27 20:00 Europe/Oslo, exit 0, output sha256 42e57d20475e6750b4ec5281f7437e473ab86b4c81fdfde9351bcc9279568801
Data source: Yahoo Finance 免费 ETF 日线，经 yfinance 读取复权开盘价和收盘价，资产为 SPY、EFA、GLD、IEF、TLT、BIL。SPY 为买入持有基准，BIL 日收益也用于计算超额夏普，作为无风险代理。
Data fetched: 2026-11-27T19:00:52.751616+00:00
Sample period: 实际回测交易日为 2010-01-04 至 2025-12-31。预热数据请求从 2008-06-01 开始。   data_range (from output): {'start': '2010-01-04', 'is_end': '2017-12-29', 'oos_start': '2018-01-02', 'end': '2025-12-31'}
IS/OOS split: 日历切分日为 2017-12-31（含于 IS）。实际 IS 为 2010-01-04 至 2017-12-29，OOS 为 2018-01-02 至 2025-12-31。默认配置连续模拟，样本内头寸延续至样本外，逐月滚动估计。六组敏感性案例仅在 IS 运行。
Costs and slippage: 默认每笔买入和卖出按成交金额计 5 bps 成本及 3 bps 固定滑点，合计 8 bps。滑点作为现金摩擦建模，卖出净收入为成交金额 × (1 − 0.0008)，买入扣款为成交金额 × (1 + 0.0008)。两项均计入开盘调仓预算并适用于初始买入和 SPY 基准。未模拟市场冲击或税费。
Signal lag: 在回测首个交易日及每月首个数据交易日 t 的收盘计算目标权重，滚动窗口包含 t 的收盘价。默认在下一交易日 t+1 的开盘模拟成交。
Price adjustment: yfinance auto_adjust total-return Open and Close, repair=False
Survivorship bias: 固定选用当前仍存续的六只 ETF，存在资产池选择和幸存者偏差。代码全期使用同一配置资产池，无逐时点成分或退市资产纳入规则。
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.07226670054346118, ann_vol=0.09615056454967019, sharpe=0.7738992606310303, sharpe_excess=0.7684336021526856, max_drawdown=-0.1274832853313922, total_return=0.7460706204597298, turnover=2.887284075860543, n_trades=322
    oos: cagr=0.0820695768982469, ann_vol=0.10459835140766069, sharpe=0.8065002027655708, sharpe_excess=0.5743850041649713, max_drawdown=-0.16386103844055044, total_return=0.8765572891309794, turnover=3.185216181710447, n_trades=299
    full: cagr=0.07715455106475844, ann_vol=0.10044919585007439, sharpe=0.7902726193311763, sharpe_excess=0.6669062070132834, max_drawdown=-0.16386103844054745, total_return=2.2766015501611916, turnover=3.0361760899917947, n_trades=621
    benchmark: cagr=0.1392476082074563, ann_vol=0.1720759533102992, sharpe=0.8440643823286108, sharpe_excess=0.7717644941407711, max_drawdown=-0.3371725386608825, total_return=7.018490274862035, turnover=0.06257419511706738, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 681_dynamic_portfolio_selection_for_nonlinea\nuniverse: [SPY, EFA, GLD, IEF, TLT, BIL]\nbenchmark: SPY\ndefensive_etf: BIL\ndates:\n  warmup_start: 2008-06-01\n  start: 2010-01-04\n  is_end: 2017-12-31\n  end: 2025-12-31\nmodel:\n  lookback_sessions: 252\n  scenario_sessions: 21\n  stress_quantile: 0.90\n  base_mix: 0.25\n  stress_loading: 0.60\n  downside_power: 5.0\n  risky_weight_cap: 0.40\nexecution:\n  signal_lag_sessions: 1\n  cost_bps: 5.0\n  slippage_bps: 3.0\nrobustness:\n  lookback_sessions: [189, 315]\n  stress_loading: [0.45, 0.75]\n  higher_friction_multiplier: 2.0\n'
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
"""Monthly ETF allocation with a drawdown-sensitive mixed-CRRA preference.

Source (2023 arXiv, sections 4.1 and 4.5): https://arxiv.org/abs/2311.06745
Author's publication list: https://fy-yuan.github.io/publications/
Related implementations reviewed for ideas, with no source code copied:
https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/pypfopt/efficient_frontier/efficient_cdar.py (MIT)
https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/LICENSE
https://pypi.org/project/pyportfolioopt/
https://github.com/chawla2255/portfolio-optimizer/blob/main/optim/backtest.py (MIT)
https://github.com/chawla2255/portfolio-optimizer/blob/main/optim/optimizers.py (MIT)
https://github.com/chawla2255/portfolio-optimizer/blob/main/LICENSE
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
TRADING_DAYS = 252


@dataclass(frozen=True)
class Model:
    lookback_sessions: int
    scenario_sessions: int
    stress_quantile: float
    base_mix: float
    stress_loading: float
    downside_power: float
    risky_weight_cap: float

    def validate(self):
        if not 0 < self.scenario_sessions < self.lookback_sessions:
            raise ValueError("Scenario length must be shorter than the trailing window")
        if not 0 < self.stress_quantile < 1 or not 0 < self.risky_weight_cap <= 1:
            raise ValueError("Invalid stress quantile or asset cap")
        if not 0 <= self.base_mix <= self.base_mix + self.stress_loading <= 1:
            raise ValueError("Mix weights must stay between zero and one")
        if not self.downside_power > 0:
            raise ValueError("Downside power must be positive")


@dataclass(frozen=True)
class Execution:
    signal_lag_sessions: int
    cost_bps: float
    slippage_bps: float

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000

    def validate(self):
        if type(self.signal_lag_sessions) is not int or self.signal_lag_sessions < 1:
            raise ValueError("Execution must lag the signal by at least one session")
        if not (self.cost_bps >= 0 and self.slippage_bps >= 0 and self.friction < 1):
            raise ValueError("Invalid execution friction")


def validate_prices(opens, closes, start, end):
    index = closes.index
    if index.empty or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Yahoo prices must have a complete, chronological session index")
    if not start <= index[0] <= start + pd.Timedelta(days=7):
        raise ValueError("Yahoo prices do not cover the start of the warmup request")
    if not end - pd.Timedelta(days=7) <= index[-1] <= end:
        raise ValueError("Yahoo prices do not cover the end of the backtest request")
    for prices in (opens, closes):
        values = prices.to_numpy()
        bad_rows = ~np.isfinite(values).all(axis=1) | (values <= 0).any(axis=1)
        if bad_rows.any():
            raise ValueError(f"Missing or invalid Yahoo prices on {index[bad_rows][0].date()}")


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    bundle = CACHE / f"yahoo_adjusted_v1_{start.date()}_{end.date()}_{'-'.join(tickers)}.npz"
    if bundle.exists():
        with np.load(bundle, allow_pickle=False) as cached:
            index = pd.DatetimeIndex(cached["sessions"])
            opens = pd.DataFrame(cached["opens"], index=index, columns=tickers)
            closes = pd.DataFrame(cached["closes"], index=index, columns=tickers)
            fetched_at = datetime.fromisoformat(str(cached["fetched_at"].item()))
        validate_prices(opens, closes, start, end)
        return opens, closes, fetched_at

    with TemporaryDirectory(dir=CACHE) as staging:
        yf.set_tz_cache_location(staging)
        bars = yf.download(
            tickers,
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d",
            auto_adjust=True,
            repair=False,
            keepna=True,
            progress=False,
            threads=False,
            timeout=12,
        )
        fetched_at = datetime.now(timezone.utc)
        opens = bars["Open"].reindex(columns=tickers)
        closes = bars["Close"].reindex(columns=tickers)
        validate_prices(opens, closes, start, end)
        staged_bundle = Path(staging) / bundle.name
        np.savez_compressed(
            staged_bundle,
            sessions=closes.index.to_numpy(dtype="datetime64[ns]"),
            opens=opens.to_numpy(),
            closes=closes.to_numpy(),
            fetched_at=fetched_at.isoformat(),
        )
        staged_bundle.replace(bundle)
    return opens, closes, fetched_at


def drawdown_stress(window, risky, quantile):
    risky_prices = window[:, risky]
    daily_gross = (risky_prices[1:] / risky_prices[:-1]).mean(axis=1)
    path = np.r_[1.0, np.cumprod(daily_gross)]
    drawdowns = 1.0 - path / np.maximum.accumulate(path)
    threshold = np.quantile(drawdowns, quantile)
    return min(1.0, drawdowns[-1] / threshold) if threshold > 0 else 0.0


def mixed_crra_objective(weights, scenarios, mix, downside_power):
    gross = scenarios @ weights
    log_gross = np.log(gross)

    def average_utility(log_equivalent):
        log_ratio = log_gross - log_equivalent
        downside_utility = -np.expm1(-downside_power * log_ratio) / downside_power
        return np.mean((1 - mix) * log_ratio + mix * downside_utility)

    # The root lies between the smallest and largest scenario gross returns.
    log_equivalent = brentq(
        average_utility, log_gross.min(), log_gross.max(), xtol=1e-13
    )
    # Implicit derivative of the root with respect to each asset weight.
    marginal = (1 - mix) + mix * np.exp(downside_power * (log_equivalent - log_gross))
    gradient = (marginal / gross) @ scenarios / marginal.sum()
    return -log_equivalent, -gradient


def target_weights(window, risky, model):
    horizon = model.scenario_sessions
    scenarios = window[horizon:] / window[:-horizon]
    stress = drawdown_stress(window, risky, model.stress_quantile)
    mix = model.base_mix + model.stress_loading * stress
    caps = np.where(risky, model.risky_weight_cap, 1.0)
    result = minimize(
        mixed_crra_objective,
        caps / caps.sum(),
        args=(scenarios, mix, model.downside_power),
        jac=True,
        method="SLSQP",
        bounds=list(zip(np.zeros(caps.size), caps)),
        constraints={
            "type": "eq",
            "fun": lambda weights: weights.sum() - 1.0,
            "jac": lambda weights: np.ones_like(weights),
        },
        options={"ftol": 1e-12, "maxiter": 150},
    )
    weights = result.x
    if not (
        result.success and np.isfinite(weights).all() and abs(weights.sum() - 1.0) < 1e-9
        and (weights >= -1e-9).all() and (weights <= caps + 1e-9).all()
    ):
        raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    weights[risky] = np.clip(weights[risky], 0.0, caps[risky])
    # Assign numerical residue to the defensive ETF, keeping the capped weights intact.
    weights[~risky] = max(0.0, 1.0 - weights[risky].sum())
    return weights / weights.sum()


def build_targets(closes, start, risky, model, lag):
    first = closes.index.searchsorted(start)
    if first < model.lookback_sessions:
        raise ValueError("Insufficient trailing history at the start of the backtest")
    months = closes.index.to_period("M")
    values = closes.to_numpy()
    targets = {}
    for row in range(first, len(closes) - lag):
        # The first observed session of a month is identifiable at its own close.
        if row == first or months[row] != months[row - 1]:
            window = values[row - model.lookback_sessions : row + 1]
            try:
                targets[closes.index[row + lag]] = target_weights(window, risky, model)
            except RuntimeError as error:
                raise RuntimeError(f"Signal on {closes.index[row].date()}: {error}") from error
    return targets


def fill_at_open(positions, cash, weights, friction):
    open_value = positions.sum() + cash
    previous_fractions = positions / open_value

    def budget(remaining):
        traded_fraction = np.abs(remaining * weights - previous_fractions).sum()
        return remaining + friction * traded_fraction - 1.0

    investable = brentq(budget, 0.0, 1.0, xtol=1e-14)
    changes = open_value * investable * weights - positions
    tolerance = open_value * 1e-10
    sells = np.where(changes < -tolerance, -changes, 0.0)
    buys = np.where(changes > tolerance, changes, 0.0)
    cash += sells.sum() * (1.0 - friction)
    buy_cost = buys.sum() * (1.0 + friction)
    if buy_cost > 0:
        cash_spent = min(cash, buy_cost)
        buys *= cash_spent / buy_cost
        cash -= cash_spent
    traded = sells + buys
    return positions - sells + buys, cash, traded.sum() / open_value, np.count_nonzero(traded)


def simulate(opens, closes, start, targets, friction):
    dates = closes.loc[start:].index
    raw_opens = opens.loc[dates].to_numpy()
    overnight = raw_opens / closes.shift(1).loc[dates].to_numpy()
    intraday = closes.loc[dates].to_numpy() / raw_opens
    positions = np.zeros(closes.shape[1])
    cash = previous_value = 1.0
    rows = []
    for date, open_move, day_move in zip(dates, overnight, intraday):
        positions *= open_move
        turnover, trades = 0.0, 0
        if date in targets:
            positions, cash, turnover, trades = fill_at_open(
                positions, cash, targets[date], friction
            )
        positions *= day_move
        value = positions.sum() + cash
        rows.append((value / previous_value - 1.0, turnover, trades))
        previous_value = value
    return pd.DataFrame(rows, index=dates, columns=["net_return", "turnover", "n_trades"])


def period_metrics(daily_results, risk_free_returns):
    returns = daily_results["net_return"].to_numpy()
    excess = returns - risk_free_returns.loc[daily_results.index].to_numpy()
    path = np.r_[1.0, np.cumprod(1.0 + returns)]
    years = len(returns) / TRADING_DAYS

    def sharpe(series):
        volatility = np.std(series, ddof=1)
        return float(np.sqrt(TRADING_DAYS) * np.mean(series) / volatility) if volatility else None

    return {
        "cagr": float(path[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(np.sqrt(TRADING_DAYS) * np.std(returns, ddof=1)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(excess),
        "max_drawdown": float(np.min(path / np.maximum.accumulate(path) - 1.0)),
        "total_return": float(path[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def summarize(daily_results, risk_free_returns, is_end):
    in_sample = daily_results.index <= is_end
    return {
        "is": period_metrics(daily_results.loc[in_sample], risk_free_returns),
        "oos": period_metrics(daily_results.loc[~in_sample], risk_free_returns),
        "full": period_metrics(daily_results, risk_free_returns),
    }


def robustness_cases(model, execution, config):
    multiple = config["higher_friction_multiplier"]
    return [
        *((f"lookback_{value}", replace(model, lookback_sessions=value), execution)
          for value in config["lookback_sessions"]),
        *((f"stress_loading_{value}", replace(model, stress_loading=value), execution)
          for value in config["stress_loading"]),
        ("fixed_mix", replace(model, stress_loading=0.0), execution),
        (f"friction_{multiple:g}x", model, replace(
            execution, cost_bps=execution.cost_bps * multiple,
            slippage_bps=execution.slippage_bps * multiple,
        )),
    ]


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    dates = {key: pd.Timestamp(value) for key, value in config["dates"].items()}
    start, is_end, end = dates["start"], dates["is_end"], dates["end"]
    if not dates["warmup_start"] < start < is_end < end:
        raise ValueError("Expected warmup, start, in-sample end, and end in chronological order")

    tickers = config["universe"]
    defensive = config["defensive_etf"]
    benchmark = config["benchmark"]
    if (len(set(tickers)) != len(tickers) or benchmark == defensive
            or not {defensive, benchmark} <= set(tickers)):
        raise ValueError("The ETF universe must be unique and contain the benchmark and defensive ETF")
    risky = np.asarray(tickers) != defensive
    model, execution = Model(**config["model"]), Execution(**config["execution"])
    alternatives = robustness_cases(model, execution, config["robustness"])
    pairs = [(model, execution), *((variant, fills) for _, variant, fills in alternatives)]
    if len(set(pairs)) != len(pairs):
        raise ValueError("The default and each sensitivity case must be distinct")
    for variant, fills in pairs:
        variant.validate()
        fills.validate()

    opens, closes, fetched_at = load_prices(tickers, dates["warmup_start"], end)
    risk_free_returns = closes[defensive].pct_change(fill_method=None)
    evaluation = closes.loc[start:].index
    if min((evaluation <= is_end).sum(), (evaluation > is_end).sum()) < 2:
        raise ValueError("Both evaluation periods require at least two sessions")

    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return Open and Close, repair=False")
    print("ROBUSTNESS (IS only)")
    trials = 0
    is_opens, is_closes = opens.loc[:is_end], closes.loc[:is_end]
    is_risk_free = risk_free_returns.loc[:is_end]
    for label, variant, fills in alternatives:
        targets = build_targets(is_closes, start, risky, variant, fills.signal_lag_sessions)
        variant_daily = simulate(is_opens, is_closes, start, targets, fills.friction)
        scores = period_metrics(variant_daily, is_risk_free)
        trials += 1
        print(f"  {label}: {json.dumps(scores, allow_nan=False)}")

    targets = build_targets(closes, start, risky, model, execution.signal_lag_sessions)
    daily_results = simulate(opens, closes, start, targets, execution.friction)
    trials += 1
    results = summarize(daily_results, risk_free_returns, is_end)
    if any(results[label]["n_trades"] == 0 for label in ("is", "oos")):
        raise RuntimeError("The strategy must have fills in both sample periods")

    benchmark_weights = (np.asarray(tickers) == benchmark).astype(float)
    benchmark_daily = simulate(
        opens, closes, start, {min(targets): benchmark_weights}, execution.friction
    )
    benchmark_results = summarize(benchmark_daily, risk_free_returns, is_end)
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(results[period], allow_nan=False)}")
        print(f"BENCHMARK {benchmark} {period.upper()}: "
              f"{json.dumps(benchmark_results[period], allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {trials}")
    print("已回测 · 待审")
    payload = {
        "strategy": config["strategy"],
        "data_range": {
            "start": evaluation[0].date().isoformat(),
            "is_end": evaluation[evaluation <= is_end][-1].date().isoformat(),
            "oos_start": evaluation[evaluation > is_end][0].date().isoformat(),
            "end": evaluation[-1].date().isoformat(),
        },
        **results,
        "benchmark": {"ticker": benchmark, **benchmark_results["full"], **benchmark_results},
    }
    print("RESULTS_JSON: " + json.dumps(payload, allow_nan=False), flush=True)


if __name__ == "__main__":
    signal.alarm(14 * 60)
    main()
