"""574_objective_and_subjective_entropy_measure
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/574_objective_and_subjective_entropy_measure.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 25a9179813de3addbbaee65a8c08579eb6e27d08 on 2026-11-27 19:08 Europe/Oslo, exit 0, output sha256 dacde285b5c74e42f47ab23f1ea36cbdb3f45171f10ee8eb45dcab5cf2b20d14
Data source: Yahoo Finance 免费日线，经 yfinance 读取开盘价、收盘价、股息及可用的资本利得分配。投资池为 SPY、EFA、IEF、TLT、GLD、VNQ，SPY 同时作买入持有基准，BIL 作短期国债收益代理。数据抓取时间记录为 2026-11-27T17:09:07+00:00。
Data fetched: 2026-11-27T18:08:32+00:00
Sample period: 行情请求范围为 2008-01-01 至 2026-10-31（含结束日期），2008 年 1 月开始预热，配置从 2010-01-04 起允许产生信号。该次运行的实际计分区间为 2010-01-05 开盘至 2026-10-30 收盘。   data_range (from output): {'start': '2010-01-05', 'end': '2026-10-30'}
IS/OOS split: 样本内为 2010-01-05 至 2019-12-31（含），样本外为 2020-01-02 至 2026-10-30（含）。正式组合跨分界连续持仓，样本外沿用正式规格并继续滚动估计。七组设定均在截断的样本内行情上运行，正式规格的样本内结果与全期路径的样本内对应部分做逐项精确比较。
Costs and slippage: 默认交易成本 5 bp，滑点以固定 5 bp 模拟。每笔买入或卖出按其成交金额扣除合计 10 bp，在开盘求解扣费后的可投资金额，并从开盘资金中扣款。较高摩擦的样本内变体为成本和滑点各 10 bp，合计 20 bp。SPY 基准首次建仓按默认费率扣款。
Signal lag: 每月首个交易日 t 收盘后，使用包含 t 日总收益的 120 交易日窗口估计目标。首次入场或满足调仓条件时，默认于下一交易日开盘成交，成交额按该开盘估值确定。
Price adjustment: yfinance split-normalized Open/Close, ex-date distributions reinvested at open
Survivorship bias: 配置使用固定 ETF 池，README 记录这些 ETF 在样本开始前均已上市。仍存在基金存续与事后选池偏差。代码固定使用配置中的证券，并要求整个行情样本的开盘、收盘和分配数据有效。
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.052487615831784185, ann_vol=0.12451687255233748, sharpe=0.4732413654965694, sharpe_excess=0.43955857782197505, max_drawdown=-0.23852132291745587, total_return=0.6662076743150096, turnover=5.460374553083423, n_trades=153
    oos: cagr=0.07995587964862616, ann_vol=0.14849589979328365, sharpe=0.5923634509933415, sharpe_excess=0.40837010076524055, max_drawdown=-0.36958910283610513, total_return=0.6889298293898682, turnover=4.9799948981979805, n_trades=96
    full: cagr=0.06354674925372472, ann_vol=0.13474695645936915, sharpe=0.5247419577071145, sharpe_excess=0.4240003621920986, max_drawdown=-0.369589102836105, total_return=1.8141078431089213, turnover=5.265475718622577, n_trades=249
    benchmark SPY full: cagr=0.1394169574183226, ann_vol=0.17024170364403837, sharpe=0.852176994505426, sharpe_excess=0.7721091778413668, max_drawdown=-0.3372273754868176, total_return=7.951774014962075, turnover=0.059486826972589744, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 574_objective_and_subjective_entropy_measure\nuniverse: [SPY, EFA, IEF, TLT, GLD, VNQ]\nbenchmark: SPY\nrisk_free_proxy: BIL\n\ndates:\n  price_start: '2008-01-01'\n  start: '2010-01-04'\n  is_end: '2019-12-31'\n  end: '2026-10-31'\n\nmodel:\n  lookback_sessions: 120\n  covariance_diagonal_weight: 0.10\n  max_weight: 0.50\n  forecast_horizon_years: 0.08333333333333333\n  entropy_buffer_bps: 5.0\n  use_entropy_gate: true\n\nexecution:\n  lag_sessions: 1\n  cost_bps: 5.0\n  slippage_bps: 5.0\n\nrobustness:\n  lookback_sessions: [100, 140]\n  entropy_buffer_bps: [2.5, 7.5]\n  high_cost_bps: 10.0\n  high_slippage_bps: 10.0\n"
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
"""Research backtest of long-only Kelly allocation with an entropy hurdle.

Ati S Sharma (2026): https://arxiv.org/abs/2607.09505
Research references (formulas and comparison only, independently written code):
KellyPortfolio (MIT): https://github.com/thk3421-models/KellyPortfolio
https://raw.githubusercontent.com/thk3421-models/KellyPortfolio/main/kelly.py
https://raw.githubusercontent.com/thk3421-models/KellyPortfolio/main/license.txt
Riskfolio-Lib (BSD-3-Clause): https://pypi.org/project/riskfolio-lib/
https://raw.githubusercontent.com/dcajasn/Riskfolio-Lib/master/riskfolio/src/Portfolio.py
Risk-Constrained Kelly Gambling: https://stanford.edu/~boyd/papers/kelly.html
Replication (GPL-3.0): https://github.com/cvxgrp/kelly_code
https://notebook.community/d00d/quantNotebooks/Notebooks/quantopian_research_public/presentations/LECTURE_Stanford_Quantopian_Tutorial_and_Markowitz_Optimization
https://github.com/QuantConnect/Lean/blob/master/Algorithm.Framework/Portfolio/RiskParityPortfolioOptimizer.py
"""

import signal

if __name__ == "__main__":
    signal.alarm(14 * 60)

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
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class Model:
    lookback_sessions: int
    covariance_diagonal_weight: float
    max_weight: float
    forecast_horizon_years: float
    entropy_buffer_bps: float
    use_entropy_gate: bool

    def check(self, asset_count):
        if type(self.lookback_sessions) is not int or self.lookback_sessions < 2:
            raise ValueError("The rolling window must contain at least two sessions.")
        if not 0 < self.covariance_diagonal_weight <= 1:
            raise ValueError("Covariance shrinkage must be in (0, 1].")
        if not 1 / asset_count <= self.max_weight <= 1:
            raise ValueError("The upper weight bound must permit full investment.")
        if not (np.isfinite(self.forecast_horizon_years) and self.forecast_horizon_years > 0
                and np.isfinite(self.entropy_buffer_bps) and self.entropy_buffer_bps >= 0
                and isinstance(self.use_entropy_gate, bool)):
            raise ValueError("Invalid entropy hurdle settings.")


@dataclass(frozen=True)
class Execution:
    lag_sessions: int
    cost_bps: float
    slippage_bps: float

    @property
    def friction(self):
        if type(self.lag_sessions) is not int or self.lag_sessions < 1:
            raise ValueError("Fills require at least one session of signal lag.")
        if not (self.cost_bps >= 0 and self.slippage_bps >= 0
                and self.cost_bps + self.slippage_bps < 5_000):
            raise ValueError("Cost and slippage must be nonnegative and total less than 5,000 bps.")
        return (self.cost_bps + self.slippage_bps) / 10_000


def load_market(tickers, start, end):
    """Load Yahoo's split-normalized OHLC and effective-date distributions."""
    CACHE.mkdir(exist_ok=True)
    end_exclusive = end + pd.Timedelta(days=1)
    key = json.dumps(["yahoo-open-close-distributions-v1", tickers, str(start), str(end)])
    path = CACHE / f"prices_{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"
    from_cache = path.exists()
    if from_cache:
        with np.load(path, allow_pickle=False) as snapshot:
            dates = pd.DatetimeIndex(snapshot["dates"])
            opens = snapshot["opens"]
            closes = snapshot["closes"]
            distributions = snapshot["distributions"]
            fetched_at = snapshot["fetched_at"].item()
        datetime.fromisoformat(fetched_at)
    else:
        with TemporaryDirectory(dir=CACHE, prefix="yahoo_") as request_dir:
            yf.set_tz_cache_location(str(request_dir))
            raw = yf.download(
                tickers, start=start.strftime("%Y-%m-%d"), end=end_exclusive.strftime("%Y-%m-%d"),
                interval="1d", auto_adjust=False, actions=True, repair=False, keepna=True,
                progress=False, threads=4, ignore_tz=True, timeout=12, multi_level_index=True,
            )
        if raw is None or raw.empty:
            raise RuntimeError("Yahoo Finance returned no daily bars.")
        dates = pd.DatetimeIndex(raw.index)
        opens = raw["Open"].reindex(columns=tickers).to_numpy(dtype=float)
        closes = raw["Close"].reindex(columns=tickers).to_numpy(dtype=float)
        dividends = raw["Dividends"].reindex(columns=tickers).to_numpy(dtype=float)
        gains = np.zeros_like(dividends)
        if "Capital Gains" in raw:
            gains = raw["Capital Gains"].reindex(columns=tickers, fill_value=0).fillna(0).to_numpy(dtype=float)
        distributions = dividends + gains
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if (dates.empty or dates.has_duplicates or dates.hasnans or not dates.is_monotonic_increasing
            or not start <= dates[0] <= start + pd.Timedelta(days=7)
            or not end - pd.Timedelta(days=7) <= dates[-1] <= end):
        raise RuntimeError("Yahoo Finance bars do not cover the requested dates in session order.")
    for name, data, valid_minimum in (("open", opens, opens > 0), ("close", closes, closes > 0),
                                      ("distribution", distributions, distributions >= 0)):
        bad = np.argwhere(~np.isfinite(data) | ~valid_minimum)
        if len(bad):
            i, j = bad[0]
            raise RuntimeError(f"Invalid or missing {name} for {tickers[j]} on {dates[i].date()}.")

    if not from_cache:
        with NamedTemporaryFile(dir=CACHE, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                np.savez_compressed(temporary, dates=dates.to_numpy(), opens=opens, closes=closes,
                                    distributions=distributions, fetched_at=fetched_at)
                temporary.close()
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)

    # Previous positions earn the ex-date distribution, reinvested at the open.
    # The fill path uses only this overnight factor. Today's close is used afterwards.
    overnight = pd.DataFrame((opens[1:] + distributions[1:]) / closes[:-1], dates[1:], tickers)
    intraday = pd.DataFrame(closes[1:] / opens[1:], dates[1:], tickers)
    return overnight, intraday, fetched_at


def kelly_target(window, risk_free_window, model):
    drift = (window.mean(axis=0) - risk_free_window.mean()) * SESSIONS_PER_YEAR
    sample_covariance = np.cov(window, rowvar=False, ddof=1) * SESSIONS_PER_YEAR
    shrink = model.covariance_diagonal_weight
    covariance = (1 - shrink) * sample_covariance + shrink * np.diag(np.diag(sample_covariance))
    if np.any(np.diag(covariance) <= 0):
        raise RuntimeError("The rolling covariance requires positive variance for each ETF.")

    def loss(weights):
        return 0.5 * weights @ covariance @ weights - drift @ weights

    count = len(drift)
    initial = np.full(count, 1 / count)
    solution = minimize(
        loss, initial, jac=lambda weights: covariance @ weights - drift, method="SLSQP",
        bounds=[(0.0, model.max_weight)] * count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1,
                     "jac": lambda weights: np.ones(count)},
        options={"ftol": 1e-12, "maxiter": 100},
    )
    weights = solution.x
    if (not solution.success or not np.isfinite(weights).all() or abs(weights.sum() - 1) > 1e-9
            or weights.min() < -1e-9 or weights.max() > model.max_weight + 1e-9):
        raise RuntimeError(f"Kelly optimization failed: {solution.message}")
    weights = np.clip(weights, 0, model.max_weight)
    return weights / weights.sum(), drift, covariance


def should_rebalance(current, target, drift, covariance, model, friction):
    # D(current) - D(target) from the paper's quadratic, with its constant canceled.
    change = target - current
    annual_gap = change @ (drift - covariance @ ((target + current) / 2))
    estimated_drag = -np.log1p(-friction * np.abs(change).sum())
    hurdle = estimated_drag + model.entropy_buffer_bps / 10_000
    return annual_gap * model.forecast_horizon_years > hurdle


def fill_at_open(open_positions, cash, target, friction):
    """Allocate at open marks after paying costs on both buy and sell amounts."""
    opening_value = open_positions.sum() + cash
    previous_weights = open_positions / opening_value

    def funding_error(fraction):
        gross_turnover = np.abs(fraction * target - previous_weights).sum()
        return fraction + friction * gross_turnover - 1.0

    fraction = brentq(funding_error, 0.0, 1.0, xtol=1e-14)
    # Leave a tiny positive cash remainder to tolerate floating point rounding.
    next_positions = (fraction - 1e-12) * opening_value * target
    trade_values = np.abs(next_positions - open_positions)
    turnover = trade_values.sum() / opening_value
    remaining_cash = (opening_value - next_positions.sum()) - friction * trade_values.sum()
    if remaining_cash < 0 or np.any(next_positions < 0):
        raise RuntimeError("Opening rebalance exceeded available funds.")
    n_trades = int(np.count_nonzero(trade_values > opening_value * 1e-10))
    return next_positions, remaining_cash, turnover, n_trades


def simulate(overnight, intraday, universe, risk_free, start, model, execution):
    dates = overnight.index
    months = dates.to_period("M")
    monthly = np.r_[False, months[1:] != months[:-1]]
    eligible = np.flatnonzero(monthly & (dates >= start))
    if not len(eligible) or eligible[0] < model.lookback_sessions - 1:
        raise ValueError("The first monthly signal requires a complete rolling window.")
    first_signal = int(eligible[0])
    first_fill = first_signal + execution.lag_sessions
    if first_fill >= len(dates):
        raise ValueError("The sample ends before the first fill.")

    overnight_moves = overnight[universe].to_numpy()
    intraday_moves = intraday[universe].to_numpy()
    closes = overnight_moves * intraday_moves - 1.0
    risk_free_closes = (overnight[risk_free] * intraday[risk_free]).to_numpy() - 1.0
    friction = execution.friction
    positions = np.zeros(len(universe))
    cash = 1.0
    pending_index = -1
    pending_target = None
    daily_results = []
    for i in range(first_signal, len(dates)):
        prior_value = positions.sum() + cash
        positions *= overnight_moves[i]
        turnover, trades = 0.0, 0
        if i == pending_index:
            positions, cash, turnover, trades = fill_at_open(positions, cash, pending_target, friction)
            pending_target = None
        positions *= intraday_moves[i]
        closing_value = positions.sum() + cash
        if i >= first_fill:
            daily_results.append((closing_value / prior_value - 1.0, turnover, trades))

        if monthly[i] and pending_target is None:
            window = slice(i - model.lookback_sessions + 1, i + 1)
            target, drift, covariance = kelly_target(closes[window], risk_free_closes[window], model)
            if (i == first_signal or not model.use_entropy_gate
                    or should_rebalance(positions / positions.sum(), target, drift, covariance, model, friction)):
                pending_target = target
                pending_index = i + execution.lag_sessions

    return pd.DataFrame(daily_results, index=dates[first_fill:], columns=["return", "turnover", "n_trades"])


def buy_and_hold(overnight, intraday, ticker, active_dates, friction):
    gross = (overnight[ticker] * intraday[ticker]).loc[active_dates].copy()
    gross.iloc[0] = intraday.at[active_dates[0], ticker]
    positions, cash, turnover, trades = fill_at_open(np.zeros(1), 1.0, np.ones(1), friction)
    values = positions[0] * gross.cumprod() + cash
    daily_returns = values / values.shift(1, fill_value=1.0) - 1.0
    result = pd.DataFrame({"return": daily_returns, "turnover": 0.0, "n_trades": 0})
    result.iloc[0, 1:] = turnover, trades
    return result


def metrics(daily_results, risk_free_returns):
    returns = daily_results["return"].to_numpy()
    if len(returns) < 2:
        raise ValueError("Each reporting period requires at least two sessions.")
    excess = returns - risk_free_returns.loc[daily_results.index].to_numpy()
    growth = np.r_[1.0, np.cumprod(1.0 + returns)]
    daily_vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    scale = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1.0),
        "ann_vol": float(daily_vol * scale),
        "sharpe": float(returns.mean() * scale / daily_vol) if daily_vol else None,
        "sharpe_excess": float(excess.mean() * scale / excess_vol) if excess_vol else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1.0).min()),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def report(label, daily_results, risk_free_returns):
    summary = metrics(daily_results, risk_free_returns)
    print(f"{label}: {json.dumps(summary, allow_nan=False)}", flush=True)
    return summary


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe, benchmark, risk_free = config["universe"], config["benchmark"], config["risk_free_proxy"]
    tickers = list(dict.fromkeys([*universe, benchmark, risk_free]))
    if len(set(universe)) != len(universe) or len(universe) < 2 or risk_free != "BIL":
        raise ValueError("Use distinct ETFs with at least two assets and BIL as the reference rate.")
    if any(not ticker.isascii() or not ticker.isalpha() or not ticker.isupper() for ticker in tickers):
        raise ValueError("Use public US ETF ticker symbols.")
    dates = config["dates"]
    price_start, start, is_end, end = (pd.Timestamp(dates[key]) for key in ("price_start", "start", "is_end", "end"))
    if not price_start < start < is_end < end:
        raise ValueError("Warmup, score start, IS cutoff and score end must be strictly ordered.")

    model, execution = Model(**config["model"]), Execution(**config["execution"])
    sensitivity = config["robustness"]
    cases = [("default", model, execution)]
    cases += [(f"lookback_{days}", replace(model, lookback_sessions=days), execution)
              for days in sensitivity["lookback_sessions"]]
    cases += [(f"buffer_{bps:g}bp", replace(model, entropy_buffer_bps=bps), execution)
              for bps in sensitivity["entropy_buffer_bps"]]
    cases += [
        ("higher_cost_and_slippage", model, replace(execution, cost_bps=sensitivity["high_cost_bps"],
                                                    slippage_bps=sensitivity["high_slippage_bps"])),
        ("monthly_without_hurdle", replace(model, use_entropy_gate=False), execution),
    ]
    if len({(case_model, case_execution) for _, case_model, case_execution in cases}) != len(cases):
        raise ValueError("Each robustness trial needs its own parameter combination.")
    for _, case_model, case_execution in cases:
        case_model.check(len(universe))
        _ = case_execution.friction

    overnight, intraday, fetched_at = load_market(tickers, price_start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance split-normalized Open/Close, ex-date distributions reinvested at open", flush=True)
    bil_returns = overnight[risk_free] * intraday[risk_free] - 1.0

    print("ROBUSTNESS (IS only)", flush=True)
    overnight_is, intraday_is = overnight.loc[:is_end], intraday.loc[:is_end]
    bil_is = bil_returns.loc[:is_end].copy()
    is_default = None
    for label, case_model, case_execution in cases:
        daily = simulate(overnight_is, intraday_is, universe, risk_free, start, case_model, case_execution)
        if label == "default":
            is_default = daily
            first_day = daily.index[0]
            bil_is.loc[first_day] = intraday_is.at[first_day, risk_free] - 1.0
        report(f"ROBUSTNESS_IS {label}", daily, bil_is)

    official = simulate(overnight, intraday, universe, risk_free, start, model, execution)
    pd.testing.assert_frame_equal(is_default, official.loc[:is_end], check_exact=True)
    bil_returns.loc[first_day] = bil_is.loc[first_day]
    if official.loc[:is_end, "n_trades"].sum() <= 0 or official.loc[official.index > is_end, "n_trades"].sum() <= 0:
        raise RuntimeError("Both IS and OOS must contain simulated trades.")
    print("LOOK_AHEAD_CHECK: exact IS prefix replay, prior-close signals, lagged open sizing, effective-date distributions")
    print("BACKTEST_STATUS: 已回测 / 待复核")
    print(f"PARAM_COMBOS_TRIED: {len(cases)}")

    baseline = buy_and_hold(overnight, intraday, benchmark, official.index, execution.friction)
    result = {"strategy": config["strategy"],
              "data_range": {"start": official.index[0].date().isoformat(), "end": official.index[-1].date().isoformat()}}
    baseline_metrics = {"ticker": benchmark}
    in_sample = official.index <= is_end
    for label, selection in (("is", in_sample), ("oos", ~in_sample), ("full", slice(None))):
        result[label] = report(label.upper(), official.loc[selection], bil_returns)
        baseline_metrics[label] = report(f"BENCHMARK {benchmark} {label.upper()}", baseline.loc[selection],
                                         bil_returns)
    result["benchmark"] = baseline_metrics
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
