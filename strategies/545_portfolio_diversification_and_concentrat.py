"""545_portfolio_diversification_and_concentrat
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/545_portfolio_diversification_and_concentrat.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c0a8f76240a0431b019ab49c34040a95a765860a on 2026-11-27 10:04 Europe/Oslo, exit 0, output sha256 1767cce06c3ceb33ee03b0ad1169db8960f4f1d4e05470ca9cc2a81a1d2d1cb4
Data source: Yahoo Finance（yfinance）日度复权收盘价。配置资产为 SPY、TLT、GLD、VNQ、EFA，BIL 仅用于计算超额夏普比率（config.yaml:2-3，strategy.py:70-87、227-241、307-310，README.md:20-21）。
Data fetched: 2026-11-27T09:04:41.260423+00:00
Sample period: 回测收益区间为 2010-01-04 至 2025-12-31，含首尾交易日。预热价格请求始于 2007-06-01（config.yaml:4-7，strategy.py:52-55、307-311，logs/run_20261127_092016.log:17）。   data_range (from output): {'start': '2010-01-04', 'end': '2025-12-31'}
IS/OOS split: 样本内为 2010-01-04 至 2018-12-31，样本外为 2019-01-02 至 2025-12-31。默认参数下连续模拟，样本外承接样本内期末仓位，按日期分窗计算指标。五个备选规格仅在样本内评估，正式结果使用默认规格（config.yaml:5-7，README.md:22、31，strategy.py:249-255、315-344）。
Costs and slippage: 默认每笔模拟 ETF 买入或卖出按成交金额收取 2 bps 费用和 3 bps 固定滑点，共 5 bps。两者合并按买卖金额之和从组合净值扣除，首次建仓计费，期末没有强制平仓。样本内高摩擦情形使用 4 bps 费用和 6 bps 滑点（config.yaml:12-17，strategy.py:44-46、191-224、270-276，README.md:24，logs/run_20261127_092016.log:11）。
Signal lag: 月末最后一个交易日收盘后，用截至该收盘的 504 个日收益率计算目标权重，于下一交易日收盘模拟成交。成交日的收盘到收盘价格收益先由旧仓位承担，成交后启用新权重（config.yaml:8-9，strategy.py:168-188、207-224、308-310，README.md:23）。
Price adjustment: yfinance auto_adjust=True, dividend- and split-adjusted close
Survivorship bias: 资产池固定，事后选取现存且历史较长的 ETF，存在生存者偏差和资产选择偏差（config.yaml:2，strategy.py:290-310，README.md:35）。
Parameter combinations tried: 6
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.09412835595954472, ann_vol=0.07664992481529631, sharpe=1.2121657472569205, sharpe_excess=1.1814472009622246, max_drawdown=-0.10451479168453304, total_return=1.2438526325438621, turnover=0.9627849366482054, n_trades=304
    oos: cagr=0.08291655075652438, ann_vol=0.1166847219122516, sharpe=0.7411187144061572, sharpe_excess=0.5244993674344766, max_drawdown=-0.322317579233243, total_return=0.7442784380737586, turnover=1.4298732240629148, n_trades=277
    full: cagr=0.08921037236855245, ann_vol=0.09621874243647964, sharpe=0.9363853492044876, sharpe_excess=0.807745731366995, max_drawdown=-0.32231757923324145, total_return=2.9139037651613084, turnover=1.1670780245830683, n_trades=581
    benchmark  full: cagr=0.08769607550537772, ann_vol=0.10862830322000282, sharpe=0.8283785329796435, sharpe_excess=0.7139551222473253, max_drawdown=-0.25349084062255023, total_return=2.827912613838105, turnover=0.38188583441400115, n_trades=960
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "545_ portfolio_diversification_and_concentrat"\nuniverse: [SPY, TLT, GLD, VNQ, EFA]\nrisk_free: BIL\nprice_start: 2007-06-01\nstart_date: 2010-01-04\nis_end: 2018-12-31\nend_date: 2025-12-31\nexecution_lag_sessions: 1\nlookback_sessions: 504\nes_confidence: 0.95\nrobust_weight: 0.50\ncost_bps: 2.0\nslippage_bps: 3.0\nrobustness:\n  robust_weights: [0.25, 0.75]\n  es_confidences: [0.925, 0.975]\n  friction_multiplier: 2.0\n'
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
"""Monthly long-only allocation using the weighted ES objective of Liu and Liu."""

import hashlib
import json
import math
import signal
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, linprog
from scipy.sparse import csr_matrix, eye, hstack


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
PRICE_ADJUSTMENT = "yfinance auto_adjust=True, dividend- and split-adjusted close"
SESSIONS_PER_YEAR = 252
TRADE_TOL = 1e-8
RUN_LIMIT_SECONDS = 13 * 60


@dataclass(frozen=True)
class Parameters:
    lookback_sessions: int
    es_confidence: float
    robust_weight: float
    cost_bps: float
    slippage_bps: float

    def __post_init__(self):
        if type(self.lookback_sessions) is not int or self.lookback_sessions < 2:
            raise ValueError("lookback_sessions must be an integer of at least two.")
        if not 0 < self.es_confidence < 1 or not 0 <= self.robust_weight <= 1:
            raise ValueError("Invalid ES confidence or dependence robustness weight.")
        if not (self.cost_bps >= 0 and self.slippage_bps >= 0 and self.friction < 0.5):
            raise ValueError("Cost and slippage must be nonnegative and sum to < 5000 bps.")

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def load_prices(symbols, price_start, end_date):
    """Cache the exact Yahoo query and its fetch time in a single local CSV."""
    CACHE_DIR.mkdir(exist_ok=True)
    query = {
        "symbols": symbols,
        "start": price_start.date().isoformat(),
        "end": (end_date + pd.Timedelta(days=1)).date().isoformat(),
        "basis": PRICE_ADJUSTMENT,
    }
    key = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"yfinance_close_{key}.csv"
    if cache_file.exists():
        with cache_file.open(encoding="utf-8") as stream:
            timestamp = stream.readline().strip()
            fetched_at = datetime.fromisoformat(timestamp.removeprefix("# "))
            prices = pd.read_csv(
                stream, index_col="Date", parse_dates=["Date"], float_precision="round_trip"
            )
        check_prices(prices, symbols, price_start, end_date)
        return prices, fetched_at

    # yfinance's anonymous HTTP cache is transient; keep only price data.
    with tempfile.TemporaryDirectory(dir=CACHE_DIR) as download_dir:
        yf.set_tz_cache_location(download_dir)
        data = yf.download(
            symbols,
            start=query["start"],
            end=query["end"],
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=12,
            multi_level_index=True,
        )
    if data is None or data.empty or "Close" not in data:
        raise RuntimeError("yfinance returned no daily adjusted closes.")
    prices = data["Close"].reindex(columns=symbols).sort_index()
    prices.index.name = "Date"
    check_prices(prices, symbols, price_start, end_date)
    fetched_at = datetime.now(timezone.utc)
    with tempfile.NamedTemporaryFile(
        dir=CACHE_DIR, mode="w", encoding="utf-8", delete=False
    ) as stream:
        temporary_path = Path(stream.name)
        try:
            stream.write(f"# {fetched_at.isoformat()}\n")
            prices.to_csv(stream)
            stream.close()
            temporary_path.replace(cache_file)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def check_prices(prices, symbols, price_start, end_date):
    dates = prices.index
    if (
        not isinstance(dates, pd.DatetimeIndex)
        or len(dates) < 2
        or dates.hasnans
        or not dates.is_unique
        or not dates.is_monotonic_increasing
        or list(prices.columns) != symbols
        or dates[0] < price_start
        or dates[-1] > end_date
        or dates[-1] < end_date - pd.Timedelta(days=7)
    ):
        raise ValueError("Adjusted price panel does not cover the requested dates and symbols.")
    values = prices.to_numpy(dtype=float)
    if not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError("Missing or invalid adjusted prices on the common ETF calendar.")


def empirical_es(losses, confidence):
    """Average the worst tail, including any fractional observation at its cutoff."""
    tail_size = len(losses) * (1 - confidence)
    fractions = np.clip(tail_size - np.arange(len(losses)), 0, 1)
    return fractions @ np.sort(losses, axis=0)[::-1] / tail_size


def optimal_weights(returns, parameters):
    """Solve the empirical version of equation (19), with losses scaled to bps."""
    losses = -10_000 * returns
    n_days, n_assets = losses.shape
    confidence = parameters.es_confidence
    omega = parameters.robust_weight
    marginal_es = empirical_es(losses, confidence)
    if omega == 1:
        weights = np.zeros(n_assets)
        weights[np.argmin(marginal_es)] = 1
        return weights

    # Variables are ETF weights, a free loss threshold, and positive excess losses.
    objective = np.r_[
        omega * marginal_es,
        1 - omega,
        np.full(n_days, (1 - omega) / (n_days * (1 - confidence))),
    ]
    tail_constraints = hstack(
        (losses, -np.ones((n_days, 1)), -eye(n_days, format="csr")), format="csr"
    )
    simplex = csr_matrix(np.r_[np.ones(n_assets), np.zeros(n_days + 1)][None, :])
    solution = linprog(
        objective,
        A_ub=tail_constraints,
        b_ub=np.zeros(n_days),
        A_eq=simplex,
        b_eq=[1.0],
        bounds=[(0, 1)] * n_assets + [(None, None)] + [(0, None)] * n_days,
        method="highs",
        options={"time_limit": 5.0},
    )
    if not solution.success:
        raise RuntimeError(f"Expected Shortfall optimization failed: {solution.message}")
    weights = np.clip(solution.x[:n_assets], 0, 1)
    return weights / weights.sum()


def monthly_signals(dates, start_date, lag):
    """Return (signal row, fill date) pairs for month-end signals."""
    months = dates.to_period("M")
    signals = np.flatnonzero(months[:-1] != months[1:])
    return [
        (row, dates[row + lag])
        for row in signals
        if row + lag < len(dates) and dates[row + lag] >= start_date
    ]


def build_targets(returns, signals, parameters):
    if signals and signals[0][0] + 1 < parameters.lookback_sessions:
        raise ValueError("Insufficient warm-up prices for the first scheduled signal.")
    scenarios = returns.to_numpy()
    targets = {}
    for signal_row, fill_date in signals:
        stop = signal_row + 1
        window = scenarios[stop - parameters.lookback_sessions : stop]
        targets[fill_date] = optimal_weights(window, parameters)
    return targets


def fill_at_close(weights, target, friction):
    """Fund fees from the portfolio while reaching post-fee target weights."""
    def traded_fraction(remaining):
        return np.abs(remaining * target - weights)

    remaining = 1.0
    if friction and np.abs(target - weights).sum() > 0:
        remaining = brentq(
            lambda fraction: fraction + friction * traded_fraction(fraction).sum() - 1,
            0.0,
            1.0,
        )
    traded = traded_fraction(remaining)
    return remaining, traded.sum(), np.count_nonzero(traded > TRADE_TOL)


def simulate(returns, targets, friction):
    """Accrue each session's return on pre-fill weights, then execute at its close."""
    weights = np.zeros(returns.shape[1])
    daily_rows = []
    for date, asset_returns in zip(returns.index, returns.to_numpy()):
        gross = 1 + weights @ asset_returns
        weights *= (1 + asset_returns) / gross
        turnover = 0.0
        trades = 0
        target = targets.get(date)
        if target is not None:
            remaining, turnover, trades = fill_at_close(weights, target, friction)
            gross *= remaining
            weights = target.copy()
        daily_rows.append((gross - 1, turnover, trades))
    return pd.DataFrame(
        daily_rows, index=returns.index, columns=["return", "turnover", "n_trades"]
    )


def metrics(daily_results, bill_returns):
    daily = daily_results["return"]
    if len(daily) < 2:
        raise ValueError("Each reporting period needs at least two return observations.")
    excess = daily - bill_returns.reindex(daily.index)
    growth = (1 + daily).cumprod().to_numpy()
    high_water = np.maximum.accumulate(np.r_[1, growth])[1:]
    daily_vol = daily.std()
    excess_vol = excess.std()
    scale = math.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(daily)) - 1),
        "ann_vol": float(scale * daily_vol),
        "sharpe": float(scale * daily.mean() / daily_vol) if daily_vol > 0 else None,
        "sharpe_excess": float(scale * excess.mean() / excess_vol) if excess_vol > 0 else None,
        "max_drawdown": float(np.min(growth / high_water - 1)),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def report_periods(daily_results, bill_returns, is_end):
    in_sample = daily_results.index <= is_end
    return {
        "is": metrics(daily_results.loc[in_sample], bill_returns),
        "oos": metrics(daily_results.loc[~in_sample], bill_returns),
        "full": metrics(daily_results, bill_returns),
    }


def robustness_cases(default, settings):
    variants = [
        ("robust_weight", replace(default, robust_weight=weight))
        for weight in settings["robust_weights"]
    ]
    variants.extend(
        ("es_confidence", replace(default, es_confidence=confidence))
        for confidence in settings["es_confidences"]
    )
    multiplier = settings["friction_multiplier"]
    if not math.isfinite(multiplier) or multiplier <= 1:
        raise ValueError("The alternative friction multiplier must exceed one.")
    variants.append((
        "higher_cost_and_slippage",
        replace(
            default,
            cost_bps=default.cost_bps * multiplier,
            slippage_bps=default.slippage_bps * multiplier,
        ),
    ))
    tried = {default}
    for label, parameters in variants:
        if parameters not in tried:
            tried.add(parameters)
            yield label, parameters


def print_summary(label, summary):
    print(f"{label}: {json.dumps(summary, allow_nan=False)}")


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    strategy = config["strategy"]
    universe = config["universe"]
    risk_free = config["risk_free"]
    symbols = [*universe, risk_free]
    if risk_free != "BIL" or len(universe) < 2 or len(set(symbols)) != len(symbols):
        raise ValueError("Use distinct ETF symbols and reserve BIL for the cash proxy.")
    if any(not symbol.isascii() or not symbol.isalpha() or not symbol.isupper() for symbol in symbols):
        raise ValueError("Use uppercase US ETF ticker symbols.")
    price_start, start_date, is_end, end_date = (
        pd.Timestamp(config[key]) for key in ("price_start", "start_date", "is_end", "end_date")
    )
    lag = config["execution_lag_sessions"]
    if not price_start < start_date < is_end < end_date or type(lag) is not int or lag < 1:
        raise ValueError("Check date ordering and set the execution lag to at least one session.")
    default = Parameters(**{key: config[key] for key in Parameters.__annotations__})
    cases = list(robustness_cases(default, config["robustness"]))
    prices, fetched_at = load_prices(symbols, price_start, end_date)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    bill_returns = returns[risk_free]
    asset_returns = returns[universe]
    study_returns = asset_returns.loc[start_date:end_date]
    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")

    # Complete the sensitivity cases using IS histories before evaluating OOS.
    is_returns = asset_returns.loc[:is_end]
    is_signals = monthly_signals(is_returns.index, start_date, lag)
    is_study = study_returns.loc[:is_end]
    is_bills = bill_returns.loc[:is_end]
    if not is_signals:
        raise ValueError("The in-sample period must contain a scheduled fill.")
    is_targets = build_targets(is_returns, is_signals, default)
    print("ROBUSTNESS (IS only, fixed defaults for official metrics)")
    print_summary("default", metrics(simulate(is_study, is_targets, default.friction), is_bills))
    for label, parameters in cases:
        case_targets = (
            is_targets if label == "higher_cost_and_slippage"
            else build_targets(is_returns, is_signals, parameters)
        )
        case_results = simulate(is_study, case_targets, parameters.friction)
        print_summary(label, {
            "es_confidence": parameters.es_confidence,
            "robust_weight": parameters.robust_weight,
            "cost_bps": parameters.cost_bps,
            "slippage_bps": parameters.slippage_bps,
            **metrics(case_results, is_bills),
        })
    print(f"PARAM_COMBOS_TRIED: {1 + len(cases)}")

    signals = monthly_signals(asset_returns.index, start_date, lag)
    oos_signals = [(row, date) for row, date in signals if date > is_end]
    targets = is_targets | build_targets(asset_returns, oos_signals, default)
    daily_results = simulate(study_returns, targets, default.friction)
    summary = report_periods(daily_results, bill_returns, is_end)
    even_weights = np.full(len(universe), 1 / len(universe))
    bench_targets = {date: even_weights for date in targets}
    bench_results = simulate(study_returns, bench_targets, default.friction)
    benchmark = {
        "name": "Monthly equal-weight ETF universe",
        **report_periods(bench_results, bill_returns, is_end),
    }
    for period, values in summary.items():
        print_summary(period.upper(), values)
    print_summary("BENCHMARK", benchmark)
    results = {
        "strategy": strategy,
        "data_range": {
            "start": study_returns.index[0].date().isoformat(),
            "end": study_returns.index[-1].date().isoformat(),
        },
        **summary,
        "benchmark": benchmark,
    }
    print_summary("RESULTS_JSON", results)


if __name__ == "__main__":
    signal.alarm(RUN_LIMIT_SECONDS)
    main()
    signal.alarm(0)
