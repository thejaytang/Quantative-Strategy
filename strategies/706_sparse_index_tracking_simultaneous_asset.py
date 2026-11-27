"""706_sparse_index_tracking_simultaneous_asset
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/706_sparse_index_tracking_simultaneous_asset.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 9532cd9f815cc2d43ceb143b2f1416acf34e9864 on 2026-11-27 20:01 Europe/Oslo, exit 0, output sha256 8b215a0c075ea538a66bc8d04e559cba32983129c5fc9065344e1df10104c076
Data source: yfinance 获取的免费 Yahoo Finance ETF 日线，含开盘、收盘和公司行动，本地缓存于 data_cache/。资产池为 XLB、XLE、XLF、XLI、XLK、XLP、XLU、XLV、XLY，SPY 为基准，BIL 为超额收益计算的现金代理。
Data fetched: 2026-11-27T19:01:59+00:00
Sample period: 实际回测为 2012-01-03 至 2025-12-31，请求的预热数据自 2010-01-01 起。   data_range (from output): ['2012-01-03', '2025-12-31']
IS/OOS split: IS 为 2012-01-03 至 2018-12-31，OOS 为 2019-01-02 至 2025-12-31。敏感性测试仅用 IS 数据。正式路径使用默认参数跨界延续持仓，OOS 继续以当时可用数据滚动拟合。程序逐日精确比较独立 IS 结果与连续路径的 IS 前段。
Costs and slippage: 默认每笔买入或卖出按开盘成交金额扣除 2 bp 成本及 5 bp 固定滑点，含首笔建仓。滑点建模为现金扣减。两项费率均乘以买卖绝对金额之和，并纳入自融资预算求解。除息再投资按无摩擦处理。IS 双倍摩擦变体使用 4 bp 成本及 10 bp 滑点。
Signal lag: 默认在 t 日收盘后，使用截至该收盘的 200 个交易日的日收益与当日收盘持仓计算信号，目标于下一交易日 t+1 开盘模拟成交。
Price adjustment: yfinance split-adjusted OHLC; explicit ex-date distributions reinvested at the open
Survivorship bias: 据 README，资产池固定为 1998 年推出的九只初始 Select Sector SPDR ETF，名单在样本前可知。代码对固定名单取数，发现日线字段不完整时中止。范围局限于这一发行批次，发行家族的选择和已退市产品仍可能带来幸存者偏差。
Parameter combinations tried: 12
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.12392895877573484, ann_vol=0.1282074944385956, sharpe=0.9756501204267417, sharpe_excess=0.9503533485386687, max_drawdown=-0.1959726614697871, total_return=1.2613440878979842, turnover=2.2413778420655985, n_trades=56
    oos: cagr=0.16681852395183694, ann_vol=0.19883699990152803, sharpe=0.8758914813775883, sharpe_excess=0.7486068425706528, max_drawdown=-0.3428814514005153, total_return=1.9373768085306837, turnover=1.5351410163245245, n_trades=54
    full: cagr=0.1451729689027006, ann_vol=0.1672755029489052, sharpe=0.8944683665858016, sharpe_excess=0.8092321251368877, max_drawdown=-0.34288145140051496, total_return=5.642419679899429, turnover=3.776518858390123, n_trades=110
    benchmark: cagr=0.14744348304141042, ann_vol=0.16667109665888327, sharpe=0.9089487857562801, sharpe_excess=0.8234151995789022, max_drawdown=-0.3372273754869103, total_return=5.828762680210312, turnover=0.9993004896572397, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 706_sparse_index_tracking_simultaneous_asset\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nbenchmark: SPY\nrisk_free_proxy: BIL\n\ndates:\n  download_start: '2010-01-01'\n  start: '2012-01-01'\n  is_end: '2018-12-31'\n  end: '2025-12-31'\n\nparameters:\n  lookback_sessions: 200\n  rebalance_sessions: 100\n  signal_lag_sessions: 1\n  initial_max_assets: 5\n  max_updates: 3\n  max_weight: 0.50\n  cost_bps: 2.0\n  slippage_bps: 5.0\n\n# Each entry overrides the default parameters for one IS-only run.\nrobustness:\n  max_updates_2: {max_updates: 2}\n  max_updates_4: {max_updates: 4}\n  lookback_160: {lookback_sessions: 160}\n  lookback_240: {lookback_sessions: 240}\n  double_friction: {cost_bps: 4.0, slippage_bps: 10.0}\n\n# Completed portfolio-sparse prototype trials before adopting turnover sparsity.\nprior_research_combos: 6\n"
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
#!/usr/bin/env python3
"""C273, a cash-funded sparse ETF tracker for research.

Paper: Yamagata and Ono (2023), https://arxiv.org/abs/2309.10152
Related implementations consulted for research only, no code or data copied:
  sparseIndexTracking (GPL-3.0): https://github.com/dppalomar/sparseIndexTracking
  SparseIndex (license not identified): https://github.com/ericwayman/SparseIndex
  Sparse Index Tracker (MIT): https://github.com/PratyushGupta7/Sparse-Index-Tracker
  QuantConnect tutorial: https://www.quantconnect.com/docs/v2/research-environment/applying-research/sparse-optimization
  Riskfolio-Lib (BSD-3-Clause): https://pypi.org/project/riskfolio-lib/
Additional references screened:
  CRAN sparseIndexTracking: https://cran.r-project.org/package=sparseIndexTracking
  Sparse group regression: https://github.com/axsaucedo/sparse
  Graph-based Sparse Index Tracking: https://github.com/Ananya05iit/Sparse-Index-Tracking
"""

import gzip
import hashlib
import json
import signal
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
QUOTE_FIELDS = ("Open", "Close", "Dividends", "Capital Gains", "Stock Splits")
SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class Parameters:
    lookback_sessions: int
    rebalance_sessions: int
    signal_lag_sessions: int
    initial_max_assets: int
    max_updates: int
    max_weight: float
    cost_bps: float
    slippage_bps: float

    def validate(self, universe_size):
        if self.lookback_sessions < 2 or min(self.rebalance_sessions, self.signal_lag_sessions) < 1:
            raise ValueError("Use a positive rebalance interval and lag, and at least two training returns.")
        if not 1 <= self.initial_max_assets <= universe_size or not 1 <= self.max_updates <= universe_size:
            raise ValueError("Cardinality limits must be within the size of the universe.")
        if not 1 / self.initial_max_assets <= self.max_weight <= 1:
            raise ValueError("The weight cap cannot fund the starting portfolio.")
        if not (0 <= self.cost_bps and 0 <= self.slippage_bps and self.cost_bps + self.slippage_bps < 10_000):
            raise ValueError("Costs and slippage must be nonnegative and add to less than 10,000 bps.")


@dataclass(frozen=True)
class Target:
    signal_row: int
    adjustable: np.ndarray
    weights: np.ndarray


def prepare_bars(raw, tickers, start, end):
    if raw.empty:
        raise ValueError("Empty Yahoo response.")
    columns = pd.MultiIndex.from_product([QUOTE_FIELDS, tickers], names=["Price", "Ticker"])
    bars = raw.reindex(columns=columns).sort_index()
    if "Capital Gains" not in raw.columns.get_level_values(0):
        bars.loc[:, "Capital Gains"] = 0.0
    if not bars.index.is_unique or not np.isfinite(bars.to_numpy(dtype=float)).all():
        raise ValueError("Duplicate sessions or incomplete daily data for the fixed ETF set.")
    if (bars[["Open", "Close"]] <= 0).any().any() or (bars[list(QUOTE_FIELDS[2:])] < 0).any().any():
        raise ValueError("Nonpositive price or negative cash distribution or split ratio.")
    first, last = bars.index[[0, -1]]
    if not 0 <= (first - start).days <= 7 or not 0 <= (end - last).days <= 7:
        raise ValueError(f"Yahoo data do not cover the requested range ({first.date()} to {last.date()}).")
    return bars


def load_bars(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    key = "|".join(["yf-daily-actions-v1", start.date().isoformat(), end.date().isoformat(), *sorted(tickers)])
    cache_file = CACHE / f"yahoo_{hashlib.sha256(key.encode()).hexdigest()[:16]}.csv.gz"
    if cache_file.exists():
        with gzip.open(cache_file, "rt", encoding="utf-8") as cached:
            fetched_at = datetime.fromisoformat(cached.readline().strip()).isoformat()
            raw = pd.read_csv(cached, header=[0, 1], index_col=0, parse_dates=True, dtype=str).astype(float)
        return prepare_bars(raw, tickers, start, end), fetched_at

    with TemporaryDirectory(dir=CACHE) as yf_cache:
        yf.set_tz_cache_location(yf_cache)
        for attempt in range(3):
            raw = yf.download(
                tickers, start=start.date().isoformat(), end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d", auto_adjust=False, actions=True, progress=False, timeout=15,
            )
            try:
                bars = prepare_bars(raw, tickers, start, end)
                break
            except ValueError:
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(dir=CACHE, suffix=".csv.gz", delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        with gzip.open(temp_path, "wt", encoding="utf-8") as output:
            output.write(fetched_at + "\n")
            bars.to_csv(output)
        temp_path.replace(cache_file)
    finally:
        temp_path.unlink(missing_ok=True)
    return bars, fetched_at


def choose_updates(asset_returns, index_returns, reference, max_changes, max_weight):
    """Fit a bounded least-squares allocation for each permitted set of changes."""
    asset_returns = 100.0 * asset_returns
    index_returns = 100.0 * index_returns
    bounds = [(0.0, max_weight)] * max_changes
    best_loss = np.inf
    best = None
    for names in combinations(range(asset_returns.shape[1]), max_changes):
        adjustable = np.asarray(names)
        fixed_weights = reference.copy()
        fixed_weights[adjustable] = 0.0
        budget = max(0.0, 1.0 - fixed_weights.sum())
        if np.any(fixed_weights > max_weight + 1e-12) or budget > max_changes * max_weight + 1e-12:
            continue
        remaining_target = index_returns - asset_returns @ fixed_weights
        subset = asset_returns[:, adjustable]

        def loss(weights):
            residual = subset @ weights - remaining_target
            return (residual @ residual) / len(residual), 2.0 * (subset.T @ residual) / len(residual)

        if budget <= 1e-12:
            weights = np.zeros(max_changes)
        else:
            constraint = {"type": "eq", "fun": lambda w: w.sum() - budget, "jac": lambda w: np.ones_like(w)}
            result = minimize(
                loss, np.full(max_changes, budget / max_changes), jac=True, method="SLSQP",
                constraints=constraint, bounds=bounds, options={"ftol": 1e-12, "maxiter": 100},
            )
            if not result.success or abs(result.x.sum() - budget) > 1e-8 or np.any(result.x < -1e-10):
                raise RuntimeError(f"Bounded tracking solve failed on support {names}: {result.message}")
            weights = result.x.clip(0.0, max_weight)
            weights[weights < 1e-12] = 0.0
        error, _ = loss(weights)
        if error < best_loss:
            best_loss = error
            best = adjustable, weights / weights.sum() if budget > 1e-12 else weights
    if best is None:
        raise RuntimeError("No feasible set of changes for the weight and turnover caps.")
    return best


def opening_rebalance(open_values, cash, weights, cost_rate, slippage_rate):
    """Find the post-friction investment value using opening notionals only."""
    total = open_values.sum() + cash
    lower, upper = 0.0, total
    for _ in range(52):
        proposed = (lower + upper) / 2.0
        change = proposed * weights - open_values
        gross = np.abs(change).sum()
        remaining = total - (proposed * weights).sum() - gross * cost_rate - gross * slippage_rate
        if remaining >= 0.0:
            lower = proposed
        else:
            upper = proposed
    next_values = lower * weights
    changes = next_values - open_values
    gross = np.abs(changes).sum()
    next_cash = total - next_values.sum() - gross * cost_rate - gross * slippage_rate
    if next_cash < 0.0:
        raise RuntimeError("Rebalance would produce negative cash.")
    trades = np.count_nonzero(np.abs(changes) > 1e-10 * total)
    return next_values, next_cash, gross, trades


def simulate(bars, tickers, first_row, params, total_returns=None, benchmark=None):
    opens = bars["Open"][tickers].to_numpy()
    closes = bars["Close"][tickers].to_numpy()
    distributions = (bars["Dividends"][tickers] + bars["Capital Gains"][tickers]).to_numpy()
    lag = params.signal_lag_sessions
    first_signal = first_row - lag
    if first_signal < 0:
        raise ValueError("Insufficient data for the configured signal lag.")
    signals = {}
    targets = {}
    if total_returns is None:
        targets[first_row] = Target(first_signal, np.array([0]), np.ones(1))
    else:
        if first_signal - params.lookback_sessions + 1 < 1:
            raise ValueError("Insufficient warm-up for the first lagged training window.")
        signals = {row - lag: row for row in range(first_row, len(bars), params.rebalance_sessions)}
        asset_returns = total_returns[tickers].to_numpy()
        index_returns = total_returns[benchmark].to_numpy()
    positions = np.zeros(len(tickers))
    cash, previous_value = 1.0, 1.0
    peak_trades_after_entry = 0
    cost_rate, slippage_rate = params.cost_bps / 10_000.0, params.slippage_bps / 10_000.0
    records = []
    for row in range(first_signal, len(bars)):
        opening_prices = opens[row]
        # Reinvest ex-date distributions for positions carried into this session, before new fills.
        positions += positions * (distributions[row] / opening_prices)
        open_values = positions * opening_prices
        turnover, trades = 0.0, 0
        if row in targets:
            target = targets.pop(row)
            if row - target.signal_row != lag:
                raise RuntimeError("Target violates the configured signal lag.")
            if target.weights.any():
                total_at_open = open_values.sum() + cash
                adjustable = target.adjustable
                next_values, cash, gross, trades = opening_rebalance(
                    open_values[adjustable], cash, target.weights, cost_rate, slippage_rate,
                )
                positions[adjustable] = next_values / opening_prices[adjustable]
                turnover = gross / total_at_open
            if row != first_row:
                peak_trades_after_entry = max(peak_trades_after_entry, trades)
        closing_values = positions * closes[row]
        closing_value = closing_values.sum() + cash
        if row >= first_row:
            records.append((closing_value / previous_value - 1.0, turnover, trades))
        previous_value = closing_value
        if row in signals:
            fill_row = signals[row]
            invested_value = closing_values.sum()
            reference = closing_values / invested_value if invested_value > 0.0 else closing_values
            change_limit = params.initial_max_assets if fill_row == first_row else params.max_updates
            window = slice(row - params.lookback_sessions + 1, row + 1)
            adjustable, weights = choose_updates(
                asset_returns[window], index_returns[window], reference, change_limit, params.max_weight,
            )
            targets[fill_row] = Target(row, adjustable, weights)
    if peak_trades_after_entry > params.max_updates:
        raise RuntimeError("Simulated fills exceed the turnover cardinality cap.")
    return pd.DataFrame(records, index=bars.index[first_row:], columns=["return", "turnover", "n_trades"])


def annualized_sharpe(returns):
    volatility = returns.std(ddof=1)
    return float(np.sqrt(SESSIONS_PER_YEAR) * returns.mean() / volatility) if volatility > 0 else None


def metrics(daily_results, risk_free):
    returns = daily_results["return"]
    growth = np.r_[1.0, (1.0 + returns).cumprod().to_numpy()]
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1.0),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": annualized_sharpe(returns),
        "sharpe_excess": annualized_sharpe(returns - risk_free.loc[returns.index]),
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1.0).min()),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def tracking_error(daily_results, comparison):
    active = daily_results["return"] - comparison.loc[daily_results.index, "return"]
    return active.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)


def run():
    started = time.monotonic()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe, benchmark, proxy = cfg["universe"], cfg["benchmark"], cfg["risk_free_proxy"]
    tickers = [*universe, benchmark, proxy]
    if len(set(tickers)) != len(tickers):
        raise ValueError("Each ETF, the benchmark and the risk-free proxy must be distinct.")
    params = Parameters(**cfg["parameters"])
    variants = {name: replace(params, **overrides) for name, overrides in cfg["robustness"].items()}
    for case in [params, *variants.values()]:
        case.validate(len(universe))
    start, is_end, end = (pd.Timestamp(cfg["dates"][key]) for key in ("start", "is_end", "end"))
    download_start = pd.Timestamp(cfg["dates"]["download_start"])
    if not download_start < start < is_end < end:
        raise ValueError("Require download_start < start < is_end < end.")

    bars, fetched_at = load_bars(tickers, download_start, end)
    first_row = bars.index.searchsorted(start)
    is_stop = bars.index.searchsorted(is_end, side="right")
    if min(is_stop - first_row, len(bars) - is_stop) < 2:
        raise ValueError("Both reporting periods need at least two market sessions.")
    opens, closes = bars["Open"], bars["Close"]
    distributions = bars["Dividends"] + bars["Capital Gains"]
    total_returns = (opens + distributions) / closes.shift(1) * (closes / opens) - 1.0
    risk_free = total_returns[proxy].iloc[first_row:].copy()
    risk_free.iloc[0] = closes[proxy].iloc[first_row] / opens[proxy].iloc[first_row] - 1.0
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance split-adjusted OHLC; explicit ex-date distributions reinvested at the open")

    is_bars = bars.iloc[:is_stop]
    is_returns = total_returns.iloc[:is_stop]
    is_benchmark = simulate(is_bars, [benchmark], first_row, params)
    default_is = simulate(is_bars, universe, first_row, params, is_returns, benchmark)
    trials = 1
    print("ROBUSTNESS (IS only; annualized tracking error versus baseline SPY)")
    print(f"default: TE={tracking_error(default_is, is_benchmark):.2%} {json.dumps(metrics(default_is, risk_free))}")
    for name, variant in variants.items():
        daily_results = simulate(is_bars, universe, first_row, variant, is_returns, benchmark)
        trials += 1
        print(f"{name}: TE={tracking_error(daily_results, is_benchmark):.2%} {json.dumps(metrics(daily_results, risk_free))}")
    print(f"PARAM_COMBOS_THIS_RUN: {trials}")
    print(f"PARAM_COMBOS_TRIED: {trials + cfg['prior_research_combos']}")

    full = simulate(bars, universe, first_row, params, total_returns, benchmark)
    comparison = simulate(bars, [benchmark], first_row, params)
    pd.testing.assert_frame_equal(full.iloc[:len(default_is)], default_is, check_exact=True)
    print("LOOK_AHEAD_SELF_CHECK: trailing closes through signal session; lag checked on every fill; "
          "opening prices size fills; no delayed releases used; split-adjusted prices and ex-date actions; "
          "IS-only rerun matches the full run's IS prefix exactly")

    periods = {"is": default_is, "oos": full.loc[full.index > is_end], "full": full}
    period_metrics = {name: metrics(period, risk_free) for name, period in periods.items()}
    if any(period["n_trades"] <= 0 for period in period_metrics.values()):
        raise RuntimeError("Expected simulated fills in each reported period.")
    print(f"EXECUTION: entry fills {full['n_trades'].iloc[0]}, "
          f"maximum fills per later opening {full['n_trades'].iloc[1:].max()}")
    result = {
        "strategy": cfg["strategy"],
        "data_range": [full.index[0].date().isoformat(), full.index[-1].date().isoformat()],
        **period_metrics,
        "benchmark": metrics(comparison, risk_free),
    }
    for name, period in periods.items():
        benchmark_metrics = metrics(comparison.loc[period.index], risk_free)
        print(f"{name.upper()} {period.index[0].date()} to {period.index[-1].date()}: "
              f"TE={tracking_error(period, comparison):.2%} {json.dumps(period_metrics[name])}")
        print(f"BENCHMARK {benchmark} {name.upper()}: {json.dumps(benchmark_metrics)}")
    print(f"已回测 | pending review | runtime {time.monotonic() - started:.2f}s")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    run()
    signal.alarm(0)
