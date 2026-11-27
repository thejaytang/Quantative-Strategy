"""671_high_dimensional_covariance_matrix_estim
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/671_high_dimensional_covariance_matrix_estim.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 9532cd9f815cc2d43ceb143b2f1416acf34e9864 on 2026-11-27 19:59 Europe/Oslo, exit 0, output sha256 7f5e5ce8b6bb6ac623c839cc4c949e80f4da12553723c2ed201ab4256f5e52a2
Data source: Yahoo Finance 免费日线（yfinance）。使用开盘价、收盘价、现金股息和资本利得分派，交易日取 SPY 有收盘价的日期。52 只配置 ETF 为 SPY IWM MDY QQQ VTV VUG XLB XLE XLF XLI XLK XLP XLU XLV XLY XRT XHB SMH EWJ EWG EWU EWC EWA EWH EWT EWY EEM EFA VEA VWO EWW EWZ INDA EZA EWS IEI IEF TLT TIP LQD HYG MUB EMB BND BNDX GLD SLV DBC USO VNQ RWX DBA。另取 BIL 日总回报，仅用于计算超额夏普。证据：。
Data fetched: 2026-11-27T18:59:36+00:00
Sample period: 正式回测交易日为 2016-01-04 至 2025-12-31（含首尾）。行情请求自 2015-01-01 起，为滚动估计提供预热。证据：。   data_range (from output): {'start': '2016-01-04', 'end': '2025-12-31'}
IS/OOS split: 以 2022-01-01 为界，IS 为此前交易日（2016-01-04 至 2021-12-31），OOS 为该日起至 2025-12-31 的交易日。固定正式配置只模拟一条连续路径，跨界延续头寸。四个稳健性变体只运行 IS。OOS 是历史时间切分，部分日期早于论文发表。证据：。
Costs and slippage: 正式配置对买入和卖出的名义金额均计 5 bp 手续费和 5 bp 固定滑点，合计单边 10 bp。按开盘估值求扣费后的可投资金额，并从现金扣除总摩擦率乘以买入加卖出的名义金额。滑点作为固定成本建模，成交估值仍用开盘价。IS 双倍摩擦变体为单边 10 bp 手续费加 10 bp 滑点。证据：。
Signal lag: 每月最后一个交易日 t 收盘后，用截至 t 的日收盘总回报估计目标权重。配置延迟 1 个交易日，在下一 SPY 交易日开盘调仓。证据：。
Price adjustment: yfinance split-adjusted OHLC, forward total return with ex-date open reinvestment
Survivorship bias: 资产池由固定 ETF 名单指定，全样本沿用同一名单并要求各标的行情有效。README 明确记录事后选样及基金幸存者偏差。证据：。
Parameter combinations tried: 5
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.03306185451380084, ann_vol=0.04517891713608152, sharpe=0.7427271161119084, sharpe_excess=0.5634049498131498, max_drawdown=-0.1503110316667, total_return=0.21535148806227422, turnover=1.7010089952569478, n_trades=1338
    oos: cagr=0.019071194617018783, ann_vol=0.05539146597356734, sharpe=0.3687387210092714, sharpe_excess=-0.327860854191684, max_drawdown=-0.1488694869833268, total_return=0.07809073861441607, turnover=1.7901723036245578, n_trades=838
    full: cagr=0.027457176956952933, ann_vol=0.04949758143418061, sharpe=0.5720872497919134, sharpe_excess=0.16341324102625363, max_drawdown=-0.1503110316667, total_return=0.3102591834411732, turnover=1.7365821051585837, n_trades=2176
    benchmark: cagr=0.0987897134710074, ann_vol=0.1348717606002169, sharpe=0.7664726634231989, sharpe_excess=0.6164112484477966, max_drawdown=-0.3047536332729611, total_return=1.5595975958888455, turnover=0.44163590400554925, n_trades=6240
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: '671_ high_dimensional_covariance_matrix_estim'\n\ndates:\n  data_start: '2015-01-01'\n  backtest_start: '2016-01-01'\n  oos_start: '2022-01-01'\n  end: '2025-12-31'\n\nuniverse:\n  calendar: SPY\n  risk_free: BIL\n  tickers: >-\n    SPY IWM MDY QQQ VTV VUG\n    XLB XLE XLF XLI XLK XLP XLU XLV XLY XRT XHB SMH\n    EWJ EWG EWU EWC EWA EWH EWT EWY EEM EFA VEA VWO\n    EWW EWZ INDA EZA EWS\n    IEI IEF TLT TIP LQD HYG MUB EMB BND BNDX\n    GLD SLV DBC USO VNQ RWX DBA\n\nexecution:\n  lag_sessions: 1\n  cost_bps: 5\n  slippage_bps: 5\n\nparameters:\n  lookback_sessions: 104\n  max_weight: 0.10\n  estimator: alca_lp\n\nrobustness:\n  shorter_window: {lookback_sessions: 84}\n  longer_window: {lookback_sessions: 126}\n  alca_only: {estimator: alca}\n  double_friction: {cost_bps: 10, slippage_bps: 10}\n"
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
"""Monthly ALCA and nonlinear spectral shrinkage for a long-only ETF portfolio.

Paper (García-Medina, 2024): https://arxiv.org/abs/2412.08756
Author's data reference: https://github.com/agarciam/High-dimensional-covariance-matrix-estimators-on-simulated-portfolios_empirical_data
Research references for the hierarchy (BAHC, PyPI MIT metadata):
https://github.com/bongiornoc/BAHC/blob/main/bahc/bahc.py
https://pypi.org/project/bahc/2.0.0/
Nonlinear shrinkage reference (Jarosz, MIT):
https://github.com/yedrek/shrinkage/blob/main/shrinkage/rie_lp.py
https://github.com/yedrek/shrinkage
Allocation comparisons:
https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/pypfopt/hierarchical_portfolio.py
https://github.com/QuantConnect/Lean/blob/master/Algorithm.Framework/Portfolio/RiskParityPortfolioOptimizer.py
Implementation and backtest are independent of those projects.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.optimize import brentq, minimize
from scipy.spatial.distance import squareform
from threadpoolctl import threadpool_limits
import yaml
import yfinance as yf


BASE = Path(__file__).resolve().parent
CACHE = BASE / "data_cache"
SESSIONS_PER_YEAR = 252
PRICE_FIELDS = ("Open", "Close", "Dividends", "Capital Gains")


@dataclass(frozen=True)
class Parameters:
    lookback_sessions: int
    max_weight: float
    estimator: str
    cost_bps: float
    slippage_bps: float

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000

    def validate(self, size):
        if not isinstance(self.lookback_sessions, int) or self.lookback_sessions <= size:
            raise ValueError("This estimator requires more observations than assets")
        if self.estimator not in {"alca", "alca_lp"}:
            raise ValueError(f"Unknown covariance estimator: {self.estimator}")
        if not 1 / size <= self.max_weight <= 1:
            raise ValueError("Infeasible target weight limit")
        if not (0 <= self.cost_bps and 0 <= self.slippage_bps and self.friction < 1):
            raise ValueError("Invalid per-side execution friction")


def download_prices(tickers, data_start, end, calendar):
    request = f"ohlc-actions-v1|{data_start.date()}|{end.date()}|{calendar}|{','.join(tickers)}"
    cache_file = CACHE / f"yahoo_{hashlib.sha256(request.encode()).hexdigest()[:16]}.csv.gz"
    if cache_file.exists():
        with gzip.open(cache_file, "rt", encoding="utf-8") as cached:
            fetched_at = datetime.fromisoformat(cached.readline().strip()).isoformat()
            raw = pd.read_csv(cached, header=[0, 1], index_col=0, parse_dates=True)
        validate_prices(raw, tickers, data_start, end, calendar)
        return raw, fetched_at

    CACHE.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=CACHE) as download_cache:
        yf.set_tz_cache_location(download_cache)
        raw = yf.download(
            tickers, start=data_start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(), interval="1d",
            auto_adjust=False, actions=True, ignore_tz=True,
            threads=6, progress=False, timeout=12,
        )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no price data")
    raw = raw.sort_index()
    equity_days = raw["Close"][calendar].dropna().index
    required = pd.MultiIndex.from_product([PRICE_FIELDS[:-1], tickers])
    optional = pd.MultiIndex.from_product([PRICE_FIELDS[-1:], tickers])
    raw = raw.reindex(index=equity_days, columns=required.append(optional))
    raw["Capital Gains"] = raw["Capital Gains"].fillna(0.0)
    validate_prices(raw, tickers, data_start, end, calendar)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with NamedTemporaryFile(dir=CACHE, delete_on_close=False) as temp:
        with gzip.open(temp.name, "wt", encoding="utf-8", newline="") as output:
            output.write(f"{fetched_at}\n")
            raw.to_csv(output)
        Path(temp.name).replace(cache_file)
    return raw, fetched_at


def validate_prices(raw, tickers, data_start, end, calendar):
    if (raw.empty or not raw.index.is_monotonic_increasing or not raw.index.is_unique
            or raw.index[0] < data_start or raw.index[-1] > end
            or raw.index[0] - data_start > pd.Timedelta(days=7)
            or end - raw.index[-1] > pd.Timedelta(days=7)):
        raise RuntimeError(f"Invalid {calendar} session range in Yahoo Finance data")
    for field in PRICE_FIELDS:
        values = raw[field].loc[:, tickers]
        invalid = ~np.isfinite(values) | (values <= 0 if field in ("Open", "Close") else values < 0)
        if invalid.to_numpy().any():
            row, column = np.argwhere(invalid.to_numpy())[0]
            raise RuntimeError(f"Missing or invalid {field} for {values.columns[column]} "
                               f"on {values.index[row].date()}")


def total_return_prices(raw):
    opens = raw["Open"]
    distributions = raw["Dividends"] + raw["Capital Gains"]
    # Reinvest ex-date entitlements at the open using split-consistent dollars.
    units_per_base_unit = (1.0 + distributions / opens).cumprod()
    return opens * units_per_base_unit, raw["Close"] * units_per_base_unit


def estimate_covariance(window, estimator):
    sample = np.cov(window, rowvar=False, ddof=1)
    sigmas = np.sqrt(np.diag(sample))
    if not np.all(np.isfinite(sample)) or np.any(sigmas <= 0):
        raise ValueError("Covariance window has missing or constant returns")

    scale = np.outer(sigmas, sigmas)
    distance = np.clip(1.0 - sample / scale, 0.0, 2.0)
    condensed = distance[np.triu_indices_from(distance, k=1)]
    tree = linkage(condensed, method="average")
    filtered = scale * (1.0 - squareform(cophenet(tree)))
    eigenvalues, eigenvectors = np.linalg.eigh(filtered)
    eigenvalues = np.maximum(eigenvalues, 1e-6 * eigenvalues[-1])

    if estimator == "alca_lp":
        # Eq. (16) evaluated off the real axis with a finite, relative bandwidth.
        sample_size = len(window) - 1
        aspect = len(eigenvalues) / sample_size
        evaluation_points = eigenvalues * (1.0 - 1j * sample_size ** (-1 / 3))
        transform = (1.0 / (evaluation_points[:, None] - eigenvalues[None, :])).mean(axis=1)
        eigenvalues = eigenvalues / np.abs(1.0 - aspect + aspect * eigenvalues * transform) ** 2
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def minimum_variance(covariance, max_weight):
    size = len(covariance)
    scaled = covariance / np.diag(covariance).mean()
    result = minimize(
        lambda weights: 0.5 * weights @ scaled @ weights,
        np.full(size, 1 / size), jac=lambda weights: scaled @ weights,
        bounds=[(0.0, max_weight)] * size,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1,
                     "jac": lambda weights: np.ones_like(weights)},
        method="SLSQP", options={"ftol": 1e-12, "maxiter": 200},
    )
    if (not result.success or abs(result.x.sum() - 1) > 1e-8
            or result.x.min() < -1e-8 or result.x.max() > max_weight + 1e-8):
        raise RuntimeError(f"Minimum-variance optimization failed: {result.message}")
    weights = np.clip(result.x, 0, max_weight)
    weights[weights < 1e-12] = 0.0
    return weights / weights.sum()


def make_targets(returns, start, lag, parameters, equal_weight=False):
    sessions = returns.index
    months = sessions.to_period("M")
    # A month boundary identifies the previous session as its last trading day.
    month_ends = np.flatnonzero(months[1:] != months[:-1])
    targets = {}
    for observation in month_ends:
        fill = observation + lag
        if fill >= len(sessions) or sessions[fill] < start:
            continue
        if observation < parameters.lookback_sessions:
            raise ValueError("Insufficient pre-start price history for the configured window")
        window = returns.iloc[observation + 1 - parameters.lookback_sessions:observation + 1]
        if equal_weight:
            weights = np.full(returns.shape[1], 1.0 / returns.shape[1])
        else:
            weights = minimum_variance(
                estimate_covariance(window.to_numpy(), parameters.estimator), parameters.max_weight,
            )
        targets[sessions[fill]] = (sessions[observation], weights)
    return targets


def rebalance(positions, cash, opening_prices, weights, friction):
    existing = positions * opening_prices
    available = existing.sum() + cash
    existing_fraction = existing / available

    def budget(invested_fraction):
        changed_fraction = np.abs(invested_fraction * weights - existing_fraction).sum()
        return invested_fraction + friction * changed_fraction - 1.0

    # A tiny buffer covers floating-point rounding after the self-financing solve.
    invested = available * brentq(budget, 0.0, 1.0, xtol=1e-14) * (1.0 - 1e-12)
    target_values = invested * weights
    trade_sizes = np.abs(target_values - existing)
    cash = available - target_values.sum() - friction * trade_sizes.sum()
    if cash < 0:
        raise RuntimeError("Rebalance would leave negative cash")
    return (target_values / opening_prices, cash, trade_sizes.sum() / available,
            np.count_nonzero(trade_sizes > 1e-10 * available))


def simulate(opens, closes, targets, friction):
    positions = np.zeros(opens.shape[1])
    cash = 1.0
    previous_value = cash
    daily_results = []
    for date, opening, closing in zip(opens.index, opens.to_numpy(), closes.to_numpy(), strict=True):
        turnover = 0.0
        trades = 0
        if date in targets:
            signal_date, weights = targets[date]
            if signal_date >= date:
                raise RuntimeError("Signal timestamp must precede its execution day")
            positions, cash, turnover, trades = rebalance(positions, cash, opening, weights, friction)
        end_value = cash + positions @ closing
        daily_results.append((end_value / previous_value - 1.0, turnover, trades))
        previous_value = end_value
    return pd.DataFrame(daily_results, index=opens.index, columns=["return", "turnover", "trades"])


def metrics(daily_results, risk_free):
    returns = daily_results["return"].to_numpy()
    excess = returns - risk_free.loc[daily_results.index].to_numpy()
    path = np.cumprod(np.r_[1.0, 1.0 + returns])
    years = len(returns) / SESSIONS_PER_YEAR
    if len(returns) < 2:
        raise ValueError("Metrics require at least two trading sessions")
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(path[-1] ** (1 / years) - 1.0),
        "ann_vol": float(volatility * annualizer),
        "sharpe": float(returns.mean() / volatility * annualizer) if volatility else None,
        "sharpe_excess": float(excess.mean() / excess_volatility * annualizer) if excess_volatility else None,
        "max_drawdown": float(np.min(path / np.maximum.accumulate(path) - 1.0)),
        "total_return": float(path[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["trades"].sum()),
    }


def summarize(daily_results, oos_start, risk_free):
    return {
        "is": metrics(daily_results.loc[daily_results.index < oos_start], risk_free),
        "oos": metrics(daily_results.loc[daily_results.index >= oos_start], risk_free),
        "full": metrics(daily_results, risk_free),
    }


def report(label, result):
    print(f"{label}: {json.dumps(result, allow_nan=False, ensure_ascii=False)}", flush=True)


def main():
    config = yaml.safe_load((BASE / "config.yaml").read_text(encoding="utf-8"))
    dates = {key: pd.Timestamp(value) for key, value in config["dates"].items()}
    data_start, start = dates["data_start"], dates["backtest_start"]
    oos_start, end = dates["oos_start"], dates["end"]
    if not data_start < start < oos_start <= end:
        raise ValueError("Invalid date sequence")
    universe = config["universe"]
    tickers = universe["tickers"].split()
    calendar, risk_free_ticker = universe["calendar"], universe["risk_free"]
    if (len(tickers) < 2 or len(tickers) != len(set(tickers)) or calendar not in tickers
            or risk_free_ticker in tickers):
        raise ValueError("Invalid universe, calendar, or risk-free proxy")

    execution = config["execution"]
    lag = execution["lag_sessions"]
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("Signal lag must be at least one trading session")
    official = Parameters(**config["parameters"], cost_bps=execution["cost_bps"],
                          slippage_bps=execution["slippage_bps"])
    variants = {name: replace(official, **changes) for name, changes in config["robustness"].items()}
    trials = {official, *variants.values()}
    if len(trials) != len(variants) + 1:
        raise ValueError("Robustness cases must be distinct from each other and the official parameters")
    for parameters in trials:
        parameters.validate(len(tickers))

    raw, fetched_at = download_prices([*tickers, risk_free_ticker], data_start, end, calendar)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance split-adjusted OHLC, forward total return with ex-date open reinvestment")
    opens, closes = total_return_prices(raw)
    asset_returns = closes[tickers].pct_change(fill_method=None)
    risk_free = closes[risk_free_ticker].pct_change(fill_method=None)
    opens, closes = opens.loc[start:, tickers], closes.loc[start:, tickers]
    is_sessions = opens.index[opens.index < oos_start]
    if is_sessions.empty or opens.index[-1] < oos_start:
        raise ValueError("Both evaluation periods require trading sessions")

    with threadpool_limits(limits=1, user_api="blas"):
        for name, parameters in variants.items():
            targets = make_targets(asset_returns.loc[:is_sessions[-1]], start, lag, parameters)
            daily = simulate(opens.loc[is_sessions], closes.loc[is_sessions], targets, parameters.friction)
            report(f"ROBUSTNESS_IS_{name}", metrics(daily, risk_free))
        targets = make_targets(asset_returns, start, lag, official)
        daily = simulate(opens, closes, targets, official.friction)
        baseline = make_targets(asset_returns, start, lag, official, equal_weight=True)
        benchmark = simulate(opens, closes, baseline, official.friction)

    official_results = summarize(daily, oos_start, risk_free)
    benchmark_results = summarize(benchmark, oos_start, risk_free)
    if min(official_results[period]["n_trades"] for period in ("is", "oos")) <= 0:
        raise RuntimeError("Official backtest requires executions in both periods")
    for period in ("is", "oos", "full"):
        report(period.upper(), official_results[period])
        report(f"BENCHMARK_{period.upper()}", benchmark_results[period])
    print(f"PARAM_COMBOS_TRIED: {len(trials)}")
    print("STATUS: 已回测")
    report("RESULTS_JSON", {
        "strategy": config["strategy"],
        "data_range": {"start": daily.index[0].date().isoformat(), "end": daily.index[-1].date().isoformat()},
        **official_results,
        "benchmark": {"name": "Same-universe monthly equal weight", **benchmark_results["full"]},
    })


if __name__ == "__main__":
    signal.alarm(14 * 60)
    main()
