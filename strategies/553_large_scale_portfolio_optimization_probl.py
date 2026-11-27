"""553_large_scale_portfolio_optimization_probl
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/553_large_scale_portfolio_optimization_probl.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c0a8f76240a0431b019ab49c34040a95a765860a on 2026-11-27 10:04 Europe/Oslo, exit 0, output sha256 7842e04cd84f0f3272914400c002351b89c4a466b25fda656a689ed2b3555937
Data source: Yahoo Finance 日线，经 yfinance 下载。固定股票池含 64 只美国大盘股，SPY 为基准，BIL 为无风险收益代理。流动性代理为 Yahoo 报告的 Close × Volume。行情及下载时间缓存在 data_cache/。（config.yaml:2-68，strategy.py:53-77、84-100，README.md:20、30）
Data fetched: 2026-11-27T09:05:04+00:00
Sample period: 请求取数区间为 2010-01-01 至 2025-12-31，其中回测开始前的数据用于预热。日志记录的回测区间为 2012-01-03 至 2025-12-31。（config.yaml:69-72，strategy.py:53-60，README.md:21，logs/run_20261127_094127.log:11）   data_range (from output): {'start': '2012-01-03', 'end': '2025-12-31'}
IS/OOS split: 样内为 2012-01-03 至 2019-12-31，样外为 2020-01-02 至 2025-12-31。默认设定连续模拟，再按日期切分报告，因此样外继承样内末日的组合状态，估计窗口继续滚动。7 组稳健性变体仅在样内运行。（config.yaml:69-72、84-91，strategy.py:155-174、257-265、280-283、295-308，README.md:21、28）
Costs and slippage: 默认按每笔买卖的成交名义金额扣除 5 bps 交易成本和 5 bps 固定滑点，合计 10 bps，包含初次建仓。模拟按实际买卖金额之和从组合净值扣减费用，按扣费后净值配置目标权重。SPY 首日建仓用相同费率。样内加倍摩擦情景为交易成本和滑点各 10 bps，合计 20 bps。（config.yaml:82-83、91，strategy.py:36-38、198-218、250-254、303-307，README.md:23）
Signal lag: 月末交易日 t 收盘确定信号，默认在下一交易日收盘模拟成交。持仓或目标股票缺价时，顺延至所需收盘价齐备，期间若有新目标则覆盖待执行目标。成交日的收盘到收盘收益按调仓前仓位计，随后一个收盘到收盘区间起使用新仓位。（config.yaml:81，strategy.py:155-185、189-218，README.md:22-23）
Price adjustment: yfinance Adj Close total-return (splits and distributions)
Survivorship bias: 64 只美国大盘股的候选池为事后固定名单，缺少历史逐期标普 500 成分和已退市股票，存在幸存者偏差及成分选择偏差。筛选要求完整收益估计窗口，持仓缺价以最近报价估值，未模拟退市处置。（config.yaml:2-66，strategy.py:171-179、189-207，README.md:23、30）
Parameter combinations tried: 8
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.27091294449884806, ann_vol=0.2841775936866729, sharpe=0.9840068414595214, sharpe_excess=0.9648842267575976, max_drawdown=-0.35738267259073253, total_return=5.780688489381333, turnover=70.42003584159593, n_trades=1211
    oos: cagr=0.3036301809926476, ann_vol=0.31988952207825117, sharpe=0.9898214763386136, sharpe_excess=0.9079088140295679, max_drawdown=-0.34979228021940323, total_return=3.8876330310593508, turnover=53.68654525303154, n_trades=898
    full: cagr=0.28482755210992283, ann_vol=0.29995598945062435, sharpe=0.9850917986496949, sharpe_excess=0.93733602688618, max_drawdown=-0.46006028479831185, total_return=32.14151703402397, turnover=124.10658109462747, n_trades=2109
    benchmark SPY full: cagr=0.14758920092268113, ann_vol=0.16666963664545795, sharpe=0.9097175930464678, sharpe_excess=0.8241637566105885, max_drawdown=-0.337172753211498, total_return=5.8408860495792805, turnover=0.9990009990009991, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "553_ large_scale_portfolio_optimization_probl"\nuniverse:\n  - AAPL\n  - ABBV\n  - ABT\n  - ADBE\n  - AMD\n  - AMGN\n  - AMZN\n  - AVGO\n  - BAC\n  - BRK-B\n  - CAT\n  - CMCSA\n  - COP\n  - COST\n  - CRM\n  - CSCO\n  - CVX\n  - DE\n  - DIS\n  - DUK\n  - F\n  - GE\n  - GILD\n  - GOOG\n  - GS\n  - HD\n  - IBM\n  - INTC\n  - JNJ\n  - JPM\n  - KO\n  - LIN\n  - LLY\n  - LOW\n  - MA\n  - MCD\n  - MDLZ\n  - META\n  - MMM\n  - MRK\n  - MS\n  - MSFT\n  - NEE\n  - NFLX\n  - NVDA\n  - ORCL\n  - PEP\n  - PFE\n  - PG\n  - QCOM\n  - SBUX\n  - T\n  - TGT\n  - TMO\n  - TSLA\n  - TXN\n  - UNH\n  - UNP\n  - UPS\n  - V\n  - VZ\n  - WFC\n  - WMT\n  - XOM\nbenchmark: SPY\nrisk_free_proxy: BIL\ndata_start: 2010-01-01\nbacktest_start: 2012-01-03\nis_end: 2019-12-31\nend: 2025-12-31\nparameters:\n  lookback_sessions: 252\n  liquidity_sessions: 63\n  liquidity_top_n: 50\n  cardinality: 10\n  risk_aversion: 6.0\n  min_weight: 0.01\n  max_weight: 0.99\n  lag_sessions: 1\n  transaction_cost_bps: 5\n  slippage_bps: 5\nrobustness:\n  cardinality_8: {cardinality: 8}\n  cardinality_12: {cardinality: 12}\n  lookback_189: {lookback_sessions: 189}\n  lookback_315: {lookback_sessions: 315}\n  risk_aversion_4: {risk_aversion: 4.0}\n  risk_aversion_8: {risk_aversion: 8.0}\n  double_friction: {transaction_cost_bps: 10, slippage_bps: 10}\n'
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
"""Monthly cardinality-constrained mean-variance research backtest."""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
TRADING_DAYS = 252


@dataclass(frozen=True)
class Parameters:
    lookback_sessions: int
    liquidity_sessions: int
    liquidity_top_n: int
    cardinality: int
    risk_aversion: float
    min_weight: float
    max_weight: float
    lag_sessions: int
    transaction_cost_bps: float
    slippage_bps: float

    @property
    def friction(self):
        return (self.transaction_cost_bps + self.slippage_bps) / 10_000

    def validate(self, universe_size):
        if self.lag_sessions < 1 or self.lookback_sessions < 2 or self.liquidity_sessions < 1:
            raise ValueError("Use a positive execution lag and valid estimation windows.")
        if not 1 <= self.cardinality <= self.liquidity_top_n <= universe_size:
            raise ValueError("Invalid cardinality or liquidity shortlist size.")
        equal_weight = 1 / self.cardinality
        if not 0 < self.min_weight <= equal_weight <= self.max_weight <= 1:
            raise ValueError("Weight bounds are incompatible with full investment.")
        if not (self.risk_aversion > 0 and self.transaction_cost_bps >= 0
                and self.slippage_bps >= 0 and self.friction < 1):
            raise ValueError("Use positive risk aversion and nonnegative friction below 100%.")


def download_panel(universe, tickers, start, end):
    end_exclusive = (end + pd.Timedelta(days=1)).date().isoformat()
    remaining = list(tickers)
    panels = []
    for _ in range(2):
        raw = yf.download(
            remaining, start=start.date().isoformat(), end=end_exclusive,
            interval="1d", auto_adjust=False, repair=False, ignore_tz=True,
            multi_level_index=True, threads=4, progress=False, timeout=12,
        )
        if raw is None or raw.empty:
            continue
        adjusted = raw["Adj Close"].reindex(columns=remaining)
        adjusted = adjusted.where((adjusted > 0) & np.isfinite(adjusted))
        dollar_volume = (raw["Close"] * raw["Volume"]).reindex(columns=remaining)
        dollar_volume = dollar_volume.where((dollar_volume > 0) & np.isfinite(dollar_volume))
        available = [
            ticker for ticker in remaining
            if adjusted[ticker].notna().any()
            and (ticker not in universe or dollar_volume[ticker].notna().any())
        ]
        panels.append(pd.concat({
            "adjusted_close": adjusted[available],
            "dollar_volume": dollar_volume[available],
        }, axis=1))
        remaining = [ticker for ticker in remaining if ticker not in available]
        if not remaining:
            return pd.concat(panels, axis=1).sort_index()
    raise RuntimeError(f"Yahoo Finance returned no usable history for {', '.join(remaining)}.")


def load_panel(universe, benchmark, risk_free, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    tickers = [*universe, benchmark, risk_free]
    cache_key = json.dumps(["daily-v1", tickers, str(start.date()), str(end.date())])
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    path = cache_dir / f"yfinance_{digest}.csv.gz"

    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as source:
            fetched_at = source.readline().strip()
            panel = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True)
    else:
        with TemporaryDirectory(dir=cache_dir, prefix="yfinance_session_") as session_dir:
            yf.set_tz_cache_location(session_dir)
            panel = download_panel(universe, tickers, start, end)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    adjusted = panel["adjusted_close"].reindex(columns=tickers)
    proxy_prices = adjusted[[benchmark, risk_free]].dropna(how="all")
    if proxy_prices.empty or proxy_prices.isna().any().any():
        raise ValueError("Benchmark and risk-free proxy must share complete daily observations.")
    if (not proxy_prices.index.is_unique or not proxy_prices.index.is_monotonic_increasing
            or proxy_prices.index[0] - start > pd.Timedelta(days=7)
            or end - proxy_prices.index[-1] > pd.Timedelta(days=7)):
        raise ValueError("Benchmark calendar does not cover the requested data range.")

    if not path.exists():
        with NamedTemporaryFile(dir=cache_dir, suffix=".tmp", delete=False) as staged:
            staged_path = Path(staged.name)
        try:
            with gzip.open(staged_path, "wt", encoding="utf-8") as destination:
                destination.write(f"{fetched_at}\n")
                panel.to_csv(destination)
            staged_path.replace(path)
        finally:
            staged_path.unlink(missing_ok=True)

    adjusted = adjusted.reindex(proxy_prices.index)
    liquidity = panel["dollar_volume"].reindex(index=proxy_prices.index, columns=universe)
    return adjusted, liquidity, fetched_at


def fit_weights(history, params, date):
    expected = history.mean(axis=0) * TRADING_DAYS
    covariance = np.atleast_2d(np.cov(history, rowvar=False, ddof=1)) * TRADING_DAYS
    risk_matrix = covariance * params.risk_aversion

    def objective(weights):
        return 0.5 * weights @ risk_matrix @ weights - expected @ weights

    def gradient(weights):
        return risk_matrix @ weights - expected

    equal = np.full(params.cardinality, 1 / params.cardinality)
    result = minimize(
        objective, equal, jac=gradient, method="SLSQP",
        bounds=[(params.min_weight, params.max_weight)] * params.cardinality,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1,
                     "jac": lambda weights: np.ones_like(weights)},
        options={"ftol": 1e-11, "maxiter": 200},
    )
    weights = result.x
    if (not result.success or not np.isfinite(weights).all()
            or abs(weights.sum() - 1) > 1e-8
            or weights.min() < params.min_weight - 1e-8
            or weights.max() > params.max_weight + 1e-8):
        raise RuntimeError(f"{date.date()} mean-variance optimization failed: {result.message}")
    return weights


def monthly_targets(prices, dollar_volume, proxy_returns, start, params):
    stock_returns = prices.pct_change(fill_method=None)
    liquidity = dollar_volume.rolling(params.liquidity_sessions).median()
    dates = stock_returns.index
    months = dates.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])
    targets = {}
    for row in month_ends:
        execution = row + params.lag_sessions
        if execution >= len(dates) or dates[execution] < start:
            continue
        date = dates[row]
        first = row - params.lookback_sessions + 1
        if first < 1:
            raise ValueError(f"Download more warm-up history before {date.date()}.")

        history = stock_returns.iloc[first:row + 1].dropna(axis=1)
        liquid = (liquidity.iloc[row].reindex(history.columns).dropna()
                  .nlargest(params.liquidity_top_n).index)
        excess = history[liquid].sub(proxy_returns.iloc[first:row + 1], axis=0)
        volatility = excess.std(ddof=1)
        scores = excess.mean().div(volatility.where(volatility > 0)).dropna()
        selected = scores.nlargest(params.cardinality).index
        if len(selected) < params.cardinality:
            raise ValueError(f"{date.date()} has too few stocks with complete estimation windows.")

        target = np.zeros(prices.shape[1])
        target[prices.columns.get_indexer(selected)] = fit_weights(
            history[selected].to_numpy(), params, date,
        )
        targets[execution] = target
    return targets


def simulate(prices, targets, start, friction):
    """Mark existing weights through the close, then apply any eligible rebalance."""
    first_row = prices.index.searchsorted(start)
    observed = prices.notna().to_numpy()
    changes = prices.ffill().pct_change(fill_method=None).fillna(0.0).to_numpy()
    weights = np.zeros(prices.shape[1])
    cash_weight = 1.0
    pending = None
    returns, turnover, trades = [], [], []
    for row in range(first_row, len(prices)):
        marked_weights = weights * (1 + changes[row])
        gross_growth = cash_weight + marked_weights.sum()
        weights = marked_weights / gross_growth
        if row in targets:
            pending = targets[row]
        remaining = 1.0
        gross_turnover = 0.0
        trade_count = 0
        if pending is not None and observed[row, (weights > 0) | (pending > 0)].all():
            # Target weights apply to the value left after proportional costs.
            # Solve for that value using actual buy plus sell notional.
            def cost_equation(fraction):
                return fraction + friction * np.abs(fraction * pending - weights).sum() - 1

            remaining = brentq(cost_equation, 0.0, 1.0) if friction else 1.0
            traded = np.abs(remaining * pending - weights)
            gross_turnover = traded.sum()
            trade_count = np.count_nonzero(traded > 1e-10)
            weights, cash_weight, pending = pending, 0.0, None
        returns.append(gross_growth * remaining - 1)
        turnover.append(gross_turnover)
        trades.append(trade_count)
    return pd.DataFrame(
        {"return": returns, "turnover": turnover, "n_trades": trades},
        index=prices.index[first_row:],
    )


def period_metrics(daily_results, proxy_returns):
    returns = daily_results["return"].to_numpy()
    growth_curve = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(np.r_[1.0, growth_curve])[1:]
    volatility = returns.std(ddof=1)
    excess = returns - proxy_returns.reindex(daily_results.index).to_numpy()
    excess_volatility = excess.std(ddof=1)
    annualizer = np.sqrt(TRADING_DAYS)
    return {
        "cagr": float(growth_curve[-1] ** (TRADING_DAYS / len(returns)) - 1),
        "ann_vol": float(volatility * annualizer),
        "sharpe": float(returns.mean() / volatility * annualizer) if volatility > 0 else None,
        "sharpe_excess": (
            float(excess.mean() / excess_volatility * annualizer)
            if excess_volatility > 0 else None
        ),
        "max_drawdown": float((growth_curve / peaks - 1).min()),
        "total_return": float(growth_curve[-1] - 1),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def run_strategy(prices, dollar_volume, proxy_returns, start, params):
    targets = monthly_targets(prices, dollar_volume, proxy_returns, start, params)
    if not targets:
        raise ValueError("No monthly signals can execute within the requested period.")
    return simulate(prices, targets, start, params.friction)


def summarize(daily_results, proxy_returns, is_end):
    periods = {
        "is": daily_results.loc[:is_end],
        "oos": daily_results.loc[daily_results.index > is_end],
        "full": daily_results,
    }
    if any(len(period) < 2 for period in periods.values()):
        raise ValueError("Each reporting period needs at least two sessions.")
    return {name: period_metrics(period, proxy_returns) for name, period in periods.items()}


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    universe = config["universe"]
    benchmark, risk_free = config["benchmark"], config["risk_free_proxy"]
    if len(set([*universe, benchmark, risk_free])) != len(universe) + 2:
        raise ValueError("Stock, benchmark and risk-free symbols must be distinct.")
    data_start, start, is_end, end = (
        pd.Timestamp(config[name]) for name in ("data_start", "backtest_start", "is_end", "end")
    )
    if not data_start < start < is_end < end:
        raise ValueError("Expected data_start < backtest_start < is_end < end.")
    official = Parameters(**config["parameters"])
    cases = {
        name: replace(official, **overrides) for name, overrides in config["robustness"].items()
    }
    combinations = {official, *cases.values()}
    if len(combinations) != 1 + len(cases):
        raise ValueError("Each robustness case must specify a distinct parameter combination.")
    for params in combinations:
        params.validate(len(universe))

    adjusted, liquidity, fetched_at = load_panel(universe, benchmark, risk_free, data_start, end)
    stocks = adjusted[universe]
    proxy_returns = adjusted[risk_free].pct_change(fill_method=None)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance Adj Close total-return (splits and distributions)")
    print("ROBUSTNESS (IS ONLY)")
    for name, params in cases.items():
        daily_results = run_strategy(
            stocks.loc[:is_end], liquidity.loc[:is_end], proxy_returns.loc[:is_end], start, params,
        )
        metrics = period_metrics(daily_results, proxy_returns.loc[:is_end])
        print(f"  {name} {json.dumps(config['robustness'][name])} {json.dumps(metrics, allow_nan=False)}")

    daily_results = run_strategy(stocks, liquidity, proxy_returns, start, official)
    index_results = simulate(
        adjusted[[benchmark]], {adjusted.index.searchsorted(start): np.ones(1)},
        start, official.friction,
    )
    strategy_metrics = summarize(daily_results, proxy_returns, is_end)
    benchmark_metrics = summarize(index_results, proxy_returns, is_end)
    data_range = {
        "start": daily_results.index[0].date().isoformat(),
        "end": daily_results.index[-1].date().isoformat(),
    }
    print(f"Date Range: {data_range['start']} to {data_range['end']}")
    print(f"OFFICIAL_PARAMETERS: {json.dumps(asdict(official))}")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()} strategy: {json.dumps(strategy_metrics[period], allow_nan=False)}")
        print(f"{period.upper()} {benchmark}: {json.dumps(benchmark_metrics[period], allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {len(combinations)}")
    results = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **strategy_metrics,
        "benchmark": {"ticker": benchmark, **benchmark_metrics},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)
