"""512_nested_clustered_optimization_is_one_end
WARNING: known defects (see DEEP_REVIEW_FINDINGS.md); numbers invalid.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/512_nested_clustered_optimization_is_one_end.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c0a8f76240a0431b019ab49c34040a95a765860a on 2026-11-27 10:04 Europe/Oslo, exit 0, output sha256 07f1bcb4d1080926ad17b8ca696b95dcd66d4649996a920a201cb9853f0bc7ed
Data source: yfinance 提供的美元日线 ETF 价格。配置的行业 ETF 为 XLB、XLE、XLF、XLI、XLK、XLP、XLU、XLV、XLY，基准为 SPY，事后计算超额夏普时使用 BIL 作为无风险代理。来源为。
Data fetched: 2026-11-27T09:04:16+00:00
Sample period: 请求行情区间为 2007-06-01 至 2025-12-31，评估开始日前的记录用于预热。配置评估区间为 2009-01-01 至 2025-12-31，日志所示实际评估区间为 2009-01-02 至 2025-12-31。来源为。 data_range (from output): {'start': '2009-01-02', 'end': '2025-12-31'}
IS/OOS split: 配置的样本内区间为 2009-01-01 至 2016-12-31（含边界），样本外区间为 2017-01-01 至 2025-12-31，均按实际交易日取样。正式配置全段连续模拟，样本外承接持仓并继续滚动估计。六组替代配置只在样本内评估。来源为。
Costs and slippage: 正式配置按实际买入与卖出名义金额各扣 2 bps 费用及 3 bps 固定滑点，合计 5 bps，包含初始建仓。成交前用当日收盘收益更新持仓，随后求解扣费后资产价值，并按扣费后目标头寸与成交前头寸的绝对差计费。SPY 在策略首次成交的收盘按同率支付一次建仓成本。样本内另测试 4 bps 费用加 6 bps 滑点。来源为。
Signal lag: 使用截至月末最后一个交易日 t 收盘的日收益形成信号，模拟在下一交易日 t+1 收盘调仓。旧权重承担截至成交收盘的收益，新权重在随后一个收盘到收盘区间开始承担价格变动。来源为。
Price adjustment: yfinance auto_adjust total-return closes (dividends and splits)
Survivorship bias: 资产池由现存的九只行业 ETF 回溯选取。README 披露幸存者偏差和事后选择偏差，注明后来成立的独立房地产及通信服务行业基金未纳入，且行业指数分类与成分可能变更。来源为。
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.13031841917705683, ann_vol=0.12495207110566968, sharpe=1.0431262734037199, sharpe_excess=1.0407566566969984, max_drawdown=-0.2023668334162421, total_return=1.6618538655971935, turnover=1.680709581283218, n_trades=387
    oos: cagr=0.08607911369028266, ann_vol=0.14678127473985117, sharpe=0.6362446116592514, sharpe_excess=0.4838299377665047, max_drawdown=-0.3117239847714956, total_return=1.0984533194366026, turnover=1.6795482124266563, n_trades=652
    full: cagr=0.10669591931434075, ann_vol=0.1369231186504212, sharpe=0.8091637086121385, sharpe_excess=0.7220613362789606, max_drawdown=-0.3117239847714958, total_return=4.585776080117574, turnover=1.6800952182445037, n_trades=1039
    benchmark full: cagr=0.14532884041642066, ann_vol=0.17887786202358763, sharpe=0.8483910852335355, sharpe_excess=0.7818029237902376, max_drawdown=-0.33717286856294393, total_return=8.998677147358071, turnover=0.05890413072229087, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 512_nested_clustered_optimization_is_one_end\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nbenchmark: SPY\nrisk_free_proxy: BIL\n\ndata_start: 2007-06-01\nstart: 2009-01-01\nis_end: 2016-12-31\nend: 2025-12-31\nexecution_lag_sessions: 1\n\nparameters:\n  lookback_sessions: 252\n  n_clusters: 3\n  gamma: 0.5\n  covariance_ridge: 0.000001\n  cost_bps: 2.0\n  slippage_bps: 3.0\n\n# Each override is applied independently and evaluated through the IS boundary.\nrobustness:\n  - {gamma: 0.35}\n  - {gamma: 0.65}\n  - {lookback_sessions: 210}\n  - {lookback_sessions: 294}\n  - {gamma: 0.0}\n  - {cost_bps: 4.0, slippage_bps: 6.0}\n'
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
import hashlib
import json
import os
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
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.linalg import solve_triangular
from scipy.optimize import brentq, nnls
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class Parameters:
    lookback_sessions: int
    n_clusters: int
    gamma: float
    covariance_ridge: float
    cost_bps: float
    slippage_bps: float

    @property
    def fee_rate(self):
        return (self.cost_bps + self.slippage_bps) / 10_000

    def check(self, size):
        if type(self.lookback_sessions) is not int or self.lookback_sessions < 2:
            raise ValueError("lookback_sessions must be an integer of at least two.")
        if type(self.n_clusters) is not int or not 1 <= self.n_clusters <= size:
            raise ValueError("n_clusters must be between one and the universe size.")
        if not 0 <= self.gamma <= 1 or not 0 < self.covariance_ridge <= 1:
            raise ValueError("gamma must be in [0, 1] and covariance_ridge in (0, 1].")
        if not (self.cost_bps >= 0 and self.slippage_bps >= 0 and self.fee_rate < 1):
            raise ValueError("Costs and slippage must be nonnegative and sum to under 10,000 bps.")


def check_prices(prices, tickers, start, end):
    if prices.empty or list(prices.columns) != tickers:
        raise ValueError("Adjusted closes are missing for the configured tickers.")
    dates = prices.index
    if not isinstance(dates, pd.DatetimeIndex) or dates.hasnans:
        raise ValueError("Adjusted closes must have valid dates.")
    if not dates.is_monotonic_increasing or not dates.is_unique:
        raise ValueError("Adjusted closes must have unique, increasing session dates.")
    values = prices.to_numpy(dtype=float)
    invalid = ~np.isfinite(values) | (values <= 0)
    if invalid.any():
        names = prices.columns[invalid.any(axis=0)]
        raise ValueError(f"Missing or invalid adjusted closes for {', '.join(names)}.")
    if any(not 0 <= gap.days <= 5 for gap in (dates[0] - start, end - dates[-1])):
        raise ValueError("Adjusted closes do not cover the requested dates.")


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    key = f"yfinance:1d:auto_adjust:{start.date()}:{end.date()}:{','.join(tickers)}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    path = CACHE / f"adjusted_close_{digest}.csv"
    if path.exists():
        try:
            with path.open(encoding="utf-8") as source:
                header = source.readline().strip()
                fetched_at = header.removeprefix("# fetched_at=")
                if header == fetched_at or datetime.fromisoformat(fetched_at).tzinfo is None:
                    raise ValueError("Missing fetch timestamp.")
                prices = pd.read_csv(source, index_col="date", parse_dates=["date"],
                                     converters=dict.fromkeys(tickers, float))
            check_prices(prices, tickers, start, end)
            return prices, fetched_at
        except (OSError, ValueError, TypeError) as error:
            print(f"DATA_CACHE_REFRESH: {error}", flush=True)

    with TemporaryDirectory(dir=CACHE, prefix="yfinance_") as download_dir:
        yf.set_tz_cache_location(download_dir)
        try:
            data = yf.download(
                tickers,
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d",
                auto_adjust=True,
                ignore_tz=True,
                keepna=True,
                threads=False,
                progress=False,
                timeout=12,
            )
        finally:
            yf.set_tz_cache_location(str(CACHE))
    if data.empty or "Close" not in data.columns.get_level_values(0):
        raise RuntimeError("yfinance returned no daily adjusted closes.")
    prices = data["Close"].reindex(columns=tickers).sort_index()
    check_prices(prices, tickers, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(dir=CACHE, mode="w", encoding="utf-8", delete=False) as sink:
        pending = Path(sink.name)
        try:
            sink.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(sink, index_label="date")
            sink.close()
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)
    return prices, fetched_at


def nonnegative_direction(covariance, objective):
    # With Q = LL', NNLS minimizes x'Qx/2 - b'x over x >= 0.
    factor = np.linalg.cholesky(covariance)
    transformed = solve_triangular(factor, objective, lower=True)
    direction, _ = nnls(factor.T, transformed)
    return direction


def bridge_weights(window, params):
    size = window.shape[1]
    covariance = np.cov(window, rowvar=False, ddof=1)
    scale = np.trace(covariance) / size
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The covariance window has no finite positive variance.")
    covariance += (params.covariance_ridge * scale) * np.eye(size)
    deviation = np.sqrt(np.diag(covariance))
    correlation = np.clip(covariance / np.outer(deviation, deviation), -1, 1)
    distance = np.sqrt((1 - correlation) / 2)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method="average")
    labels = cut_tree(tree, n_clusters=params.n_clusters).ravel()
    clusters = [np.flatnonzero(labels == label) for label in np.unique(labels)]

    knots = []
    for members in clusters:
        _, eigenvectors = np.linalg.eigh(correlation[np.ix_(members, members)])
        knots.append(members[np.argmax(np.abs(eigenvectors[:, -1]))])

    inner_weights = np.zeros((size, len(clusters)))
    for column, members in enumerate(clusters):
        block = covariance[np.ix_(members, members)]
        companion = np.ones(len(members))
        if params.gamma > 0 and len(clusters) > 1:
            others = [knot for index, knot in enumerate(knots) if index != column]
            cross = covariance[np.ix_(members, others)]
            regression = np.linalg.solve(covariance[np.ix_(others, others)], cross.T).T
            # Cotton (2026), equation (8), with u = 1.
            block = block - params.gamma * (regression @ cross.T)
            block = (block + block.T) / 2
            companion -= params.gamma * regression.sum(axis=1)
        direction = nonnegative_direction(block, companion)
        if direction.sum() > 0:
            inner_weights[members, column] = direction / direction.sum()

    # A zero direction has no positive exposure to its companion vector.
    inner_weights = inner_weights[:, inner_weights.any(axis=0)]
    if not inner_weights.shape[1]:
        raise ValueError("No cluster has a positive long-only Schur direction.")
    outer = inner_weights.T @ covariance @ inner_weights
    budgets = nonnegative_direction((outer + outer.T) / 2, np.ones(outer.shape[0]))
    return inner_weights @ (budgets / budgets.sum())


def rebalance_targets(returns, start, lag, params):
    dates = returns.index
    months = dates.to_period("M")
    signal_rows = np.flatnonzero(months[:-1] != months[1:])
    targets = {}
    for signal_row in signal_rows:
        fill_row = signal_row + lag
        if fill_row >= len(dates) or dates[fill_row] < start:
            continue
        first_row = signal_row + 1 - params.lookback_sessions
        if first_row < 0:
            raise ValueError(f"Insufficient lookback at {dates[signal_row].date()}.")
        window = returns.iloc[first_row:signal_row + 1].to_numpy()
        targets[dates[fill_row]] = bridge_weights(window, params)
    if not targets:
        raise ValueError("The evaluation interval has no executable rebalance.")
    return targets


def simulate(returns, targets, fee_rate):
    weights = np.zeros(returns.shape[1])
    growth = np.empty(len(returns))
    turnover = np.zeros(len(returns))
    trades = np.zeros(len(returns), dtype=int)
    for row, (date, price_relatives) in enumerate(zip(returns.index, 1 + returns.to_numpy())):
        # Prior weights earn the return up to today's close, before any fill.
        gross = 1 + weights @ (price_relatives - 1)
        weights = weights * price_relatives / gross
        target = targets.get(date)
        remaining = 1.0
        if target is not None:
            if fee_rate:
                # Solve for post-cost value as a fraction of the pre-fill value.
                def fee_balance(fraction):
                    return fraction + fee_rate * np.abs(fraction * target - weights).sum() - 1

                remaining = brentq(fee_balance, 0.0, 1.0)
            traded = np.abs(remaining * target - weights)
            turnover[row] = traded.sum()
            trades[row] = np.count_nonzero(traded > 1e-10)
            weights = target
        growth[row] = gross * remaining
    return pd.DataFrame({"return": growth - 1, "turnover": turnover, "trades": trades},
                        index=returns.index)


def metrics(daily_results, bill_returns):
    daily = daily_results["return"].to_numpy()
    if len(daily) < 2:
        raise ValueError("Each evaluated period needs at least two sessions.")
    excess = daily - bill_returns.loc[daily_results.index].to_numpy()
    log_growth = np.log1p(daily).cumsum()
    peak = np.maximum.accumulate(np.r_[0.0, log_growth])[1:]
    vol = daily.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(np.expm1(log_growth[-1] * SESSIONS_PER_YEAR / len(daily))),
        "ann_vol": float(vol * annualizer),
        "sharpe": float(daily.mean() / vol * annualizer) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * annualizer) if excess_vol > 0 else None,
        "max_drawdown": float(np.expm1(log_growth - peak).min()),
        "total_return": float(np.expm1(log_growth[-1])),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["trades"].sum()),
    }


def period_metrics(daily_results, bill_returns, is_end):
    in_sample = daily_results.index <= is_end
    return {
        "is": metrics(daily_results.loc[in_sample], bill_returns),
        "oos": metrics(daily_results.loc[~in_sample], bill_returns),
        "full": metrics(daily_results, bill_returns),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    universe = config["universe"]
    benchmark_ticker = config["benchmark"]
    bill_ticker = config["risk_free_proxy"]
    tickers = [*universe, benchmark_ticker, bill_ticker]
    if len(universe) < 2 or len(set(tickers)) != len(tickers):
        raise ValueError("Use at least two sector ETFs and keep all ticker symbols distinct.")
    if not all(re.fullmatch(r"[A-Z][A-Z0-9]{0,5}(?:-[A-Z])?", ticker) for ticker in tickers):
        raise ValueError("Use US equity or ETF ticker symbols.")
    data_start, start, is_end, end = (
        pd.Timestamp(config[key]) for key in ("data_start", "start", "is_end", "end")
    )
    lag = config["execution_lag_sessions"]
    if not data_start < start <= is_end < end or type(lag) is not int or lag < 1:
        raise ValueError("Check date ordering and use a positive integer execution lag.")
    params = Parameters(**config["parameters"])
    alternatives = [(changes, replace(params, **changes)) for changes in config["robustness"]]
    cases = [params, *(case for _, case in alternatives)]
    if len(set(cases)) != len(cases):
        raise ValueError("Every robustness case must be distinct from the default and the other cases.")
    for case in cases:
        case.check(len(universe))

    prices, fetched_at = load_prices(tickers, data_start, end)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    sector_returns = returns[universe]
    bill_returns = returns[bill_ticker]
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return closes (dividends and splits)")
    print("ROBUSTNESS (IS only)")
    is_returns = sector_returns.loc[:is_end]
    for changes, case in alternatives:
        targets = rebalance_targets(is_returns, start, lag, case)
        daily_results = simulate(is_returns.loc[start:], targets, case.fee_rate)
        values = metrics(daily_results, bill_returns.loc[:is_end])
        print(json.dumps({"override": changes, "is": values}, allow_nan=False))

    targets = rebalance_targets(sector_returns, start, lag, params)
    sample = returns.loc[start:end]
    daily_results = simulate(sample[universe], targets, params.fee_rate)
    benchmark_targets = {next(iter(targets)): np.ones(1)}
    benchmark_results = simulate(sample[[benchmark_ticker]], benchmark_targets, params.fee_rate)
    official = period_metrics(daily_results, bill_returns, is_end)
    benchmark = period_metrics(benchmark_results, bill_returns, is_end)
    data_range = {"start": sample.index[0].date().isoformat(), "end": sample.index[-1].date().isoformat()}
    print(f"Date Range: {data_range['start']} to {data_range['end']}")
    for label, values in official.items():
        print(f"{label.upper()}: {json.dumps(values, allow_nan=False)}")
    print(f"BENCHMARK: {benchmark_ticker} buy and hold from the first strategy fill")
    for label, values in benchmark.items():
        print(f"BENCHMARK_{label.upper()}: {json.dumps(values, allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {len(cases)}")
    result = {"strategy": config["strategy"], "data_range": data_range, **official, "benchmark": benchmark}
    print(f"RESULTS_JSON: {json.dumps(result, allow_nan=False)}")


if __name__ == "__main__":
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)