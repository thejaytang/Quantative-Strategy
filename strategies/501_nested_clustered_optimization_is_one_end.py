"""501_nested_clustered_optimization_is_one_end
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/501_nested_clustered_optimization_is_one_end.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c0a8f76240a0431b019ab49c34040a95a765860a on 2026-11-27 10:03 Europe/Oslo, exit 0, output sha256 7d058ca5f119229656a05f807995bd1e6939435442a14fccd69be1d84d113fd7
Data source: yfinance 免费日频 ETF 复权收盘价。配置的行业 ETF 为 XLB、XLE、XLF、XLI、XLK、XLP、XLU、XLV、XLY。BIL 仅用于事后计算超额夏普。
Data fetched: 2026-11-27T09:03:44+00:00
Sample period: 预热数据请求从 2007-06-01 开始，配置的回测区间为 2009-01-01 至 2025-12-31，日志记录的首末回测交易日为 2009-01-02 和 2025-12-31。 data_range (from output): {'start': '2009-01-02', 'end': '2025-12-31'}
IS/OOS split: 样本内为 2009-01-02 至 2017-12-29，按配置包含截至 2017-12-31 的交易日。样本外为 2018-01-02 至 2025-12-31，承接样本内结束时的持仓。正式结果全程使用固定默认参数，七个变体仅在样本内运行。
Costs and slippage: 默认按每笔买入或卖出的成交金额计 2 bps 交易成本和 3 bps 固定滑点，合计 5 bps。滑点作为费率扣款建模。调仓收盘按调仓前已漂移的持仓与扣费后目标持仓之间的绝对变动计费，包括首次从现金建仓。等权基准使用相同费率。样本内费用压力测试使用 4 bps 成本加 6 bps 滑点。
Signal lag: 每月用截至上月最后一个交易日 t 收盘的收益计算目标权重，默认回看 252 个交易日。模拟在下一交易日 t+1 收盘调仓。旧持仓承担截至该次收盘的价格变动，当日扣费，新权重从该次收盘后的第一个收盘到收盘区间起计入收益。
Price adjustment: yfinance auto_adjust total-return
Survivorship bias: 固定选用九只行业 ETF。README 记载这些基金在回测期前已成立，且事后仍能取得长期历史，并指出固定资产池存在幸存者偏差和事后选池风险。代码要求配置资产池的价格完整且有效。
Parameter combinations tried: 8
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.13373214568868796, ann_vol=0.11937254474417407, sharpe=1.1114452609850176, sharpe_excess=1.102604569901029, max_drawdown=-0.20236685737363225, total_return=2.0899056080757106, turnover=1.678076889957888, n_trades=475
    oos: cagr=0.07657910555638625, ann_vol=0.1543514371200629, sharpe=0.5555053400211863, sharpe_excess=0.3980743495037256, max_drawdown=-0.3117247636266096, total_return=0.8019043813886413, turnover=1.678221078484321, n_trades=563
    full: cagr=0.10648470363991147, ann_vol=0.13693200108985826, sharpe=0.8077256474256108, sharpe_excess=0.720630570286935, max_drawdown=-0.3117247636266093, total_return=4.567714453268953, turnover=1.6781447017274522, n_trades=1038
    benchmark full: cagr=0.13253729981339935, ann_vol=0.17361920151678253, sharpe=0.8040986988749034, sharpe_excess=0.7354591002632082, max_drawdown=-0.367154672069144, total_return=7.263742768983503, turnover=0.3473511765729799, n_trades=1836
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 501_nested_clustered_optimization_is_one_end\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nrisk_free_ticker: BIL\ndata_start: 2007-06-01\nstart: 2009-01-01\nis_end: 2017-12-31\nend: 2025-12-31\nexecution_lag_sessions: 1\n\nallocation:\n  lookback_sessions: 252\n  n_clusters: 3\n  gamma: 0.5\n  covariance_ridge: 0.000001\n\ncost_bps: 2.0\nslippage_bps: 3.0\n\n# Individual changes to the default, evaluated only in sample.\nrobustness:\n  - {gamma: 0.0}\n  - {gamma: 0.25}\n  - {gamma: 0.75}\n  - {lookback_sessions: 189}\n  - {lookback_sessions: 315}\n  - {n_clusters: 4}\n  - {cost_bps: 4.0, slippage_bps: 6.0}\n'
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
"""Monthly long-only Schur bridge for a fixed universe of sector ETFs."""

import hashlib
import json
import os
import signal
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.linalg import solve_triangular
from scipy.optimize import brentq, nnls
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class Parameters:
    lookback_sessions: int
    n_clusters: int
    gamma: float
    covariance_ridge: float
    cost_bps: float
    slippage_bps: float

    def __post_init__(self):
        if self.lookback_sessions < 2 or not 0 <= self.gamma <= 1:
            raise ValueError("Require a lookback of at least two sessions and gamma in [0, 1].")
        if not np.isfinite(self.covariance_ridge) or self.covariance_ridge <= 0:
            raise ValueError("The covariance ridge must be positive and finite.")
        if min(self.cost_bps, self.slippage_bps) < 0 or not 0 <= self.fee_rate < 1:
            raise ValueError("Fees and slippage must be nonnegative and total under 10,000 bps.")

    @property
    def fee_rate(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def check_prices(prices, tickers, data_start, end):
    if list(prices.columns) != tickers or prices.empty:
        raise ValueError("Adjusted closes are missing for the configured universe.")
    dates = prices.index
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Adjusted closes must have unique, increasing session dates.")
    values = prices.to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        bad = prices.columns[(~np.isfinite(values) | (values <= 0)).any(axis=0)]
        raise ValueError(f"Missing or invalid adjusted closes for {', '.join(bad)}.")
    # Configured boundaries can fall on non-session dates.
    coverage = [dates[0] - data_start, end - dates[-1]]
    if any(not pd.Timedelta(0) <= gap <= pd.Timedelta(days=6) for gap in coverage):
        raise ValueError("Adjusted closes do not cover the configured date range.")


def load_prices(tickers, data_start, end):
    CACHE_DIR.mkdir(exist_ok=True)
    request = f"yfinance:1d:auto_adjust:{data_start.date()}:{end.date()}:{','.join(tickers)}"
    digest = hashlib.sha256(request.encode()).hexdigest()[:16]
    path = CACHE_DIR / f"adjusted_close_{digest}.csv"
    if path.exists():
        try:
            with path.open(encoding="utf-8") as source:
                fetched_at = source.readline().strip().removeprefix("# fetched_at=")
                datetime.fromisoformat(fetched_at)
                prices = pd.read_csv(
                    source, index_col="date", parse_dates=["date"], float_precision="round_trip"
                )
            check_prices(prices, tickers, data_start, end)
            return prices, fetched_at
        except (OSError, ValueError, TypeError):
            pass

    # Keep the library's temporary state inside this project.
    with tempfile.TemporaryDirectory(dir=CACHE_DIR) as download_dir:
        yf.set_tz_cache_location(download_dir)
        try:
            raw = yf.download(
                tickers,
                start=data_start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d",
                auto_adjust=True,
                group_by="column",
                ignore_tz=True,
                keepna=True,
                threads=False,
                progress=False,
                timeout=10,
            )
        finally:
            yf.set_tz_cache_location(str(CACHE_DIR))
    if raw is None or raw.empty or "Close" not in raw.columns.get_level_values(0):
        raise RuntimeError("yfinance returned no adjusted closes.")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prices = raw["Close"].reindex(columns=tickers).sort_index()
    check_prices(prices, tickers, data_start, end)
    with tempfile.NamedTemporaryFile(dir=CACHE_DIR, mode="w", encoding="utf-8", delete=False) as sink:
        pending = Path(sink.name)
        try:
            sink.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(sink, index_label="date")
            sink.flush()
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)
    return prices, fetched_at


def long_only_weights(covariance, budget_direction):
    # NNLS solves min_{x >= 0} (x' Q x / 2 - b' x) using Q = L L'.
    # Its normalized solution has the shape of the generalized minimum-variance direction.
    cholesky = np.linalg.cholesky(covariance)
    target = solve_triangular(cholesky, budget_direction, lower=True)
    direction, _ = nnls(cholesky.T, target)
    if direction.sum() == 0:
        # A nonpositive budget vector cannot give b' x = 1 with x >= 0.
        target = solve_triangular(cholesky, np.ones(len(direction)), lower=True)
        direction, _ = nnls(cholesky.T, target)
    return direction / direction.sum()


def bridge_weights(window, parameters):
    sample_covariance = np.cov(window, rowvar=False)
    standard_deviations = np.sqrt(sample_covariance.diagonal())
    if (standard_deviations <= 0).any():
        raise ValueError("A zero-variance series makes correlation clustering undefined.")
    correlation = sample_covariance / np.outer(standard_deviations, standard_deviations)
    distances = np.sqrt((1 - np.clip(correlation, -1, 1)) / 2)
    np.fill_diagonal(distances, 0)
    tree = linkage(squareform(distances, checks=False), method="average")
    labels = cut_tree(tree, n_clusters=[parameters.n_clusters]).ravel()
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    # Select the member nearest the others on average as each cluster's knot.
    knots = [group[np.argmin(distances[np.ix_(group, group)].sum(axis=1))] for group in groups]

    ridge = parameters.covariance_ridge * sample_covariance.diagonal().mean()
    covariance = sample_covariance + ridge * np.eye(window.shape[1])
    cluster_weights = np.zeros((window.shape[1], len(groups)))
    for column, group in enumerate(groups):
        conditioned = covariance[np.ix_(group, group)]
        budget_direction = np.ones(len(group))
        other_knots = knots[:column] + knots[column + 1 :]
        if parameters.gamma and other_knots:
            cross = covariance[np.ix_(group, other_knots)]
            knot_covariance = covariance[np.ix_(other_knots, other_knots)]
            regression = np.linalg.solve(knot_covariance, cross.T).T
            # Equation (8) damps both the conditional covariance and budget vector.
            conditioned -= parameters.gamma * (regression @ cross.T)
            budget_direction -= parameters.gamma * regression.sum(axis=1)
        cluster_weights[group, column] = long_only_weights(conditioned, budget_direction)

    outer_covariance = cluster_weights.T @ covariance @ cluster_weights
    outer_weights = long_only_weights(outer_covariance, np.ones(len(groups)))
    return cluster_weights @ outer_weights


def rebalance_schedule(dates, start, lag):
    months = dates.to_period("M")
    # On a new month's first session the previous month's last close is known.
    signals = np.flatnonzero(months[:-1] != months[1:])
    return {
        dates[signal_position + lag]: signal_position
        for signal_position in signals
        if signal_position + lag < len(dates) and dates[signal_position + lag] >= start
    }


def strategy_targets(returns, schedule, parameters):
    targets = {}
    for fill_date, signal_position in schedule.items():
        stop = signal_position + 1
        start = stop - parameters.lookback_sessions
        if start < 0:
            raise ValueError(f"Insufficient warm-up history for the rebalance on {fill_date.date()}.")
        targets[fill_date] = bridge_weights(returns.iloc[start:stop].to_numpy(), parameters)
    return targets


def simulate(returns, targets, fee_rate):
    positions = np.zeros(returns.shape[1])
    returns_net = np.zeros(len(returns))
    turnover = np.zeros(len(returns))
    n_trades = np.zeros(len(returns), dtype=int)
    for row, (date, asset_returns) in enumerate(zip(returns.index, returns.to_numpy())):
        # The old positions earn the move through this close.
        gross = 1 + positions @ asset_returns
        positions = positions * (1 + asset_returns) / gross
        remaining = 1.0
        if date in targets:
            target = targets[date]
            # Charge on actual buys and sells, including the first investment from cash.
            remaining = brentq(
                lambda fraction: fraction + fee_rate * np.abs(fraction * target - positions).sum() - 1,
                0.0,
                1.0,
            )
            changes = np.abs(remaining * target - positions)
            turnover[row] = changes.sum()
            n_trades[row] = np.count_nonzero(changes > 1e-10)
            positions = target
        returns_net[row] = gross * remaining - 1
    return pd.DataFrame(
        {"return": returns_net, "turnover": turnover, "n_trades": n_trades}, index=returns.index
    )


def metrics(daily_results, bill_returns):
    daily = daily_results["return"]
    if len(daily) < 2:
        raise ValueError("Each reported period needs at least two return observations.")
    annualization = SESSIONS_PER_YEAR / len(daily)
    growth = np.r_[1.0, (1 + daily).cumprod().to_numpy()]
    sigma = daily.std()
    excess = daily - bill_returns.loc[daily.index]
    excess_sigma = excess.std()
    return {
        "cagr": float(growth[-1] ** annualization - 1),
        "ann_vol": float(sigma * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": float(daily.mean() / sigma * np.sqrt(SESSIONS_PER_YEAR)) if sigma > 0 else None,
        "sharpe_excess": (
            float(excess.mean() / excess_sigma * np.sqrt(SESSIONS_PER_YEAR))
            if excess_sigma > 0
            else None
        ),
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].sum() * annualization),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, bill_returns, is_end):
    within_is = daily_results.index <= is_end
    periods = {"is": daily_results.loc[within_is], "oos": daily_results.loc[~within_is], "full": daily_results}
    return {label: metrics(period, bill_returns) for label, period in periods.items()}


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    tickers = config["universe"]
    bill = config["risk_free_ticker"]
    data_start, start, is_end, end = (
        pd.Timestamp(config[key]) for key in ("data_start", "start", "is_end", "end")
    )
    lag = config["execution_lag_sessions"]
    if not data_start < start <= is_end < end or not isinstance(lag, int) or lag < 1:
        raise ValueError("Check the configured period boundaries and positive execution lag.")
    if len(set([*tickers, bill])) != len(tickers) + 1:
        raise ValueError("The ETF universe and its cash proxy must have distinct tickers.")

    base = Parameters(**config["allocation"], cost_bps=config["cost_bps"], slippage_bps=config["slippage_bps"])
    alternatives = [(changes, replace(base, **changes)) for changes in config["robustness"]]
    cases = {base, *(parameters for _, parameters in alternatives)}
    if any(not 1 <= case.n_clusters <= len(tickers) for case in cases) or len(tickers) < 2:
        raise ValueError("Check the universe size and configured cluster counts.")
    if len(cases) != 1 + len(alternatives):
        raise ValueError("Robustness specifications must be distinct from the default and each other.")

    prices, fetched_at = load_prices([*tickers, bill], data_start, end)
    asset_returns = prices.pct_change(fill_method=None).iloc[1:]
    sector_returns = asset_returns[tickers]
    bill_returns = asset_returns[bill]
    evaluation_returns = sector_returns.loc[start:end]
    schedule = rebalance_schedule(sector_returns.index, start, lag)
    if not schedule or next(iter(schedule)) > is_end:
        raise ValueError("The sample contains no executable in-sample rebalance.")

    targets = strategy_targets(sector_returns, schedule, base)
    strategy_results = simulate(evaluation_returns, targets, base.fee_rate)
    equal_weights = np.full(len(tickers), 1 / len(tickers))
    benchmark_results = simulate(
        evaluation_returns, dict.fromkeys(schedule, equal_weights), base.fee_rate
    )
    official = period_metrics(strategy_results, bill_returns, is_end)
    benchmark = period_metrics(benchmark_results, bill_returns, is_end)
    data_range = {
        "start": evaluation_returns.index[0].date().isoformat(),
        "end": evaluation_returns.index[-1].date().isoformat(),
    }
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return")
    for label, values in official.items():
        print(f"{label.upper()}: {json.dumps(values, allow_nan=False)}")
    print("BENCHMARK: monthly equal-weight sector ETFs with the same lag, costs and slippage")
    for label, values in benchmark.items():
        print(f"BENCHMARK_{label.upper()}: {json.dumps(values, allow_nan=False)}")

    print("ROBUSTNESS (IS only)")
    in_sample = sector_returns.loc[:is_end]
    is_evaluation = evaluation_returns.loc[:is_end]
    is_schedule = rebalance_schedule(in_sample.index, start, lag)
    print(f"DEFAULT: {json.dumps(official['is'], allow_nan=False)}")
    for changes, parameters in alternatives:
        alternative_targets = strategy_targets(in_sample, is_schedule, parameters)
        alternative = simulate(is_evaluation, alternative_targets, parameters.fee_rate)
        values = metrics(alternative, bill_returns)
        print(f"{json.dumps(changes)}: {json.dumps(values, allow_nan=False)}")

    print(f"PARAM_COMBOS_TRIED: {len(cases)}")
    result = {"strategy": config["strategy"], "data_range": data_range, **official, "benchmark": benchmark}
    print(f"RESULTS_JSON: {json.dumps(result, allow_nan=False)}")


if __name__ == "__main__":
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)