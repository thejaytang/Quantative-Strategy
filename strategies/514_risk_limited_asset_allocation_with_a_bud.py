"""514_risk_limited_asset_allocation_with_a_bud
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/514_risk_limited_asset_allocation_with_a_bud.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c0a8f76240a0431b019ab49c34040a95a765860a on 2026-11-27 10:04 Europe/Oslo, exit 0, output sha256 24709d43ebe800eb76dcc753d8db0eb9cfdd849729609cc8118148aa7b5c361e
Data source: yfinance 美元 ETF 日收盘价。风险资产为 SPY、QQQ、EFA、TLT、GLD、VNQ，BIL 为现金替代，SPY 同时用作买入持有基准。运行日志记录的数据抓取时间为 2026-11-27T08:00:19+00:00。
Data fetched: 2026-11-27T09:04:19+00:00
Sample period: 价格覆盖 2007-10-01 至 2025-12-31，2007-10-01 起的数据用于预热。回测收益区间为 2009-01-02 至 2025-12-31，含首尾交易日。 data_range (from output): {'start': '2009-01-02', 'end': '2025-12-31'}
IS/OOS split: 样本内（IS）为 2009-01-02 至 2017-12-29；样本外（OOS）为 2018-01-02 至 2025-12-31，均含首尾交易日。两段从同一次默认参数模拟的连续持仓和收益路径切分。OOS 为历史留出期，其中大部分日期早于论文发表。五组稳健性对照仅运行至 IS 截止日。
Costs and slippage: 默认每笔买卖按成交额计 5 bps 成本和 5 bps 固定滑点，两项合计 10 bps，买卖两边及 BIL 均计入。调仓时以费后资产价值确定目标持仓金额，按其与成交前持仓金额的绝对差之和计算扣费，并从净值扣除；滑点以固定费率建模。样本内双倍成本对照将成本和滑点各提高至 10 bps。
Signal lag: 月末最后一个交易日收盘 t 生成信号，默认在下一交易日收盘 t+1 模拟成交。t 至 t+1 的收盘到收盘收益按原持仓计入，随后扣费并更新权重。SPY 基准在回测首日收盘按预先设定的目标从 BIL 转为 SPY。
Price adjustment: yfinance auto_adjust=True adjusted close, total-return basis
Survivorship bias: 固定资产池选自研究时仍存续且具有长历史的 ETF，README 明示存在事后选择与存续偏差。加载器按配置读取同一组标的，并在发现任一标的缺价或价格无效时停止。
Parameter combinations tried: 6
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.05815685244260593, ann_vol=0.07720268842020143, sharpe=0.770900074161784, sharpe_excess=0.7616039971689724, max_drawdown=-0.12346042848999783, total_return=0.6621039279319347, turnover=3.344732880658356, n_trades=295
    oos: cagr=0.052023056046132776, ann_vol=0.09691245898784251, sharpe=0.5722335272034725, sharpe_excess=0.32163466163332377, max_drawdown=-0.2365405010576883, total_return=0.49887375058351746, turnover=2.9762307541275472, n_trades=232
    full: cagr=0.05526769010108934, ann_vol=0.08701960100425812, sharpe=0.6619958322659767, sharpe_excess=0.5263230736183157, max_drawdown=-0.23654050105768853, total_return=1.491283948318939, turnover=3.1714265718525896, n_trades=527
    benchmark SPY full: cagr=0.14524231948766286, ann_vol=0.17887852829174625, sharpe=0.8479660123111793, sharpe_excess=0.7813775640669344, max_drawdown=-0.337172719764255, total_return=8.98586833816174, turnover=0.11774941615914487, n_trades=2
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: '514_ risk_limited_asset_allocation_with_a_bud'\nuniverse: [SPY, QQQ, EFA, TLT, GLD, VNQ]\ncash_proxy: BIL\nbenchmark: SPY\n\ndates:\n  price_start: 2007-10-01\n  backtest_start: 2009-01-02\n  is_end: 2017-12-29\n  backtest_end: 2025-12-31\n\nsignal:\n  lookback_sessions: 252\n  horizon_sessions: 21\n  risk_barrier: 0.40\n  downside_aversion: 8.0\n  max_asset_weight: 0.30\n\nexecution:\n  lag_sessions: 1\n  cost_bps: 5.0\n  slippage_bps: 5.0\n\nrobustness:\n  - label: risk_barrier_0.35\n    signal: {risk_barrier: 0.35}\n  - label: risk_barrier_0.45\n    signal: {risk_barrier: 0.45}\n  - label: downside_aversion_6\n    signal: {downside_aversion: 6.0}\n  - label: downside_aversion_10\n    signal: {downside_aversion: 10.0}\n  - label: double_execution_cost\n    execution: {cost_bps: 10.0, slippage_bps: 10.0}\n"
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
"""Monthly long-only ETF backtest based on Giller's budget threshold."""

import hashlib
import json
import signal
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
TIME_LIMIT_SECONDS = 720


class DeadlineReached(BaseException):
    pass


def stop_at_deadline(_signum, _frame):
    raise DeadlineReached


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    cache_key = f"auto_adjust daily close {start.date()} {end.date()} {' '.join(tickers)}"
    key = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    path = CACHE / f"yfinance_{key}.csv"
    stamp_prefix = "# DATA_FETCHED_AT: "

    if path.is_file():
        with path.open(encoding="utf-8") as source:
            header = source.readline().strip()
            if not header.startswith(stamp_prefix):
                raise ValueError(f"Missing fetch timestamp in {path.name}")
            fetched_at = header.removeprefix(stamp_prefix)
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(source, index_col="date", parse_dates=["date"])
    else:
        with tempfile.TemporaryDirectory(dir=CACHE, prefix="yfinance_") as yf_cache:
            yf.set_tz_cache_location(yf_cache)
            data = yf.download(
                tickers=tickers,
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d",
                auto_adjust=True,
                keepna=True,
                threads=False,
                progress=False,
                timeout=15,
            )
        if data is None or data.empty or "Close" not in data.columns.get_level_values(0):
            raise RuntimeError("yfinance returned no daily adjusted closes")
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        prices = data["Close"].sort_index()

    if prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Price history must contain unique, sorted dates")
    if prices.index[0] != start or prices.index[-1] != end:
        raise ValueError(f"Missing boundary price dates for {start.date()} through {end.date()}")
    prices = prices.reindex(columns=tickers).rename_axis(columns=None)
    prices.index = prices.index.as_unit("ns").rename("date")
    values = prices.to_numpy(dtype=float)
    good = np.isfinite(values) & (values > 0)
    if not good.all():
        row, col = np.argwhere(~good)[0]
        raise ValueError(f"Missing or invalid adjusted close for {tickers[col]} on {prices.index[row].date()}")

    if not path.is_file():
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as out:
            temp_path = Path(out.name)
            try:
                out.write(f"{stamp_prefix}{fetched_at}\n")
                prices.to_csv(out, index_label="date", float_format="%.17g")
                out.flush()
                temp_path.replace(path)
            finally:
                temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def target_weights(scenarios, risk_barrier, downside_aversion, max_asset_weight):
    """Combine the paper's per-asset barrier with a joint downside penalty."""
    expected = scenarios.mean(axis=0)
    score = expected - risk_barrier * scenarios.std(axis=0, ddof=1)
    assets = len(score)
    if score.max() <= 0:
        return np.zeros(assets)

    centered = scenarios - expected

    def objective(weights):
        downside = np.minimum(centered @ weights, 0.0)
        penalty = 0.5 * downside_aversion * np.mean(downside**2)
        gradient = downside_aversion * (centered.T @ downside) / len(downside)
        return penalty - score @ weights, gradient - score

    result = minimize(
        objective,
        np.zeros(assets),
        jac=True,
        method="SLSQP",
        bounds=[(0.0, max_asset_weight)] * assets,
        constraints={"type": "ineq", "fun": lambda w: 1.0 - w.sum(), "jac": lambda w: -np.ones(assets)},
        options={"ftol": 1e-12, "maxiter": 150},
    )
    weights = result.x
    if not result.success or not np.isfinite(weights).all() or weights.sum() > 1.0 + 1e-8:
        raise RuntimeError(f"Allocation solve failed: {result.message}")
    weights = np.clip(weights, 0.0, max_asset_weight)
    return weights / max(1.0, weights.sum())


def execution_targets(prices, universe, cash_proxy, backtest_start, signal_cfg, lag):
    lookback = signal_cfg["lookback_sessions"]
    horizon = signal_cfg["horizon_sessions"]
    if not (1 <= horizon < lookback and 0 < signal_cfg["max_asset_weight"] <= 1):
        raise ValueError("Need 1 <= horizon < lookback and 0 < max_asset_weight <= 1")
    if not (signal_cfg["risk_barrier"] >= 0 and signal_cfg["downside_aversion"] >= 0):
        raise ValueError("Risk penalties must be nonnegative")
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("Execution lag must be at least one session")

    dates = prices.index
    months = dates.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])
    values = prices[[*universe, cash_proxy]].to_numpy()
    targets = {}
    for pos in month_ends:
        fill_pos = pos + lag
        if fill_pos >= len(dates) or dates[fill_pos] < backtest_start:
            continue
        if pos < lookback:
            raise ValueError(f"Insufficient price history for the {dates[pos].date()} signal")
        window = values[pos - lookback : pos + 1]
        horizon_growth = window[horizon:] / window[:-horizon]
        scenarios = horizon_growth[:, :-1] - horizon_growth[:, -1:]
        try:
            risky = target_weights(
                scenarios,
                signal_cfg["risk_barrier"],
                signal_cfg["downside_aversion"],
                signal_cfg["max_asset_weight"],
            )
        except RuntimeError as error:
            raise RuntimeError(f"Signal on {dates[pos].date()}: {error}") from error
        targets[dates[fill_pos]] = np.append(risky, max(0.0, 1.0 - risky.sum()))
    return targets


def rebalance(current, desired, drag):
    """Find the post-fee size that funds both trade legs without leverage."""
    remaining = 1.0
    for _ in range(100):
        changes = remaining * desired - current
        after_fees = 1.0 - drag * np.abs(changes).sum()
        if abs(after_fees - remaining) <= 1e-14:
            trades = np.abs(after_fees * desired - current)
            return after_fees, float(trades.sum()), int(np.count_nonzero(trades > 1e-10))
        remaining = after_fees
    raise RuntimeError("Execution fee calculation did not converge")


def simulate(returns, targets, execution_cfg):
    cost = execution_cfg["cost_bps"]
    slippage = execution_cfg["slippage_bps"]
    if not (cost >= 0 and slippage >= 0 and cost + slippage < 5000):
        raise ValueError("Costs must be nonnegative and sum to less than 5000 bps")
    drag = (cost + slippage) / 10_000
    weights = np.zeros(returns.shape[1])
    weights[-1] = 1.0
    daily_results = []
    for date, row in zip(returns.index, returns.to_numpy(), strict=True):
        marked = weights * (1.0 + row)
        growth = marked.sum()
        weights = marked / growth
        turnover, trades = 0.0, 0
        if date in targets:
            desired = targets[date]
            remaining, turnover, trades = rebalance(weights, desired, drag)
            growth *= remaining
            weights = desired
        daily_results.append((growth - 1.0, row[-1], turnover, trades))
    return pd.DataFrame(daily_results, index=returns.index, columns=["net", "bills", "turnover", "trades"])


def metrics(daily_results):
    net = daily_results["net"]
    if len(net) < 2:
        raise ValueError("Each measurement period needs at least two return observations")
    growth = np.r_[1.0, (1.0 + net.to_numpy()).cumprod()]
    excess = net - daily_results["bills"]
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    ann_vol = net.std(ddof=1) * annualizer
    excess_vol = excess.std(ddof=1) * annualizer
    years = len(net) / SESSIONS_PER_YEAR
    return {
        "cagr": float(growth[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(ann_vol),
        "sharpe": float(SESSIONS_PER_YEAR * net.mean() / ann_vol) if ann_vol > 1e-12 else None,
        "sharpe_excess": float(SESSIONS_PER_YEAR * excess.mean() / excess_vol) if excess_vol > 1e-12 else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1.0).min()),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["trades"].sum()),
    }


def periods(daily_results, is_end):
    return {
        "is": metrics(daily_results.loc[:is_end]),
        "oos": metrics(daily_results.loc[daily_results.index > is_end]),
        "full": metrics(daily_results),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    cash_proxy = config["cash_proxy"]
    benchmark_ticker = config["benchmark"]
    if not universe or cash_proxy in universe or len(set(universe)) != len(universe) or cash_proxy == benchmark_ticker:
        raise ValueError("Use distinct risky ETFs, a cash proxy, and a risky benchmark")
    price_start, start, is_end, end = (
        pd.Timestamp(config["dates"][key]) for key in ("price_start", "backtest_start", "is_end", "backtest_end")
    )
    if not price_start < start < is_end < end:
        raise ValueError("Dates must satisfy price_start < backtest_start < is_end < backtest_end")
    tickers = list(dict.fromkeys([*universe, cash_proxy, benchmark_ticker]))
    prices, fetched_at = load_prices(tickers, price_start, end)
    if start not in prices.index or is_end not in prices.index:
        raise ValueError("The backtest start and IS end must be observed price dates")
    returns = prices.pct_change(fill_method=None).loc[start:]
    strategy_returns = returns[[*universe, cash_proxy]]
    signal_cfg = config["signal"]
    execution_cfg = config["execution"]

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True adjusted close, total-return basis")
    print("ROBUSTNESS: in-sample only, fixed one-at-a-time cases", flush=True)
    attempts = 0
    for case in config["robustness"]:
        case_execution = {**execution_cfg, **case.get("execution", {})}
        attempts += 1
        targets = execution_targets(
            prices.loc[:is_end], universe, cash_proxy, start,
            {**signal_cfg, **case.get("signal", {})}, case_execution["lag_sessions"],
        )
        result = simulate(strategy_returns.loc[:is_end], targets, case_execution)
        print(f"ROBUSTNESS_IS {case['label']}: {json.dumps(metrics(result), allow_nan=False)}")

    attempts += 1
    targets = execution_targets(prices, universe, cash_proxy, start, signal_cfg, execution_cfg["lag_sessions"])
    official = periods(simulate(strategy_returns, targets, execution_cfg), is_end)
    benchmark_returns = returns[[benchmark_ticker, cash_proxy]]
    benchmark_targets = {start: np.array([1.0, 0.0])}
    benchmark = periods(simulate(benchmark_returns, benchmark_targets, execution_cfg), is_end)
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        **official,
        "benchmark": {"ticker": benchmark_ticker, **benchmark},
    }
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(official[period], allow_nan=False)}")
        print(f"BENCHMARK {benchmark_ticker} {period.upper()}: {json.dumps(benchmark[period], allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {attempts}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_at_deadline)
    signal.alarm(TIME_LIMIT_SECONDS)
    try:
        main()
    except DeadlineReached:
        raise SystemExit(f"Backtest exceeded its {TIME_LIMIT_SECONDS}-second time limit") from None
    finally:
        signal.alarm(0)