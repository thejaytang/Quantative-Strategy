"""476_conditional_value_at_risk_under_reward_p
WARNING: known defects (see DEEP_REVIEW_FINDINGS.md); numbers invalid.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/476_conditional_value_at_risk_under_reward_p.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit e0683ac343a175cb2458afc860f8d1a1066c8e3a on 2026-11-27 08:56 Europe/Oslo, exit 0, output sha256 3608cb98cdc72b13deae2bf0fda80ee212e8b2353f229ebd14fbbfd70d8457eb
Data source: Yahoo Finance 日频 ETF 收盘价，通过 yfinance 下载或读取对应的本地快照。组合资产为 SPY、IWM、EFA、EEM、IEF、GLD。BIL 的日总回报用于事后计算超额收益指标。
Data fetched: 2026-11-27T07:56:27+00:00
Sample period: 预热数据请求自 2014-01-01 开始，回测评估期为 2017-03-01 至 2025-02-28（含首尾），以数据中的交易日为准。 data_range (from output): {'start': '2017-03-01', 'end': '2025-02-28'}
IS/OOS split: 按日期切分历史模拟。IS 为 2017-03-01 至 2020-02-29，OOS 为 2020-03-01 至 2025-02-28，均取区间内的数据交易日。先连续模拟全评估期再切分指标，OOS 延续既有持仓和滚动窗口。
Costs and slippage: 每笔模拟买入和卖出均按成交金额计 2 bps 成本和 3 bps 滑点，合计 5 bps。用当日收益漂移后的持仓与扣费后资金按目标权重分配所得的持仓之差计算双边成交额，并求解扣费后资金。策略和每日再平衡等权基准使用同一费率。
Signal lag: 用截至交易日 t 收盘的 756 个日收益观测计算目标权重，于下一数据交易日 t+1 收盘模拟成交。至 t+1 收盘的收益归旧权重，新权重从 t+1 收盘开始承担下一段收盘到收盘收益。
Price adjustment: yfinance auto_adjust=True total-return adjusted close
Survivorship bias: 使用固定的六只仍存续且历史较长的 ETF，存在资产选择与幸存者偏差。代码直接采用配置中的固定名单，未按历史时点重建资产池。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.06311070855038636, ann_vol=0.034605457271774005, sharpe=1.785979696167371, sharpe_excess=1.3451567112623295, max_drawdown=-0.03219749931936455, total_return=0.20124061610366217, turnover=1.3092415066104692, n_trades=2792
    oos: cagr=0.007505410656637013, ann_vol=0.07316951843593093, sharpe=0.13875749322834804, sharpe_excess=-0.1907670631314807, max_drawdown=-0.20012932478601775, total_return=0.038002205525201393, turnover=1.1368187787036572, n_trades=4886
    full: cagr=0.028021860173570445, ann_vol=0.061605742018182426, sharpe=0.4794210465838699, sharpe_excess=0.1423158461269029, max_drawdown=-0.20012932478601764, total_return=0.24689040888205072, turnover=1.2015201502591457, n_trades=7678
    benchmark: cagr=0.07718174945463652, ann_vol=0.12857631125987595, sharpe=0.6428707138461003, sharpe_excess=0.4813332361393508, max_drawdown=-0.23671483656319103, total_return=0.8105035023250353, turnover=1.448787633897517, n_trades=12072
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "476_ conditional_value_at_risk_under_reward_p"\nuniverse: [SPY, IWM, EFA, EEM, IEF, GLD]\nrisk_free_ticker: BIL\ndata_start: 2014-01-01\nstart: 2017-03-01\nis_end: 2020-02-29\nend: 2025-02-28\nlookback_sessions: 756\nsignal_lag_sessions: 1\ncvar_confidence: 0.90\ndownside_weight: 5.0\nloss_threshold: 0.03\ncost_bps: 2.0\nslippage_bps: 3.0\nannualization_sessions: 252\n'
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
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent


def load_prices(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    request = [tickers, start.isoformat(), end.isoformat(), "daily_auto_adjust_close"]
    key = sha256(json.dumps(request).encode()).hexdigest()[:16]
    snapshot = cache_dir / f"yahoo_{key}.csv"

    if snapshot.exists():
        with snapshot.open() as source:
            fetched_at = source.readline().strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(source, index_col=0, parse_dates=True)
    else:
        with TemporaryDirectory(dir=cache_dir) as session_cache:
            yf.set_tz_cache_location(session_cache)
            quotes = yf.download(
                tickers,
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=12,
            )
        if quotes.empty:
            raise RuntimeError("Yahoo Finance returned no daily prices")
        prices = quotes["Close"].reindex(columns=tickers)
        prices.index = prices.index.tz_localize(None)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if (prices.empty or list(prices.columns) != tickers or not prices.index.is_unique
            or not prices.index.is_monotonic_increasing):
        raise RuntimeError("Unexpected Yahoo Finance price layout")
    if (prices.index[0] < start or prices.index[0] > start + pd.Timedelta(days=7)
            or prices.index[-1] > end or prices.index[-1] < end - pd.Timedelta(days=7)):
        raise RuntimeError("Yahoo Finance prices do not cover the configured dates")
    bad_prices = ~np.isfinite(prices) | (prices <= 0)
    if bad_prices.to_numpy().any():
        missing = bad_prices.sum()
        raise RuntimeError(f"Missing or invalid adjusted closes: {missing[missing > 0].to_dict()}")

    if not snapshot.exists():
        with NamedTemporaryFile(mode="w", dir=cache_dir, delete=False) as output:
            pending = Path(output.name)
            try:
                output.write(fetched_at + "\n")
                prices.to_csv(output)
                output.close()
                pending.replace(snapshot)
            finally:
                pending.unlink(missing_ok=True)
    return prices, fetched_at


def robust_weights(losses, confidence, downside_weight, threshold):
    # Work in percentage points to keep the numerical objective well scaled.
    losses = 100 * losses
    threshold *= 100
    mean_loss = losses.mean(axis=0)
    covariance = np.cov(losses, rowvar=False)
    tail = 1 - confidence
    upper_slope = 1 + downside_weight
    midpoint = (1 + upper_slope) / 2
    radius_squared = midpoint**2 - tail * upper_slope
    size = mean_loss.size
    equal_weights = np.full(size, 1 / size)

    def objective(weights):
        mean = weights @ mean_loss
        projected = covariance @ weights
        sigma = np.sqrt(max(weights @ projected, 0.0))
        gap = mean - threshold
        if sigma == 0:
            slope = upper_slope if gap > 0 else 1.0
            return mean + downside_weight * max(gap, 0.0), slope * mean_loss

        # Lemma 3.1, equations (3.5)–(3.6), scaled for model (4.3).
        # The two positive tail slopes are 1 and upper_slope. Their probability-
        # weighted sum lies in [tail, tail * upper_slope]. The scalar maximizer
        # gives the same three regimes as Theorem 3.1(b).
        tail_slope = midpoint + np.sqrt(radius_squared) * gap / np.hypot(gap, sigma)
        tail_slope = np.clip(tail_slope, tail, tail * upper_slope)
        slope_std = np.sqrt(radius_squared - (tail_slope - midpoint)**2)
        value = threshold + (gap * tail_slope + sigma * slope_std) / tail
        gradient = (tail_slope * mean_loss + slope_std * projected / sigma) / tail
        return value, gradient

    optimum = minimize(
        objective,
        equal_weights,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * size,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(size)},
        options={"ftol": 1e-10, "maxiter": 200},
    )
    if not optimum.success or not np.isfinite(optimum.x).all():
        raise RuntimeError(f"Risk optimization failed: {optimum.message}")
    weights = optimum.x.clip(min=0.0)
    return weights / weights.sum()


def simulate(returns, first, lookback, lag, cost_rate, model=None):
    asset_returns = returns.to_numpy()
    size = asset_returns.shape[1]
    positions = np.zeros(size)
    equal_weights = np.full(size, 1 / size)
    daily_results = []

    for fill in range(first, len(returns)):
        signal = fill - lag
        # The current session's return belongs to the pre-fill weights.
        pre_fill_growth = 1 + positions @ asset_returns[fill]
        drifted = positions * (1 + asset_returns[fill]) / pre_fill_growth
        target = equal_weights
        if model is not None:
            losses = -asset_returns[signal - lookback + 1:signal + 1]
            target = robust_weights(losses, *model)

        # Solve for post-cost capital; fees apply to both buys and sells.
        def cost_equation(remaining):
            return remaining + cost_rate * np.abs(remaining * target - drifted).sum() - 1

        remaining = brentq(cost_equation, 0.0, 1.0)
        changes = np.abs(remaining * target - drifted)
        daily_results.append((pre_fill_growth * remaining - 1, changes.sum(), (changes > 1e-8).sum()))
        positions = target

    return pd.DataFrame(daily_results, index=returns.index[first:], columns=["return", "turnover", "n_trades"])


def metrics(rows, risk_free, annualization):
    daily = rows["return"]
    excess = daily - risk_free.loc[rows.index]
    growth = np.r_[1.0, (1 + daily).cumprod().to_numpy()]
    vol = daily.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    root_year = np.sqrt(annualization)
    return {
        "cagr": float(growth[-1]**(annualization / len(rows)) - 1),
        "ann_vol": float(vol * root_year),
        "sharpe": float(daily.mean() / vol * root_year) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * root_year) if excess_vol > 0 else None,
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1)),
        "total_return": float(growth[-1] - 1),
        "turnover": float(rows["turnover"].mean() * annualization),
        "n_trades": int(rows["n_trades"].sum()),
    }


def main():
    with (ROOT / "config.yaml").open() as source:
        config = yaml.safe_load(source)

    universe = config["universe"]
    risk_free_ticker = config["risk_free_ticker"]
    tickers = [*universe, risk_free_ticker]
    data_start, start, split, end = (pd.Timestamp(config[key]) for key in ("data_start", "start", "is_end", "end"))
    if len(universe) < 2 or len(set(tickers)) != len(tickers):
        raise ValueError("Use at least two distinct universe symbols and a separate risk-free symbol")
    if risk_free_ticker != "BIL":
        raise ValueError("The risk-free proxy must be BIL adjusted returns")
    if not data_start < start <= split < end:
        raise ValueError("Invalid warm-up, evaluation or split dates")
    lag = config["signal_lag_sessions"]
    lookback = config["lookback_sessions"]
    annualization = config["annualization_sessions"]
    if any(type(value) is not int for value in (lag, lookback, annualization)):
        raise ValueError("Lag, lookback and annualization must be integer session counts")
    if lag < 1 or lookback < 2 or annualization < 1:
        raise ValueError("Invalid lag, lookback or annualization session counts")
    confidence = config["cvar_confidence"]
    downside_weight = config["downside_weight"]
    threshold = config["loss_threshold"]
    if not (0 < confidence < 1 and 0 < downside_weight < np.inf and np.isfinite(threshold)):
        raise ValueError("Invalid reward-penalty model parameters")
    cost_bps, slippage_bps = config["cost_bps"], config["slippage_bps"]
    cost_rate = (cost_bps + slippage_bps) / 10_000
    if not (cost_bps >= 0 and slippage_bps >= 0 and cost_rate < 1):
        raise ValueError("Invalid cost or slippage")

    prices, fetched_at = load_prices(tickers, data_start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True total-return adjusted close")
    returns = prices.pct_change(fill_method=None).iloc[1:]
    first = returns.index.searchsorted(start)
    boundary = returns.index.searchsorted(split, side="right")
    if first - lag - lookback + 1 < 0:
        raise ValueError("Insufficient warm-up prices for the first evaluation session")
    if min(boundary - first, len(returns) - boundary) < 2:
        raise ValueError("Both IS and OOS need at least two evaluation sessions")

    model = (confidence, downside_weight, threshold)
    print("PARAM_COMBOS_TRIED: 1")
    daily_results = simulate(returns[universe], first, lookback, lag, cost_rate, model)
    benchmark = simulate(returns[universe], first, lookback, lag, cost_rate)
    risk_free = returns[risk_free_ticker]
    report = {
        "strategy": config["strategy"],
        "data_range": {"start": str(daily_results.index[0].date()), "end": str(daily_results.index[-1].date())},
        "is": metrics(daily_results.iloc[:boundary - first], risk_free, annualization),
        "oos": metrics(daily_results.iloc[boundary - first:], risk_free, annualization),
        "full": metrics(daily_results, risk_free, annualization),
        "benchmark": metrics(benchmark, risk_free, annualization),
    }
    for period, label in (("is", "IS"), ("oos", "OOS"), ("full", "FULL"), ("benchmark", "BENCHMARK (daily equal weight, full period)")):
        print(f"{label}: {json.dumps(report[period], allow_nan=False)}")
    print(f"RESULTS_JSON: {json.dumps(report, allow_nan=False)}")


if __name__ == "__main__":
    main()