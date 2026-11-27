"""470_entropic_value_at_risk_parity_for_temper
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/470_entropic_value_at_risk_parity_for_temper.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit e0683ac343a175cb2458afc860f8d1a1066c8e3a on 2026-11-27 08:55 Europe/Oslo, exit 0, output sha256 055a0fdd51a8a52c3f2ed62143c9a7518c393d48e40c3bb297b039506b4e9d0d
Data source: Yahoo Finance 的免费 ETF 日线，经 yfinance 获取。组合标的为 VTI、EFA、VWO、VNQ、AGG、TIP、GLD。BIL 仅作为超额夏普比率的现金收益代理（config.yaml:2-3；README.md:17；strategy.py:36-44、177-191、223-225、236-254）。
Data fetched: 2026-11-27T07:55:57+00:00
Sample period: 配置的行情请求区间为 2005-04-01 至 2026-03-31，回测前的历史行情用于滚动估计。日志记录的全样本交易日区间为 2008-01-02 至 2026-03-31（config.yaml:4-7；README.md:19；strategy.py:39-42、127-143、236-239；logs/run_20261127_080829.log:6）。   data_range (from output): 2008-01-02 to 2026-03-31
IS/OOS split: 以 2016-12-31 为界，样本内包含分界日及之前的交易日，样本外包含其后的交易日。实际 IS 为 2008-01-02 至 2016-12-30，OOS 为 2017-01-03 至 2026-03-31。组合连续模拟后按日期分段报告（config.yaml:5-7；strategy.py:236-254；logs/run_20261127_080829.log:4-5）。
Costs and slippage: 每笔 ETF 买卖按成交金额建模 5 bps 费用和 5 bps 固定滑点，合计 10 bps，于成交日从组合净值扣除。成交额以扣费后目标持仓和收盘漂移后持仓的差额绝对值计算。初次建仓、后续调仓及等权基准均适用（config.yaml:11-12；README.md:23；strategy.py:151-172、210-212、236-241）。
Signal lag: 用截至月末最后一个交易日 t 收盘的过去 12 个日历月 ETF 日收益计算目标权重，配置为下一交易日收盘执行。执行日收盘前的价格变化按旧权重计入，新权重从该收盘之后的收益区间起生效（config.yaml:8-10；README.md:22；strategy.py:127-143、151-172、236-239）。
Price adjustment: yfinance auto_adjust total-return adjusted closes
Survivorship bias: 固定使用论文选定的七只长期存续 ETF。README 明确披露早期样本存在存续偏差和事后选择偏差，OOS 也是早于论文发表的历史切分（config.yaml:2；README.md:9-11、30；strategy.py:200-202、223-225、236-241）。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.039296018615857964, ann_vol=0.07285814361397339, sharpe=0.565431502327354, sharpe_excess=0.5370261400907219, max_drawdown=-0.2287461248431547, total_return=0.41444787166127894, turnover=4.395899848130754, n_trades=756
    oos: cagr=0.0610244715794912, ann_vol=0.06981099577775819, sharpe=0.8835317939150763, sharpe_excess=0.5583974741966912, max_drawdown=-0.19116266690077266, total_return=0.726407350549823, turnover=3.0044885157994012, n_trades=777
    full: cagr=0.05023660103164107, ann_vol=0.07132738935806081, sharpe=0.7229090950496083, sharpe_excess=0.5474183822527304, max_drawdown=-0.2287461248431547, total_return=1.4419132026055874, turnover=7.400388363930155, n_trades=1533
    benchmark: cagr=0.06437400554575001, ann_vol=0.13130431145736843, sharpe=0.5408785470822973, sharpe_excess=0.4455053420186708, max_drawdown=-0.40102898627722106, total_return=2.1153194353379825, turnover=3.8702385366040506, n_trades=1533
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 470_entropic_value_at_risk_parity_for_temper\nuniverse: [VTI, EFA, VWO, VNQ, AGG, TIP, GLD]\nrisk_free_symbol: BIL\ndata_start: 2005-04-01\nbacktest_start: 2008-01-02\nis_end: 2016-12-31\nend_date: 2026-03-31\nlookback_months: 12\ntail_probability: 0.05\nsignal_lag_sessions: 1\ncost_bps: 5\nslippage_bps: 5\n'
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
"""Monthly equal contributions to empirical EVaR deviation."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, minimize


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252


def price_cache_path(symbols, start, end):
    cache_key = json.dumps([sorted(symbols), str(start.date()), str(end.date())])
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"adjusted_close_{digest}.csv"


def load_prices(symbols, start, end, cache_file):
    CACHE_DIR.mkdir(exist_ok=True)
    if cache_file.exists():
        with cache_file.open(encoding="utf-8") as source:
            fetched_at = source.readline().strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(source, index_col="Date", parse_dates=["Date"])
        return prices[symbols], fetched_at

    with TemporaryDirectory(prefix="yf_", dir=CACHE_DIR) as temp_dir:
        # Keep yfinance's transient session files inside the strategy folder.
        yf.set_tz_cache_location(temp_dir)
        prices = yf.download(
            symbols, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=True, ignore_tz=True,
            progress=False, threads=False, timeout=20,
        )["Close"].reindex(columns=symbols).sort_index()

    missing = prices.columns[prices.notna().sum() == 0].tolist()
    if missing:
        raise RuntimeError(f"Yahoo Finance returned no adjusted closes for {missing}")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return prices, fetched_at


def cache_prices(prices, fetched_at, cache_file):
    if cache_file.exists():
        return
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE_DIR, delete=False) as output:
        temp_path = Path(output.name)
        try:
            output.write(f"{fetched_at}\n")
            prices.to_csv(output, index_label="Date")
            output.flush()
            temp_path.replace(cache_file)
        finally:
            temp_path.unlink(missing_ok=True)


def evar_tilt(losses, tail_probability):
    """Return weights for the entropy-constrained worst-case mean loss."""
    gaps = losses - losses.max()
    worst = gaps == 0
    if worst.mean() >= tail_probability:
        return worst.astype(float) / worst.sum()

    gaps /= -gaps.min()
    entropy_limit = -np.log(tail_probability)

    def tilted(inverse_temperature):
        logits = inverse_temperature * gaps
        values = np.exp(logits)
        normalizer = values.sum()
        probabilities = values / normalizer
        divergence = probabilities @ logits - np.log(normalizer / len(losses))
        return probabilities, divergence

    def entropy_gap(inverse_temperature):
        return tilted(inverse_temperature)[1] - entropy_limit

    upper = 1.0
    while entropy_gap(upper) <= 0:
        upper *= 2
    optimum = brentq(entropy_gap, 0, upper)
    return tilted(optimum)[0]


def equal_risk_weights(window, tail_probability):
    losses = window.mean(axis=0) - window
    scale = losses.std(axis=0)
    if (scale <= 0).any():
        raise ValueError("EVaR risk budgeting requires nonconstant return series")
    scenarios = losses / scale
    budget = 1.0 / window.shape[1]

    def marginal_risk(scaled_weights):
        probabilities = evar_tilt(scenarios @ scaled_weights, tail_probability)
        return probabilities @ scenarios

    def objective(scaled_weights):
        marginal = marginal_risk(scaled_weights)
        risk = scaled_weights @ marginal
        # At the minimum, each unnormalized Euler contribution equals its budget.
        value = risk - budget * np.log(scaled_weights).sum()
        gradient = marginal - budget / scaled_weights
        return value, gradient

    result = minimize(
        objective, np.full(window.shape[1], budget), jac=True, method="L-BFGS-B",
        bounds=[(1e-10, None)] * window.shape[1],
        options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 400},
    )
    contributions = result.x * marginal_risk(result.x)
    if not result.success or not np.allclose(contributions, budget, rtol=1e-4, atol=1e-8):
        raise RuntimeError(f"EVaR risk budgeting did not converge: {result.message}")
    weights = result.x / scale
    return weights / weights.sum()


def monthly_targets(returns, backtest_start, lookback_months, lag, tail_probability):
    dates = returns.index
    months = dates.to_period("M")
    new_months = np.flatnonzero(months[1:] != months[:-1]) + 1
    targets = {}
    for first_session in new_months:
        fill_row = first_session + lag - 1
        if fill_row >= len(dates) or dates[fill_row] < backtest_start:
            continue

        signal_date = dates[first_session - 1]
        window_start = signal_date - pd.DateOffset(months=lookback_months)
        if dates[0] > window_start:
            continue
        window = returns.loc[(dates > window_start) & (dates <= signal_date)]
        try:
            targets[dates[fill_row]] = equal_risk_weights(window.to_numpy(), tail_probability)
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(f"Signal on {signal_date.date()}: {error}") from error
    if not targets:
        raise ValueError("No monthly signals with enough estimation history before the end date")
    return targets


def simulate(returns, targets, fee_rate):
    daily_results = []
    weights = np.zeros(returns.shape[1])
    for date, asset_returns in zip(returns.index, returns.to_numpy()):
        growth = 1.0 + weights @ asset_returns
        weights = weights * (1.0 + asset_returns) / growth
        turnover = 0.0
        trades = 0
        target = targets.get(date)
        if target is not None:
            # Price movement through this close applies to the previous weights.
            turnover = 0.5 * (np.abs(target - weights).sum() + abs(1.0 - weights.sum()))

            def fee_balance(net_fraction):
                gross_traded = np.abs(net_fraction * target - weights).sum()
                return net_fraction + fee_rate * gross_traded - 1.0

            net_fraction = brentq(fee_balance, 0.0, 1.0)
            trades = np.count_nonzero(np.abs(net_fraction * target - weights) > 1e-10)
            growth *= net_fraction
            weights = target
        daily_results.append((growth - 1.0, turnover, trades))
    return pd.DataFrame(daily_results, index=returns.index,
                        columns=["return", "turnover", "n_trades"])


def metrics(daily_results, risk_free):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise ValueError("Each reporting period needs at least two sessions")
    excess = returns - risk_free.reindex(returns.index)
    cumulative = (1.0 + returns).cumprod()
    volatility = float(returns.std())
    excess_volatility = float(excess.std())
    annualize = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(cumulative.iloc[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(volatility * annualize),
        "sharpe": float(returns.mean() / volatility * annualize) if volatility > 0 else None,
        "sharpe_excess": (float(excess.mean() / excess_volatility * annualize)
                          if excess_volatility > 0 else None),
        "max_drawdown": float((cumulative / cumulative.cummax().clip(lower=1.0) - 1).min()),
        "total_return": float(cumulative.iloc[-1] - 1),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    risk_free_symbol = config["risk_free_symbol"]
    data_start = pd.Timestamp(config["data_start"])
    backtest_start = pd.Timestamp(config["backtest_start"])
    is_end = pd.Timestamp(config["is_end"])
    end = pd.Timestamp(config["end_date"])
    lookback = config["lookback_months"]
    lag = config["signal_lag_sessions"]
    tail_probability = config["tail_probability"]
    cost = config["cost_bps"]
    slippage = config["slippage_bps"]
    fee_rate = (cost + slippage) / 10_000

    if not data_start < backtest_start <= is_end < end:
        raise ValueError("Dates must allow estimation history and distinct IS and OOS periods")
    if len(set(universe)) != len(universe) or not universe or risk_free_symbol in universe:
        raise ValueError("Choose a nonempty, unique universe separate from the cash proxy")
    if not isinstance(lookback, int) or lookback < 1 or not isinstance(lag, int) or lag < 1:
        raise ValueError("Lookback months and signal lag sessions must be positive integers")
    if not 0 < tail_probability < 1 or not 0 <= fee_rate < 1 or min(cost, slippage) < 0:
        raise ValueError("Invalid tail probability or fee rates")

    symbols = [*universe, risk_free_symbol]
    cache_file = price_cache_path(symbols, data_start, end)
    prices, fetched_at = load_prices(symbols, data_start, end, cache_file)
    if (not prices.index.is_monotonic_increasing or not prices.index.is_unique
            or not prices.index[0] <= data_start + pd.Timedelta(days=7)
            or not prices.index[-1] >= end - pd.Timedelta(days=7)):
        raise ValueError("Adjusted closes do not cover the requested date range")
    arrays = (prices[universe].to_numpy(), prices.loc[backtest_start:, risk_free_symbol].to_numpy())
    for values in arrays:
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Missing or invalid adjusted closes in the required sample")
    cache_prices(prices, fetched_at, cache_file)

    asset_returns = prices[universe].pct_change(fill_method=None).iloc[1:]
    targets = monthly_targets(asset_returns, backtest_start, lookback, lag, tail_probability)
    evaluated = asset_returns.loc[min(targets):]
    daily_results = simulate(evaluated, targets, fee_rate)
    equal_weight = np.full(len(universe), 1.0 / len(universe))
    benchmark_results = simulate(evaluated, dict.fromkeys(targets, equal_weight), fee_rate)

    risk_free = prices.loc[evaluated.index, risk_free_symbol].pct_change(fill_method=None)
    risk_free.iloc[0] = 0.0
    periods = {
        "is": daily_results.loc[daily_results.index <= is_end],
        "oos": daily_results.loc[daily_results.index > is_end],
        "full": daily_results,
    }
    results = {
        "strategy": config["strategy"],
        "data_range": f"{evaluated.index[0].date()} to {evaluated.index[-1].date()}",
        **{period: metrics(series, risk_free) for period, series in periods.items()},
        "benchmark": {"name": "Monthly equal weight", **metrics(benchmark_results, risk_free)},
    }

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return adjusted closes")
    print("PARAM_COMBOS_TRIED: 1")
    for period, series in periods.items():
        print(f"{period.upper()} ({series.index[0].date()} to {series.index[-1].date()}): "
              + json.dumps(results[period], allow_nan=False))
    print("BENCHMARK: " + json.dumps(results["benchmark"], allow_nan=False))
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()