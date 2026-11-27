"""541_entropy_regularization_under_no_shorting
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/541_entropy_regularization_under_no_shorting.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 69a306c0280e31031e1c1674f053aa22730544ac on 2026-11-27 09:47 Europe/Oslo, exit 0, output sha256 88a54e22f26eb76114ebd8d787f6c1397e3f13bf99024dae739b8147e546f88d
Data source: yfinance 日度 ETF 收盘价：SPY、TLT、GLD、VNQ、IEF 用于配置，BIL 用作无风险代理，其日收益仅用于事后计算超额夏普。行情优先从本地 CSV 快照读取，缺少快照时通过 yfinance 下载。见 config.yaml:2-3、strategy.py:64-79、strategy.py:85-105、strategy.py:247-252、strategy.py:200-213、README.md:23-27。
Data fetched: 2026-11-27T08:47:42+00:00
Sample period: 配置的取价区间为 2008-01-01 至 2025-12-31，配置的回测区间为 2010-01-01 至 2025-12-31，2010 年前数据用于预热。该次运行记录的实际回测区间为 2010-01-04 至 2025-12-31。见 config.yaml:5-8、README.md:24、logs/run_20261127_091936.log:16、backtest_results.md:30。   data_range (from output): {'start': '2010-01-04', 'end': '2025-12-31'}
IS/OOS split: 样本内从实际回测首日 2010-01-04 至 2018-12-31（含），样本外为 2018-12-31 之后的交易日至 2025-12-31（名义起点 2019-01-01）。稳健性变体只在样本内运行。全期与样本外使用默认参数，均值和协方差按可用历史滚动估计，切分时延续已有持仓。见 config.yaml:6-8、README.md:15-25、README.md:31、strategy.py:185-197、strategy.py:254-273、logs/run_20261127_091936.log:16。
Costs and slippage: 默认费用 5 bps、滑点 5 bps，合计按 10 bps 乘以每次调仓的买入和卖出金额之和（含初始建仓），从执行日组合价值扣减。目标权重按扣减后的组合价值计算。滑点以固定线性费率建模。样本内高摩擦变体将费用和滑点分别设为 10 bps。见 config.yaml:18-19、config.yaml:27、strategy.py:40-42、strategy.py:157-181、strategy.py:254-261、README.md:26-27。
Signal lag: 以月末最后一个交易日 t 的收盘为信号时点，使用截至该收盘的滚动日收益计算目标权重。默认在下一交易日 t+1 的收盘模拟成交。原权重承担截至执行日收盘的收益，新权重从 t+1 收盘至下一交易日收盘的区间开始计收益。见 config.yaml:10-11、README.md:25、strategy.py:114-124、strategy.py:168-181、strategy.py:185-197。
Price adjustment: yfinance auto_adjust=True, daily dividend- and split-adjusted closes
Survivorship bias: 资产池事后固定为仍存续的 SPY、TLT、GLD、VNQ、IEF。README 明确指出可能存在存续偏差和资产选择偏差。见 config.yaml:2、strategy.py:229-233、README.md:35。
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.0869726100155439, ann_vol=0.11809740774830306, sharpe=0.7654871263356343, sharpe_excess=0.7450862235928994, max_drawdown=-0.17697635995192773, total_return=1.1154000379547977, turnover=4.1549897691364714, n_trades=535
    oos: cagr=0.14469798077235052, ann_vol=0.14606466094834714, sharpe=0.9985159190254501, sharpe_excess=0.8258283543291571, max_drawdown=-0.2718390712505643, total_return=1.5698324884403956, turnover=4.771512340375485, n_trades=418
    full: cagr=0.11185319061228483, ann_vol=0.13106022746503562, sharpe=0.8748093514371832, sharpe_excess=0.7803606326414971, max_drawdown=-0.27183907125056594, total_return=4.436223743584278, turnover=4.424641788366257, n_trades=953
    benchmark: cagr=0.07907405830814174, ann_vol=0.08679068539563733, sharpe=0.9204224325014991, sharpe_excess=0.7772566449663841, max_drawdown=-0.23683031459702442, total_return=2.3710933949079287, turnover=0.3658828339568781, n_trades=960
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "541_entropy_regularization_under_no_shorting"\nuniverse: [SPY, TLT, GLD, VNQ, IEF]\nrisk_free_ticker: BIL\n\ndata_start: 2008-01-01\nsample_start: 2010-01-01\nis_end: 2018-12-31\nsample_end: 2025-12-31\n\nrebalance_period: M\nsignal_lag_sessions: 1\nannualization_sessions: 252\n\ndefault:\n  lookback_sessions: 252\n  risk_aversion: 5.0\n  entropy_strength: 0.02\n  cost_bps: 5.0\n  slippage_bps: 5.0\n\nrobustness:\n  entropy_lower: {entropy_strength: 0.01}\n  entropy_higher: {entropy_strength: 0.03}\n  risk_aversion_lower: {risk_aversion: 4.0}\n  risk_aversion_higher: {risk_aversion: 6.0}\n  longer_lookback: {lookback_sessions: 315}\n  higher_friction: {cost_bps: 10.0, slippage_bps: 10.0}\n'
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
import json
import math
import signal
from dataclasses import asdict, dataclass, replace
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
PRICE_ADJUSTMENT = "yfinance auto_adjust=True, daily dividend- and split-adjusted closes"


@dataclass(frozen=True)
class Parameters:
    lookback_sessions: int
    risk_aversion: float
    entropy_strength: float
    cost_bps: float
    slippage_bps: float

    def __post_init__(self):
        if not isinstance(self.lookback_sessions, int) or self.lookback_sessions < 2:
            raise ValueError("lookback_sessions must be an integer of at least 2.")
        values = (self.risk_aversion, self.entropy_strength, self.cost_bps, self.slippage_bps)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("Risk aversion, entropy strength, costs and slippage must be finite and nonnegative.")
        if self.entropy_strength == 0:
            raise ValueError("entropy_strength must be positive.")
        if self.friction >= 1:
            raise ValueError("Combined cost and slippage must be below 10,000 bps.")

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def check_prices(prices, tickers, start, end):
    if prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Expected nonempty adjusted prices with unique, sorted dates.")
    missing = set(tickers) - set(prices.columns)
    if missing:
        raise ValueError(f"Missing adjusted closes for {', '.join(sorted(missing))}.")
    prices = prices.loc[:, tickers]
    # Allow weekends and exchange holidays at the request boundaries.
    first, last = prices.index[[0, -1]]
    if not (0 <= (first - start).days <= 7 and 0 <= (end - last).days <= 7):
        raise ValueError(f"Adjusted close coverage is incomplete: {first.date()} to {last.date()}.")
    invalid = ~np.isfinite(prices) | (prices <= 0)
    if invalid.any().any():
        date = invalid.index[invalid.any(axis=1)][0]
        symbols = ", ".join(invalid.columns[invalid.loc[date]])
        raise ValueError(f"Missing or invalid adjusted close on {date.date()}: {symbols}.")
    return prices


def download_prices(tickers, start, end):
    for attempt in range(2):
        try:
            raw = yf.download(
                tickers,
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=True,
                interval="1d",
                threads=False,
                progress=False,
                timeout=8,
            )
            if raw is None or raw.empty:
                raise ValueError("yfinance returned no daily prices.")
            return check_prices(raw["Close"], tickers, start, end)
        except (KeyError, ValueError) as error:
            if attempt == 1:
                raise RuntimeError(f"Could not load adjusted closes from yfinance: {error}") from error


def load_prices(tickers, start, end):
    key = "_".join(sorted(tickers))
    path = CACHE / f"adjusted_close_{key}_{start.date()}_{end.date()}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as source:
            fetched_at = datetime.fromisoformat(source.readline().strip()).isoformat()
            prices = pd.read_csv(
                source, index_col="Date", parse_dates=["Date"], converters=dict.fromkeys(tickers, float)
            )
        return check_prices(prices, tickers, start, end), fetched_at

    CACHE.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=CACHE, prefix="yfinance-") as yf_cache:
        yf.set_tz_cache_location(yf_cache)
        prices = download_prices(tickers, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Replace the complete snapshot atomically, including its fetch time.
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as destination:
        temporary_path = Path(destination.name)
        try:
            destination.write(fetched_at + "\n")
            prices.to_csv(destination, index_label="Date")
            destination.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def rebalance_schedule(dates, start, period, lag):
    periods = dates.to_period(period)
    period_ends = np.flatnonzero(periods[:-1] != periods[1:])
    schedule = {}
    for signal_row in period_ends:
        fill_row = signal_row + lag
        if fill_row < len(dates) and dates[fill_row] >= start:
            schedule[fill_row] = signal_row
    if not schedule:
        raise ValueError("No scheduled fills fall within the sample.")
    return schedule


def optimal_weights(window, parameters, annual_sessions):
    n_assets = window.shape[1]
    mean = window.mean(axis=0) * annual_sessions
    risk = np.cov(window, rowvar=False) * (annual_sessions * parameters.risk_aversion)
    entropy_strength = parameters.entropy_strength
    ones = np.ones(n_assets)
    equal = ones / n_assets

    def objective(weights):
        log_ratio = np.log(weights * n_assets)
        quadratic_gradient = risk @ weights
        value = weights @ (0.5 * quadratic_gradient - mean + entropy_strength * log_ratio)
        gradient = quadratic_gradient - mean + entropy_strength * (log_ratio + 1.0)
        return value, gradient

    fit = minimize(
        objective,
        equal,
        jac=True,
        method="SLSQP",
        bounds=[(1e-12, 1.0)] * n_assets,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0, "jac": lambda weights: ones},
        options={"ftol": 1e-12, "maxiter": 250},
    )
    weights = fit.x
    if not fit.success or not np.isfinite(weights).all() or weights.min() < 0 or abs(weights.sum() - 1) > 1e-9:
        raise RuntimeError(f"Weight optimization failed: {fit.message}")
    return weights / weights.sum()


def fill_at_close(weights, target, friction):
    def balance(retained):
        traded = np.abs(retained * target - weights).sum()
        return retained + friction * traded - 1.0

    # Targets are fractions of the post-cost value. Solve for that value first.
    retained = brentq(balance, 0.0, 1.0, xtol=1e-14) if friction else 1.0
    trades = np.abs(retained * target - weights)
    return retained, float(trades.sum()), int(np.count_nonzero(trades > 1e-8))


def simulate(asset_returns, start, targets, friction):
    first_row = asset_returns.index.searchsorted(start)
    weights = np.zeros(asset_returns.shape[1])
    rows = []
    for row, returns in enumerate(asset_returns.to_numpy()[first_row:], start=first_row):
        gross = 1.0 + weights @ returns
        weights = weights * (1.0 + returns) / gross
        traded = n_trades = 0
        if row in targets:
            target = targets[row]
            retained, traded, n_trades = fill_at_close(weights, target, friction)
            gross *= retained
            weights = target
        rows.append((gross - 1.0, traded, n_trades))
    return pd.DataFrame(rows, index=asset_returns.index[first_row:], columns=["return", "traded", "n_trades"])


def run_strategy(asset_returns, start, schedule, parameters, annual_sessions):
    returns = asset_returns.to_numpy()
    targets = {}
    for fill_row, signal_row in schedule.items():
        window_start = signal_row + 1 - parameters.lookback_sessions
        if window_start < 0:
            raise ValueError(f"Insufficient history at signal close {asset_returns.index[signal_row].date()}.")
        window = returns[window_start : signal_row + 1]
        try:
            targets[fill_row] = optimal_weights(window, parameters, annual_sessions)
        except RuntimeError as error:
            raise RuntimeError(f"Signal close {asset_returns.index[signal_row].date()}: {error}") from error
    return simulate(asset_returns, start, targets, parameters.friction)


def metrics(daily_results, risk_free_returns, annual_sessions):
    returns = daily_results["return"].to_numpy()
    if len(returns) < 2:
        raise ValueError("Each reported period must have at least two daily returns.")
    excess = returns - risk_free_returns.loc[daily_results.index].to_numpy()
    annual_scale = math.sqrt(annual_sessions)
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    growth = np.r_[1.0, np.cumprod(1.0 + returns)]
    return {
        "cagr": float(growth[-1] ** (annual_sessions / len(returns)) - 1.0),
        "ann_vol": float(volatility * annual_scale),
        "sharpe": float(returns.mean() / volatility * annual_scale) if volatility else None,
        "sharpe_excess": float(excess.mean() / excess_volatility * annual_scale) if excess_volatility else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1.0).min()),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["traded"].mean() * annual_sessions),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    data_start, start, is_end, end = (
        pd.Timestamp(config[key]) for key in ("data_start", "sample_start", "is_end", "sample_end")
    )
    if not data_start < start < is_end < end:
        raise ValueError("Expected data_start < sample_start < is_end < sample_end.")
    universe = config["universe"]
    risk_free_ticker = config["risk_free_ticker"]
    tickers = [*universe, risk_free_ticker]
    if len(universe) < 2 or len(set(tickers)) != len(tickers) or any(not ticker.isalnum() for ticker in tickers):
        raise ValueError("Set at least two distinct ETF symbols and a separate risk-free proxy.")
    lag = config["signal_lag_sessions"]
    annual_sessions = config["annualization_sessions"]
    if not isinstance(lag, int) or lag < 1 or not isinstance(annual_sessions, int) or annual_sessions < 1:
        raise ValueError("Execution lag and annualization must be positive session counts.")
    period = config["rebalance_period"]
    official = Parameters(**config["default"])
    cases = {"default": official}
    for label, adjustments in config["robustness"].items():
        case = replace(official, **adjustments)
        if label in cases or case in cases.values():
            raise ValueError(f"Repeated parameter combination or label: {label}.")
        cases[label] = case

    prices, fetched_at = load_prices(tickers, data_start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    daily_returns = prices.pct_change(fill_method=None).iloc[1:]
    asset_returns = daily_returns[universe]
    risk_free_returns = daily_returns[risk_free_ticker]

    is_returns = asset_returns.loc[:is_end]
    is_risk_free = risk_free_returns.loc[:is_end]
    is_schedule = rebalance_schedule(is_returns.index, start, period, lag)
    print("ROBUSTNESS (IS only)")
    for label, case in cases.items():
        daily_results = run_strategy(is_returns, start, is_schedule, case, annual_sessions)
        result = {"case": label, **asdict(case), **metrics(daily_results, is_risk_free, annual_sessions)}
        print(json.dumps(result, allow_nan=False))
    print(f"PARAM_COMBOS_TRIED: {len(cases)}")

    schedule = rebalance_schedule(asset_returns.index, start, period, lag)
    daily_results = run_strategy(asset_returns, start, schedule, official, annual_sessions)
    equal = np.full(len(universe), 1.0 / len(universe))
    benchmark = simulate(asset_returns, start, dict.fromkeys(schedule, equal), official.friction)
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": str(daily_results.index[0].date()), "end": str(daily_results.index[-1].date())},
        "is": metrics(daily_results.loc[:is_end], risk_free_returns, annual_sessions),
        "oos": metrics(daily_results.loc[daily_results.index > is_end], risk_free_returns, annual_sessions),
        "full": metrics(daily_results, risk_free_returns, annual_sessions),
        "benchmark": metrics(benchmark, risk_free_returns, annual_sessions),
    }
    for label in ("is", "oos", "full"):
        print(f"{label.upper()}: {json.dumps(results[label], allow_nan=False)}")
    print(f"BENCHMARK (equal weight, same universe and schedule): {json.dumps(results['benchmark'], allow_nan=False)}")
    print(f"RESULTS_JSON: {json.dumps(results, allow_nan=False)}")


if __name__ == "__main__":
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)