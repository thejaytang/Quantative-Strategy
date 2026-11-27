"""546_generalizing_markowitz_portfolio_optimiz
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/546_generalizing_markowitz_portfolio_optimiz.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c0a8f76240a0431b019ab49c34040a95a765860a on 2026-11-27 10:04 Europe/Oslo, exit 0, output sha256 c94893a3bcaa98e6b32d8948b63e1c622785efbd6741cbd85d81996722afec24
Data source: Yahoo Finance 公开 ETF 日线，经 yfinance 获取。配置池为 XLK、XLF、XLE、XLV，SPY 为买入持有基准，BIL 的日收益仅用于事后计算超额夏普。（config.yaml:2-4；strategy.py:66-94、201-214、239-245、260-274；README.md:21-22；logs/run_20261127_092129.log:2-4）
Data fetched: 2026-11-27T09:04:47+00:00
Sample period: 预热数据请求从 2008-01-01 开始。配置计分起止为 2010-01-01 至 2025-12-31，日志记录实际计分区间为 2010-01-04 至 2025-12-31，首次模拟成交为 2010-01-04。（config.yaml:5-8；strategy.py:156-164、188-189、275-277；README.md:23；logs/run_20261127_092129.log:3）   data_range (from output): 2010-01-04 to 2025-12-31
IS/OOS split: 按日期留出。样本内截至配置界点 2017-12-31（含），实际为 2010-01-04 至 2017-12-29。样本外为 2018-01-02 至 2025-12-31，沿用持仓和固定的默认超参数并继续滚动估计。七组额外参数仅在样本内运行。（config.yaml:6-8、11-27；strategy.py:222-228、255-258、263-274；README.md:23、34；logs/run_20261127_092129.log:3、12-22）
Costs and slippage: 默认买入和卖出均按各 ETF 的模拟成交金额计 5 bps 交易成本和 2 bps 滑点，合计 7 bps，从组合价值中扣除。滑点以固定费率建模。费用基数为各 ETF 买卖金额绝对值之和，成交后的目标金额按扣费后组合价值计算。SPY 初次建仓使用相同费率。样本内成本压力组为 10 bps 成本加 5 bps 滑点。（config.yaml:16-17、27；strategy.py:45-47、145-153、173-182、192-198、265-272；README.md:29、34、38）
Signal lag: 月末最后一个可用交易日收盘 t，用截至该收盘的滚动日收益计算目标权重。默认在下一交易日 t+1 的收盘执行。t+1 收盘前的日收益归原持仓，新权重从该收盘之后开始产生收益。（config.yaml:9、12；strategy.py:156-186、251-253；README.md:9、28）
Price adjustment: yfinance auto_adjust=True, split- and distribution-adjusted closes
Survivorship bias: 固定选择四只现今仍存续的行业 ETF，存在存续和事后选样偏差。配置列出固定名单，样本外是按时间留出的回溯检验。（config.yaml:2；strategy.py:239-241；README.md:23-24）
Parameter combinations tried: 8
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.13683136782811234, ann_vol=0.16873627971418564, sharpe=0.8448807658052685, sharpe_excess=0.8413485887579799, max_drawdown=-0.28672675533071545, total_return=1.7855139518766103, turnover=18.592557266375877, n_trades=187
    oos: cagr=0.18255869180720952, ann_vol=0.24018820235752011, sharpe=0.8189244542825435, sharpe_excess=0.7176050124917068, max_drawdown=-0.311532222062224, total_return=2.811858238955921, turnover=20.08406574480178, n_trades=188
    full: cagr=0.1594582638016564, ann_vol=0.20752370637900133, sharpe=0.8173310175967543, sharpe_excess=0.7573541665065936, max_drawdown=-0.31153222206222386, total_return=9.61798430718746, turnover=38.676623011177654, n_trades=375
    benchmark: cagr=0.13921067884502558, ann_vol=0.17207629552497677, sharpe=0.8438745576772824, sharpe_excess=0.7715746873059892, max_drawdown=-0.337172555955769, total_return=7.01434075006029, turnover=0.99930048965724, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "546_generalizing_markowitz_portfolio_optimiz"\nuniverse: [XLK, XLF, XLE, XLV]\nbenchmark: SPY\ncash_proxy: BIL\ndata_start: "2008-01-01"\nsample_start: "2010-01-01"\nis_end: "2017-12-31"\nsample_end: "2025-12-31"\nsignal_lag_sessions: 1\n\nmodel:\n  lookback_sessions: 252\n  risk_aversion: 5.0\n  quadratic_turnover_penalty: 0.20\n  covariance_ridge: 0.001\n  transaction_cost_bps: 5.0\n  slippage_bps: 2.0\n\n# Each case inherits the default model and runs through the IS cutoff only.\nrobustness:\n  lookback_shorter: {lookback_sessions: 210}\n  lookback_longer: {lookback_sessions: 294}\n  risk_aversion_lower: {risk_aversion: 4.0}\n  risk_aversion_higher: {risk_aversion: 6.0}\n  turnover_penalty_lower: {quadratic_turnover_penalty: 0.15}\n  turnover_penalty_higher: {quadratic_turnover_penalty: 0.25}\n  higher_cost_and_slippage: {transaction_cost_bps: 10.0, slippage_bps: 5.0}\n'
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
"""Monthly long-only allocation using a generalized quadratic risk measure."""

import hashlib
import json
import os
import re
import signal
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
TRADE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class Model:
    lookback_sessions: int
    risk_aversion: float
    quadratic_turnover_penalty: float
    covariance_ridge: float
    transaction_cost_bps: float
    slippage_bps: float

    def __post_init__(self):
        if type(self.lookback_sessions) is not int or self.lookback_sessions < 2:
            raise ValueError("lookback_sessions must be an integer of at least two")
        positive = (self.risk_aversion, self.covariance_ridge)
        nonnegative = (self.quadratic_turnover_penalty, self.transaction_cost_bps, self.slippage_bps)
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise ValueError("Risk aversion and covariance ridge must be finite and positive")
        if not all(np.isfinite(value) and value >= 0 for value in nonnegative) or self.fee_rate >= 1:
            raise ValueError("Penalties and fees must be finite, nonnegative and fees must total under 100%")

    @property
    def fee_rate(self):
        return (self.transaction_cost_bps + self.slippage_bps) / 10_000


def prepare_prices(prices, tickers, start, end):
    prices = prices.reindex(columns=tickers).astype(float).dropna(how="all")
    dates = prices.index
    if (prices.empty or not isinstance(dates, pd.DatetimeIndex) or dates.tz is not None
            or not dates.is_unique or not dates.is_monotonic_increasing):
        raise ValueError("Expected a nonempty, sorted panel of unique daily closes")
    first, last = dates[[0, -1]]
    if not (start <= first <= start + timedelta(days=7) and end - timedelta(days=7) <= last <= end):
        raise ValueError(f"Requested price coverage {start.date()} to {end.date()}, got {first.date()} to {last.date()}")
    invalid = ~np.isfinite(prices) | (prices <= 0)
    if invalid.any().any():
        date, column = np.argwhere(invalid.to_numpy())[0]
        raise ValueError(f"Missing or invalid close for {tickers[column]} on {prices.index[date].date()}")
    return prices


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    cache_key = "|".join(["yf-adjusted-close-v1", start.date().isoformat(), end.date().isoformat(), *tickers])
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    path = CACHE / f"adjusted_close_{digest}.csv"
    if path.is_file():
        try:
            with path.open(encoding="utf-8") as source:
                fetched_at = source.readline().strip().removeprefix("# fetched_at=")
                if datetime.fromisoformat(fetched_at).utcoffset() != timedelta(0):
                    raise ValueError("Expected a UTC fetch time")
                cached = pd.read_csv(source, index_col=0, parse_dates=[0])
            return prepare_prices(cached, tickers, start, end), fetched_at
        except (OSError, ValueError) as error:
            print(f"CACHE_RELOAD: {error}", flush=True)

    with TemporaryDirectory(prefix="yf_", dir=CACHE) as yf_cache:
        yf.set_tz_cache_location(yf_cache)
        try:
            downloaded = yf.download(
                tickers, start=start.date().isoformat(), end=(end + timedelta(days=1)).date().isoformat(),
                interval="1d", auto_adjust=True, ignore_tz=True, keepna=True,
                threads=False, progress=False, timeout=12,
            )
        finally:
            yf.set_tz_cache_location(str(CACHE))
    if downloaded is None or downloaded.empty or "Close" not in downloaded:
        raise RuntimeError("yfinance did not return adjusted closes")
    prices = prepare_prices(downloaded["Close"], tickers, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as out:
        temporary_path = Path(out.name)
        try:
            out.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(out, index_label="date")
            out.close()
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def budget_solution(quadratic, reward):
    """Proposition 8 with reward = mean - linear risk term and sum(weights) = 1."""
    ones = np.ones(len(reward))
    center, direction = np.linalg.solve(quadratic, np.column_stack((reward, ones))).T
    return center + direction * ((1.0 - center.sum()) / direction.sum())


def target_weights(window, reference, model):
    mean = window.mean(axis=0) * SESSIONS_PER_YEAR
    covariance = np.cov(window, rowvar=False) * SESSIONS_PER_YEAR
    penalty = model.quadratic_turnover_penalty
    quadratic = model.risk_aversion * covariance + (model.covariance_ridge + penalty) * np.eye(len(mean))
    reward = mean + penalty * reference
    unrestricted = budget_solution(quadratic, reward)
    if unrestricted.min() >= 0:
        return unrestricted

    # Projection in the quadratic metric preserves the original utility objective.
    def distance(weights):
        difference = weights - unrestricted
        gradient = quadratic @ difference
        return 0.5 * (difference @ gradient), gradient

    ones = np.ones(len(mean))
    start = unrestricted.clip(min=0.0)
    result = minimize(
        distance, start / start.sum(), jac=True, method="SLSQP",
        bounds=[(0.0, 1.0)] * len(mean),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1, "jac": lambda _: ones},
        options={"ftol": 1e-13, "maxiter": 150},
    )
    if not result.success or abs(result.x.sum() - 1) > 1e-8:
        raise RuntimeError(f"Long-only optimization failed: {result.message}")
    weights = result.x.clip(min=0.0)
    return weights / weights.sum()


def fill_at_close(drifted, target, fee_rate):
    # In units of value before execution, q is the fraction left after both charges.
    # ETF amounts change from drifted to q * target, with the remainder paying fees.
    def budget_gap(q):
        return q + fee_rate * np.abs(q * target - drifted).sum() - 1.0

    fraction_left = brentq(budget_gap, 0.0, 1.0, xtol=1e-14)
    trade_sizes = np.abs(fraction_left * target - drifted)
    return fraction_left, trade_sizes.sum(), int(np.count_nonzero(trade_sizes > TRADE_TOLERANCE))


def simulate(asset_returns, start, lag, model):
    dates = asset_returns.index
    values = asset_returns.to_numpy()
    first = int(dates.searchsorted(start))
    month_end = np.flatnonzero(dates[:-1].month != dates[1:].month)
    fill_for_signal = {
        i: i + lag for i in month_end
        if i >= model.lookback_sessions and first <= i + lag < len(dates)
    }
    if not fill_for_signal:
        raise ValueError("No month-end signal with enough warm-up and an execution inside the sample")

    first_fill = next(iter(fill_for_signal.values()))
    pending = {}
    weights = np.zeros(values.shape[1])
    daily = []
    for i in range(min(first, min(fill_for_signal)), len(dates)):
        if i >= first:
            # The position carried into the session earns the close-to-close move.
            gross_return = weights @ values[i]
            weights = weights * (1 + values[i]) / (1 + gross_return)
            fraction_left, turnover, trades = 1.0, 0.0, 0
            if i in pending:
                target = pending.pop(i)
                fraction_left, turnover, trades = fill_at_close(weights, target, model.fee_rate)
                weights = target
            daily.append((gross_return - (1 + gross_return) * (1 - fraction_left), turnover, trades))

        if i in fill_for_signal:
            window = values[i + 1 - model.lookback_sessions : i + 1]
            pending[fill_for_signal[i]] = target_weights(window, weights, model)

    daily_results = pd.DataFrame(daily, index=dates[first:], columns=["return", "turnover", "n_trades"])
    return daily_results, dates[first_fill]


def buy_and_hold(returns, first_fill, fee_rate):
    daily_results = pd.DataFrame({"return": 0.0, "turnover": 0.0, "n_trades": 0}, index=returns.index)
    after_fill = returns.index > first_fill
    daily_results.loc[after_fill, "return"] = returns.loc[after_fill]
    left, turnover, trades = fill_at_close(np.zeros(1), np.ones(1), fee_rate)
    daily_results.loc[first_fill] = (left - 1, turnover, trades)
    return daily_results


def metrics(daily_results, cash_returns):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise ValueError("Each performance period needs at least two sessions")
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    volatility = returns.std()
    excess = returns - cash_returns.loc[returns.index]
    excess_volatility = excess.std()
    growth = np.r_[1.0, (1 + returns.to_numpy()).cumprod()]
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(volatility * annualizer),
        "sharpe": float(returns.mean() / volatility * annualizer) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_volatility * annualizer) if excess_volatility > 0 else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, cash_returns, is_end):
    in_sample = daily_results.index <= is_end
    return {
        "is": metrics(daily_results.loc[in_sample], cash_returns),
        "oos": metrics(daily_results.loc[~in_sample], cash_returns),
        "full": metrics(daily_results, cash_returns),
    }


def print_metrics(rows):
    print(pd.DataFrame.from_dict(rows, orient="index").to_string(float_format=lambda value: f"{value:.6f}"))


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)

    universe = config["universe"]
    benchmark, cash_proxy = config["benchmark"], config["cash_proxy"]
    tickers = list(dict.fromkeys([*universe, benchmark, cash_proxy]))
    if len(set(universe)) != len(universe) or len(universe) < 2:
        raise ValueError("The universe must contain at least two distinct sector ETFs")
    if not all(re.fullmatch(r"[A-Z]{1,5}", ticker) for ticker in tickers) or cash_proxy != "BIL":
        raise ValueError("Use U.S. ETF tickers and BIL as the cash proxy")
    data_start, sample_start, is_end, sample_end = (
        pd.Timestamp(config[key]) for key in ("data_start", "sample_start", "is_end", "sample_end")
    )
    if not data_start < sample_start < is_end < sample_end:
        raise ValueError("Expected data_start < sample_start < is_end < sample_end")
    lag = config["signal_lag_sessions"]
    if type(lag) is not int or lag < 1:
        raise ValueError("signal_lag_sessions must be an integer of at least one")

    official = Model(**config["model"])
    alternatives = {name: replace(official, **overrides) for name, overrides in config["robustness"].items()}
    if len(set(alternatives.values()) | {official}) != 1 + len(alternatives):
        raise ValueError("Robustness models must be distinct from each other and from the default")

    prices, fetched_at = load_prices(tickers, data_start, sample_end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, split- and distribution-adjusted closes")
    returns = prices.pct_change(fill_method=None)
    cash_returns = returns[cash_proxy]
    is_asset_returns = returns.loc[:is_end, universe]
    robustness = {}
    for name, model in alternatives.items():
        daily_results, _ = simulate(is_asset_returns, sample_start, lag, model)
        robustness[name] = metrics(daily_results, cash_returns.loc[:is_end])

    daily_results, first_fill = simulate(returns[universe], sample_start, lag, official)
    benchmark_results = buy_and_hold(returns.loc[daily_results.index, benchmark], first_fill, official.fee_rate)
    strategy_summary = period_metrics(daily_results, cash_returns, is_end)
    benchmark_summary = period_metrics(benchmark_results, cash_returns, is_end)
    first_date, last_date = daily_results.index[[0, -1]]
    data_range = f"{first_date.date()} to {last_date.date()}"
    print(f"SAMPLE: {data_range} | IS through {is_end.date()} | first fill {first_fill.date()}")
    print(f"STRATEGY AND {benchmark} BENCHMARK | excess Sharpe vs {cash_proxy}")
    print_metrics({
        **{f"Strategy {period.upper()}": summary for period, summary in strategy_summary.items()},
        **{f"{benchmark} {period.upper()}": summary for period, summary in benchmark_summary.items()},
    })
    print("ROBUSTNESS | IS only | default fixed in config.yaml")
    print_metrics({"default": strategy_summary["is"], **robustness})
    print(f"PARAM_COMBOS_TRIED: {1 + len(robustness)}")
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **strategy_summary,
        "benchmark": {"name": benchmark, **benchmark_summary["full"], **benchmark_summary},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(840)
    main()