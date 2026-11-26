"""339_dancing_dynamic_rebalancing_mechanism
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/339_dancing_dynamic_rebalancing_mechanism.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 3140bbcffa77faae04956fc55adbc5be3b33f896 on 2026-11-26 16:06 Europe/Oslo, exit 0, output sha256 a7d60070f5504489d925506bc42f5212b5ea656d603b30675309580225de918d
Data source: 公开 Yahoo Finance 日线，经 yfinance 获取 SPY、QQQ、IWM、TLT、GLD、DBC 的开盘价与收盘价。若存在对应日期范围的 CSV 缓存则优先读取缓存（config.yaml:2，strategy.py:23-27、46-63，README.md:13）。
Data fetched: 2026-11-26 16:06 Europe/Oslo (run time)
Sample period: 价格请求从 2008-10-01 开始，供波动率预热。配置的初始训练期为 2010-01-01 至 2014-12-31。评估原定截至 2026-11-25，程序按纽约时间 16:00 截断价格请求上限。本次日志记录的实际评估区间为 2015-01-02 至 2026-11-24（config.yaml:3-10、16-17，strategy.py:279-294，README.md:21-22，logs/run_20261125_151354.log:4、11）。   data_range (from output): 2015-01-02 to 2026-11-25
IS/OOS split: 配置的 IS 为 2015-01-01 至 2019-12-31，OOS 为 2020-01-01 至 2026-11-25。日志中的实际交易日区间分别为 2015-01-02 至 2019-12-31、2020-01-02 至 2026-11-24。每年首次开盘前用截至上一交易日收盘的扩展窗口重新拟合，较早的 OOS 数据可进入后续年份的训练，持仓跨年及跨 IS/OOS 边界连续（config.yaml:5-10，strategy.py:184-210、236-245，README.md:22-23，logs/run_20261125_151354.log:2-4）。
Costs and slippage: 每笔买入和卖出各按成交金额收取 5 bps，从组合净值扣除。首次建仓投入开盘现金的 1/(1+0.0005)，其余用于买入费用。再平衡按买卖金额绝对值之和计费，迭代求解扣费后恰好达到目标权重的持仓。开盘滑点未建模，成交采用调整后开盘价（config.yaml:11，strategy.py:91-100、124-151、288，README.md:16、28）。
Signal lag: 交易日 t 收盘形成 Q 策略及漂移阈值信号，最早在下一交易日开盘成交。隔夜收益归原持仓，开盘至收盘收益归开盘操作后的持仓。月度基准按日历在月首交易日开盘调整。评估首日统一按预设等权目标从现金开盘建仓（strategy.py:91-100、124-151、184-210、304，README.md:16）。
Price adjustment: yfinance 下载使用 auto_adjust=True，取拆股和分配调整后的开盘价及收盘价。分配再投资为调整价近似。数据商可能修订历史，代码未保存逐日当时可见的数据版本（strategy.py:23-27、46-75、91-100，README.md:13、28）。
Survivorship bias: 使用固定的六只美国上市 ETF，缺失或无效的开收盘价会触发报错。未构建逐时点基金池，事后选择存续基金仍可能产生选择和存续偏差（config.yaml:2，strategy.py:23-43、294、304，README.md:9、28）。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.07705928285023757, ann_vol=0.08159287599564008, sharpe=0.9507409930607503, max_drawdown=-0.1318451090159828, total_return=0.4485789526218767, turnover=3.458191048759806, n_trades=335
    oos: cagr=0.11034353709461356, ann_vol=0.1346867573901172, sharpe=0.8447532068800416, max_drawdown=-0.22110600414668247, total_return=1.0557503748229098, turnover=7.565664497626324, n_trades=841
    full: cagr=0.09623018976702165, ann_vol=0.11537518382284695, sharpe=0.8542589301354799, max_drawdown=-0.22110600414668255, total_return=1.977916724813007, turnover=11.02385554638613, n_trades=1176
    benchmark monthly_equal_weight full: cagr=0.09610973735333345, ann_vol=0.1164188688755853, sharpe=0.8467079593570729, max_drawdown=-0.22334288209549685, total_return=1.9740327807485714, turnover=5.15097199254995, n_trades=143
    benchmark drift_threshold full: cagr=0.09881675696487854, ann_vol=0.11724475358563109, sharpe=0.8626305352960147, max_drawdown=-0.2253898422052596, total_return=2.062448733866528, turnover=2.784409850945692, n_trades=18
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 339_dancing_dynamic_rebalancing_mechanism\ntickers: [SPY, QQQ, IWM, TLT, GLD, DBC]\ndata_start: 2008-10-01\ntraining_start: 2010-01-01\nin_sample:\n  start: 2015-01-01\n  end: 2019-12-31\nout_of_sample:\n  start: 2020-01-01\n  end: 2026-11-25\ncost_bps: 5.0\nsessions_per_year: 252\nvolatility_window: 20\nstate_median_window: 252\ndrift_benchmark_threshold: 0.10\nq_learning:\n  initial_training_years: 5\n  passes: 10\n  discount: 0.98\n  learning_rate: 0.05\n  exploration: 0.20\n  seed: 339\n'
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
"""Walk-forward test of C130's equal-weight rebalancing policy."""

import json
from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
HOLD, REBALANCE = range(2)


def load_prices(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"yahoo_adjusted_{'-'.join(tickers)}_{start}_{end}.csv"
    columns = pd.MultiIndex.from_product([["Open", "Close"], tickers])

    def validate(raw):
        if raw.empty or not isinstance(raw.index, pd.DatetimeIndex):
            raise ValueError("Yahoo Finance returned no dated bars")
        prices = raw.reindex(columns=columns).sort_index()
        prices.index = prices.index.tz_localize(None).normalize()
        prices = prices.loc[str(start):str(end)]
        values = prices.to_numpy(dtype=float)
        if prices.empty or prices.index.has_duplicates:
            raise ValueError("Yahoo Finance returned no bars or duplicate dates")
        invalid = ~np.isfinite(values) | (values <= 0)
        if invalid.any():
            row, column = np.argwhere(invalid)[0]
            raise ValueError(
                f"Missing or invalid {columns[column]} on {prices.index[row].date()}"
            )
        return prices

    if cache.is_file():
        return validate(pd.read_csv(cache, header=[0, 1], index_col=0, parse_dates=True))

    with TemporaryDirectory(dir=cache_dir, prefix="yahoo_") as yahoo_cache:
        yf.set_tz_cache_location(yahoo_cache)
        for _ in range(3):
            try:
                raw = yf.download(
                    list(tickers),
                    start=str(start),
                    end=str(end + timedelta(days=1)),
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    timeout=15,
                    threads=False,
                    multi_level_index=True,
                )
                prices = validate(raw)
                break
            except (OSError, ValueError) as error:
                last_error = error
        else:
            raise RuntimeError("Could not download a complete Yahoo Finance price panel") from last_error

    with NamedTemporaryFile(dir=cache_dir, suffix=".csv", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        prices.to_csv(temporary_path, index_label="Date")
        temporary_path.replace(cache)
    finally:
        temporary_path.unlink(missing_ok=True)
    return prices


@dataclass
class Market:
    overnight: np.ndarray
    intraday: np.ndarray
    volatility_high: np.ndarray
    year: np.ndarray
    month: np.ndarray

    @classmethod
    def from_prices(cls, prices, volatility_window, median_window, training_start):
        opens, closes = prices["Open"], prices["Close"]
        basket_return = closes.pct_change(fill_method=None).mean(axis=1)
        volatility = basket_return.rolling(volatility_window).std()
        prior_median = volatility.shift(1).rolling(median_window).median()
        if prior_median.iloc[training_start:].isna().any():
            raise ValueError("Insufficient price history to warm up the volatility bins")
        return cls(
            (opens / closes.shift(1)).to_numpy(),
            (closes / opens).to_numpy(),
            (volatility > prior_median).to_numpy(),
            prices.index.year.to_numpy(),
            prices.index.month.to_numpy(),
        )


class StateBins:
    def __init__(self, target, window):
        self.target = target
        self.window = window
        self.history = deque()
        self.sorted_history = []

    def observe(self, weights, volatility_high):
        drift = float(np.abs(weights - self.target).sum())
        drift_high = bool(self.history) and drift > median(self.sorted_history)
        if len(self.history) == self.window:
            expired = self.history.popleft()
            self.sorted_history.pop(bisect_left(self.sorted_history, expired))
        self.history.append(drift)
        insort(self.sorted_history, drift)
        return 2 * int(drift_high) + int(volatility_high)


def fund_portfolio(target, intraday, cost_rate):
    invested = 1.0 / (1.0 + cost_rate)
    close_positions = target * intraday
    close_value = close_positions.sum()
    return close_positions / close_value, np.log(invested * close_value), invested


def trade_to_target(open_weights, target, cost_rate):
    # Post-cost NAV must fund exactly the target holdings and the trading fee.
    retained = 1.0
    while True:
        turnover = float(np.abs(retained * target - open_weights).sum())
        next_retained = 1.0 - cost_rate * turnover
        if abs(next_retained - retained) < 1e-13:
            return next_retained, turnover
        retained = next_retained


def advance(weights, overnight, intraday, target, action, cost_rate):
    open_positions = weights * overnight
    open_value = open_positions.sum()
    turnover = 0.0
    if action == REBALANCE:
        retained, turnover = trade_to_target(open_positions / open_value, target, cost_rate)
        open_positions = target * (open_value * retained)
    close_positions = open_positions * intraday
    close_value = close_positions.sum()
    return close_positions / close_value, np.log(close_value), turnover


def fit_policy(market, start, stop, target, cost_rate, config):
    settings = config["q_learning"]
    q_values = np.zeros((4, 2))
    rng = np.random.default_rng(settings["seed"])
    for _ in range(settings["passes"]):
        bins = StateBins(target, config["state_median_window"])
        weights, _, _ = fund_portfolio(target, market.intraday[start], cost_rate)
        state = bins.observe(weights, market.volatility_high[start])
        for day in range(start + 1, stop):
            if rng.random() < settings["exploration"]:
                action = rng.integers(2)
            else:
                action = q_values[state].argmax()
            weights, reward, _ = advance(
                weights, market.overnight[day], market.intraday[day], target, action, cost_rate
            )
            next_state = bins.observe(weights, market.volatility_high[day])
            continuation = q_values[next_state].max() if day + 1 < stop else 0.0
            error = reward + settings["discount"] * continuation - q_values[state, action]
            q_values[state, action] += settings["learning_rate"] * error
            state = next_state
    return q_values


def evaluate(market, dates, training_start, first_day, target, cost_rate, config):
    names = ("q_policy", "monthly_equal_weight", "drift_threshold")
    results = {
        name: pd.DataFrame(0.0, index=dates[first_day:], columns=["log_return", "turnover"])
        for name in names
    }
    positions = {}
    for name in names:
        positions[name], log_return, turnover = fund_portfolio(
            target, market.intraday[first_day], cost_rate
        )
        results[name].iloc[0] = (log_return, turnover)

    bins = StateBins(target, config["state_median_window"])
    q_values = fit_policy(market, training_start, first_day, target, cost_rate, config)
    for day in range(first_day + 1, len(dates)):
        previous = day - 1
        if market.year[day] != market.year[previous]:
            # The training slice ends at the signal close, before the next open.
            q_values = fit_policy(market, training_start, day, target, cost_rate, config)
        state = bins.observe(positions["q_policy"], market.volatility_high[previous])
        threshold_drift = np.abs(positions["drift_threshold"] - target).sum()
        actions = (
            q_values[state].argmax(),
            REBALANCE if market.month[day] != market.month[previous] else HOLD,
            REBALANCE if threshold_drift >= config["drift_benchmark_threshold"] else HOLD,
        )
        for name, action in zip(names, actions):
            positions[name], log_return, turnover = advance(
                positions[name], market.overnight[day], market.intraday[day],
                target, action, cost_rate,
            )
            results[name].iloc[day - first_day] = (log_return, turnover)
    return results


def metrics(history, sessions_per_year):
    if history.empty:
        raise ValueError("No completed trading sessions in a reporting period")
    log_returns = history["log_return"].to_numpy()
    returns = np.expm1(log_returns)
    cumulative = np.r_[0.0, log_returns.cumsum()]
    volatility = returns.std(ddof=1) if len(returns) > 1 else np.nan
    annualizer = np.sqrt(sessions_per_year)
    values = {
        "cagr": np.expm1(cumulative[-1] * sessions_per_year / len(returns)),
        "ann_vol": volatility * annualizer,
        "sharpe": returns.mean() * annualizer / volatility if volatility > 0 else np.nan,
        "max_drawdown": np.expm1(cumulative - np.maximum.accumulate(cumulative)).min(),
        "total_return": np.expm1(cumulative[-1]),
        "turnover": history["turnover"].sum(),
    }
    return {
        **{key: float(value) if np.isfinite(value) else None for key, value in values.items()},
        "n_trades": int((history["turnover"] > 1e-12).sum()),
    }


def report(config, histories):
    ranges = {"is": config["in_sample"], "oos": config["out_of_sample"]}
    measures = {}
    rows = []
    for name, history in histories.items():
        slices = {
            period: history.loc[str(bounds["start"]):str(bounds["end"])]
            for period, bounds in ranges.items()
        }
        slices["full"] = history
        measures[name] = {}
        for period, sample in slices.items():
            result = metrics(sample, config["sessions_per_year"])
            measures[name][period] = result
            rows.append({
                "Policy": name, "Period": period.upper(),
                "From": str(sample.index[0].date()), "To": str(sample.index[-1].date()),
                **result,
            })
    summary = pd.DataFrame(rows).rename(columns={
        "cagr": "CAGR", "ann_vol": "Ann. vol.", "sharpe": "Sharpe",
        "max_drawdown": "Max DD", "total_return": "Total ret.",
        "turnover": "Turnover", "n_trades": "Trades",
    })
    print(summary.to_string(index=False, formatters={
        **{column: "{:.2%}".format for column in
           ("CAGR", "Ann. vol.", "Max DD", "Total ret.", "Turnover")},
        "Sharpe": "{:.2f}".format,
    }))
    q_results = measures.pop("q_policy")
    dates = histories["q_policy"].index
    result = {
        "strategy": config["strategy"],
        "data_range": f"{dates[0].date()} to {dates[-1].date()}",
        **q_results,
        "benchmark": measures,
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


def main():
    with (ROOT / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    is_dates, oos_dates = config["in_sample"], config["out_of_sample"]
    train_date = config["training_start"]
    first_scored = pd.Timestamp(train_date) + pd.DateOffset(
        years=config["q_learning"]["initial_training_years"]
    )
    if first_scored.date() != is_dates["start"] or not (
        is_dates["start"] <= is_dates["end"] < oos_dates["start"] <= oos_dates["end"]
    ) or oos_dates["start"] - is_dates["end"] != timedelta(days=1):
        raise ValueError("Use adjacent IS and OOS ranges beginning after the training warmup")
    cost_rate = config["cost_bps"] / 10_000
    if not 0 <= cost_rate <= 0.1:
        raise ValueError("cost_bps must be between 0 and 1000")

    now = datetime.now(ZoneInfo("America/New_York"))
    last_complete = now.date() if now.hour >= 16 else now.date() - timedelta(days=1)
    prices = load_prices(config["tickers"], config["data_start"], min(oos_dates["end"], last_complete))
    training_start = int(prices.index.searchsorted(pd.Timestamp(train_date)))
    first_day = int(prices.index.searchsorted(first_scored))
    if not 0 <= training_start < first_day < len(prices):
        raise ValueError("Price history does not cover the training and evaluation periods")
    if (prices.index[training_start] - pd.Timestamp(train_date)).days > 7:
        raise ValueError("Price history does not cover the start of training")
    market = Market.from_prices(
        prices, config["volatility_window"], config["state_median_window"], training_start
    )
    target = np.full(len(config["tickers"]), 1.0 / len(config["tickers"]))
    histories = evaluate(market, prices.index, training_start, first_day, target, cost_rate, config)
    report(config, histories)


if __name__ == "__main__":
    main()
