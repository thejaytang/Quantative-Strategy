"""549_drawdown_risk_beyond_brownian_motion_a_m
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/549_drawdown_risk_beyond_brownian_motion_a_m.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit f0c72714f025c64b653010840b4be414cbe39932 on 2026-11-27 12:28 Europe/Oslo, exit 0, output sha256 590cb1fc714e01bf0b8383a42e326481bd53a3fb3a763766bb424556ae599815
Data source: Yahoo Finance（yfinance）的 SPY、BIL ETF 日收盘价，分别用于权益仓位和现金代理。优先读取 data_cache/yfinance_SPY_BIL_2008-01-01_2025-12-31.json，缓存缺失时下载。。
Data fetched: 2026-11-27T11:28:28+00:00
Sample period: 实际回测评估期为 2010-01-04 至 2025-12-31。请求的数据起始日为 2008-01-01，评估前的数据用于信号预热。。 data_range (from output): {'start': '2010-01-04', 'is_end': '2017-12-29', 'oos_start': '2018-01-02', 'end': '2025-12-31'}
IS/OOS split: 样本内 2010-01-04 至 2017-12-29，样本外 2018-01-02 至 2025-12-31，以 2017-12-31 为配置切分点。默认参数下先连续模拟，再分段统计，跨期保留仓位；六组稳健性对照仅使用样本内和预热数据。。
Costs and slippage: 每条买卖腿计 1 bp 成本和 2 bp 固定滑点，合并为每腿 3 bp。首次买入仅计买入腿，ETF 切换先从卖出所得扣除卖出费用，再用剩余资金支付买入金额及买入费用，持仓不变时不收费。SPY 买入持有基准采用同一费用模型。。
Signal lag: 每周首个有价格记录的交易日收盘 t 用截至该收盘的数据计算信号，在下一交易日收盘执行。执行当日截至收盘的收益归原仓位，新仓位自执行收盘后开始计收益。首个评估日收盘按当时已生效的滞后目标建仓。。
Price adjustment: Yahoo Adj Close for signals and BIL excess returns, as-quoted Close for fills with separate splits and distributions
Survivorship bias: SPY 和 BIL 是事后选定且持续存续的两只 ETF，存在资产选择与存续偏差。回测使用 ETF 历史价格，未用当前指数成分股名单重建历史。。
Parameter combinations tried: 8
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.07105854918671284, ann_vol=0.11920129563521199, sharpe=0.6357128580954825, sharpe_excess=0.630494167938349, max_drawdown=-0.2172737718295069, total_return=0.7304170180004383, turnover=6.659506019727342, n_trades=61
    oos: cagr=0.0738945235579227, ann_vol=0.12326402482035934, sharpe=0.6404091859745082, sharpe_excess=0.4434573441878667, max_drawdown=-0.2481494130319568, total_return=0.7663578925262886, turnover=4.826472516556935, n_trades=44
    full: cagr=0.07247489420330977, ann_vol=0.12123363827195012, sharpe=0.6380887390535539, sharpe_excess=0.535506758313145, max_drawdown=-0.24814941303195726, total_return=2.056535757106855, turnover=5.743444793366585, n_trades=105
    benchmark full: cagr=0.11996015207307731, ann_vol=0.1481659687957519, sharpe=0.8390521431485154, sharpe_excess=0.7548901592435677, max_drawdown=-0.2914823885160718, total_return=5.104908331540547, turnover=0.056027178856323565, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 549_drawdown_risk_beyond_brownian_motion_a_m\nuniverse:\n  equity: SPY\n  cash_proxy: BIL\ndates:\n  data_start: "2008-01-01"\n  evaluation_start: "2010-01-01"\n  is_end: "2017-12-31"\n  end: "2025-12-31"\nexecution:\n  lag_sessions: 1\n  cost_bps: 1.0\n  slippage_bps: 2.0\n  cash_buffer: 0.12\n  distribution_delay_days: 60\nsignal:\n  lookback_days: 252\n  horizon_days: 63\n  quantile: 0.90\n  paths: 2048\n  seed: 164\n  hurst_scales: [1, 2, 4, 8, 16]\n  hurst_weight: 0.50\n  hurst_min: 0.40\n  hurst_max: 0.70\nrobustness:\n  lookbacks: [210, 294]\n  quantiles: [0.85, 0.95]\n  high_cost_multiplier: 2.0\n'
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
"""Weekly SPY/BIL momentum with a simulated drawdown threshold."""

import json
import signal
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
TRADING_DAYS = 252
CALENDAR_START = pd.Timestamp("2008-01-01")
CALENDAR_END = pd.Timestamp("2025-12-31")
EXCEPTIONAL_CLOSURES = pd.DatetimeIndex(["2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])
DATA_FIELDS = ("signal_close", "close", "distributions", "splits")


class NyseHolidays(AbstractHolidayCalendar):
    # Early closes remain trading sessions.
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


@dataclass(frozen=True)
class SignalSpec:
    lookback_days: int
    horizon_days: int
    quantile: float
    paths: int
    seed: int
    hurst_scales: tuple[int, ...]
    hurst_weight: float
    hurst_min: float
    hurst_max: float
    drawdown_filter: bool = True

    def validate(self):
        if (
            len(self.hurst_scales) < 2
            or tuple(sorted(set(self.hurst_scales))) != self.hurst_scales
            or self.hurst_scales[0] < 1
            or self.lookback_days <= self.hurst_scales[-1]
        ):
            raise ValueError("Hurst scales must be increasing and shorter than the lookback.")
        if not 0 <= self.hurst_weight <= 1 or not (
            0 < self.hurst_min <= 0.5 <= self.hurst_max < 1
        ):
            raise ValueError("Invalid Hurst bounds or estimation weight.")
        if self.horizon_days < 2 or self.paths < 2 or not 0 < self.quantile < 1:
            raise ValueError("Invalid drawdown horizon, path count, or quantile.")


def check_market_data(market, data_start, end):
    index = market["close"].index
    if index.empty or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Yahoo prices are empty or their dates are invalid.")
    holidays = NyseHolidays().holidays(data_start, end).union(EXCEPTIONAL_CLOSURES)
    expected = pd.bdate_range(data_start, end).difference(holidays)
    if not index.equals(expected):
        missing = expected.difference(index).strftime("%Y-%m-%d").tolist()
        extra = index.difference(expected).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Yahoo data differ from NYSE sessions. Missing: {missing}; extra: {extra}.")
    for field, frame in market.items():
        values = frame.to_numpy(dtype=float)
        valid = values >= 0 if field == "distributions" else values > 0
        if not np.isfinite(values).all() or not valid.all():
            raise ValueError(f"Yahoo {field} data contain incomplete or invalid values.")


def load_market_data(tickers, data_start, end):
    CACHE.mkdir(exist_ok=True)
    start_date = data_start.date().isoformat()
    end_date = end.date().isoformat()
    cache_file = CACHE / f"yfinance_{'_'.join(tickers)}_{start_date}_{end_date}_actions_v1.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        dates = pd.DatetimeIndex(cached["dates"], dtype="datetime64[ns]", name="date")
        market = {
            field: pd.DataFrame(cached[field], index=dates, columns=list(tickers))
            for field in DATA_FIELDS
        }
        check_market_data(market, data_start, end)
        return market, cached["fetched_at"]

    # Include later corporate actions to undo Yahoo's split restatement of historical quotes.
    with TemporaryDirectory(dir=CACHE, prefix=".yfinance-") as download_cache:
        yf.set_tz_cache_location(download_cache)
        daily = yf.download(
            list(tickers),
            start=start_date,
            interval="1d",
            auto_adjust=False,
            actions=True,
            ignore_tz=True,
            threads=False,
            progress=False,
            timeout=15,
        )
    if daily is None or daily.empty:
        raise RuntimeError("yfinance returned no daily prices for " + ", ".join(tickers))
    daily.index = daily.index.astype("datetime64[ns]").rename("date")
    splits = daily["Stock Splits"].loc[:, list(tickers)].replace(0, 1.0)
    if not np.isfinite(splits).to_numpy().all() or not splits.gt(0).to_numpy().all():
        raise ValueError("Yahoo split data contain incomplete or invalid values.")
    split_factor = splits.shift(-1, fill_value=1.0).iloc[::-1].cumprod().iloc[::-1]
    dividends = daily["Dividends"].loc[:, list(tickers)]
    if "Capital Gains" in daily.columns.get_level_values(0):
        dividends = dividends + daily["Capital Gains"].reindex(columns=list(tickers), fill_value=0.0)
    market = {
        "signal_close": daily["Adj Close"].loc[:, list(tickers)].loc[:end],
        "close": (daily["Close"].loc[:, list(tickers)] * split_factor).loc[:end],
        "distributions": (dividends * split_factor).loc[:end],
        "splits": splits.loc[:end],
    }
    check_market_data(market, data_start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "fetched_at": fetched_at,
        "dates": market["close"].index.strftime("%Y-%m-%d").tolist(),
        **{field: frame.to_dict(orient="list") for field, frame in market.items()},
    }
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            json.dump(payload, temp, allow_nan=False)
            temp.close()
            temp_path.replace(cache_file)
        finally:
            temp_path.unlink(missing_ok=True)
    return market, fetched_at


def estimate_hurst(window, spec):
    if spec.hurst_weight == 0:
        return 0.5
    path = np.r_[0.0, np.cumsum(window - window.mean())]
    scales = spec.hurst_scales
    dispersions = np.array(
        [np.std(path[width:] - path[:-width], ddof=1) for width in scales]
    )
    if (dispersions <= 0).any():
        return 0.5
    raw_hurst = np.polyfit(np.log(scales), np.log(dispersions), 1)[0]
    shrunk = 0.5 + spec.hurst_weight * (raw_hurst - 0.5)
    return float(np.clip(shrunk, spec.hurst_min, spec.hurst_max))


def drawdown_limit(window, hurst, resample_indices, spec):
    # Use common uniform ranks, mapped to the current empirical return distribution.
    drift = window.mean()
    shocks = np.sort(window)[resample_indices] - drift
    scale = spec.horizon_days ** (hurst - 0.5)
    paths = np.cumsum(drift + shocks * scale, axis=1)
    # Include the zero log-return at the path's starting close.
    peaks = np.maximum(0.0, np.maximum.accumulate(paths, axis=1))
    return float(np.quantile((peaks - paths).max(axis=1), spec.quantile))


def execution_targets(prices, equity, cash_proxy, spec, lag):
    log_prices = np.log(prices)
    log_returns = log_prices.diff()
    equity_returns = log_returns[equity].to_numpy()
    excess = (log_returns[equity] - log_returns[cash_proxy]).to_numpy()
    equity_log_prices = log_prices[equity].to_numpy()

    # Week membership and its first observed session are known by that close.
    weeks = prices.index.to_period("W-SUN")
    first_sessions = np.flatnonzero(np.r_[True, weeks[1:] != weeks[:-1]])
    picks = pd.Series(pd.NA, index=prices.index, dtype="string")
    warmup = spec.lookback_days
    if spec.drawdown_filter:
        rng = np.random.default_rng(spec.seed)
        uniforms = rng.random((spec.paths, spec.horizon_days))
        resample_indices = (uniforms * spec.lookback_days).astype(np.intp)
        warmup = max(warmup, spec.horizon_days)
    for day in first_sessions[first_sessions >= warmup]:
        past = slice(day + 1 - spec.lookback_days, day + 1)
        excess_window = excess[past]
        # A positive mean with nonzero dispersion is a positive trailing Sharpe.
        chosen = cash_proxy
        if excess_window.mean() > 0 and excess_window.std(ddof=1) > 0:
            chosen = equity
            if spec.drawdown_filter:
                window = equity_returns[past]
                hurst = estimate_hurst(window, spec)
                limit = drawdown_limit(window, hurst, resample_indices, spec)
                peak = equity_log_prices[day - spec.horizon_days : day + 1].max()
                if peak - equity_log_prices[day] >= limit:
                    chosen = cash_proxy
        picks.iloc[day] = chosen
    return picks.shift(lag).ffill()


def simulate(market, targets, cash_proxy, fee, cash_buffer, distribution_delay_days):
    closes = market["close"]
    instruments = closes.columns.get_indexer(targets)
    if (instruments < 0).any():
        raise ValueError("Missing or unknown execution targets. Check the signal warm-up.")
    first = closes.index.get_loc(targets.index[0])
    stop = first + len(targets)
    if first == 0 or not targets.index.equals(closes.index[first:stop]):
        raise ValueError("Execution requires consecutive market sessions and a prior close.")

    prices = closes.to_numpy()
    splits = market["splits"].to_numpy()
    distributions = market["distributions"].to_numpy()
    cash_returns = market["signal_close"][cash_proxy].pct_change(fill_method=None)
    risk_free = cash_returns.iloc[first:stop].to_numpy(copy=True)
    risk_free[0] = 0.0  # Evaluation starts at the first session's close.
    distribution_delay = pd.Timedelta(days=distribution_delay_days)
    pending_distributions = deque()
    active = None
    shares = receivables = 0.0
    cash = previous_value = 1.0
    rows = []

    for day, (date, target) in enumerate(zip(targets.index, instruments), start=first):
        change = target != active
        buy_shares = planned_buy_notional = 0.0
        # Size using prior-close cash and prices.
        if change:
            estimated_sale = shares * prices[day - 1, active] if active is not None else 0.0
            available = cash + estimated_sale * (1.0 - fee)
            planned_buy_notional = available * (1.0 - cash_buffer) / (1.0 + fee)
            buy_shares = planned_buy_notional / prices[day - 1, target]

        # Convert to the fill session's share units for splits effective by its open.
        if active is not None:
            shares *= splits[day, active]
        sale_shares = 0.0
        if change:
            sale_shares = shares
            buy_shares *= splits[day, target]

        old_value = 0.0
        if active is not None:
            distribution = shares * distributions[day, active]
            if distribution:
                pending_distributions.append((date + distribution_delay, distribution))
                receivables += distribution
            old_value = shares * prices[day, active]
        before_fills = cash + receivables + old_value

        fees = turnover = 0.0
        n_trades = 0
        position_value = old_value
        if change:
            purchase = buy_shares * prices[day, target]
            fees = (old_value + purchase) * fee
            cash += old_value - purchase - fees
            if cash < 0:
                raise RuntimeError(f"Insufficient cash for fixed share quantities on {date.date()}: {cash:.8g}")
            turnover = (old_value + purchase) / before_fills
            n_trades = int(sale_shares > 0) + int(buy_shares > 0)
            active, shares = target, buy_shares
            position_value = purchase

        while pending_distributions and pending_distributions[0][0] <= date:
            _, payment = pending_distributions.popleft()
            receivables -= payment
            cash += payment
        value = cash + receivables + position_value
        rows.append((
            value / previous_value - 1.0, turnover, n_trades, cash, fees,
            buy_shares, sale_shares, planned_buy_notional,
        ))
        previous_value = value

    daily_results = pd.DataFrame(
        rows,
        columns=[
            "return", "turnover", "n_trades", "cash", "fees",
            "buy_shares", "sale_shares", "planned_buy_notional",
        ],
        index=targets.index,
    )
    daily_results["risk_free"] = risk_free
    return daily_results


def metrics(daily_results):
    returns = daily_results["return"].to_numpy()
    excess = returns - daily_results["risk_free"].to_numpy()
    compounded = np.r_[1.0, np.cumprod(1.0 + returns)]
    vol = np.std(returns, ddof=1)
    excess_vol = np.std(excess, ddof=1)
    annualizer = np.sqrt(TRADING_DAYS)
    return {
        "cagr": float(compounded[-1] ** (TRADING_DAYS / len(returns)) - 1.0),
        "ann_vol": float(annualizer * vol),
        "sharpe": float(annualizer * returns.mean() / vol) if vol else None,
        "sharpe_excess": float(annualizer * excess.mean() / excess_vol) if excess_vol else None,
        "max_drawdown": float((compounded / np.maximum.accumulate(compounded) - 1.0).min()),
        "total_return": float(compounded[-1] - 1.0),
        "turnover": float(TRADING_DAYS * daily_results["turnover"].mean()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def by_period(daily_results, is_end):
    is_rows = daily_results.index <= is_end
    if min(int(is_rows.sum()), int((~is_rows).sum())) < 2:
        raise ValueError("Both evaluation periods must contain at least two sessions.")
    return {
        "is": metrics(daily_results.loc[is_rows]),
        "oos": metrics(daily_results.loc[~is_rows]),
        "full": metrics(daily_results),
    }


def run_robustness(market, sessions, tickers, spec, execution, settings):
    print("ROBUSTNESS (IS only)")
    fee = (execution["cost_bps"] + execution["slippage_bps"]) * 1e-4
    cases = {
        (replace(spec, lookback_days=lookback), fee): f"lookback={lookback}"
        for lookback in settings["lookbacks"]
    }
    cases.update({
        (replace(spec, quantile=quantile), fee): f"quantile={quantile}"
        for quantile in settings["quantiles"]
    })
    cases[(replace(spec, hurst_weight=0.0), fee)] = "H=0.5, empirical shocks"
    multiplier = settings["high_cost_multiplier"]
    cases[(spec, fee * multiplier)] = f"cost and slippage x{multiplier:g}"
    cases[(replace(spec, drawdown_filter=False), fee)] = "momentum only, no drawdown filter"
    cases.pop((spec, fee), None)
    for (scenario, case_fee), name in cases.items():
        scenario.validate()
        if not 0 <= case_fee < 1:
            raise ValueError("Invalid robustness cost and slippage.")
        targets = execution_targets(market["signal_close"], *tickers, scenario, execution["lag_sessions"])
        runs = simulate(
            market, targets.reindex(sessions), tickers[1], case_fee,
            execution["cash_buffer"], execution["distribution_delay_days"],
        )
        print(f"  {name}  {json.dumps(metrics(runs), allow_nan=False)}")
    return len(cases)


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    tickers = config["universe"]["equity"], config["universe"]["cash_proxy"]
    if tickers != ("SPY", "BIL"):
        raise ValueError("The research universe is SPY and BIL.")
    equity, cash_proxy = tickers
    dates = config["dates"]
    data_start, start, is_end, end = (
        pd.Timestamp(dates[key])
        for key in ("data_start", "evaluation_start", "is_end", "end")
    )
    if not CALENDAR_START <= data_start < start < is_end < end <= CALENDAR_END:
        raise ValueError("Invalid warm-up, evaluation or split dates for the 2008–2025 NYSE calendar.")
    signal_settings = config["signal"].copy()
    signal_settings["hurst_scales"] = tuple(signal_settings["hurst_scales"])
    spec = SignalSpec(**signal_settings)
    spec.validate()
    execution = config["execution"]
    lag = execution["lag_sessions"]
    cost, slippage = execution["cost_bps"], execution["slippage_bps"]
    fee = (cost + slippage) * 1e-4
    if not isinstance(lag, int) or lag < 1 or not (0 <= fee < 1 and cost >= 0 and slippage >= 0):
        raise ValueError("Use a positive whole-session lag and nonnegative execution charges.")

    cash_buffer = execution["cash_buffer"]
    distribution_delay_days = execution["distribution_delay_days"]
    if not 0 < cash_buffer < 1 or not isinstance(distribution_delay_days, int) or distribution_delay_days < 1:
        raise ValueError("Use a positive fractional cash buffer and a positive distribution delay in days.")

    market, fetched_at = load_market_data(tickers, data_start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(
        "PRICE_ADJUSTMENT: Yahoo Adj Close for signals and BIL excess returns, "
        "as-quoted Close for fills with separate splits and distributions"
    )
    prices = market["signal_close"]
    sessions = prices.loc[start:end].index
    is_sessions = prices.loc[start:is_end].index
    cases = run_robustness(
        {field: frame.loc[:is_end] for field, frame in market.items()},
        is_sessions, tickers, spec, execution, config["robustness"],
    )

    targets = execution_targets(prices, *tickers, spec, lag).reindex(sessions)
    daily_results = simulate(market, targets, cash_proxy, fee, cash_buffer, distribution_delay_days)
    strategy_periods = by_period(daily_results, is_end)
    benchmark_targets = pd.Series(equity, index=sessions, dtype="string")
    benchmark_results = simulate(market, benchmark_targets, cash_proxy, fee, cash_buffer, distribution_delay_days)
    benchmark = by_period(benchmark_results, is_end)
    data_range = {
        "start": sessions[0].date().isoformat(),
        "is_end": is_sessions[-1].date().isoformat(),
        "oos_start": sessions[len(is_sessions)].date().isoformat(),
        "end": sessions[-1].date().isoformat(),
    }
    for label in ("is", "oos", "full"):
        print(f"STRATEGY {label.upper()} {json.dumps(strategy_periods[label], allow_nan=False)}")
        print(f"BENCHMARK {label.upper()} {json.dumps(benchmark[label], allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {1 + cases}")
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **strategy_periods,
        "benchmark": {"name": f"{equity} buy and hold with cash reserve and cash distributions", **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(9 * 60)
    main()
