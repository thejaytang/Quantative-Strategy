"""452_simple_dynamic_stock_bond_gold_portfolio
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/452_simple_dynamic_stock_bond_gold_portfolio.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 9e9ff3660187503321abd71fccfdc940b2ed36ad on 2026-11-27 08:24 Europe/Oslo, exit 0, output sha256 7872580637ffff9810db3a72df90b3c17d79eeb2ed1adde2a5ef354e4ce3eb7a
Data source: Yahoo Finance 经 yfinance 提供 SPY、AGG、GLD 的日收盘价及 ^IRX 的日收盘收益率报价。FRED 公共 CSV 提供 DFF（联邦基金有效利率），用于模拟现金计息。^IRX 是超额夏普比率所用的 13 周国库券贴现率报价。config.yaml:2-7，strategy.py:19-21、48-59、78-87、90-105、168-175、218-232，README.md:13-14。
Data fetched: 2026-11-27T07:24:44+00:00
Sample period: 配置回测区间为 2006-01-01 至 2025-12-31，含首尾日期内的交易日。日志所报首末交易日为 2006-01-03 至 2025-12-31。预热数据的请求起点为 2005-11-01。config.yaml:8-12，strategy.py:220-235、250-252，logs/run_20261127_080229.log:13。   data_range (from output): {'start': '2006-01-03', 'end': '2025-12-31'}
IS/OOS split: 以 2016-01-01 为切分点。IS 为 2006-01-01 至 2015-12-31 区间内的交易日，OOS 为 2016-01-01 至 2025-12-31 区间内的交易日。策略和基准各模拟一次全区间，持仓与资金路径跨切分点延续，再分段计算指标。README 将 OOS 定义为本实现的时间切分，并披露论文参数来自既有历史样本研究。config.yaml:8-12，strategy.py:235-247，README.md:15-16。
Costs and slippage: 策略和基准每笔 ETF 买入或卖出按成交额计 1.0 bps 交易成本及 1.5 bps 固定滑点，合计 2.5 bps，从调仓净值扣减。费用以调仓前漂移持仓与扣费后目标持仓之差的绝对值计算，模拟现金不单独收费。滑点作为固定费率建模，未模拟随规模变化的市场冲击。config.yaml:18-19，strategy.py:138-165、202-203、240-241，README.md:20、25。
Signal lag: 策略使用月末最后一个交易日 t 收盘及此前的 ETF 收益计算目标权重，在下一交易日 t+1 收盘模拟成交。t 到 t+1 的收盘至收盘收益归调仓前持仓，新权重从成交后的下一个收益区间起生效。固定权重基准与策略同日建仓，之后按年末信号使用相同执行滞后。config.yaml:16，strategy.py:108-135、138-163，README.md:9、20。
Price adjustment: yfinance auto_adjust total-return ETF closes; ^IRX yield quotes
Survivorship bias: SPY、AGG、GLD 是事后发表的论文所选的固定 ETF 名单。README 记录三只基金在样本开始前已有历史记录，并披露选取仍存续基金可能带来幸存者偏差和事后选择偏差。config.yaml:2-5，strategy.py:193-195、218-225，README.md:25。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.05864280872500083, ann_vol=0.07284150211318174, sharpe=0.818885066542572, sharpe_excess=0.6658531590949813, max_drawdown=-0.1565099423741625, total_return=0.7668508143853647, turnover=0.9265544965341469, n_trades=360
    oos: cagr=0.09613915590157851, ann_vol=0.07301451318130689, sharpe=1.2939439092678238, sharpe_excess=0.9921142070839735, max_drawdown=-0.1278223743521435, total_return=1.4986632786956742, turnover=0.6973565881638201, n_trades=360
    full: cagr=0.07721666821953521, ann_vol=0.07292900995746855, sharpe=1.056539332259873, sharpe_excess=0.8290793363748842, max_drawdown=-0.1565099423741625, total_return=3.4147652488382505, turnover=0.8120238780402091, n_trades=720
    benchmark: cagr=0.0903564840164175, ann_vol=0.10293742270181125, sharpe=0.8919925094723811, sharpe_excess=0.7307823484873418, max_drawdown=-0.2728429932774945, total_return=4.623782011017771, turnover=0.06238085853492763, n_trades=60
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 452_simple_dynamic_stock_bond_gold_portfolio\nuniverse:\n  SPY: 0.50\n  AGG: 0.30\n  GLD: 0.20\ncash_rate_series: DFF\nexcess_sharpe_ticker: ^IRX\ndates:\n  warmup_start: '2005-11-01'\n  start: '2006-01-01'\n  oos_start: '2016-01-01'\n  end: '2025-12-31'\nlookback_sessions: 11\ntarget_annual_vol: 0.07\nsessions_per_year: 252\nexecution_lag_sessions: 1\ndff_publication_lag_business_days: 2\ncost_bps: 1.0\nslippage_bps: 1.5\n"
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
"""Monthly 50/30/20 volatility control for historical research."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
PARAM_COMBOS_TRIED = 1


def cached_frame(source, symbols, start, end, fetch):
    key = hashlib.sha256(repr((symbols, str(start), str(end))).encode()).hexdigest()[:16]
    path = CACHE / f"{source}_{key}.csv"
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            timestamp = datetime.fromisoformat(handle.readline().strip())
            frame = pd.read_csv(handle, index_col="date", parse_dates=["date"])
        return frame, timestamp

    frame = fetch()
    timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(timestamp.isoformat() + "\n")
            frame.to_csv(handle, index_label="date")
            handle.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return frame, timestamp


def fetch_closes(tickers, start, end):
    with TemporaryDirectory(dir=CACHE) as transport_cache:
        yf.set_tz_cache_location(transport_cache)
        history = yf.download(
            tickers, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=True, ignore_tz=True,
            threads=False, progress=False, timeout=12,
        )
    if history is None or history.empty or "Close" not in history:
        raise RuntimeError("Yahoo Finance returned no daily closes")
    return history["Close"].reindex(columns=tickers).sort_index()


def split_closes(frame, risky_tickers, excess_ticker, start, end):
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Yahoo Finance close dates must be unique and sorted")
    frame = frame.loc[start:end]
    risky_closes = frame[risky_tickers].dropna(how="all")
    bill_quote = frame[excess_ticker].dropna()
    for series in (risky_closes, bill_quote):
        if (series.empty or (series.index[0] - start).days > 7
                or (end - series.index[-1]).days > 7
                or not np.isfinite(series.to_numpy()).all()):
            raise ValueError("Incomplete Yahoo Finance close history for the requested dates")
    if (risky_closes <= 0).any().any():
        raise ValueError("Risky-asset closes must be positive")
    return risky_closes, bill_quote


def fetch_fred(series, start, end):
    query = urlencode({"id": series, "cosd": start.date().isoformat(), "coed": end.date().isoformat()})
    with urlopen(f"{FRED_CSV}?{query}", timeout=20) as response:
        frame = pd.read_csv(response, index_col=0, parse_dates=[0], na_values=["."])
    frame = frame[[series]].sort_index()
    available = frame[series].dropna()
    if (available.empty or (available.index[0] - start).days > 7
            or (end - available.index[-1]).days > 7 or not np.isfinite(available).all()):
        raise ValueError(f"Incomplete FRED observations for {series}")
    return frame


def cash_and_bill_returns(sessions, dff, bill_quote, publication_lag):
    if not dff.index.is_unique or dff.empty:
        raise ValueError("FRED cash-rate observations must be available and unique")
    business_day = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
    release_calendar = pd.date_range(dff.index[0], sessions[-1], freq=business_day)
    observations = dff.reindex(release_calendar).shift(publication_lag).dropna()
    cash_rate_at_close = observations.reindex(sessions, method="ffill")
    bill_rate_at_close = bill_quote.reindex(sessions, method="ffill")
    days = sessions.to_series().diff().dt.days
    cash_yield = cash_rate_at_close.shift(1) / 100.0
    bill_discount = bill_rate_at_close.shift(1) / 100.0
    cash = (1.0 + cash_yield / 360.0) ** days - 1.0
    # A 13-week bill's discount quote implies a price per dollar of face value.
    bill_price = 1.0 - bill_discount * 91.0 / 360.0
    bill = bill_price ** (-days / 91.0) - 1.0
    return cash, bill


def rebalance_targets(returns, base_weights, start, lag, lookback, target_vol, sessions_per_year):
    sessions = returns.index
    blend = returns @ base_weights
    volatility = blend.rolling(lookback).std() * np.sqrt(sessions_per_year)
    dates = sessions.to_series()
    fill_dates = dates.shift(-lag)
    month_ends = dates.groupby(sessions.to_period("M")).last()
    year_ends = dates.groupby(sessions.to_period("Y")).last()

    dynamic = {}
    for signal in month_ends:
        fill = fill_dates.loc[signal]
        if pd.isna(fill) or fill < start:
            continue
        sigma = volatility.loc[signal]
        if not np.isfinite(sigma):
            raise ValueError(f"Insufficient volatility history at the {signal.date()} signal")
        scale = target_vol / max(sigma, target_vol)
        dynamic[fill] = base_weights * scale
    if not dynamic:
        raise ValueError("No monthly rebalances fall inside the sample")

    benchmark = {next(iter(dynamic)): base_weights}
    for signal in year_ends:
        fill = fill_dates.loc[signal]
        if pd.notna(fill) and fill >= start:
            benchmark[fill] = base_weights
    return dynamic, benchmark


def simulate(returns, cash_returns, targets, fee_rate):
    risky_weights = np.zeros(returns.shape[1])
    rows = []
    for session, moves, cash_move in zip(returns.index, returns.to_numpy(), cash_returns, strict=True):
        # Existing holdings earn the entire close-to-close move before a close fill.
        risky_value = risky_weights * (1.0 + moves)
        cash_value = (1.0 - risky_weights.sum()) * (1.0 + cash_move)
        gross_value = risky_value.sum() + cash_value
        risky_weights = risky_value / gross_value
        retained = 1.0
        turnover = 0.0
        trades = 0

        if session in targets:
            target = targets[session]
            # Fees reduce NAV, and target weights apply to the NAV after fees.
            def balance(remaining):
                traded = np.abs(remaining * target - risky_weights).sum()
                return remaining + fee_rate * traded - 1.0

            retained = brentq(balance, 0.0, 1.0)
            traded = np.abs(retained * target - risky_weights)
            turnover = 0.5 * traded.sum()
            trades = np.count_nonzero(traded > 1e-12)
            risky_weights = target
        rows.append((gross_value * retained - 1.0, turnover, trades))

    return pd.DataFrame(rows, index=returns.index, columns=["return", "turnover", "n_trades"])


def metrics(daily_results, bill_returns, sessions_per_year):
    daily = daily_results["return"].to_numpy()
    excess = daily - bill_returns.loc[daily_results.index].to_numpy()
    wealth = np.r_[1.0, np.cumprod(1.0 + daily)]
    annual_factor = sessions_per_year / len(daily)
    vol_scale = np.sqrt(sessions_per_year)
    std = daily.std(ddof=1)
    excess_std = excess.std(ddof=1)
    return {
        "cagr": float(wealth[-1] ** annual_factor - 1.0),
        "ann_vol": float(std * vol_scale),
        "sharpe": float(daily.mean() / std * vol_scale) if std > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_std * vol_scale) if excess_std > 0 else None,
        "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1.0).min()),
        "total_return": float(wealth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum() * annual_factor),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    date_keys = ("warmup_start", "start", "oos_start", "end")
    warmup, start, split, end = (pd.Timestamp(config["dates"][key]) for key in date_keys)
    universe = config["universe"]
    risky_tickers = list(universe)
    base_weights = pd.Series(universe, dtype=float)
    excess_ticker = config["excess_sharpe_ticker"]
    lag = config["execution_lag_sessions"]
    publication_lag = config["dff_publication_lag_business_days"]
    lookback = config["lookback_sessions"]
    target_vol = config["target_annual_vol"]
    sessions_per_year = config["sessions_per_year"]
    cost, slippage = config["cost_bps"], config["slippage_bps"]
    fee_rate = (cost + slippage) / 10_000.0
    if not warmup < start < split < end:
        raise ValueError("Expected warmup_start < start < oos_start < end")
    if (not np.isfinite(base_weights).all() or (base_weights < 0).any()
            or not np.isclose(base_weights.sum(), 1.0)):
        raise ValueError("Risky-asset base weights must be long-only and sum to one")
    base_weights /= base_weights.sum()
    if lag < 1 or publication_lag < 1 or lookback < 2 or not 0 < target_vol < np.inf or sessions_per_year <= 0:
        raise ValueError("Invalid timing or volatility parameters")
    if excess_ticker != "^IRX":
        raise ValueError("The excess-return calculation requires the ^IRX bill discount quote")
    if cost < 0 or slippage < 0 or not 0 <= fee_rate < 1:
        raise ValueError("Invalid fee or slippage assumptions")

    CACHE.mkdir(exist_ok=True)
    tickers = [*risky_tickers, excess_ticker]

    def download_prices():
        raw = fetch_closes(tickers, warmup, end)
        split_closes(raw, risky_tickers, excess_ticker, warmup, end)
        return raw

    closes, price_timestamp = cached_frame("yfinance_adjusted_close", tickers, warmup, end, download_prices)
    cash_series = config["cash_rate_series"]
    rates, rate_timestamp = cached_frame(
        "fred_csv", cash_series, warmup, end, lambda: fetch_fred(cash_series, warmup, end),
    )
    prices, bill_quote = split_closes(closes, risky_tickers, excess_ticker, warmup, end)
    returns = prices.pct_change()
    cash, bill = cash_and_bill_returns(prices.index, rates[cash_series], bill_quote, publication_lag)
    dynamic, benchmark = rebalance_targets(returns, base_weights, start, lag, lookback, target_vol, sessions_per_year)

    daily = returns.loc[start:end]
    cash = cash.loc[daily.index]
    bill = bill.loc[daily.index]
    if not (np.isfinite(daily.to_numpy()).all() and np.isfinite(cash).all() and np.isfinite(bill).all()):
        raise ValueError("Incomplete return inputs inside the evaluation dates")
    dynamic_daily_results = simulate(daily, cash, dynamic, fee_rate)
    benchmark_daily_results = simulate(daily, cash, benchmark, fee_rate)
    masks = {"is": daily.index < split, "oos": daily.index >= split, "full": daily.index.notna()}
    if min(mask.sum() for mask in masks.values()) < 2:
        raise ValueError("Each evaluation period must contain at least two sessions")

    dynamic_periods = {name: metrics(dynamic_daily_results.loc[mask], bill, sessions_per_year) for name, mask in masks.items()}
    benchmark_periods = {name: metrics(benchmark_daily_results.loc[mask], bill, sessions_per_year) for name, mask in masks.items()}
    weight_label = "/".join(f"{weight * 100:g}" for weight in base_weights)
    benchmark_name = f"Annual fixed-weight {'/'.join(risky_tickers)} {weight_label}"
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": daily.index[0].date().isoformat(), "end": daily.index[-1].date().isoformat()},
        **dynamic_periods,
        "benchmark": {"name": benchmark_name, **benchmark_periods["full"], **benchmark_periods},
    }
    print(f"DATA_FETCHED_AT: {max(price_timestamp, rate_timestamp).isoformat()}")
    print(f"Source timestamps | yfinance {price_timestamp.isoformat()} | FRED {rate_timestamp.isoformat()}")
    print(f"PRICE_ADJUSTMENT: yfinance auto_adjust total-return ETF closes; {excess_ticker} yield quotes")
    print(f"PARAM_COMBOS_TRIED: {PARAM_COMBOS_TRIED}")
    print(f"Benchmark | {benchmark_name}")
    rows = {
        **{f"Strategy {k.upper()}": v for k, v in dynamic_periods.items()},
        **{f"Benchmark {k.upper()}": v for k, v in benchmark_periods.items()},
    }
    print(pd.DataFrame.from_dict(rows, orient="index").to_string(float_format=lambda value: f"{value:.6f}"))
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()