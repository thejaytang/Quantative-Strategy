"""303_timing_moving_average_distance
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/303_timing_moving_average_distance.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 2b23c5bbf706aea933c0cba5f5805efb450ee206 on 2026-11-26 14:27 Europe/Oslo, exit 0, output sha256 94e3a1ed619c739628db1df69abedbdf1879c804e7dd9246ec1eac8851ab4ede
Data source: Yahoo Finance 美国上市国家股票 ETF 日频复权 Open/Close，由 yfinance 获取或从 CSV 缓存读取。标的为 EWC、EWA、EWG、EWJ、EWU、EWL、EWS、EWT、EWY、EWP（config.yaml:2；strategy.py:20-30、34-54；README.md:11-15）。
Data fetched: 2026-11-26 14:27 Europe/Oslo (run time)
Sample period: 行情请求窗口为 2008-09-01 至 2026-11-24，含配置首尾日期，2008-09-01 起的样本前数据供信号预热（config.yaml:3-9；strategy.py:45-51、175-179、190-196）。运行日志记录的完整评估区间为 2010-01-04 至 2026-11-24（logs/run_20261125_132843.log:4、15）。   data_range (from output): 2010-01-04 to 2026-11-24
IS/OOS split: 回溯性时间切分，配置边界含首尾。IS 配置为 2010-01-01 至 2020-12-31，实际交易日为 2010-01-04 至 2020-12-31。OOS 配置为 2021-01-01 至 2026-11-24，实际交易日为 2021-01-04 至 2026-11-24。每个组合先连续模拟，再按区间统计，切分时持仓延续（config.yaml:4-9；README.md:19-23；strategy.py:190-209；logs/run_20261125_132843.log:2-3）。
Costs and slippage: 单边 5 bps，按每次调仓中实际买入和卖出的金额分别收费，首次建仓也计费。程序按调仓前开盘净值求解扣费后的目标敞口，并以当日复权开盘价计算份额。未建模滑点、买卖价差或市场冲击（config.yaml:15；strategy.py:107-132、180、205；README.md:18、27）。
Signal lag: 在月末最后一个美国交易日 t 收盘后用复权收盘价计算 SMA(21) / SMA(252) − 1。调仓日期设为下一美国交易日，即下月首个交易日，并按该日复权开盘价模拟成交（config.yaml:10-11；strategy.py:79-87、94-104、107-132、190-205；README.md:5）。
Price adjustment: 下载设置为 auto_adjust=True。信号、模拟开盘成交和收盘估值使用经分红及拆股复权的价格，缓存也保存复权 Open/Close（strategy.py:20-30、34-54、62-70、190-205；README.md:11）。
Survivorship bias: 全期采用同一份固定的十只 ETF 名单。README 明示事后选择和基金幸存者偏差，名单遗漏已关闭或历史上无法交易的国家基金（config.yaml:2；strategy.py:173-174、190-205；README.md:27）。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.049928169237897, ann_vol=0.19117697594076852, sharpe=0.35113538096854463, max_drawdown=-0.3728434361492935, total_return=0.7080617472947066, turnover=57.77387854888789
    oos: cagr=0.12419871083374656, ann_vol=0.22492898158080815, sharpe=0.6332078915920224, max_drawdown=-0.33075870133818763, total_return=0.9897750979554982, turnover=32.28558144605719
    full: cagr=0.07523504995979335, ann_vol=0.2035609350754299, sharpe=0.45867381657213385, max_drawdown=-0.3728434361492935, total_return=2.398658730537353, turnover=90.05945999494509
    benchmark  full: cagr=0.07406300868113891, ann_vol=0.1827707804481326, sharpe=0.48279661702527366, max_drawdown=-0.36786516152282456, total_return=2.336716866788497, turnover=5.659666331594233
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 303_timing_moving_average_distance\ntickers: [EWC, EWA, EWG, EWJ, EWU, EWL, EWS, EWT, EWY, EWP]\ndata_start: 2008-09-01  # Signal warmup only.\nin_sample:\n  start: 2010-01-01\n  end: 2020-12-31\nout_of_sample:\n  start: 2021-01-01\n  end: 2026-11-24\nshort_sma_sessions: 21\nlong_sma_sessions: 252\ntop_n: 3\nmomentum_lookback_months: 12\nmomentum_skip_months: 1\ntransaction_cost_bps: 5  # One-way, per dollar bought or sold.\nsessions_per_year: 252\n'
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
"""C94 country ETF rotation using daily Yahoo Finance bars."""

from math import isfinite, sqrt
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
import json

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"


def validate_bars(bars, tickers):
    columns = pd.MultiIndex.from_product([["Open", "Close"], tickers])
    if bars is None or bars.empty:
        raise ValueError("Yahoo Finance returned no price bars.")
    if not isinstance(bars.index, pd.DatetimeIndex) or not bars.index.is_unique:
        raise ValueError("Expected one daily price bar per U.S. trading session.")
    bars = bars.reindex(columns=columns).sort_index()
    invalid = (~np.isfinite(bars) | bars.le(0)).any()
    if invalid.any():
        missing = sorted({ticker for _, ticker in invalid[invalid].index})
        raise ValueError(f"Missing or invalid adjusted Open/Close bars for: {', '.join(missing)}")
    return bars


def load_bars(tickers, start, end):
    CACHE_DIR.mkdir(exist_ok=True)
    symbols = "-".join(sorted(tickers))
    path = CACHE_DIR / f"yahoo_adjusted_{symbols}_{start.date()}_{end.date()}.csv"
    if path.exists():
        try:
            cached = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
            return validate_bars(cached, tickers)
        except (ValueError, TypeError) as error:
            print(f"Refreshing invalid price cache: {error}", flush=True)

    yf.set_tz_cache_location(str(CACHE_DIR / "yfinance"))
    stop = end + pd.Timedelta(days=1)  # Yahoo's end date is exclusive.
    for attempt in range(2):
        try:
            downloaded = yf.download(
                tickers, start=str(start.date()), end=str(stop.date()),
                interval="1d", auto_adjust=True, ignore_tz=True,
                threads=4, progress=False, timeout=10,
            )
            bars = validate_bars(downloaded, tickers)
            break
        except (ValueError, OSError) as error:
            if attempt == 1:
                raise RuntimeError("Could not retrieve a complete Yahoo Finance ETF panel.") from error
            print(f"Retrying Yahoo Finance download: {error}", flush=True)
            sleep(2)

    with NamedTemporaryFile(mode="w", dir=CACHE_DIR, delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            bars.to_csv(temp)
            temp.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    return bars


def top_weights(scores, count):
    # Column order from config breaks tied ranks consistently.
    winners = scores.rank(axis=1, ascending=False, method="first").le(count)
    return winners.astype(float).div(winners.sum(axis=1), axis=0)


def monthly_targets(close, config):
    months = close.index.to_period("M")
    # Each month starts with the immediately preceding session's closing signal.
    starts = np.flatnonzero(months[1:] != months[:-1]) + 1
    signal_dates = close.index[starts - 1]
    execution_dates = close.index[starts]
    short = close.rolling(config["short_sma_sessions"]).mean()
    long = close.rolling(config["long_sma_sessions"]).mean()
    mad = (short / long - 1).loc[signal_dates]

    month_end_close = close.loc[signal_dates]
    recent = month_end_close.shift(config["momentum_skip_months"])
    earlier = month_end_close.shift(config["momentum_lookback_months"])
    momentum = recent / earlier - 1

    count = config["top_n"]
    targets = {
        "MAD rotation": top_weights(mad, count),
        "Equal-weight countries": pd.DataFrame(
            1.0 / close.shape[1], index=signal_dates, columns=close.columns
        ),
        "12–1 momentum": top_weights(momentum, count),
    }
    for weights in targets.values():
        weights.index = execution_dates
    return targets


def simulate(open_prices, close_prices, targets, fee_rate):
    units = np.zeros(close_prices.shape[1])
    cash = previous_nav = 1.0
    rows = []
    # Index alignment marks non-rebalance days with missing targets.
    targets = targets.reindex(close_prices.index)
    for opening, closing, target in zip(
        open_prices.to_numpy(), close_prices.to_numpy(), targets.to_numpy(), strict=True
    ):
        turnover = 0.0
        if np.isfinite(target).all():
            opening_values = units * opening
            opening_nav = cash + opening_values.sum()
            prior_weights = opening_values / opening_nav
            # Solve for the NAV fraction remaining after paying for actual fills.
            remaining = brentq(
                lambda fraction: (
                    fraction + fee_rate * np.abs(fraction * target - prior_weights).sum() - 1
                ),
                0.0, 1.0,
            )
            new_weights = remaining * target
            turnover = np.abs(new_weights - prior_weights).sum()
            units = new_weights * opening_nav / opening
            cash = 0.0
        closing_nav = cash + units @ closing
        rows.append((closing_nav / previous_nav - 1.0, turnover))
        previous_nav = closing_nav
    return pd.DataFrame(rows, index=close_prices.index, columns=["return", "turnover"])


def metrics(daily_results, sessions_per_year):
    returns = daily_results["return"]
    equity = (1.0 + returns).cumprod().to_numpy()
    drawdown = equity / np.maximum.accumulate(np.maximum(equity, 1.0)) - 1.0
    vol = returns.std(ddof=1)
    values = {
        "cagr": equity[-1] ** (sessions_per_year / len(returns)) - 1.0,
        "ann_vol": vol * sqrt(sessions_per_year),
        "sharpe": returns.mean() / vol * sqrt(sessions_per_year) if vol > 0 else np.nan,
        "max_drawdown": drawdown.min(),
        "total_return": equity[-1] - 1.0,
        "turnover": daily_results["turnover"].sum(),
    }
    return {key: float(value) if isfinite(value) else None for key, value in values.items()}


def show_metrics(results, periods):
    print("Net performance by portfolio and period")
    for period, dates in periods.items():
        print(f"  {period.upper():4} {dates[0].date()} through {dates[-1].date()}")
    print(f"{'Portfolio':<24} {'Period':<6} {'CAGR':>9} {'Ann. vol':>9} {'Sharpe':>8}"
          f" {'Max DD':>9} {'Total':>10} {'Turnover':>10}")
    formats = (".2%", ".2%", ".2f", ".2%", ".2%", ".2f")
    for name, segments in results.items():
        for period, stats in segments.items():
            fields = [
                "n/a" if value is None else format(value, fmt) + ("x" if key == "turnover" else "")
                for (key, value), fmt in zip(stats.items(), formats, strict=True)
            ]
            cagr, vol, sharpe, drawdown, total, turnover = fields
            print(f"{name:<24} {period.upper():<6} {cagr:>9} {vol:>9} {sharpe:>8}"
                  f" {drawdown:>9} {total:>10} {turnover:>10}")


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    tickers = config["tickers"]
    is_start = pd.Timestamp(config["in_sample"]["start"])
    is_end = pd.Timestamp(config["in_sample"]["end"])
    oos_start = pd.Timestamp(config["out_of_sample"]["start"])
    oos_end = pd.Timestamp(config["out_of_sample"]["end"])
    data_start = pd.Timestamp(config["data_start"])
    fee_rate = float(config["transaction_cost_bps"]) / 10_000
    if not (data_start < is_start <= is_end == oos_start - pd.Timedelta(days=1) <= oos_end):
        raise ValueError("Expected warmup data followed by contiguous IS and OOS date ranges.")
    if len(set(tickers)) != len(tickers) or not 1 <= config["top_n"] <= len(tickers):
        raise ValueError("Tickers must be unique and top_n must fit the universe.")
    if not (0 < config["short_sma_sessions"] < config["long_sma_sessions"]
            and 0 <= config["momentum_skip_months"] < config["momentum_lookback_months"]
            and 0 <= fee_rate < 1 and config["sessions_per_year"] > 0):
        raise ValueError("Invalid signal windows, transaction costs, or sessions_per_year.")

    bars = load_bars(tickers, data_start, oos_end)
    targets = monthly_targets(bars["Close"], config)
    # If IS starts mid-month, enter at that month's first open before measuring IS.
    simulation = bars.loc[is_start.replace(day=1):oos_end]
    sessions = simulation.loc[is_start:].index
    periods = {"is": sessions[sessions <= is_end], "oos": sessions[sessions >= oos_start],
               "full": sessions}
    if not len(periods["is"]) or not len(periods["oos"]):
        raise ValueError("Both IS and OOS must contain trading sessions.")

    results = {}
    for name, weights in targets.items():
        first_session = simulation.index[0]
        if first_session not in weights.index or weights.loc[first_session].isna().any():
            raise ValueError(f"Insufficient warmup history for {name} on {first_session.date()}.")
        daily_results = simulate(simulation["Open"], simulation["Close"], weights, fee_rate)
        results[name] = {
            period: metrics(daily_results.loc[dates], config["sessions_per_year"])
            for period, dates in periods.items()
        }

    show_metrics(results, periods)
    payload = {
        "strategy": config["strategy"],
        "data_range": f"{sessions[0].date()} to {sessions[-1].date()}",
        **results["MAD rotation"],
        "benchmark": {
            "name": "equal_weight_countries",
            **results["Equal-weight countries"],
            "momentum_12_1": results["12–1 momentum"],
        },
    }
    print("RESULTS_JSON: " + json.dumps(payload, allow_nan=False))


if __name__ == "__main__":
    main()
