"""460_understanding_the_long_only_minimum_vari
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/460_understanding_the_long_only_minimum_vari.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c71364ad99535bbf59be998b004cff431ee83048 on 2026-11-27 11:54 Europe/Oslo, exit 0, output sha256 33c62470601cd27a0cf43b6200b179689d9198389c9679fc58f4da59d65bff2d
Data source: Yahoo Finance 经 yfinance 提供日频调整收盘价，覆盖行业 ETF XLB、XLE、XLF、XLI、XLK、XLP、XLU、XLV、XLY，以及市场因子 SPY 和现金代理 BIL。程序优先读取按标的和请求日期区间标识的本地数据缓存。（config.yaml:2-4，strategy.py:35-62，strategy.py:199-203）
Data fetched: 2026-11-27T10:54:23.732229+00:00
Sample period: 请求数据区间为 2008-01-01 至 2025-12-31（含末日），2008 年用于预热。配置的回测起点为 2009-01-01，日志记录实际回测交易日范围为 2009-01-02 至 2025-12-31。（config.yaml:5-8，strategy.py:54-57，strategy.py:199-202，README.md:18，logs/run_20261127_080341.log:4）   data_range (from output): 2009-01-02 to 2025-12-31
IS/OOS split: 样本内为回测起点至 2016-12-31（含切分日，按实际交易日），样本外为 2017-01-01 至 2025-12-31（按实际交易日）。程序对同一条连续持仓路径分段统计，切分处持仓延续。（config.yaml:6-8，strategy.py:171-175，strategy.py:202-209，README.md:18）
Costs and slippage: 每笔模拟买入或卖出均按成交名义金额计 5 bps 交易成本和 5 bps 固定滑点，共 10 bps。二者合并为费用率，在执行收盘按各 ETF 买卖名义金额绝对值之和扣减净值，目标名义金额按扣费后的净值计算。策略与等权基准采用相同费率。（config.yaml:11-12，strategy.py:124-145，strategy.py:195-207，README.md:25）
Signal lag: 每月首个交易日 t 收盘后，用截至该收盘的 126 个交易日的日收益计算目标权重。配置规定在下一交易日收盘执行。执行日的收盘到收盘收益按旧持仓计入，新权重从执行收盘后的收益区间起生效。（config.yaml:9-10，strategy.py:106-118，strategy.py:124-147，README.md:23）
Price adjustment: yfinance auto_adjust total-return close
Survivorship bias: 固定资产池由九只现仍存续的早期行业 ETF 构成。README 明确记录存在幸存者偏差，ETF 历史价格反映成分与指数规则随时间变化的结果。（config.yaml:2，strategy.py:181-186，strategy.py:199-205，README.md:29）
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.11952065161262149, ann_vol=0.1188344052883519, sharpe=1.0097347093564235, sharpe_excess=1.0073812596431344, max_drawdown=-0.19171674768757774, total_return=1.4652883934201397, turnover=1.8852807582504005, n_trades=303
    oos: cagr=0.08416030641226646, ann_vol=0.13877636649785036, sharpe=0.6519325950898799, sharpe_excess=0.4907296247347426, max_drawdown=-0.2787231321800252, total_return=1.0654085233910422, turnover=1.8528002514005895, n_trades=529
    full: cagr=0.10067372562894272, ann_vol=0.1297544455899462, sharpe=0.8044119346668672, sharpe_excess=0.7125285189152458, max_drawdown=-0.27872313218002454, total_return=4.091827660387021, turnover=1.8680986005108609, n_trades=832
    benchmark  full: cagr=0.12677910695128913, ann_vol=0.16460883506747565, sharpe=0.8078345783258826, sharpe_excess=0.7354399475196651, max_drawdown=-0.349994109843658, total_return=6.579030924953063, turnover=0.33115119324016795, n_trades=1836
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 460_understanding_the_long_only_minimum_vari\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nmarket_factor: SPY\ncash_proxy: BIL\ndata_start: 2008-01-01\nbacktest_start: 2009-01-01\nis_end: 2016-12-31\ndata_end: 2025-12-31\nlookback_sessions: 126\nsignal_lag_sessions: 1\ncost_bps: 5\nslippage_bps: 5\ncash_buffer_fraction: 0.05\n'
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
"""One-factor long-only minimum variance research backtest."""

import json
from datetime import datetime, timezone
from hashlib import sha256
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
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SECTOR_ETFS = {"XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLC", "XLRE"}
SESSIONS_PER_YEAR = 252
FETCH_HEADER = "# DATA_FETCHED_AT: "


class NyseHolidayCalendar(AbstractHolidayCalendar):
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


# Unscheduled full-day closures in the requested sample.
NYSE_SPECIAL_CLOSURES = pd.DatetimeIndex(["2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])


def check_prices(prices, tickers, start, end):
    if prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Daily closes must have a nonempty, increasing, unique date index.")
    holidays = NyseHolidayCalendar().holidays(start, end).union(NYSE_SPECIAL_CLOSURES)
    expected = pd.bdate_range(start, end).difference(holidays)
    if not prices.index.equals(expected):
        missing = expected.difference(prices.index)
        unexpected = prices.index.difference(expected)
        raise ValueError(
            f"Daily closes disagree with the NYSE calendar. "
            f"Missing {len(missing)} dates {missing[:10].strftime('%Y-%m-%d').tolist()}. "
            f"Unexpected {len(unexpected)} dates {unexpected[:10].strftime('%Y-%m-%d').tolist()}."
        )
    for ticker in tickers:
        values = prices[ticker].to_numpy()
        if not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError(f"Missing or invalid adjusted close in {ticker}.")


def load_prices(tickers, start, end):
    tickers = sorted(tickers)
    key = sha256(f"{tickers}|{start.date()}|{end.date()}".encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"yfinance_adjusted_close_{key}.csv"
    if cache_file.exists():
        with cache_file.open(encoding="utf-8") as handle:
            stamp = handle.readline().strip()
            if not stamp.startswith(FETCH_HEADER):
                raise ValueError(f"Missing fetch timestamp in {cache_file.name}.")
            fetched_at = datetime.fromisoformat(stamp.removeprefix(FETCH_HEADER))
            prices = pd.read_csv(
                handle, index_col="Date", parse_dates=["Date"], converters=dict.fromkeys(tickers, float),
            )
        check_prices(prices, tickers, start, end)
        return prices, fetched_at

    CACHE_DIR.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=CACHE_DIR, prefix="yfinance_") as library_cache:
        yf.set_tz_cache_location(library_cache)
        raw = yf.download(
            tickers, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=True, progress=False, timeout=15,
        )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no daily prices.")
    fetched_at = datetime.now(timezone.utc)
    prices = raw["Close"].reindex(columns=tickers)
    prices.index = prices.index.tz_localize(None)
    check_prices(prices, tickers, start, end)

    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE_DIR, delete_on_close=False) as handle:
        handle.write(FETCH_HEADER + fetched_at.isoformat() + "\n")
        prices.to_csv(handle, index_label="Date")
        handle.flush()
        Path(handle.name).replace(cache_file)
    return prices, fetched_at


def minimum_variance_weights(asset_excess, market_excess):
    assets = asset_excess - asset_excess.mean(axis=0)
    market = market_excess - market_excess.mean()
    market_variance = (market @ market) / len(market)
    if market_variance <= 0:
        raise ValueError("The market factor requires positive variance.")
    beta = (market @ assets) / (len(market) * market_variance)
    residual = assets - np.outer(market, beta)
    residual_variance = np.mean(residual ** 2, axis=0)
    variance_scale = max(market_variance, np.mean(assets ** 2, axis=0).max())
    precision = 1 / np.maximum(residual_variance, np.finfo(float).eps * variance_scale)

    # Theorem 2 orients the betas so their precision-weighted sum is positive.
    if beta @ precision < 0:
        beta = -beta
    exposure_sum = exposure_squared_sum = 0.0
    active = []
    for i in np.argsort(beta):
        inclusion = 1 + market_variance * (exposure_squared_sum - beta[i] * exposure_sum)
        if inclusion <= 0:
            break
        active.append(i)
        exposure_sum += beta[i] * precision[i]
        exposure_squared_sum += beta[i] ** 2 * precision[i]

    # Woodbury formula for the inverse covariance solution on the active set.
    h = market_variance * exposure_sum / (1 + market_variance * exposure_squared_sum)
    weights = np.zeros_like(beta)
    weights[active] = precision[active] * np.maximum(1 - beta[active] * h, 0)
    return weights / weights.sum()


def rebalance_targets(returns, universe, market, cash, start, lookback, lag):
    dates = returns.index
    months = dates.to_period("M")
    excess = returns[[*universe, market]].sub(returns[cash], axis=0)
    targets = {}
    for signal in np.flatnonzero(months[1:] != months[:-1]) + 1:
        execution = signal + lag
        if signal < lookback or execution >= len(dates) or dates[execution] < start:
            continue
        window = excess.iloc[signal - lookback + 1:signal + 1]
        targets[dates[execution]] = minimum_variance_weights(
            window[universe].to_numpy(), window[market].to_numpy(),
        )
    if not targets:
        raise ValueError("No execution dates with a complete trailing return window.")
    return targets


def size_positions(positions, cash_value, reference_close, target, fee_rate, cash_buffer):
    marked = positions * reference_close
    reference_value = marked.sum() + cash_value
    current_weights = marked / reference_value
    buffered_target = (1 - cash_buffer) * target

    def funding_gap(value_fraction):
        traded = np.abs(value_fraction * buffered_target - current_weights).sum()
        return value_fraction + fee_rate * traded - 1

    # Reserve the buffer as a fraction of reference value after estimated fees.
    value_after_fees = reference_value * brentq(funding_gap, 0, 1)
    return value_after_fees * buffered_target / reference_close


def simulate(prices, cash_returns, targets, fee_rate, cash_buffer):
    positions = np.zeros(prices.shape[1])
    cash_value = previous_value = 1.0
    reference_closes = prices.shift(1).loc[cash_returns.index].to_numpy()
    execution_closes = prices.loc[cash_returns.index].to_numpy()
    rows = []
    sessions = zip(cash_returns.index, reference_closes, execution_closes, cash_returns)
    for date, reference_close, execution_close, cash_return in sessions:
        if date in targets:
            next_positions = size_positions(
                positions, cash_value, reference_close, targets[date], fee_rate, cash_buffer,
            )
            share_changes = next_positions - positions

        cash_value *= 1 + cash_return
        value_before_fill = positions @ execution_close + cash_value
        turnover, n_trades = 0.0, 0
        if date in targets:
            traded = share_changes * execution_close
            gross_traded = np.abs(traded).sum()
            cash_after_fill = cash_value - traded.sum() - fee_rate * gross_traded
            if cash_after_fill < 0:
                raise RuntimeError(f"Predetermined shares would leave negative cash on {date.date()}.")
            cash_value = cash_after_fill
            positions = next_positions
            turnover = gross_traded / value_before_fill
            n_trades = np.count_nonzero(share_changes)

        closing_value = positions @ execution_close + cash_value
        rows.append((closing_value / previous_value - 1, turnover, n_trades, cash_value / closing_value))
        previous_value = closing_value
    return pd.DataFrame(rows, index=cash_returns.index, columns=["return", "turnover", "n_trades", "cash_fraction"])


def metrics(path, cash_returns):
    returns = path["return"]
    growth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    years = len(returns) / SESSIONS_PER_YEAR
    scale = np.sqrt(SESSIONS_PER_YEAR)
    vol = returns.std(ddof=1)
    excess = returns - cash_returns.loc[returns.index]
    excess_vol = excess.std(ddof=1)
    return {
        "cagr": float(growth[-1] ** (1 / years) - 1),
        "ann_vol": float(vol * scale),
        "sharpe": float(returns.mean() / vol * scale) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * scale) if excess_vol > 0 else None,
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1)),
        "total_return": float(growth[-1] - 1),
        "turnover": float(path["turnover"].sum() / years),
        "n_trades": int(path["n_trades"].sum()),
    }


def period_metrics(path, cash_returns, split):
    periods = {"is": path.loc[:split], "oos": path.loc[path.index > split], "full": path}
    if any(len(period) < 2 for period in periods.values()):
        raise ValueError("IS and OOS each require at least two sessions.")
    return {name: metrics(period, cash_returns) for name, period in periods.items()}


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    universe = config["universe"]
    market, cash = config["market_factor"], config["cash_proxy"]
    if len(set(universe)) != len(universe) or len(universe) < 2 or not set(universe) <= SECTOR_ETFS:
        raise ValueError("Choose at least two distinct sector ETFs from the supported list.")
    if (market, cash) != ("SPY", "BIL"):
        raise ValueError("This specification uses SPY for market returns and BIL for cash returns.")
    data_start, start, split, end = (
        pd.Timestamp(config[key]) for key in ("data_start", "backtest_start", "is_end", "data_end")
    )
    if not data_start < start <= split < end:
        raise ValueError("Dates must order the warm-up, IS and OOS periods.")
    lookback, lag = config["lookback_sessions"], config["signal_lag_sessions"]
    if type(lookback) is not int or lookback < 3 or type(lag) is not int or lag < 1:
        raise ValueError("Lookback must be at least three sessions and execution lag at least one.")
    cost, slippage = config["cost_bps"], config["slippage_bps"]
    if not (cost >= 0 and slippage >= 0 and cost + slippage < 10_000):
        raise ValueError("Cost and slippage must be nonnegative and sum to less than 10,000 bps.")
    cash_buffer = config["cash_buffer_fraction"]
    if not 0 < cash_buffer < 1:
        raise ValueError("The cash buffer fraction must be between zero and one.")

    prices, fetched_at = load_prices([*universe, market, cash], data_start, end)
    if start <= prices.index[0]:
        raise ValueError("At least one closing price must precede the backtest start.")
    returns = prices.pct_change(fill_method=None)
    targets = rebalance_targets(returns, universe, market, cash, start, lookback, lag)
    cash_returns = returns.loc[start:, cash]
    fee_rate = (cost + slippage) / 10_000
    sector_prices = prices[universe]
    path = simulate(sector_prices, cash_returns, targets, fee_rate, cash_buffer)
    equal_weight = np.full(len(universe), 1 / len(universe))
    benchmark_targets = dict.fromkeys(targets, equal_weight)
    benchmark = simulate(sector_prices, cash_returns, benchmark_targets, fee_rate, cash_buffer)
    strategy_results = period_metrics(path, cash_returns, split)
    benchmark_results = period_metrics(benchmark, cash_returns, split)
    result = {
        "strategy": config["strategy"],
        "data_range": f"{path.index[0].date()} to {path.index[-1].date()}",
        **strategy_results,
        "benchmark": {"name": "Monthly equal-weight sector ETFs", **benchmark_results},
    }
    print("PARAM_COMBOS_TRIED: 1")
    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return close")
    print(f"PERFORMANCE_RANGE: {result['data_range']} | excess returns vs {cash}")
    print(f"MIN_CASH_FRACTION: strategy={path['cash_fraction'].min():.8f} benchmark={benchmark['cash_fraction'].min():.8f}")
    for name in ("is", "oos", "full"):
        print(f"{name.upper()}: {json.dumps(strategy_results[name], allow_nan=False)}")
        print(f"BENCHMARK {name.upper()}: {json.dumps(benchmark_results[name], allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
