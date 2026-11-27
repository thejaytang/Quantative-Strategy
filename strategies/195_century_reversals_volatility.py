"""195_century_reversals_volatility
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/195_century_reversals_volatility.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 10f82382008748cd9ff2f56e2b8ef160caa66e38 on 2026-11-27 11:24 Europe/Oslo, exit 0, output sha256 530a6c44d469bc5fac7660ff093a608f5c5f3dd8209c521f6cf8702912508ddd
Data source: 免费 Yahoo Finance 日频行情，经 yfinance 获取 SPY、BIL 的开盘价和收盘价，以及 ^VIX 的收盘值。优先读取本地行情缓存。见 config.yaml:2-5、strategy.py:18-38、README.md:17。
Data fetched: 2026-11-27T10:24:17+00:00
Sample period: 行情请求区间为 2006-10-01 至 2025-12-31（含截止日），回测开始前的数据用于预热。日志记录的实际执行区间为 2008-01-02 至 2025-12-31，共 4,529 个交易日。见 config.yaml:6-12、strategy.py:33-36、strategy.py:230-232、logs/run_20261125_123225.log:1、logs/run_20261125_123225.log:87。   data_range (from output): 2008-01-02 to 2025-12-31
IS/OOS split: 按执行日期划分，样本内为 2008-01-02 至 2017-12-31（2,518 个交易日），样本外为 2018-01-01 至 2025-12-31（2,011 个交易日）。样本外为历史留出期，承接样本内期末仓位。README 记载信号采用冻结候选 C41，论文的历史研究窗口覆盖部分留出期。见 config.yaml:7-12、strategy.py:230-248、README.md:18-19、logs/run_20261125_123225.log:87。
Costs and slippage: 默认每笔单边成交收取 5 bps，敏感性情景为 0 和 15 bps。设 r 为 bps/10000，买入资金乘以 1/(1+r)，卖出所得乘以 1−r。常规信号日两买两卖，全期首日从现金建仓时为信号日三笔、非信号日一笔，基准仅收首笔买入成本。固定成本合并近似点差、佣金与滑点，滑点未单独建模。见 config.yaml:18-20、strategy.py:108-128、strategy.py:224-228、README.md:19、README.md:24。
Signal lag: 交易日 t 收盘后，以 SPY 复权收盘到收盘收益严格低于 −0.75% 且 t−1 的 VIX 严格高于截至 t−1 的 252 个 SPY 交易日滚动第 75 百分位为主策略信号。t+1 指下一交易日，在该日开盘切换至 SPY，当日收盘切回 BIL。隔夜与非信号时段持有 BIL，全期首日从现金建仓。见 config.yaml:13-16、strategy.py:78-100、strategy.py:119-127、README.md:11-13、README.md:19。
Price adjustment: Yahoo auto_adjust=True (ETF split- and distribution-adjusted open/close)
Survivorship bias: 使用固定的 SPY、BIL 实际 ETF 历史，避免用当前成分股回建历史。事后选用存续 ETF 仍有产品选择偏差。见 config.yaml:2-5、strategy.py:18-20、strategy.py:33-38、README.md:24。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.009571186078770832, ann_vol=0.09041090148694818, sharpe=0.15072657998820965, sharpe_excess=0.12586433073569514, max_drawdown=-0.19422415265261483, total_return=0.09985804779958318, turnover=54.40022752999471, n_trades=549
    oos: cagr=0.030857041760043113, ann_vol=0.07357007911915298, sharpe=0.44917232391986606, sharpe_excess=0.11904313804199465, max_drawdown=-0.16058385450047907, total_return=0.2744583926136921, turnover=58.80969201483848, n_trades=474
    full: cagr=0.018967890521761177, ann_vol=0.08334758366804589, sharpe=0.26694943743596733, sharpe_excess=0.12254206346809463, max_drawdown=-0.21557554098352194, total_return=0.40172331970187747, turnover=56.35815048848904, n_trades=1023
    benchmark  full: cagr=0.10920347103687522, ann_vol=0.19775947243912897, sharpe=0.6231252953654971, sharpe_excess=0.5606937373469207, max_drawdown=-0.5189881093699988, total_return=5.440975502971304, turnover=0.055079499777997476, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 195_century_reversals_volatility\nuniverse:\n  market: SPY\n  reserve: BIL\n  volatility: '^VIX'\ndata_start: '2006-10-01'  # Includes the VIX warm-up before the first test session.\noriginal_snapshot:\n  sha256: 04c232667c448bccf8a2b48ebbdce7daca9d89d12b39f7bf0c4e60f886497669\n  fetched_at: '2026-11-25T12:31:39+01:00'  # Timestamp preserved on the original Yahoo CSV.\nin_sample:\n  start: '2008-01-02'\n  end: '2017-12-31'\nout_of_sample:\n  start: '2018-01-01'\n  end: '2025-12-31'\nsignal:\n  drop_threshold: -0.0075\n  vix_window_sessions: 252\n  vix_quantile: 0.75\nevent_drop_band_edges: [0.01, 0.02, 0.03]  # Absolute loss sizes, for diagnostics only.\ncosts:\n  commission_bps: 1\n  slippage_bps: 4\n  sensitivity_bps: [0, 15]\ncash_buffer: 0.01\nannualization_sessions: 252\n"
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
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
FETCH_HEADER = "# DATA_FETCHED_AT: "
PRICE_ADJUSTMENT = "Yahoo auto_adjust=True (ETF split- and distribution-adjusted open/close)"


class NyseHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def nyse_sessions(start, end):
    if start.year < 2006 or end.year > 2025:
        raise ValueError("The historical NYSE schedule covers 2006 through 2025.")
    holidays = NyseHolidays().holidays(start, end)
    # Ford, Hurricane Sandy, George H. W. Bush, and Carter full-day closures.
    special = pd.to_datetime(["2007-01-02", "2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])
    return pd.bdate_range(start, end).difference(holidays.union(special))


def read_price_cache(cache_file, original_snapshot):
    with cache_file.open(encoding="utf-8") as source:
        first_line = source.readline()
        if first_line.startswith(FETCH_HEADER):
            fetched_at = datetime.fromisoformat(first_line.removeprefix(FETCH_HEADER).strip())
            if fetched_at.utcoffset() is None:
                raise ValueError("Cached download timestamp needs a timezone.")
        else:
            if hashlib.sha256(cache_file.read_bytes()).hexdigest() != original_snapshot["sha256"]:
                raise ValueError("Price cache has no matching download provenance.")
            fetched_at = datetime.fromisoformat(original_snapshot["fetched_at"])
            source.seek(0)
        prices = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True)
    return prices, fetched_at


def download_prices(market, reserve, volatility, start, end, first_test, expected, original_snapshot):
    columns = [(ticker, field) for ticker in (market, reserve) for field in ("Open", "Close")]
    columns.append((volatility, "Close"))
    cache_key = hashlib.sha256(repr((columns, start, end)).encode()).hexdigest()[:16]
    cache_file = CACHE / f"yahoo_adjusted_{cache_key}.csv"
    cache_error = ""
    if cache_file.exists():
        try:
            prices, fetched_at = read_price_cache(cache_file, original_snapshot)
            return align_prices(prices, columns, expected, first_test), fetched_at
        except (OSError, ValueError) as error:
            cache_error = f" Cache rejected: {error}"

    CACHE.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(CACHE / "yfinance"))
    last_error = ""
    for _ in range(2):
        try:
            raw = yf.download(
                [market, reserve, volatility], start=start, end=end + pd.Timedelta(days=1),
                auto_adjust=True, group_by="ticker", multi_level_index=True,
                interval="1d", threads=False, progress=False, timeout=12,
            )
            prices = raw.reindex(columns=pd.MultiIndex.from_tuples(columns)).sort_index()
            prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
            aligned = align_prices(prices, columns, expected, first_test)
            fetched_at = datetime.now(timezone.utc).replace(microsecond=0)
            break
        except (OSError, ValueError) as error:
            last_error = str(error)
    else:
        raise ValueError(f"Yahoo Finance download failed after two attempts. {last_error}{cache_error}")

    temp = NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, suffix=".csv", delete=False)
    temp_path = Path(temp.name)
    try:
        with temp:
            temp.write(FETCH_HEADER + fetched_at.isoformat() + "\n")
            prices.to_csv(temp, index_label="Date")
        temp_path.replace(cache_file)
    finally:
        temp_path.unlink(missing_ok=True)
    return aligned, fetched_at


def align_prices(prices, columns, expected, first_test):
    if (not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans
            or not prices.index.is_unique or not prices.index.is_monotonic_increasing):
        raise ValueError("Price dates must be valid, unique and sorted.")
    prices = prices.reindex(index=expected, columns=pd.MultiIndex.from_tuples(columns))
    prices = prices.apply(pd.to_numeric)
    test_dates = expected[expected >= first_test]
    for ticker, field in columns[:-1]:
        required = expected if ticker == columns[0][0] else test_dates
        quote = prices.loc[required, (ticker, field)]
        missing = quote.index[~(np.isfinite(quote) & quote.gt(0))]
        if len(missing):
            raise ValueError(f"Missing or invalid {ticker} {field} on NYSE session {missing[0].date()}.")
    vix = prices[columns[-1]].dropna()
    if vix.empty or not (np.isfinite(vix) & vix.gt(0)).all():
        raise ValueError(f"Missing or invalid {columns[-1][0]} close history.")
    return prices


def make_sessions(prices, market, reserve, volatility, signal, start, end):
    spy = prices[market]
    bil = prices[reserve]
    vix = prices[(volatility, "Close")]
    cutoff = vix.rolling(signal["vix_window_sessions"]).quantile(signal["vix_quantile"])
    lagged_vix = vix.shift(1)
    cutoff = cutoff.shift(1)
    known = lagged_vix.notna() & cutoff.notna()
    high = lagged_vix.gt(cutoff)
    spy_daily = spy["Close"].div(spy["Close"].shift()).sub(1)
    drop = spy_daily.lt(signal["drop_threshold"]) & known

    sessions = pd.DataFrame({
        "high_vix": (drop & high).shift(fill_value=False),
        "lower_vix": (drop & ~high).shift(fill_value=False),
        "drop_only": drop.shift(fill_value=False),
        "vix_ready": known.shift(fill_value=False),
        "signal_return": spy_daily.shift(),
        "spy_overnight": spy["Open"].div(spy["Close"].shift()),
        "spy_intraday": spy["Close"].div(spy["Open"]),
        "spy_open": spy["Open"],
        "spy_close": spy["Close"],
        "bil_open": bil["Open"],
        "bil_close": bil["Close"],
        "cash_proxy_return": bil["Close"].div(bil["Close"].shift()).sub(1),
    }).loc[start:end].copy()
    if sessions.empty:
        raise ValueError("The backtest needs at least one NYSE session.")
    if not np.isfinite(sessions["signal_return"].iloc[0]):
        raise ValueError("The first execution session needs a prior SPY close-to-close return.")
    first = sessions.iloc[0]
    sessions.loc[sessions.index[0], "cash_proxy_return"] = first.bil_close / first.bil_open - 1
    return sessions


def buy_at_open(cash, price, commission, slippage, cash_buffer):
    spendable = cash * (1 - cash_buffer)
    requested_units = spendable / price
    fill_price = price * (1 + slippage)
    per_unit_cost = fill_price * (1 + commission)
    units = min(requested_units, spendable / per_unit_cost)
    return units, units * fill_price, cash - units * per_unit_cost


def sell_units(units, price, commission, slippage):
    traded = units * price * (1 - slippage)
    return traded, traded * (1 - commission)


def simulate(sessions, model, commission, slippage, cash_buffer):
    buy_and_hold = {"benchmark": "spy", "cash_benchmark": "bil"}.get(model)
    positions = {"spy": 0.0, "bil": 0.0}
    cash = previous_value = 1.0
    daily_results = []
    for day in sessions.itertuples(index=False):
        target = buy_and_hold or ("spy" if getattr(day, model) else "bil")
        trades = 0
        traded = 0.0

        if target == "spy" and positions["bil"] > 0:
            sale, proceeds = sell_units(positions["bil"], day.bil_open, commission, slippage)
            cash += proceeds
            traded += sale
            trades += 1
            positions["bil"] = 0.0
        if positions[target] == 0 and cash > 0:
            open_price = day.spy_open if target == "spy" else day.bil_open
            units, purchase, cash = buy_at_open(cash, open_price, commission, slippage, cash_buffer)
            positions[target] = units
            traded += purchase
            trades += int(units > 0)

        if buy_and_hold is None and positions["spy"] > 0:
            # Sell the units fixed at the open. The proceeds stay in cash through this close.
            sale, proceeds = sell_units(positions["spy"], day.spy_close, commission, slippage)
            cash += proceeds
            traded += sale
            trades += 1
            positions["spy"] = 0.0

        value = cash + positions["spy"] * day.spy_close + positions["bil"] * day.bil_close
        if cash < 0 or not math.isfinite(value) or value <= 0:
            raise ValueError("Simulation produced negative cash or an invalid portfolio value.")
        daily_results.append((value / previous_value - 1, traded / previous_value, trades))
        previous_value = value
    return pd.DataFrame(daily_results, index=sessions.index, columns=["return", "turnover", "trades"])


def metrics(daily, cash_proxy_returns, year):
    returns = daily["return"]
    growth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    vol = returns.std(ddof=1)
    excess = returns - cash_proxy_returns
    excess_vol = excess.std(ddof=1)
    return {
        "cagr": growth[-1] ** (year / len(returns)) - 1,
        "ann_vol": vol * math.sqrt(year),
        "sharpe": returns.mean() / vol * math.sqrt(year) if vol > 0 else None,
        "sharpe_excess": excess.mean() / excess_vol * math.sqrt(year) if excess_vol > 0 else None,
        "max_drawdown": (growth / np.maximum.accumulate(growth) - 1).min(),
        "total_return": growth[-1] - 1,
        "turnover": daily["turnover"].mean() * year,
        "n_trades": int(daily["trades"].sum()),
        "n_sessions": len(returns),
    }


def event_metrics(sessions, event_column):
    events = sessions.loc[sessions[event_column]]
    return {
        "n_events": len(events),
        "mean_signal_return": events["signal_return"].mean(),
        "mean_overnight_return": events["spy_overnight"].mean() - 1,
        "mean_intraday_return": events["spy_intraday"].mean() - 1,
        "mean_close_to_close_return": (events["spy_overnight"] * events["spy_intraday"]).mean() - 1,
    }


def summarize_events(sessions, windows, names, lower_edge, band_edges):
    diagnostic_sets = {"All": sessions}
    magnitude = -sessions["signal_return"]
    for left, right in zip([lower_edge, *band_edges], [*band_edges, math.inf]):
        label = f"{left:.2%}+" if math.isinf(right) else f"{left:.2%}–<{right:.2%}"
        diagnostic_sets[label] = sessions.loc[magnitude.ge(left) & magnitude.lt(right)]

    results = {}
    rows = []
    for label, subset in diagnostic_sets.items():
        results[label] = {}
        for model in ("high_vix", "lower_vix", "drop_only"):
            results[label][model] = {}
            for period, window in windows.items():
                stats = event_metrics(subset.loc[window], model)
                results[label][model][period] = stats
                rows.append({
                    "Drop size": label, "Model": names[model], "Period": period.upper(),
                    "Events": stats["n_events"], "Signal drop": stats["mean_signal_return"],
                    "Overnight": stats["mean_overnight_return"], "Intraday": stats["mean_intraday_return"],
                    "Close-close": stats["mean_close_to_close_return"],
                })
    events = results.pop("All")
    return events, results, rows


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def print_tables(portfolio_rows, event_rows):
    def percent(value):
        return f"{value:.2%}" if pd.notna(value) else "-"

    def ratio(value):
        return f"{value:.2f}" if pd.notna(value) else "-"

    results_table = pd.DataFrame(portfolio_rows)
    print(results_table.to_string(index=False, na_rep="-", formatters={
        "CAGR": percent, "Ann. vol": percent, "Sharpe": ratio, "Excess Sharpe": ratio,
        "Max DD": percent, "Total return": percent, "Turnover": percent,
    }))
    print("\nSPY event windows (gross mean returns, assigned by next-session execution date)")
    print(pd.DataFrame(event_rows).to_string(index=False, na_rep="-", formatters={
        "Signal drop": percent, "Overnight": percent, "Intraday": percent, "Close-close": percent,
    }))


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    market, reserve, volatility = (universe[key] for key in ("market", "reserve", "volatility"))
    start_is, end_is = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    start_oos, end_oos = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    data_start = pd.Timestamp(config["data_start"])
    if not data_start < start_is <= end_is < start_oos <= end_oos:
        raise ValueError("Expected warm-up, in-sample and out-of-sample dates in chronological order.")
    if start_oos != end_is + pd.Timedelta(days=1):
        raise ValueError("The test periods must be contiguous in calendar time.")

    signal = config["signal"]
    lower_edge = -signal["drop_threshold"]
    band_edges = config["event_drop_band_edges"]
    if not all(left < right for left, right in zip([lower_edge, *band_edges], band_edges)):
        raise ValueError("Diagnostic drop-band edges must increase above the absolute drop threshold.")
    costs = config["costs"]
    commission_bps = costs["commission_bps"]
    slippage_bps = costs["slippage_bps"]
    base_bps = commission_bps + slippage_bps
    scenarios = [base_bps, *dict.fromkeys(bps for bps in costs["sensitivity_bps"] if bps != base_bps)]
    if min(commission_bps, slippage_bps) < 0 or not all(0 <= bps < 10_000 for bps in scenarios):
        raise ValueError("One-way commission and slippage must be nonnegative and total below 10,000 bps.")
    if base_bps == 0 and len(scenarios) > 1:
        raise ValueError("A positive base cost is needed to split sensitivity costs.")
    cash_buffer = config["cash_buffer"]
    if not 0 < cash_buffer < 1:
        raise ValueError("The cash buffer must be between zero and one.")

    expected = nyse_sessions(data_start, end_oos)
    prices, fetched_at = download_prices(
        market, reserve, volatility, data_start, end_oos, start_is, expected, config["original_snapshot"],
    )
    sessions = make_sessions(prices, market, reserve, volatility, signal, start_is, end_oos)
    windows = {"is": slice(start_is, end_is), "oos": slice(start_oos, end_oos), "full": slice(None)}
    if any(sessions.loc[window].empty for window in windows.values()):
        raise ValueError("Each test period needs at least one ETF session.")

    names = {
        "high_vix": "High VIX reversal", "lower_vix": "Lower VIX reversal",
        "drop_only": "Drop-only reversal", "benchmark": f"{market} buy-and-hold",
        "cash_benchmark": f"{reserve} buy-and-hold",
    }
    commission_share = commission_bps / base_bps if base_bps else 0.0
    results = {}
    portfolio_rows = []
    for bps in scenarios:
        results[bps] = {}
        commission = bps * commission_share / 10_000
        slippage = bps * (1 - commission_share) / 10_000
        for model, name in names.items():
            daily = simulate(sessions, model, commission, slippage, cash_buffer)
            model_results = {}
            for period, window in windows.items():
                stats = metrics(daily.loc[window], sessions.loc[window, "cash_proxy_return"],
                                config["annualization_sessions"])
                model_results[period] = stats
                portfolio_rows.append({
                    "Model": name, "Period": period.upper(), "Cost bps/leg": bps,
                    "CAGR": stats["cagr"], "Ann. vol": stats["ann_vol"], "Sharpe": stats["sharpe"],
                    "Excess Sharpe": stats["sharpe_excess"], "Turnover": stats["turnover"],
                    "Max DD": stats["max_drawdown"], "Total return": stats["total_return"],
                    "Trades": stats["n_trades"],
                })
            results[bps][model] = model_results

    events, event_size_bands, event_rows = summarize_events(sessions, windows, names, lower_edge, band_edges)
    data_range = f"{sessions.index[0].date()} to {sessions.index[-1].date()}"
    ready_sessions = int(sessions["vix_ready"].sum())
    print("PARAM_COMBOS_TRIED: 1")
    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"{config['strategy']} | {data_range} | {ready_sessions}/{len(sessions)} sessions with VIX state")
    print_tables(portfolio_rows, event_rows)
    primary = results[base_bps]
    payload = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **primary["high_vix"],
        "benchmark": {"name": names["benchmark"], **primary["benchmark"]},
        "cash_benchmark": {"name": names["cash_benchmark"], **primary["cash_benchmark"]},
        "cost_bps_per_leg": base_bps,
        "comparisons": {model: primary[model] for model in ("lower_vix", "drop_only")},
        "events": events,
        "event_size_bands": event_size_bands,
        "cost_sensitivity": {str(bps): results[bps] for bps in scenarios if bps != base_bps},
    }
    print("RESULTS_JSON: " + json.dumps(json_safe(payload), allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        sys.exit(f"Backtest failed: {exc}")