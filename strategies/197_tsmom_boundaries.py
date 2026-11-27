"""197_tsmom_boundaries
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/197_tsmom_boundaries.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit da9754ee51e8d74f175ab50e14ef28c47c764ad1 on 2026-11-27 12:03 Europe/Oslo, exit 0, output sha256 61a52fec22095d3c064c2a5b1cf57be874a4cecc57e6e9b0b9afd37797f35d73
Data source: Yahoo Finance（yfinance）的 SPY、BIL 日频开盘价、收盘价、复权收盘价及除息日现金分红，另用免费 FRED CSV 的 DGS10（10 年期）和 DGS3MO（3 个月期）美国国债收益率。config.yaml:3-7，strategy.py:18、56-70、79-95，README.md:22。
Data fetched: 2026-11-27
Sample period: 实际回测交易区间为 2009-01-02 至 2026-10-30。数据请求自 2002-01-01 起用于预热，截止 2026-10-30。config.yaml:9-15，strategy.py:214-217、255-256，README.md:20，logs/codex_fix_d2_run_20261126_181450.log:4。   data_range (from output): 2009-01-02 to 2026-10-30
IS/OOS split: 样本内为 2009-01-01 至 2018-12-31，样本外为 2019-01-01 至 2026-10-30。先连续模拟，再按日期切分报告，样本边界延续持仓。README 记载两段均未用于参数搜索，并说明这属于历史时间切分，研究论文的美国样本与样本外区间部分重叠。config.yaml:10-15，strategy.py:221-227、236-253，README.md:20、34，logs/codex_fix_d2_run_20261126_181450.log:5-6。
Costs and slippage: 每边 5 bps，由 1 bp 手续费与 4 bps 固定滑点组成，滑点已建模为固定费用并包含价差影响。费率 c = 0.0005。首次从现金建仓乘以 1/(1+c)，换仓日卖出和买入各计一边，乘以 (1−c)/(1+c)，其余日成本因子为 1。策略及两个基准使用同一费率。config.yaml:24-25，strategy.py:144-156、203-208、236-245，README.md:16，logs/codex_fix_d2_run_20261126_181450.log:4。
Signal lag: 每月最后一个 SPY 交易日收盘 t 形成信号。月末目标前向填充到日频后再滞后一整个交易日，最早在下一 SPY 交易日 t+1 开盘按目标换仓。换仓日的隔夜收益计入原持仓，开盘至收盘收益计入新持仓。strategy.py:112-139、144-156、159-165、221-226，README.md:16。
Price adjustment: Yahoo split-adjusted Open for sizing and fills, Close for marking and valuation; dividends tracked separately; dividend- and split-adjusted Adj Close for momentum and BIL cash reference
Survivorship bias: 使用固定的 SPY 和 BIL 历史，不按当前标普 500 成分股回填。两只 ETF 由研究者事后选定，仍存在产品层面的选择和幸存者偏差。config.yaml:3-5，strategy.py:211-215，README.md:36。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.11956947485326985, ann_vol=0.15101010346081545, sharpe=0.8236254808795336, sharpe_excess=0.8074576182506913, max_drawdown=-0.2686896337586938, total_return=2.08838831597446, turnover=24.593058794896642, n_trades=25
    oos: cagr=0.14006656636278558, ann_vol=0.14937651167973853, sharpe=0.9526645288262168, sharpe_excess=0.775690910011263, max_drawdown=-0.2799076966262053, total_return=1.7849906524142432, turnover=35.44200508608044, n_trades=36
    full: cagr=0.12852231171139783, ann_vol=0.15027944220383807, sharpe=0.8800095903435303, sharpe_excess=0.7936798817358752, max_drawdown=-0.2799076966262052, total_return=7.6011325910141885, turnover=60.03506388097708, n_trades=61
    benchmark: cagr=0.1329219095231713, ann_vol=0.160834446321925, sharpe=0.8566970336283034, sharpe_excess=0.7760256372515544, max_drawdown=-0.29930415890686735, total_return=8.217868738172784, turnover=0.9895052473763118, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 197_tsmom_boundaries\n\nuniverse:\n  equity: SPY\n  cash: BIL\n  treasury_long: DGS10\n  treasury_short: DGS3MO\n\ndata_start: '2002-01-01'\nin_sample:\n  start: '2009-01-01'\n  end: '2018-12-31'\nout_of_sample:\n  start: '2019-01-01'\n  end: '2026-10-30'\n\nsignal:\n  dividend_lookback_months: 12\n  treasury_lag_sessions: 2\n  valuation_lookback_months: 60\n  boundary_threshold: 4.0\n  trend_lookback_sessions: 252\n\ncommission_bps_per_side: 1.0\nslippage_bps_per_side: 4.0\ncash_buffer_fraction: 0.01\ndividend_cash_delay_days: 60\nparam_combos_tried: 1\n"
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
"""C46 valuation-conditioned U.S. equity time-series momentum backtest."""

import io
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import requests
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
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SESSIONS_PER_YEAR = 252
PRICE_ADJUSTMENT = (
    "Yahoo split-adjusted Open for sizing and fills, Close for marking and valuation; "
    "dividends tracked separately; dividend- and split-adjusted Adj Close for momentum and BIL cash reference"
)


def cached_history(source, symbol, start, end, download):
    path = CACHE / f"{source}_{symbol}_{start.date()}_{end.date()}.csv"
    cache_hit = path.exists()
    if cache_hit:
        data = pd.read_csv(path, index_col="date", parse_dates=["date"])
        fetched_on = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).date()
    else:
        data = download().sort_index()
        fetched_on = datetime.now(timezone.utc).date()

    if data.index.has_duplicates or not data.index.is_monotonic_increasing or data.index.hasnans:
        raise ValueError(f"Invalid {source} observation dates for {symbol}")
    data = data.loc[start:end]
    last_observation = data.dropna().index.max()
    if pd.isna(last_observation) or last_observation < end - pd.Timedelta(days=7):
        raise ValueError(f"Missing, incomplete or unordered {source} history for {symbol}")

    if not cache_hit:
        with NamedTemporaryFile(dir=CACHE, suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
        try:
            data.to_csv(temporary_path, index_label="date")
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    data.attrs["fetched_on"] = fetched_on
    return data


def fetch_etf(symbol, start, end):
    def download():
        data = yf.Ticker(symbol).history(
            start=str(start.date()),
            end=str((end + pd.Timedelta(days=1)).date()),
            auto_adjust=False,
            actions=True,
            timeout=15,
        )
        if data.empty:
            raise RuntimeError(f"Yahoo Finance returned no history for {symbol}")
        data.index = data.index.tz_localize(None)
        return data[["Open", "Close", "Adj Close", "Dividends"]].astype(float)

    data = cached_history("yahoo", symbol, start, end, download)
    prices = data[["Open", "Close", "Adj Close"]]
    if not np.isfinite(prices).all().all() or (prices <= 0).any().any():
        raise ValueError(f"Invalid daily prices for {symbol}")
    if not np.isfinite(data["Dividends"]).all() or (data["Dividends"] < 0).any():
        raise ValueError(f"Invalid dividend actions for {symbol}")
    return data


def fetch_treasury(symbol, start, end):
    def download():
        response = requests.get(
            FRED_CSV,
            params={"id": symbol, "cosd": str(start.date()), "coed": str(end.date())},
            timeout=20,
        )
        response.raise_for_status()
        data = pd.read_csv(io.StringIO(response.text), na_values=["."])
        dates = pd.to_datetime(data.iloc[:, 0])
        return data[[symbol]].astype(float).set_index(dates)

    data = cached_history("fred", symbol, start, end, download)
    rates = data[symbol].dropna()
    if not np.isfinite(rates).all():
        raise ValueError(f"Invalid Treasury yields for {symbol}")
    return rates, data.attrs["fetched_on"]


def market_sessions(start, end):
    """U.S. equity sessions for the configured history, including full-day exceptional closures."""
    calendar = AbstractHolidayCalendar(rules=[
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ])
    exceptional_closures = pd.to_datetime([
        "2004-06-11", "2007-01-02", "2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09",
    ])
    closed = calendar.holidays(start, end).union(exceptional_closures)
    return pd.bdate_range(start, end).difference(closed)


def available_treasury_spread(sessions, long_rates, short_rates, lag_sessions):
    spread = long_rates.sub(short_rates).dropna().sort_index()
    observations = spread.index
    # Allow through the following joint Treasury observation day for the H.15
    # release. Require a later SPY session even if that day is a market holiday.
    after_release = sessions.searchsorted(observations[1:], side="right")
    after_lag = sessions.searchsorted(observations[:-1]) + lag_sessions
    effective = np.maximum(after_release, after_lag)
    valid = effective < len(sessions)
    released = pd.Series(spread.to_numpy()[:-1][valid], index=sessions[effective[valid]])
    released = released[~released.index.duplicated(keep="last")]
    return released.reindex(sessions, method="ffill", tolerance=pd.Timedelta(days=7))


def month_end_signals(equity, cash, long_rates, short_rates, settings):
    close = equity["Close"]
    sessions = close.index
    dividend_start = sessions - pd.DateOffset(months=settings["dividend_lookback_months"])
    first_in_window = sessions.searchsorted(dividend_start, side="right")
    cumulative_dividends = np.r_[0.0, equity["Dividends"].cumsum().to_numpy()]
    dividends = cumulative_dividends[1:] - cumulative_dividends[first_in_window]
    dividend_yield = close.rdiv(dividends).where(dividend_start >= sessions[0])

    spread = available_treasury_spread(sessions, long_rates, short_rates, settings["treasury_lag_sessions"])
    valuations = pd.DataFrame({"dividend_yield": dividend_yield, "term_spread": spread})

    month_ends = sessions[~sessions.to_period("M").duplicated(keep="last")]
    monthly = valuations.loc[month_ends]
    previous = monthly.shift(1).rolling(settings["valuation_lookback_months"])
    scale = previous.std(ddof=0)
    z_scores = (monthly - previous.mean()) / scale.where(scale > 0)
    score = z_scores.pow(2).sum(axis=1, min_count=2)

    total_return_prices = pd.DataFrame({"equity": equity["Adj Close"], "cash": cash["Adj Close"]})
    past_returns = total_return_prices.pct_change(settings["trend_lookback_sessions"], fill_method=None)
    excess = (past_returns["equity"] - past_returns["cash"]).reindex(month_ends)
    extreme = score > settings["boundary_threshold"]
    ready = score.notna() & excess.notna()
    return pd.DataFrame(
        {
            "strategy": excess.mask(extreme, -excess).gt(0).where(ready),
            "plain_tsmom": excess.gt(0).where(ready),
        }
    )


def simulate(target, equity, cash, cost, cash_buffer, dividend_delay):
    """Size on the open, fill fixed shares at the modeled open, and mark at the close."""
    equity = equity.loc[target.index]
    cash = cash.loc[target.index]
    opens = np.column_stack((equity["Open"], cash["Open"]))
    closes = np.column_stack((equity["Close"], cash["Close"]))
    dividends = np.column_stack((equity["Dividends"], cash["Dividends"]))
    positions = np.zeros(2)
    free_cash = prior_value = 1.0
    current_asset = None
    pending_dividends = deque()
    daily_results = []

    for i, (session, use_equity) in enumerate(target.items()):
        asset = 0 if use_equity else 1
        while pending_dividends and pending_dividends[0][0] <= session:
            _, paid = pending_dividends.popleft()
            free_cash += paid
        # Entitlement belongs to the position carried into the ex-dividend day.
        new_dividend = positions @ dividends[i]
        if new_dividend > 0:
            pending_dividends.append((session + dividend_delay, new_dividend))
        unpaid_dividends = sum(amount for _, amount in pending_dividends)

        opening_prices = opens[i]
        opening_value = free_cash + positions @ opening_prices + unpaid_dividends
        trade_legs = 0
        traded_notional = 0.0
        if asset != current_asset:
            sale_notional = positions @ opening_prices
            available_cash = free_cash + sale_notional * (1 - cost)
            buy_shares = available_cash * (1 - cash_buffer) / (opening_prices[asset] * (1 + cost))

            # Apply costs at the opening mark after sizing the trade.
            buy_notional = buy_shares * opening_prices[asset]
            free_cash = available_cash - buy_notional * (1 + cost)
            if not np.isfinite(free_cash) or free_cash < 0:
                raise ValueError(f"Opening fills would leave invalid cash on {session.date()}: {free_cash}")
            positions.fill(0.0)
            positions[asset] = buy_shares
            trade_legs = int(sale_notional > 0) + int(buy_notional > 0)
            traded_notional = sale_notional + buy_notional
            current_asset = asset

        closing_value = free_cash + positions @ closes[i] + unpaid_dividends
        daily_results.append({
            "return": closing_value / prior_value - 1,
            "trade_legs": trade_legs,
            "turnover": traded_notional / opening_value,
            "traded_notional": traded_notional,
            "equity_shares": positions[0],
            "cash_etf_shares": positions[1],
            "free_cash": free_cash,
            "unpaid_dividends": unpaid_dividends,
        })
        prior_value = closing_value

    return pd.DataFrame(daily_results, index=target.index)


def annualized_sharpe(daily):
    if not np.isfinite(daily).all() or daily.nunique() < 2:
        return np.nan
    volatility = daily.std(ddof=1)
    return daily.mean() * np.sqrt(SESSIONS_PER_YEAR) / volatility if volatility > 0 else np.nan


def metrics(simulation, risk_free):
    daily = simulation["return"]
    value_path = np.r_[1.0, (1 + daily).cumprod().to_numpy()]
    excess = daily - risk_free.reindex(daily.index)
    values = {
        "cagr": value_path[-1] ** (SESSIONS_PER_YEAR / len(daily)) - 1,
        "ann_vol": daily.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR),
        "sharpe": annualized_sharpe(daily),
        "sharpe_excess": annualized_sharpe(excess),
        "max_drawdown": (value_path / np.maximum.accumulate(value_path) - 1).min(),
        "total_return": value_path[-1] - 1,
    }
    result = {key: float(value) if np.isfinite(value) else None for key, value in values.items()}
    result["n_trades"] = int(simulation["trade_legs"].sum())
    result["turnover"] = float(simulation["turnover"].sum())
    return result


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    is_start, is_end = [pd.Timestamp(config["in_sample"][key]) for key in ("start", "end")]
    oos_start, oos_end = [pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end")]
    data_start = pd.Timestamp(config["data_start"])
    if not data_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Use ordered, non-overlapping IS/OOS windows and earlier warm-up data")
    commission_bps = float(config["commission_bps_per_side"])
    slippage_bps = float(config["slippage_bps_per_side"])
    cost_bps = commission_bps + slippage_bps
    cost = cost_bps / 10_000
    if not (commission_bps >= 0 and slippage_bps >= 0 and cost < 1):
        raise ValueError("Commission and slippage must be nonnegative and total less than 10000 bps")
    cash_buffer = float(config["cash_buffer_fraction"])
    if not 0 < cash_buffer < 1:
        raise ValueError("The cash buffer must be a fraction strictly between zero and one")
    dividend_delay_days = config["dividend_cash_delay_days"]
    if type(dividend_delay_days) is not int or dividend_delay_days < 1:
        raise ValueError("The dividend cash delay must be a positive integer number of calendar days")
    dividend_delay = pd.Timedelta(days=dividend_delay_days)

    CACHE.mkdir(exist_ok=True)
    universe = config["universe"]
    with TemporaryDirectory(dir=CACHE, prefix="yahoo-") as metadata_dir:
        yf.set_tz_cache_location(metadata_dir)
        equity = fetch_etf(universe["equity"], data_start, oos_end)
        cash = fetch_etf(universe["cash"], data_start, oos_end)
    long_rates, long_fetched = fetch_treasury(universe["treasury_long"], data_start, oos_end)
    short_rates, short_fetched = fetch_treasury(universe["treasury_short"], data_start, oos_end)
    expected_sessions = market_sessions(data_start, oos_end)
    for symbol, prices, first_required in (
        (universe["equity"], equity, data_start),
        (universe["cash"], cash, cash.index[0]),
    ):
        expected = expected_sessions[expected_sessions >= first_required]
        missing = expected.difference(prices.index)
        unexpected = prices.index.difference(expected)
        if len(missing) or len(unexpected):
            raise ValueError(f"{symbol} session mismatch: missing {list(missing.date)}, extra {list(unexpected.date)}")

    monthly = month_end_signals(equity, cash, long_rates, short_rates, config["signal"])
    # Forward-fill only between monthly signals. Missing month-end inputs stay missing.
    targets = monthly.reindex(equity.index, method="ffill").shift(1).loc[is_start:oos_end]
    if targets.empty or targets.isna().any().any() or not targets.index.isin(cash.index).all():
        raise ValueError("Evaluation requires a valid prior month-end signal on every session")
    targets = targets.astype(bool)
    targets["buy_hold"] = True
    risk_free = cash["Adj Close"].pct_change(fill_method=None).loc[targets.index].copy()
    first_cash = cash.loc[targets.index[0]]
    risk_free.iloc[0] = first_cash["Close"] / first_cash["Open"] - 1

    windows = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    labels = {
        "strategy": "Valuation boundaries",
        "plain_tsmom": "Plain time-series momentum",
        "buy_hold": f"{universe['equity']} buy-and-hold",
    }
    reports = {}
    rows = []
    for method, label in labels.items():
        simulation = simulate(targets[method], equity, cash, cost, cash_buffer, dividend_delay)
        reports[method] = {}
        for window, (start, end) in windows.items():
            sample = simulation.loc[start:end]
            if sample.empty:
                raise ValueError(f"No trading sessions in {window.upper()}")
            report = metrics(sample, risk_free)
            reports[method][window] = report
            rows.append({"Portfolio": label, "Window": window.upper(), **report})

    index = targets.index
    data_range = f"{index[0].date()} to {index[-1].date()}"
    fetched_on = {equity.attrs["fetched_on"], cash.attrs["fetched_on"], long_fetched, short_fetched}
    print(f"DATA_FETCHED_AT: {', '.join(day.isoformat() for day in sorted(fetched_on))}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"PARAM_COMBOS_TRIED: {config['param_combos_tried']}")
    print(
        f"{config['strategy']} | Evaluation {data_range} | Cost {cost_bps:g} bps per side "
        f"({commission_bps:g} commission + {slippage_bps:g} slippage) | Cash proxy {universe['cash']}"
    )
    for name in ("is", "oos"):
        start, end = windows[name]
        print(f"{name.upper()} {start.date()} to {end.date()}")
    table = pd.DataFrame(rows)
    percent_columns = ["cagr", "ann_vol", "max_drawdown", "total_return"]
    formatters = {column: "{:.2%}".format for column in percent_columns}
    formatters.update(sharpe="{:.2f}".format, sharpe_excess="{:.2f}".format, turnover="{:.1f}x".format)
    print(table.to_string(index=False, formatters=formatters, na_rep="n/a"))

    buy_hold = reports["buy_hold"]
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **reports["strategy"],
        "benchmark": {
            "name": labels["buy_hold"],
            **buy_hold["full"],
            **buy_hold,
            "plain_tsmom": {"name": labels["plain_tsmom"], **reports["plain_tsmom"]},
        },
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
