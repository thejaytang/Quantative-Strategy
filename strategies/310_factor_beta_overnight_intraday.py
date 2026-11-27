# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 310_factor_beta_overnight_intraday\nuniverse:\n  quality: QUAL\n  market: SPY\n  growth: IWF\ncash_proxy: BIL\n\n# Date bounds are inclusive. The earlier data is used only to initialize the hedge.\ndata_start: '2014-01-01'\nin_sample:\n  start: '2015-01-01'\n  end: '2023-05-31'\nout_of_sample:\n  start: '2023-06-01'\n  end: '2025-12-31'\n\nbeta_lookback_sessions: 126\ngross_exposure: 1.0\n# Extra room for the price move between open sizing and close settlement.\ncash_buffer: 0.15\ntransaction_cost_bps_per_fill: 1.0\nslippage_bps_per_fill: 1.0\nsessions_per_year: 252\n# The 10% buffer attempt stopped on a gross exposure guard before reporting metrics.\nparam_combos_tried: 2\n"
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
"""Backtest the C101 US ETF proxy for session-specific profitability exposure."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
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
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_COLUMNS = ["Open", "Close"]
PRICE_ADJUSTMENT = "Yahoo Finance yfinance auto_adjust=True (split/dividend-adjusted OHLC)"


class NyseHolidays(AbstractHolidayCalendar):
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


def expected_sessions(start, end):
    # Full-day NYSE Group closures in the study interval. Early closes remain sessions.
    special_closures = pd.DatetimeIndex(["2018-12-05", "2025-01-09"])
    closures = NyseHolidays().holidays(start, end).union(special_closures)
    return pd.bdate_range(start, end).difference(closures)


def load_prices(symbols, start, end):
    CACHE.mkdir(exist_ok=True)
    sessions = expected_sessions(start, end)
    prices = {}
    fetched_at = {}
    # Keep yfinance's auxiliary cache local and separate from the price CSVs.
    with TemporaryDirectory(dir=CACHE) as yahoo_cache:
        yf.set_tz_cache_location(yahoo_cache)
        for symbol in symbols:
            path = CACHE / f"{symbol}_{start.date()}_{end.date()}_adjusted.csv"
            cached = path.exists()
            if cached:
                frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
                stamp = frame.get("fetched_at_utc", pd.Series(dtype="object")).dropna()
                if stamp.empty:
                    modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                    fetched_at[symbol] = modified.isoformat(timespec="seconds") + " (legacy CSV mtime estimate)"
                else:
                    fetched_at[symbol] = stamp.iloc[0]
            else:
                frame = yf.Ticker(symbol).history(
                    start=start.date().isoformat(),
                    end=(end + pd.Timedelta(days=1)).date().isoformat(),
                    interval="1d",
                    auto_adjust=True,
                    actions=False,
                    timeout=15,
                )
                if frame.empty:
                    raise ValueError(f"Yahoo Finance returned no daily prices for {symbol}")
                frame.index = frame.index.tz_localize(None)
                fetched_at[symbol] = datetime.now(timezone.utc).isoformat(timespec="seconds")

            frame = frame[PRICE_COLUMNS].loc[start:end]
            values = frame.to_numpy()
            if not frame.index.equals(sessions):
                missing = sessions.difference(frame.index).strftime("%Y-%m-%d")[:5].tolist()
                extra = frame.index.difference(sessions).strftime("%Y-%m-%d")[:5].tolist()
                raise ValueError(
                    f"{symbol} dates must match the ordered NYSE sessions "
                    f"(first missing dates {missing}, first extra dates {extra})"
                )
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"Invalid adjusted daily prices for {symbol}")

            if not cached:
                temp = Path(yahoo_cache) / path.name
                snapshot = frame.assign(fetched_at_utc="")
                snapshot.at[frame.index[0], "fetched_at_utc"] = fetched_at[symbol]
                snapshot.to_csv(temp, index_label="date")
                temp.replace(path)
            prices[symbol] = frame

    panel = pd.concat(prices, axis=1)
    return panel.xs("Open", axis=1, level=1), panel.xs("Close", axis=1, level=1), fetched_at


def month_starts(index):
    month = index.to_period("M")
    return np.r_[True, month[1:] != month[:-1]]


def hedge_betas(opens, closes, market, long_legs, lookback):
    overnight = opens.div(closes.shift(1)).sub(1)
    intraday = closes.div(opens).sub(1)
    # Pool two observations per date, ordered by when each return is known.
    sessions = pd.concat({"overnight": overnight, "intraday": intraday})
    dates_and_sessions = pd.MultiIndex.from_product(
        [closes.index, ["overnight", "intraday"]]
    )
    sessions = sessions.swaplevel().reindex(dates_and_sessions)
    market_returns = sessions[market]
    window = 2 * lookback
    variance = market_returns.rolling(window).var()
    beta = sessions[long_legs].rolling(window).cov(market_returns).div(
        variance.where(variance > 0), axis=0
    )
    # At the first trading day's open only the prior month's closing estimate is usable.
    previous_close = beta.xs("intraday", level=1).shift(1)
    updates = previous_close.loc[month_starts(closes.index)]
    return updates.reindex(closes.index, method="ffill").clip(lower=0)


def fill_positions(positions, next_positions, prices, cash, fee_rate, slippage_rate, context):
    changes = next_positions - positions
    # Buy above the reference quote and sell below it, including when covering a short.
    fill_values = changes * prices * (1 + np.sign(changes) * slippage_rate)
    notional = np.abs(fill_values).sum()
    cash_after = cash - fill_values.sum() - notional * fee_rate
    marked_positions = next_positions * prices
    equity_after = cash_after + marked_positions.sum()
    if cash_after < 0:
        raise ValueError(f"Predetermined fills would make cash negative ({context}: {cash_after})")
    if np.abs(marked_positions).sum() > equity_after:
        raise ValueError(f"Predetermined fills would exceed 100% gross exposure ({context})")
    return cash_after, notional, np.count_nonzero(changes)


def simulate(opens, closes, targets, mode, cash_buffer, fee_rate, slippage_rate):
    open_targets = targets.copy()
    close_targets = targets.copy()
    if mode == "overnight":
        open_targets[:] = 0.0
    elif mode == "intraday":
        close_targets[:] = 0.0
    else:
        rebalance = (
            month_starts(opens.index) if mode == "continuous" else np.arange(len(targets)) == 0
        )
        open_targets[~rebalance] = np.nan
        close_targets[:] = np.nan
    close_targets[-1] = 0.0

    cash = equity = 1.0
    positions = np.zeros(opens.shape[1])
    daily_results = []
    for date, morning, evening, at_open, at_close in zip(
        opens.index, opens.to_numpy(), closes.to_numpy(), open_targets, close_targets
    ):
        start_equity = equity
        opening_equity = cash + np.dot(positions, morning)
        if opening_equity <= 0:
            raise ValueError(f"Nonpositive opening equity in {mode} on {date.date()}")
        notional = 0.0
        trades = 0

        if not np.isnan(at_open).all():
            opening_positions = at_open * (opening_equity * (1 - cash_buffer)) / morning
            cash, notional, trades = fill_positions(
                positions, opening_positions, morning, cash, fee_rate, slippage_rate,
                f"{mode} {date.date()} open",
            )
            positions = opening_positions
            opening_equity = cash + np.dot(positions, morning)

        # Set the closing quantities using information available just after the open.
        if not np.isnan(at_close).all():
            closing_positions = at_close * (opening_equity * (1 - cash_buffer)) / morning
            cash, close_notional, close_trades = fill_positions(
                positions, closing_positions, evening, cash, fee_rate, slippage_rate,
                f"{mode} {date.date()} close",
            )
            positions = closing_positions
            notional += close_notional
            trades += close_trades

        equity = cash + np.dot(positions, evening)
        if equity <= 0:
            raise ValueError(f"Nonpositive closing equity in {mode} on {date.date()}")
        daily_results.append((equity / start_equity - 1, notional / start_equity, trades))
    return pd.DataFrame(daily_results, index=opens.index, columns=["return", "turnover", "n_trades"])


def metrics(frame, cash_returns, sessions_per_year):
    returns = frame["return"]
    wealth = (1 + returns).cumprod()
    vol_scale = np.sqrt(sessions_per_year)
    volatility = returns.std() * vol_scale
    excess = returns - cash_returns
    excess_volatility = excess.std() * vol_scale
    result = {
        "cagr": wealth.iloc[-1] ** (sessions_per_year / len(frame)) - 1,
        "ann_vol": volatility,
        "sharpe": returns.mean() * sessions_per_year / volatility if volatility > 0 else np.nan,
        "sharpe_excess": (
            excess.mean() * sessions_per_year / excess_volatility if excess_volatility > 0 else np.nan
        ),
        "max_drawdown": (wealth / wealth.cummax().clip(lower=1.0) - 1).min(),
        "total_return": wealth.iloc[-1] - 1,
        "turnover": frame["turnover"].mean() * sessions_per_year,
    }
    return {
        **{key: float(value) if np.isfinite(value) else None for key, value in result.items()},
        "n_trades": int(frame["n_trades"].sum()),
        "n_sessions": len(frame),
    }


def print_table(reports, periods):
    columns = {
        "cagr": ("CAGR", ".2%"),
        "ann_vol": ("Ann. vol", ".2%"),
        "sharpe": ("Sharpe", ".2f"),
        "sharpe_excess": ("Ex. SR", ".2f"),
        "max_drawdown": ("Max DD", ".2%"),
        "total_return": ("Total", ".2%"),
        "turnover": ("Ann. TO", ".2f"),
        "n_trades": ("Fills", ",d"),
    }
    for period, dates in periods.items():
        print(f"{period.upper():<4} {dates[0].date()} to {dates[-1].date()} ({len(dates)} sessions)")
    print(f"{'Portfolio':<23} {'Period':<6}" + "".join(f"{title:>10}" for title, _ in columns.values()))
    for name, report in reports.items():
        for period, values in report.items():
            cells = [
                format(values[key], spec) if values[key] is not None else "n/a"
                for key, (_, spec) in columns.items()
            ]
            print(f"{name:<23} {period.upper():<6}" + "".join(f"{cell:>10}" for cell in cells))


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)

    quality, market, growth = (config["universe"][leg] for leg in ("quality", "market", "growth"))
    cash_proxy = config["cash_proxy"]
    symbols = [quality, market, growth, cash_proxy]
    if len(set(symbols)) != 4 or not all(re.fullmatch(r"[A-Z]{1,5}", s) for s in symbols):
        raise ValueError("Configure four distinct U.S. ETF ticker symbols")
    data_start = pd.Timestamp(config["data_start"])
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not (data_start < is_start <= is_end < oos_start <= oos_end):
        raise ValueError("Ensure that the data warmup, IS and OOS ranges are ordered and disjoint")
    lookback = config["beta_lookback_sessions"]
    gross = float(config["gross_exposure"])
    cash_buffer = float(config["cash_buffer"])
    fee_rate = float(config["transaction_cost_bps_per_fill"]) / 10_000
    slippage_rate = float(config["slippage_bps_per_fill"]) / 10_000
    sessions_per_year = config["sessions_per_year"]
    if not (
        isinstance(lookback, int)
        and lookback >= 2
        and 0 < gross <= 1
        and 0 < cash_buffer < 1
        and 0 <= fee_rate < 1
        and 0 <= slippage_rate < 1
        and sessions_per_year > 0
    ):
        raise ValueError("Check lookback, exposure, cash buffer, costs and annualization settings")

    opens, closes, fetched_at = load_prices(symbols, data_start, oos_end)
    cash_returns = closes[cash_proxy].pct_change(fill_method=None)
    hedge_symbols = [quality, market, growth]
    betas = hedge_betas(
        opens[hedge_symbols], closes[hedge_symbols], market, [quality, growth], lookback
    ).loc[is_start:oos_end]
    if betas.empty or not np.isfinite(betas.to_numpy()).all():
        raise ValueError("A valid hedge is needed for every evaluation date; check warmup and prices")
    opens, closes = opens.loc[betas.index], closes.loc[betas.index]
    cash_returns = cash_returns.loc[betas.index]
    if not np.isfinite(cash_returns.to_numpy()).all():
        raise ValueError("A valid cash proxy return is needed for every evaluation date")
    periods = {
        "is": betas.loc[is_start:is_end].index,
        "oos": betas.loc[oos_start:oos_end].index,
        "full": betas.index,
    }
    if not all(len(dates) for dates in periods.values()):
        raise ValueError("Each sample range must contain trading sessions")

    def spread_targets(asset):
        beta = betas[asset].to_numpy()
        return np.column_stack((np.ones(len(beta)), -beta)) * (gross / (1 + beta))[:, None]

    quality_targets = spread_targets(quality)
    benchmark_name = f"{market} buy-and-hold"
    portfolios = {
        f"{quality} overnight": ([quality, market], quality_targets, "overnight"),
        f"{quality} intraday": ([quality, market], quality_targets, "intraday"),
        f"{quality} continuous": ([quality, market], quality_targets, "continuous"),
        f"{growth} overnight": ([growth, market], spread_targets(growth), "overnight"),
        benchmark_name: ([market], np.ones((len(betas), 1)), "buy_hold"),
    }
    reports = {}
    for name, (legs, targets, mode) in portfolios.items():
        frame = simulate(opens[legs], closes[legs], targets, mode, cash_buffer, fee_rate, slippage_rate)
        reports[name] = {
            period: metrics(frame.loc[dates], cash_returns.loc[dates], sessions_per_year)
            for period, dates in periods.items()
        }

    strategy_name = f"{quality} overnight"
    results = {
        "strategy": config["strategy"],
        "data_range": f"{betas.index[0].date()} to {betas.index[-1].date()}",
        **reports[strategy_name],
        "benchmark": {
            "name": benchmark_name,
            "cash_proxy": cash_proxy,
            **reports[benchmark_name],
            "comparisons": {
                name: reports[name] for name in (f"{quality} intraday", f"{quality} continuous", f"{growth} overnight")
            },
        },
    }
    print(f"PARAM_COMBOS_TRIED: {config['param_combos_tried']}")
    print("DATA_FETCHED_AT: " + ", ".join(f"{symbol}={stamp}" for symbol, stamp in fetched_at.items()))
    print("PRICE_ADJUSTMENT: " + PRICE_ADJUSTMENT)
    print_table(reports, periods)
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
