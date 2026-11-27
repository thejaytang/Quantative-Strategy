# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 194_breaking_beta_defensive\ncategory: factor\n\ndata:\n  warmup_start: 2012-01-01\n  market: SPY\n  defensive_etfs: [USMV, SPLV]\n  cash_proxy: BIL\n  breadth_stocks:\n    - AAPL\n    - MSFT\n    - AMZN\n    - GOOGL\n    - META\n    - NVDA\n    - JPM\n    - BAC\n    - GS\n    - MS\n    - XOM\n    - CVX\n    - COP\n    - SLB\n    - WMT\n    - COST\n    - PG\n    - KO\n    - PEP\n    - JNJ\n    - PFE\n    - MRK\n    - HD\n    - LOW\n\nin_sample:\n  start: 2013-01-01\n  end: 2021-12-31\nout_of_sample:\n  start: 2022-01-01\n  end: 2026-11-24\n\nsignal:\n  beta_window_sessions: 126\n  min_eligible_stocks: 16\n  negative_beta_breadth_threshold: 0.15\n  defensive_beta_trigger: 0.75\n  target_beta: 0.85\n  max_spy_weight: 0.75\n\nexecution:\n  cash_buffer_bps: 50\n  distribution_cash_lag_days: 60\n\ncosts:\n  commission_bps: 5\n  slippage_bps: 2\n  # [commission, slippage] per side in basis points\n  sensitivity_bps: [[0, 0], [15, 2], [5, 5]]\n\nannualization_sessions: 252\n'
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
"""Monthly breadth-conditioned beta restoration for a U.S. defensive ETF sleeve."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
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
TOLERANCE = 1e-12
PARAM_COMBOS_TRIED = 1
PRICE_ADJUSTMENT = (
    "Yahoo split/dividend-adjusted closes for signals and reference returns; "
    "Yahoo Open/Close without dividend adjustment for execution and valuation; "
    "ETF distributions accrued separately on ex-dates"
)


class NyseHolidays(AbstractHolidayCalendar):
    # NYSE Group equity-market schedule, including full-day unscheduled closures in the sample.
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
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
        Holiday("Hurricane Sandy (Monday)", year=2012, month=10, day=29),
        Holiday("Hurricane Sandy (Tuesday)", year=2012, month=10, day=30),
        Holiday("National day of mourning (Bush)", year=2018, month=12, day=5),
        Holiday("National day of mourning (Carter)", year=2025, month=1, day=9),
    ]


def require_sessions(symbol, prices, sessions, columns):
    missing = sessions.difference(prices.index)
    extra = prices.index.difference(sessions)
    if len(missing) or len(extra):
        raise ValueError(
            f"{symbol} differs from the NYSE session calendar: "
            f"missing {missing[:5].strftime('%Y-%m-%d').tolist()}, "
            f"unexpected {extra[:5].strftime('%Y-%m-%d').tolist()}."
        )
    if prices[columns].isna().any().any():
        raise ValueError(f"{symbol} has missing or invalid required prices or distributions.")


def download_prices(symbol, start, end):
    """Read the public Yahoo Finance daily chart feed used by yfinance."""
    params = {
        "period1": int(start.tz_localize("UTC").timestamp()),
        "period2": int((end + pd.Timedelta(days=1)).tz_localize("UTC").timestamp()),
        "interval": "1d",
        "events": "div,splits,capitalGains",
    }
    failures = []
    with requests.Session() as session:
        session.trust_env = False
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
            url = f"https://{host}/v8/finance/chart/{symbol}"
            try:
                response = session.get(url, params=params, timeout=(7, 20))
                response.raise_for_status()
                chart = response.json()["chart"]["result"][0]
                meta = chart["meta"]
                if (
                    meta["dataGranularity"] != "1d" or meta["currency"] != "USD"
                    or meta["instrumentType"] not in ("EQUITY", "ETF")
                ):
                    raise ValueError("expected daily USD equity or ETF data")

                def session_dates(timestamps):
                    return (
                        pd.to_datetime(timestamps, unit="s", utc=True)
                        .tz_convert(meta["exchangeTimezoneName"])
                        .tz_localize(None)
                        .normalize()
                    )

                indicators = chart["indicators"]
                quote = indicators["quote"][0]
                prices = pd.DataFrame(
                    {
                        "open": quote["open"],
                        "close": quote["close"],
                        "adjusted_close": indicators["adjclose"][0]["adjclose"],
                    },
                    index=session_dates(chart["timestamp"]),
                    dtype=float,
                ).sort_index().loc[start:end]
                prices.index.name = "date"
                prices = prices.where(np.isfinite(prices) & prices.gt(0)).dropna(how="all")
                if prices.empty or not prices.index.is_unique:
                    raise ValueError("empty or duplicated daily data")
                events = chart.get("events", {})
                distributions = [
                    event
                    for kind in ("dividends", "capitalGains")
                    for event in events.get(kind, {}).values()
                ]
                prices["distribution"] = 0.0
                if distributions:
                    amounts = pd.Series(
                        [event["amount"] for event in distributions],
                        index=session_dates([event["date"] for event in distributions]),
                        dtype=float,
                    ).sort_index().loc[start:end]
                    if (
                        not np.isfinite(amounts).all() or amounts.lt(0).any()
                        or len(amounts.index.difference(prices.index))
                    ):
                        raise ValueError("invalid cash distribution amount or ex-date")
                    prices["distribution"] = amounts.groupby(level=0).sum().reindex(prices.index, fill_value=0.0)
                splits = session_dates([event["date"] for event in events.get("splits", {}).values()])
                prices.attrs["has_split"] = ((splits >= start) & (splits <= end)).any()
                return prices
            except (requests.RequestException, KeyError, TypeError, IndexError, ValueError) as error:
                failures.append(f"{host}: {error}")
    raise RuntimeError(f"Could not download {symbol}. " + " | ".join(failures))


def load_prices(symbols, start, end):
    with ThreadPoolExecutor(max_workers=6) as pool:
        frames = list(pool.map(lambda symbol: download_prices(symbol, start, end), symbols))
    return dict(zip(symbols, frames))


def month_open_decisions(closes, market, defensive_etfs, stocks, settings):
    returns = closes.pct_change(fill_method=None)
    market_returns = returns[market]
    blend = returns[defensive_etfs].mean(axis=1, skipna=False).rename("blend")
    candidates = pd.concat([returns[stocks], blend], axis=1)
    window = settings["beta_window_sessions"]
    market_variance = market_returns.rolling(window).var()
    betas = candidates.rolling(window).cov(market_returns).div(
        market_variance.where(market_variance.gt(TOLERANCE)), axis=0
    )
    stock_betas = betas[stocks]
    eligible = stock_betas.notna().sum(axis=1)
    breadth = stock_betas.lt(0).sum(axis=1).div(
        eligible.where(eligible.ge(settings["min_eligible_stocks"]))
    )
    beta = betas["blend"]
    low_beta = beta.lt(settings["defensive_beta_trigger"])
    trigger = low_beta & breadth.ge(settings["negative_beta_breadth_threshold"])
    restoration = (
        (settings["target_beta"] - beta) / (1 - beta.where(low_beta))
    ).clip(lower=0, upper=settings["max_spy_weight"])
    signals = pd.DataFrame(
        {
            "beta": beta,
            "breadth": breadth,
            "strategy": restoration.where(trigger, 0.0),
            "beta_only": restoration.where(low_beta, 0.0),
        }
    )
    month = closes.index.to_period("M")
    # The first new-month open consumes the previous session's completed signal.
    return signals.shift(1).loc[~month.duplicated()].iloc[1:]


def fill_at_open(positions, target_positions, cash, open_prices, commission, slippage):
    """Fill predetermined share changes, reducing buys only to fit opening cash."""
    sells = np.maximum(positions - target_positions, 0.0)
    buys = np.maximum(target_positions - positions, 0.0)
    sell_prices = open_prices * (1 - slippage)
    buy_prices = open_prices * (1 + slippage)
    buy_unit_cost = buy_prices * (1 + commission)
    sale_proceeds = (sells @ sell_prices) * (1 - commission)
    available = cash + sale_proceeds
    requested_cost = buys @ buy_unit_cost
    limited = requested_cost > available
    if limited:
        buys *= available / requested_cost
    remaining = available - buys @ buy_unit_cost
    if remaining < -TOLERANCE * max(1.0, available):
        raise RuntimeError("Opening fills exceeded available cash.")
    traded = sells @ sell_prices + buys @ buy_prices
    n_trades = np.count_nonzero((buys + sells) > 0)
    return positions - sells + buys, max(0.0, remaining), traded, n_trades, limited


def simulate(targets, opens, closes, previous_closes, distributions, release_rows,
             commission, slippage, cash_buffer):
    """Keep ETF shares and cash between rebalances and value them at each close."""
    positions = np.zeros(targets.shape[1])
    cash = 1.0
    accrued = 0.0
    releases = np.zeros(len(targets) + 1)
    previous_value = 1.0
    daily_results = np.zeros((len(targets), 5))
    for row, target in enumerate(targets):
        # Ex-date rights belong to the position carried into the session.
        prior_cash = cash
        cash += releases[row]
        distribution = positions @ distributions[row]
        accrued = max(0.0, accrued - releases[row]) + distribution
        releases[release_rows[row]] += distribution
        if np.isfinite(target[0]):
            close_prices = previous_closes[row]
            budget = (positions @ close_prices + prior_cash) * (1 - cash_buffer)
            target_positions = target * budget / close_prices
            opening_value = positions @ opens[row] + cash + accrued
            positions, cash, traded, n_trades, limited = fill_at_open(
                positions, target_positions, cash, opens[row], commission, slippage
            )
            daily_results[row, 1:4] = traded / opening_value, n_trades, limited
        closing_value = positions @ closes[row] + cash + accrued
        daily_results[row, 0] = closing_value / previous_value - 1
        daily_results[row, 4] = cash / closing_value
        previous_value = closing_value
    return daily_results


def metrics(results, market_returns, cash_returns, sessions):
    daily = results["return"]
    wealth = (1 + daily).cumprod()
    annual_vol = daily.std() * np.sqrt(sessions)
    excess = daily - cash_returns
    excess_vol = excess.std() * np.sqrt(sessions)
    benchmark_variance = market_returns.var()
    values = {
        "cagr": wealth.iloc[-1] ** (sessions / len(daily)) - 1,
        "ann_vol": annual_vol,
        "sharpe": daily.mean() * sessions / annual_vol if annual_vol > 0 else None,
        "sharpe_excess": excess.mean() * sessions / excess_vol if excess_vol > 0 else None,
        "max_drawdown": (wealth / wealth.cummax().clip(lower=1) - 1).min(),
        "total_return": wealth.iloc[-1] - 1,
        "tracking_error": (daily - market_returns).std() * np.sqrt(sessions),
        "realized_beta": (
            daily.cov(market_returns) / benchmark_variance if benchmark_variance > 0 else None
        ),
        "turnover": results["turnover"].sum(),
    }
    clean = {key: float(val) if val is not None and np.isfinite(val) else None
             for key, val in values.items()}
    return {**clean, "n_trades": int(results["n_trades"].sum()), "n_sessions": len(daily)}


def print_metrics(runs, data_dates):
    print(f"Actual return dates: {data_dates}")
    print(
        f"{'Portfolio':<17} {'Fee':>4} {'Slip':>4}  {'Period':<6} {'CAGR':>8} {'Ann vol':>8} "
        f"{'Sharpe':>7} {'Shp ex.':>7} {'Max DD':>8} {'Total':>9} {'Track err':>9} "
        f"{'Beta':>6} {'Turnover':>9} {'Trades':>7}"
    )
    fields = (
        ("cagr", "8.2%"), ("ann_vol", "8.2%"), ("sharpe", "7.2f"), ("sharpe_excess", "7.2f"),
        ("max_drawdown", "8.2%"), ("total_return", "9.2%"), ("tracking_error", "9.2%"),
        ("realized_beta", "6.2f"), ("turnover", "9.2f"), ("n_trades", "7d"),
    )
    for (commission, slippage), portfolios in runs.items():
        for label, periods in portfolios.items():
            for period, values in periods.items():
                rendered = " ".join(
                    format(values[key], style) if values[key] is not None
                    else "n/a".rjust(int(style.split(".")[0].rstrip("d")))
                    for key, style in fields
                )
                print(f"{label:<17} {commission:>4g} {slippage:>4g}  {period:<6} {rendered}")


def reference_returns(prices, report_index):
    daily = prices["adjusted_close"].pct_change(fill_method=None).loc[report_index]
    first = report_index[0]
    daily.iloc[0] = prices.at[first, "close"] / prices.at[first, "open"] - 1
    return daily


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data = config["data"]
    market = data["market"]
    defensive_etfs = data["defensive_etfs"]
    stocks = data["breadth_stocks"]
    cash_proxy = data["cash_proxy"]
    instruments = [market, *defensive_etfs]
    symbols = [*instruments, *stocks, cash_proxy]
    warmup = pd.Timestamp(data["warmup_start"])
    start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not warmup < start <= is_end < oos_start <= end:
        raise ValueError("Expected warm-up, followed by disjoint chronological IS and OOS ranges.")
    if len(set(symbols)) != len(symbols) or len(defensive_etfs) != 2:
        raise ValueError("Need two defensive ETFs and unique tickers across ETF and breadth inputs.")
    settings = config["signal"]
    if not (
        settings["beta_window_sessions"] >= 2
        and 1 <= settings["min_eligible_stocks"] <= len(stocks)
        and 0 <= settings["negative_beta_breadth_threshold"] <= 1
        and settings["defensive_beta_trigger"] <= settings["target_beta"] <= 1
        and settings["defensive_beta_trigger"] < 1
        and 0 <= settings["max_spy_weight"] <= 1
    ):
        raise ValueError("Invalid beta signal settings.")
    costs = config["costs"]
    primary_cost = (costs["commission_bps"], costs["slippage_bps"])
    cost_levels = dict.fromkeys([primary_cost, *map(tuple, costs["sensitivity_bps"])])
    if any(not 0 <= value < 10_000 for level in cost_levels for value in level):
        raise ValueError("Commission and slippage must be finite and in [0, 10000) bps.")
    cash_buffer = config["execution"]["cash_buffer_bps"] / 10_000
    release_days = config["execution"]["distribution_cash_lag_days"]
    if not 0 < cash_buffer < 1 or not isinstance(release_days, int) or release_days < 1:
        raise ValueError("Expected a positive cash buffer and distribution cash lag.")

    prices = load_prices(symbols, warmup, end)
    data_fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sessions = pd.bdate_range(warmup, end, name="date").difference(
        NyseHolidays().holidays(start=warmup, end=end)
    )
    for symbol in [*instruments, cash_proxy]:
        require_sessions(symbol, prices[symbol], sessions, ["open", "close", "adjusted_close"])
    for symbol in instruments:
        if prices[symbol].attrs["has_split"]:
            raise ValueError(f"Point-in-time split handling is required for {symbol} in this sample.")
    signal_closes = pd.DataFrame({symbol: prices[symbol]["adjusted_close"] for symbol in symbols
                                  if symbol != cash_proxy}, index=sessions)
    decisions = month_open_decisions(signal_closes, market, defensive_etfs, stocks, settings)

    report_index = sessions[(sessions >= start) & (sessions <= end)]
    periods = {"is": (start, is_end), "oos": (oos_start, end), "full": (start, end)}
    for label, (first, last) in periods.items():
        if not ((report_index >= first) & (report_index <= last)).any():
            raise ValueError(f"No ETF sessions for {label}.")
    first_open = report_index[0]
    initial = decisions.loc[:first_open].tail(1)
    if initial.empty or pd.isna(initial["beta"].iloc[0]):
        raise ValueError("Insufficient warm-up for a defensive beta estimate at the first fill.")
    schedule = decisions.reindex(report_index)
    schedule.loc[first_open] = initial.iloc[0]

    spy_weights = schedule[["strategy", "beta_only"]].copy()
    spy_weights["defensive_blend"] = 0.0
    spy_weights.loc[schedule["strategy"].isna(), "defensive_blend"] = np.nan
    spy_weights["spy_buy_hold"] = np.nan
    spy_weights.loc[first_open, "spy_buy_hold"] = 1.0
    opens = pd.DataFrame({asset: prices[asset]["open"] for asset in instruments}, index=sessions)
    closes = pd.DataFrame({asset: prices[asset]["close"] for asset in instruments}, index=sessions)
    distributions = pd.DataFrame({asset: prices[asset]["distribution"] for asset in instruments},
                                 index=sessions)
    opening_prices = opens.loc[report_index].to_numpy()
    closing_prices = closes.loc[report_index].to_numpy()
    previous_closes = closes.shift(1).loc[report_index].to_numpy()
    cash_distributions = distributions.loc[report_index].to_numpy()
    release_rows = report_index.searchsorted(report_index + pd.Timedelta(days=release_days))
    market_returns = reference_returns(prices[market], report_index)
    cash_returns = reference_returns(prices[cash_proxy], report_index)

    runs = {}
    for commission_bps, slippage_bps in cost_levels:
        portfolios = {}
        for label, weights in spy_weights.items():
            sleeve = (1 - weights) / 2
            targets = np.column_stack([weights, sleeve, sleeve])
            daily = pd.DataFrame(
                simulate(
                    targets, opening_prices, closing_prices, previous_closes, cash_distributions,
                    release_rows, commission_bps / 10_000, slippage_bps / 10_000, cash_buffer,
                ),
                index=report_index,
                columns=["return", "turnover", "n_trades", "cash_limited", "cash_weight"],
            )
            if (commission_bps, slippage_bps) == primary_cost and label == "strategy":
                primary_daily = daily
            portfolios[label] = {
                period: metrics(
                    daily.loc[first:last], market_returns.loc[first:last], cash_returns.loc[first:last],
                    config["annualization_sessions"],
                )
                for period, (first, last) in periods.items()
            }
        runs[commission_bps, slippage_bps] = portfolios

    diagnostics = {}
    for period, (first, last) in periods.items():
        fills = schedule.loc[first:last].dropna(subset=["strategy"])
        diagnostics[period] = {
            "scheduled_rebalances": len(fills),
            "breadth_available_rebalances": int(fills["breadth"].notna().sum()),
            "breadth_overlay_rebalances": int(fills["strategy"].gt(0).sum()),
            "beta_only_overlay_rebalances": int(fills["beta_only"].gt(0).sum()),
        }
    data_dates = f"{report_index[0].date()} to {report_index[-1].date()}"
    primary = runs[primary_cost]
    output = {
        "strategy": config["strategy"],
        "data_range": data_dates,
        **primary["strategy"],
        "benchmark": {"name": "defensive_blend", **primary["defensive_blend"]},
        "comparisons": {key: primary[key] for key in ("beta_only", "spy_buy_hold")},
        "cash_reference": cash_proxy,
        "cost_bps": primary_cost[0],
        "slippage_bps": primary_cost[1],
        "cost_sensitivity": {
            f"{fee:g}+{slip:g}": run for (fee, slip), run in runs.items() if (fee, slip) != primary_cost
        },
        "signal_diagnostics": diagnostics,
    }
    print(f"PARAM_COMBOS_TRIED: {PARAM_COMBOS_TRIED}")
    print(f"DATA_FETCHED_AT: {data_fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"Cash reference for excess Sharpe: {cash_proxy}")
    print_metrics(runs, data_dates)
    print(f"Cash-limited buy rebalances: {int(primary_daily['cash_limited'].sum())}")
    print(f"Minimum closing cash weight: {primary_daily['cash_weight'].min():.6%}")
    for period, stats in diagnostics.items():
        print(
            f"{period.upper()} overlay rebalances were {stats['breadth_overlay_rebalances']} (breadth rule) "
            f"and {stats['beta_only_overlay_rebalances']} (beta only). Breadth was available at "
            f"{stats['breadth_available_rebalances']}/{stats['scheduled_rebalances']} rebalances."
        )
    print("RESULTS_JSON: " + json.dumps(output, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
