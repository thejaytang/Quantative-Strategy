# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 316_hidden_cost_etf_investing\nhedge_ticker: SPY\npair_tickers: [IVV, VOO, RSP, VTI]\ncash_ticker: BIL\n\n# Date bounds are inclusive. The download starts before the reporting window.\ndata_start: 2011-01-01\nin_sample:\n  start: 2012-01-01\n  end: 2018-12-31\nout_of_sample:\n  start: 2019-01-01\n  end: 2025-12-31\n\nlookback_sessions: 126\ngross_cap: 1.0  # Reference gross weight before the sizing buffer.\ncash_buffer: 0.10  # Reserved when converting target weights into shares.\ntransaction_cost_bps: 2.0  # Per side, on the dollar amount traded in each ETF.\nslippage_bps: 1.0  # Additional adverse fill cost per side.\nsessions_per_year: 252\n'
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
"""Backtest the C107 ETF relative-value spread by trading session."""

import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

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


class EquityHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def expected_sessions(start, end):
    # Exceptional closures during the configured warmup and sample.
    exceptional_closures = pd.to_datetime([
        "2012-10-29", "2012-10-30",  # Hurricane Sandy
        "2018-12-05", "2025-01-09",  # National days of mourning
    ])
    holidays = EquityHolidays().holidays(start, end).union(exceptional_closures)
    return pd.bdate_range(start, end).difference(holidays)


def validate_prices(prices, sessions):
    index = prices.index
    if (not isinstance(index, pd.DatetimeIndex) or index.hasnans
            or not index.is_unique or not index.is_monotonic_increasing):
        raise RuntimeError("Yahoo Finance returned invalid, duplicate or unordered dates")
    if not prices.index.equals(sessions):
        missing = sessions.difference(prices.index).strftime("%Y-%m-%d").tolist()
        extra = prices.index.difference(sessions).strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"Yahoo Finance dates do not match the exchange calendar "
                           f"(missing {len(missing)} {missing[:5]}, extra {len(extra)} {extra[:5]})")
    if not np.isfinite(prices.to_numpy()).all():
        raise RuntimeError("Yahoo Finance returned missing or nonfinite values")
    if (prices[["Open", "Close", "Adj Close"]] <= 0).any().any():
        raise RuntimeError("Yahoo Finance returned a nonpositive price")
    if (prices[["Dividends", "Capital Gains"]] < 0).any().any():
        raise RuntimeError("Yahoo Finance returned a negative distribution")


def load_prices(tickers, start, end):
    sessions = expected_sessions(start, end)
    columns = pd.MultiIndex.from_product(
        [["Open", "Close", "Adj Close", "Dividends", "Capital Gains"], tickers]
    )
    with TemporaryDirectory(dir=ROOT, prefix=".yahoo_metadata_") as metadata:
        yf.set_tz_cache_location(metadata)
        for attempt in range(2):
            data = yf.download(
                tickers, start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d", actions=True, auto_adjust=False, ignore_tz=True,
                threads=False, progress=False,
                timeout=15, multi_level_index=True,
            )
            fetched_at = datetime.now(ZoneInfo("Europe/Oslo")).isoformat(timespec="seconds")
            prices = data.reindex(columns=columns)
            try:
                validate_prices(prices, sessions)
            except RuntimeError as error:
                if attempt == 1:
                    raise
                print(f"Retrying the Yahoo Finance download: {error}", flush=True)
            else:
                return prices, fetched_at


def fit_session(returns):
    market = returns[:, 0]
    candidates = returns[:, 1:]
    centered_market = market - market.mean()
    sum_squares = centered_market @ centered_market
    if sum_squares <= 0:
        raise ValueError("Cannot estimate a session beta with zero SPY variance")
    beta = centered_market @ (candidates - candidates.mean(axis=0)) / sum_squares
    # The mean of y - beta*x is the fitted intercept. Subtracting the intercept
    # as well would force every sample mean residual to zero.
    premium = (candidates - market[:, None] * beta).mean(axis=0)
    return beta, premium


def monthly_targets(overnight, intraday, dates, lookback, gross_cap):
    night_targets = np.zeros_like(overnight)
    day_targets = np.zeros_like(intraday)
    months = dates.to_period("M")
    month_starts = np.flatnonzero(months[1:] != months[:-1]) + 1
    for start, end in zip(month_starts, np.r_[month_starts[1:], len(dates)]):
        if start <= lookback:  # The first close-to-open return needs a previous close.
            continue
        history = slice(start - lookback, start)
        night_beta, night_premium = fit_session(overnight[history])
        day_beta, day_premium = fit_session(intraday[history])
        ranking = np.argsort(night_premium - day_premium, kind="stable")
        short, long = ranking[[0, -1]]
        for targets, beta in ((night_targets, night_beta), (day_targets, day_beta)):
            weights = np.zeros(overnight.shape[1])
            weights[long + 1] = 1.0
            weights[short + 1] = -1.0
            weights[0] = beta[short] - beta[long]
            targets[start:end] = weights * (gross_cap / np.abs(weights).sum())
    return night_targets, day_targets


def size_positions(positions, target, equity, prices, cash_buffer):
    if target is None:
        return positions
    return target * (equity * (1.0 - cash_buffer)) / prices


def fill(positions, next_positions, cash, prices, cost_rate, date, session):
    changes = next_positions - positions
    notional = np.abs(changes) @ prices
    cost = cost_rate * notional
    next_cash = cash - changes @ prices - cost
    equity = next_cash + next_positions @ prices
    gross = np.abs(next_positions) @ prices
    if not np.isfinite(equity) or equity <= 0 or next_cash < 0 or gross > equity:
        raise RuntimeError(f"Insufficient capital for fixed quantities on {date.date()} {session} "
                           f"(cash={next_cash:.8g}, equity={equity:.8g}, gross={gross:.8g})")
    return next_cash, equity, notional, cost, np.count_nonzero(changes)


def simulate(dates, opens, closes, previous_closes, distributions,
             open_targets, close_targets, cash_buffer, cost_rate):
    cash = equity = 1.0
    positions = np.zeros(opens.shape[1])
    daily_results = []
    for date, opening, closing, previous_close, distribution, open_target, close_target in zip(
        dates, opens, closes, previous_closes, distributions, open_targets, close_targets,
        strict=True,
    ):
        previous_equity = equity
        open_positions = size_positions(positions, open_target, equity, previous_close, cash_buffer)
        # Ex-date distributions belong to positions carried from the prior close.
        cash += positions @ distribution
        cash, open_equity, open_notional, open_cost, open_trades = fill(
            positions, open_positions, cash, opening, cost_rate, date, "open"
        )
        close_positions = size_positions(
            open_positions, close_target, open_equity, opening, cash_buffer
        )
        cash, equity, close_notional, close_cost, close_trades = fill(
            open_positions, close_positions, cash, closing, cost_rate, date, "close"
        )
        positions = close_positions
        net_return = equity / previous_equity - 1.0
        cost_fraction = (open_cost + close_cost) / previous_equity
        daily_results.append((net_return, net_return + cost_fraction, cost_fraction,
                              (open_notional + close_notional) / previous_equity,
                              open_trades + close_trades))
    return pd.DataFrame(daily_results, index=dates,
                        columns=["return", "gross_return", "cost_fraction", "turnover", "n_trades"])


def finite_float(value):
    return float(value) if np.isfinite(value) else None


def annualized_sharpe(returns, annualization):
    if len(returns) < 2 or np.ptp(returns) == 0:
        return None
    return finite_float(returns.mean() / returns.std(ddof=1) * annualization)


def metrics(daily_results, cash_returns, annual_sessions):
    returns = daily_results["return"].to_numpy()
    excess = returns - cash_returns
    equity = np.r_[1.0, np.cumprod(1.0 + returns)]
    std = returns.std(ddof=1) if len(returns) > 1 else np.nan
    annualization = np.sqrt(annual_sessions)
    cagr_exponent = annual_sessions / len(returns)
    gross_growth = (1.0 + daily_results["gross_return"]).prod()
    return {
        "cagr": finite_float(equity[-1] ** cagr_exponent - 1.0),
        "ann_vol": finite_float(std * annualization),
        "sharpe": annualized_sharpe(returns, annualization),
        "sharpe_excess": annualized_sharpe(excess, annualization),
        "max_drawdown": finite_float((equity / np.maximum.accumulate(equity) - 1.0).min()),
        "total_return": finite_float(equity[-1] - 1.0),
        "n_trades": int(daily_results["n_trades"].sum()),
        "turnover": finite_float(daily_results["turnover"].sum()),
        "gross_cagr": finite_float(gross_growth ** cagr_exponent - 1.0),
        "cost_sum": finite_float(daily_results["cost_fraction"].sum()),
    }


def summarize(daily_results, cash_returns, periods, annual_sessions):
    result = {}
    for name, (start, end) in periods.items():
        period = daily_results.loc[start:end]
        if period.empty:
            raise ValueError(f"No sessions in reporting period {name}")
        result[name] = metrics(period, cash_returns.loc[period.index].to_numpy(), annual_sessions)
    return result


def print_table(reports):
    columns = (
        ("cagr", "CAGR", 9, ".2%"), ("ann_vol", "Ann vol", 9, ".2%"),
        ("sharpe", "Sharpe", 8, ".2f"), ("sharpe_excess", "Excess SR", 9, ".2f"),
        ("max_drawdown", "Max DD", 9, ".2%"),
        ("total_return", "Total", 9, ".2%"), ("n_trades", "Trades", 9, "d"),
        ("turnover", "Turnover", 9, ".2f"),
        ("gross_cagr", "Gross CAGR", 10, ".2%"), ("cost_sum", "Cost sum", 9, ".2%"),
    )
    headings = " ".join(label.rjust(width) for _, label, width, _ in columns)
    print(f"{'Portfolio':<26} {'Period':<6} {headings}")
    for portfolio, report in reports.items():
        for period, row in report.items():
            values = [(format(row[key], spec) if row[key] is not None else "-").rjust(width)
                      for key, _, width, spec in columns]
            print(f"{portfolio:<26} {period.upper():<6} {' '.join(values)}")


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    hedge = config["hedge_ticker"]
    cash_ticker = config["cash_ticker"]
    tickers = [hedge, *config["pair_tickers"]]
    lookback = config["lookback_sessions"]
    gross_cap = config["gross_cap"]
    fee_bps = config["transaction_cost_bps"]
    slippage_bps = config["slippage_bps"]
    cost_rate = (fee_bps + slippage_bps) / 10_000
    cash_buffer = config["cash_buffer"]
    annual_sessions = config["sessions_per_year"]
    data_start = pd.Timestamp(config["data_start"])
    periods = {}
    for label, section in (("is", "in_sample"), ("oos", "out_of_sample")):
        periods[label] = tuple(pd.Timestamp(config[section][key]) for key in ("start", "end"))
    is_start, is_end = periods["is"]
    oos_start, oos_end = periods["oos"]
    if not data_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Check the date ordering of the warmup, IS and OOS windows")
    if len(tickers) < 3 or len(set(tickers)) != len(tickers) or cash_ticker in tickers:
        raise ValueError("Configure distinct symbols for the hedge, cash proxy and at least two candidates")
    if (lookback < 2 or not 0 < gross_cap <= 1 or not 0 < cash_buffer < 1
            or not 0 <= fee_bps or not 0 <= slippage_bps or not 0 <= cost_rate <= 0.05
            or annual_sessions <= 0):
        raise ValueError("Invalid lookback, exposure, costs or annualization setting")
    periods["full"] = (is_start, oos_end)

    prices, fetched_at = load_prices([*tickers, cash_ticker], data_start, oos_end)
    opens = prices["Open"][tickers]
    closes = prices["Close"][tickers]
    adjusted_closes = prices["Adj Close"][tickers]
    adjusted_opens = opens * (adjusted_closes / closes)
    overnight = (adjusted_opens / adjusted_closes.shift(1) - 1.0).to_numpy()
    intraday = (adjusted_closes / adjusted_opens - 1.0).to_numpy()
    night, day = monthly_targets(overnight, intraday, prices.index, lookback, gross_cap)
    cash_returns = prices["Adj Close"][cash_ticker].pct_change(fill_method=None)
    distributions = (prices["Dividends"] + prices["Capital Gains"])[tickers]
    sample = (prices.index >= is_start) & (prices.index <= oos_end)
    dates = prices.index[sample]
    night, day = night[sample], day[sample]
    if not len(dates) or not night[0].any():
        raise ValueError("Download more warmup data to fit the first month's signal")
    previous_closes = closes.shift(1).loc[dates].to_numpy()
    sample_opens = opens.loc[dates].to_numpy()
    sample_closes = closes.loc[dates].to_numpy()
    sample_distributions = distributions.loc[dates].to_numpy()
    flat = np.zeros_like(night)
    night[-1] = 0.0  # End at the last close, without starting another overnight trade.

    buy = np.zeros(len(tickers))
    buy[0] = 1.0
    benchmark_opens = [None] * len(dates)
    benchmark_closes = [None] * len(dates)
    benchmark_opens[0] = buy
    benchmark_closes[-1] = flat[-1]
    benchmark_name = f"{hedge} buy and hold"
    portfolios = {
        "Overnight spread": (flat, night),
        "Intraday spread": (day, flat),
        "Continuous spread": (day, night),
        benchmark_name: (benchmark_opens, benchmark_closes),
    }
    reports = {}
    for name, (open_targets, close_targets) in portfolios.items():
        daily_results = simulate(
            dates, sample_opens, sample_closes, previous_closes, sample_distributions,
            open_targets, close_targets, cash_buffer, cost_rate,
        )
        reports[name] = summarize(daily_results, cash_returns, periods, annual_sessions)

    strategy_report = reports["Overnight spread"]
    benchmark = reports[benchmark_name]
    print("PARAM_COMBOS_TRIED: 1")
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: Yahoo split-adjusted fills, ex-date distributions in cash; "
          "dividend/split-adjusted signals and cash proxy")
    print(f"DATA_SHA256: {sha256(prices.to_csv().encode()).hexdigest()}")
    print(f"{config['strategy']} | {dates[0].date()} to {dates[-1].date()} | "
          f"{fee_bps:g} bps fee + {slippage_bps:g} bps slippage per side | "
          f"{cash_buffer:.0%} sizing buffer | excess Sharpe vs {cash_ticker}")
    print_table(reports)
    result = {
        "strategy": config["strategy"],
        "data_range": f"{dates[0].date()} to {dates[-1].date()}",
        **strategy_report,
        "benchmark": {"name": benchmark_name, **benchmark["full"], **benchmark},
        "comparisons": {
            "intraday": reports["Intraday spread"],
            "continuous": reports["Continuous spread"],
        },
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
