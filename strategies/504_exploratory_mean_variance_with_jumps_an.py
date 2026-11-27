"""504_exploratory_mean_variance_with_jumps_an
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/504_exploratory_mean_variance_with_jumps_an.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit da9754ee51e8d74f175ab50e14ef28c47c764ad1 on 2026-11-27 12:05 Europe/Oslo, exit 0, output sha256 38570417eb8e24668bc90bc615615dd137c17a644271606c761fb25c20277c21
Data source: Yahoo Finance via yfinance for daily closes of SPY, IEF and SHY and the daily ^IRX 13-week Treasury-bill discount yield, used for the excess-return metric. FRED public CSV for daily DFF (effective federal funds rate), used for the cash return and the signal’s excess-return estimate.
Data fetched: 2026-11-27T11:05:23.189367+00:00
Sample period: Yahoo data requested from 2003-01-01 through 2025-12-31, including pre-backtest warm-up. The FRED request starts 14 calendar days earlier, on 2002-12-18. The evaluation start is configured as 2005-01-01 and the logged evaluation sessions run from 2005-01-03 through 2025-12-31, inclusive.   data_range (from output): {'start': '2005-01-03', 'end': '2025-12-31'}
IS/OOS split: In sample 2005-01-03 through 2014-12-31. Out of sample 2015-01-02 through 2025-12-31. The report slices a continuous default simulation at 2014-12-31. The OOS segment carries forward existing holdings, uses the same rules and updates rolling inputs. The eight sensitivity variants run only in sample. The README states that the split was chosen retrospectively.
Costs and slippage: Default commission of 1 bp and fixed slippage of 2 bps per dollar of ETF purchases and sales, for 3 bps total. At each simulated fill, the solver deducts charges from portfolio value and computes absolute ETF trade notionals using the resulting post-cost target values and drifted pre-trade holdings. The doubled-cost IS variant uses 2 bps commission and 4 bps slippage. The SPY benchmark pays the default rates on its initial purchase at the strategy’s first fill.
Signal lag: At the last observed trading close of each month, the default rule computes target weights using 252 ETF returns through that close and the lagged DFF series as of the signal date. The fill occurs at the next trading session’s close. The old holdings earn the fill day’s close-to-close return before the target weights are applied.
Price adjustment: yfinance auto_adjust=True total-return ETF closes and adjusted units; ^IRX raw yield closes
Survivorship bias: The universe is the fixed SPY, IEF and SHY list. The README identifies fund survivorship and asset-class selection bias because the ETFs were chosen retrospectively for continued existence and long histories.
Parameter combinations tried: 9
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.05834116935251976, ann_vol=0.07885680883659202, sharpe=0.7586290069006257, sharpe_excess=0.577619545189694, max_drawdown=-0.12701683975134226, total_return=0.7618289487480361, turnover=2.388472031191391, n_trades=360
    oos: cagr=0.05498176286633871, ann_vol=0.07758606485723214, sharpe=0.7287989021998976, sharpe_excess=0.4701212970869022, max_drawdown=-0.13418022323781786, total_return=0.7994551340544738, turnover=2.0866084824682547, n_trades=396
    full: cagr=0.05658096575636318, ann_vol=0.07818672040431657, sharpe=0.7431772376620975, sharpe_excess=0.5218106150362875, max_drawdown=-0.13418022323781764, total_return=2.1703321471504218, turnover=2.2304264934726334, n_trades=756
    benchmark: cagr=0.10183451851207681, ann_vol=0.174631639522508, sharpe=0.6428259374093233, sharpe_excess=0.5436833823300142, max_drawdown=-0.5007710860670267, total_return=6.637297287426039, turnover=0.04271975294565473, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "504_exploratory_mean_variance_with_jumps_an"\nuniverse: [SPY, IEF, SHY]\nbenchmark: SPY\nexcess_rate_ticker: "^IRX"\nfred_series: DFF\n\ndata_start: 2003-01-01\nbacktest_start: 2005-01-01\nis_end: 2014-12-31\nbacktest_end: 2025-12-31\nexecution_lag_sessions: 1\n# Minimum cash fraction at sizing prices before execution costs.\ncash_buffer_fraction: 0.10\nfred_lag_calendar_days: 7\n\nmodel:\n  lookback_sessions: 252\n  jump_z: 3.0\n  downside_jump_weight: 1.0\n  risk_aversion: 5.0\n  entropy_temperature: 0.01\n  grid_divisions: 20\n  commission_bps: 1.0\n  slippage_bps: 2.0\n\nrobustness:\n  lookback_shorter: {lookback_sessions: 189}\n  lookback_longer: {lookback_sessions: 315}\n  jump_threshold_lower: {jump_z: 2.5}\n  jump_threshold_higher: {jump_z: 3.5}\n  temperature_lower: {entropy_temperature: 0.005}\n  temperature_higher: {entropy_temperature: 0.02}\n  no_extra_jump_penalty: {downside_jump_weight: 0.0}\n  double_costs: {commission_bps: 2.0, slippage_bps: 4.0}\n'
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
import json
import signal
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from io import StringIO
from itertools import combinations
from math import comb
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
    USFederalHolidayCalendar,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from scipy.special import softmax


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SESSIONS_PER_YEAR = 252
PRICE_ADJUSTMENT = "yfinance auto_adjust=True total-return ETF closes and adjusted units; ^IRX raw yield closes"


class NYSEHolidays(AbstractHolidayCalendar):
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


# Full-session closures in the supported calendar range beyond the recurring holidays.
NYSE_SPECIAL_CLOSURES = pd.DatetimeIndex([
    "2004-06-11",  # Reagan day of mourning
    "2007-01-02",  # Ford day of mourning
    "2012-10-29", "2012-10-30",  # Hurricane Sandy
    "2018-12-05",  # Bush day of mourning
    "2025-01-09",  # Carter day of mourning
])


@dataclass(frozen=True)
class Rules:
    lookback_sessions: int
    jump_z: float
    downside_jump_weight: float
    risk_aversion: float
    entropy_temperature: float
    grid_divisions: int
    commission_bps: float
    slippage_bps: float

    def __post_init__(self):
        if type(self.lookback_sessions) is not int or self.lookback_sessions < 2:
            raise ValueError("lookback_sessions must be an integer of at least two.")
        if type(self.grid_divisions) is not int or self.grid_divisions < 1:
            raise ValueError("grid_divisions must be a positive integer.")
        positive = (self.jump_z, self.risk_aversion, self.entropy_temperature)
        nonnegative = (self.downside_jump_weight, self.commission_bps, self.slippage_bps)
        if not np.isfinite(positive + nonnegative).all() or min(positive) <= 0 or min(nonnegative) < 0:
            raise ValueError("Check jump, risk aversion, entropy, and friction settings.")
        if self.commission_bps + self.slippage_bps >= 10_000:
            raise ValueError("Combined friction must be below 100% of ETF notional.")

    @property
    def friction(self):
        return (self.commission_bps + self.slippage_bps) / 10_000


def cached_frame(key, fetch):
    path = CACHE_DIR / f"{key}.csv"
    if not path.is_file():
        frame = fetch()
        fetched_at = datetime.now(timezone.utc)
        with NamedTemporaryFile(mode="w", dir=CACHE_DIR, encoding="utf-8", delete=False) as target:
            target_path = Path(target.name)
            try:
                target.write(f"{fetched_at.isoformat()}\n")
                frame.to_csv(target, index_label="date")
                target.close()
                target_path.replace(path)
            finally:
                target_path.unlink(missing_ok=True)
    with path.open(encoding="utf-8") as source:
        fetched_at = datetime.fromisoformat(source.readline().strip())
        frame = pd.read_csv(source, index_col=0, parse_dates=True)
    return frame, fetched_at


def validate_quotes(frame, symbols, start, end, *, adjusted, expected_sessions=None):
    if frame.empty or list(frame.columns) != list(symbols):
        raise RuntimeError(f"Missing requested data for {symbols}.")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise RuntimeError(f"Unsorted or duplicate dates for {symbols}.")
    week = pd.Timedelta(days=7)
    if frame.index[0] > start + week or frame.index[-1] < end - week:
        raise RuntimeError(f"Incomplete requested range for {symbols}.")
    if not np.isfinite(frame.to_numpy()).all() or (adjusted and frame.le(0).any().any()):
        raise RuntimeError(f"Invalid close data for {symbols}.")
    if expected_sessions is not None:
        validate_etf_sessions(frame, expected_sessions)


def exchange_schedule(start, end):
    if start.year < 2003 or end.year > 2025:
        raise ValueError("The NYSE closure calendar covers 2003 through 2025.")
    calendar_end = end + pd.offsets.MonthEnd(0)
    holidays = NYSEHolidays().holidays(start, calendar_end)
    scheduled = pd.bdate_range(start, calendar_end).difference(holidays)
    # A cancelled month-end session cannot cause a signal on an earlier close.
    month_ends = scheduled[~scheduled.to_period("M").duplicated(keep="last")]
    sessions = scheduled[scheduled <= end].difference(NYSE_SPECIAL_CLOSURES)
    return sessions, month_ends.intersection(sessions)


def validate_etf_sessions(closes, sessions):
    missing = sessions.difference(closes.index)
    unexpected = closes.index.difference(sessions)
    if len(missing) or len(unexpected):
        raise RuntimeError(
            f"ETF dates differ from the NYSE calendar. Missing ({len(missing)}): "
            f"{missing.strftime('%Y-%m-%d')[:5].tolist()}; unexpected ({len(unexpected)}): "
            f"{unexpected.strftime('%Y-%m-%d')[:5].tolist()}."
        )


def yahoo_closes(symbols, start, end, *, adjusted, expected_sessions=None):
    def fetch():
        raw = yf.download(
            list(symbols), start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=adjusted, ignore_tz=True, progress=False,
            threads=False, timeout=15,
        )
        if raw is None or raw.empty:
            raise RuntimeError(f"Yahoo Finance returned an empty download for {symbols}.")
        closes = raw["Close"].reindex(columns=symbols).dropna(how="all")
        validate_quotes(closes, symbols, start, end, adjusted=adjusted, expected_sessions=expected_sessions)
        return closes

    ticker_key = "_".join(symbol.replace("^", "index_") for symbol in symbols)
    basis = "adjusted" if adjusted else "raw"
    key = f"yahoo_{basis}_{ticker_key}_{start.date()}_{end.date()}"
    closes, fetched_at = cached_frame(key, fetch)
    validate_quotes(closes, symbols, start, end, adjusted=adjusted, expected_sessions=expected_sessions)
    return closes, fetched_at


def fred_rates(series, start, end):
    def fetch():
        response = requests.get(
            FRED_CSV, params={"id": series, "cosd": str(start.date()), "coed": str(end.date())},
            timeout=(10, 25),
        )
        response.raise_for_status()
        raw = pd.read_csv(StringIO(response.text), parse_dates=["observation_date"])
        raw[series] = pd.to_numeric(raw[series], errors="raise")
        frame = raw.set_index("observation_date")[[series]].dropna()
        validate_quotes(frame, [series], start, end, adjusted=False)
        return frame

    frame, fetched_at = cached_frame(f"fred_{series}_{start.date()}_{end.date()}", fetch)
    validate_quotes(frame, [series], start, end, adjusted=False)
    return frame[series], fetched_at


def last_available(series, cutoffs):
    quotes = series.reindex(cutoffs, method="ffill", tolerance=pd.Timedelta(days=7))
    if quotes.isna().any():
        raise RuntimeError(f"No recent {series.name} observation for {quotes.index[quotes.isna()][0].date()}.")
    return quotes.to_numpy()


def build_market(closes, rates, bill_yields, publication_lag):
    dates = closes.index
    intervals = dates.to_series().diff().dt.days
    observations = rates.index.to_numpy(dtype="datetime64[D]")
    holidays = USFederalHolidayCalendar().holidays(
        start=rates.index[0], end=rates.index[-1] + pd.Timedelta(days=14),
    ).to_numpy(dtype="datetime64[D]")
    next_business_day = np.busday_offset(observations, 1, roll="backward", holidays=holidays)
    available_on = np.maximum(next_business_day, observations + np.timedelta64(publication_lag, "D"))
    released_rates = pd.Series(rates.to_numpy(), index=pd.DatetimeIndex(available_on), name=rates.name)
    released_rates = released_rates.groupby(level=0).last()
    annual_cash_rate = pd.Series(
        last_available(released_rates, dates) / 100,
        index=dates,
    )
    cash_return = annual_cash_rate.shift(1) * intervals / 365.25
    # Use a quote dated before the ETF close that starts each return interval.
    bill_discount = pd.Series(last_available(bill_yields, dates[:-2]) / 100, index=dates[2:])
    bill_price = 1 - bill_discount * 91 / 360
    bill_return = np.expm1(-np.log(bill_price) * intervals / 91)
    return closes.pct_change(fill_method=None), annual_cash_rate, cash_return, bill_return


def simplex_grid(n_etfs, divisions):
    # Stars and bars enumerates a grid with cash as the last, residual weight.
    if comb(divisions + n_etfs, n_etfs) > 100_000:
        raise ValueError("Reduce grid_divisions for this universe.")
    separators = np.array(list(combinations(range(divisions + n_etfs), n_etfs)))
    return (np.diff(separators, axis=1, prepend=-1) - 1) / divisions


def target_weights(window, cash_rate, grid, rules):
    mean = window.mean(axis=0)
    centered = window - mean
    covariance = centered.T @ centered / (len(window) - 1)
    cutoff = rules.jump_z * np.sqrt(np.diag(covariance))
    lower_jumps = np.where(centered < -cutoff, centered, 0.0)
    jump_moment = lower_jumps.T @ lower_jumps / len(window)
    annual_risk = SESSIONS_PER_YEAR * (covariance + rules.downside_jump_weight * jump_moment)
    excess_mean = SESSIONS_PER_YEAR * mean - cash_rate
    variance = ((grid @ annual_risk) * grid).sum(axis=1)
    utility = grid @ excess_mean - 0.5 * rules.risk_aversion * variance
    etf_weights = softmax(utility / rules.entropy_temperature) @ grid
    cash_weight = max(0.0, 1.0 - etf_weights.sum())
    return np.append(etf_weights, cash_weight)


def closing_targets(etf_returns, annual_cash_rate, month_ends, start, execution_lag, rules):
    grid = simplex_grid(etf_returns.shape[1], rules.grid_divisions)
    dates = etf_returns.index
    targets = {}
    for row in np.flatnonzero(dates.isin(month_ends)):
        execution_row = row + execution_lag
        if execution_row >= len(dates) or dates[execution_row] < start:
            continue
        first_row = row - rules.lookback_sessions + 1
        if first_row < 1:
            raise ValueError("The first signal needs more warm-up history.")
        window = etf_returns.iloc[first_row:row + 1].to_numpy()
        targets[dates[execution_row]] = target_weights(window, annual_cash_rate.iloc[row], grid, rules)
    return targets


def size_trade(positions, previous_closes, previous_value, target, cash_buffer):
    target_values = previous_value * target[:-1]
    etf_value = target_values.sum()
    etf_limit = previous_value * (1 - cash_buffer)
    if etf_value > etf_limit:
        target_values *= etf_limit / etf_value
    return target_values / previous_closes - positions


def simulate(closes, cash_return, bill_return, start, targets, friction, cash_buffer):
    start_row = closes.index.searchsorted(start)
    if not 0 < start_row < len(closes):
        raise ValueError("The evaluation period requires a prior close and at least one session.")
    dates = closes.index[start_row:]
    prices = closes.to_numpy()
    cash_factors = 1 + cash_return.reindex(dates).to_numpy()
    bill_returns = bill_return.reindex(dates)
    if (not np.isfinite(cash_factors).all() or (cash_factors <= 0).any()
            or not np.isfinite(bill_returns).all()):
        raise RuntimeError("Missing or invalid cash proxy return in the evaluation period.")

    positions = np.zeros(closes.shape[1])
    cash = previous_value = 1.0
    net_returns = np.empty(len(dates))
    turnover = np.zeros(len(dates))
    n_trades = np.zeros(len(dates), dtype=int)
    for row, date in enumerate(dates):
        target = targets.get(date)
        trade_units = None
        if target is not None:
            trade_units = size_trade(positions, prices[start_row + row - 1], previous_value,
                                     target, cash_buffer)

        cash *= cash_factors[row]
        fill_prices = prices[start_row + row]
        before_trade = cash + positions @ fill_prices
        value = before_trade
        if trade_units is not None:
            trade_values = trade_units * fill_prices
            traded = np.abs(trade_values).sum()
            fees = friction * traded
            cash_after_fill = cash - trade_values.sum() - fees
            if cash_after_fill < 0:
                raise RuntimeError(f"Fixed quantities exceed available cash on {date.date()} after costs.")
            cash = cash_after_fill
            positions += trade_units
            value = before_trade - fees
            turnover[row] = traded / before_trade
            n_trades[row] = np.count_nonzero(trade_units)
        net_returns[row] = value / previous_value - 1
        previous_value = value
    return pd.DataFrame(
        {"return": net_returns, "bill_return": bill_returns,
         "turnover": turnover, "n_trades": n_trades}, index=dates,
    )


def metrics(daily_results):
    returns = daily_results["return"].to_numpy()
    if len(returns) < 2:
        raise ValueError("Each evaluation period requires at least two return observations.")
    excess = returns - daily_results["bill_return"].to_numpy()
    growth = np.r_[1.0, (1 + returns).cumprod()]
    vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    scale = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(scale * vol),
        "sharpe": float(scale * returns.mean() / vol) if vol > 0 else None,
        "sharpe_excess": float(scale * excess.mean() / excess_vol) if excess_vol > 0 else None,
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1)),
        "total_return": float(growth[-1] - 1),
        "turnover": float(SESSIONS_PER_YEAR * daily_results["turnover"].mean()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def print_json(label, payload):
    print(f"{label}: {json.dumps(payload, allow_nan=False)}", flush=True)


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    data_start, start, split, end = map(pd.Timestamp, (
        config["data_start"], config["backtest_start"], config["is_end"], config["backtest_end"],
    ))
    if not data_start < start < split < end:
        raise ValueError("Check the warm-up, evaluation, and sample split dates.")
    universe = config["universe"]
    if not universe or len(set(universe)) != len(universe) or config["benchmark"] not in universe:
        raise ValueError("The universe needs distinct ETFs and must include the benchmark.")
    if any(not symbol.isascii() or not symbol.isalpha() or not symbol.isupper() for symbol in universe):
        raise ValueError("Use US ETF ticker symbols in the universe.")
    if config["excess_rate_ticker"] != "^IRX" or config["fred_series"] != "DFF":
        raise ValueError("The rate conversions use ^IRX and the DFF series.")
    lag = config["execution_lag_sessions"]
    publication_lag = config["fred_lag_calendar_days"]
    cash_buffer = config["cash_buffer_fraction"]
    if type(lag) is not int or lag < 1 or type(publication_lag) is not int or publication_lag < 0:
        raise ValueError("Execution lag must be at least one session and FRED lag must be nonnegative.")
    if not 0 < cash_buffer < 1:
        raise ValueError("cash_buffer_fraction must be strictly between zero and one.")
    sessions, month_ends = exchange_schedule(data_start, end)
    rules = Rules(**config["model"])
    seen = {rules}
    cases = {}
    for label, overrides in config["robustness"].items():
        variant = replace(rules, **overrides)
        if variant not in seen:
            cases[label] = variant
            seen.add(variant)

    CACHE_DIR.mkdir(exist_ok=True)
    bill_ticker = config["excess_rate_ticker"]
    with TemporaryDirectory(prefix="yfinance_", dir=CACHE_DIR) as source_cache:
        yf.set_tz_cache_location(source_cache)
        closes, price_time = yahoo_closes(universe, data_start, end, adjusted=True, expected_sessions=sessions)
        bill_quotes, bill_time = yahoo_closes([bill_ticker], data_start, end, adjusted=False)
    fred_start = data_start - pd.Timedelta(days=publication_lag + 7)
    rates, rate_time = fred_rates(config["fred_series"], fred_start, end)
    print(f"DATA_FETCHED_AT: {max(price_time, bill_time, rate_time).isoformat()}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")

    is_closes = closes.loc[:split]
    market = build_market(is_closes, rates.loc[:split], bill_quotes.loc[:split, bill_ticker], publication_lag)
    etf_returns, annual_cash_rate, cash_return, bill_return = market
    print("ROBUSTNESS (IS only, fixed alternatives):", flush=True)
    for label, variant in cases.items():
        targets = closing_targets(etf_returns, annual_cash_rate, month_ends, start, lag, variant)
        daily_results = simulate(is_closes, cash_return, bill_return, start, targets, variant.friction, cash_buffer)
        print_json(f"ROBUSTNESS {label}", metrics(daily_results))
    print(f"PARAM_COMBOS_TRIED: {1 + len(cases)}", flush=True)

    market = build_market(closes, rates, bill_quotes[bill_ticker], publication_lag)
    etf_returns, annual_cash_rate, cash_return, bill_return = market
    targets = closing_targets(etf_returns, annual_cash_rate, month_ends, start, lag, rules)
    full_results = simulate(closes, cash_return, bill_return, start, targets, rules.friction, cash_buffer)
    benchmark_target = np.zeros(len(universe) + 1)
    benchmark_target[universe.index(config["benchmark"])] = 1.0
    benchmark_entry = min(targets, default=full_results.index[0])
    benchmark = simulate(closes, cash_return, bill_return, start,
                         {benchmark_entry: benchmark_target}, rules.friction, cash_buffer)

    results = {
        "strategy": config["strategy"],
        "data_range": {"start": full_results.index[0].date().isoformat(),
                       "end": full_results.index[-1].date().isoformat()},
    }
    for period, selected in (("is", full_results.index <= split),
                             ("oos", full_results.index > split), ("full", slice(None))):
        results[period] = metrics(full_results.loc[selected])
        benchmark_metrics = metrics(benchmark.loc[selected])
        print_json(period.upper(), results[period])
        print_json(f"BENCHMARK_{period.upper()}", benchmark_metrics)
    results["benchmark"] = {"name": config["benchmark"], **benchmark_metrics}
    print_json("RESULTS_JSON", results)


if __name__ == "__main__":
    signal.alarm(540)
    main()
