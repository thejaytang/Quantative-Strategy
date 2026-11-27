# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "504_ exploratory_mean_variance_with_jumps_an"\nuniverse: [SPY, IEF, SHY]\nbenchmark: SPY\nexcess_rate_ticker: "^IRX"\nfred_series: DFF\n\ndata_start: 2003-01-01\nbacktest_start: 2005-01-01\nis_end: 2014-12-31\nbacktest_end: 2025-12-31\nexecution_lag_sessions: 1\nfred_lag_calendar_days: 7\n\nmodel:\n  lookback_sessions: 252\n  jump_z: 3.0\n  downside_jump_weight: 1.0\n  risk_aversion: 5.0\n  entropy_temperature: 0.01\n  grid_divisions: 20\n  commission_bps: 1.0\n  slippage_bps: 2.0\n\nrobustness:\n  lookback_shorter: {lookback_sessions: 189}\n  lookback_longer: {lookback_sessions: 315}\n  jump_threshold_lower: {jump_z: 2.5}\n  jump_threshold_higher: {jump_z: 3.5}\n  temperature_lower: {entropy_temperature: 0.005}\n  temperature_higher: {entropy_temperature: 0.02}\n  no_extra_jump_penalty: {downside_jump_weight: 0.0}\n  double_costs: {commission_bps: 2.0, slippage_bps: 4.0}\n'
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
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.optimize import brentq
from scipy.special import softmax


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SESSIONS_PER_YEAR = 252
PRICE_ADJUSTMENT = "yfinance auto_adjust=True total-return ETF closes; ^IRX raw yield closes"


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


def validate_quotes(frame, symbols, start, end, *, adjusted):
    if frame.empty or list(frame.columns) != list(symbols):
        raise RuntimeError(f"Missing requested data for {symbols}.")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise RuntimeError(f"Unsorted or duplicate dates for {symbols}.")
    week = pd.Timedelta(days=7)
    if frame.index[0] > start + week or frame.index[-1] < end - week:
        raise RuntimeError(f"Incomplete requested range for {symbols}.")
    if not np.isfinite(frame.to_numpy()).all() or (adjusted and frame.le(0).any().any()):
        raise RuntimeError(f"Invalid close data for {symbols}.")


def yahoo_closes(symbols, start, end, *, adjusted):
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
        validate_quotes(closes, symbols, start, end, adjusted=adjusted)
        return closes

    ticker_key = "_".join(symbol.replace("^", "index_") for symbol in symbols)
    basis = "adjusted" if adjusted else "raw"
    key = f"yahoo_{basis}_{ticker_key}_{start.date()}_{end.date()}"
    closes, fetched_at = cached_frame(key, fetch)
    validate_quotes(closes, symbols, start, end, adjusted=adjusted)
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
    bill_discount = pd.Series(last_available(bill_yields, dates) / 100, index=dates)
    bill_price = 1 - bill_discount.shift(1) * 91 / 360
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


def closing_targets(etf_returns, annual_cash_rate, start, execution_lag, rules):
    grid = simplex_grid(etf_returns.shape[1], rules.grid_divisions)
    dates = etf_returns.index
    months = dates.to_period("M")
    targets = {}
    for row in np.flatnonzero(months[:-1] != months[1:]):
        execution_row = row + execution_lag
        if execution_row >= len(dates) or dates[execution_row] < start:
            continue
        first_row = row - rules.lookback_sessions + 1
        if first_row < 1:
            raise ValueError("The first signal needs more warm-up history.")
        window = etf_returns.iloc[first_row:row + 1].to_numpy()
        targets[dates[execution_row]] = target_weights(window, annual_cash_rate.iloc[row], grid, rules)
    return targets


def simulate(etf_returns, cash_return, bill_return, start, targets, friction):
    returns = etf_returns.loc[start:].assign(cash=cash_return)
    gross_factors = 1 + returns.to_numpy()
    if not np.isfinite(gross_factors).all() or (gross_factors <= 0).any():
        raise RuntimeError("Missing or invalid return in the evaluation period.")
    positions = np.zeros(returns.shape[1])
    positions[-1] = 1.0
    net_returns = np.empty(len(returns))
    turnover = np.zeros(len(returns))
    n_trades = np.zeros(len(returns), dtype=int)
    for row, (date, factors) in enumerate(zip(returns.index, gross_factors)):
        grown = positions * factors
        gross = grown.sum()
        positions = grown / gross
        target = targets.get(date)
        if target is not None:
            before = positions[:-1]
            target_etfs = target[:-1]
            # Solve for the value left after paying both charges on actual ETF fills.
            remaining = 1.0
            if friction > 0:
                remaining = brentq(
                    lambda value: value + friction * np.abs(value * target_etfs - before).sum() - 1,
                    0, 1, xtol=1e-14,
                )
            traded = np.abs(remaining * target_etfs - before)
            turnover[row] = traded.sum()
            n_trades[row] = np.count_nonzero(traded > 1e-10)
            gross *= remaining
            positions = target
        net_returns[row] = gross - 1
    return pd.DataFrame(
        {"return": net_returns, "bill_return": bill_return.reindex(returns.index),
         "turnover": turnover, "n_trades": n_trades}, index=returns.index,
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
    if type(lag) is not int or lag < 1 or type(publication_lag) is not int or publication_lag < 0:
        raise ValueError("Execution lag must be at least one session and FRED lag must be nonnegative.")
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
        closes, price_time = yahoo_closes(universe, data_start, end, adjusted=True)
        bill_quotes, bill_time = yahoo_closes([bill_ticker], data_start, end, adjusted=False)
    fred_start = data_start - pd.Timedelta(days=publication_lag + 7)
    rates, rate_time = fred_rates(config["fred_series"], fred_start, end)
    print(f"DATA_FETCHED_AT: {max(price_time, bill_time, rate_time).isoformat()}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")

    market = build_market(closes.loc[:split], rates.loc[:split], bill_quotes.loc[:split, bill_ticker], publication_lag)
    etf_returns, annual_cash_rate, cash_return, bill_return = market
    print("ROBUSTNESS (IS only, fixed alternatives):", flush=True)
    for label, variant in cases.items():
        targets = closing_targets(etf_returns, annual_cash_rate, start, lag, variant)
        daily_results = simulate(etf_returns, cash_return, bill_return, start, targets, variant.friction)
        print_json(f"ROBUSTNESS {label}", metrics(daily_results))
    print(f"PARAM_COMBOS_TRIED: {1 + len(cases)}", flush=True)

    market = build_market(closes, rates, bill_quotes[bill_ticker], publication_lag)
    etf_returns, annual_cash_rate, cash_return, bill_return = market
    targets = closing_targets(etf_returns, annual_cash_rate, start, lag, rules)
    if not targets:
        raise ValueError("The backtest requires at least one scheduled rebalance.")
    official = simulate(etf_returns, cash_return, bill_return, start, targets, rules.friction)
    benchmark_target = np.zeros(len(universe) + 1)
    benchmark_target[universe.index(config["benchmark"])] = 1.0
    benchmark = simulate(etf_returns, cash_return, bill_return, start,
                         {min(targets): benchmark_target}, rules.friction)

    results = {
        "strategy": config["strategy"],
        "data_range": {"start": official.index[0].date().isoformat(),
                       "end": official.index[-1].date().isoformat()},
    }
    for period, selected in (("is", official.index <= split), ("oos", official.index > split), ("full", slice(None))):
        results[period] = metrics(official.loc[selected])
        benchmark_metrics = metrics(benchmark.loc[selected])
        print_json(period.upper(), results[period])
        print_json(f"BENCHMARK_{period.upper()}", benchmark_metrics)
    results["benchmark"] = {"name": config["benchmark"], **benchmark_metrics}
    print_json("RESULTS_JSON", results)


if __name__ == "__main__":
    signal.alarm(840)
    main()
