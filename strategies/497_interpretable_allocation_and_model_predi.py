# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "497_interpretable_allocation_and_model_predi"\ndates:\n  warmup_start: 2012-12-01\n  start: 2014-01-03\n  is_end: 2024-12-31\n  end: 2026-09-04\n\nbenchmark: SPY\nexcess_reference: BIL\n\n# Fixed large-cap research universe. Rank liquidity again at each signal.\nuniverse: [\n  AAPL, ABBV, ABT, ACN, ADBE, ADI, ADP, AFL, AIG, AMAT, AMD, AMGN, AMT, AMZN,\n  ANET, APD, AVGO, AXP, AZO, BA, BAC, BAX, BDX, BK, BKNG, BLK, BMY, BRK-B, BSX,\n  C, CAT, CB, CCI, CDNS, CL, CMCSA, CME, CMG, CMI, COP, COST, CRM, CSCO, CSX,\n  CTAS, CVS, CVX, DE, DHR, DIS, DUK, DVN, ECL, EMR, EOG, EQIX, ETN, EXC,\n  F, FDX, FICO, GD, GE, GILD, GIS, GM, GOOG, GS, HD, HON, IBM, INTC, INTU,\n  ISRG, JCI, JNJ, JPM, KMB, KO, KR, LIN, LLY, LMT, LOW, LVS, MA, MAR, MCD,\n  MCK, MDT, MET, MMM, MO, MRK, MS, MSFT, MU, NEE, NFLX, NKE, NOC, NOW, NSC,\n  NVDA, O, ORCL, OXY, PEP, PFE, PG, PGR, PH, PLD, PM, PNC, PYPL, QCOM,\n  REGN, ROST, RSG, RTX, SBUX, SCHW, SHW, SLB, SO, SPGI, SYK, T, TGT, TJX,\n  TMUS, TRV, TSCO, TXN, UBER, UNH, UNP, UPS, USB, V, VLO, VRTX, VZ, WFC,\n  WM, WMT, XEL, XOM, ZTS\n]\n\nsymbol_changes:\n  BK:\n    to: BNY\n    effective: 2026-05-21\n\nexecution:\n  cost_bps: 5.0\n  slippage_bps: 3.0\n  signal_lag_sessions: 1\n  cash_buffer_bps: 10.0\n  max_missing_sessions: 5\n\nparameters:\n  regression_days: 252\n  regression_half_life_days: 63\n  market_growth_span_days: 252\n  market_fast_span_days: 21\n  market_slow_span_days: 63\n  market_state_gain: 21.23\n  beta_floor: 0.10\n  liquidity_days: 63\n  liquid_names: 100\n  selection_fraction: 0.20\n  risk_days: 126\n  annual_volatility_cap: 0.173\n  weight_cap: 0.10\n  drawdown_stop: 0.08\n\nrobustness:\n  half_life_days: [50, 75]\n  selection_fractions: [0.15, 0.25]\n  higher_slippage_bps: 10.0\n'
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
"""Research backtest for a liquidity-screened trend and market-beta strategy."""

import hashlib
import json
import math
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
YAHOO_FIELDS = ["Open", "Close", "Volume", "Dividends", "Capital Gains"]


@dataclass
class MarketData:
    dates: pd.DatetimeIndex
    tickers: list[str]
    stock_open: np.ndarray
    stock_close: np.ndarray
    stock_signal: np.ndarray
    stock_distributions: np.ndarray
    dollar_volume: np.ndarray
    market_close: pd.Series
    market_signal: pd.Series
    market_open: pd.Series
    cash_to_open: pd.Series
    cash_to_close: pd.Series


def write_cache(path, write):
    with NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as file:
        temporary_path = Path(file.name)
    try:
        write(temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def download_prices(tickers, start, end):
    def fetch(symbols):
        frame = yf.download(
            symbols, start=start, end=end, auto_adjust=False, progress=False,
            actions=True, keepna=True, timeout=12, threads=8, ignore_tz=True,
        )
        columns = pd.MultiIndex.from_product([YAHOO_FIELDS, symbols])
        return frame.reindex(columns=columns)

    with TemporaryDirectory(dir=CACHE_DIR, prefix=".yahoo-") as yahoo_cache:
        yf.set_tz_cache_location(yahoo_cache)
        batches = [fetch(tickers[i:i + 24]) for i in range(0, len(tickers), 24)]
        prices = pd.concat(batches, axis=1).sort_index()
        missing = prices["Close"].columns[prices["Close"].isna().all()]
        if len(missing):
            retry = fetch(missing.tolist())
            prices = pd.concat([prices.drop(columns=retry.columns), retry], axis=1)
    return prices.sort_index()


def load_sources(config):
    dates = config["dates"]
    start = str(dates["warmup_start"])
    end = str((pd.Timestamp(dates["end"]) + pd.Timedelta(days=1)).date())
    symbols = {change["to"]: old for old, change in config["symbol_changes"].items()}
    sources = {old: new for new, old in symbols.items()}
    research_names = [*config["universe"], config["benchmark"], config["excess_reference"]]
    tickers = sorted(sources.get(name, name) for name in research_names)
    if len(set(tickers)) != len(tickers):
        raise ValueError("The universe and reference symbols must be distinct.")
    cache_key = hashlib.sha256(" ".join(tickers).encode()).hexdigest()[:12]

    CACHE_DIR.mkdir(exist_ok=True)
    price_path = CACHE_DIR / f"yahoo_actions_{start}_{end}_{cache_key}.csv.gz"
    if not price_path.is_file():
        prices = download_prices(tickers, start, end)
        absent = prices["Close"].columns[prices["Close"].isna().all()]
        if len(absent):
            raise ValueError(f"Yahoo Finance returned no prices for {', '.join(absent)}.")
        write_cache(price_path, lambda path: prices.to_csv(path))

    prices = pd.read_csv(price_path, header=[0, 1], index_col=0, parse_dates=True)
    fetched_at = datetime.fromtimestamp(
        price_path.stat().st_mtime, tz=timezone.utc
    ).isoformat(timespec="seconds")
    source_hash = hashlib.sha256(price_path.read_bytes()).hexdigest()
    return prices.rename(columns=symbols, level=1), fetched_at, source_hash


def prepare_market(prices, config):
    market = config["benchmark"]
    reference = config["excess_reference"]
    tickers = sorted(config["universe"])
    closes = prices["Close"]
    closes = closes.where(closes.gt(0) & np.isfinite(closes))
    opens = prices["Open"]
    opens = opens.where(opens.gt(0) & np.isfinite(opens))
    dates = prices.index
    if dates[-1] < pd.Timestamp(config["dates"]["end"]) - pd.Timedelta(days=3):
        raise ValueError("Yahoo Finance data end before the configured sample.")
    if opens[reference].isna().any() or closes[reference].isna().any():
        raise ValueError("The cash proxy has missing bars on the Yahoo Finance session index.")
    distributions = prices["Dividends"].fillna(0) + prices["Capital Gains"].fillna(0)
    if not (np.isfinite(distributions) & distributions.ge(0)).all().all():
        raise ValueError("Yahoo Finance returned an invalid cash distribution.")

    dollar_volume = closes[tickers] * prices["Volume"][tickers]
    dollar_volume = dollar_volume.where(dollar_volume.gt(0) & np.isfinite(dollar_volume))
    # Yahoo Open, Close and cash distributions use the same split-adjusted units.
    marks = closes.fillna(opens).ffill(limit=config["execution"]["max_missing_sessions"])
    if marks[market].isna().any():
        raise ValueError("The benchmark has no usable mark within the missing-price limit.")
    total_return = marks * (1 + (distributions / marks).fillna(0)).cumprod()
    stock_signal = total_return[tickers].where(closes[tickers].notna())
    cash_to_open = (opens[reference] + distributions[reference]) / closes[reference].shift()
    cash_to_close = closes[reference] / opens[reference]
    return MarketData(
        dates, tickers, opens[tickers].to_numpy(), closes[tickers].to_numpy(), stock_signal.to_numpy(),
        distributions[tickers].to_numpy(), dollar_volume.to_numpy(), marks[market],
        total_return[market], opens[market], cash_to_open, cash_to_close,
    )


class Allocator:
    def __init__(self, data, parameters, last):
        self.parameters = parameters
        self.log_returns = np.diff(np.log(data.stock_signal[:last + 1]), axis=0, prepend=np.nan)
        self.dollar_volume = data.dollar_volume[:last + 1]
        market_close = data.market_signal.iloc[:last + 1]
        market_log = np.log(market_close).diff()
        self.market_log = market_log.to_numpy()
        self.market_growth = (
            market_log.ewm(span=parameters["market_growth_span_days"], adjust=False)
            .mean().to_numpy() * SESSIONS_PER_YEAR
        )
        fast = market_close.ewm(span=parameters["market_fast_span_days"], adjust=False).mean()
        slow = market_close.ewm(span=parameters["market_slow_span_days"], adjust=False).mean()
        self.market_state = (-parameters["market_state_gain"] * (fast / slow - 1)).to_numpy()
        days = parameters["regression_days"]
        ages = np.arange(days - 1, -1, -1)
        decay = np.exp2(-ages / parameters["regression_half_life_days"])
        self.decay = decay / decay.sum()

    def target(self, day):
        p = self.parameters
        weights = np.zeros(self.log_returns.shape[1])
        regression_start = day + 1 - p["regression_days"]
        recent = self.log_returns[regression_start:day + 1]
        volume = self.dollar_volume[day + 1 - p["liquidity_days"]:day + 1]
        eligible = np.flatnonzero(np.isfinite(recent).all(axis=0) & np.isfinite(volume).all(axis=0))
        if not len(eligible):
            return weights
        liquidity = np.median(volume[:, eligible], axis=0)
        liquid = eligible[np.argsort(-liquidity, kind="stable")[:p["liquid_names"]]]
        stock = recent[:, liquid]
        market = self.market_log[regression_start:day + 1]
        mean_market = self.decay @ market
        centered_market = market - mean_market
        weighted_market = self.decay * centered_market
        variance = weighted_market @ centered_market
        if variance <= 0:
            return weights

        beta = (weighted_market @ stock) / variance
        alpha = (self.decay @ stock - beta * mean_market) * SESSIONS_PER_YEAR
        trend = alpha + beta * self.market_growth[day]
        penalty = np.maximum(np.abs(beta), p["beta_floor"]) ** self.market_state[day]
        scores = np.tanh(trend / penalty)
        count = math.ceil(len(liquid) * p["selection_fraction"])
        top = np.argsort(-scores, kind="stable")[:count]
        top = top[scores[top] > 0]
        if not len(top):
            return weights

        selected = liquid[top]
        positive = scores[top]
        sized = np.minimum(positive / positive.sum(), p["weight_cap"])
        risk_window = self.log_returns[day + 1 - p["risk_days"]:day + 1, selected]
        portfolio_returns = np.expm1(risk_window) @ sized
        forecast_vol = np.std(portfolio_returns, ddof=1) * np.sqrt(SESSIONS_PER_YEAR)
        if forecast_vol > p["annual_volatility_cap"]:
            sized *= p["annual_volatility_cap"] / forecast_vol
        weights[selected] = sized
        return weights


def execution_rate(execution):
    return (execution["cost_bps"] + execution["slippage_bps"]) / 10_000


def size_at_close(weights, close, value, reserve):
    desired = np.zeros_like(weights)
    selected = weights > 0
    if selected.any():
        scale = min(1.0, (1 - reserve) / weights.sum())
        desired[selected] = weights[selected] * (value * scale) / close[selected]
    return desired


def execute(positions, cash, opens, desired, rate, ready):
    prices = opens[ready]
    previous = positions[ready]
    planned = desired[ready]
    after_sales = np.minimum(previous, planned)
    buying = np.maximum(planned - previous, 0.0)
    cash += (1 - rate) * ((previous - after_sales) @ prices)
    buy_cost = (1 + rate) * (buying @ prices)
    updated = planned
    if buy_cost > cash:
        # Leave room for floating-point rounding when the cash constraint binds.
        buying *= cash / buy_cost * (1 - 1e-12)
        updated = np.minimum(np.nextafter(after_sales + buying, after_sales), planned)

    # Debit the cost of the actual share change using the same opening quote.
    cash -= (1 + rate) * (np.maximum(updated - previous, 0.0) @ prices)
    if cash < 0:
        raise ArithmeticError("Opening fills exceeded the available cash.")
    positions[ready] = updated
    change = updated - previous
    return cash, float(np.abs(change) @ prices), int(np.count_nonzero(change))


def symbols_on(data, mask, day, symbol_changes):
    date = data.dates[day].date()
    names = [data.tickers[index] for index in np.flatnonzero(mask)]
    return [
        symbol_changes[name]["to"]
        if name in symbol_changes and date >= symbol_changes[name]["effective"] else name
        for name in names
    ]


def simulate(data, first, last, parameters, execution, symbol_changes):
    lag = execution["signal_lag_sessions"]
    max_missing = execution["max_missing_sessions"]
    if first - lag < max(parameters["regression_days"], parameters["liquidity_days"] - 1):
        raise ValueError("The configured warmup does not cover the first decision.")
    if parameters["risk_days"] > parameters["regression_days"]:
        raise ValueError("The risk window must fit inside the regression window.")
    allocator = Allocator(data, parameters, last)
    rate = execution_rate(execution)
    reserve = execution["cash_buffer_bps"] / 10_000 + 2 * rate
    dates = data.dates
    to_open = data.cash_to_open.to_numpy()
    to_close = data.cash_to_close.to_numpy()
    positions = np.zeros(data.stock_open.shape[1])
    last_marks = np.zeros_like(positions)
    stale = np.zeros_like(positions, dtype=int)
    cash = peak = previous_value = 1.0
    signal_day = first - lag
    desired = size_at_close(allocator.target(signal_day), data.stock_close[signal_day], cash, reserve)
    due = first
    exit_pending = False
    deferred = expired = 0
    rows = []

    for day in range(first, last + 1):
        if day > first:
            cash *= to_open[day]
        opening = data.stock_open[day]
        quoted = np.isfinite(opening)
        last_marks[quoted] = opening[quoted]
        # Only positions carried into the ex-date earn the distribution.
        distributions = positions @ data.stock_distributions[day]
        turnover = 0.0
        deals = 0
        stopped_today = False
        if desired is not None and day >= due:
            changed = np.isfinite(desired) & (desired != positions)
            ready = changed & quoted
            if ready.any():
                opening_value = cash + positions @ last_marks
                cash, traded, deals = execute(positions, cash, opening, desired, rate, ready)
                turnover = traded / opening_value
                desired[ready] = np.nan
            outstanding = changed & ~quoted
            if not outstanding.any():
                stopped_today = exit_pending
                desired = None
                exit_pending = False
            else:
                deferred += 1
                if day - due >= max_missing:
                    if exit_pending:
                        names = ", ".join(symbols_on(data, outstanding, day, symbol_changes))
                        raise ValueError(f"Stop exit has no opening quote on {dates[day].date()} for {names}.")
                    expired += int(outstanding.sum())
                    desired = None

        cash = cash * to_close[day] + distributions
        close = data.stock_close[day]
        present = np.isfinite(close)
        last_marks[present] = close[present]
        fresh = present | quoted
        stale[fresh] = 0
        stale[~fresh] += 1
        invested = positions > 0
        unpriceable = invested & (stale > max_missing)
        if unpriceable.any():
            names = ", ".join(symbols_on(data, unpriceable, day, symbol_changes))
            raise ValueError(f"An open position has stale marks on {dates[day].date()} for {names}.")
        value = cash + positions @ last_marks
        rows.append((value / previous_value - 1, turnover, deals))
        previous_value = value
        peak = value if stopped_today else max(peak, value)

        if day + lag > last or stopped_today:
            continue
        if invested.any() and value <= peak * (1 - parameters["drawdown_stop"]) and not exit_pending:
            due = day + lag
            desired = np.zeros_like(positions)
            exit_pending = True
        elif desired is None and day > first and dates[day].month != dates[day - 1].month:
            due = day + lag
            desired = size_at_close(allocator.target(day), close, value, reserve)

    daily = pd.DataFrame(rows, index=dates[first:last + 1], columns=["return", "turnover", "trades"])
    return daily, deferred, expired


def performance(daily, reference):
    returns = daily["return"]
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    excess = returns - reference
    volatility = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    growth = (1 + returns).cumprod()
    peak = growth.cummax().clip(lower=1.0)
    return {
        "cagr": float(growth.iloc[-1] ** (SESSIONS_PER_YEAR / len(daily)) - 1),
        "ann_vol": float(volatility * annualizer) if np.isfinite(volatility) else None,
        "sharpe": float(returns.mean() / volatility * annualizer) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * annualizer) if excess_vol > 0 else None,
        "max_drawdown": float((growth / peak - 1).min()),
        "total_return": float(growth.iloc[-1] - 1),
        "turnover": float(daily["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily["trades"].sum()),
    }


def benchmark_path(data, first, last, cash_reference, execution):
    opening = data.market_open.iloc[first]
    if not np.isfinite(opening):
        raise ValueError("The benchmark has no opening quote on the first fill date.")
    buffer = execution["cash_buffer_bps"] / 10_000
    invested = (1 - buffer) / (1 + execution_rate(execution))
    close_series = data.market_signal.iloc[first:last + 1]
    growth = close_series / close_series.iloc[0] * (data.market_close.iloc[first] / opening)
    value = invested * growth + buffer * (1 + cash_reference).cumprod()
    returns = value.pct_change().fillna(value.iloc[0] - 1)
    daily = pd.DataFrame({"return": returns, "turnover": 0.0, "trades": 0})
    daily.iloc[0, 1:] = [invested, 1]
    return daily


def run_robustness(data, first, split, parameters, execution, settings, reference, symbol_changes):
    cases = []
    for half_life in settings["half_life_days"]:
        cases.append((f"half_life={half_life}", dict(parameters, regression_half_life_days=half_life), execution))
    for fraction in settings["selection_fractions"]:
        cases.append((f"top_fraction={fraction}", dict(parameters, selection_fraction=fraction), execution))
    slippage = settings["higher_slippage_bps"]
    cases.append((f"slippage_bps={slippage}", parameters, dict(execution, slippage_bps=slippage)))

    print("ROBUSTNESS — IS only")
    for label, case_parameters, case_execution in cases:
        daily, _, _ = simulate(data, first, split, case_parameters, case_execution, symbol_changes)
        print(f"  {label} {json.dumps(performance(daily, reference.loc[daily.index]), allow_nan=False)}")
    return len(cases)


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    execution = config["execution"]
    parameters = config["parameters"]
    buffer = execution["cash_buffer_bps"] / 10_000
    worst_rate = (execution["cost_bps"] + config["robustness"]["higher_slippage_bps"]) / 10_000
    if (
        execution["signal_lag_sessions"] < 1
        or execution["max_missing_sessions"] < 1
        or not 0 < buffer < 1 - 2 * max(execution_rate(execution), worst_rate)
        or min(execution["cost_bps"], execution["slippage_bps"], config["robustness"]["higher_slippage_bps"]) < 0
        or not 0 < parameters["selection_fraction"] <= 1
        or not 0 < parameters["weight_cap"] <= 1
        or not 0 < parameters["drawdown_stop"] < 1
    ):
        raise ValueError("Check the execution lag, missing-price limit, cash buffer, costs and signal parameters.")

    prices, fetched_at, source_hash = load_sources(config)
    data = prepare_market(prices, config)
    dates = config["dates"]
    first = int(data.dates.searchsorted(pd.Timestamp(dates["start"])))
    split = int(data.dates.searchsorted(pd.Timestamp(dates["is_end"]), side="right")) - 1
    last = int(data.dates.searchsorted(pd.Timestamp(dates["end"]), side="right")) - 1
    if not 0 < first < split < last - 1:
        raise ValueError("The source data do not cover both configured periods.")

    reference = (data.cash_to_open * data.cash_to_close).iloc[first:last + 1] - 1
    reference.iloc[0] = data.cash_to_close.iloc[first] - 1
    data_range = {"start": str(data.dates[first].date()), "end": str(data.dates[last].date())}
    print(f"Date Range: {data_range['start']} to {data_range['end']}")
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"DATA_SNAPSHOT_SHA256: {source_hash}")
    print("PRICE_ADJUSTMENT: Yahoo split-adjusted Open/Close; explicit ex-date cash distributions")
    print(f"Excess Sharpe reference: {config['excess_reference']} daily total returns")

    symbol_changes = config["symbol_changes"]
    cases = run_robustness(data, first, split, parameters, execution, config["robustness"], reference, symbol_changes)
    daily, deferred, expired = simulate(data, first, last, parameters, execution, symbol_changes)
    benchmark = benchmark_path(data, first, last, reference, execution)
    results = {"strategy": config["strategy"], "data_range": data_range}
    ranges = {"is": slice(0, split - first + 1), "oos": slice(split - first + 1, None), "full": slice(None)}
    for period, window in ranges.items():
        results[period] = performance(daily.iloc[window], reference.iloc[window])
        comparison = performance(benchmark.iloc[window], reference.iloc[window])
        print(f"{period.upper()}: {json.dumps(results[period], allow_nan=False)}")
        print(f"Benchmark {config['benchmark']} {period.upper()}: {json.dumps(comparison, allow_nan=False)}")
        if period == "full":
            results["benchmark"] = comparison
    print(f"OPEN_FILL_DEFERRALS: {deferred}")
    print(f"EXPIRED_REBALANCE_NAMES: {expired}")
    print(f"PARAM_COMBOS_TRIED: {1 + cases}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


def stop_for_timeout(*_):
    sys.exit("Backtest exceeded its 9-minute runtime limit.")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_for_timeout)
    signal.alarm(9 * 60)
    try:
        main()
    finally:
        signal.alarm(0)
