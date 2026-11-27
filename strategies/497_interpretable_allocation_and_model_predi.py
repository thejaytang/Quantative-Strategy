"""497_interpretable_allocation_and_model_predi
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/497_interpretable_allocation_and_model_predi.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit fb92b5054b1596d6a39ea87072bbd5a288327fb7 on 2026-11-27 09:41 Europe/Oslo, exit 0, output sha256 ca04f075b9e021511b49c2bab8a907a2682161eebc0024e39c28a9975bbf393f
Data source: Yahoo Finance via yfinance supplies daily open, close, adjusted close and volume for the fixed U.S. large-cap equity universe and the ETFs SPY (benchmark) and BIL (excess-return reference). FRED's public CSV supplies the daily DTB3 Treasury bill discount rate for idle cash.
Data fetched: 2026-11-27T08:42:02+00:00
Sample period: 2014-01-03 through 2026-09-04 inclusive. The data request starts on 2012-12-01 for indicator warm-up.   data_range (from output): {'start': '2014-01-03', 'end': '2026-09-04'}
IS/OOS split: IS 2014-01-03 through 2024-12-31. OOS 2025-01-02 through 2026-09-04. The default simulation runs continuously across the boundary and the code slices its daily results for reporting. All five sensitivity runs end at the IS cutoff.
Costs and slippage: The default applies 5 bps cost and 3 bps slippage, totaling 8 bps of absolute traded notional on both buys and sells. The simulator deducts this charge when solving for the post-cost portfolio value at the open. Slippage is modeled as a fixed proportional charge. The SPY benchmark pays the same rate on its single initial opening purchase. The IS slippage sensitivity uses 10 bps slippage plus 5 bps cost.
Signal lag: When no trade is pending, the regular monthly target is computed using data through the first trading day's close and scheduled for the next session's open. A drawdown exit triggered at the close uses the same one-session delay. The first sample open uses a target computed through the prior session's close. A missing required opening price defers the fill.
Price adjustment: yfinance Adj Close total-return; Open scaled by Adj Close / Close
Survivorship bias: The fixed large-cap universe carries retrospective selection and survivorship bias. It lacks point-in-time constituent histories and many historically delisted names. At each signal the code requires complete trailing return and liquidity windows and ranks eligible names by trailing median dollar volume.
Parameter combinations tried: 6
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.1266236693119116, ann_vol=0.1589391552595198, sharpe=0.8301172727398809, sharpe_excess=0.7387829620553656, max_drawdown=-0.1915348233582138, total_return=2.7028849940217765, turnover=9.217166355762563, n_trades=3741
    oos: cagr=-0.0003187287654636828, ann_vol=0.17058878456033383, sharpe=0.08383750651349318, sharpe_excess=-0.14396936373347485, max_drawdown=-0.23666603797986274, total_return=-0.000531158169315038, turnover=10.82967240219116, n_trades=644
    full: cagr=0.10901374322318902, ann_vol=0.1605144016634109, sharpe=0.7253889960565298, sharpe_excess=0.6148704576646892, max_drawdown=-0.23666603797986285, total_return=2.7009181764071717, turnover=9.429671074777316, n_trades=4385
    benchmark: cagr=0.13896385329693972, ann_vol=0.17148203545653862, sharpe=0.8449293378566034, sharpe_excess=0.7415226191861988, max_drawdown=-0.3371727693638181, total_return=4.184076015155199, turnover=0.07900802044276095, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "497_ interpretable_allocation_and_model_predi"\ndates:\n  warmup_start: 2012-12-01\n  start: 2014-01-03\n  is_end: 2024-12-31\n  end: 2026-09-04\n\nbenchmark: SPY\nexcess_reference: BIL\ncash_rate_series: DTB3\n\n# Fixed large-cap research universe. Rank liquidity again at each signal.\nuniverse: [\n  AAPL, ABBV, ABT, ACN, ADBE, ADI, ADP, AFL, AIG, AMAT, AMD, AMGN, AMT, AMZN,\n  ANET, APD, AVGO, AXP, AZO, BA, BAC, BAX, BDX, BKNG, BLK, BMY, BNY, BRK-B, BSX,\n  C, CAT, CB, CCI, CDNS, CL, CMCSA, CME, CMG, CMI, COP, COST, CRM, CSCO, CSX,\n  CTAS, CVS, CVX, DE, DHR, DIS, DUK, DVN, ECL, EMR, EOG, EQIX, ETN, EXC,\n  F, FDX, FICO, GD, GE, GILD, GIS, GM, GOOG, GS, HD, HON, IBM, INTC, INTU,\n  ISRG, JCI, JNJ, JPM, KMB, KO, KR, LIN, LLY, LMT, LOW, LVS, MA, MAR, MCD,\n  MCK, MDT, MET, MMM, MO, MRK, MS, MSFT, MU, NEE, NFLX, NKE, NOC, NOW, NSC,\n  NVDA, O, ORCL, OXY, PEP, PFE, PG, PGR, PH, PLD, PM, PNC, PYPL, QCOM,\n  REGN, ROST, RSG, RTX, SBUX, SCHW, SHW, SLB, SO, SPGI, SYK, T, TGT, TJX,\n  TMUS, TRV, TSCO, TXN, UBER, UNH, UNP, UPS, USB, V, VLO, VRTX, VZ, WFC,\n  WM, WMT, XEL, XOM, ZTS\n]\n\nexecution:\n  cost_bps: 5.0\n  slippage_bps: 3.0\n  signal_lag_sessions: 1\n  cash_rate_lag_sessions: 2\n  max_missing_sessions: 5\n\nparameters:\n  regression_days: 252\n  regression_half_life_days: 63\n  market_growth_span_days: 252\n  market_fast_span_days: 21\n  market_slow_span_days: 63\n  market_state_gain: 21.23\n  beta_floor: 0.10\n  liquidity_days: 63\n  liquid_names: 100\n  selection_fraction: 0.20\n  risk_days: 126\n  annual_volatility_cap: 0.173\n  weight_cap: 0.10\n  drawdown_stop: 0.08\n\nrobustness:\n  half_life_days: [50, 75]\n  selection_fractions: [0.15, 0.25]\n  higher_slippage_bps: 10.0\n'
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
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
YAHOO_FIELDS = ["Open", "Close", "Adj Close", "Volume"]


@dataclass
class MarketData:
    dates: pd.DatetimeIndex
    stock_open: np.ndarray
    stock_close: np.ndarray
    dollar_volume: np.ndarray
    market_close: pd.Series
    market_open: pd.Series
    reference_close: pd.Series
    reference_open: pd.Series
    cash_returns: np.ndarray


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
            timeout=12, threads=8, ignore_tz=True,
        )
        columns = pd.MultiIndex.from_product([YAHOO_FIELDS, symbols])
        return frame.reindex(columns=columns)

    with TemporaryDirectory(dir=CACHE_DIR, prefix=".yahoo-") as yahoo_cache:
        yf.set_tz_cache_location(yahoo_cache)
        batches = [fetch(tickers[i:i + 24]) for i in range(0, len(tickers), 24)]
        prices = pd.concat(batches, axis=1).sort_index()
        missing = prices["Adj Close"].columns[prices["Adj Close"].isna().all()]
        if len(missing):
            retry = fetch(missing.tolist())
            prices = pd.concat([prices.drop(columns=retry.columns), retry], axis=1)
    return prices.sort_index()


def load_sources(config):
    dates = config["dates"]
    start = str(dates["warmup_start"])
    end = str((pd.Timestamp(dates["end"]) + pd.Timedelta(days=1)).date())
    tickers = sorted([*config["universe"], config["benchmark"], config["excess_reference"]])
    if len(set(tickers)) != len(tickers):
        raise ValueError("The universe and reference symbols must be distinct.")
    cache_key = hashlib.sha256(" ".join(tickers).encode()).hexdigest()[:12]

    CACHE_DIR.mkdir(exist_ok=True)
    price_path = CACHE_DIR / f"yahoo_{start}_{end}_{cache_key}.csv.gz"
    if price_path.is_file():
        prices = pd.read_csv(
            price_path, header=[0, 1], index_col=0, parse_dates=True, float_precision="round_trip"
        )
    else:
        prices = download_prices(tickers, start, end)

    absent = prices["Adj Close"].columns[prices["Adj Close"].isna().all()]
    if len(absent):
        raise ValueError(f"Yahoo Finance returned no prices for {', '.join(absent)}.")
    if not price_path.is_file():
        write_cache(price_path, lambda path: prices.to_csv(path))

    series = config["cash_rate_series"]
    rate_path = CACHE_DIR / f"fred_{series}_{start}_{end}.csv"
    if rate_path.is_file():
        rate_bytes = rate_path.read_bytes()
    else:
        response = requests.get(
            f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}",
            params={"cosd": start, "coed": str(dates["end"])},
            timeout=(5, 20),
        )
        response.raise_for_status()
        rate_bytes = response.content
    raw_rates = pd.read_csv(BytesIO(rate_bytes), index_col=0, parse_dates=True)
    rates = pd.to_numeric(raw_rates[series], errors="coerce").dropna().sort_index()
    if rates.empty:
        raise ValueError(f"FRED returned no observations for {series}.")
    if not rate_path.is_file():
        write_cache(rate_path, lambda path: path.write_bytes(rate_bytes))

    fetched_at = datetime.fromtimestamp(
        max(price_path.stat().st_mtime, rate_path.stat().st_mtime), tz=timezone.utc
    ).isoformat(timespec="seconds")
    return prices, rates, fetched_at


def prepare_market(prices, rates, config):
    market = config["benchmark"]
    reference = config["excess_reference"]
    tickers = sorted(config["universe"])
    adjusted = prices["Adj Close"]
    closes = adjusted.where(adjusted.gt(0) & np.isfinite(adjusted))
    opens = prices["Open"] * closes / prices["Close"]
    opens = opens.where(opens.gt(0) & np.isfinite(opens))
    dates = closes[market].dropna().index
    closes = closes.loc[dates]
    opens = opens.loc[dates]
    if dates[-1] < pd.Timestamp(config["dates"]["end"]) - pd.Timedelta(days=3):
        raise ValueError("Yahoo Finance benchmark data end before the configured sample.")
    if (
        opens[[market, reference]].isna().any().any()
        or closes[reference].isna().any()
    ):
        raise ValueError("The benchmark or excess-return reference has missing bars.")
    dollar_volume = (prices["Close"] * prices["Volume"]).loc[dates, tickers]
    dollar_volume = dollar_volume.where(dollar_volume.gt(0))

    # The rate used for an interval must have been published before it began.
    rate_lag = config["execution"]["cash_rate_lag_sessions"]
    discount = rates.reindex(dates, method="ffill").shift(rate_lag) / 100
    calendar_days = dates.to_series().diff().dt.days
    cash_returns = np.expm1(-np.log1p(-discount * 91 / 360) * calendar_days / 91)
    return MarketData(
        dates, opens[tickers].to_numpy(), closes[tickers].to_numpy(),
        dollar_volume.to_numpy(), closes[market], opens[market], closes[reference],
        opens[reference], cash_returns.to_numpy(),
    )


class Allocator:
    def __init__(self, data, parameters, last):
        self.parameters = parameters
        self.log_returns = np.diff(np.log(data.stock_close[:last + 1]), axis=0, prepend=np.nan)
        self.dollar_volume = data.dollar_volume[:last + 1]
        market_close = data.market_close.iloc[:last + 1]
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


def execute(positions, cash, opens, target, rate, active):
    marked = positions[active] * opens[active]
    opening_value = cash + marked.sum()
    weights = target[active]
    # The post-cost value must finance both the target positions and their fees.
    low, high = opening_value * (1 - 2 * rate), opening_value
    for _ in range(40):
        value = (low + high) / 2
        if value + rate * np.abs(weights * value - marked).sum() > opening_value:
            high = value
        else:
            low = value
    desired = weights * low
    traded = np.abs(desired - marked)
    positions[active] = desired / opens[active]
    return low * max(0.0, 1 - weights.sum()), traded.sum() / opening_value, int(
        (traded > opening_value * 1e-10).sum()
    )


def simulate(data, first, last, parameters, execution):
    lag = execution["signal_lag_sessions"]
    max_missing = execution["max_missing_sessions"]
    if first - lag < max(parameters["regression_days"], parameters["liquidity_days"] - 1):
        raise ValueError("The configured warmup does not cover the first decision.")
    if parameters["risk_days"] > parameters["regression_days"]:
        raise ValueError("The risk window must fit inside the regression window.")
    allocator = Allocator(data, parameters, last)
    rate = execution_rate(execution)
    dates = data.dates
    positions = np.zeros(data.stock_open.shape[1])
    last_marks = np.zeros_like(positions)
    stale = np.zeros_like(positions, dtype=int)
    cash = peak = previous_value = 1.0
    scheduled = (first, allocator.target(first - lag), False)
    deferred = consecutive_deferrals = 0
    rows = []

    for day in range(first, last + 1):
        if day > first:
            cash *= 1 + data.cash_returns[day]
        opening = data.stock_open[day]
        turnover = 0.0
        deals = 0
        stopped_today = False
        if scheduled is not None and day >= scheduled[0]:
            _, target, exit_all = scheduled
            active = (positions > 0) | (target > 0)
            if np.isfinite(opening[active]).all():
                if active.any():
                    cash, turnover, deals = execute(positions, cash, opening, target, rate, active)
                    last_marks[active] = opening[active]
                    stale[active] = 0
                stopped_today = exit_all
                scheduled = None
                consecutive_deferrals = 0
            else:
                deferred += 1
                consecutive_deferrals += 1
                if consecutive_deferrals > max_missing:
                    raise ValueError(f"Orders remain unfilled due to missing opens on {dates[day].date()}.")

        close = data.stock_close[day]
        present = np.isfinite(close)
        last_marks[present] = close[present]
        stale[present] = 0
        stale[~present] += 1
        invested = positions > 0
        if (stale[invested] > max_missing).any():
            raise ValueError(f"An open position has stale closes on {dates[day].date()}.")
        value = cash + positions @ last_marks
        rows.append((value / previous_value - 1, turnover, deals))
        previous_value = value
        peak = value if stopped_today else max(peak, value)

        if day + lag > last or stopped_today:
            continue
        stop_pending = scheduled is not None and scheduled[2]
        if invested.any() and value <= peak * (1 - parameters["drawdown_stop"]) and not stop_pending:
            scheduled = (day + lag, np.zeros_like(positions), True)
            consecutive_deferrals = 0
        elif scheduled is None and day > first and dates[day].month != dates[day - 1].month:
            scheduled = (day + lag, allocator.target(day), False)

    daily = pd.DataFrame(rows, index=dates[first:last + 1], columns=["return", "turnover", "trades"])
    return daily, deferred


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
        "ann_vol": float(volatility * annualizer),
        "sharpe": float(returns.mean() / volatility * annualizer) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * annualizer) if excess_vol > 0 else None,
        "max_drawdown": float((growth / peak - 1).min()),
        "total_return": float(growth.iloc[-1] - 1),
        "turnover": float(daily["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily["trades"].sum()),
    }


def benchmark_path(data, first, last, rate):
    returns = data.market_close.pct_change().iloc[first:last + 1].copy()
    returns.iloc[0] = data.market_close.iloc[first] / (data.market_open.iloc[first] * (1 + rate)) - 1
    daily = pd.DataFrame({"return": returns, "turnover": 0.0, "trades": 0})
    daily.iloc[0, 1:] = [1 / (1 + rate), 1]
    return daily


def run_robustness(data, first, split, parameters, execution, settings, reference):
    cases = []
    for half_life in settings["half_life_days"]:
        cases.append((f"half_life={half_life}", dict(parameters, regression_half_life_days=half_life), execution))
    for fraction in settings["selection_fractions"]:
        cases.append((f"top_fraction={fraction}", dict(parameters, selection_fraction=fraction), execution))
    slippage = settings["higher_slippage_bps"]
    cases.append((f"slippage_bps={slippage}", parameters, dict(execution, slippage_bps=slippage)))

    print("ROBUSTNESS — IS only")
    for label, case_parameters, case_execution in cases:
        daily, _ = simulate(data, first, split, case_parameters, case_execution)
        print(f"  {label} {json.dumps(performance(daily, reference.loc[daily.index]), allow_nan=False)}")
    return len(cases)


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    execution = config["execution"]
    parameters = config["parameters"]
    if (
        execution["signal_lag_sessions"] < 1
        or execution["cash_rate_lag_sessions"] < 2
        or not 0 <= execution_rate(execution) < 0.5
        or min(execution["cost_bps"], execution["slippage_bps"]) < 0
        or not 0 < parameters["selection_fraction"] <= 1
        or not 0 < parameters["weight_cap"] <= 1
        or not 0 < parameters["drawdown_stop"] < 1
    ):
        raise ValueError("Check the execution lags, costs, selection fraction, weight cap and stop.")

    prices, rates, fetched_at = load_sources(config)
    data = prepare_market(prices, rates, config)
    dates = config["dates"]
    first = int(data.dates.searchsorted(pd.Timestamp(dates["start"])))
    split = int(data.dates.searchsorted(pd.Timestamp(dates["is_end"]), side="right")) - 1
    last = int(data.dates.searchsorted(pd.Timestamp(dates["end"]), side="right")) - 1
    if not 0 < first < split < last - 1 or not np.isfinite(data.cash_returns[first + 1:last + 1]).all():
        raise ValueError("The date ranges or lagged cash observations do not cover both periods.")

    reference = data.reference_close.pct_change().iloc[first:last + 1].copy()
    reference.iloc[0] = data.reference_close.iloc[first] / data.reference_open.iloc[first] - 1
    data_range = {"start": str(data.dates[first].date()), "end": str(data.dates[last].date())}
    print(f"Date Range: {data_range['start']} to {data_range['end']}")
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance Adj Close total-return; Open scaled by Adj Close / Close")
    print(f"Excess Sharpe reference: {config['excess_reference']} daily total returns")

    cases = run_robustness(data, first, split, parameters, execution, config["robustness"], reference)
    daily, deferred = simulate(data, first, last, parameters, execution)
    benchmark = benchmark_path(data, first, last, execution_rate(execution))
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
    print(f"PARAM_COMBOS_TRIED: {1 + cases}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


def stop_for_timeout(*_):
    sys.exit("Backtest exceeded its 14-minute runtime limit.")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_for_timeout)
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)