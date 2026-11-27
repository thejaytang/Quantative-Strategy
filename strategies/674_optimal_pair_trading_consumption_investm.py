# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 674_optimal_pair_trading_consumption_investm\nuniverse: [XOM, CVX]\ncash_reference: BIL\ndata_start: '2008-01-02'\nbacktest_start: '2009-07-01'\noos_start: '2018-01-02'\nend: '2025-12-31'\n\nlookback: 252\nrebalance_sessions: 5\nsignal_lag_sessions: 1\ncointegration_pvalue: 0.10\ncointegration_lags: 1\nmin_half_life: 5\nmax_half_life: 63\nhorizon_sessions: 63\nutility_exponent: 0.50\nterminal_weight: 1.0\ncontrol_scale: 0.025\ngross_limit: 0.75\nuse_intertemporal_term: true\n\ntransaction_cost_bps: 2.0\nslippage_bps: 3.0\nshort_borrow_bps_annual: 100.0\n\n# Declared before the first run. These cases run only on the IS slice.\nrobustness:\n  lookback_189: {lookback: 189}\n  lookback_315: {lookback: 315}\n  utility_035: {utility_exponent: 0.35}\n  utility_065: {utility_exponent: 0.65}\n  myopic_only: {use_intertemporal_term: false}\n  double_friction:\n    transaction_cost_bps: 4.0\n    slippage_bps: 6.0\n    short_borrow_bps_annual: 200.0\n"
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
"""C250 research backtest of finite-horizon OU consumption-investment feedback.

Paper: https://arxiv.org/abs/2312.06842
Research references (ideas only, independently implemented):
ArbitrageLab (BSD-3-Clause): https://github.com/hudson-and-thames/arbitragelab
OU source: https://github.com/hudson-and-thames/arbitragelab/blob/master/arbitragelab/optimal_mean_reversion/ou_model.py
Package: https://pypi.org/project/arbitragelab/
QuantConnect tutorial: https://www.quantconnect.com/research/15294/optimal-pairs-trading/
"""

import gzip
import hashlib
import json
import math
import re
import signal
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from statsmodels.tsa.stattools import coint


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
QUOTE_FIELDS = ["Open", "Close", "Dividends", "Capital Gains", "Stock Splits"]
QUADRATURE_NODES, QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(16)


@dataclass(frozen=True)
class Settings:
    strategy: str
    universe: list[str]
    cash_reference: str
    data_start: str
    backtest_start: str
    oos_start: str
    end: str
    lookback: int
    rebalance_sessions: int
    signal_lag_sessions: int
    cointegration_pvalue: float
    cointegration_lags: int
    min_half_life: float
    max_half_life: float
    horizon_sessions: int
    utility_exponent: float
    terminal_weight: float
    control_scale: float
    gross_limit: float
    use_intertemporal_term: bool
    transaction_cost_bps: float
    slippage_bps: float
    short_borrow_bps_annual: float

    @property
    def friction(self):
        return (self.transaction_cost_bps + self.slippage_bps) / 10_000

    def validate(self):
        symbols = [*self.universe, self.cash_reference]
        if len(self.universe) != 2 or len(set(symbols)) != 3 or any(
            not re.fullmatch(r"[A-Z]{1,5}", symbol) for symbol in symbols
        ):
            raise ValueError("Configure two distinct US stock symbols and a cash-reference ETF")
        dates = pd.to_datetime([self.data_start, self.backtest_start, self.oos_start, self.end])
        if not dates[0] < dates[1] < dates[2] <= dates[3]:
            raise ValueError("Dates must be in chronological order with separate IS and OOS")
        if not 1 <= self.signal_lag_sessions or not 1 <= self.rebalance_sessions:
            raise ValueError("Rebalancing and signal lag must be positive session counts")
        if not 0 <= self.cointegration_lags < self.lookback // 2 - 2:
            raise ValueError("The lookback must support the specified cointegration lags")
        if not (0 < self.cointegration_pvalue < 1 and 0 < self.min_half_life <= self.max_half_life):
            raise ValueError("Invalid cointegration or half-life limits")
        if not (0 < self.utility_exponent < 1 and self.horizon_sessions > 0 and self.terminal_weight > 0):
            raise ValueError("Invalid utility or horizon parameters")
        if not (0 < self.gross_limit < 1 and self.control_scale > 0):
            raise ValueError("The pair requires a positive control scale and a gross limit below one")
        costs = [self.transaction_cost_bps, self.slippage_bps, self.short_borrow_bps_annual]
        if not all(math.isfinite(cost) and cost >= 0 for cost in costs) or self.friction >= 1:
            raise ValueError("Invalid friction or borrowing costs")


def fetch_snapshot(settings):
    quotes = {}
    end_exclusive = (pd.Timestamp(settings.end) + pd.Timedelta(days=1)).date().isoformat()
    yf.config.debug.hide_exceptions = False
    for symbol in [*settings.universe, settings.cash_reference]:
        ticker = yf.Ticker(symbol)
        bars = ticker.history(
            start=settings.data_start, end=end_exclusive, interval="1d", auto_adjust=False,
            keepna=True, repair=False, actions=True, timeout=20,
        )
        expected = "EQUITY" if symbol in settings.universe else "ETF"
        metadata = ticker.history_metadata
        if metadata.get("instrumentType") != expected or metadata.get("currency") != "USD":
            raise ValueError(f"{symbol} is not a USD {expected}")
        if not set(QUOTE_FIELDS) - {"Capital Gains"} <= set(bars):
            raise ValueError(f"{symbol} is missing prices or corporate actions")
        bars = bars.reindex(columns=QUOTE_FIELDS, fill_value=0.0)
        dates = bars.index.tz_localize(None).strftime("%Y-%m-%d")
        quotes[symbol] = [(day, *row) for day, row in zip(dates, bars.to_numpy(dtype=float), strict=True)]
    return {"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "quotes": quotes}


def total_return_marks(snapshot, settings):
    opens, closes = {}, {}
    first_allowed = pd.Timestamp(settings.data_start)
    last_allowed = pd.Timestamp(settings.end)
    for symbol, records in snapshot["quotes"].items():
        bars = pd.DataFrame(records, columns=["date", *QUOTE_FIELDS]).set_index("date")
        bars.index = pd.to_datetime(bars.index)
        if (len(bars) < 2 or not bars.index.is_unique or not bars.index.is_monotonic_increasing
                or not np.isfinite(bars.to_numpy()).all()):
            raise ValueError(f"Invalid price or action history for {symbol}")
        if not (
            first_allowed <= bars.index[0] <= first_allowed + pd.Timedelta(days=7)
            and last_allowed - pd.Timedelta(days=7) <= bars.index[-1] <= last_allowed
        ):
            raise ValueError(f"Incomplete date coverage for {symbol}")
        if opens and not bars.index.equals(next(iter(opens.values())).index):
            raise ValueError(f"{symbol} has missing or extra sessions compared with the other series")
        open_price, close_price = bars["Open"], bars["Close"]
        if (open_price <= 0).any() or (close_price <= 0).any():
            raise ValueError(f"Nonpositive market price for {symbol}")
        # Yahoo OHLC is already split-adjusted. Use effective-date distributions
        # to reinvest at the open. The opening mark uses only the opening quote,
        # the previous close and distributions effective that morning.
        distribution = bars["Dividends"] + bars["Capital Gains"]
        overnight = (open_price + distribution) / close_price.shift(1)
        intraday = close_price / open_price
        growth = overnight * intraday
        growth.iloc[0] = intraday.iloc[0]
        closes[symbol] = growth.cumprod()
        opening_marks = closes[symbol].shift(1) * overnight
        opening_marks.iloc[0] = 1.0
        opens[symbol] = opening_marks
        if (growth <= 0).any() or not np.isfinite(closes[symbol]).all():
            raise ValueError(f"Invalid total-return growth for {symbol}")
    return pd.DataFrame(opens), pd.DataFrame(closes)


def load_market(settings):
    CACHE.mkdir(exist_ok=True)
    key = json.dumps([settings.universe, settings.cash_reference, settings.data_start, settings.end])
    cache_file = CACHE / ("yahoo_daily_" + hashlib.sha256(key.encode()).hexdigest()[:16] + ".json.gz")
    if cache_file.is_file():
        with gzip.open(cache_file, "rt", encoding="utf-8") as source:
            snapshot = json.load(source)
        return snapshot, *total_return_marks(snapshot, settings)

    with TemporaryDirectory(dir=CACHE) as library_cache:
        yf.set_tz_cache_location(library_cache)
        snapshot = fetch_snapshot(settings)
    opens, closes = total_return_marks(snapshot, settings)
    with NamedTemporaryFile(dir=CACHE, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8") as target:
            json.dump(snapshot, target, allow_nan=False, separators=(",", ":"))
        temporary_path.replace(cache_file)
    finally:
        temporary_path.unlink(missing_ok=True)
    return snapshot, opens, closes


def feedback(spread, speed, variance, settings):
    """Riccati kernel in time to maturity, with paper rate r = 0."""
    gamma = settings.utility_exponent
    adjusted_speed = speed / (1 - gamma)
    if not settings.use_intertemporal_term:
        return -spread * adjusted_speed / variance

    horizon = settings.horizon_sessions / SESSIONS_PER_YEAR
    times = np.r_[horizon, (1 + QUADRATURE_NODES) * horizon / 2]
    distance = speed / math.sqrt(1 - gamma)
    upper = adjusted_speed + distance
    lower = gamma * adjusted_speed**2 / upper
    root_ratio = lower / upper
    decay = np.exp(-2 * distance * times)
    denominator = 1 - root_ratio * decay
    q = lower * (-np.expm1(-2 * distance * times)) / denominator
    integrated_q = lower * times - np.log(denominator / (1 - root_ratio))

    # Differentiate the terminal kernel and the consumption integral together.
    log_weights = np.r_[math.log(settings.terminal_weight) / (1 - gamma),
                        np.log(QUADRATURE_WEIGHTS * horizon / 2)]
    log_weights += 0.5 * (integrated_q + spread**2 * q / variance)
    weights = np.exp(log_weights - log_weights.max())
    return spread * (np.average(q, weights=weights) - adjusted_speed) / variance


def target_units(closing_window, settings):
    reference = closing_window[0]
    first, second = (closing_window / reference).T
    design = np.column_stack([np.ones(len(second)), second])
    intercept, hedge = np.linalg.lstsq(design, first, rcond=None)[0]
    spread = first - intercept - hedge * second
    if hedge <= 0 or np.std(spread) < 1e-8:
        return np.zeros(2)

    # Conditional Gaussian OU likelihood can be fitted as an AR(1) regression.
    prior = np.column_stack([np.ones(len(spread) - 1), spread[:-1]])
    intercept, phi = np.linalg.lstsq(prior, spread[1:], rcond=None)[0]
    if not 0 < phi < 1:
        return np.zeros(2)
    speed = -math.log(phi) * SESSIONS_PER_YEAR
    half_life = math.log(2) * SESSIONS_PER_YEAR / speed
    if not settings.min_half_life <= half_life <= settings.max_half_life:
        return np.zeros(2)
    pvalue = coint(first, second, trend="c", maxlag=settings.cointegration_lags, autolag=None)[1]
    if not pvalue <= settings.cointegration_pvalue:
        return np.zeros(2)

    innovations = spread[1:] - (intercept + phi * spread[:-1])
    variance = np.mean(innovations**2) * 2 * speed / (1 - phi**2)
    if variance <= 1e-12:
        return np.zeros(2)
    state = spread[-1] - intercept / (1 - phi)
    control = feedback(state, speed, variance, settings)
    return settings.control_scale * control * np.array([1.0, -hedge]) / reference


def rebalance(cash, positions, target, open_price, gross_limit, friction):
    marked = positions * open_price
    value = cash + marked.sum()
    old_gross = np.abs(marked).sum()
    available = value - friction * old_gross
    if available <= 0:
        raise ValueError("Opening value cannot cover liquidation costs")
    # This fee reserve bounds fees by friction times old plus new gross exposure.
    limit = gross_limit * available / (1 + gross_limit * friction)
    desired = target * value
    desired_gross = np.abs(desired * open_price).sum()
    if desired_gross > limit:
        desired *= limit / desired_gross
    fills = (desired - positions) * open_price
    volume = np.abs(fills).sum()
    next_cash = cash - fills.sum() - volume * friction
    collateral = 2 * np.maximum(-desired * open_price, 0).sum()
    tolerance = 1e-12 * value
    if next_cash < collateral - tolerance:
        raise RuntimeError("Basket fill exceeded the cash or collateral budget")
    return max(0.0, next_cash), desired, volume / value, int(np.count_nonzero(fills))


def simulate(opens, closes, settings, *, passive=False):
    dates = closes.index
    first = dates.searchsorted(pd.Timestamp(settings.backtest_start))
    lag = settings.signal_lag_sessions
    if first - lag + 1 < settings.lookback or first >= len(dates):
        raise ValueError("Insufficient warm-up or performance dates")
    open_prices = opens[settings.universe].to_numpy()
    close_prices = closes[settings.universe].to_numpy()
    positions = np.zeros(2)
    cash = prior_value = 1.0
    daily_results = []

    for i in range(first, len(dates)):
        target = None
        open_price = open_prices[i]
        if passive:
            if i == first:
                target = 0.5 / open_price
        else:
            prior_shorts = np.maximum(-positions * close_prices[i - 1], 0).sum()
            days = (dates[i] - dates[i - 1]).days
            cash -= prior_shorts * (settings.short_borrow_bps_annual / 10_000) * days / 365
            if (i - first) % settings.rebalance_sessions == 0:
                end = i - lag + 1
                target = target_units(close_prices[end - settings.lookback:end], settings)
            elif cash < 2 * np.maximum(-positions * open_price, 0).sum():
                target = np.zeros(2)

        turnover, trades = 0.0, 0
        if target is not None:
            cash, positions, turnover, trades = rebalance(
                cash, positions, target, open_price, 1.0 if passive else settings.gross_limit, settings.friction,
            )
        value = cash + positions @ close_prices[i]
        if cash < 0 or value <= 0 or not math.isfinite(value):
            raise RuntimeError(f"Infeasible close on {dates[i].date()}")
        daily_results.append((value / prior_value - 1, turnover, trades, cash))
        prior_value = value

    return pd.DataFrame(daily_results, index=dates[first:], columns=["return", "turnover", "n_trades", "cash"])


def metrics(results, cash_returns):
    returns = results["return"]
    if len(returns) < 2 or not cash_returns.index.equals(returns.index):
        raise ValueError("Metrics require aligned portfolio and reference returns")
    excess = returns - cash_returns
    values = np.r_[1.0, (1 + returns).cumprod()]
    annualizer = math.sqrt(SESSIONS_PER_YEAR)
    vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    return {
        "cagr": float(values[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(vol * annualizer),
        "sharpe": float(returns.mean() / vol * annualizer) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * annualizer) if excess_vol > 0 else None,
        "max_drawdown": float(np.min(values / np.maximum.accumulate(values) - 1)),
        "total_return": float(values[-1] - 1),
        "turnover": float(results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(results["n_trades"].sum()),
    }


def print_json(label, value):
    print(f"{label}: {json.dumps(value, allow_nan=False)}")


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    robustness = config.pop("robustness")
    settings = Settings(**config)
    settings.validate()
    variants = {name: replace(settings, **overrides) for name, overrides in robustness.items()}
    for variant in variants.values():
        variant.validate()

    snapshot, opens, closes = load_market(settings)
    split = pd.Timestamp(settings.oos_start)
    in_sample_dates = closes.index < split
    cash_opens, cash_closes = opens[settings.cash_reference], closes[settings.cash_reference]
    cash_returns = cash_closes.pct_change().loc[settings.backtest_start:].copy()
    initial_date = cash_returns.index[0]
    cash_returns.iloc[0] = cash_closes.loc[initial_date] / cash_opens.loc[initial_date] - 1
    is_cash = cash_returns.loc[cash_returns.index < split]

    print(f"DATA_FETCHED_AT: {snapshot['fetched_at']}")
    print("PRICE_ADJUSTMENT: yfinance split-adjusted OHLC, forward total return with ex-date open reinvestment")
    print(f"DATA_RANGE_WITH_WARMUP: {closes.index[0].date()} to {closes.index[-1].date()}")

    # Freeze the configured policy and run all alternatives before evaluating OOS.
    is_opens, is_closes = opens.loc[in_sample_dates], closes.loc[in_sample_dates]
    baseline_is = simulate(is_opens, is_closes, settings)
    print_json("ROBUSTNESS IS official_config", metrics(baseline_is, is_cash))
    for name, variant in variants.items():
        case = simulate(is_opens, is_closes, variant)
        print_json(f"ROBUSTNESS IS {name}", metrics(case, is_cash))

    official = simulate(opens, closes, settings)
    pd.testing.assert_frame_equal(baseline_is, official.loc[official.index < split], check_exact=True)
    passive = simulate(opens, closes, settings, passive=True)
    periods = {"is": official.index[official.index < split],
               "oos": official.index[official.index >= split], "full": official.index}
    summaries, benchmark = {}, {}
    for name, dates in periods.items():
        reference = cash_returns.loc[dates]
        summaries[name] = metrics(official.loc[dates], reference)
        benchmark[name] = metrics(passive.loc[dates], reference)
        print_json(f"STRATEGY {name.upper()} {dates[0].date()} to {dates[-1].date()}", summaries[name])
        print_json(f"BENCHMARK {name.upper()} {dates[0].date()} to {dates[-1].date()}", benchmark[name])
    if not summaries["full"]["n_trades"]:
        raise RuntimeError("The official configuration produced no fills")

    print(f"PARAM_COMBOS_TRIED: {1 + len(variants)}")
    print("LOOK_AHEAD_CHECK: positive close-to-open lag, open-only sizing, effective-date actions, identical isolated IS path")
    print("RUN_STATUS: 已回测")
    print_json("RESULTS_JSON", {
        "strategy": settings.strategy,
        "data_range": f"{official.index[0].date()} to {official.index[-1].date()}",
        **summaries,
        "benchmark": {"name": "/".join(settings.universe) + " equal-weight buy and hold", **benchmark},
    })


if __name__ == "__main__":
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)
