# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 511_dynamic_portfolio_optimization_under_cva\n\nuniverse:\n  risky: SPY\n  defensive: BIL\n\ndates:\n  price_start: 2007-10-01\n  start: 2009-01-02\n  is_end: 2017-12-31\n  end: 2025-12-31\n\nmodel:\n  lookback_sessions: 252\n  risk_horizon_sessions: 21\n  drawdown_lookback_sessions: 252\n  alpha: 0.95\n  risk_aversion: 5.0\n  cvar_limit: 0.05\n  cvar_floor: 0.01\n  drawdown_slope: 0.50\n\nexecution:\n  lag_sessions: 1\n  cost_bps: 2.0\n  slippage_bps: 3.0\n  cash_buffer: 0.05  # Adverse move on each leg's prior-close notional; fees are reserved separately\n\nrobustness:\n  lookback_offsets: [-21, 21]\n  cvar_limit_offsets: [-0.01, 0.01]\n  friction_multiplier: 2.0\n"
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
"""Monthly SPY/BIL allocation with a drawdown-sensitive empirical CVaR limit."""

import argparse
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory

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
from scipy.optimize import brentq, minimize_scalar
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_ADJUSTMENT = "yfinance auto_adjust=True, adjusted Close with distributions and splits"
TRADE_TOLERANCE = 1e-12
SIZING_ROUNDOFF = 1e-11
SESSIONS_PER_YEAR = 252


class NyseArcaHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


EXTRA_CLOSURES = pd.to_datetime([
    "2007-01-02",  # Ford national day of mourning
    "2012-10-29", "2012-10-30",  # Hurricane Sandy
    "2018-12-05",  # Bush national day of mourning
    "2025-01-09",  # Carter national day of mourning
])


@dataclass(frozen=True)
class Settings:
    lookback_sessions: int
    risk_horizon_sessions: int
    drawdown_lookback_sessions: int
    alpha: float
    risk_aversion: float
    cvar_limit: float
    cvar_floor: float
    drawdown_slope: float
    lag_sessions: int
    cost_bps: float
    slippage_bps: float
    cash_buffer: float

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000

    @property
    def sizing_reserve_rate(self):
        # Cover a move against each ETF leg and the change in its fee and slippage.
        return self.friction + self.cash_buffer * (1 + self.friction)

    def validate(self):
        counts = (
            self.lookback_sessions, self.risk_horizon_sessions,
            self.drawdown_lookback_sessions, self.lag_sessions,
        )
        if any(type(count) is not int or count < 1 for count in counts):
            raise ValueError("Session counts and execution lag must be positive integers.")
        if self.lookback_sessions <= self.risk_horizon_sessions:
            raise ValueError("Lookback must exceed the scenario horizon.")
        if not 0 < self.alpha < 1 or not 0 < self.cvar_floor <= self.cvar_limit < 1:
            raise ValueError("Invalid CVaR confidence or limits.")
        if not np.isfinite(self.risk_aversion) or self.risk_aversion <= 0:
            raise ValueError("Risk aversion must be positive and finite.")
        if not np.isfinite(self.drawdown_slope) or self.drawdown_slope < 0:
            raise ValueError("Drawdown slope must be nonnegative and finite.")
        if not (self.cost_bps >= 0 and self.slippage_bps >= 0 and self.friction < 1):
            raise ValueError("Invalid costs or slippage in basis points.")
        if not 0 < self.cash_buffer < 1 or self.sizing_reserve_rate + SIZING_ROUNDOFF >= 1:
            raise ValueError("Cash buffer and friction must give a reserve rate below one.")


def validate_prices(prices, symbols, start, end):
    if list(prices.columns) != list(symbols) or len(prices) < 2:
        raise ValueError("Adjusted-close series for SPY and BIL are required.")
    dates = prices.index
    if (not isinstance(dates, pd.DatetimeIndex) or dates.tz is not None or dates.hasnans
            or not dates.is_monotonic_increasing or dates.has_duplicates):
        raise ValueError("Price dates must be unique and chronological.")
    holidays = NyseArcaHolidays().holidays(start, end).union(EXTRA_CLOSURES)
    expected = pd.bdate_range(start, end).difference(holidays)
    if not dates.equals(expected):
        missing = expected.difference(dates).strftime("%Y-%m-%d").tolist()
        extra = dates.difference(expected).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"NYSE Arca session mismatch (missing: {missing[:5]}, extra: {extra[:5]}).")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("A common session has a missing or invalid adjusted close.")


def load_prices(symbols, start, end):
    CACHE.mkdir(exist_ok=True)
    cache_path = CACHE / f"{'_'.join(symbols)}_{start.date()}_{end.date()}_adjusted.csv"
    if cache_path.exists():
        try:
            with cache_path.open(encoding="utf-8") as snapshot:
                header = snapshot.readline().strip()
                if not header.startswith("# fetched_at="):
                    raise ValueError("Missing fetch timestamp.")
                fetched_at = header.removeprefix("# fetched_at=")
                if datetime.fromisoformat(fetched_at).tzinfo is None:
                    raise ValueError("Fetch timestamp has no time zone.")
                prices = pd.read_csv(snapshot, index_col="Date", parse_dates=["Date"],
                                     converters=dict.fromkeys(symbols, float))
            validate_prices(prices, symbols, start, end)
            return prices, fetched_at
        except (OSError, ValueError) as error:
            print(f"Refreshing incomplete price cache ({error})", file=sys.stderr)

    with TemporaryDirectory(dir=CACHE, prefix=".yfinance-") as download_cache:
        yf.set_tz_cache_location(download_cache)
        for attempt in range(2):
            raw = yf.download(
                list(symbols), start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
                interval="1d", auto_adjust=True, ignore_tz=True, multi_level_index=True,
                progress=False, threads=False, timeout=15,
            )
            try:
                prices = raw["Close"].reindex(columns=symbols).astype(float).sort_index()
                validate_prices(prices, symbols, start, end)
            except (KeyError, ValueError) as error:
                if attempt == 1:
                    raise RuntimeError(f"Could not obtain the requested yfinance closes: {error}") from error
            else:
                break

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as snapshot:
        temporary_path = Path(snapshot.name)
        try:
            snapshot.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(snapshot, index_label="Date")
            snapshot.flush()
            temporary_path.replace(cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def empirical_cvar(losses, alpha):
    """Include the fraction of the boundary observation needed to fill the tail."""
    tail_mass = (1 - alpha) * len(losses)
    whole = int(tail_mass)
    fraction = tail_mass - whole
    losses = np.sort(losses)[::-1]
    return float((losses[:whole].sum() + fraction * losses[whole:whole + 1].sum()) / tail_mass)


def choose_weight(scenarios, drawdown, settings):
    """Maximize mean minus a variance penalty over the convex feasible interval."""
    defensive = scenarios[:, 1]
    spread = scenarios[:, 0] - defensive
    centered_spread = spread - spread.mean()
    variance = np.mean(centered_spread**2)
    covariance = np.mean(centered_spread * (defensive - defensive.mean()))
    slope = spread.mean() - settings.risk_aversion * covariance
    if variance == 0:
        preferred = float(slope > 0)
    else:
        preferred = float(np.clip(slope / (settings.risk_aversion * variance), 0, 1))

    budget = max(settings.cvar_floor, settings.cvar_limit - settings.drawdown_slope * drawdown)

    def residual(weight):
        return empirical_cvar(-(defensive + weight * spread), settings.alpha) - budget

    preferred_residual = residual(preferred)
    if preferred_residual <= 0:
        return preferred, budget + preferred_residual, budget, False

    anchor = 0.0
    if residual(anchor) > 0:
        minimum = minimize_scalar(residual, bounds=(0.0, 1.0), method="bounded",
                                  options={"xatol": 1e-12})
        anchor = min((0.0, float(minimum.x), 1.0), key=residual)
        anchor_residual = residual(anchor)
        if anchor_residual >= 0:
            return anchor, budget + anchor_residual, budget, anchor_residual > 0

    lower, upper = sorted((anchor, preferred))
    weight = float(brentq(residual, lower, upper, xtol=1e-12))
    return weight, budget + residual(weight), budget, False


def size_order(positions, cash, target, reference_prices, settings):
    """Fix signed quantities at the reference close, allowing for adverse fills."""
    marked = positions * reference_prices
    value = cash + marked.sum()
    weights = marked / value

    def residual(scale):
        turnover = np.abs(scale * target - weights).sum()
        return scale + settings.sizing_reserve_rate * turnover - (1.0 - SIZING_ROUNDOFF)

    scale = brentq(residual, 0.0, 1.0)
    changes = scale * value * target / reference_prices - positions
    changes[np.abs(changes) * reference_prices <= TRADE_TOLERANCE * value] = 0.0
    return changes


def apply_fill(positions, cash, changes, fill_prices, friction, fill_date):
    """Settle the predetermined share changes at the close, including both costs."""
    prefill_value = cash + positions @ fill_prices
    notionals = changes * fill_prices
    gross_traded = np.abs(notionals).sum()
    remaining_cash = cash - notionals.sum() - friction * gross_traded
    if remaining_cash < 0:
        raise RuntimeError(f"{fill_date.date()}: fixed quantities would leave negative cash "
                           f"({remaining_cash:.8g}).")
    return positions + changes, remaining_cash, gross_traded / prefill_value


def simulate(prices, start, settings, execution_details=None):
    settings.validate()
    dates = prices.index
    first = dates.searchsorted(start)
    if first < settings.lookback_sessions or first >= len(prices) - 1:
        raise ValueError("Prices must cover the lookback and at least two evaluation sessions.")

    closes = prices.to_numpy()
    scenarios = (prices / prices.shift(settings.risk_horizon_sessions) - 1).to_numpy()
    positions = np.zeros(2)
    cash = previous_value = 1.0
    recent_values = deque([previous_value], maxlen=settings.drawdown_lookback_sessions + 1)
    targets_to_size = {}
    initial_target = np.array([0.0, 1.0])
    initial_changes = size_order(positions, cash, initial_target, closes[first - 1], settings)
    order = (None, initial_target, initial_changes, (None, None, None, False))
    rows = []

    for i in range(first, len(prices)):
        close = closes[i]
        turnover, trades, weight_error = 0.0, 0, None
        if order is not None:
            signal_date, target, changes, fill_signal = order
            positions, cash, turnover = apply_fill(
                positions, cash, changes, close, settings.friction, dates[i]
            )
            trades = int(np.count_nonzero(changes))
        marked = positions * close
        value = cash + marked.sum()
        if order is not None:
            filled_weights = marked / value
            weight_error = float(np.max(np.abs(filled_weights - target)))
            if execution_details is not None:
                _, estimated_cvar, risk_budget, infeasible = fill_signal
                execution_details.append({
                    "signal_date": str(signal_date.date()) if signal_date is not None else None,
                    "sizing_date": str(dates[i - 1].date()),
                    "fill_date": str(dates[i].date()),
                    "reference_prices": closes[i - 1].tolist(),
                    "fill_prices": close.tolist(),
                    "share_changes": changes.tolist(),
                    "target_weights": target.tolist(),
                    "estimated_cvar": estimated_cvar,
                    "cvar_budget": risk_budget,
                    "cvar_infeasible": infeasible,
                    "filled_weights": filled_weights.tolist(),
                    "cash_after": float(cash),
                })
        recent_values.append(value)
        proxy_return = 0.0 if i == first else close[1] / closes[i - 1, 1] - 1.0

        signal_diagnostics = (None, None, None, False)
        if dates[i].month != dates[i - 1].month:
            drawdown = 1.0 - value / max(recent_values)
            first_scenario = i - settings.lookback_sessions + settings.risk_horizon_sessions
            signal_diagnostics = choose_weight(scenarios[first_scenario:i + 1], drawdown, settings)
            sizing_day = i + settings.lag_sessions - 1
            targets_to_size[sizing_day] = (dates[i], signal_diagnostics)

        # Today's closing state may size tomorrow's order, including with a longer signal lag.
        signal_to_size = targets_to_size.pop(i, None)
        order = None
        if signal_to_size is not None and i + 1 < len(prices):
            signal_date, fill_signal = signal_to_size
            risky_weight = fill_signal[0]
            target = np.array([risky_weight, 1.0 - risky_weight])
            changes = size_order(positions, cash, target, close, settings)
            order = (signal_date, target, changes, fill_signal)
        rows.append((value / previous_value - 1.0, proxy_return, turnover, trades,
                     cash, cash / value, weight_error, *signal_diagnostics))
        previous_value = value

    return pd.DataFrame(
        rows, index=dates[first:],
        columns=["return", "proxy_return", "turnover", "n_trades", "cash", "cash_weight",
                 "weight_error", "signal_spy_weight", "signal_cvar", "signal_budget", "infeasible"],
    )


def buy_and_hold(prices, daily_results, settings):
    baseline = daily_results[["proxy_return"]].copy()
    first = baseline.index[0]
    closes = prices.to_numpy()
    i = prices.index.get_loc(first)
    positions = np.zeros(2)
    changes = size_order(positions, 1.0, np.array([1.0, 0.0]), closes[i - 1], settings)
    positions, cash, turnover = apply_fill(positions, 1.0, changes, closes[i], settings.friction, first)
    values = cash + closes[i:] @ positions
    baseline["return"] = values / np.r_[1.0, values[:-1]] - 1.0
    baseline["turnover"] = 0.0
    baseline["n_trades"] = 0
    baseline.loc[first, "turnover"] = turnover
    baseline.loc[first, "n_trades"] = int(np.count_nonzero(changes))
    return baseline


def cvar_diagnostics(daily, settings):
    signals = daily.loc[daily["signal_budget"].notna()]
    fallbacks = signals.loc[signals["infeasible"]]
    excess = fallbacks["signal_cvar"] - fallbacks["signal_budget"]
    horizon = settings.risk_horizon_sessions
    factors = 1 + daily["return"].to_numpy()
    realized_cvar = None
    windows = max(0, len(factors) - horizon + 1)
    if windows:
        horizon_returns = np.lib.stride_tricks.sliding_window_view(factors, horizon).prod(axis=1) - 1
        realized_cvar = empirical_cvar(-horizon_returns, settings.alpha)
    return {
        "signals": len(signals),
        "infeasible_signals": len(fallbacks),
        "max_budget_excess": float(excess.max()) if len(excess) else 0.0,
        "horizon_sessions": horizon,
        "alpha": settings.alpha,
        "realized_windows": windows,
        "realized_loss_cvar": realized_cvar,
    }


def metrics(daily, starting_close):
    years = (daily.index[-1] - starting_close).days / 365.2425 if len(daily) else 0

    def sharpe(returns):
        deviation = returns.std(ddof=1)
        if np.isfinite(deviation) and deviation > 0:
            return float(np.sqrt(SESSIONS_PER_YEAR) * returns.mean() / deviation)
        return None

    returns = daily["return"]
    curve = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    return {
        "cagr": float(curve[-1] ** (1 / years) - 1) if years > 0 else None,
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)) if len(daily) > 1 else None,
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(returns - daily["proxy_return"]),
        "max_drawdown": float(np.min(curve / np.maximum.accumulate(curve) - 1)) if len(daily) else None,
        "total_return": float(curve[-1] - 1) if len(daily) else None,
        "turnover": float(daily["turnover"].sum() / years) if years > 0 else None,
        "n_trades": int(daily["n_trades"].sum()),
    }


def period_metrics(daily_results, split):
    insample = daily_results.loc[:split]
    outsample = daily_results.loc[daily_results.index > split]
    inception = daily_results.index[0]
    oos_starting_close = insample.index[-1] if len(insample) else inception
    return {
        "is": metrics(insample, inception),
        "oos": metrics(outsample, oos_starting_close),
        "full": metrics(daily_results, inception),
    }


def robustness_cases(official, options):
    for offset in options["lookback_offsets"]:
        yield replace(official, lookback_sessions=official.lookback_sessions + offset)
    for offset in options["cvar_limit_offsets"]:
        yield replace(official, cvar_limit=round(official.cvar_limit + offset, 12))
    scale = options["friction_multiplier"]
    yield replace(official, cost_bps=official.cost_bps * scale,
                  slippage_bps=official.slippage_bps * scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-details", action="store_true",
                        help="Print the strategy's fixed-quantity fill records in SPY, BIL order.")
    args = parser.parse_args()
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    symbols = (config["universe"]["risky"], config["universe"]["defensive"])
    if symbols != ("SPY", "BIL"):
        raise ValueError("This specification uses SPY and BIL.")
    dates = config["dates"]
    price_start, start, split, end = (
        pd.Timestamp(dates[key]) for key in ("price_start", "start", "is_end", "end")
    )
    if not price_start < start <= split < end:
        raise ValueError("Invalid warmup, evaluation, or split dates.")

    official = Settings(**config["model"], **config["execution"])
    official.validate()
    prices, fetched_at = load_prices(symbols, price_start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")

    # The original parameters and alternatives are fixed before either sample is scored.
    cases = dict.fromkeys(robustness_cases(official, config["robustness"]))
    cases.pop(official, None)
    insample_prices = prices.loc[:split]
    print("ROBUSTNESS (IS only)")
    for case in cases:
        daily_results = simulate(insample_prices, start, case)
        parameters = (f"lookback={case.lookback_sessions}, cvar_limit={case.cvar_limit:g}, "
                      f"cost_bps={case.cost_bps:g}, slippage_bps={case.slippage_bps:g}")
        print(f"  {parameters} " + json.dumps(metrics(daily_results, daily_results.index[0]),
                                               allow_nan=False))

    execution_details = [] if args.execution_details else None
    daily_results = simulate(prices, start, official, execution_details)
    if execution_details is not None:
        print("EXECUTION SYMBOLS: " + json.dumps(symbols))
        for detail in execution_details:
            print("EXECUTION DETAIL: " + json.dumps(detail, allow_nan=False))
    performance = period_metrics(daily_results, split)
    benchmark = buy_and_hold(prices, daily_results, official)
    benchmark_performance = period_metrics(benchmark, split)
    data_range = {"start": str(daily_results.index[0].date()),
                  "end": str(daily_results.index[-1].date())}
    print(f"DATE RANGE: {data_range['start']} to {data_range['end']}")
    print(f"IS through {split.date()} | OOS after {split.date()} | Excess Sharpe vs BIL")
    for period, daily in (("IS", daily_results.loc[:split]),
                          ("OOS", daily_results.loc[daily_results.index > split]),
                          ("FULL", daily_results)):
        print(f"CVAR DIAGNOSTICS {period}: " + json.dumps(cvar_diagnostics(daily, official), allow_nan=False))
    print("EXECUTION FULL: " + json.dumps({
        "cash_buffer": official.cash_buffer,
        "min_cash_weight": float(daily_results["cash_weight"].min()),
        "max_abs_etf_weight_error": float(daily_results["weight_error"].max()),
    }, allow_nan=False))
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: " + json.dumps(performance[period], allow_nan=False))
        print(f"SPY BUY-AND-HOLD {period.upper()}: "
              + json.dumps(benchmark_performance[period], allow_nan=False))

    print(f"PARAM_COMBOS_TRIED: {1 + len(cases)}")
    results = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **performance,
        "benchmark": {"name": "SPY buy-and-hold", **benchmark_performance},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(570)
    main()
