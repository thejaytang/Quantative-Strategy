# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 587_bayesian_portfolio_optimization_by_predi\nuniverse: [SPY, TLT, GLD, VNQ]\nrisk_free_proxy: BIL\n\ndates:\n  data_start: 2007-12-01\n  sample_start: 2013-01-01\n  is_end: 2018-12-31\n  sample_end: 2025-12-31\n\nmodel:\n  short_mean_months: 12\n  long_lookback_months: 36\n  ar_orders: [1, 2, 3]\n  min_filter_updates: 12\n  state_discount: 0.99\n  residual_discount: 0.95\n  prior_intercept_sd: 0.01\n  prior_expert_sd: 0.35\n  prior_residual_sd: 0.04\n  residual_sd_floor: 0.005\n  student_df: 7\n  cvar_confidence: 0.95\n  max_weight: 0.60\n\nexecution:\n  lag_sessions: 1\n  cost_bps: 5.0\n  slippage_bps: 5.0\n  distribution_delay_days: 30\n  initial_value: 100000.0\n\nrobustness:\n  cvar_confidences: [0.90, 0.975]\n  higher_cost_bps: 20.0\n  higher_slippage_bps: 15.0\n'
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
"""Cross-asset ETF allocation using dynamic predictive synthesis and tail risk.

Paper: https://arxiv.org/abs/2510.07180 (Kato, Baba, Kaibuchi, Inokuchi, 2025).
Conceptual references (independent implementation, no copied source):
https://github.com/jasa-acs/Multivariate-Bayesian-Predictive-Synthesis-in-Macroeconomic-Forecasting
https://www2.stat.duke.edu/~mw/mwsoftware/BPS/index.html (authors' MATLAB package)
https://github.com/josephrilling/dynbps (Apache-2.0)
https://pypi.org/project/dynbps/ (Apache-2.0)
https://josephrilling.github.io/dynbps/bps.html
https://www.mcalinn.com/research/bps/tutorial.html
https://masakat0.github.io/research
Adjacent results reviewed for scope:
https://github.com/emtall/BPDS-2023
https://github.com/quantopian/bayesalpha
https://github.com/QuantConnect/HandsOnAITradingBook/blob/master/08%20AI%20for%20Risk%20Management%20and%20Optimization/01%20Conditional%20Portfolio%20Optimization%20Applied/main.py
"""

from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
from pathlib import Path
import json
import re
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
    USMartinLutherKingJr, USMemorialDay, USPresidentsDay, USThanksgivingDay,
    nearest_workday, sunday_to_monday,
)
from scipy.optimize import minimize
from scipy.stats import t as student_t
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
QUOTE_FIELDS = ("Open", "Close", "Dividends", "Capital Gains", "Stock Splits")
SPECIAL_CLOSURES = pd.to_datetime(["2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])
SESSIONS_PER_YEAR = 252


class EquityHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def expected_sessions(start, end):
    holidays = EquityHolidays().holidays(start, end).union(SPECIAL_CLOSURES)
    return pd.bdate_range(start, end).difference(holidays).rename("date")


@dataclass(frozen=True)
class Model:
    short_mean_months: int
    long_lookback_months: int
    ar_orders: tuple[int, ...]
    min_filter_updates: int
    state_discount: float
    residual_discount: float
    prior_intercept_sd: float
    prior_expert_sd: float
    prior_residual_sd: float
    residual_sd_floor: float
    student_df: float
    cvar_confidence: float
    max_weight: float

    def __post_init__(self):
        counts = (self.short_mean_months, self.long_lookback_months, self.min_filter_updates, *self.ar_orders)
        if not all(type(value) is int and value >= 1 for value in counts) or not self.ar_orders:
            raise ValueError("Forecast windows, AR orders and filter warm-up must be positive integers")
        if (not 2 <= self.short_mean_months < self.long_lookback_months
                or self.long_lookback_months <= 2 * max(self.ar_orders) + 1
                or len(set(self.ar_orders)) != len(self.ar_orders)):
            raise ValueError("Forecast windows must support the distinct AR orders")
        scales = (self.prior_intercept_sd, self.prior_expert_sd, self.prior_residual_sd, self.residual_sd_floor)
        if not all(np.isfinite(value) and value > 0 for value in scales):
            raise ValueError("Prior scales and the residual floor must be positive and finite")
        if not 0 < self.state_discount <= 1 or not 0 < self.residual_discount < 1:
            raise ValueError("Invalid state or residual discount")
        if not np.isfinite(self.student_df) or not self.student_df > 2:
            raise ValueError("Student-t needs finite degrees of freedom greater than two")
        if not 0.5 < self.cvar_confidence < 1 or not 0 < self.max_weight <= 1:
            raise ValueError("Invalid CVaR confidence or concentration cap")


@dataclass(frozen=True)
class Execution:
    lag_sessions: int
    cost_bps: float
    slippage_bps: float
    distribution_delay_days: int
    initial_value: float

    def __post_init__(self):
        if type(self.lag_sessions) is not int or self.lag_sessions < 1:
            raise ValueError("Execution lag must be at least one trading session")
        if type(self.distribution_delay_days) is not int or self.distribution_delay_days < 1:
            raise ValueError("Distribution payment delay must be positive")
        if not (0 <= self.cost_bps < 10_000 and 0 <= self.slippage_bps < 10_000):
            raise ValueError("Cost and slippage must each be in [0, 10000) bps")
        if not np.isfinite(self.initial_value) or self.initial_value <= 0:
            raise ValueError("Initial capital must be positive and finite")

    @property
    def buy_multiplier(self):
        return (1 + self.slippage_bps / 10_000) * (1 + self.cost_bps / 10_000)

    @property
    def sell_multiplier(self):
        return (1 - self.slippage_bps / 10_000) * (1 - self.cost_bps / 10_000)


def validate_quotes(quotes, symbols, sessions):
    if not quotes.index.equals(sessions) or quotes.columns.has_duplicates:
        missing = sessions.difference(quotes.index).strftime("%Y-%m-%d").tolist()
        unexpected = quotes.index.difference(sessions).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Quote/calendar mismatch. Missing {missing[:8]}, unexpected {unexpected[:8]}")
    columns = pd.MultiIndex.from_product([QUOTE_FIELDS, symbols])
    quotes = quotes.reindex(columns=columns).astype(float)
    bad = ~np.isfinite(quotes) | (quotes < 0)
    for field in ("Open", "Close"):
        bad[field] = bad[field] | (quotes[field] == 0)
    if bad.to_numpy().any():
        row, col = np.argwhere(bad.to_numpy())[0]
        raise ValueError(f"Invalid quote or action for {columns[col]} on {sessions[row].date()}")
    return quotes


def read_snapshot(snapshot, symbols, sessions):
    header, body = snapshot.decode("utf-8").split("\n", 1)
    if not header.startswith("# fetched_at="):
        raise ValueError("Snapshot is missing its acquisition time")
    stamp = header.removeprefix("# fetched_at=")
    if datetime.fromisoformat(stamp).utcoffset() != pd.Timedelta(0):
        raise ValueError("Snapshot acquisition time must be UTC")
    quotes = pd.read_csv(StringIO(body), header=[0, 1], index_col=0, parse_dates=[0])
    return validate_quotes(quotes, symbols, sessions), stamp, sha256(snapshot).hexdigest()


def load_quotes(symbols, start, end):
    sessions = expected_sessions(start, end)
    CACHE.mkdir(exist_ok=True)
    cache_key = f"yf-raw-actions-v1|{start.date()}|{end.date()}|{'|'.join(symbols)}"
    path = CACHE / f"quotes_{sha256(cache_key.encode()).hexdigest()[:16]}.csv"
    if path.is_file():
        try:
            return read_snapshot(path.read_bytes(), symbols, sessions)
        except (ValueError, OSError) as error:
            print(f"CACHE_RELOAD: {error}", flush=True)

    # Avoid persisting provider cookies or timezone databases alongside the price snapshot.
    with TemporaryDirectory(prefix="yf_", dir=CACHE) as temporary:
        yf.set_tz_cache_location(temporary)
        quotes = yf.download(
            symbols, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=False, actions=True, repair=False,
            ignore_tz=True, keepna=True, threads=True, progress=False, timeout=12,
        )
    quotes = validate_quotes(quotes, symbols, sessions)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as out:
        temporary_path = Path(out.name)
        try:
            out.write(f"# fetched_at={stamp}\n")
            quotes.to_csv(out, index_label="date")
            out.flush()
            loaded = read_snapshot(temporary_path.read_bytes(), symbols, sessions)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return loaded


def split_adjusted_panels(quotes):
    closes = quotes["Close"]
    distributions = quotes["Dividends"] + quotes["Capital Gains"]
    # Yahoo has already expressed OHLC and cash distributions in split-adjusted units.
    # Check that a reported split does not also appear as a near-mechanical price jump.
    splits = quotes["Stock Splits"]
    events = (splits > 0) & (splits != 1) & closes.shift().notna()
    gross = (closes + distributions) / closes.shift()
    expected_jump = 1 / splits.where(events)
    if (events & (np.abs(np.log(gross / expected_jump)) + 0.05 < np.abs(np.log(gross)))).to_numpy().any():
        raise ValueError("A reported split appears inconsistent with split-adjusted prices")
    return quotes["Open"], closes, distributions, gross


def expert_forecasts(history, model):
    """Estimate each expert's conditional mean and its estimation variance."""
    sample = history[-model.long_lookback_months:]
    short = sample[-model.short_mean_months:]
    means = [short.mean(axis=0), sample.mean(axis=0)]
    variances = [window.var(axis=0, ddof=1) / len(window) for window in (short, sample)]
    for order in model.ar_orders:
        ar_mean, ar_var = [], []
        for values in sample.T:
            # Each completed target at row i has features from rows i-1 through i-order.
            design = np.column_stack([
                np.ones(len(values) - order),
                *[values[order - lag:len(values) - lag] for lag in range(1, order + 1)],
            ])
            target = values[order:]
            next_row = np.r_[1.0, values[-order:][::-1]]
            coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
            if rank < design.shape[1]:
                ar_mean.append(values.mean())
                ar_var.append(values.var(ddof=1) / len(values))
                continue
            residual = target - design @ coefficients
            leverage = np.linalg.norm(next_row @ np.linalg.pinv(design)) ** 2
            ar_mean.append(next_row @ coefficients)
            ar_var.append((residual @ residual) / (len(target) - rank) * leverage)
        means.append(ar_mean)
        variances.append(ar_var)
    return np.column_stack(means), np.column_stack(variances)


class Synthesis:
    """Joint Gaussian moment projection for the dynamic coefficients."""

    def __init__(self, asset_count, model):
        experts = 2 + len(model.ar_orders)
        self.block_size = experts + 1
        self.model = model
        self.state = np.tile(np.r_[0, np.full(experts, 1 / experts)], asset_count)
        prior_sd = np.r_[model.prior_intercept_sd, np.full(experts, model.prior_expert_sd)]
        self.state_cov = np.diag(np.tile(prior_sd**2, asset_count))
        self.residual_cov = np.eye(asset_count) * model.prior_residual_sd**2
        self.design = np.zeros((asset_count, self.state.size))
        self.asset_rows = np.arange(asset_count)[:, None]
        self.asset_blocks = self.asset_rows * self.block_size + np.arange(self.block_size)

    def forecast(self, expert_means, expert_variances):
        self.design[self.asset_rows, self.asset_blocks] = np.column_stack([
            np.ones(len(expert_means)), expert_means,
        ])
        prior_cov = self.state_cov / self.model.state_discount
        coefficient_second_moment = (self.state**2 + np.diag(prior_cov)).reshape(-1, self.block_size)
        agent_noise = np.sum(coefficient_second_moment[:, 1:] * expert_variances, axis=1)
        noise_cov = self.residual_cov + np.diag(agent_noise)
        state_cross_cov = prior_cov @ self.design.T
        mean = self.design @ self.state
        covariance = self.design @ state_cross_cov + noise_cov
        return mean, covariance, prior_cov, state_cross_cov, noise_cov

    def observe(self, realized, prediction):
        mean, covariance, prior_cov, state_cross_cov, noise_cov = prediction
        innovation = realized - mean
        gain = np.linalg.solve(covariance, state_cross_cov.T).T
        self.state += gain @ innovation
        transform = np.eye(len(self.state)) - gain @ self.design
        self.state_cov = transform @ prior_cov @ transform.T + gain @ noise_cov @ gain.T
        self.state_cov = (self.state_cov + self.state_cov.T) / 2

        # Discount the conditional second moment of the observation residual.
        residual_gain = np.linalg.solve(covariance, self.residual_cov).T
        residual_mean = residual_gain @ innovation
        conditional = self.residual_cov - residual_gain @ self.residual_cov
        second_moment = conditional + np.outer(residual_mean, residual_mean)
        discount = self.model.residual_discount
        blended = discount * self.residual_cov + (1 - discount) * second_moment
        values, vectors = np.linalg.eigh((blended + blended.T) / 2)
        self.residual_cov = (vectors * np.maximum(values, self.model.residual_sd_floor**2)) @ vectors.T


def tail_weights(mean, covariance, model):
    """Minimize expected loss in the worst tail of a covariance-matched Student-t."""
    tail = 1 - model.cvar_confidence
    df = model.student_df
    quantile = student_t.ppf(model.cvar_confidence, df)
    tail_factor = (student_t.pdf(quantile, df) * (df + quantile**2) / ((df - 1) * tail)
                   * np.sqrt((df - 2) / df))

    def loss(weights):
        marginal = covariance @ weights
        sigma = np.sqrt(weights @ marginal)
        return -weights @ mean + tail_factor * sigma, -mean + tail_factor * marginal / sigma

    initial = np.full(mean.size, 1 / mean.size)
    if model.max_weight == initial[0]:
        return initial
    solved = minimize(
        loss, initial, jac=True, method="SLSQP", bounds=[(0, model.max_weight)] * mean.size,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
        options={"ftol": 1e-12, "maxiter": 100},
    )
    if not solved.success or not np.isclose(solved.x.sum(), 1, atol=1e-9, rtol=0):
        raise RuntimeError(f"CVaR allocation failed: {solved.message}")
    weights = np.clip(solved.x, 0, model.max_weight)
    weights[weights < 1e-12] = 0
    return weights / weights.sum()


def scheduled_signals(sessions, sample_start, lag):
    calendar = expected_sessions(sessions[0], sessions[-1] + pd.offsets.MonthEnd(0))
    month_ends = calendar.to_series().groupby(calendar.to_period("M")).last()
    return {
        day: sessions[i + lag]
        for i, day in enumerate(sessions)
        if day in month_ends.values and i + lag < len(sessions) and sessions[i + lag] >= sample_start
    }


def model_targets(daily_gross, schedule, model):
    # An incomplete final month can enter this series, but never gets a target or a later fit.
    monthly = daily_gross.groupby(daily_gross.index.to_period("M")).prod(min_count=1) - 1
    # The acquisition start may cut the first month short or lack its preceding close.
    monthly = monthly.iloc[1:]
    end_dates = daily_gross.index.to_series().groupby(daily_gross.index.to_period("M")).last()
    values = monthly.to_numpy()
    synthesis = Synthesis(values.shape[1], model)
    predictions = None
    updates = 0
    targets = {}
    for row in range(model.long_lookback_months - 1, len(values)):
        if predictions is not None:
            synthesis.observe(values[row], predictions)
            updates += 1
        expert_means, expert_vars = expert_forecasts(values[:row + 1], model)
        predictions = synthesis.forecast(expert_means, expert_vars)
        day = end_dates.loc[monthly.index[row]]
        if day in schedule and updates >= model.min_filter_updates:
            mean, covariance, *_ = predictions
            targets[schedule[day]] = tail_weights(mean, covariance, model)
    if not targets or next(iter(targets)) != next(iter(schedule.values()), None):
        raise ValueError("Insufficient data before the first planned fill. Extend the warm-up.")
    return targets


def open_rebalance(positions, cash, open_prices, weights, execution):
    """Fund opening purchases from cleared cash and sale proceeds at the same open."""
    opening_values = positions * open_prices
    liquid = cash + opening_values.sum()
    buy_multiplier, sell_multiplier = execution.buy_multiplier, execution.sell_multiplier

    def cash_left(deltas):
        sales = -np.minimum(deltas, 0).sum()
        buys = np.maximum(deltas, 0).sum()
        return cash + sales * sell_multiplier - buys * buy_multiplier

    lower, upper = 0.0, liquid
    for _ in range(48):
        midpoint = (lower + upper) / 2
        if cash_left(weights * midpoint - opening_values) >= 0:
            lower = midpoint
        else:
            upper = midpoint
    target_values = weights * lower
    deltas = target_values - opening_values
    remaining = cash_left(deltas)
    if not remaining >= 0:
        raise RuntimeError("Opening purchases exceed available cash")
    return target_values / open_prices, remaining, deltas, liquid


def simulate(opens, closes, distributions, targets, start, execution):
    first = opens.index.searchsorted(start)
    days = opens.index[first:]
    delay = pd.Timedelta(days=execution.distribution_delay_days)
    opens = opens.to_numpy()[first:]
    closes = closes.to_numpy()[first:]
    distributions = distributions.to_numpy()[first:]
    positions = np.zeros(opens.shape[1])
    cash = value = execution.initial_value
    receivables = deque()
    unpaid = 0.0
    daily_results = []
    for day, open_prices, close_prices, payouts in zip(days, opens, closes, distributions, strict=True):
        while receivables and receivables[0][0] <= day:
            _, amount = receivables.popleft()
            cash += amount
            unpaid -= amount
        # Ex-date entitlement belongs to the positions carried into the session.
        entitlement = positions @ payouts
        if entitlement > 0:
            receivables.append((day + delay, entitlement))
            unpaid += entitlement
        turnover = trades = 0
        if day in targets:
            positions, cash, deltas, liquid = open_rebalance(positions, cash, open_prices, targets[day], execution)
            turnover = float(np.abs(deltas).sum() / liquid)
            trades = int(np.count_nonzero(np.abs(deltas) > liquid * 1e-10))
        close_value = cash + positions @ close_prices + unpaid
        daily_results.append((close_value / value - 1, turnover, trades, cash))
        value = close_value
    return pd.DataFrame(daily_results, index=days, columns=["return", "turnover", "n_trades", "cash"])


def metrics(daily_results, cash_returns):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise ValueError("Each reporting period needs at least two sessions")
    excess = returns - cash_returns.loc[returns.index]
    growth = np.r_[1.0, (1 + returns.to_numpy()).cumprod()]
    volatility = returns.std()
    excess_vol = excess.std()
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(volatility * annualizer),
        "sharpe": float(returns.mean() / volatility * annualizer) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * annualizer) if excess_vol > 0 else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def summarize(daily_results, cash_returns, is_end):
    before = daily_results.index <= is_end
    return {
        "is": metrics(daily_results.loc[before], cash_returns),
        "oos": metrics(daily_results.loc[~before], cash_returns),
        "full": metrics(daily_results, cash_returns),
    }


def evaluate(quotes, universe, sample_start, model, execution, *, equal_weight=False):
    opens, closes, distributions, daily_gross = split_adjusted_panels(quotes)
    schedule = scheduled_signals(opens.index, sample_start, execution.lag_sessions)
    if not schedule:
        raise ValueError("No scheduled opening fills in the reporting period")
    if equal_weight:
        targets = {day: np.full(len(universe), 1 / len(universe)) for day in schedule.values()}
    else:
        targets = model_targets(daily_gross[universe], schedule, model)
    daily_results = simulate(
        opens[universe], closes[universe], distributions[universe], targets, sample_start, execution,
    )
    return daily_results, targets, schedule


def causal_checks(quotes, universe, sample_start, is_end, model, execution,
                  is_results, is_targets, full_results, full_targets, schedule):
    pd.testing.assert_frame_equal(is_results, full_results.loc[:is_end], check_exact=True)
    if is_targets.keys() != {day for day in full_targets if day <= is_end}:
        raise RuntimeError("IS and full runs disagree on IS fill dates")
    for day, weights in is_targets.items():
        np.testing.assert_array_equal(weights, full_targets[day])

    # Mimic a later split rescaling every earlier field expressed per share.
    scaled = quotes.loc[:is_end].copy()
    for field in ("Open", "Close", "Dividends", "Capital Gains"):
        scaled[(field, universe[0])] *= 0.25
    scaled_results, scaled_targets, _ = evaluate(scaled, universe, sample_start, model, execution)
    pd.testing.assert_frame_equal(is_results, scaled_results, check_exact=False, rtol=1e-10, atol=1e-9)
    for day, weights in is_targets.items():
        np.testing.assert_allclose(weights, scaled_targets[day], atol=1e-10, rtol=0)

    # Shock a later fill's close and ex-date payout. The preceding positions can
    # earn a receivable, but that payout and close cannot change opening purchases.
    fill_days = iter(full_targets)
    first_fill = next(fill_days)
    shock_day = next(fill_days, first_fill)
    symbol = universe[int(np.argmax(full_targets[first_fill]))]
    shocked = quotes.loc[:shock_day].copy()
    shocked.loc[shock_day, ("Close", symbol)] *= 1.15
    shocked.loc[shock_day, ("Dividends", symbol)] += 1.0
    shocked_results, shocked_targets, _ = evaluate(shocked, universe, sample_start, model, execution)
    np.testing.assert_array_equal(full_targets[shock_day], shocked_targets[shock_day])
    cols = ["turnover", "n_trades", "cash"]
    pd.testing.assert_series_equal(
        full_results.loc[shock_day, cols], shocked_results.loc[shock_day, cols], check_exact=True,
    )
    for day, fill in schedule.items():
        if quotes.index.get_loc(fill) - quotes.index.get_loc(day) != execution.lag_sessions:
            raise RuntimeError("Signal and fill dates violate the configured lag")
    if (full_results["cash"] < 0).any() or not np.isfinite(full_results.to_numpy()).all():
        raise RuntimeError("Invalid simulated values or negative cash")


def print_metrics(rows):
    print(pd.DataFrame.from_dict(rows, orient="index").to_string(float_format=lambda value: f"{value:.6f}"))


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    cash_proxy = config["risk_free_proxy"]
    if (cash_proxy != "BIL" or len(universe) < 2 or len(set(universe)) != len(universe)
            or cash_proxy in universe or not all(re.fullmatch("[A-Z]{1,5}", name) for name in universe)):
        raise ValueError("Use distinct US ETF symbols and a separate BIL risk-free proxy")
    bounds = config["dates"]
    data_start, sample_start, is_end, sample_end = (
        pd.Timestamp(bounds[name]) for name in ("data_start", "sample_start", "is_end", "sample_end")
    )
    if not pd.Timestamp("2007-05-31") <= data_start < sample_start < is_end < sample_end:
        raise ValueError("Dates must be increasing and within BIL's history")
    if sample_end >= pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None):
        raise ValueError("Sample end must precede the current US trading date")
    model_config = config["model"]
    model = Model(**(model_config | {"ar_orders": tuple(model_config["ar_orders"])}))
    if model.max_weight < 1 / len(universe):
        raise ValueError("The weight cap makes full investment impossible")
    execution = Execution(**config["execution"])
    cases = config["robustness"]
    alternative_models = {
        f"CVaR {confidence:.1%}": replace(model, cvar_confidence=confidence)
        for confidence in cases["cvar_confidences"]
    }
    high_friction = replace(execution, cost_bps=cases["higher_cost_bps"], slippage_bps=cases["higher_slippage_bps"])
    if (len(alternative_models) != len(cases["cvar_confidences"])
            or len(alternative_models) < 2 or model in alternative_models.values()
            or high_friction.buy_multiplier <= execution.buy_multiplier
            or high_friction.sell_multiplier >= execution.sell_multiplier):
        raise ValueError("Robustness needs distinct confidence levels and higher execution friction")

    quotes, fetched_at, digest = load_quotes([*universe, cash_proxy], data_start, sample_end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"DATA_SHA256: {digest}")
    print("PRICE_ADJUSTMENT: yfinance split-adjusted Open/Close; ex-date cash distributions in total-return "
          "signals and BIL; simulation accrues distributions separately")
    _, _, _, is_gross = split_adjusted_panels(quotes.loc[:is_end])
    is_cash_returns = is_gross[cash_proxy] - 1
    is_quotes = quotes.loc[:is_end]
    baseline_is, is_targets, _ = evaluate(is_quotes, universe, sample_start, model, execution)
    robustness = {"default": metrics(baseline_is, is_cash_returns)}
    for name, alternate in alternative_models.items():
        alternate_results, _, _ = evaluate(is_quotes, universe, sample_start, alternate, execution)
        robustness[name] = metrics(alternate_results, is_cash_returns)
    alternate_results, _, _ = evaluate(is_quotes, universe, sample_start, model, high_friction)
    robustness["higher cost + slippage"] = metrics(alternate_results, is_cash_returns)
    print("ROBUSTNESS | IS only | official configuration fixed")
    print_metrics(robustness)
    print(f"PARAM_COMBOS_TRIED: {len(robustness)}")

    daily_results, targets, schedule = evaluate(quotes, universe, sample_start, model, execution)
    benchmark_results, _, _ = evaluate(quotes, universe, sample_start, model, execution, equal_weight=True)
    causal_checks(
        quotes, universe, sample_start, is_end, model, execution,
        baseline_is, is_targets, daily_results, targets, schedule,
    )
    if (benchmark_results["cash"] < 0).any():
        raise RuntimeError("Benchmark has negative cash")
    if daily_results["n_trades"].sum() <= 0:
        raise RuntimeError("The official simulation made no trades")

    _, _, _, gross = split_adjusted_panels(quotes)
    cash_returns = gross[cash_proxy] - 1
    periods = summarize(daily_results, cash_returns, is_end)
    benchmark = summarize(benchmark_results, cash_returns, is_end)
    data_range = f"{daily_results.index[0].date()} to {daily_results.index[-1].date()}"
    print(f"SAMPLE: {data_range} | IS through {is_end.date()} | first fill {next(iter(targets)).date()}")
    print(f"EXECUTION: lag={execution.lag_sessions} sessions | sizing=fill-session open | "
          f"fill=open | commission={execution.cost_bps:g} bps | slippage={execution.slippage_bps:g} bps")
    print("LOOK_AHEAD_CHECK: passed signal lag, price-only availability, split scaling, "
          "fill-day close and distribution shocks, and independently truncated IS replay")
    print(f"MIN_CASH: strategy={daily_results['cash'].min():.8f} | benchmark={benchmark_results['cash'].min():.8f}")
    print("已回测 | pending review | strategy and monthly equal-weight benchmark | excess Sharpe vs BIL")
    print_metrics({
        **{f"Strategy {period.upper()}": summary for period, summary in periods.items()},
        **{f"Equal weight {period.upper()}": summary for period, summary in benchmark.items()},
    })
    result = {
        "strategy": config["strategy"], "data_range": data_range, **periods,
        "benchmark": {"name": "Monthly equal weight", **benchmark["full"], **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(840)
    main()
