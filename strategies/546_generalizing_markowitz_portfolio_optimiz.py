# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "546_generalizing_markowitz_portfolio_optimiz"\nuniverse: [XLK, XLF, XLE, XLV]\nbenchmark: SPY\ncash_proxy: BIL\ndata_start: "2008-01-01"\nsample_start: "2010-01-01"\nis_end: "2017-12-31"\nsample_end: "2025-12-31"\nsignal_lag_sessions: 1  # Fill at this many sessions after the signal close. Lock shares at the preceding close.\n\nmodel:\n  lookback_sessions: 252\n  risk_aversion: 5.0\n  quadratic_turnover_penalty: 0.20\n  covariance_ridge: 0.001\n  transaction_cost_bps: 5.0\n  slippage_bps: 2.0\n  cash_buffer_bps: 300.0  # Reserve from prior-close positions and spendable cash, before allowing for both sides\' fees.\n  cash_distribution_delay_days: 60  # Assumed calendar days from ex-date. Release only after the eligible session\'s fills.\n\n# Each case inherits the default model and runs through the IS cutoff only.\nrobustness:\n  lookback_shorter: {lookback_sessions: 210}\n  lookback_longer: {lookback_sessions: 294}\n  risk_aversion_lower: {risk_aversion: 4.0}\n  risk_aversion_higher: {risk_aversion: 6.0}\n  turnover_penalty_lower: {quadratic_turnover_penalty: 0.15}\n  turnover_penalty_higher: {quadratic_turnover_penalty: 0.25}\n  higher_cost_and_slippage: {transaction_cost_bps: 10.0, slippage_bps: 5.0}\n'
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
"""Monthly long-only allocation using a generalized quadratic risk measure."""

import hashlib
import json
import os
import signal
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)
from scipy.optimize import minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
TRADE_TOLERANCE = 1e-8
QUOTE_FIELDS = ("Close", "Adj Close", "Dividends", "Capital Gains", "Stock Splits")
SECTOR_ETFS = frozenset({"XLK", "XLF", "XLE", "XLV"})
SPECIAL_CLOSURES = pd.to_datetime(["2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])
XLF_EX_DATE = pd.Timestamp("2016-09-19")
XLRE_SALE_DATE = pd.Timestamp("2016-09-23")  # First session after the September 22 pay date.
# XLF declared the ratio on September 16, as recorded in SEC filing 0001104659-16-148813.
XLRE_PER_XLF_SHARE = 0.139146


class USEquityHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


@dataclass(frozen=True)
class Model:
    lookback_sessions: int
    risk_aversion: float
    quadratic_turnover_penalty: float
    covariance_ridge: float
    transaction_cost_bps: float
    slippage_bps: float
    cash_buffer_bps: float
    cash_distribution_delay_days: int

    def __post_init__(self):
        if type(self.lookback_sessions) is not int or self.lookback_sessions < 2:
            raise ValueError("lookback_sessions must be an integer of at least two")
        positive = (self.risk_aversion, self.covariance_ridge)
        nonnegative = (self.quadratic_turnover_penalty, self.transaction_cost_bps, self.slippage_bps)
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise ValueError("Risk aversion and covariance ridge must be finite and positive")
        if not all(np.isfinite(value) and value >= 0 for value in nonnegative) or self.fee_rate >= 1:
            raise ValueError("Penalties and fees must be finite, nonnegative and fees must total under 100%")
        if not np.isfinite(self.cash_buffer_bps) or not 0 < self.cash_buffer_bps < 10_000:
            raise ValueError("cash_buffer_bps must be strictly between 0 and 10000")
        if type(self.cash_distribution_delay_days) is not int or self.cash_distribution_delay_days < 1:
            raise ValueError("cash_distribution_delay_days must be a positive integer")

    @property
    def fee_rate(self):
        return (self.transaction_cost_bps + self.slippage_bps) / 10_000


def expected_sessions(start, end):
    holidays = USEquityHolidays().holidays(start, end).union(SPECIAL_CLOSURES)
    return pd.bdate_range(start, end).difference(holidays)


def prepare_quotes(quotes, tickers, sessions):
    dates = quotes.index
    if (quotes.empty or not isinstance(dates, pd.DatetimeIndex) or dates.tz is not None
            or not dates.is_unique or not dates.is_monotonic_increasing):
        raise ValueError("Expected a nonempty, sorted panel of unique daily quotes")
    if not dates.equals(sessions):
        missing = sessions.difference(dates).strftime("%Y-%m-%d").tolist()
        extra = dates.difference(sessions).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"NYSE Arca session mismatch: {len(missing)} missing {missing[:5]}, {len(extra)} unexpected {extra[:5]}")
    columns = pd.MultiIndex.from_product([QUOTE_FIELDS, tickers])
    quotes = quotes.reindex(columns=columns).astype(float)
    invalid = ~np.isfinite(quotes) | (quotes < 0)
    invalid.loc[:, ["Close", "Adj Close"]] |= quotes.loc[:, ["Close", "Adj Close"]] == 0
    if invalid.any().any():
        date, column = np.argwhere(invalid.to_numpy())[0]
        field, ticker = columns[column]
        raise ValueError(f"Missing or invalid {field} for {ticker} on {dates[date].date()}")
    return quotes


def read_snapshot(snapshot, tickers, sessions):
    with BytesIO(snapshot) as source:
        header = source.readline().decode("utf-8").strip()
        if not header.startswith("# fetched_at="):
            raise ValueError("Expected a fetch timestamp in the first line of the snapshot")
        fetched_at = header.removeprefix("# fetched_at=")
        if datetime.fromisoformat(fetched_at).utcoffset() != timedelta(0):
            raise ValueError("Expected a UTC fetch time")
        quotes = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=[0])
    return prepare_quotes(quotes, tickers, sessions), fetched_at, hashlib.sha256(snapshot).hexdigest()


def load_quotes(tickers, start, end):
    sessions = expected_sessions(start, end)
    CACHE.mkdir(exist_ok=True)
    cache_key = "|".join(["yf-close-actions-v1", start.date().isoformat(), end.date().isoformat(), *tickers])
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    path = CACHE / f"close_actions_{digest}.csv"
    if path.is_file():
        try:
            return read_snapshot(path.read_bytes(), tickers, sessions)
        except (OSError, ValueError) as error:
            print(f"CACHE_RELOAD: {error}", flush=True)

    downloaded = yf.download(
        tickers, start=start.date().isoformat(), end=(end + timedelta(days=1)).date().isoformat(),
        interval="1d", auto_adjust=False, actions=True, ignore_tz=True, keepna=True,
        threads=False, progress=False, timeout=12,
    )
    if downloaded is None or downloaded.empty or "Close" not in downloaded:
        raise RuntimeError("yfinance did not return daily quotes")
    quotes = prepare_quotes(downloaded, tickers, sessions)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as out:
        temporary_path = Path(out.name)
        try:
            out.write(f"# fetched_at={fetched_at}\n")
            quotes.to_csv(out, index_label="date")
            out.close()
            snapshot = read_snapshot(temporary_path.read_bytes(), tickers, sessions)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return snapshot


def load_price_panels(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    xlre_closes = None
    # Both downloads share one temporary yfinance cache for this process.
    with TemporaryDirectory(prefix="yf_", dir=CACHE) as yf_cache:
        yf.set_tz_cache_location(yf_cache)
        try:
            quotes, fetched_at, checksum = load_quotes(tickers, start, end)
            print(f"DATA_FETCHED_AT: {fetched_at}")
            print(f"DATA_SHA256: {checksum}")
            if "XLF" in tickers and XLF_EX_DATE in quotes.index:
                xlre_quotes, xlre_fetched_at, xlre_checksum = load_quotes(
                    ["XLRE"], XLF_EX_DATE, min(XLRE_SALE_DATE, end),
                )
                if (xlre_quotes.loc[:, ["Dividends", "Capital Gains", "Stock Splits"]] != 0).any().any():
                    raise ValueError("Unexpected XLRE action during the XLF distribution")
                xlre_closes = xlre_quotes["Close", "XLRE"]
                print(f"XLRE_DATA_FETCHED_AT: {xlre_fetched_at}")
                print(f"XLRE_DATA_SHA256: {xlre_checksum}")
        finally:
            yf.set_tz_cache_location(str(CACHE))
    return quotes, xlre_closes


@dataclass(frozen=True)
class XLFDistribution:
    xlre_shares_per_xlf_unit: float
    xlre_closes: pd.Series


def execution_prices(quotes, xlre_closes):
    # Yahoo Close uses split-adjusted units, so an ordinary split leaves those units unchanged.
    closes = quotes["Close"].copy()
    distributions = quotes["Dividends"] + quotes["Capital Gains"]
    if xlre_closes is None:
        return closes, distributions, None

    # Yahoo encodes the distribution as a split. Keep the pre-event quoted
    # units throughout, where one normalized unit represents 1 / factor XLF shares.
    factor = quotes.loc[XLF_EX_DATE, ("Stock Splits", "XLF")]
    if not 1 < factor < 2 or distributions.at[XLF_EX_DATE, "XLF"] != 0:
        raise ValueError("Expected Yahoo's XLF in-kind distribution in its split column only")
    closes.loc[XLF_EX_DATE:, "XLF"] /= factor
    distributions.loc[XLF_EX_DATE:, "XLF"] /= factor
    return closes, distributions, XLFDistribution(XLRE_PER_XLF_SHARE / factor, xlre_closes)


def budget_solution(quadratic, reward):
    """Proposition 8 with reward = mean - linear risk term and sum(weights) = 1."""
    ones = np.ones(len(reward))
    center, direction = np.linalg.solve(quadratic, np.column_stack((reward, ones))).T
    return center + direction * ((1.0 - center.sum()) / direction.sum())


def target_weights(window, reference, model):
    mean = window.mean(axis=0) * SESSIONS_PER_YEAR
    covariance = np.cov(window, rowvar=False) * SESSIONS_PER_YEAR
    penalty = model.quadratic_turnover_penalty
    quadratic = model.risk_aversion * covariance + (model.covariance_ridge + penalty) * np.eye(len(mean))
    reward = mean + penalty * reference
    unrestricted = budget_solution(quadratic, reward)
    if unrestricted.min() >= 0:
        return unrestricted

    # Projection in the quadratic metric preserves the original utility objective.
    def distance(weights):
        difference = weights - unrestricted
        gradient = quadratic @ difference
        return 0.5 * (difference @ gradient), gradient

    ones = np.ones(len(mean))
    start = unrestricted.clip(min=0.0)
    result = minimize(
        distance, start / start.sum(), jac=True, method="SLSQP",
        bounds=[(0.0, 1.0)] * len(mean),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1, "jac": lambda _: ones},
        options={"ftol": 1e-13, "maxiter": 150},
    )
    if not result.success or abs(result.x.sum() - 1) > 1e-8:
        raise RuntimeError(f"Long-only optimization failed: {result.message}")
    weights = result.x.clip(min=0.0)
    return weights / weights.sum()


def size_at_close(positions, sizing_close, cash, target, model):
    """Lock signed share changes for the next session's close."""
    position_value = positions @ sizing_close
    available_value = position_value + cash
    reserve = model.cash_buffer_bps / 10_000
    # Budget for selling all existing units and buying the whole new basket.
    # Actual buys and sells can be smaller when a position is carried forward.
    spendable_value = available_value * (1 - reserve)
    target_value = max(0.0, (spendable_value - position_value * model.fee_rate) / (1 + model.fee_rate))
    share_changes = target_value * target / sizing_close - positions
    share_changes[np.abs(share_changes) * sizing_close <= TRADE_TOLERANCE * available_value] = 0.0
    share_changes.setflags(write=False)
    return share_changes


def fill_at_close(share_changes, fill_close, cash, fee_rate, date):
    """Price the fixed share changes and stop before posting an unaffordable fill."""
    trade_values = share_changes * fill_close
    gross_traded = np.abs(trade_values).sum()
    cash_after_fills = cash - trade_values.sum() - gross_traded * fee_rate
    if not np.isfinite(cash_after_fills) or cash_after_fills < 0:
        raise ArithmeticError(
            f"{date.date()}: cash after predetermined close fills would be {cash_after_fills:.12g} "
            f"(cash before fills {cash:.12g}, gross traded {gross_traded:.12g})"
        )
    return cash_after_fills, gross_traded


def simulate_positions(closes, distributions, first, model, fill_for_signal, choose_target, xlf_distribution=None):
    dates = closes.index
    close_values = closes.to_numpy()
    distribution_values = distributions.to_numpy()
    positions = np.zeros(close_values.shape[1])
    cash = previous_value = 1.0
    distribution_delay = timedelta(days=model.cash_distribution_delay_days)
    pending_distributions = deque()
    pending_targets = {}
    share_changes = None
    xlre_shares = 0.0
    if xlf_distribution is not None:
        xlf_index = closes.columns.get_loc("XLF")
        xlre_closes = xlf_distribution.xlre_closes
    daily_results = []
    for i in range(min(first, min(fill_for_signal, default=first)), len(dates)):
        close = close_values[i]
        date = dates[i]
        earned = positions @ distribution_values[i]
        if earned:
            pending_distributions.append((date + distribution_delay, earned))
        unpaid_distributions = sum(amount for _, amount in pending_distributions)
        if xlf_distribution is not None and date == XLF_EX_DATE:
            xlre_shares = positions[xlf_index] * xlf_distribution.xlre_shares_per_xlf_unit
        xlre_value = xlre_shares * xlre_closes.at[date] if xlre_shares else 0.0
        before_fill = positions @ close + cash + unpaid_distributions + xlre_value
        turnover, trades = 0.0, 0
        if share_changes is not None:
            cash, gross_traded = fill_at_close(share_changes, close, cash, model.fee_rate, date)
            positions += share_changes
            turnover = gross_traded / before_fill
            trades = np.count_nonzero(share_changes)
        if xlre_shares and date == XLRE_SALE_DATE:
            # Sell the entire received quantity. The close only values the sale.
            cash += xlre_value * (1 - model.fee_rate)
            turnover += xlre_value / before_fill
            trades += 1
            xlre_shares = xlre_value = 0.0
        position_values = positions * close
        value = position_values.sum() + cash + unpaid_distributions + xlre_value
        if i >= first:
            cash_fraction = cash / value if trades else np.nan
            daily_results.append((value / previous_value - 1, turnover, trades, cash_fraction))
        previous_value = value

        # Release ordinary distributions after fills on the first eligible session.
        while pending_distributions and pending_distributions[0][0] <= date:
            cash += pending_distributions.popleft()[1]
        if i in fill_for_signal:
            pending_targets[fill_for_signal[i]] = choose_target(i, position_values / value)
        # Lock quantities now for the next session's close, even when the signal was earlier.
        next_target = pending_targets.pop(i + 1, None)
        share_changes = None if next_target is None else size_at_close(positions, close, cash, next_target, model)

    return pd.DataFrame(daily_results, index=dates[first:], columns=[
        "return", "turnover", "n_trades", "cash_fraction_after_fills",
    ])


def simulate(asset_returns, closes, distributions, start, lag, model, xlf_distribution=None):
    dates = asset_returns.index
    values = asset_returns.to_numpy()
    first = int(dates.searchsorted(start))
    # The quote index has already been matched to the exchange calendar.
    month_end = np.flatnonzero(dates[:-1].month != dates[1:].month)
    fill_for_signal = {
        i: i + lag for i in month_end
        if i >= model.lookback_sessions and first <= i + lag < len(dates)
    }

    def choose_target(i, reference):
        window = values[i + 1 - model.lookback_sessions : i + 1]
        return target_weights(window, reference, model)

    daily_results = simulate_positions(
        closes, distributions, first, model, fill_for_signal, choose_target, xlf_distribution,
    )
    return daily_results, next(iter(fill_for_signal.values()), first)


def metrics(daily_results, cash_returns):
    returns = daily_results["return"]
    if returns.empty:
        return dict.fromkeys(("cagr", "ann_vol", "sharpe", "sharpe_excess", "max_drawdown", "total_return")) | {
            "turnover": 0.0, "n_trades": 0,
        }
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    volatility = returns.std()
    excess = returns - cash_returns.loc[returns.index]
    excess_volatility = excess.std()
    growth = np.r_[1.0, (1 + returns.to_numpy()).cumprod()]
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(volatility * annualizer) if np.isfinite(volatility) else None,
        "sharpe": float(returns.mean() / volatility * annualizer) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_volatility * annualizer) if excess_volatility > 0 else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, cash_returns, is_end):
    in_sample = daily_results.index <= is_end
    return {
        "is": metrics(daily_results.loc[in_sample], cash_returns),
        "oos": metrics(daily_results.loc[~in_sample], cash_returns),
        "full": metrics(daily_results, cash_returns),
    }


def print_metrics(rows):
    print(pd.DataFrame.from_dict(rows, orient="index").to_string(float_format=lambda value: f"{value:.6f}"))


def minimum_fill_cash_fraction(daily_results):
    minimum = daily_results["cash_fraction_after_fills"].min()
    return f"{minimum:.6f}" if pd.notna(minimum) else "none"


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)

    universe = config["universe"]
    benchmark, cash_proxy = config["benchmark"], config["cash_proxy"]
    tickers = list(dict.fromkeys([*universe, benchmark, cash_proxy]))
    if not set(universe) <= SECTOR_ETFS or len(set(universe)) != len(universe) or len(universe) < 2:
        raise ValueError("The universe must contain at least two distinct ETFs among XLK, XLF, XLE and XLV")
    if benchmark != "SPY" or cash_proxy != "BIL":
        raise ValueError("Use SPY as the benchmark and BIL as the cash proxy")
    data_start, sample_start, is_end, sample_end = (
        pd.Timestamp(config[key]) for key in ("data_start", "sample_start", "is_end", "sample_end")
    )
    if not data_start < sample_start < is_end < sample_end:
        raise ValueError("Expected data_start < sample_start < is_end < sample_end")
    lag = config["signal_lag_sessions"]
    if type(lag) is not int or lag < 1:
        raise ValueError("signal_lag_sessions must be an integer of at least one")

    official = Model(**config["model"])
    alternatives = {name: replace(official, **overrides) for name, overrides in config["robustness"].items()}
    if len(set(alternatives.values()) | {official}) != 1 + len(alternatives):
        raise ValueError("Robustness models must be distinct from each other and from the default")

    quotes, xlre_closes = load_price_panels(tickers, data_start, sample_end)
    closes, distributions, xlf_distribution = execution_prices(quotes, xlre_closes)
    print("PRICE_ADJUSTMENT: Yahoo Adj Close for signals and BIL; split-adjusted Close for fills and marks, "
          "with cash distributions and the XLF in-kind XLRE distribution modeled separately")
    returns = quotes["Adj Close"].pct_change(fill_method=None)
    cash_returns = returns[cash_proxy]
    is_asset_returns = returns.loc[:is_end, universe]
    strategy_distribution = xlf_distribution if "XLF" in universe else None
    is_distribution = None
    if strategy_distribution is not None and XLF_EX_DATE <= is_end:
        is_distribution = replace(strategy_distribution, xlre_closes=strategy_distribution.xlre_closes.loc[:is_end])
    robustness = {}
    for name, model in alternatives.items():
        daily_results, _ = simulate(
            is_asset_returns, closes.loc[:is_end, universe], distributions.loc[:is_end, universe],
            sample_start, lag, model, is_distribution,
        )
        robustness[name] = metrics(daily_results, cash_returns.loc[:is_end])

    daily_results, first_scheduled_fill = simulate(
        returns[universe], closes[universe], distributions[universe], sample_start, lag, official, strategy_distribution,
    )
    first = closes.index.searchsorted(sample_start)
    benchmark_fill = max(1, first_scheduled_fill)
    benchmark_schedule = {benchmark_fill - 1: benchmark_fill} if benchmark_fill < len(closes) else {}
    benchmark_results = simulate_positions(
        closes[[benchmark]], distributions[[benchmark]], first, official, benchmark_schedule,
        lambda _i, _reference: np.ones(1),
    )
    strategy_summary = period_metrics(daily_results, cash_returns, is_end)
    benchmark_summary = period_metrics(benchmark_results, cash_returns, is_end)
    first_date, last_date = daily_results.index[[0, -1]]
    data_range = f"{first_date.date()} to {last_date.date()}"
    fills = daily_results.index[daily_results["n_trades"] > 0]
    first_fill = fills[0].date().isoformat() if len(fills) else "none"
    print(f"SAMPLE: {data_range} | IS through {is_end.date()} | first fill {first_fill}")
    print(f"EXECUTION: lag={lag} sessions | sizing=previous close | fill=close | "
          f"cash_buffer_bps={official.cash_buffer_bps:g}")
    print(f"MIN_POST_FILL_CASH_FRACTION: strategy={minimum_fill_cash_fraction(daily_results)} | "
          f"{benchmark}={minimum_fill_cash_fraction(benchmark_results)}")
    print(f"STRATEGY AND {benchmark} BENCHMARK | excess Sharpe vs {cash_proxy}")
    print_metrics({
        **{f"Strategy {period.upper()}": summary for period, summary in strategy_summary.items()},
        **{f"{benchmark} {period.upper()}": summary for period, summary in benchmark_summary.items()},
    })
    print("ROBUSTNESS | IS only | default fixed in config.yaml")
    print_metrics({"default": strategy_summary["is"], **robustness})
    print(f"PARAM_COMBOS_TRIED: {1 + len(robustness)}")
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **strategy_summary,
        "benchmark": {"name": benchmark, **benchmark_summary["full"], **benchmark_summary},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(570)
    main()
