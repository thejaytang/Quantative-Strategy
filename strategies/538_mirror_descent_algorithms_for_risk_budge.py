# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "538_mirror_descent_algorithms_for_risk_budge"\nuniverse: [SPY, QQQ, TLT, GLD]\nrisk_free: BIL\ndata_start: 2007-06-01\nstart: 2009-01-02\nis_end: 2018-12-31\nend: 2025-12-31\n\nlookback_sessions: 252\nexecution_lag_sessions: 1\ncash_buffer_bps: 200.0  # Prior-close cash reserve shared by strategy and benchmark.\ncost_bps: 2.0\nslippage_bps: 3.0\n\nsolver:\n  step_exponent: 0.55\n  max_iterations: 20000\n  contribution_tolerance: 0.000001\n\nrobustness:\n  lookback_sessions: [189, 315]\n  friction_multiplier: 3.0\n'
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
"""Monthly volatility risk budgeting with deterministic entropic mirror descent."""

import json
import signal
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
FETCH_HEADER = "# fetched_at="
SOURCE_HEADER = "# source="
CALENDAR_START = pd.Timestamp("2007-01-01")
CALENDAR_END = pd.Timestamp("2025-12-31")
SPECIAL_CLOSURES = pd.DatetimeIndex([
    "2007-01-02",  # Gerald Ford national day of mourning
    "2012-10-29", "2012-10-30",  # Hurricane Sandy
    "2018-12-05",  # George H.W. Bush national day of mourning
    "2025-01-09",  # Jimmy Carter national day of mourning
])


class ExchangeHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


@dataclass(frozen=True)
class Specification:
    lookback_sessions: int
    cost_bps: float
    slippage_bps: float

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def price_cache_path(tickers, start, end):
    symbols = "-".join(tickers)
    return CACHE_DIR / f"yf_adjusted_{symbols}_{start.date()}_{end.date()}.csv"


def validate_prices(prices, tickers, start, end):
    if start < CALENDAR_START or end > CALENDAR_END:
        raise ValueError("The exchange calendar covers 2007 through 2025.")
    if not set(tickers).issubset(prices.columns):
        raise ValueError("Adjusted closing prices are missing for one or more symbols.")
    dates = prices.index
    if not isinstance(dates, pd.DatetimeIndex) or not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Price history has invalid, unsorted or duplicate dates.")
    prices = prices.loc[start:end, tickers]
    holidays = ExchangeHolidays().holidays(start, end).union(SPECIAL_CLOSURES)
    expected = pd.bdate_range(start, end).difference(holidays)
    if expected.empty or not prices.index.equals(expected):
        missing = expected.difference(prices.index).strftime("%Y-%m-%d").tolist()
        unexpected = prices.index.difference(expected).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Price dates do not match the exchange calendar. "
                         f"Missing: {missing[:5]}; unexpected: {unexpected[:5]}.")
    values = prices.to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Price history contains missing or nonpositive closes.")
    return prices


def save_prices(path, prices, fetched_at, source):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                            prefix=".prices_", delete=False) as stream:
        candidate = Path(stream.name)
    try:
        with candidate.open("w", encoding="utf-8") as stream:
            stream.write(f"{FETCH_HEADER}{fetched_at}\n{SOURCE_HEADER}{source}\n")
            prices.to_csv(stream, index_label="date")
        digest = sha256(candidate.read_bytes()).hexdigest()
        candidate.replace(path)
        return digest
    finally:
        candidate.unlink(missing_ok=True)


def load_prices(tickers, start, end):
    path = price_cache_path(tickers, start, end)
    if path.is_file():
        contents = path.read_bytes()
        with StringIO(contents.decode("utf-8")) as stream:
            fetched_at = stream.readline().strip().removeprefix(FETCH_HEADER)
            source = stream.readline().strip().removeprefix(SOURCE_HEADER)
            if datetime.fromisoformat(fetched_at).tzinfo is None or not source.startswith("yfinance"):
                raise ValueError(f"Missing timestamp or yfinance provenance in {path.name}.")
            prices = pd.read_csv(stream, index_col=0, parse_dates=True,
                                 dtype=dict.fromkeys(tickers, str)).astype(float)
        return (validate_prices(prices, tickers, start, end), fetched_at,
                f"{source} (cached snapshot)", sha256(contents).hexdigest())

    import yfinance as yf

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=CACHE_DIR, prefix="yf_") as session_cache:
        yf.set_tz_cache_location(session_cache)
        bars = yf.download(
            tickers, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=True, ignore_tz=True, multi_level_index=True,
            progress=False, threads=False, timeout=10,
        )
    if bars is None or bars.empty or "Close" not in bars:
        raise RuntimeError("yfinance returned no adjusted closing prices.")
    prices = validate_prices(bars["Close"].sort_index(), tickers, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    source = "yfinance download"
    digest = save_prices(path, prices, fetched_at, source)
    return prices, fetched_at, source, digest


def risk_budget_weights(history, *, step_exponent, max_iterations, contribution_tolerance):
    covariance = np.cov(history, rowvar=False, ddof=1)
    variances = covariance.diagonal()
    if not np.isfinite(covariance).all() or np.min(variances) <= 0:
        raise ValueError("Each asset must have positive finite volatility.")

    # Scaling covariance changes the optimizer's scale but leaves its proportions intact.
    covariance = covariance / variances.mean()
    smallest_eigenvalue = np.linalg.eigvalsh(covariance)[0]
    if smallest_eigenvalue <= 0:
        raise ValueError("The covariance matrix must be positive definite.")

    n_assets = covariance.shape[0]
    budget = np.full(n_assets, 1 / n_assets)
    # At the optimum y.T @ covariance @ y is one. This radius contains that optimum.
    radius = np.sqrt(n_assets / smallest_eigenvalue)
    log_radius = np.log(radius)
    scaled_weights = 1 / np.sqrt(covariance.diagonal())
    scaled_weights /= np.sqrt(scaled_weights @ covariance @ scaled_weights)

    marginal = covariance @ scaled_weights
    for iteration in range(1, max_iterations + 1):
        # Negative entropy step with the paper's gradient taming and L1 projection.
        tame = min(1.0, scaled_weights.min())
        step = iteration ** -step_exponent
        gradient = marginal - budget / scaled_weights
        log_candidate = np.log(scaled_weights) - step * tame * gradient
        excess = max(0.0, np.logaddexp.reduce(log_candidate) - log_radius)
        scaled_weights = np.exp(log_candidate - excess)
        marginal = covariance @ scaled_weights
        contributions = scaled_weights * marginal
        error = np.max(np.abs(contributions / contributions.sum() - budget))
        if error <= contribution_tolerance:
            return scaled_weights / scaled_weights.sum()

    raise RuntimeError(f"Mirror descent exceeded its iteration limit with contribution error {error:.3g}.")


def schedule_rebalances(dates, start, lag):
    months = dates.to_period("M")
    # The session before each monthly boundary supplies the signal.
    first_sessions = np.flatnonzero(months[1:] != months[:-1]) + 1
    execution_sessions = first_sessions + lag - 1
    valid = (execution_sessions >= dates.searchsorted(start)) & (execution_sessions < len(dates))
    return zip(first_sessions[valid] - 1, execution_sessions[valid], strict=True)


def make_targets(asset_returns, schedule, lookback, solver):
    returns = asset_returns.to_numpy()
    targets = {}
    for signal_session, execution_session in schedule:
        window_start = signal_session + 1 - lookback
        date = asset_returns.index[signal_session].date()
        if window_start < 1:
            raise ValueError(f"The {date} signal needs {lookback} returns. Set an earlier data_start.")
        try:
            targets[execution_session] = risk_budget_weights(
                returns[window_start:signal_session + 1], **solver,
            )
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(f"Failed to compute the {date} signal: {exc}") from exc
    return targets


def size_orders(positions, cash, previous_close, target, friction, cash_buffer):
    previous_value = cash + positions @ previous_close
    reserve = cash_buffer * previous_value
    desired = (previous_value - reserve) * target / previous_close
    changes = desired - positions
    sells = np.minimum(changes, 0)
    buys = np.maximum(changes, 0)
    cash_after_sells = cash - (sells @ previous_close) * (1 - friction)
    buy_budget = max(0.0, cash_after_sells - reserve)
    estimated_debit = (buys @ previous_close) * (1 + friction)
    if estimated_debit > buy_budget:
        buys *= buy_budget / estimated_debit
    orders = sells + buys
    orders.setflags(write=False)
    return orders


def settle_orders(orders, cash, close, friction):
    traded_amounts = orders * close
    traded_value = np.abs(traded_amounts).sum()
    remaining_cash = cash - traded_amounts.sum() - friction * traded_value
    if remaining_cash < 0:
        raise ArithmeticError(f"Predetermined fills exceeded available cash by {-remaining_cash:.6g}.")
    return remaining_cash, traded_value


def simulate(asset_prices, targets, start, friction, cash_buffer):
    dates = asset_prices.index
    first_session = dates.searchsorted(start)
    closes = asset_prices.to_numpy()
    positions = np.zeros(closes.shape[1])
    cash = previous_value = 1.0
    daily_results = []
    for session in range(first_session, len(dates)):
        turnover = 0.0
        n_trades = 0
        if session in targets:
            orders = size_orders(positions, cash, closes[session - 1], targets[session],
                                 friction, cash_buffer)
            close = closes[session]
            before_fill = cash + positions @ close
            try:
                cash, traded_value = settle_orders(orders, cash, close, friction)
            except ArithmeticError as exc:
                raise ArithmeticError(f"{dates[session].date()}: {exc}") from exc
            positions += orders
            turnover = traded_value / before_fill
            n_trades = np.count_nonzero(orders)
        closing_value = cash + positions @ closes[session]
        daily_results.append((closing_value / previous_value - 1, turnover, n_trades))
        previous_value = closing_value
    return pd.DataFrame(daily_results, index=dates[first_session:],
                        columns=["return", "turnover", "n_trades"])


def metrics(daily_results, risk_free):
    returns = daily_results["return"]
    excess = returns - risk_free.loc[returns.index]
    curve = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    ann_scale = np.sqrt(SESSIONS_PER_YEAR)
    volatility = returns.std()
    excess_volatility = excess.std()
    return {
        "cagr": float(curve[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1) if len(returns) else None,
        "ann_vol": float(volatility * ann_scale) if len(returns) > 1 else None,
        "sharpe": float(returns.mean() * ann_scale / volatility) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() * ann_scale / excess_volatility) if excess_volatility > 0 else None,
        "max_drawdown": float(np.min(curve / np.maximum.accumulate(curve) - 1)) if len(returns) else None,
        "total_return": float(curve[-1] - 1) if len(returns) else None,
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, risk_free, is_end):
    return {
        "is": metrics(daily_results.loc[:is_end], risk_free),
        "oos": metrics(daily_results.loc[daily_results.index > is_end], risk_free),
        "full": metrics(daily_results, risk_free),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    tickers = config["universe"]
    risk_free_ticker = config["risk_free"]
    price_symbols = [*tickers, risk_free_ticker]
    if len(tickers) < 2 or len(set(price_symbols)) != len(price_symbols):
        raise ValueError("Use at least two distinct assets and a separate cash proxy.")
    data_start, start, is_end, end = (pd.Timestamp(config[key]) for key in
                                      ("data_start", "start", "is_end", "end"))
    lag = config["execution_lag_sessions"]
    if not data_start < start < is_end < end or type(lag) is not int or lag < 1:
        raise ValueError("Check the date order and use a delay of at least one session.")
    cash_buffer = config["cash_buffer_bps"] / 10_000
    if not 0 < cash_buffer < 1:
        raise ValueError("The sizing cash buffer must be between zero and 10,000 bps.")
    solver = config["solver"]
    if (not 0.5 < solver["step_exponent"] <= 1 or solver["max_iterations"] < 1
            or not 0 < solver["contribution_tolerance"] < 1):
        raise ValueError("Invalid mirror descent settings.")

    base = Specification(config["lookback_sessions"], config["cost_bps"], config["slippage_bps"])
    robustness = config["robustness"]
    cost_multiplier = robustness["friction_multiplier"]
    high_friction = Specification(base.lookback_sessions, base.cost_bps * cost_multiplier,
                                  base.slippage_bps * cost_multiplier)
    alternatives = dict.fromkeys(
        [Specification(window, base.cost_bps, base.slippage_bps)
         for window in robustness["lookback_sessions"]] + [high_friction]
    )
    alternatives.pop(base, None)
    for spec in (base, *alternatives):
        if (type(spec.lookback_sessions) is not int or spec.lookback_sessions <= len(tickers)
                or not 0 <= spec.cost_bps or not 0 <= spec.slippage_bps
                or not 0 <= spec.friction < 1):
            raise ValueError(f"Invalid lookback or friction: {spec}")

    prices, fetched_at, source, digest = load_prices(price_symbols, data_start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"DATA_SOURCE: {source}")
    print(f"DATA_CACHE_SHA256: {digest}")
    print(f"PRICE_SESSIONS: {len(prices)}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, total-return daily closes and adjusted units")
    print(f"SIZING_CASH_BUFFER_BPS: {config['cash_buffer_bps']}")
    returns = prices.pct_change(fill_method=None)
    assets = returns[tickers]
    asset_prices = prices[tickers]
    risk_free = returns[risk_free_ticker]
    print(f"RISK_FREE: {risk_free_ticker}")

    is_assets = assets.loc[:is_end]
    is_risk_free = risk_free.loc[:is_end]
    print("ROBUSTNESS (IS only)")
    combos_tried = 0
    for spec in alternatives:
        combos_tried += 1
        targets = make_targets(is_assets, schedule_rebalances(is_assets.index, start, lag),
                               spec.lookback_sessions, solver)
        daily_results = simulate(asset_prices.loc[:is_end], targets, start, spec.friction, cash_buffer)
        print(json.dumps({**asdict(spec), "is": metrics(daily_results, is_risk_free)}, allow_nan=False))

    combos_tried += 1
    schedule = schedule_rebalances(assets.index, start, lag)
    targets = make_targets(assets, schedule, base.lookback_sessions, solver)
    daily_results = simulate(asset_prices, targets, start, base.friction, cash_buffer)
    if daily_results.empty:
        raise ValueError("The backtest period has no exchange sessions.")
    benchmark_name = "Monthly equal weight"
    equal = np.full(len(tickers), 1 / len(tickers))
    benchmark_daily = simulate(asset_prices, dict.fromkeys(targets, equal), start, base.friction,
                               cash_buffer)
    summaries = period_metrics(daily_results, risk_free, is_end)
    benchmark = period_metrics(benchmark_daily, risk_free, is_end)
    print(f"PARAM_COMBOS_TRIED: {combos_tried}")
    for period, summary in summaries.items():
        print(f"{period.upper()}: {json.dumps(summary, allow_nan=False)}")
        print(f"BENCHMARK {period.upper()} ({benchmark_name}): "
              f"{json.dumps(benchmark[period], allow_nan=False)}")

    print("RESULTS_JSON: " + json.dumps({
        "strategy": config["strategy"],
        "data_range": {
            "start": daily_results.index[0].date().isoformat(),
            "end": daily_results.index[-1].date().isoformat(),
        },
        **summaries,
        "benchmark": {"name": benchmark_name, **benchmark},
    }, allow_nan=False))


def stop_at_deadline(_signal_number, _frame):
    raise TimeoutError("The backtest exceeded its 590-second limit.")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_at_deadline)
    signal.alarm(590)
    try:
        main()
    finally:
        signal.alarm(0)
