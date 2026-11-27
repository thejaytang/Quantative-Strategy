"""548_entropic_value_at_risk_portfolio_optimiz
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/548_entropic_value_at_risk_portfolio_optimiz.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 53c18ef949f91367c58358829370e449e492628c on 2026-11-27 18:41 Europe/Oslo, exit 0, output sha256 148ab103bd32ec741c91d8173bd2e6cb97acb0217bce7242244b8bcec54ea3c0
Data source: Yahoo Finance via yfinance daily bars for 11 Select Sector SPDR ETFs (XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY) and ^IRX quotes used as 91-day Treasury-bill discount rates for excess Sharpe. The recorded snapshot was fetched at 2026-11-27T08:20:08+00:00.
Data fetched: 2026-11-27T17:41:36+00:00
Sample period: Scored sessions span 2001-07-02 through 2026-03-31 inclusive. The daily price history starts on 1998-12-22 for warm-up.   data_range (from output): {'start': '2001-07-02', 'end': '2026-03-31'}
IS/OOS split: IS runs from 2001-07-02 through 2014-12-31. OOS runs from 2015-01-02 through 2026-03-31. The split partitions the returns of one continuous default simulation, so OOS inherits the IS closing holdings and cash. The five alternatives run only on IS. This is a historical holdout, with no documented evidence that the split or parameters were fixed before OOS began.
Costs and slippage: The default models 5 bps of cost plus 2 bps of fixed slippage per buy or sell, including initial entry. Sizing estimates the deduction at the prior close. At execution, cash is debited by buy notional times 1.0007 and credited by sell notional times 0.9993, using absolute simulated fill notionals. The IS higher-cost variant uses 20 bps of cost plus 5 bps of slippage per side. The fixed-bps model omits variable spreads and market impact.
Signal lag: After the last NYSE session closes each month, targets use ETF returns through that close t. Under the configured one-session lag, quantities are fixed using the close of t and filled at the next NYSE session's close.
Price adjustment: yfinance auto_adjust=True total-return close (sector ETFs)
Survivorship bias: Survivorship bias remains because the universe is a fixed list of existing sector ETFs. The code enforces each ETF's hard-coded first trading date and requires a complete finite return window for eligibility, including for later launches XLRE and XLC. The sources provide no historical point-in-time membership or data on exits, suspensions and liquidations.
Parameter combinations tried: 6
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.07502622495672057, ann_vol=0.13960405859805924, sharpe=0.5880999188520267, sharpe_excess=0.48416133664635846, max_drawdown=-0.35252907337855566, total_return=1.6509969124335053, turnover=0.7739998833444394, n_trades=543
    oos: cagr=0.07959994956623605, ann_vol=0.13726914636036353, sharpe=0.6268182989496728, sharpe_excess=0.47851242553438156, max_drawdown=-0.2798761644937254, total_return=1.361299971035992, turnover=0.9046422888476047, n_trades=430
    full: cagr=0.07710158123968727, ann_vol=0.13853718176666552, sharpe=0.6055541875243194, sharpe_excess=0.4816402929955504, max_drawdown=-0.35252907337855566, total_return=5.259798932545732, turnover=0.8333484419749148, n_trades=973
    benchmark  full: cagr=0.08650558617186906, ann_vol=0.17232186407027347, sharpe=0.5678990763306287, sharpe_excess=0.4682931137993772, max_drawdown=-0.5138241677272652, total_return=6.758706474917533, turnover=0.19220943733352355, n_trades=2843
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 548_entropic_value_at_risk_portfolio_optimiz\nuniverse: [XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY]\n# Excess Sharpe uses quotes dated 1–10 calendar days before each return\'s starting close.\nrisk_free: "^IRX"\nprice_start: 1998-12-22\nstart: 2001-07-02\nis_end: 2014-12-31\nend: 2026-03-31\n# Count NYSE sessions from signal close to fill close. Lock quantities one session before the fill.\nexecution_lag_sessions: 1\n# Reserve 5% of estimated post-cost value at the prior close. Unspent cash stays idle.\n# Fill the fixed quantities or stop if their cost exceeds available cash.\ncash_buffer_fraction: 0.05\n\ndefault:\n  lookback_sessions: 504\n  confidence: 0.95\n  cost_bps: 5\n  slippage_bps: 2\n\nrobustness_is:\n  lower_confidence: {confidence: 0.925}\n  higher_confidence: {confidence: 0.975}\n  shorter_lookback: {lookback_sessions: 462}\n  longer_lookback: {lookback_sessions: 546}\n  higher_cost: {cost_bps: 20, slippage_bps: 5}\n'
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
"""Monthly long-only allocation by empirical Entropic Value-at-Risk."""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from pandas.tseries import holiday
from pandas.tseries.offsets import CustomBusinessDay
from scipy.optimize import brentq, minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
FIRST_CLOSE = dict.fromkeys("XLB XLE XLF XLI XLK XLP XLU XLV XLY".split(), "1998-12-22")
FIRST_CLOSE.update(XLRE="2015-10-08", XLC="2018-06-19")
RISK_FREE = "^IRX"
MIN_TRADE_FRACTION = 1e-8


class NYSEHolidays(holiday.AbstractHolidayCalendar):
    rules = [
        holiday.Holiday("New Year's Day", month=1, day=1, observance=holiday.sunday_to_monday),
        holiday.USMartinLutherKingJr,
        holiday.USPresidentsDay,
        holiday.GoodFriday,
        holiday.USMemorialDay,
        holiday.Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                        observance=holiday.nearest_workday),
        holiday.Holiday("Independence Day", month=7, day=4, observance=holiday.nearest_workday),
        holiday.USLaborDay,
        holiday.USThanksgivingDay,
        holiday.Holiday("Christmas", month=12, day=25, observance=holiday.nearest_workday),
    ]


# Full-day closures since the first configured listings, excluding regular holidays.
NYSE_SPECIAL_CLOSURES = pd.to_datetime([
    "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14",
    "2004-06-11", "2007-01-02", "2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09",
])


@dataclass(frozen=True)
class Specification:
    lookback_sessions: int
    confidence: float
    cost_bps: float
    slippage_bps: float

    def __post_init__(self):
        if type(self.lookback_sessions) is not int or self.lookback_sessions < 2:
            raise ValueError("lookback_sessions must be an integer of at least two")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between zero and one")
        if not (0 <= self.cost_bps and 0 <= self.slippage_bps and self.drag < 1):
            raise ValueError("Costs and slippage must be nonnegative and sum to less than 10000 bps")

    @property
    def drag(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def load_config():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for key in ("price_start", "start", "is_end", "end"):
        config[key] = pd.Timestamp(config[key])
    if not config["price_start"] < config["start"] <= config["is_end"] < config["end"]:
        raise ValueError("Dates must satisfy price_start < start <= is_end < end")
    universe = config["universe"]
    if not universe or not set(universe) <= FIRST_CLOSE.keys() or len(universe) != len(set(universe)):
        raise ValueError("universe must contain unique Select Sector SPDR tickers")
    if config["risk_free"] != RISK_FREE:
        raise ValueError(f"risk_free must be {RISK_FREE} for the discount-rate conversion")
    lag = config["execution_lag_sessions"]
    if type(lag) is not int or lag < 1:
        raise ValueError("execution_lag_sessions must be an integer of at least one")
    if not 0 < config["cash_buffer_fraction"] < 1:
        raise ValueError("cash_buffer_fraction must be between zero and one")
    default = Specification(**config["default"])
    alternatives = {
        name: replace(default, **changes)
        for name, changes in config["robustness_is"].items()
    }
    if "default" in alternatives:
        raise ValueError("default is reserved for the official specification")
    if len({default, *alternatives.values()}) != 1 + len(alternatives):
        raise ValueError("Strategy specifications must be distinct")
    return config, default, alternatives


def check_prices(close, sectors, price_start, end):
    if not isinstance(close.index, pd.DatetimeIndex) or close.index.hasnans:
        raise ValueError("Price snapshot has invalid dates")
    close = close.loc[(close.index >= price_start) & (close.index <= end)]
    if close.empty or close.index.has_duplicates or not close.index.is_monotonic_increasing:
        raise ValueError("Price snapshot has no rows or has invalid dates")
    listings = {ticker: pd.Timestamp(FIRST_CLOSE[ticker]) for ticker in sectors}
    first_session = max(price_start, min(listings.values()))
    sessions = pd.date_range(first_session, end, freq=CustomBusinessDay(calendar=NYSEHolidays()))
    sessions = sessions.difference(NYSE_SPECIAL_CLOSURES)
    sector_close = close[sectors]
    quoted_sessions = sector_close.index[sector_close.notna().any(axis=1)]
    missing = sessions.difference(quoted_sessions)
    unexpected = quoted_sessions.difference(sessions)
    if len(missing) or len(unexpected):
        raise ValueError(f"ETF dates differ from NYSE sessions: "
                         f"{len(missing)} missing {missing[:5].strftime('%Y-%m-%d').tolist()}, "
                         f"{len(unexpected)} unexpected {unexpected[:5].strftime('%Y-%m-%d').tolist()}")
    sector_close = sector_close.reindex(sessions)
    for ticker in sectors:
        prices = sector_close[ticker]
        listed = sessions >= listings[ticker]
        if prices.loc[~listed].notna().any():
            raise ValueError(f"{ticker} has prices before its first trading session")
        listed_prices = prices.loc[listed]
        if not (np.isfinite(listed_prices) & (listed_prices > 0)).all():
            raise ValueError(f"{ticker} has missing or invalid prices on or after listing")
    quotes = close[RISK_FREE]
    if not np.isfinite(quotes).any():
        raise ValueError(f"Missing price history for {RISK_FREE}")
    return sector_close, quotes


def load_prices(config):
    price_start, end = config["price_start"], config["end"]
    sectors = [t for t in config["universe"] if pd.Timestamp(FIRST_CLOSE[t]) <= end]
    if not sectors:
        raise ValueError("No sectors had launched by the configured end date")
    tickers = [*sectors, RISK_FREE]
    universe_key = sha256(",".join(sorted(tickers)).encode()).hexdigest()[:12]
    CACHE_DIR.mkdir(exist_ok=True)
    snapshot = CACHE_DIR / f"yahoo_adjusted_{price_start.date()}_{end.date()}_{universe_key}.csv"
    if snapshot.exists():
        with snapshot.open(encoding="utf-8") as source:
            fetched_at = source.readline().strip().removeprefix("# fetched_at=")
            datetime.fromisoformat(fetched_at)
            close = pd.read_csv(
                source, index_col=0, parse_dates=True,
                converters=dict.fromkeys(tickers, lambda value: float(value or "nan")),
            )
        sector_close, quotes = check_prices(close, sectors, price_start, end)
        return sector_close, quotes, fetched_at

    with TemporaryDirectory(dir=CACHE_DIR) as vendor_cache:
        yf.set_tz_cache_location(vendor_cache)
        prices = yf.download(
            tickers, start=price_start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True, interval="1d", threads=4, progress=False, timeout=10,
        )
    if prices is None or prices.empty:
        raise RuntimeError("Yahoo Finance returned no daily prices")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    close = prices["Close"].reindex(columns=tickers)
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    sector_close, quotes = check_prices(close, sectors, price_start, end)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            tmp.write(f"# fetched_at={fetched_at}\n")
            close.to_csv(tmp)
            tmp.flush()
            tmp_path.replace(snapshot)
        finally:
            tmp_path.unlink(missing_ok=True)
    return sector_close, quotes, fetched_at


def risk_free_returns(quotes, sessions):
    # Allow a calendar day for publication because the feed has no quote timestamps.
    publication_lag = pd.Timedelta(days=1)
    max_quote_age = pd.Timedelta(days=10)
    quotes = quotes[np.isfinite(quotes)]
    dated_rates = quotes.reindex(
        sessions - publication_lag, method="ffill", tolerance=max_quote_age - publication_lag,
    )
    dated_rates.index = sessions
    bill_price = 1.0 - (dated_rates / 100) * 91 / 360
    bill_price = bill_price.where(bill_price > 0)
    daily_rate = np.expm1(-np.log(bill_price) * 365.25 / (91 * SESSIONS_PER_YEAR))
    # Attribute the starting close's cash proxy to the return ending at the next close.
    return daily_rate.shift(1)


def minimum_evar_weights(losses, confidence):
    # In percentage points, v = 1/u gives a joint convex perspective.
    size = losses.shape[1]
    uniform = np.full(size, 1.0 / size)
    if size == 1:
        return uniform
    tail_penalty = -np.log1p(-confidence)
    min_v = 1e-6
    start_v = max(10 * min_v, np.std(losses @ uniform) / np.sqrt(2 * tail_penalty))

    def objective(choice):
        weights, v = choice[:-1], choice[-1]
        portfolio_losses = losses @ weights
        maximum = portfolio_losses.max()
        shifted = (portfolio_losses - maximum) / v
        mass = np.exp(shifted)
        mass_sum = mass.sum()
        probabilities = mass / mass_sum
        log_term = np.log(mass_sum / len(losses)) + tail_penalty
        value = maximum + v * log_term
        gradient = np.r_[probabilities @ losses, log_term - probabilities @ shifted]
        return value, gradient

    budget_jacobian = np.r_[np.ones(size), 0.0]
    solution = minimize(
        objective, np.r_[uniform, start_v], jac=True, method="SLSQP",
        bounds=[(0, 1)] * size + [(min_v, None)],
        constraints=[{
            "type": "eq",
            "fun": lambda choice: budget_jacobian @ choice - 1.0,
            "jac": lambda _choice: budget_jacobian,
        }],
        options={"ftol": 1e-10, "maxiter": 200},
    )
    if not solution.success or not np.isfinite(solution.fun):
        raise RuntimeError(f"EVaR optimization failed: {solution.message}")
    weights = np.clip(solution.x[:-1], 0, 1)
    if abs(weights.sum() - 1.0) > 1e-8:
        raise RuntimeError("EVaR optimization missed the budget constraint")
    return weights / weights.sum()


def build_targets(returns, start, lag, spec, include_benchmark=False):
    sessions = returns.index
    months = sessions.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])
    series = returns.to_numpy()
    targets, equal_weights = {}, {}
    for signal_index in month_ends:
        fill_index = signal_index + lag
        if fill_index >= len(sessions) or sessions[fill_index] < start:
            continue
        window_start = signal_index + 1 - spec.lookback_sessions
        if window_start < 0:
            continue
        window = series[window_start:signal_index + 1]
        eligible = np.isfinite(window).all(axis=0)
        if not eligible.any():
            continue
        fill_date = sessions[fill_index]
        weights = np.zeros(len(returns.columns))
        try:
            weights[eligible] = minimum_evar_weights(-100.0 * window[:, eligible], spec.confidence)
        except RuntimeError as error:
            raise RuntimeError(f"Signal on {sessions[signal_index].date()}: {error}") from error
        targets[fill_date] = weights
        if include_benchmark:
            equal_weights[fill_date] = eligible.astype(float) / eligible.sum()
    return targets, equal_weights


def size_changes(previous_prices, positions, cash, target, drag, cash_buffer):
    required_prices = previous_prices[(target > 0) | (positions > 0)]
    if not np.all(np.isfinite(required_prices) & (required_prices > 0)):
        raise ValueError("A held or target ETF has no valid prior close for sizing")
    previous_values = positions * previous_prices
    previous_value = cash + previous_values.sum()
    previous_weights = previous_values / previous_value
    funded_target = (1.0 - cash_buffer) * target

    # Estimate both sides' costs at the prior close and reserve cash after those costs.
    def funding_gap(fraction):
        turnover = np.abs(fraction * funded_target - previous_weights).sum()
        return fraction + drag * turnover - 1.0

    fraction = brentq(funding_gap, 0.0, 1.0)
    desired_values = previous_value * fraction * funded_target
    desired_positions = np.divide(
        desired_values, previous_prices, out=np.zeros_like(positions), where=previous_prices > 0,
    )
    changes = desired_positions - positions
    changes[np.abs(changes) * previous_prices <= MIN_TRADE_FRACTION * previous_value] = 0.0
    changes.setflags(write=False)
    return changes


def execute_changes(fixed_changes, closing_prices, positions, cash, drag, before):
    traded = fixed_changes != 0
    fill_prices = closing_prices[traded]
    if not np.all(np.isfinite(fill_prices) & (fill_prices > 0)):
        raise RuntimeError("A fixed-quantity fill has no valid closing price")
    fill_values = np.zeros_like(fixed_changes)
    fill_values[traded] = fixed_changes[traded] * fill_prices
    sell_value = np.maximum(-fill_values, 0.0).sum()
    buy_value = np.maximum(fill_values, 0.0).sum()
    remaining_cash = cash + sell_value * (1.0 - drag) - buy_value * (1.0 + drag)
    if not np.isfinite(remaining_cash):
        raise RuntimeError("Fixed-quantity fills produced a nonfinite cash balance")
    if remaining_cash < 0.0:
        raise RuntimeError(
            f"Fixed-quantity fills would exceed available cash by {-remaining_cash / before:.6%} of portfolio value"
        )
    turnover = max(buy_value, sell_value) / before
    next_positions = positions + fixed_changes
    if not np.all(np.isfinite(next_positions) & (next_positions >= 0.0)):
        raise RuntimeError("Fixed-quantity fills must leave finite, nonnegative positions")
    return next_positions, remaining_cash, turnover, np.count_nonzero(traded)


def simulate(close, start, targets, spec, risk_free, cash_buffer):
    sessions = close.index
    first = sessions.searchsorted(start)
    if first == 0 and len(sessions) and targets.get(sessions[0]) is not None:
        raise ValueError("A trade requires a prior session for sizing")
    # Price validation leaves NaNs only before listing, when positions must be zero.
    prices = close.fillna(0.0).to_numpy()
    positions = np.zeros(len(close.columns))
    cash = previous_value = 1.0
    daily_results = []
    pending_changes = None
    # Include the preceding unscored session to size any first-day fill.
    for day in range(max(first - 1, 0), len(sessions)):
        closing_prices = prices[day]
        before = cash + positions @ closing_prices
        turnover, trades = 0.0, 0
        if pending_changes is not None:
            try:
                positions, cash, turnover, trades = execute_changes(
                    pending_changes, closing_prices, positions, cash, spec.drag, before,
                )
            except RuntimeError as error:
                raise RuntimeError(f"Execution on {sessions[day].date()}: {error}") from error
        value = cash + positions @ closing_prices
        if day >= first:
            daily_results.append((value / previous_value - 1.0, turnover, trades))
        previous_value = value

        # Once this session is complete, lock the quantities for the next close.
        next_target = targets.get(sessions[day + 1]) if day + 1 < len(sessions) else None
        pending_changes = None
        if next_target is not None:
            try:
                pending_changes = size_changes(
                    closing_prices, positions, cash, next_target, spec.drag, cash_buffer,
                )
            except ValueError as error:
                raise ValueError(f"Sizing on {sessions[day].date()}: {error}") from error
    result = pd.DataFrame(
        daily_results, index=sessions[first:], columns=["return", "turnover", "n_trades"]
    )
    result["risk_free"] = risk_free.reindex(result.index)
    return result


def metrics(period):
    if period.empty:
        return {
            "cagr": None, "ann_vol": None, "sharpe": None, "sharpe_excess": None,
            "max_drawdown": None, "total_return": None, "turnover": None, "n_trades": 0,
        }
    returns = period["return"]
    excess = returns - period["risk_free"]
    vol, excess_vol = float(returns.std()), float(excess.std(skipna=False))
    path = np.r_[0.0, np.log1p(returns.to_numpy()).cumsum()]
    years = len(period) / SESSIONS_PER_YEAR
    return {
        "cagr": float(np.expm1(path[-1] / years)),
        "ann_vol": vol * np.sqrt(SESSIONS_PER_YEAR) if np.isfinite(vol) else None,
        "sharpe": float(returns.mean()) * np.sqrt(SESSIONS_PER_YEAR) / vol if vol > 0 else None,
        "sharpe_excess": float(excess.mean()) * np.sqrt(SESSIONS_PER_YEAR) / excess_vol if excess_vol > 0 else None,
        "max_drawdown": float(np.expm1(path - np.maximum.accumulate(path)).min()),
        "total_return": float(np.expm1(path[-1])),
        "turnover": float(period["turnover"].sum()) / years,
        "n_trades": int(period["n_trades"].sum()),
    }


def split_metrics(daily_results, is_end):
    return {
        "is": metrics(daily_results.loc[:is_end]),
        "oos": metrics(daily_results.loc[daily_results.index > is_end]),
        "full": metrics(daily_results),
    }


def main():
    config, default, alternatives = load_config()
    close, quotes, fetched_at = load_prices(config)
    start, is_end, lag = config["start"], config["is_end"], config["execution_lag_sessions"]
    cash_buffer = config["cash_buffer_fraction"]
    returns = close.pct_change(fill_method=None)
    scored_sessions = close.loc[start:].index
    risk_free = risk_free_returns(quotes, close.index)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True total-return close (sector ETFs)")
    missing_risk_free = risk_free.reindex(scored_sessions).isna().sum()
    if missing_risk_free:
        print(f"RISK_FREE_MISSING_SESSIONS: {missing_risk_free}")

    targets, equal_weights = build_targets(returns, start, lag, default, include_benchmark=True)
    daily_results = simulate(close, start, targets, default, risk_free, cash_buffer)
    official = split_metrics(daily_results, is_end)
    benchmark = split_metrics(simulate(close, start, equal_weights, default, risk_free, cash_buffer), is_end)
    for name, report in (("STRATEGY", official), ("BENCHMARK monthly equal-weight sectors", benchmark)):
        for period, values in report.items():
            print(f"{name} {period.upper()}: {json.dumps(values, allow_nan=False)}")

    print("ROBUSTNESS (IS only)")
    print(f"default IS {default}: {json.dumps(official['is'], allow_nan=False)}")
    is_close = close.loc[:is_end]
    is_returns = returns.loc[:is_end]
    base_is_targets = {date: weights for date, weights in targets.items() if date <= is_end}
    for name, spec in alternatives.items():
        same_signal = (spec.lookback_sessions, spec.confidence) == (default.lookback_sessions, default.confidence)
        if same_signal:
            is_targets = base_is_targets
        else:
            is_targets, _ = build_targets(is_returns, start, lag, spec)
        values = metrics(simulate(is_close, start, is_targets, spec, risk_free, cash_buffer))
        print(f"{name} IS {spec}: {json.dumps(values, allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {1 + len(alternatives)}")

    result = {
        "strategy": config["strategy"],
        "data_range": {"start": str(scored_sessions[0].date()), "end": str(scored_sessions[-1].date())},
        **official,
        "benchmark": {"name": "monthly_equal_weight_sectors", **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(9 * 60)
    main()
