"""543_robust_enhanced_index_tracking_portfolio
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/543_robust_enhanced_index_tracking_portfolio.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit fc4032f1dc84592dc32f39915df656911d60b83b on 2026-11-27 18:52 Europe/Oslo, exit 0, output sha256 7b63e774fa96f677889d334c3842bdce54c81757bb7834715277e35321144a00
Data source: Free Yahoo Finance daily adjusted closes via yfinance for 25 US stocks (AAPL, NVDA, MSFT, AVGO, ORCL, JNJ, ABBV, LLY, UNH, ABT, BRK-B, JPM, V, BAC, MA, AMZN, TSLA, HD, MCD, BKNG, WMT, COST, KO, PG, PM), SPY (tracking and buy-and-hold benchmark), and BIL (excess-Sharpe reference only). The run records a snapshot fetched at 2026-11-27 08:17:44 UTC.
Data fetched: 2026-11-27T17:52:46+00:00
Sample period: Daily price coverage is 2009-12-01 through 2024-12-31. The first 60-month return window uses month-end closes from December 2009 through December 2014. The configured evaluation starts 2015-01-01 and the reported trading-session range is 2015-01-02 through 2024-12-31. data_range (from output): {'start': '2015-01-02', 'end': '2024-12-31'}
IS/OOS split: IS is 2015-01-02 through 2019-12-31. OOS is 2020-01-02 through 2024-12-31. The default portfolio carries positions and cash across the boundary and re-estimates monthly using the trailing 60 months with fixed configured parameters. The seven robustness variants are run through the IS cutoff only. The paper calibrated some defaults on the same IS period and also studied the same OOS period, so the split is retrospective.
Costs and slippage: Baseline cost is 5 bp and modeled slippage is a flat 5 bp per buy or sell, both deducted from cash on actual gross fill notional, including initial entry. The same rates apply to the SPY benchmark entry. The higher-friction IS case uses 15 bp cost plus 10 bp slippage. The 500 bp buy/sell price stress applies to the preceding-close cash budget for sizing.
Signal lag: Target weights use monthly returns through the last trading session's close t. With the configured one-session lag, quantities are fixed after close t and filled at the next trading session's adjusted close. With a longer lag, sizing occurs at the close immediately before the fill.
Price adjustment: yfinance auto_adjust=True, dividend- and split-adjusted daily Close
Survivorship bias: The fixed 25-stock list comes from the 2026 paper's retrospective selection. Historical candidate, exit, and delisting settlement records are absent, leaving survivorship and retrospective selection bias. A stock enters the eligible set when the trailing 60 monthly returns are complete. Missing prices needed for a fill or valuation stop the simulation.
Parameter combinations tried: 8
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.20905243136653673, ann_vol=0.1495686997369626, sharpe=1.3446018169471403, sharpe_excess=1.28507945667254, max_drawdown=-0.19443587753529867, total_return=1.5797127327493037, turnover=1.1495181286618026, n_trades=1372
    oos: cagr=0.27013442776638485, ann_vol=0.21004376343789344, sharpe=1.2444372105423476, sharpe_excess=1.1339468598218276, max_drawdown=-0.30435130511888153, total_return=2.2993184537238864, turnover=1.0118161526102545, n_trades=1390
    full: cagr=0.23921713918638687, ann_vol=0.18230462559008276, sharpe=1.2684719847129582, sharpe_excess=1.1804899407596772, max_drawdown=-0.30435130511888153, total_return=7.511293824466261, turnover=1.0806671406360284, n_trades=2762
    benchmark: cagr=0.12624349324664763, ann_vol=0.1706990973211719, sharpe=0.7823302457284171, sharpe_excess=0.6884679070519432, max_drawdown=-0.32803752527501917, total_return=2.2772033399715332, turnover=0.09524317041191006, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 543_robust_enhanced_index_tracking_portfolio\n# Cai and Song, section 5. The benchmark and cash proxy are separate from the universe.\nuniverse:\n  [AAPL, NVDA, MSFT, AVGO, ORCL, JNJ, ABBV, LLY, UNH, ABT, BRK-B, JPM, V,\n   BAC, MA, AMZN, TSLA, HD, MCD, BKNG, WMT, COST, KO, PG, PM]\nbenchmark: SPY\nexcess_sharpe_reference: BIL\ndates:\n  data_start: 2009-12-01\n  evaluation_start: 2015-01-01\n  is_end: 2019-12-31\n  end: 2024-12-31\n# Count exchange sessions from the signal close. Size at the close immediately before the fill.\nrebalance_lag_sessions: 1\n# Stress buy prices up and sell prices down by 500 bp using the preceding close.\n# Include fees in that budget. The fill uses the locked quantity or stops on a cash shortfall.\ncash_buffer_bps: 500.0\nmodel:\n  lookback_months: 60\n  kappa: 5.0\n  mean_radius: 0.22\n  covariance_radius: 0.025\n  target_monthly_excess: 0.04\n  worst_monthly_return_floor: -0.01\n  max_weight: 0.10\n  cost_bps: 5.0\n  slippage_bps: 5.0\n# Each case overrides only the listed baseline parameters and runs through is_end.\nrobustness:\n  kappa_4: {kappa: 4.0}\n  kappa_6: {kappa: 6.0}\n  mean_radius_0.18: {mean_radius: 0.18}\n  mean_radius_0.26: {mean_radius: 0.26}\n  covariance_radius_0.020: {covariance_radius: 0.020}\n  covariance_radius_0.030: {covariance_radius: 0.030}\n  higher_friction: {cost_bps: 15.0, slippage_bps: 10.0}\n'
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
"""Monthly distributionally robust mean–target semivariance index tracking."""

import signal

if __name__ == "__main__":
    signal.alarm(570)

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
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
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
FEASIBILITY_TOLERANCE = 1e-8
TRADE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class Model:
    lookback_months: int
    kappa: float
    mean_radius: float
    covariance_radius: float
    target_monthly_excess: float
    worst_monthly_return_floor: float
    max_weight: float
    cost_bps: float
    slippage_bps: float

    def __post_init__(self):
        if (
            type(self.lookback_months) is not int or self.lookback_months < 2
            or not np.isfinite(list(vars(self).values())).all()
            or self.kappa <= 0
            or min(self.mean_radius, self.covariance_radius, self.cost_bps, self.slippage_bps) < 0
            or not 0 < self.max_weight <= 1
            or self.friction >= 1
        ):
            raise ValueError("Invalid model parameters in config.yaml")

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


class NyseHolidays(AbstractHolidayCalendar):
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


def expected_sessions(start, end):
    holidays = NyseHolidays().holidays(start, end)
    # Full-day exceptional closures within the configured 2009–2024 window.
    exceptional = pd.to_datetime(["2012-10-29", "2012-10-30", "2018-12-05"])
    return pd.bdate_range(start, end).difference(holidays.union(exceptional))


def validate_prices(prices, tickers, benchmark, reference, start, end):
    if list(prices.columns) != tickers or prices.empty:
        raise RuntimeError("Adjusted close data is missing or the ticker columns differ from the request")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise RuntimeError("Adjusted close dates must be unique and ordered")
    sessions = expected_sessions(start, end)
    missing = sessions.difference(prices.index)
    unexpected = prices.index.difference(sessions)
    if len(missing) or len(unexpected):
        missing_dates = missing[:5].strftime("%Y-%m-%d").tolist()
        unexpected_dates = unexpected[:5].strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"NYSE calendar mismatch: missing={missing_dates}, unexpected={unexpected_dates}")
    for ticker in tickers:
        series = prices[ticker]
        observed = series.dropna()
        if observed.empty or (ticker in (benchmark, reference) and len(observed) != len(sessions)):
            raise RuntimeError(f"Missing adjusted close history for {ticker}")
        if not np.isfinite(observed).all() or (observed <= 0).any():
            raise RuntimeError(f"Invalid adjusted closes for {ticker}")


def load_prices(tickers, benchmark, reference, start, end):
    start_date = start.date().isoformat()
    end_exclusive = (end + pd.Timedelta(days=1)).date().isoformat()
    key = sha256(json.dumps([tickers, start_date, end_exclusive]).encode()).hexdigest()[:16]
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"yfinance_adjusted_close_{key}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            fetched_at = handle.readline().strip().removeprefix("# fetched_at=")
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(handle, index_col="date", parse_dates=["date"])
        validate_prices(prices, tickers, benchmark, reference, start, end)
        return prices, fetched_at

    import yfinance as yf

    with TemporaryDirectory(dir=CACHE, prefix=".yfinance-") as transient_cache:
        yf.set_tz_cache_location(transient_cache)
        for attempt in range(2):
            raw = yf.download(
                tickers, start=start_date, end=end_exclusive, interval="1d",
                auto_adjust=True, ignore_tz=True, multi_level_index=True,
                progress=False, threads=False, timeout=8,
            )
            try:
                if raw is None or raw.empty or "Close" not in raw.columns.get_level_values(0):
                    raise RuntimeError("Yahoo Finance returned no adjusted daily closes")
                prices = raw["Close"].reindex(columns=tickers).sort_index()
                validate_prices(prices, tickers, benchmark, reference, start, end)
                break
            except RuntimeError:
                if attempt == 1:
                    raise

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as handle:
        temporary_path = Path(handle.name)
        try:
            handle.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(handle, index_label="date")
            handle.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def optimize_weights(monthly_returns, model):
    """Solve paper equation (4.8) after substituting its tight risk bounds."""
    n_assets = monthly_returns.shape[1] - 1
    if n_assets * model.max_weight < 1:
        raise ValueError("The eligible universe is too small for the configured weight cap")
    means = -monthly_returns.mean(axis=0)
    covariance = np.cov(monthly_returns, rowvar=False, ddof=1)
    covariance += 1e-10 * np.eye(n_assets + 1)
    asset_covariance = covariance[:-1, :-1]
    robust_covariance = covariance + model.covariance_radius * np.eye(n_assets + 1)
    mean_penalty = np.sqrt(model.mean_radius)
    max_worst_loss = -model.worst_monthly_return_floor

    def worst_loss(weights):
        covariance_product = asset_covariance @ weights
        volatility = np.sqrt(weights @ covariance_product)
        loss = means[:-1] @ weights + mean_penalty * volatility
        extra_gradient = mean_penalty * covariance_product / volatility if volatility else 0.0
        return loss, means[:-1] + extra_gradient

    def objective(weights):
        loss, loss_gradient = worst_loss(weights)
        shortfall = max(loss - means[-1] + model.target_monthly_excess, 0.0)
        active = np.append(weights, -1.0)
        covariance_product = robust_covariance @ active
        semivariance = shortfall**2 + active @ covariance_product
        gradient = loss_gradient + 2 * model.kappa * (
            shortfall * loss_gradient + covariance_product[:-1]
        )
        return 100 * (loss + model.kappa * semivariance), 100 * gradient

    ones = np.ones(n_assets)
    budget = {
        "type": "eq",
        "fun": lambda weights: weights.sum() - 1,
        "jac": lambda weights: ones,
    }
    floor = {
        "type": "ineq",
        "fun": lambda weights: 100 * (max_worst_loss - worst_loss(weights)[0]),
        "jac": lambda weights: -100 * worst_loss(weights)[1],
    }

    def solve(function, initial, constraints):
        result = minimize(
            function, initial, method="SLSQP", jac=True, constraints=constraints,
            bounds=[(0.0, model.max_weight)] * n_assets,
            options={"ftol": 1e-11, "maxiter": 150},
        )
        if (
            not result.success or not np.isfinite(result.x).all()
            or abs(result.x.sum() - 1) > FEASIBILITY_TOLERANCE
        ):
            raise RuntimeError(f"Weight optimization failed: {result.message}")
        return result.x

    equal_weights = ones / n_assets
    unconstrained_floor = solve(objective, equal_weights, [budget])
    if worst_loss(unconstrained_floor)[0] <= max_worst_loss:
        return unconstrained_floor, False

    def loss_objective(weights):
        loss, gradient = worst_loss(weights)
        return 100 * loss, 100 * gradient

    best_floor = solve(loss_objective, equal_weights, [budget])
    if worst_loss(best_floor)[0] >= max_worst_loss - FEASIBILITY_TOLERANCE:
        # Use the best attainable worst-case return at or beyond the feasibility boundary.
        return best_floor, bool(worst_loss(best_floor)[0] > max_worst_loss + FEASIBILITY_TOLERANCE)

    weights = solve(objective, best_floor, [budget, floor])
    if worst_loss(weights)[0] > max_worst_loss + FEASIBILITY_TOLERANCE:
        raise RuntimeError("Optimized weights violate the worst-case return floor")
    return weights, False


def monthly_targets(prices, start, lag, model):
    # The price index has already been checked against the exchange calendar.
    month = prices.index.to_period("M")
    month_end_rows = np.flatnonzero(month[:-1] != month[1:])
    monthly_returns = prices.iloc[month_end_rows].pct_change(fill_method=None)
    n_assets = prices.shape[1] - 1
    targets = {}
    relaxed_floors = []
    for row, signal_row in enumerate(month_end_rows):
        execution_row = signal_row + lag
        if execution_row >= len(prices) or prices.index[execution_row] < start:
            continue
        # The first month-end price is the base for the first monthly return.
        # Check warm-up at the signal date, including when lag crosses the evaluation start.
        if row < model.lookback_months:
            continue
        window = monthly_returns.iloc[row + 1 - model.lookback_months:row + 1]
        if window.iloc[:, -1].isna().any():
            raise ValueError(f"Missing monthly benchmark returns as of {prices.index[signal_row].date()}")
        present = window.iloc[:, :-1].notna().all(axis=0).to_numpy()
        eligible = window.iloc[:, np.append(present, True)]
        execution_date = prices.index[execution_row]
        try:
            selected_weights, relaxed = optimize_weights(eligible.to_numpy(), model)
        except (ValueError, RuntimeError) as error:
            raise RuntimeError(f"{prices.index[signal_row].date()} signal: {error}") from error
        weights = np.zeros(n_assets)
        weights[present] = selected_weights
        targets[execution_date] = weights
        if relaxed:
            relaxed_floors.append(execution_date)
    return targets, relaxed_floors


def size_changes(positions, sizing_marks, previous_value, target, cash, friction, cash_buffer):
    """Lock signed quantities against stressed prices from the preceding close."""
    desired = np.divide(
        previous_value * target, sizing_marks, out=positions.copy(), where=sizing_marks > 0,
    )
    changes = desired - positions
    changes[np.abs(changes) * sizing_marks <= TRADE_TOLERANCE * previous_value] = 0.0
    sales = np.minimum(changes, 0.0)
    purchases = np.maximum(changes, 0.0)
    sale_proceeds = -(sales @ sizing_marks) * (1 - cash_buffer) * (1 - friction)
    buy_budget = cash + sale_proceeds
    buy_cost = (purchases @ sizing_marks) * (1 + cash_buffer) * (1 + friction)
    if buy_cost > buy_budget:
        purchases *= (buy_budget / buy_cost) * (1 - 8 * np.finfo(float).eps)
    locked_changes = sales + purchases
    locked_changes.setflags(write=False)
    return locked_changes


def simulate(prices, start, targets, friction, cash_buffer):
    """Track fractional adjusted-price units and cash across closing fills."""
    first_row = prices.index.searchsorted(start)
    if not 0 < first_row < len(prices):
        raise ValueError("Evaluation requires at least one session and a preceding close")
    closing_prices = prices.to_numpy()
    positions = np.zeros(prices.shape[1])
    cash = previous_value = 1.0
    locked_changes = None
    daily_results = []
    # Include the preceding session so the first fill can be sized before evaluation starts.
    for row in range(first_row - 1, len(prices)):
        date = prices.index[row]
        needs_close = positions > 0
        if locked_changes is not None:
            needs_close |= locked_changes != 0

        close = closing_prices[row]
        quoted = np.isfinite(close)
        missing_close = needs_close & ~quoted
        if missing_close.any():
            tickers = ", ".join(prices.columns[missing_close])
            raise RuntimeError(f"Cannot value or fill {tickers} on {date.date()}: missing close or delisting proceeds")
        marks = np.where(quoted, close, 0.0)
        before_fill_value = cash + positions @ marks
        traded_value = 0.0
        trade_count = 0
        if locked_changes is not None:
            notionals = locked_changes * marks
            traded_value = np.abs(notionals).sum()
            closing_cash = cash - (notionals.sum() + traded_value * friction)
            if not np.isfinite(closing_cash):
                raise RuntimeError(f"Non-finite cash after pricing fixed fill quantities on {date.date()}")
            if closing_cash < 0:
                raise RuntimeError(
                    f"Fixed fill quantities exceed cash on {date.date()}: shortfall {-closing_cash:.8g}"
                )
            positions += locked_changes
            cash = closing_cash
            trade_count = int(np.count_nonzero(locked_changes))

        value = cash + positions @ marks if trade_count else before_fill_value
        if row >= first_row:
            daily_results.append((
                value / previous_value - 1, traded_value / before_fill_value, trade_count, cash / value,
            ))
        previous_value = value

        # Lock the next session's changes using this session's settled cash and closing marks.
        locked_changes = None
        if row + 1 == len(prices):
            continue
        execution_date = prices.index[row + 1]
        target = targets.get(execution_date)
        if target is None:
            continue
        missing_sizing = (target > 0) & ~quoted
        if missing_sizing.any():
            tickers = ", ".join(prices.columns[missing_sizing])
            raise RuntimeError(f"Cannot size {tickers} for {execution_date.date()}: missing preceding close")
        locked_changes = size_changes(positions, marks, value, target, cash, friction, cash_buffer)
    return pd.DataFrame(
        daily_results, index=prices.index[first_row:], columns=["return", "turnover", "n_trades", "cash_weight"],
    )


def run_model(prices, start, lag, model, cash_buffer):
    targets, relaxed_floors = monthly_targets(prices, start, lag, model)
    daily_results = simulate(prices.iloc[:, :-1], start, targets, model.friction, cash_buffer)
    return daily_results, min(targets, default=None), relaxed_floors


def metrics(daily_results, reference):
    returns = daily_results["return"]
    if returns.empty:
        return dict.fromkeys(
            ("cagr", "ann_vol", "sharpe", "sharpe_excess", "max_drawdown", "total_return", "turnover"),
        ) | {"n_trades": 0}
    excess = returns - reference.reindex(returns.index)
    if not np.isfinite(excess).all():
        raise ValueError("Missing returns for the excess-Sharpe reference")
    growth_index = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    years = len(returns) / SESSIONS_PER_YEAR
    volatility = float(returns.std())
    excess_volatility = float(excess.std())
    return {
        "cagr": float(growth_index[-1] ** (1 / years) - 1),
        "ann_vol": float(volatility * np.sqrt(SESSIONS_PER_YEAR)) if np.isfinite(volatility) else None,
        "sharpe": (
            float(returns.mean() * np.sqrt(SESSIONS_PER_YEAR) / volatility)
            if np.isfinite(volatility) and volatility > 0 else None
        ),
        "sharpe_excess": (
            float(excess.mean() * np.sqrt(SESSIONS_PER_YEAR) / excess_volatility)
            if np.isfinite(excess_volatility) and excess_volatility > 0 else None
        ),
        "max_drawdown": float(np.min(growth_index / np.maximum.accumulate(growth_index) - 1)),
        "total_return": float(growth_index[-1] - 1),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    dates = config["dates"]
    data_start = pd.Timestamp(dates["data_start"])
    start = pd.Timestamp(dates["evaluation_start"])
    is_end = pd.Timestamp(dates["is_end"])
    end = pd.Timestamp(dates["end"])
    lag = config["rebalance_lag_sessions"]
    if not data_start < start <= is_end < end or type(lag) is not int or lag < 1:
        raise ValueError("Check chronology and set rebalance_lag_sessions to a positive integer")
    cash_buffer = config["cash_buffer_bps"] / 10_000
    if not 0 < cash_buffer < 1:
        raise ValueError("Set cash_buffer_bps between 0 and 10,000, exclusive")
    benchmark = config["benchmark"]
    reference_ticker = config["excess_sharpe_reference"]
    tickers = [*config["universe"], benchmark, reference_ticker]
    if len(tickers) < 3 or len(tickers) != len(set(tickers)):
        raise ValueError("Use a nonempty universe and distinct benchmark and reference tickers")
    model = Model(**config["model"])
    robustness = [(label, replace(model, **overrides)) for label, overrides in config["robustness"].items()]
    if len({model, *(variant for _, variant in robustness)}) != 1 + len(robustness):
        raise ValueError("Every robustness case must specify a distinct parameter combination")
    prices, fetched_at = load_prices(tickers, benchmark, reference_ticker, data_start, end)
    reference = prices[reference_ticker].pct_change(fill_method=None)
    model_prices = prices.drop(columns=reference_ticker)

    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, dividend- and split-adjusted daily Close")
    print(f"BENCHMARK: {benchmark} buy-and-hold")
    print(f"EXCESS_SHARPE_REFERENCE: {reference_ticker} daily adjusted return")

    robustness_results = []
    with threadpool_limits(limits=1, user_api="blas"):
        for label, variant in robustness:
            sample, _, relaxed = run_model(model_prices.loc[:is_end], start, lag, variant, cash_buffer)
            robustness_results.append((label, metrics(sample, reference.loc[:is_end]), len(relaxed)))
        daily_results, first_execution, relaxed = run_model(model_prices, start, lag, model, cash_buffer)

    sessions = daily_results.index
    benchmark_entry = first_execution if first_execution is not None else sessions[0]
    benchmark_results = simulate(
        prices[[benchmark]], start, {benchmark_entry: np.ones(1)}, model.friction, cash_buffer,
    )
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": sessions[0].date().isoformat(), "end": sessions[-1].date().isoformat()},
    }
    benchmark_metrics = {}
    for name, period in (
        ("is", sessions[sessions <= is_end]),
        ("oos", sessions[sessions > is_end]),
        ("full", sessions),
    ):
        result[name] = metrics(daily_results.loc[period], reference)
        benchmark_metrics[name] = metrics(benchmark_results.loc[period], reference)
        print(f"{name.upper()}: {json.dumps(result[name], allow_nan=False)}")
        print(f"BENCHMARK_{name.upper()}: {json.dumps(benchmark_metrics[name], allow_nan=False)}")
    result["benchmark"] = benchmark_metrics["full"]

    print(
        f"MIN_CLOSING_CASH_WEIGHT: strategy={daily_results['cash_weight'].min():.8f} "
        f"benchmark={benchmark_results['cash_weight'].min():.8f}"
    )
    relaxed_is = sum(date <= is_end for date in relaxed)
    print(f"RETURN_FLOOR_INFEASIBLE_REBALANCES: IS={relaxed_is} OOS={len(relaxed) - relaxed_is}")
    print("ROBUSTNESS (IS ONLY)")
    print(f"default: {json.dumps(result['is'], allow_nan=False)}; floor_infeasible={relaxed_is}")
    for label, row, relaxed_count in robustness_results:
        print(f"{label}: {json.dumps(row, allow_nan=False)}; floor_infeasible={relaxed_count}")
    print(f"PARAM_COMBOS_TRIED: {1 + len(robustness_results)}")
    print(f"RESULTS_JSON: {json.dumps(result, allow_nan=False)}")


if __name__ == "__main__":
    main()
