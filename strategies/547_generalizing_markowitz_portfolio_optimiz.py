"""547_generalizing_markowitz_portfolio_optimiz
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/547_generalizing_markowitz_portfolio_optimiz.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 53c18ef949f91367c58358829370e449e492628c on 2026-11-27 18:41 Europe/Oslo, exit 0, output sha256 6edfdb1a2dfa4a8709d81ae3f7963f24cb2f867a31458db52e383a0f09acdd30
Data source: Free Yahoo Finance daily adjusted ETF closes via yfinance or the corresponding local CSV cache. XLK, XLF, XLE and XLV are the portfolio universe. BIL daily total returns are used only as the cash proxy for ex-post excess-return metrics.
Data fetched: 2026-11-27T17:41:34+00:00
Sample period: Backtest sessions 2010-01-04 through 2025-12-31 inclusive. Price history is requested from 2008-01-01, with 2008-01-02 documented as the first price session and pre-backtest history used for estimation.   data_range (from output): {'start': '2010-01-04', 'end': '2025-12-31'}
IS/OOS split: IS 2010-01-04 through 2017-12-29, OOS 2018-01-02 through 2025-12-31, all endpoints inclusive. The retrospective time split slices a single continuous simulation, carrying portfolio state and rolling estimation into OOS. The README records no contemporaneous strategy freeze.
Costs and slippage: Base costs are 2 bps and slippage is modeled as a flat 3 bps. The simulation deducts 5 bps in cash on the sum of buy and sell notionals at the fill close, including initial entry. The same rate is used to estimate proceeds and purchase costs at sizing. The IS friction stress case uses 4 bps cost and 6 bps slippage.
Signal lag: The first backtest session and subsequent first sessions of each month are signal dates. A target computed using data through close t is sized at close t and filled at close t+1 under the default one-session lag. With a larger lag L, quantities are fixed at close t+L-1 and filled at close t+L.
Price adjustment: yfinance auto_adjust=True, dividend- and split-adjusted total-return closes
Survivorship bias: The strategy retrospectively uses a fixed list of four surviving sector ETFs. The README documents missing historical candidate lists, exited funds and liquidation returns, and identifies survivorship and hindsight selection bias. The equal-weight benchmark uses the same universe.
Parameter combinations tried: 6
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.12280461278124877, ann_vol=0.15268682014835686, sharpe=0.8353633822874356, sharpe_excess=0.8313836431051645, max_drawdown=-0.24404469446690324, total_return=1.5225201606964762, turnover=2.6449441353067034, n_trades=226
    oos: cagr=0.15870083318051598, ann_vol=0.21072838935476645, sharpe=0.8049591023854212, sharpe_excess=0.6895260835289794, max_drawdown=-0.2972566524456701, total_return=2.239673786060522, turnover=2.6865985191648187, n_trades=229
    full: cagr=0.1406026002094094, ann_vol=0.18397814231634374, sharpe=0.8075845206820491, sharpe_excess=0.7399372254838601, max_drawdown=-0.2972566524456699, total_return=7.172142439417515, turnover=2.6657609757487184, n_trades=455
    benchmark: cagr=0.1231036802652059, ann_vol=0.1753262111071283, sharpe=0.7503311578817682, sharpe_excess=0.6793855656798914, max_drawdown=-0.38074597587003134, total_return=5.384354235542732, turnover=0.3402236514414474, n_trades=768
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 547_generalizing_markowitz_portfolio_optimiz\nuniverse: [XLK, XLF, XLE, XLV]\nrisk_free_ticker: BIL\n\nhistory_start: 2008-01-01\nstart: 2010-01-04\nis_end: 2017-12-29\nend: 2025-12-31\n# Signal at close t, lock quantities one session before the scheduled fill close.\nsignal_lag_sessions: 1\n# Reserve this fraction of sizing-close value. A fill that would make cash negative stops the simulation.\ncash_buffer_fraction: 0.05\n\nlookback_sessions: 252\nrisk_aversion: 8.0\ncovariance_shrinkage: 0.10\nturnover_penalty: 0.15\ncost_bps: 2.0\nslippage_bps: 3.0\n\nrobustness:\n  lookback_sessions: [210, 294]\n  turnover_penalty: [0.10, 0.20]\n  friction_multiplier: 2.0\n'
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
"""Monthly long-only sector allocation with a quadratic turnover penalty."""

import argparse
import json
import signal
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
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


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
TIME_LIMIT_SECONDS = 9 * 60
MIN_ORDER_VALUE_FRACTION = 1e-10


class NyseHolidayCalendar(AbstractHolidayCalendar):
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


def nyse_sessions(first_date, last_date):
    start, end = pd.Timestamp(first_date), pd.Timestamp(last_date)
    if start < pd.Timestamp("2008-01-01") or end > pd.Timestamp("2025-12-31"):
        raise ValueError("The historical closure calendar covers 2008 through 2025.")
    holidays = NyseHolidayCalendar().holidays(start, end)
    # Hurricane Sandy and two national days of mourning.
    closures = pd.DatetimeIndex(["2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])
    return pd.bdate_range(start, end).difference(holidays.union(closures))


def validate_price_panel(prices, expected_sessions):
    if prices.index.has_duplicates:
        raise ValueError("Daily adjusted closes contain duplicate dates.")
    missing = expected_sessions.difference(prices.index)
    unexpected = prices.index.difference(expected_sessions)
    if len(missing) or len(unexpected):
        raise ValueError(
            f"Daily adjusted close dates disagree with the NYSE calendar. "
            f"Missing ({len(missing)}): {missing[:5].strftime('%Y-%m-%d').tolist()}. "
            f"Unexpected ({len(unexpected)}): {unexpected[:5].strftime('%Y-%m-%d').tolist()}."
        )
    values = prices.to_numpy(dtype=float)
    if prices.empty or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Expected a complete panel of positive adjusted closes for all configured ETFs.")


def adjusted_closes(tickers, first_date, last_date):
    expected_sessions = nyse_sessions(first_date, last_date)
    CACHE_DIR.mkdir(exist_ok=True)
    cache_name = f"yfinance_adjusted_close_{'_'.join(tickers)}_{first_date}_{last_date}.csv"
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        with cache_path.open() as source:
            fetched_at = source.readline().strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(source, index_col=0, parse_dates=True)
    else:
        end_exclusive = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).date()
        with TemporaryDirectory(dir=CACHE_DIR) as download_cache:
            yf.set_tz_cache_location(download_cache)
            downloaded = yf.download(
                tickers, start=str(first_date), end=str(end_exclusive), interval="1d",
                auto_adjust=True, ignore_tz=True, progress=False, threads=False, timeout=15,
            )
        if downloaded is None or downloaded.empty or "Close" not in downloaded.columns:
            raise RuntimeError("yfinance did not return daily adjusted closes.")
        prices = downloaded["Close"]
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    prices = prices.reindex(columns=tickers).sort_index().loc[str(first_date):str(last_date)]
    validate_price_panel(prices, expected_sessions)
    if not cache_path.exists():
        with NamedTemporaryFile(mode="w", dir=CACHE_DIR, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                temporary.write(f"{fetched_at}\n")
                prices.to_csv(temporary, index_label="Date")
                temporary.close()
                temporary_path.replace(cache_path)
            finally:
                temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def optimal_weights(sample, current_weights, settings):
    expected_return = sample.mean(axis=0) * SESSIONS_PER_YEAR
    covariance = np.cov(sample, rowvar=False) * SESSIONS_PER_YEAR
    identity = np.eye(len(current_weights))
    shrinkage = settings["covariance_shrinkage"]
    shrunk_covariance = (1 - shrinkage) * covariance + shrinkage * covariance.diagonal().mean() * identity
    penalty = settings["turnover_penalty"]
    quadratic = settings["risk_aversion"] * shrunk_covariance + penalty * identity
    adjusted_return = expected_return + penalty * current_weights

    # Proposition 8 gives the budget-constrained stationary point.
    ones = np.ones(len(current_weights))
    return_direction, budget_direction = np.linalg.solve(
        quadratic, np.column_stack((adjusted_return, ones)),
    ).T
    unconstrained = return_direction + (1 - return_direction.sum()) / budget_direction.sum() * budget_direction
    if unconstrained.min() >= 0:
        return unconstrained

    # Project in the metric induced by the objective's Hessian.
    starting_weights = unconstrained.clip(min=0)
    projection = minimize(
        lambda weights: 0.5 * weights @ quadratic @ weights - adjusted_return @ weights,
        starting_weights / starting_weights.sum(),
        jac=lambda weights: quadratic @ weights - adjusted_return,
        bounds=[(0.0, 1.0)] * len(current_weights),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1, "jac": lambda weights: ones},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 100},
    )
    if not projection.success or abs(projection.x.sum() - 1) > 1e-8:
        raise RuntimeError(f"Long-only quadratic optimization failed: {projection.message}")
    weights = projection.x.clip(min=0)
    return weights / weights.sum()


def size_rebalance(target, positions, cash, sizing_close, friction, cash_buffer):
    """Fix both sides of the rebalance using the close before execution."""
    value = cash + positions @ sizing_close
    reserve = value * cash_buffer
    changes = (value - reserve) * target / sizing_close - positions
    changes[np.abs(changes) * sizing_close <= value * MIN_ORDER_VALUE_FRACTION] = 0.0
    sells = np.maximum(-changes, 0.0)
    buys = np.maximum(changes, 0.0)
    estimated_sale_proceeds = (sells @ sizing_close) * (1 - friction)
    spendable_cash = max(0.0, cash + estimated_sale_proceeds - reserve)
    estimated_buy_cost = (buys @ sizing_close) * (1 + friction)
    # Apply the cash limit while sizing, using the same close as the target quantities.
    if estimated_buy_cost > spendable_cash:
        buys *= spendable_cash / estimated_buy_cost
    sells.setflags(write=False)
    buys.setflags(write=False)
    return sells, buys


def fill_rebalance(sells, buys, cash, fill_close, friction):
    """Settle the fixed quantities at the execution close."""
    sale_value = sells @ fill_close
    buy_value = buys @ fill_close
    traded_value = sale_value + buy_value
    cash_after_fill = cash + sale_value - buy_value - friction * traded_value
    if not np.isfinite(cash_after_fill):
        raise ArithmeticError("Settlement produced a nonfinite cash balance.")
    if cash_after_fill < 0:
        raise ArithmeticError(f"Locked quantities exceed cash by {-cash_after_fill:.6g}.")
    return cash_after_fill, traded_value


def simulate(prices, start, settings, *, equal_weight=False, audit_label=None):
    first_session = prices.index.searchsorted(start)
    lag = settings["signal_lag_sessions"]
    lookback = settings["lookback_sessions"]
    if not lookback <= first_session < len(prices):
        raise ValueError("Expected a backtest session preceded by a full estimation window.")
    friction = (settings["cost_bps"] + settings["slippage_bps"]) / 10_000
    cash_buffer = settings["cash_buffer_fraction"]
    returns = prices.pct_change(fill_method=None).to_numpy()
    closes = prices.to_numpy()
    months = prices.index.to_period("M")
    equal_target = np.full(prices.shape[1], 1 / prices.shape[1])
    positions = np.zeros(prices.shape[1])
    cash = previous_value = 1.0
    scheduled = {}
    pending_rebalance = None
    records = []

    for session in range(first_session, len(prices)):
        close = closes[session]
        value_before_fill = cash + positions @ close
        turnover = 0.0
        trades = 0
        if pending_rebalance is not None:
            signal_session, sizing_session, sells, buys = pending_rebalance
            try:
                cash, traded_value = fill_rebalance(sells, buys, cash, close, friction)
            except ArithmeticError as error:
                raise ArithmeticError(f"{prices.index[session].date()} close: {error}") from error
            positions = positions - sells + buys
            trades = int(np.count_nonzero(sells) + np.count_nonzero(buys))
            turnover = traded_value / value_before_fill
            if audit_label is not None:
                audit = {
                    "simulation": audit_label,
                    "signal_date": str(prices.index[signal_session].date()),
                    "sizing_date": str(prices.index[sizing_session].date()),
                    "fill_date": str(prices.index[session].date()),
                    "sizing_closes": dict(zip(prices.columns, closes[sizing_session].tolist())),
                    "fill_closes": dict(zip(prices.columns, close.tolist())),
                    "quantity_changes": dict(zip(prices.columns, (buys - sells).tolist())),
                    "traded_value": float(traded_value),
                    "cash_after_fill": float(cash),
                }
                print("FILL_AUDIT: " + json.dumps(audit, allow_nan=False))
            pending_rebalance = None

        value = cash + positions @ close
        records.append((value / previous_value - 1, turnover, trades))
        previous_value = value
        is_signal_session = session == first_session or months[session] != months[session - 1]
        if is_signal_session and session + lag < len(prices):
            sample = returns[session - lookback + 1:session + 1]
            current_weights = positions * close / value
            target = equal_target if equal_weight else optimal_weights(sample, current_weights, settings)
            scheduled[session + lag - 1] = (session, target)

        # Lock quantities at this close for execution at the next session's close.
        scheduled_target = scheduled.pop(session, None)
        if scheduled_target is not None:
            signal_session, target = scheduled_target
            sells, buys = size_rebalance(target, positions, cash, close, friction, cash_buffer)
            pending_rebalance = (signal_session, session, sells, buys)

    return pd.DataFrame(records, index=prices.index[first_session:], columns=["return", "turnover", "n_trades"])


def metrics(daily_results, risk_free_returns):
    returns = daily_results["return"]
    excess = returns - risk_free_returns.loc[returns.index]
    growth = (1 + returns).cumprod().to_numpy()
    running_peak = np.maximum.accumulate(np.r_[1.0, growth])[1:]
    annual_scale = np.sqrt(SESSIONS_PER_YEAR)
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(volatility * annual_scale) if pd.notna(volatility) else None,
        "sharpe": float(returns.mean() / volatility * annual_scale) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_volatility * annual_scale) if excess_volatility > 0 else None,
        "max_drawdown": float((growth / running_peak - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def robustness_cases(settings):
    robustness = settings["robustness"]
    for name in ("lookback_sessions", "turnover_penalty"):
        for value in robustness[name]:
            yield f"{name}={value}", {**settings, name: value}
    multiplier = robustness["friction_multiplier"]
    yield f"cost_and_slippage={multiplier:g}x", {
        **settings,
        "cost_bps": settings["cost_bps"] * multiplier,
        "slippage_bps": settings["slippage_bps"] * multiplier,
    }


def main(*, audit_fills=False):
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        settings = yaml.safe_load(source)
    start = pd.Timestamp(settings["start"])
    split = pd.Timestamp(settings["is_end"])
    end = pd.Timestamp(settings["end"])
    if not pd.Timestamp(settings["history_start"]) < start < split < end:
        raise ValueError("Expected history_start < start < is_end < end.")
    lag = settings["signal_lag_sessions"]
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("Execution lag must be at least one full trading session.")
    if not 0 < settings["cash_buffer_fraction"] < 1:
        raise ValueError("The cash buffer fraction must be positive and below 1.")
    universe = settings["universe"]
    risk_free_ticker = settings["risk_free_ticker"]
    tickers = [*universe, risk_free_ticker]
    if len(universe) < 2 or len(set(tickers)) != len(tickers) or any(not ticker.isalpha() for ticker in tickers):
        raise ValueError("Use distinct ETF symbols for the sector universe and cash proxy.")
    cases = list(robustness_cases(settings))
    for _, case in [("default", settings), *cases]:
        if (case["lookback_sessions"] < 2 or case["risk_aversion"] <= 0
                or not 0 < case["covariance_shrinkage"] <= 1 or case["turnover_penalty"] < 0):
            raise ValueError("Invalid estimation or quadratic penalty settings.")
        if min(case["cost_bps"], case["slippage_bps"]) < 0 or case["cost_bps"] + case["slippage_bps"] >= 10_000:
            raise ValueError("Combined costs must be nonnegative and below 10,000 bps.")

    prices, fetched_at = adjusted_closes(tickers, settings["history_start"], settings["end"])
    sector_prices = prices[universe]
    risk_free_returns = prices[risk_free_ticker].pct_change(fill_method=None)
    print(f"RUN_DIRECTORY: {Path.cwd()}")
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, dividend- and split-adjusted total-return closes")

    daily_results = simulate(sector_prices, start, settings, audit_label="strategy" if audit_fills else None)
    benchmark = simulate(
        sector_prices, start, settings, equal_weight=True, audit_label="benchmark" if audit_fills else None,
    )
    is_results = daily_results.loc[:split]
    oos_results = daily_results.loc[daily_results.index > split]
    if is_results.empty or oos_results.empty:
        raise ValueError("Both sample periods need at least one session.")
    data_range = {"start": str(daily_results.index[0].date()), "end": str(daily_results.index[-1].date())}
    results = {
        "strategy": settings["strategy"],
        "data_range": data_range,
        "is": metrics(is_results, risk_free_returns),
        "oos": metrics(oos_results, risk_free_returns),
        "full": metrics(daily_results, risk_free_returns),
        "benchmark": metrics(benchmark, risk_free_returns),
    }
    print(f"Date Range: {data_range['start']} to {data_range['end']}")
    print(f"IS through {is_results.index[-1].date()}, OOS from {oos_results.index[0].date()}")
    print(f"SHARPE_EXCESS_PROXY: {risk_free_ticker} daily total return")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(results[period], allow_nan=False)}")
    benchmark_periods = {
        "IS": metrics(benchmark.loc[is_results.index], risk_free_returns),
        "OOS": metrics(benchmark.loc[oos_results.index], risk_free_returns),
        "FULL": results["benchmark"],
    }
    for period, values in benchmark_periods.items():
        print(f"BENCHMARK (monthly equal weight, {period}): {json.dumps(values, allow_nan=False)}")

    print("ROBUSTNESS (IS only, no parameter selection)")
    for name, case in cases:
        case_results = simulate(sector_prices.loc[:split], start, case, audit_label=name if audit_fills else None)
        print(f"  {name}: {json.dumps(metrics(case_results, risk_free_returns), allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {1 + len(cases)}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


def time_limit_reached(_signum, _frame):
    raise SystemExit(f"Backtest exceeded the {TIME_LIMIT_SECONDS}-second time limit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-fills", action="store_true", help="Print the locked quantities and prices at each rebalance.")
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, time_limit_reached)
    signal.alarm(TIME_LIMIT_SECONDS)
    try:
        main(audit_fills=args.audit_fills)
    finally:
        signal.alarm(0)
