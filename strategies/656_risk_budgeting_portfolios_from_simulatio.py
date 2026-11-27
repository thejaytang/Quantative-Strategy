"""656_risk_budgeting_portfolios_from_simulatio
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/656_risk_budgeting_portfolios_from_simulatio.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 9532cd9f815cc2d43ceb143b2f1416acf34e9864 on 2026-11-27 19:49 Europe/Oslo, exit 0, output sha256 7ae973c948347ccdcc3a0626cdd2cf4059e9ec3c088470383f6ceffe6900f144
Data source: Yahoo Finance via yfinance, daily ETF opens, closes and cash distributions for SPY, IWM, EFA, TLT, DBC and GLD. BIL is downloaded from the same source and used as the risk-free proxy for excess Sharpe. Inputs are cached under data_cache/.
Data fetched: 2026-11-27T18:50:01+00:00
Sample period: Market data including warm-up spans 2007-06-01 through 2025-12-31. The reported backtest spans 2010-01-04 through 2025-12-31. Dates are inclusive.   data_range (from output): {'start': '2010-01-04', 'end': '2025-12-31'}
IS/OOS split: IS 2010-01-04 through 2018-12-31; OOS 2019-01-02 through 2025-12-31. The main backtest runs continuously across the split, carrying positions and cash forward. The main specification is fixed in config.yaml, and the five robustness variants run on the IS prefix only. OOS is a historical holdout.
Costs and slippage: The main model applies 2 bps fees plus 3 bps modeled slippage per buy or sell leg, including initial entry. Both are charged on the opening trade notional as a combined cash-flow adjustment: buys consume 1.0005 times notional and sells realize 0.9995 times notional. The IS higher-friction variant uses 6 bps fees plus 9 bps slippage per leg.
Signal lag: The signal uses returns through the last trading session's close each month. Configured lag is one trading session, so the target is sized and filled at the next session's open.
Price adjustment: yfinance auto_adjust=False; Yahoo split-adjusted quotes with cash distributions reinvested at the ex-date open
Survivorship bias: The fixed ETF universe was selected retrospectively. It excludes liquidated funds and historical alternatives, so the sample is subject to survivorship and selection bias.
Parameter combinations tried: 6
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.05594533731014906, ann_vol=0.0734398611107552, sharpe=0.7780449969928676, sharpe_excess=0.7454790352761943, max_drawdown=-0.13589305197558388, total_return=0.6307886140854666, turnover=9.57942194333214, n_trades=648
    oos: cagr=0.09575187598161317, ann_vol=0.10376190722688877, sharpe=0.9333114716032316, sharpe_excess=0.6894814854563328, max_drawdown=-0.2108229513048484, total_return=0.8938919901162627, turnover=6.550998728591479, n_trades=504
    full: cagr=0.07317465845640303, ann_vol=0.08799393846110845, sharpe=0.8467003424327567, sharpe_excess=0.7057481441441805, max_drawdown=-0.21082295130484963, total_return=2.0885374937892327, turnover=16.13042067192362, n_trades=1152
    benchmark  full: cagr=0.07821906782076371, ann_vol=0.1093658879538267, sharpe=0.7434988122290275, sharpe_excess=0.6297723632713742, max_drawdown=-0.23445675371182606, total_return=2.328693501166563, turnover=6.282764911810185, n_trades=1152
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 656_risk_budgeting_portfolios_from_simulatio\nuniverse: [SPY, IWM, EFA, TLT, DBC, GLD]\nrisk_free: BIL\ndata_start: '2007-06-01'\nstart: '2010-01-04'\nis_end: '2018-12-31'\nend: '2025-12-31'\nsignal_lag_sessions: 1\nseed: 656234\n\nmodel:\n  lookback_sessions: 500\n  horizon_sessions: 5\n  confidence: 0.95\n  scenarios: 1000\n  cost_bps: 2\n  slippage_bps: 3\n\nrobustness:\n  shorter_window: {lookback_sessions: 400}\n  longer_window: {lookback_sessions: 600}\n  lower_confidence: {confidence: 0.925}\n  higher_confidence: {confidence: 0.975}\n  higher_friction: {cost_bps: 6, slippage_bps: 9}\n"
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
"""Monthly cross-asset Expected Shortfall risk budgeting, for research.

Paper: https://arxiv.org/abs/2302.01196
SamplingRB.jl (GPL-3.0-or-later): https://github.com/bfpc/SamplingRB.jl
https://raw.githubusercontent.com/bfpc/SamplingRB.jl/main/src/cvar_cp.jl
Riskfolio-Lib (BSD-3-Clause): https://pypi.org/project/riskfolio-lib/
https://github.com/dcajasn/Riskfolio-Lib/blob/master/examples/Tutorial%209%20-%20Risk%20Parity%20Portfolio%20Optimization.ipynb
https://github.com/dcajasn/Riskfolio-Lib/blob/master/riskfolio/src/Portfolio.py
Related LEAN and Empyrical examples (Apache-2.0):
https://github.com/QuantConnect/Lean/blob/master/Algorithm.Framework/Portfolio/RiskParityPortfolioOptimizer.py
https://github.com/quantopian/empyrical/blob/master/empyrical/stats.py
The Python optimization and execution are independent implementations.
"""

import hashlib
import json
import signal
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from pandas.tseries import holiday
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
PER_YEAR = 252
EXTRA_CLOSURES = ["2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"]


@dataclass(frozen=True)
class Model:
    lookback_sessions: int
    horizon_sessions: int
    confidence: float
    scenarios: int
    cost_bps: float
    slippage_bps: float

    def __post_init__(self):
        if not 1 <= self.horizon_sessions <= self.lookback_sessions:
            raise ValueError("The risk horizon must fit inside the lookback window.")
        if not 0 < self.confidence < 1 or self.scenarios * (1 - self.confidence) < 1 - 1e-10:
            raise ValueError("The simulated tail must contain at least one scenario.")
        if not 0 <= self.cost_bps or not 0 <= self.slippage_bps or not self.friction < 1:
            raise ValueError("Costs and slippage must be nonnegative and sum to less than 10,000 bps.")

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def exchange_sessions(start, end):
    """Regular U.S. equity sessions and exceptional closures for this sample."""
    if not pd.Timestamp("2007-01-01") <= start <= end <= pd.Timestamp("2025-12-31"):
        raise ValueError("The U.S. equity calendar in this script covers 2007 through 2025.")
    calendar = holiday.AbstractHolidayCalendar(rules=[
        holiday.Holiday("New Year", month=1, day=1, observance=holiday.sunday_to_monday),
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
    ])
    sessions = pd.date_range(start, end, freq=pd.offsets.CustomBusinessDay(calendar=calendar))
    return sessions.difference(pd.DatetimeIndex(EXTRA_CLOSURES))


def fetch_prices(tickers, sessions, start, end, temporary_path):
    yf.set_tz_cache_location(str(temporary_path))
    yf.config.debug.hide_exceptions = False
    bars = []
    for ticker in tickers:
        history = yf.Ticker(ticker).history(
            start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
            interval="1d", auto_adjust=False, actions=True, repair=False,
            timeout=12,
        )
        history.index = history.index.tz_localize(None)
        if not history.index.equals(sessions):
            missing = sessions.difference(history.index).strftime("%Y-%m-%d").tolist()
            extra = history.index.difference(sessions).strftime("%Y-%m-%d").tolist()
            raise ValueError(f"{ticker} has unexpected session coverage. Missing={missing[:8]}, extra={extra[:8]}")
        distributions = history["Dividends"]
        if "Capital Gains" in history:
            distributions = distributions + history["Capital Gains"]
        bars.append(np.column_stack([history["Open"], history["Close"], distributions]))
    return np.stack(bars, axis=1)


def market_data(tickers, start, end):
    sessions = exchange_sessions(start, end)
    request = json.dumps(["yfinance_split_quotes_and_cash_v1", tickers, str(start), str(end)])
    key = hashlib.sha256(request.encode()).hexdigest()[:16]
    CACHE_DIR.mkdir(exist_ok=True)
    snapshot = CACHE_DIR / f"yahoo_{key}.npz"
    if not snapshot.exists():
        with TemporaryDirectory(dir=CACHE_DIR) as temp_dir:
            temporary_path = Path(temp_dir)
            bars = fetch_prices(tickers, sessions, start, end, temporary_path)
            validate_prices(bars, sessions, tickers)
            fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
            pending = temporary_path / "prices.npz"
            np.savez_compressed(pending, bars=bars, fetched_at=fetched_at,
                                dates=sessions.to_numpy(dtype="datetime64[D]"))
            pending.replace(snapshot)

    with np.load(snapshot, allow_pickle=False) as data:
        bars = data["bars"]
        fetched_at = data["fetched_at"].item()
        if not np.array_equal(data["dates"], sessions.to_numpy(dtype="datetime64[D]")):
            raise ValueError(f"The dates in {snapshot.name} do not match the exchange calendar.")
    validate_prices(bars, sessions, tickers)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=False; Yahoo split-adjusted quotes with "
          "cash distributions reinvested at the ex-date open")
    print(f"DATA_SHA256: {hashlib.sha256(snapshot.read_bytes()).hexdigest()}")
    return sessions, bars


def validate_prices(bars, sessions, tickers):
    if bars.shape != (len(sessions), len(tickers), 3) or not np.isfinite(bars).all():
        raise ValueError("The snapshot must contain finite prices and distributions for every session.")
    if np.any(bars[:, :, :2] <= 0) or np.any(bars[:, :, 2] < 0):
        raise ValueError("The snapshot contains a nonpositive quote or a negative cash distribution.")


def price_factors(bars):
    opening, closing, distribution = np.moveaxis(bars, 2, 0)
    overnight = np.full_like(opening, np.nan)
    overnight[1:] = (opening[1:] + distribution[1:]) / closing[:-1]
    return overnight, closing / opening


def schedule_rebalances(dates, start, lag):
    first = dates.searchsorted(start)
    month_ends = np.flatnonzero(dates.month[:-1] != dates.month[1:])
    return [(close_idx, close_idx + lag) for close_idx in month_ends
            if first <= close_idx + lag < len(dates)]


def bootstrap_losses(log_growth, model, seed, close_date):
    # Every scenario draws a whole multivariate block, so its assets share the same dates.
    horizon = model.horizon_sessions
    sums = np.vstack([np.zeros(log_growth.shape[1]), log_growth.cumsum(axis=0)])
    block_losses = -np.expm1(sums[horizon:] - sums[:-horizon])
    rng = np.random.default_rng(np.random.SeedSequence([seed, close_date.toordinal()]))
    return block_losses[rng.integers(len(block_losses), size=model.scenarios)]


def tail_cut(losses, exposure, mass):
    """Return a supporting ES subgradient, including fractional mass at the cutoff."""
    ranks = np.argsort(-(losses @ exposure), kind="stable")
    whole = int(mass)
    gradient = losses[ranks[:whole]].sum(axis=0)
    if mass > whole:
        gradient += (mass - whole) * losses[ranks[whole]]
    return gradient / mass


def risk_weights(losses, confidence):
    """Solve empirical ES minus the log budget penalty using supporting planes."""
    count = losses.shape[1]
    budget = np.full(count, 1 / count)
    mass = round((1 - confidence) * len(losses), 10)
    scale = tail_cut(losses, budget, mass) @ budget
    if not scale > 0:
        raise ValueError("Equal-weight ES is nonpositive and the risk-budget solution may not exist.")
    losses = losses / scale

    def objective(point):
        exposure = point[:-1]
        return point[-1] - budget @ np.log(exposure), np.r_[-budget / exposure, 1.0]

    bounds = [(1e-5, 100.0)] * count + [(None, None)]
    point = np.r_[budget, 1.0]
    cuts = [tail_cut(losses, direction, mass) for direction in np.vstack([budget, np.eye(count)])]
    for _ in range(160):
        # The epigraph of the maximum of the cuts gives a lower bound on the full problem.
        cut_matrix = np.column_stack([-np.asarray(cuts), np.ones(len(cuts))])
        result = minimize(
            objective, point, jac=True, method="SLSQP", bounds=bounds,
            constraints={"type": "ineq", "fun": lambda x: cut_matrix @ x,
                         "jac": lambda x: cut_matrix},
            options={"ftol": 1e-10, "maxiter": 150},
        )
        if not result.success:
            raise RuntimeError(f"Risk-budget subproblem failed: {result.message}")
        exposure = result.x[:-1]
        supporting_cut = tail_cut(losses, exposure, mass)
        risk = supporting_cut @ exposure
        if abs(risk - result.x[-1]) <= 1e-8:
            if risk <= 0 or np.any(exposure < 1.01e-5) or np.any(exposure > 99.9):
                raise RuntimeError("Risk-budget optimum has nonpositive ES or touches a numerical bound.")
            # When losses tie at VaR a blend of active cuts supplies the marginal risks.
            multipliers = result.multipliers
            if abs(multipliers.sum() - 1) > 1e-3 or multipliers.min() < -1e-10:
                raise RuntimeError("Risk-budget cut multipliers do not form a valid blend.")
            blend = np.maximum(multipliers, 0)
            marginal = (blend / blend.sum()) @ np.asarray(cuts)
            error = float(np.max(np.abs(exposure * marginal / risk - budget)))
            if abs(risk - exposure @ marginal) > 1e-6 or error > 5e-4:
                raise RuntimeError(f"Risk contributions miss their budgets by {error:.3g}.")
            return exposure / exposure.sum(), error
        cuts.append(supporting_cut)
        point = np.r_[exposure, risk]

    raise RuntimeError("The empirical ES risk-budget problem did not converge.")


def target_weights(dates, log_growth, schedule, model, seed):
    targets = {}
    max_error = 0.0
    for signal_idx, fill_idx in schedule:
        left = signal_idx + 1 - model.lookback_sessions
        if left < 1:
            raise ValueError(f"Insufficient warmup for the {dates[signal_idx].date()} signal.")
        history = log_growth[left:signal_idx + 1]
        losses = bootstrap_losses(history, model, seed, dates[signal_idx])
        try:
            targets[fill_idx], error = risk_weights(losses, model.confidence)
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(f"Risk-budget solve failed on {dates[signal_idx].date()}: {exc}") from exc
        max_error = max(max_error, error)
    return targets, max_error


def fill_at_open(positions, cash, weights, friction):
    """Find post-fee targets using opening values and self-financing cash flows."""
    opening_value = positions.sum() + cash
    low, high = 0.0, opening_value
    for _ in range(54):
        candidate = (low + high) / 2
        changes = weights * candidate - positions
        proceeds = -np.minimum(changes, 0).sum() * (1 - friction)
        spent = np.maximum(changes, 0).sum() * (1 + friction)
        if spent <= cash + proceeds:
            low = candidate
        else:
            high = candidate

    new_positions = weights * low
    changes = new_positions - positions
    bought = np.maximum(changes, 0).sum()
    sold = -np.minimum(changes, 0).sum()
    new_cash = (cash + sold * (1 - friction)) - bought * (1 + friction)
    if new_cash < 0:
        raise RuntimeError("Simulated opening fills exceeded available cash.")
    return new_positions, new_cash, (bought + sold) / opening_value, np.count_nonzero(changes)


def simulate(dates, overnight, intraday, targets, start, friction):
    first = dates.searchsorted(start)
    positions = np.zeros(overnight.shape[1])
    cash = 1.0
    previous_value = cash
    rows = []
    for idx in range(first, len(dates)):
        positions = positions * overnight[idx]
        turnover, n_trades = 0.0, 0
        if idx in targets:
            positions, cash, turnover, n_trades = fill_at_open(positions, cash, targets[idx], friction)
        positions = positions * intraday[idx]
        closing_value = cash + positions.sum()
        rows.append((closing_value / previous_value - 1, turnover, n_trades, cash))
        previous_value = closing_value
    return pd.DataFrame(rows, index=dates[first:], columns=["return", "turnover", "n_trades", "cash"])


def sharpe(returns):
    volatility = returns.std(ddof=1)
    return float(np.sqrt(PER_YEAR) * returns.mean() / volatility) if volatility > 0 else None


def metrics(daily_results, risk_free):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise ValueError("Each reporting period needs at least two sessions.")
    growth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    return {
        "cagr": float(growth[-1] ** (PER_YEAR / len(returns)) - 1),
        "ann_vol": float(np.sqrt(PER_YEAR) * returns.std(ddof=1)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(returns - risk_free.loc[returns.index]),
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, risk_free, is_end):
    in_sample = daily_results.index <= is_end
    return {
        "is": metrics(daily_results.loc[in_sample], risk_free),
        "oos": metrics(daily_results.loc[~in_sample], risk_free),
        "full": metrics(daily_results, risk_free),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    tickers = config["universe"] + [config["risk_free"]]
    data_start, start, is_end, end = (pd.Timestamp(config[key])
                                      for key in ("data_start", "start", "is_end", "end"))
    lag, seed = config["signal_lag_sessions"], config["seed"]
    if not data_start < start <= is_end < end or not isinstance(lag, int) or not 1 <= lag <= 20:
        raise ValueError("Use ordered sample dates and a signal lag of 1 to 20 sessions.")
    if len(tickers) < 3 or len(set(tickers)) != len(tickers) or tickers[-1] != "BIL":
        raise ValueError("Use unique ETF symbols and BIL as the cash proxy.")
    model = Model(**config["model"])
    variants = {name: replace(model, **overrides) for name, overrides in config["robustness"].items()}
    if len({model, *variants.values()}) != len(variants) + 1:
        raise ValueError("Robustness specifications must differ from the main model and each other.")

    dates, bars = market_data(tickers, data_start, end)
    overnight, intraday = price_factors(bars)
    growth = overnight * intraday
    risk_free = pd.Series(growth[:, -1] - 1, index=dates, name=tickers[-1])
    overnight, intraday, log_growth = overnight[:, :-1], intraday[:, :-1], np.log(growth[:, :-1])

    # Run only on the truncated prefix, with parameters set before evaluating the main result.
    split = dates.searchsorted(is_end, side="right")
    is_dates = dates[:split]
    is_schedule = schedule_rebalances(is_dates, start, lag)
    if not is_schedule:
        raise ValueError("The in-sample period contains no scheduled fills.")
    combos_tried = 0
    print("ROBUSTNESS (IS only)")
    for name, variant in variants.items():
        combos_tried += 1
        targets, _ = target_weights(is_dates, log_growth[:split], is_schedule, variant, seed)
        daily_results = simulate(is_dates, overnight[:split], intraday[:split], targets, start, variant.friction)
        print(f"{name}: {json.dumps(metrics(daily_results, risk_free.iloc[:split]), allow_nan=False)}")

    combos_tried += 1
    schedule = schedule_rebalances(dates, start, lag)
    targets, max_error = target_weights(dates, log_growth, schedule, model, seed)
    daily_results = simulate(dates, overnight, intraday, targets, start, model.friction)
    benchmark_name = "Monthly equal weight, same ETFs"
    equal = np.full(len(tickers) - 1, 1 / (len(tickers) - 1))
    benchmark_daily = simulate(dates, overnight, intraday, dict.fromkeys(targets, equal), start, model.friction)
    summaries = period_metrics(daily_results, risk_free, is_end)
    benchmark = period_metrics(benchmark_daily, risk_free, is_end)
    if min(summaries["is"]["n_trades"], summaries["oos"]["n_trades"]) == 0:
        raise RuntimeError("The main backtest requires fills in both reporting periods.")

    print(f"PARAM_COMBOS_TRIED: {combos_tried}")
    print(f"OPEN_REBALANCES: {len(schedule)}")
    print(f"MAX_RISK_BUDGET_ERROR: {max_error:.8g}")
    print(f"MIN_CASH: {daily_results['cash'].min():.8g}")
    for period, summary in summaries.items():
        print(f"{period.upper()}: {json.dumps(summary, allow_nan=False)}")
        print(f"BENCHMARK {period.upper()}: {json.dumps(benchmark[period], allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps({
        "strategy": config["strategy"],
        "data_range": {"start": str(daily_results.index[0].date()),
                       "end": str(daily_results.index[-1].date())},
        **summaries,
        "benchmark": {"name": benchmark_name, **benchmark},
    }, allow_nan=False))


def stop_at_deadline(_signum, _frame):
    raise TimeoutError("The backtest exceeded its 840-second time limit.")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_at_deadline)
    signal.alarm(840)
    try:
        main()
    finally:
        signal.alarm(0)
