"""514_risk_limited_asset_allocation_with_a_bud
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/514_risk_limited_asset_allocation_with_a_bud.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 5df80ea472f17ed604c7c4cdce92d6afd1dc3e36 on 2026-11-27 19:24 Europe/Oslo, exit 0, output sha256 2a4ff9e91dd41b3c7d1570cda9a3949af25d484bb6f8984dad3c94d84d46d45b
Data source: Free Yahoo Finance daily USD adjusted closing prices via yfinance for SPY, QQQ, EFA, TLT, GLD and VNQ, with BIL as the cash proxy and SPY as the benchmark. The recorded run used the cached snapshot fetched at 2026-11-27T08:00:19+00:00.
Data fetched: 2026-11-27T18:24:19+00:00
Sample period: Price history runs from 2007-10-01 through 2025-12-31, with data before 2009-01-02 used for warm-up. The backtest runs from 2009-01-02 through 2025-12-31, inclusive.   data_range (from output): {'start': '2009-01-02', 'end': '2025-12-31'}
IS/OOS split: IS is 2009-01-02 through 2017-12-29. OOS is 2018-01-02 through 2025-12-31. The default strategy runs continuously across the split and slices daily returns for reporting. The five sensitivity cases run only on IS. OOS is a historical holdout, and the README says the record cannot establish whether the specification was frozen before the first inspection of OOS results.
Costs and slippage: Each simulated buy or sell, including BIL, incurs 5 bps of transaction cost plus 5 bps of modeled slippage on its fill-close notional. The combined 10 bps reduces sell proceeds and increases buy outlay. The doubled-cost IS case charges 10 bps of each, or 20 bps per side. The 500 bps cash buffer is a sizing reserve and the sizing step budgets estimated fees separately.
Signal lag: The last trading session of each month generates a signal using a rolling price window through that session's close. The default lag is one trading session, so the target fills at the next session's close. Shares are fixed using the session immediately before the fill.
Price adjustment: yfinance auto_adjust=True adjusted close, fractional total-return units
Survivorship bias: The fixed universe consists of currently surviving ETFs with long histories, which creates retrospective selection and survivorship bias in both IS and OOS. The README states that point-in-time membership, ticker mappings, delistings and liquidation returns were not reconstructed. Yahoo Finance supplies the stitched history under the current QQQ ticker. The adjusted data are a retrospective snapshot and the impact of historical revisions is unknown.
Parameter combinations tried: 6
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.05517078973182454, ann_vol=0.07333356298854941, sharpe=0.7690568616377207, sharpe_excess=0.7592349363330637, max_drawdown=-0.11781278117149607, total_return=0.6204184871866858, turnover=3.1773092749252654, n_trades=316
    oos: cagr=0.04965508487667236, ann_vol=0.09222743359373665, sharpe=0.5720005047981821, sharpe_excess=0.3086632915564615, max_drawdown=-0.2263885181520372, total_return=0.4721610404699208, turnover=2.8316784737050584, n_trades=258
    full: cagr=0.052573156425622214, ann_vol=0.08274851532780331, sharpe=0.6608472010901224, sharpe_excess=0.518153640180404, max_drawdown=-0.22638851815203664, total_return=1.3855169660934443, turnover=3.014759335436529, n_trades=574
    benchmark SPY full: cagr=0.14391632108682018, ann_vol=0.1771159173513372, sharpe=0.8480856955353564, sharpe_excess=0.7808392890505457, max_drawdown=-0.335539996630962, total_return=8.791485236875316, turnover=0.11647722335832132, n_trades=2
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: '514_risk_limited_asset_allocation_with_a_bud'\nuniverse: [SPY, QQQ, EFA, TLT, GLD, VNQ]\ncash_proxy: BIL\nbenchmark: SPY\n\ndates:\n  price_start: 2007-10-01\n  backtest_start: 2009-01-02\n  is_end: 2017-12-29\n  backtest_end: 2025-12-31\n\nsignal:\n  lookback_sessions: 252\n  horizon_sessions: 21\n  risk_barrier: 0.40\n  downside_aversion: 8.0\n  max_asset_weight: 0.30\n\nexecution:\n  # Size at the close immediately before the fill session, using the lagged signal.\n  lag_sessions: 1  # Month-end close signal fills at the next session's close.\n  cost_bps: 5.0\n  slippage_bps: 5.0\n  # Reserve 5% of prior-close value, including at SPY entry, and budget fees separately.\n  # Keep any proceeds left after settlement in zero-interest cash.\n  cash_buffer_bps: 500.0\n\nrobustness:\n  - label: risk_barrier_0.35\n    signal: {risk_barrier: 0.35}\n  - label: risk_barrier_0.45\n    signal: {risk_barrier: 0.45}\n  - label: downside_aversion_6\n    signal: {downside_aversion: 6.0}\n  - label: downside_aversion_10\n    signal: {downside_aversion: 10.0}\n  - label: double_execution_cost\n    execution: {cost_bps: 10.0, slippage_bps: 10.0}\n"
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
"""Monthly long-only ETF backtest based on Giller's budget threshold."""

import hashlib
import json
import signal
import tempfile
from datetime import datetime, timezone
from pathlib import Path

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
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
TIME_LIMIT_SECONDS = 570
CALENDAR_START = pd.Timestamp("2007-10-01")
CALENDAR_END = pd.Timestamp("2025-12-31")
EXTRA_MARKET_CLOSURES = pd.DatetimeIndex([
    "2012-10-29", "2012-10-30",  # Hurricane Sandy
    "2018-12-05",  # George H. W. Bush national day of mourning
    "2025-01-09",  # Jimmy Carter national day of mourning
])
US_EQUITY_HOLIDAYS = AbstractHolidayCalendar(rules=[
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
])


class DeadlineReached(BaseException):
    pass


def stop_at_deadline(_signum, _frame):
    raise DeadlineReached


def validate_sessions(dates, start, end):
    if not CALENDAR_START <= start <= end <= CALENDAR_END:
        raise ValueError(f"Equity calendar covers {CALENDAR_START.date()} through {CALENDAR_END.date()}")
    if dates.empty or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Price history must contain unique, sorted dates")
    closures = US_EQUITY_HOLIDAYS.holidays(start, end).union(EXTRA_MARKET_CLOSURES)
    expected = pd.date_range(start, end, freq="B").difference(closures)
    missing = expected.difference(dates)
    unexpected = dates.difference(expected)
    if len(missing) or len(unexpected):
        raise ValueError(
            f"Price dates differ from the U.S. equity calendar: "
            f"missing {list(missing.strftime('%Y-%m-%d')[:5])}, "
            f"unexpected {list(unexpected.strftime('%Y-%m-%d')[:5])}"
        )
    if dates[0] != start or dates[-1] != end:
        raise ValueError(f"Start and end must be trading sessions: {start.date()} through {end.date()}")


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    cache_key = f"auto_adjust daily close {start.date()} {end.date()} {' '.join(tickers)}"
    key = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    path = CACHE / f"yfinance_{key}.csv"
    stamp_prefix = "# DATA_FETCHED_AT: "

    if path.is_file():
        with path.open(encoding="utf-8") as source:
            header = source.readline().strip()
            if not header.startswith(stamp_prefix):
                raise ValueError(f"Missing fetch timestamp in {path.name}")
            fetched_at = header.removeprefix(stamp_prefix)
            if datetime.fromisoformat(fetched_at).utcoffset() is None:
                raise ValueError(f"Fetch timestamp in {path.name} needs a UTC offset")
            prices = pd.read_csv(source, index_col="date", parse_dates=["date"])
    else:
        with tempfile.TemporaryDirectory(dir=CACHE, prefix="yfinance_") as yf_cache:
            yf.set_tz_cache_location(yf_cache)
            data = yf.download(
                tickers=tickers,
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d",
                auto_adjust=True,
                ignore_tz=True,
                keepna=True,
                threads=False,
                progress=False,
                timeout=15,
            )
        if data is None or data.empty or "Close" not in data.columns.get_level_values(0):
            raise RuntimeError("yfinance returned no daily adjusted closes")
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        prices = data["Close"].sort_index()

    prices = prices.reindex(columns=tickers).rename_axis(columns=None)
    prices.index = prices.index.as_unit("ns").rename("date")
    validate_sessions(prices.index, start, end)
    values = prices.to_numpy(dtype=float)
    good = np.isfinite(values) & (values > 0)
    if not good.all():
        row, col = np.argwhere(~good)[0]
        raise ValueError(f"Missing or invalid adjusted close for {tickers[col]} on {prices.index[row].date()}")

    if not path.is_file():
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as out:
            temp_path = Path(out.name)
            try:
                out.write(f"{stamp_prefix}{fetched_at}\n")
                prices.to_csv(out, index_label="date", float_format="%.17g")
                out.flush()
                temp_path.replace(path)
            finally:
                temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def target_weights(scenarios, risk_barrier, downside_aversion, max_asset_weight):
    """Combine the paper's per-asset barrier with a joint downside penalty."""
    expected = scenarios.mean(axis=0)
    score = expected - risk_barrier * scenarios.std(axis=0, ddof=1)
    assets = len(score)
    if score.max() <= 0:
        return np.zeros(assets)

    centered = scenarios - expected

    def objective(weights):
        downside = np.minimum(centered @ weights, 0.0)
        penalty = 0.5 * downside_aversion * np.mean(downside**2)
        gradient = downside_aversion * (centered.T @ downside) / len(downside)
        return penalty - score @ weights, gradient - score

    result = minimize(
        objective,
        np.zeros(assets),
        jac=True,
        method="SLSQP",
        bounds=[(0.0, max_asset_weight)] * assets,
        constraints={"type": "ineq", "fun": lambda w: 1.0 - w.sum(), "jac": lambda w: -np.ones(assets)},
        options={"ftol": 1e-12, "maxiter": 150},
    )
    weights = result.x
    if not result.success or not np.isfinite(weights).all() or weights.sum() > 1.0 + 1e-8:
        raise RuntimeError(f"Allocation solve failed: {result.message}")
    weights = np.clip(weights, 0.0, max_asset_weight)
    return weights / max(1.0, weights.sum())


def execution_targets(prices, universe, cash_proxy, backtest_start, signal_cfg, lag):
    lookback = signal_cfg["lookback_sessions"]
    horizon = signal_cfg["horizon_sessions"]
    if not (1 <= horizon < lookback and 0 < signal_cfg["max_asset_weight"] <= 1):
        raise ValueError("Need 1 <= horizon < lookback and 0 < max_asset_weight <= 1")
    if not (signal_cfg["risk_barrier"] >= 0 and signal_cfg["downside_aversion"] >= 0):
        raise ValueError("Risk penalties must be nonnegative")
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("Execution lag must be at least one session")

    dates = prices.index
    months = dates.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])
    values = prices[[*universe, cash_proxy]].to_numpy()
    targets = {}
    for pos in month_ends:
        fill_pos = pos + lag
        if fill_pos >= len(dates) or dates[fill_pos] < backtest_start:
            continue
        if pos < lookback:
            raise ValueError(f"Insufficient price history for the {dates[pos].date()} signal")
        window = values[pos - lookback : pos + 1]
        horizon_growth = window[horizon:] / window[:-horizon]
        scenarios = horizon_growth[:, :-1] - horizon_growth[:, -1:]
        try:
            risky = target_weights(
                scenarios,
                signal_cfg["risk_barrier"],
                signal_cfg["downside_aversion"],
                signal_cfg["max_asset_weight"],
            )
        except RuntimeError as error:
            raise RuntimeError(f"Signal on {dates[pos].date()}: {error}") from error
        targets[dates[fill_pos]] = np.append(risky, max(0.0, 1.0 - risky.sum()))
    return targets


def size_orders(positions, cash, previous_close, target, fee_rate, buffer_rate):
    """Lock share quantities using the previous session's close and cash."""
    previous_value = cash + float(positions @ previous_close)
    reserve = previous_value * buffer_rate
    changes = (previous_value - reserve) * target / previous_close - positions
    changes[np.abs(changes) * previous_close <= previous_value * 1e-10] = 0.0
    sell_shares = np.maximum(-changes, 0.0)
    buy_shares = np.maximum(changes, 0.0)
    estimated_proceeds = float(sell_shares @ (previous_close * (1.0 - fee_rate)))
    estimated_spend = float(buy_shares @ (previous_close * (1.0 + fee_rate)))
    budget = max(0.0, cash + estimated_proceeds - reserve)
    if estimated_spend > budget:
        buy_shares *= budget / estimated_spend
    # Settlement can read these quantities to price the fill and fees.
    sell_shares.setflags(write=False)
    buy_shares.setflags(write=False)
    return sell_shares, buy_shares


def fill_orders(cash, sell_shares, buy_shares, close, fee_rate, date):
    """Settle the fixed quantities at the close, including fees and slippage."""
    proceeds = float(sell_shares @ (close * (1.0 - fee_rate)))
    spent = float(buy_shares @ (close * (1.0 + fee_rate)))
    remaining_cash = cash + proceeds - spent
    if not np.isfinite(remaining_cash):
        raise RuntimeError(f"{date.date()}: scheduled fill would leave nonfinite cash")
    if remaining_cash < 0.0:
        raise RuntimeError(f"{date.date()}: scheduled fill would leave negative cash ({remaining_cash:.8g})")
    return remaining_cash


def simulate(prices, start, targets, execution_cfg):
    cost = execution_cfg["cost_bps"]
    slippage = execution_cfg["slippage_bps"]
    if not (cost >= 0 and slippage >= 0 and cost + slippage < 5000):
        raise ValueError("Costs must be nonnegative and sum to less than 5000 bps")
    fee_rate = (cost + slippage) / 10_000
    buffer_rate = execution_cfg["cash_buffer_bps"] / 10_000
    if not 0 < buffer_rate < 1:
        raise ValueError("Cash buffer must be between 0 and 10000 bps")
    start_pos = prices.index.get_loc(start)
    if start_pos == 0:
        raise ValueError("Simulation needs the close of the session before its start")
    dates = prices.index[start_pos:]
    # A unit is the total-return value of one dollar at the initial close.
    price_scale = prices.iloc[start_pos - 1].to_numpy(dtype=float)
    closes = prices.iloc[start_pos:].to_numpy(dtype=float)
    previous_close = np.ones_like(price_scale)
    positions = np.zeros(prices.shape[1])
    positions[-1] = 1.0
    cash = 0.0
    daily_results = []
    for offset, date in enumerate(dates):
        previous_value = cash + float(positions @ previous_close)
        turnover, trades = 0.0, 0
        target = targets.get(date)
        planned = None
        if target is not None:
            sell_shares, buy_shares = size_orders(positions, cash, previous_close, target, fee_rate, buffer_rate)
            next_positions = positions - sell_shares + buy_shares
            if not np.isfinite(next_positions).all() or (next_positions < 0.0).any():
                raise RuntimeError(f"{date.date()}: scheduled quantities would leave invalid positions")
            planned = sell_shares, buy_shares, next_positions

        # The current close first enters the simulation after quantities are locked.
        close = closes[offset] / price_scale
        if planned is not None:
            sell_shares, buy_shares, next_positions = planned
            remaining_cash = fill_orders(cash, sell_shares, buy_shares, close, fee_rate, date)
            value_before_fill = cash + float(positions @ close)
            traded_shares = sell_shares + buy_shares
            turnover = float(traded_shares @ close) / value_before_fill
            trades = int(np.count_nonzero(traded_shares))
            positions, cash = next_positions, remaining_cash
        close_value = cash + float(positions @ close)
        net_return = close_value / previous_value - 1.0
        bills_return = close[-1] / previous_close[-1] - 1.0
        daily_results.append((net_return, bills_return, turnover, trades))
        previous_close = close
    return pd.DataFrame(daily_results, index=dates, columns=["net", "bills", "turnover", "trades"])


def metrics(daily_results):
    net = daily_results["net"]
    if net.empty:
        raise ValueError("Each measurement period needs at least one return observation")
    growth = np.r_[1.0, (1.0 + net.to_numpy()).cumprod()]
    excess = net - daily_results["bills"]
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    ann_vol = net.std(ddof=1) * annualizer
    excess_vol = excess.std(ddof=1) * annualizer
    years = len(net) / SESSIONS_PER_YEAR
    return {
        "cagr": float(growth[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(ann_vol) if np.isfinite(ann_vol) else None,
        "sharpe": float(SESSIONS_PER_YEAR * net.mean() / ann_vol) if ann_vol > 1e-12 else None,
        "sharpe_excess": float(SESSIONS_PER_YEAR * excess.mean() / excess_vol) if excess_vol > 1e-12 else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1.0).min()),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["trades"].sum()),
    }


def periods(daily_results, is_end):
    return {
        "is": metrics(daily_results.loc[:is_end]),
        "oos": metrics(daily_results.loc[daily_results.index > is_end]),
        "full": metrics(daily_results),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    cash_proxy = config["cash_proxy"]
    benchmark_ticker = config["benchmark"]
    if not universe or cash_proxy in universe or len(set(universe)) != len(universe) or cash_proxy == benchmark_ticker:
        raise ValueError("Use distinct risky ETFs, a cash proxy, and a risky benchmark")
    price_start, start, is_end, end = (
        pd.Timestamp(config["dates"][key]) for key in ("price_start", "backtest_start", "is_end", "backtest_end")
    )
    if not price_start < start < is_end < end:
        raise ValueError("Dates must satisfy price_start < backtest_start < is_end < backtest_end")
    tickers = list(dict.fromkeys([*universe, cash_proxy, benchmark_ticker]))
    prices, fetched_at = load_prices(tickers, price_start, end)
    if start not in prices.index or is_end not in prices.index:
        raise ValueError("The backtest start and IS end must be observed price dates")
    strategy_prices = prices[[*universe, cash_proxy]]
    signal_cfg = config["signal"]
    execution_cfg = config["execution"]

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True adjusted close, fractional total-return units")
    print("ROBUSTNESS: in-sample only, fixed one-at-a-time cases", flush=True)
    attempts = 0
    for case in config["robustness"]:
        case_execution = {**execution_cfg, **case.get("execution", {})}
        attempts += 1
        targets = execution_targets(
            prices.loc[:is_end], universe, cash_proxy, start,
            {**signal_cfg, **case.get("signal", {})}, case_execution["lag_sessions"],
        )
        result = simulate(strategy_prices.loc[:is_end], start, targets, case_execution)
        print(f"ROBUSTNESS_IS {case['label']}: {json.dumps(metrics(result), allow_nan=False)}")

    attempts += 1
    targets = execution_targets(prices, universe, cash_proxy, start, signal_cfg, execution_cfg["lag_sessions"])
    default_periods = periods(simulate(strategy_prices, start, targets, execution_cfg), is_end)
    benchmark_prices = prices[[benchmark_ticker, cash_proxy]]
    benchmark_targets = {start: np.array([1.0, 0.0])}
    benchmark = periods(simulate(benchmark_prices, start, benchmark_targets, execution_cfg), is_end)
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        **default_periods,
        "benchmark": {"ticker": benchmark_ticker, **benchmark},
    }
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(default_periods[period], allow_nan=False)}")
        print(f"BENCHMARK {benchmark_ticker} {period.upper()}: {json.dumps(benchmark[period], allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {attempts}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_at_deadline)
    signal.alarm(TIME_LIMIT_SECONDS)
    try:
        main()
    except DeadlineReached:
        raise SystemExit(f"Backtest exceeded its {TIME_LIMIT_SECONDS}-second time limit") from None
    finally:
        signal.alarm(0)
