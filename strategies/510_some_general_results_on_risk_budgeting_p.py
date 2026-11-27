# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "510_ some_general_results_on_risk_budgeting_p"\nuniverse: [SPY, IEF, TLT, GLD, DBC]\nrisk_free_proxy: BIL\ndata_start: "2008-06-01"\nstart: "2010-01-01"\nis_end: "2017-12-31"\nend: "2025-12-31"\nlookback_sessions: 252\nrebalance: monthly\nexecution_lag_sessions: 1\ncost_bps: 5.0\nslippage_bps: 5.0\nsolver:\n  contraction: 0.5\n  tolerance: 1.0e-6\n  max_iterations: 1000\nrobustness:\n  lookback_sessions: [210, 294]\n  higher_cost_bps: 10.0\n  higher_slippage_bps: 10.0\n'
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
"""Monthly standard-deviation risk budgeting, using public ETF prices."""

import json
import signal
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
PRICE_ADJUSTMENT = "yfinance auto_adjust=True daily Close (total-return proxy)"


def check_prices(prices, symbols, start, end):
    prices = prices.loc[start:end].reindex(columns=symbols)
    if prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be nonempty, unique and ordered.")
    if prices.index[0] > start + pd.Timedelta(days=5) or prices.index[-1] < end - pd.Timedelta(days=5):
        raise ValueError("Price data do not span the configured dates.")
    values = prices.to_numpy(dtype=float)
    invalid = ~np.isfinite(values) | (values <= 0)
    if invalid.any():
        row, col = np.argwhere(invalid)[0]
        raise ValueError(f"Missing or invalid close for {symbols[col]} on {prices.index[row].date()}.")
    return prices


def load_prices(symbols, start, end):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_key = "_".join(sorted(symbols))
    path = CACHE_DIR / f"yf_adjusted_{cache_key}_{start.date()}_{end.date()}.csv"
    if path.is_file():
        with path.open(encoding="utf-8") as source:
            fetched_at = source.readline().strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(source, index_col="Date", parse_dates=["Date"])
        return check_prices(prices, symbols, start, end), fetched_at

    with TemporaryDirectory(dir=CACHE_DIR, prefix="yfinance_") as provider_cache:
        yf.set_tz_cache_location(provider_cache)
        for threads in (True, False):
            raw = yf.download(
                symbols, start=start.date().isoformat(), end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d", auto_adjust=True, ignore_tz=True, multi_level_index=True,
                progress=False, threads=threads, timeout=8,
            )
            try:
                prices = check_prices(raw["Close"], symbols, start, end)
                break
            except (KeyError, ValueError) as error:
                if not threads:
                    raise RuntimeError("Could not obtain complete adjusted closes from Yahoo Finance.") from error

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE_DIR, delete=False) as target:
        temporary_path = Path(target.name)
        try:
            target.write(f"{fetched_at}\n")
            prices.to_csv(target, index_label="Date")
            target.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def risk_parity(covariance, contraction, tolerance, max_iterations):
    """Follow the paper's residual direction on the simplex for volatility risk."""
    np.linalg.cholesky(covariance)
    covariance = covariance / np.mean(np.diag(covariance))
    budget = np.full(len(covariance), 1.0 / len(covariance))
    weights = budget.copy()

    for iteration in range(max_iterations + 1):
        marginal = covariance @ weights
        contributions = weights * marginal
        variance = contributions.sum()
        residual = contributions - budget * variance
        error = np.linalg.norm(residual)
        if error <= tolerance * variance:
            return weights
        if iteration == max_iterations:
            break

        # Normalizing the direction rescales the scalar step and the polynomial roots.
        direction = residual / error
        lower = 0.999999 * np.max(-weights[direction > 0] / direction[direction > 0])
        upper = 0.999999 * np.min(-weights[direction < 0] / direction[direction < 0])
        cov_direction = covariance @ direction
        linear = direction * marginal + weights * cov_direction - 2 * budget * (direction @ marginal)
        quadratic = direction * cov_direction - budget * (direction @ cov_direction)

        # First try the step from the linear term. If needed, minimize the exact quartic
        # along the feasible line by checking both endpoints and all stationary points.
        slope_squared = linear @ linear
        trial = np.clip(-(residual @ linear) / slope_squared, lower, upper) if slope_squared else 0.0
        trial_error = np.linalg.norm(residual + trial * (linear + trial * quadratic))
        if trial_error > contraction * error:
            roots = np.roots([
                2 * (quadratic @ quadratic), 3 * (linear @ quadratic),
                slope_squared + 2 * (residual @ quadratic), residual @ linear,
            ])
            candidates = [lower, upper, trial]
            candidates.extend(
                root.real for root in roots if abs(root.imag) <= 1e-10 and lower <= root.real <= upper
            )
            step_errors = [np.linalg.norm(residual + step * (linear + step * quadratic)) for step in candidates]
            best = int(np.argmin(step_errors))
            trial, trial_error = candidates[best], step_errors[best]
            if trial_error >= error:
                raise RuntimeError("No feasible residual-reducing step for the risk budget.")

        weights = weights + trial * direction
        weights /= weights.sum()

    raise RuntimeError(f"Risk budget exceeded {max_iterations} iterations.")


def monthly_schedule(dates, start, lag):
    months = dates.to_period("M")
    # Only session labels identify the prior month-end. Its price window ends there.
    signal_rows = np.flatnonzero(months[:-1] != months[1:])
    schedule = {}
    for signal_row in signal_rows:
        fill_row = signal_row + lag
        if fill_row < len(dates) and dates[fill_row] >= start:
            schedule[fill_row] = signal_row
    return schedule


def risk_targets(asset_returns, dates, schedule, lookback, solver):
    targets = {}
    for fill_row, signal_row in schedule.items():
        first_row = signal_row - lookback + 1
        if first_row < 1:
            raise ValueError(f"Insufficient warm-up for the {lookback}-session lookback on {dates[signal_row].date()}.")
        window = asset_returns[first_row : signal_row + 1]
        covariance = np.cov(window, rowvar=False, ddof=1)
        try:
            targets[fill_row] = risk_parity(covariance, **solver)
        except (RuntimeError, np.linalg.LinAlgError) as error:
            raise RuntimeError(f"Risk-budget solve failed for {dates[signal_row].date()}.") from error
    return targets


def simulate(asset_returns, dates, start, targets, cost_bps, slippage_bps):
    first_row = dates.searchsorted(start)
    rate = (cost_bps + slippage_bps) / 10_000.0
    weights = np.zeros(asset_returns.shape[1])
    daily_results = []
    for row in range(first_row, len(dates)):
        # The old weights earn the move to this close. New weights earn the next move.
        asset_moves = weights * (1.0 + asset_returns[row])
        gross = 1.0 + weights @ asset_returns[row]
        weights = asset_moves / gross
        scale, turnover, trades = 1.0, 0.0, 0
        if row in targets:
            target = targets[row]

            def settlement(fraction):
                return fraction - 1.0 + rate * np.abs(fraction * target - weights).sum()

            scale = brentq(settlement, 0.0, 1.0, xtol=1e-14)
            changes = np.abs(scale * target - weights)
            turnover = changes.sum()
            trades = np.count_nonzero(changes > 1e-10)
            weights = target
        daily_results.append((gross * scale - 1.0, turnover, trades))

    return pd.DataFrame(daily_results, index=dates[first_row:], columns=["return", "turnover", "n_trades"])


def metrics(daily_results, risk_free):
    returns = daily_results["return"].to_numpy()
    if len(returns) < 2:
        raise ValueError("Each evaluation period needs at least two sessions.")
    growth = np.r_[1.0, np.cumprod(1.0 + returns)]
    excess = returns - risk_free.loc[daily_results.index].to_numpy()
    vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    years = len(returns) / SESSIONS_PER_YEAR
    return {
        "cagr": float(growth[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(vol * annualizer),
        "sharpe": float(returns.mean() / vol * annualizer) if vol else None,
        "sharpe_excess": float(excess.mean() / excess_vol * annualizer) if excess_vol else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1.0).min()),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    symbols = config["universe"]
    proxy = config["risk_free_proxy"]
    data_start, start, split, end = (pd.Timestamp(config[key]) for key in ("data_start", "start", "is_end", "end"))
    lookback, lag = config["lookback_sessions"], config["execution_lag_sessions"]
    solver = config["solver"]
    robustness = config["robustness"]
    fees = config["cost_bps"], config["slippage_bps"]
    higher_fees = robustness["higher_cost_bps"], robustness["higher_slippage_bps"]
    cases = [(window, fees) for window in robustness["lookback_sessions"]] + [(lookback, higher_fees)]
    if not data_start < start <= split < end or len(set([*symbols, proxy])) != len(symbols) + 1 or len(symbols) < 2:
        raise ValueError("Check the date ordering and the distinct asset and cash proxy symbols.")
    if config["rebalance"] != "monthly" or not isinstance(lag, int) or lag < 1:
        raise ValueError("Use monthly rebalancing and a positive whole-session execution lag.")
    for window, (cost, slippage) in [(lookback, fees), *cases]:
        if (
            not isinstance(window, int) or window <= len(symbols)
            or not (0 <= cost and 0 <= slippage and cost + slippage < 10_000)
        ):
            raise ValueError("Check covariance lookbacks and nonnegative cost and slippage in bps.")
    if not 0 < solver["contraction"] < 1 or not 0 < solver["tolerance"] < 1 or solver["max_iterations"] < 1:
        raise ValueError("Check the fixed-point solver settings.")

    prices, fetched_at = load_prices([*symbols, proxy], data_start, end)
    returns = prices.pct_change(fill_method=None)
    dates = returns.index
    asset_returns = returns[symbols].to_numpy()
    risk_free = returns[proxy]

    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"BENCHMARK: monthly equal weight, same universe, execution lag and fees | EXCESS_SHARPE_PROXY: {proxy}")

    is_dates = dates[dates <= split]
    is_returns = asset_returns[:len(is_dates)]
    is_schedule = monthly_schedule(is_dates, start, lag)
    is_targets = {
        window: risk_targets(is_returns, is_dates, is_schedule, window, solver)
        for window in dict.fromkeys(window for window, _ in cases)
    }
    for window, (cost, slippage) in cases:
        daily_case = simulate(is_returns, is_dates, start, is_targets[window], cost, slippage)
        case = {
            "lookback_sessions": window, "cost_bps": cost, "slippage_bps": slippage,
            "is": metrics(daily_case, risk_free.loc[is_dates]),
        }
        print("ROBUSTNESS_IS: " + json.dumps(case, allow_nan=False))

    schedule = monthly_schedule(dates, start, lag)
    if not schedule:
        raise ValueError("The selected dates leave no monthly rebalances.")
    targets = risk_targets(asset_returns, dates, schedule, lookback, solver)
    daily_results = simulate(asset_returns, dates, start, targets, *fees)
    equal = np.full(len(symbols), 1.0 / len(symbols))
    benchmark = simulate(asset_returns, dates, start, dict.fromkeys(schedule, equal), *fees)
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": str(daily_results.index[0].date()), "end": str(daily_results.index[-1].date())},
        "is": metrics(daily_results.loc[:split], risk_free),
        "oos": metrics(daily_results.loc[daily_results.index > split], risk_free),
        "full": metrics(daily_results, risk_free),
        "benchmark": metrics(benchmark, risk_free),
    }
    for key in ("is", "oos", "full", "benchmark"):
        print(f"{key.upper()}: {json.dumps(results[key], allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {1 + len(cases)}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    main()
