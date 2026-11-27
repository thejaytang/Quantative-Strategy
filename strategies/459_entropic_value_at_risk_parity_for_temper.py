# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 459_ entropic_value_at_risk_parity_for_temper\nuniverse: [VTI, EFA, VWO, VNQ, AGG, TIP, GLD]\ncash_proxy: BIL\ndata_start: 2005-04-01\nbacktest_start: 2007-07-01\nis_end: 2016-12-31\ndata_end: 2026-03-31\nlookback_months: 12\ntail_probability: 0.05\nsignal_lag_sessions: 1\ncost_bps: 5\nslippage_bps: 5\n'
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
"""Research backtest for empirical EVaR-deviation equal risk contribution."""

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, minimize
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
PAPER_ETFS = {"VTI", "EFA", "VWO", "VNQ", "AGG", "TIP", "GLD"}
SESSIONS_PER_YEAR = 252
FETCH_HEADER = "# DATA_FETCHED_AT: "


def validate_prices(prices, universe, cash_proxy, start, end):
    if prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Adjusted closes must have a nonempty, sorted, unique date index.")
    if not start <= prices.index[0] < start + pd.Timedelta(days=7):
        raise ValueError("Price history does not reach the requested start.")
    if not end - pd.Timedelta(days=7) < prices.index[-1] <= end:
        raise ValueError("Price history does not reach the requested end.")
    cash_start = prices[cash_proxy].first_valid_index()
    if cash_start is None:
        raise ValueError(f"No adjusted closes for {cash_proxy}.")
    risky = prices[universe].to_numpy()
    cash = prices.loc[cash_start:, cash_proxy].to_numpy()
    if not (np.isfinite(risky).all() and np.isfinite(cash).all()
            and (risky > 0).all() and (cash > 0).all()):
        raise ValueError("A required adjusted close is invalid or missing.")


def load_prices(universe, cash_proxy, start, end):
    tickers = sorted([*universe, cash_proxy])
    key = sha256(f"{tickers}|{start.date()}|{end.date()}".encode()).hexdigest()[:16]
    path = CACHE_DIR / f"yfinance_adjusted_close_{key}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as cached:
            header = cached.readline().strip()
            if not header.startswith(FETCH_HEADER):
                raise ValueError(f"Missing fetch timestamp in {path.name}.")
            fetched_at = datetime.fromisoformat(header.removeprefix(FETCH_HEADER))
            prices = pd.read_csv(cached, index_col="Date", parse_dates=["Date"])
        validate_prices(prices, universe, cash_proxy, start, end)
        return prices, fetched_at

    CACHE_DIR.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=CACHE_DIR, prefix="yfinance_") as library_cache:
        yf.set_tz_cache_location(library_cache)
        downloaded = yf.download(
            tickers, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=True, progress=False, timeout=15,
        )
    fetched_at = datetime.now(timezone.utc)
    if downloaded.empty:
        raise RuntimeError("Yahoo Finance returned no daily prices.")
    prices = downloaded["Close"].reindex(columns=tickers)
    prices.index = prices.index.tz_localize(None)
    validate_prices(prices, universe, cash_proxy, start, end)

    with NamedTemporaryFile(mode="w", dir=CACHE_DIR, encoding="utf-8", delete_on_close=False) as cached:
        cached.write(FETCH_HEADER + fetched_at.isoformat() + "\n")
        prices.to_csv(cached, index_label="Date")
        cached.flush()
        Path(cached.name).replace(path)
    return prices, fetched_at


def evar_and_gradient(weights, loss_scenarios, entropy_budget):
    losses = loss_scenarios @ weights
    worst = losses.max()
    relative_losses = losses - worst
    at_worst = relative_losses == 0
    log_count = np.log(len(losses))

    # When the entropy constraint admits all mass on the worst losses, EVaR is their maximum.
    if entropy_budget >= log_count - np.log(np.count_nonzero(at_worst)):
        return worst, loss_scenarios[at_worst].mean(axis=0)

    spread = -relative_losses.min()
    scaled_losses = relative_losses / spread

    def tilted_distribution(tilt):
        # The largest scaled loss is zero, so these exponential weights cannot overflow.
        weights = np.exp(tilt * scaled_losses)
        normalizer = weights.sum()
        return weights / normalizer, np.log(normalizer)

    def entropy_gap(tilt):
        probabilities, log_normalizer = tilted_distribution(tilt)
        return tilt * (probabilities @ scaled_losses) - log_normalizer + log_count - entropy_budget

    upper = 1.0
    while entropy_gap(upper) < 0:
        upper *= 2
    tilt = brentq(entropy_gap, 0, upper)
    probabilities, _ = tilted_distribution(tilt)
    return probabilities @ losses, probabilities @ loss_scenarios


def equal_risk_weights(returns, tail_probability):
    # Center each asset over the trailing window and use percentage units for optimization.
    losses = -100 * (returns - returns.mean(axis=0))
    entropy_budget = -np.log(tail_probability)
    budget = np.full(losses.shape[1], 1 / losses.shape[1])
    risk, _ = evar_and_gradient(budget, losses, entropy_budget)
    if risk <= 0:
        raise ValueError("Cannot allocate a positive risk budget to a riskless portfolio.")

    def objective(weights):
        risk, gradient = evar_and_gradient(weights, losses, entropy_budget)
        return risk - budget @ np.log(weights), gradient - budget / weights

    # Positive homogeneous risk plus the log barrier gives x_i dEVaR/dx_i = budget_i.
    solution = minimize(
        objective, budget / risk, jac=True, method="L-BFGS-B",
        bounds=[(1e-10, None)] * len(budget),
        options={"ftol": 1e-15, "gtol": 1e-9, "maxiter": 1000},
    )
    weights = solution.x / solution.x.sum()
    risk, gradient = evar_and_gradient(weights, losses, entropy_budget)
    if risk <= 0 or not np.allclose(weights * gradient / risk, budget, atol=1e-5, rtol=0):
        raise RuntimeError(f"EVaR risk budgets did not converge: {solution.message}")
    return weights


def rebalance_schedule(returns, start, lookback, tail_probability, lag):
    dates = returns.index
    first_price = dates[0]
    month_ends = dates.to_series().groupby(dates.to_period("M")).last()
    targets = {}
    for signal in month_ends:
        fill_index = dates.get_loc(signal) + lag
        if fill_index >= len(dates) or dates[fill_index] < start:
            continue
        cutoff = signal - pd.DateOffset(months=lookback)
        if first_price > cutoff:
            continue
        window = returns.iloc[dates.searchsorted(cutoff, side="right"):dates.get_loc(signal) + 1]
        try:
            targets[dates[fill_index]] = equal_risk_weights(window.to_numpy(), tail_probability)
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(f"Rebalance signal on {signal.date()} failed: {error}") from error
    if not targets:
        raise ValueError("No rebalance has both a full estimation window and a fill session.")
    return targets


def simulate(returns, targets, fee_rate):
    weights = np.zeros(returns.shape[1])
    rows = []
    for date, asset_returns in zip(returns.index, returns.to_numpy()):
        # Old holdings earn the entire close-to-close move before the close fill.
        drifted = weights * (1 + asset_returns)
        gross_factor = 1 + weights @ asset_returns
        weights = drifted / gross_factor
        turnover, n_trades = 0.0, 0
        if date in targets:
            target = targets[date]
            # Target amounts use NAV after paying fees on both purchases and sales.
            def fee_balance(nav_fraction):
                traded = np.abs(nav_fraction * target - weights).sum()
                return nav_fraction + fee_rate * traded - 1

            nav_fraction = brentq(fee_balance, 0, 1)
            trades = np.abs(nav_fraction * target - weights)
            turnover = trades.sum()
            n_trades = np.count_nonzero(trades > 1e-10)
            gross_factor *= nav_fraction
            weights = target
        rows.append((gross_factor - 1, turnover, n_trades))
    return pd.DataFrame(rows, index=returns.index, columns=["return", "turnover", "n_trades"])


def metrics(path, cash_returns):
    returns = path["return"]
    nav = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    years = len(returns) / SESSIONS_PER_YEAR
    vol = returns.std(ddof=1)
    excess = returns - cash_returns.loc[returns.index]
    excess_vol = excess.std(ddof=1)
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(nav[-1] ** (1 / years) - 1),
        "ann_vol": float(vol * annualizer),
        "sharpe": float(returns.mean() / vol * annualizer) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * annualizer) if excess_vol > 0 else None,
        "max_drawdown": float(np.min(nav / np.maximum.accumulate(nav) - 1)),
        "total_return": float(nav[-1] - 1),
        "turnover": float(path["turnover"].sum() / years),
        "n_trades": int(path["n_trades"].sum()),
    }


def period_metrics(path, cash_returns, split):
    periods = {"is": path.loc[:split], "oos": path.loc[path.index > split], "full": path}
    if any(len(period) < 2 for period in periods.values()):
        raise ValueError("Both IS and OOS need at least two evaluated sessions.")
    return {name: metrics(period, cash_returns) for name, period in periods.items()}


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    universe, cash_proxy = config["universe"], config["cash_proxy"]
    if len(universe) < 2 or len(set(universe)) != len(universe) or not set(universe) <= PAPER_ETFS:
        raise ValueError("Choose at least two distinct ETFs from the paper's XASSET universe.")
    if cash_proxy != "BIL":
        raise ValueError("The excess-return reference is BIL.")
    data_start, start, split, end = (
        pd.Timestamp(config[field]) for field in ("data_start", "backtest_start", "is_end", "data_end")
    )
    if not data_start < start <= split < end:
        raise ValueError("Dates must be ordered as data start, backtest start, IS end, data end.")
    lag, lookback = config["signal_lag_sessions"], config["lookback_months"]
    tail = config["tail_probability"]
    cost, slippage = config["cost_bps"], config["slippage_bps"]
    if type(lag) is not int or type(lookback) is not int or min(lag, lookback) < 1:
        raise ValueError("Execution lag and lookback must be positive integers.")
    if not (0 < tail < 1 and cost >= 0 and slippage >= 0 and cost + slippage < 10_000):
        raise ValueError("Invalid tail probability, transaction cost or slippage.")

    prices, fetched_at = load_prices(universe, cash_proxy, data_start, end)
    returns = prices.pct_change(fill_method=None)
    # Each solve has at most seven weights, so BLAS thread startup dominates its runtime.
    with threadpool_limits(limits=1, user_api="blas"):
        targets = rebalance_schedule(returns[universe], start, lookback, tail, lag)
    evaluated = returns.loc[min(targets):]
    cash_returns = evaluated[cash_proxy]
    if cash_returns.isna().any():
        raise ValueError("BIL needs an adjusted close on and before the first evaluated session.")
    equal = np.full(len(universe), 1 / len(universe))
    benchmark_targets = dict.fromkeys(targets, equal)
    fee_rate = (cost + slippage) / 10_000
    path = simulate(evaluated[universe], targets, fee_rate)
    benchmark = simulate(evaluated[universe], benchmark_targets, fee_rate)
    strategy_results = period_metrics(path, cash_returns, split)
    benchmark_results = period_metrics(benchmark, cash_returns, split)
    result = {
        "strategy": config["strategy"],
        "data_range": f"{path.index[0].date()} to {path.index[-1].date()}",
        **strategy_results,
        "benchmark": {"name": "Monthly equal weight", **benchmark_results},
    }
    print("PARAM_COMBOS_TRIED: 1")
    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, split- and distribution-adjusted total-return close")
    print(f"PERFORMANCE_RANGE: {result['data_range']} | excess return vs {cash_proxy}")
    for name in ("is", "oos", "full"):
        print(f"{name.upper()}: {json.dumps(strategy_results[name], allow_nan=False)}")
        print(f"BENCHMARK {name.upper()}: {json.dumps(benchmark_results[name], allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
