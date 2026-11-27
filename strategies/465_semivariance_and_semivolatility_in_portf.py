# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 465_semivariance_and_semivolatility_in_portf\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nrisk_free_ticker: BIL\ndata_start: '2009-01-01'\ndata_end: '2025-12-31'\nis_end: '2017-12-31'\nlookback_sessions: 504\nrebalance_sessions: 21\nexecution_lag_sessions: 1\ntransaction_cost_bps: 50.0\nslippage_bps: 2.0\n"
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
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252


def load_prices(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    key = json.dumps([sorted(tickers), str(start.date()), str(end.date())])
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    cache_path = cache_dir / f"adjusted_close_{digest}.csv"

    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as handle:
            fetched_at = datetime.fromisoformat(handle.readline().strip()).isoformat()
            close = pd.read_csv(handle, index_col="Date", parse_dates=["Date"])
    else:
        with TemporaryDirectory(dir=cache_dir, prefix="yfinance_") as session_dir:
            yf.set_tz_cache_location(session_dir)
            data = yf.download(
                tickers,
                start=start.date().isoformat(),
                end=(end.date() + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=True,
                ignore_tz=True,
                multi_level_index=True,
                progress=False,
                timeout=15,
            )
        close = data.get("Close", pd.DataFrame())
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if (
        not isinstance(close.index, pd.DatetimeIndex)
        or close.index.has_duplicates
        or not close.index.is_monotonic_increasing
    ):
        raise ValueError("Daily prices must have unique dates in ascending order.")
    close = close.reindex(columns=tickers).loc[start:end]
    values = close.to_numpy(dtype=float)
    if close.empty or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Adjusted closes are missing or invalid for the configured universe.")

    if not cache_path.exists():
        with NamedTemporaryFile(mode="w", dir=cache_dir, encoding="utf-8", delete=False) as handle:
            temp_path = Path(handle.name)
            try:
                handle.write(f"{fetched_at}\n")
                close.to_csv(handle, index_label="Date")
                handle.flush()
                temp_path.replace(cache_path)
            finally:
                temp_path.unlink(missing_ok=True)
    return close, fetched_at


def minimum_semivariance(window, equal_weights):
    # Express returns in percent to improve numerical conditioning.
    scenarios = 100.0 * window

    def objective(weights):
        downside = np.minimum(scenarios @ weights, 0.0)
        return downside @ downside / len(downside), 2 * (scenarios.T @ downside) / len(downside)

    solution = minimize(
        objective,
        equal_weights,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(equal_weights),
        constraints={
            "type": "eq",
            "fun": lambda weights: weights.sum() - 1.0,
            "jac": lambda weights: np.ones_like(weights),
        },
        options={"ftol": 1e-12, "maxiter": 200},
    )
    if not solution.success:
        return equal_weights, solution.message
    weights = np.clip(solution.x, 0.0, 1.0)
    return weights / weights.sum(), None


def simulate(returns, targets, trading_cost):
    weights = np.zeros(returns.shape[1])
    rows = []
    first_fill = min(targets)
    for position in range(first_fill, len(returns)):
        marked = weights * (1.0 + returns.iloc[position].to_numpy())
        gross = 1.0 + marked.sum() - weights.sum()
        weights = marked / gross
        capital_fraction, turnover, trades = 1.0, 0.0, 0

        if position in targets:
            target = targets[position]

            def fee_balance(fraction):
                return fraction + trading_cost * np.abs(fraction * target - weights).sum() - 1.0

            # Solve for the post-cost capital that can fund the target weights.
            capital_fraction = brentq(fee_balance, 0.0, 1.0, xtol=1e-14)
            traded = np.abs(capital_fraction * target - weights)
            turnover = traded.sum()
            trades = np.count_nonzero(traded > 1e-10)
            weights = target

        rows.append((gross * capital_fraction - 1.0, turnover, trades))

    return pd.DataFrame(rows, index=returns.index[first_fill:], columns=["return", "turnover", "trades"])


def metrics(history, cash_returns):
    daily = history["return"].to_numpy()
    excess = daily - cash_returns.loc[history.index].to_numpy()
    wealth = np.cumprod(np.r_[1.0, 1.0 + daily])
    vol = daily.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(wealth[-1] ** (SESSIONS_PER_YEAR / len(daily)) - 1.0),
        "ann_vol": float(vol * annualizer),
        "sharpe": float(daily.mean() / vol * annualizer) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * annualizer) if excess_vol > 0 else None,
        "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1.0).min()),
        "total_return": float(wealth[-1] - 1.0),
        "turnover": float(history["turnover"].sum() * SESSIONS_PER_YEAR / len(daily)),
        "n_trades": int(history["trades"].sum()),
    }


def summarize(history, cash_returns, is_end):
    return {
        "is": metrics(history.loc[history.index <= is_end], cash_returns),
        "oos": metrics(history.loc[history.index > is_end], cash_returns),
        "full": metrics(history, cash_returns),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    universe = config["universe"]
    cash_ticker = config["risk_free_ticker"]
    start, end, is_end = (pd.Timestamp(config[key]) for key in ("data_start", "data_end", "is_end"))
    lookback = config["lookback_sessions"]
    interval = config["rebalance_sessions"]
    lag = config["execution_lag_sessions"]
    cost, slippage = (float(config[key]) for key in ("transaction_cost_bps", "slippage_bps"))
    if not universe or len(set(universe)) != len(universe) or cash_ticker != "BIL" or cash_ticker in universe:
        raise ValueError("Configure unique portfolio tickers and use BIL solely for excess performance.")
    if not start < is_end < end or lookback < 2 or interval < 1 or lag < 1:
        raise ValueError("Invalid dates, lookback, rebalance interval, or execution lag.")
    if not (cost >= 0 and slippage >= 0 and cost + slippage < 10000):
        raise ValueError("Invalid transaction cost or slippage in basis points.")

    prices, fetched_at = load_prices([*universe, cash_ticker], start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return adjusted close")
    print("PARAM_COMBOS_TRIED: 1")

    returns = prices.pct_change(fill_method=None).iloc[1:]
    assets = returns[universe]
    first_fill = lookback - 1 + lag
    evaluation_dates = returns.index[first_fill:]
    if (evaluation_dates <= is_end).sum() < 2 or (evaluation_dates > is_end).sum() < 2:
        raise ValueError("The configured dates must leave two or more observations in each period.")

    equal_weights = np.full(len(universe), 1.0 / len(universe))
    targets = {}
    benchmark_targets = {}
    for fill in range(first_fill, len(returns), interval):
        signal = fill - lag
        window = assets.iloc[signal - lookback + 1 : signal + 1].to_numpy()
        targets[fill], solver_error = minimum_semivariance(window, equal_weights)
        if solver_error is not None:
            print(f"SOLVER_FALLBACK: {assets.index[signal].date()} {solver_error}")
        benchmark_targets[fill] = equal_weights

    rate = (cost + slippage) / 10000.0
    history = simulate(assets, targets, rate)
    benchmark_history = simulate(assets, benchmark_targets, rate)
    summary = summarize(history, returns[cash_ticker], is_end)
    benchmark = summarize(benchmark_history, returns[cash_ticker], is_end)
    for period in ("is", "oos", "full"):
        print(f"{period.upper()} MSV_0: {json.dumps(summary[period], allow_nan=False)}")
        print(f"{period.upper()} equal-weight: {json.dumps(benchmark[period], allow_nan=False)}")

    result = {
        "strategy": config["strategy"],
        "data_range": [str(prices.index[0].date()), str(prices.index[-1].date())],
        **summary,
        "benchmark": {"name": "Equal-weight sector ETFs", **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
