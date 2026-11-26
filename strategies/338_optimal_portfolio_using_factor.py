# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 338_optimal_portfolio_using_factor\n\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nmarket_factor: SPY\ndownload_start: 2006-01-01\n\nin_sample:\n  start: 2008-01-01\n  end: 2018-12-31\nout_of_sample:\n  start: 2019-01-01\n  end: 2025-12-31\n\nlookback_sessions: 252\ngraphical_lasso_alpha: 0.08\nmax_weight: 0.25\nrebalance: monthly\ntransaction_cost_bps: 10  # Per dollar bought or sold, including initial purchases.\nsessions_per_year: 252\n'
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
"""Monthly sector allocation using a market factor and sparse residual precision."""

import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq, minimize
from sklearn.covariance import graphical_lasso
from sklearn.exceptions import ConvergenceWarning


ROOT = Path(__file__).resolve().parent
FACTOR_GMV = "Factor graphical-lasso GMV"
SAMPLE_GMV = "Sample covariance GMV"
EQUAL_WEIGHT = "Equal weight"


def download_prices(symbol, start, end):
    params = urlencode({
        "period1": int(start.tz_localize("America/New_York").timestamp()),
        "period2": int((end + pd.Timedelta(days=1)).tz_localize("America/New_York").timestamp()),
        "interval": "1d",
        "includeAdjustedClose": "true",
    })
    path = f"/v8/finance/chart/{quote(symbol, safe='')}?{params}"
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        try:
            request = Request(f"https://{host}{path}", headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=20) as response:
                result = json.load(response)["chart"]["result"][0]
            dates = pd.to_datetime(result["timestamp"], unit="s", utc=True)
            dates = dates.tz_convert("America/New_York").tz_localize(None).normalize()
            indicators = result["indicators"]
            raw = indicators["quote"][0]
            close = np.asarray(indicators["adjclose"][0]["adjclose"], dtype=float)
            adjusted_open = np.asarray(raw["open"], dtype=float) * close / np.asarray(raw["close"], dtype=float)
            prices = pd.DataFrame({"open": adjusted_open, "close": close}, index=dates)
            return prices.loc[start:end]
        except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
            last_error = error
    raise RuntimeError(f"Yahoo Finance download failed for {symbol}") from last_error


def load_prices(symbols, start, end):
    cache = ROOT / "data_cache" / f"yahoo_{start:%Y%m%d}_{end:%Y%m%d}_{'-'.join(sorted(symbols))}.csv"
    if cache.exists():
        prices = pd.read_csv(cache, index_col=0, header=[0, 1], parse_dates=True)
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            history = list(pool.map(lambda symbol: download_prices(symbol, start, end), symbols))
        prices = pd.concat(history, axis=1, keys=symbols).swaplevel(axis=1).sort_index()

    columns = pd.MultiIndex.from_product([("open", "close"), symbols])
    prices = prices.reindex(columns=columns)
    values = prices.to_numpy(dtype=float)
    if (prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing
            or prices.index[0] < start or prices.index[0] > start + pd.Timedelta(days=7)
            or prices.index[-1] > end or prices.index[-1] < end - pd.Timedelta(days=7)
            or not np.isfinite(values).all() or (values <= 0).any()):
        raise ValueError("Expected complete, aligned, positive prices across the requested dates.")

    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        with NamedTemporaryFile(mode="w", dir=cache.parent, delete=False) as temp:
            temp_path = Path(temp.name)
            try:
                prices.to_csv(temp, index_label="session")
                temp.close()
                temp_path.replace(cache)
            finally:
                temp_path.unlink(missing_ok=True)
    return prices


def factor_covariance(returns, market, alpha):
    assets = returns - returns.mean(axis=0)
    factor = market - market.mean()
    factor_ss = factor @ factor
    if factor_ss <= 0:
        raise ValueError("The market factor must have positive variance.")
    beta = (factor @ assets) / factor_ss
    residuals = assets - np.outer(factor, beta)
    scales = residuals.std(axis=0, ddof=1)
    if (scales <= 1e-12).any():
        raise ValueError("Each portfolio ETF needs nonzero residual variance.")
    standardized = residuals / scales
    correlation = standardized.T @ standardized / (len(market) - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        residual_cov, _ = graphical_lasso(correlation, alpha=alpha, max_iter=500, tol=1e-7, enet_tol=1e-8)
    return residual_cov * np.outer(scales, scales) + np.outer(beta, beta) * factor.var(ddof=1)


def minimum_variance(covariance, max_weight):
    covariance = covariance / np.mean(np.diag(covariance))
    n_assets = len(covariance)
    equal = np.full(n_assets, 1 / n_assets)
    result = minimize(
        fun=lambda w: 0.5 * w @ covariance @ w,
        x0=equal,
        jac=lambda w: covariance @ w,
        bounds=[(0, max_weight)] * n_assets,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 300},
    )
    weights = result.x
    if (not result.success or not np.isfinite(weights).all() or abs(weights.sum() - 1) > 1e-9
            or (weights < -1e-9).any() or (weights > max_weight + 1e-9).any()):
        raise RuntimeError(f"Minimum-variance optimization failed: {result.message}")
    return weights


def monthly_targets(closes, universe, market_factor, simulation_start, lookback, alpha, max_weight):
    daily_returns = closes.pct_change(fill_method=None)
    sessions = daily_returns.index
    months = sessions.to_period("M")
    targets = {name: {} for name in (FACTOR_GMV, SAMPLE_GMV, EQUAL_WEIGHT)}
    for position in np.flatnonzero(months[:-1] != months[1:]):
        execution = sessions[position + 1]
        if execution < simulation_start:
            continue
        if position < lookback:
            raise ValueError(f"Need {lookback} trailing daily returns by {sessions[position].date()}.")
        window = daily_returns.iloc[position - lookback + 1:position + 1]
        assets = window[universe].to_numpy()
        market = window[market_factor].to_numpy()
        factor_cov = factor_covariance(assets, market, alpha)
        sample_cov = np.cov(assets, rowvar=False, ddof=1)
        targets[FACTOR_GMV][execution] = minimum_variance(factor_cov, max_weight)
        targets[SAMPLE_GMV][execution] = minimum_variance(sample_cov, max_weight)
        targets[EQUAL_WEIGHT][execution] = np.full(len(universe), 1 / len(universe))
    return targets


def simulate(opens, closes, targets, simulation_start, cost_rate):
    # Old holdings earn the overnight return. New holdings earn the open-to-close return.
    overnight = (opens / closes.shift()).loc[simulation_start:]
    intraday = (closes / opens).loc[simulation_start:]
    if overnight.empty or overnight.index[0] not in targets:
        raise ValueError("Need a prior month-end signal for the first simulated session.")
    weights = np.zeros(len(opens.columns))
    records = []
    for session, night, day in zip(overnight.index, overnight.to_numpy(), intraday.to_numpy(), strict=True):
        open_positions = weights * night
        open_growth = 1 + weights @ (night - 1)
        weights = open_positions / open_growth
        nav_after_fees = 1.0
        turnover = 0.0
        if session in targets:
            target = targets[session]

            # Target weights are fractions of NAV after fees. Solve for that NAV fraction.
            def traded(fraction):
                return np.abs(fraction * target - weights).sum()

            if cost_rate:
                nav_after_fees = brentq(lambda fraction: fraction + cost_rate * traded(fraction) - 1, 0, 1)
            turnover = traded(nav_after_fees)
            weights = target

        close_growth = 1 + weights @ (day - 1)
        weights = weights * day / close_growth
        records.append((open_growth * nav_after_fees * close_growth - 1, turnover))
    return pd.DataFrame(records, index=overnight.index, columns=["return", "turnover"])


def metrics(results, sessions_per_year):
    returns = results["return"]
    growth = (1 + returns).cumprod()
    monthly = (1 + returns).groupby(returns.index.to_period("M")).prod() - 1
    daily_vol = returns.std(ddof=1)
    years = len(returns) / sessions_per_year
    values = {
        "cagr": growth.iloc[-1] ** (1 / years) - 1,
        "ann_vol": daily_vol * np.sqrt(sessions_per_year),
        "sharpe": returns.mean() / daily_vol * np.sqrt(sessions_per_year) if daily_vol > 0 else np.nan,
        "max_drawdown": (growth / growth.clip(lower=1).cummax() - 1).min(),
        "total_return": growth.iloc[-1] - 1,
        "turnover": results["turnover"].sum() / years,
        "mean_monthly_return": monthly.mean(),
        "monthly_variance": monthly.var(ddof=1),
    }
    return {key: float(value) if np.isfinite(value) else None for key, value in values.items()}


def print_table(summary):
    columns = {
        "cagr": "CAGR",
        "ann_vol": "Ann. vol",
        "sharpe": "Sharpe",
        "max_drawdown": "Max DD",
        "total_return": "Total return",
        "turnover": "Ann. turnover",
        "mean_monthly_return": "Mean mo. ret",
        "monthly_variance": "Mo. variance",
    }
    formats = {"sharpe": ".3f", "monthly_variance": ".3e"}
    print(f"{'Portfolio':<28} {'Period':<6}" + "".join(f" {label:>14}" for label in columns.values()))
    for portfolio, periods in summary.items():
        for period, stats in periods.items():
            row = []
            for key in columns:
                form = formats.get(key, ".2%")
                row.append(format(stats[key], form) if stats[key] is not None else "n/a")
            print(f"{portfolio:<28} {period.upper():<6}" + "".join(f" {value:>14}" for value in row))


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    universe, market_factor = config["universe"], config["market_factor"]
    symbols = [*universe, market_factor]
    download_start = pd.Timestamp(config["download_start"])
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    lookback = int(config["lookback_sessions"])
    alpha = float(config["graphical_lasso_alpha"])
    max_weight = float(config["max_weight"])
    cost_rate = float(config["transaction_cost_bps"]) / 10_000
    sessions_per_year = int(config["sessions_per_year"])
    if not download_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Expected chronological price history, IS and OOS dates.")
    if len(universe) < 2 or len(set(symbols)) != len(symbols):
        raise ValueError("Use distinct sector ETFs and a separate market factor.")
    if not (lookback >= 3 and 0 < alpha and 1 / len(universe) <= max_weight <= 1
            and 0 <= cost_rate < 1 and sessions_per_year > 0 and config["rebalance"] == "monthly"):
        raise ValueError("Invalid model, allocation, execution or annualization settings.")

    prices = load_prices(symbols, download_start, oos_end)
    simulation_start = is_start.replace(day=1)
    targets = monthly_targets(prices["close"], universe, market_factor, simulation_start, lookback, alpha, max_weight)
    ranges = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    summary = {}
    for portfolio, schedule in targets.items():
        results = simulate(prices["open"][universe], prices["close"][universe], schedule, simulation_start, cost_rate)
        summary[portfolio] = {}
        for period, (start, end) in ranges.items():
            segment = results.loc[start:end]
            if segment.empty:
                raise ValueError(f"No trading sessions in the {period} date range.")
            summary[portfolio][period] = metrics(segment, sessions_per_year)

    scored = results.loc[is_start:oos_end].index
    benchmark = summary[EQUAL_WEIGHT]
    report = {
        "strategy": config["strategy"],
        "data_range": f"{scored[0].date()} to {scored[-1].date()}",
        **summary[FACTOR_GMV],
        "benchmark": {"name": EQUAL_WEIGHT, **benchmark["full"], **benchmark},
        "sample_covariance": summary[SAMPLE_GMV],
    }
    print_table(summary)
    print("RESULTS_JSON: " + json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
