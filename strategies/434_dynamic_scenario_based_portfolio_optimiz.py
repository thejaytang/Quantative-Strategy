# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 434_dynamic_scenario_based_portfolio_optimiz\nuniverse:\n  [VT, VTI, ACWI, ACWX, SUSA, ROUS, IYG, BND, FPE, BYLD, NTSX, GAL, MDIV,\n   AOM, AOR, INKM, HNDL, ALTY, IYLD, CVY, PDBC, QAI, HDG, MNA, RAAX, RLY,\n   SRET, YYY, GAA, GMOM]\nrisk_free_ticker: BIL\ndata_start: '2018-12-19'\ndata_end: '2026-09-30'\nbacktest_start: '2022-12-21'\nis_end: '2024-12-31'\noos_start: '2025-01-01'\nlookback_sessions: 1008\nexecution_lag_sessions: 1\nsessions_per_year: 252\ntarget_turnover_cap: 0.00013227513227513228  # 1 / (30 × 252)\ncost_bps: 10.0\nslippage_bps: 5.0\n"
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
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize
from threadpoolctl import threadpool_limits
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
TRADE_TOLERANCE = 1e-8


def load_config():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key in ("data_start", "data_end", "backtest_start", "is_end", "oos_start"):
        config[key] = pd.Timestamp(config[key]).normalize()
    if not (config["data_start"] < config["backtest_start"] <= config["is_end"]
            < config["oos_start"] <= config["data_end"]):
        raise ValueError("Invalid data or IS/OOS dates")
    if len(set(config["universe"])) != len(config["universe"]) or len(config["universe"]) < 2:
        raise ValueError("The universe must contain at least two distinct ETFs")
    if config["risk_free_ticker"] != "BIL":
        raise ValueError("The risk-free proxy must be BIL for ETF total-return accounting")
    if config["risk_free_ticker"] in config["universe"]:
        raise ValueError("The risk-free proxy must be outside the strategy universe")
    for key, minimum in (("lookback_sessions", 2), ("execution_lag_sessions", 1),
                         ("sessions_per_year", 1)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer of at least {minimum}")
    if not 0 < config["target_turnover_cap"] <= 1:
        raise ValueError("target_turnover_cap must be in (0, 1]")
    rates = [config["cost_bps"], config["slippage_bps"]]
    if any(not np.isfinite(rate) or rate < 0 for rate in rates) or sum(rates) >= 10_000:
        raise ValueError("Invalid trading cost or slippage")
    return config


def check_prices(prices, symbols, start, end):
    prices = prices.reindex(columns=symbols).sort_index()
    prices = prices.loc[start:end]
    if len(prices) < 2 or prices.index.has_duplicates:
        raise ValueError("Price history is empty, too short, or has duplicate dates")
    invalid = (~np.isfinite(prices)) | (prices <= 0)
    if invalid.any().any():
        missing = invalid.sum().loc[lambda counts: counts > 0].to_dict()
        raise ValueError(f"Missing or invalid daily closing prices: {missing}")
    if prices.index[0] - start > pd.Timedelta(days=7) or end - prices.index[-1] > pd.Timedelta(days=7):
        raise ValueError("Price history does not cover the requested date boundaries")
    return prices


def load_prices(config):
    symbols = sorted([*config["universe"], config["risk_free_ticker"]])
    start, end = config["data_start"], config["data_end"]
    key = sha256(f"yf-adjusted-close-v1|{start.date()}|{end.date()}|{symbols}".encode()).hexdigest()[:16]
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"prices_{key}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            fetched_at = datetime.fromisoformat(handle.readline().strip()).isoformat()
            prices = pd.read_csv(handle, index_col="Date", parse_dates=["Date"])
        return check_prices(prices, symbols, start, end), fetched_at

    with TemporaryDirectory(dir=CACHE) as session_cache:
        yf.set_tz_cache_location(session_cache)
        raw = yf.download(
            symbols, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True, actions=False, progress=False, timeout=12, threads=6,
        )
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prices = check_prices(raw["Close"], symbols, start, end)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as handle:
        temp_path = Path(handle.name)
        try:
            handle.write(f"{fetched_at}\n")
            prices.to_csv(handle, index_label="Date")
            handle.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def minimum_variance(scenarios, previous, cap, constraints):
    covariance = np.cov(scenarios * 100.0, rowvar=False)
    previous_covariance = covariance @ previous
    n_assets = len(previous)

    # Split target changes into purchases and sales measured in units of the cap.
    def objective(changes):
        direction = changes[:n_assets] - changes[n_assets:]
        curvature = cap * (covariance @ direction)
        value = direction @ (2 * previous_covariance + curvature)
        gradient = 2 * (previous_covariance + curvature)
        return value, np.concatenate((gradient, -gradient))

    bounds = [(0, upper) for upper in np.minimum(1.0, np.r_[1 - previous, previous] / cap)]
    result = minimize(
        objective, np.zeros(2 * n_assets), jac=True, method="SLSQP",
        bounds=bounds, constraints=constraints, options={"ftol": 1e-12, "maxiter": 150},
    )
    if not result.success or not np.isfinite(result.x).all():
        return previous, False

    target = previous + cap * (result.x[:n_assets] - result.x[n_assets:])
    if abs(target.sum() - 1) > 1e-8 or target.min() < -1e-8 or target.max() > 1 + 1e-8:
        return previous, False
    target = np.clip(target, 0, 1)
    target /= target.sum()
    turnover = 0.5 * np.abs(target - previous).sum()
    if turnover > cap + 1e-8:
        return previous, False
    if turnover > cap:
        target = previous + (cap / turnover) * (target - previous)
    return target, True


def drift(weights, returns):
    positions = weights * (1 + returns)
    gross_growth = float(positions.sum())
    return positions / gross_growth, gross_growth


def simulate(prices, config):
    returns = prices.pct_change(fill_method=None).iloc[1:]
    evaluation_start = returns.index.searchsorted(config["backtest_start"])
    index = returns.index[evaluation_start:]
    window, lag = config["lookback_sessions"], config["execution_lag_sessions"]
    if evaluation_start - lag + 1 < window:
        raise ValueError("Insufficient price history for the configured rolling window and fill lag")
    if len(index[index <= config["is_end"]]) < 2 or len(index[index >= config["oos_start"]]) < 2:
        raise ValueError("IS and OOS both need at least two observations")

    asset_returns = returns[config["universe"]].to_numpy()
    n_assets = asset_returns.shape[1]
    equal = np.full(n_assets, 1 / n_assets)
    target = holdings = benchmark_holdings = equal
    cap = config["target_turnover_cap"]
    cost_rate = (config["cost_bps"] + config["slippage_bps"]) / 10_000
    constraints = [
        LinearConstraint(np.r_[np.ones(n_assets), -np.ones(n_assets)], 0, 0),
        LinearConstraint(np.ones(2 * n_assets), 0, 2),
    ]
    strategy, benchmark = [], []
    solver_fallbacks = 0
    for row in range(evaluation_start, len(returns)):
        decision = row - lag
        scenarios = asset_returns[decision - window + 1:decision + 1]
        target, success = minimum_variance(scenarios, target, cap, constraints)
        solver_fallbacks += int(not success)

        if row == evaluation_start:
            # Both portfolios fund their initial positions at this close from cash.
            pre_trade, gross_growth = np.zeros(n_assets), 1.0
            turnover = 1.0
            benchmark.append((-cost_rate, 1.0, n_assets))
        else:
            # Today's close-to-close move accrues to yesterday's holdings.
            pre_trade, gross_growth = drift(holdings, asset_returns[row])
            turnover = 0.5 * np.abs(target - pre_trade).sum()
            benchmark_holdings, benchmark_growth = drift(benchmark_holdings, asset_returns[row])
            benchmark.append((benchmark_growth - 1, 0.0, 0))
        n_trades = np.count_nonzero(np.abs(target - pre_trade) > TRADE_TOLERANCE)
        net_return = gross_growth * (1 - cost_rate * turnover) - 1
        strategy.append((net_return, turnover, n_trades))
        holdings = target

    columns = ["net_return", "turnover", "n_trades"]
    risk_free = returns.loc[index, config["risk_free_ticker"]].copy()
    risk_free.iloc[0] = 0.0  # The evaluation starts at the initial execution close.
    return (pd.DataFrame(strategy, index=index, columns=columns),
            pd.DataFrame(benchmark, index=index, columns=columns), risk_free, solver_fallbacks)


def metrics(period, risk_free, sessions_per_year):
    net = period["net_return"].to_numpy()
    excess = net - risk_free.loc[period.index].to_numpy()
    nav = np.r_[1.0, np.cumprod(1 + net)]
    years = len(net) / sessions_per_year
    vol = net.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    root_sessions = np.sqrt(sessions_per_year)
    return {
        "cagr": float(nav[-1] ** (1 / years) - 1),
        "ann_vol": float(vol * root_sessions),
        "sharpe": float(net.mean() * root_sessions / vol) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() * root_sessions / excess_vol) if excess_vol > 0 else None,
        "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1).min()),
        "total_return": float(nav[-1] - 1),
        "turnover": float(period["turnover"].sum() / years),
        "n_trades": int(period["n_trades"].sum()),
    }


def main():
    config = load_config()
    prices, fetched_at = load_prices(config)
    with threadpool_limits(limits=1, user_api="blas"):
        strategy, benchmark, risk_free, solver_fallbacks = simulate(prices, config)
    is_sample = strategy.loc[:config["is_end"]]
    oos_sample = strategy.loc[config["oos_start"]:]
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": str(strategy.index[0].date()), "end": str(strategy.index[-1].date())},
    }
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return adjusted close")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"SOLVER_FALLBACKS: {solver_fallbacks}")
    for key, period in (("is", is_sample), ("oos", oos_sample), ("full", strategy), ("benchmark", benchmark)):
        result[key] = metrics(period, risk_free, config["sessions_per_year"])
        label = "BENCHMARK (equal-weight buy and hold)" if key == "benchmark" else key.upper()
        print(f"{label} {period.index[0].date()} to {period.index[-1].date()}: {json.dumps(result[key])}")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
