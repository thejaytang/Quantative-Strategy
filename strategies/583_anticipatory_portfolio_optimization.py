"""583_anticipatory_portfolio_optimization
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/583_anticipatory_portfolio_optimization.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 25a9179813de3addbbaee65a8c08579eb6e27d08 on 2026-11-27 19:08 Europe/Oslo, exit 0, output sha256 3c2e8f5415bebde7dde34f78199282b2f8732a7719e29f8ae69fce53874f680b
Data source: yfinance public daily Open and Close bars for nine U.S. sector ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY) and the Treasury-bill ETF BIL. BIL close-to-close returns are used only for ex-post excess Sharpe.
Data fetched: 2026-11-27T18:08:54+00:00
Sample period: Requested daily data from 2007-07-01 through 2025-12-31, with pre-2014 history used for warm-up. The configured backtest starts on 2014-01-01 and the logged evaluation sessions span 2014-01-02 through 2025-12-31.   data_range (from output): ['2014-01-02', '2025-12-31']
IS/OOS split: Chronological split. IS is 2014–2019 through 2019-12-31 inclusive. OOS consists of sessions after that cutoff through 2025-12-31. Sensitivity runs receive data through the IS cutoff. The default strategy runs continuously through the split, carries its holdings forward, and updates its rolling estimates with completed observations under the same parameter rules.
Costs and slippage: Every simulated purchase and sale incurs 2 bps in fees and 3 bps in modeled slippage. The combined 5 bps is charged to cash on the sum of absolute opening trade notionals, with the charge included in the self-financing sizing calculation. Slippage is implemented as a cash charge at the recorded opening price. The 20 bps impact coefficient and 80 bps turnover coefficient enter the optimizer only. One IS sensitivity run doubles the combined fee and slippage charge to 10 bps.
Signal lag: The signal uses the prior month’s last trading session close. With lag_sessions=1, it fills at the next trading session’s open, the first session of the new month. The configured lag counts trading sessions from the signal close and still fills at the selected session’s open.
Price adjustment: yfinance auto_adjust=True, total-return adjusted Open and Close
Survivorship bias: The universe is the fixed nine-fund sector ETF list in config. The README attributes the complete list to a 2013 prospectus that predates the 2014 backtest start. Its single-family coverage excludes a broader universe with delisted or later-launched funds, so market-wide survivorship bias is unassessed.
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.1443546639441553, ann_vol=0.12775305097413084, sharpe=1.119663741992008, sharpe_excess=1.0626669056565219, max_drawdown=-0.20255898833947472, total_return=1.2433606448649637, turnover=9.418649331611027, n_trades=499
    oos: cagr=0.10135117541371752, ann_vol=0.21353802212060685, sharpe=0.5591461038311495, sharpe_excess=0.43660799830796987, max_drawdown=-0.3806307480632002, total_return=0.7819250314332107, turnover=8.971427016993793, n_trades=494
    full: cagr=0.12266127833614116, ann_vol=0.1758984171844894, sharpe=0.7460416170768135, sharpe_excess=0.6510258737254768, max_drawdown=-0.3806307480632005, total_return=2.9975004876170144, turnover=9.195186359297315, n_trades=993
    benchmark: cagr=0.11896260587854424, ann_vol=0.16931920627784508, sharpe=0.7488770250021367, sharpe_excess=0.650180300707796, max_drawdown=-0.33147167560738156, total_return=2.842594915210742, turnover=1.2036449902010202, n_trades=839
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 583_anticipatory_portfolio_optimization\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nrisk_free: BIL\ndata_start: '2007-07-01'\nbacktest_start: '2014-01-01'\nis_end: '2019-12-31'\nbacktest_end: '2025-12-31'\nlag_sessions: 1\ncost_bps: 2.0\nslippage_bps: 3.0\nmodel:\n  lookback_months: 72\n  ar_lags: 2\n  ar_ridge: 0.5\n  mean_weight: 0.5\n  covariance_diagonal_weight: 0.25\n  horizon_months: 3\n  monthly_discount: 0.97\n  risk_aversion: 3.0\n  turnover_penalty_bps: 80.0\n  impact_bps: 20.0\n  impact_group_weight: 0.25\n  impact_groups:\n    - [XLB, XLE, XLI]\n    - [XLF, XLK, XLY]\n    - [XLP, XLU, XLV]\n  max_weight: 0.35\nrobustness:\n  horizons: [2, 4]\n  impact_bps: [10.0, 30.0]\n  friction_multiplier: 2.0\n"
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
"""Rolling ETF adaptation of Anticipatory Portfolio Optimization (2026).

Paper: https://arxiv.org/abs/2606.04258
Implementation references (design only):
- Cvxportfolio (GPL-3.0): https://github.com/cvxgrp/cvxportfolio
  https://www.cvxportfolio.com/en/stable/examples/etfs.html
  https://pypi.org/project/cvxportfolio/
- mpo-dev (Apache-2.0): https://github.com/akxen/mpo-dev
  https://github.com/akxen/mpo-dev/blob/master/project/api/optimisation/model.py
- QuantConnect ETF workflow: https://www.quantconnect.com/forum/discussion/3682
"""

import hashlib
import json
import signal
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, minimize
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
TRADE_THRESHOLD = 1e-8


def load_prices(symbols, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    key = hashlib.sha256(repr((symbols, start.date(), end.date())).encode()).hexdigest()[:16]
    path = cache_dir / f"yfinance_adjusted_open_close_{key}.csv"
    fields = pd.MultiIndex.from_product([["Open", "Close"], symbols])

    if path.is_file():
        with path.open() as stream:
            fetched_at = stream.readline().strip()
            prices = pd.read_csv(stream, header=[0, 1], index_col=0, parse_dates=True)
    else:
        with TemporaryDirectory(dir=cache_dir) as download_cache:
            yf.set_tz_cache_location(download_cache)
            prices = yf.download(
                symbols,
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d",
                auto_adjust=True,
                group_by="column",
                multi_level_index=True,
                ignore_tz=True,
                threads=False,
                progress=False,
                timeout=10,
            )
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if prices.empty or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("No daily yfinance bars returned for the configured dates")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Expected unique, chronological session dates")
    if not start <= prices.index[0] <= start + pd.Timedelta(days=7) or not (
        end - pd.Timedelta(days=7) <= prices.index[-1] <= end
    ):
        raise ValueError("Daily bars do not cover the requested start and end dates")
    prices = prices.reindex(columns=fields)
    invalid = ~(np.isfinite(prices) & (prices > 0))
    if invalid.any().any():
        row, column = np.argwhere(invalid.to_numpy())[0]
        raise ValueError(f"Missing or invalid {fields[column]} bar on {prices.index[row].date()}")

    if not path.is_file():
        with NamedTemporaryFile(mode="w", dir=cache_dir, delete=False) as stream:
            pending = Path(stream.name)
            try:
                stream.write(fetched_at + "\n")
                prices.to_csv(stream)
                stream.flush()
                pending.replace(path)
            finally:
                pending.unlink(missing_ok=True)
    return prices, fetched_at


def month_returns(closes, signal_date, lookback):
    available = closes.loc[:signal_date]
    month_closes = available.groupby(available.index.to_period("M")).last().tail(lookback + 1)
    expected = pd.period_range(end=signal_date.to_period("M"), periods=lookback + 1, freq="M")
    if not month_closes.index.equals(expected):
        raise ValueError(f"Need {lookback} consecutive monthly returns by {signal_date.date()}")
    return month_closes.pct_change().iloc[1:].to_numpy()


def forecasts_and_risk(history, model, restricted):
    horizon = model["horizon_months"]
    means = history.mean(axis=0)
    covariance = np.cov(history, rowvar=False)
    diagonal = np.diag(np.diag(covariance))
    shrink = model["covariance_diagonal_weight"]
    covariance = (1 - shrink) * covariance + shrink * diagonal
    forecasts = np.tile(model["mean_weight"] * means, (horizon, 1))
    if restricted:
        return forecasts, covariance

    lags = model["ar_lags"]
    for asset, (series, mean) in enumerate(zip(history.T, means)):
        scale = series.std(ddof=1)
        if scale <= np.finfo(float).eps:
            continue
        centered = (series - mean) / scale
        rows = np.lib.stride_tricks.sliding_window_view(centered, lags + 1)
        features, response = rows[:, :-1], rows[:, -1]
        regularizer = len(response) * model["ar_ridge"] * np.eye(lags)
        coefficients = np.linalg.solve(features.T @ features + regularizer, features.T @ response)
        state = centered[-lags:].copy()
        for step in range(horizon):
            prediction = state @ coefficients
            forecasts[step, asset] += prediction * scale
            state = np.append(state[1:], prediction)
    return forecasts, covariance


def impact_matrix(universe, model):
    groups = [symbol for group in model["impact_groups"] for symbol in group]
    if sorted(groups) != sorted(universe):
        raise ValueError("Impact groups must partition the ETF universe")
    membership = np.array([[symbol in group for group in model["impact_groups"]] for symbol in universe], float)
    group_weight = model["impact_group_weight"]
    return model["impact_bps"] / 10_000 * (
        (1 - group_weight) * np.eye(len(universe)) + group_weight * (membership @ membership.T)
    )


def plan_weights(history, current_weights, model, impact, restricted):
    forecasts, covariance = forecasts_and_risk(history, model, restricted)
    horizon, assets = forecasts.shape
    discount = model["monthly_discount"] ** np.arange(horizon)
    difference = np.kron(np.eye(horizon) - np.eye(horizon, k=-1), np.eye(assets))
    change_penalty = np.repeat(discount, assets) * model["turnover_penalty_bps"] / 10_000
    starting = np.zeros(horizon * assets)
    starting[:assets] = current_weights

    within_period = model["risk_aversion"] * covariance + (1 if restricted else 2) * impact
    precision = (
        np.kron(np.diag(discount), within_period) + difference.T @ (change_penalty[:, None] * difference)
    )
    score = (discount[:, None] * forecasts).ravel() + difference.T @ (change_penalty * starting)
    budget = np.kron(np.eye(horizon), np.ones((1, assets)))
    cap = model["max_weight"]
    result = minimize(
        lambda weights: 1000 * (0.5 * weights @ precision @ weights - score @ weights),
        np.full(horizon * assets, 1 / assets),
        jac=lambda weights: 1000 * (precision @ weights - score),
        method="SLSQP",
        bounds=[(0, cap)] * (horizon * assets),
        constraints={"type": "eq", "fun": lambda weights: budget @ weights - 1, "jac": lambda _: budget},
        options={"ftol": 1e-10, "maxiter": 200},
    )
    if not result.success or not np.allclose(budget @ result.x, 1, rtol=0, atol=1e-8):
        raise RuntimeError(f"Allocation did not converge: {result.message}")
    first = np.clip(result.x[:assets], 0, cap)
    return first / first.sum()


def rebalance_at_open(positions, cash, open_prices, target, friction):
    before = positions * open_prices
    total = before.sum() + cash
    relative_before = before / total

    def budget_gap(scale):
        return 1 - scale - friction * np.abs(scale * target - relative_before).sum()

    # Solve for the asset value left after both sides of trading have been charged.
    # A tiny downward buffer keeps the cash residue positive in floating-point arithmetic.
    scale = max(0.0, brentq(budget_gap, 0, 1, xtol=1e-14) - 1e-13)
    after = target * (scale * total)
    traded = np.abs(after - before)
    cash = total - after.sum() - friction * traded.sum()
    if cash < 0:
        raise RuntimeError("Open rebalance exceeded the cash budget")
    return after / open_prices, cash, traded.sum() / total, np.count_nonzero(traded > TRADE_THRESHOLD * total)


def simulate(opens, closes, start, universe, model, lag, friction, restricted=False):
    sessions = opens.index
    months = sessions.to_period("M")
    first_sessions = np.flatnonzero(months[1:] != months[:-1]) + 1
    schedule = {}
    for first in first_sessions:
        fill = first + lag - 1
        if fill < len(sessions) and months[fill] == months[first]:
            schedule[fill] = first - 1

    opens_array = opens.to_numpy()
    closes_array = closes.to_numpy()
    impact = impact_matrix(universe, model)
    positions = np.zeros(len(universe))
    cash = previous_value = 1.0
    daily_results = []
    start_row = sessions.searchsorted(start)
    for row in range(start_row, len(sessions)):
        turnover = trades = 0
        if row in schedule:
            signal_row = schedule[row]
            if row - signal_row != lag or signal_row >= row:
                raise RuntimeError("Signal and execution timing are inconsistent")
            signal_date = sessions[signal_row]
            history = month_returns(closes, signal_date, model["lookback_months"])
            signal_values = positions * closes_array[signal_row]
            current_weights = signal_values / (signal_values.sum() + cash)
            target = plan_weights(history, current_weights, model, impact, restricted)
            positions, cash, turnover, trades = rebalance_at_open(
                positions, cash, opens_array[row], target, friction
            )
        value = positions @ closes_array[row] + cash
        daily_results.append((value / previous_value - 1, turnover, trades))
        previous_value = value
    return pd.DataFrame(daily_results, index=sessions[start_row:], columns=["return", "turnover", "trades"])


def metrics(daily_results, risk_free):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise ValueError("Need at least two daily observations for period metrics")
    excess = returns - risk_free.loc[returns.index]
    values = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    scale = np.sqrt(SESSIONS_PER_YEAR)
    vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    return {
        "cagr": float(values[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(vol * scale),
        "sharpe": float(scale * returns.mean() / vol) if vol else 0.0,
        "sharpe_excess": float(scale * excess.mean() / excess_vol) if excess_vol else 0.0,
        "max_drawdown": float(np.min(values / np.maximum.accumulate(values) - 1)),
        "total_return": float(values[-1] - 1),
        "turnover": float(SESSIONS_PER_YEAR * daily_results["turnover"].mean()),
        "n_trades": int(daily_results["trades"].sum()),
    }


def main():
    with (ROOT / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    universe = config["universe"]
    risk_free_symbol = config["risk_free"]
    data_start, start, is_end, end = (
        pd.Timestamp(config[key]) for key in ("data_start", "backtest_start", "is_end", "backtest_end")
    )
    model = config["model"]
    lag = config["lag_sessions"]
    cost_bps, slippage_bps = config["cost_bps"], config["slippage_bps"]
    robustness = config["robustness"]
    multiplier = robustness["friction_multiplier"]
    friction = (cost_bps + slippage_bps) / 10_000
    if not data_start < start < is_end < end or not isinstance(lag, int) or not 1 <= lag <= 15:
        raise ValueError("Need chronological dates and a lag of 1 to 15 sessions")
    if min(cost_bps, slippage_bps) < 0 or multiplier <= 1 or not 0 <= friction * multiplier < 1:
        raise ValueError("Need nonnegative costs and stressed friction below 100%")
    if (
        len(set(universe)) != len(universe)
        or len(universe) < 2
        or risk_free_symbol in universe
        or not 1 / len(universe) <= model["max_weight"] <= 1
    ):
        raise ValueError("Need distinct ETF symbols, a separate risk-free series and a feasible weight cap")
    if not (
        1 <= model["ar_lags"] < model["lookback_months"]
        and model["ar_ridge"] > 0
        and 0 < model["covariance_diagonal_weight"] <= 1
        and 0 < model["monthly_discount"] <= 1
        and model["risk_aversion"] > 0
        and 0 <= model["impact_group_weight"] <= 1
        and model["turnover_penalty_bps"] >= 0
        and min(model["horizon_months"], *robustness["horizons"]) >= 1
        and min(model["impact_bps"], *robustness["impact_bps"]) >= 0
    ):
        raise ValueError("Forecast, risk or impact settings are outside supported ranges")

    prices, fetched_at = load_prices([*universe, risk_free_symbol], data_start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, total-return adjusted Open and Close")
    print(f"BENCHMARK: restricted constant-mean forecast with one unit of impact; excess returns vs {risk_free_symbol}")
    opens, closes = prices["Open"][universe], prices["Close"][universe]
    risk_free = prices["Close"][risk_free_symbol].pct_change()
    variants = [
        (f"horizon_months={horizon}", {**model, "horizon_months": horizon}, friction)
        for horizon in robustness["horizons"]
    ] + [
        (f"impact_bps={impact}", {**model, "impact_bps": impact}, friction)
        for impact in robustness["impact_bps"]
    ] + [(f"cost_and_slippage_x{multiplier:g}", model, friction * multiplier)]

    print("ROBUSTNESS (IS ONLY)")
    is_opens, is_closes = opens.loc[:is_end], closes.loc[:is_end]
    for name, variant, variant_friction in variants:
        result = simulate(is_opens, is_closes, start, universe, variant, lag, variant_friction)
        print(f"{name}: " + json.dumps(metrics(result, risk_free.loc[:is_end]), allow_nan=False))

    daily_results = simulate(opens, closes, start, universe, model, lag, friction)
    benchmark = simulate(opens, closes, start, universe, model, lag, friction, restricted=True)
    sessions = daily_results.index
    result = {
        "strategy": config["strategy"],
        "data_range": [sessions[0].date().isoformat(), sessions[-1].date().isoformat()],
    }
    for period, mask in (("is", sessions <= is_end), ("oos", sessions > is_end), ("full", slice(None))):
        result[period] = metrics(daily_results.loc[mask], risk_free)
        benchmark_period = metrics(benchmark.loc[mask], risk_free)
        print(f"{period.upper()} strategy: " + json.dumps(result[period], allow_nan=False))
        print(f"{period.upper()} benchmark: " + json.dumps(benchmark_period, allow_nan=False))
    result["benchmark"] = benchmark_period
    if result["is"]["n_trades"] == 0 or result["oos"]["n_trades"] == 0:
        raise RuntimeError("Need executed trades in both evaluation periods")
    print("LOOK_AHEAD_CHECK: prior month-end close; ETF prices only; aligned adjusted Open/Close; IS-only variants")
    print(f"PARAM_COMBOS_TRIED: {len(variants) + 2}")
    print("STATUS: 已回测 · 待审")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    # The monthly programs are small; BLAS thread startup can dominate their runtime.
    with threadpool_limits(limits=1, user_api="blas"):
        main()
