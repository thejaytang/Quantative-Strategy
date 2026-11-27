"""580_anticipatory_portfolio_optimization
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/580_anticipatory_portfolio_optimization.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 25a9179813de3addbbaee65a8c08579eb6e27d08 on 2026-11-27 19:08 Europe/Oslo, exit 0, output sha256 818e3e39dacc653909b6f0a71e07a42449b356656930a9bfffc100340461b9ec
Data source: Yahoo Finance via yfinance, daily Open and Close plus dividends and capital-gain distributions when supplied. Sector ETFs: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY. SPY is the benchmark and BIL supplies the reference returns for ex-post excess Sharpe..
Data fetched: 2026-11-27T18:08:42+00:00
Sample period: Data requested from 2004-01-01 through 2025-12-31 for warm-up and evaluation. BIL may start later and must cover the session preceding evaluation. The logged evaluation spans 2010-01-04 through 2025-12-31.   data_range (from output): {'start': '2010-01-04', 'end': '2025-12-31'}
IS/OOS split: IS consists of evaluation sessions before 2018-01-01 (2010–2017). OOS consists of sessions on or after 2018-01-01 (2018–2025). The default simulation carries its portfolio across the cutoff, and rolling estimation can use OOS observations through each signal close. The six robustness cases run on IS only..
Costs and slippage: Each buy or sell incurs a 2 bp fee and 3 bp of modeled slippage on absolute traded notional at the open. The combined 5 bp is reserved when sizing and deducted from cash. Slippage is a fixed cash charge and the simulated fill price is the adjusted open. The IS higher-friction case charges 4 bp in fees and 6 bp in slippage using the same mechanism. The SPY benchmark entry uses the default rate..
Signal lag: The strategy computes target weights after the last trading session close of a month using returns through that close and queues them for the open one trading session later under the default configuration. The first logged signal close was 2009-12-31 and the first fill open was 2010-01-04..
Price adjustment: yfinance split-normalized OHLC with forward ex-date total-return adjustment
Survivorship bias: The same nine sector ETFs are selected for the whole historical period. Selecting surviving funds can introduce fund-selection and survivorship bias. Historical security exposure comes from the funds' own price histories..
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.15500637682581808, ann_vol=0.1368103846812054, sharpe=1.122119509504016, sharpe_excess=1.117508378056549, max_drawdown=-0.14495003139461748, total_return=2.1617680295151738, turnover=9.999302822590948, n_trades=670
    oos: cagr=0.061878059151794895, ann_vol=0.2020746294412541, sharpe=0.3991070242924952, sharpe_excess=0.27882668149591333, max_drawdown=-0.4503706451614168, total_return=0.6146553088110231, turnover=11.76804450343826, n_trades=656
    full: cagr=0.10748687408246083, ann_vol=0.17253402142030916, sharpe=0.6787160603599569, sharpe_excess=0.6066147153459392, max_drawdown=-0.45037064516141634, total_return=4.105165534085686, turnover=10.883234114883182, n_trades=1326
    benchmark SPY full: cagr=0.13983198877820424, ann_vol=0.17208618360719966, sharpe=0.8470061253062938, sharpe_excess=0.7747026624360458, max_drawdown=-0.3371726392349519, total_return=7.0844221945885355, turnover=0.06259295799410133, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 580_anticipatory_portfolio_optimization\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nbenchmark: SPY\nrisk_free: BIL\ndata_start: '2004-01-01'\nbacktest_start: '2010-01-01'\noos_start: '2018-01-01'\nbacktest_end: '2025-12-31'\nexecution_lag_sessions: 1\ncost_bps: 2.0\nslippage_bps: 3.0\n\nmodel:\n  lookback_months: 60\n  ar_lags: 2\n  ridge: 0.50\n  mean_shrinkage: 0.50\n  covariance_shrinkage: 0.20\n  horizon_months: 3\n  discount: 0.97\n  risk_aversion: 3.0\n  impact_scale: 0.50\n  turnover_penalty: 0.012\n  max_weight: 0.35\n\nrobustness:\n  horizon_months: [2, 4]\n  impact_scale: [0.25, 0.75]\n  high_cost_bps: 4.0\n  high_slippage_bps: 6.0\n"
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
"""Finite-horizon sector ETF allocation with impact curvature and trading friction.

Paper: Miquel Noguer i Alonso (2026), https://arxiv.org/abs/2606.04258
Implementation references (ideas only, no source code copied):
https://github.com/cvxgrp/cvxportfolio (GPL-3.0-or-later)
https://raw.githubusercontent.com/cvxgrp/cvxportfolio/master/cvxportfolio/policies.py
https://www.cvxportfolio.com/en/stable/examples/case_shiller.html
https://pypi.org/project/cvxportfolio/
https://github.com/dynamic-trading-RL/dynamic-trading (license unclear)
https://raw.githubusercontent.com/dynamic-trading-RL/dynamic-trading/main/dynamic_trading/benchmark_agents/agents.py
https://github.com/QuantConnect/Research#analysis-examples (Apache-2.0)
https://github.com/QuantConnect/Research/blob/master/Analysis/03%20Mean-Variance%20Portfolio%20Optimization%20.ipynb
"""

import hashlib
import json
import math
import signal
from dataclasses import dataclass, replace
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
CACHE = ROOT / "data_cache"
PRICE_ADJUSTMENT = "yfinance split-normalized OHLC with forward ex-date total-return adjustment"
SESSIONS_PER_YEAR = 252
MAX_RUN_SECONDS = 850
BAR_FIELDS = ("Open", "Close", "Distribution")


@dataclass(frozen=True)
class Model:
    lookback_months: int
    ar_lags: int
    ridge: float
    mean_shrinkage: float
    covariance_shrinkage: float
    horizon_months: int
    discount: float
    risk_aversion: float
    impact_scale: float
    turnover_penalty: float
    max_weight: float

    def validate(self, n_assets):
        counts = (self.lookback_months, self.ar_lags, self.horizon_months)
        if any(type(value) is not int or value < 1 for value in counts):
            raise ValueError("Window, autoregressive lags and planning horizon must be positive integers.")
        if self.lookback_months <= self.ar_lags + 2:
            raise ValueError("The monthly estimation window is too short for the autoregression.")
        if not (0 <= self.mean_shrinkage <= 1 and 0 < self.covariance_shrinkage <= 1
                and 0 < self.discount <= 1 and 1 / n_assets <= self.max_weight <= 1):
            raise ValueError("Invalid shrinkage, discount or allocation constraint.")
        positive = (self.ridge, self.risk_aversion, self.turnover_penalty)
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Ridge, risk aversion and turnover penalty must be positive and finite.")
        if not math.isfinite(self.impact_scale) or self.impact_scale < 0:
            raise ValueError("Impact scale must be nonnegative and finite.")


def friction_rate(cost_bps, slippage_bps):
    if not (0 <= cost_bps and 0 <= slippage_bps and cost_bps + slippage_bps < 10_000):
        raise ValueError("Cost and slippage must be nonnegative and sum to less than 10,000 bps.")
    return (cost_bps + slippage_bps) / 10_000


def read_config():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    date_fields = ("data_start", "backtest_start", "oos_start", "backtest_end")
    for key in date_fields:
        config[key] = pd.Timestamp(config[key])
    if not (config["data_start"] < config["backtest_start"]
            < config["oos_start"] < config["backtest_end"]):
        raise ValueError("Expected data start < backtest start < OOS start < backtest end.")
    universe = config["universe"]
    tickers = [*universe, config["benchmark"], config["risk_free"]]
    if len(universe) < 2 or len(tickers) != len(set(tickers)):
        raise ValueError("Use at least two unique ETFs and separate benchmark and reference ETFs.")
    lag = config["execution_lag_sessions"]
    if type(lag) is not int or lag < 1:
        raise ValueError("The fill must lag the signal by at least one trading session.")

    model = Model(**config["model"])
    rate = friction_rate(config["cost_bps"], config["slippage_bps"])
    robustness = config["robustness"]
    cases = []
    for name in ("horizon_months", "impact_scale"):
        for value in robustness[name]:
            cases.append((f"{name}={value}", replace(model, **{name: value}), rate, False))
    cases.append(("flat_forecast_path", model, rate, True))
    high_cost = robustness["high_cost_bps"]
    high_slippage = robustness["high_slippage_bps"]
    high_rate = friction_rate(high_cost, high_slippage)
    if high_rate <= rate:
        raise ValueError("The higher-friction case must increase combined costs.")
    cases.append((f"cost_bps={high_cost}, slippage_bps={high_slippage}", model, high_rate, False))
    combinations = [(model, rate, False)] + [case[1:] for case in cases]
    if len(set(combinations)) != len(combinations):
        raise ValueError("Expected distinct robustness cases.")
    for specification, _, _ in combinations:
        specification.validate(len(universe))
    return config, model, rate, cases


def check_bars(bars, tickers, config):
    start, end = config["data_start"], config["backtest_end"]
    if not isinstance(bars.index, pd.DatetimeIndex) or not bars.index.is_unique:
        raise ValueError("Expected unique daily session dates.")
    columns = pd.MultiIndex.from_product([BAR_FIELDS, tickers])
    bars = bars.sort_index().loc[start:end].reindex(columns=columns)
    week = pd.Timedelta(days=7)
    if (bars.empty or bars.index[0] - start > week or end - bars.index[-1] > week
            or (bars.index.to_series().diff() > week).any()):
        raise ValueError("Daily bars do not cover the configured dates.")
    first_sample = bars.index.searchsorted(config["backtest_start"])
    for ticker in tickers:
        frame = bars.xs(ticker, axis=1, level=1)
        if ticker == config["risk_free"]:
            # BIL may start after the sector model's warm-up begins.
            frame = frame.loc[frame["Close"].first_valid_index():]
            if frame.empty or first_sample < 1 or frame.index[0] > bars.index[first_sample - 1]:
                raise ValueError(f"{ticker} needs coverage from before the first evaluation session.")
        if (not np.isfinite(frame.to_numpy()).all()
                or (frame[["Open", "Close"]] <= 0).any().any()
                or (frame["Distribution"] < 0).any()):
            raise ValueError(f"Missing or invalid daily price or distribution data for {ticker}.")
    return bars


def download_bars(tickers, config):
    start = config["data_start"].date().isoformat()
    end = (config["backtest_end"] + pd.Timedelta(days=1)).date().isoformat()
    with TemporaryDirectory(dir=CACHE, prefix="yfinance-") as client_dir:
        yf.set_tz_cache_location(client_dir)
        for threaded in (True, False):
            raw = yf.download(tickers, start=start, end=end, interval="1d", auto_adjust=False,
                              actions=True, repair=False, ignore_tz=True, keepna=True,
                              progress=False, timeout=12, threads=threaded)
            try:
                distributions = raw["Dividends"].copy()
                if "Capital Gains" in raw:
                    gains = raw["Capital Gains"].reindex(columns=distributions.columns).fillna(0)
                    distributions = distributions + gains
                bars = pd.concat({"Open": raw["Open"], "Close": raw["Close"],
                                  "Distribution": distributions}, axis=1)
                return check_bars(bars, tickers, config)
            except (KeyError, ValueError) as exc:
                if not threaded:
                    raise RuntimeError(f"Unable to load public Yahoo Finance prices: {exc}") from exc
                print(f"Retrying Yahoo Finance download after: {exc}", flush=True)


def load_bars(config):
    tickers = [*config["universe"], config["benchmark"], config["risk_free"]]
    key = json.dumps(["yfinance-bars-v1", tickers, str(config["data_start"]), str(config["backtest_end"])])
    path = CACHE / f"bars-{hashlib.sha256(key.encode()).hexdigest()[:16]}.csv"
    CACHE.mkdir(exist_ok=True)
    if path.is_file():
        with path.open(encoding="utf-8") as source:
            fetched_at = source.readline().strip()
            datetime.fromisoformat(fetched_at)
            bars = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True)
        return check_bars(bars, tickers, config), fetched_at

    bars = download_bars(tickers, config)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(fetched_at + "\n")
            bars.to_csv(stream)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return bars, fetched_at


def total_return_prices(bars):
    raw_close = bars["Close"]
    # Only the prior close and the distribution effective at today's open enter today's factor.
    prior_close = raw_close.shift(1)
    adjustment_step = 1 - bars["Distribution"] / prior_close
    if adjustment_step.le(0).any().any():
        raise ValueError("A distribution meets or exceeds the prior split-normalized close.")
    factor = (1 / adjustment_step.fillna(1)).cumprod()
    return bars["Open"] * factor, raw_close * factor


def month_schedule(closes, first_sample, lag):
    months = closes.index.to_period("M")
    month_starts = np.flatnonzero(months[1:] != months[:-1]) + 1
    month_ends = month_starts - 1
    monthly_returns = closes.iloc[month_ends].pct_change(fill_method=None).iloc[1:]
    schedule = {}
    for signal_row in month_ends:
        fill_row = int(signal_row + lag)
        if first_sample <= fill_row < len(closes):
            schedule[int(signal_row)] = fill_row
    return monthly_returns, schedule


def forecast_and_risk(history, model, flat_path):
    sample_mean = history.mean(axis=0)
    sample_cov = np.cov(history, rowvar=False, ddof=1)
    variance = sample_cov.diagonal()
    if np.any(variance <= 0):
        raise ValueError("The forecast window contains a constant monthly return series.")
    stdev = np.sqrt(variance)
    standardized = (history - sample_mean) / stdev
    lags = model.ar_lags
    rows = len(standardized)
    features = np.hstack([standardized[lags - lag:rows - lag] for lag in range(1, lags + 1)])
    penalty = len(features) * model.ridge * np.eye(features.shape[1])
    coefficients = np.linalg.solve(features.T @ features + penalty,
                                   features.T @ standardized[lags:])

    long_run = sample_mean + model.mean_shrinkage * (sample_mean.mean() - sample_mean)
    state = standardized[-lags:][::-1].ravel()
    forecasts = []
    for _ in range(model.horizon_months):
        predicted = state @ coefficients
        forecasts.append(long_run + stdev * predicted)
        state = np.concatenate((predicted, state[:-len(sample_mean)]))
    if flat_path:
        forecasts = [forecasts[0]] * model.horizon_months

    diagonal_cov = np.diag(variance)
    covariance = (1 - model.covariance_shrinkage) * sample_cov + model.covariance_shrinkage * diagonal_cov
    risk_and_impact = model.risk_aversion * covariance + 2 * model.impact_scale * diagonal_cov
    return np.asarray(forecasts), risk_and_impact


def first_target(history, prior_weights, model, flat_path):
    forecasts, risk_and_impact = forecast_and_risk(history, model, flat_path)
    horizon, n_assets = forecasts.shape
    discounts = model.discount ** np.arange(horizon)
    discount_matrix = np.diag(discounts)
    differences = np.eye(horizon) - np.eye(horizon, k=-1)
    precision = (np.kron(discount_matrix, risk_and_impact)
                 + model.turnover_penalty * np.kron(differences.T @ discount_matrix @ differences,
                                                   np.eye(n_assets)))
    score = (discounts[:, None] * forecasts).ravel()
    score[:n_assets] += model.turnover_penalty * prior_weights

    # Rescale to keep the solver tolerances independent of the return units.
    scale = max(np.abs(score).max(), precision.diagonal().max())
    precision = precision / scale
    score = score / scale
    budgets = np.kron(np.eye(horizon), np.ones((1, n_assets)))
    solution = minimize(
        lambda weights: 0.5 * weights @ precision @ weights - score @ weights,
        np.full(horizon * n_assets, 1 / n_assets),
        jac=lambda weights: precision @ weights - score,
        method="SLSQP",
        bounds=[(0, model.max_weight)] * (horizon * n_assets),
        constraints=[{"type": "eq", "fun": lambda weights: budgets @ weights - 1,
                      "jac": lambda _weights: budgets}],
        options={"ftol": 1e-11, "maxiter": 150},
    )
    if (not solution.success or not np.isfinite(solution.x).all()
            or not np.allclose(budgets @ solution.x, 1, rtol=0, atol=1e-9)):
        raise RuntimeError(f"Constrained allocation failed: {solution.message}")
    target = solution.x[:n_assets].copy()
    target[target < 1e-12] = 0
    target /= target.sum()
    if target.max() > model.max_weight + 1e-9:
        raise RuntimeError("The allocation exceeded the configured weight cap.")
    return target


def rebalance_at_open(positions, cash, open_prices, target, rate):
    open_values = positions * open_prices
    open_total = cash + open_values.sum()
    open_weights = open_values / open_total

    def budget(fraction):
        return fraction + rate * np.abs(fraction * target - open_weights).sum() - 1

    # Solve for the capital left after charging both sides, then size at the open.
    fraction = max(0.0, brentq(budget, 0.0, 1.0, xtol=1e-14) - 1e-12)
    target_values = open_total * fraction * target
    next_positions = target_values / open_prices
    trade_values = np.abs((next_positions - positions) * open_prices)
    remaining_cash = open_total - target_values.sum() - rate * trade_values.sum()
    if remaining_cash < 0 or (next_positions < 0).any():
        raise RuntimeError("The rebalance exceeded available cash or created a short position.")
    turnover = trade_values.sum() / open_total
    trades = int(np.count_nonzero(trade_values > open_total * 1e-10))
    return next_positions, remaining_cash, turnover, trades


def simulate(opens, closes, first_sample, schedule, allocate, rate):
    open_prices = opens.to_numpy()
    close_prices = closes.to_numpy()
    positions = np.zeros(closes.shape[1])
    cash = previous_total = 1.0
    pending = {}
    rows = []
    first_row = min(first_sample, min(schedule))
    for row in range(first_row, len(closes)):
        turnover = trades = 0
        if row in pending:
            positions, cash, turnover, trades = rebalance_at_open(
                positions, cash, open_prices[row], pending.pop(row), rate)
        close_values = positions * close_prices[row]
        total = cash + close_values.sum()
        if row >= first_sample:
            rows.append((total / previous_total - 1, turnover, trades))
        previous_total = total
        if row in schedule:
            pending[schedule[row]] = allocate(closes.index[row], close_values / total)
    return pd.DataFrame(rows, index=closes.index[first_sample:], columns=["return", "turnover", "trades"])


def strategy_run(opens, closes, start, lag, model, rate, flat_path=False):
    first_sample = closes.index.searchsorted(start)
    monthly_returns, schedule = month_schedule(closes, first_sample, lag)
    if not schedule:
        raise ValueError("No eligible monthly rebalance in this evaluation period.")

    def allocate(day, prior_weights):
        history = monthly_returns.loc[:day].tail(model.lookback_months)
        if len(history) != model.lookback_months or history.index[-1] != day:
            raise ValueError(f"Insufficient causal monthly history at {day.date()}.")
        return first_target(history.to_numpy(), prior_weights, model, flat_path)

    daily = simulate(opens, closes, first_sample, schedule, allocate, rate)
    return daily, min(schedule), schedule[min(schedule)]


def metrics(daily, risk_free):
    returns = daily["return"]
    if len(returns) < 2:
        raise ValueError("A reporting period needs at least two daily observations.")
    excess = returns - risk_free.loc[returns.index]
    if not np.isfinite(excess.to_numpy()).all():
        raise ValueError("Incomplete BIL reference returns.")
    growth = (1 + returns).cumprod()
    peak = growth.cummax().clip(lower=1)
    vol = returns.std()
    excess_vol = excess.std()
    annualizer = math.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(growth.iloc[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(vol * annualizer),
        "sharpe": float(returns.mean() / vol * annualizer) if vol else None,
        "sharpe_excess": float(excess.mean() / excess_vol * annualizer) if excess_vol else None,
        "max_drawdown": float((growth / peak - 1).min()),
        "total_return": float(growth.iloc[-1] - 1),
        "turnover": float(daily["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily["trades"].sum()),
    }


def period_metrics(daily, risk_free, oos_start):
    is_mask = daily.index < oos_start
    return {
        "is": metrics(daily.loc[is_mask], risk_free),
        "oos": metrics(daily.loc[~is_mask], risk_free),
        "full": metrics(daily, risk_free),
    }


def main():
    config, model, rate, cases = read_config()
    bars, fetched_at = load_bars(config)
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}", flush=True)
    opens, closes = total_return_prices(bars)
    universe, benchmark_ticker = config["universe"], config["benchmark"]
    asset_opens, asset_closes = opens[universe], closes[universe]
    risk_free = closes[config["risk_free"]].pct_change(fill_method=None)
    start, oos_start = config["backtest_start"], config["oos_start"]
    lag = config["execution_lag_sessions"]

    print("ROBUSTNESS (IS ONLY)", flush=True)
    is_mask = asset_closes.index < oos_start
    for label, case_model, case_rate, flat_path in cases:
        daily, _, _ = strategy_run(asset_opens.loc[is_mask], asset_closes.loc[is_mask],
                                   start, lag, case_model, case_rate, flat_path)
        print(f"IS {label}: {json.dumps(metrics(daily, risk_free), allow_nan=False)}", flush=True)

    daily, first_signal, first_fill = strategy_run(asset_opens, asset_closes, start, lag, model, rate)
    benchmark = simulate(opens[[benchmark_ticker]], closes[[benchmark_ticker]],
                         closes.index.searchsorted(start), {first_signal: first_fill},
                         lambda _day, _weights: np.ones(1), rate)
    official = period_metrics(daily, risk_free, oos_start)
    benchmark_periods = period_metrics(benchmark, risk_free, oos_start)
    if any(official[period]["n_trades"] <= 0 for period in ("is", "oos")):
        raise RuntimeError("The official strategy must record trades in both sample periods.")

    print(f"PARAM_COMBOS_TRIED: {1 + len(cases)}")
    print(f"FIRST_SIGNAL_CLOSE: {closes.index[first_signal].date()}")
    print(f"FIRST_FILL_OPEN: {closes.index[first_fill].date()}")
    print("BACKTEST_STATUS: 已回测 | pending review")
    for period in official:
        print(f"STRATEGY {period.upper()}: {json.dumps(official[period], allow_nan=False)}")
        print(f"BENCHMARK {benchmark_ticker} {period.upper()}: "
              f"{json.dumps(benchmark_periods[period], allow_nan=False)}")
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": daily.index[0].date().isoformat(), "end": daily.index[-1].date().isoformat()},
        **official,
        "benchmark": {"ticker": benchmark_ticker, **benchmark_periods},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False), flush=True)


def stop_on_timeout(_signal, _frame):
    raise SystemExit(f"Run exceeded {MAX_RUN_SECONDS} seconds.")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_on_timeout)
    signal.alarm(MAX_RUN_SECONDS)
    try:
        # These small monthly solves are faster without multithreaded BLAS overhead.
        with threadpool_limits(limits=1, user_api="blas"):
            main()
    finally:
        signal.alarm(0)
