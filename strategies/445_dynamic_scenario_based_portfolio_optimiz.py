# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "445_ dynamic_scenario_based_portfolio_optimiz"\nuniverse:\n  [VT, VTI, ACWI, ACWX, SUSA, ROUS, IYG, BND, FPE, BYLD, NTSX, GAL, MDIV,\n   AOM, AOR, INKM, HNDL, ALTY, IYLD, CVY, PDBC, QAI, HDG, MNA, RAAX, RLY,\n   SRET, YYY, GAA, GMOM]\nrisk_free_ticker: BIL\ndates:\n  price_start: 2018-12-19\n  evaluation_start: 2022-12-21\n  is_end: 2024-12-31\n  end: 2026-09-30\nmodel:\n  lookback_sessions: 1008\n  cvar_confidence: 0.95\n  target_turnover_cap: 0.00013227513227513228  # 1 / (30 * 252)\n  execution_lag_sessions: 1\ncosts:\n  commission_bps: 10.0\n  slippage_bps: 5.0\nannualization_sessions: 252\n'
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
import math
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
TRADE_TOLERANCE = 1e-10


def download_closes(tickers, start, end):
    settings = dict(start=start.date().isoformat(),
                    end=(end + pd.Timedelta(days=1)).date().isoformat(),
                    auto_adjust=True, progress=False, timeout=12)
    frame = yf.download(tickers, threads=8, **settings)
    closes = frame["Close"].reindex(columns=tickers)
    for ticker in tickers:
        if closes[ticker].notna().any():
            continue
        retry = yf.download(ticker, threads=False, **settings)["Close"]
        closes = closes.combine_first(retry)
    return closes


def check_closes(closes, tickers, start, end, calendar_ticker):
    closes = closes.reindex(columns=tickers).sort_index().rename_axis(index="Date", columns=None)
    closes.index = pd.DatetimeIndex(closes.index, dtype="datetime64[ns]")
    if closes.index.has_duplicates:
        raise ValueError("Yahoo Finance returned duplicate session dates")
    closes = closes.loc[start:end].dropna(how="all")
    calendar = closes[calendar_ticker].dropna().index
    if (calendar.empty or calendar[0] - start > pd.Timedelta(days=7)
            or end - calendar[-1] > pd.Timedelta(days=7)):
        available = "empty" if calendar.empty else f"{calendar[0].date()} to {calendar[-1].date()}"
        raise ValueError(f"{calendar_ticker} price coverage is incomplete: {available}")
    invalid = ~np.isfinite(closes) | (closes <= 0)
    if invalid.any().any():
        missing = invalid.sum()
        raise ValueError(f"Incomplete adjusted close panel (rows by ticker): "
                         f"{missing[missing > 0].to_dict()}")
    return closes


def load_closes(tickers, start, end, calendar_ticker):
    CACHE.mkdir(exist_ok=True)
    key = hashlib.sha256(f"auto_adjust_close|{tickers}|{start}|{end}".encode()).hexdigest()[:16]
    path = CACHE / f"yahoo_{key}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            fetched_at = datetime.fromisoformat(handle.readline().strip()).isoformat()
            closes = pd.read_csv(handle, index_col="Date", parse_dates=["Date"])
        return check_closes(closes, tickers, start, end, calendar_ticker), fetched_at

    with TemporaryDirectory(dir=CACHE, prefix=".yfinance_") as transient_cache:
        yf.set_tz_cache_location(transient_cache)
        closes = download_closes(tickers, start, end)
    closes = check_closes(closes, tickers, start, end, calendar_ticker)
    fetched_at = datetime.now(timezone.utc).isoformat()
    with NamedTemporaryFile(mode="w", dir=CACHE, encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(fetched_at + "\n")
            closes.to_csv(handle, index_label="Date")
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return closes, fetched_at


class HistoricalCVaR:
    def __init__(self, n_assets, lookback, confidence, target_cap):
        self.n_assets = n_assets
        self.target_cap = target_cap
        # LP variables are weights, absolute target changes, VaR, and scenario excess losses.
        self.objective = np.r_[np.zeros(2 * n_assets), 1,
                               np.full(lookback, 1 / ((1 - confidence) * lookback))]
        identity = eye(n_assets, format="csr")
        zeros = csr_matrix((n_assets, 1 + lookback))
        self.change_constraints = vstack([
            hstack([identity, -identity, zeros]),
            hstack([-identity, -identity, zeros]),
            csr_matrix(np.r_[np.zeros(n_assets), np.ones(n_assets), np.zeros(1 + lookback)])
        ], format="csr")
        self.scenario_slack = hstack([csr_matrix((lookback, n_assets)),
                                      csr_matrix(-np.ones((lookback, 1))),
                                      -eye(lookback, format="csr")], format="csr")
        self.budget = csr_matrix(np.r_[np.ones(n_assets), np.zeros(n_assets + 1 + lookback)])
        self.bounds = ([(0, 1)] * n_assets + [(0, None)] * n_assets
                       + [(None, None)] + [(0, None)] * lookback)
        self.scenario_rhs = np.zeros(lookback)

    def target(self, scenarios, previous):
        # Percent units keep portfolio loss and VaR away from solver precision limits.
        loss_constraints = hstack([csr_matrix(-100 * scenarios), self.scenario_slack])
        result = linprog(
            self.objective,
            A_ub=vstack([self.change_constraints, loss_constraints], format="csr"),
            b_ub=np.r_[previous, -previous, 2 * self.target_cap, self.scenario_rhs],
            A_eq=self.budget, b_eq=[1.0], bounds=self.bounds, method="highs-ds",
        )
        if not result.success:
            raise RuntimeError(f"CVaR optimization failed: {result.message}")
        weights = np.maximum(result.x[:self.n_assets], 0)
        weights /= weights.sum()
        change = 0.5 * np.abs(weights - previous).sum()
        if change > self.target_cap + 1e-7:
            raise RuntimeError("CVaR target exceeds the adjustment cap")
        if change > self.target_cap:
            weights = previous + (weights - previous) * (self.target_cap / change)
        return weights


def close_fill(drifted, target, fee_rate):
    def order_sizes(fee):
        return np.abs((1 - fee) * target - drifted)

    # The target is applied to post-cost NAV, so fees and order notionals are self-consistent.
    fee = brentq(lambda fee: fee - fee_rate * order_sizes(fee).sum(), 0, 2 * fee_rate)
    orders = order_sizes(fee)
    return fee, orders.sum(), np.count_nonzero(orders > TRADE_TOLERANCE)


def backtest(returns, universe, evaluation_start, model, fee_rate):
    lookback = model["lookback_sessions"]
    lag = model["execution_lag_sessions"]
    first_fill = returns.index.searchsorted(evaluation_start)
    if first_fill < lookback - 1 + lag or first_fill >= len(returns):
        raise ValueError("The evaluation start needs a full estimation window before the first fill")

    n_assets = len(universe)
    optimizer = HistoricalCVaR(n_assets, lookback, model["cvar_confidence"],
                               model["target_turnover_cap"])
    equal = np.full(n_assets, 1 / n_assets)
    previous_target = equal.copy()
    weights, benchmark_weights = np.zeros((2, n_assets))
    targets = {}
    asset_returns = returns[universe].to_numpy()
    rows = []

    for i in range(first_fill - lag, len(returns)):
        if i >= first_fill:
            daily = asset_returns[i]
            gross = 1 + weights @ daily
            benchmark_gross = 1 + benchmark_weights @ daily
            drifted = weights * (1 + daily) / gross
            benchmark_weights = benchmark_weights * (1 + daily) / benchmark_gross
            weights = targets.pop(i)
            fee, turnover, trades = close_fill(drifted, weights, fee_rate)
            benchmark_fee = benchmark_turnover = benchmark_trades = 0
            if i == first_fill:
                benchmark_fee, benchmark_turnover, benchmark_trades = close_fill(
                    benchmark_weights, equal, fee_rate)
                benchmark_weights = equal.copy()
            rows.append((gross * (1 - fee) - 1, turnover, trades,
                         benchmark_gross * (1 - benchmark_fee) - 1,
                         benchmark_turnover, benchmark_trades))

        if i + lag < len(returns):
            scenarios = asset_returns[i - lookback + 1:i + 1]
            previous_target = optimizer.target(scenarios, previous_target)
            targets[i + lag] = previous_target

    columns = pd.MultiIndex.from_product(
        [["strategy", "benchmark"], ["return", "turnover", "n_trades"]])
    return pd.DataFrame(rows, index=returns.index[first_fill:], columns=columns)


def metrics(frame, risk_free, sessions_per_year):
    daily = frame["return"].to_numpy()
    excess = daily - risk_free.reindex(frame.index).to_numpy()
    years = len(daily) / sessions_per_year
    wealth = np.r_[1.0, np.cumprod(1 + daily)]
    vol = float(daily.std(ddof=1))
    excess_vol = float(excess.std(ddof=1))
    annualizer = math.sqrt(sessions_per_year)
    return {
        "cagr": float(wealth[-1] ** (1 / years) - 1),
        "ann_vol": vol * annualizer,
        "sharpe": float(annualizer * daily.mean() / vol) if vol else None,
        "sharpe_excess": float(annualizer * excess.mean() / excess_vol) if excess_vol else None,
        "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
        "total_return": float(wealth[-1] - 1),
        "turnover": float(frame["turnover"].sum() / years),
        "n_trades": int(frame["n_trades"].sum()),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    universe = config["universe"]
    risk_free_ticker = config["risk_free_ticker"]
    model = config["model"]
    start, evaluation_start, is_end, end = (
        pd.Timestamp(config["dates"][key]) for key in ("price_start", "evaluation_start", "is_end", "end"))
    commission, slippage = (config["costs"][key] for key in ("commission_bps", "slippage_bps"))
    sessions_per_year = config["annualization_sessions"]
    if not start < evaluation_start <= is_end < end:
        raise ValueError("Expected price_start < evaluation_start <= is_end < end")
    if len(set(universe)) != len(universe) or not universe or risk_free_ticker != "BIL":
        raise ValueError("Expected a unique ETF universe and BIL for the risk-free proxy")
    if (model["lookback_sessions"] < 2 or model["execution_lag_sessions"] < 1
            or not 0 < model["cvar_confidence"] < 1 or not 0 <= model["target_turnover_cap"] <= 1):
        raise ValueError("Invalid rolling window, execution lag, confidence level, or adjustment cap")
    if not (commission >= 0 and slippage >= 0 and commission + slippage < 5000
            and sessions_per_year > 0):
        raise ValueError("Invalid costs or annualization convention")

    tickers = list(dict.fromkeys([*universe, risk_free_ticker]))
    closes, fetched_at = load_closes(tickers, start, end, risk_free_ticker)
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True adjusted-close total-return proxy", flush=True)
    print("PARAM_COMBOS_TRIED: 1", flush=True)
    returns = closes.pct_change(fill_method=None).iloc[1:]
    daily = backtest(returns, universe, evaluation_start, model, (commission + slippage) / 10_000)
    periods = {"is": daily.loc[daily.index <= is_end], "oos": daily.loc[daily.index > is_end], "full": daily}
    if min(len(period) for period in periods.values()) < 2:
        raise ValueError("IS and OOS each need at least two daily observations")

    risk_free = returns[risk_free_ticker]
    strategy_stats = {name: metrics(period["strategy"], risk_free, sessions_per_year)
                      for name, period in periods.items()}
    benchmark_stats = {name: metrics(period["benchmark"], risk_free, sessions_per_year)
                       for name, period in periods.items()}
    for name in periods:
        print(f"{name.upper()}: {json.dumps(strategy_stats[name], allow_nan=False)}")
        print(f"BENCHMARK {name.upper()} (equal-weight buy and hold): "
              f"{json.dumps(benchmark_stats[name], allow_nan=False)}")

    results = {
        "strategy": config["strategy"],
        "data_range": {"start": daily.index[0].date().isoformat(),
                       "end": daily.index[-1].date().isoformat(),
                       "price_start": closes.index[0].date().isoformat(),
                       "price_end": closes.index[-1].date().isoformat()},
        **strategy_stats,
        "benchmark": {"name": "equal-weight buy and hold (same ETF universe)", **benchmark_stats},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
