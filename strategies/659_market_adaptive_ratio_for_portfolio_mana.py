"""659_market_adaptive_ratio_for_portfolio_mana
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/659_market_adaptive_ratio_for_portfolio_mana.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 9532cd9f815cc2d43ceb143b2f1416acf34e9864 on 2026-11-27 19:50 Europe/Oslo, exit 0, output sha256 e473fcd48971b33ac4d1f7c3748cd7dfb744e6926bc45d87722f448029ac8f92
Data source: Public Yahoo Finance daily ETF opens, closes and distributions via yfinance. The portfolio universe is SPY, QQQ, IWM, TLT and IEF. SPY is the market proxy and BIL is the risk-free proxy. Distributions include dividends and capital gains when the latter field is returned. The recorded data fetch timestamp is 2026-11-27T17:58:32+00:00.
Data fetched: 2026-11-27T18:50:23+00:00
Sample period: Daily data from 2009-01-02 through 2025-12-31 inclusive, with warmup before the reported backtest of 2010-01-04 through 2025-12-31 inclusive.   data_range (from output): ['2010-01-04', '2025-12-31']
IS/OOS split: IS is 2010-01-04 through 2014-12-31 inclusive. OOS is 2015-01-02 through 2025-12-31 inclusive. The default run carries positions and cash across the split. Target estimation continues rolling through OOS with fixed default hyperparameters. The six robustness variants run only on IS, and the code checks the default full run's IS rows against the standalone IS run for exact equality.
Costs and slippage: Default commission is 2 bps and modeled slippage is 3 bps. Both are charged as a combined 5 bps on gross buy plus sell notional at the open, through the self-financing rebalance budget. Slippage is a fixed proportional cash charge. The IS doubled-cost variant uses 4 bps commission and 6 bps slippage.
Signal lag: The strategy rebalances at the open of the first trading session of each month. With the default one-session lag, a target calculated using observations through close t is filled at the next trading session's open when that is the scheduled rebalance.
Price adjustment: yfinance auto_adjust=False, Yahoo split-normalized open and close, total return with distributions reinvested at the ex-date open
Survivorship bias: The universe is a fixed list of existing ETFs selected retrospectively. The README explicitly identifies fund survivorship and universe-selection bias. The loader requests this fixed universe and the configured proxies for the full date range.
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.10299392136135621, ann_vol=0.11254228107176235, sharpe=0.9275351807280455, sharpe_excess=0.9314437649100206, max_drawdown=-0.13038516667623412, total_return=0.6312770012395024, turnover=41.68624709185328, n_trades=144
    oos: cagr=0.06446505455465945, ann_vol=0.1456389966874061, sharpe=0.5019877962539483, sharpe_excess=0.3765657703696137, max_drawdown=-0.313621378160992, total_return=0.9851782468098609, turnover=91.46217316493299, n_trades=270
    full: cagr=0.07636325101519215, ann_vol=0.1361483752931868, sharpe=0.6088007816472565, sharpe_excess=0.5175998497332701, max_drawdown=-0.31362137816099334, total_return=2.238375617381844, turnover=133.14842025678627, n_trades=414
    benchmark  full: cagr=0.10229472853804689, ann_vol=0.10935838685505662, sharpe=0.9455013227378561, sharpe_excess=0.8314337948397459, max_drawdown=-0.2727375168265743, total_return=3.736100429189607, turnover=5.734076260712848, n_trades=960
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 659_market_adaptive_ratio_for_portfolio_mana\nuniverse: [SPY, QQQ, IWM, TLT, IEF]\nmarket_proxy: SPY\nrisk_free_proxy: BIL\ndata_start: '2009-01-02'\nstart: '2010-01-04'\nis_end: '2014-12-31'\nend: '2025-12-31'\nsignal_lag_sessions: 1\ncommission_bps: 2\nslippage_bps: 3\nparameters:\n  trend_sessions: 200\n  lookback_sessions: 120\n  alpha: 5.0\nrobustness:\n  - {name: alpha_4, parameters: {alpha: 4.0}}\n  - {name: alpha_6, parameters: {alpha: 6.0}}\n  - {name: lookback_100, parameters: {lookback_sessions: 100}}\n  - {name: lookback_140, parameters: {lookback_sessions: 140}}\n  - {name: fixed_rho_1, parameters: {alpha: 0.0}}\n  - {name: doubled_costs, commission_bps: 4, slippage_bps: 6}\n"
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
"""C236 research backtest, Market-Adaptive Ratio for Portfolio Management.

Paper: https://arxiv.org/abs/2312.13719
Revised full text: https://arxiv.org/html/2312.13719
Related implementations consulted for design only:
- PeDiot RRL (no license located): https://github.com/PeDiot/rrl-trading-system
  https://raw.githubusercontent.com/PeDiot/rrl-trading-system/master/rrl_trading/model/optimize.py
- FinRL (MIT): https://github.com/AI4Finance-Foundation/FinRL
  https://finrl.readthedocs.io/en/latest/tutorial/Introduction/PortfolioAllocation.html
  https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/env_portfolio_optimization/env_portfolio_optimization.py
  https://pypi.org/project/FinRL/
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
from scipy.optimize import LinearConstraint, brentq, minimize
from scipy.special import expit


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
FIELDS = ["Open", "Close", "Distribution"]
PRICE_ADJUSTMENT = (
    "yfinance auto_adjust=False, Yahoo split-normalized open and close, "
    "total return with distributions reinvested at the ex-date open"
)


def load_bars(cfg, tickers):
    data_start = pd.Timestamp(cfg["data_start"])
    end = pd.Timestamp(cfg["end"])
    end_exclusive = (end + pd.Timedelta(days=1)).date().isoformat()
    columns = pd.MultiIndex.from_product([FIELDS, tickers], names=["Field", "Ticker"])
    key = json.dumps(["split-normalized-actions-v1", tickers, cfg["data_start"], end_exclusive])
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"yahoo_daily_{digest}.csv"
    fetched_from_cache = path.exists()
    if fetched_from_cache:
        with path.open(encoding="utf-8") as cached:
            fetched_at = datetime.fromisoformat(cached.readline().strip()).isoformat()
            bars = pd.read_csv(cached, header=[0, 1], index_col=0, parse_dates=True)
    else:
        with TemporaryDirectory(dir=CACHE) as session_files:
            yf.set_tz_cache_location(session_files)
            quotes = yf.download(
                tickers, start=cfg["data_start"], end=end_exclusive, interval="1d",
                auto_adjust=False, actions=True, repair=False, ignore_tz=True,
                progress=False, threads=4, timeout=15,
            )
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        expected = pd.MultiIndex.from_product([["Open", "Close", "Dividends"], tickers])
        if quotes.empty or not expected.isin(quotes.columns).all():
            raise RuntimeError("Yahoo Finance did not return prices and dividends for the full universe")
        bars = quotes.loc[:, expected].rename(columns={"Dividends": "Distribution"}, level=0)
        for ticker in tickers:
            gains_key = ("Capital Gains", ticker)
            if gains_key in quotes:
                bars[("Distribution", ticker)] += quotes[gains_key].fillna(0.0)

    if (
        bars.empty or not isinstance(bars.index, pd.DatetimeIndex)
        or not bars.index.is_unique or not bars.index.is_monotonic_increasing
        or bars.index[0] != data_start or bars.index[-1] != end
        or not columns.isin(bars.columns).all()
    ):
        raise ValueError("Incomplete Yahoo daily history or unexpected session dates")
    bars = bars.reindex(columns=columns).rename_axis("session")
    if (
        not np.isfinite(bars.to_numpy()).all()
        or (bars[["Open", "Close"]] <= 0.0).any().any()
        or (bars["Distribution"] < 0.0).any().any()
    ):
        raise ValueError("Invalid or missing price/distribution in the ETF panel")

    if not fetched_from_cache:
        temporary_path = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as temp:
                temporary_path = Path(temp.name)
                temp.write(fetched_at + "\n")
                bars.to_csv(temp)
            temporary_path.replace(path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    return bars, fetched_at


def market_inputs(bars):
    opens = bars["Open"]
    closes = bars["Close"]
    # These ratios preserve comparable units across splits and credit
    # ex-date distributions to positions carried from the previous close.
    overnight = (opens + bars["Distribution"]) / closes.shift(1)
    intraday = closes / opens
    return overnight, intraday, overnight * intraday - 1.0


def adaptive_weights(history, rho):
    expected = history.mean(axis=0) * SESSIONS_PER_YEAR
    covariance = np.cov(history, rowvar=False) * SESSIONS_PER_YEAR
    variance_scale = covariance.diagonal().max()
    if variance_scale <= 0.0:
        raise ValueError("A rolling window has no measurable portfolio risk")
    covariance /= variance_scale
    # Numerical ridge in variance-relative units.
    covariance += 1e-10 * np.eye(expected.size)

    n_assets = expected.size
    equal_weights = np.full(n_assets, 1.0 / n_assets)
    bounds = [(0.0, 1.0)] * n_assets
    simplex = LinearConstraint(np.ones(n_assets), 1.0, 1.0)
    solver_options = {"ftol": 1e-11, "maxiter": 150}

    def variance(weights):
        return weights @ covariance @ weights, 2.0 * (covariance @ weights)

    if expected.max() <= 0:
        result = minimize(
            variance, equal_weights, jac=True, method="SLSQP", bounds=bounds,
            constraints=[simplex], options=solver_options,
        )
        if not result.success:
            raise RuntimeError(f"Minimum-variance optimization failed: {result.message}")
        weights = np.clip(result.x, 0.0, 1.0)
        return weights / weights.sum()

    # For positive excess means, maximizing log(MAR) preserves its ordering
    # while avoiding the unstable range of the original powers.
    expected /= expected.max()
    floor = 1e-8

    def negative_log_ratio(weights):
        mean = weights @ expected
        risk, risk_gradient = variance(weights)
        # Feasible solutions are above the floor. This extension remains
        # finite when SLSQP evaluates a point outside the return constraint.
        log_mean = np.log(max(mean, floor)) + min(mean - floor, 0.0) / floor
        loss = np.log(risk) / (2.0 * rho) - rho * log_mean
        gradient = risk_gradient / (2.0 * rho * risk) - rho * expected / max(mean, floor)
        return loss, gradient

    excess_constraint = LinearConstraint(expected, floor, np.inf)
    positive_vertices = np.eye(n_assets)[expected >= floor]
    starts = list(positive_vertices)
    if equal_weights @ expected >= floor:
        starts.append(equal_weights)

    solutions = []
    for starting_weights in starts:
        result = minimize(
            negative_log_ratio, starting_weights, jac=True, method="SLSQP", bounds=bounds,
            constraints=[simplex, excess_constraint], options=solver_options,
        )
        if result.success:
            weights = np.clip(result.x, 0.0, 1.0)
            weights /= weights.sum()
            if weights @ expected >= floor:
                solutions.append(weights)
    if not solutions:
        raise RuntimeError("No feasible market-adaptive ratio solution")
    return min(solutions, key=lambda weights: negative_log_ratio(weights)[0])


def monthly_targets(bars, close_returns, cfg, equal_weight=False):
    dates = bars.index
    first_sessions = ~dates.to_period("M").duplicated()
    fill_rows = np.flatnonzero(first_sessions & (dates >= pd.Timestamp(cfg["start"])))
    universe = cfg["universe"]
    if equal_weight:
        return {dates[row]: np.full(len(universe), 1.0 / len(universe)) for row in fill_rows}

    settings = cfg["parameters"]
    lookback = settings["lookback_sessions"]
    lag = cfg["signal_lag_sessions"]
    market_close = bars["Close"][cfg["market_proxy"]]
    trend = market_close / market_close.rolling(settings["trend_sessions"]).mean() - 1.0
    rhos = (2.0 * expit(settings["alpha"] * trend)).to_numpy()
    excess = close_returns[universe].sub(close_returns[cfg["risk_free_proxy"]], axis=0)
    excess = excess.to_numpy()

    targets = {}
    for fill_row in fill_rows:
        signal_row = fill_row - lag
        first_return_row = signal_row - lookback + 1
        if first_return_row < 1 or not 0.0 < rhos[signal_row] < 2.0:
            raise ValueError(f"Insufficient or invalid signal history for {dates[fill_row].date()}")
        history = excess[first_return_row : signal_row + 1]
        targets[dates[fill_row]] = adaptive_weights(history, rhos[signal_row])
    return targets


def fill_targets(open_positions, cash, weights, cost_rate):
    # Solve post-cost invested fraction in open-marked units. Subtracting
    # the gross buy/sell cost inside this equation keeps the fill self-financing.
    open_value = open_positions.sum() + cash
    current = open_positions / open_value

    def budget_gap(invested_fraction):
        gross_fraction = np.abs(weights * invested_fraction - current).sum()
        return invested_fraction + cost_rate * gross_fraction - 1.0

    invested = brentq(budget_gap, 0.0, 1.0, xtol=1e-15)
    target = weights * invested
    trade_fractions = np.abs(target - current)
    gross = trade_fractions.sum()
    cash_fraction = 1.0 - target.sum() - cost_rate * gross
    if cash_fraction < -1e-12:
        raise RuntimeError("Open fill exceeded its cash budget")
    trades = int(np.count_nonzero(trade_fractions > 1e-10))
    return target * open_value, max(cash_fraction, 0.0) * open_value, gross, trades


def simulate(bars, cfg, equal_weight=False):
    overnight, intraday, close_returns = market_inputs(bars)
    targets = monthly_targets(bars, close_returns, cfg, equal_weight=equal_weight)
    start = pd.Timestamp(cfg["start"])
    universe = cfg["universe"]
    overnight_values = overnight.loc[start:, universe].to_numpy()
    intraday_values = intraday.loc[start:, universe].to_numpy()
    dates = bars.loc[start:].index
    cost_bps = np.array([cfg["commission_bps"], cfg["slippage_bps"]], dtype=float)
    cost_rate = cost_bps.sum() / 10_000.0
    if (cost_bps < 0.0).any() or not 0.0 <= cost_rate < 1.0:
        raise ValueError("Costs must be nonnegative and total less than 10,000 bps")

    positions = np.zeros(len(universe))
    cash = 1.0
    previous_close_value = 1.0
    records = []
    for row, date in enumerate(dates):
        positions *= overnight_values[row]
        turnover = 0.0
        trades = 0
        if date in targets:
            positions, cash, turnover, trades = fill_targets(
                positions, cash, targets[date], cost_rate,
            )
        positions *= intraday_values[row]
        close_value = positions.sum() + cash
        records.append((close_value / previous_close_value - 1.0, turnover, trades, cash))
        previous_close_value = close_value

    daily_results = pd.DataFrame(
        records, index=dates, columns=["return", "turnover", "n_trades", "cash"],
    )
    cash_proxy = cfg["risk_free_proxy"]
    daily_results["cash_proxy_return"] = close_returns.loc[dates, cash_proxy]
    daily_results.loc[dates[0], "cash_proxy_return"] = intraday.at[dates[0], cash_proxy] - 1.0
    return daily_results


def metrics(daily_results):
    returns = daily_results["return"].to_numpy()
    excess = returns - daily_results["cash_proxy_return"].to_numpy()
    growth = np.r_[1.0, np.cumprod(1.0 + returns)]
    ann_scale = np.sqrt(SESSIONS_PER_YEAR)
    vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1.0),
        "ann_vol": float(vol * ann_scale),
        "sharpe": float(returns.mean() / vol * ann_scale) if vol else None,
        "sharpe_excess": float(excess.mean() / excess_vol * ann_scale) if excess_vol else None,
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1.0)),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, is_end):
    return {
        "is": metrics(daily_results.loc[:is_end]),
        "oos": metrics(daily_results.loc[daily_results.index > is_end]),
        "full": metrics(daily_results),
    }


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    lag = cfg["signal_lag_sessions"]
    if not isinstance(lag, int) or isinstance(lag, bool) or lag < 1:
        raise ValueError("Signal lag must be at least one whole session")
    universe = cfg["universe"]
    if len(universe) < 2 or len(set(universe)) != len(universe):
        raise ValueError("The ETF universe must have at least two distinct tickers")
    tickers = list(dict.fromkeys(universe + [cfg["market_proxy"], cfg["risk_free_proxy"]]))
    dates = [pd.Timestamp(cfg[key]) for key in ["data_start", "start", "is_end", "end"]]
    if any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("Warmup, start, IS end and sample end must be in chronological order")
    bars, fetched_at = load_bars(cfg, tickers)
    is_end = dates[2]
    is_bars = bars.loc[:is_end]
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")

    print("ROBUSTNESS (IS only)")
    default_is = simulate(is_bars, cfg)
    print("default " + json.dumps(metrics(default_is), allow_nan=False))
    trials = 1
    for variant in cfg["robustness"]:
        variant_cfg = cfg | variant | {
            "parameters": cfg["parameters"] | variant.get("parameters", {}),
        }
        variant_is = simulate(is_bars, variant_cfg)
        trials += 1
        print(variant["name"] + " " + json.dumps(metrics(variant_is), allow_nan=False))
    print(f"PARAM_COMBOS_TRIED: {trials}")

    official = simulate(bars, cfg)
    pd.testing.assert_frame_equal(official.loc[:is_end], default_is, check_exact=True)
    benchmark = simulate(bars, cfg, equal_weight=True)
    strategy_periods = period_metrics(official, is_end)
    benchmark_periods = period_metrics(benchmark, is_end)
    if any(strategy_periods[period]["n_trades"] <= 0 for period in ("is", "oos")):
        raise RuntimeError("Expected strategy fills in both evaluation periods")
    print("LOOKAHEAD_CHECK: prior close signals, open sizing, ex-date distributions, exact IS prefix match")
    benchmark_name = "monthly_equal_weight_same_etfs"
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: " + json.dumps(strategy_periods[period], allow_nan=False))
        print(f"BENCHMARK {period.upper()}: " + json.dumps(benchmark_periods[period], allow_nan=False))
    report = {
        "strategy": cfg["strategy"],
        "data_range": [date.date().isoformat() for date in (official.index[0], official.index[-1])],
        **strategy_periods,
        "benchmark": {"name": benchmark_name, **benchmark_periods},
    }
    print("RESULTS_JSON: " + json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    main()
