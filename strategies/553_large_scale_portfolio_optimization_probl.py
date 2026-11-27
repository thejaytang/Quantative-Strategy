# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "553_ large_scale_portfolio_optimization_probl"\nuniverse:\n  - AAPL\n  - ABBV\n  - ABT\n  - ADBE\n  - AMD\n  - AMGN\n  - AMZN\n  - AVGO\n  - BAC\n  - BRK-B\n  - CAT\n  - CMCSA\n  - COP\n  - COST\n  - CRM\n  - CSCO\n  - CVX\n  - DE\n  - DIS\n  - DUK\n  - F\n  - GE\n  - GILD\n  - GOOG\n  - GS\n  - HD\n  - IBM\n  - INTC\n  - JNJ\n  - JPM\n  - KO\n  - LIN\n  - LLY\n  - LOW\n  - MA\n  - MCD\n  - MDLZ\n  - META\n  - MMM\n  - MRK\n  - MS\n  - MSFT\n  - NEE\n  - NFLX\n  - NVDA\n  - ORCL\n  - PEP\n  - PFE\n  - PG\n  - QCOM\n  - SBUX\n  - T\n  - TGT\n  - TMO\n  - TSLA\n  - TXN\n  - UNH\n  - UNP\n  - UPS\n  - V\n  - VZ\n  - WFC\n  - WMT\n  - XOM\nbenchmark: SPY\nrisk_free_proxy: BIL\ndata_start: 2010-01-01\nbacktest_start: 2012-01-03\nis_end: 2019-12-31\nend: 2025-12-31\nparameters:\n  lookback_sessions: 252\n  liquidity_sessions: 63\n  liquidity_top_n: 50\n  cardinality: 10\n  risk_aversion: 6.0\n  min_weight: 0.01\n  max_weight: 0.99\n  lag_sessions: 1\n  transaction_cost_bps: 5\n  slippage_bps: 5\nrobustness:\n  cardinality_8: {cardinality: 8}\n  cardinality_12: {cardinality: 12}\n  lookback_189: {lookback_sessions: 189}\n  lookback_315: {lookback_sessions: 315}\n  risk_aversion_4: {risk_aversion: 4.0}\n  risk_aversion_8: {risk_aversion: 8.0}\n  double_friction: {transaction_cost_bps: 10, slippage_bps: 10}\n'
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
"""Monthly cardinality-constrained mean-variance research backtest."""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
TRADING_DAYS = 252


@dataclass(frozen=True)
class Parameters:
    lookback_sessions: int
    liquidity_sessions: int
    liquidity_top_n: int
    cardinality: int
    risk_aversion: float
    min_weight: float
    max_weight: float
    lag_sessions: int
    transaction_cost_bps: float
    slippage_bps: float

    @property
    def friction(self):
        return (self.transaction_cost_bps + self.slippage_bps) / 10_000

    def validate(self, universe_size):
        if self.lag_sessions < 1 or self.lookback_sessions < 2 or self.liquidity_sessions < 1:
            raise ValueError("Use a positive execution lag and valid estimation windows.")
        if not 1 <= self.cardinality <= self.liquidity_top_n <= universe_size:
            raise ValueError("Invalid cardinality or liquidity shortlist size.")
        equal_weight = 1 / self.cardinality
        if not 0 < self.min_weight <= equal_weight <= self.max_weight <= 1:
            raise ValueError("Weight bounds are incompatible with full investment.")
        if not (self.risk_aversion > 0 and self.transaction_cost_bps >= 0
                and self.slippage_bps >= 0 and self.friction < 1):
            raise ValueError("Use positive risk aversion and nonnegative friction below 100%.")


def download_panel(universe, tickers, start, end):
    end_exclusive = (end + pd.Timedelta(days=1)).date().isoformat()
    remaining = list(tickers)
    panels = []
    for _ in range(2):
        raw = yf.download(
            remaining, start=start.date().isoformat(), end=end_exclusive,
            interval="1d", auto_adjust=False, repair=False, ignore_tz=True,
            multi_level_index=True, threads=4, progress=False, timeout=12,
        )
        if raw is None or raw.empty:
            continue
        adjusted = raw["Adj Close"].reindex(columns=remaining)
        adjusted = adjusted.where((adjusted > 0) & np.isfinite(adjusted))
        dollar_volume = (raw["Close"] * raw["Volume"]).reindex(columns=remaining)
        dollar_volume = dollar_volume.where((dollar_volume > 0) & np.isfinite(dollar_volume))
        available = [
            ticker for ticker in remaining
            if adjusted[ticker].notna().any()
            and (ticker not in universe or dollar_volume[ticker].notna().any())
        ]
        panels.append(pd.concat({
            "adjusted_close": adjusted[available],
            "dollar_volume": dollar_volume[available],
        }, axis=1))
        remaining = [ticker for ticker in remaining if ticker not in available]
        if not remaining:
            return pd.concat(panels, axis=1).sort_index()
    raise RuntimeError(f"Yahoo Finance returned no usable history for {', '.join(remaining)}.")


def load_panel(universe, benchmark, risk_free, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    tickers = [*universe, benchmark, risk_free]
    cache_key = json.dumps(["daily-v1", tickers, str(start.date()), str(end.date())])
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    path = cache_dir / f"yfinance_{digest}.csv.gz"

    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as source:
            fetched_at = source.readline().strip()
            panel = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True)
    else:
        with TemporaryDirectory(dir=cache_dir, prefix="yfinance_session_") as session_dir:
            yf.set_tz_cache_location(session_dir)
            panel = download_panel(universe, tickers, start, end)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    adjusted = panel["adjusted_close"].reindex(columns=tickers)
    proxy_prices = adjusted[[benchmark, risk_free]].dropna(how="all")
    if proxy_prices.empty or proxy_prices.isna().any().any():
        raise ValueError("Benchmark and risk-free proxy must share complete daily observations.")
    if (not proxy_prices.index.is_unique or not proxy_prices.index.is_monotonic_increasing
            or proxy_prices.index[0] - start > pd.Timedelta(days=7)
            or end - proxy_prices.index[-1] > pd.Timedelta(days=7)):
        raise ValueError("Benchmark calendar does not cover the requested data range.")

    if not path.exists():
        with NamedTemporaryFile(dir=cache_dir, suffix=".tmp", delete=False) as staged:
            staged_path = Path(staged.name)
        try:
            with gzip.open(staged_path, "wt", encoding="utf-8") as destination:
                destination.write(f"{fetched_at}\n")
                panel.to_csv(destination)
            staged_path.replace(path)
        finally:
            staged_path.unlink(missing_ok=True)

    adjusted = adjusted.reindex(proxy_prices.index)
    liquidity = panel["dollar_volume"].reindex(index=proxy_prices.index, columns=universe)
    return adjusted, liquidity, fetched_at


def fit_weights(history, params, date):
    expected = history.mean(axis=0) * TRADING_DAYS
    covariance = np.atleast_2d(np.cov(history, rowvar=False, ddof=1)) * TRADING_DAYS
    risk_matrix = covariance * params.risk_aversion

    def objective(weights):
        return 0.5 * weights @ risk_matrix @ weights - expected @ weights

    def gradient(weights):
        return risk_matrix @ weights - expected

    equal = np.full(params.cardinality, 1 / params.cardinality)
    result = minimize(
        objective, equal, jac=gradient, method="SLSQP",
        bounds=[(params.min_weight, params.max_weight)] * params.cardinality,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1,
                     "jac": lambda weights: np.ones_like(weights)},
        options={"ftol": 1e-11, "maxiter": 200},
    )
    weights = result.x
    if (not result.success or not np.isfinite(weights).all()
            or abs(weights.sum() - 1) > 1e-8
            or weights.min() < params.min_weight - 1e-8
            or weights.max() > params.max_weight + 1e-8):
        raise RuntimeError(f"{date.date()} mean-variance optimization failed: {result.message}")
    return weights


def monthly_targets(prices, dollar_volume, proxy_returns, start, params):
    stock_returns = prices.pct_change(fill_method=None)
    liquidity = dollar_volume.rolling(params.liquidity_sessions).median()
    dates = stock_returns.index
    months = dates.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])
    targets = {}
    for row in month_ends:
        execution = row + params.lag_sessions
        if execution >= len(dates) or dates[execution] < start:
            continue
        date = dates[row]
        first = row - params.lookback_sessions + 1
        if first < 1:
            raise ValueError(f"Download more warm-up history before {date.date()}.")

        history = stock_returns.iloc[first:row + 1].dropna(axis=1)
        liquid = (liquidity.iloc[row].reindex(history.columns).dropna()
                  .nlargest(params.liquidity_top_n).index)
        excess = history[liquid].sub(proxy_returns.iloc[first:row + 1], axis=0)
        volatility = excess.std(ddof=1)
        scores = excess.mean().div(volatility.where(volatility > 0)).dropna()
        selected = scores.nlargest(params.cardinality).index
        if len(selected) < params.cardinality:
            raise ValueError(f"{date.date()} has too few stocks with complete estimation windows.")

        target = np.zeros(prices.shape[1])
        target[prices.columns.get_indexer(selected)] = fit_weights(
            history[selected].to_numpy(), params, date,
        )
        targets[execution] = target
    return targets


def simulate(prices, targets, start, friction):
    """Mark existing weights through the close, then apply any eligible rebalance."""
    first_row = prices.index.searchsorted(start)
    observed = prices.notna().to_numpy()
    changes = prices.ffill().pct_change(fill_method=None).fillna(0.0).to_numpy()
    weights = np.zeros(prices.shape[1])
    cash_weight = 1.0
    pending = None
    returns, turnover, trades = [], [], []
    for row in range(first_row, len(prices)):
        marked_weights = weights * (1 + changes[row])
        gross_growth = cash_weight + marked_weights.sum()
        weights = marked_weights / gross_growth
        if row in targets:
            pending = targets[row]
        remaining = 1.0
        gross_turnover = 0.0
        trade_count = 0
        if pending is not None and observed[row, (weights > 0) | (pending > 0)].all():
            # Target weights apply to the value left after proportional costs.
            # Solve for that value using actual buy plus sell notional.
            def cost_equation(fraction):
                return fraction + friction * np.abs(fraction * pending - weights).sum() - 1

            remaining = brentq(cost_equation, 0.0, 1.0) if friction else 1.0
            traded = np.abs(remaining * pending - weights)
            gross_turnover = traded.sum()
            trade_count = np.count_nonzero(traded > 1e-10)
            weights, cash_weight, pending = pending, 0.0, None
        returns.append(gross_growth * remaining - 1)
        turnover.append(gross_turnover)
        trades.append(trade_count)
    return pd.DataFrame(
        {"return": returns, "turnover": turnover, "n_trades": trades},
        index=prices.index[first_row:],
    )


def period_metrics(daily_results, proxy_returns):
    returns = daily_results["return"].to_numpy()
    growth_curve = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(np.r_[1.0, growth_curve])[1:]
    volatility = returns.std(ddof=1)
    excess = returns - proxy_returns.reindex(daily_results.index).to_numpy()
    excess_volatility = excess.std(ddof=1)
    annualizer = np.sqrt(TRADING_DAYS)
    return {
        "cagr": float(growth_curve[-1] ** (TRADING_DAYS / len(returns)) - 1),
        "ann_vol": float(volatility * annualizer),
        "sharpe": float(returns.mean() / volatility * annualizer) if volatility > 0 else None,
        "sharpe_excess": (
            float(excess.mean() / excess_volatility * annualizer)
            if excess_volatility > 0 else None
        ),
        "max_drawdown": float((growth_curve / peaks - 1).min()),
        "total_return": float(growth_curve[-1] - 1),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def run_strategy(prices, dollar_volume, proxy_returns, start, params):
    targets = monthly_targets(prices, dollar_volume, proxy_returns, start, params)
    if not targets:
        raise ValueError("No monthly signals can execute within the requested period.")
    return simulate(prices, targets, start, params.friction)


def summarize(daily_results, proxy_returns, is_end):
    periods = {
        "is": daily_results.loc[:is_end],
        "oos": daily_results.loc[daily_results.index > is_end],
        "full": daily_results,
    }
    if any(len(period) < 2 for period in periods.values()):
        raise ValueError("Each reporting period needs at least two sessions.")
    return {name: period_metrics(period, proxy_returns) for name, period in periods.items()}


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    universe = config["universe"]
    benchmark, risk_free = config["benchmark"], config["risk_free_proxy"]
    if len(set([*universe, benchmark, risk_free])) != len(universe) + 2:
        raise ValueError("Stock, benchmark and risk-free symbols must be distinct.")
    data_start, start, is_end, end = (
        pd.Timestamp(config[name]) for name in ("data_start", "backtest_start", "is_end", "end")
    )
    if not data_start < start < is_end < end:
        raise ValueError("Expected data_start < backtest_start < is_end < end.")
    official = Parameters(**config["parameters"])
    cases = {
        name: replace(official, **overrides) for name, overrides in config["robustness"].items()
    }
    combinations = {official, *cases.values()}
    if len(combinations) != 1 + len(cases):
        raise ValueError("Each robustness case must specify a distinct parameter combination.")
    for params in combinations:
        params.validate(len(universe))

    adjusted, liquidity, fetched_at = load_panel(universe, benchmark, risk_free, data_start, end)
    stocks = adjusted[universe]
    proxy_returns = adjusted[risk_free].pct_change(fill_method=None)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance Adj Close total-return (splits and distributions)")
    print("ROBUSTNESS (IS ONLY)")
    for name, params in cases.items():
        daily_results = run_strategy(
            stocks.loc[:is_end], liquidity.loc[:is_end], proxy_returns.loc[:is_end], start, params,
        )
        metrics = period_metrics(daily_results, proxy_returns.loc[:is_end])
        print(f"  {name} {json.dumps(config['robustness'][name])} {json.dumps(metrics, allow_nan=False)}")

    daily_results = run_strategy(stocks, liquidity, proxy_returns, start, official)
    index_results = simulate(
        adjusted[[benchmark]], {adjusted.index.searchsorted(start): np.ones(1)},
        start, official.friction,
    )
    strategy_metrics = summarize(daily_results, proxy_returns, is_end)
    benchmark_metrics = summarize(index_results, proxy_returns, is_end)
    data_range = {
        "start": daily_results.index[0].date().isoformat(),
        "end": daily_results.index[-1].date().isoformat(),
    }
    print(f"Date Range: {data_range['start']} to {data_range['end']}")
    print(f"OFFICIAL_PARAMETERS: {json.dumps(asdict(official))}")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()} strategy: {json.dumps(strategy_metrics[period], allow_nan=False)}")
        print(f"{period.upper()} {benchmark}: {json.dumps(benchmark_metrics[period], allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {len(combinations)}")
    results = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **strategy_metrics,
        "benchmark": {"ticker": benchmark, **benchmark_metrics},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)
