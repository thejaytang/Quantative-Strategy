"""454_semivariance_and_semivolatility_in_portf
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/454_semivariance_and_semivolatility_in_portf.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 6df04a4d89c55b7a1b162dbdcff0dafc1e2bd682 on 2026-11-27 08:11 Europe/Oslo, exit 0, output sha256 b415039b8f91f26ccf52df805365afebc45af4efb1c3141ff38a874f28c97cac
Data source: Yahoo Finance free daily ETF prices via yfinance for XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV and XLY. Same-day BIL returns are used only to calculate the excess Sharpe ratio. The loader uses the local price cache when available.
Data fetched: 2026-11-27T07:11:31+00:00
Sample period: Price request from 2008-03-01 through 2026-11-06 inclusive. The run evaluated 2010-03-04 through 2026-11-06 after a 504-session estimation window and the execution lag.   data_range (from output): {'start': '2010-03-04', 'end': '2026-11-06'}
IS/OOS split: The cutoff is 2017-12-31. Evaluated IS runs from 2010-03-04 through 2017-12-29, and OOS runs from 2018-01-02 through 2026-11-06. The code simulates continuously and then slices the daily results. Both periods use the same fixed settings. The README describes OOS as a chronological holdout that predates publication of the cited paper.
Costs and slippage: Each simulated buy and sell incurs 50 bps of transaction cost and 5 bps of fixed slippage, for 55 bps applied to absolute executed notional. The simulator solves for post-cost wealth using the actual simulated position changes and deducts the friction from execution-day NAV, including at initial entry.
Signal lag: A signal computed using prices through the close of session t is filled at the next session's close. The close-to-close return ending at execution is applied to the prior holdings, and the new target starts earning returns over the following close-to-close interval.
Price adjustment: yfinance auto_adjust=True total-return adjusted close
Survivorship bias: The fixed set of early sector ETFs uses funds with data still available. The README identifies survivorship bias and retrospective universe selection risk.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.12071893648488552, ann_vol=0.11195555609486672, sharpe=1.07425136638517, sharpe_excess=1.0678815758467421, max_drawdown=-0.11924589004606534, total_return=1.4396717564584973, turnover=8.850114519745436, n_trades=334
    oos: cagr=0.0706192261495795, ann_vol=0.1536540121937392, sharpe=0.5212043628167224, sharpe_excess=0.35580765176895546, max_drawdown=-0.34134355332982247, total_return=0.8266635850538124, turnover=10.442525810413375, n_trades=492
    full: cagr=0.09387382894578122, ann_vol=0.13565762902257691, sharpe=0.7295249752825032, sharpe_excess=0.6279179852129508, max_drawdown=-0.341343553329822, total_return=3.4564595570069967, turnover=19.29264033015881, n_trades=826
    benchmark  full: cagr=0.12729590749791142, ann_vol=0.1631597659619706, sharpe=0.8164206449505296, sharpe_excess=0.7318205653018149, max_drawdown=-0.3672100734441718, total_return=6.356637219803166, turnover=5.8581919437023355, n_trades=1800
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 454_semivariance_and_semivolatility_in_portf\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nrisk_free: BIL\ndata_start: 2008-03-01\ndata_end: 2026-11-06\nis_end: 2017-12-31\nlookback_sessions: 504\nrebalance_sessions: 21\nexecution_lag_sessions: 1\ntransaction_cost_bps: 50\nslippage_bps: 5\nsessions_per_year: 252\n'
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
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_ADJUSTMENT = "yfinance auto_adjust=True total-return adjusted close"
WEIGHT_TOLERANCE = 1e-10


def validate_prices(prices, symbols, start, end):
    prices = prices.sort_index().reindex(columns=symbols).loc[start:end]
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("Empty price panel or duplicate price dates.")
    if (prices.index[0] - start).days > 7 or (end - prices.index[-1]).days > 7:
        raise ValueError("The price panel does not cover the requested date range.")
    valid = np.isfinite(prices) & prices.gt(0)
    if not valid.all().all():
        first_bad_date = valid.index[~valid.all(axis=1)][0]
        missing = valid.columns[~valid.loc[first_bad_date]].tolist()
        raise ValueError(f"Missing or invalid prices on {first_bad_date.date()} for {missing}.")
    return prices


def load_prices(symbols, start, end):
    key = json.dumps([PRICE_ADJUSTMENT, sorted(symbols), str(start.date()), str(end.date())])
    cache_file = CACHE / f"close_{hashlib.sha256(key.encode()).hexdigest()[:16]}.csv"
    if cache_file.exists():
        with cache_file.open(encoding="utf-8") as handle:
            prefix, _, fetched_at = handle.readline().strip().partition("=")
            if prefix != "# fetched_at" or datetime.fromisoformat(fetched_at).utcoffset() is None:
                raise ValueError(f"Invalid source timestamp in {cache_file.name}.")
            prices = pd.read_csv(handle, index_col="Date", parse_dates=["Date"])
        return validate_prices(prices, symbols, start, end), fetched_at

    yf_cache = CACHE / "yfinance"
    yf_cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(yf_cache))
    downloaded = yf.download(
        symbols,
        start=start.date().isoformat(),
        end=(end + pd.Timedelta(days=1)).date().isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=4,
        timeout=15,
    )
    if downloaded.empty or "Close" not in downloaded:
        raise RuntimeError("yfinance returned no adjusted closing prices.")
    prices = validate_prices(downloaded["Close"], symbols, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as handle:
        temp_path = Path(handle.name)
        try:
            handle.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(handle, index_label="Date")
            handle.close()
            temp_path.replace(cache_file)
        finally:
            temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def minimum_semivariance(window, equal_weights):
    # Percent units improve numerical conditioning without changing the minimizer.
    scenarios = 100.0 * window

    def objective(weights):
        downside = np.minimum(scenarios @ weights, 0.0)
        risk = np.mean(downside**2)
        gradient = (2.0 / len(scenarios)) * (scenarios.T @ downside)
        return risk, gradient

    result = minimize(
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
        options={"ftol": 1e-10, "maxiter": 300},
    )
    weights = result.x
    if (not result.success or not np.isfinite(weights).all()
            or weights.min() < -WEIGHT_TOLERANCE or abs(weights.sum() - 1.0) > 1e-8):
        return equal_weights, False
    weights = np.where(weights > WEIGHT_TOLERANCE, weights, 0.0)
    return weights / weights.sum(), True


def schedule_targets(returns, lookback, cadence, lag):
    equal_weights = np.full(returns.shape[1], 1.0 / returns.shape[1])
    strategy = {}
    benchmark = {}
    fallbacks = 0
    for signal in range(lookback - 1, len(returns) - lag, cadence):
        window = returns[signal + 1 - lookback:signal + 1]
        target, success = minimum_semivariance(window, equal_weights)
        execution = signal + lag
        strategy[execution] = target
        benchmark[execution] = equal_weights
        fallbacks += int(not success)
    if not strategy:
        raise ValueError("Insufficient price history for the lookback and execution lag.")
    return strategy, benchmark, fallbacks


def simulate(returns, targets, friction):
    first = min(targets)
    weights = np.zeros(returns.shape[1])
    rows = []
    for day in range(first, len(returns)):
        marked = weights * (1.0 + returns[day])
        gross = 1.0 + weights @ returns[day]
        weights = marked / gross
        turnover, trades = 0.0, 0
        target = targets.get(day)
        if target is not None:
            # Solve for post-cost wealth using the dollar turnover of the fill.
            def budget(net):
                return net + friction * np.abs(net * target - weights).sum() - 1.0

            net = brentq(budget, 0.0, 1.0) if friction else 1.0
            trade_sizes = np.abs(net * target - weights)
            turnover = trade_sizes.sum()
            trades = np.count_nonzero(trade_sizes > WEIGHT_TOLERANCE)
            gross *= net
            weights = target
        rows.append((gross - 1.0, turnover, trades))
    return rows


def metrics(period, annualization):
    returns = period["return"].to_numpy()
    excess = returns - period["risk_free"].to_numpy()
    wealth = np.r_[1.0, np.cumprod(1.0 + returns)]

    def sharpe(series):
        volatility = series.std(ddof=1)
        return float(series.mean() / volatility * np.sqrt(annualization)) if volatility > 0 else None

    return {
        "cagr": float(wealth[-1] ** (annualization / len(returns)) - 1.0),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(annualization)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(excess),
        "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0)),
        "total_return": float(wealth[-1] - 1.0),
        "turnover": float(period["turnover"].sum()),
        "n_trades": int(period["n_trades"].sum()),
    }


def summarize(daily, is_end, annualization):
    segments = {"is": daily.loc[:is_end], "oos": daily.loc[daily.index > is_end], "full": daily}
    if any(len(period) < 2 for period in segments.values()):
        raise ValueError("The IS/OOS split must leave at least two evaluated sessions in each period.")
    return {label: metrics(period, annualization) for label, period in segments.items()}


def print_metrics(name, summary):
    for period, values in summary.items():
        formatted = []
        for metric, value in values.items():
            if value is None:
                display = "n/a"
            elif metric == "n_trades":
                display = str(value)
            elif metric.startswith("sharpe"):
                display = f"{value:.3f}"
            else:
                display = f"{value:.2%}"
            formatted.append(f"{metric}={display}")
        print(f"{name} {period.upper()} " + "  ".join(formatted))


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    risk_free = config["risk_free"]
    symbols = [*universe, risk_free]
    start, end, is_end = (pd.Timestamp(config[key]) for key in ("data_start", "data_end", "is_end"))
    lookback = config["lookback_sessions"]
    cadence = config["rebalance_sessions"]
    lag = config["execution_lag_sessions"]
    annualization = config["sessions_per_year"]
    cost, slippage = (float(config[key]) for key in ("transaction_cost_bps", "slippage_bps"))
    if not start < is_end < end or len(universe) < 2 or len(symbols) != len(set(symbols)):
        raise ValueError("Use chronological dates and distinct ETFs for the portfolio and cash reference.")
    if not all(type(value) is int and value >= 1 for value in (lookback, cadence, lag, annualization)):
        raise ValueError("Lookback, rebalance cadence, lag and annualization must be positive integers.")
    if not (cost >= 0 and slippage >= 0 and cost + slippage < 10_000):
        raise ValueError("Friction must be nonnegative and total less than 10,000 bps.")
    friction = (cost + slippage) / 10_000.0

    prices, fetched_at = load_prices(symbols, start, end)
    daily_returns = prices.pct_change(fill_method=None).iloc[1:]
    asset_returns = daily_returns[universe].to_numpy()
    strategy_targets, benchmark_targets, fallbacks = schedule_targets(asset_returns, lookback, cadence, lag)
    first_day = min(strategy_targets)
    summaries = {}
    for label, targets in (("MSV0", strategy_targets), ("EW", benchmark_targets)):
        daily = pd.DataFrame(
            simulate(asset_returns, targets, friction),
            index=daily_returns.index[first_day:],
            columns=["return", "turnover", "n_trades"],
        )
        daily["risk_free"] = daily_returns[risk_free]
        summaries[label] = summarize(daily, is_end, annualization)

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"OPTIMIZER_FALLBACKS: {fallbacks}")
    for label, summary in summaries.items():
        print_metrics(label, summary)
    result = {
        "strategy": config["strategy"],
        "data_range": {
            "start": str(daily_returns.index[first_day].date()),
            "end": str(daily_returns.index[-1].date()),
        },
        **summaries["MSV0"],
        "benchmark": summaries["EW"],
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
