# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 672_randomized_signature_methods_in_optimal\ndata:\n  universe: [SPY, IWM, EFA, EEM, TLT, IEF, GLD, VNQ, DBC]\n  risk_free: BIL\n  start: 2008-01-02\n  end: 2025-12-31\nsample:\n  evaluation_start: 2011-01-01\n  is_end: 2023-12-31\nexecution:\n  lag_sessions: 1\n  cost_bps: 2.0\n  slippage_bps: 3.0\nmodel:\n  path_days: 22\n  horizon_days: 21\n  covariance_days: 252\n  min_train_rows: 500\n  reservoir_size: 70\n  matrix_mean: 0.0\n  matrix_variance: 0.03\n  seeds: [1729, 1730, 1731]\n  return_unit: 0.01\n  ridge_alpha: 0.001\n  risk_aversion: 5.0\n  max_weight: 0.20\nrobustness:\n  path_days: [18, 26]\n  matrix_variance: [0.01, 0.05]\n  trading_cost_multiplier: 2.0\n'
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
"""Randomized-signature forecasts for a monthly cross-asset ETF portfolio.

Paper: https://arxiv.org/abs/2312.16448
Research references (ideas only; implementation written here):
https://github.com/janka-moeller/Sig-SPT (no explicit license)
https://github.com/ThotDjehuty/optimiz-rs (MIT)
https://docs.rs/optimiz-rs/latest/src/optimizr/signatures/random_signature.rs.html
https://pypi.org/project/pysiglib/ (Apache-2.0; background)
https://github.com/pdebus/randomized_signatures (unavailable)
"""

import hashlib
import json
import signal
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "data_cache"
SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class Market:
    dates: pd.DatetimeIndex
    opens: np.ndarray
    closes: np.ndarray
    bil_returns: np.ndarray

    def through(self, date):
        stop = self.dates.searchsorted(date, side="right")
        return Market(
            self.dates[:stop], self.opens[:stop], self.closes[:stop], self.bil_returns[:stop]
        )


def check_bars(dates, opens, closes, tickers, start, end):
    if not dates.is_unique or not dates.is_monotonic_increasing or len(dates) < 2:
        raise ValueError("Expected ordered, unique daily sessions")
    if not (start <= dates[0] <= start + pd.Timedelta(days=7)) or not (
        end - pd.Timedelta(days=7) <= dates[-1] <= end
    ):
        raise ValueError(f"Incomplete date range from Yahoo: {dates[0]} to {dates[-1]}")
    for values in (opens, closes):
        if values.shape != (len(dates), len(tickers)):
            raise ValueError("ETF price columns do not match the requested universe")
        invalid = np.argwhere(~np.isfinite(values) | (values <= 0.0))
        if len(invalid):
            day, ticker = invalid[0]
            raise ValueError(f"Missing or invalid bar for {tickers[ticker]} on {dates[day].date()}")


def price_units(opens, closes):
    # Common factors on historical bars cancel against the first opening price.
    first_open = opens[0]
    return 100.0 * (opens / first_open), 100.0 * (closes / first_open)


def load_market(data):
    tickers = [*data["universe"], data["risk_free"]]
    if len(set(tickers)) != len(tickers) or data["risk_free"] != "BIL":
        raise ValueError("ETF tickers must be distinct and the excess-return proxy must be BIL")
    start, end = pd.Timestamp(data["start"]), pd.Timestamp(data["end"])
    cache_key = f"yfinance-adjusted-v1:{','.join(tickers)}:{start.date()}:{end.date()}"
    cache_id = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"etf_daily_{cache_id}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as cached:
            dates = pd.DatetimeIndex(cached["dates"])
            opens, closes = cached["opens"], cached["closes"]
            fetched_at = cached["fetched_at"].item()
        check_bars(dates, opens, closes, tickers, start, end)
    else:
        # Keep only market bars. The public HTTP client's transient state stays temporary.
        with TemporaryDirectory(prefix=".yahoo-", dir=CACHE_DIR) as transient:
            yf.set_tz_cache_location(transient)
            bars = yf.download(
                tickers,
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=15,
            )
        dates = pd.DatetimeIndex(bars.index).tz_localize(None)
        opens = bars["Open"].reindex(columns=tickers).to_numpy(dtype=float)
        closes = bars["Close"].reindex(columns=tickers).to_numpy(dtype=float)
        check_bars(dates, opens, closes, tickers, start, end)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with NamedTemporaryFile(dir=CACHE_DIR, suffix=".npz", delete=False) as temp:
            temp_path = Path(temp.name)
            try:
                np.savez_compressed(
                    temp, dates=dates.to_numpy(), opens=opens, closes=closes, fetched_at=fetched_at
                )
                temp.flush()
                temp_path.replace(path)
            finally:
                temp_path.unlink(missing_ok=True)

    # A later split or distribution can apply a common ticker-specific scale to a prefix.
    prefix_len = min(64, len(dates))
    prefix_open, prefix_close = opens[:prefix_len], closes[:prefix_len]
    factor = np.linspace(0.25, 0.95, len(tickers))
    expected = price_units(prefix_open, prefix_close)
    rescaled = price_units(prefix_open * factor, prefix_close * factor)
    np.testing.assert_allclose(rescaled, expected, rtol=1e-13)

    opens, closes = price_units(opens, closes)
    bil = closes[:, -1]
    bil_returns = np.r_[np.nan, bil[1:] / bil[:-1] - 1.0]
    return Market(dates, opens[:, :-1], closes[:, :-1], bil_returns), fetched_at


def reservoir_features(log_returns, model, seed):
    """Run independent trailing windows in parallel, keeping their terminal states."""
    days = model["path_days"]
    size = model["reservoir_size"]
    channels = log_returns.shape[1] + 1
    rng = np.random.default_rng(seed)
    matrices = rng.normal(model["matrix_mean"], np.sqrt(model["matrix_variance"]),
                          size=(channels, size, size))
    biases = rng.normal(size=(channels, 1, size))
    initial = rng.normal(size=size)
    window_count = len(log_returns) - days + 1
    states = np.tile(initial, (window_count, 1))
    for offset in range(days):
        increments = log_returns[offset:offset + window_count]
        activated = np.tanh(states @ matrices + biases)
        states += activated[0] / days + np.einsum("wa,awr->wr", increments, activated[1:])
    # If C[j] is the jth close, this row ends on C[days + j], inclusive.
    return states


def expected_returns(market, features, signal_day, model, lag):
    path_days, horizon = model["path_days"], model["horizon_days"]
    last_label_start = signal_day - horizon
    label_starts = np.arange(path_days + lag, last_label_start + 1)
    if len(label_starts) < model["min_train_rows"]:
        raise ValueError(f"Insufficient mature labels at {market.dates[signal_day].date()}")
    label_ends = label_starts + horizon
    if label_ends[-1] > signal_day:
        raise RuntimeError("Readout label extends beyond the signal close")
    labels = market.opens[label_ends] / market.opens[label_starts] - 1.0
    forecast = []
    for states in features:
        train = states[label_starts - lag - path_days]
        mean = train.mean(axis=0)
        scale = np.maximum(train.std(axis=0), 1e-8)
        ridge = Ridge(alpha=model["ridge_alpha"] * len(train), solver="cholesky")
        ridge.fit((train - mean) / scale, labels)
        signal_state = states[signal_day - path_days]
        forecast.append(ridge.predict(((signal_state - mean) / scale)[None, :])[0])
    return np.mean(forecast, axis=0)


def optimal_weights(mu, covariance, model):
    cap = model["max_weight"]
    risk_aversion = model["risk_aversion"]

    def loss(weights):
        return 100.0 * (0.5 * risk_aversion * weights @ covariance @ weights - mu @ weights)

    def gradient(weights):
        return 100.0 * (risk_aversion * (covariance @ weights) - mu)

    result = minimize(
        loss, np.full(len(mu), 1.0 / len(mu)), jac=gradient, method="SLSQP",
        bounds=[(0.0, cap)] * len(mu),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0,
                     "jac": lambda w: np.ones_like(w)},
        options={"ftol": 1e-11, "maxiter": 100},
    )
    weights = result.x
    if (not result.success or not np.isfinite(weights).all() or
            abs(weights.sum() - 1.0) > 1e-8 or weights.min() < -1e-8 or
            weights.max() > cap + 1e-8):
        raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    weights = np.clip(weights, 0.0, cap)
    return weights / weights.sum()


def monthly_fills(market, evaluation_start):
    months = market.dates.to_period("M")
    first_session = np.r_[False, months[1:] != months[:-1]]
    return np.flatnonzero(first_session & (market.dates >= evaluation_start))


def form_targets(market, cfg, fills):
    model = cfg["model"]
    lag = cfg["execution"]["lag_sessions"]
    cov_days, path_days = model["covariance_days"], model["path_days"]
    close_ratios = market.closes[1:] / market.closes[:-1]
    log_returns = np.log(close_ratios) / (model["return_unit"] * np.sqrt(path_days))
    features = [reservoir_features(log_returns, model, seed) for seed in model["seeds"]]
    targets = {}
    for fill_day in fills:
        signal_day = fill_day - lag
        if signal_day < max(cov_days, path_days) or market.dates[signal_day] >= market.dates[fill_day]:
            raise ValueError("A scheduled opening lacks a strictly earlier, sufficiently long signal history")
        mu = expected_returns(market, features, signal_day, model, lag)
        trailing_returns = close_ratios[signal_day - cov_days:signal_day] - 1.0
        covariance = LedoitWolf().fit(trailing_returns).covariance_ * model["horizon_days"]
        targets[fill_day] = optimal_weights(mu, covariance, model)
    return targets


def rebalance_at_open(positions, cash, opening_prices, target, fee_rate):
    opening_values = positions * opening_prices
    opening_total = cash + opening_values.sum()
    low, high = 0.0, opening_total
    # Solve for the largest target investment whose buys, sells and costs self-finance.
    for _ in range(48):
        candidate = (low + high) / 2.0
        invested = candidate * target
        fee = fee_rate * np.abs(invested - opening_values).sum()
        if invested.sum() + fee <= opening_total:
            low = candidate
        else:
            high = candidate
    invested = low * target
    traded = np.abs(invested - opening_values)
    cash = opening_total - (invested.sum() + fee_rate * traded.sum())
    if cash < 0.0:
        raise RuntimeError("Opening rebalance exceeded available cash")
    trades = int(np.count_nonzero(traded > 1e-10 * opening_total))
    return invested / opening_prices, cash, traded.sum() / opening_total, trades


def simulate(market, evaluation_start, targets, execution):
    fee_rate = (execution["cost_bps"] + execution["slippage_bps"]) / 10_000.0
    if not 0.0 <= fee_rate < 1.0:
        raise ValueError("Total one-way friction must be between 0 and 10,000 bps")
    start = market.dates.searchsorted(evaluation_start)
    positions = np.zeros(market.opens.shape[1])
    previous_total = cash = 1.0
    daily_results = []
    for day in range(start, len(market.dates)):
        turnover, trades = 0.0, 0
        if day in targets:
            positions, cash, turnover, trades = rebalance_at_open(
                positions, cash, market.opens[day], targets[day], fee_rate
            )
        close_total = cash + positions @ market.closes[day]
        daily_results.append((close_total / previous_total - 1.0, turnover, trades))
        previous_total = close_total
    return pd.DataFrame(daily_results, index=market.dates[start:],
                        columns=["return", "turnover", "n_trades"])


def metrics(daily_results, bil_returns):
    returns = daily_results["return"].to_numpy()
    if len(returns) < 2 or not np.isfinite(returns).all() or np.any(returns <= -1.0):
        raise ValueError("Expected at least two finite daily portfolio returns above -100%")
    excess = returns - bil_returns.reindex(daily_results.index).to_numpy()
    if not np.isfinite(excess).all():
        raise ValueError("Missing BIL return within the evaluation sample")
    growth = np.r_[1.0, np.cumprod(1.0 + returns)]
    volatility = returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)
    excess_volatility = excess.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1.0),
        "ann_vol": float(volatility),
        "sharpe": float(returns.mean() * SESSIONS_PER_YEAR / volatility) if volatility else None,
        "sharpe_excess": (float(excess.mean() * SESSIONS_PER_YEAR / excess_volatility)
                          if excess_volatility else None),
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1.0)),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def report(daily_results, is_end, bil_returns):
    return {
        "is": metrics(daily_results.loc[:is_end], bil_returns),
        "oos": metrics(daily_results.loc[daily_results.index > is_end], bil_returns),
        "full": metrics(daily_results, bil_returns),
    }


def print_summary(label, values):
    print(f"{label}: {json.dumps(values, allow_nan=False)}", flush=True)


def run_robustness(sample, cfg, evaluation_start, bil_returns):
    fills = monthly_fills(sample, evaluation_start)
    default_targets = form_targets(sample, cfg, fills)
    default_daily = simulate(sample, evaluation_start, default_targets, cfg["execution"])
    print("ROBUSTNESS (IS only)", flush=True)
    print_summary("default IS", metrics(default_daily, bil_returns))
    trials = 1
    for field in ("path_days", "matrix_variance"):
        for value in cfg["robustness"][field]:
            variant = {**cfg, "model": {**cfg["model"], field: value}}
            targets = form_targets(sample, variant, fills)
            daily_results = simulate(sample, evaluation_start, targets, cfg["execution"])
            print_summary(f"IS {field}={value}", metrics(daily_results, bil_returns))
            trials += 1
    multiplier = cfg["robustness"]["trading_cost_multiplier"]
    execution = cfg["execution"].copy()
    for field in ("cost_bps", "slippage_bps"):
        execution[field] *= multiplier
    stressed = simulate(sample, evaluation_start, default_targets, execution)
    print_summary(f"IS both cost components x{multiplier:g}", metrics(stressed, bil_returns))
    return default_daily, trials + 1


def main():
    cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    execution, model = cfg["execution"], cfg["model"]
    size = len(cfg["data"]["universe"])
    lag = execution["lag_sessions"]
    if not isinstance(lag, int) or isinstance(lag, bool) or lag < 1:
        raise ValueError("Execution lag must be a positive integer number of sessions")
    for field in ("cost_bps", "slippage_bps"):
        if execution[field] < 0:
            raise ValueError(f"{field} must be nonnegative")
    if not 1.0 / size <= model["max_weight"] <= 1.0:
        raise ValueError("Asset weight cap is infeasible for this universe")
    start = pd.Timestamp(cfg["sample"]["evaluation_start"])
    is_end = pd.Timestamp(cfg["sample"]["is_end"])
    market, fetched_at = load_market(cfg["data"])
    if not market.dates[0] < start < is_end < market.dates[-1]:
        raise ValueError("Expected a warm-up followed by ordered IS and OOS windows")
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print(f"PRICE_DATA_RANGE: {market.dates[0].date()} to {market.dates[-1].date()}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return OHLC, normalized at first open", flush=True)
    bil_returns = pd.Series(market.bil_returns, index=market.dates)

    with threadpool_limits(limits=1, user_api="blas"):
        default_is, trials = run_robustness(market.through(is_end), cfg, start, bil_returns.loc[:is_end])
        fills = monthly_fills(market, start)
        targets = form_targets(market, cfg, fills)
        official = simulate(market, start, targets, execution)
    pd.testing.assert_frame_equal(official.loc[:is_end], default_is, rtol=1e-10, atol=1e-12)
    equal_weights = np.full(size, 1.0 / size)
    benchmark = simulate(market, start, dict.fromkeys(fills, equal_weights), execution)
    strategy_periods = report(official, is_end, bil_returns)
    benchmark_periods = report(benchmark, is_end, bil_returns)
    if any(values["n_trades"] == 0 for values in strategy_periods.values()):
        raise RuntimeError("Expected actual simulated executions in each strategy period")
    print(f"PARAM_COMBOS_TRIED: {trials}", flush=True)
    print("LOOKAHEAD_CHECK: signal close precedes fill open by the configured lag; "
          "labels mature by the signal close; ETF bar inputs require no separate release calendar; "
          "share sizing uses opening prices and cash stays nonnegative; "
          "split/dividend prefix-scale invariance and IS-only replay matched", flush=True)
    for period in ("is", "oos", "full"):
        print_summary(f"STRATEGY {period.upper()}", strategy_periods[period])
        print_summary(f"BENCHMARK Equal Weight {period.upper()}", benchmark_periods[period])
    print("STATUS: 已回测", flush=True)
    print_summary("RESULTS_JSON", {
        "strategy": cfg["strategy"],
        "data_range": {"start": official.index[0].date().isoformat(),
                       "end": official.index[-1].date().isoformat()},
        **strategy_periods,
        "benchmark": benchmark_periods["full"],
    })


if __name__ == "__main__":
    signal.alarm(840)
    try:
        main()
    finally:
        signal.alarm(0)
