# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 436_idiosyncratic_market_effects_in_global_t\nuniverse: [SPY, EFA, EEM, EWJ, EWU, EWG, IEF, TLT, GLD, SLV, DBC, USO, FXE, FXY]\ncash_proxy: BIL\ndata_start: '2007-06-01'\nevaluation_start: '2012-01-03'\nis_end: '2018-12-31'\nend: '2026-06-30'\ncost_bps: 2.0\nslippage_bps: 3.0\nexecution_lag_sessions: 1\nlookbacks: [21, 126, 252]\nforecast_horizons: [1, 5, 21]\nvolatility_span: 40\nfeature_clip: 10.0\nfeature_bins: 256\nkernel_bandwidth: 0.30\nbootstrap_draws: 100\nbootstrap_fraction: 0.10\nseed: 436\n"
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
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
PARAM_COMBOS_TRIED = 1


def load_prices(config: dict) -> tuple[pd.DataFrame, str]:
    symbols = [*config["universe"], config["cash_proxy"]]
    start = pd.Timestamp(config["data_start"])
    end = pd.Timestamp(config["end"])
    key = json.dumps([symbols, str(start.date()), str(end.date()), "adjusted-close-v1"])
    cache_file = CACHE / f"prices-{hashlib.sha256(key.encode()).hexdigest()[:16]}.csv"
    if cache_file.is_file():
        with cache_file.open() as handle:
            fetched_at = handle.readline().strip().removeprefix("# fetched_at=")
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(handle, index_col=0, parse_dates=True)
        return validate_prices(prices, symbols, start, end), fetched_at

    CACHE.mkdir(exist_ok=True)
    prices = pd.DataFrame()
    missing = symbols
    with TemporaryDirectory(dir=CACHE) as download_cache:
        yf.set_tz_cache_location(download_cache)
        for _ in range(2):
            bars = yf.download(
                missing, start=start.date(), end=(end + pd.Timedelta(days=1)).date(),
                interval="1d", auto_adjust=True, progress=False, timeout=15,
            )
            if not bars.empty:
                close = bars["Close"].dropna(axis=1, how="all")
                prices = prices.join(close, how="outer")
            missing = [symbol for symbol in symbols if symbol not in prices]
            if not missing:
                break

    prices = validate_prices(prices, symbols, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", dir=CACHE, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(handle)
            handle.close()
            temporary.replace(cache_file)
        finally:
            temporary.unlink(missing_ok=True)
    return prices, fetched_at


def validate_prices(
    prices: pd.DataFrame, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp,
) -> pd.DataFrame:
    missing = [symbol for symbol in symbols if symbol not in prices]
    if missing:
        raise ValueError(f"No adjusted close history for {', '.join(missing)}")
    prices = prices.loc[:, symbols].sort_index().loc[start:end]
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("Adjusted close data is empty or has duplicate sessions")
    if prices.index[0] > start + pd.Timedelta(days=7) or prices.index[-1] < end - pd.Timedelta(days=7):
        raise ValueError(f"Incomplete history from {prices.index[0].date()} to {prices.index[-1].date()}")
    invalid = ~np.isfinite(prices) | prices.le(0)
    if invalid.any().any():
        session, symbol = invalid.stack().loc[lambda values: values].index[0]
        raise ValueError(f"Invalid adjusted close for {symbol} on {session.date()}")
    return prices


def signal_inputs(excess: pd.DataFrame, config: dict) -> tuple[dict, dict, pd.DataFrame]:
    clip = config["feature_clip"]
    bins = config["feature_bins"]
    volatility = excess.ewm(span=config["volatility_span"], min_periods=config["volatility_span"]).std()
    volatility = volatility.where(volatility.gt(0))
    features = {}
    for lookback in config["lookbacks"]:
        window = excess.rolling(lookback)
        score = window.sum() / (window.std().where(lambda std: std.gt(0)) * np.sqrt(lookback))
        bucket = np.floor((score.clip(-clip, clip) + clip) * bins / (2 * clip))
        features[lookback] = bucket.clip(upper=bins - 1).to_numpy()

    lag = config["execution_lag_sessions"]
    scaled_returns = excess / volatility.shift(lag + 1)
    responses = {}
    for horizon in config["forecast_horizons"]:
        # A signal at t first earns close-to-close return ending at t + lag + 1.
        responses[horizon] = (
            scaled_returns.rolling(horizon).sum().shift(-(lag + horizon)) / np.sqrt(horizon)
        ).to_numpy()
    return features, responses, volatility


def fit_curve(
    feature: np.ndarray, response: np.ndarray, kernel: np.ndarray,
    config: dict, rng: np.random.Generator,
) -> np.ndarray:
    valid = np.isfinite(feature) & np.isfinite(response)
    feature = feature[valid].astype(int)
    response = response[valid]
    if not len(feature):
        raise ValueError("Insufficient feature and response history for model training")

    bins = config["feature_bins"]
    draws = config["bootstrap_draws"]
    sample_size = max(1, int(len(feature) * config["bootstrap_fraction"]))
    sample = rng.integers(len(feature), size=(draws, sample_size))
    sampled_bins = feature[sample]
    sampled_returns = response[sample]
    # Each bootstrap sample includes the sign-reflected observations.
    offsets = np.arange(draws)[:, None] * bins
    indices = np.concatenate((sampled_bins + offsets, bins - 1 - sampled_bins + offsets), axis=1)
    values = np.concatenate((sampled_returns, -sampled_returns), axis=1)
    counts = np.bincount(indices.ravel(), minlength=draws * bins).reshape(draws, bins)
    sums = np.bincount(indices.ravel(), values.ravel(), minlength=draws * bins).reshape(draws, bins)
    occupied = counts.any(axis=0)
    weights = kernel[:, occupied]
    denominator = np.einsum("ij,kj->ik", counts[:, occupied], weights)
    numerator = np.einsum("ij,kj->ik", sums[:, occupied], weights)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0).mean(axis=0)


def forecasts(excess: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    features, responses, volatility = signal_inputs(excess, config)
    index = excess.index
    rng = np.random.default_rng(config["seed"])
    bins = config["feature_bins"]
    bin_width = 2 * config["feature_clip"] / bins
    distance = (np.arange(bins)[:, None] - np.arange(bins)[None, :]) * bin_width
    kernel = np.exp(-0.5 * (distance / config["kernel_bandwidth"]) ** 2)
    lag = config["execution_lag_sessions"]
    is_cutoff = pd.Timestamp(config["is_end"])
    evaluation_start = pd.Timestamp(config["evaluation_start"])
    first_signal_year = index[index.searchsorted(evaluation_start) - lag].year
    cutoffs = [pd.Timestamp(year=year, month=1, day=1) for year in range(first_signal_year, is_cutoff.year + 1)]
    cutoffs.append(is_cutoff + pd.Timedelta(days=1))
    pred = {h: np.full((2, *excess.shape), np.nan) for h in responses}
    for block, cutoff in enumerate(cutoffs):
        start = index.searchsorted(cutoff)
        end = index.searchsorted(cutoffs[block + 1]) if block + 1 < len(cutoffs) else len(index)
        if start == end:
            continue
        # Use one embargo for every horizon, covering the last possible label close.
        train_end = start - lag - max(responses)
        if train_end <= 0:
            raise ValueError("Increase the training history before the first evaluation session")
        for horizon, target in responses.items():
            for feature in features.values():
                train_x = feature[:train_end]
                train_y = target[:train_end]
                pooled = fit_curve(train_x, train_y, kernel, config, rng)
                for column in range(excess.shape[1]):
                    market = fit_curve(train_x[:, column], train_y[:, column], kernel, config, rng)
                    signal = feature[start:end, column]
                    valid = np.isfinite(signal)
                    bucket = signal[valid].astype(int)
                    for kind, curve in enumerate((market, pooled)):
                        buffer = pred[horizon][kind, start:end, column]
                        previous = np.nan_to_num(buffer[valid], nan=0.0)
                        buffer[valid] = previous + curve[bucket] / len(features)

    weights = []
    for kind in range(2):
        smoothed = [
            pd.DataFrame(values[kind], index=index, columns=excess.columns).rolling(horizon).mean()
            for horizon, values in pred.items()
        ]
        exposure = (sum(smoothed) / len(smoothed)) / volatility
        gross = exposure.abs().sum(axis=1)
        weights.append(exposure.div(gross.where(gross.gt(0)), axis=0).fillna(0.0))
    return tuple(weights)


def backtest(
    returns: pd.DataFrame, cash: pd.Series, weights: pd.DataFrame, config: dict,
) -> pd.DataFrame:
    start = pd.Timestamp(config["evaluation_start"])
    lag = config["execution_lag_sessions"]
    rate = (config["cost_bps"] + config["slippage_bps"]) / 10_000
    targets = weights.shift(lag, fill_value=0.0).loc[start:]
    market_returns = returns.loc[targets.index]
    cash_returns = cash.loc[targets.index]
    positions = np.zeros(returns.shape[1])
    results = []
    for market_return, cash_return, target in zip(market_returns.to_numpy(), cash_returns, targets.to_numpy()):
        cash_weight = 1.0 - positions.sum()
        marked = positions * (1.0 + market_return)
        equity = marked.sum() + cash_weight * (1.0 + cash_return)
        if equity <= 0:
            raise ValueError("Portfolio equity became nonpositive")
        drifted = marked / equity
        # Target weights refer to equity after the one-way costs paid at this close.
        equity_fraction = 1.0
        for _ in range(12):
            traded = np.abs(target * equity_fraction - drifted)
            equity_fraction = 1.0 - rate * traded.sum()
        traded = np.abs(target * equity_fraction - drifted)
        if equity_fraction <= 0:
            raise ValueError("Execution costs exhausted portfolio equity")
        positions = target
        results.append((equity * equity_fraction - 1.0, traded.sum(), np.count_nonzero(traded > 1e-10)))
    return pd.DataFrame(results, index=targets.index, columns=["return", "turnover", "n_trades"])


def period_metrics(period: pd.DataFrame, cash: pd.Series) -> dict:
    returns = period["return"].to_numpy()
    excess = returns - cash.loc[period.index].to_numpy()
    equity = np.r_[1.0, np.cumprod(1.0 + returns)]
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    return {
        "cagr": float(equity[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1.0),
        "ann_vol": float(volatility * annualizer),
        "sharpe": float(returns.mean() * annualizer / volatility) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() * annualizer / excess_volatility) if excess_volatility > 0 else None,
        "max_drawdown": float((equity / np.maximum.accumulate(equity) - 1.0).min()),
        "total_return": float(equity[-1] - 1.0),
        "turnover": float(period["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(period["n_trades"].sum()),
    }


def summarize(results: pd.DataFrame, cash: pd.Series, is_end: pd.Timestamp) -> dict:
    periods = {"is": results.loc[results.index <= is_end], "oos": results.loc[results.index > is_end], "full": results}
    if any(len(period) < 2 for period in periods.values()):
        raise ValueError("IS and OOS must each contain at least two evaluation sessions")
    return {label: period_metrics(period, cash) for label, period in periods.items()}


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    lag = config["execution_lag_sessions"]
    if not isinstance(lag, int) or isinstance(lag, bool) or lag < 1:
        raise ValueError("execution_lag_sessions must be an integer of at least 1")
    if min(config["cost_bps"], config["slippage_bps"]) < 0 or config["cost_bps"] + config["slippage_bps"] >= 1000:
        raise ValueError("Trading costs must be nonnegative and total less than 1000 bps")
    prices, fetched_at = load_prices(config)
    start = pd.Timestamp(config["evaluation_start"])
    is_end = pd.Timestamp(config["is_end"])
    if not prices.index[lag + 1] <= start <= is_end < prices.index[-1]:
        raise ValueError("Evaluation dates must leave room for a lagged signal and an OOS period")

    returns = prices.pct_change(fill_method=None)
    cash = returns.pop(config["cash_proxy"])
    local, pooled = forecasts(returns.sub(cash, axis=0), config)
    strategy = backtest(returns, cash, local, config)
    benchmark = backtest(returns, cash, pooled, config)
    periods = summarize(strategy, cash, is_end)
    benchmark_periods = summarize(benchmark, cash, is_end)
    results = {
        "strategy": config["strategy"],
        "data_range": [str(strategy.index[0].date()), str(strategy.index[-1].date())],
        **periods,
        "benchmark": {"name": "Pooled-market nonlinear momentum", **benchmark_periods},
    }
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return close")
    print(f"PARAM_COMBOS_TRIED: {PARAM_COMBOS_TRIED}")
    print(f"PRICE_HISTORY: {prices.index[0].date()} through {prices.index[-1].date()}")
    for label, metrics in periods.items():
        print(f"{label.upper()}: {json.dumps(metrics, allow_nan=False)}")
    for label, metrics in benchmark_periods.items():
        print(f"BENCHMARK {label.upper()}: {json.dumps(metrics, allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
