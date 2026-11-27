# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 400_noxtrader_lstm_based_stock_return_moment\nuniverse: [AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, BAC, WMT, COST, PG, JNJ, UNH, XOM, CVX, HD, V, MA]\nbenchmark: SPY\nrisk_free: BIL\n\n# Dates are inclusive. Calibration is a subset of IS.\ndata_start: 2017-01-03\nstart: 2018-10-04\ncalibration_end: 2022-12-30\nis_end: 2023-10-31\nend: 2025-12-31\n\n# Bps per dollar of entry, exit, or rebalance turnover.\ncost_bps: 1.0\nslippage_bps: 5.0\nsignal_lag_sessions: 1\n\nmodel:\n  week_sessions: 5\n  month_sessions: 20\n  lookback_sessions: 10\n  training_sessions: 240\n  refit_sessions: 10\n  hidden_units: 32\n  epochs: 3\n  batch_size: 256\n  learning_rate: 0.001\n  seed: 400\n\n# Select by net excess Sharpe through calibration_end. Frozen thereafter.\nscore_quantiles: [0.70, 0.80, 0.90]\n'
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
"""Walk-forward NoxTrader research backtest using public daily data."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import yaml


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252


def load_config():
    with (ROOT / "config.yaml").open(encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)

    dates = [
        pd.Timestamp(cfg[key])
        for key in ("data_start", "start", "calibration_end", "is_end", "end")
    ]
    if any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("Expected data_start < start < calibration_end < is_end < end.")

    tickers = [*cfg["universe"], cfg["benchmark"], cfg["risk_free"]]
    if (
        len(cfg["universe"]) < 2
        or len(tickers) != len(set(tickers))
        or any(not re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z])?", ticker) for ticker in tickers)
    ):
        raise ValueError("Use distinct U.S. equity/ETF symbols for the universe and references.")
    lag = cfg["signal_lag_sessions"]
    if type(lag) is not int or lag < 1:
        raise ValueError("signal_lag_sessions must be a positive integer.")
    costs_bps = [cfg["cost_bps"], cfg["slippage_bps"]]
    if not all(np.isfinite(bps) and bps >= 0 for bps in costs_bps) or sum(costs_bps) >= 1000:
        raise ValueError("Costs and slippage must be nonnegative and sum to less than 1000 bps.")

    model = cfg["model"]
    for key in (
        "week_sessions", "month_sessions", "lookback_sessions", "training_sessions",
        "refit_sessions", "hidden_units", "epochs", "batch_size",
    ):
        if type(model[key]) is not int or model[key] < 1:
            raise ValueError(f"model.{key} must be a positive integer.")
    if not np.isfinite(model["learning_rate"]) or model["learning_rate"] <= 0:
        raise ValueError("model.learning_rate must be positive and finite.")
    quantiles = cfg["score_quantiles"]
    if (
        not quantiles
        or len(quantiles) != len(set(quantiles))
        or any(not 0 < quantile < 1 for quantile in quantiles)
    ):
        raise ValueError("score_quantiles must contain distinct values in (0, 1).")
    return cfg


def price_history(cfg):
    tickers = [*cfg["universe"], cfg["benchmark"], cfg["risk_free"]]
    start = pd.Timestamp(cfg["data_start"])
    end = pd.Timestamp(cfg["end"])
    # yfinance's end is exclusive.
    end_exclusive = (end + pd.Timedelta(days=1)).date().isoformat()
    request = json.dumps([tickers, start.date().isoformat(), end_exclusive, "auto_adjust"])
    key = hashlib.sha256(request.encode()).hexdigest()[:16]
    cache_path = CACHE / f"daily_{key}.csv.gz"

    if cache_path.exists():
        quotes = pd.read_csv(cache_path, header=[0, 1], index_col=0, parse_dates=True)
        fetched_at = datetime.fromtimestamp(cache_path.stat().st_mtime, timezone.utc)
    else:
        import yfinance as yf

        CACHE.mkdir(exist_ok=True)
        yf.set_tz_cache_location(str(CACHE / "yfinance"))
        quotes = yf.download(
            tickers,
            start=start.date().isoformat(),
            end=end_exclusive,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=4,
            timeout=15,
        )
        fetched_at = datetime.now(timezone.utc)

    if quotes.empty or not {"Close", "Volume"} <= set(quotes.columns.get_level_values(0)):
        raise RuntimeError("yfinance returned no usable adjusted prices and volume.")
    close = quotes["Close"].reindex(columns=tickers).sort_index()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    if close.index.has_duplicates:
        raise RuntimeError("Duplicate exchange dates in yfinance history.")
    close = close.loc[start:end]
    if (
        close.empty
        or close.index[0] - start > pd.Timedelta(days=7)
        or end - close.index[-1] > pd.Timedelta(days=7)
    ):
        raise RuntimeError("yfinance history does not cover the requested date range.")
    bad_prices = (~np.isfinite(close) | (close <= 0)).any()
    if bad_prices.any():
        missing = ", ".join(bad_prices[bad_prices].index)
        raise RuntimeError(f"Missing or invalid adjusted closes for {missing}.")
    volume = quotes["Volume"].reindex(columns=cfg["universe"]).sort_index()
    volume.index = pd.DatetimeIndex(volume.index).tz_localize(None).normalize()
    volume = volume.reindex(close.index)
    bad_volume = (~np.isfinite(volume) | (volume <= 0)).all()
    if bad_volume.any():
        missing = ", ".join(bad_volume[bad_volume].index)
        raise RuntimeError(f"No usable volume history for {missing}.")

    if not cache_path.exists():
        with NamedTemporaryFile(dir=CACHE, suffix=".csv.gz", delete=False) as temporary:
            temp_path = Path(temporary.name)
        try:
            quotes.to_csv(temp_path, compression="gzip")
            temp_path.replace(cache_path)
        finally:
            temp_path.unlink(missing_ok=True)
    return close, volume, fetched_at.isoformat(timespec="seconds")


def make_examples(close, volume, cfg):
    model = cfg["model"]
    returns = close.pct_change(fill_method=None)
    momentum = returns.diff()
    volume = volume.where(volume > 0)
    fields = [
        returns,
        momentum,
        momentum.diff(),
        close.pct_change(model["week_sessions"], fill_method=None),
        close.pct_change(model["month_sessions"], fill_method=None),
        volume.pct_change(fill_method=None),
    ]
    features = np.stack([field.to_numpy(dtype=np.float32) for field in fields], axis=-1)
    lookback = model["lookback_sessions"]
    windows = np.lib.stride_tricks.sliding_window_view(features, lookback, axis=0)
    windows = windows.transpose(0, 1, 3, 2)
    # Signal at t, fill at t+lag, and realize the holding return at t+lag+1.
    # The label is that holding return minus the return of the prior session.
    label_delay = cfg["signal_lag_sessions"] + 1
    labels = momentum.shift(-label_delay).iloc[lookback - 1:].to_numpy(dtype=np.float32)
    return windows, labels, label_delay


class LSTM:
    """Single LSTM layer and linear readout, trained with MSE and Adam."""

    def __init__(self, n_features, hidden_units, rng):
        self.hidden_units = hidden_units
        width = n_features + hidden_units
        self.kernel = rng.standard_normal((width, 4 * hidden_units), dtype=np.float32)
        self.kernel *= 1 / np.sqrt(width)
        self.bias = np.zeros(4 * hidden_units, dtype=np.float32)
        self.bias[hidden_units:2 * hidden_units] = 1
        self.readout = rng.standard_normal((hidden_units, 1), dtype=np.float32)
        self.readout *= 1 / np.sqrt(hidden_units)
        self.intercept = np.zeros(1, dtype=np.float32)
        self.parameters = (self.kernel, self.bias, self.readout, self.intercept)

    def forward(self, x, training=False):
        hidden = np.zeros((len(x), self.hidden_units), dtype=np.float32)
        cell = np.zeros_like(hidden)
        history = []
        for observation in x.transpose(1, 0, 2):
            joined = np.concatenate((observation, hidden), axis=1)
            input_gate, forget, candidate, output_gate = np.split(
                joined @ self.kernel + self.bias, 4, axis=1,
            )
            input_gate = 1 / (1 + np.exp(-np.clip(input_gate, -30, 30)))
            forget = 1 / (1 + np.exp(-np.clip(forget, -30, 30)))
            candidate = np.tanh(candidate)
            output_gate = 1 / (1 + np.exp(-np.clip(output_gate, -30, 30)))
            previous_cell = cell
            cell = forget * previous_cell + input_gate * candidate
            tanh_cell = np.tanh(cell)
            hidden = output_gate * tanh_cell
            if training:
                history.append((joined, previous_cell, input_gate, forget,
                                candidate, output_gate, tanh_cell))
        return hidden @ self.readout + self.intercept, hidden, history

    def gradients(self, x, y):
        prediction, hidden, history = self.forward(x, training=True)
        error_gradient = 2 * (prediction - y) / len(x)
        readout_gradient = hidden.T @ error_gradient
        intercept_gradient = error_gradient.sum(axis=0)
        hidden_gradient = error_gradient @ self.readout.T
        cell_gradient = np.zeros_like(hidden)
        kernel_gradient = np.zeros_like(self.kernel)
        bias_gradient = np.zeros_like(self.bias)
        recurrent_kernel = self.kernel[-self.hidden_units:]

        for joined, previous, input_gate, forget, candidate, output, tanh_cell in reversed(history):
            output_gradient = hidden_gradient * tanh_cell
            cell_gradient += hidden_gradient * output * (1 - tanh_cell ** 2)
            input_gradient = cell_gradient * candidate
            forget_gradient = cell_gradient * previous
            candidate_gradient = cell_gradient * input_gate
            gates_gradient = np.concatenate(
                (
                    input_gradient * input_gate * (1 - input_gate),
                    forget_gradient * forget * (1 - forget),
                    candidate_gradient * (1 - candidate ** 2),
                    output_gradient * output * (1 - output),
                ),
                axis=1,
            )
            kernel_gradient += joined.T @ gates_gradient
            bias_gradient += gates_gradient.sum(axis=0)
            hidden_gradient = gates_gradient @ recurrent_kernel.T
            cell_gradient *= forget
        return kernel_gradient, bias_gradient, readout_gradient, intercept_gradient

    def fit(self, x, y, cfg, rng):
        first_moment = [np.zeros_like(param) for param in self.parameters]
        second_moment = [np.zeros_like(param) for param in self.parameters]
        step = 0
        batch_size = cfg["batch_size"]
        for _ in range(cfg["epochs"]):
            order = rng.permutation(len(x))
            for offset in range(0, len(order), batch_size):
                rows = order[offset:offset + batch_size]
                gradients = self.gradients(x[rows], y[rows])
                step += 1
                for param, gradient, first, second in zip(
                    self.parameters, gradients, first_moment, second_moment,
                ):
                    first *= 0.9
                    first += 0.1 * gradient
                    second *= 0.999
                    second += 0.001 * gradient ** 2
                    mean = first / (1 - 0.9 ** step)
                    variance = second / (1 - 0.999 ** step)
                    param -= cfg["learning_rate"] * mean / (np.sqrt(variance) + 1e-8)


def fit_window(windows, labels, valid, stop, cfg):
    start = stop - cfg["training_sessions"]
    if start < 0:
        raise ValueError("Increase history before the first execution to fit the training window.")
    usable = valid[start:stop] & np.isfinite(labels[start:stop])
    if np.count_nonzero(usable.any(axis=1)) < cfg["training_sessions"]:
        raise RuntimeError("Each training session must have at least one complete labeled sequence.")
    x = windows[start:stop][usable]
    y = labels[start:stop][usable][:, None]
    center = x.mean(axis=(0, 1))
    scale = np.maximum(x.std(axis=(0, 1)), 1e-6)
    y_center, y_scale = y.mean(), max(y.std(), 1e-6)
    rng = np.random.default_rng(cfg["seed"])
    model = LSTM(x.shape[-1], cfg["hidden_units"], rng)
    model.fit((x - center) / scale, (y - y_center) / y_scale, cfg, rng)
    return model, center, scale, y_center, y_scale


def execution_scores(windows, labels, label_delay, first, last, cfg):
    model_cfg = cfg["model"]
    lag = cfg["signal_lag_sessions"]
    # Convert execution dates into signal indices in the sequence array.
    first_signal = first - lag - (model_cfg["lookback_sessions"] - 1)
    last_signal = last - lag - (model_cfg["lookback_sessions"] - 1)
    valid = np.isfinite(windows).all(axis=(2, 3))
    scores = np.full((last - first + 1, windows.shape[1]), np.nan)
    with threadpool_limits(limits=1, user_api="blas"):
        for i, signal in enumerate(range(first_signal, last_signal + 1)):
            if i % model_cfg["refit_sessions"] == 0:
                # The latest usable label resolves at this signal's close.
                stop = signal - label_delay + 1
                model, center, scale, y_center, y_scale = fit_window(
                    windows, labels, valid, stop, model_cfg,
                )
            eligible = valid[signal]
            if eligible.any():
                x = (windows[signal, eligible] - center) / scale
                prediction, _, _ = model.forward(x)
                prediction = prediction[:, 0] * y_scale + y_center
                if not np.isfinite(prediction).all():
                    raise RuntimeError("LSTM generated a nonfinite prediction.")
                scores[i, eligible] = prediction
    return scores


def target_weights(scores, quantile):
    weights = np.zeros_like(scores)
    for i, row in enumerate(scores):
        eligible = np.isfinite(row)
        available = row[eligible]
        if len(available) < 2:
            continue
        cutoff = np.quantile(np.abs(available), quantile)
        shrunk = np.sign(available) * np.maximum(np.abs(available) - cutoff, 0)
        gross = np.abs(shrunk).sum()
        if gross > 0:
            weights[i, eligible] = shrunk / gross
    return weights


def simulate(asset_returns, targets, cost_rate, dates):
    """Mark existing holdings, then rebalance at that session's close."""
    holdings = np.zeros(targets.shape[1])
    history = []
    for daily_returns, target in zip(asset_returns, targets):
        marked = holdings * (1 + daily_returns)
        gross_equity = 1 + holdings @ daily_returns
        if gross_equity <= 0:
            raise RuntimeError("Portfolio equity exhausted before rebalancing.")

        # Fees change post-trade equity and therefore target dollar notionals.
        equity = gross_equity
        for _ in range(12):
            traded = np.abs(target * equity - marked).sum()
            equity = gross_equity - cost_rate * traded
        if equity <= 0:
            raise RuntimeError("Portfolio equity exhausted by trading costs.")
        change = target * equity - marked
        turnover = np.abs(change).sum() / gross_equity
        n_trades = np.count_nonzero(np.abs(change) > 1e-10 * gross_equity)
        history.append((equity - 1, turnover, n_trades))
        holdings = target
    return pd.DataFrame(history, index=dates, columns=["return", "turnover", "n_trades"])


def sharpe(returns):
    volatility = returns.std(ddof=1)
    if volatility == 0 or not np.isfinite(volatility):
        return None
    return float(np.sqrt(SESSIONS_PER_YEAR) * returns.mean() / volatility)


def metrics(history, risk_free):
    returns = history["return"]
    if len(returns) < 2:
        raise ValueError("Each metric period needs at least two sessions.")
    equity = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    return {
        "cagr": float(equity[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(returns - risk_free.loc[history.index]),
        "max_drawdown": float(np.min(equity / np.maximum.accumulate(equity) - 1)),
        "total_return": float(equity[-1] - 1),
        "turnover": float(history["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(history["n_trades"].sum()),
    }


def main():
    cfg = load_config()
    close, volume, fetched_at = price_history(cfg)
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return", flush=True)
    universe = cfg["universe"]
    first = close.index.searchsorted(pd.Timestamp(cfg["start"]))
    last = len(close) - 1
    dates = close.index[first:last + 1]
    is_end = pd.Timestamp(cfg["is_end"])
    calibration_stop = dates.searchsorted(pd.Timestamp(cfg["calibration_end"]), side="right")
    if calibration_stop < 2 or (dates > is_end).sum() < 2:
        raise ValueError("The data must span calibration, IS and OOS.")

    windows, labels, delay = make_examples(close[universe], volume, cfg)
    scores = execution_scores(windows, labels, delay, first, last, cfg)
    market_returns = close.pct_change(fill_method=None).loc[dates].copy()
    market_returns.iloc[0] = 0.0  # All portfolios start flat and first fill at this close.
    risk_free = market_returns[cfg["risk_free"]]
    asset_returns = market_returns[universe].to_numpy()
    cost_rate = (cfg["cost_bps"] + cfg["slippage_bps"]) / 10_000

    trials = []
    for quantile in cfg["score_quantiles"]:
        calibration_scores = scores[:calibration_stop]
        trial = simulate(
            asset_returns[:calibration_stop], target_weights(calibration_scores, quantile),
            cost_rate, dates[:calibration_stop],
        )
        excess_sharpe = sharpe(trial["return"] - risk_free.loc[trial.index])
        trials.append((excess_sharpe if excess_sharpe is not None else -np.inf, quantile))
    _, selected = max(trials, key=lambda trial: trial[0])
    print(f"PARAM_COMBOS_TRIED: {len(trials)}")
    print(f"IS_CALIBRATION_SELECTED_SCORE_QUANTILE: {selected:.2f}")

    portfolio = simulate(asset_returns, target_weights(scores, selected), cost_rate, dates)
    benchmark = simulate(
        market_returns[[cfg["benchmark"]]].to_numpy(), np.ones((len(dates), 1)), cost_rate, dates,
    )
    benchmark_name = f'{cfg["benchmark"]} buy and hold'
    result = {
        "strategy": cfg["strategy"],
        "data_range": f"{dates[0].date()} to {dates[-1].date()}",
        "is": metrics(portfolio.loc[:is_end], risk_free),
        "oos": metrics(portfolio.loc[dates > is_end], risk_free),
        "full": metrics(portfolio, risk_free),
        "benchmark": {"name": benchmark_name, **metrics(benchmark, risk_free)},
    }
    print(f'TEST_DATES: {result["data_range"]} (IS ends {is_end.date()})')
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(result[period], allow_nan=False)}")
    print(f"BENCHMARK: {json.dumps(result['benchmark'], allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
