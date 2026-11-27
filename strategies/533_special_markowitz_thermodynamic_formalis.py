# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 533_special_markowitz_thermodynamic_formalis\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nbenchmark: SPY\nrisk_free: BIL\n\nhistory_start: '2007-06-01'\nevaluation_start: '2009-01-01'\nis_end: '2016-12-31'\nevaluation_end: '2025-12-31'\nsessions_per_year: 252\nexecution_lag_sessions: 1\ndownload_timeout_seconds: 12\n\ndefault:\n  lookback_sessions: 252\n  persistence_floor: 0.05\n  risk_aversion: 5.0\n  max_weight: 0.35\n  cost_bps: 2.0\n  slippage_bps: 3.0\n\nrobustness:\n  shorter_lookback: {lookback_sessions: 210}\n  longer_lookback: {lookback_sessions: 294}\n  lower_persistence_floor: {persistence_floor: 0.025}\n  higher_persistence_floor: {persistence_floor: 0.10}\n  lower_risk_aversion: {risk_aversion: 4.0}\n  higher_risk_aversion: {risk_aversion: 6.0}\n  doubled_friction: {cost_bps: 4.0, slippage_bps: 6.0}\n"
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
import gzip
import hashlib
import json
import re
import signal
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, minimize


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
PRICE_ADJUSTMENT = "yfinance auto_adjust=True, dividend- and split-adjusted close (total-return proxy)"


@dataclass(frozen=True)
class Model:
    lookback_sessions: int
    persistence_floor: float
    risk_aversion: float
    max_weight: float
    cost_bps: float
    slippage_bps: float

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000

    def validate(self, n_assets):
        if type(self.lookback_sessions) is not int or self.lookback_sessions <= n_assets:
            raise ValueError("lookback_sessions must be an integer greater than the universe size")
        if not np.isfinite([self.persistence_floor, self.risk_aversion, self.max_weight,
                            self.cost_bps, self.slippage_bps]).all():
            raise ValueError("Model parameters must be finite")
        if not (0 < self.persistence_floor <= 1 and self.risk_aversion > 0):
            raise ValueError("Use a persistence floor in (0, 1] and positive risk aversion")
        if not 1 / n_assets <= self.max_weight <= 1:
            raise ValueError("max_weight must permit a fully invested long-only portfolio")
        if min(self.cost_bps, self.slippage_bps) < 0 or self.friction >= 0.5:
            raise ValueError("Costs and slippage must be nonnegative and sum to less than 5000 bps")


def check_prices(close, start, end):
    if (close.empty or not close.index.is_unique or not close.index.is_monotonic_increasing
            or close.index.hasnans):
        raise ValueError("Price dates must be present, unique and in ascending order")
    if not (start <= close.index[0] <= start + pd.Timedelta(days=7)
            and end - pd.Timedelta(days=7) <= close.index[-1] <= end):
        raise ValueError(f"Incomplete price date range ({close.index[0].date()} to {close.index[-1].date()})")
    values = close.to_numpy(dtype=float)
    bad = ~np.isfinite(values) | (values <= 0)
    if bad.any():
        row, col = np.argwhere(bad)[0]
        raise ValueError(f"Missing or invalid close for {close.columns[col]} on {close.index[row].date()}")


def load_prices(tickers, start, end, timeout):
    CACHE_DIR.mkdir(exist_ok=True)
    request = json.dumps([sorted(tickers), start.date().isoformat(), end.date().isoformat(), PRICE_ADJUSTMENT])
    cache_path = CACHE_DIR / f"closes_{hashlib.sha256(request.encode()).hexdigest()[:20]}.csv.gz"

    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as saved:
            fetched_at = saved.readline().strip()
            close = pd.read_csv(saved, index_col="Date", parse_dates=["Date"])
        datetime.fromisoformat(fetched_at)
        close = close.reindex(columns=tickers)
        check_prices(close, start, end)
        return close, fetched_at

    with TemporaryDirectory(prefix="yf_", dir=CACHE_DIR) as client_cache:
        yf.set_tz_cache_location(client_cache)
        bars = yf.download(
            tickers, start=start.date().isoformat(), end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True, interval="1d", progress=False, threads=False, timeout=timeout,
        )
    if bars.empty:
        raise RuntimeError("yfinance returned no daily bars")
    close = bars["Close"].reindex(columns=tickers).dropna(how="all")
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).rename("Date")
    check_prices(close, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with NamedTemporaryFile(dir=CACHE_DIR, delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        with gzip.open(temp_path, "wt", encoding="utf-8") as saved:
            saved.write(fetched_at + "\n")
            close.to_csv(saved)
        temp_path.replace(cache_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return close, fetched_at


def spectral_persistence(eigenvalues, concentration, floor):
    # Reinhardt, Remark 4.8. Invert the upper-spike map and estimate squared overlap.
    edge = (1 + np.sqrt(concentration)) ** 2
    upper = eigenvalues > edge
    distance = eigenvalues[upper] - 1 - concentration
    discriminant = np.maximum(distance**2 - 4 * concentration, 0)
    spike = (distance + np.sqrt(discriminant)) / 2
    overlap = (1 - concentration / spike**2) / (1 + concentration / spike)
    persistence = np.full_like(eigenvalues, floor)
    persistence[upper] = np.clip(overlap, floor, 1)
    return persistence


def regularized_moments(window, model, sessions_per_year):
    n_obs, n_assets = window.shape
    sample_mean = window.mean(axis=0) * sessions_per_year
    sample_cov = np.cov(window, rowvar=False, ddof=1) * sessions_per_year
    ref_variance = sample_cov.diagonal()
    if np.any(ref_variance <= 0):
        raise ValueError("A sector has zero variance in the estimation window")
    ref_scale = np.sqrt(ref_variance)
    relative_cov = sample_cov / np.outer(ref_scale, ref_scale)
    eigenvalues, modes = np.linalg.eigh(relative_cov)
    persistence = spectral_persistence(eigenvalues, n_assets / n_obs, model.persistence_floor)

    # Theorem 7.1 uses the same persistence for each whitened deviation.
    excess_mean = sample_mean - sample_mean.mean()
    whitened_mean = modes.T @ (excess_mean / ref_scale)
    regularized_excess = ref_scale * (modes @ (persistence * whitened_mean))
    regularized_eigenvalues = 1 + persistence * (eigenvalues - 1)
    if np.any(regularized_eigenvalues <= 0):
        raise ValueError("The regularized relative covariance must be positive definite")
    unwhitened_modes = ref_scale[:, None] * modes
    regularized_cov = (unwhitened_modes * regularized_eigenvalues) @ unwhitened_modes.T
    return regularized_excess, regularized_cov


def target_weights(window, model, sessions_per_year):
    excess_mean, covariance = regularized_moments(window, model, sessions_per_year)
    equal_weights = np.full(window.shape[1], 1 / window.shape[1])
    risk = model.risk_aversion * covariance

    def loss(weights):
        active = weights - equal_weights
        return float(active @ (0.5 * (risk @ active) - excess_mean))

    def gradient(weights):
        return risk @ (weights - equal_weights) - excess_mean

    optimum = minimize(
        loss, equal_weights, jac=gradient, method="SLSQP",
        bounds=[(0, model.max_weight)] * len(equal_weights),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
        options={"ftol": 1e-12, "maxiter": 200},
    )
    weights = optimum.x
    if (not optimum.success or not np.isfinite(weights).all() or abs(weights.sum() - 1) > 1e-9
            or weights.min() < -1e-9 or weights.max() > model.max_weight + 1e-9):
        raise RuntimeError(f"Weight optimization failed: {optimum.message}")
    weights = np.clip(weights, 0, model.max_weight)
    return weights / weights.sum()


def schedule_targets(sector_returns, model, evaluation_start, lag, sessions_per_year):
    months = sector_returns.index.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])
    asset_returns = sector_returns.to_numpy()
    targets = {}
    for signal_row in month_ends:
        fill_row = signal_row + lag
        if fill_row >= len(months) or sector_returns.index[fill_row] < evaluation_start:
            continue
        first_row = signal_row + 1 - model.lookback_sessions
        signal_day = sector_returns.index[signal_row].date()
        if first_row < 0:
            raise ValueError(f"Insufficient history for {signal_day}: move history_start earlier")
        try:
            targets[sector_returns.index[fill_row]] = target_weights(
                asset_returns[first_row:signal_row + 1], model, sessions_per_year,
            )
        except (ValueError, RuntimeError) as error:
            raise RuntimeError(f"Cannot compute weights for {signal_day}: {error}") from error
    if not targets:
        raise ValueError("The evaluated period contains no scheduled fills")
    return targets


def simulate(asset_returns, cash_returns, targets, friction):
    weights = np.zeros(asset_returns.shape[1])
    daily = []
    for day, asset_return, cash_return in zip(asset_returns.index, asset_returns.to_numpy(),
                                             cash_returns.to_numpy(), strict=True):
        # Mark the previous weights through today's close, then fill today's targets.
        asset_values = weights * (1 + asset_return)
        growth = asset_values.sum() + max(0.0, 1 - weights.sum()) * (1 + cash_return)
        weights = asset_values / growth
        turnover = 0.0
        n_trades = 0
        target = targets.get(day)
        if target is not None:
            # Fraction left after paying fees on actual buy and sell notionals.
            def fee_equation(remaining):
                traded = np.abs(remaining * target - weights).sum()
                return remaining + friction * traded - 1

            remaining = brentq(fee_equation, 0, 1, xtol=1e-14)
            traded = np.abs(remaining * target - weights)
            turnover = traded.sum()
            n_trades = np.count_nonzero(traded > 1e-8)
            growth *= remaining
            weights = target
        daily.append((growth - 1, turnover, n_trades))

    return pd.DataFrame(daily, index=asset_returns.index, columns=["net_return", "turnover", "n_trades"])


def assess(daily, cash_returns, sessions_per_year):
    if len(daily) < 2:
        raise ValueError("Each reported period needs at least two return observations")
    net = daily["net_return"]
    excess = net - cash_returns.loc[net.index]
    growth = np.concatenate(([1.0], (1 + net.to_numpy()).cumprod()))
    scale = np.sqrt(sessions_per_year)
    daily_vol = net.std(ddof=1)
    daily_excess_vol = excess.std(ddof=1)
    return {
        "cagr": float(growth[-1] ** (sessions_per_year / len(net)) - 1),
        "ann_vol": float(daily_vol * scale),
        "sharpe": float(net.mean() * scale / daily_vol) if daily_vol > 0 else None,
        "sharpe_excess": float(excess.mean() * scale / daily_excess_vol) if daily_excess_vol > 0 else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily["turnover"].sum()),
        "n_trades": int(daily["n_trades"].sum()),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    universe = config["universe"]
    benchmark, risk_free = config["benchmark"], config["risk_free"]
    tickers = [*universe, benchmark, risk_free]
    if (len(universe) < 2 or len(set(tickers)) != len(tickers)
            or any(re.fullmatch(r"[A-Z]{1,5}", ticker) is None for ticker in tickers)):
        raise ValueError("Use distinct U.S. ETF symbols for the universe, benchmark and cash proxy")
    start, evaluation_start, is_end, end = (
        pd.Timestamp(config[field]) for field in ("history_start", "evaluation_start", "is_end", "evaluation_end")
    )
    if not start < evaluation_start < is_end < end:
        raise ValueError("Require history_start < evaluation_start < is_end < evaluation_end")
    lag, sessions_per_year = config["execution_lag_sessions"], config["sessions_per_year"]
    if type(lag) is not int or lag < 1 or type(sessions_per_year) is not int or sessions_per_year < 1:
        raise ValueError("Lag and annualization must be positive integers")
    timeout = config["download_timeout_seconds"]
    if not 0 < timeout <= 30:
        raise ValueError("Set the download timeout in (0, 30] seconds")

    official = Model(**config["default"])
    variants = {name: replace(official, **changes) for name, changes in config["robustness"].items()}
    combinations = {official, *variants.values()}
    if len(combinations) != 1 + len(variants):
        raise ValueError("Default and robustness model combinations must be unique")
    for model in combinations:
        model.validate(len(universe))

    prices, fetched_at = load_prices(tickers, start, end, timeout)
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"DATA_HISTORY: {prices.index[0].date()} to {prices.index[-1].date()}")
    returns = prices.pct_change(fill_method=None).iloc[1:]
    evaluation = returns.loc[evaluation_start:]
    cash = evaluation[risk_free]

    # ROBUSTNESS (IS only). Supply only the prefix through the IS cutoff.
    is_returns = returns.loc[:is_end]
    is_evaluation = is_returns.loc[evaluation_start:]
    print(f"ROBUSTNESS (IS only through {is_end.date()})")
    for name, model in variants.items():
        targets = schedule_targets(is_returns[universe], model, evaluation_start, lag, sessions_per_year)
        daily = simulate(is_evaluation[universe], is_evaluation[risk_free], targets, model.friction)
        metrics = assess(daily, is_evaluation[risk_free], sessions_per_year)
        print(f"  {name}: {json.dumps(metrics, allow_nan=False)}")

    targets = schedule_targets(returns[universe], official, evaluation_start, lag, sessions_per_year)
    daily_results = simulate(evaluation[universe], cash, targets, official.friction)
    benchmark_targets = {next(iter(targets)): np.ones(1)}
    benchmark_results = simulate(evaluation[[benchmark]], cash, benchmark_targets, official.friction)
    periods = {"is": evaluation.index <= is_end, "oos": evaluation.index > is_end, "full": slice(None)}
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": evaluation.index[0].date().isoformat(),
                       "end": evaluation.index[-1].date().isoformat()},
        "benchmark": {"ticker": benchmark},
    }
    for period, date_mask in periods.items():
        result[period] = assess(daily_results.loc[date_mask], cash, sessions_per_year)
        result["benchmark"][period] = assess(benchmark_results.loc[date_mask], cash, sessions_per_year)
        print(f"{period.upper()} strategy: {json.dumps(result[period], allow_nan=False)}")
        print(f"{period.upper()} benchmark {benchmark}: {json.dumps(result['benchmark'][period], allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {len(combinations)}")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


def stop_on_deadline(_signal_number, _frame):
    raise SystemExit("Backtest stopped at the 14-minute runtime limit")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_on_deadline)
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)
