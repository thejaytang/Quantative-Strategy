# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "456_ an_entropic_factor_model_for_robust_port"\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nbenchmark: SPY\nrisk_free: BIL\ndates:\n  price_start: 2017-12-01\n  start: 2019-01-01\n  is_end: 2021-12-31\n  end: 2025-12-31\nmodel:\n  lookback_sessions: 252\n  rebalance_every_sessions: 21\n  target_beta: 1.0\n  residual_buffer: 1.05\n  residual_vol_floor: 0.5\nexecution:\n  lag_sessions: 1\n  cost_bps: 10.0\n  slippage_bps: 2.0\n'
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
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq
from scipy.special import expit, logit, xlogy


HERE = Path(__file__).resolve().parent
CACHE = HERE / "data_cache"
TRADING_DAYS = 252


def load_closes(symbols, start, end):
    CACHE.mkdir(exist_ok=True)
    key = hashlib.sha256(",".join(symbols).encode()).hexdigest()[:12]
    path = CACHE / f"adjusted_close_{start.date()}_{end.date()}_{key}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as source:
            fetched_at = datetime.fromisoformat(source.readline().strip().removeprefix("# "))
            close = pd.read_csv(source, index_col=0, parse_dates=True)
    else:
        with TemporaryDirectory(dir=CACHE) as session_dir:
            yf.set_tz_cache_location(session_dir)
            raw = yf.download(
                symbols, start=start.date(), end=(end + pd.Timedelta(days=1)).date(),
                auto_adjust=True, progress=False, timeout=15, ignore_tz=True,
            )
        if raw.empty:
            raise RuntimeError("Yahoo Finance returned no prices.")
        close = raw["Close"]
        fetched_at = datetime.now(timezone.utc)

    close = close.reindex(columns=symbols).sort_index().loc[start:end]
    valid = np.isfinite(close) & close.gt(0)
    if close.empty or close.index.has_duplicates or not valid.all().all():
        bad = valid.columns[~valid.all()].tolist()
        raise ValueError(f"Incomplete or invalid adjusted closes. Check dates and symbols: {bad}")
    if close.index[-1] < end - pd.Timedelta(days=7):
        raise ValueError("Price data ends more than seven days before the requested end.")

    if not path.exists():
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as output:
            temporary_path = Path(output.name)
            try:
                output.write(f"# {fetched_at.isoformat()}\n")
                close.to_csv(output, index_label="date")
                output.flush()
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)
    return close, fetched_at


def entropy(centered_coordinates):
    probability = (1.0 + centered_coordinates) / 2.0
    return np.sum(xlogy(probability, 2.0 * probability)
                  + xlogy(1.0 - probability, 2.0 * (1.0 - probability)))


def fit_beta(asset, factor, model):
    # Paper equations 23-28. A zero factor change defines no slope.
    factor_change = np.diff(factor)
    usable = factor_change != 0.0
    if not usable.any():
        raise ValueError("Cannot estimate a loading for a constant factor.")
    slopes = np.diff(asset)[usable] / factor_change[usable]
    low_beta, high_beta = slopes.min(), slopes.max()
    lower = np.array([(asset[1:] - high_beta * factor[1:]).min(), low_beta])
    upper = np.array([(asset[1:] - low_beta * factor[1:]).max(), high_beta])
    if np.any(lower > upper):
        raise ValueError("Invalid empirical intercept bounds.")
    midpoint = (lower + upper) / 2.0
    half_width = (upper - lower) / 2.0
    if half_width[1] == 0.0:
        return float(midpoint[1])

    design = np.column_stack((np.ones(len(factor)), factor))
    central_residual = asset - design @ midpoint
    envelope = max(
        model["residual_buffer"] * np.max(np.abs(central_residual)),
        model["residual_vol_floor"] * np.std(asset, ddof=1),
    )

    # Eliminate epsilon in X = design @ beta + epsilon. The entropy of all
    # coefficients and residuals is now a convex function of two coordinates.
    # Each coordinate has midpoint zero and its box endpoints at -1 and +1.
    offset = np.r_[np.zeros(2), central_residual / envelope]
    transform = np.vstack((np.eye(2), -design * half_width / envelope))
    coefficients = np.zeros(2)
    objective_roundoff = 8.0 * np.finfo(float).eps * len(offset)
    for _ in range(100):
        z = offset + transform @ coefficients
        gradient = transform.T @ np.arctanh(z)
        hessian = transform.T @ (transform / (1.0 - z**2)[:, None])
        step = np.linalg.solve(hessian, -gradient)
        if np.max(np.abs(transform @ step)) <= 1e-9:
            return float(midpoint[1] + half_width[1] * coefficients[1])

        current = entropy(z)
        rate = 1.0
        while rate >= 1e-12:
            candidate = coefficients + rate * step
            trial = offset + transform @ candidate
            if (np.max(np.abs(trial)) < 1.0 and
                    entropy(trial) <= current + 1e-4 * rate * (gradient @ step) + objective_roundoff):
                coefficients = candidate
                break
            rate /= 2.0
        else:
            raise RuntimeError("Stage 1 entropy line search failed.")
    raise RuntimeError("Stage 1 entropy solver did not converge.")


def replication_weights(betas, target):
    # Paper equations 35-38, with long-only boxes [0, 1].
    # If necessary, project the target onto the available beta interval.
    beta_min, beta_max = betas.min(), betas.max()
    if target <= beta_min or target >= beta_max:
        match = betas == np.clip(target, beta_min, beta_max)
        return match / match.sum(), target < beta_min or target > beta_max

    scaled_beta = (betas - target) / (beta_max - beta_min)
    equal_weight_logit = logit(1.0 / len(betas))

    def budget_weights(slope):
        tilted = slope * scaled_beta
        intercept = brentq(
            lambda shift: expit(shift + tilted).sum() - 1.0,
            equal_weight_logit - tilted.max(),
            equal_weight_logit - tilted.min(),
        )
        return expit(intercept + tilted)

    def exposure_error(slope):
        return float(scaled_beta @ budget_weights(slope))

    lower, upper = -1.0, 1.0
    while exposure_error(lower) > 0.0:
        lower *= 2.0
    while exposure_error(upper) < 0.0:
        upper *= 2.0
    weights = budget_weights(brentq(exposure_error, lower, upper))
    if abs(weights.sum() - 1.0) > 1e-9 or abs(weights @ betas - target) > 1e-8:
        raise RuntimeError("Stage 2 entropy solution missed a constraint.")
    return weights, False


def simulate(returns, targets, fee_rate):
    weights = np.zeros(returns.shape[1])
    daily = []
    for day, asset_returns in zip(returns.index, returns.to_numpy()):
        gross = 1.0 + weights @ asset_returns
        weights = weights * (1.0 + asset_returns) / gross
        traded, trades = 0.0, 0
        target = targets.get(day)
        if target is not None:
            # Charge fees against actual buy and sell notionals and size the new
            # portfolio on post-fee equity so the rebalance is self-financing.
            equity_fraction = brentq(
                lambda fraction: fraction + fee_rate * np.abs(fraction * target - weights).sum() - 1.0,
                0.0, 1.0,
            )
            orders = np.abs(equity_fraction * target - weights)
            traded = orders.sum()
            trades = np.count_nonzero(orders > 1e-10)
            gross *= equity_fraction
            weights = target
        daily.append((gross - 1.0, traded, trades))
    return pd.DataFrame(daily, index=returns.index, columns=["return", "turnover", "n_trades"])


def metrics(daily, risk_free):
    returns = daily["return"]
    wealth = (1.0 + returns).cumprod().to_numpy()
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    excess = returns - risk_free.reindex(returns.index)
    vol, excess_vol = returns.std(), excess.std()
    annualizer = np.sqrt(TRADING_DAYS)
    return {
        "cagr": float(wealth[-1] ** (TRADING_DAYS / len(returns)) - 1.0),
        "ann_vol": float(vol * annualizer),
        "sharpe": float(returns.mean() / vol * annualizer) if vol > 0.0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * annualizer) if excess_vol > 0.0 else None,
        "max_drawdown": float(np.min(wealth / peak - 1.0)),
        "total_return": float(wealth[-1] - 1.0),
        "turnover": float(daily["turnover"].sum()),
        "n_trades": int(daily["n_trades"].sum()),
    }


def main():
    config = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    assets = config["universe"]
    benchmark, risk_free = config["benchmark"], config["risk_free"]
    symbols = assets + [benchmark, risk_free]
    if len(assets) < 2 or len(set(symbols)) != len(symbols):
        raise ValueError("Use at least two distinct constituents, a separate factor and a cash proxy.")
    dates = {key: pd.Timestamp(value) for key, value in config["dates"].items()}
    if not dates["price_start"] < dates["start"] <= dates["is_end"] < dates["end"]:
        raise ValueError("Dates must run from warmup through the IS and OOS periods.")

    model, execution = config["model"], config["execution"]
    lookback, spacing = model["lookback_sessions"], model["rebalance_every_sessions"]
    lag = execution["lag_sessions"]
    costs = (execution["cost_bps"], execution["slippage_bps"])
    if lookback < 3 or spacing < 1 or lag < 1:
        raise ValueError("Use at least three fitting sessions and a positive rebalance interval and lag.")
    if model["residual_buffer"] <= 1.0 or model["residual_vol_floor"] <= 0.0:
        raise ValueError("The residual envelope must have a buffer above one and a positive floor.")
    if min(costs) < 0.0 or sum(costs) >= 10_000:
        raise ValueError("Costs and slippage must be positive or zero and total below 10,000 bps.")

    closes, fetched_at = load_closes(symbols, dates["price_start"], dates["end"])
    returns = closes.pct_change(fill_method=None).iloc[1:]
    first_fill = returns.index.searchsorted(dates["start"])
    if first_fill - lag + 1 < lookback:
        raise ValueError("Insufficient warmup returns for the first signal.")
    if first_fill >= len(returns) or returns.index[first_fill] > dates["is_end"]:
        raise ValueError("No in-sample evaluation dates.")
    if (returns.index[first_fill:] <= dates["is_end"]).sum() < 2 or (returns.index > dates["is_end"]).sum() < 2:
        raise ValueError("Each evaluation period needs at least two sessions.")

    asset_returns = returns[assets].to_numpy()
    factor_returns = returns[benchmark].to_numpy()
    targets = {}
    projections = 0
    for fill in range(first_fill, len(returns), spacing):
        cutoff = fill - lag + 1
        window = slice(cutoff - lookback, cutoff)
        betas = np.array([
            fit_beta(asset, factor_returns[window], model) for asset in asset_returns[window].T
        ])
        weights, projected = replication_weights(betas, model["target_beta"])
        targets[returns.index[fill]] = weights
        projections += int(projected)

    sample = returns.iloc[first_fill:]
    fee_rate = sum(costs) / 10_000.0
    strategy_daily = simulate(sample[assets], targets, fee_rate)
    benchmark_daily = simulate(sample[[benchmark]], {sample.index[0]: np.ones(1)}, fee_rate)
    periods = {
        "is": sample.index <= dates["is_end"],
        "oos": sample.index > dates["is_end"],
        "full": slice(None),
    }
    full_range = f"{sample.index[0].date()} to {sample.index[-1].date()}"
    results = {"strategy": config["strategy"], "data_range": full_range}

    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"BETA_TARGET_PROJECTIONS: {projections}/{len(targets)}")
    for period, mask in periods.items():
        strategy_period = strategy_daily.loc[mask]
        benchmark_period = benchmark_daily.loc[mask]
        results[period] = metrics(strategy_period, sample[risk_free])
        benchmark_metrics = metrics(benchmark_period, sample[risk_free])
        tracking_error = (strategy_period["return"] - benchmark_period["return"]).std() * np.sqrt(TRADING_DAYS)
        print(f"{period.upper()} {strategy_period.index[0].date()} to {strategy_period.index[-1].date()}")
        print(f"  Strategy {json.dumps(results[period], allow_nan=False)}")
        print(f"  {benchmark} buy and hold {json.dumps(benchmark_metrics, allow_nan=False)}")
        print(f"  Annualized net tracking error {tracking_error:.6f}")
        if period == "full":
            results["benchmark"] = {"ticker": benchmark, **benchmark_metrics}
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
