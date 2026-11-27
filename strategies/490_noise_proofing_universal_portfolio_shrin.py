# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "490_ noise_proofing_universal_portfolio_shrin"\nuniverse: [SPY, QQQ, IWM, EFA, EEM, IEF, TLT, LQD, TIP, GLD, VNQ, XLU]\ncash_proxy: BIL\ndata_start: 2005-01-01\nevaluation_start: 2019-01-01\nis_end: 2021-12-31\nevaluation_end: 2026-10-31\n\ncost_bps: 2.0\nslippage_bps: 3.0\nexecution_lag_sessions: 1\n\nmodel:\n  lookback_months: 120\n  oracle_test_months: 6\n  ao_half_life_months: 24\n  ridge_min: 1.0e-8\n  ridge_max: 0.1\n  ridge_points: 20\n  ridge_validation_months: 24\n\nrobustness:\n  lookback_months: [108, 132]\n  ao_half_life_months: [18, 30]\n  cost_multiplier: 3.0\n'
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
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252


def load_config():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for key in ("data_start", "evaluation_start", "is_end", "evaluation_end"):
        config[key] = pd.Timestamp(config[key])
    start, first, split, end = (
        config[key] for key in ("data_start", "evaluation_start", "is_end", "evaluation_end")
    )
    if not start < first <= split < end or not all(
        (start.is_month_start, first.is_month_start, split.is_month_end, end.is_month_end)
    ):
        raise ValueError("Use complete, ordered calendar months for data and evaluation dates.")
    if end.date() >= datetime.now(timezone.utc).date():
        raise ValueError("The evaluation end must be a completed calendar month.")
    symbols = config["universe"]
    if (
        len(symbols) < 2 or len(symbols) != len(set(symbols))
        or any(not re.fullmatch(r"[A-Z]{1,5}", symbol) for symbol in symbols)
        or config["cash_proxy"] != "BIL" or "BIL" in symbols
    ):
        raise ValueError("Use unique US ETF symbols and BIL as a separate cash proxy.")
    if type(config["execution_lag_sessions"]) is not int or config["execution_lag_sessions"] < 1:
        raise ValueError("Execution lag must be at least one session.")
    for key in ("cost_bps", "slippage_bps"):
        if not np.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    return config


def check_prices(prices, config):
    start, end = config["data_start"], config["evaluation_end"]
    if (
        prices.empty or prices.index.has_duplicates or not prices.index.is_monotonic_increasing
        or not start <= prices.index[0] <= start + pd.Timedelta(days=7)
        or not end - pd.Timedelta(days=7) <= prices.index[-1] <= end
    ):
        raise ValueError("The price panel does not cover the requested history.")
    required = prices[config["universe"]]
    first = prices.index.searchsorted(config["evaluation_start"])
    cash = prices[config["cash_proxy"]].iloc[first - 1:]
    for observed in (required, cash):
        bad = (~np.isfinite(observed) | (observed <= 0)).to_numpy()
        if bad.any():
            row = np.argwhere(bad)[0, 0]
            raise ValueError(f"Missing or invalid required price on {observed.index[row].date()}.")
    months = prices.index.to_period("M").unique()
    if not months.equals(pd.period_range(start, end, freq="M")):
        raise ValueError("Missing calendar month in the price panel.")


def load_prices(config):
    symbols = config["universe"] + [config["cash_proxy"]]
    start = config["data_start"].date().isoformat()
    end = (config["evaluation_end"] + pd.Timedelta(days=1)).date().isoformat()
    key = sha256(",".join(sorted(symbols)).encode()).hexdigest()[:12]
    cache_file = CACHE / f"yahoo_adjusted_{start}_{end}_{key}.csv"
    if cache_file.exists():
        with cache_file.open(encoding="utf-8") as stream:
            fetched_at = stream.readline().strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(stream, index_col="Date", parse_dates=["Date"]).loc[:, symbols]
        check_prices(prices, config)
        return prices, fetched_at

    CACHE.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=CACHE, prefix=".yfinance-") as client_cache:
        yf.set_tz_cache_location(client_cache)
        downloaded = yf.download(
            symbols, start=start, end=end, interval="1d", auto_adjust=True,
            ignore_tz=True, threads=4, timeout=12, progress=False, multi_level_index=True,
        )
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if downloaded is None or downloaded.empty:
        raise RuntimeError("Yahoo Finance returned no daily prices.")
    prices = downloaded["Close"].reindex(columns=symbols).sort_index().rename_axis("Date")
    check_prices(prices, config)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(f"{fetched_at}\n")
            prices.to_csv(stream)
            stream.close()
            temporary.replace(cache_file)
        finally:
            temporary.unlink(missing_ok=True)
    return prices, fetched_at


def sample_correlation(returns):
    covariance = np.cov(returns, rowvar=False)
    volatility = np.sqrt(np.diag(covariance))
    if np.any(volatility <= 1e-12):
        raise ValueError("A monthly estimation window has a constant price series.")
    scales = np.outer(volatility, volatility)
    return covariance / scales, scales


def oracle_spectrum(calibration, subsequent):
    corr_cal, _ = sample_correlation(calibration)
    corr_next, _ = sample_correlation(subsequent)
    _, vectors = np.linalg.eigh(corr_cal)
    return np.diag(vectors.T @ corr_next @ vectors)


def ridge_basis(returns, oracle, penalties):
    corr, scales = sample_correlation(returns)
    _, vectors = np.linalg.eigh(corr)
    filtered = (vectors * np.maximum(oracle, 1e-10)) @ vectors.T
    diagonal = np.sqrt(np.diag(filtered))
    covariance = scales * filtered / np.outer(diagonal, diagonal)
    values, vectors = np.linalg.eigh(covariance)
    rotated_mean = vectors.T @ returns.mean(axis=0)
    raw = vectors @ (rotated_mean[:, None] / (np.maximum(values[:, None], 0) + penalties))
    positive = np.maximum(raw, 0)
    return np.divide(
        positive, positive.sum(axis=0), out=np.full_like(positive, 1.0 / len(positive)),
        where=positive.sum(axis=0) > 0,
    )


def fit_mixture(realized, initial):
    mean = realized.mean(axis=0)
    covariance = np.cov(realized, rowvar=False)
    # Scale the objective for numerical conditioning. The simplex optimum is unchanged.
    scale = 100.0
    result = minimize(
        lambda weights: scale * (0.5 * weights @ covariance @ weights - mean @ weights),
        initial, jac=lambda weights: scale * (covariance @ weights - mean),
        method="SLSQP", bounds=[(0.0, 1.0)] * len(initial),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0,
                     "jac": lambda weights: np.ones_like(weights)},
        options={"ftol": 1e-11},
    )
    if not result.success:
        raise RuntimeError(f"Ridge mixture fit failed: {result.message}")
    weights = np.maximum(result.x, 0)
    return weights / weights.sum()


def month_end_rows(dates):
    months = dates.to_period("M")
    return np.flatnonzero(np.r_[months[1:] != months[:-1], True])


def model_targets(prices, signals, fills, model):
    lookback = model["lookback_months"]
    oracle_test = model["oracle_test_months"]
    validation = model["ridge_validation_months"]
    half_life = model["ao_half_life_months"]
    if min(lookback, oracle_test, validation, model["ridge_points"]) < 2 or half_life <= 0:
        raise ValueError("The model requires positive window lengths and multiple ridge basis points.")
    if not 0 < model["ridge_min"] < model["ridge_max"]:
        raise ValueError("Ridge grid bounds must be strictly positive and increasing.")
    penalties = np.geomspace(model["ridge_min"], model["ridge_max"], model["ridge_points"])
    decay = 2.0 ** (-1.0 / half_life)
    monthly_prices = prices[signals]
    monthly_returns = monthly_prices[1:] / monthly_prices[:-1] - 1.0
    oracle = None
    first_model = lookback + oracle_test + 1
    next_score = first_model
    bases = {}
    scores = []
    mixture = np.full(len(penalties), 1.0 / len(penalties))
    mixture_sum = np.zeros(len(penalties))
    mixture_count = 0
    targets = {}

    for month in range(first_model, len(signals)):
        if fills[month] >= len(prices):
            break
        # The oracle test block ends one month before this signal.
        test_end = month - 1
        test_start = test_end - oracle_test
        observed = oracle_spectrum(
            monthly_returns[test_start - lookback:test_start], monthly_returns[test_start:test_end]
        )
        oracle = observed if oracle is None else decay * oracle + (1.0 - decay) * observed
        basis = ridge_basis(monthly_returns[month - lookback:month], oracle, penalties)
        bases[month] = basis

        # A basis score becomes usable only after its entire fill-to-fill interval ends.
        while next_score + 1 < month and fills[next_score + 1] <= signals[month]:
            start, end = fills[next_score:next_score + 2]
            forward_returns = prices[end] / prices[start] - 1.0
            scores.append(forward_returns @ bases.pop(next_score))
            next_score += 1
        if len(scores) >= validation:
            mixture = fit_mixture(np.asarray(scores[-validation:]), mixture)
            mixture_sum += mixture
            mixture_count += 1
            targets[fills[month]] = basis @ (mixture_sum / mixture_count)
    return targets


def simulate(daily_returns, targets, config, cost_multiplier):
    cost_rate = cost_multiplier * (config["cost_bps"] + config["slippage_bps"]) / 10_000
    if not np.isfinite(cost_rate) or not 0 <= cost_rate < 1:
        raise ValueError("Total transaction friction must be less than 100% of traded value.")
    first = daily_returns.index.searchsorted(config["evaluation_start"])
    positions = np.zeros(daily_returns.shape[1])
    records = []
    for row, asset_returns in enumerate(daily_returns.to_numpy()[first:], start=first):
        gross_factor = 1.0 + positions @ asset_returns
        positions = positions * (1.0 + asset_returns) / gross_factor
        remaining = 1.0
        turnover = 0.0
        trades = 0
        if row in targets:
            target = targets[row]
            # Targets apply to the post-cost value. Charge on the resulting executions.
            if cost_rate > 0:
                remaining = brentq(
                    lambda fraction: fraction + cost_rate * np.abs(fraction * target - positions).sum() - 1.0,
                    0.0, 1.0,
                )
            traded = np.abs(remaining * target - positions)
            turnover = traded.sum()
            trades = np.count_nonzero(traded > 1e-10)
            positions = target
        records.append((gross_factor * remaining - 1.0, turnover, trades))
    return pd.DataFrame(records, index=daily_returns.index[first:],
                        columns=["return", "turnover", "n_trades"])


def evaluate(prices, config, model, cost_multiplier=1.0, equal_weight=False):
    etfs = prices[config["universe"]]
    signals = month_end_rows(etfs.index)
    fills = signals + config["execution_lag_sessions"]
    first = etfs.index.searchsorted(config["evaluation_start"])
    due = fills[(fills >= first) & (fills < len(etfs))]
    if equal_weight:
        weights = np.repeat(1 / etfs.shape[1], etfs.shape[1])
        targets = dict.fromkeys(due, weights)
    else:
        targets = model_targets(etfs.to_numpy(), signals, fills, model)
    if len(due) == 0 or not all(row in targets for row in due):
        raise ValueError("Insufficient warm-up history for the configured evaluation start.")
    return simulate(etfs.pct_change(fill_method=None), targets, config, cost_multiplier)


def metrics(daily_results, cash_returns):
    returns = daily_results["return"].to_numpy()
    excess = returns - cash_returns.loc[daily_results.index].to_numpy()
    if len(returns) < 2:
        raise ValueError("Each period requires at least two daily observations.")
    growth = np.r_[1.0, np.cumprod(1.0 + returns)]
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    years = len(returns) / SESSIONS_PER_YEAR
    return {
        "cagr": float(growth[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(volatility * annualizer),
        "sharpe": float(returns.mean() * annualizer / volatility) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() * annualizer / excess_volatility) if excess_volatility > 0 else None,
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1.0)),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, cash_returns, split):
    is_mask = daily_results.index <= split
    return {
        "is": metrics(daily_results.loc[is_mask], cash_returns),
        "oos": metrics(daily_results.loc[~is_mask], cash_returns),
        "full": metrics(daily_results, cash_returns),
    }


def main():
    config = load_config()
    prices, fetched_at = load_prices(config)
    model, robustness = config["model"], config["robustness"]
    split = config["is_end"]
    cash_returns = prices[config["cash_proxy"]].pct_change(fill_method=None)
    is_prices = prices.loc[:split]
    cases = [
        (f"{parameter}={value}", {**model, parameter: value}, 1.0)
        for parameter in ("lookback_months", "ao_half_life_months")
        for value in robustness[parameter]
    ]
    cases.append((f"friction_multiplier={robustness['cost_multiplier']}", model, robustness["cost_multiplier"]))
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return close in USD")
    print(f"HISTORY: {prices.index[0].date()} to {prices.index[-1].date()}")
    print("ROBUSTNESS (IS ONLY)")
    for label, case_model, multiplier in cases:
        case = evaluate(is_prices, config, case_model, multiplier)
        print(f"  {label}: {json.dumps(metrics(case, cash_returns.loc[:split]), allow_nan=False)}")

    daily_results = evaluate(prices, config, model)
    results = period_metrics(daily_results, cash_returns, split)
    is_official = evaluate(is_prices, config, model)
    pd.testing.assert_frame_equal(is_official, daily_results.loc[:split])
    benchmark = evaluate(prices, config, model, equal_weight=True)
    benchmark_results = period_metrics(benchmark, cash_returns, split)
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(results[period], allow_nan=False)}")
        print(f"BENCHMARK {period.upper()} (monthly equal-weight ETFs): "
              f"{json.dumps(benchmark_results[period], allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {1 + len(cases)}")
    payload = {
        "strategy": config["strategy"],
        "data_range": f"{daily_results.index[0].date()} to {daily_results.index[-1].date()}",
        **results,
        "benchmark": {"name": "Monthly equal-weight ETFs", **benchmark_results},
    }
    print("RESULTS_JSON: " + json.dumps(payload, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    main()
