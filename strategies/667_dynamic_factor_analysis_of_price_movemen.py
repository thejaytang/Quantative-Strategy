# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "667_ dynamic_factor_analysis_of_price_movemen"\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nbenchmark: SPY\nrisk_free: BIL\n\ndates:\n  data_start: "2007-07-02"\n  start: "2010-01-05"\n  is_end: "2019-12-31"\n  end: "2025-12-31"\n\nexecution:\n  signal_lag_sessions: 1\n  cost_bps: 5.0\n  slippage_bps: 5.0\n\nmodel:\n  lookback_sessions: 252\n  factors: 2\n  factor_order: 2\n  forecast_sessions: 5\n  selected_sectors: 3\n  tilt_weight: 0.50\n  max_iterations: 150\n\nrobustness:\n  lookback_sessions: [210, 294]\n  cost_multiplier: 2.0\n'
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
"""Rolling dynamic-factor sector allocation, C243 (research only).

Paper: https://arxiv.org/abs/2510.15938
Authors' implementation (MIT): https://github.com/briangodwinlim/DynamicFactorAnalysis
https://github.com/briangodwinlim/DynamicFactorAnalysis/blob/main/examples/DynamicFactorModel/DynamicFactorModel.ipynb
DynamicFactor API and example (BSD-3-Clause): https://github.com/statsmodels/statsmodels
https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_dfm_coincident.html
https://pypi.org/project/statsmodels/
Related approaches consulted, no code copied: https://github.com/MiniKetch/dynamic-factors
https://www.quantconnect.com/docs/v2/research-environment/applying-research/kalman-filters-and-stat-arb
https://raw.githubusercontent.com/quantopian/research_public/master/notebooks/lectures/Factor_Based_Risk_Management/notebook.ipynb
"""

import gzip
import hashlib
import json
import signal
import warnings
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
PRICE_ADJUSTMENT = (
    "yfinance auto_adjust=False split-normalized OHLC; "
    "dividends and capital gains reinvested in the originating ETF on the ex-date open"
)
PRICE_FIELDS = ["Open", "Close", "Dividends", "Capital Gains"]


def validate_prices(prices):
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and increasing")
    if prices.empty or not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Incomplete Yahoo Finance panel; refusing to impute prices or actions")
    for field in ("Open", "Close"):
        if (prices.xs(field, axis=1, level=1) <= 0).any().any():
            raise ValueError(f"Nonpositive Yahoo Finance {field}")


def load_prices(symbols, start, end):
    CACHE.mkdir(exist_ok=True)
    key = hashlib.sha256(f"{symbols}:{start}:{end}:actions-v1".encode()).hexdigest()[:16]
    path = CACHE / f"yfinance_{key}.csv.gz"
    if path.exists():
        with gzip.open(path, "rt") as stream:
            fetched_at = datetime.fromisoformat(stream.readline().strip())
            prices = pd.read_csv(
                stream, header=[0, 1], index_col=0, parse_dates=True, float_precision="round_trip"
            )
        validate_prices(prices)
        return prices, fetched_at

    end_exclusive = (end + pd.Timedelta(days=1)).date().isoformat()
    with TemporaryDirectory(prefix="download-", dir=CACHE) as temporary:
        yf.set_tz_cache_location(temporary)
        frames = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            for _ in range(2):
                frame = ticker.history(
                    start=start.date().isoformat(), end=end_exclusive, interval="1d",
                    auto_adjust=False, actions=True, repair=False, timeout=10,
                )
                if not frame.empty:
                    break
            if frame.empty:
                raise RuntimeError(f"No Yahoo Finance daily history for {symbol}")
            frame.index = frame.index.tz_localize(None).normalize()
            if "Capital Gains" not in frame:
                frame["Capital Gains"] = 0.0
            frames[symbol] = frame[PRICE_FIELDS]

        prices = pd.concat(frames, axis=1).sort_index()
        validate_prices(prices)
        fetched_at = datetime.now(timezone.utc)
        temporary_path = Path(temporary) / path.name
        with gzip.open(temporary_path, "wt") as stream:
            stream.write(fetched_at.isoformat() + "\n")
            prices.to_csv(stream)
        temporary_path.replace(path)
    return prices, fetched_at


def price_relatives(prices):
    opening = prices.xs("Open", axis=1, level=1)
    closing = prices.xs("Close", axis=1, level=1)
    distributions = (
        prices.xs("Dividends", axis=1, level=1) + prices.xs("Capital Gains", axis=1, level=1)
    )
    overnight = ((opening + distributions) / closing.shift(1)).iloc[1:]
    intraday = (closing / opening).iloc[1:]
    if (overnight <= 0).any().any():
        raise ValueError("Nonpositive overnight total-return price relative")
    return overnight, intraday


def fit_forecast(window, spec, lag, start_params):
    model = DynamicFactor(
        window, k_factors=spec["factors"], factor_order=spec["factor_order"],
        error_order=0, error_cov_type="diagonal", enforce_stationarity=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = model.fit(start_params=start_params, maxiter=spec["max_iterations"],
                           method="lbfgs", disp=False, cov_type="none")
    if not result.mle_retvals["converged"] or not np.isfinite(result.llf):
        raise ValueError("Dynamic-factor likelihood fit did not converge")

    state = result.filtered_state[:, -1]
    transition = result.filter_results.transition[:, :, 0]
    design = result.filter_results.design[:, :, 0]
    common_prediction = np.zeros(window.shape[1])
    for step in range(1, lag + spec["forecast_sessions"] + 1):
        state = transition @ state
        # Skip the close-to-close forecast spanning the execution open.
        if step > lag:
            common_prediction += design @ state
    if not np.isfinite(common_prediction).all():
        raise ValueError("Nonfinite common-return forecast")
    return common_prediction, result.params


def monthly_targets(returns, start, lag, spec):
    dates = returns.index
    new_months = np.flatnonzero(~dates.to_period("M").duplicated())
    targets = {}
    fallbacks = []
    last_params = None
    lookback = spec["lookback_sessions"]
    values = returns.to_numpy() * 100.0
    neutral = np.ones(returns.shape[1]) / returns.shape[1]
    for signal_index in new_months:
        fill_index = signal_index + lag
        if fill_index >= len(dates):
            break
        if dates[fill_index] < start or signal_index + 1 < lookback:
            continue
        window = values[signal_index + 1 - lookback:signal_index + 1]
        try:
            forecast, last_params = fit_forecast(window, spec, lag, last_params)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            fallbacks.append((str(dates[fill_index].date()), str(error)))
            targets[dates[fill_index]] = neutral
            last_params = None
            continue

        selected = np.argsort(-forecast, kind="stable")[:spec["selected_sectors"]]
        weights = neutral * (1.0 - spec["tilt_weight"])
        weights[selected] += spec["tilt_weight"] / len(selected)
        targets[dates[fill_index]] = weights
    return targets, fallbacks


def rebalance(positions, cash, weights, friction):
    open_value = cash + positions.sum()
    changes = weights * open_value - positions
    threshold = open_value * 1e-12
    changes[np.abs(changes) <= threshold] = 0.0
    sells = np.maximum(-changes, 0.0)
    buys = np.maximum(changes, 0.0)
    cash += sells.sum() * (1.0 - friction)
    buy_cost = buys.sum() * (1.0 + friction)
    if buy_cost > cash:
        buys *= cash / buy_cost
    cash -= buys.sum() * (1.0 + friction)
    if cash < -threshold:
        raise RuntimeError("Rebalance exceeded cash budget")
    traded = buys + sells
    return (positions - sells + buys, max(cash, 0.0),
            traded.sum() / open_value, np.count_nonzero(traded > 0.0))


def simulate(overnight, intraday, targets, start, end, execution):
    friction = (execution["cost_bps"] + execution["slippage_bps"]) / 10_000.0
    positions = np.zeros(overnight.shape[1])
    cash = 1.0
    daily_rows = []
    dates = overnight.loc[start:end].index
    for date, gap, session in zip(dates, overnight.loc[dates].to_numpy(),
                                  intraday.loc[dates].to_numpy()):
        previous_value = cash + positions.sum()
        positions *= gap
        turnover, trades = 0.0, 0
        if date in targets:
            positions, cash, turnover, trades = rebalance(positions, cash, targets[date], friction)
        positions *= session
        daily_rows.append(((cash + positions.sum()) / previous_value - 1.0, turnover, trades))
    return pd.DataFrame(daily_rows, index=dates, columns=["return", "turnover", "n_trades"])


def metrics(daily_results, risk_free):
    returns = daily_results["return"]
    excess = returns - risk_free.loc[returns.index]
    growth = np.r_[1.0, (1.0 + returns.to_numpy()).cumprod()]
    scale = np.sqrt(SESSIONS_PER_YEAR)
    sigma = returns.std(ddof=1)
    sigma_excess = excess.std(ddof=1)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1.0),
        "ann_vol": float(sigma * scale),
        "sharpe": float(returns.mean() * scale / sigma) if sigma > 0.0 else None,
        "sharpe_excess": float(excess.mean() * scale / sigma_excess) if sigma_excess > 0.0 else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1.0).min()),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def print_robustness(lookbacks, cost_multiplier, returns, overnight, intraday,
                     risk_free, targets, config, start, is_end):
    spec = config["model"]
    execution = config["execution"]
    lag = execution["signal_lag_sessions"]
    print("ROBUSTNESS (IS only, default configuration fixed)", flush=True)
    for lookback in lookbacks:
        variant = {**spec, "lookback_sessions": lookback}
        alternative, fallbacks = monthly_targets(returns, start, lag, variant)
        daily = simulate(overnight, intraday, alternative, start, is_end, execution)
        outcome = {**metrics(daily, risk_free), "fit_fallbacks": len(fallbacks)}
        print(f"lookback_sessions={lookback}: {json.dumps(outcome, allow_nan=False)}", flush=True)

    higher_costs = {**execution, "cost_bps": execution["cost_bps"] * cost_multiplier,
                    "slippage_bps": execution["slippage_bps"] * cost_multiplier}
    daily = simulate(overnight, intraday, targets, start, is_end, higher_costs)
    print(f"cost_and_slippage_x{cost_multiplier:g}: "
          f"{json.dumps(metrics(daily, risk_free), allow_nan=False)}", flush=True)
    return len(lookbacks) + 1


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assets, spec, execution = config["universe"], config["model"], config["execution"]
    benchmark, risk_free_symbol = config["benchmark"], config["risk_free"]
    data_start, start, is_end, end = [pd.Timestamp(config["dates"][key]) for key in
                                     ("data_start", "start", "is_end", "end")]
    lag = execution["signal_lag_sessions"]
    if not isinstance(lag, int) or lag < 1 or not data_start < start < is_end < end:
        raise ValueError("Require a positive session lag and increasing sample dates")
    if not 0 < spec["factors"] < len(assets) or len(set(assets)) != len(assets):
        raise ValueError("Require a unique universe with more series than common factors")
    if (not 1 <= spec["selected_sectors"] <= len(assets) or
            not 0.0 < spec["tilt_weight"] <= 1.0 or
            min(spec["factor_order"], spec["forecast_sessions"], spec["max_iterations"]) < 1):
        raise ValueError("Invalid dynamic-factor model or allocation parameters")
    lookbacks = list(dict.fromkeys(config["robustness"]["lookback_sessions"]))
    cost_multiplier = config["robustness"]["cost_multiplier"]
    costs = [execution["cost_bps"], execution["slippage_bps"]]
    if min(costs) < 0.0 or not 1.0 < cost_multiplier or sum(costs) * cost_multiplier >= 10_000.0:
        raise ValueError("Invalid trading costs or cost multiplier")
    if (len(lookbacks) < 2 or spec["lookback_sessions"] in lookbacks or
            min(*lookbacks, spec["lookback_sessions"]) <= spec["factor_order"] + len(assets)):
        raise ValueError("Require at least two distinct, sufficiently long alternative windows")

    symbols = list(dict.fromkeys([*assets, benchmark, risk_free_symbol]))
    prices, fetched_at = load_prices(symbols, data_start, end)
    if not {data_start, start, is_end, end}.issubset(prices.index):
        raise ValueError("Sample dates do not align with available Yahoo Finance sessions")
    overnight, intraday = price_relatives(prices)
    total_returns = overnight * intraday - 1.0
    risk_free = total_returns[risk_free_symbol]
    assert prices.index[-1] == end
    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}", flush=True)
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}", flush=True)

    with threadpool_limits(limits=1):
        targets, fallbacks = monthly_targets(total_returns[assets], start, lag, spec)
        is_targets = {date: weights for date, weights in targets.items() if date <= is_end}
        n_variants = print_robustness(
            lookbacks, cost_multiplier, total_returns.loc[:is_end, assets],
            overnight.loc[:is_end, assets], intraday.loc[:is_end, assets], risk_free.loc[:is_end],
            is_targets, config, start, is_end,
        )

    daily = simulate(overnight[assets], intraday[assets], targets, start, end, execution)
    official = {
        "is": metrics(daily.loc[:is_end], risk_free),
        "oos": metrics(daily.loc[daily.index > is_end], risk_free),
        "full": metrics(daily, risk_free),
    }
    if min(official[period]["n_trades"] for period in ("is", "oos")) == 0:
        raise RuntimeError("Official run needs fills in both evaluation periods")
    failed_dates = {pd.Timestamp(date) for date, _ in fallbacks}
    model_dates = targets.keys() - failed_dates
    if not any(date <= is_end for date in model_dates) or not any(date > is_end for date in model_dates):
        raise RuntimeError("Official run needs model-driven allocation in both evaluation periods")

    baseline = simulate(overnight[[benchmark]], intraday[[benchmark]], {start: np.ones(1)},
                        start, end, execution)
    benchmark_metrics = {"ticker": benchmark, **metrics(baseline, risk_free)}
    for period, result in official.items():
        print(f"{period.upper()}: {json.dumps(result, allow_nan=False)}")
    print(f"BENCHMARK: {json.dumps(benchmark_metrics, allow_nan=False)}")
    print(f"MODEL_FIT_FALLBACKS: {len(fallbacks)} / {len(targets)}")
    for date, reason in fallbacks:
        print(f"FIT_FALLBACK: {date} {reason}")
    print(f"PARAM_COMBOS_TRIED: {1 + n_variants}")
    print(f"FILLS: {official['full']['n_trades']} across {len(targets)} scheduled rebalances")
    print("STATUS: 已回测 | pending review")
    output = {
        "strategy": config["strategy"],
        "data_range": {"start": str(daily.index[0].date()), "end": str(daily.index[-1].date())},
        **official,
        "benchmark": benchmark_metrics,
    }
    print(f"RESULTS_JSON: {json.dumps(output, ensure_ascii=False, allow_nan=False)}")


if __name__ == "__main__":
    signal.alarm(840)
    main()
