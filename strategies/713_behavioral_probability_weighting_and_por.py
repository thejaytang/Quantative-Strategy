# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 713_behavioral_probability_weighting_and_por\ndata_start: "2022-12-01"\nstart: "2024-02-26"\nis_end: "2025-07-31"\nend: "2026-10-30"\nbenchmark: DIA\nexcess_reference: BIL\nsignal_lag_sessions: 1\ncost_bps: 5\nslippage_bps: 5\nuniverse:\n  effective: "2024-02-26"\n  announced: "2024-02-20"\n  initial: [AAPL, AMGN, AMZN, AXP, BA, CAT, CRM, CSCO, CVX, DIS,\n            DOW, GS, HD, HON, IBM, INTC, JNJ, JPM, KO, MCD,\n            MMM, MRK, MSFT, NKE, PG, TRV, UNH, V, VZ, WMT]\n  changes:\n    - effective: "2024-11-08"\n      announced: "2024-11-01"\n      add: [NVDA, SHW]\n      remove: [INTC, DOW]\n    - effective: "2026-06-29"\n      announced: "2026-06-23"\n      add: [GOOGL]\n      remove: [VZ]\nmodel:\n  lookback_sessions: 252\n  prelec_gamma: 0.70\n  tail_confidence: 0.99\n  risk_penalty: 0.50\n  max_weight: 0.15\nrobustness:\n  prelec_gamma: [0.60, 0.80]\n  risk_penalty: [0.35, 0.65]\n  equal_probability_gamma: 1.00\n'
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
"""Research backtest for C278, a behavioral mean–CVaR portfolio.

Paper: https://arxiv.org/abs/2507.04208
Prelec weighting: https://doi.org/10.2307/2998573
Research references (ideas only, independent implementation):
https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/pypfopt/efficient_frontier/efficient_cvar.py
https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/LICENSE (MIT)
https://pypi.org/project/pyportfolioopt/
https://github.com/cvxgrp/cptopt
https://github.com/cvxgrp/cptopt/blob/master/cptopt/utility.py
https://github.com/cvxgrp/cptopt/blob/master/setup.cfg (Apache 2.0 metadata)
https://www.quantconnect.com/forum/discussion/11712/mean-cvar-portfolio-construction-model/
Related distributional comparison: https://github.com/akashdeepo/NTS_PWF (MIT)
https://github.com/akashdeepo/NTS_PWF/blob/main/src/nts_utils.py
https://github.com/akashdeepo/NTS_PWF/blob/main/LICENSE
"""

from datetime import UTC, datetime
import gzip
from hashlib import sha256
import json
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq, linprog
from scipy.sparse import csr_matrix, eye, hstack
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
BPS = 10_000
SESSIONS_PER_YEAR = 252
PRICE_ADJUSTMENT = (
    "yfinance split-adjusted Open/Close (auto_adjust=False), "
    "cash distributions reinvested at ex-date open, vendor spin-off adjustments"
)


def download_prices(tickers, start, end):
    end_exclusive = str((pd.Timestamp(end) + pd.Timedelta(days=1)).date())
    for attempt in range(2):
        price_frame = yf.download(
            tickers, start=start, end=end_exclusive, interval="1d", actions=True,
            auto_adjust=False, repair=False, progress=False, threads=False, timeout=12,
        )
        if price_frame is not None and not price_frame.empty:
            price_fields = price_frame.columns.get_level_values(0)
            for field in ("Dividends", "Capital Gains"):
                if field not in price_fields:
                    for ticker in tickers:
                        price_frame[field, ticker] = 0.0
            price_frame = price_frame.reindex(
                columns=pd.MultiIndex.from_product(
                    [["Open", "Close", "Dividends", "Capital Gains"], tickers]
                )
            ).sort_index()
            if price_frame[["Open", "Close"]].notna().all().all():
                return price_frame
        print(f"Yahoo price coverage incomplete after attempt {attempt + 1}", flush=True)
    raise RuntimeError("Missing Yahoo price bars for the configured universe.")


def load_market(config):
    universe = config["universe"]
    tickers = set(universe["initial"])
    for change in universe["changes"]:
        if pd.Timestamp(change["effective"]) <= pd.Timestamp(config["end"]):
            tickers.update(change["add"])
    tickers.update((config["benchmark"], config["excess_reference"]))
    tickers = sorted(tickers)
    start, end = config["data_start"], config["end"]
    cache_key = sha256(json.dumps(["ohlc-distributions-v1", tickers, start, end]).encode())
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"prices_{cache_key.hexdigest()[:16]}.csv.gz"

    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as cached:
            fetched_at = cached.readline().strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(cached, header=[0, 1], index_col=0)
        prices.index = pd.to_datetime(prices.index)
    else:
        with TemporaryDirectory(dir=CACHE, prefix="yahoo_") as transient:
            yf.set_tz_cache_location(transient)
            prices = download_prices(tickers, start, end)
        fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
        write_price_cache(path, prices, fetched_at)

    opens, closes = prices["Open"], prices["Close"]
    index = prices.index
    if (
        not index.is_unique or not index.is_monotonic_increasing
        or index[0] < pd.Timestamp(start) or index[-1] > pd.Timestamp(end)
        or pd.Timestamp(end) - index[-1] > pd.Timedelta(days=4)
        or not np.isfinite(opens).all().all() or not np.isfinite(closes).all().all()
        or (opens <= 0).any().any() or (closes <= 0).any().any()
    ):
        raise RuntimeError("Incomplete or invalid Yahoo OHLC data.")

    distributions = prices["Dividends"].fillna(0) + prices["Capital Gains"].fillna(0)
    # Mark yesterday's shares at the open and reinvest any ex-date distribution there.
    # Split ratios already appear in both Yahoo prices in the same share units.
    open_growth = (opens + distributions) / closes.shift(1)
    intraday_growth = closes / opens
    return open_growth, intraday_growth, fetched_at


def write_price_cache(path, prices, fetched_at):
    with NamedTemporaryFile(dir=CACHE, suffix=".csv.gz", delete=False) as temp:
        pending = Path(temp.name)
    try:
        with gzip.open(pending, "wt", encoding="utf-8", newline="") as compressed:
            compressed.write(fetched_at + "\n")
            prices.to_csv(compressed)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def prelec(probabilities, gamma):
    if gamma == 1:
        return probabilities
    with np.errstate(divide="ignore"):
        return np.exp(-(-np.log(probabilities)) ** gamma)


def scenario_probabilities(reference_returns, gamma):
    # Equal reference returns share the total probability of their rank interval.
    _, group, counts = np.unique(reference_returns, return_inverse=True, return_counts=True)
    cumulative = np.r_[0, counts.cumsum()] / len(reference_returns)
    return (np.diff(prelec(cumulative, gamma)) / counts)[group]


def allocate(returns, benchmark, assets, model):
    scenarios = returns.loc[:, assets].to_numpy() * BPS
    probabilities = scenario_probabilities(returns[benchmark].to_numpy(), model["prelec_gamma"])
    tail_mass = prelec(1 - model["tail_confidence"], model["prelec_gamma"])
    penalty = model["risk_penalty"]
    cap = model["max_weight"]
    n_days, n_assets = scenarios.shape

    # Minimize -E[Rw] + penalty * (threshold + E[excess losses] / tail_mass).
    # Each scenario imposes -Rw - threshold - excess_loss <= 0.
    objective = np.r_[-(probabilities @ scenarios), penalty, penalty * probabilities / tail_mass]
    tail_constraints = hstack(
        [csr_matrix(-scenarios), csr_matrix(-np.ones((n_days, 1))), -eye(n_days)],
        format="csr",
    )
    result = linprog(
        objective, A_ub=tail_constraints, b_ub=np.zeros(n_days),
        A_eq=np.r_[np.ones(n_assets), np.zeros(n_days + 1)][None, :], b_eq=[1.0],
        bounds=[(0, cap)] * n_assets + [(None, None)] + [(0, None)] * n_days,
        method="highs", options={"time_limit": 5.0},
    )
    if not result.success:
        raise RuntimeError(f"Portfolio solve failed at {returns.index[-1].date()}: {result.message}")
    weights = np.clip(result.x[:n_assets], 0.0, cap)
    if not np.isclose(weights.sum(), 1, atol=1e-8, rtol=0):
        raise RuntimeError("Optimizer weights do not sum to one.")
    return weights / weights.sum()


def build_targets(close_returns, config, model):
    dates = close_returns.index
    start = pd.Timestamp(config["start"])
    lag = config["signal_lag_sessions"]
    lookback = model["lookback_sessions"]
    universe = config["universe"]
    first_fill = dates.searchsorted(start)
    if first_fill - lag < lookback or first_fill >= len(dates):
        raise ValueError("Insufficient warm-up or an empty evaluation period.")

    active = set(universe["initial"])
    initial_date = pd.Timestamp(universe["effective"])
    initial_announcement = pd.Timestamp(universe["announced"])
    if dates[first_fill] < initial_date or dates[first_fill - lag] <= initial_announcement:
        raise ValueError("Initial membership is unavailable for the requested execution dates.")

    # For a longer lag, postpone an effective-date rebalance until the signal can
    # incorporate the publicly announced change.
    changes = {}
    previous_date = initial_date
    for change in universe["changes"]:
        effective = pd.Timestamp(change["effective"])
        announced = pd.Timestamp(change["announced"])
        if not previous_date < effective or not announced < effective:
            raise ValueError("Invalid constituent history dates.")
        previous_date = effective
        fill = max(dates.searchsorted(effective), dates.searchsorted(announced, side="right") + lag)
        if fill < len(dates):
            changes.setdefault(fill, []).append(change)

    targets = {}
    for fill, date in enumerate(dates):
        for change in changes.get(fill, ()):
            if not set(change["remove"]) <= active or active.intersection(change["add"]):
                raise ValueError("Invalid constituent change.")
            active.difference_update(change["remove"])
            active.update(change["add"])
        if fill < first_fill:
            continue
        if fill != first_fill and fill not in changes and date.month == dates[fill - 1].month:
            continue
        signal_row = fill - lag
        window = close_returns.iloc[signal_row - lookback + 1:signal_row + 1]
        assets = sorted(active)
        target = pd.Series(0.0, index=close_returns.columns)
        target.loc[assets] = allocate(window, config["benchmark"], assets, model)
        targets[date] = target.to_numpy()
    return targets


def rebalance(positions, cash, target, friction):
    # These positions are already marked and sized at today's open.
    open_value = positions.sum() + cash

    def amount_due(invested):
        return invested + friction * np.abs(target * invested - positions).sum() - open_value

    invested = brentq(amount_due, 0, open_value, xtol=1e-14)
    new_positions = target * invested
    changed = np.abs(new_positions - positions)
    traded_value = changed.sum()
    remaining = open_value - new_positions.sum() - friction * traded_value
    if remaining < -1e-12 * open_value:
        raise RuntimeError("Negative cash after rebalancing.")
    return new_positions, max(0.0, remaining), traded_value / open_value, int(
        np.count_nonzero(changed > 1e-10 * open_value)
    )


def simulate(open_growth, intraday_growth, targets, friction):
    positions = np.zeros(len(open_growth.columns))
    cash = 1.0
    previous_value = cash
    rows = []
    for date, overnight, intraday in zip(
        open_growth.index, open_growth.to_numpy(), intraday_growth.to_numpy(), strict=True,
    ):
        positions *= overnight
        turnover = 0.0
        trades = 0
        if date in targets:
            positions, cash, turnover, trades = rebalance(positions, cash, targets[date], friction)
        # The closing price enters only after all opening executions are complete.
        positions *= intraday
        closing_value = positions.sum() + cash
        rows.append((closing_value / previous_value - 1, turnover, trades))
        previous_value = closing_value
    return pd.DataFrame(rows, index=open_growth.index, columns=["return", "turnover", "n_trades"])


def run_strategy(open_growth, intraday_growth, config, model, friction):
    close_returns = open_growth * intraday_growth - 1
    targets = build_targets(close_returns, config, model)
    return simulate(
        open_growth.loc[config["start"]:], intraday_growth.loc[config["start"]:], targets, friction,
    )


def summarize(daily_results, reference):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise ValueError("A performance period needs at least two sessions.")
    curve = np.r_[1.0, (1 + returns).cumprod().to_numpy()]

    def sharpe(daily):
        deviation = daily.std(ddof=1)
        return float(np.sqrt(SESSIONS_PER_YEAR) * daily.mean() / deviation) if deviation > 0 else None

    return {
        "cagr": float(curve[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(returns - reference.loc[returns.index]),
        "max_drawdown": float(np.min(curve / np.maximum.accumulate(curve) - 1)),
        "total_return": float(curve[-1] - 1),
        "turnover": float(daily_results["turnover"].sum() * SESSIONS_PER_YEAR / len(returns)),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def split_summary(daily_results, reference, is_end):
    is_dates = daily_results.index <= pd.Timestamp(is_end)
    return {
        "is": summarize(daily_results.loc[is_dates], reference),
        "oos": summarize(daily_results.loc[~is_dates], reference),
        "full": summarize(daily_results, reference),
    }


def validate_model(model):
    if (
        model["lookback_sessions"] < 2 or not 0 < model["prelec_gamma"] <= 1
        or not 0 < model["tail_confidence"] < 1 or model["risk_penalty"] <= 0
        or not 0 < model["max_weight"] <= 1
    ):
        raise ValueError("Invalid model parameters.")


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    lag = config["signal_lag_sessions"]
    friction = (config["cost_bps"] + config["slippage_bps"]) / BPS
    if (
        not isinstance(lag, int) or lag < 1
        or min(config["cost_bps"], config["slippage_bps"]) < 0 or not friction < 1
        or not (
            pd.Timestamp(config["data_start"]) < pd.Timestamp(config["start"])
            < pd.Timestamp(config["is_end"]) < pd.Timestamp(config["end"])
        )
    ):
        raise ValueError("Invalid dates, lag or execution costs.")
    model = config["model"]
    validate_model(model)
    overnight, intraday, fetched_at = load_market(config)
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}", flush=True)

    risk_ticker = config["excess_reference"]
    reference = (overnight[risk_ticker] * intraday[risk_ticker] - 1).loc[config["start"]:]
    reference.iloc[0] = intraday.loc[reference.index[0], risk_ticker] - 1

    is_end = config["is_end"]
    robust = config["robustness"]
    print("ROBUSTNESS (IS ONLY)", flush=True)
    variants = [
        (field, value)
        for field in ("prelec_gamma", "risk_penalty")
        for value in robust[field]
    ]
    variants.append(("prelec_gamma", robust["equal_probability_gamma"]))
    trials = 0
    seen = {tuple(sorted(model.items()))}
    for field, value in variants:
        variant = {**model, field: value}
        key = tuple(sorted(variant.items()))
        if key in seen:
            continue
        seen.add(key)
        validate_model(variant)
        is_results = run_strategy(overnight.loc[:is_end], intraday.loc[:is_end], config, variant, friction)
        print(f"IS {field}={value}: " + json.dumps(summarize(is_results, reference), allow_nan=False),
              flush=True)
        trials += 1

    daily_results = run_strategy(overnight, intraday, config, model, friction)
    trials += 1
    ticker = config["benchmark"]
    first_date = daily_results.index[0]
    benchmark = simulate(
        overnight.loc[first_date:, [ticker]], intraday.loc[first_date:, [ticker]],
        {first_date: np.ones(1)}, friction,
    )
    official = split_summary(daily_results, reference, is_end)
    baseline = split_summary(benchmark, reference, is_end)
    if official["is"]["n_trades"] == 0 or official["oos"]["n_trades"] == 0:
        raise RuntimeError("Expected actual executions in both evaluation periods.")

    print(f"PARAM_COMBOS_TRIED: {trials}")
    print(
        f"LOOK_AHEAD_CHECK: trailing close windows and {lag}-session execution lag; "
        "public membership announcements precede signals; BIL is scored ex post; "
        "split-adjusted relative prices and ex-date cash at the open; IS-only variants"
    )
    for period in ("is", "oos", "full"):
        print(f"STRATEGY {period.upper()}: " + json.dumps(official[period], allow_nan=False))
        print(f"BENCHMARK {ticker} {period.upper()}: " + json.dumps(baseline[period], allow_nan=False))
    last_date = daily_results.index[-1]
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": first_date.date().isoformat(), "end": last_date.date().isoformat()},
        **official,
        "benchmark": baseline["full"],
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    signal.alarm(14 * 60)
    main()
