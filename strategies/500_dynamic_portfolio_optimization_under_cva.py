# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "500_ dynamic_portfolio_optimization_under_cva"\nuniverse:\n  risky: SPY\n  defensive: BIL\ndata_start: 2007-06-01\nbacktest_start: 2008-07-01\nis_end: 2017-12-31\nbacktest_end: 2025-12-31\n\nexecution_lag_sessions: 1\ncommission_bps: 1.0\nslippage_bps: 2.0\n\nmodel:\n  lookback_sessions: 252\n  horizon_sessions: 21\n  cvar_confidence: 0.95\n  risk_aversion: 5.0\n  cvar_limit: 0.04\n  drawdown_threshold: 0.15\n  max_budget_cut: 0.50\n  weight_intervals: 200\n\nrobustness:\n  cvar_limits: [0.03, 0.05]\n  drawdown_thresholds: [0.10, 0.20]\n  friction_multiplier: 5.0\n'
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
"""Research backtest of monthly state-dependent CVaR sizing for SPY and BIL."""

import json
import re
import signal
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
PRICE_ADJUSTMENT = "yfinance auto_adjust=True daily close, split and dividend adjusted"


@dataclass(frozen=True)
class Rules:
    lookback_sessions: int
    horizon_sessions: int
    cvar_confidence: float
    risk_aversion: float
    cvar_limit: float
    drawdown_threshold: float
    max_budget_cut: float
    weight_intervals: int
    commission_bps: float
    slippage_bps: float

    def __post_init__(self):
        if not 1 <= self.horizon_sessions < self.lookback_sessions:
            raise ValueError("The scenario horizon must be shorter than the lookback.")
        if not 0 < self.cvar_confidence < 1 or self.weight_intervals < 1:
            raise ValueError("Use a confidence in (0, 1) and a positive grid size.")
        if not 0 < self.cvar_limit < 1 or not 0 <= self.risk_aversion < np.inf:
            raise ValueError("Use a positive risk limit and finite nonnegative risk aversion.")
        if not 0 < self.drawdown_threshold <= 1 or not 0 <= self.max_budget_cut < 1:
            raise ValueError("Invalid drawdown scaling parameters.")
        if not (self.commission_bps >= 0 and self.slippage_bps >= 0 and self.friction < 1):
            raise ValueError("Costs must be nonnegative and sum to less than 10,000 bps.")

    @property
    def friction(self):
        return (self.commission_bps + self.slippage_bps) / 10_000


def check_prices(prices, tickers, start, end):
    if prices.empty or list(prices.columns) != tickers:
        raise ValueError(f"Missing adjusted close data for {', '.join(tickers)}.")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and sorted.")
    if prices.index[0] < start or prices.index[-1] > end:
        raise ValueError("Price dates fall outside the requested range.")
    margin = pd.Timedelta(days=7)
    if prices.index[0] - start > margin or end - prices.index[-1] > margin:
        raise ValueError("The price history does not cover the requested dates.")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Aligned adjusted closes contain a missing or invalid price.")


def load_prices(tickers, start, end):
    CACHE_DIR.mkdir(exist_ok=True)
    stem = "_".join([*tickers, start.date().isoformat(), end.date().isoformat()])
    cache_path = CACHE_DIR / f"adjusted_close_{stem}.csv"
    if cache_path.is_file():
        try:
            with cache_path.open(encoding="utf-8") as cached:
                stamp = cached.readline().strip().removeprefix("# fetched_at=")
                if datetime.fromisoformat(stamp).tzinfo is None:
                    raise ValueError("Fetch timestamp must include its time zone.")
                prices = pd.read_csv(cached, index_col="date", parse_dates=["date"])
            check_prices(prices, tickers, start, end)
            return prices, stamp
        except (KeyError, OSError, TypeError, ValueError) as exc:
            print(f"CACHE_RELOAD: {exc}", flush=True)

    with TemporaryDirectory(dir=CACHE_DIR) as fetch_dir:
        yf.set_tz_cache_location(fetch_dir)
        data = yf.download(
            tickers, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=True, repair=False, keepna=True,
            ignore_tz=True, threads=False, progress=False, timeout=15,
        )
    if data is None or data.empty or "Close" not in data.columns.get_level_values(0):
        raise RuntimeError("yfinance returned no adjusted close history.")
    prices = data["Close"].reindex(columns=tickers).sort_index()
    check_prices(prices, tickers, start, end)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE_DIR, delete=False) as out:
            temporary_path = Path(out.name)
            out.write(f"# fetched_at={stamp}\n")
            prices.to_csv(out, index_label="date")
        temporary_path.replace(cache_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return prices, stamp


def upper_tail_mean(losses, confidence):
    """Empirical CVaR with fractional mass at the cutoff, one row per weight."""
    tail_mass = (1 - confidence) * losses.shape[1]
    count = int(np.ceil(tail_mass))
    cutoff = losses.shape[1] - count
    tail = np.partition(losses, cutoff, axis=1)[:, cutoff:]
    return (tail.sum(axis=1) - (count - tail_mass) * tail[:, 0]) / tail_mass


def choose_target(price_window, drawdown, rules, candidates):
    horizon = rules.horizon_sessions
    scenarios = price_window[horizon:] / price_window[:-horizon] - 1
    portfolio_returns = candidates @ scenarios.T
    tail_risk = upper_tail_mean(-portfolio_returns, rules.cvar_confidence)
    budget_cut = rules.max_budget_cut * min(drawdown / rules.drawdown_threshold, 1)
    budget = rules.cvar_limit * (1 - budget_cut)
    feasible = tail_risk <= budget
    if not feasible.any():
        return candidates[np.argmin(tail_risk)]
    score = portfolio_returns.mean(axis=1) - rules.risk_aversion / 2 * portfolio_returns.var(axis=1)
    return candidates[np.argmax(np.where(feasible, score, -np.inf))]


def execute(arriving, target, friction):
    """Return remaining growth fraction, two-sided turnover, and executed leg count."""
    if np.allclose(arriving, target, rtol=0, atol=1e-12):
        return 1.0, 0.0, 0

    def residual(remaining):
        return remaining + friction * np.abs(target * remaining - arriving).sum() - 1

    remaining = brentq(residual, 0, 1, xtol=1e-14)
    traded = np.abs(target * remaining - arriving)
    return remaining, traded.sum(), int(np.count_nonzero(traded > 1e-12))


def simulate(prices, start, lag, rules, buy_and_hold=False):
    levels = prices.to_numpy(dtype=float)
    first = prices.index.searchsorted(start)
    if first < rules.lookback_sessions:
        raise ValueError("Extend data_start to provide the complete historical lookback.")
    if len(prices) - first <= lag:
        raise ValueError("The evaluation period is shorter than the execution lag.")

    months = prices.index.to_period("M")
    asset_returns = levels[1:] / levels[:-1] - 1
    risk_weights = np.linspace(0, 1, rules.weight_intervals + 1)
    candidates = np.column_stack((risk_weights, 1 - risk_weights))
    weights = np.zeros(2)
    growth = peak = 1.0
    pending = {}
    rows = []
    for i in range(first, len(prices)):
        daily_move = asset_returns[i - 1]
        gross = 1 + weights @ daily_move
        weights = weights * (1 + daily_move) / gross
        traded, trades = 0.0, 0
        if i in pending:
            target = pending.pop(i)
            remaining, traded, trades = execute(weights, target, rules.friction)
            gross *= remaining
            weights = target

        growth *= gross
        peak = max(peak, growth)
        rows.append((gross - 1, daily_move[1], traded, trades))

        # The return ending at a fill already belongs to the earlier weights.
        signal_day = i == first or months[i] != months[i - 1]
        if signal_day and i + lag < len(prices):
            if buy_and_hold:
                if i == first:
                    pending[i + lag] = np.array([1.0, 0.0])
            else:
                window = levels[i - rules.lookback_sessions:i + 1]
                target = choose_target(window, 1 - growth / peak, rules, candidates)
                pending[i + lag] = target

    return pd.DataFrame(rows, index=prices.index[first:],
                        columns=["return", "bill_return", "turnover", "n_trades"])


def metrics(daily_results):
    returns = daily_results["return"].to_numpy()
    if len(returns) < 2:
        raise ValueError("Each evaluation period must contain at least two sessions.")
    excess = returns - daily_results["bill_return"].to_numpy()
    growth = np.r_[1.0, np.cumprod(1 + returns)]
    daily_vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    scale = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(scale * daily_vol),
        "sharpe": float(scale * returns.mean() / daily_vol) if daily_vol > 0 else None,
        "sharpe_excess": float(scale * excess.mean() / excess_vol) if excess_vol > 0 else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def robustness_cases(rules, settings):
    cases = {}
    for limit in settings["cvar_limits"]:
        cases[f"cvar_limit={limit:g}"] = replace(rules, cvar_limit=limit)
    for threshold in settings["drawdown_thresholds"]:
        cases[f"drawdown_threshold={threshold:g}"] = replace(rules, drawdown_threshold=threshold)
    multiplier = settings["friction_multiplier"]
    if multiplier <= 1:
        raise ValueError("The higher-friction case requires a multiplier above one.")
    cases[f"commission_and_slippage_x{multiplier:g}"] = replace(
        rules, commission_bps=rules.commission_bps * multiplier,
        slippage_bps=rules.slippage_bps * multiplier,
    )
    seen = {rules}
    for label, case in cases.items():
        if case not in seen:
            seen.add(case)
            yield label, case


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    risky, defensive = (config["universe"][key] for key in ("risky", "defensive"))
    if defensive != "BIL" or risky == defensive or not re.fullmatch(r"[A-Z]{1,5}", risky):
        raise ValueError("The universe requires a US equity ETF and BIL.")
    data_start, start, split, end = (
        pd.Timestamp(config[key]) for key in ("data_start", "backtest_start", "is_end", "backtest_end")
    )
    if not data_start < start <= split < end:
        raise ValueError("Check the configured data, evaluation, and split dates.")
    lag = config["execution_lag_sessions"]
    if type(lag) is not int or lag < 1:
        raise ValueError("Execution requires at least one full session of lag.")
    rules = Rules(**config["model"], commission_bps=config["commission_bps"],
                  slippage_bps=config["slippage_bps"])
    cases = list(robustness_cases(rules, config["robustness"]))

    prices, fetched_at = load_prices([risky, defensive], data_start, end)
    official = simulate(prices, start, lag, rules)
    comparison = simulate(prices, start, lag, rules, buy_and_hold=True)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print("ROBUSTNESS (IS only, one change per case):")
    in_sample_prices = prices.loc[:split]
    for label, variant in cases:
        variant_results = simulate(in_sample_prices, start, lag, variant)
        print(f"ROBUSTNESS {label}: {json.dumps(metrics(variant_results), allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {1 + len(cases)}")

    results = {
        "strategy": config["strategy"],
        "data_range": {"start": official.index[0].date().isoformat(),
                       "end": official.index[-1].date().isoformat()},
    }
    print(f"BENCHMARK: {risky} buy and hold, same initial fill date and friction")
    is_mask = official.index <= split
    for period, selected in (("is", is_mask), ("oos", ~is_mask), ("full", slice(None))):
        results[period] = metrics(official.loc[selected])
        benchmark = metrics(comparison.loc[selected])
        print(f"{period.upper()}: {json.dumps(results[period], allow_nan=False)}")
        print(f"BENCHMARK_{period.upper()}: {json.dumps(benchmark, allow_nan=False)}")
    results["benchmark"] = {"name": risky, **benchmark}
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(840)
    main()
