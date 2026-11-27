# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 453_relief_gated_relative_rotation_for_qqq_d\nuniverse:\n  growth: QQQ\n  dow: DIA\n  market: SPY\n  credit_risk: HYG\n  short_treasury: SHY\n  volatility: '^VIX'\n  risk_free: BIL\n  treasury_series: DGS10\ndates:\n  data_start: '2007-06-01'\n  backtest_start: '2013-01-02'\n  oos_start: '2026-07-10'\n  end_inclusive: '2026-11-25'\nexecution:\n  signal_lag_sessions: 1\n  fred_lag_sessions: 3\n  vix_lag_sessions: 1\n  cost_bps: 10.0\n  slippage_bps: 2.0\nmodel:\n  standardization_min_sessions: 252\n  momentum_sessions: 126\n  relief_sessions: 21\n  drawdown_sessions: 126\n  vix_percentile_sessions: 252\n  max_tilt: 0.50\n  tanh_scale: 0.75\n  smoothing: 0.05\n  lambda_grid: [0.25, 0.50, 0.75, 1.00]\n  fit_sessions: 756\n  annual_turnover_target: 3.0\n  turnover_penalty: 0.08\n"
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

import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252

# Published interaction terms and directions, from Tables 12 and 22.
SECOND_ORDER = (
    (-1, ("vix_relief", "credit_stress")),
    (-1, ("rel_momentum", "credit_relief")),
    (-1, ("rate_relief", "low_vix")),
    (-1, ("rate_relief", "credit_relief")),
    (-1, ("rel_reversal", "high_vix")),
    (1, ("rate_relief", "spy_drawdown")),
    (1, ("rel_reversal", "rate_relief")),
    (-1, ("credit_relief", "credit_stress")),
    (1, ("rel_reversal", "credit_relief")),
)
THIRD_ORDER = (
    (-1, ("rel_momentum", "rel_reversal", "rate_relief")),
    (1, ("rel_reversal", "rate_relief", "spy_drawdown")),
)


def store_csv(frame: pd.DataFrame, path: Path) -> None:
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        temp_path = Path(output.name)
        try:
            frame.to_csv(output, index_label="date")
            output.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=["date"]).sort_index()


def fetch_data(universe, start, end):
    CACHE.mkdir(exist_ok=True)
    symbols = sorted(set(universe.values()) - {universe["treasury_series"]})
    price_key = "_".join(symbols).replace("^", "index-")
    price_path = CACHE / f"yahoo_adjusted_{price_key}_{start.date()}_{end.date()}.csv"
    if price_path.exists():
        prices = read_csv(price_path)
    else:
        with TemporaryDirectory(dir=CACHE) as session_cache:
            yf.set_tz_cache_location(session_cache)
            for _ in range(3):
                raw = yf.download(
                    symbols, start=str(start.date()),
                    end=str((end + pd.Timedelta(days=1)).date()),
                    auto_adjust=True, progress=False, timeout=15,
                )
                prices = raw["Close"].reindex(columns=symbols)
                # Use ETF sessions even when the volatility feed has holiday rows.
                prices = prices.dropna(subset=[universe["growth"], universe["dow"]], how="all")
                if not prices.empty and prices.notna().all().all():
                    break

    incomplete = (
        prices.empty or not prices.index.is_unique
        or prices.index[0] > start + pd.Timedelta(days=7)
        or prices.index[-1] < end - pd.Timedelta(days=4)
        or not np.isfinite(prices.to_numpy()).all()
        or (prices <= 0).any().any()
    )
    if incomplete:
        raise RuntimeError("Adjusted market close history is incomplete.")
    if not price_path.exists():
        store_csv(prices, price_path)

    series = universe["treasury_series"]
    rate_path = CACHE / f"fred_{series}_{start.date()}_{end.date()}.csv"
    if rate_path.exists():
        rates = read_csv(rate_path)
    else:
        query = urlencode({"id": series, "cosd": str(start.date()), "coed": str(end.date())})
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{query}"
        for attempt in range(3):
            try:
                with urlopen(url, timeout=15) as response:
                    rates = pd.read_csv(response, index_col=0, parse_dates=True)
                break
            except OSError:
                if attempt == 2:
                    raise
        rates = rates[[series]].apply(pd.to_numeric, errors="coerce").dropna().sort_index()
    if (rates.empty or not rates.index.is_unique
            or rates.index[0] > start + pd.Timedelta(days=7)
            or rates.index[-1] < end - pd.Timedelta(days=14)
            or not np.isfinite(rates.to_numpy()).all()):
        raise RuntimeError(f"FRED {series} history is incomplete.")
    if not rate_path.exists():
        store_csv(rates, rate_path)

    latest_write = max(price_path.stat().st_mtime, rate_path.stat().st_mtime)
    fetched_at = datetime.fromtimestamp(latest_write, timezone.utc).isoformat()
    return prices.loc[start:end, symbols], rates[series], fetched_at


def expanding_z(values, minimum):
    history = values.expanding(min_periods=minimum)
    mean = history.mean().shift(1)
    deviation = history.std(ddof=0).shift(1).replace(0.0, np.nan)
    return (values - mean) / deviation


def drawdown_depth(values, lookback):
    return 1.0 - values / values.rolling(lookback).max()


def group_scores(prices, rates, universe, model, execution):
    ratio = prices[universe["growth"]] / prices[universe["dow"]]
    vix = prices[universe["volatility"]].shift(execution["vix_lag_sessions"])
    available_rate = rates.reindex(prices.index, method="ffill").shift(execution["fred_lag_sessions"])
    relief = model["relief_sessions"]
    credit = prices[universe["credit_risk"]].pct_change(relief)
    treasury = prices[universe["short_treasury"]].pct_change(relief)
    raw = pd.DataFrame({
        "rel_momentum": ratio.pct_change(model["momentum_sessions"]),
        "rel_reversal": drawdown_depth(ratio, model["drawdown_sessions"]),
        "rate_relief": -available_rate.diff(relief),
        "spy_drawdown": drawdown_depth(prices[universe["market"]], model["drawdown_sessions"]),
        "high_vix": vix.rolling(model["vix_percentile_sessions"]).rank(pct=True),
        "vix_relief": -vix.diff(relief),
        "credit_relief": credit - treasury,
    })
    minimum = model["standardization_min_sessions"]
    states = expanding_z(raw, minimum)
    states["low_vix"] = -states["high_vix"]
    states["credit_stress"] = -states["credit_relief"]

    groups = {"main": states["rate_relief"]}
    for group, terms in (("ix2", SECOND_ORDER), ("ix3", THIRD_ORDER)):
        interactions = pd.DataFrame({
            factors: direction * states[list(factors)].prod(axis=1, skipna=False)
            for direction, factors in terms
        })
        groups[group] = expanding_z(interactions, minimum).mean(axis=1, skipna=False)
    return expanding_z(pd.DataFrame(groups), minimum)


def target_weights(groups, lambdas, model, signal_lag):
    score = expanding_z(groups.dot(np.asarray(lambdas)), model["standardization_min_sessions"])
    target = 0.5 + model["max_tilt"] * np.tanh(score / model["tanh_scale"])
    return target.shift(signal_lag)


def simulate(asset_returns, targets, smoothing, cost_rate, initial_target):
    sessions = asset_returns.index
    asset_values = asset_returns.to_numpy()
    target_values = targets.reindex(sessions).to_numpy()
    if not np.isfinite(target_values).all():
        raise RuntimeError("Increase the feature warmup before starting the backtest.")
    model_weight = initial_target
    held_weight = initial_target
    records = []
    for i, ((growth_ret, dow_ret), target) in enumerate(zip(asset_values, target_values)):
        model_weight += smoothing * (target - model_weight)
        if i == 0:
            gross = 1.0
            turnover = 1.0
            trades = int(model_weight > 0.0) + int(model_weight < 1.0)
        else:
            growth_value = held_weight * (1.0 + growth_ret)
            gross = growth_value + (1.0 - held_weight) * (1.0 + dow_ret)
            drifted_weight = growth_value / gross
            turnover = 2.0 * abs(model_weight - drifted_weight)
            trades = 2 if turnover > 1e-12 else 0
        records.append((gross * (1.0 - cost_rate * turnover) - 1.0, turnover, trades))
        held_weight = model_weight
    return pd.DataFrame(records, index=sessions, columns=["net", "turnover", "trades"])


def metrics(path, cash_returns):
    daily = path["net"]
    excess = daily - cash_returns.loc[daily.index]
    scale = np.sqrt(SESSIONS_PER_YEAR)
    wealth = (1.0 + daily).cumprod()
    volatility = daily.std(ddof=0)
    excess_volatility = excess.std(ddof=0)
    return {
        "cagr": float(wealth.iloc[-1] ** (SESSIONS_PER_YEAR / len(daily)) - 1.0),
        "ann_vol": float(volatility * scale),
        "sharpe": float(daily.mean() / volatility * scale) if volatility > 0.0 else None,
        "sharpe_excess": (float(excess.mean() / excess_volatility * scale)
                          if excess_volatility > 0.0 else None),
        "max_drawdown": float((wealth / wealth.cummax().clip(lower=1.0) - 1.0).min()),
        "total_return": float(wealth.iloc[-1] - 1.0),
        "turnover": float(path["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(path["trades"].sum()),
    }


def fit_lambdas(groups, returns, cash_returns, model, lag, cost_rate):
    combinations = list(product(dict.fromkeys(model["lambda_grid"]), repeat=3))
    rows = []
    for lambdas in combinations:
        targets = target_weights(groups, lambdas, model, lag)
        path = simulate(returns, targets, model["smoothing"], cost_rate, 0.5)
        rows.append(metrics(path.tail(model["fit_sessions"]), cash_returns))
    stats = pd.DataFrame(rows)
    stats["calmar"] = stats["cagr"] / stats["max_drawdown"].abs().replace(0.0, np.nan)
    stats["low_turnover"] = -stats["turnover"]
    rank_columns = ["cagr", "sharpe", "calmar", "max_drawdown", "low_turnover"]
    stats["rank_score"] = stats[rank_columns].rank(pct=True).fillna(0.0).mean(axis=1)
    excess_turnover = (stats["turnover"] - model["annual_turnover_target"]).clip(lower=0.0)
    stats["objective"] = stats["rank_score"] - model["turnover_penalty"] * excess_turnover
    winner = stats.sort_values(["objective", "rank_score", "cagr"], ascending=False).index[0]
    return combinations[winner], len(combinations)


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    model = config["model"]
    execution = config["execution"]
    dates = config["dates"]
    start, test_start, split, end = (
        pd.Timestamp(dates[key]) for key in ("data_start", "backtest_start", "oos_start", "end_inclusive")
    )
    if not start < test_start < split <= end:
        raise ValueError("Expected data_start < backtest_start < oos_start <= end_inclusive.")
    lag = execution["signal_lag_sessions"]
    if lag < 1 or execution["vix_lag_sessions"] < 1 or execution["fred_lag_sessions"] < 3:
        raise ValueError("Execution and VIX lags must be at least one session, and FRED lag at least three.")
    fees = execution["cost_bps"], execution["slippage_bps"]
    cost_rate = sum(fees) / 10_000
    if min(fees) < 0.0 or not 0.0 <= cost_rate < 0.5:
        raise ValueError("Invalid execution cost or slippage.")
    prices, rates, fetched_at = fetch_data(universe, start, end)
    closes = prices[[universe["growth"], universe["dow"], universe["risk_free"]]]
    returns = closes.pct_change().loc[test_start:end].dropna()
    if returns.empty:
        raise RuntimeError("No market sessions in the requested backtest.")
    asset_returns = returns[[universe["growth"], universe["dow"]]]
    cash_returns = returns[universe["risk_free"]].copy()
    cash_returns.iloc[0] = 0.0  # The portfolio is invested at the first close.
    is_mask = returns.index < split
    training = asset_returns.loc[is_mask].iloc[:-1]
    if len(training) < model["fit_sessions"] or (~is_mask).sum() < 2:
        raise RuntimeError("Insufficient IS training sessions or OOS observations.")

    groups = group_scores(prices, rates, universe, model, execution)
    # Leave the last IS close for the first post-fit allocation before OOS returns start.
    lambdas, trials = fit_lambdas(
        groups.loc[:training.index[-1]], training, cash_returns, model, lag, cost_rate,
    )
    targets = target_weights(groups, lambdas, model, lag)
    strategy = simulate(asset_returns, targets, model["smoothing"], cost_rate, 0.5)
    benchmark = simulate(asset_returns, pd.Series(1.0, index=returns.index), 1.0, cost_rate, 1.0)
    date_range = {"start": str(returns.index[0].date()), "end": str(returns.index[-1].date())}
    result = {
        "strategy": config["strategy"],
        "data_range": date_range,
        "is": metrics(strategy.loc[is_mask], cash_returns),
        "oos": metrics(strategy.loc[~is_mask], cash_returns),
        "full": metrics(strategy, cash_returns),
        "benchmark": metrics(benchmark, cash_returns),
    }

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, ETF split- and distribution-adjusted closes")
    print(f"PARAM_COMBOS_TRIED: {trials}")
    print(f"IS_LAMBDAS: {lambdas} (fit through {training.index[-1].date()})")
    print(f"Date Range: {date_range['start']} to {date_range['end']}")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(result[period], allow_nan=False)}")
    for period, mask in (("IS", is_mask), ("OOS", ~is_mask)):
        benchmark_metrics = metrics(benchmark.loc[mask], cash_returns)
        print(f"BENCHMARK ({universe['growth']} buy and hold, {period}): "
              f"{json.dumps(benchmark_metrics, allow_nan=False)}")
    print(f"BENCHMARK ({universe['growth']} buy and hold, FULL): "
          f"{json.dumps(result['benchmark'], allow_nan=False)}")
    print(f"RESULTS_JSON: {json.dumps(result, allow_nan=False)}")


if __name__ == "__main__":
    main()
