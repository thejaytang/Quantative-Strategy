# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "323_information_transmission_across_tail"\nuniverse:\n  oil: USO\n  equity: SPY\n  defensive: BIL\n  implied_vol: OVXCLS\ndata_start: "2008-01-01"\nin_sample:\n  start: "2014-01-01"\n  end: "2019-12-31"\nout_of_sample:\n  start: "2020-01-01"\n  end: "2026-11-20"\nmodel:\n  training_sessions: 504\n  forecast_sessions: 5\n  oil_variance_windows: [1, 5, 22]\n  equity_quantile_window: 252\n  equity_tail_quantile: 0.05\n  equity_tail_window: 20\n  ovx_lag_sessions: 1\n  ovx_max_fill_sessions: 5\n  ridge_alpha: 1.0\nallocation:\n  gate_history_weeks: 52\n  gate_quantile: 0.80\n  defensive_weight: 0.50\n  transaction_cost_bps: 5.0\nsessions_per_year: 252\n'
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
"""Weekly USO allocation using an incremental SPY downside-tail forecast."""

import sys

sys.dont_write_bytecode = True

import json
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import requests
import yaml


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
HTTP_TIMEOUT = (10, 30)


def public_csv(url, params):
    response = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text), index_col=0, parse_dates=True)


def yahoo_prices(symbol, start, end):
    """Read the keyless Yahoo Finance date-filtered historical price table."""
    period1 = int(start.tz_localize("UTC").timestamp())
    period2 = int((end + pd.Timedelta(days=1)).tz_localize("UTC").timestamp())
    response = requests.get(
        f"https://finance.yahoo.com/quote/{symbol}/history/",
        params={"period1": period1, "period2": period2},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    table = pd.read_html(StringIO(response.text), flavor="lxml")[0]
    close_column = next(c for c in table if c.startswith("Close "))
    adjusted_column = next(c for c in table if c.startswith("Adj Close "))
    # Corporate-action rows have a date but no numeric OHLC fields.
    raw_open, close, adjusted = (
        pd.to_numeric(table[column], errors="coerce")
        for column in ("Open", close_column, adjusted_column)
    )
    dates = pd.to_datetime(table["Date"], format="%b %d, %Y", errors="coerce")
    prices = pd.DataFrame(
        {"open": (raw_open * (adjusted / close)).to_numpy(), "close": adjusted.to_numpy()},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    return prices.loc[prices.index.notna()].dropna(how="all")


def cached_data(source, symbol, start, end):
    path = CACHE / f"{source}_{symbol}_{start.date()}_{end.date()}.csv"
    cache_hit = path.is_file()
    if cache_hit:
        frame = pd.read_csv(path, index_col="date", parse_dates=True)
    elif source == "yahoo":
        frame = yahoo_prices(symbol, start, end)
    else:
        frame = public_csv(
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            {"id": symbol, "cosd": str(start.date()), "coed": str(end.date())},
        )
        frame[symbol] = pd.to_numeric(frame[symbol], errors="coerce")

    frame = frame.sort_index().loc[start:end].rename_axis("date")
    observations = frame.dropna(how="all")
    grace = pd.Timedelta(days=5)
    if (observations.empty or frame.index.has_duplicates
            or observations.index[0] > start + grace
            or observations.index[-1] < end - grace):
        raise ValueError(f"Incomplete or duplicate {symbol} history for {start.date()} to {end.date()}")
    if source == "yahoo" and (not np.isfinite(frame).all().all() or (frame <= 0).any().any()):
        raise ValueError(f"Invalid OHLC history for {symbol}")
    if not cache_hit:
        with TemporaryDirectory(dir=CACHE) as directory:
            temporary_path = Path(directory) / path.name
            frame.to_csv(temporary_path)
            temporary_path.replace(path)
    return frame


def load_market_data(universe, start, end):
    CACHE.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        downloads = {
            role: pool.submit(
                cached_data, "fred" if role == "implied_vol" else "yahoo", symbol, start, end
            )
            for role, symbol in universe.items()
        }
        data = {role: future.result() for role, future in downloads.items()}

    sessions = data["oil"].index
    for role in ("equity", "defensive"):
        if not data[role].index.equals(sessions):
            raise ValueError(f"{universe[role]} and {universe['oil']} have different price sessions")
    return data


def ridge_predict(training_features, targets, current_features, alpha):
    center = training_features.mean(axis=0)
    scale = training_features.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (training_features - center) / scale
    current = (current_features - center) / scale
    target_mean = targets.mean()
    penalty = alpha * np.eye(standardized.shape[1])
    coefficients = np.linalg.solve(
        standardized.T @ standardized + penalty,
        standardized.T @ (targets - target_mean),
    )
    return float(target_mean + current @ coefficients)


def weekly_forecasts(data, model, annual_sessions):
    oil_variance = data["oil"]["close"].pct_change().pow(2)
    sessions = oil_variance.index
    equity_returns = data["equity"]["close"].pct_change()
    cutoff = equity_returns.shift(1).rolling(model["equity_quantile_window"]).quantile(
        model["equity_tail_quantile"]
    )
    tail_days = (equity_returns < cutoff).astype(float).where(cutoff.notna())
    equity_tail = tail_days.rolling(model["equity_tail_window"]).mean()
    ovx = data["implied_vol"].iloc[:, 0].reindex(sessions)
    ovx = ovx.ffill(limit=model["ovx_max_fill_sessions"]).shift(model["ovx_lag_sessions"])
    features = pd.DataFrame({
        f"oil_{n}": oil_variance.rolling(n).mean() for n in model["oil_variance_windows"]
    })
    features["ovx"] = (ovx / 100).pow(2) / annual_sessions
    features["equity_tail"] = equity_tail

    horizon = model["forecast_sessions"]
    target = oil_variance.rolling(horizon).mean().shift(-horizon).to_numpy()
    feature_values = features.to_numpy()
    width = model["training_sessions"]
    week = sessions.to_period("W-FRI")
    # The first session of a new week trades on the preceding session's signal.
    prior_week_closes = np.flatnonzero(week[:-1] != week[1:])
    forecasts = []
    for position in prior_week_closes:
        # Origin s has a known label at t exactly when s + horizon <= t.
        train_end = position - horizon + 1
        train_start = train_end - width
        if train_start < 0:
            continue
        train = feature_values[train_start:train_end]
        labels = target[train_start:train_end]
        current = feature_values[position]
        if not (np.isfinite(train).all() and np.isfinite(labels).all() and np.isfinite(current).all()):
            continue
        own = ridge_predict(train[:, :-1], labels, current[:-1], model["ridge_alpha"])
        cross = ridge_predict(train, labels, current, model["ridge_alpha"])
        target_end = position + horizon
        forecasts.append({
            "signal": sessions[position],
            "execution": sessions[position + 1],
            "target_end": sessions[target_end] if target_end < len(sessions) else pd.NaT,
            "realized": target[position],
            "own": own,
            "cross": cross,
        })
    if not forecasts:
        raise ValueError("There are no weekly forecasts with a complete training window")
    weekly = pd.DataFrame(forecasts).set_index("execution")
    expected = sessions[prior_week_closes + 1]
    return weekly.reindex(expected[expected >= weekly.index[0]])


def allocation_signals(weekly, config):
    history = config["gate_history_weeks"]
    incremental = weekly["cross"] - weekly["own"]
    tail_cutoff = incremental.shift(1).rolling(history).quantile(config["gate_quantile"])
    tail_gate = (incremental > tail_cutoff).astype(float).where(
        incremental.notna() & tail_cutoff.notna()
    )

    # Match the cross gate's observed past activation rate using own forecast ranks.
    target_rate = tail_gate.shift(1).rolling(history).mean()
    own_gate = pd.Series(index=weekly.index, dtype=float)
    own_forecasts = weekly["own"].to_numpy()
    for i in np.flatnonzero(target_rate.notna() & weekly["own"].notna()):
        rate = target_rate.iloc[i]
        past_forecasts = own_forecasts[i - history:i]
        if rate == 0 or rate == 1:
            own_gate.iloc[i] = rate
        else:
            own_gate.iloc[i] = float(own_forecasts[i] > np.quantile(past_forecasts, 1 - rate))

    gates = pd.DataFrame({"tail": tail_gate, "own": own_gate})
    return 1 - config["defensive_weight"] * gates


def rebalance(holdings, equity, weights, fee):
    # Solve for invested wealth after paying fees on the resulting dollar orders.
    invested = equity
    for _ in range(12):
        desired = invested * weights
        orders = np.abs(desired - holdings)
        after_fee = equity - fee * orders.sum()
        if abs(after_fee - invested) <= equity * 1e-12:
            return desired, orders
        invested = after_fee
    raise ArithmeticError("Transaction cost calculation failed to converge")


def simulate(oil, defensive, oil_target, rebalance_dates, fee):
    opening = np.column_stack((oil["open"], defensive["open"]))
    closing = np.column_stack((oil["close"], defensive["close"]))
    holdings = np.zeros(2)
    wealth = 1.0
    records = []
    for i, target in enumerate(oil_target):
        if i:
            holdings *= opening[i] / closing[i - 1]
        equity = holdings.sum() if i else wealth
        orders = np.zeros(2)
        if i == 0 or oil_target.index[i] in rebalance_dates:
            holdings, orders = rebalance(holdings, equity, np.array([target, 1 - target]), fee)
        oil_weight = holdings[0] / holdings.sum()
        holdings *= closing[i] / opening[i]
        close_wealth = holdings.sum()
        records.append((close_wealth / wealth - 1, orders.sum() / equity,
                        (orders > equity * 1e-10).sum(), oil_weight, target < 1))
        wealth = close_wealth
    return pd.DataFrame(records, index=oil_target.index, columns=[
        "return", "turnover", "n_trades", "avg_uso_weight", "de_risk_share",
    ])


def performance(daily, annual_sessions):
    returns = daily["return"]
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax().clip(lower=1) - 1
    daily_vol = returns.std()
    return {
        "cagr": float(equity.iloc[-1] ** (annual_sessions / len(returns)) - 1),
        "ann_vol": float(daily_vol * np.sqrt(annual_sessions)) if np.isfinite(daily_vol) else None,
        "sharpe": float(returns.mean() / daily_vol * np.sqrt(annual_sessions)) if daily_vol > 0 else None,
        "max_drawdown": float(drawdown.min()),
        "total_return": float(equity.iloc[-1] - 1),
        "n_trades": int(daily["n_trades"].sum()),
        "turnover": float(daily["turnover"].sum()),
        "avg_uso_weight": float(daily["avg_uso_weight"].mean()),
        "de_risk_share": float(daily["de_risk_share"].mean()),
    }


def forecast_error(weekly, periods):
    rows = {}
    for period, (start, end) in periods.items():
        sample = weekly.loc[weekly["signal"].between(start, end) & (weekly["target_end"] <= end)]
        scores = {"n_forecasts": len(sample)}
        for model in ("cross", "own"):
            residuals = (sample[model].clip(lower=0) - sample["realized"]) * 10_000
            scores[f"{model}_rmse"] = float(np.sqrt(residuals.pow(2).mean())) if len(sample) else None
            scores[f"{model}_mae"] = float(residuals.abs().mean()) if len(sample) else None
        rows[period] = scores
    return rows


def print_results(results, portfolios):
    rows = [
        {"portfolio": name, "period": period.upper(), **values}
        for name, metrics in portfolios.items()
        for period, values in metrics.items()
    ]
    table = pd.DataFrame(rows).set_index(["portfolio", "period"])
    percentage = ["cagr", "ann_vol", "max_drawdown", "total_return", "avg_uso_weight", "de_risk_share"]
    formatters = {column: "{:.2%}".format for column in percentage}
    print(f"{results['strategy']} | return dates {results['data_range']}")
    print("\nNet portfolio performance (turnover in multiples of capital)")
    print(table.to_string(formatters=formatters, float_format="{:.3f}".format, na_rep="n/a"))
    print("\nWeekly five-session variance forecast errors (squared percentage points)")
    print(pd.DataFrame.from_dict(results["forecast_error"], orient="index").rename(index=str.upper).to_string(
        float_format="{:.4f}".format, na_rep="n/a"
    ))
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not is_start < is_end < oos_start < oos_end or oos_start - is_end != pd.Timedelta(days=1):
        raise ValueError("Configure adjacent, non-overlapping IS and OOS date ranges")
    periods = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    annual_sessions = config["sessions_per_year"]
    data = load_market_data(config["universe"], pd.Timestamp(config["data_start"]), oos_end)
    weekly = weekly_forecasts(data, config["model"], annual_sessions)
    signals = allocation_signals(weekly, config["allocation"])

    oil = data["oil"].loc[is_start:oos_end]
    defensive = data["defensive"].reindex(oil.index)
    fee = config["allocation"]["transaction_cost_bps"] / 10_000
    if not 0 <= fee < 0.01:
        raise ValueError("transaction_cost_bps must be in the range [0, 100)")
    if not 0 <= config["allocation"]["defensive_weight"] <= 1:
        raise ValueError("defensive_weight must be in the range [0, 1]")
    portfolios = {}
    for name, scheduled in {
        "tail_gate": signals["tail"].dropna(),
        "own_market_gate": signals["own"].dropna(),
        "buy_and_hold_uso": pd.Series(1.0, index=oil.index[:1]),
    }.items():
        target = scheduled.reindex(oil.index, method="ffill")
        if target.empty or pd.isna(target.iloc[0]):
            raise ValueError(f"Insufficient pre-IS history for {name}")
        daily = simulate(oil, defensive, target, frozenset(scheduled.index), fee)
        portfolios[name] = {
            period: performance(daily.loc[start:end], annual_sessions)
            for period, (start, end) in periods.items()
        }

    results = {
        "strategy": config["strategy"],
        "data_range": f"{oil.index[0].date()} to {oil.index[-1].date()}",
        **portfolios["tail_gate"],
        "benchmark": {name: portfolios[name] for name in ("buy_and_hold_uso", "own_market_gate")},
        "forecast_error": forecast_error(weekly, periods),
    }
    print_results(results, portfolios)


if __name__ == "__main__":
    main()
