# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 195_century_reversals_volatility\nuniverse:\n  market: SPY\n  reserve: BIL\n  volatility: '^VIX'\ndata_start: '2006-10-01'  # Includes the VIX warm-up before the first test session.\nin_sample:\n  start: '2008-01-02'\n  end: '2017-12-31'\nout_of_sample:\n  start: '2018-01-01'\n  end: '2025-12-31'\nsignal:\n  drop_threshold: -0.0075\n  vix_window_sessions: 252\n  vix_quantile: 0.75\nevent_drop_band_edges: [0.01, 0.02, 0.03]  # Absolute loss sizes, for diagnostics only.\ncosts:\n  one_way_bps: 5\n  sensitivity_bps: [0, 15]\nannualization_sessions: 252\n"
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
import math
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"


def download_prices(market, reserve, volatility, start, end, first_test):
    columns = [(ticker, field) for ticker in (market, reserve) for field in ("Open", "Close")]
    columns.append((volatility, "Close"))
    cache_key = hashlib.sha256(repr((columns, start, end)).encode()).hexdigest()[:16]
    cache_file = CACHE / f"yahoo_adjusted_{cache_key}.csv"
    if cache_file.exists():
        prices = pd.read_csv(cache_file, header=[0, 1], index_col=0, parse_dates=True)
        validate_prices(prices, market, reserve, volatility, first_test, end)
        return prices

    CACHE.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(CACHE / "yfinance"))
    last_error = ""
    for _ in range(2):
        try:
            raw = yf.download(
                [market, reserve, volatility], start=start, end=end + pd.Timedelta(days=1),
                auto_adjust=True, group_by="ticker", multi_level_index=True,
                interval="1d", threads=True, progress=False, timeout=12,
            )
            prices = raw.reindex(columns=pd.MultiIndex.from_tuples(columns)).sort_index()
            prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
            validate_prices(prices, market, reserve, volatility, first_test, end)
            break
        except (OSError, ValueError) as error:
            last_error = str(error)
    else:
        raise ValueError(f"Yahoo Finance download failed after two attempts. {last_error}")

    with NamedTemporaryFile(dir=CACHE, suffix=".csv", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        prices.to_csv(temp_path, index_label="Date")
        temp_path.replace(cache_file)
    finally:
        temp_path.unlink(missing_ok=True)
    return prices


def validate_prices(prices, market, reserve, volatility, start, end):
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and sorted.")
    market_prices = prices[market]
    market_sessions = market_prices.dropna(how="all").loc[start:end]
    if (market_sessions.empty
            or market_sessions.index[0] > start + pd.Timedelta(days=7)
            or market_sessions.index[-1] < end - pd.Timedelta(days=7)):
        raise ValueError(f"Incomplete {market} history for {start.date()} through {end.date()}.")
    reserve_sessions = prices[reserve].loc[start:end].dropna(how="all")
    sessions = market_sessions.index.union(reserve_sessions.index)
    for ticker in (market, reserve):
        etf = prices[ticker].reindex(sessions)
        if not (np.isfinite(etf) & etf.gt(0)).to_numpy().all():
            raise ValueError(f"Missing or invalid {ticker} open/close on an ETF session.")
    vix = prices[(volatility, "Close")].dropna()
    if (vix.empty or vix.index[-1] < end - pd.Timedelta(days=7)
            or not (np.isfinite(vix) & vix.gt(0)).all()):
        raise ValueError(f"Incomplete or invalid {volatility} close history.")


def make_sessions(prices, market, reserve, volatility, signal, start, end):
    spy = prices[market].dropna(how="all")
    bil = prices[reserve].reindex(spy.index)
    vix = prices[(volatility, "Close")].reindex(spy.index)
    cutoff = vix.rolling(signal["vix_window_sessions"]).quantile(signal["vix_quantile"])
    lagged_vix = vix.shift(1)
    cutoff = cutoff.shift(1)
    known = lagged_vix.notna() & cutoff.notna()
    high = lagged_vix.gt(cutoff)
    spy_daily = spy["Close"].div(spy["Close"].shift()).sub(1)
    drop = spy_daily.lt(signal["drop_threshold"]) & known

    sessions = pd.DataFrame({
        "high_vix": (drop & high).shift(fill_value=False),
        "lower_vix": (drop & ~high).shift(fill_value=False),
        "drop_only": drop.shift(fill_value=False),
        "vix_ready": known.shift(fill_value=False),
        "signal_return": spy_daily.shift(),
        "spy_overnight": spy["Open"].div(spy["Close"].shift()),
        "spy_intraday": spy["Close"].div(spy["Open"]),
        "bil_overnight": bil["Open"].div(bil["Close"].shift()),
        "bil_intraday": bil["Close"].div(bil["Open"]),
    }).loc[start:end].copy()
    if not sessions["vix_ready"].iloc[0]:
        raise ValueError("The first execution session needs a complete lagged VIX warm-up.")
    if not np.isfinite(sessions["signal_return"].iloc[0]):
        raise ValueError("The first execution session needs a prior SPY close-to-close return.")
    return sessions


def simulate(sessions, event_column, cost_bps):
    rate = cost_bps / 10_000
    buy = 1 / (1 + rate)
    sell = 1 - rate
    if event_column is None:
        overnight = sessions["spy_overnight"].copy()
        overnight.iloc[0] = 1.0  # All portfolios start in cash at the first test open.
        growth = overnight * sessions["spy_intraday"]
        growth.iloc[0] *= buy
        trades = pd.Series(0, index=sessions.index)
        trades.iloc[0] = 1
    else:
        event = sessions[event_column]
        overnight = sessions["bil_overnight"].copy()
        overnight.iloc[0] = 1.0
        growth = overnight * sessions["spy_intraday"].where(event, sessions["bil_intraday"])
        growth.loc[event] *= (sell * buy) ** 2
        growth.iloc[0] *= 1 / sell if event.iloc[0] else buy
        trades = event.astype(int) * 4
        trades.iloc[0] = 3 if event.iloc[0] else 1
    return pd.DataFrame({"return": growth - 1, "trades": trades})


def metrics(daily, year):
    returns = daily["return"]
    wealth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    vol = returns.std(ddof=1)
    return {
        "cagr": wealth[-1] ** (year / len(returns)) - 1,
        "ann_vol": vol * math.sqrt(year),
        "sharpe": returns.mean() / vol * math.sqrt(year) if vol > 0 else None,
        "max_drawdown": (wealth / np.maximum.accumulate(wealth) - 1).min(),
        "total_return": wealth[-1] - 1,
        "n_trades": int(daily["trades"].sum()),
        "n_sessions": len(returns),
    }


def event_metrics(sessions, event_column):
    events = sessions.loc[sessions[event_column]]
    return {
        "n_events": len(events),
        "mean_signal_return": events["signal_return"].mean(),
        "mean_overnight_return": events["spy_overnight"].mean() - 1,
        "mean_intraday_return": events["spy_intraday"].mean() - 1,
        "mean_close_to_close_return": (events["spy_overnight"] * events["spy_intraday"]).mean() - 1,
    }


def summarize_events(sessions, windows, names, lower_edge, band_edges):
    diagnostic_sets = {"All": sessions}
    magnitude = -sessions["signal_return"]
    for left, right in zip([lower_edge, *band_edges], [*band_edges, math.inf]):
        label = f"{left:.2%}+" if math.isinf(right) else f"{left:.2%}–<{right:.2%}"
        diagnostic_sets[label] = sessions.loc[magnitude.ge(left) & magnitude.lt(right)]

    results = {}
    rows = []
    for label, subset in diagnostic_sets.items():
        results[label] = {}
        for model in ("high_vix", "lower_vix", "drop_only"):
            results[label][model] = {}
            for period, window in windows.items():
                stats = event_metrics(subset.loc[window], model)
                results[label][model][period] = stats
                rows.append({
                    "Drop size": label, "Model": names[model], "Period": period.upper(),
                    "Events": stats["n_events"], "Signal drop": stats["mean_signal_return"],
                    "Overnight": stats["mean_overnight_return"], "Intraday": stats["mean_intraday_return"],
                    "Close-close": stats["mean_close_to_close_return"],
                })
    events = results.pop("All")
    return events, results, rows


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def print_tables(portfolio_rows, event_rows):
    def percent(value):
        return f"{value:.2%}" if pd.notna(value) else "-"

    results_table = pd.DataFrame(portfolio_rows)
    print(results_table.to_string(index=False, na_rep="-", formatters={
        "CAGR": percent, "Ann. vol": percent,
        "Sharpe": lambda value: f"{value:.2f}" if pd.notna(value) else "-",
        "Max DD": percent, "Total return": percent,
    }))
    print("\nSPY event windows (gross mean returns, assigned by next-session execution date)")
    print(pd.DataFrame(event_rows).to_string(index=False, na_rep="-", formatters={
        "Signal drop": percent, "Overnight": percent, "Intraday": percent, "Close-close": percent,
    }))


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    market, reserve, volatility = (universe[key] for key in ("market", "reserve", "volatility"))
    start_is, end_is = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    start_oos, end_oos = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    data_start = pd.Timestamp(config["data_start"])
    if not data_start < start_is <= end_is < start_oos <= end_oos:
        raise ValueError("Expected warm-up, in-sample and out-of-sample dates in chronological order.")
    if start_oos != end_is + pd.Timedelta(days=1):
        raise ValueError("The test periods must be contiguous in calendar time.")

    signal = config["signal"]
    lower_edge = -signal["drop_threshold"]
    band_edges = config["event_drop_band_edges"]
    if not all(left < right for left, right in zip([lower_edge, *band_edges], band_edges)):
        raise ValueError("Diagnostic drop-band edges must increase above the absolute drop threshold.")
    costs = config["costs"]
    base_bps = costs["one_way_bps"]
    scenarios = [base_bps, *dict.fromkeys(bps for bps in costs["sensitivity_bps"] if bps != base_bps)]
    if not all(0 <= bps < 10_000 for bps in scenarios):
        raise ValueError("One-way costs must be at least 0 and below 10,000 bps.")

    prices = download_prices(market, reserve, volatility, data_start, end_oos, start_is)
    sessions = make_sessions(prices, market, reserve, volatility, signal, start_is, end_oos)
    windows = {"is": slice(start_is, end_is), "oos": slice(start_oos, end_oos), "full": slice(None)}
    if any(sessions.loc[window].empty for window in windows.values()):
        raise ValueError("Each test period needs at least one ETF session.")

    names = {
        "high_vix": "High VIX reversal", "lower_vix": "Lower VIX reversal",
        "drop_only": "Drop-only reversal", "benchmark": f"{market} buy-and-hold",
    }
    results = {}
    portfolio_rows = []
    for bps in scenarios:
        results[bps] = {}
        for model, name in names.items():
            daily = simulate(sessions, None if model == "benchmark" else model, bps)
            model_results = {}
            for period, window in windows.items():
                stats = metrics(daily.loc[window], config["annualization_sessions"])
                model_results[period] = stats
                portfolio_rows.append({
                    "Model": name, "Period": period.upper(), "Cost bps/leg": bps,
                    "CAGR": stats["cagr"], "Ann. vol": stats["ann_vol"], "Sharpe": stats["sharpe"],
                    "Max DD": stats["max_drawdown"], "Total return": stats["total_return"],
                    "Trades": stats["n_trades"],
                })
            results[bps][model] = model_results

    events, event_size_bands, event_rows = summarize_events(sessions, windows, names, lower_edge, band_edges)
    data_range = f"{sessions.index[0].date()} to {sessions.index[-1].date()}"
    ready_sessions = int(sessions["vix_ready"].sum())
    print(f"{config['strategy']} | {data_range} | {ready_sessions}/{len(sessions)} sessions with VIX state")
    print_tables(portfolio_rows, event_rows)
    primary = results[base_bps]
    payload = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **primary["high_vix"],
        "benchmark": {"name": names["benchmark"], **primary["benchmark"]},
        "cost_bps_per_leg": base_bps,
        "comparisons": {model: primary[model] for model in ("lower_vix", "drop_only")},
        "events": events,
        "event_size_bands": event_size_bands,
        "cost_sensitivity": {str(bps): results[bps] for bps in scenarios if bps != base_bps},
    }
    print("RESULTS_JSON: " + json.dumps(json_safe(payload), allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        sys.exit(f"Backtest failed: {exc}")
