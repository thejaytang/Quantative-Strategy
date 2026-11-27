# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 197_tsmom_boundaries\n\nuniverse:\n  equity: SPY\n  cash: BIL\n  treasury_long: DGS10\n  treasury_short: DGS3MO\n\ndata_start: '2002-01-01'\nin_sample:\n  start: '2009-01-01'\n  end: '2018-12-31'\nout_of_sample:\n  start: '2019-01-01'\n  end: '2026-10-30'\n\nsignal:\n  dividend_lookback_months: 12\n  treasury_lag_sessions: 2\n  valuation_lookback_months: 60\n  boundary_threshold: 4.0\n  trend_lookback_sessions: 252\n\ncommission_bps_per_side: 1.0\nslippage_bps_per_side: 4.0\nparam_combos_tried: 1\n"
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
"""C46 valuation-conditioned U.S. equity time-series momentum backtest."""

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SESSIONS_PER_YEAR = 252
PRICE_ADJUSTMENT = (
    "Yahoo split-adjusted Open/Close for execution and valuation; "
    "dividend- and split-adjusted Adj Close for momentum and total returns; "
    "adjusted open = Open * Adj Close / Close"
)


def cached_history(source, symbol, start, end, download):
    path = CACHE / f"{source}_{symbol}_{start.date()}_{end.date()}.csv"
    cache_hit = path.exists()
    if cache_hit:
        data = pd.read_csv(path, index_col="date", parse_dates=["date"])
        fetched_on = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).date()
    else:
        data = download().sort_index()
        fetched_on = datetime.now(timezone.utc).date()

    if data.index.has_duplicates or not data.index.is_monotonic_increasing or data.index.hasnans:
        raise ValueError(f"Invalid {source} observation dates for {symbol}")
    data = data.loc[start:end]
    last_observation = data.dropna().index.max()
    if pd.isna(last_observation) or last_observation < end - pd.Timedelta(days=7):
        raise ValueError(f"Missing, incomplete or unordered {source} history for {symbol}")

    if not cache_hit:
        with NamedTemporaryFile(dir=CACHE, suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
        try:
            data.to_csv(temporary_path, index_label="date")
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    data.attrs["fetched_on"] = fetched_on
    return data


def fetch_etf(symbol, start, end):
    def download():
        data = yf.Ticker(symbol).history(
            start=str(start.date()),
            end=str((end + pd.Timedelta(days=1)).date()),
            auto_adjust=False,
            actions=True,
            timeout=15,
        )
        if data.empty:
            raise RuntimeError(f"Yahoo Finance returned no history for {symbol}")
        data.index = data.index.tz_localize(None)
        return data[["Open", "Close", "Adj Close", "Dividends"]].astype(float)

    data = cached_history("yahoo", symbol, start, end, download)
    prices = data[["Open", "Close", "Adj Close"]]
    if not np.isfinite(prices).all().all() or (prices <= 0).any().any():
        raise ValueError(f"Invalid daily prices for {symbol}")
    if not np.isfinite(data["Dividends"]).all() or (data["Dividends"] < 0).any():
        raise ValueError(f"Invalid dividend actions for {symbol}")
    return data


def fetch_treasury(symbol, start, end):
    def download():
        response = requests.get(
            FRED_CSV,
            params={"id": symbol, "cosd": str(start.date()), "coed": str(end.date())},
            timeout=20,
        )
        response.raise_for_status()
        data = pd.read_csv(io.StringIO(response.text), na_values=["."])
        dates = pd.to_datetime(data.iloc[:, 0])
        return data[[symbol]].astype(float).set_index(dates)

    data = cached_history("fred", symbol, start, end, download)
    rates = data[symbol].dropna()
    if not np.isfinite(rates).all():
        raise ValueError(f"Invalid Treasury yields for {symbol}")
    return rates, data.attrs["fetched_on"]


def available_treasury_spread(sessions, long_rates, short_rates, lag_sessions):
    spread = long_rates.sub(short_rates).dropna().sort_index()
    observations = spread.index
    # Allow through the following joint Treasury observation day for the H.15
    # release. Require a later SPY session even if that day is a market holiday.
    after_release = sessions.searchsorted(observations[1:], side="right")
    after_lag = sessions.searchsorted(observations[:-1]) + lag_sessions
    effective = np.maximum(after_release, after_lag)
    valid = effective < len(sessions)
    released = pd.Series(spread.to_numpy()[:-1][valid], index=sessions[effective[valid]])
    released = released[~released.index.duplicated(keep="last")]
    return released.reindex(sessions, method="ffill", tolerance=pd.Timedelta(days=7))


def month_end_signals(equity, cash, long_rates, short_rates, settings):
    close = equity["Close"]
    sessions = close.index
    dividend_start = sessions - pd.DateOffset(months=settings["dividend_lookback_months"])
    first_in_window = sessions.searchsorted(dividend_start, side="right")
    cumulative_dividends = np.r_[0.0, equity["Dividends"].cumsum().to_numpy()]
    dividends = cumulative_dividends[1:] - cumulative_dividends[first_in_window]
    dividend_yield = close.rdiv(dividends).where(dividend_start >= sessions[0])

    spread = available_treasury_spread(sessions, long_rates, short_rates, settings["treasury_lag_sessions"])
    valuations = pd.DataFrame({"dividend_yield": dividend_yield, "term_spread": spread})

    month_ends = sessions[~sessions.to_period("M").duplicated(keep="last")]
    monthly = valuations.loc[month_ends]
    previous = monthly.shift(1).rolling(settings["valuation_lookback_months"])
    scale = previous.std(ddof=0)
    z_scores = (monthly - previous.mean()) / scale.where(scale > 0)
    score = z_scores.pow(2).sum(axis=1, min_count=2)

    total_return_prices = pd.DataFrame({"equity": equity["Adj Close"], "cash": cash["Adj Close"]})
    past_returns = total_return_prices.pct_change(settings["trend_lookback_sessions"], fill_method=None)
    excess = (past_returns["equity"] - past_returns["cash"]).reindex(month_ends)
    extreme = score > settings["boundary_threshold"]
    ready = score.notna() & excess.notna()
    return pd.DataFrame(
        {
            "strategy": excess.mask(extreme, -excess).gt(0).where(ready),
            "plain_tsmom": excess.gt(0).where(ready),
        }
    )


def simulate(target, equity, cash, cost):
    """Mark at each close with the prior asset held overnight and the target held intraday."""
    prior = target.shift(1, fill_value=False)
    overnight = equity["overnight"].where(prior, cash["overnight"])
    intraday = equity["intraday"].where(target, cash["intraday"])
    switches = target.ne(prior)
    fees_factor = pd.Series(1.0, index=target.index)
    fees_factor.loc[switches] = (1 - cost) / (1 + cost)
    fees_factor.iloc[0] = 1 / (1 + cost)
    trade_legs = switches.astype(int) * 2
    trade_legs.iloc[0] = 1
    overnight.iloc[0] = 1.0
    return pd.DataFrame({"return": overnight * intraday * fees_factor - 1, "trade_legs": trade_legs})


def return_factors(prices):
    adjusted_open = prices["Open"] * (prices["Adj Close"] / prices["Close"])
    return pd.DataFrame(
        {
            "overnight": adjusted_open / prices["Adj Close"].shift(1),
            "intraday": prices["Adj Close"] / adjusted_open,
        }
    )


def annualized_sharpe(daily):
    if not np.isfinite(daily).all() or daily.nunique() < 2:
        return np.nan
    volatility = daily.std(ddof=1)
    return daily.mean() * np.sqrt(SESSIONS_PER_YEAR) / volatility if volatility > 0 else np.nan


def metrics(simulation, risk_free):
    daily = simulation["return"]
    nav = np.r_[1.0, (1 + daily).cumprod().to_numpy()]
    excess = daily - risk_free.reindex(daily.index)
    values = {
        "cagr": nav[-1] ** (SESSIONS_PER_YEAR / len(daily)) - 1,
        "ann_vol": daily.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR),
        "sharpe": annualized_sharpe(daily),
        "sharpe_excess": annualized_sharpe(excess),
        "max_drawdown": (nav / np.maximum.accumulate(nav) - 1).min(),
        "total_return": nav[-1] - 1,
    }
    result = {key: float(value) if np.isfinite(value) else None for key, value in values.items()}
    result["n_trades"] = int(simulation["trade_legs"].sum())
    result["turnover"] = float(result["n_trades"])
    return result


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    is_start, is_end = [pd.Timestamp(config["in_sample"][key]) for key in ("start", "end")]
    oos_start, oos_end = [pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end")]
    data_start = pd.Timestamp(config["data_start"])
    if not data_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Use ordered, non-overlapping IS/OOS windows and earlier warm-up data")
    commission_bps = float(config["commission_bps_per_side"])
    slippage_bps = float(config["slippage_bps_per_side"])
    cost_bps = commission_bps + slippage_bps
    cost = cost_bps / 10_000
    if not (commission_bps >= 0 and slippage_bps >= 0 and cost < 1):
        raise ValueError("Commission and slippage must be nonnegative and total less than 10000 bps")

    CACHE.mkdir(exist_ok=True)
    universe = config["universe"]
    with TemporaryDirectory(dir=CACHE, prefix="yahoo-") as metadata_dir:
        yf.set_tz_cache_location(metadata_dir)
        equity = fetch_etf(universe["equity"], data_start, oos_end)
        cash = fetch_etf(universe["cash"], data_start, oos_end)
    long_rates, long_fetched = fetch_treasury(universe["treasury_long"], data_start, oos_end)
    short_rates, short_fetched = fetch_treasury(universe["treasury_short"], data_start, oos_end)
    if not equity.loc[cash.index[0]:].index.equals(cash.index):
        raise ValueError("ETF sessions differ over their common history")

    monthly = month_end_signals(equity, cash, long_rates, short_rates, config["signal"])
    # Forward-fill only between monthly signals. Missing month-end inputs stay missing.
    targets = monthly.reindex(equity.index, method="ffill").shift(1).loc[is_start:oos_end]
    if targets.empty or targets.isna().any().any():
        raise ValueError("Evaluation requires a valid prior month-end signal on every session")
    targets = targets.astype(bool)
    targets["buy_hold"] = True
    equity_factors = return_factors(equity).reindex(targets.index)
    cash_factors = return_factors(cash).reindex(targets.index)
    for factors in (equity_factors, cash_factors):
        if not np.isfinite(factors).all().all():
            raise ValueError("Missing ETF returns over the evaluation interval")
    risk_free = cash_factors["overnight"] * cash_factors["intraday"] - 1
    risk_free.iloc[0] = cash_factors["intraday"].iloc[0] - 1

    windows = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    labels = {
        "strategy": "Valuation boundaries",
        "plain_tsmom": "Plain time-series momentum",
        "buy_hold": f"{universe['equity']} buy-and-hold",
    }
    reports = {}
    rows = []
    for method, label in labels.items():
        simulation = simulate(targets[method], equity_factors, cash_factors, cost)
        reports[method] = {}
        for window, (start, end) in windows.items():
            sample = simulation.loc[start:end]
            if sample.empty:
                raise ValueError(f"No trading sessions in {window.upper()}")
            report = metrics(sample, risk_free)
            reports[method][window] = report
            rows.append({"Portfolio": label, "Window": window.upper(), **report})

    index = targets.index
    data_range = f"{index[0].date()} to {index[-1].date()}"
    fetched_on = {equity.attrs["fetched_on"], cash.attrs["fetched_on"], long_fetched, short_fetched}
    print(f"DATA_FETCHED_AT: {', '.join(day.isoformat() for day in sorted(fetched_on))}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"PARAM_COMBOS_TRIED: {config['param_combos_tried']}")
    print(
        f"{config['strategy']} | Evaluation {data_range} | Cost {cost_bps:g} bps per side "
        f"({commission_bps:g} commission + {slippage_bps:g} slippage) | Cash proxy {universe['cash']}"
    )
    for name in ("is", "oos"):
        start, end = windows[name]
        print(f"{name.upper()} {start.date()} to {end.date()}")
    table = pd.DataFrame(rows)
    percent_columns = ["cagr", "ann_vol", "max_drawdown", "total_return"]
    formatters = {column: "{:.2%}".format for column in percent_columns}
    formatters.update(sharpe="{:.2f}".format, sharpe_excess="{:.2f}".format, turnover="{:.1f}x".format)
    print(table.to_string(index=False, formatters=formatters, na_rep="n/a"))

    buy_hold = reports["buy_hold"]
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **reports["strategy"],
        "benchmark": {
            "name": labels["buy_hold"],
            **buy_hold["full"],
            **buy_hold,
            "plain_tsmom": {"name": labels["plain_tsmom"], **reports["plain_tsmom"]},
        },
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
