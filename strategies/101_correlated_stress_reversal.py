# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 101_correlated_stress_reversal\n\nuniverse:\n  risk_tickers: [SPY, USO, GLD]\n  treasury_ticker: IEF\n  cash_ticker: BIL\n  benchmark_ticker: SPY\n\nsignal:\n  risk_drop_pct: 1.0\n  treasury_rise_pct: 1.0\n\ndata_start: 2007-12-14  # Warm-up history before the first reported session.\nin_sample:\n  start: 2008-01-02\n  end: 2025-04-25\nout_of_sample:\n  start: 2025-04-26\n  end: 2026-11-24\n\ntransaction_cost_bps_per_side: 5.0\nsessions_per_year: 252\n'
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
"""Backtest next-session cross-asset stress reversals using public ETF prices."""

import json
import math
from pathlib import Path
import re
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_COLUMNS = ["Open", "Close"]


def get_prices(ticker, start, end):
    if not re.fullmatch(r"[A-Z0-9^._=-]+", ticker):
        raise ValueError(f"Invalid Yahoo Finance ticker: {ticker!r}")

    path = CACHE / f"yahoo_adjusted_{ticker}_{start.date()}_{end.date()}.csv"
    if path.exists():
        prices = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    else:
        history = yf.Ticker(ticker).history(
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True,
            actions=False,
            timeout=15,
        )
        if history.empty:
            raise RuntimeError(f"Yahoo Finance returned no history for {ticker}")
        prices = history[PRICE_COLUMNS].copy()
        prices.index = prices.index.tz_localize(None).normalize()

    prices = prices[PRICE_COLUMNS].sort_index().loc[start:end]
    if prices.empty or prices.index.has_duplicates:
        raise ValueError(f"Missing or duplicate trading dates for {ticker}")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"Invalid adjusted opening or closing prices for {ticker}")
    if (prices.index[0] - start).days > 7 or (end - prices.index[-1]).days > 7:
        raise ValueError(f"{ticker} history does not cover {start.date()} to {end.date()}")

    if not path.exists():
        with NamedTemporaryFile(mode="w", dir=CACHE, suffix=".tmp", delete=False) as tmp:
            temporary = Path(tmp.name)
            try:
                prices.to_csv(tmp, index_label="Date")
                tmp.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    return prices


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="yahoo_", dir=CACHE) as metadata_cache:
        yf.set_tz_cache_location(metadata_cache)
        histories = {ticker: get_prices(ticker, start, end) for ticker in tickers}
    opens = pd.DataFrame({ticker: data["Open"] for ticker, data in histories.items()})
    closes = pd.DataFrame({ticker: data["Close"] for ticker, data in histories.items()})
    missing = closes.isna().sum()
    if missing.any():
        detail = ", ".join(f"{ticker} ({count})" for ticker, count in missing.items() if count)
        raise ValueError(f"Missing sessions in the shared ETF calendar: {detail}")
    return opens, closes


def reversal_returns(opens, closes, selected, cash_ticker, cost):
    """Hold the cash ETF overnight and rotate into the selected basket during the session."""
    n_selected = selected.sum(axis=1)
    risk_day = n_selected.gt(0)
    intraday = closes / opens
    risk_gross = intraday[selected.columns].where(selected, 0).sum(axis=1) / n_selected.where(risk_day)
    session_gross = risk_gross.where(risk_day, intraday[cash_ticker])

    overnight_gross = opens[cash_ticker] / closes[cash_ticker].shift()
    overnight_gross.iloc[0] = 1.0  # The backtest starts in cash at the first open.

    # A sale leaves (1 - cost) of proceeds. A purchase spends price * (1 + cost).
    switch_factor = (1 - cost) / (1 + cost)
    entry_factor = pd.Series(np.where(risk_day, switch_factor, 1.0), index=opens.index)
    entry_factor.iloc[0] = 1 / (1 + cost)
    exit_factor = np.where(risk_day, switch_factor, 1.0)
    returns = overnight_gross * entry_factor * session_gross * exit_factor - 1

    # Count each asset-side fill, including BIL. The first open has only buys.
    trades = (2 * (n_selected + 1)).where(risk_day, 0)
    trades.iloc[0] += -1 if risk_day.iloc[0] else 1
    return pd.DataFrame({"returns": returns, "trades": trades})


def buy_and_hold(opens, closes, ticker, cost):
    gross = closes[ticker] / closes[ticker].shift()
    gross.iloc[0] = closes[ticker].iloc[0] / opens[ticker].iloc[0] / (1 + cost)
    trades = pd.Series(0, index=opens.index)
    trades.iloc[0] = 1
    return pd.DataFrame({"returns": gross - 1, "trades": trades})


def finite_or_none(value):
    return float(value) if math.isfinite(value) else None


def metrics(daily, sessions_per_year):
    returns = daily["returns"]
    wealth = (1 + returns).cumprod()
    terminal_wealth = wealth.iloc[-1]
    volatility = returns.std(ddof=1)
    sharpe = returns.mean() / volatility * math.sqrt(sessions_per_year) if volatility > 0 else math.nan
    return {
        "cagr": finite_or_none(terminal_wealth ** (sessions_per_year / len(returns)) - 1),
        "ann_vol": finite_or_none(volatility * math.sqrt(sessions_per_year)),
        "sharpe": finite_or_none(sharpe),
        "max_drawdown": finite_or_none((wealth / wealth.cummax().clip(lower=1.0) - 1).min()),
        "total_return": finite_or_none(terminal_wealth - 1),
        "n_trades": int(daily["trades"].sum()),
        "n_sessions": len(returns),
    }


def report(daily, periods, sessions_per_year):
    return {
        name: metrics(daily.loc[start:end], sessions_per_year)
        for name, (start, end) in periods.items()
    }


def print_results(results):
    print(f"{results['strategy']}  |  {results['data_range']}")
    print(f"{'Portfolio':<23} {'Period':<6} {'CAGR':>9} {'Ann. vol':>9} {'Sharpe':>8} "
          f"{'Max DD':>9} {'Total':>10} {'Trades':>8} {'Sessions':>9}")
    portfolios = {
        "Correlated stress": results,
        results["benchmark"]["name"]: results["benchmark"],
        "Individual drop": results["benchmark"]["unconfirmed_drop"],
    }
    for label, portfolio in portfolios.items():
        for period in ("is", "oos", "full"):
            stats = portfolio[period]
            displayed = []
            for key in ("cagr", "ann_vol", "sharpe", "max_drawdown", "total_return"):
                value = stats[key]
                fmt = ".2f" if key == "sharpe" else ".2%"
                displayed.append(format(value, fmt) if value is not None else "n/a")
            cagr, volatility, sharpe, drawdown, total = displayed
            print(f"{label:<23} {period.upper():<6} {cagr:>9} {volatility:>9} {sharpe:>8} "
                  f"{drawdown:>9} {total:>10} {stats['n_trades']:>8} {stats['n_sessions']:>9}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    periods = {
        key: tuple(pd.Timestamp(cfg[section][bound]) for bound in ("start", "end"))
        for key, section in (("is", "in_sample"), ("oos", "out_of_sample"))
    }
    is_start, is_end = periods["is"]
    oos_start, oos_end = periods["oos"]
    data_start = pd.Timestamp(cfg["data_start"])
    if not data_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Expected warm-up, in-sample, and out-of-sample dates in order")
    periods["full"] = (is_start, oos_end)

    universe = cfg["universe"]
    risk = universe["risk_tickers"]
    treasury = universe["treasury_ticker"]
    cash = universe["cash_ticker"]
    benchmark = universe["benchmark_ticker"]
    if not risk or len(set([*risk, treasury, cash])) != len(risk) + 2:
        raise ValueError("The risk, Treasury, and cash roles require distinct tickers")
    tickers = list(dict.fromkeys([*risk, treasury, cash, benchmark]))

    cost = float(cfg["transaction_cost_bps_per_side"]) / 10_000
    risk_drop = float(cfg["signal"]["risk_drop_pct"]) / 100
    treasury_rise = float(cfg["signal"]["treasury_rise_pct"]) / 100
    sessions_per_year = int(cfg["sessions_per_year"])
    if not (0 <= cost < 1 and 0 < risk_drop < 1 and 0 < treasury_rise < 1 and sessions_per_year > 0):
        raise ValueError("Invalid threshold, transaction cost, or annualization parameter")

    opens, closes = load_prices(tickers, data_start, oos_end)
    if (closes.index < is_start).sum() < 2:
        raise ValueError("At least two warm-up sessions are required before in-sample starts")
    for name, (start, end) in periods.items():
        if closes.loc[start:end].empty:
            raise ValueError(f"No trading sessions in {name}")

    close_returns = closes.pct_change(fill_method=None)
    drops = close_returns[risk].le(-risk_drop)
    confirmed = drops.sum(axis=1).ge(2) | close_returns[treasury].ge(treasury_rise)
    confirmed_drops = drops.where(confirmed, False, axis=0)
    trade_dates = closes.loc[is_start:oos_end].index
    opens, closes = opens.loc[trade_dates], closes.loc[trade_dates]

    def run_reversal(signal):
        next_session = signal.shift(1, fill_value=False).loc[trade_dates]
        return report(reversal_returns(opens, closes, next_session, cash, cost), periods, sessions_per_year)

    results = {
        "strategy": cfg["strategy"],
        "data_range": f"{trade_dates[0].date()} to {trade_dates[-1].date()}",
        **run_reversal(confirmed_drops),
        "benchmark": {
            "name": f"{benchmark} buy and hold",
            "ticker": benchmark,
            **report(buy_and_hold(opens, closes, benchmark, cost), periods, sessions_per_year),
            "unconfirmed_drop": {
                "name": "Individual risk ETF drop",
                **run_reversal(drops),
            },
        },
    }
    print_results(results)


if __name__ == "__main__":
    main()
