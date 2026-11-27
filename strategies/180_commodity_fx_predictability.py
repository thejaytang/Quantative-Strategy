# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 180_commodity_fx_predictability\ndata_start: 2012-01-01\n\nin_sample:\n  start: 2016-01-01\n  end: 2022-12-31\nout_of_sample:\n  start: 2023-01-01\n  end: 2025-12-31\n\nuniverse:\n  cash: BIL\n  sleeves:\n    - currency: FXC\n      commodity: USO\n      weight: 0.50\n    - currency: FXA\n      commodity: CPER\n      weight: 0.50\n\ncommodity_lookbacks: [5, 21]\ncurrency_lookback: 5\ntraining_weeks: 156\nridge_alpha: 1.0\nforecast_threshold: 0.0\ntransaction_cost_bps: 5.0  # Per side, on every ETF purchase or sale.\nslippage_bps: 5.0  # Additional execution cost per side from the opening mark.\nsessions_per_year: 252\n'
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

import argparse
import json
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_ADJUSTMENT = "Yahoo Adj Close; adjusted Open = Open * (Adj Close / Close)"
PARAM_COMBOS_TRIED = 1  # One fixed strategy specification, with no search.
TRADE_WEIGHT_TOLERANCE = 1e-10


def load_prices(ticker, start, end):
    cache_file = CACHE / f"{ticker}_{start.date()}_{end.date()}.csv"
    from_cache = cache_file.is_file()
    if from_cache:
        prices = pd.read_csv(cache_file, index_col="Date", parse_dates=True)
    else:
        history = yf.Ticker(ticker).history(
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            auto_adjust=False,
            actions=False,
            timeout=15,
        )
        if history.empty:
            raise RuntimeError(f"Yahoo Finance returned no daily prices for {ticker}.")
        # Yahoo provides an adjusted close but no adjusted open.
        prices = pd.DataFrame({
            "open": history["Open"] * (history["Adj Close"] / history["Close"]),
            "close": history["Adj Close"],
        })
        prices.index = prices.index.tz_localize(None)

    values = prices.to_numpy()
    if (prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing
            or not np.isfinite(values).all() or (values <= 0).any()):
        raise ValueError(f"Invalid dates or adjusted prices for {ticker}.")
    if not from_cache:
        with NamedTemporaryFile(dir=CACHE, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
        try:
            prices.to_csv(temporary, index_label="Date")
            temporary.replace(cache_file)
        finally:
            temporary.unlink(missing_ok=True)
    # The original and current downloaders write the cache immediately after download.
    fetched_date = datetime.fromtimestamp(cache_file.stat().st_mtime, ZoneInfo("Europe/Oslo")).date()
    return prices, fetched_date


def weekly_targets(opens, closes, config):
    sessions = opens.index
    # A holiday Friday uses the last close from that week. Entry is strictly after Friday.
    signal_closes = pd.Series(sessions, index=sessions).resample("W-FRI").last().dropna()
    entry_positions = sessions.searchsorted(signal_closes.index, side="right")
    has_entry = entry_positions < len(sessions)
    signal_closes = signal_closes.loc[has_entry]
    entry_dates = pd.Series(sessions[entry_positions[has_entry]], index=signal_closes.index)
    exit_dates = entry_dates.shift(-1)

    cash = config["universe"]["cash"]
    sleeves = config["universe"]["sleeves"]
    currencies = [sleeve["currency"] for sleeve in sleeves]
    entry_opens = opens.loc[entry_dates].set_axis(entry_dates.index)
    weekly_returns = entry_opens.shift(-1).div(entry_opens).sub(1)
    labels = weekly_returns[currencies].sub(weekly_returns[cash], axis=0)

    strategy = pd.DataFrame(index=entry_dates.index, columns=currencies, dtype=float)
    own_currency = strategy.copy()
    for sleeve in sleeves:
        currency = sleeve["currency"]
        commodity = sleeve["commodity"]
        features = {
            f"commodity_{lookback}": closes[commodity].pct_change(lookback, fill_method=None)
            for lookback in config["commodity_lookbacks"]
        }
        features["currency"] = closes[currency].pct_change(config["currency_lookback"], fill_method=None)
        features = pd.DataFrame(features).loc[signal_closes].set_axis(entry_dates.index)
        pairs = features.assign(label=labels[currency]).dropna()
        specifications = [(strategy, list(features)), (own_currency, ["currency"])]

        for _, quarter in features.groupby(features.index.to_period("Q")):
            fit_as_of = signal_closes.loc[quarter.index[0]]
            # Both prices in a training label must have been observed by the fit close.
            completed = exit_dates.reindex(pairs.index).le(fit_as_of)
            train = pairs.loc[completed].tail(config["training_weeks"])
            if len(train) < config["training_weeks"]:
                continue
            for forecasts, columns in specifications:
                model = make_pipeline(StandardScaler(), Ridge(alpha=config["ridge_alpha"]))
                model.fit(train[columns], train["label"])
                valid = quarter[columns].dropna()
                forecasts.loc[valid.index, currency] = model.predict(valid)

    weights = pd.Series({sleeve["currency"]: sleeve["weight"] for sleeve in sleeves})
    targets = {}
    for name, forecasts in [("strategy", strategy), ("own_currency_only", own_currency)]:
        active = forecasts.dropna().gt(config["forecast_threshold"])
        allocation = active.mul(weights, axis=1)
        allocation[cash] = 1.0 - allocation.sum(axis=1)
        targets[name] = allocation.set_axis(entry_dates.loc[allocation.index])

    passive = pd.DataFrame(1.0 / len(currencies), index=targets["strategy"].index, columns=currencies)
    passive[cash] = 0.0
    targets["equal_weight_fx"] = passive
    return targets


def simulate(open_relatives, targets, fee):
    first_open = open_relatives.index[0]
    prior_targets = targets.loc[:first_open]
    if prior_targets.empty:
        raise ValueError("Insufficient completed training history before the first backtest session.")
    schedule = targets.reindex(columns=open_relatives.columns).to_dict("index")
    schedule[first_open] = prior_targets.iloc[-1].to_dict()

    # The portfolio begins with uninvested USD. Carry drifted weights between rebalances.
    holdings = np.zeros(open_relatives.shape[1])
    records = []
    for session, relatives in zip(open_relatives.index, open_relatives.to_numpy()):
        nav_after_fee = 1.0
        turnover = 0.0
        n_trades = 0
        if session in schedule:
            target = np.array([schedule[session][ticker] for ticker in open_relatives.columns])
            # Solve for post-cost NAV so traded notional and fees are self-financing.
            nav_after_fee = brentq(
                lambda nav: nav + fee * np.abs(nav * target - holdings).sum() - 1.0, 0.0, 1.0
            )
            traded_weights = np.abs(nav_after_fee * target - holdings)
            turnover = traded_weights.sum()
            n_trades = np.count_nonzero(traded_weights > TRADE_WEIGHT_TOLERANCE)
            holdings = target
        marked_holdings = holdings * relatives
        gross_growth = marked_holdings.sum()
        holdings = marked_holdings / gross_growth
        records.append((nav_after_fee * gross_growth - 1.0, turnover, n_trades))
    return pd.DataFrame(records, index=open_relatives.index, columns=["return", "turnover", "n_trades"])


def metrics(performance, cash_returns, next_open, sessions_per_year):
    returns = performance["return"]
    excess = returns - cash_returns.loc[returns.index]
    wealth = (1 + returns).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    years = (next_open.loc[returns.index[-1]] - returns.index[0]).days / 365.25
    annualization = np.sqrt(sessions_per_year)
    vol = returns.std()
    excess_vol = excess.std()
    numbers = {
        "cagr": wealth.iloc[-1] ** (1 / years) - 1,
        "ann_vol": vol * annualization,
        "sharpe": returns.mean() / vol * annualization if vol > 0 else np.nan,
        "sharpe_excess": excess.mean() / excess_vol * annualization if excess_vol > 0 else np.nan,
        "max_drawdown": (wealth / peak - 1).min(),
        "total_return": wealth.iloc[-1] - 1,
        "turnover": performance["turnover"].sum(),
    }
    report = {name: float(value) if np.isfinite(value) else None for name, value in numbers.items()}
    report["n_trades"] = int(performance["n_trades"].sum())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily", action="store_true", help="Print daily net returns and BIL returns as CSV.")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    periods = {
        name: tuple(pd.Timestamp(config[key][bound]) for bound in ("start", "end"))
        for name, key in [("is", "in_sample"), ("oos", "out_of_sample")]
    }
    first, is_end = periods["is"]
    oos_start, last = periods["oos"]
    data_start = pd.Timestamp(config["data_start"])
    if not data_start < first <= is_end < oos_start <= last:
        raise ValueError("Configure chronological, non-overlapping evaluation periods with training prehistory.")
    periods["full"] = (first, last)

    cash = config["universe"]["cash"]
    sleeves = config["universe"]["sleeves"]
    currencies = [sleeve["currency"] for sleeve in sleeves]
    commodities = [sleeve["commodity"] for sleeve in sleeves]
    transaction_cost = config["transaction_cost_bps"]
    slippage = config["slippage_bps"]
    fee = (transaction_cost + slippage) / 10_000
    if not (transaction_cost >= 0 and slippage >= 0 and 0 <= fee < 1):
        raise ValueError("Costs and slippage must be nonnegative and total less than 10000 basis points.")
    weights = [sleeve["weight"] for sleeve in sleeves]
    if not weights or min(weights) <= 0 or sum(weights) > 1:
        raise ValueError("Sleeve weights must be positive and sum to at most one.")

    CACHE.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(CACHE / "yahoo_metadata"))
    # The next opening mark is needed to value the final session's holding interval.
    data_end = last + pd.Timedelta(days=8)
    prices = {}
    fetched_dates = set()
    for ticker in dict.fromkeys([cash, *currencies, *commodities]):
        prices[ticker], fetched_date = load_prices(ticker, data_start, data_end)
        fetched_dates.add(fetched_date.isoformat())
    sessions = prices[cash].index
    if sessions[-1] <= last:
        raise ValueError(f"{cash} has no opening mark after the configured end date {last.date()}.")
    opens = pd.DataFrame({ticker: prices[ticker]["open"] for ticker in [*currencies, cash]}, index=sessions)
    closes = pd.DataFrame({ticker: prices[ticker]["close"] for ticker in [*currencies, *commodities]}, index=sessions)
    for field, frame in [("open", opens), ("close", closes)]:
        for ticker in frame:
            missing = sessions[frame[ticker].isna()]
            if len(missing):
                raise ValueError(f"Missing {ticker} {field} on {missing[0].date()} ({len(missing)} sessions).")
    targets = weekly_targets(opens, closes, config)
    open_relatives = opens.shift(-1).div(opens).loc[first:last]
    next_open = pd.Series(sessions, index=sessions).shift(-1)
    cash_returns = open_relatives[cash].sub(1)
    reports = {}
    daily = {}
    for name, allocation in targets.items():
        performance = simulate(open_relatives, allocation, fee)
        if args.daily:
            daily[name] = performance.rename(columns={"return": "net_return"}).assign(
                next_open=next_open, cash_return=cash_returns
            )
        reports[name] = {}
        for period, (start, end) in periods.items():
            subset = performance.loc[start:end]
            if subset.empty:
                raise ValueError(f"No trading sessions in the {period} date range.")
            reports[name][period] = metrics(subset, cash_returns, next_open, config["sessions_per_year"])

    data_range = f"{open_relatives.index[0].date()} to {open_relatives.index[-1].date()}"
    print(f"{config['strategy']} | {data_range} | {transaction_cost:g} bps costs + {slippage:g} bps slippage per side")
    print(f"PARAM_COMBOS_TRIED: {PARAM_COMBOS_TRIED}")
    print(f"DATA_FETCHED_AT: {', '.join(sorted(fetched_dates))} (Europe/Oslo; Yahoo Finance cache write dates)")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"{'Portfolio':<21} {'Set':<5} {'CAGR':>9} {'Ann vol':>9} {'Sharpe':>9} {'Excess SR':>9} "
          f"{'Max DD':>9} {'Total':>9} {'Turnover':>9} {'N trades':>9}")
    for name, report in reports.items():
        for period, row in report.items():
            values = []
            for key, value in row.items():
                if value is None:
                    values.append(f"{'n/a':>9}")
                elif key == "n_trades":
                    values.append(f"{value:>9d}")
                elif key in ("sharpe", "sharpe_excess", "turnover"):
                    values.append(f"{value:>9.2f}")
                else:
                    values.append(f"{value:>9.2%}")
            print(f"{name:<21} {period.upper():<5} {' '.join(values)}")
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **reports["strategy"],
        "benchmark": {**reports["equal_weight_fx"], "own_currency_only": reports["own_currency_only"]},
    }
    if args.daily:
        print("DAILY_RETURNS_CSV:")
        print(pd.concat(daily, names=["portfolio", "session"]).to_csv(), end="")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
