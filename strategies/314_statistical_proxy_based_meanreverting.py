# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 314_statistical_proxy_based_meanreverting\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nmarket_benchmark: SPY\ndata_start: 2014-12-01\nin_sample:\n  start: 2016-01-01\n  end: 2020-12-31\nout_of_sample:\n  start: 2021-01-01\n  end: 2025-12-31\n\nformation_sessions: 252\nvolatility_quantiles: [0.25, 0.75]\nentry_z: 1.5\nmax_holding_sessions: 10\ngross_exposure: 1.0\ntransaction_cost_bps: 5.0  # Per dollar traded, on every entry and exit leg.\nannualization_sessions: 252\n'
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
"""Quarterly sector ETF spread selection and next-open execution backtest."""

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"


def load_prices(tickers, start, end):
    """Read a fixed-date slice of Yahoo Finance adjusted daily opens/closes."""
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / f"yahoo_adjusted_{'-'.join(tickers)}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    columns = [f"{field}_{ticker}" for field in ("Open", "Close") for ticker in tickers]

    def validate(prices):
        prices = prices.loc[:, columns].sort_index()
        if (
            prices.empty
            or not prices.index.is_unique
            or not start <= prices.index[0] <= start + pd.Timedelta(days=7)
            or not end - pd.Timedelta(days=7) <= prices.index[-1] <= end
        ):
            raise ValueError("Yahoo Finance data is missing or does not cover the configured dates")
        values = prices.to_numpy(dtype=float)
        valid_columns = (np.isfinite(values) & (values > 0)).all(axis=0)
        if not valid_columns.all():
            missing = prices.columns[~valid_columns]
            raise ValueError(f"Invalid or missing adjusted prices in {', '.join(missing)}")
        return prices

    if cache_file.is_file():
        cached = pd.read_csv(cache_file, index_col="Date", parse_dates=True)
        return validate(cached)

    yf.set_tz_cache_location(str(CACHE / "yfinance"))
    for attempt in range(2):
        try:
            raw = yf.download(
                tickers,
                start=start.date(),
                end=(end + pd.Timedelta(days=1)).date(),  # Yahoo end is exclusive.
                auto_adjust=True,
                keepna=True,
                progress=False,
                timeout=15,
            )
            prices = pd.DataFrame(
                {f"{field}_{ticker}": raw[field][ticker] for field in ("Open", "Close") for ticker in tickers}
            )
            prices = validate(prices)
            break
        except (KeyError, ValueError) as exc:
            if attempt == 1:
                raise RuntimeError("Unable to download complete adjusted OHLC for the configured universe") from exc

    with NamedTemporaryFile(dir=CACHE, suffix=".csv", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        prices.to_csv(temp_path, index_label="Date")
        temp_path.replace(cache_file)
    finally:
        temp_path.unlink(missing_ok=True)
    return prices


def candidate_weights(n_assets):
    # Holding each four-name set fixed gives three distinct splits up to sign.
    weights = []
    for first, *others in combinations(range(n_assets), 4):
        for partner in others:
            basket = np.zeros(n_assets)
            basket[[first, *others]] = -0.25
            basket[[first, partner]] = 0.25
            weights.append(basket)
    return np.array(weights)


@dataclass(frozen=True)
class Spread:
    weights: np.ndarray
    deviation: float
    sigma: float


def fit_spreads(returns, weights, formation_sessions, volatility_quantiles):
    crossing_fits, ar1_fits = {}, {}
    quarters = returns.index.to_period("Q")
    quarter_ends = np.flatnonzero(quarters[:-1] != quarters[1:])
    values = returns.to_numpy()
    for i in quarter_ends:
        if i < formation_sessions:
            continue
        paths = (values[i - formation_sessions + 1 : i + 1] @ weights.T).cumsum(axis=0)
        centered = paths - np.median(paths, axis=0)
        sigma = centered.std(axis=0, ddof=1)
        low, high = np.quantile(sigma, volatility_quantiles)
        eligible = np.flatnonzero((sigma >= low) & (sigma <= high) & (sigma > 0))
        if eligible.size == 0:
            raise ValueError(f"No spreads meet the volatility constraint on {returns.index[i].date()}")

        # Carry the preceding sign through exact zeros so a crossing counts once.
        signs = pd.DataFrame(np.sign(centered[:, eligible])).replace(0, np.nan).ffill().fillna(0).to_numpy()
        crossing_rates = (signs[:-1] * signs[1:] < 0).mean(axis=0)
        crossing = eligible[np.argmax(crossing_rates)]

        lagged = centered[:-1, eligible]
        current = centered[1:, eligible]
        lagged = lagged - lagged.mean(axis=0)
        current = current - current.mean(axis=0)
        lag_variance = np.square(lagged).sum(axis=0)
        ar1 = np.divide(
            (lagged * current).sum(axis=0),
            lag_variance,
            out=np.full(eligible.size, np.inf),
            where=lag_variance > 0,
        )
        if not np.isfinite(ar1).any():
            raise ValueError(f"No eligible AR(1) fit on {returns.index[i].date()}")
        persistence = eligible[np.argmin(ar1)]

        crossing_fits[i] = Spread(weights[crossing], float(centered[-1, crossing]), float(sigma[crossing]))
        ar1_fits[i] = Spread(weights[persistence], float(centered[-1, persistence]), float(sigma[persistence]))
    return crossing_fits, ar1_fits


def basket_targets(returns, fits, first_session, entry_z, max_hold, gross):
    """Return next-open target weights. A NaN row keeps the existing shares."""
    values = returns.to_numpy()
    targets = np.full_like(values, np.nan)
    signal_start = first_session - 1
    last_fit = max(i for i in fits if i <= signal_start)
    spread = fits[last_fit]
    deviation = spread.deviation + values[last_fit + 1 : signal_start + 1].sum(axis=0) @ spread.weights
    direction = 0
    entry_session = None

    for i in range(signal_start, len(returns) - 1):
        new_fit = i in fits
        if new_fit:
            spread = fits[i]
            deviation = spread.deviation
        elif i > signal_start:
            deviation += values[i] @ spread.weights

        if direction:
            at_equilibrium = direction * deviation >= 0
            at_time_limit = i + 1 - entry_session >= max_hold
            if new_fit or at_equilibrium or at_time_limit:
                targets[i + 1] = 0.0
                direction = 0
        elif abs(deviation) >= entry_z * spread.sigma:
            direction = -int(np.sign(deviation))
            entry_session = i + 1
            targets[entry_session] = direction * gross * spread.weights
    return targets[first_session:]


def simulate(opens, closes, targets, cost):
    """Mark fixed shares to the daily close and charge open execution costs."""
    cash = previous_nav = 1.0
    shares = np.zeros(opens.shape[1])
    records = []
    for open_prices, close_prices, target in zip(opens.to_numpy(), closes.to_numpy(), targets, strict=True):
        turnover = 0.0
        entry = 0
        if not np.isnan(target[0]):
            open_nav = cash + shares @ open_prices
            if open_nav <= 0:
                raise ValueError("Portfolio equity is nonpositive at the open")
            new_shares = open_nav * target / open_prices
            traded = (new_shares - shares) * open_prices
            traded_notional = np.abs(traded).sum()
            turnover = traded_notional / open_nav
            cash -= traded.sum() + cost * traded_notional
            shares = new_shares
            entry = int(np.any(target))

        nav = cash + shares @ close_prices
        if nav <= 0 or not np.isfinite(nav):
            raise ValueError("Portfolio equity is invalid at the close")
        records.append((nav / previous_nav - 1.0, turnover, entry))
        previous_nav = nav
    return pd.DataFrame(records, index=opens.index, columns=["return", "turnover", "entries"])


def metrics(book, sessions_per_year):
    returns = book["return"].to_numpy()
    equity = np.r_[1.0, np.cumprod(1.0 + returns)]
    sigma = float(returns.std(ddof=1)) if len(returns) > 1 else None
    ann_vol = sigma * np.sqrt(sessions_per_year) if sigma is not None else None
    sharpe = returns.mean() * np.sqrt(sessions_per_year) / sigma if sigma else None
    values = {
        "cagr": equity[-1] ** (sessions_per_year / len(returns)) - 1.0,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": (equity / np.maximum.accumulate(equity) - 1.0).min(),
        "total_return": equity[-1] - 1.0,
        "turnover": book["turnover"].sum(),
    }
    return {key: float(value) if value is not None and np.isfinite(value) else None for key, value in values.items()} | {
        "n_trades": int(book["entries"].sum()),
    }


def report(book, ranges, sessions_per_year):
    results = {}
    for key, (start, end) in ranges.items():
        segment = book.loc[start:end]
        if segment.empty:
            raise ValueError(f"No trading sessions for {key} between {start.date()} and {end.date()}")
        results[key] = metrics(segment, sessions_per_year)
    return results


def print_results(name, dates, strategy, persistence, market_name, market):
    print(f"{name}  |  {dates}")
    columns = ["cagr", "ann_vol", "sharpe", "max_drawdown", "total_return", "n_trades", "turnover"]
    table = []
    for label, results in [("Crossings", strategy), ("AR(1)", persistence), (f"{market_name} buy/hold", market)]:
        for period, stats in results.items():
            row = {"Book": label, "Period": period.upper()}
            row.update({key: stats[key] for key in columns})
            table.append(row)
    formats = dict.fromkeys(["cagr", "ann_vol", "max_drawdown", "total_return"], "{:.2%}".format)
    formats.update(dict.fromkeys(["sharpe", "turnover"], "{:.2f}".format))
    print(pd.DataFrame(table).to_string(index=False, formatters=formats, na_rep="n/a"))

    payload = {
        "strategy": name,
        "data_range": dates,
        **strategy,
        "benchmark": {
            "name": "AR(1) persistence",
            **persistence["full"],
            **persistence,
            "buy_and_hold": {"ticker": market_name, **market},
        },
    }
    print("RESULTS_JSON: " + json.dumps(payload, allow_nan=False))


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    ranges = {
        key: (pd.Timestamp(config[section]["start"]), pd.Timestamp(config[section]["end"]))
        for key, section in [("is", "in_sample"), ("oos", "out_of_sample")]
    }
    first_day, is_end = ranges["is"]
    oos_start, last_day = ranges["oos"]
    if not first_day <= is_end < oos_start <= last_day:
        raise ValueError("IS and OOS dates must be ordered, non-overlapping ranges")
    ranges["full"] = first_day, last_day
    universe = config["universe"]
    benchmark = config["market_benchmark"]
    if len(set(universe)) < 4 or len(set(universe)) != len(universe) or benchmark in universe:
        raise ValueError("Use at least four distinct spread tickers and a separate market benchmark")

    tickers = [*universe, benchmark]
    prices = load_prices(tickers, pd.Timestamp(config["data_start"]), last_day)
    opens = prices[[f"Open_{ticker}" for ticker in tickers]].set_axis(tickers, axis=1)
    closes = prices[[f"Close_{ticker}" for ticker in tickers]].set_axis(tickers, axis=1)
    returns = closes[universe].pct_change(fill_method=None)
    first_session = prices.index.searchsorted(first_day)
    candidates = candidate_weights(len(universe))
    crossing, persistence = fit_spreads(returns, candidates, config["formation_sessions"], config["volatility_quantiles"])
    if not any(i < first_session for i in crossing):
        raise ValueError("Start data earlier so a full formation window ends before the first IS session")

    cost = config["transaction_cost_bps"] / 10_000
    sessions_per_year = config["annualization_sessions"]
    opens, closes = opens.iloc[first_session:], closes.iloc[first_session:]
    results = []
    for fits in (crossing, persistence):
        targets = basket_targets(
            returns, fits, first_session, config["entry_z"], config["max_holding_sessions"], config["gross_exposure"]
        )
        book = simulate(opens[universe], closes[universe], targets, cost)
        results.append(report(book, ranges, sessions_per_year))

    market_targets = np.full((len(opens), 1), np.nan)
    market_targets[0, 0] = 1.0 / (1.0 + cost)
    market_book = simulate(opens[[benchmark]], closes[[benchmark]], market_targets, cost)
    market_results = report(market_book, ranges, sessions_per_year)
    dates = f"{opens.index[0]:%Y-%m-%d} to {opens.index[-1]:%Y-%m-%d}"
    print_results(config["strategy"], dates, *results, benchmark, market_results)


if __name__ == "__main__":
    main()
