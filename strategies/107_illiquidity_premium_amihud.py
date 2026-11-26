# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 107_illiquidity_premium_amihud\nuniverse:\n  [AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, BAC, GS, MS, XOM, CVX,\n   COP, SLB, WMT, COST, PG, KO, PEP, JNJ, PFE, MRK, HD, LOW]\n\n# Inclusive dates. Pre-IS data are used only for signal warmup.\ndata_start: 2015-09-01\nin_sample:\n  start: 2016-01-01\n  end: 2020-12-31\nout_of_sample:\n  start: 2021-01-01\n  end: 2025-12-31\n\nlookback_sessions: 63\nmin_observations: 50\nquartile_fraction: 0.25\nannualization_sessions: 252\n\n# One-way basis points per dollar bought or sold. A round trip pays twice.\ntransaction_cost_bps: 5\nsensitivity_cost_bps: [0, 15]\n'
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
"""C08 Amihud illiquidity sort on the fixed US24 stock basket."""

import hashlib
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
FIELDS = ["Open", "Close", "Adj Close", "Volume"]
BOOKS = ("High illiquidity", "Low illiquidity", "US24 equal weight")


def load_prices(tickers, start, end):
    cache = ROOT / "data_cache"
    cache.mkdir(exist_ok=True)
    universe_id = hashlib.sha256(",".join(sorted(tickers)).encode()).hexdigest()[:12]
    path = cache / f"yahoo_{start.date()}_{end.date()}_{universe_id}.csv"
    if path.exists():
        return pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)

    yf.set_tz_cache_location(str(cache))
    prices = {}
    for _ in range(2):
        missing = [ticker for ticker in tickers if ticker not in prices]
        if not missing:
            break
        downloaded = yf.download(
            missing, start=start.date(), end=(end + pd.Timedelta(days=1)).date(),
            auto_adjust=False, ignore_tz=True, group_by="column", progress=False,
            threads=8, timeout=15, multi_level_index=True,
        )
        for ticker in missing:
            if ticker not in downloaded.columns.get_level_values(1):
                continue
            bars = downloaded.xs(ticker, axis=1, level=1).reindex(columns=FIELDS)
            if bars.notna().any().all():
                prices[ticker] = bars.dropna(how="all")

    missing = [ticker for ticker in tickers if ticker not in prices]
    if missing:
        raise RuntimeError(f"Yahoo Finance returned incomplete data for {', '.join(missing)}")
    panel = pd.concat(prices, axis=1).swaplevel(axis=1).sort_index().sort_index(axis=1)
    with NamedTemporaryFile(mode="w", dir=cache, suffix=".tmp", delete=False) as temporary:
        temp_path = Path(temporary.name)
        try:
            panel.to_csv(temporary, lineterminator="\n")
            temporary.flush()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    return panel


def portfolio_targets(scores, adjusted_close, fraction):
    targets = {name: {} for name in BOOKS}
    months = scores.index.to_period("M")
    # A new month executes the preceding month's last closing signal at its open.
    for next_session in np.flatnonzero(months[1:] != months[:-1]) + 1:
        close = adjusted_close.iloc[next_session - 1]
        ranked = scores.iloc[next_session - 1].where(close.notna()).dropna()
        ranked = ranked.sort_values(kind="stable")
        size = math.floor(len(ranked) * fraction)
        groups = (
            ranked.index[-size:] if size else ranked.index[:0],
            ranked.index[:size],
            close.dropna().index,
        )
        for name, chosen in zip(BOOKS, groups):
            weights = pd.Series(0.0, index=scores.columns)
            if len(chosen):
                weights.loc[chosen] = 1.0 / len(chosen)
            targets[name][next_session] = weights.to_numpy()
    return targets


def mark_to_market(weights, price_relatives, date, tickers):
    held = weights > 0
    invalid = held & (~np.isfinite(price_relatives) | (price_relatives <= 0))
    if invalid.any():
        raise RuntimeError(f"Missing or invalid held price on {date.date()} for {', '.join(tickers[invalid])}")
    marked = weights.copy()
    marked[held] *= price_relatives[held]
    growth = 1.0 + (marked - weights).sum()
    return marked / growth, growth


def simulate(overnight, intraday, targets, start, end, cost_bps):
    cost = cost_bps / 10_000.0
    dates = overnight.index
    tickers = overnight.columns
    first = dates.searchsorted(start, side="left")
    stop = dates.searchsorted(end, side="right")
    overnight_values = overnight.to_numpy()
    intraday_values = intraday.to_numpy()
    weights = np.zeros(len(tickers))
    daily_returns = np.empty(stop - first)
    daily_turnover = np.zeros(stop - first)

    for row in range(first, stop):
        date = dates[row]
        weights, overnight_growth = mark_to_market(weights, overnight_values[row], date, tickers)
        fee_factor = 1.0
        if row in targets:
            target = targets[row]
            if cost:
                # Targets are fractions of post-fee NAV. Solve for that NAV's
                # fraction of pre-trade NAV so costs are funded by the portfolio.
                def balance(nav_fraction):
                    dollars_traded = np.abs(nav_fraction * target - weights).sum()
                    return nav_fraction + cost * dollars_traded - 1.0

                fee_factor = brentq(balance, 0.0, 1.0)
            daily_turnover[row - first] = np.abs(fee_factor * target - weights).sum()
            weights = target
        weights, intraday_growth = mark_to_market(weights, intraday_values[row], date, tickers)
        daily_returns[row - first] = overnight_growth * fee_factor * intraday_growth - 1.0

    return pd.DataFrame(
        {"return": daily_returns, "turnover": daily_turnover}, index=dates[first:stop],
    )


def metrics(daily, sessions_per_year):
    returns = daily["return"]
    n = len(returns)
    if not n:
        raise ValueError("A configured reporting period has no trading sessions")
    equity = np.r_[1.0, (1.0 + returns).cumprod().to_numpy()]
    volatility = returns.std(ddof=1) * math.sqrt(sessions_per_year)
    values = {
        "cagr": equity[-1] ** (sessions_per_year / n) - 1.0,
        "ann_vol": volatility,
        "sharpe": returns.mean() * sessions_per_year / volatility if volatility > 0 else np.nan,
        "max_drawdown": (equity / np.maximum.accumulate(equity) - 1.0).min(),
        "total_return": equity[-1] - 1.0,
        "turnover": daily["turnover"].sum(),
    }
    return {key: float(value) if np.isfinite(value) else None for key, value in values.items()}


def print_metrics(results):
    print(f"{'Portfolio':<20} {'Cost bps':>8} {'Period':<6} {'CAGR':>9} {'Ann vol':>9} "
          f"{'Sharpe':>8} {'Max DD':>9} {'Total':>9} {'Turnover':>10}")
    for cost, books in results.items():
        for book, periods in books.items():
            for period, values in periods.items():
                formats = (".2%", ".2%", ".2f", ".2%", ".2%", ".2f")
                cells = [format(value, spec) if value is not None else "n/a"
                         for value, spec in zip(values.values(), formats)]
                cagr, vol, sharpe, drawdown, total, turnover = cells
                print(f"{book:<20} {cost:>8g} {period.upper():<6} {cagr:>9} {vol:>9} "
                      f"{sharpe:>8} {drawdown:>9} {total:>9} {turnover:>9}x")


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    tickers = config["universe"]
    data_start = pd.Timestamp(config["data_start"])
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not data_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Configure warmup data and chronologically ordered IS and OOS ranges")
    lookback = config["lookback_sessions"]
    minimum = config["min_observations"]
    fraction = config["quartile_fraction"]
    if not 2 <= minimum <= lookback or not 0 < fraction <= 0.5:
        raise ValueError("Invalid Amihud lookback, observation minimum, or quartile fraction")
    cost = config["transaction_cost_bps"]
    costs = dict.fromkeys([cost, *config["sensitivity_cost_bps"]])
    if not all(0 <= rate < 10_000 for rate in costs):
        raise ValueError("Costs must be nonnegative and less than 10,000 bps")

    panel = load_prices(tickers, data_start, oos_end)
    if panel.empty or panel.index[-1].to_period("M") != oos_end.to_period("M"):
        raise RuntimeError("Yahoo Finance data do not reach the configured final month")
    open_, close, adj_close, volume = (
        panel[field].reindex(columns=tickers).where(lambda values: values > 0)
        for field in FIELDS
    )
    adj_open = open_ * (adj_close / close)
    # Yahoo's Close and Volume use a consistent split-adjusted share basis.
    daily_impact = adj_close.pct_change(fill_method=None).abs() * 1_000_000 / (close * volume)
    scores = daily_impact.replace([np.inf, -np.inf], np.nan).rolling(lookback, min_periods=minimum).mean()
    targets = portfolio_targets(scores, adj_close, fraction)
    overnight = adj_open / adj_close.shift(1)
    intraday = adj_close / adj_open

    results = {}
    for rate in costs:
        results[rate] = {}
        for name in BOOKS:
            daily = simulate(overnight, intraday, targets[name], is_start, oos_end, rate)
            periods = {"is": daily.loc[is_start:is_end], "oos": daily.loc[oos_start:oos_end], "full": daily}
            results[rate][name] = {
                period: metrics(sample, config["annualization_sessions"])
                for period, sample in periods.items()
            }

    full_dates = panel.loc[is_start:oos_end].index
    data_range = f"{full_dates[0].date()} to {full_dates[-1].date()}"
    high, low, benchmark = BOOKS
    base = results[cost]
    output = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **base[high],
        "benchmark": {"name": benchmark, **base[benchmark]},
        "low_illiquidity": {"name": low, **base[low]},
        "transaction_cost_bps": cost,
        "cost_sensitivity": {str(rate): results[rate] for rate in costs if rate != cost},
    }
    print(f"{config['strategy']} | Daily returns {data_range}")
    print(f"IS {is_start.date()} to {is_end.date()} | OOS {oos_start.date()} to {oos_end.date()}")
    print_metrics(results)
    print("RESULTS_JSON: " + json.dumps(output, allow_nan=False))


if __name__ == "__main__":
    main()
