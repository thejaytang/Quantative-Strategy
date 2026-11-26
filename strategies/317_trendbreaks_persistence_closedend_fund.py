# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 317_trendbreaks_persistence_closedend_fund\n\nuniverse:\n  gold: [PHYS, GLD]\n  silver: [PSLV, SLV]\n\ndata_start: 2010-11-01\nin_sample:\n  start: 2012-01-01\n  end: 2020-12-31\nout_of_sample:\n  start: 2021-01-01\n  end: 2025-12-31\n\nsignal:\n  lookback_sessions: 252\n  break_ages_sessions: [63, 126]\n  min_bic_improvement: 6.0\n  entry_z: 2.0\n  exit_z: 0.5\n  max_holding_sessions: 10\n\nportfolio:\n  gross_fraction_at_entry: 1.0\n  transaction_cost_bps: 5.0\n  annual_short_borrow_bps: 100.0\n  sessions_per_year: 252\n'
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
"""Walk-forward matched bullion pairs with a weekly trend-break model."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent


def load_prices(symbols, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"yahoo_adjusted_{'_'.join(symbols)}_{start.date()}_{end.date()}.csv"
    columns = pd.MultiIndex.from_product([["Open", "Close"], symbols])

    def validate(prices):
        prices = prices.reindex(columns=columns).sort_index().loc[start:end]
        values = prices.to_numpy(dtype=float)
        if not len(prices) or not prices.index.is_unique:
            raise ValueError("Yahoo prices are empty or have duplicate dates.")
        if not np.all(np.isfinite(values) & (values > 0)):
            invalid = prices.index[~(np.isfinite(values) & (values > 0)).all(axis=1)]
            raise ValueError(f"Yahoo prices are incomplete or invalid starting {invalid[0].date()}.")
        week = pd.Timedelta(days=7)
        if prices.index[0] > start + week or prices.index[-1] < end - week:
            raise ValueError("Yahoo prices do not cover the configured date range.")
        return prices

    if cache.exists():
        return validate(pd.read_csv(cache, header=[0, 1], index_col=0, parse_dates=True))

    import yfinance as yf

    with TemporaryDirectory(dir=cache_dir, prefix="yahoo_download_") as temp:
        yf.set_tz_cache_location(temp)
        downloaded = yf.download(
            symbols,
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True,
            progress=False,
            timeout=15,
        )
        if downloaded is None:
            raise RuntimeError("Yahoo Finance did not return price history.")
        prices = validate(downloaded)
        staging = Path(temp) / cache.name
        prices.to_csv(staging, index_label="Date")
        staging.replace(cache)
    return prices


def fit_line(y, x):
    """Fit an intercept and slope with an RSS expressed in spread units."""
    centered_x = x - x.mean()
    mean_y = y.mean()
    slope = centered_x @ (y - mean_y) / (centered_x @ centered_x)
    intercept = mean_y - slope * x.mean()
    residual = y - (intercept + slope * x)
    rss = residual @ residual
    return intercept, slope, rss, np.sqrt(rss / (len(y) - 2))


def trend_scores(spread, rules):
    """Return daily scores from the break-aware and single-line weekly fits."""
    lookback = rules["lookback_sessions"]
    ages = rules["break_ages_sessions"]
    x = np.arange(lookback, dtype=float)
    values = spread.to_numpy()
    weeks = spread.index.to_period("W-SUN")
    scores = np.full((len(values), 2), np.nan)
    tiny = np.finfo(float).tiny

    for t in range(lookback, len(values)):
        if t == lookback or weeks[t] != weeks[t - 1]:
            fit_day = t
            # Freeze both fits for the week before scoring its first close.
            y = values[t - lookback:t]
            single = fit_line(y, x)
            selected = single
            single_bic = lookback * np.log(max(single[2] / lookback, tiny)) + 2 * np.log(lookback)
            best_bic = single_bic - rules["min_bic_improvement"]
            for age in ages:
                split = lookback - age
                first = fit_line(y[:split], x[:split])
                last = fit_line(y[split:], x[split:])
                # Two intercepts, two slopes and one selected breakpoint.
                rss = first[2] + last[2]
                bic = lookback * np.log(max(rss / lookback, tiny)) + 5 * np.log(lookback)
                if bic <= best_bic:
                    selected, best_bic = last, bic

        step = lookback + t - fit_day
        for model, (intercept, slope, _, sigma) in enumerate((selected, single)):
            if sigma > 1e-12:
                scores[t, model] = (values[t] - (intercept + slope * step)) / sigma
    return pd.DataFrame(scores, index=spread.index, columns=["break_aware", "single_trend"])


def next_open_directions(scores, rules):
    """Convert closing scores into target directions at the following open."""
    closing_decisions = np.zeros(len(scores), dtype=int)
    direction, age = 0, 0
    for t, z in enumerate(scores):
        if direction:
            age += 1
            if age >= rules["max_holding_sessions"] or direction * z >= -rules["exit_z"]:
                direction, age = 0, 0
        elif abs(z) > rules["entry_z"]:
            direction = 1 if z < 0 else -1
        closing_decisions[t] = direction
    return pd.Series(closing_decisions).shift(1, fill_value=0).to_numpy()


def simulate(prices, scores, config):
    """Mark to close, changing each pair's fixed shares only at its entry or exit."""
    targets = np.column_stack([next_open_directions(z, config["signal"]) for z in scores])
    n_pairs = targets.shape[1]
    opens = prices["Open"].to_numpy().reshape(-1, n_pairs, 2)
    closes = prices["Close"].to_numpy().reshape(-1, n_pairs, 2)
    days = prices.index.to_series().diff().dt.days.fillna(0).to_numpy()
    portfolio = config["portfolio"]
    leg_weight = portfolio["gross_fraction_at_entry"] / (n_pairs * 2)
    cost_rate = portfolio["transaction_cost_bps"] / 10_000
    daily_borrow = portfolio["annual_short_borrow_bps"] / (10_000 * 365)

    shares = np.zeros((n_pairs, 2))
    previous_targets = np.zeros(n_pairs, dtype=int)
    cash = previous_equity = 1.0
    short_value = 0.0
    results = np.zeros((len(prices), 3))
    leg_signs = np.array([1, -1])

    for t, (target, opening, closing) in enumerate(zip(targets, opens, closes)):
        cash -= short_value * days[t] * daily_borrow
        opening_equity = cash + (shares * opening).sum()
        if opening_equity <= 0:
            raise ValueError(f"Portfolio equity depleted at the open on {prices.index[t].date()}.")

        changes = target != previous_targets
        new_shares = np.outer(target[changes], leg_signs) * (opening_equity * leg_weight) / opening[changes]
        traded = (new_shares - shares[changes]) * opening[changes]
        gross_traded = np.abs(traded).sum()
        cash -= traded.sum() + gross_traded * cost_rate
        shares[changes] = new_shares

        positions = shares * closing
        equity = cash + positions.sum()
        if equity <= 0:
            raise ValueError(f"Portfolio equity depleted at the close on {prices.index[t].date()}.")
        pair_entries = np.count_nonzero(changes & (target != 0))
        results[t] = (equity / previous_equity - 1, gross_traded / opening_equity, pair_entries)
        short_value = -np.minimum(positions, 0).sum()
        previous_targets, previous_equity = target, equity

    return pd.DataFrame(results, index=prices.index, columns=["return", "turnover", "n_trades"])


def metrics(results, sessions_per_year):
    daily = results["return"].to_numpy()
    if not len(daily):
        raise ValueError("No common price sessions in the configured evaluation period.")
    equity = np.r_[1.0, np.cumprod(1.0 + daily)]
    annual_vol = daily.std(ddof=1) * np.sqrt(sessions_per_year) if len(daily) > 1 else np.nan
    summary = {
        "cagr": equity[-1] ** (sessions_per_year / len(daily)) - 1,
        "ann_vol": annual_vol,
        "sharpe": daily.mean() * sessions_per_year / annual_vol if annual_vol > 0 else np.nan,
        "max_drawdown": (equity / np.maximum.accumulate(equity) - 1).min(),
        "total_return": equity[-1] - 1,
        "turnover": results["turnover"].sum(),
    }
    return {key: float(value) if np.isfinite(value) else None for key, value in summary.items()} | {
        "n_trades": int(results["n_trades"].sum())
    }


def print_table(strategy_summary, benchmark_summary):
    def fmt(value, spec):
        return format(value, spec) if value is not None else "n/a"

    print(f"{'Model':<15} {'Period':<6} {'CAGR':>9} {'Ann. vol':>9} {'Sharpe':>8} "
          f"{'Max DD':>9} {'Total':>9} {'Trades':>7} {'Turnover':>9}")
    for name, summary in (("Break-aware", strategy_summary), ("Single trend", benchmark_summary)):
        for period, row in summary.items():
            print(f"{name:<15} {period.upper():<6} {fmt(row['cagr'], '.2%'):>9} "
                  f"{fmt(row['ann_vol'], '.2%'):>9} {fmt(row['sharpe'], '.2f'):>8} "
                  f"{fmt(row['max_drawdown'], '.2%'):>9} {fmt(row['total_return'], '.2%'):>9} "
                  f"{row['n_trades']:>7} {fmt(row['turnover'], '.2f'):>9}")


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    data_start = pd.Timestamp(config["data_start"])
    if not data_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Configure ordered, nonoverlapping IS and OOS ranges after the warm-up start.")

    pairs = list(config["universe"].values())
    symbols = [symbol for pair in pairs for symbol in pair]
    if not pairs or any(len(pair) != 2 for pair in pairs) or len(set(symbols)) != len(symbols):
        raise ValueError("Each pair needs two tickers and each ticker must be unique.")
    rules = config["signal"]
    lookback = rules["lookback_sessions"]
    if not all(3 <= age <= lookback - 3 for age in rules["break_ages_sessions"]):
        raise ValueError("Each breakpoint must leave at least three sessions on both sides.")

    prices = load_prices(symbols, data_start, oos_end)
    if (prices.index < is_start).sum() < lookback:
        raise ValueError("Insufficient price history for the first IS trend fit.")
    log_close = np.log(prices["Close"])
    pair_scores = [trend_scores(log_close[trust] - log_close[hedge], rules).loc[is_start:]
                   for trust, hedge in pairs]
    evaluation_prices = prices.loc[is_start:]
    actual_start, actual_end = evaluation_prices.index[[0, -1]].date
    data_range = f"{actual_start} to {actual_end}"

    summaries = {}
    for model in ("break_aware", "single_trend"):
        results = simulate(evaluation_prices, [score[model] for score in pair_scores], config)
        annualization = config["portfolio"]["sessions_per_year"]
        summaries[model] = {
            "is": metrics(results.loc[is_start:is_end], annualization),
            "oos": metrics(results.loc[oos_start:oos_end], annualization),
            "full": metrics(results, annualization),
        }

    strategy_summary = summaries["break_aware"]
    benchmark_summary = summaries["single_trend"]
    print(f"{config['strategy']} | {data_range} | Yahoo Finance adjusted prices")
    print(f"IS {is_start.date()} to {is_end.date()} | OOS {oos_start.date()} to {oos_end.date()}")
    print("Fixed C108 signal parameters, no performance tuning. Turnover is in multiples of equity.")
    print_table(strategy_summary, benchmark_summary)
    payload = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **strategy_summary,
        "benchmark": {"name": "single_trend", **benchmark_summary},
    }
    print("RESULTS_JSON: " + json.dumps(payload, allow_nan=False))


if __name__ == "__main__":
    main()
