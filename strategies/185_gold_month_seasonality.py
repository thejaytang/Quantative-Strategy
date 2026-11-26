# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 185_gold_month_seasonality\nuniverse:\n  gold: GLD\n  treasury_bills: BIL\ndata_start: 2007-12-01\nin_sample:\n  start: 2016-01-01\n  end: 2020-12-31\nout_of_sample:\n  start: 2021-01-01\n  end: 2025-12-31\nlookback_years: 8\ntop_months: 4\ntransaction_cost_bps: 5\npermutation_draws: 999\nrandom_seed: 185\n'
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
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"


def load_prices(ticker, start, end):
    """Read adjusted daily prices, with end exclusive."""
    cache_file = CACHE / f"{ticker}_{start:%Y%m%d}_{end:%Y%m%d}_adjusted.csv"
    if cache_file.exists():
        prices = pd.read_csv(cache_file, index_col="Date", parse_dates=["Date"])
    else:
        daily = yf.Ticker(ticker).history(
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            auto_adjust=True,
            actions=False,
            timeout=20,
        )
        if daily.empty:
            raise RuntimeError(f"Yahoo Finance returned no prices for {ticker}")
        prices = daily[["Open", "Close"]].copy()
        prices.index = prices.index.tz_localize(None)
    prices = prices.sort_index()
    prices.index.name = "Date"
    if (
        prices.empty
        or prices.index.has_duplicates
        or prices.index[0] < start
        or prices.index[0] > start + pd.Timedelta(days=7)
        or prices.index[-1] >= end
        or not np.isfinite(prices.to_numpy()).all()
        or (prices <= 0).any().any()
    ):
        raise ValueError(f"Invalid adjusted daily prices for {ticker}")
    if not cache_file.exists():
        with NamedTemporaryFile(dir=CACHE, suffix=".tmp", delete=False) as temporary:
            temporary_file = Path(temporary.name)
        try:
            prices.to_csv(temporary_file)
            temporary_file.replace(cache_file)
        finally:
            temporary_file.unlink(missing_ok=True)
    return prices


def load_market_data(universe, start, terminal_month):
    CACHE.mkdir(exist_ok=True)
    end = terminal_month.start_time + pd.Timedelta(days=10)
    with TemporaryDirectory(dir=CACHE) as yahoo_cache:
        yf.set_tz_cache_location(yahoo_cache)
        prices = {asset: load_prices(ticker, start, end) for asset, ticker in universe.items()}

    daily = pd.concat(prices, axis=1)
    if daily.isna().any().any():
        raise ValueError("Gold and Treasury-bill price dates do not match")
    months = daily.index.to_period("M")
    expected_months = pd.period_range(start.to_period("M"), terminal_month, freq="M")
    if not months.unique().equals(expected_months):
        raise ValueError("Daily prices do not cover every required calendar month")
    terminal_open = daily.index[months == terminal_month][0]
    if terminal_open > terminal_month.start_time + pd.Timedelta(days=7):
        raise ValueError("Missing first open of the terminal month")
    return daily.loc[:terminal_open]


def seasonal_weights(daily_closes, months, lookback_years, top_months):
    close_to_close = daily_closes.groupby(daily_closes.index.to_period("M")).last().pct_change()
    excess = close_to_close["gold"] - close_to_close["treasury_bills"]
    calendar_returns = pd.DataFrame({"year": excess.index.year, "month": excess.index.month, "excess": excess})
    annual = calendar_returns.pivot(index="year", columns="month", values="excess")

    weights = pd.Series(0.0, index=months)
    for year in months.year.unique():
        # The prior December close completes this window before the first January open.
        history = annual.reindex(range(year - lookback_years, year))
        if history.isna().any().any():
            raise ValueError(f"Missing complete lookback years for {year}")
        scores = history.mean() / history.std(ddof=1)
        if not np.isfinite(scores).all():
            raise ValueError(f"Undefined seasonal score for {year}")
        ranked = scores.sort_index().sort_values(ascending=False, kind="stable")
        weights.loc[(months.year == year) & months.month.isin(ranked.index[:top_months])] = 1.0
    return weights


def monthly_portfolios(first_opens, weights, cost_rate):
    """Calculate each portfolio's monthly net factor and summed trade notionals."""
    growth = first_opens.shift(-1).div(first_opens).iloc[:-1]
    gold = growth["gold"].to_numpy()[:, None]
    bills = growth["treasury_bills"].to_numpy()[:, None]
    target = weights.to_numpy()
    gross = target * gold + (1 - target) * bills
    end_gold = target * gold / gross
    # One initial purchase from cash. Thereafter both legs incur one-way cost.
    turnover = np.ones_like(target)
    turnover[1:] = 2 * np.abs(target[1:] - end_gold[:-1])
    turnover = pd.DataFrame(turnover, index=weights.index, columns=weights.columns)
    return gross * (1 - cost_rate * turnover), turnover


def daily_portfolios(opens, first_opens, weights, turnover, cost_rate):
    """Mark the drifting holdings at each open, with returns dated at interval start."""
    dates = opens.index[:-1]
    months = dates.to_period("M")
    base = first_opens.loc[months].to_numpy()
    start = opens.iloc[:-1].to_numpy() / base
    end = opens.iloc[1:].to_numpy() / base
    gold_weight = weights.loc[months].to_numpy()
    bills_weight = 1 - gold_weight
    start_value = start[:, :1] * gold_weight + start[:, 1:] * bills_weight
    end_value = end[:, :1] * gold_weight + end[:, 1:] * bills_weight
    net = end_value / start_value
    first_session = ~months.duplicated()
    net[first_session] *= 1 - cost_rate * turnover.to_numpy()
    return pd.DataFrame(net - 1, index=dates, columns=weights.columns)


def placebo_weights(weights, draws, seed):
    """Independently rotate each year's known seasonal schedule in 12-month blocks."""
    annual = weights.to_numpy().reshape(-1, 12)
    offsets = np.random.default_rng(seed).integers(12, size=(len(annual), 1, draws))
    shifted_months = (np.arange(12)[None, :, None] - offsets) % 12
    shifted = np.take_along_axis(annual[:, :, None], shifted_months, axis=1)
    return pd.DataFrame(shifted.reshape(len(weights), draws), index=weights.index)


def metrics(returns, terminal_open, turnover, weights):
    equity = (1 + returns).cumprod().to_numpy()
    path = np.r_[1.0, equity]
    years = (terminal_open - returns.index[0]).days / 365.2425
    volatility = returns.std(ddof=1)
    values = {
        "cagr": equity[-1] ** (1 / years) - 1,
        "ann_vol": volatility * np.sqrt(252),
        "sharpe": returns.mean() / volatility * np.sqrt(252) if volatility > 0 else None,
        "max_drawdown": (path / np.maximum.accumulate(path) - 1).min(),
        "total_return": equity[-1] - 1,
        "turnover": turnover.sum(),
        "gold_target_mean": weights.mean(),
    }
    return {key: float(value) if value is not None and np.isfinite(value) else None for key, value in values.items()}


def evaluate(daily, periods, weights, cost_rate, draws, seed):
    opens = daily.xs("Open", level=1, axis=1)[["gold", "treasury_bills"]]
    opens = opens.loc[opens.index.to_period("M") >= weights.index[0]]
    first_opens = opens.groupby(opens.index.to_period("M")).first()
    main_growth, turnover = monthly_portfolios(first_opens, weights, cost_rate)
    returns = daily_portfolios(opens, first_opens, weights, turnover, cost_rate)
    placebos = placebo_weights(weights["strategy"], draws, seed)
    placebo_growth, _ = monthly_portfolios(first_opens, placebos, cost_rate)

    open_dates = opens.index[~opens.index.to_period("M").duplicated()]
    month_open_dates = pd.Series(open_dates, index=first_opens.index)
    results = {column: {} for column in weights}
    for name, (start, end) in periods.items():
        mask = (weights.index >= start.to_period("M")) & (weights.index <= end.to_period("M"))
        closing_mark = month_open_dates.loc[end.to_period("M") + 1]
        for column in weights:
            results[column][name] = metrics(
                returns.loc[start:end, column], closing_mark, turnover.loc[mask, column], weights.loc[mask, column]
            )
        log_growth = np.log(main_growth.loc[mask]).sum()
        control = log_growth["static_mix"]
        observed = log_growth["strategy"] - control
        null = np.log(placebo_growth.loc[mask]).sum() - control
        results["strategy"][name]["placebo_p_value"] = float((1 + (null >= observed - 1e-12).sum()) / (draws + 1))
    return results, f"{opens.index[0].date()} to {opens.index[-1].date()}"


def print_table(results, gold_ticker):
    labels = {"strategy": "Seasonal", "buy_and_hold": f"{gold_ticker} buy-and-hold", "static_mix": "Static mix"}
    fields = {
        "CAGR": ("cagr", ".2%"),
        "Vol": ("ann_vol", ".2%"),
        "Sharpe": ("sharpe", ".2f"),
        "Max DD": ("max_drawdown", ".2%"),
        "Total": ("total_return", ".2%"),
        "Turnover": ("turnover", ".2f"),
        "Gold wt": ("gold_target_mean", ".2%"),
        "Block p": ("placebo_p_value", ".3f"),
    }
    rows = []
    for portfolio, splits in results.items():
        for split, values in splits.items():
            row = {"Portfolio": labels[portfolio], "Period": split.upper()}
            for label, (key, fmt) in fields.items():
                value = values.get(key)
                row[label] = format(value, fmt) if value is not None else "-"
            rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    periods = {
        name: (pd.Timestamp(config[key]["start"]), pd.Timestamp(config[key]["end"]))
        for name, key in (("is", "in_sample"), ("oos", "out_of_sample"))
    }
    for start, end in periods.values():
        if start > end or (start.month, start.day, end.month, end.day) != (1, 1, 12, 31):
            raise ValueError("Both reporting periods must cover whole calendar years")
    if periods["is"][1] + pd.Timedelta(days=1) != periods["oos"][0]:
        raise ValueError("The out-of-sample period must start the day after the in-sample period")
    start, end = periods["full"] = (periods["is"][0], periods["oos"][1])
    cost_bps = config["transaction_cost_bps"]
    lookback, top_months = config["lookback_years"], config["top_months"]
    draws = config["permutation_draws"]
    if lookback < 2 or not 1 <= top_months <= 12 or draws < 1 or not 0 <= cost_bps < 5000:
        raise ValueError("Invalid lookback, month count, permutation count, or cost")

    months = pd.period_range(start, end, freq="M")
    daily = load_market_data(universe, pd.Timestamp(config["data_start"]), months[-1] + 1)
    seasonal = seasonal_weights(daily.xs("Close", level=1, axis=1), months, lookback, top_months)
    weights = pd.DataFrame({"strategy": seasonal, "buy_and_hold": 1.0, "static_mix": top_months / 12})
    results, data_range = evaluate(daily, periods, weights, cost_bps / 10_000, draws, config["random_seed"])
    buy_and_hold = results["buy_and_hold"]
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **results["strategy"],
        "benchmark": {
            "name": f"{universe['gold']} buy-and-hold",
            **buy_and_hold["full"],
            "is": buy_and_hold["is"],
            "oos": buy_and_hold["oos"],
            "static_mix": results["static_mix"],
        },
    }
    print(f"{config['strategy']} | {data_range} | {cost_bps:g} bps one-way")
    print_table(results, universe["gold"])
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
