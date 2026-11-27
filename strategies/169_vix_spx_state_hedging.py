# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 169_vix_spx_state_hedging\ncategory: volatility\n\ntickers:\n  equity: SPY\n  hedge: VIXM\n  treasury: BIL\n  volatility: ^VIX\n  volatility_of_volatility: ^VVIX\n\ndata_start: '2011-01-01'\nin_sample:\n  start: '2016-01-01'\n  end: '2019-12-31'\nout_of_sample:\n  start: '2020-01-01'\n  end: '2025-12-31'\n\nstate_lookback_sessions: 252\ncovariance_lookback_sessions: 756\nminimum_state_observations: 40\nequity_weight: 0.85\n# Maximum VIXM budget. Scale it by the hedge ratio per unit of equity (clipped to [0, 1]).\nmax_hedge_weight: 0.15\n\n# One-way bps on each ETF's traded notional, including initial purchases.\ntransaction_cost_bps:\n  equity: 2\n  hedge: 10\n  treasury: 2\n\n# Extra one-way execution slippage in bps per traded notional.\nslippage_bps:\n  equity: 1\n  hedge: 5\n  treasury: 1\n"
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
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
ETF_ROLES = ("equity", "hedge", "treasury")
INDEX_ROLES = ("volatility", "volatility_of_volatility")
SESSIONS_PER_YEAR = 252
PARAM_COMBOS_TRIED = 1
OSLO = ZoneInfo("Europe/Oslo")


def cached_series(ticker, field, start, end):
    symbol = ticker.replace("^", "index_")
    path = CACHE / f"yahoo_{symbol}_{field}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    from_cache = path.exists()
    if from_cache:
        series = pd.read_csv(path, index_col="date", parse_dates=["date"])[field]
    else:
        bars = yf.Ticker(ticker).history(
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=(field == "Open"),
            timeout=15,
        )
        if bars.empty:
            raise ValueError(f"No Yahoo history for {ticker}")
        series = bars[field].copy()
        series.index = series.index.tz_localize(None).normalize()

    series = pd.to_numeric(series, errors="coerce")
    series = series[~series.index.duplicated(keep="last")].sort_index().loc[start:end]
    series = series[series.gt(0) & np.isfinite(series)].rename(field)
    if series.empty or series.index[-1] < end - pd.Timedelta(days=7):
        raise ValueError(f"Yahoo {field} history for {ticker} is empty or ends before the test window")
    if not from_cache:
        with TemporaryDirectory(dir=CACHE) as staging:
            temporary = Path(staging) / path.name
            series.to_csv(temporary, index_label="date")
            temporary.replace(path)
    snapshot_date = datetime.fromtimestamp(path.stat().st_mtime, tz=OSLO).date()
    return series, snapshot_date


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(CACHE / "yfinance_metadata"))
    data = {}
    fetch_dates = set()
    for roles, field in [(ETF_ROLES, "Open"), (INDEX_ROLES, "Close")]:
        for role in roles:
            data[role], snapshot_date = cached_series(tickers[role], field, start, end)
            fetch_dates.add(snapshot_date)

    etfs = {role: data[role] for role in ETF_ROLES}
    # Trim pre-inception rows and require an open for each ETF on every observed session.
    opens = pd.DataFrame(etfs)
    opens = opens.loc[max(price.index.min() for price in etfs.values()):]
    if opens.empty or opens.isna().any().any():
        raise ValueError("ETF opening history has missing prices")

    closes = pd.DataFrame({role: data[role] for role in INDEX_ROLES}, index=opens.index)
    return opens, closes, sorted(fetch_dates)


def hedge_targets(opens, closes, config):
    # Use only jointly observed closes for the medians, while keeping the trading calendar.
    observed = closes.dropna()
    medians = observed.rolling(config["state_lookback_sessions"]).median()
    above = observed.gt(medians)
    state = (2 * above[INDEX_ROLES[0]] + above[INDEX_ROLES[1]]).where(
        medians.notna().all(axis=1)
    ).reindex(opens.index).rename("state")

    # A signal at close s forecasts the interval from open s+1 to open s+2.
    # Index the pairs by s+2 so a signal at t can use only pairs ending by t.
    realized = opens.loc[:, ["equity", "hedge"]].pct_change(fill_method=None)
    completed = realized.join(state.shift(2)).dropna()
    equity_weight = config["equity_weight"]
    lookback = config["covariance_lookback_sessions"]
    minimum = config["minimum_state_observations"]
    hedge_budget = config["max_hedge_weight"]
    if not 2 <= minimum <= lookback:
        raise ValueError("Require 2 <= minimum_state_observations <= covariance_lookback_sessions")
    residual_weight = 1 - equity_weight
    if not (0 <= equity_weight <= 1 and 0 <= hedge_budget <= residual_weight + 1e-12):
        raise ValueError("Equity and hedge budget must fit in an unlevered portfolio")
    hedge_budget = min(hedge_budget, residual_weight)

    def hedge_weight(sample):
        variance = sample["hedge"].var()
        if not np.isfinite(variance) or variance <= 0:
            return None
        covariance = sample["equity"].cov(sample["hedge"])
        hedge_per_unit_equity = -covariance / variance
        return float(hedge_budget * np.clip(hedge_per_unit_equity, 0, 1))

    conditional = pd.DataFrame(np.nan, index=opens.index, columns=list(ETF_ROLES))
    pooled = conditional.copy()
    week_ends = state.groupby(state.index.to_period("W-FRI")).tail(1)
    for date, current_state in week_ends.items():
        history = completed.loc[:date].tail(lookback)
        if len(history) < lookback or pd.isna(current_state):
            continue
        pooled_weight = hedge_weight(history)
        if pooled_weight is None:
            pooled_weight = 0.0
        subset = history.loc[history["state"].eq(current_state)]
        state_weight = hedge_weight(subset) if len(subset) >= minimum else None
        if state_weight is None:
            state_weight = pooled_weight

        conditional.loc[date] = equity_weight, state_weight, residual_weight - state_weight
        pooled.loc[date] = equity_weight, pooled_weight, residual_weight - pooled_weight

    return conditional.shift(1), pooled.shift(1)


def simulate(opens, targets, cost_rates):
    """Rebalance at the opening print and mark NAV at the following open."""
    asset_values = np.zeros(len(ETF_ROLES))
    asset_growth = (opens.shift(-1) / opens).to_numpy()
    rows = []
    for growth, target in zip(asset_growth[:-1], targets.to_numpy()[:-1]):
        turnover = 0.0
        trades = 0
        remaining_nav = 1.0
        if np.isfinite(target).all():
            # Fees are assessed on the actual trades to reach post-fee weights.
            def budget(nav):
                return nav + cost_rates @ np.abs(nav * target - asset_values) - 1.0

            remaining_nav = brentq(budget, 0.0, 1.0)
            post_trade = remaining_nav * target
            traded = np.abs(post_trade - asset_values)
            turnover = traded.sum()
            trades = np.count_nonzero(traded > 1e-10)
            asset_values = post_trade

        ending_assets = asset_values * growth
        ending_nav = remaining_nav + (ending_assets - asset_values).sum()
        asset_values = ending_assets / ending_nav
        rows.append((ending_nav - 1.0, turnover, trades))

    return pd.DataFrame(rows, index=opens.index[1:], columns=["return", "turnover", "n_trades"])


def safe_float(value):
    return float(value) if np.isfinite(value) else None


def annualized_sharpe(returns):
    daily_vol = returns.std()
    if daily_vol > 0:
        return safe_float(returns.mean() / daily_vol * np.sqrt(SESSIONS_PER_YEAR))
    return None


def metrics(path, initial_date):
    returns = path["return"]
    nav = (1 + returns).cumprod()
    elapsed_days = (nav.index[-1] - initial_date).days
    drawdown = nav / nav.cummax().clip(lower=1.0) - 1
    return {
        "cagr": safe_float(nav.iloc[-1] ** (365.25 / elapsed_days) - 1),
        "ann_vol": safe_float(returns.std() * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": annualized_sharpe(returns),
        "sharpe_excess": annualized_sharpe(returns - path["cash_return"]),
        "max_drawdown": safe_float(drawdown.min()),
        "total_return": safe_float(nav.iloc[-1] - 1),
        "n_trades": int(path["n_trades"].sum()),
        "turnover": safe_float(path["turnover"].sum()),
    }


def report_periods(path, initial_date, oos_open):
    in_sample = path.loc[:oos_open]
    out_of_sample = path.loc[path.index > oos_open]
    if in_sample.empty or out_of_sample.empty:
        raise ValueError("Both IS and OOS need at least one completed return interval")
    return {
        "is": metrics(in_sample, initial_date),
        "oos": metrics(out_of_sample, oos_open),
        "full": metrics(path, initial_date),
    }


def print_hedge_diagnostics(conditional, pooled, hedge_budget):
    # The final open is the valuation endpoint and does not execute a new target.
    conditional = conditional.iloc[:-1].dropna()
    pooled = pooled.loc[conditional.index]
    for name, targets in [("state", conditional), ("pooled", pooled)]:
        hedge = targets["hedge"]
        print(
            f"HEDGE_WEIGHTS {name} n={len(hedge)} min={hedge.min():.4%} "
            f"p25={hedge.quantile(0.25):.4%} median={hedge.median():.4%} "
            f"p75={hedge.quantile(0.75):.4%} max={hedge.max():.4%} "
            f"at_cap={np.isclose(hedge, hedge_budget, rtol=0, atol=1e-12).sum()} "
            f"BIL_min={targets['treasury'].min():.4%} BIL_max={targets['treasury'].max():.4%}"
        )
    gap = (conditional["hedge"] - pooled["hedge"]).abs()
    print(
        f"HEDGE_WEIGHT_GAP n={len(gap)} different={gap.gt(1e-12).sum()} "
        f"median_abs={gap.median():.4%} max_abs={gap.max():.4%}"
    )


def print_table(portfolios):
    header = f"{'Portfolio':<22} {'Period':<6} {'CAGR':>10} {'Ann. vol':>10} {'Sharpe':>8} "
    print(f"{header}{'Sharpe-BIL':>10} {'Max DD':>10} {'Return':>10} {'Trades':>7} {'Turnover':>10}")
    for name, periods in portfolios.items():
        for period, values in periods.items():
            columns = []
            for key, width, spec in [
                ("cagr", 10, ".2%"), ("ann_vol", 10, ".2%"), ("sharpe", 8, ".2f"),
                ("sharpe_excess", 10, ".2f"),
                ("max_drawdown", 10, ".2%"), ("total_return", 10, ".2%"),
                ("n_trades", 7, "d"), ("turnover", 10, ".2f"),
            ]:
                value = values[key]
                display = format(value, spec) if value is not None else "—"
                columns.append(display.rjust(width))
            print(f"{name:<22} {period.upper():<6} " + " ".join(columns))


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_start = pd.Timestamp(config["data_start"])
    is_start = pd.Timestamp(config["in_sample"]["start"])
    is_end = pd.Timestamp(config["in_sample"]["end"])
    oos_start = pd.Timestamp(config["out_of_sample"]["start"])
    oos_end = pd.Timestamp(config["out_of_sample"]["end"])
    if not data_start < is_start < is_end < oos_start < oos_end:
        raise ValueError("Expected warmup, IS and OOS dates in chronological order")

    opens, closes, fetch_dates = load_prices(config["tickers"], data_start, oos_end)
    conditional, pooled = hedge_targets(opens, closes, config)
    opens = opens.loc[is_start:oos_end]
    if opens.empty or opens.index[0] > is_start + pd.Timedelta(days=7):
        raise ValueError("ETF opening history does not cover the configured IS start")
    oos_location = opens.index.searchsorted(oos_start)
    if not 0 < oos_location < len(opens) - 1 or opens.index[oos_location - 1] > is_end:
        raise ValueError("The IS and OOS windows must cover adjacent open-to-open intervals")
    oos_open = opens.index[oos_location]

    targets = {"State-dependent hedge": conditional, "Pooled hedge": pooled}
    for name, target in targets.items():
        targets[name] = target.reindex(opens.index).copy()
        initial = target.loc[:opens.index[0]].dropna()
        if initial.empty:
            raise ValueError("Insufficient warmup to establish the first weekly hedge")
        targets[name].iloc[0] = initial.iloc[-1]

    benchmark_name = f"{config['tickers']['equity']} buy-and-hold"
    benchmark = pd.DataFrame(np.nan, index=opens.index, columns=list(ETF_ROLES))
    benchmark.iloc[0] = 1.0, 0.0, 0.0
    targets[benchmark_name] = benchmark
    fees = np.array([config["transaction_cost_bps"][role] for role in ETF_ROLES])
    slippage = np.array([config["slippage_bps"][role] for role in ETF_ROLES])
    cost_rates = (fees + slippage) / 10_000
    if not np.all((fees >= 0) & (slippage >= 0) & (cost_rates < 1)):
        raise ValueError("Costs and slippage must be nonnegative and total less than 10000 bps per ETF")
    cash_returns = opens["treasury"].pct_change(fill_method=None).rename("cash_return")
    portfolios = {}
    for name, target in targets.items():
        path = simulate(opens, target, cost_rates).join(cash_returns)
        portfolios[name] = report_periods(path, opens.index[0], oos_open)

    data_range = f"{opens.index[0].date()} to {opens.index[-1].date()}"
    print(f"{config['strategy']} | {data_range} | OOS begins at the {oos_open.date()} open")
    print(f"PARAM_COMBOS_TRIED: {PARAM_COMBOS_TRIED}")
    print("DATA_SOURCE: Yahoo Finance via yfinance")
    print("DATA_FETCHED_AT: " + ", ".join(day.isoformat() for day in fetch_dates))
    print("DATA_FETCH_DATE_BASIS: CSV cache modification dates (Europe/Oslo)")
    print("PRICE_ADJUSTMENT: yfinance dividend- and split-adjusted ETF opens; index closes")
    print("MISSING_INDEX_CLOSES: " + ", ".join(
        f"{config['tickers'][role]}={closes[role].isna().sum()}" for role in INDEX_ROLES
    ))
    print_hedge_diagnostics(
        targets["State-dependent hedge"], targets["Pooled hedge"], config["max_hedge_weight"]
    )
    print_table(portfolios)
    buy_and_hold = portfolios[benchmark_name]
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **portfolios["State-dependent hedge"],
        "benchmark": {
            "name": benchmark_name,
            **buy_and_hold["full"],
            **buy_and_hold,
            "pooled_hedge": portfolios["Pooled hedge"],
        },
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
