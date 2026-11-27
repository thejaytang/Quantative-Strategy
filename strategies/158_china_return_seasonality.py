# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 158_china_return_seasonality\nmarket_label: US large-cap equities (fixed US24 basket)\ncategory: seasonality\n\nuniverse: [AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, BAC, GS, MS, XOM, CVX,\n           COP, SLB, WMT, COST, PG, KO, PEP, JNJ, PFE, MRK, HD, LOW]\ncalendar_ticker: SPY\ncash_ticker: BIL\nhistory_start: 2003-01-01\n\n# Inclusive dates for the starts of open-to-open holding intervals.\nin_sample:\n  start: 2012-01-01\n  end: 2018-12-31\nout_of_sample:\n  start: 2019-01-01\n  end: 2025-12-31\n\nlookback_years: 8\nminimum_occurrences: 6\nminimum_eligible: 16\nholdings: 6\nplacebo_month_shift: 6\n\ntransaction_fee_bps: 3\nslippage_bps: 2\n# Alternative all-in rates per dollar bought or sold.\ncost_sensitivity_bps: [0, 15]\n'
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

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
STRATEGY = "Same calendar month"
BENCHMARK = "Eligible US24 equal weight"
PRIOR_MONTH = "Prior available month"
PLACEBO = "Calendar permutation"
PORTFOLIOS = (STRATEGY, BENCHMARK, PRIOR_MONTH, PLACEBO)
PRICE_ADJUSTMENT = "Yahoo Finance via yfinance auto_adjust=True; split- and dividend-adjusted Open and Close"


def load_prices(config, end):
    start = pd.Timestamp(config["history_start"])
    # Include the next trading open for the final open-to-open return.
    download_end = end + pd.Timedelta(days=10)
    symbols = sorted(set(config["universe"] + [config["cash_ticker"], config["calendar_ticker"]]))
    fields = ["Open", "Close"]
    columns = pd.MultiIndex.from_product([fields, symbols])
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    key = f"yf_adjusted_open_close_v1_{start.date()}_{download_end.date()}_{','.join(symbols)}"
    cache = cache_dir / f"prices_{hashlib.sha256(key.encode()).hexdigest()[:16]}.csv.gz"
    from_cache = cache.exists()

    if from_cache:
        with gzip.open(cache, "rb") as snapshot:
            prices = pd.read_csv(snapshot, header=[0, 1], index_col=0, parse_dates=True)
            fetched_at = snapshot.mtime
        if not fetched_at:
            raise RuntimeError(f"Yahoo Finance snapshot has no acquisition timestamp: {cache.name}")
    else:
        yf.set_tz_cache_location(str(cache_dir))
        prices = pd.DataFrame(columns=columns, index=pd.DatetimeIndex([]), dtype=float)
        missing = symbols
        for _ in range(2):
            batch = yf.download(
                missing, start=str(start.date()), end=str(download_end.date()),
                auto_adjust=True, progress=False, threads=8, timeout=10,
                multi_level_index=True,
            )
            batch = batch.reindex(columns=columns)
            prices = prices.combine_first(batch.where(np.isfinite(batch) & (batch > 0)))
            available = prices.notna().any()
            missing = [ticker for ticker in missing if not all(available[field, ticker] for field in fields)]
            if not missing:
                break
        if missing:
            raise RuntimeError(f"Yahoo Finance returned no usable prices for {', '.join(missing)}")
        fetched_at = int(datetime.now(timezone.utc).timestamp())

    prices = prices.reindex(columns=columns).sort_index()
    prices = prices.where(np.isfinite(prices) & (prices > 0))
    sessions = prices["Open"][config["calendar_ticker"]].dropna().index
    sessions = sessions[sessions >= start]
    if (sessions.empty or sessions[0] > start + pd.Timedelta(days=7)
            or sessions[-1] <= end):
        raise RuntimeError("Yahoo Finance prices do not span the warm-up and evaluation dates")
    if not from_cache:
        with NamedTemporaryFile(dir=cache_dir, delete=False) as temporary_file:
            temporary_cache = Path(temporary_file.name)
        try:
            # Store acquisition time in the gzip header so cache copies preserve it.
            prices.to_csv(temporary_cache, compression={"method": "gzip", "mtime": fetched_at})
            temporary_cache.replace(cache)
        finally:
            temporary_cache.unlink(missing_ok=True)
    timestamp = datetime.fromtimestamp(fetched_at, timezone.utc).isoformat()
    return prices["Open"].loc[sessions], prices["Close"].loc[sessions], timestamp


def seasonal_sample(completed, month, years):
    return completed.loc[completed.index.month == month].tail(years)


def matched_scores(history, counts):
    # Use each stock's most recent valid observations, matching its signal count.
    recent_first = history.iloc[::-1]
    matched = recent_first.where(recent_first.notna().cumsum().le(counts, axis="columns"))
    return matched.mean().where(matched.count().eq(counts))


def monthly_targets(opens, closes, config, start, end):
    stocks = config["universe"]
    years = config["lookback_years"]
    min_eligible = config["minimum_eligible"]
    first_dates = opens.index[~opens.index.to_period("M").duplicated()]
    first_opens = opens.loc[first_dates, stocks].set_axis(first_dates.to_period("M"))
    month_returns = first_opens.shift(-1).div(first_opens).sub(1)
    exit_dates = pd.Series(first_dates, index=first_opens.index).shift(-1)
    targets = {name: {} for name in PORTFOLIOS}

    for execution in first_dates[(first_dates >= start) & (first_dates <= end)]:
        open_position = opens.index.get_loc(execution)
        if open_position == 0:
            raise ValueError("history_start must provide prices before the first evaluation month")
        signal_date = opens.index[open_position - 1]
        completed = month_returns.loc[exit_dates.le(signal_date)]
        same_month = seasonal_sample(completed, execution.month, years)
        counts = same_month.count()
        eligible = closes.loc[signal_date, stocks].notna() & counts.ge(config["minimum_occurrences"])
        score = same_month.mean().where(eligible).dropna()
        prior_month = completed.index[-1].month if not completed.empty else execution.month
        placebo_month = (execution.month - 1 + config["placebo_month_shift"]) % 12 + 1
        selections = {
            STRATEGY: score.nlargest(config["holdings"], keep="first").index,
            BENCHMARK: score.index,
        }
        for name, month in ((PRIOR_MONTH, prior_month), (PLACEBO, placebo_month)):
            comparison = matched_scores(seasonal_sample(completed, month, years), counts)
            ranked = comparison.where(eligible).dropna()
            selections[name] = ranked.nlargest(config["holdings"], keep="first").index
            if len(ranked) < min_eligible:
                selections[name] = ranked.index[:0]

        for name, members in selections.items():
            target = pd.Series(0.0, index=opens.columns)
            if len(score) >= min_eligible and len(members):
                target.loc[members] = 1.0 / len(members)
            elif pd.notna(closes.at[signal_date, config["cash_ticker"]]):
                target.loc[config["cash_ticker"]] = 1.0
            targets[name][execution] = target.to_numpy()
    return targets


def simulate(growth, targets, cost_bps):
    cost = cost_bps / 10_000
    weights = np.zeros(len(growth.columns))
    records = []
    for date, relatives in zip(growth.index, growth.to_numpy()):
        after_fee = 1.0
        turnover = 0.0
        trades = 0
        if date in targets:
            target = targets[date]
            # Solve for post-cost NAV, in units of pre-trade NAV, so purchases
            # and sales fund both the target allocation and the exact fee.
            if cost:
                after_fee = brentq(lambda nav: nav + cost * np.abs(nav * target - weights).sum() - 1, 0, 1)
            trade_sizes = np.abs(after_fee * target - weights)
            turnover = trade_sizes.sum()
            trades = np.count_nonzero(trade_sizes > 1e-10)
            weights = target

        held = weights > 0
        missing = held & ~np.isfinite(relatives)
        if missing.any():
            symbols = growth.columns[missing]
            raise RuntimeError(f"Missing open-to-open prices from {date.date()} for {', '.join(symbols)}")
        market_growth = 1.0 + np.dot(weights[held], relatives[held] - 1.0)
        weights = weights.copy()
        weights[held] *= relatives[held] / market_growth
        records.append((after_fee * market_growth - 1.0, turnover, trades))
    return pd.DataFrame(records, index=growth.index, columns=["return", "turnover", "n_trades"])


def metrics(result, next_open, cash_returns):
    returns = result["return"]
    if returns.empty:
        raise ValueError("A configured evaluation period has no trading sessions")
    equity = (1 + returns.to_numpy()).cumprod()
    with_initial = np.r_[1.0, equity]
    years = (next_open.at[returns.index[-1]] - returns.index[0]).days / 365.2425
    std = returns.std()
    excess_returns = returns - cash_returns.reindex(returns.index)
    excess_std = excess_returns.std(skipna=False)
    stats = {
        "cagr": equity[-1] ** (1 / years) - 1,
        "ann_vol": std * np.sqrt(252),
        "sharpe": returns.mean() / std * np.sqrt(252) if std > 0 else np.nan,
        "sharpe_excess": excess_returns.mean() / excess_std * np.sqrt(252) if excess_std > 0 else np.nan,
        "max_drawdown": (with_initial / np.maximum.accumulate(with_initial) - 1).min(),
        "total_return": equity[-1] - 1,
        "turnover": result["turnover"].sum(),
    }
    return {**{key: float(value) if np.isfinite(value) else None for key, value in stats.items()},
            "n_trades": int(result["n_trades"].sum())}


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not (is_start <= is_end and is_end + pd.Timedelta(days=1) == oos_start <= oos_end):
        raise ValueError("IS and OOS must be nonempty, chronological, contiguous date ranges")
    if not 1 <= config["holdings"] <= config["minimum_eligible"] <= len(config["universe"]):
        raise ValueError("Holdings and minimum eligibility must fit the stock universe")
    if not 1 <= config["minimum_occurrences"] <= config["lookback_years"]:
        raise ValueError("Minimum occurrences must fit the annual lookback")
    if not 1 <= config["placebo_month_shift"] <= 11:
        raise ValueError("Use a nonzero calendar permutation of 1 to 11 months")
    fee = config["transaction_fee_bps"]
    slippage = config["slippage_bps"]
    base_cost = fee + slippage
    costs = sorted(set([base_cost, *config["cost_sensitivity_bps"]]))
    if any(not 0 <= cost < 10_000 for cost in [fee, slippage, *costs]):
        raise ValueError("Transaction costs must be in [0, 10000) basis points")

    opens, closes, fetched_at = load_prices(config, oos_end)
    start = is_start.to_period("M").start_time
    targets = monthly_targets(opens, closes, config, start, oos_end)
    growth = opens.shift(-1).div(opens).loc[start:oos_end]
    cash_returns = growth[config["cash_ticker"]] - 1
    next_open = pd.Series(opens.index, index=opens.index).shift(-1)
    periods = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    reports = {}
    print(f"{config['strategy']} | {config['market_label']} | base cost {base_cost:g} bps "
          f"({fee:g} fee + {slippage:g} slippage per dollar traded)")
    print("PARAM_COMBOS_TRIED: 1")
    print("COST_SCENARIOS_BPS: " + ", ".join(f"{cost:g}" for cost in costs))
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"Sharpe excess cash proxy: {config['cash_ticker']} adjusted open-to-open total return")
    print(f"{'Portfolio':<29} {'Cost bp':>7} {'Period':>6} {'CAGR':>9} {'Ann vol':>9} "
          f"{'Sharpe':>8} {'Ex Sharpe':>9} {'Max DD':>9} {'Total':>9} {'Trades':>7} {'Turnover':>9}")
    for name in PORTFOLIOS:
        for cost in costs:
            run = simulate(growth, targets[name], cost)
            report = {}
            for period, (first, last) in periods.items():
                stat = metrics(run.loc[first:last], next_open, cash_returns)
                report[period] = stat
                values = [stat[key] for key in ("cagr", "ann_vol", "sharpe", "sharpe_excess", "max_drawdown", "total_return")]
                formats = (".2%", ".2%", ".2f", ".2f", ".2%", ".2%")
                cells = [format(value, spec) if value is not None else "n/a" for value, spec in zip(values, formats)]
                cagr, vol, sharpe, excess, drawdown, total = cells
                print(f"{name:<29} {cost:>7g} {period.upper():>6} {cagr:>9} {vol:>9} "
                      f"{sharpe:>8} {excess:>9} {drawdown:>9} {total:>9} {stat['n_trades']:>7d} {stat['turnover']:>9.2f}")
            if cost == base_cost:
                reports[name] = report

    evaluation = growth.loc[is_start:oos_end]
    last_valuation = next_open.at[evaluation.index[-1]]
    benchmark = reports[BENCHMARK]
    result = {
        "strategy": config["strategy"],
        "data_range": f"{evaluation.index[0].date()} to {last_valuation.date()}",
        **reports[STRATEGY],
        "benchmark": {"name": BENCHMARK, **benchmark["full"], **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
