# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 315_leveraged_funds_robust_replication\n\ndata_start: '2011-01-01'\nin_sample:\n  start: '2012-01-01'\n  end: '2018-12-31'\nout_of_sample:\n  start: '2019-01-01'\n  end: '2025-12-31'\n\npairs:\n  - {fund: SSO, hedge: SPY, leverage: 2}\n  - {fund: UPRO, hedge: SPY, leverage: 3}\n  - {fund: QLD, hedge: QQQ, leverage: 2}\n  - {fund: TQQQ, hedge: QQQ, leverage: 3}\nbenchmark: SPY\ncash_proxy: BIL\nfunding:\n  target_range: [DFEDTARL, DFEDTARU]\n  availability_lag_days: 1\n\nsignal:\n  cumulative_sessions: 20\n  tracking_error_sessions: 60\n  entry_below: -2.0\n  hold_sessions: 5\n\ncosts:\n  synthetic_funding_spread_bps_annual: 100\n  commission_bps_one_way: 1\n  slippage_bps_one_way: 4\n  short_borrow_bps_annual: 30\n  financing_day_count: 360\n  cash_buffer_fraction: 0.02  # Reserve at sizing for close moves and fees.\n\nannualization_sessions: 252\n"
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
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)


ROOT = Path(__file__).resolve().parent
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
BPS = 10_000
PRICE_ADJUSTMENT = "Yahoo Finance dividend- and split-adjusted daily close; fractional adjusted units"


class USEquityHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
        Holiday("Carter mourning day", year=2025, month=1, day=9),
        Holiday("Bush mourning day", year=2018, month=12, day=5),
        Holiday("Hurricane Sandy day one", year=2012, month=10, day=29),
        Holiday("Hurricane Sandy day two", year=2012, month=10, day=30),
    ]


def load_prices(tickers, start, end, sessions):
    with TemporaryDirectory(dir=ROOT, prefix="yahoo-") as metadata_cache:
        yf.set_tz_cache_location(metadata_cache)
        for _ in range(2):
            download = yf.download(
                tickers, start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
                auto_adjust=True, progress=False, threads=False, timeout=10, multi_level_index=True,
            )
            if download is None or download.empty or "Close" not in download:
                continue
            prices = download["Close"].sort_index().reindex(columns=tickers)
            values = prices.to_numpy(dtype=float)
            if prices.index.equals(sessions) and np.isfinite(values).all() and (values > 0).all():
                return prices, datetime.now(timezone.utc).isoformat(timespec="seconds")
    raise RuntimeError(f"Yahoo Finance did not return complete daily adjusted closes for {tickers}.")


def load_policy_rate(funding, sessions):
    lower, upper = funding["target_range"]
    # Use the announced target range, allowing a full calendar-date lag on its effective dates.
    observation_dates = sessions - pd.Timedelta(days=funding["availability_lag_days"])
    response = requests.get(
        FRED_CSV, params={"id": f"{lower},{upper}", "cosd": str(observation_dates[0].date()),
                          "coed": str(observation_dates[-1].date())}, timeout=(5, 25),
    )
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text), index_col=0, parse_dates=True)
    limits = frame.reindex(index=observation_dates, columns=[lower, upper])
    if not np.isfinite(limits.to_numpy(dtype=float)).all() or (limits[lower] > limits[upper]).any():
        raise RuntimeError("FRED is missing a valid federal funds target range for a required date.")
    rates = limits.mean(axis=1).set_axis(sessions) / 100
    return rates, datetime.now(timezone.utc).isoformat(timespec="seconds")


def compounded_return(returns, window):
    return np.expm1(np.log1p(returns).rolling(window).sum())


def tracking_scores(returns, pairs, funding_cost, signal):
    scores = {}
    for pair in pairs:
        fund, hedge, leverage = pair["fund"], pair["hedge"], pair["leverage"]
        actual = returns[fund]
        replica = leverage * returns[hedge] - (leverage - 1) * funding_cost
        if (replica <= -1).any():
            raise RuntimeError(f"The synthetic {fund} replica lost all its equity within one session.")
        window = signal["cumulative_sessions"]
        gap = compounded_return(actual, window) - compounded_return(replica, window)
        error = (actual - replica).rolling(signal["tracking_error_sessions"]).std()
        scores[fund] = gap.div(error.where(error > 0))
    return pd.DataFrame(scores)


def trade_targets(scores, weekly_closes, dates, pair_weights, signal):
    qualifiers = scores.loc[weekly_closes].where(lambda score: score < signal["entry_below"])
    winners = qualifiers.dropna(how="all").idxmin(axis=1)
    executions = winners.reindex(scores.index).shift(1).reindex(dates)
    targets = pd.DataFrame(0.0, index=dates, columns=pair_weights.columns)
    entries = pd.Series(0, index=dates, dtype=int)
    next_available = 1  # The initial close is only used to initialize the simulation.
    for idx, fund in enumerate(executions):
        if idx >= next_available and pd.notna(fund):
            next_available = idx + signal["hold_sessions"]
            targets.iloc[idx:next_available] = pair_weights.loc[fund].to_numpy()
            entries.iloc[idx] = 1
    return targets, entries


def settle_fills(quantities, cash, close, friction):
    trade_values = quantities * close
    traded = np.abs(trade_values).sum()
    return cash - (trade_values.sum() + friction * traded), traded


def run_execution(prices, day_fraction, targets, entries, costs, *, rebalance=True):
    friction = (costs["commission_bps_one_way"] + costs["slippage_bps_one_way"]) / BPS
    borrow = costs["short_borrow_bps_annual"] / BPS
    invested_fraction = 1 - costs["cash_buffer_fraction"]
    positions = np.zeros(prices.shape[1])
    cash = equity = 1.0
    output = []
    closes = prices.to_numpy()
    for idx, (previous, close, elapsed, target) in enumerate(zip(
        closes[:-1], closes[1:], day_fraction.to_numpy(), targets.to_numpy(), strict=True
    )):
        cash += (np.minimum(positions, 0) @ previous) * borrow * elapsed
        traded, trades = 0.0, 0
        if rebalance or idx == 0:
            # Prior-close prices and equity lock every leg before the closing fill.
            next_positions = (invested_fraction * equity) * target / previous
            quantities = next_positions - positions
            cash, traded = settle_fills(quantities, cash, close, friction)
            trades = np.count_nonzero(quantities)
            positions = next_positions
        # Keep cash equal to the full marked short liability as collateral.
        free_cash = cash + np.minimum(positions, 0) @ close
        if free_cash < 0:
            raise RuntimeError(f"Insufficient cash after fees and short collateral on "
                               f"{targets.index[idx].date()}: {free_cash / equity:.4%} of prior equity.")
        ending_equity = cash + positions @ close
        if not np.isfinite(ending_equity) or ending_equity <= 0:
            raise RuntimeError(f"Nonpositive or invalid equity on {targets.index[idx].date()}.")
        output.append((ending_equity / equity - 1, traded / equity, trades, free_cash / equity,
                       np.count_nonzero(positions)))
        equity = ending_equity

    daily_results = pd.DataFrame(output, index=targets.index,
                                 columns=["return", "turnover", "n_trades", "free_cash_fraction", "open_legs"])
    daily_results["entries"] = entries
    # Allocate the initial benchmark purchase and its cost to the first reported interval.
    initial, first = daily_results.index[:2]
    daily_results.at[first, "return"] = (
        (1 + daily_results.at[initial, "return"]) * (1 + daily_results.at[first, "return"]) - 1
    )
    for column in ("turnover", "n_trades", "entries"):
        daily_results.at[first, column] += daily_results.at[initial, column]
    daily_results.at[first, "free_cash_fraction"] = min(
        daily_results.at[initial, "free_cash_fraction"], daily_results.at[first, "free_cash_fraction"]
    )
    return daily_results.iloc[1:]


def summarize(period, cash_returns, sessions_per_year):
    if period.empty:
        raise ValueError("No observations in a configured reporting period.")
    returns = period["return"]
    excess_returns = returns - cash_returns.loc[returns.index]
    annualization = np.sqrt(sessions_per_year)
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax().clip(lower=1) - 1
    wealth = equity.iloc[-1]
    volatility = returns.std()
    excess_volatility = excess_returns.std()
    values = {
        "cagr": wealth ** (sessions_per_year / len(period)) - 1,
        "ann_vol": volatility * annualization,
        "sharpe": returns.mean() / volatility * annualization if volatility > 0 else None,
        "sharpe_excess": (excess_returns.mean() / excess_volatility * annualization
                          if excess_volatility > 0 else None),
        "max_drawdown": drawdown.min(),
        "total_return": wealth - 1,
        "turnover": period["turnover"].mean() * sessions_per_year,
    }
    metrics = {key: float(value) if value is not None and np.isfinite(value) else None
               for key, value in values.items()}
    metrics["n_trades"] = int(period["n_trades"].sum())
    return metrics


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    start = pd.Timestamp(config["data_start"])
    in_sample, out_of_sample = config["in_sample"], config["out_of_sample"]
    is_start, is_end = (pd.Timestamp(in_sample[key]) for key in ("start", "end"))
    oos_start, end = (pd.Timestamp(out_of_sample[key]) for key in ("start", "end"))
    if not start < is_start <= is_end < oos_start <= end:
        raise ValueError("Configure warm-up, in-sample and out-of-sample dates in chronological order.")
    costs = config["costs"]
    commission, slippage = costs["commission_bps_one_way"], costs["slippage_bps_one_way"]
    if (min(commission, slippage, costs["short_borrow_bps_annual"]) < 0 or commission + slippage >= BPS
            or costs["financing_day_count"] <= 0 or not 0 < costs["cash_buffer_fraction"] < 1):
        raise ValueError("Invalid transaction costs, financing day count, or cash buffer.")
    funding = config["funding"]
    if funding["availability_lag_days"] < 1:
        raise ValueError("The policy target must be lagged by at least one calendar day.")

    pairs, benchmark = config["pairs"], config["benchmark"]
    cash_proxy = config["cash_proxy"]
    tickers = sorted({benchmark, *(pair[key] for pair in pairs for key in ("fund", "hedge"))})
    equity_day = pd.offsets.CustomBusinessDay(calendar=USEquityHolidays())
    # Extend the scheduled calendar through the week containing the end date.
    schedule = pd.date_range(start, end + pd.Timedelta(days=7), freq=equity_day)
    week = schedule.to_period("W-FRI")
    week_ends = schedule[~week.duplicated(keep="last")]
    sessions = schedule[schedule <= end]
    prices, yahoo_fetched_at = load_prices(sorted(set(tickers) | {cash_proxy}), start, end, sessions)
    rates, fred_fetched_at = load_policy_rate(funding, sessions)
    all_returns = prices.pct_change(fill_method=None)
    returns, cash_returns = all_returns[tickers], all_returns[cash_proxy]
    day_fraction = sessions.to_series().diff().dt.days / costs["financing_day_count"]
    funding_cost = (rates.shift(1) + costs["synthetic_funding_spread_bps_annual"] / BPS) * day_fraction

    first = sessions.searchsorted(is_start)
    if first < 2 or first == len(sessions):
        raise ValueError("The simulation needs two sessions before the first in-sample return.")
    dates = sessions[first - 1:]
    execution_prices = prices.loc[sessions[first - 2]:, tickers]
    execution_days = day_fraction.loc[dates]

    pair_weights = pd.DataFrame(0.0, index=[pair["fund"] for pair in pairs], columns=tickers)
    for pair in pairs:
        fund, hedge, leverage = pair["fund"], pair["hedge"], pair["leverage"]
        pair_weights.at[fund, fund] = 1 / (1 + leverage)
        pair_weights.at[fund, hedge] = -leverage / (1 + leverage)

    signal = config["signal"]
    weekly_closes = sessions.intersection(week_ends)
    results = {}
    signal_variants = {"Financed signal": funding_cost, "Unfinanced signal": 0.0}
    for label, daily_cost in signal_variants.items():
        scores = tracking_scores(returns, pairs, daily_cost, signal)
        targets, entries = trade_targets(scores, weekly_closes, dates, pair_weights, signal)
        results[label] = run_execution(execution_prices, execution_days, targets, entries, costs)

    entries = pd.Series(0, index=dates, dtype=int)
    entries.iloc[0] = 1
    always_held = pd.DataFrame([pair_weights.mean().to_numpy()] * len(dates), index=dates, columns=tickers)
    buy_hold = pd.DataFrame(0.0, index=dates, columns=tickers)
    buy_hold[benchmark] = 1.0
    benchmark_label = f"{benchmark} buy and hold"
    results["Always-held hedged"] = run_execution(execution_prices, execution_days, always_held, entries, costs)
    results[benchmark_label] = run_execution(
        execution_prices, execution_days, buy_hold, entries, costs, rebalance=False
    )

    table, summary = [], {}
    for label, daily in results.items():
        periods = {"is": daily.loc[is_start:is_end], "oos": daily.loc[oos_start:end], "full": daily}
        summary[label] = {key: summarize(period, cash_returns, config["annualization_sessions"])
                          for key, period in periods.items()}
        for key, metrics in summary[label].items():
            table.append({"Portfolio": label, "Period": key.upper(), **metrics})
    data_range = f"{dates[1].date()} to {dates[-1].date()}"
    results_json = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **summary["Financed signal"],
        "benchmark": {
            "name": benchmark_label,
            **summary[benchmark_label],
            "unfinanced_signal": summary["Unfinanced signal"],
            "always_held_hedged": summary["Always-held hedged"],
        },
    }
    print(f"{config['strategy']} | {data_range}")
    print(f"IS {is_start.date()} to {is_end.date()} | OOS {oos_start.date()} to {end.date()}")
    print(f"PARAM_COMBOS_TRIED: {len(signal_variants)} (one main specification, one zero-financing control)")
    print(f"DATA_FETCHED_AT: Yahoo Finance {yahoo_fetched_at}; FRED {fred_fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"CASH_PROXY: {cash_proxy} adjusted close-to-close return for excess Sharpe")
    display = pd.DataFrame(table)
    for key in ("cagr", "ann_vol", "max_drawdown", "total_return", "turnover"):
        display[key] = display[key].map(lambda value: "—" if pd.isna(value) else f"{value:.2%}")
    print(display.to_string(index=False, float_format=lambda value: f"{value:.3f}", na_rep="—"))
    main_results = results["Financed signal"]
    print(f"FINANCED_ENTRY_OR_RENEWAL: IS {main_results.loc[is_start:is_end, 'entries'].sum()}, "
          f"OOS {main_results.loc[oos_start:end, 'entries'].sum()}, FULL {main_results['entries'].sum()}")
    for label, daily in results.items():
        print(f"EXECUTION: {label} | min_free_cash_pct={daily['free_cash_fraction'].min():.4%} "
              f"| end_open_legs={daily['open_legs'].iloc[-1]}")
    print("RESULTS_JSON: " + json.dumps(results_json, allow_nan=False))


if __name__ == "__main__":
    main()
