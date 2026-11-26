# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 315_leveraged_funds_robust_replication\n\ndata_start: '2011-01-01'\nin_sample:\n  start: '2012-01-01'\n  end: '2018-12-31'\nout_of_sample:\n  start: '2019-01-01'\n  end: '2025-12-31'\n\npairs:\n  - {fund: SSO, hedge: SPY, leverage: 2}\n  - {fund: UPRO, hedge: SPY, leverage: 3}\n  - {fund: QLD, hedge: QQQ, leverage: 2}\n  - {fund: TQQQ, hedge: QQQ, leverage: 3}\nbenchmark: SPY\nfunding_series: DFF\n\nsignal:\n  cumulative_sessions: 20\n  tracking_error_sessions: 60\n  entry_below: -2.0\n  hold_sessions: 5\n\ncosts:\n  synthetic_funding_spread_bps_annual: 100\n  transaction_bps_one_way: 5\n  short_borrow_bps_annual: 30\n  financing_day_count: 360\n\nannualization_sessions: 252\n"
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
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USFederalHolidayCalendar,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
BPS = 10_000


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


def save_cache(frame, path):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as file:
        temporary = Path(file.name)
        try:
            frame.to_csv(file, float_format="%.17g")
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def load_prices(tickers, start, end, sessions):
    path = CACHE / f"yahoo_adjusted_{start.date()}_{end.date()}_{'-'.join(tickers)}.csv"
    if path.exists():
        prices = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        with TemporaryDirectory(dir=CACHE, prefix="yahoo-") as metadata_cache:
            yf.set_tz_cache_location(metadata_cache)
            download = yf.download(
                tickers, start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
                auto_adjust=True, progress=False, threads=False, timeout=10,
            )
        if download is None or download.empty or "Close" not in download:
            raise RuntimeError("Yahoo Finance returned no daily adjusted closes.")
        prices = download["Close"]
    prices = prices.sort_index().reindex(columns=tickers)
    values = prices.to_numpy(dtype=float)
    if (not prices.index.equals(sessions) or not np.isfinite(values).all()
            or (values <= 0).any()):
        missing_dates = sessions.difference(prices.index)
        extra_dates = prices.index.difference(sessions)
        missing_tickers = prices.columns[prices.isna().any()].tolist()
        raise RuntimeError(
            f"Incomplete Yahoo adjusted-close panel. Missing sessions: {list(missing_dates[:5])}; "
            f"unexpected sessions: {list(extra_dates[:5])}; missing prices: {missing_tickers}."
        )
    if not path.exists():
        save_cache(prices, path)
    return prices


def load_funding(series, start, end, closes):
    start -= pd.Timedelta(days=14)
    path = CACHE / f"fred_{series}_{start.date()}_{end.date()}.csv"
    if path.exists():
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        response = requests.get(
            FRED_CSV, params={"id": series, "cosd": str(start.date()), "coed": str(end.date())},
            timeout=(5, 25),
        )
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text), index_col=0, parse_dates=True)
    if series not in frame or frame.empty:
        raise RuntimeError(f"FRED returned no observations for {series}.")
    fed_days = pd.date_range(
        start, end + pd.Timedelta(days=7),
        freq=pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar()),
    )
    # A business day's observation becomes available the following Fed business day.
    observations = pd.to_numeric(frame[series], errors="coerce").reindex(fed_days[:-1])
    published = observations.set_axis(fed_days[1:]).dropna()
    known_at_close = published.reindex(closes, method="ffill", tolerance=pd.Timedelta(days=7))
    if not np.isfinite(known_at_close).all():
        raise RuntimeError(f"FRED {series} is missing or stale for one or more trading sessions.")
    if not path.exists():
        save_cache(frame, path)
    return known_at_close / 100


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


def run_book(returns, day_fraction, targets, entries, costs):
    fee = costs["transaction_bps_one_way"] / BPS
    borrow = costs["short_borrow_bps_annual"] / BPS
    weights = np.zeros(returns.shape[1])
    output = []
    for daily_return, elapsed, target in zip(
        returns.to_numpy(), day_fraction.to_numpy(), targets.to_numpy(), strict=True
    ):
        before_trade = 1 + weights @ daily_return + np.minimum(weights, 0).sum() * borrow * elapsed
        if before_trade <= 0:
            raise RuntimeError("The portfolio exhausted its equity.")
        drifted = weights * (1 + daily_return) / before_trade
        if fee:
            # Solve for post-cost equity so the target gross stays at or below 100%.
            def cost_balance(equity):
                return equity + fee * np.abs(equity * target - drifted).sum() - 1

            after_trade = brentq(cost_balance, 0.0, 1.0)
        else:
            after_trade = 1.0
        turnover = np.abs(after_trade * target - drifted).sum()
        output.append((before_trade * after_trade - 1, turnover))
        weights = target

    result = pd.DataFrame(output, index=returns.index, columns=["return", "turnover"])
    result["entries"] = entries
    # Allocate the initial benchmark purchase and its cost to the first reported interval.
    initial, first = result.index[:2]
    result.at[first, "return"] = (1 + result.at[initial, "return"]) * (1 + result.at[first, "return"]) - 1
    result.loc[first, ["turnover", "entries"]] += result.loc[initial, ["turnover", "entries"]]
    return result.iloc[1:]


def summarize(period, sessions_per_year):
    if period.empty:
        raise ValueError("No observations in a configured reporting period.")
    returns = period["return"]
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax().clip(lower=1) - 1
    wealth = equity.iloc[-1]
    volatility = returns.std()
    values = {
        "cagr": wealth ** (sessions_per_year / len(period)) - 1,
        "ann_vol": volatility * np.sqrt(sessions_per_year),
        "sharpe": returns.mean() / volatility * np.sqrt(sessions_per_year) if volatility > 0 else None,
        "max_drawdown": drawdown.min(),
        "total_return": wealth - 1,
        "annualized_turnover": period["turnover"].mean() * sessions_per_year,
    }
    metrics = {key: float(value) if value is not None and np.isfinite(value) else None
               for key, value in values.items()}
    metrics["n_trades"] = int(period["entries"].sum())
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
    if not 0 <= costs["transaction_bps_one_way"] < BPS or costs["financing_day_count"] <= 0:
        raise ValueError("Invalid transaction fee or financing day count.")

    pairs, benchmark = config["pairs"], config["benchmark"]
    tickers = sorted({benchmark, *(pair[key] for pair in pairs for key in ("fund", "hedge"))})
    equity_day = pd.offsets.CustomBusinessDay(calendar=USEquityHolidays())
    # Extend the scheduled calendar through the week containing the end date.
    schedule = pd.date_range(start, end + pd.Timedelta(days=7), freq=equity_day)
    week = schedule.to_period("W-FRI")
    week_ends = schedule[~week.duplicated(keep="last")]
    sessions = schedule[schedule <= end]
    CACHE.mkdir(exist_ok=True)
    prices = load_prices(tickers, start, end, sessions)
    rates = load_funding(config["funding_series"], start, end, sessions)
    returns = prices.pct_change(fill_method=None)
    day_fraction = sessions.to_series().diff().dt.days / costs["financing_day_count"]
    funding_cost = (rates.shift(1) + costs["synthetic_funding_spread_bps_annual"] / BPS) * day_fraction

    first = sessions.searchsorted(is_start)
    if first == 0 or first == len(sessions):
        raise ValueError("The simulation needs a session before the first in-sample return.")
    dates = sessions[first - 1:]
    returns_for_book = returns.loc[dates].copy()
    days_for_book = day_fraction.loc[dates].copy()
    returns_for_book.iloc[0] = 0.0
    days_for_book.iloc[0] = 0.0

    pair_weights = pd.DataFrame(0.0, index=[pair["fund"] for pair in pairs], columns=tickers)
    for pair in pairs:
        fund, hedge, leverage = pair["fund"], pair["hedge"], pair["leverage"]
        pair_weights.at[fund, fund] = 1 / (1 + leverage)
        pair_weights.at[fund, hedge] = -leverage / (1 + leverage)

    signal = config["signal"]
    weekly_closes = sessions.intersection(week_ends)
    results = {}
    for label, daily_cost in (("Financed signal", funding_cost), ("Unfinanced signal", 0.0)):
        scores = tracking_scores(returns, pairs, daily_cost, signal)
        targets, entries = trade_targets(scores, weekly_closes, dates, pair_weights, signal)
        results[label] = run_book(returns_for_book, days_for_book, targets, entries, costs)

    entries = pd.Series(0, index=dates, dtype=int)
    entries.iloc[0] = 1
    always_held = pd.DataFrame([pair_weights.mean().to_numpy()] * len(dates), index=dates, columns=tickers)
    buy_hold = pd.DataFrame(0.0, index=dates, columns=tickers)
    buy_hold[benchmark] = 1.0
    benchmark_label = f"{benchmark} buy and hold"
    results["Always-held hedged"] = run_book(returns_for_book, days_for_book, always_held, entries, costs)
    results[benchmark_label] = run_book(returns_for_book, days_for_book, buy_hold, entries, costs)

    table, summary = [], {}
    for label, daily in results.items():
        periods = {"is": daily.loc[is_start:is_end], "oos": daily.loc[oos_start:end], "full": daily}
        summary[label] = {key: summarize(period, config["annualization_sessions"])
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
    display = pd.DataFrame(table)
    for key in ("cagr", "ann_vol", "max_drawdown", "total_return", "annualized_turnover"):
        display[key] = display[key].map(lambda value: "—" if pd.isna(value) else f"{value:.2%}")
    print(display.to_string(index=False, float_format=lambda value: f"{value:.3f}", na_rep="—"))
    print("RESULTS_JSON: " + json.dumps(results_json, allow_nan=False))


if __name__ == "__main__":
    main()
