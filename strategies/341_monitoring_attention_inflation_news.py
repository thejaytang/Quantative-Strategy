# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 341_monitoring_attention_inflation_news\nparameter_combos_tried: 1\nuniverse:\n  inflation_linked: TIP\n  nominal: IEF\n  cash_proxy: BIL\nseries:\n  attention: EMVMACROINFLATION\n  breakeven: T10YIE\ndata:\n  archived_macro_fetch_date: 2026-11-25\nin_sample:\n  start: 2019-11-01\n  end: 2022-12-31\nout_of_sample:\n  start: 2023-01-01\n  end: 2025-12-31\nsignal:\n  attention_lag_months: 2\n  attention_window_months: 60\n  attention_quantile: 0.80\n  breakeven_lookback_sessions: 20\n  breakeven_lag_sessions: 1\n  breakeven_max_quote_age_days: 7\nweights:\n  neutral_tip: 0.50\n  tilted_tip: 0.75\nexecution:\n  commission_bps: 1.0\n  slippage_bps: 1.0\n  # Reserve from prior-close sizing value before locking quantities for the next close.\n  cash_buffer_bps: 50.0\n  distribution_delay_sessions: 10\n'
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
from datetime import datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zoneinfo import ZoneInfo
import json
import math
import time

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
DAY = pd.Timedelta(days=1)
SESSIONS_PER_YEAR = 252
FETCH_STAMP = "# DATA_FETCHED_AT: "
PRICE_ADJUSTMENT = (
    "TIP/IEF Yahoo Close (split-adjusted, no distribution adjustment) plus cash distributions | "
    "BIL Yahoo Adj Close (total-return proxy)"
)


def save_prices(frame, path, fetched_at):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        with temp_path.open("w", encoding="utf-8") as output:
            output.write(FETCH_STAMP + fetched_at + "\n")
            frame.to_csv(output)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def read_csv(source):
    return pd.read_csv(source, index_col=0, parse_dates=True, na_values=["."])


def nyse_sessions(start, end):
    """Cash-equity session calendar for the 2019–2025 data, including early closes."""
    if start < pd.Timestamp("2019-01-01") or end > pd.Timestamp("2025-12-31"):
        raise ValueError("The NYSE Arca closure calendar covers 2019–2025.")
    holidays = AbstractHolidayCalendar(rules=[
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
        Holiday("Carter Day of Mourning", year=2025, month=1, day=9),
    ]).holidays(start, end)
    return pd.date_range(start, end, freq="B").difference(holidays)


def check_market_data(frame, sessions, price_columns, distribution_columns):
    if not frame.index.equals(sessions):
        missing = sessions.difference(frame.index).strftime("%Y-%m-%d").tolist()
        extra = frame.index.difference(sessions).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Yahoo dates differ from NYSE Arca sessions: missing={missing}, extra={extra}.")
    if (not np.isfinite(frame.to_numpy()).all() or (frame[price_columns] <= 0).any().any()
            or (frame[distribution_columns] < 0).any().any()):
        raise ValueError("Yahoo returned invalid prices or distribution amounts.")


def load_prices(tickers, cash_ticker, start, end, sessions):
    symbols = [*tickers, cash_ticker]
    cache = CACHE / f"yahoo_closes_distributions_{'-'.join(symbols)}_{start.date()}_{end.date()}.csv"
    close_columns = [f"{ticker}_close" for ticker in tickers]
    distribution_columns = [f"{ticker}_distribution" for ticker in tickers]
    cash_column = f"{cash_ticker}_adjusted_close"
    price_columns = [*close_columns, cash_column]
    if not cache.exists():
        with TemporaryDirectory(dir=CACHE) as scratch:
            yf.set_tz_cache_location(scratch)
            for attempt in range(3):
                raw = yf.download(
                    symbols, start=str(start.date()), end=str((end + DAY).date()),
                    auto_adjust=False, actions=True, threads=False, progress=False, timeout=15,
                )
                try:
                    closes = raw["Close"][tickers].set_axis(close_columns, axis=1)
                    distributions = raw["Dividends"][tickers] + raw["Capital Gains"][tickers]
                    distributions = distributions.set_axis(distribution_columns, axis=1)
                    cash_prices = raw["Adj Close"][cash_ticker].rename(cash_column)
                    frame = pd.concat([closes, distributions, cash_prices], axis=1)
                    frame.index = frame.index.tz_localize(None)
                    check_market_data(frame, sessions, price_columns, distribution_columns)
                    if not raw["Stock Splits"][tickers].eq(0).all().all():
                        raise ValueError("The fixed-share simulation requires a sample without ETF splits.")
                    break
                except (KeyError, ValueError) as error:
                    if attempt == 2:
                        raise RuntimeError(f"Incomplete Yahoo Finance data for {symbols}: {error}") from error
                    time.sleep(2 ** attempt)
        fetched_at = datetime.now(ZoneInfo("Europe/Oslo")).isoformat(timespec="seconds")
        save_prices(frame, cache, fetched_at)

    snapshot = cache.read_bytes()
    with BytesIO(snapshot) as source:
        fetched_line = source.readline().decode("utf-8").strip()
        if not fetched_line.startswith(FETCH_STAMP):
            raise ValueError(f"Missing fetch date in {cache.name}.")
        fetched_at = fetched_line.removeprefix(FETCH_STAMP)
        frame = read_csv(source)
    check_market_data(frame, sessions, price_columns, distribution_columns)
    provenance = {"path": cache.relative_to(ROOT).as_posix(), "sha256": sha256(snapshot).hexdigest()}
    return frame[close_columns], frame[distribution_columns], frame[cash_column], fetched_at, provenance


def load_vintages(series, vintage_dates, last_dates, first_dates=None):
    """Read the supplied public ALFRED graph CSV snapshots for each decision."""
    frames = []
    batch_size = 12
    for offset in range(0, len(vintage_dates), batch_size):
        batch = slice(offset, offset + batch_size)
        dates = vintage_dates[batch]
        params = {
            "id": ",".join([series] * len(dates)),
            "vintage_date": ",".join(dates.strftime("%Y-%m-%d")),
            "coed": ",".join(last_dates[batch].strftime("%Y-%m-%d")),
        }
        if first_dates is not None:
            params["cosd"] = ",".join(first_dates[batch].strftime("%Y-%m-%d"))
        digest = sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        path = CACHE / f"alfred_{series}_{digest}.csv"
        columns = [f"{series}_{date:%Y%m%d}" for date in dates]

        if not path.is_file():
            raise FileNotFoundError(f"Required archived public ALFRED snapshot is missing: {path.name}.")
        frame = read_csv(path)
        if (frame.empty or not isinstance(frame.index, pd.DatetimeIndex)
                or not frame.index.is_unique or list(frame.columns) != columns):
            raise ValueError(f"Unexpected ALFRED data for {series}, {dates[0]:%Y-%m-%d}.")
        frame = frame.apply(pd.to_numeric, errors="raise").sort_index()
        if np.isinf(frame.to_numpy()).any():
            raise ValueError(f"Non-finite value in ALFRED data for {series}, {dates[0]:%Y-%m-%d}.")
        frame.columns = dates
        frames.append(frame)
    return pd.concat(frames, axis=1).sort_index()


def make_targets(sessions, evaluation_start, config):
    """Create closing targets indexed by execution date, one session after the signal."""
    months = sessions.to_period("M")
    month_end_positions = np.flatnonzero(months[:-1] != months[1:])
    execution_dates = sessions[month_end_positions + 1]
    first = max(0, execution_dates.searchsorted(evaluation_start, side="right") - 1)
    month_end_positions = month_end_positions[first:]
    execution_dates = execution_dates[first:]
    decisions = sessions[month_end_positions]

    rules = config["signal"]
    vintage_dates = decisions - DAY
    attention_cutoffs = (decisions.to_period("M") - rules["attention_lag_months"]).to_timestamp()
    recent_positions = month_end_positions - rules["breakeven_lag_sessions"]
    prior_positions = recent_positions - rules["breakeven_lookback_sessions"]
    if not len(prior_positions) or prior_positions[0] < 0:
        raise ValueError("Insufficient ETF sessions to construct the monthly signals.")
    quote_dates = sessions[recent_positions]
    prior_dates = sessions[prior_positions]
    quote_age = pd.Timedelta(days=rules["breakeven_max_quote_age_days"])

    attention = load_vintages(config["series"]["attention"], vintage_dates, attention_cutoffs)
    breakeven = load_vintages(
        config["series"]["breakeven"], vintage_dates, quote_dates, prior_dates - quote_age,
    )

    neutral = config["weights"]["neutral_tip"]
    tilt = config["weights"]["tilted_tip"]
    targets = pd.DataFrame({
        "strategy": neutral, "static_50_50": 0.50, "breakevens_only": neutral,
    }, index=execution_dates)
    for execution, vintage, cutoff, recent_date, prior_date in zip(
        execution_dates, vintage_dates, attention_cutoffs, quote_dates, prior_dates,
    ):
        coverage = attention[vintage].loc[:cutoff].dropna()
        attention_high = False
        if not coverage.empty:
            window = pd.date_range(end=coverage.index[-1], periods=rules["attention_window_months"], freq="MS")
            history = coverage.reindex(window)
            attention_high = history.notna().all() and history.iloc[-1] > history.quantile(rules["attention_quantile"])

        quotes = breakeven[vintage].dropna().reindex(
            [prior_date, recent_date], method="ffill", tolerance=quote_age,
        )
        if quotes.iloc[1] > quotes.iloc[0]:
            targets.loc[execution, "breakevens_only"] = tilt
            if attention_high:
                targets.loc[execution, "strategy"] = tilt

    # If the sample opens mid-month, enter with the last available target.
    targets.index = targets.index.where(targets.index >= evaluation_start, evaluation_start)
    return targets


def fill_at_close(cash, requested_shares, close, commission_rate, slippage_rate, session):
    """Settle the predetermined quantities or stop before applying the rebalance."""
    fill_prices = close * (1.0 + np.sign(requested_shares) * slippage_rate)
    trade_values = requested_shares * fill_prices
    traded_value = np.abs(trade_values).sum()
    remaining_cash = cash - trade_values.sum() - commission_rate * traded_value
    if remaining_cash < 0:
        raise ValueError(f"Insufficient cash for fixed quantities at the {session:%Y-%m-%d} close "
                         f"(cash shortfall {abs(remaining_cash):.6g} in initial-capital units).")
    return remaining_cash, traded_value


def simulate(closes, distributions, tip_targets, commission_rate, slippage_rate, cash_buffer, distribution_delay):
    """Use the first row for sizing. All remaining rows are evaluated sessions."""
    prices = closes.to_numpy()
    per_share_distributions = distributions.to_numpy()
    targets = tip_targets.reindex(closes.index).to_numpy()
    positions = np.zeros(prices.shape[1])
    cash, previous_value = 1.0, 1.0
    distribution_schedule = np.zeros(len(closes) + distribution_delay)
    records = []

    for i in range(1, len(closes)):
        target = targets[i]
        rebalance = pd.notna(target)
        if rebalance:
            prior_close = prices[i - 1]
            sizing_value = cash + positions @ prior_close
            weights = np.array([target, 1.0 - target])
            target_shares = weights * sizing_value * (1.0 - cash_buffer) / prior_close
            requested_shares = target_shares - positions

        # Only shares carried into the ex-date earn the distribution.
        distribution_schedule[i + distribution_delay] = positions @ per_share_distributions[i]
        cash += distribution_schedule[i]
        pending_distributions = distribution_schedule[i + 1:i + distribution_delay + 1].sum()
        close = prices[i]
        pre_fill_value = cash + positions @ close + pending_distributions
        turnover, trades = 0.0, 0
        weight_gap = np.nan
        if rebalance:
            cash, traded_value = fill_at_close(
                cash, requested_shares, close, commission_rate, slippage_rate, closes.index[i],
            )
            positions = target_shares
            turnover = traded_value / pre_fill_value
            trades = np.count_nonzero(requested_shares)

        value = cash + positions @ close + pending_distributions
        if rebalance:
            weight_gap = abs(positions[0] * close[0] / value - target)
        records.append((value / previous_value - 1.0, turnover, trades, cash / value, weight_gap))
        previous_value = value

    return pd.DataFrame(records, index=closes.index[1:], columns=[
        "return", "turnover", "n_trades", "cash_weight", "tip_weight_gap",
    ])


def finite(value):
    return float(value) if math.isfinite(value) else None


def sharpe_ratio(daily):
    daily_vol = daily.std(ddof=1)
    return finite(daily.mean() / daily_vol * np.sqrt(SESSIONS_PER_YEAR)) if daily_vol > 0 else None


def metrics(daily_results, cash_returns, start, end):
    segment = daily_results.loc[start:end]
    daily = segment["return"]
    if daily.empty:
        raise ValueError(f"No return observations from {start.date()} to {end.date()}.")
    wealth = (1.0 + daily).cumprod()
    drawdown = wealth / wealth.cummax().clip(lower=1.0) - 1.0
    vol = daily.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": finite(wealth.iloc[-1] ** (SESSIONS_PER_YEAR / len(daily)) - 1.0),
        "ann_vol": finite(vol),
        "sharpe": sharpe_ratio(daily),
        "sharpe_excess": sharpe_ratio(daily - cash_returns.loc[daily.index]),
        "max_drawdown": finite(drawdown.min()),
        "total_return": finite(wealth.iloc[-1] - 1.0),
        "n_trades": int(segment["n_trades"].sum()),
        "turnover": finite(segment["turnover"].sum()),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    periods = {key: tuple(pd.Timestamp(config[field][bound]) for bound in ("start", "end"))
               for key, field in (("is", "in_sample"), ("oos", "out_of_sample"))}
    start, end = periods["is"][0], periods["oos"][1]
    if not start <= periods["is"][1] < periods["oos"][0] <= end:
        raise ValueError("IS and OOS must be in chronological order and must not overlap.")
    periods["full"] = start, end

    execution = config["execution"]
    commission_rate, slippage_rate, cash_buffer = (
        float(execution[key]) / 10_000 for key in ("commission_bps", "slippage_bps", "cash_buffer_bps")
    )
    distribution_delay = execution["distribution_delay_sessions"]
    neutral, tilted = (config["weights"][key] for key in ("neutral_tip", "tilted_tip"))
    if (not all(0 <= rate < 1 for rate in (commission_rate, slippage_rate, cash_buffer))
            or not 0 <= neutral < tilted <= 1 or not isinstance(distribution_delay, int) or distribution_delay < 1):
        raise ValueError("Invalid execution settings or TIP weights.")
    tickers = [config["universe"][role] for role in ("inflation_linked", "nominal")]
    cash_ticker = config["universe"]["cash_proxy"]
    lookback = config["signal"]["breakeven_lookback_sessions"] + config["signal"]["breakeven_lag_sessions"]
    warmup = (start.to_period("M") - (lookback // 15 + 3)).start_time
    CACHE.mkdir(exist_ok=True)
    sessions = nyse_sessions(warmup, end)
    closes, distributions, cash_prices, fetched_at, market_snapshot = load_prices(
        tickers, cash_ticker, warmup, end, sessions,
    )
    start_position = sessions.searchsorted(start)
    evaluation = sessions[start_position:]
    simulation_start = sessions[start_position - 1]
    targets = make_targets(sessions, evaluation[0], config)
    cash_returns = cash_prices.pct_change().loc[evaluation]

    reports = {}
    diagnostics = {}
    for model in targets:
        daily_results = simulate(
            closes.loc[simulation_start:], distributions.loc[simulation_start:], targets[model],
            commission_rate, slippage_rate, cash_buffer, distribution_delay,
        )
        reports[model] = {period: metrics(daily_results, cash_returns, *bounds) for period, bounds in periods.items()}
        diagnostics[model] = {
            "min_cash_weight": finite(daily_results["cash_weight"].min()),
            "max_tip_weight_gap_at_rebalance": finite(daily_results["tip_weight_gap"].max()),
        }

    data_range = f"{evaluation[0]:%Y-%m-%d} to {evaluation[-1]:%Y-%m-%d}"
    print(f"{config['strategy']} | {data_range} | {execution['commission_bps']:g} bp commission"
          f" + {execution['slippage_bps']:g} bp slippage per traded side | Cash proxy: {cash_ticker}")
    print(f"PARAM_COMBOS_TRIED: {config['parameter_combos_tried']}")
    print(f"DATA_FETCHED_AT: Yahoo Finance {fetched_at} | ALFRED archives "
          f"{config['data']['archived_macro_fetch_date']} (Europe/Oslo)")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"RUN_CWD: {Path.cwd()}")
    print(f"MARKET_DATA_SNAPSHOT: {json.dumps(market_snapshot, allow_nan=False)}")
    print(f"FILL_DIAGNOSTICS: {json.dumps(diagnostics, allow_nan=False)}")
    table = pd.DataFrame.from_dict(
        {(model, period.upper()): values for model, report in reports.items() for period, values in report.items()},
        orient="index",
    )
    table.index.names = ["portfolio", "period"]
    print(table.to_string(na_rep="n/a", formatters={
        "cagr": "{:.2%}".format, "ann_vol": "{:.2%}".format,
        "sharpe": "{:.2f}".format, "sharpe_excess": "{:.2f}".format,
        "max_drawdown": "{:.2%}".format, "total_return": "{:.2%}".format, "turnover": "{:.2f}x".format,
    }))

    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **reports["strategy"],
        "benchmark": {model: reports[model] for model in ("static_50_50", "breakevens_only")},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
