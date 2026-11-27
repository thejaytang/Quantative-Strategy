# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 309_intangible_value_sixth_factor\ncategory: factor\n\nuniverse:\n  market: SPY\n  intangible_value: ITAN\n  conventional_value: IWD\n  growth: IWF\n\ndata:\n  start: 2021-06-28\n  first_common_close: 2021-06-29  # Required first common US equity session for this sample.\n  end: 2026-11-20  # Inclusive. Last complete trading week before the job date.\n  cash_proxy: BIL  # Adjusted close returns, for excess Sharpe only.\n  legacy_snapshot:\n    sha256: 471b28ab62325a98dc6245382951d9cdb3069d03245ef5d68efc4fc425951fb0\n    file_timestamp: '2026-11-25T13:43:38+01:00'  # Original cached file timestamp in Europe/Oslo.\nin_sample:\n  start: 2021-06-28\n  end: 2023-12-31\nout_of_sample:\n  start: 2024-01-01\n  end: 2026-11-20\n\nsleeve_weight: 0.20  # ITAN/IWD fraction of the long-only invested amount.\nhedge_lookback_sessions: 126\ntransaction_cost_bps: 5  # One way, per dollar bought or sold.\nslippage_bps: 5  # One-way adverse fill-price adjustment.\ncash_buffer: 0.05  # Reserve on prior-close sizing for the long-only portfolios.\nhedge_cash_buffer: 0.15  # Collateral headroom on the gross hedge exposure.\nshort_borrow_bps_per_year: 100  # Flat modeling assumption, per dollar short.\ntrading_days_per_year: 252\n"
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
"""Backtest an intangible-value ETF sleeve and a collateralized ETF spread."""

from datetime import datetime
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
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
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
OSLO = ZoneInfo("Europe/Oslo")
FETCH_HEADER = "# fetched_at="
PARAM_COMBOS_TRIED = 1


class NyseHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
        Holiday("Carter national day of mourning", year=2025, month=1, day=9),
    ]


def trading_sessions(start, end):
    if not 2021 <= start.year <= end.year <= 2026:
        raise ValueError("NYSE closure calendar covers 2021 through 2026")
    return pd.date_range(start, end, freq=pd.offsets.CustomBusinessDay(calendar=NyseHolidays()))


def validate_prices(prices, tickers, sessions):
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("ETF price dates must be unique and increasing")
    prices = prices.loc[sessions[0]:sessions[-1], tickers]
    if not prices.index.equals(sessions):
        missing = sessions.difference(prices.index)
        extra = prices.index.difference(sessions)
        first_missing = missing[:5].strftime("%Y-%m-%d").tolist()
        first_extra = extra[:5].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"ETF prices do not match NYSE sessions: "
                         f"{len(missing)} missing (first dates {first_missing}), "
                         f"{len(extra)} extra (first dates {first_extra})")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"Missing or invalid adjusted ETF close for {', '.join(tickers)}")
    return prices


def download_prices(cache_path, tickers, sessions):
    for attempt in range(3):
        try:
            history = yf.download(
                tickers,
                start=sessions[0].date().isoformat(),
                end=(sessions[-1] + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d",
                auto_adjust=True,
                progress=False,
                timeout=10,
            )
            prices = validate_prices(history["Close"], tickers, sessions)
        except (KeyError, ValueError, RuntimeError, OSError) as error:
            if attempt == 2:
                raise RuntimeError(f"Unable to obtain complete Yahoo Finance prices for {tickers}") from error
            time.sleep(2 ** attempt)
        else:
            fetched_at = datetime.now(OSLO).isoformat(timespec="seconds")
            with TemporaryDirectory(dir=cache_path.parent) as directory:
                temporary_path = Path(directory) / cache_path.name
                with temporary_path.open("w", encoding="utf-8", newline="") as handle:
                    handle.write(f"{FETCH_HEADER}{fetched_at}\n")
                    prices.to_csv(handle, index_label="Date")
                temporary_path.replace(cache_path)
            return


def load_prices(tickers, start, end, sessions, legacy_snapshot=None):
    CACHE.mkdir(exist_ok=True)
    symbols = "-".join(sorted(tickers))
    cache_path = CACHE / f"yahoo_adjusted_{symbols}_{start.date()}_{end.date()}.csv"
    if not cache_path.is_file():
        download_prices(cache_path, tickers, sessions)
    snapshot = cache_path.read_bytes()
    snapshot_hash = sha256(snapshot).hexdigest()
    with BytesIO(snapshot) as handle:
        header = handle.readline().decode("utf-8")
        if header.startswith(FETCH_HEADER):
            fetched_at = header.removeprefix(FETCH_HEADER).strip()
        else:
            handle.seek(0)
            fetched_at = "unknown (legacy CSV without fetch metadata)"
            if legacy_snapshot and snapshot_hash == legacy_snapshot["sha256"]:
                fetched_at = f"{legacy_snapshot['file_timestamp']} (legacy CSV file timestamp)"
        prices = pd.read_csv(handle, index_col="Date", parse_dates=["Date"])
    return validate_prices(prices, tickers, sessions), fetched_at, snapshot_hash


def hedge_targets(returns, rebalance_dates, universe, lookback):
    market = universe["market"]
    intangible = universe["intangible_value"]
    value = universe["conventional_value"]
    growth = universe["growth"]
    factors = pd.DataFrame({
        "market": returns[market],
        "growth_minus_value": returns[growth] - returns[value],
    })
    targets = {}
    for date in rebalance_dates:
        end = returns.index.get_loc(date)
        if end < lookback:
            continue
        # The execution day's close-to-close return belongs to the old positions.
        window = slice(end - lookback, end)
        design = np.column_stack((np.ones(lookback), factors.iloc[window].to_numpy()))
        observations = returns[[intangible, value]].iloc[window].to_numpy()
        coefficients, _, rank, _ = np.linalg.lstsq(design, observations, rcond=None)
        if rank < design.shape[1]:
            raise ValueError(f"Hedge regressors are collinear for execution on {date.date()}")
        market_beta, style_beta = coefficients[1:, 0] - coefficients[1:, 1]
        targets[date] = {
            market: -market_beta,
            intangible: 1.0,
            value: -1.0 + style_beta,
            growth: -style_beta,
        }
    return pd.DataFrame.from_dict(targets, orient="index", columns=returns.columns).sort_index()


def simulate(prices, targets, cost_rate, slippage_rate, cash_buffer, borrow_rate=0.0):
    """Fix share quantities from the prior close, then fill at the next close."""
    columns = ["return", "turnover", "n_trades"]
    if targets.empty:
        return pd.DataFrame(columns=columns, index=prices.index[:0])

    positions = np.zeros(len(prices.columns))
    cash = previous_value = 1.0
    targets_by_date = dict(zip(targets.index, targets.to_numpy()))
    start = prices.index.get_loc(targets.index[0])
    if start == 0:
        raise ValueError("The first fill requires a previous close")
    dates = prices.index[start - 1:]
    closes = prices.iloc[start - 1:].to_numpy()
    records = []
    for i, date in enumerate(dates[1:], 1):
        prior_close = closes[i - 1]
        target = targets_by_date.get(date)
        shares_to_fill = np.zeros_like(positions)
        if target is not None:
            gross_weight = np.abs(target).sum()
            budget = previous_value * (1.0 - cash_buffer) / max(1.0, gross_weight)
            shares_to_fill = budget * target / prior_close - positions

        short_positions = np.maximum(-positions, 0.0)
        elapsed_days = (date - dates[i - 1]).days
        cash -= (short_positions @ prior_close) * borrow_rate * elapsed_days / 365.0
        close = closes[i]
        pre_fill_value = cash + positions @ close
        if not np.isfinite(pre_fill_value) or pre_fill_value <= 0:
            raise ValueError(f"Nonpositive or invalid portfolio value on {date.date()}")
        if cash < 2.0 * (short_positions @ close):
            raise ValueError(f"Insufficient cash or short collateral before the fill on {date.date()}")

        fill_prices = close * (1.0 + slippage_rate * np.sign(shares_to_fill))
        fill_amounts = shares_to_fill * fill_prices
        traded = np.abs(fill_amounts).sum()
        cash -= fill_amounts.sum() + traded * cost_rate
        positions += shares_to_fill
        if cash < 0:
            raise ValueError(f"Predetermined shares would make cash negative on {date.date()}")
        # Retain short proceeds and post the same amount of own cash at the close.
        if cash < 2.0 * (np.maximum(-positions, 0.0) @ close):
            raise ValueError(f"Predetermined shares exceed short collateral on {date.date()}")
        closing_value = cash + positions @ close
        if not np.isfinite(closing_value) or closing_value <= 0:
            raise ValueError(f"Nonpositive or invalid portfolio value after the fill on {date.date()}")
        records.append((closing_value / previous_value - 1.0, traded / pre_fill_value,
                        int(np.count_nonzero(shares_to_fill))))
        previous_value = closing_value
    return pd.DataFrame(records, index=dates[1:], columns=columns)


def metrics(daily_results, cash_returns, start, end, periods_per_year):
    segment = daily_results.loc[start:end]
    result = dict.fromkeys(["cagr", "ann_vol", "sharpe", "sharpe_excess",
                            "max_drawdown", "total_return", "turnover"])
    result.update(n_trades=0, n_days=len(segment), data_range=None)
    if segment.empty:
        return result

    returns = segment["return"]
    excess = returns - cash_returns.loc[segment.index]
    wealth = (1.0 + returns).cumprod()
    std = returns.std(ddof=1)
    excess_std = excess.std(ddof=1)
    annual_scale = np.sqrt(periods_per_year)
    values = {
        "cagr": wealth.iloc[-1] ** (periods_per_year / len(returns)) - 1.0,
        "ann_vol": std * annual_scale,
        "sharpe": returns.mean() / std * annual_scale if std > 0 else np.nan,
        "sharpe_excess": excess.mean() / excess_std * annual_scale if excess_std > 0 else np.nan,
        "max_drawdown": (wealth / wealth.cummax().clip(lower=1.0) - 1.0).min(),
        "total_return": wealth.iloc[-1] - 1.0,
        "turnover": segment["turnover"].sum(),
    }
    result.update({name: float(value) if np.isfinite(value) else None for name, value in values.items()})
    result.update(
        n_trades=int(segment["n_trades"].sum()),
        data_range=f"{segment.index[0].date()} to {segment.index[-1].date()}",
    )
    return result


def print_table(reports):
    print(f"{'Portfolio':<25} {'Period':<6} {'CAGR':>9} {'Ann. vol':>9} {'Sharpe':>8} {'Excess':>8} "
          f"{'Max DD':>9} {'Total':>9} {'Turnover':>10} {'Trades':>6} {'Days':>5}")
    for name, split in reports.items():
        for period, row in split.items():
            display = []
            for key in ("cagr", "ann_vol", "sharpe", "sharpe_excess", "max_drawdown", "total_return", "turnover"):
                value = row[key]
                display.append("n/a" if value is None else format(value, ".2f" if key.startswith("sharpe") else ".2%"))
            cagr, vol, sharpe, excess, drawdown, total, turnover = display
            print(f"{name:<25} {period.upper():<6} {cagr:>9} {vol:>9} {sharpe:>8} {excess:>8} "
                  f"{drawdown:>9} {total:>9} {turnover:>10} {row['n_trades']:6d} {row['n_days']:5d}")


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data, in_sample, out_of_sample = (config[key] for key in ("data", "in_sample", "out_of_sample"))
    start, end = pd.Timestamp(data["start"]), pd.Timestamp(data["end"])
    first_close = pd.Timestamp(data["first_common_close"])
    is_start, is_end = pd.Timestamp(in_sample["start"]), pd.Timestamp(in_sample["end"])
    oos_start, oos_end = pd.Timestamp(out_of_sample["start"]), pd.Timestamp(out_of_sample["end"])
    if not start <= is_start <= is_end < oos_start <= oos_end <= end:
        raise ValueError("IS and OOS ranges must be ordered, disjoint and within the data range")
    if not start <= first_close < end:
        raise ValueError("First common close must lie within the sample, before its end")
    sessions = trading_sessions(first_close, end)
    if len(sessions) < 2 or sessions[0] != first_close:
        raise ValueError("At least two sessions are required, starting at the first common close")
    splits = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (start, end)}
    universe = config["universe"]
    market, intangible, value, growth = (
        universe[role] for role in ("market", "intangible_value", "conventional_value", "growth")
    )
    tickers = [market, intangible, value, growth]
    if len(set(tickers)) != len(tickers):
        raise ValueError("Use four distinct ETF tickers")
    sleeve = float(config["sleeve_weight"])
    lookback = int(config["hedge_lookback_sessions"])
    cost_bps = float(config["transaction_cost_bps"])
    slippage_bps = float(config["slippage_bps"])
    cash_buffer = float(config["cash_buffer"])
    hedge_cash_buffer = float(config["hedge_cash_buffer"])
    short_borrow_bps = float(config["short_borrow_bps_per_year"])
    periods_per_year = int(config["trading_days_per_year"])
    if not (0 < sleeve < 1 and lookback >= 3 and periods_per_year > 0):
        raise ValueError("Invalid sleeve weight, lookback or annualization factor")
    if not (cost_bps >= 0 and slippage_bps >= 0 and cost_bps + slippage_bps < 10000
            and 0 <= short_borrow_bps < 10000 and 0 < cash_buffer < 1 and 0 < hedge_cash_buffer < 1):
        raise ValueError("Invalid fees, slippage, borrow rate or cash buffer")

    cash_proxy = data["cash_proxy"]
    with TemporaryDirectory(dir=ROOT) as directory:
        yf.set_tz_cache_location(directory)
        prices, equity_fetch, equity_hash = load_prices(tickers, start, end, sessions, data.get("legacy_snapshot"))
        cash_prices, cash_fetch, cash_hash = load_prices([cash_proxy], first_close, end, sessions)
    cash_returns = cash_prices[cash_proxy].pct_change(fill_method=None).iloc[1:]
    returns = prices.pct_change(fill_method=None).iloc[1:]
    months = returns.index.to_period("M")
    rebalance_dates = returns.index[:1].union(returns.index[1:][months[1:] != months[:-1]])
    strategy_name = f"{market}/{intangible} sleeve"
    value_name = f"{market}/{value} sleeve"
    market_name = f"{market} buy and hold"
    hedge_name = f"Hedged {intangible}-{value}"
    targets = {
        strategy_name: pd.DataFrame({market: 1 - sleeve, intangible: sleeve}, index=rebalance_dates),
        value_name: pd.DataFrame({market: 1 - sleeve, value: sleeve}, index=rebalance_dates),
        market_name: pd.DataFrame({market: 1.0}, index=rebalance_dates[:1]),
        hedge_name: hedge_targets(returns, rebalance_dates, universe, lookback),
    }
    reports = {}
    for name, weights in targets.items():
        weights = weights.reindex(columns=tickers, fill_value=0.0)
        is_hedge = name == hedge_name
        daily_results = simulate(prices, weights, cost_bps / 10000.0, slippage_bps / 10000.0,
                                 hedge_cash_buffer if is_hedge else cash_buffer,
                                 short_borrow_bps / 10000.0 if is_hedge else 0.0)
        reports[name] = {
            period: metrics(daily_results, cash_returns, left, right, periods_per_year)
            for period, (left, right) in splits.items()
        }

    print(f"PARAM_COMBOS_TRIED: {PARAM_COMBOS_TRIED}")
    print(f"DATA_FETCHED_AT: Equity ETFs {equity_fetch} | {cash_proxy} {cash_fetch}")
    print(f"DATA_SHA256: Equity ETFs {equity_hash} | {cash_proxy} {cash_hash}")
    print("PRICE_ADJUSTMENT: Yahoo Finance daily close adjusted for splits and distributions (auto_adjust=True)")
    print(f"Adjusted closes: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"IS: {is_start.date()} to {is_end.date()} | OOS: {oos_start.date()} to {oos_end.date()}")
    print(f"One-way fee: {cost_bps:g} bps | One-way slippage: {slippage_bps:g} bps | "
          f"Short borrow: {short_borrow_bps:g} bps/year | Excess Sharpe cash proxy: {cash_proxy}")
    print(f"Sizing buffers: {cash_buffer:.0%} long-only, {hedge_cash_buffer:.0%} hedge | "
          f"Hedge evaluated: {reports[hedge_name]['full']['data_range'] or 'n/a'}")
    print_table(reports)
    strategy = reports[strategy_name]
    benchmark = {"name": value_name, **reports[value_name]}
    benchmark["buy_and_hold"] = {"name": market_name, **reports[market_name]}
    results = {
        "strategy": config["strategy"],
        "data_range": strategy["full"]["data_range"],
        **{period: {**row, "hedged_spread": reports[hedge_name][period]} for period, row in strategy.items()},
        "benchmark": benchmark,
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
