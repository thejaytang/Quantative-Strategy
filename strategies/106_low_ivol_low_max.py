# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 106_low_ivol_low_max\nuniverse:\n  name: US24\n  tickers: [AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, BAC, GS, MS, XOM, CVX,\n            COP, SLB, WMT, COST, PG, KO, PEP, JNJ, PFE, MRK, HD, LOW]\nmarket_proxy: SPY\ncash_proxy: BIL\ndata_start: 2012-06-01\nin_sample:\n  start: 2013-01-01\n  end: 2019-12-31\nout_of_sample:\n  start: 2020-01-01\n  end: 2025-12-31\nsignals:\n  ivol_lookback_sessions: 63\n  max_lookback_sessions: 21\n  portfolio_size: 6\ncosts:\n  commission_bps: 3\n  slippage_bps: 2\n  sensitivity:\n    - {commission_bps: 0, slippage_bps: 0}\n    - {commission_bps: 3, slippage_bps: 12}\n# Reserve this fraction of the prior close's investable value for opening gaps and costs.\ncash_buffer_fraction: 0.05\nannualization_sessions: 252\n"
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
"""Monthly US24 low-IVOL, low-MAX and combined backtest."""

from datetime import datetime
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
SIGNAL_PORTFOLIOS = ("low_ivol", "low_max", "combined")
BENCHMARK = "equal_weight"
FIELDS = ("Open", "Close", "Dividends", "Capital Gains")
PARAM_COMBOS_TRIED = 1
PRICE_ADJUSTMENT = (
    "Yahoo split-adjusted, dividend-unadjusted Open/Close. "
    "Distributions valued at ex-date close and excluded from buying power"
)


class USStockHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def trading_sessions(start, end):
    # Unscheduled full-market closures during the research window.
    closures = pd.to_datetime(["2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])
    weekdays = pd.bdate_range(start, end)
    return weekdays.difference(USStockHolidays().holidays(start, end).union(closures))


def align_prices(prices, sessions, reference_tickers):
    prices = prices.reindex(sessions)
    missing = prices["Close"][reference_tickers].isna().any(axis=1)
    if missing.any():
        dates = ", ".join(prices.index[missing].strftime("%Y-%m-%d")[:8])
        raise ValueError(f"Missing market or cash proxy close on {missing.sum()} exchange sessions: {dates}")
    for field in ("Dividends", "Capital Gains"):
        actions = prices[field]
        invalid = ~np.isfinite(actions.fillna(0.0)) | actions.lt(0)
        if (invalid | (actions.isna() & prices["Close"].notna())).any().any():
            raise ValueError(f"Missing or invalid Yahoo cash distribution field: {field}")
    return prices


def download_prices(tickers, start, end, sessions, reference_tickers):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_key = hashlib.sha256(
        f"{start.date()}:{end.date()}:{','.join(sorted(tickers))}".encode()
    ).hexdigest()[:16]
    cache = cache_dir / f"yfinance_split_only_v1_{cache_key}.csv.gz"
    columns = pd.MultiIndex.from_product([FIELDS, tickers])
    if cache.exists():
        with gzip.open(cache, "rt") as handle:
            fetched_at = datetime.fromisoformat(handle.readline().strip()).isoformat(timespec="seconds")
            prices = pd.read_csv(handle, header=[0, 1], index_col=0, parse_dates=True)
        return align_prices(prices.reindex(columns=columns), sessions, reference_tickers), fetched_at

    frames = []
    pending = list(tickers)
    with TemporaryDirectory(prefix="yf_session_", dir=cache_dir) as session_cache:
        yf.set_tz_cache_location(session_cache)
        for _ in range(3):
            bars = yf.download(
                pending, start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=False, actions=True, keepna=True,
                progress=False, threads=8, timeout=10, multi_level_index=True,
            )
            for ticker in pending:
                if ("Capital Gains", ticker) not in bars.columns:
                    bars[("Capital Gains", ticker)] = 0.0
            bars = bars.reindex(columns=pd.MultiIndex.from_product([FIELDS, pending]))
            for field in ("Open", "Close"):
                quotes = bars[field]
                bars[field] = quotes.where(np.isfinite(quotes) & quotes.gt(0))
            usable = (bars["Open"].notna() & bars["Close"].notna() & bars["Dividends"].notna()).any()
            received = usable.index[usable].tolist()
            if received:
                frames.append(bars.loc[:, bars.columns.get_level_values(1).isin(received)])
            pending = [ticker for ticker in pending if ticker not in received]
            if not pending:
                break
    if pending:
        raise RuntimeError(f"Yahoo Finance returned no usable bars for {', '.join(pending)}")

    prices = pd.concat(frames, axis=1).reindex(columns=columns).sort_index()
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = align_prices(prices, sessions, reference_tickers)
    fetched_at = datetime.now(ZoneInfo("Europe/Oslo")).isoformat(timespec="seconds")
    with NamedTemporaryFile(dir=cache_dir, suffix=".csv.gz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with gzip.open(temporary, "wt") as handle:
            handle.write(f"{fetched_at}\n")
            prices.to_csv(handle)
        temporary.replace(cache)
    finally:
        temporary.unlink(missing_ok=True)
    return prices, fetched_at


def total_returns(closes, distributions):
    return (closes + distributions).div(closes.shift()).sub(1)


def build_targets(stock_returns, market_returns, first_day, last_day, signals):
    ivol_window = signals["ivol_lookback_sessions"]
    max_window = signals["max_lookback_sessions"]
    size = signals["portfolio_size"]
    lookback = max(ivol_window, max_window)
    days = stock_returns.index

    # A month changes on the exchange calendar after the signal close.
    months = days.to_period("M")
    signal_indices = np.flatnonzero(months[:-1] != months[1:])
    signal_indices = [
        i for i in signal_indices if i >= lookback and first_day <= days[i + 1] <= last_day
    ]
    if not signal_indices:
        raise ValueError("No monthly rebalance has sufficient signal history in the requested dates")
    fill_days = days[np.asarray(signal_indices) + 1]
    targets = {name: pd.DataFrame(0.0, index=fill_days, columns=stock_returns.columns)
               for name in (*SIGNAL_PORTFOLIOS, BENCHMARK)}

    for i, fill_day in zip(signal_indices, fill_days):
        stock_window = stock_returns.iloc[i - lookback + 1:i + 1].dropna(axis=1)
        market_window = market_returns.iloc[i - ivol_window + 1:i + 1].to_numpy()
        eligible = stock_window.columns
        if len(eligible) < size:
            continue
        targets[BENCHMARK].loc[fill_day, eligible] = 1 / len(eligible)
        max_return = stock_window.iloc[-max_window:].max()
        selected = max_return.sort_index().nsmallest(size).index
        targets["low_max"].loc[fill_day, selected] = 1 / size
        if not np.isfinite(market_window).all() or np.ptp(market_window) == 0:
            continue
        design = np.column_stack((np.ones(ivol_window), market_window))
        stock_ivol_window = stock_window.iloc[-ivol_window:].to_numpy()
        coefficients = np.linalg.lstsq(design, stock_ivol_window, rcond=None)[0]
        ivol = pd.Series((stock_ivol_window - design @ coefficients).std(axis=0, ddof=1), index=eligible)
        scores = {
            "low_ivol": ivol,
            "combined": (ivol.rank(pct=True) + max_return.rank(pct=True)) / 2,
        }
        for name, score in scores.items():
            selected = score.sort_index().nsmallest(size).index
            targets[name].loc[fill_day, selected] = 1 / size

    return targets


def plan_shares(shares, target, previous_close, previous_investable_value, cash_buffer):
    requested = np.zeros_like(shares)
    selected = target > 0
    budget = previous_investable_value * (1 - cash_buffer)
    requested[selected] = budget * target[selected] / previous_close[selected]
    return requested


def fill_rebalance(shares, cash, requested, opening, restricted_distributions,
                   commission_rate, slippage_rate, date):
    # Unquoted new stocks cannot fill. Every quoted stock fills the planned quantity.
    quoted = np.isfinite(opening)
    updated_shares = np.where(quoted, requested, shares)
    changes = updated_shares - shares
    opening = np.where(quoted, opening, 0.0)
    opening_value = cash + shares @ opening + restricted_distributions
    sell_value = (np.maximum(-changes, 0) @ opening) * (1 - slippage_rate)
    buy_value = (np.maximum(changes, 0) @ opening) * (1 + slippage_rate)
    remaining_cash = cash + sell_value * (1 - commission_rate) - buy_value * (1 + commission_rate)
    if remaining_cash < 0:
        raise RuntimeError(
            f"Predetermined rebalance on {date.date()} exceeded available cash by "
            f"{-remaining_cash / opening_value:.2%} of the opening portfolio value"
        )

    turnover = (sell_value + buy_value) / opening_value
    return updated_shares, remaining_cash, turnover, np.count_nonzero(changes)


def simulate(opens, closes, previous_closes, distributions, targets, costs, cash_buffer):
    """Retain split-adjusted share units between rebalances and value at each close."""
    opening_prices = opens.to_numpy()
    closing_prices = closes.to_numpy()
    sizing_prices = previous_closes.to_numpy()
    distributions = distributions.to_numpy()
    target_rows = targets.reindex(closes.index).to_numpy()
    shares = np.zeros(closes.shape[1])
    cash = previous_investable_value = previous_value = 1.0
    restricted_distributions = 0.0
    commission_rate = costs["commission_bps"] / 10_000
    slippage_rate = costs["slippage_bps"] / 10_000
    daily = np.zeros((len(closes), 4))
    fill_positions = {}

    for i, date in enumerate(closes.index):
        prior_shares = shares
        target = target_rows[i]
        if np.isfinite(target[0]):
            requested = plan_shares(shares, target, sizing_prices[i], previous_investable_value, cash_buffer)
            opening = opening_prices[i]
            held = shares > 0
            if not np.isfinite(opening[held]).all():
                missing = closes.columns[held & ~np.isfinite(opening)].tolist()
                raise ValueError(f"Missing held-stock opening quote on {date.date()} for {missing}")
            shares, cash, turnover, trades = fill_rebalance(
                shares, cash, requested, opening, restricted_distributions,
                commission_rate, slippage_rate, date,
            )
            daily[i, 1:3] = turnover, trades
            fill_positions[date] = shares

        # Payment dates are absent. Include the prior positions' entitlement in value only.
        restricted_distributions += prior_shares @ distributions[i]
        closing = closing_prices[i]
        held = shares > 0
        if not np.isfinite(closing[held]).all():
            missing = closes.columns[held & ~np.isfinite(closing)].tolist()
            raise ValueError(f"Missing held-stock closing quote on {date.date()} for {missing}")
        previous_investable_value = cash + shares[held] @ closing[held]
        value = previous_investable_value + restricted_distributions
        daily[i, 0] = value / previous_value - 1
        daily[i, 3] = cash
        previous_value = value

    daily_results = pd.DataFrame(daily, index=closes.index, columns=["return", "turnover", "n_trades", "cash"])
    positions = pd.DataFrame.from_dict(fill_positions, orient="index", columns=closes.columns)
    return daily_results, positions


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def metrics(daily, market_returns, cash_returns, sessions):
    returns = daily["return"]
    if returns.empty:
        raise ValueError("An evaluation period has no market sessions")
    growth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    std = returns.std()
    excess = returns - cash_returns.reindex(returns.index)
    excess_std = excess.std() if excess.notna().all() else np.nan
    paired = pd.concat([returns, market_returns.reindex(returns.index)], axis=1).dropna()
    market_var = paired.iloc[:, 1].var()
    beta = paired.iloc[:, 0].cov(paired.iloc[:, 1]) / market_var if market_var > 0 else np.nan
    return {
        "cagr": finite_or_none(growth[-1] ** (sessions / len(returns)) - 1),
        "ann_vol": finite_or_none(std * np.sqrt(sessions)),
        "sharpe": finite_or_none(returns.mean() / std * np.sqrt(sessions) if std > 0 else np.nan),
        "sharpe_excess": finite_or_none(excess.mean() / excess_std * np.sqrt(sessions) if excess_std > 0 else np.nan),
        "max_drawdown": finite_or_none(np.min(growth / np.maximum.accumulate(growth) - 1)),
        "total_return": finite_or_none(growth[-1] - 1),
        "turnover": finite_or_none(daily["turnover"].mean() * sessions),
        "n_trades": int(daily["n_trades"].sum()),
        "spy_beta": finite_or_none(beta),
    }


def portfolio_overlap(positions, periods):
    overlap = {}
    for name, (start, end) in periods.items():
        overlap[name] = {}
        for left, right in combinations(SIGNAL_PORTFOLIOS, 2):
            a = positions[left].loc[start:end].gt(0)
            b = positions[right].loc[start:end].gt(0)
            both_invested = a.any(axis=1) & b.any(axis=1)
            shared = (a & b).sum(axis=1)[both_invested]
            overlap[name][f"{left}_vs_{right}"] = {
                "mean_shared_names": finite_or_none(shared.mean()),
                "n_rebalances": len(shared),
            }
    return overlap


def print_report(result, reports, baseline_bps, costs, cash_proxy, fetched_at):
    def display(value, percent=False):
        if value is None:
            return "n/a"
        return f"{value:.2%}" if percent else f"{value:.2f}"

    print(f"PARAM_COMBOS_TRIED: {PARAM_COMBOS_TRIED}")
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"{result['strategy']} | {result['data_range']} | "
          f"{costs['commission_bps']:g} bps commission + {costs['slippage_bps']:g} bps slippage per side")
    print(f"Portfolios and benchmark (net of costs, excess Sharpe vs {cash_proxy} total returns)")
    rows = []
    for portfolio, periods in reports[baseline_bps].items():
        for period, values in periods.items():
            rows.append({
                "Portfolio": portfolio, "Period": period.upper(),
                "CAGR": display(values["cagr"], True), "Ann. vol": display(values["ann_vol"], True),
                "Sharpe": display(values["sharpe"]), "Excess Sharpe": display(values["sharpe_excess"]),
                "Max DD": display(values["max_drawdown"], True),
                "Total return": display(values["total_return"], True),
                "Ann. turnover": display(values["turnover"], True),
                "Trades": values["n_trades"], "SPY beta": display(values["spy_beta"]),
            })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nMean shared positions per rebalance")
    overlap = {period.upper(): {pair: display(values["mean_shared_names"])
                               for pair, values in pairs.items()}
               for period, pairs in result["overlap"].items()}
    print(pd.DataFrame(overlap).to_string())

    print("\nCost sensitivity (CAGR by period, 0 bps is gross)")
    cost_rows = []
    for cost, portfolios in reports.items():
        for portfolio, periods in portfolios.items():
            cost_rows.append({"Bps": cost, "Portfolio": portfolio, **{
                period.upper(): display(values["cagr"], True) for period, values in periods.items()
            }})
    print(pd.DataFrame(cost_rows).to_string(index=False))
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False, separators=(",", ":")))


def main():
    with (ROOT / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    universe = config["universe"]
    tickers = sorted(universe["tickers"])
    market = config["market_proxy"]
    cash_proxy = config["cash_proxy"]
    start = pd.Timestamp(config["data_start"])
    in_sample, out_of_sample = (
        tuple(pd.Timestamp(config[period][bound]) for bound in ("start", "end"))
        for period in ("in_sample", "out_of_sample")
    )
    first_day, last_day = in_sample[0], out_of_sample[1]
    if not start < first_day <= in_sample[1] < out_of_sample[0] <= last_day:
        raise ValueError("Expected a warmup followed by chronological, non-overlapping IS and OOS ranges")
    signals = config["signals"]
    if (len(set([*tickers, market, cash_proxy])) != len(tickers) + 2
            or not 1 <= signals["portfolio_size"] <= len(tickers)
            or signals["ivol_lookback_sessions"] < 3 or signals["max_lookback_sessions"] < 1):
        raise ValueError("Invalid universe, portfolio size or lookback")
    costs = config["costs"]
    cash_buffer = config["cash_buffer_fraction"]
    if not 0 < cash_buffer < 1:
        raise ValueError("Cash buffer must be a positive fraction below one")
    baseline_bps = costs["commission_bps"] + costs["slippage_bps"]
    cost_levels = {}
    for case in (costs, *costs["sensitivity"]):
        commission, slippage = case["commission_bps"], case["slippage_bps"]
        total = commission + slippage
        if not (commission >= 0 and slippage >= 0 and total < 1000) or total in cost_levels:
            raise ValueError("Cost scenarios must have distinct nonnegative totals below 1000 bps")
        cost_levels[total] = case

    sessions = trading_sessions(start, last_day)
    references = [market, cash_proxy]
    prices, fetched_at = download_prices([*tickers, *references], start, last_day, sessions, references)
    close_prices = prices["Close"]
    distributions = (prices["Dividends"] + prices["Capital Gains"]).fillna(0.0)
    returns = total_returns(close_prices, distributions)
    market_returns = returns[market]
    cash_returns = returns[cash_proxy]
    if not np.isfinite(cash_returns.loc[first_day:last_day]).all():
        raise ValueError(f"Missing {cash_proxy} total returns during the evaluation period")
    targets = build_targets(returns[tickers], market_returns, first_day, last_day, signals)
    opens = prices["Open"][tickers].loc[first_day:last_day]
    closes = close_prices[tickers].loc[first_day:last_day]
    previous_closes = close_prices[tickers].shift().loc[first_day:last_day]
    stock_distributions = distributions[tickers].loc[first_day:last_day]
    periods = {"is": in_sample, "oos": out_of_sample, "full": (first_day, last_day)}

    reports = {}
    baseline_positions = {}
    for cost, case in sorted(cost_levels.items()):
        reports[cost] = {}
        for portfolio, weights in targets.items():
            daily_results, positions = simulate(
                opens, closes, previous_closes, stock_distributions, weights, case, cash_buffer,
            )
            if cost == baseline_bps:
                baseline_positions[portfolio] = positions
            reports[cost][portfolio] = {
                period: metrics(
                    daily_results.loc[left:right], market_returns, cash_returns, config["annualization_sessions"],
                )
                for period, (left, right) in periods.items()
            }

    net = reports[baseline_bps]
    result = {
        "strategy": config["strategy"],
        "data_range": f"{closes.index[0].date()} to {closes.index[-1].date()}",
        **net["combined"],
        "benchmark": {"name": f"{universe['name']}_equal_weight", **net[BENCHMARK]},
        "books": {portfolio: net[portfolio] for portfolio in SIGNAL_PORTFOLIOS if portfolio != "combined"},
        "overlap": portfolio_overlap(baseline_positions, periods),
        "cost_bps_per_dollar_traded": baseline_bps,
        "cost_sensitivity": {f"{cost:g}": report for cost, report in reports.items() if cost != baseline_bps},
    }
    print_report(result, reports, baseline_bps, costs, cash_proxy, fetched_at)


if __name__ == "__main__":
    main()
