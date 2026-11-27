# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 177_overnight_news_returns\ncategory: text-sentiment\n\n# Fixed C82 stock universe with issuer CIKs.\nuniverse:\n  AAPL: "0000320193"\n  MSFT: "0000789019"\n  AMZN: "0001018724"\n  GOOGL: "0001652044"\n  META: "0001326801"\n  NVDA: "0001045810"\n  JPM: "0000019617"\n  BAC: "0000070858"\n  XOM: "0000034088"\n  CVX: "0000093410"\n  WMT: "0000104169"\n  COST: "0000909832"\ncash_ticker: BIL\nbenchmark_ticker: SPY\n\n# Inclusive dates for execution sessions. Earlier bars are used for signal warmup.\nin_sample:\n  start: 2016-01-01\n  end: 2020-12-31\nout_of_sample:\n  start: 2021-01-01\n  end: 2025-12-31\n\nminimum_gap: 0.01\nmax_stock_weight: 0.25\ngap_match_cutoffs: [0.02, 0.03, 0.05, 0.10]\n# Costs are in basis points per buy or sell, applied to both stocks and BIL.\ncommission_bps: 0\nslippage_bps: 5\n# Alternative all-in per-leg costs, used only for sensitivity analysis.\ncost_sensitivity_bps: [0, 15]\n'
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
"""Backtest next-session returns after an upward gap and an overnight 8-K."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SNAPSHOT_FETCHED_ON = "2026-11-25"
PRICE_SNAPSHOT = CACHE / (
    "adjusted_2015-12-18_2025-12-31_AAPL-MSFT-AMZN-GOOGL-META-NVDA-JPM-BAC-XOM-CVX-WMT-COST-BIL-SPY.csv"
)
FILING_SNAPSHOT_DATES = "2015-12-18_2026-01-07"
EASTERN = "America/New_York"
OSLO = "Europe/Oslo"
SESSIONS_PER_YEAR = 252
DAY = pd.Timedelta(days=1)
EARLY_CLOSES = pd.to_datetime([
    "2015-12-24", "2016-11-25", "2017-07-03", "2017-11-24",
    "2018-07-03", "2018-11-23", "2018-12-24",
    "2019-07-03", "2019-11-29", "2019-12-24", "2020-11-27", "2020-12-24",
    "2021-11-26", "2022-11-25", "2023-07-03", "2023-11-24",
    "2024-07-03", "2024-11-29", "2024-12-24",
    "2025-07-03", "2025-11-28", "2025-12-24",
])
ITEM_202 = r"(?<![\d.])2\.02(?![\d.])"


def load_prices(tickers, start, end):
    columns = pd.MultiIndex.from_product([["Open", "Close"], tickers])
    if PRICE_SNAPSHOT.is_file():
        prices = pd.read_csv(PRICE_SNAPSHOT, header=[0, 1], index_col=0, parse_dates=True)
        fetched_on = SNAPSHOT_FETCHED_ON
    else:
        with TemporaryDirectory(dir=ROOT) as yahoo_cache:
            yf.set_tz_cache_location(yahoo_cache)
            prices = yf.download(
                tickers, start=str(start.date()), end=str((end + DAY).date()),
                auto_adjust=True, keepna=True, ignore_tz=True, progress=False,
                threads=4, timeout=12,
            )
        fetched_on = pd.Timestamp.now(tz=OSLO).date().isoformat()
    if prices is None or prices.empty or not columns.isin(prices.columns).all():
        raise RuntimeError("Yahoo Finance data lacks adjusted open and close for one or more tickers")
    prices = prices.loc[:, columns]
    prices.index = pd.DatetimeIndex(prices.index).as_unit("ns")
    prices = prices.sort_index().loc[start:end]
    prices = prices.where(np.isfinite(prices) & prices.gt(0)).dropna(how="all")
    if not prices.notna().any().all() or not prices.index.is_unique or len(prices.loc[:start + 7 * DAY]) < 2:
        raise RuntimeError("Price data has invalid dates, missing tickers or insufficient warmup")
    if prices.index[-1] < end - 4 * DAY:
        raise RuntimeError("Price data ends before the requested sample")
    return prices["Open"], prices["Close"], fetched_on


def earnings_filings(cik):
    """Use the bundled public submissions and original archive timestamps."""
    submissions = CACHE / f"submissions_{cik}_{FILING_SNAPSHOT_DATES}.csv"
    acceptance = CACHE / f"acceptance_{cik}_{FILING_SNAPSHOT_DATES}.csv"
    if not submissions.is_file() or not acceptance.is_file():
        raise RuntimeError(f"Missing bundled EDGAR snapshot for CIK {cik}")
    filings = pd.read_csv(submissions, dtype=str, keep_default_na=False)
    accessions = filings.loc[
        filings["form"].eq("8-K") & filings["items"].str.contains(ITEM_202),
        "accession",
    ]
    timestamps = pd.read_csv(acceptance, dtype=str, index_col="accession")["accepted_et"]
    if filings.empty or not accessions.is_unique or not timestamps.index.is_unique:
        raise RuntimeError(f"Invalid EDGAR snapshot for CIK {cik}")
    selected = timestamps.reindex(accessions)
    if not selected.str.fullmatch(r"[0-9]{14}", na=False).all():
        raise RuntimeError(f"Missing or invalid Item 2.02 acceptance timestamp for CIK {cik}")
    return pd.to_datetime(pd.Index(selected), format="%Y%m%d%H%M%S").tz_localize(EASTERN)


def overnight_sessions(sessions, arrivals):
    """Map each filing strictly inside a close-to-open window to that opening session."""
    opens = (sessions + pd.Timedelta(hours=9, minutes=30)).tz_localize(EASTERN)
    close_hours = np.where(sessions.isin(EARLY_CLOSES), 13, 16)
    closes = (sessions + pd.to_timedelta(close_hours, unit="h")).tz_localize(EASTERN)
    following = opens.searchsorted(arrivals, side="right")
    within_range = (following > 0) & (following < len(sessions))
    following = following[within_range]
    arrivals = arrivals[within_range]
    overnight = (arrivals > closes[following - 1]) & (arrivals < opens[following])
    return sessions[following[overnight]].unique()


def load_events(sessions, universe):
    events = pd.DataFrame(False, index=sessions, columns=list(universe))
    for ticker, cik in universe.items():
        arrivals = earnings_filings(cik)
        overnight = overnight_sessions(sessions, arrivals)
        events.loc[overnight, ticker] = True
        print(f"EDGAR {ticker}: {len(overnight)} overnight sessions from {len(arrivals)} original Item 2.02 filings")
    return events


def daily_stock_book(weights, open_to_close, cash_overnight, cash_intraday, cost_bps):
    """Start and end each session in BIL. Swap the day's stock sleeve at each auction."""
    invested = weights.sum(axis=1)
    active = weights.gt(0)
    if (active & open_to_close.isna()).any().any():
        raise RuntimeError("A stock with an observed entry lacks an exit close")
    equity_at_close = weights.mul(open_to_close.where(active, 1.0)).sum(axis=1)

    # Each swap sells one asset and buys another, with both costs funded by sale proceeds.
    cost_rate = cost_bps / 10_000
    swap = (1 - cost_rate) / (1 + cost_rate)
    net_value = cash_overnight * ((1 - invested) * cash_intraday + equity_at_close * swap**2)
    turnover = cash_overnight * (invested * (1 + swap) + equity_at_close * swap * (1 + swap))
    stock_round_trips = active.sum(axis=1)
    trades = 2 * stock_round_trips + 2 * invested.gt(0)
    return pd.DataFrame({"returns": net_value - 1, "turnover": turnover, "trades": trades})


def buy_and_hold(open_price, close_price, cost_bps):
    returns = close_price.pct_change(fill_method=None)
    entry = 1 / (1 + cost_bps / 10_000)
    returns.iloc[0] = close_price.iloc[0] / open_price.iloc[0] * entry - 1
    book = pd.DataFrame({"returns": returns, "turnover": 0.0, "trades": 0})
    book.iloc[0, book.columns.get_loc("turnover")] = entry
    book.iloc[0, book.columns.get_loc("trades")] = 1
    return book


def annualized_sharpe(returns):
    std = returns.std(ddof=1)
    return float(returns.mean() / std * np.sqrt(SESSIONS_PER_YEAR)) if std > 0 else None


def metrics(book, cash_returns):
    returns = book["returns"]
    if returns.empty or not np.isfinite(returns).all():
        raise RuntimeError("Metric calculation requires a nonempty period of finite returns")
    wealth = (1 + returns).cumprod()
    std = returns.std(ddof=1)
    return {
        "cagr": float(wealth.iloc[-1] ** (SESSIONS_PER_YEAR / len(wealth)) - 1),
        "ann_vol": float(std * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": annualized_sharpe(returns),
        "sharpe_excess": annualized_sharpe(returns - cash_returns),
        "max_drawdown": float((wealth / wealth.cummax().clip(lower=1) - 1).min()),
        "total_return": float(wealth.iloc[-1] - 1),
        "n_trades": int(book["trades"].sum()),
        "turnover": float(book["turnover"].mean() * SESSIONS_PER_YEAR),
    }


def matched_gap_stats(gaps, after_filing, all_gaps, stock_gross, bins):
    """Descriptive gross stock returns standardized to the filing cohort's gap sizes."""
    observations = pd.DataFrame({
        "gap": gaps.where(all_gaps).stack(),
        "filing": after_filing.where(all_gaps).stack(),
        "return": stock_gross.where(all_gaps).stack() - 1,
    }).dropna()
    observations.index = observations.index.set_names(["session", "ticker"])
    observations["bucket"] = pd.cut(observations["gap"], bins)
    counts = observations.groupby(["ticker", "bucket", "filing"], observed=True)["return"].agg(["mean", "count"])
    if set(counts.index.get_level_values("filing")) != {False, True}:
        return {"matched_filing_trades": 0, "filing_gross_mean": None, "matched_no_filing_gross_mean": None}
    groups = counts.unstack("filing").dropna()
    matched = int(groups["count"][True].sum())
    return {
        "matched_filing_trades": matched,
        "filing_gross_mean": float(np.average(groups["mean"][True], weights=groups["count"][True])) if matched else None,
        "matched_no_filing_gross_mean": float(np.average(groups["mean"][False], weights=groups["count"][True])) if matched else None,
    }


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def percentage(value):
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.2%}"


def main():
    with (ROOT / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    in_start, in_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    out_start, out_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not (pd.Timestamp("2016-01-01") <= in_start <= in_end < out_start <= out_end <= pd.Timestamp("2025-12-31")):
        raise ValueError("Use an ordered IS/OOS split within the 2016–2025 exchange calendar")
    if out_start != in_end + DAY:
        raise ValueError("The IS and OOS date ranges must be contiguous")
    periods = {"is": slice(in_start, in_end), "oos": slice(out_start, out_end), "full": slice(in_start, out_end)}
    universe = config["universe"]
    stocks = list(universe)
    cash, benchmark = config["cash_ticker"], config["benchmark_ticker"]
    if len(set(stocks + [cash, benchmark])) != len(stocks) + 2:
        raise ValueError("Stock, cash and benchmark symbols must be distinct")
    min_gap = float(config["minimum_gap"])
    max_weight = float(config["max_stock_weight"])
    bins = [min_gap, *map(float, config["gap_match_cutoffs"]), np.inf]
    if min_gap <= 0 or not 0 < max_weight <= 1 or not all(lo < hi for lo, hi in zip(bins, bins[1:])):
        raise ValueError("Require positive gap, increasing gap cutoffs and cap in (0, 1]")
    commission = float(config["commission_bps"])
    slippage = float(config["slippage_bps"])
    primary_cost = commission + slippage
    fees = sorted({primary_cost, *map(float, config["cost_sensitivity_bps"])})
    if not all(0 <= fee < 10_000 for fee in [commission, slippage, *fees]):
        raise ValueError("Commission, slippage and total costs must be in [0, 10000) bps")

    open_price, close_price, prices_fetched_on = load_prices(
        stocks + [cash, benchmark], in_start - 14 * DAY, out_end,
    )
    if open_price[[cash, benchmark]].isna().any().any() or close_price[[cash, benchmark]].isna().any().any():
        raise RuntimeError("Missing cash or benchmark quote on an observed exchange session")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"DATA_FETCHED_AT: {prices_fetched_on} (yfinance); {SNAPSHOT_FETCHED_ON} (SEC EDGAR snapshot)")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True; split- and dividend-adjusted Open and Close")
    events = load_events(open_price.index, universe)

    signal_gap = open_price[stocks] / close_price[stocks].shift(1) - 1
    # A signal formed at today's close is eligible only at tomorrow's open.
    lagged_gap = signal_gap.shift(1).loc[periods["full"]]
    lagged_event = events.shift(1, fill_value=False).loc[periods["full"]]
    trade_open = open_price[stocks].loc[periods["full"]]
    stock_gross = close_price[stocks].loc[periods["full"]] / trade_open
    cash_overnight = (open_price[cash] / close_price[cash].shift(1)).loc[periods["full"]]
    cash_intraday = (close_price[cash] / open_price[cash]).loc[periods["full"]]
    cash_book = pd.DataFrame({
        "returns": cash_overnight * cash_intraday - 1,
        "turnover": 0.0,
        "trades": 0,
    })
    all_gaps = lagged_gap.gt(min_gap)
    signals = {
        "Item 2.02 gap": all_gaps & lagged_event,
        "No Item 2.02 gap": all_gaps & ~lagged_event,
        "All gaps": all_gaps,
    }
    weights = {
        label: selected.div(selected.sum(axis=1).clip(lower=1), axis=0)
        .clip(upper=max_weight).where(trade_open.notna(), 0)
        for label, selected in signals.items()
    }

    report = {}
    rows = []
    for cost in fees:
        books = {
            label: daily_stock_book(weight, stock_gross, cash_overnight, cash_intraday, cost)
            for label, weight in weights.items()
        }
        books[f"{benchmark} buy & hold"] = buy_and_hold(
            open_price[benchmark].loc[periods["full"]], close_price[benchmark].loc[periods["full"]], cost,
        )
        if cost == primary_cost:
            books[f"{cash} cash hold"] = cash_book
        report[cost] = {
            label: {
                period: metrics(book.loc[dates], cash_book["returns"].loc[dates])
                for period, dates in periods.items()
            }
            for label, book in books.items()
        }
        for label, split in report[cost].items():
            for period, values in split.items():
                rows.append({"Portfolio": label, "Bps": f"{cost:g}", "Period": period.upper(), **values})

    table = pd.DataFrame(rows, columns=[
        "Portfolio", "Bps", "Period", "cagr", "ann_vol", "sharpe_excess", "sharpe",
        "max_drawdown", "total_return", "n_trades", "turnover",
    ])
    formatters = {key: percentage for key in ("cagr", "ann_vol", "max_drawdown", "total_return", "turnover")}
    formatters.update({key: "{:.2f}".format for key in ("sharpe_excess", "sharpe")})
    print(f"\nPrimary cost per leg: {commission:g} bp commission + {slippage:g} bp slippage")
    print(f"Portfolio metrics (annualized turnover; sharpe_excess vs. {cash}; sharpe vs. zero):")
    print(table.to_string(index=False, formatters=formatters, na_rep="n/a"))

    stock_activity = weights["Item 2.02 gap"].gt(0).sum(axis=1)
    for period, dates in periods.items():
        activity = stock_activity.loc[dates]
        print(
            f"STOCK_ACTIVITY {period.upper()}: {int(activity.sum())} stock round trips "
            f"across {int(activity.gt(0).sum())} active sessions of {len(activity)}"
        )

    matching = {
        label: matched_gap_stats(
            lagged_gap.loc[dates], lagged_event.loc[dates],
            (all_gaps & trade_open.notna()).loc[dates], stock_gross.loc[dates], bins,
        )
        for label, dates in periods.items()
    }
    print("\nGross next-session stock returns matched by ticker and gap-size bin:")
    matching_format = {key: percentage for key in ("filing_gross_mean", "matched_no_filing_gross_mean")}
    print(pd.DataFrame.from_dict(matching, orient="index").to_string(formatters=matching_format, na_rep="n/a"))

    net = report[primary_cost]
    benchmark_splits = net[f"{benchmark} buy & hold"]
    start, end = all_gaps.index[[0, -1]]
    result = {
        "strategy": config["strategy"],
        "data_range": f"{start.date()} to {end.date()}",
        **net["Item 2.02 gap"],
        "benchmark": {
            "ticker": benchmark,
            **benchmark_splits["full"],
            **benchmark_splits,
            "cash": {"ticker": cash, **net[f"{cash} cash hold"]},
            "no_item_202": net["No Item 2.02 gap"],
            "all_gaps": net["All gaps"],
            "gap_size_matched_diagnostic": matching,
        },
    }
    print("RESULTS_JSON: " + json.dumps(json_value(result), allow_nan=False))


if __name__ == "__main__":
    main()
