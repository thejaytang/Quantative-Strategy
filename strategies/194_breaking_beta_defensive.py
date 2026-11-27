# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 194_breaking_beta_defensive\ncategory: factor\n\ndata:\n  warmup_start: 2012-01-01\n  cache_dir: data_cache\n  market: SPY\n  defensive_etfs: [USMV, SPLV]\n  breadth_stocks:\n    - AAPL\n    - MSFT\n    - AMZN\n    - GOOGL\n    - META\n    - NVDA\n    - JPM\n    - BAC\n    - GS\n    - MS\n    - XOM\n    - CVX\n    - COP\n    - SLB\n    - WMT\n    - COST\n    - PG\n    - KO\n    - PEP\n    - JNJ\n    - PFE\n    - MRK\n    - HD\n    - LOW\n\nin_sample:\n  start: 2013-01-01\n  end: 2021-12-31\nout_of_sample:\n  start: 2022-01-01\n  end: 2026-11-24\n\nsignal:\n  beta_window_sessions: 126\n  min_eligible_stocks: 16\n  negative_beta_breadth_threshold: 0.15\n  defensive_beta_trigger: 0.75\n  target_beta: 0.85\n  max_spy_weight: 0.75\n\ncosts:\n  bps_per_dollar_traded: 5\n  sensitivity_bps: [0, 15]\n\nannualization_sessions: 252\n'
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
"""Monthly breadth-conditioned beta restoration for a U.S. defensive ETF sleeve."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import requests
import yaml


ROOT = Path(__file__).resolve().parent
TOLERANCE = 1e-12


def download_prices(symbol, start, end):
    """Read the public Yahoo Finance daily chart feed used by yfinance."""
    params = {
        "period1": int(start.tz_localize("UTC").timestamp()),
        "period2": int((end + pd.Timedelta(days=1)).tz_localize("UTC").timestamp()),
        "interval": "1d",
        "events": "history",
    }
    failures = []
    with requests.Session() as session:
        session.trust_env = False
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
            url = f"https://{host}/v8/finance/chart/{symbol}"
            try:
                response = session.get(url, params=params, timeout=(7, 20))
                response.raise_for_status()
                chart = response.json()["chart"]["result"][0]
                meta = chart["meta"]
                if meta["dataGranularity"] != "1d" or meta["currency"] != "USD":
                    raise ValueError("expected daily USD data")
                dates = (
                    pd.to_datetime(chart["timestamp"], unit="s", utc=True)
                    .tz_convert(meta["exchangeTimezoneName"])
                    .tz_localize(None)
                    .normalize()
                )
                indicators = chart["indicators"]
                raw = indicators["quote"][0]
                close = np.asarray(raw["close"], dtype=float)
                close[close <= 0] = np.nan
                adjusted_close = np.asarray(indicators["adjclose"][0]["adjclose"], dtype=float)
                adjusted_close[adjusted_close <= 0] = np.nan
                adjusted_open = np.asarray(raw["open"], dtype=float) * (adjusted_close / close)
                prices = pd.DataFrame(
                    {"open": adjusted_open, "close": adjusted_close}, index=dates
                ).sort_index().loc[start:end]
                prices.index.name = "date"
                prices = prices.where(np.isfinite(prices) & prices.gt(0)).dropna(how="all")
                if prices.empty or not prices.index.is_unique:
                    raise ValueError("empty or duplicated daily data")
                return prices
            except (requests.RequestException, KeyError, TypeError, IndexError, ValueError) as error:
                failures.append(f"{host}: {error}")
    raise RuntimeError(f"Could not download {symbol}. " + " | ".join(failures))


def load_prices(symbols, start, end, cache_dir):
    cache_dir.mkdir(exist_ok=True, parents=True)

    def load(symbol):
        path = cache_dir / f"yahoo_daily_{symbol}_{start.date()}_{end.date()}.csv"
        if path.exists():
            return pd.read_csv(path, index_col="date", parse_dates=["date"])
        prices = download_prices(symbol, start, end)
        with NamedTemporaryFile(mode="w", dir=cache_dir, delete=False) as cache_file:
            temp_path = Path(cache_file.name)
        try:
            prices.to_csv(temp_path)
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
        return prices

    with ThreadPoolExecutor(max_workers=6) as pool:
        frames = list(pool.map(load, symbols))
    return dict(zip(symbols, frames))


def month_open_decisions(closes, market, defensive_etfs, stocks, settings):
    returns = closes.pct_change(fill_method=None)
    market_returns = returns[market]
    blend = returns[defensive_etfs].mean(axis=1, skipna=False).rename("blend")
    candidates = pd.concat([returns[stocks], blend], axis=1)
    window = settings["beta_window_sessions"]
    market_variance = market_returns.rolling(window).var()
    betas = candidates.rolling(window).cov(market_returns).div(
        market_variance.where(market_variance.gt(TOLERANCE)), axis=0
    )
    stock_betas = betas[stocks]
    eligible = stock_betas.notna().sum(axis=1)
    breadth = stock_betas.lt(0).sum(axis=1).div(
        eligible.where(eligible.ge(settings["min_eligible_stocks"]))
    )
    beta = betas["blend"]
    low_beta = beta.lt(settings["defensive_beta_trigger"])
    trigger = low_beta & breadth.ge(settings["negative_beta_breadth_threshold"])
    restoration = (
        (settings["target_beta"] - beta) / (1 - beta.where(low_beta))
    ).clip(lower=0, upper=settings["max_spy_weight"])
    signals = pd.DataFrame(
        {
            "beta": beta,
            "breadth": breadth,
            "strategy": restoration.where(trigger, 0.0),
            "beta_only": restoration.where(low_beta, 0.0),
        }
    )
    month = closes.index.to_period("M")
    # The first new-month open consumes the previous session's completed signal.
    return signals.shift(1).loc[~month.duplicated()].iloc[1:]


def fill_at_open(holdings, target, cash, fee_rate):
    """Find post-fee NAV such that fees equal rate times actual traded notional."""
    nav = holdings.sum() + cash
    if fee_rate:
        lower, upper = 0.0, nav
        for _ in range(42):
            value = (lower + upper) / 2
            traded = np.abs(target * value - holdings).sum()
            if value + fee_rate * traded > nav:
                upper = value
            else:
                lower = value
        target_nav = (lower + upper) / 2
    else:
        target_nav = nav
    new_holdings = target * target_nav
    orders = np.abs(new_holdings - holdings) / nav
    return new_holdings, orders.sum(), np.count_nonzero(orders > TOLERANCE)


def simulate(targets, overnight, intraday, fee_rate):
    """Carry actual ETF weights between monthly opening fills and value at closes."""
    holdings = np.zeros(targets.shape[1])
    cash = 1.0
    results = np.zeros((len(targets), 3))
    for row, target in enumerate(targets):
        holdings *= overnight[row]
        if np.isfinite(target[0]):
            holdings, results[row, 1], results[row, 2] = fill_at_open(
                holdings, target, cash, fee_rate
            )
            cash = 0.0
        holdings *= intraday[row]
        nav = holdings.sum() + cash
        results[row, 0] = nav - 1
        holdings /= nav
        cash /= nav
    return results


def metrics(results, market_returns, sessions):
    daily = results["return"]
    wealth = (1 + daily).cumprod()
    annual_vol = daily.std() * np.sqrt(sessions)
    benchmark_variance = market_returns.var()
    values = {
        "cagr": wealth.iloc[-1] ** (sessions / len(daily)) - 1,
        "ann_vol": annual_vol,
        "sharpe": daily.mean() * sessions / annual_vol if annual_vol > 0 else None,
        "max_drawdown": (wealth / wealth.cummax().clip(lower=1) - 1).min(),
        "total_return": wealth.iloc[-1] - 1,
        "tracking_error": (daily - market_returns).std() * np.sqrt(sessions),
        "realized_beta": (
            daily.cov(market_returns) / benchmark_variance if benchmark_variance > 0 else None
        ),
        "turnover": results["turnover"].sum(),
    }
    clean = {key: float(val) if val is not None and np.isfinite(val) else None
             for key, val in values.items()}
    return {**clean, "n_trades": int(results["n_trades"].sum()), "n_sessions": len(daily)}


def print_metrics(runs, data_dates):
    print(f"Actual return dates: {data_dates}")
    print(
        f"{'Portfolio':<17} {'bps':>4}  {'Period':<6} {'CAGR':>8} {'Ann vol':>8} "
        f"{'Sharpe':>7} {'Max DD':>8} {'Total':>9} {'Track err':>9} "
        f"{'Beta':>6} {'Turnover':>9} {'Trades':>7}"
    )
    fields = (
        ("cagr", "8.2%"), ("ann_vol", "8.2%"), ("sharpe", "7.2f"),
        ("max_drawdown", "8.2%"), ("total_return", "9.2%"), ("tracking_error", "9.2%"),
        ("realized_beta", "6.2f"), ("turnover", "9.2f"), ("n_trades", "7d"),
    )
    for bps, portfolios in runs.items():
        for label, periods in portfolios.items():
            for period, values in periods.items():
                rendered = " ".join(
                    format(values[key], style) if values[key] is not None
                    else "n/a".rjust(int(style.split(".")[0].rstrip("d")))
                    for key, style in fields
                )
                print(f"{label:<17} {bps:>4g}  {period:<6} {rendered}")


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data = config["data"]
    market = data["market"]
    defensive_etfs = data["defensive_etfs"]
    stocks = data["breadth_stocks"]
    instruments = [market, *defensive_etfs]
    symbols = [*instruments, *stocks]
    warmup = pd.Timestamp(data["warmup_start"])
    start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not warmup < start <= is_end < oos_start <= end:
        raise ValueError("Expected warm-up, followed by disjoint chronological IS and OOS ranges.")
    if len(set(symbols)) != len(symbols) or len(defensive_etfs) != 2:
        raise ValueError("Need two defensive ETFs and unique tickers across ETF and breadth inputs.")
    settings = config["signal"]
    if not (
        settings["beta_window_sessions"] >= 2
        and 1 <= settings["min_eligible_stocks"] <= len(stocks)
        and 0 <= settings["negative_beta_breadth_threshold"] <= 1
        and settings["defensive_beta_trigger"] <= settings["target_beta"] <= 1
        and settings["defensive_beta_trigger"] < 1
        and 0 <= settings["max_spy_weight"] <= 1
    ):
        raise ValueError("Invalid beta signal settings.")
    primary_bps = config["costs"]["bps_per_dollar_traded"]
    cost_levels = dict.fromkeys([primary_bps, *config["costs"]["sensitivity_bps"]])
    if any(not 0 <= value < 10_000 for value in cost_levels):
        raise ValueError("Fee rates must be finite and in [0, 10000) bps.")

    cache_dir = (ROOT / data["cache_dir"]).resolve()
    if ROOT not in cache_dir.parents:
        raise ValueError("The data cache must be inside the strategy folder.")
    prices = load_prices(symbols, warmup, end, cache_dir)
    # A session present in any traded ETF must be present for all traded ETFs.
    opens = pd.concat({asset: prices[asset]["open"] for asset in instruments}, axis=1)
    closes = pd.concat({asset: prices[asset]["close"] for asset in instruments}, axis=1)
    if (
        not opens.index.is_monotonic_increasing
        or not opens.index.is_unique
        or not opens.index.equals(closes.index)
        or not (np.isfinite(opens).all().all() and np.isfinite(closes).all().all())
        or not (opens.gt(0).all().all() and closes.gt(0).all().all())
    ):
        raise ValueError("Daily adjusted open and close are required for every traded ETF session.")
    if len(pd.bdate_range(opens.index[-1], end)) > 5:
        raise ValueError(f"ETF data end on {opens.index[-1].date()}, before the configured end {end.date()}.")
    stock_closes = pd.concat({stock: prices[stock]["close"] for stock in stocks}, axis=1)
    signal_closes = closes.join(stock_closes.where(np.isfinite(stock_closes) & stock_closes.gt(0)))
    decisions = month_open_decisions(signal_closes, market, defensive_etfs, stocks, settings)

    report_index = closes.loc[start:end].index
    periods = {"is": (start, is_end), "oos": (oos_start, end), "full": (start, end)}
    for label, (first, last) in periods.items():
        if not ((report_index >= first) & (report_index <= last)).any():
            raise ValueError(f"No ETF sessions for {label}.")
    first_open = report_index[0]
    initial = decisions.loc[:first_open].tail(1)
    if initial.empty or pd.isna(initial["beta"].iloc[0]):
        raise ValueError("Insufficient warm-up for a defensive beta estimate at the first fill.")
    schedule = decisions.reindex(report_index)
    schedule.loc[first_open] = initial.iloc[0]

    spy_weights = schedule[["strategy", "beta_only"]].copy()
    spy_weights["defensive_blend"] = 0.0
    spy_weights.loc[schedule["strategy"].isna(), "defensive_blend"] = np.nan
    spy_weights["spy_buy_hold"] = np.nan
    spy_weights.loc[first_open, "spy_buy_hold"] = 1.0
    intraday = (closes / opens).loc[report_index].to_numpy()
    overnight = (opens / closes.shift(1)).loc[report_index].to_numpy()
    overnight[0] = 1.0
    market_returns = pd.Series(overnight[:, 0] * intraday[:, 0] - 1, index=report_index)

    runs = {}
    for bps in cost_levels:
        portfolios = {}
        for label, weights in spy_weights.items():
            sleeve = (1 - weights) / 2
            targets = np.column_stack([weights, sleeve, sleeve])
            daily = pd.DataFrame(
                simulate(targets, overnight, intraday, bps / 10_000),
                index=report_index,
                columns=["return", "turnover", "n_trades"],
            )
            portfolios[label] = {
                period: metrics(
                    daily.loc[first:last], market_returns.loc[first:last], config["annualization_sessions"]
                )
                for period, (first, last) in periods.items()
            }
        runs[bps] = portfolios

    diagnostics = {}
    for period, (first, last) in periods.items():
        fills = schedule.loc[first:last].dropna(subset=["strategy"])
        diagnostics[period] = {
            "scheduled_rebalances": len(fills),
            "breadth_available_rebalances": int(fills["breadth"].notna().sum()),
            "breadth_overlay_rebalances": int(fills["strategy"].gt(0).sum()),
            "beta_only_overlay_rebalances": int(fills["beta_only"].gt(0).sum()),
        }
    data_dates = f"{report_index[0].date()} to {report_index[-1].date()}"
    primary = runs[primary_bps]
    output = {
        "strategy": config["strategy"],
        "data_range": data_dates,
        **primary["strategy"],
        "benchmark": {"name": "defensive_blend", **primary["defensive_blend"]},
        "comparisons": {key: primary[key] for key in ("beta_only", "spy_buy_hold")},
        "cost_bps": primary_bps,
        "cost_sensitivity": {str(bps): run for bps, run in runs.items() if bps != primary_bps},
        "signal_diagnostics": diagnostics,
    }
    print_metrics(runs, data_dates)
    for period, stats in diagnostics.items():
        print(
            f"{period.upper()} overlay rebalances were {stats['breadth_overlay_rebalances']} (breadth rule) "
            f"and {stats['beta_only_overlay_rebalances']} (beta only). Breadth was available at "
            f"{stats['breadth_available_rebalances']}/{stats['scheduled_rebalances']} rebalances."
        )
    print("RESULTS_JSON: " + json.dumps(output, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
