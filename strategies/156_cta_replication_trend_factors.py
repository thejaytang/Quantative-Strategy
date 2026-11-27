# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 156_cta_replication_trend_factors\n\ndata_start: 2016-01-01\nin_sample:\n  start: 2020-01-02\n  end: 2025-07-17\nout_of_sample:\n  start: 2025-07-18\n  end: 2026-10-30\n\nuniverse:\n  equity: [SPY, QQQ, EFA, EEM]\n  treasury: [SHY, IEF]\n  currency: [FXE, FXY]\n  commodity: [GLD, DBC]\ncta_proxy: DBMF\ncta_proxy_first_trade: 2019-05-08\nsecondary_benchmark: SPY\ncash_proxy: BIL\n\ntrend:\n  short_windows: [10, 20, 40, 60]\n  long_window: 500\n\nfilter:\n  initial_loading_sd: 0.25\n  daily_loading_step_sd: 0.004\n  observation_noise_bps: 60\n\nportfolio:\n  max_gross_exposure: 1.00\n  max_asset_exposure: 0.40\n  cash_buffer: 0.05\n  transaction_bps: 3.0  # One way, per dollar traded\n  slippage_bps: 2.0  # One way, per dollar traded\n  annual_short_borrow_bps: 50.0\n  sessions_per_year: 252\n'
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
"""Forward-only Bayesian replication of a managed-futures ETF."""

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
    USMartinLutherKingJr, USMemorialDay, USPresidentsDay, USThanksgivingDay,
    nearest_workday, sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay
from scipy.special import ndtr
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
BPS = 10_000


class ExchangeHolidays(AbstractHolidayCalendar):
    # US equity full-day holidays for the configured sample. Early closes trade.
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
        Holiday("George H. W. Bush national day of mourning", year=2018, month=12, day=5),
        Holiday("Jimmy Carter national day of mourning", year=2025, month=1, day=9),
    ]


def validate_prices(prices, sessions, tickers, first_trade):
    prices = prices.sort_index().loc[sessions[0]:sessions[-1]].reindex(columns=tickers)
    if prices.index.has_duplicates:
        raise ValueError("Yahoo prices contain duplicate dates")
    missing = sessions.difference(prices.index)
    unexpected = prices.index.difference(sessions)
    if len(missing) or len(unexpected):
        raise ValueError(f"Yahoo calendar mismatch (missing {missing[:5].date}, "
                         f"unexpected {unexpected[:5].date})")
    for ticker in tickers:
        series = prices[ticker]
        listed = sessions >= first_trade.get(ticker, sessions[0])
        active = series.loc[listed]
        if series.loc[~listed].notna().any():
            raise ValueError(f"Unexpected {ticker} prices before its first trading session")
        if active.empty or not (np.isfinite(active) & active.gt(0)).all():
            raise ValueError(f"Missing or invalid Yahoo closes for {ticker}")
    return prices


def download_prices(tickers, sessions, first_trade):
    for attempt in range(2):
        try:
            download = yf.download(
                tickers,
                start=sessions[0].date().isoformat(),
                end=(sessions[-1] + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=True,
                progress=False,
                timeout=12,
            )
            prices = validate_prices(download["Close"], sessions, tickers, first_trade)
            return prices, datetime.now(timezone.utc).isoformat(timespec="seconds")
        except (KeyError, ValueError) as exc:
            if attempt == 1:
                raise RuntimeError("Unable to load a complete public Yahoo Finance price panel") from exc
            print(f"Retrying Yahoo Finance download ({exc})")


def load_market_prices(tickers, sessions, first_trade, start, end):
    cache_file = CACHE / f"yahoo_adjusted_close_{start.date()}_{end.date()}.csv"
    if cache_file.exists():
        try:
            cached = pd.read_csv(cache_file, index_col="date", parse_dates=["date"])
            prices = validate_prices(cached, sessions, tickers, first_trade)
            saved_at = datetime.fromtimestamp(cache_file.stat().st_mtime, timezone.utc)
            return prices, f"{saved_at.isoformat(timespec='seconds')} (cache mtime)"
        except ValueError as exc:
            print(f"Downloading the Yahoo panel ({exc})")
    return download_prices(tickers, sessions, first_trade)


def trend_score(log_price, squared_log_returns, window):
    high = log_price.rolling(window).max()
    low = log_price.rolling(window).min()
    scale = np.sqrt(squared_log_returns.rolling(window).sum() * window / (window - 1))
    scale = scale.mask(scale == 0, np.inf)
    return ndtr((log_price - low) / scale) - ndtr((high - log_price) / scale)


def target_weights(close, proxy_returns, trend, model, portfolio):
    log_price = np.log(close)
    squared_log_returns = log_price.diff().pow(2)
    short = sum(trend_score(log_price, squared_log_returns, n) for n in trend["short_windows"])
    short /= len(trend["short_windows"])
    long = trend_score(log_price, squared_log_returns, trend["long_window"])

    # The straddle scores lie between -0.5 and +0.5. Rescale both trend sleeves
    # to the same unit position range as the raw market sleeve.
    positions = np.stack([2 * short, 2 * long, np.ones(close.shape)], axis=2)
    market_returns = close.pct_change(fill_method=None).to_numpy()
    observations = positions[:-2] * market_returns[2:, :, None]

    n_assets = len(close.columns)
    mean = np.zeros(3 * n_assets)
    identity = np.eye(mean.size)
    covariance = identity * model["initial_loading_sd"] ** 2
    process_noise = identity * model["daily_loading_step_sd"] ** 2
    observation_variance = (model["observation_noise_bps"] / BPS) ** 2
    asset_cap = portfolio["max_asset_exposure"]
    gross_cap = portfolio["max_gross_exposure"]
    targets = np.full(close.shape, np.nan)

    for day, (factor_returns, proxy_return) in enumerate(
        zip(observations, proxy_returns.to_numpy()[2:]), start=2
    ):
        design = factor_returns.ravel()
        if not np.isfinite(proxy_return) or not np.isfinite(design).all():
            continue

        # One online posterior update. No date after this close enters the state.
        covariance += process_noise
        cross_covariance = covariance @ design
        innovation_variance = design @ cross_covariance + observation_variance
        mean += cross_covariance * (proxy_return - design @ mean) / innovation_variance
        covariance -= np.outer(cross_covariance, cross_covariance) / innovation_variance

        weight = (positions[day] * mean.reshape(n_assets, 3)).sum(axis=1)
        weight = np.clip(weight, -asset_cap, asset_cap)
        targets[day] = weight / max(1.0, np.abs(weight).sum() / gross_cap)

    return pd.DataFrame(targets, index=close.index, columns=close.columns)


def backtest(close, close_targets, portfolio):
    fee = (portfolio["transaction_bps"] + portfolio["slippage_bps"]) / BPS
    borrow = portfolio["annual_short_borrow_bps"] / BPS / portfolio["sessions_per_year"]
    reserve = portfolio["cash_buffer"]
    if not 0 < reserve < 1 or not 0 <= fee < reserve or not 0 <= borrow < reserve:
        raise ValueError("Invalid cash buffer, execution cost or daily borrowing cost")
    sizing_dates, fill_dates = close.index[:-1], close.index[1:]
    if (not close_targets.columns.equals(close.columns) or close_targets.index.has_duplicates
            or not close_targets.index.isin(sizing_dates).all()
            or not np.isfinite(close_targets.to_numpy()).all()
            or (close_targets.abs().sum(axis=1) > 1 + 1e-12).any()):
        raise ValueError("Targets must be finite, aligned to sizing dates, and at most 100% gross")

    daily_results = []
    positions = np.zeros(len(close.columns))
    previous_value = cash = 1.0
    prices = close.to_numpy()
    targets = close_targets.reindex(sizing_dates).to_numpy()
    rebalance = sizing_dates.isin(close_targets.index)
    for day, fill_date in enumerate(fill_dates):
        sizing_close = prices[day]
        next_positions = positions
        if rebalance[day]:
            # Adjusted-share units, fixed after the preceding close.
            next_positions = previous_value * (1 - reserve) * targets[day] / sizing_close
        order = next_positions - positions

        fill_close = prices[day + 1]
        traded = np.abs(order) @ fill_close
        borrow_cost = borrow * (np.maximum(-positions, 0) @ sizing_close)
        cash -= order @ fill_close + fee * traded + borrow_cost
        value = cash + next_positions @ fill_close
        # Keep all short proceeds and an equal amount of portfolio funds as collateral.
        free_cash = value - np.abs(next_positions) @ fill_close
        if value <= 0 or cash < 0 or free_cash < 0:
            raise ValueError(f"Insufficient cash at the {fill_date.date()} close with fixed share quantities")

        daily_results.append((value / previous_value - 1, traded / previous_value,
                              np.count_nonzero(order), free_cash / value))
        positions = next_positions
        previous_value = value

    return pd.DataFrame(daily_results, index=fill_dates,
                        columns=["return", "turnover", "n_trades", "free_cash_fraction"])


def metrics(performance, proxy_returns, cash_returns, annual_sessions):
    daily = performance["return"]
    growth = (1 + daily).cumprod().to_numpy()
    peaks = np.maximum.accumulate(np.r_[1.0, growth])[1:]
    vol = daily.std(ddof=1) * np.sqrt(annual_sessions)
    excess = daily - cash_returns
    excess_vol = excess.std(ddof=1) * np.sqrt(annual_sessions)
    values = {
        "cagr": growth[-1] ** (annual_sessions / len(daily)) - 1,
        "ann_vol": vol,
        "sharpe": daily.mean() * annual_sessions / vol if vol > 0 else np.nan,
        "sharpe_excess": excess.mean() * annual_sessions / excess_vol if excess_vol > 0 else np.nan,
        "max_drawdown": np.min(growth / peaks - 1),
        "total_return": growth[-1] - 1,
        "turnover": performance["turnover"].mean() * annual_sessions,
        "cta_proxy_corr": daily.corr(proxy_returns) if vol > 0 and proxy_returns.std() > 0 else np.nan,
    }
    return {
        **{key: float(value) if np.isfinite(value) else None for key, value in values.items()},
        "n_trades": int(performance["n_trades"].sum()),
    }


def print_metrics(rows):
    columns = (
        ("cagr", "CAGR", ".2%", 10), ("ann_vol", "Ann. vol", ".2%", 10),
        ("sharpe", "Sharpe", ".2f", 8), ("sharpe_excess", "Excess SR", ".2f", 10),
        ("max_drawdown", "Max DD", ".2%", 10), ("total_return", "Total ret.", ".2%", 11),
        ("turnover", "Ann. turn.", ".2%", 11), ("n_trades", "Trades", "d", 7),
        ("cta_proxy_corr", "CTA corr.", ".2f", 10),
    )
    print(f"{'Portfolio / period':<18} " + " ".join(title.rjust(width) for _, title, _, width in columns))
    for label, row in rows.items():
        values = [format(row[key], spec) if row[key] is not None else "n/a"
                  for key, _, spec, _ in columns]
        print(f"{label:<18} " + " ".join(value.rjust(column[3]) for value, column in zip(values, columns)))


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    start = pd.Timestamp(config["data_start"])
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("IS and OOS must be ordered and disjoint, with earlier warm-up data")

    assets = list(dict.fromkeys(ticker for group in config["universe"].values() for ticker in group))
    proxy = config["cta_proxy"]
    secondary = config["secondary_benchmark"]
    cash_proxy = config["cash_proxy"]
    portfolio = config["portfolio"]
    if proxy in assets or proxy == secondary or cash_proxy in {*assets, proxy, secondary}:
        raise ValueError("Choose distinct ETF roles for the CTA target, comparison and cash proxy")
    if not 0 < portfolio["max_asset_exposure"] <= portfolio["max_gross_exposure"] <= 1:
        raise ValueError("Exposure limits must be positive and at most 100% gross")
    tickers = list(dict.fromkeys([*assets, proxy, secondary]))
    sessions = pd.date_range(start, oos_end, freq=CustomBusinessDay(calendar=ExchangeHolidays()))
    first_fill = sessions.searchsorted(is_start)
    if not 0 < first_fill < len(sessions) or sessions[first_fill] > is_end or sessions[-1] < oos_start:
        raise ValueError("Both periods need trading sessions and an earlier sizing close")
    evaluation_prices = sessions[first_fill - 1:]
    first_trade = {proxy: pd.Timestamp(config["cta_proxy_first_trade"])}
    with TemporaryDirectory(prefix="yfinance_", dir=ROOT) as yf_cache:
        yf.set_tz_cache_location(yf_cache)
        prices, market_fetched_at = load_market_prices(tickers, sessions, first_trade, start, oos_end)
        cash_prices, cash_fetched_at = download_prices([cash_proxy], evaluation_prices, {})
    cash_returns = cash_prices[cash_proxy].pct_change(fill_method=None)
    proxy_returns = prices[proxy].pct_change(fill_method=None)
    targets = target_weights(prices[assets], proxy_returns, config["trend"], config["filter"], portfolio)

    # Include the prior session for sizing the first fill. No target uses the last close.
    strategy = backtest(prices.loc[evaluation_prices, assets], targets.loc[evaluation_prices[:-1]], portfolio)
    comparisons = {}
    for ticker in (proxy, secondary):
        initial_target = pd.DataFrame(1.0, index=evaluation_prices[:1], columns=[ticker])
        comparisons[ticker] = backtest(prices.loc[evaluation_prices, [ticker]], initial_target, portfolio)

    periods = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    reports = {
        label: {period: metrics(daily.loc[a:b], proxy_returns.loc[a:b], cash_returns.loc[a:b],
                                portfolio["sessions_per_year"])
                for period, (a, b) in periods.items()}
        for label, daily in {"Strategy": strategy, **comparisons}.items()
    }

    data_range = f"{strategy.index[0].date()} to {strategy.index[-1].date()}"
    print(f"{config['strategy']} | {data_range} | Benchmark {proxy} buy and hold | Excess vs {cash_proxy}")
    for period, (a, b) in periods.items():
        dates = strategy.loc[a:b].index
        print(f"{period.upper():<4} {dates[0].date()} to {dates[-1].date()} ({len(dates)} sessions)")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"DATA_FETCHED_AT: ETF_PANEL={market_fetched_at}; {cash_proxy}={cash_fetched_at}")
    print("PRICE_ADJUSTMENT: Yahoo Finance auto_adjust=True, distributions and splits, USD adjusted closes")
    print(f"MIN_STRATEGY_FREE_CASH: {strategy['free_cash_fraction'].min():.2%}")
    print_metrics({f"{label} {period.upper()}": values
                   for label, report in reports.items() for period, values in report.items()})
    results = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **reports["Strategy"],
        "benchmark": {
            "name": f"{proxy} buy and hold",
            **reports[proxy]["full"],
            **reports[proxy],
        },
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
