"""156_cta_replication_trend_factors
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/156_cta_replication_trend_factors.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit da9754ee51e8d74f175ab50e14ef28c47c764ad1 on 2026-11-27 12:03 Europe/Oslo, exit 0, output sha256 b6a7eb04669e390ba5ffebfc2926ca842b5217d85244a8adeaf29c866826c0fd
Data source: Yahoo Finance daily USD ETF closes via yfinance or its local adjusted-close CSV cache. Strategy universe: SPY, QQQ, EFA, EEM, SHY, IEF, FXE, FXY, GLD, DBC. CTA proxy: DBMF. Benchmark: SPY.
Data fetched: ETF_PANEL=2026-11-27T11:03:43+00:00; BIL=2026-11-27T11:03:43+00:00
Sample period: Configured warm-up data request starts 2016-01-01. The reported evaluation spans 2020-01-02 through 2026-10-30, inclusive. DBMF enters the filter when a finite proxy return and finite factor observations are available.   data_range (from output): 2020-01-02 to 2026-10-30
IS/OOS split: IS 2020-01-02 through 2025-07-17 and OOS 2025-07-18 through 2026-10-30, both inclusive. The filter continues to update daily in OOS with the same settings. The strategy and benchmark are simulated across the boundary and sliced for reporting.
Costs and slippage: 3 bp one way per dollar traded, deducted on the rebalance date on absolute trades against marked holdings, with targets sized on equity after fees. Short borrow costs 50 bp per year on beginning-of-session short notional, charged at 50/252 bp per session. The SPY benchmark uses the same transaction fee. Slippage is not modelled.
Signal lag: A target computed after session t closes is filled in the simulation at session t+1's close. Existing holdings earn the return through that close before the rebalance. The new holdings first earn the close-to-close return ending at t+2.
Price adjustment: Yahoo Finance auto_adjust=True, distributions and splits, USD adjusted closes
Survivorship bias: The README states that the backtest uses retrospectively selected surviving ETFs and is subject to fund-selection and survivorship bias. The implementation uses the fixed ticker list from the configuration.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.006743753491951665, ann_vol=0.05253237185473903, sharpe=0.15425031781680637, sharpe_excess=-0.3204381619355861, max_drawdown=-0.13677086998330523, total_return=0.037823949785030786, turnover=21.39409111499946, n_trades=13920
    oos: cagr=0.06790213637441678, ann_vol=0.05253345844707779, sharpe=1.2769532588371535, sharpe_excess=0.5642744459566502, max_drawdown=-0.03881752605220368, total_return=0.08842001583873249, turnover=19.31567773580487, n_trades=3250
    full: cagr=0.01804499452049435, ann_vol=0.05253744365592874, sharpe=0.36672920387125707, sharpe_excess=-0.15299490729995735, max_drawdown=-0.13677086998330523, total_return=0.12958835986284156, turnover=21.000681477120455, n_trades=17170
    benchmark: cagr=0.08521333774232698, ann_vol=0.1210396841764258, sharpe=0.7363844730891789, sharpe_excess=0.5099716300556497, max_drawdown=-0.198872424938626, total_return=0.7457498253390369, turnover=0.14126801308117914, n_trades=1
"""
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
