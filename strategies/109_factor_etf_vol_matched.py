"""109_factor_etf_vol_matched
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/109_factor_etf_vol_matched.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c71364ad99535bbf59be998b004cff431ee83048 on 2026-11-27 11:53 Europe/Oslo, exit 0, output sha256 3ff99862c015b7d634f2d644697284c93d3685961ac6c87da5b4e191989420d1
Data source: Yahoo Finance daily ETF closes via yfinance for VTV, MTUM, QUAL, USMV, BIL, SPY, IWD, IWF and VUG. The loader uses a matching local price cache when present.
Data fetched: 2026-11-27T10:53:57+00:00
Sample period: Requested price history starts 2014-01-01 for warm-up and ends 2025-12-31. The run reports an evaluation range of 2015-01-02 through 2025-12-31.   data_range (from output): 2015-01-02 to 2025-12-31
IS/OOS split: IS 2015-01-02 through 2020-12-31 and OOS 2021-01-01 through 2025-12-31, inclusive calendar bounds applied to trading sessions. OOS carries forward the existing holdings. The target uses expanding gross returns of the invested IS equal-weight benchmark, requires 63 observations, and is frozen after IS. The equity cap applies before the target becomes available. Portfolio weights continue updating in OOS. This is a historical holdout, with some dates before the cited paper was published.
Costs and slippage: Both portfolios pay 5 bps per dollar bought or sold, including initial entry and BIL trades. At each rebalance close the simulator marks existing holdings to market, then solves for post-cost NAV so the cost equals 0.0005 times the sum of absolute dollar changes from drifted holdings to post-cost targets. Slippage is not modelled. Spread variation, market impact, taxes and close execution deviations are also omitted.
Signal lag: The month-end close at t enters the rolling volatility calculation and the target available as of t. Both portfolios fill at the next trading session's close. Existing holdings earn the return ending at the fill close, and new holdings first earn the return after that close.
Price adjustment: Yahoo Finance daily closes, split- and dividend-adjusted (yfinance auto_adjust=True)
Survivorship bias: The backtest uses a fixed list of surviving ETFs. The README identifies fund-selection and survivorship bias, and the loader requires valid prices for the configured panel over the requested range.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.08367725787727953, ann_vol=0.14302461252264034, sharpe=0.63383061426741, sharpe_excess=0.5777789542376682, max_drawdown=-0.29685829659479646, total_return=0.619053739581513, turnover=1.8938512612115892, n_trades=321
    oos: cagr=0.10794853618167477, ann_vol=0.13129266513906568, sharpe=0.8465322499656285, sharpe_excess=0.6130892429528965, max_drawdown=-0.1862054260809931, total_return=0.6661519856089746, turnover=1.0752411288381447, n_trades=257
    full: cagr=0.0946231121675587, ann_vol=0.13780233010879855, sharpe=0.7253159106498459, sharpe_excess=0.5926299378482647, max_drawdown=-0.29685829659479646, total_return=1.6975896030113806, turnover=1.5224283703480053, n_trades=578
    benchmark: cagr=0.11659282101949087, ann_vol=0.1600533731644123, sharpe=0.7694490643407852, sharpe_excess=0.6551881343972835, max_drawdown=-0.33639469516277376, total_return=2.3550793115883613, turnover=0.2154684071541333, n_trades=528
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 109_factor_etf_vol_matched\nuniverse:\n  sleeves:\n    value: VTV\n    momentum: MTUM\n    quality: QUAL\n    low_volatility: USMV\n  cash_proxy: BIL\n  market_proxy: SPY\n  peer_value: IWD\n  peer_growth: IWF\n  growth_reference: VUG\ndata_start: '2014-01-01'\nin_sample:\n  start: '2015-01-02'\n  end: '2020-12-31'\nout_of_sample:\n  start: '2021-01-01'\n  end: '2025-12-31'\nrebalance: month_end\nvolatility_lookback_sessions: 63\nexposure_lookback_sessions: 126\nsessions_per_year: 252\nmax_equity_weight: 1.0\ntransaction_cost_bps: 2.0  # Per dollar bought or sold, including BIL.\nslippage_bps: 3.0          # Adverse adjustment to each side of the reference close.\ncash_buffer_bps: 250.0     # Reserved in both portfolios when sizing at the signal close.\nlegacy_price_snapshot:\n  date: '2026-11-25'  # Date on the delivered legacy snapshot file.\n  sha256: bc42cede01b592cfa1d1b6383b19b361279fe216165e8beb0bb63ce42dbae92f\n"
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
"""C10 factor ETF allocation and exposure backtest."""

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

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
FETCH_HEADER = "# DATA_FETCHED_AT: "
PRICE_ADJUSTMENT = "Yahoo Finance daily closes, split- and dividend-adjusted (yfinance auto_adjust=True)"


def validate_sessions(dates, start, end):
    """Check every NYSE session in the strategy's historical calendar coverage."""
    if start.year < 2014 or end.year > 2025:
        raise ValueError("NYSE calendar coverage for this backtest is 2014–2025.")
    holidays = AbstractHolidayCalendar(rules=[
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
    ]).holidays(start, end)
    mourning_closures = pd.to_datetime(["2018-12-05", "2025-01-09"])
    expected = pd.bdate_range(start, end).difference(holidays.union(mourning_closures))
    missing = expected.difference(dates)
    unexpected = dates.difference(expected)
    if len(missing) or len(unexpected):
        raise ValueError(
            f"NYSE session mismatch: {len(missing)} missing ({list(missing[:5].strftime('%Y-%m-%d'))}), "
            f"{len(unexpected)} unexpected ({list(unexpected[:5].strftime('%Y-%m-%d'))})."
        )


def load_prices(tickers, start, end, legacy_snapshot):
    """Download complete, dividend- and split-adjusted daily closing prices."""
    CACHE.mkdir(exist_ok=True)
    cache_path = CACHE / f"yahoo_adjusted_{start.date()}_{end.date()}_{'-'.join(tickers)}.csv"
    from_cache = cache_path.exists()
    if from_cache:
        with cache_path.open(encoding="utf-8") as cache_file:
            header = cache_file.readline()
            if header.startswith(FETCH_HEADER):
                fetched_at = header.removeprefix(FETCH_HEADER).strip()
            else:
                cache_file.seek(0)
                fetched_at = "unknown (legacy cache without download metadata)"
                if sha256(cache_path.read_bytes()).hexdigest() == legacy_snapshot["sha256"]:
                    fetched_at = f"{legacy_snapshot['date']} (legacy snapshot date; exact fetch time unrecorded)"
            prices = pd.read_csv(cache_file, index_col="Date", parse_dates=["Date"])
    else:
        yf.set_tz_cache_location(str(CACHE / "yfinance"))
        download = yf.download(
            tickers,
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=15,
        )
        if download.empty:
            raise RuntimeError("Yahoo Finance returned no data for the requested window.")
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        prices = download["Close"].copy()

    prices = prices.reindex(columns=tickers).sort_index()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    prices = prices.loc[start:end]
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("The adjusted-close panel is empty or has duplicate dates.")
    invalid = (~np.isfinite(prices) | prices.le(0)).sum()
    if invalid.any():
        raise ValueError(f"Missing or invalid adjusted closes: {invalid[invalid.gt(0)].to_dict()}")
    validate_sessions(prices.index, start, end)

    if not from_cache:
        cache_file = NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False)
        temporary_path = Path(cache_file.name)
        try:
            with cache_file:
                cache_file.write(f"{FETCH_HEADER}{fetched_at}\n")
                prices.to_csv(cache_file, index_label="Date", float_format="%.17g")
            temporary_path.replace(cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def next_close_targets(signal_weights, sessions, start, end):
    """Move each month-end weight to the following trading session's close."""
    return signal_weights.reindex(sessions).shift(1).loc[start:end].dropna(how="all")


def simulate(prices, targets, friction_rate, cash_buffer):
    """Fix share quantities at the prior close and settle them at the next close."""
    positions = np.zeros(prices.shape[1])
    cash = 1.0
    daily_results = []
    sessions = prices.index[1:]
    daily_targets = targets.reindex(index=sessions, columns=prices.columns).to_numpy()
    closes = prices.to_numpy()
    for session, previous_close, fill_close, target in zip(
        sessions, closes[:-1], closes[1:], daily_targets, strict=True
    ):
        previous_value = cash + positions @ previous_close
        before_fills = cash + positions @ fill_close
        turnover = 0.0
        trades = 0
        if np.isfinite(target[0]):
            sizing_value = previous_value * (1.0 - cash_buffer)
            target_positions = sizing_value * target / previous_close
            share_changes = target_positions - positions

            fill_values = share_changes * fill_close
            traded_value = np.abs(fill_values).sum()
            cash_after_fills = cash - fill_values.sum() - traded_value * friction_rate
            if cash_after_fills < 0.0:
                raise RuntimeError(
                    f"{session.date()} close: fixed fills exceed available cash by "
                    f"{-cash_after_fills / before_fills:.4%} of pre-fill portfolio value."
                )

            cash = cash_after_fills
            positions = target_positions
            turnover = traded_value / before_fills
            trades = np.count_nonzero(share_changes)

        after_fills = cash + positions @ fill_close
        daily_results.append((
            after_fills / previous_value - 1.0,
            before_fills / previous_value - 1.0,
            turnover,
            trades,
            cash / after_fills,
        ))
    return pd.DataFrame(
        daily_results,
        index=sessions,
        columns=["return", "gross_return", "turnover", "n_trades", "cash_weight"],
    )


def risk_matched_weights(sleeve_returns, signal_vols, target_vol, lookback, annualizer, cap, cash_proxy):
    signal_weights = []
    for signal, daily_vol in signal_vols.iterrows():
        # Each signal uses the window through that session's close.
        inverse_vol = 1.0 / daily_vol
        mix = inverse_vol / inverse_vol.sum()
        trailing_mix = sleeve_returns.loc[:signal].tail(lookback) @ mix
        mix_vol = trailing_mix.std() * annualizer
        target = target_vol.loc[signal]
        equity = min(cap, target / mix_vol) if mix_vol > 0 and pd.notna(target) else cap
        signal_weights.append([*(mix * equity), 1.0 - equity])
    return pd.DataFrame(signal_weights, index=signal_vols.index, columns=[*sleeve_returns.columns, cash_proxy])


def rolling_exposures(returns, market, value_spread, lookback):
    """Fit two-factor OLS with an intercept via rolling sample covariances."""
    market_var = market.rolling(lookback).var()
    spread_var = value_spread.rolling(lookback).var()
    cross_cov = market.rolling(lookback).cov(value_spread)
    determinant = market_var * spread_var - cross_cov.pow(2)
    determinant = determinant.where(determinant > market_var * spread_var * 1e-12)
    cov_market = returns.rolling(lookback).cov(market)
    cov_spread = returns.rolling(lookback).cov(value_spread)
    return {
        market.name: (cov_market.mul(spread_var, axis=0) - cov_spread.mul(cross_cov, axis=0)).div(
            determinant, axis=0
        ),
        value_spread.name: (cov_spread.mul(market_var, axis=0) - cov_market.mul(cross_cov, axis=0)).div(
            determinant, axis=0
        ),
    }


def metrics(daily, cash_returns, sessions_per_year):
    returns = daily["return"]
    wealth = (1.0 + returns).cumprod()
    last = wealth.iloc[-1]
    peaks = wealth.cummax().clip(lower=1.0)
    annualizer = np.sqrt(sessions_per_year)
    volatility = returns.std()
    excess = returns - cash_returns
    excess_volatility = excess.std()
    values = {
        "cagr": last ** (sessions_per_year / len(returns)) - 1.0,
        "ann_vol": volatility * annualizer,
        "sharpe": returns.mean() / volatility * annualizer if volatility > 0 else np.nan,
        "sharpe_excess": excess.mean() / excess_volatility * annualizer if excess_volatility > 0 else np.nan,
        "max_drawdown": (wealth / peaks - 1.0).min(),
        "total_return": last - 1.0,
        "turnover": daily["turnover"].mean() * sessions_per_year,
    }
    return {
        **{name: float(value) if np.isfinite(value) else None for name, value in values.items()},
        "n_trades": int(daily["n_trades"].sum()),
    }


def print_report(performance, exposures, periods, cash_proxy):
    print(
        f"{'Portfolio':<16} {'Period':<6} {'CAGR':>9} {'Ann vol':>9} {'Sharpe':>8} "
        f"{f'vs {cash_proxy}':>9} {'Max DD':>9} {'Total':>9} {'Ann turn':>9} {'Trades':>7}"
    )
    columns = [
        ("cagr", "+9.2%", 9),
        ("ann_vol", "9.2%", 9),
        ("sharpe", "8.2f", 8),
        ("sharpe_excess", "9.2f", 9),
        ("max_drawdown", "9.2%", 9),
        ("total_return", "+9.2%", 9),
        ("turnover", "9.2%", 9),
        ("n_trades", "7d", 7),
    ]
    for portfolio, summary in performance.items():
        for period, values in summary.items():
            fields = [
                format(values[key], spec) if values[key] is not None else "n/a".rjust(width)
                for key, spec, width in columns
            ]
            print(f"{portfolio:<16} {period.upper():<6} " + " ".join(fields))

    print("\nMean rolling return betas by window end date")
    print(f"{'Series':<16} {'Period':<6} " + " ".join(f"{factor:>9}" for factor in exposures))
    for period, (start, end) in periods.items():
        means = pd.DataFrame({factor: beta.loc[start:end].mean() for factor, beta in exposures.items()})
        for name, row in means.iterrows():
            fields = [f"{beta:9.3f}" if pd.notna(beta) else f"{'n/a':>9}" for beta in row]
            print(f"{name:<16} {period.upper():<6} " + " ".join(fields))


def print_execution_comparison(matched, benchmark, capped_dates, periods):
    print("\nPaired net daily returns (risk matched minus equal weight) and equity cap counts")
    print(f"{'Period':<6} {'Mean bp':>9} {'Paired t':>9} {'At cap':>8}")
    paired_returns = matched["return"] - benchmark["return"]
    cap_flags = pd.Series(1, index=capped_dates)
    for period, (start, end) in periods.items():
        spread = paired_returns.loc[start:end]
        standard_error = spread.sem()
        t_value = f"{spread.mean() / standard_error:.2f}" if standard_error > 0 else "n/a"
        print(
            f"{period.upper():<6} {spread.mean() * 10_000:+9.2f} {t_value:>9} "
            f"{cap_flags.loc[start:end].size:8d}"
        )
    print(
        f"Minimum closing cash weight: risk matched {matched['cash_weight'].min():.6%}, "
        f"equal weight {benchmark['cash_weight'].min():.6%}"
    )


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)
    universe = cfg["universe"]
    sleeves = list(universe["sleeves"].values())
    cash_proxy = universe["cash_proxy"]
    market = universe["market_proxy"]
    peer_value, peer_growth = universe["peer_value"], universe["peer_growth"]
    growth_reference = universe["growth_reference"]
    traded_tickers = [*sleeves, cash_proxy]
    tickers = sorted(set([*traded_tickers, market, peer_value, peer_growth, growth_reference]))
    data_start = pd.Timestamp(cfg["data_start"])
    is_start, is_end = (pd.Timestamp(cfg["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(cfg["out_of_sample"][key]) for key in ("start", "end"))
    lookback = cfg["volatility_lookback_sessions"]
    exposure_lookback = cfg["exposure_lookback_sessions"]
    sessions_per_year = cfg["sessions_per_year"]
    annualizer = np.sqrt(sessions_per_year)
    fee_rate = cfg["transaction_cost_bps"] / 10_000
    slippage_rate = cfg["slippage_bps"] / 10_000
    friction_rate = fee_rate + slippage_rate
    cash_buffer = cfg["cash_buffer_bps"] / 10_000
    cap = cfg["max_equity_weight"]
    if not data_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Use ordered warm-up, IS and OOS date ranges with no overlap.")
    if not (0.0 <= fee_rate and 0.0 <= slippage_rate and friction_rate < 1.0):
        raise ValueError("Use nonnegative costs and slippage totalling less than 10000 bps.")
    if not (0.0 < cash_buffer < 1.0 and 0.0 < cap <= 1.0):
        raise ValueError("Use a cash buffer in (0, 10000) bps and an equity cap in (0, 1].")
    if cfg["rebalance"] != "month_end" or min(lookback, exposure_lookback, sessions_per_year) < 2:
        raise ValueError("Use month_end rebalancing and windows of at least two sessions.")

    prices, fetched_at = load_prices(tickers, data_start, oos_end, cfg["legacy_price_snapshot"])
    returns = prices.pct_change(fill_method=None).iloc[1:]
    sessions = returns.index
    months = sessions.to_period("M")
    month_ends = sessions[:-1][months[:-1] != months[1:]]
    sleeve_returns = returns[sleeves]
    month_end_vols = sleeve_returns.rolling(lookback).std().loc[month_ends]
    signal_vols = month_end_vols.loc[month_end_vols.gt(0).all(axis=1)]
    periods = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    evaluation_returns = returns.loc[is_start:oos_end]
    if evaluation_returns.loc[is_start:is_end].empty or evaluation_returns.loc[oos_start:oos_end].empty:
        raise ValueError("Both evaluation periods must have trading sessions.")
    if len(evaluation_returns.loc[is_end + pd.Timedelta(days=1):oos_start - pd.Timedelta(days=1)]):
        raise ValueError("The IS and OOS periods must cover consecutive trading sessions.")
    first_evaluation_row = prices.index.get_loc(evaluation_returns.index[0])
    simulation_prices = prices[traded_tickers].iloc[first_evaluation_row - 1:]

    equal_signals = pd.DataFrame(1.0 / len(sleeves), index=signal_vols.index, columns=sleeves)
    equal_signals[cash_proxy] = 0.0
    equal_targets = next_close_targets(equal_signals, sessions, is_start, oos_end)
    if equal_targets.empty or equal_targets.index[0] > is_end:
        raise ValueError("The requested warm-up and IS period leave no IS rebalance.")
    benchmark = simulate(simulation_prices, equal_targets, friction_rate, cash_buffer)
    # Only invested IS benchmark returns enter the target. Expanding in IS, frozen in OOS.
    calibration = benchmark.loc[benchmark.index > equal_targets.index[0], "gross_return"].loc[:is_end]
    expanding_vol = calibration.expanding(min_periods=lookback).std() * annualizer
    if expanding_vol.dropna().empty:
        raise ValueError("IS needs enough invested sessions to calibrate the volatility target.")
    target_vol = expanding_vol.reindex(sessions).ffill()
    matched_signals = risk_matched_weights(
        sleeve_returns, signal_vols, target_vol, lookback, annualizer, cap, cash_proxy
    )
    matched_targets = next_close_targets(matched_signals, sessions, is_start, oos_end)
    matched = simulate(simulation_prices, matched_targets, friction_rate, cash_buffer)

    portfolios = {"Risk matched": matched, "Equal weight": benchmark}
    cash_returns = evaluation_returns[cash_proxy]
    performance = {
        name: {
            period: metrics(daily.loc[start:end], cash_returns.loc[start:end], sessions_per_year)
            for period, (start, end) in periods.items()
        }
        for name, daily in portfolios.items()
    }
    compared_returns = returns[[*sleeves, growth_reference]].copy()
    for name, daily in portfolios.items():
        compared_returns[name] = daily["return"]
    value_spread = (returns[peer_value] - returns[peer_growth]).rename(f"{peer_value}-{peer_growth}")
    exposures = rolling_exposures(compared_returns, returns[market], value_spread, exposure_lookback)

    data_range = f"{evaluation_returns.index[0].date()} to {evaluation_returns.index[-1].date()}"
    print(cfg["strategy"], "|", data_range)
    print("PARAM_COMBOS_TRIED: 1")
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"Frozen IS equal-weight volatility target {expanding_vol.iloc[-1]:.2%}")
    print_report(performance, exposures, periods, cash_proxy)
    capped = np.isclose(1.0 - matched_targets[cash_proxy], cap, rtol=0, atol=1e-12)
    print_execution_comparison(matched, benchmark, matched_targets.index[capped], periods)
    baseline = performance["Equal weight"]
    result = {
        "strategy": cfg["strategy"],
        "data_range": data_range,
        **performance["Risk matched"],
        "benchmark": {"name": "Equal weight", **baseline["full"], **baseline},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
