"""472_breaking_the_trend_how_to_avoid_cherry_p
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/472_breaking_the_trend_how_to_avoid_cherry_p.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 53c18ef949f91367c58358829370e449e492628c on 2026-11-27 18:41 Europe/Oslo, exit 0, output sha256 15b1aea614825fc494a1945f8b2b5e4014bbb1dd5fda520086e6c7a1dce1afbf
Data source: Free Yahoo Finance daily prices via yfinance for SPY, EFA, IEF, TLT, GLD, DBC, UUP and FXE, with BIL as the cash-return proxy. The run reports a data snapshot fetched at 2026-11-27T07:06:46+00:00.
Data fetched: 2026-11-27T17:41:14+00:00
Sample period: Requested price window 2008-01-01 through 2026-10-31, inclusive. The 2008–2011 history is used for indicator warm-up. Actual backtest sessions run from 2012-01-03 through 2026-10-30.   data_range (from output): {'start': '2012-01-03', 'end': '2026-10-30'}
IS/OOS split: Retrospective chronological split. IS is 2012-01-03 through 2023-12-29, using dates on or before 2023-12-31. OOS is 2024-01-02 through 2026-10-30, using dates after 2023-12-31. Strategy and benchmark holdings carry across the split. A separate retrospective OOS_POST_PUBLICATION run starts from cash and uses the first eligible signal at the 2025-04-16 close after the configured 2025-04-15 paper publication date. Its first fill is at the 2025-04-17 close and it ends on 2026-10-30. The historical parameter lock date is undocumented.
Costs and slippage: 1 bp transaction cost plus 2 bps fixed slippage per side for the strategy and benchmark. The simulator deducts a combined cash charge of 0.0003 times the sum of absolute share changes valued at the fill-day close. The fixed slippage is modeled through this charge.
Signal lag: A target computed from session t's close sets the quantities for a simulated fill at the next U.S. market session's close. Old positions earn the price move through that fill close.
Price adjustment: yfinance auto_adjust=True total-return adjusted close
Survivorship bias: The eight ETFs were fixed retrospectively from surviving funds with long available histories. The strategy and benchmark are exposed to survivorship and asset-selection bias. Historical candidate lists, inclusion and replacement rules, and suspension or delisting settlement paths are undocumented.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.03166483616303939, ann_vol=0.04930766962259695, sharpe=0.6569490977535223, sharpe_excess=0.474916849464706, max_drawdown=-0.08305367075373415, total_return=0.4525837734573914, turnover=4.494150092817482, n_trades=24144
    oos: cagr=0.0853780249641305, ann_vol=0.05956573090371088, sharpe=1.4054404850830453, sharpe_excess=0.6855217013660234, max_drawdown=-0.04431527273879954, total_return=0.2600544687255171, turnover=4.658164866156845, n_trades=5688
    full: cagr=0.041696972443205915, ann_vol=0.05142867446284436, sharpe=0.8201330820111685, sharpe_excess=0.520018172767958, max_drawdown=-0.08305367075373415, total_return=0.8303346749431584, turnover=4.525422418868511, n_trades=29832
    benchmark: cagr=0.057916028528289365, ann_vol=0.071579920707852, sharpe=0.8224416517539389, sharpe_excess=0.6066522874581521, max_drawdown=-0.1587794369645582, total_return=1.300485505808242, turnover=0.061423762172517486, n_trades=8
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 472_breaking_the_trend_how_to_avoid_cherry_p\nuniverse:\n  equity: [SPY, EFA]\n  bonds: [IEF, TLT]\n  commodities: [GLD, DBC]\n  currencies: [UUP, FXE]\ncash_proxy: BIL\ndates:\n  data_start: 2008-01-01\n  start: 2012-01-01\n  is_end: 2023-12-31\n  end: 2026-10-31\n  source_published: 2025-04-15\nexecution:\n  lag_sessions: 1  # Close-to-close signal lag in US market sessions.\n  cost_bps: 1.0\n  slippage_bps: 2.0\n  cash_buffer: 0.10  # Cap planned gross at 90% of prior-close value for both strategy and benchmark.\nmodel:\n  trend_decay_sessions: 112\n  risk_decay_sessions: 40\n  correlation_decay_weeks: 150\n  min_correlation_weeks: 52\n  correlation_shrinkage: 0.10\n  smoothing_sessions: 20\n  annual_vol_target: 0.10\n  max_gross_weight: 1.0\n  sessions_per_year: 252\n'
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
"""Research backtest for a single-EMA cross-asset trend strategy."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
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
CACHE = ROOT / "data_cache"
PRICE_ADJUSTMENT = "yfinance auto_adjust=True total-return adjusted close"


class USMarketCalendar(AbstractHolidayCalendar):
    # Full-day US equity market closures within the backtest calendar's coverage.
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
        Holiday("Hurricane Sandy, first day", year=2012, month=10, day=29),
        Holiday("Hurricane Sandy, second day", year=2012, month=10, day=30),
        Holiday("George H. W. Bush day of mourning", year=2018, month=12, day=5),
        Holiday("Jimmy Carter day of mourning", year=2025, month=1, day=9),
    ]


def load_prices(tickers, start, end):
    end_exclusive = end + pd.Timedelta(days=1)
    request = json.dumps([sorted(tickers), str(start.date()), str(end_exclusive.date()), PRICE_ADJUSTMENT])
    cache_file = CACHE / f"prices_{hashlib.sha256(request.encode()).hexdigest()[:16]}.csv"
    if cache_file.exists():
        with cache_file.open() as source:
            fetched_at = source.readline().removeprefix("# fetched_at=").strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(source, index_col=0, parse_dates=True)
        return check_prices(prices, tickers, start, end), fetched_at

    CACHE.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(CACHE / "yfinance"))
    error = None
    for _ in range(3):
        try:
            downloaded = yf.download(
                tickers,
                start=str(start.date()),
                end=str(end_exclusive.date()),
                interval="1d",
                auto_adjust=True,
                ignore_tz=True,
                progress=False,
                timeout=8,
            )
            prices = check_prices(downloaded["Close"], tickers, start, end)
            break
        except (KeyError, RuntimeError, ValueError, OSError) as exc:
            error = exc
    else:
        raise RuntimeError(f"Unable to download complete yfinance prices: {error}") from error

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as target:
        temp_path = Path(target.name)
        target.write(f"# fetched_at={fetched_at}\n")
        prices.to_csv(target, index_label="date")
    temp_path.replace(cache_file)
    return prices, fetched_at


def check_prices(prices, tickers, start, end):
    prices = prices.reindex(columns=tickers).sort_index()
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.has_duplicates:
        raise ValueError("Expected a unique daily price index")
    prices.index = prices.index.as_unit("ns").rename("date")
    prices.columns.name = None
    prices = prices.loc[start:end]
    if start.year < 2008 or end.year > 2026:
        raise ValueError("US market calendar covers 2008–2026")
    calendar = USMarketCalendar()
    expected = pd.bdate_range(start, end).difference(calendar.holidays(start, end))
    missing = expected.difference(prices.index)
    unexpected = prices.index.difference(expected)
    if len(missing):
        raise ValueError(f"Missing US market sessions: {', '.join(missing.strftime('%Y-%m-%d')[:10])}")
    if len(unexpected):
        raise ValueError(f"Unexpected price dates: {', '.join(unexpected.strftime('%Y-%m-%d')[:10])}")
    invalid = ~np.isfinite(prices) | (prices <= 0)
    if invalid.any().any():
        raise ValueError(f"Missing or invalid prices for {', '.join(invalid.columns[invalid.any()])}")
    return prices


def trend_targets(prices, tickers, cash_proxy, model):
    returns = prices.pct_change(fill_method=None)
    excess = returns[tickers].sub(returns[cash_proxy], axis=0)
    trend_days = model["trend_decay_sessions"]
    risk_days = model["risk_decay_sessions"]
    squared = excess.pow(2)
    signal_vol = np.sqrt(squared.ewm(alpha=1 / trend_days, adjust=False, min_periods=trend_days).mean())
    normalized = excess.div(signal_vol.shift(1).where(signal_vol.shift(1) > 0))
    # In equation (6), the newest observation has weight sqrt(alpha).
    signal = normalized.ewm(alpha=1 / trend_days, adjust=False, min_periods=trend_days).mean()
    signal *= np.sqrt(trend_days)
    risk_vol = np.sqrt(squared.ewm(alpha=1 / risk_days, adjust=False, min_periods=risk_days).mean())

    weekly = prices.resample("W-FRI").last().pct_change(fill_method=None)
    weekly_excess = weekly[tickers].sub(weekly[cash_proxy], axis=0)
    weekly_corr = weekly_excess.ewm(
        alpha=1 / model["correlation_decay_weeks"],
        adjust=False,
        min_periods=model["min_correlation_weeks"],
    ).corr()
    # A short week's estimate first becomes usable on or after its Friday label.
    corr_dates = weekly.index.get_indexer(prices.index, method="pad")
    corr_values = weekly_corr.to_numpy().reshape(len(weekly), len(tickers), len(tickers))
    identity = np.eye(len(tickers))
    shrinkage = model["correlation_shrinkage"]
    smoothing = 1 / model["smoothing_sessions"]
    daily_vol_target = model["annual_vol_target"] / np.sqrt(model["sessions_per_year"])
    max_gross = model["max_gross_weight"]
    smoothed = np.zeros(len(tickers))
    targets = np.full(signal.shape, np.nan)

    for row, (current_signal, vol, corr_row) in enumerate(
        zip(signal.to_numpy(), risk_vol.to_numpy(), corr_dates)
    ):
        if corr_row < 0 or not np.isfinite(current_signal).all() or not (vol > 0).all():
            continue
        correlation = corr_values[corr_row]
        if not np.isfinite(correlation).all():
            continue
        correlation = (1 - shrinkage) * correlation + shrinkage * identity
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        decorrelated = eigenvectors @ ((eigenvectors.T @ current_signal) / np.sqrt(eigenvalues))
        smoothed = (1 - smoothing) * smoothed + smoothing * decorrelated / vol
        gross = np.abs(smoothed).sum()
        if gross == 0:
            targets[row] = 0.0
            continue
        risk_units = vol * smoothed
        predicted_vol = np.sqrt(risk_units @ correlation @ risk_units)
        targets[row] = smoothed * min(daily_vol_target / predicted_vol, max_gross / gross)

    return pd.DataFrame(targets, index=prices.index, columns=tickers)


def size_positions(previous_close, previous_value, target, gross_limit):
    size_fraction = gross_limit / max(gross_limit, np.abs(target).sum())
    return target * (previous_value * size_fraction) / previous_close


def simulate(closes, previous_closes, cash_returns, targets, friction, cash_buffer):
    positions = np.zeros(closes.shape[1])
    gross_limit = 1 - cash_buffer
    cash = previous_value = 1.0
    rows = []
    for date, close, previous_close, cash_return, target in zip(
        closes.index, closes.to_numpy(), previous_closes.to_numpy(),
        cash_returns.to_numpy(), targets.to_numpy(),
    ):
        # Lock the next positions using only the information at the sizing close.
        next_positions = positions
        if np.isfinite(target).all():
            next_positions = size_positions(
                previous_close, previous_value, target, gross_limit,
            )
        planned_changes = next_positions - positions

        # Mark the old positions and accrue interest before applying the fixed fill.
        previous_long_value = np.maximum(positions, 0) @ previous_close
        market_pnl = positions @ (close - previous_close)
        interest = (previous_value - previous_long_value) * cash_return
        cash_before_fill = cash + interest
        value_before_fill = cash_before_fill + positions @ close
        if not np.isfinite(value_before_fill) or value_before_fill <= 0:
            raise ValueError(f"Portfolio value became nonpositive or invalid on {date.date()}")

        traded = np.abs(planned_changes) @ close
        fee = friction * traded
        cash_after_fill = cash_before_fill - (planned_changes @ close + fee)
        short_collateral = 2 * (np.maximum(-next_positions, 0) @ close)
        value_after_fill = cash_after_fill + next_positions @ close
        if not np.isfinite(value_after_fill) or value_after_fill <= 0 or cash_after_fill < short_collateral:
            raise ValueError(f"Insufficient cash or short collateral for fixed shares on {date.date()}")

        daily_return = cash_return + (market_pnl - previous_long_value * cash_return - fee) / previous_value
        rows.append((daily_return, traded / value_before_fill, np.count_nonzero(planned_changes)))
        positions = next_positions
        cash = cash_after_fill
        previous_value = value_after_fill

    return pd.DataFrame(rows, index=closes.index, columns=["return", "turnover", "n_trades"])


def simulate_pair(prices, cash_returns, targets, friction, cash_buffer):
    closes = prices.loc[targets.index]
    previous_closes = prices.shift(1).loc[targets.index]
    cash_returns = cash_returns.loc[targets.index]
    passive_targets = pd.DataFrame(np.nan, index=targets.index, columns=targets.columns)
    if not passive_targets.empty:
        passive_targets.iloc[0] = 1 / len(prices.columns)
    return (
        simulate(closes, previous_closes, cash_returns, targets, friction, cash_buffer),
        simulate(closes, previous_closes, cash_returns, passive_targets, friction, cash_buffer),
    )


def metrics(daily_results, cash_returns, sessions_per_year):
    returns = daily_results["return"]
    if returns.empty:
        keys = ("cagr", "ann_vol", "sharpe", "sharpe_excess", "max_drawdown", "total_return", "turnover")
        return dict.fromkeys(keys) | {"n_trades": 0}
    excess = returns - cash_returns.loc[returns.index]
    values = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    scale = np.sqrt(sessions_per_year)
    vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    return {
        "cagr": float(values[-1] ** (sessions_per_year / len(returns)) - 1),
        "ann_vol": float(vol * scale) if np.isfinite(vol) else None,
        "sharpe": float(returns.mean() * scale / vol) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() * scale / excess_vol) if excess_vol > 0 else None,
        "max_drawdown": float((values / np.maximum.accumulate(values) - 1).min()),
        "total_return": float(values[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * sessions_per_year),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    tickers = [ticker for group in config["universe"].values() for ticker in group]
    cash_proxy = config["cash_proxy"]
    symbols = [*tickers, cash_proxy]
    if not tickers or len(set(symbols)) != len(symbols):
        raise ValueError("The universe and cash proxy must use distinct symbols")
    data_start, start, is_end, end = (
        pd.Timestamp(config["dates"][key]) for key in ("data_start", "start", "is_end", "end")
    )
    source_published = pd.Timestamp(config["dates"]["source_published"])
    execution = config["execution"]
    lag = execution["lag_sessions"]
    cost_bps, slippage_bps = execution["cost_bps"], execution["slippage_bps"]
    cash_buffer = execution["cash_buffer"]
    model = config["model"]
    sessions_per_year = model["sessions_per_year"]
    if not (data_start < start <= is_end < end):
        raise ValueError("Expected a warm-up period followed by ordered IS and OOS dates")
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("Execution lag must be at least one session")
    if min(cost_bps, slippage_bps) < 0 or cost_bps + slippage_bps >= 10_000:
        raise ValueError("Invalid transaction cost or slippage")
    if not 0 < cash_buffer < 1:
        raise ValueError("Cash buffer must be in (0, 1)")
    if not (0 < model["correlation_shrinkage"] <= 1 and 0 < model["max_gross_weight"] <= 1):
        raise ValueError("Correlation shrinkage and gross exposure cap must be in (0, 1]")
    if min(model["annual_vol_target"], sessions_per_year) <= 0:
        raise ValueError("Volatility target and annualization factor must be positive")
    friction = (cost_bps + slippage_bps) / 10_000

    prices, fetched_at = load_prices(symbols, data_start, end)
    cash_returns = prices[cash_proxy].pct_change(fill_method=None).loc[start:end]
    dates = cash_returns.index
    targets = trend_targets(prices, tickers, cash_proxy, model).shift(lag).loc[dates]
    if not np.isfinite(targets).all().all():
        raise ValueError("Insufficient warm-up history for one or more requested sessions")

    daily_results, passive_results = simulate_pair(prices[tickers], cash_returns, targets, friction, cash_buffer)
    # Start from cash on the first OOS fill whose signal follows publication.
    public_signal = pd.Series(prices.index > source_published, index=prices.index)
    public_fill = public_signal.shift(lag, fill_value=False).loc[dates]
    post_targets = targets.loc[(dates > is_end) & public_fill]
    post_results, post_passive_results = simulate_pair(
        prices[tickers], cash_returns, post_targets, friction, cash_buffer,
    )

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"EXCESS_SHARPE_REFERENCE: {cash_proxy}")
    print(f"BENCHMARK: Buy-and-hold with equal initial target weights ({', '.join(tickers)})")
    output = {
        "strategy": config["strategy"],
        "data_range": {"start": str(dates[0].date()), "end": str(dates[-1].date())},
    }
    for name, mask in (
        ("is", dates <= is_end),
        ("oos", dates > is_end),
        ("full", slice(None)),
    ):
        output[name] = metrics(daily_results.loc[mask], cash_returns, sessions_per_year)
        passive = metrics(passive_results.loc[mask], cash_returns, sessions_per_year)
        print(f"{name.upper()}: {json.dumps(output[name], allow_nan=False)}")
        print(f"BENCHMARK_{name.upper()}: {json.dumps(passive, allow_nan=False)}")
    output["benchmark"] = passive
    for label, results in (
        ("OOS_POST_PUBLICATION", post_results),
        ("BENCHMARK_OOS_POST_PUBLICATION", post_passive_results),
    ):
        stats = metrics(results, cash_returns, sessions_per_year)
        print(f"{label}: {json.dumps(stats, allow_nan=False)}")
    print(f"RESULTS_JSON: {json.dumps(output, allow_nan=False)}")


if __name__ == "__main__":
    main()
