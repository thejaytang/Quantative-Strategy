# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 109_factor_etf_vol_matched\nuniverse:\n  sleeves:\n    value: VTV\n    momentum: MTUM\n    quality: QUAL\n    low_volatility: USMV\n  cash_proxy: BIL\n  market_proxy: SPY\n  peer_value: IWD\n  peer_growth: IWF\n  growth_reference: VUG\ndata_start: '2014-01-01'\nin_sample:\n  start: '2015-01-02'\n  end: '2020-12-31'\nout_of_sample:\n  start: '2021-01-01'\n  end: '2025-12-31'\nrebalance: month_end\nvolatility_lookback_sessions: 63\nexposure_lookback_sessions: 126\nsessions_per_year: 252\nmax_equity_weight: 1.0\ntransaction_cost_bps: 5.0  # Per dollar bought or sold, including BIL.\n"
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
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from scipy.optimize import brentq
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"


def load_prices(tickers, start, end):
    """Download complete, dividend- and split-adjusted daily closing prices."""
    CACHE.mkdir(exist_ok=True)
    cache_path = CACHE / f"yahoo_adjusted_{start.date()}_{end.date()}_{'-'.join(tickers)}.csv"
    if cache_path.exists():
        prices = pd.read_csv(cache_path, index_col="Date", parse_dates=["Date"])
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
        prices = download["Close"].copy()

    prices = prices.reindex(columns=tickers).sort_index()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    prices = prices.loc[start:end]
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("The adjusted-close panel is empty or has duplicate dates.")
    invalid = (~np.isfinite(prices) | prices.le(0)).sum()
    if invalid.any():
        raise ValueError(f"Missing or invalid adjusted closes: {invalid[invalid.gt(0)].to_dict()}")
    edge_tolerance = pd.Timedelta(days=7)
    if prices.index[0] > start + edge_tolerance or prices.index[-1] < end - edge_tolerance:
        raise ValueError("Adjusted closes do not cover the requested date range.")

    if not cache_path.exists():
        cache_file = NamedTemporaryFile(mode="w", dir=CACHE, delete=False)
        temporary_path = Path(cache_file.name)
        try:
            with cache_file:
                prices.to_csv(cache_file, index_label="Date", float_format="%.17g")
            temporary_path.replace(cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices


def next_close_targets(signal_weights, sessions, start, end):
    """Move each month-end weight to the following trading session's close."""
    return signal_weights.reindex(sessions).shift(1).loc[start:end].dropna(how="all")


def simulate(asset_returns, targets, fee_rate):
    """Drift holdings daily, then fill targets at the close using remaining NAV."""
    weights = np.zeros(asset_returns.shape[1])  # Initial capital is uninvested.
    result = []
    daily_targets = targets.reindex(asset_returns.index).to_numpy()
    for daily_returns, target in zip(asset_returns.to_numpy(), daily_targets, strict=True):
        gross_factor = 1.0 + weights @ daily_returns
        weights = weights * (1.0 + daily_returns) / gross_factor
        nav_after_cost = 1.0
        turnover = 0.0
        trades = 0
        if np.isfinite(target[0]):
            # The cost budget and the post-cost dollar value of the target sum to pre-cost NAV.
            if fee_rate:
                nav_after_cost = brentq(
                    lambda nav: nav + fee_rate * np.abs(nav * target - weights).sum() - 1.0,
                    0.0,
                    1.0,
                )
            traded = np.abs(nav_after_cost * target - weights)
            turnover = traded.sum()
            trades = np.count_nonzero(traded > 1e-10)
            weights = target
        result.append((gross_factor * nav_after_cost - 1.0, gross_factor - 1.0, turnover, trades))
    return pd.DataFrame(
        result, index=asset_returns.index, columns=["return", "gross_return", "turnover", "n_trades"]
    )


def risk_matched_weights(sleeve_returns, signal_vols, target_vol, lookback, annualizer, cap):
    signal_weights = []
    for signal, daily_vol in signal_vols.iterrows():
        # Each signal uses the window through that session's close.
        inverse_vol = 1.0 / daily_vol
        mix = inverse_vol / inverse_vol.sum()
        trailing_mix = sleeve_returns.loc[:signal].tail(lookback) @ mix
        mix_vol = trailing_mix.std() * annualizer
        target = target_vol.loc[signal]
        equity = min(cap, target / mix_vol) if mix_vol > 0 and pd.notna(target) else cap
        signal_weights.append(mix * equity)
    return pd.DataFrame(signal_weights, index=signal_vols.index, columns=sleeve_returns.columns)


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


def metrics(daily, sessions_per_year):
    returns = daily["return"]
    wealth = (1.0 + returns).cumprod()
    last = wealth.iloc[-1]
    peaks = wealth.cummax().clip(lower=1.0)
    annualizer = np.sqrt(sessions_per_year)
    volatility = returns.std()
    values = {
        "cagr": last ** (sessions_per_year / len(returns)) - 1.0,
        "ann_vol": volatility * annualizer,
        "sharpe": returns.mean() / volatility * annualizer if volatility > 0 else np.nan,
        "max_drawdown": (wealth / peaks - 1.0).min(),
        "total_return": last - 1.0,
        "turnover": daily["turnover"].mean() * sessions_per_year,
    }
    return {
        **{name: float(value) if np.isfinite(value) else None for name, value in values.items()},
        "n_trades": int(daily["n_trades"].sum()),
    }


def print_report(performance, exposures, periods):
    print(
        f"{'Portfolio':<16} {'Period':<6} {'CAGR':>9} {'Ann vol':>9} {'Sharpe':>8} "
        f"{'Max DD':>9} {'Total':>9} {'Ann turn':>9} {'Trades':>7}"
    )
    columns = [
        ("cagr", "+9.2%", 9),
        ("ann_vol", "9.2%", 9),
        ("sharpe", "8.2f", 8),
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


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)
    universe = cfg["universe"]
    sleeves = list(universe["sleeves"].values())
    cash = universe["cash_proxy"]
    market = universe["market_proxy"]
    peer_value, peer_growth = universe["peer_value"], universe["peer_growth"]
    growth_reference = universe["growth_reference"]
    traded_tickers = [*sleeves, cash]
    tickers = sorted(set([*traded_tickers, market, peer_value, peer_growth, growth_reference]))
    data_start = pd.Timestamp(cfg["data_start"])
    is_start, is_end = (pd.Timestamp(cfg["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(cfg["out_of_sample"][key]) for key in ("start", "end"))
    lookback = cfg["volatility_lookback_sessions"]
    exposure_lookback = cfg["exposure_lookback_sessions"]
    sessions_per_year = cfg["sessions_per_year"]
    annualizer = np.sqrt(sessions_per_year)
    fee_rate = cfg["transaction_cost_bps"] / 10_000
    cap = cfg["max_equity_weight"]
    if not data_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Use ordered warm-up, IS and OOS date ranges with no overlap.")
    if not (0.0 <= fee_rate < 1.0 and 0.0 < cap <= 1.0):
        raise ValueError("Use a cost in [0, 10000) bps and an equity cap in (0, 1].")
    if cfg["rebalance"] != "month_end" or min(lookback, exposure_lookback, sessions_per_year) < 2:
        raise ValueError("Use month_end rebalancing and windows of at least two sessions.")

    returns = load_prices(tickers, data_start, oos_end).pct_change(fill_method=None).iloc[1:]
    sessions = returns.index
    months = sessions.to_period("M")
    month_ends = sessions[:-1][months[:-1] != months[1:]]
    sleeve_returns = returns[sleeves]
    month_end_vols = sleeve_returns.rolling(lookback).std().loc[month_ends]
    signal_vols = month_end_vols.loc[month_end_vols.gt(0).all(axis=1)]
    periods = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    asset_returns = returns[traded_tickers].loc[is_start:oos_end]
    if asset_returns.loc[is_start:is_end].empty or asset_returns.loc[oos_start:oos_end].empty:
        raise ValueError("Both evaluation periods must have trading sessions.")
    if len(asset_returns.loc[is_end + pd.Timedelta(days=1):oos_start - pd.Timedelta(days=1)]):
        raise ValueError("The IS and OOS periods must cover consecutive trading sessions.")

    equal_signals = pd.DataFrame(1.0 / len(sleeves), index=signal_vols.index, columns=sleeves)
    equal_signals[cash] = 0.0
    equal_targets = next_close_targets(equal_signals, sessions, is_start, oos_end)
    if equal_targets.empty or equal_targets.index[0] > is_end:
        raise ValueError("The requested warm-up and IS period leave no IS rebalance.")
    benchmark = simulate(asset_returns, equal_targets, fee_rate)
    # Only invested IS benchmark returns enter the target. Expanding in IS, frozen in OOS.
    calibration = benchmark.loc[benchmark.index > equal_targets.index[0], "gross_return"].loc[:is_end]
    expanding_vol = calibration.expanding(min_periods=lookback).std() * annualizer
    if expanding_vol.dropna().empty:
        raise ValueError("IS needs enough invested sessions to calibrate the volatility target.")
    target_vol = expanding_vol.reindex(sessions).ffill()
    matched_signals = risk_matched_weights(sleeve_returns, signal_vols, target_vol, lookback, annualizer, cap)
    matched_signals[cash] = (1.0 - matched_signals.sum(axis=1)).clip(lower=0.0)
    matched_targets = next_close_targets(matched_signals, sessions, is_start, oos_end)
    matched = simulate(asset_returns, matched_targets, fee_rate)

    portfolios = {"Risk matched": matched, "Equal weight": benchmark}
    performance = {
        name: {
            period: metrics(daily.loc[start:end], sessions_per_year)
            for period, (start, end) in periods.items()
        }
        for name, daily in portfolios.items()
    }
    compared_returns = returns[[*sleeves, growth_reference]].copy()
    for name, daily in portfolios.items():
        compared_returns[name] = daily["return"]
    value_spread = (returns[peer_value] - returns[peer_growth]).rename(f"{peer_value}-{peer_growth}")
    exposures = rolling_exposures(compared_returns, returns[market], value_spread, exposure_lookback)

    data_range = f"{asset_returns.index[0].date()} to {asset_returns.index[-1].date()}"
    print(cfg["strategy"], "|", data_range)
    print(f"Frozen IS equal-weight volatility target {expanding_vol.iloc[-1]:.2%}")
    print_report(performance, exposures, periods)
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
