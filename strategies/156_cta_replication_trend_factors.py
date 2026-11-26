# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 156_cta_replication_trend_factors\n\ndata_start: 2016-01-01\nin_sample:\n  start: 2020-01-02\n  end: 2025-07-17\nout_of_sample:\n  start: 2025-07-18\n  end: 2026-10-30\n\nuniverse:\n  equity: [SPY, QQQ, EFA, EEM]\n  treasury: [SHY, IEF]\n  currency: [FXE, FXY]\n  commodity: [GLD, DBC]\ncta_proxy: DBMF\nbenchmark: SPY\n\ntrend:\n  short_windows: [10, 20, 40, 60]\n  long_window: 500\n\nfilter:\n  initial_loading_sd: 0.25\n  daily_loading_step_sd: 0.004\n  observation_noise_bps: 60\n\nportfolio:\n  max_gross_exposure: 1.50\n  max_asset_exposure: 0.40\n  transaction_bps: 3.0  # One way, per dollar traded\n  annual_short_borrow_bps: 50.0\n  sessions_per_year: 252\n'
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
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.special import ndtr
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
BPS = 10_000


def validate_prices(prices, tickers, proxy, benchmark, start, end, is_start):
    prices = prices.reindex(columns=tickers).sort_index().loc[start:end]
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("Yahoo prices are empty or contain duplicate dates")
    prices = prices.replace([np.inf, -np.inf], np.nan).where(lambda p: p > 0)
    prices = prices.dropna(how="all")
    market = prices.drop(columns=proxy)
    missing = market.isna().any()
    if missing.any():
        raise ValueError(f"Missing Yahoo closes for {', '.join(missing[missing].index)}")
    if prices.empty or (prices.index[0] - start).days > 4 or (end - prices.index[-1]).days > 4:
        raise ValueError(f"Incomplete Yahoo date coverage for {benchmark}")

    proxy_start = prices[proxy].first_valid_index()
    if proxy_start is None or proxy_start >= is_start or prices.loc[proxy_start:, proxy].isna().any():
        raise ValueError(f"Missing {proxy} data before IS or after the first proxy observation")
    return prices


def load_prices(tickers, proxy, benchmark, start, end, is_start):
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / f"yahoo_adjusted_close_{start.date()}_{end.date()}.csv"
    if cache_file.exists():
        try:
            cached = pd.read_csv(cache_file, index_col="date", parse_dates=["date"])
            return validate_prices(cached, tickers, proxy, benchmark, start, end, is_start)
        except ValueError as exc:
            print(f"Refreshing price cache ({exc})")

    with TemporaryDirectory(prefix="yfinance_", dir=CACHE) as yf_cache:
        yf.set_tz_cache_location(yf_cache)
        for attempt in range(2):
            try:
                download = yf.download(
                    tickers,
                    start=start.date().isoformat(),
                    end=(end + pd.Timedelta(days=1)).date().isoformat(),
                    auto_adjust=True,
                    progress=False,
                    timeout=12,
                )
                prices = validate_prices(download["Close"], tickers, proxy, benchmark, start, end, is_start)
                break
            except (KeyError, ValueError) as exc:
                if attempt == 1:
                    raise RuntimeError("Unable to load a complete public Yahoo Finance price panel") from exc
                print(f"Retrying Yahoo Finance download ({exc})")

    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            prices.to_csv(temporary, index_label="date")
            temporary.close()
            temporary_path.replace(cache_file)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices


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


def backtest(returns, close_targets, portfolio):
    fee = portfolio["transaction_bps"] / BPS
    borrow = portfolio["annual_short_borrow_bps"] / BPS / portfolio["sessions_per_year"]
    if not np.isfinite(returns.to_numpy()).all() or not np.isfinite(close_targets.to_numpy()).all():
        raise ValueError("The evaluation range needs complete prices and warmed-up targets")
    if fee < 0 or fee * close_targets.abs().sum(axis=1).max() >= 1:
        raise ValueError("Transaction cost is outside the supported range")

    result = []
    holdings = np.zeros(len(returns.columns))
    for change, target in zip(returns.to_numpy(), close_targets.to_numpy()):
        marked = holdings * (1 + change)
        pretrade_equity = 1 + holdings @ change - borrow * np.maximum(-holdings, 0).sum()
        # Set targets relative to equity after fees. Each iteration recomputes
        # the trades against positions that have drifted with prices.
        equity = pretrade_equity
        for _ in range(20):
            traded = np.abs(target * equity - marked).sum()
            updated_equity = pretrade_equity - fee * traded
            if abs(updated_equity - equity) < 1e-12:
                break
            equity = updated_equity
        else:
            raise ValueError("Portfolio rebalancing cost did not converge")
        if updated_equity <= 0:
            raise ValueError("Portfolio equity was exhausted during the backtest")
        result.append((updated_equity - 1, traded))
        holdings = target

    return pd.DataFrame(result, index=returns.index, columns=["return", "turnover"])


def metrics(performance, proxy_returns, annual_sessions):
    daily = performance["return"]
    nav = (1 + daily).cumprod().to_numpy()
    peaks = np.maximum.accumulate(np.r_[1.0, nav])[1:]
    vol = daily.std(ddof=1) * np.sqrt(annual_sessions)
    values = {
        "cagr": nav[-1] ** (annual_sessions / len(daily)) - 1,
        "ann_vol": vol,
        "sharpe": daily.mean() * annual_sessions / vol if vol > 0 else np.nan,
        "max_drawdown": np.min(nav / peaks - 1),
        "total_return": nav[-1] - 1,
        "turnover": performance["turnover"].mean() * annual_sessions,
        "cta_proxy_corr": daily.corr(proxy_returns) if vol > 0 and proxy_returns.std() > 0 else np.nan,
    }
    return {key: float(value) if np.isfinite(value) else None for key, value in values.items()}


def print_metrics(rows):
    print(f"{'Portfolio / period':<25} {'CAGR':>10} {'Ann. vol':>10} {'Sharpe':>8} "
          f"{'Max DD':>10} {'Total ret.':>11} {'Ann. turn.':>11} {'CTA corr.':>10}")
    formats = (".2%", ".2%", ".2f", ".2%", ".2%", ".2%", ".2f")
    widths = (10, 10, 8, 10, 11, 11, 10)
    for label, row in rows.items():
        values = [format(value, spec) if value is not None else "n/a"
                  for value, spec in zip(row.values(), formats)]
        print(f"{label:<25} " + " ".join(value.rjust(width) for value, width in zip(values, widths)))


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
    benchmark = config["benchmark"]
    if proxy in assets or proxy == benchmark:
        raise ValueError("The CTA target must be separate from both the universe and the benchmark")
    tickers = list(dict.fromkeys([*assets, benchmark, proxy]))
    prices = load_prices(tickers, proxy, benchmark, start, oos_end, is_start)
    if prices[is_start:is_end].empty or prices[oos_start:oos_end].empty:
        raise ValueError("Both IS and OOS must contain trading sessions")
    returns = prices.pct_change(fill_method=None)
    proxy_returns = returns[proxy]
    target = target_weights(prices[assets], proxy_returns, config["trend"], config["filter"], config["portfolio"])
    dates = prices.loc[is_start:oos_end].index

    # A target decided at close t trades at close t+1. The simulator marks
    # the previous holdings before trading, so the first P&L is on t+2.
    strategy = backtest(returns.loc[dates, assets], target.shift(1).loc[dates], config["portfolio"])
    benchmark_targets = pd.DataFrame(1.0, index=dates, columns=[benchmark])
    passive = backtest(returns.loc[dates, [benchmark]], benchmark_targets, config["portfolio"])
    periods = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    annual_sessions = config["portfolio"]["sessions_per_year"]
    strategy_metrics = {period: metrics(strategy.loc[a:b], proxy_returns.loc[a:b], annual_sessions)
                        for period, (a, b) in periods.items()}
    benchmark_metrics = {period: metrics(passive.loc[a:b], proxy_returns.loc[a:b], annual_sessions)
                         for period, (a, b) in periods.items()}

    data_range = f"{dates[0].date()} to {dates[-1].date()}"
    print(f"{config['strategy']} | {data_range} | CTA proxy {proxy} | Benchmark {benchmark} buy and hold")
    for period, (a, b) in periods.items():
        print(f"{period.upper():<4} {a.date()} to {b.date()}")
    print_metrics({
        **{f"Strategy {key.upper()}": value for key, value in strategy_metrics.items()},
        **{f"{benchmark} {key.upper()}": value for key, value in benchmark_metrics.items()},
    })
    results = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **strategy_metrics,
        "benchmark": {
            "name": f"{benchmark} buy and hold",
            **benchmark_metrics["full"],
            "is": benchmark_metrics["is"],
            "oos": benchmark_metrics["oos"],
        },
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
