# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 125_liquidity_risk_indicator_timing\ncategory: macro\n\ntickers:\n  equity: SPY\n  high_yield: HYG\n  investment_grade: LQD\n  intermediate_treasury: IEF\n  short_treasury: SHY\n  volatility: ^VIX\n\ndata_start: '2007-07-01'  # Includes signal warm-up.\nin_sample:\n  start: '2009-01-01'\n  end: '2018-12-31'\nout_of_sample:\n  start: '2019-01-01'\n  end: '2025-12-31'\n\nsignal:\n  review_frequency: W-FRI\n  return_lookback_sessions: 21\n  standardization_sessions: 252\n  stress_threshold: 1.0\n\nallocation:\n  normal: {SPY: 0.60, LQD: 0.40, SHY: 0.00}\n  stressed: {SPY: 0.25, LQD: 0.25, SHY: 0.50}\n\ncost_bps: 5.0  # Per dollar bought or sold, including initial entry.\nsessions_per_year: 252\n"
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
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent


def load_prices(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    symbol_key = "_".join(symbol.replace("^", "index-") for symbol in tickers)
    cache_file = cache_dir / f"yahoo_adj_{symbol_key}_{start.date()}_{end.date()}.csv"
    download = not cache_file.exists()
    if download:
        with TemporaryDirectory(dir=cache_dir, prefix="yfinance_") as yf_cache:
            yf.set_tz_cache_location(yf_cache)
            raw = yf.download(
                tickers,
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=True,
                progress=False,
                timeout=15,
            )
        if raw.empty or not {"Open", "Close"}.issubset(raw.columns.get_level_values(0)):
            raise RuntimeError("Yahoo Finance returned no usable adjusted OHLC history.")
        prices = raw[["Open", "Close"]].copy()
    else:
        prices = pd.read_csv(cache_file, header=[0, 1], index_col=0)

    prices.index = pd.to_datetime(prices.index).normalize()
    prices = prices.sort_index().loc[start:end]
    closes = prices["Close"].reindex(columns=tickers)
    # VIX is an input only, so its open is not needed.
    opens = prices["Open"].reindex(columns=tickers[:-1])
    required = pd.concat([opens, closes], axis=1)
    if (
        required.empty
        or required.index.has_duplicates
        or required.index[0] > start + pd.Timedelta(days=7)
        or required.index[-1] < end - pd.Timedelta(days=7)
        or not np.isfinite(required.to_numpy()).all()
        or (required <= 0.0).any().any()
    ):
        raise RuntimeError("Price history is incomplete or invalid for the configured universe and dates.")

    if download:
        # Write only validated downloads and publish the cache atomically.
        with NamedTemporaryFile(dir=cache_dir, mode="w", encoding="utf-8", delete=False) as temp:
            temp_path = Path(temp.name)
            try:
                prices.to_csv(temp, index_label="Date")
                temp.flush()
                temp_path.replace(cache_file)
            finally:
                temp_path.unlink(missing_ok=True)
    return opens, closes


def stress_scores(closes, tickers, settings):
    returns = closes.pct_change(periods=settings["return_lookback_sessions"], fill_method=None)
    vix = closes[tickers["volatility"]]
    ig = returns[tickers["investment_grade"]]
    proxies = pd.DataFrame(
        {
            "vix": vix,
            "hy_stress": ig - returns[tickers["high_yield"]],
            "ig_stress": returns[tickers["intermediate_treasury"]] - ig,
        }
    )
    reference = proxies.shift(1).rolling(settings["standardization_sessions"])
    std = reference.std(ddof=1)
    z = (proxies - reference.mean()) / std.where(std > 0.0)
    return pd.DataFrame({"composite": z.mean(axis=1, skipna=False), "vix_only": z["vix"]})


def execution_scores(scores, frequency, start, end):
    sessions = scores.index
    reviews = pd.date_range(sessions[0], sessions[-1], freq=frequency)
    # An observed Friday holiday uses the latest close known by Friday evening.
    asof = sessions.get_indexer(reviews, method="pad")
    execute = sessions.searchsorted(reviews, side="right")
    valid = (asof >= 0) & (execute < len(sessions))
    weekly = scores.iloc[asof[valid]].set_axis(sessions[execute[valid]])
    by_session = weekly.reindex(sessions, method="ffill")
    if not np.isfinite(by_session.loc[start:end].to_numpy()).all():
        raise ValueError("Weekly scores are undefined within the report dates. Check the warm-up history.")

    # Start at the first sample open with the latest executable review, then trade weekly.
    first = scores.loc[start:end].index[0]
    execution_dates = weekly.index.union(pd.DatetimeIndex([first]))
    return by_session.loc[execution_dates].loc[start:end]


def target_weights(stress, normal, stressed):
    weights = np.where(stress.to_numpy()[:, None], stressed, normal)
    return pd.DataFrame(weights, index=stress.index)


def simulate(opens, closes, targets, fee_rate):
    overnight = opens.div(closes.shift(1)).to_numpy()
    intraday = closes.div(opens).to_numpy()
    targets = targets.reindex(opens.index).to_numpy()
    previous_nav = 1.0
    holdings = np.zeros(opens.shape[1])
    results = []

    for i, target in enumerate(targets):
        if i == 0:
            pre_trade_nav = previous_nav  # Start with cash at the first open.
        else:
            holdings *= overnight[i]
            pre_trade_nav = holdings.sum()

        turnover = 0.0
        if np.isfinite(target).all():
            # Find the post-fee NAV that finances the target holdings.
            investable = pre_trade_nav
            if fee_rate > 0.0:
                investable = brentq(
                    lambda nav: nav + fee_rate * np.abs(nav * target - holdings).sum() - pre_trade_nav,
                    0.0,
                    pre_trade_nav,
                    xtol=1e-14,
                )
            new_holdings = investable * target
            turnover = np.abs(new_holdings - holdings).sum() / pre_trade_nav
            holdings = new_holdings

        holdings *= intraday[i]
        nav = holdings.sum()
        results.append((nav / previous_nav - 1.0, turnover))
        previous_nav = nav

    return pd.DataFrame(results, index=opens.index, columns=["return", "turnover"])


def metrics(history, sessions_per_year):
    returns = history["return"].to_numpy()
    wealth = np.r_[1.0, np.cumprod(1.0 + returns)]
    std = returns.std(ddof=1) if len(returns) > 1 else np.nan
    values = {
        "cagr": wealth[-1] ** (sessions_per_year / len(returns)) - 1.0,
        "ann_vol": std * np.sqrt(sessions_per_year),
        "sharpe": returns.mean() / std * np.sqrt(sessions_per_year) if std > 0.0 else np.nan,
        "max_drawdown": (wealth / np.maximum.accumulate(wealth) - 1.0).min(),
        "total_return": wealth[-1] - 1.0,
        "turnover": history["turnover"].sum(),
    }
    clean = {key: float(value) if np.isfinite(value) else None for key, value in values.items()}
    clean["n_trades"] = int((history["turnover"] > 1e-12).sum())
    return clean


def print_results(result, reports):
    print(f"{result['strategy']} | {result['data_range']}")
    print(f"{'Portfolio':<14} {'Period':<6} {'CAGR':>9} {'Ann. vol':>9} {'Sharpe':>8} "
          f"{'Max DD':>9} {'Return':>10} {'Turnover':>10} {'Trades':>7}")
    for name, report in reports.items():
        for period, row in report.items():
            cells = []
            for key in ("cagr", "ann_vol", "sharpe", "max_drawdown", "total_return", "turnover"):
                value = row[key]
                fmt = ".2f" if key == "sharpe" else ".2%"
                cells.append(format(value, fmt) if value is not None else "n/a")
            cagr, vol, sharpe, drawdown, total, turnover = cells
            print(f"{name:<14} {period.upper():<6} {cagr:>9} {vol:>9} {sharpe:>8} "
                  f"{drawdown:>9} {total:>10} {turnover:>10} {row['n_trades']:>7}")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    periods = {
        name: tuple(pd.Timestamp(config[key][edge]) for edge in ("start", "end"))
        for name, key in (("is", "in_sample"), ("oos", "out_of_sample"))
    }
    start, is_end = periods["is"]
    oos_start, end = periods["oos"]
    data_start = pd.Timestamp(config["data_start"])
    if not (data_start < start <= is_end and is_end + pd.Timedelta(days=1) == oos_start <= end):
        raise ValueError("Use adjacent, ordered IS and OOS ranges with earlier warm-up data.")
    periods["full"] = (start, end)

    tickers = config["tickers"]
    assets = [tickers[role] for role in ("equity", "investment_grade", "short_treasury")]
    symbols = [*assets, tickers["high_yield"], tickers["intermediate_treasury"], tickers["volatility"]]
    normal, stressed = [
        pd.Series(config["allocation"][key], dtype=float).reindex(assets).to_numpy()
        for key in ("normal", "stressed")
    ]
    for weights in (normal, stressed):
        if not np.isfinite(weights).all() or (weights < 0.0).any() or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Allocations must cover all portfolio tickers, be long-only, and sum to one.")
    fee_rate = float(config["cost_bps"]) / 10_000.0
    if not 0.0 <= fee_rate < 1.0:
        raise ValueError("Transaction costs must be between zero and 10,000 bps, exclusive of the latter.")

    opens, closes = load_prices(symbols, data_start, end)
    settings = config["signal"]
    scores = stress_scores(closes, tickers, settings)
    reviews = execution_scores(scores, settings["review_frequency"], start, end)
    stress = reviews > settings["stress_threshold"]

    sample_opens = opens.loc[start:end, assets]
    sample_closes = closes.loc[start:end, assets]
    reports = {}
    for portfolio, rule in (
        ("Composite", stress["composite"]),
        ("Static", pd.Series(False, index=reviews.index)),
        ("VIX only", stress["vix_only"]),
    ):
        targets = target_weights(rule, normal, stressed)
        history = simulate(sample_opens, sample_closes, targets, fee_rate)
        reports[portfolio] = {}
        for period, (first, last) in periods.items():
            window = history.loc[first:last]
            if window.empty:
                raise ValueError(f"No trading sessions in the configured {period} window.")
            reports[portfolio][period] = metrics(window, config["sessions_per_year"])

    static = reports["Static"]
    result = {
        "strategy": config["strategy"],
        "data_range": f"{sample_opens.index[0].date()} to {sample_opens.index[-1].date()}",
        **reports["Composite"],
        "benchmark": {
            **static["full"],
            "is": static["is"],
            "oos": static["oos"],
            "vix_only": reports["VIX only"],
        },
    }
    print_results(result, reports)


if __name__ == "__main__":
    main()
