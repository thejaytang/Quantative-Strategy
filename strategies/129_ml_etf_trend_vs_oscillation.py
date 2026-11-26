# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 129_ml_etf_trend_vs_oscillation\nuniverse: [SPY, QQQ, IWM, DIA]\nvix_ticker: '^VIX'\nbenchmark_ticker: SPY\ncash_benchmark_ticker: BIL\n\ndata_start: '2009-01-01'\nin_sample:\n  start: '2015-01-01'\n  end: '2019-12-31'\nout_of_sample:\n  start: '2020-01-01'\n  end: '2025-12-31'\n\nfeatures:\n  indicator_sessions: 14\n  short_return_sessions: 5\n  trend_return_sessions: 21\nmodel:\n  move_threshold: 0.005\n  probability_threshold: 0.55\n  n_estimators: 80\n  max_depth: 5\n  training_years: 5\n  refit_months: 6\n  random_state: 129\n\ncost_bps_per_side: 2.0\n"
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
"""Walk-forward C30 ETF magnitude classifier and intraday overlay."""

import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
OHLC = ["Open", "High", "Low", "Close"]
FEATURES = ["rsi", "atr_pct", "short_return", "trend_return", "vix"]
SESSIONS_PER_YEAR = 252


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(CACHE / "yfinance"))
    prices = {}
    missing = []
    paths = {
        ticker: CACHE / f"{ticker}_{start.date()}_{end.date()}_adj.csv"
        for ticker in tickers
    }
    for ticker, path in paths.items():
        if path.is_file():
            prices[ticker] = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            missing.append(ticker)

    if missing:
        print(f"Downloading adjusted daily bars for {', '.join(missing)}", flush=True)
        downloaded = yf.download(
            missing,
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True,
            ignore_tz=True,
            group_by="ticker",
            multi_level_index=True,
            progress=False,
            threads=4,
            timeout=15,
        )

    for ticker in tickers:
        fields = ["Close"] if ticker.startswith("^") else OHLC
        frame = prices[ticker] if ticker in prices else downloaded.get(ticker)
        if frame is None or not set(fields).issubset(frame.columns):
            raise RuntimeError(f"Missing Yahoo Finance price data for {ticker}")
        frame = frame[fields].dropna(how="all").sort_index()
        if (
            frame.empty
            or not frame.index.is_unique
            or not np.isfinite(frame.to_numpy()).all()
            or frame.le(0).any().any()
            or not start <= frame.index[0] <= start + pd.Timedelta(days=7)
            or not end - pd.Timedelta(days=7) <= frame.index[-1] <= end
        ):
            raise RuntimeError(f"Invalid or incomplete price history for {ticker}: {paths[ticker]}")
        if ticker in missing:
            with NamedTemporaryFile(mode="w", dir=CACHE, delete=False) as temp:
                temp_path = Path(temp.name)
                try:
                    frame.to_csv(temp, index_label="Date")
                    temp.close()
                    temp_path.replace(paths[ticker])
                finally:
                    temp_path.unlink(missing_ok=True)
        prices[ticker] = frame
    return prices


def build_panel(prices, universe, vix_ticker, calendar, feature_config, threshold):
    # VIX is available after its own close and before the next equity open.
    vix = prices[vix_ticker]["Close"].reindex(calendar).ffill(limit=1)
    window = feature_config["indicator_sessions"]
    by_ticker = {}
    for ticker in universe:
        bars = prices[ticker].loc[calendar]
        close = bars["Close"]
        change = close.diff()
        gains = change.clip(lower=0).rolling(window).mean()
        losses = (-change.clip(upper=0)).rolling(window).mean()
        movement = gains + losses
        rsi = 100 * gains.div(movement.mask(movement.eq(0)))
        rsi = rsi.mask(movement.eq(0), 50)
        prev_close = close.shift()
        true_range = pd.concat(
            [bars["High"] - bars["Low"], (bars["High"] - prev_close).abs(),
             (bars["Low"] - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        features = pd.DataFrame({
            "rsi": rsi,
            "atr_pct": true_range.rolling(window).mean().div(close),
            "short_return": close.pct_change(feature_config["short_return_sessions"]),
            "trend_return": close.pct_change(feature_config["trend_return_sessions"]),
            "vix": vix,
        }).shift()
        # Index by execution session: predictors were known the prior evening.
        features["intraday_return"] = close.div(bars["Open"]).sub(1)
        features["large_move"] = features["intraday_return"].abs().gt(threshold)
        by_ticker[ticker] = features
    return pd.concat(by_ticker, names=["ticker", "session"]).swaplevel().sort_index()


def walk_forward(panel, sessions, config):
    dates = panel.index.get_level_values("session")
    eligible = panel[FEATURES].notna().all(axis=1).to_numpy()
    probability = pd.Series(np.nan, index=panel.index, name="probability")
    refit_bucket = (sessions.year * 12 + sessions.month - 1) // config["refit_months"]

    for _, block in pd.Series(sessions, index=sessions).groupby(refit_bucket):
        first, last = block.iloc[0], block.iloc[-1]
        start = first - pd.DateOffset(years=config["training_years"])
        training = panel.loc[eligible & (dates >= start) & (dates < first)]
        if training["large_move"].nunique() < 2:
            raise ValueError(f"Training window before {first.date()} needs both target classes")
        test = eligible & (dates >= first) & (dates <= last)
        if not test.any():
            continue
        forest = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=config["random_state"],
            n_jobs=2,
        )
        forest.fit(training[FEATURES], training["large_move"])
        probability.loc[test] = forest.predict_proba(panel.loc[test, FEATURES])[:, 1]
    return probability.loc[sessions]


def intraday_portfolio(active, intraday, cost):
    # Each active sleeve's capital pays for both its opening notional and fee.
    opening = active.astype(float) / (active.shape[1] * (1 + cost))
    closing = opening * (1 + intraday)
    return pd.DataFrame({
        "return": (closing * (1 - cost) - opening * (1 + cost)).sum(axis=1),
        "turnover": (opening + closing).sum(axis=1),
        "n_trades": 2 * active.sum(axis=1),
    })


def buy_and_hold(bars, cost):
    # Establish the position at the first open and liquidate at the last close.
    growth = bars["Close"].div(bars["Close"].shift())
    growth.iloc[0] = bars["Close"].iloc[0] / (bars["Open"].iloc[0] * (1 + cost))
    turnover = pd.Series(0.0, index=bars.index)
    turnover.iloc[0] += 1 / (1 + cost)
    turnover.iloc[-1] += growth.iloc[-1]
    growth.iloc[-1] *= 1 - cost
    n_trades = pd.Series(0, index=bars.index)
    n_trades.iloc[0] += 1
    n_trades.iloc[-1] += 1
    return pd.DataFrame({"return": growth - 1, "turnover": turnover, "n_trades": n_trades})


def metrics(portfolio):
    returns = portfolio["return"]
    equity = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    years = len(returns) / SESSIONS_PER_YEAR
    daily_vol = returns.std(ddof=1)
    metrics = {
        "cagr": equity[-1] ** (1 / years) - 1,
        "ann_vol": daily_vol * math.sqrt(SESSIONS_PER_YEAR),
        "sharpe": returns.mean() * math.sqrt(SESSIONS_PER_YEAR) / daily_vol
        if daily_vol > 0 else np.nan,
        "max_drawdown": (equity / np.maximum.accumulate(equity) - 1).min(),
        "total_return": equity[-1] - 1,
        "ann_turnover": portfolio["turnover"].sum() / years,
    }
    return {
        **{key: float(val) if math.isfinite(val) else None for key, val in metrics.items()},
        "n_trades": int(portfolio["n_trades"].sum()),
        "n_sessions": len(returns),
    }


def evaluate(portfolio, periods, scores=None):
    results = {}
    for period, (start, end) in periods.items():
        results[period] = metrics(portfolio.loc[start:end])
        if scores is not None:
            observations = scores.loc[start:end].dropna()
            results[period]["auc"] = (
                float(roc_auc_score(observations["large_move"], observations["score"]))
                if observations["large_move"].nunique() == 2 else None
            )
    return results


def print_table(results):
    def fmt(value, format_spec):
        return format(value, format_spec) if value is not None else "--"

    print(f"{'Portfolio':<21} {'Period':<6} {'CAGR':>9} {'Ann. vol':>9} {'Sharpe':>8} "
          f"{'Max DD':>9} {'Total ret':>10} {'Ann. TO':>10} {'Trades':>7} {'AUC':>7}")
    for name, periods in results.items():
        for period, values in periods.items():
            columns = [
                fmt(values["cagr"], "9.2%"), fmt(values["ann_vol"], "9.2%"),
                fmt(values["sharpe"], "8.3f"), fmt(values["max_drawdown"], "9.2%"),
                fmt(values["total_return"], "10.2%"), fmt(values["ann_turnover"], "9.2f") + "x",
                fmt(values["n_trades"], "7d"), fmt(values.get("auc"), "7.3f"),
            ]
            print(f"{name:<21} {period.upper():<6} " + " ".join(columns))


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    periods = {
        key: (pd.Timestamp(config[section]["start"]), pd.Timestamp(config[section]["end"]))
        for key, section in [("is", "in_sample"), ("oos", "out_of_sample")]
    }
    start, is_end = periods["is"]
    oos_start, end = periods["oos"]
    data_start = pd.Timestamp(config["data_start"])
    if not data_start < start <= is_end < oos_start <= end:
        raise ValueError("Expected prehistory followed by ordered, non-overlapping IS and OOS ranges")
    periods["full"] = (start, end)
    cost = config["cost_bps_per_side"] / 10_000
    if not 0 <= cost < 1:
        raise ValueError("cost_bps_per_side must be in [0, 10000)")

    universe = config["universe"]
    vix, equity, cash = (config[key] for key in
                         ["vix_ticker", "benchmark_ticker", "cash_benchmark_ticker"])
    traded = list(dict.fromkeys([*universe, equity, cash]))
    prices = load_prices([*traded, vix], data_start, end)
    calendar = prices[equity].index
    for ticker in traded:
        if not prices[ticker].index.equals(calendar):
            raise RuntimeError(f"{ticker} price dates differ from the {equity} calendar")
    for name, (period_start, period_end) in periods.items():
        if not ((calendar >= period_start) & (calendar <= period_end)).any():
            raise ValueError(f"No market sessions in {name.upper()} range")
    sessions = calendar[(calendar >= start) & (calendar <= end)]

    model_config = config["model"]
    panel = build_panel(prices, universe, vix, calendar, config["features"],
                        model_config["move_threshold"])
    probability = walk_forward(panel, sessions, model_config)
    evaluation = panel.loc[sessions]
    positive_trend = evaluation["trend_return"].gt(0)
    gated = positive_trend & probability.gt(model_config["probability_threshold"])
    intraday = evaluation["intraday_return"].unstack("ticker")
    model_scores = evaluation[["large_move"]].assign(score=probability)
    trend_scores = model_scores.assign(score=positive_trend.where(probability.notna()))

    strategy_stats = evaluate(intraday_portfolio(gated.unstack("ticker"), intraday, cost),
                              periods, model_scores)
    trend_stats = evaluate(intraday_portfolio(positive_trend.unstack("ticker"), intraday, cost),
                           periods, trend_scores)
    equity_stats = evaluate(buy_and_hold(prices[equity].loc[sessions], cost), periods)
    cash_stats = evaluate(buy_and_hold(prices[cash].loc[sessions], cost), periods)
    data_range = f"{sessions[0].date()} to {sessions[-1].date()}"
    print(f"{config['strategy']} | {data_range} | {config['cost_bps_per_side']:g} bps per side")
    print_table({
        "Random forest + trend": strategy_stats,
        "Trend only": trend_stats,
        f"{equity} buy and hold": equity_stats,
        f"{cash} buy and hold": cash_stats,
    })
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **strategy_stats,
        "benchmark": {
            "name": f"{equity} buy and hold",
            **equity_stats["full"],
            "is": equity_stats["is"],
            "oos": equity_stats["oos"],
            "trend_only": trend_stats,
            "cash_proxy": {"name": f"{cash} buy and hold", **cash_stats},
        },
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
