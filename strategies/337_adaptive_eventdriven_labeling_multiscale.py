# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 337_adaptive_eventdriven_labeling_multiscale\ntickers: [SPY, QQQ, IWM, XLF, XLK, XLE, GLD, TLT]\ncash_ticker: BIL\nbenchmark_ticker: SPY\n\ndata_start: 2008-01-01\nin_sample:\n  start: 2011-01-01\n  end: 2018-12-31\nout_of_sample:\n  start: 2019-01-01\n  end: 2025-12-31\n\nevent_sigma: 1.25\nvolatility_sessions: 20\nregime_sessions: 252\nhigh_volatility_horizon: 5\nlow_volatility_horizon: 10\nfixed_horizon: 5\nfeature_return_sessions: [1, 5, 20, 60, 120]\nlogistic_c: 1.0\nprobability_threshold: 0.55\ncost_bps_per_side: 5.0\nsessions_per_year: 252\n'
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
"""Walk-forward ETF backtest for the frozen C128 strategy candidate."""

import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def load_bars(tickers, start, end, calendar_ticker):
    cache = ROOT / "data_cache"
    cache.mkdir(exist_ok=True)
    paths = {
        ticker: cache / f"{ticker}_{start.date()}_{end.date()}.csv" for ticker in tickers
    }
    bars = {
        ticker: pd.read_csv(path, index_col=0, parse_dates=True)
        for ticker, path in paths.items()
        if path.is_file()
    }
    missing = [ticker for ticker in tickers if ticker not in bars]
    if missing:
        yf.set_tz_cache_location(str(cache / "yfinance"))
        download = yf.download(
            missing,
            start=str(start.date()),
            end=str((end + pd.Timedelta(days=1)).date()),
            auto_adjust=True,
            group_by="ticker",
            multi_level_index=True,
            progress=False,
            threads=True,
            timeout=15,
        )
        for ticker in missing:
            if ticker not in download.columns.get_level_values(0):
                raise RuntimeError(f"Yahoo Finance returned no data for {ticker}")
            bars[ticker] = download[ticker][FIELDS].dropna(how="all")

    for ticker, frame in bars.items():
        if frame.empty:
            raise RuntimeError(f"No bars for {ticker}")
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise RuntimeError(f"Invalid session index for {ticker}")

    dates = bars[calendar_ticker].loc[start:end].index
    week = pd.Timedelta(days=7)
    if dates.empty or dates[0] > start + week or dates[-1] < end - week:
        raise RuntimeError("Yahoo Finance history does not cover the requested date range")
    for ticker, frame in bars.items():
        if not frame.loc[start:end].index.equals(dates):
            raise RuntimeError(f"Session calendar mismatch for {ticker}")
        frame = frame.loc[dates, FIELDS].rename_axis(columns=None)
        if (
            not np.isfinite(frame.to_numpy()).all()
            or (frame[FIELDS[:4]] <= 0).any().any()
            or (frame["Volume"] < 0).any()
        ):
            raise RuntimeError(f"Invalid OHLCV values for {ticker}")
        bars[ticker] = frame
    for ticker in missing:
        path = paths[ticker]
        with NamedTemporaryFile(dir=cache, suffix=".tmp", mode="w", delete=False) as csv_file:
            temporary = Path(csv_file.name)
            try:
                bars[ticker].to_csv(csv_file)
                csv_file.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    return bars


def build_events(bars, cfg):
    """Use the longer exit to schedule the same events for both experiments."""
    cash_open = bars[cfg["cash_ticker"]]["Open"].to_numpy()
    window = cfg["volatility_sessions"]
    fixed_days = cfg["fixed_horizon"]
    feature_names = [f"return_{days}" for days in cfg["feature_return_sessions"]]
    feature_names += ["range", "relative_volume"]
    events = []
    for sleeve, ticker in enumerate(cfg["tickers"]):
        frame = bars[ticker]
        close = frame["Close"]
        daily_return = close.pct_change()
        volatility = daily_return.rolling(window).std().shift(1)
        median = volatility.shift(1).rolling(cfg["regime_sessions"]).median()
        features = pd.DataFrame(
            {f"return_{days}": close.pct_change(days) for days in cfg["feature_return_sessions"]}
        )
        features["range"] = frame["High"] / frame["Low"] - 1
        features["relative_volume"] = frame["Volume"] / frame["Volume"].rolling(window).mean()
        features = features.shift(1).to_numpy()
        triggers = (
            (daily_return.abs() > cfg["event_sigma"] * volatility)
            & median.notna()
            & np.isfinite(features).all(axis=1)
        )
        high_volatility = (volatility > median).to_numpy()
        opens = frame["Open"].to_numpy()
        next_entry = 0
        for signal in np.flatnonzero(triggers.to_numpy()):
            entry = signal + 1
            if entry < next_entry:
                continue
            days = (
                cfg["high_volatility_horizon"]
                if high_volatility[signal]
                else cfg["low_volatility_horizon"]
            )
            adaptive_exit, fixed_exit = entry + days, entry + fixed_days
            next_entry = max(adaptive_exit, fixed_exit)
            labels = []
            for exit_open in (adaptive_exit, fixed_exit):
                if exit_open >= len(opens):
                    labels.append(np.nan)
                else:
                    asset_growth = opens[exit_open] / opens[entry]
                    cash_growth = cash_open[exit_open] / cash_open[entry]
                    labels.append(int(asset_growth > cash_growth))
            events.append(
                (signal, sleeve, entry, adaptive_exit, fixed_exit, next_entry, *labels, *features[signal])
            )
    columns = [
        "signal", "sleeve", "entry", "adaptive_exit", "fixed_exit", "label_ready",
        "adaptive_label", "fixed_label", *feature_names,
    ]
    return pd.DataFrame(events, columns=columns), feature_names


def walk_forward(events, features, dates, first_session, cfg):
    """Refit separate logistic models on the same purged rows each quarter."""
    trade_dates = dates[first_session:]
    quarters = trade_dates.to_period("Q")
    refits = np.flatnonzero(~quarters.duplicated()) + first_session
    cash = len(cfg["tickers"])
    result = {}
    with threadpool_limits(limits=1):
        for variant in ("adaptive", "fixed"):
            holdings = np.full((len(dates), cash), cash, dtype=int)
            for refit, next_refit in zip(refits, [*refits[1:], len(dates)], strict=True):
                train = events.loc[events["label_ready"] < refit]
                label = train[f"{variant}_label"]
                current = events.loc[events["signal"].between(refit, next_refit - 1)]
                if label.nunique() < 2 or current.empty:
                    continue
                model = make_pipeline(StandardScaler(), LogisticRegression(C=cfg["logistic_c"]))
                model.fit(train[features], label)
                probability = model.predict_proba(current[features])[:, 1]
                picks = current.loc[probability > cfg["probability_threshold"]]
                windows = picks[["entry", f"{variant}_exit", "sleeve"]]
                for entry, exit_open, sleeve in windows.itertuples(index=False, name=None):
                    holdings[entry:exit_open, sleeve] = sleeve
            result[variant] = holdings[first_session:]
    return result


def simulate(opens, closes, holdings, fee):
    """Mark independent sleeves at close and charge each bought/sold dollar at the open."""
    dates = opens.index
    opens, closes = opens.to_numpy(), closes.to_numpy()
    n_sleeves = holdings.shape[1]
    nav = np.full(n_sleeves, 1 / n_sleeves)
    previous = holdings[0]
    daily = np.empty((len(opens), 3))
    for t, target in enumerate(holdings):
        prior_close_nav = nav.sum()
        if t:
            nav *= opens[t, previous] / closes[t - 1, previous]
        open_nav = nav.sum()
        changed = target != previous if t else np.ones(n_sleeves, dtype=bool)
        sold = nav[changed].sum() if t else 0.0
        nav[changed] *= (1 - fee if t else 1) / (1 + fee)
        bought = nav[changed].sum()
        nav *= closes[t, target] / opens[t, target]
        daily[t] = (
            nav.sum() / prior_close_nav - 1,
            (sold + bought) / open_nav,
            changed.sum() * (2 if t else 1),
        )
        previous = target
    return pd.DataFrame(daily, index=dates, columns=["return", "turnover", "n_trades"])


def metrics(daily, sessions):
    returns = daily["return"]
    wealth = (1 + returns).cumprod().to_numpy()
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    volatility = returns.std(ddof=1)
    values = {
        "cagr": wealth[-1] ** (sessions / len(wealth)) - 1,
        "ann_vol": volatility * math.sqrt(sessions),
        "sharpe": returns.mean() / volatility * math.sqrt(sessions) if volatility > 0 else np.nan,
        "max_drawdown": (wealth / peak - 1).min(),
        "total_return": wealth[-1] - 1,
        "turnover": daily["turnover"].mean() * sessions,
    }
    return {
        **{key: float(value) if math.isfinite(value) else None for key, value in values.items()},
        "n_trades": int(daily["n_trades"].sum()),
    }


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    is_start, is_end = (pd.Timestamp(cfg["in_sample"][edge]) for edge in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(cfg["out_of_sample"][edge]) for edge in ("start", "end"))
    data_start = pd.Timestamp(cfg["data_start"])
    if (
        not (data_start < is_start <= is_end < oos_start <= oos_end)
        or oos_start != is_end + pd.Timedelta(days=1)
    ):
        raise ValueError("Use contiguous IS and OOS periods after the data start")

    benchmark = cfg["benchmark_ticker"]
    instruments = list(dict.fromkeys([*cfg["tickers"], cfg["cash_ticker"], benchmark]))
    bars = load_bars(instruments, data_start, oos_end, benchmark)
    dates = bars[benchmark].index
    first_session = dates.searchsorted(is_start)
    spans = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    for start, end in spans.values():
        if not ((dates >= start) & (dates <= end)).any():
            raise ValueError(f"No sessions in {start.date()} through {end.date()}")

    events, features = build_events(bars, cfg)
    holdings = walk_forward(events, features, dates, first_session, cfg)
    holdings["benchmark"] = np.full((len(dates) - first_session, 1), instruments.index(benchmark))
    opens, closes = (
        pd.DataFrame({ticker: bars[ticker][field] for ticker in instruments}).iloc[first_session:]
        for field in ("Open", "Close")
    )
    fee = cfg["cost_bps_per_side"] / 10_000
    report = {}
    for variant, positions in holdings.items():
        daily = simulate(opens, closes, positions, fee)
        report[variant] = {
            period: metrics(daily.loc[start:end], cfg["sessions_per_year"])
            for period, (start, end) in spans.items()
        }

    result = {
        "strategy": cfg["strategy"],
        "data_range": f"{opens.index[0].date()} to {opens.index[-1].date()}",
        **report["adaptive"],
        "benchmark": {"name": f"{benchmark} buy-and-hold", **report["benchmark"]},
        "fixed_horizon": {"sessions": cfg["fixed_horizon"], **report["fixed"]},
    }
    print(f"{cfg['strategy']} | {result['data_range']} | {cfg['cost_bps_per_side']:g} bps per side")
    names = {
        "adaptive": "Adaptive horizon",
        "fixed": f"Fixed {cfg['fixed_horizon']}-session horizon",
        "benchmark": result["benchmark"]["name"],
    }
    rows = {
        (names[variant], period.upper()): row
        for variant, parts in report.items()
        for period, row in parts.items()
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.names = ["Portfolio", "Period"]
    percent_columns = {"cagr", "ann_vol", "max_drawdown", "total_return", "turnover"}
    formats = {column: "{:.2%}".format for column in percent_columns}
    formats["sharpe"] = "{:.2f}".format
    print(table.to_string(formatters=formats, na_rep="n/a"))
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
