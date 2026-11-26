# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 110_cointegration_etf_pairs\ndata_start: '2020-11-01'\nin_sample:\n  start: '2022-01-01'\n  end: '2023-12-31'\nout_of_sample:\n  start: '2024-01-01'\n  end: '2026-10-31'\n\n# The first ticker of each pair is the dependent series in the regression.\npairs:\n  - [SPY, IVV]\n  - [QQQ, QQQM]\n  - [IWM, VTWO]\n  - [XLE, VDE]\n  - [XLF, VFH]\n  - [XLK, VGT]\nbenchmark_ticker: SPY\n\nformation_sessions: 252\ncointegration_p_max: 0.05\nentry_z: 2.0\nexit_z: 0.0\nmax_holding_sessions: 10\ngross_target: 1.0\n\ncosts:\n  transaction_bps: 5.0 # Each dollar traded on each leg, including rehedges\n  short_borrow_bps_annual: 30.0 # Each dollar of short exposure, calendar-day basis\n  transaction_sensitivity_bps: [0.0, 15.0]\n  short_borrow_sensitivity_bps_annual: 100.0\n"
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
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from statsmodels.tsa.adfvalues import mackinnonp
from statsmodels.tsa.stattools import adfuller


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
BPS = 10_000
PERIODS = ("is", "oos", "full")


def validate_prices(data, tickers, start, end):
    prices = data.loc[:, pd.MultiIndex.from_product([("Open", "Close"), tickers])]
    if prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Expected a nonempty, strictly increasing daily price index")
    dates = prices.index
    tolerance = pd.Timedelta(days=7)
    if not (start <= dates[0] <= start + tolerance and end - tolerance <= dates[-1] <= end):
        raise ValueError(f"Yahoo Finance returned incomplete coverage: {dates[0]} to {dates[-1]}")
    values = prices.to_numpy(dtype=float)
    invalid = ~np.isfinite(values) | (values <= 0)
    if invalid.any():
        row, column = np.argwhere(invalid)[0]
        raise ValueError(f"Missing or invalid {prices.columns[column]} at {dates[row].date()}")
    return prices


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    ticker_key = hashlib.sha256(" ".join(sorted(tickers)).encode()).hexdigest()[:12]
    cache_file = CACHE / f"yahoo_adjusted_{start.date()}_{end.date()}_{ticker_key}.csv"
    if cache_file.exists():
        try:
            cached = pd.read_csv(cache_file, header=[0, 1], index_col=0, parse_dates=True)
            return validate_prices(cached, tickers, start, end)
        except (KeyError, ValueError, OSError) as exc:
            print(f"Refreshing invalid price cache ({exc})", flush=True)

    with TemporaryDirectory(dir=CACHE, prefix="yfinance-") as downloader_cache:
        yf.set_tz_cache_location(downloader_cache)
        for _ in range(2):
            downloaded = yf.download(
                tickers, start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(), interval="1d",
                auto_adjust=True, ignore_tz=True, progress=False, timeout=10,
            )
            try:
                prices = validate_prices(downloaded, tickers, start, end)
                break
            except (KeyError, ValueError) as exc:
                error = exc
        else:
            raise RuntimeError("Could not obtain a complete adjusted ETF price panel from Yahoo Finance") from error

    with NamedTemporaryFile(mode="w", dir=CACHE, suffix=".tmp", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        try:
            prices.to_csv(temp_file)
            temp_file.close()
            temp_path.replace(cache_file)
        finally:
            temp_path.unlink(missing_ok=True)
    return prices


def monthly_signals(closes, index, config, *, distance):
    """Scores and eligible betas indexed by execution session, using prior closes."""
    formation = config["formation_sessions"]
    months = index.to_period("M")
    boundaries = np.r_[0, np.flatnonzero(months[1:] != months[:-1]) + 1, len(index)]
    refit = np.zeros(len(index), dtype=bool)
    refit[boundaries[1:-1]] = True
    scores = np.full(closes.shape[:2], np.nan)
    betas = np.full_like(scores, np.nan)
    prices = closes if distance else np.log(closes)

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start < formation:
            continue
        for pair in range(closes.shape[1]):
            window = prices[start - formation:start, pair]
            prior_closes = prices[start - 1:end - 1, pair]
            if distance:
                # Normalize both series to 1 at the start of formation.
                scaled_window = window / window[0]
                spread = scaled_window[:, 0] - scaled_window[:, 1]
                scaled_signal = prior_closes / window[0]
                signal_spread = scaled_signal[:, 0] - scaled_signal[:, 1]
                beta = 1.0
            else:
                log_means = window.mean(axis=0)
                centered = window - log_means
                y, x = centered.T
                predictor_ss = x @ x
                if predictor_ss == 0:
                    continue
                beta = (x @ y) / predictor_ss
                if beta <= 0:
                    continue
                # Centering is equivalent to a regression with an intercept.
                spread = y - beta * x
                signal_y, signal_x = (prior_closes - log_means).T
                signal_spread = signal_y - beta * signal_x

            mean, sigma = spread.mean(), spread.std()
            if sigma <= 100 * np.finfo(float).eps:
                continue
            if not distance:
                # Engle–Granger step two. Standardize for near-identical ETFs and
                # use the two-series MacKinnon p-value, not the ordinary ADF p-value.
                stat = adfuller((spread - mean) / sigma, regression="n", autolag="AIC",
                                result_object=False)[0]
                p_value = mackinnonp(stat, regression="c", N=2)
                if not p_value < config["cointegration_p_max"]:
                    continue

            scores[start:end, pair] = (signal_spread - mean) / sigma
            betas[start:end, pair] = beta
    return scores, betas, refit


def simulate(index, opens, closes, signals, first, config, transaction_bps, borrow_bps):
    scores, betas, refit = signals
    count = opens.shape[1]
    units = np.zeros((count, 2))
    direction = np.zeros(count)
    entered_at = np.zeros(count, dtype=int)
    equity = np.ones(len(index) - first)
    entries = np.zeros(len(equity), dtype=int)
    turnover = np.zeros(len(equity))
    day_gaps = np.diff(index.values).astype("timedelta64[D]").astype(int)
    fee_rate = transaction_bps / BPS
    daily_borrow_rate = borrow_bps / (BPS * 365)
    cash = 1.0

    for i in range(first, len(index)):
        bar = i - first
        marks = opens[i]
        if bar > 0:
            short_notional = (-np.minimum(units, 0) * closes[i - 1]).sum()
            cash -= short_notional * daily_borrow_rate * day_gaps[i - 1]
        eligible = np.isfinite(betas[i])
        active = direction != 0
        mean_reversion = direction * scores[i] >= -config["exit_z"]
        timed_out = i - entered_at >= config["max_holding_sessions"]
        exiting = active & (~eligible | mean_reversion | timed_out)
        entering = ~active & eligible & (np.abs(scores[i]) > config["entry_z"])
        direction[exiting] = 0
        direction[entering] = -np.sign(scores[i, entering])
        entered_at[entering] = i
        entries[bar] = entering.sum()
        sizing = entering | (refit[i] & (direction != 0))

        if (exiting | sizing).any():
            open_nav = cash + (units * marks).sum()
            if open_nav <= 0:
                raise RuntimeError(f"Portfolio equity exhausted at {index[i].date()} open")
            target = units.copy()
            target[exiting] = 0
            if sizing.any():
                sleeve = open_nav * config["gross_target"] / eligible.sum()
                beta = betas[i, sizing]
                first_leg = direction[sizing] * sleeve / (1.0 + beta)
                leg_values = np.column_stack((first_leg, -beta * first_leg))
                target[sizing] = leg_values / marks[sizing]
            traded = (target - units) * marks
            traded_gross = np.abs(traded).sum()
            turnover[bar] = traded_gross / open_nav
            cash -= traded.sum() + traded_gross * fee_rate
            units = target

        equity[bar] = cash + (units * closes[i]).sum()
        if equity[bar] <= 0:
            raise RuntimeError(f"Portfolio equity exhausted at {index[i].date()} close")

    return performance_frame(index[first:], equity, entries, turnover)


def performance_frame(index, equity, entries, turnover):
    return pd.DataFrame({
        "return": equity / np.r_[1.0, equity[:-1]] - 1.0,
        "entries": entries,
        "turnover": turnover,
    }, index=index)


def buy_and_hold(prices, first, transaction_bps):
    data = prices.iloc[first:]
    units = 1.0 / (data["Open"].iloc[0] * (1 + transaction_bps / BPS))
    entries = np.zeros(len(data), dtype=int)
    turnover = np.zeros(len(data))
    entries[0] = 1
    turnover[0] = units * data["Open"].iloc[0]
    return performance_frame(data.index, units * data["Close"].to_numpy(), entries, turnover)


def metrics(data):
    returns = data["return"].to_numpy()
    path = np.r_[1.0, np.cumprod(1.0 + returns)]
    daily_vol = returns.std(ddof=1) if len(returns) > 1 else np.nan
    stats = {
        "cagr": path[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1.0,
        "ann_vol": daily_vol * np.sqrt(SESSIONS_PER_YEAR),
        "sharpe": returns.mean() / daily_vol * np.sqrt(SESSIONS_PER_YEAR) if daily_vol > 0 else np.nan,
        "max_drawdown": (path / np.maximum.accumulate(path) - 1.0).min(),
        "total_return": path[-1] - 1.0,
        "turnover": data["turnover"].sum(),
    }
    return {
        **{key: float(value) if np.isfinite(value) else None for key, value in stats.items()},
        "n_trades": int(data["entries"].sum()),
    }


def summarize(data, split):
    return dict(zip(PERIODS, (metrics(data.loc[data.index < split]),
                              metrics(data.loc[data.index >= split]), metrics(data))))


def print_table(rows):
    print(f"{'Model':<34} {'Period':<6} {'CAGR':>9} {'Ann. vol':>9} {'Sharpe':>8} "
          f"{'Max DD':>9} {'Return':>9} {'Trades':>7} {'Turnover':>9}")
    for label, summaries in rows:
        for period, stats in summaries.items():
            cells = [f"{stats[key]:{fmt}}" if stats[key] is not None else "n/a" for key, fmt in (
                ("cagr", ".2%"), ("ann_vol", ".2%"), ("sharpe", ".3f"),
                ("max_drawdown", ".2%"), ("total_return", ".2%"),
                ("n_trades", "d"), ("turnover", ".2f"),
            )]
            print(f"{label:<34} {period.upper():<6} {cells[0]:>9} {cells[1]:>9} {cells[2]:>8} "
                  f"{cells[3]:>9} {cells[4]:>9} {cells[5]:>7} {cells[6]:>9}")


def main():
    with (ROOT / "config.yaml").open() as config_file:
        config = yaml.safe_load(config_file)
    start = pd.Timestamp(config["data_start"])
    is_start = pd.Timestamp(config["in_sample"]["start"])
    is_end = pd.Timestamp(config["in_sample"]["end"])
    oos_start = pd.Timestamp(config["out_of_sample"]["start"])
    oos_end = pd.Timestamp(config["out_of_sample"]["end"])
    if not (start < is_start <= is_end < oos_start <= oos_end
            and oos_start - is_end == pd.Timedelta(days=1)):
        raise ValueError("Expected chronological, contiguous IS and OOS date ranges after warmup")
    pairs = config["pairs"]
    symbols = [ticker for pair in pairs for ticker in pair]
    if not pairs or any(len(pair) != 2 for pair in pairs) or len(set(symbols)) != len(symbols):
        raise ValueError("Expected disjoint pairs, each with two distinct tickers")
    benchmark = config["benchmark_ticker"]
    prices = load_prices(list(dict.fromkeys([*symbols, benchmark])), start, oos_end)
    index = prices.index
    first = index.searchsorted(is_start)
    split = index.searchsorted(oos_start)
    if not first < split < len(index):
        raise ValueError("Both IS and OOS must contain price observations")
    first_month = index.to_period("M").searchsorted(index[first].to_period("M"))
    if first_month < config["formation_sessions"]:
        raise ValueError("Insufficient formation history before the first evaluation month")

    opens = prices["Open"][symbols].to_numpy().reshape(len(index), len(pairs), 2)
    closes = prices["Close"][symbols].to_numpy().reshape(opens.shape)
    cointegration = monthly_signals(closes, index, config, distance=False)
    distance = monthly_signals(closes, index, config, distance=True)
    costs = config["costs"]
    fee = costs["transaction_bps"]
    borrow = costs["short_borrow_bps_annual"]

    def evaluate(signals, transaction_bps, borrow_bps, pair=slice(None)):
        scores, betas, refit = signals
        data = simulate(index, opens[:, pair], closes[:, pair], (scores[:, pair], betas[:, pair], refit),
                        first, config, transaction_bps, borrow_bps)
        return summarize(data, oos_start)

    primary = evaluate(cointegration, fee, borrow)
    benchmark_prices = prices.xs(benchmark, axis=1, level=1)
    benchmark_summary = summarize(buy_and_hold(benchmark_prices, first, fee), oos_start)
    rows = [
        (f"Engle-Granger ({fee:g}/{borrow:g} bps)", primary),
        (f"Distance ({fee:g}/{borrow:g} bps)", evaluate(distance, fee, borrow)),
        (f"{benchmark} buy & hold ({fee:g} bps)", benchmark_summary),
        ("Engle-Granger gross (0/0 bps)", evaluate(cointegration, 0.0, 0.0)),
    ]
    for stress_fee in costs["transaction_sensitivity_bps"]:
        rows.append((f"Engle-Granger ({stress_fee:g}/{borrow:g} bps)",
                     evaluate(cointegration, stress_fee, borrow)))
    stress_borrow = costs["short_borrow_sensitivity_bps_annual"]
    rows.append((f"Engle-Granger ({fee:g}/{stress_borrow:g} bps)",
                 evaluate(cointegration, fee, stress_borrow)))

    date_range = f"{index[first].date()} to {index[-1].date()}"
    print(f"{config['strategy']} | Yahoo Finance adjusted bars | {date_range}")
    print(f"IS {index[first].date()} to {index[split - 1].date()} | "
          f"OOS {index[split].date()} to {index[-1].date()}")
    print("Costs in labels are transaction bps per leg / annual short borrow bps. Turnover is in NAV multiples.")
    print_table(rows)
    print(f"\nStandalone pair results ({fee:g}/{borrow:g} bps)")
    pair_rows = []
    for j, pair in enumerate(pairs):
        label = "/".join(pair)
        columns = slice(j, j + 1)
        pair_rows.extend([
            (f"{label} Engle-Granger", evaluate(cointegration, fee, borrow, columns)),
            (f"{label} distance", evaluate(distance, fee, borrow, columns)),
        ])
    print_table(pair_rows)

    result = {
        "strategy": config["strategy"], "data_range": date_range,
        **primary, "benchmark": benchmark_summary["full"],
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
