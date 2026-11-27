# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 482_dynamic_inclusion_and_bounded_multi_fact\nuniverse: [XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY]\ncalendar_ticker: SPY\nrisk_free: BIL\ndates:\n  history_start: 2008-01-01\n  start: 2010-01-01\n  is_end: 2019-12-31\n  end: 2025-12-31\nsignals:\n  liquidity_lookback_sessions: 60\n  volatility_lookback_sessions: 60\n  momentum_lookback_sessions: 252\n  momentum_skip_sessions: 21\n  min_eligible: 5\n  momentum_weight: 0.50\n  lower_weight_multiple: 0.70\n  upper_weight_multiple: 1.30\n  is_grid:\n    min_average_dollar_volume: [5000000, 10000000]\n    max_annualized_volatility: [0.40, 0.55]\n    tilt_strength: [0.35, 0.65]\nexecution:\n  signal_lag_sessions: 1\n  cash_buffer_fraction: 0.03\n  cost_bps: 2.0\n  slippage_bps: 3.0\n'
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
import gzip
import hashlib
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
PRICE_FIELDS = ["Adj Close", "Close", "Volume"]
GRID_KEYS = ("min_average_dollar_volume", "max_annualized_volatility", "tilt_strength")


class NyseHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def check_session_prices(prices, sessions, calendar_ticker):
    if (not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_unique
            or not prices.index.is_monotonic_increasing):
        raise RuntimeError("Yahoo Finance dates must be unique and increasing.")
    if not prices.index.equals(sessions):
        missing = sessions.difference(prices.index).strftime("%Y-%m-%d").tolist()
        unexpected = prices.index.difference(sessions).strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"NYSE session mismatch: missing {missing}, unexpected {unexpected}.")
    calendar_close = prices["Adj Close", calendar_ticker]
    if not (np.isfinite(calendar_close) & calendar_close.gt(0)).all():
        raise RuntimeError(f"Missing or invalid {calendar_ticker} closes in the session calendar.")


def load_prices(symbols, start, end, calendar_ticker):
    extra_closures = pd.to_datetime(["2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])
    sessions = pd.date_range(start, end, freq=CustomBusinessDay(calendar=NyseHolidays()))
    sessions = sessions.difference(extra_closures)
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    key = hashlib.sha256(f"v1|{start}|{end}|{','.join(symbols)}".encode()).hexdigest()[:16]
    cache_file = cache_dir / f"yahoo_{key}.csv.gz"
    if cache_file.exists():
        with gzip.open(cache_file, "rt", encoding="utf-8") as source:
            fetched_at = source.readline().strip()
            prices = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True)
        check_session_prices(prices, sessions, calendar_ticker)
        return prices, fetched_at

    with TemporaryDirectory(dir=cache_dir) as scratch:
        yf.set_tz_cache_location(scratch)
        for _ in range(2):
            downloaded = yf.download(
                symbols, start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
                interval="1d", auto_adjust=False, progress=False, threads=4, timeout=12,
            )
            prices = downloaded.reindex(columns=pd.MultiIndex.from_product([PRICE_FIELDS, symbols]))
            missing = prices.columns[prices.isna().all()].tolist()
            if not missing:
                break
    if missing:
        raise RuntimeError(f"Yahoo Finance returned empty fields for {missing}")

    check_session_prices(prices, sessions, calendar_ticker)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(dir=cache_dir, suffix=".csv.gz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8") as output:
            output.write(fetched_at + "\n")
            prices.to_csv(output)
        temporary_path.replace(cache_file)
    finally:
        temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def monthly_signals(close, returns, dollar_volume, evaluation_dates, model, lag):
    lookback = model["momentum_lookback_sessions"]
    skip = model["momentum_skip_sessions"]
    has_history = close.rolling(lookback + 1).count().eq(lookback + 1)
    momentum = (close.shift(skip) / close.shift(lookback) - 1).where(has_history)
    volatility = (returns.rolling(model["volatility_lookback_sessions"]).std()
                  * np.sqrt(SESSIONS_PER_YEAR))
    liquidity = dollar_volume.rolling(model["liquidity_lookback_sessions"]).mean()

    months = close.index.to_period("M")
    execution_dates = close.index[~months.duplicated()].intersection(evaluation_dates)
    signals = {}
    for execution_date in execution_dates:
        signal_index = close.index.get_loc(execution_date) - lag
        if signal_index < lookback:
            raise ValueError("Increase the history before the first evaluation month.")
        signals[execution_date] = pd.DataFrame({
            "momentum": momentum.iloc[signal_index],
            "volatility": volatility.iloc[signal_index],
            "liquidity": liquidity.iloc[signal_index],
        })
    return signals


def bounded_multiples(raw, lower, upper):
    # Scale and clip simultaneously so the final multiples retain their bounds.
    left, right = 0.0, upper / raw.min()
    for _ in range(48):
        scale = (left + right) / 2
        multiples = np.clip(scale * raw, lower, upper)
        if multiples.mean() < 1:
            left = scale
        else:
            right = scale
    return np.clip(((left + right) / 2) * raw, lower, upper)


def build_targets(signals, model, params):
    tilted_targets = {}
    equal_targets = {}
    eligible_counts = []
    for date, features in signals.items():
        eligible = (features["momentum"].notna()
                    & features["liquidity"].ge(params["min_average_dollar_volume"])
                    & features["volatility"].le(params["max_annualized_volatility"]))
        count = int(eligible.sum())
        eligible_counts.append(count)
        equal = np.zeros(len(features))
        tilted = equal.copy()
        if count >= model["min_eligible"]:
            chosen = features.loc[eligible]
            momentum_rank = chosen["momentum"].rank()
            low_volatility_rank = (-chosen["volatility"]).rank()
            mix = model["momentum_weight"]
            rank = mix * momentum_rank + (1 - mix) * low_volatility_rank
            score = 2 * (rank.to_numpy() - 1) / (count - 1) - 1
            raw = 1 + params["tilt_strength"] * score
            multiples = bounded_multiples(raw, model["lower_weight_multiple"],
                                          model["upper_weight_multiple"])
            equal[eligible.to_numpy()] = 1 / count
            tilted[eligible.to_numpy()] = multiples / count
        tilted_targets[date] = tilted
        equal_targets[date] = equal
    return tilted_targets, equal_targets, eligible_counts


def simulate(close, dates, targets, friction, cash_buffer):
    positions = np.zeros(len(close.columns))
    cash = previous_value = 1.0
    daily_results = []
    previous_closes = close.shift(1).loc[dates].to_numpy()
    closing_prices = close.loc[dates].to_numpy()
    for date, previous_close, fill_close in zip(dates, previous_closes, closing_prices):
        next_positions = positions
        target = targets.get(date)
        if target is not None:
            selected = target > 0
            if not np.isfinite(previous_close[selected]).all():
                missing = list(close.columns[selected & ~np.isfinite(previous_close)])
                raise RuntimeError(f"Missing sizing prices before {date.date()}: {missing}")
            budget = previous_value * (1 - cash_buffer)
            next_positions = np.zeros_like(positions)
            next_positions[selected] = budget * target[selected] / previous_close[selected]

        changes = next_positions - positions
        required = (positions > 0) | (next_positions > 0)
        if not np.isfinite(fill_close[required]).all():
            missing = list(close.columns[required & ~np.isfinite(fill_close)])
            raise RuntimeError(f"Missing closing prices on {date.date()}: {missing}")
        traded = changes[required] * fill_close[required]
        traded_value = np.abs(traded).sum()
        value_before_trade = cash + positions[required] @ fill_close[required]
        cash_after = cash - traded.sum() - friction * traded_value
        if cash_after < 0:
            raise RuntimeError(f"Insufficient cash for fixed quantities on {date.date()}: {cash_after:.8g}")
        value = cash_after + next_positions[required] @ fill_close[required]
        daily_results.append((value / previous_value - 1, traded_value / (2 * value_before_trade),
                              np.count_nonzero(changes)))
        previous_value, cash, positions = value, cash_after, next_positions
    return pd.DataFrame(daily_results, index=dates,
                        columns=["return", "turnover", "n_trades"])


def sharpe(returns):
    deviation = returns.std(ddof=1)
    if not np.isfinite(deviation) or deviation < 1e-12:
        return None
    return float(np.sqrt(SESSIONS_PER_YEAR) * returns.mean() / deviation)


def metrics(daily_results, risk_free):
    returns = daily_results["return"]
    growth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    years = len(returns) / SESSIONS_PER_YEAR
    return {
        "cagr": float(growth[-1] ** (1 / years) - 1),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(returns - risk_free),
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def select_params(signals, close, dates, risk_free, model, friction, cash_buffer):
    candidates = product(*(model["is_grid"][key] for key in GRID_KEYS))
    best_params, best_score = None, -np.inf
    scores = []
    for values in candidates:
        params = dict(zip(GRID_KEYS, values))
        targets, _, eligible_counts = build_targets(signals, model, params)
        daily_results = simulate(close, dates, targets, friction, cash_buffer)
        score = sharpe(daily_results["return"] - risk_free)
        scores.append({"parameters": params, "sharpe_excess": score,
                       "monthly_eligible_counts": eligible_counts})
        score_for_selection = score if score is not None else -np.inf
        if best_params is None or score_for_selection > best_score:
            best_params, best_score = params, score_for_selection
    if best_params is None:
        raise ValueError("The in-sample grid has no candidates.")
    return best_params, scores


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    history_start, start, split, end = (pd.Timestamp(config["dates"][key])
                                       for key in ("history_start", "start", "is_end", "end"))
    assets = config["universe"]
    calendar_ticker, risk_free_ticker = config["calendar_ticker"], config["risk_free"]
    symbols = [*assets, calendar_ticker, risk_free_ticker]
    model, execution = config["signals"], config["execution"]
    lag = execution["signal_lag_sessions"]
    cost, slippage = execution["cost_bps"], execution["slippage_bps"]
    friction = (cost + slippage) / 10_000
    cash_buffer = execution["cash_buffer_fraction"]
    if not history_start < start < split < end or len(set(symbols)) != len(symbols):
        raise ValueError("Use ascending dates and distinct tickers.")
    if risk_free_ticker != "BIL":
        raise ValueError("The excess return calculation expects BIL as the risk-free proxy.")
    if not isinstance(lag, int) or lag < 1 or min(cost, slippage) < 0 or friction >= 0.1:
        raise ValueError("Use a positive session lag and nonnegative costs below 1000 bps in total.")
    if not 0 < cash_buffer < 1:
        raise ValueError("The cash buffer must be a positive fraction below one.")
    if not 2 <= model["min_eligible"] <= len(assets):
        raise ValueError("The breadth floor must be between two and the size of the universe.")
    if not 0 <= model["momentum_skip_sessions"] < model["momentum_lookback_sessions"]:
        raise ValueError("The momentum skip must be shorter than the lookback.")
    if not 0 < model["lower_weight_multiple"] <= 1 <= model["upper_weight_multiple"]:
        raise ValueError("Weight multiples must bracket one and have a positive lower bound.")
    if not 0 <= model["momentum_weight"] <= 1 or any(
        not 0 <= value < 1 for value in model["is_grid"]["tilt_strength"]
    ):
        raise ValueError("Use a convex factor mix and tilt strengths between zero and one.")

    prices, fetched_at = load_prices(symbols, history_start, end, calendar_ticker)
    adjusted_close = prices["Adj Close"]
    adjusted_close = adjusted_close.where(np.isfinite(adjusted_close) & adjusted_close.gt(0))
    close = adjusted_close[assets]
    returns = close.pct_change(fill_method=None)
    dollar_volume = prices["Close"][assets].where(prices["Close"][assets].gt(0)) * (
        prices["Volume"][assets].where(prices["Volume"][assets].gt(0))
    )
    evaluation_dates = close.loc[start:end].index
    is_dates = evaluation_dates[evaluation_dates <= split]
    oos_dates = evaluation_dates[evaluation_dates > split]
    if min(len(is_dates), len(oos_dates)) < 2:
        raise ValueError("Both evaluation periods need at least two sessions.")
    risk_free = adjusted_close[risk_free_ticker].pct_change(fill_method=None).loc[evaluation_dates]
    if not np.isfinite(risk_free).all():
        raise RuntimeError(f"Missing {risk_free_ticker} returns in the evaluation period.")

    signals = monthly_signals(close, returns, dollar_volume, evaluation_dates, model, lag)
    is_signals = {date: values for date, values in signals.items() if date <= split}
    params, scores = select_params(is_signals, close.loc[:is_dates[-1]], is_dates,
                                   risk_free.loc[is_dates], model, friction, cash_buffer)
    strategy_targets, benchmark_targets, _ = build_targets(signals, model, params)
    strategy_results = {}
    benchmark_results = {}
    for period, dates in (("is", is_dates), ("oos", oos_dates)):
        strategy_results[period] = simulate(close, dates, strategy_targets, friction, cash_buffer)
        benchmark_results[period] = simulate(close, dates, benchmark_targets, friction, cash_buffer)
    strategy_results["full"] = pd.concat([strategy_results["is"], strategy_results["oos"]])
    benchmark_results["full"] = pd.concat([benchmark_results["is"], benchmark_results["oos"]])

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance Adj Close total-return prices and units (auto_adjust=False), "
          "liquidity uses Close x Volume")
    print(f"PARAM_COMBOS_TRIED: {len(scores)}")
    print(f"IS_SELECTED_PARAMETERS: {json.dumps(params)}")
    for score in scores:
        print(f"IS_GRID: {json.dumps(score, allow_nan=False)}")
    result = {"strategy": config["strategy"], "data_range": {
        "start": evaluation_dates[0].date().isoformat(),
        "end": evaluation_dates[-1].date().isoformat(),
        "is_end": is_dates[-1].date().isoformat(),
        "oos_start": oos_dates[0].date().isoformat(),
    }}
    for period, dates in (("is", is_dates), ("oos", oos_dates), ("full", evaluation_dates)):
        result[period] = metrics(strategy_results[period], risk_free.loc[dates])
        benchmark = metrics(benchmark_results[period], risk_free.loc[dates])
        print(f"{period.upper()}: {json.dumps(result[period], allow_nan=False)}")
        print(f"BENCHMARK DYNAMIC_EQUAL_WEIGHT {period.upper()}: "
              f"{json.dumps(benchmark, allow_nan=False)}")
        if period == "full":
            result["benchmark"] = benchmark
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
