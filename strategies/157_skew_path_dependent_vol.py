# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 157_skew_path_dependent_vol\ncategory: volatility\nuniverse: [SPY]\ndata_start: '1993-01-01'\nin_sample:\n  start: '2004-01-01'\n  end: '2015-12-31'\nout_of_sample:\n  start: '2016-01-01'\n  end: '2025-12-31'\nallocation:\n  drawdown_start: 0.10\n  drawdown_full: 0.25\n  duration_full_sessions: 63\n  recovery_start: 0.25\n  recovery_full: 0.75\n  min_weight_candidates: [0.25, 0.50, 0.75]\none_way_cost_bps: 5.0\nsessions_per_year: 252\n"
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
import math
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"


def load_prices(ticker, start, end):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{ticker}_{start.date()}_{end.date()}_adjusted.csv"
    if cache_file.exists():
        prices = pd.read_csv(cache_file, index_col="Date", parse_dates=["Date"])
    else:
        import yfinance as yf

        with TemporaryDirectory(dir=CACHE_DIR) as library_cache:
            yf.set_tz_cache_location(library_cache)
            prices = yf.Ticker(ticker).history(
                start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
                interval="1d", auto_adjust=True, actions=False, timeout=15,
            )
        if prices.empty:
            raise RuntimeError(f"Yahoo Finance returned no daily prices for {ticker}")
        prices = prices[["Open", "Close"]].copy()
        prices.index = prices.index.tz_localize(None).rename("Date")

    values = prices.to_numpy(dtype=float)
    if (prices.empty or not prices.index.is_monotonic_increasing
            or not prices.index.is_unique or prices.index.hasnans
            or not np.isfinite(values).all() or (values <= 0).any()):
        raise ValueError(f"Invalid adjusted daily prices for {ticker}")
    if prices.index[0] < start or prices.index[-1] > end or (
        end - prices.index[-1] > pd.Timedelta(days=7)
    ):
        raise ValueError(f"Daily prices do not cover the requested end date for {ticker}")

    if not cache_file.exists():
        with NamedTemporaryFile(mode="w", dir=CACHE_DIR, suffix=".tmp", delete=False) as file:
            temporary_path = Path(file.name)
        try:
            prices.to_csv(temporary_path)
            temporary_path.replace(cache_file)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices


def recovery_gain(drawdown):
    return drawdown / (1 - drawdown)


def allocation_stress(close, parameters):
    """Fraction of the allowed equity reduction indicated at each close."""
    drawdown = 1 - close / close.cummax()
    episode = drawdown.eq(0).cumsum()
    by_episode = drawdown.groupby(episode, sort=False)
    worst_depth = by_episode.cummax()
    underwater_days = by_episode.cumcount()

    start, full = (recovery_gain(parameters[key]) for key in ("drawdown_start", "drawdown_full"))
    depth_scale = ((recovery_gain(worst_depth) - start) / (full - start)).clip(0, 1)
    duration_scale = (underwater_days / parameters["duration_full_sessions"]).clip(0, 1)
    recovered = 1 - drawdown / worst_depth.where(worst_depth > 0)
    start, full = (parameters[key] for key in ("recovery_start", "recovery_full"))
    restored = ((recovered - start) / (full - start)).clip(0, 1).fillna(0)
    return depth_scale * (0.5 + 0.5 * duration_scale) * (1 - restored)


def market_relatives(prices):
    return pd.DataFrame({
        "overnight": prices["Open"] / prices["Close"].shift(),
        "intraday": prices["Close"] / prices["Open"],
    })


def simulate(relatives, targets, cost_rate):
    """Hold the old allocation overnight, rebalance at the open, mark at the close."""
    close_weight = 0.0
    rows = []
    for (overnight, intraday), target in zip(relatives.to_numpy(), targets, strict=True):
        overnight_factor = 1 + close_weight * (overnight - 1)
        pretrade_weight = close_weight * overnight / overnight_factor
        weight_change = target - pretrade_weight
        if abs(weight_change) < 1e-12:
            traded = 0.0
            target = pretrade_weight
        else:
            # Signed trade value as a share of pre-trade NAV, with target weights after fees.
            traded = abs(weight_change / (1 + math.copysign(cost_rate * target, weight_change)))
        intraday_factor = 1 + target * (intraday - 1)
        close_weight = target * intraday / intraday_factor
        net_return = overnight_factor * (1 - cost_rate * traded) * intraday_factor - 1
        rows.append((net_return, traded))
    return pd.DataFrame(rows, index=relatives.index, columns=["return", "turnover"])


def risk_stats(returns, sessions_per_year):
    daily_vol = returns.std()
    scale = math.sqrt(sessions_per_year)
    return daily_vol * scale, returns.mean() * scale / daily_vol if daily_vol > 0 else np.nan


def select_min_weight(relatives, stress_at_open, candidates, cost_rate, sessions_per_year):
    scores = []
    for floor in candidates:
        trial = simulate(relatives, 1 - (1 - floor) * stress_at_open, cost_rate)
        _, sharpe = risk_stats(trial["return"], sessions_per_year)
        if math.isfinite(sharpe):
            scores.append((sharpe, floor))
    if not scores:
        raise ValueError("No candidate had a defined in-sample Sharpe ratio")
    return max(scores)[1]


def geometric_capture(returns, reference, mask):
    denominator = np.expm1(np.log1p(reference[mask]).mean())
    if not math.isfinite(denominator) or denominator == 0:
        return np.nan
    return np.expm1(np.log1p(returns[mask]).mean()) / denominator


def metrics(path, reference, sessions_per_year):
    returns = path["return"]
    log_growth = np.log1p(returns)
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax().clip(lower=1) - 1
    underwater = drawdown < 0
    ann_vol, sharpe = risk_stats(returns, sessions_per_year)
    upside = geometric_capture(returns, reference, reference > 0)
    downside = geometric_capture(returns, reference, reference < 0)
    values = {
        "cagr": np.expm1(log_growth.mean() * sessions_per_year),
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "total_return": np.expm1(log_growth.sum()),
        "turnover": path["turnover"].sum(),
        "upside_capture": upside,
        "downside_capture": downside,
        "capture_asymmetry": upside - downside,
    }
    result = {key: float(value) if math.isfinite(value) else None for key, value in values.items()}
    result["n_trades"] = int(path["turnover"].gt(0).sum())
    result["max_underwater_sessions"] = int(underwater.groupby((~underwater).cumsum()).sum().max())
    return result


def print_report(results):
    header = ("Period", "Portfolio", "CAGR", "Ann vol", "Sharpe", "Max DD", "Total return",
              "Turnover", "Trades", "Underwater d", "Up capture", "Down capture")
    layout = "{:<12} {:<16} {:>8} {:>8} {:>7} {:>8} {:>12} {:>10} {:>7} {:>12} {:>11} {:>12}"
    print(layout.format(*header))
    fields = (
        ("cagr", ".2%"), ("ann_vol", ".2%"), ("sharpe", ".2f"), ("max_drawdown", ".2%"),
        ("total_return", ".2%"), ("turnover", ".2%"), ("n_trades", "d"),
        ("max_underwater_sessions", "d"), ("upside_capture", ".2%"), ("downside_capture", ".2%"),
    )
    for period in ("is", "oos", "full"):
        for label, row in (("Strategy", results[period]),
                           ("Buy and hold", results["benchmark"][period])):
            formatted = [format(row[key], spec) if row[key] is not None else "n/a" for key, spec in fields]
            print(layout.format(period.upper(), label, *formatted))


def main():
    with (ROOT / "config.yaml").open() as file:
        config = yaml.safe_load(file)
    ticker, = config["universe"]
    parameters = config["allocation"]
    cost_rate = config["one_way_cost_bps"] / 10_000
    sessions_per_year = config["sessions_per_year"]
    data_start = pd.Timestamp(config["data_start"])
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not data_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Expected warm-up followed by ordered, disjoint IS and OOS dates")
    if not (0 <= parameters["drawdown_start"] < parameters["drawdown_full"] < 1
            and 0 <= parameters["recovery_start"] < parameters["recovery_full"] <= 1
            and parameters["duration_full_sessions"] > 0
            and all(0 <= floor <= 1 for floor in parameters["min_weight_candidates"])
            and 0 <= cost_rate < 1 and sessions_per_year > 0):
        raise ValueError("Invalid allocation, transaction cost, or annualization parameters")

    prices = load_prices(ticker, data_start, oos_end)
    relatives = market_relatives(prices).loc[is_start:oos_end]
    if relatives.loc[:is_end].empty or relatives.loc[oos_start:].empty or relatives.isna().any().any():
        raise ValueError("Need warm-up prices and at least one session in both IS and OOS")
    if not relatives.loc[is_end + pd.Timedelta(days=1):oos_start - pd.Timedelta(days=1)].empty:
        raise ValueError("IS and OOS must cover adjacent trading sessions")

    is_stress = allocation_stress(prices.loc[:is_end, "Close"], parameters).shift().loc[is_start:is_end]
    floor = select_min_weight(relatives.loc[:is_end], is_stress, parameters["min_weight_candidates"],
                              cost_rate, sessions_per_year)
    stress_at_open = allocation_stress(prices["Close"], parameters).shift().reindex(relatives.index)
    strategy = simulate(relatives, 1 - (1 - floor) * stress_at_open, cost_rate)
    benchmark = simulate(relatives, np.ones(len(relatives)), cost_rate)
    windows = {"is": slice(is_start, is_end), "oos": slice(oos_start, oos_end),
               "full": slice(is_start, oos_end)}

    results = {
        "strategy": config["strategy"],
        "data_range": f"{strategy.index[0].date()} to {strategy.index[-1].date()}",
    }
    for name, dates in windows.items():
        results[name] = metrics(strategy.loc[dates], benchmark.loc[dates, "return"], sessions_per_year)
    results["benchmark"] = {"name": f"{ticker} buy and hold"}
    for name, dates in windows.items():
        results["benchmark"][name] = metrics(benchmark.loc[dates], benchmark.loc[dates, "return"],
                                             sessions_per_year)

    print(f"{config['strategy']} | {ticker} | {results['data_range']}")
    print(f"IS {is_start.date()} to {is_end.date()} | OOS {oos_start.date()} to {oos_end.date()}")
    print(f"IS-selected minimum equity weight {floor:.0%} | {config['one_way_cost_bps']:g} bps one way")
    print_report(results)
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
