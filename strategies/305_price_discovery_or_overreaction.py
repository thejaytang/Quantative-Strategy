# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 305_price_discovery_or_overreaction\nuniverse: [EWJ, EWA, EWH, EWS, EWY]\nmarket_ticker: SPY\ndata_start: 2006-01-01\nin_sample:\n  start: 2007-01-01\n  end: 2023-06-30\nout_of_sample:\n  start: 2023-07-01\n  end: 2026-10-31\nparameters:\n  beta_sessions: 126\n  shock_sessions: 126\n  shock_quantile: 0.90\n  cost_bps_per_side: 5.0\n'
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
"""Simulate C96 using daily US ETF prices and next-session fills."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import time

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252


def validate_prices(prices, columns, start, end):
    prices = prices.loc[:, columns].sort_index().loc[start:end]
    values = prices.to_numpy()
    if prices.empty or not prices.index.is_unique or prices.index.hasnans:
        raise ValueError("Yahoo data must have dated, unique daily bars.")
    week = pd.Timedelta(days=7)
    if prices.index[0] > start + week or prices.index[-1] < end - week:
        raise ValueError("Yahoo data does not cover the requested start and end dates.")
    invalid = ~np.isfinite(values) | (values <= 0)
    if invalid.any():
        row, column = np.argwhere(invalid)[0]
        raise ValueError(f"Missing or invalid Yahoo price for {columns[column]} "
                         f"on {prices.index[row].date()}.")
    return prices


def load_prices(tickers, start, end):
    cache = ROOT / "data_cache"
    cache.mkdir(exist_ok=True)
    symbols = "_".join(tickers)
    filename = cache / f"yahoo_adjusted_{symbols}_{start.date()}_{end.date()}.csv"
    columns = [f"{field}_{ticker}" for field in ("Open", "Close") for ticker in tickers]
    if filename.exists():
        try:
            prices = pd.read_csv(filename, index_col="date", parse_dates=["date"])
            return validate_prices(prices, columns, start, end)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            print(f"Refreshing invalid price cache ({exc})", flush=True)

    with TemporaryDirectory(prefix="yf_", dir=cache) as yf_cache:
        yf.set_tz_cache_location(yf_cache)
        for attempt in range(3):
            try:
                prices = yf.download(
                    tickers, start=start.date().isoformat(),
                    end=(end + pd.Timedelta(days=1)).date().isoformat(),
                    auto_adjust=True, progress=False, threads=True, timeout=12,
                    multi_level_index=True,
                )
                prices.columns = [f"{field}_{ticker}" for field, ticker in prices.columns]
                prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
                prices = validate_prices(prices, columns, start, end)
                break
            except (OSError, KeyError, TypeError, ValueError) as exc:
                if attempt == 2:
                    raise RuntimeError(f"Unable to load complete Yahoo daily prices ({exc})") from exc
                print(f"Retrying Yahoo download ({exc})", flush=True)
                time.sleep(2 ** attempt)

    with NamedTemporaryFile(mode="w", prefix="prices_", suffix=".tmp", dir=cache,
                            delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        try:
            prices.to_csv(temp_file, index_label="date")
            temp_file.close()
            temp_path.replace(filename)
        finally:
            temp_path.unlink(missing_ok=True)
    return prices


def build_signals(session_returns, market, universe, parameters):
    spy_return = session_returns[market]
    beta_window = parameters["beta_sessions"]
    market_variance = spy_return.rolling(beta_window).var().replace(0.0, np.nan)
    covariance = session_returns[universe].rolling(beta_window).cov(spy_return)
    beta = covariance.div(market_variance, axis=0).shift(1)
    residuals = session_returns[universe] - beta.mul(spy_return, axis=0)

    ranks = residuals.rank(axis=1, method="first")
    eligible = residuals.notna().all(axis=1) & residuals.max(axis=1).gt(residuals.min(axis=1))
    fade = (ranks.eq(1).astype(float) - ranks.eq(len(universe)).astype(float)) * 0.5
    fade = fade.mul(eligible, axis=0)

    magnitude = spy_return.abs()
    cutoff = magnitude.rolling(parameters["shock_sessions"]).quantile(
        parameters["shock_quantile"]
    ).shift(1)
    triggered_fade = fade.mul(magnitude.gt(cutoff), axis=0)
    sign_rule = pd.DataFrame(1 / len(universe), index=fade.index, columns=universe)
    sign_rule = sign_rule.mul(np.sign(spy_return), axis=0)
    return triggered_fade, fade, sign_rule


def intraday_returns(close_signal, session_returns, cost_rate):
    # Signals are decided at the prior US close and filled at this US open.
    weights = close_signal.shift(1, fill_value=0.0)
    absolute = weights.abs()
    gross = (weights * session_returns).sum(axis=1)
    turnover = absolute.sum(axis=1) + (absolute * (1 + session_returns)).sum(axis=1)
    return pd.DataFrame({
        "gross_return": gross,
        "net_return": gross - cost_rate * turnover,
        "turnover": turnover,
        "trades": absolute.gt(0).sum(axis=1),
    })


def buy_and_hold(market_open, market_close, cost_rate):
    # Each report buys at the first open and sells at the last close.
    gross_factor = market_close.div(market_close.shift(1, fill_value=market_open.iloc[0]))
    net_factor = gross_factor.copy()
    net_factor.iloc[0] /= 1 + cost_rate
    net_factor.iloc[-1] *= 1 - cost_rate
    turnover = pd.Series(0.0, index=market_close.index)
    entry_weight = 1 / (1 + cost_rate)
    turnover.iloc[0] += entry_weight
    turnover.iloc[-1] += gross_factor.iloc[-1] * (entry_weight if len(turnover) == 1 else 1)
    trades = pd.Series(0, index=market_close.index)
    trades.iloc[-1] = 1
    return pd.DataFrame({"net_return": net_factor - 1, "turnover": turnover, "trades": trades})


def finite_float(value):
    return float(value) if np.isfinite(value) else None


def metrics(daily):
    returns = daily["net_return"]
    if returns.empty or not np.isfinite(returns).all() or returns.le(-1).any():
        raise ValueError("Performance requires nonempty, finite daily returns above -100%.")
    wealth = (1 + returns).cumprod()
    std = returns.std(ddof=1)
    max_drawdown = (wealth / wealth.cummax().clip(lower=1.0) - 1).min()
    return {
        "cagr": finite_float(wealth.iloc[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": finite_float(std * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": finite_float(returns.mean() * np.sqrt(SESSIONS_PER_YEAR) / std)
        if std > 0 else None,
        "max_drawdown": finite_float(max_drawdown),
        "total_return": finite_float(wealth.iloc[-1] - 1),
        "n_trades": int(daily["trades"].sum()),
        "turnover": finite_float(daily["turnover"].mean()),
    }


def print_results(reports):
    def display(value, fmt):
        return "n/a" if value is None else format(value, fmt)

    print(f"{'Portfolio':<20} {'Period':<6} {'CAGR':>9} {'Ann Vol':>9} {'Sharpe':>8} "
          f"{'Max DD':>9} {'Total':>9} {'Trades':>8} {'Turn/day':>9}")
    for name, periods in reports.items():
        for period, row in periods.items():
            print(f"{name:<20} {period.upper():<6} "
                  f"{display(row['cagr'], '.2%'):>9} {display(row['ann_vol'], '.2%'):>9} "
                  f"{display(row['sharpe'], '.2f'):>8} {display(row['max_drawdown'], '.2%'):>9} "
                  f"{display(row['total_return'], '.2%'):>9} {row['n_trades']:>8} "
                  f"{display(row['turnover'], '.2%'):>9}")


def main():
    with (ROOT / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    universe = config["universe"]
    market = config["market_ticker"]
    parameters = config["parameters"]
    start = pd.Timestamp(config["data_start"])
    is_start = pd.Timestamp(config["in_sample"]["start"])
    is_end = pd.Timestamp(config["in_sample"]["end"])
    oos_start = pd.Timestamp(config["out_of_sample"]["start"])
    oos_end = pd.Timestamp(config["out_of_sample"]["end"])
    periods = {"is": slice(is_start, is_end), "oos": slice(oos_start, oos_end),
               "full": slice(is_start, oos_end)}
    cost_rate = float(parameters["cost_bps_per_side"]) / 10_000
    if not start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Configure warm-up dates followed by ordered, nonoverlapping IS and OOS.")
    if len(universe) < 2 or len(set(universe)) != len(universe) or market in universe:
        raise ValueError("Use at least two distinct ETFs and a separate market ticker.")
    if parameters["beta_sessions"] < 2 or parameters["shock_sessions"] < 1:
        raise ValueError("Configure valid beta and shock lookbacks.")
    if not 0 < parameters["shock_quantile"] < 1 or not 0 <= cost_rate < 1:
        raise ValueError("Configure a quantile between 0 and 1 and valid transaction costs.")

    tickers = [*universe, market]
    prices = load_prices(tickers, start, oos_end)
    warmup_sessions = max(parameters["beta_sessions"], parameters["shock_sessions"]) + 1
    if (prices.index < is_start).sum() < warmup_sessions:
        raise ValueError("Move data_start earlier to warm up signals before the first IS open.")
    if any(prices.loc[period].empty for period in periods.values()):
        raise ValueError("Each report period must contain a US trading session.")

    open_prices = prices[[f"Open_{ticker}" for ticker in tickers]].set_axis(tickers, axis=1)
    close_prices = prices[[f"Close_{ticker}" for ticker in tickers]].set_axis(tickers, axis=1)
    session_returns = close_prices / open_prices - 1
    trigger, fade, sign = build_signals(session_returns, market, universe, parameters)
    strategy_name = "Shock fade"
    hold_name = f"{market} buy-and-hold"
    signals = {strategy_name: trigger, "All-day fade": fade, f"{market} sign (ETFs)": sign}
    simulations = {
        name: intraday_returns(weights, session_returns[universe], cost_rate)
        for name, weights in signals.items()
    }
    reports = {
        name: {label: metrics(daily.loc[period]) for label, period in periods.items()}
        for name, daily in simulations.items()
    }
    reports[hold_name] = {
        label: metrics(buy_and_hold(open_prices[market].loc[period],
                                    close_prices[market].loc[period], cost_rate))
        for label, period in periods.items()
    }

    dates = prices.loc[periods["full"]].index
    data_range = f"{dates[0].date()} to {dates[-1].date()}"
    print(f"{config['strategy']} | {data_range} | {cost_rate * 10_000:g} bps per side")
    print_results(reports)

    gap = open_prices[universe] / close_prices[universe].shift(1) - 1
    diagnostic = simulations[strategy_name].assign(
        overnight_gap=(trigger.shift(1) * gap).sum(axis=1, min_count=1)
    )
    print("\nShock-event diagnostics, grouped by next US session (gross means in bps)")
    print(f"{'Period':<6} {'Events':>8} {'Prior close to open':>20} {'Open to close':>16}")
    for label, period in periods.items():
        events = diagnostic.loc[period].query("trades > 0")
        gap_bps = 10_000 * events["overnight_gap"].mean()
        session_bps = 10_000 * events["gross_return"].mean()
        gap_text = f"{gap_bps:.2f}" if len(events) else "n/a"
        session_text = f"{session_bps:.2f}" if len(events) else "n/a"
        print(f"{label.upper():<6} {len(events):>8} {gap_text:>20} {session_text:>16}")

    hold = reports[hold_name]
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **reports[strategy_name],
        "benchmark": {
            "name": hold_name, **hold["full"], "is": hold["is"], "oos": hold["oos"],
            "controls": {name: reports[name] for name in signals if name != strategy_name},
        },
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
