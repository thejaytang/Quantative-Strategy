# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 192_downside_upside_rv_forecast\nuniverse:\n  risk_tickers: [DBC, GLD, IEF]\n  reserve_ticker: BIL\ndata_start: 2008-01-01\nin_sample:\n  start: 2014-01-01\n  end: 2019-12-31\nout_of_sample:\n  start: 2020-01-01\n  end: 2025-12-31\nmodel:\n  lookback_years: 5\n  long_window_weeks: 4\n  variance_floor: 1.0e-8\n  ridge_alpha: 1.0\nallocation:\n  downside_share_threshold: 0.65\n  reduced_sleeve_fraction: 0.5\ncost_bps: 5.0\ntrading_days_per_year: 252\n'
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
"""C58 weekly downside semivariance allocation backtest."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
SIDES = ("up", "down")


def load_prices(tickers, start, end):
    cache = HERE / "data_cache"
    cache.mkdir(exist_ok=True)
    path = cache / f"yahoo_adjusted_{'_'.join(tickers)}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    columns = pd.MultiIndex.from_product([["Open", "Close"], tickers])
    cached = path.exists()
    if cached:
        prices = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True, float_precision="round_trip")
    else:
        with TemporaryDirectory(prefix=".yahoo-", dir=cache) as session_cache:
            yf.set_tz_cache_location(session_cache)
            prices = yf.download(
                tickers, start=start.date(), end=(end + pd.Timedelta(days=1)).date(),
                auto_adjust=True, interval="1d", progress=False, timeout=15,
            )
    if prices is None:
        raise RuntimeError("Yahoo Finance did not return a price table.")
    prices = prices.reindex(columns=columns).sort_index()
    if (
        prices.empty
        or prices.index.has_duplicates
        or prices.index[0] > start + pd.Timedelta(days=7)
        or prices.index[-1] < end - pd.Timedelta(days=7)
        or not np.isfinite(prices.to_numpy()).all()
        or (prices <= 0).any().any()
    ):
        raise RuntimeError("Incomplete or invalid adjusted Yahoo Finance bars for the configured dates.")
    if not cached:
        with NamedTemporaryFile(mode="w", suffix=".tmp", dir=cache, delete=False, encoding="utf-8") as file:
            temp = Path(file.name)
            try:
                prices.to_csv(file, index_label="Date")
                file.close()
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
    return prices["Open"], prices["Close"]


def weekly_components(returns):
    weeks = returns.index.to_period("W-FRI")
    signed = {"up": returns.clip(lower=0), "down": returns.clip(upper=0)}
    variance = {side: values.pow(2).groupby(weeks).sum(min_count=1)
                for side, values in signed.items()}
    sums = {side: values.groupby(weeks).sum(min_count=1) for side, values in signed.items()}
    return variance, sums


def forecast_risk(opens, closes, model, start):
    sessions = opens.index.to_series()
    grouped = sessions.groupby(sessions.index.to_period("W-FRI"))
    first_session = grouped.first()
    last_session = grouped.last()
    if not (np.diff(first_session.index.asi8) == 1).all():
        raise RuntimeError("At least one trading week is missing from the price data.")

    trade_open = first_session.shift(-1)
    label_end = first_session.shift(-2)
    close_variance, close_sums = weekly_components(np.log(closes).diff())
    forward_open_returns = np.log(opens).diff().shift(-1)
    open_variance, _ = weekly_components(forward_open_returns)
    floor = model["variance_floor"]
    features = {}
    labels = {}
    for side in SIDES:
        one_week = close_variance[side]
        long_window = one_week.rolling(model["long_window_weeks"]).sum()
        features[side] = np.stack([
            np.log(one_week.clip(lower=floor)),
            np.log(long_window.clip(lower=floor)),
            close_sums["up"],
            close_sums["down"],
        ], axis=-1)
        labels[side] = np.log(open_variance[side].shift(-1).clip(lower=floor)).to_numpy()

    active = (trade_open >= start) & (trade_open < opens.index[-1])
    forecasts = {side: [] for side in SIDES}
    execution_dates = pd.DatetimeIndex(trade_open.loc[active])
    fit_year = None
    regressions = {}
    for row, execution_date in zip(np.flatnonzero(active), execution_dates):
        if execution_date.year != fit_year:
            signal_close = last_session.iloc[row]
            cutoff = signal_close - pd.DateOffset(years=model["lookback_years"])
            train = (last_session >= cutoff) & (label_end <= signal_close)
            for side in SIDES:
                x = features[side][train].reshape(-1, features[side].shape[-1])
                y = labels[side][train].ravel()
                valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
                if valid.sum() <= x.shape[1]:
                    raise RuntimeError(f"Too few completed training rows for {execution_date:%Y-%m-%d}.")
                regressions[side] = make_pipeline(StandardScaler(), Ridge(alpha=model["ridge_alpha"]))
                regressions[side].fit(x[valid], y[valid])
            fit_year = execution_date.year
        for side in SIDES:
            log_prediction = regressions[side].predict(features[side][row])
            forecasts[side].append(np.exp(np.maximum(log_prediction, np.log(floor))))

    predicted = {side: pd.DataFrame(forecasts[side], index=execution_dates, columns=opens.columns)
                 for side in SIDES}
    trailing = {side: close_variance[side].loc[active].set_axis(execution_dates).clip(lower=floor)
                for side in SIDES}
    return predicted, trailing


def within(index, period):
    return (index >= pd.Timestamp(period["start"])) & (index <= pd.Timestamp(period["end"]))


def matched_total_gate(total, downside_gate):
    # Preserve the signal's total risk budget and assign cuts by total variance.
    ranks = total.rank(axis=1, method="first", ascending=False)
    return ranks.le(downside_gate.sum(axis=1), axis="index")


def target_weights(gate, reserve, reduced_fraction):
    weights = (1.0 - gate * (1.0 - reduced_fraction)) / len(gate.columns)
    weights[reserve] = gate.sum(axis=1) * (1.0 - reduced_fraction) / len(gate.columns)
    return weights


def simulate(opens, targets, fee_rate, reserve):
    """Buy at target opens, drift between rebalances, mark at the next open."""
    opens = opens.loc[targets.index[0]:, targets.columns]
    relatives = opens.to_numpy()[1:] / opens.to_numpy()[:-1]
    weights = np.zeros(len(targets.columns))
    rebalance = targets.reindex(opens.index[:-1]).to_numpy()
    reserve_column = targets.columns.get_loc(reserve)
    rows = []
    for asset_growth, target in zip(relatives, rebalance):
        fee = turnover = 0.0
        if np.isfinite(target).all():
            if fee_rate:
                # Post-cost target holdings and fees must sum to pre-trade equity.
                fee = brentq(
                    lambda cost: cost - fee_rate * np.abs((1.0 - cost) * target - weights).sum(),
                    0.0, 1.0,
                )
            turnover = np.abs((1.0 - fee) * target - weights).sum()
            weights = target
        risk_weight = 1.0 - weights[reserve_column]
        growth = weights @ asset_growth
        rows.append(((1.0 - fee) * growth - 1.0, turnover, risk_weight))
        weights = weights * asset_growth / growth
    result = pd.DataFrame(rows, index=opens.index[1:], columns=["return", "turnover", "risk_weight"])
    result["start_open"] = opens.index[:-1]
    return result


def metrics(path, trading_days):
    returns = path["return"]
    equity = np.r_[1.0, (1.0 + returns).cumprod().to_numpy()]
    years = (returns.index[-1] - path["start_open"].iloc[0]).days / 365.2425
    daily_vol = returns.std()
    values = {
        "cagr": equity[-1] ** (1.0 / years) - 1.0 if years > 0 else None,
        "ann_vol": daily_vol * np.sqrt(trading_days),
        "sharpe": returns.mean() / daily_vol * np.sqrt(trading_days) if daily_vol > 0 else None,
        "max_drawdown": (equity / np.maximum.accumulate(equity) - 1.0).min(),
        "total_return": equity[-1] - 1.0,
        "turnover": path["turnover"].sum(),
        "avg_risk_weight": path["risk_weight"].mean(),
        "n_obs": len(returns),
    }
    return {key: (None if value is None or not np.isfinite(value) else
                  int(value) if key == "n_obs" else float(value))
            for key, value in values.items()}


def period_metrics(path, cfg):
    segments = {
        "is": path.loc[within(path.index, cfg["in_sample"])],
        "oos": path.loc[within(path.index, cfg["out_of_sample"])],
        "full": path,
    }
    if any(segment.empty for segment in segments.values()):
        raise RuntimeError("Each configured sample must contain at least one complete open-to-open return.")
    return {name: metrics(segment, cfg["trading_days_per_year"]) for name, segment in segments.items()}


def print_table(results):
    def fmt(value, spec):
        return format(value, spec) if value is not None else "n/a"

    print(f"{'Portfolio':<24} {'Period':<7} {'CAGR':>10} {'Ann vol':>10} {'Sharpe':>8} "
          f"{'Max DD':>10} {'Tot return':>11} {'Turnover':>10} {'Avg risk':>10} {'Days':>6}")
    for name, periods in results.items():
        for period, row in periods.items():
            print(f"{name:<24} {period.upper():<7} {fmt(row['cagr'], '.2%'):>10} "
                  f"{fmt(row['ann_vol'], '.2%'):>10} {fmt(row['sharpe'], '.2f'):>8} "
                  f"{fmt(row['max_drawdown'], '.2%'):>10} {fmt(row['total_return'], '.2%'):>11} "
                  f"{fmt(row['turnover'], '.2f'):>10} {fmt(row['avg_risk_weight'], '.2%'):>10} "
                  f"{row['n_obs']:>6}")


def main():
    with (HERE / "config.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    risk = cfg["universe"]["risk_tickers"]
    reserve = cfg["universe"]["reserve_ticker"]
    tickers = [*risk, reserve]
    is_start = pd.Timestamp(cfg["in_sample"]["start"])
    is_end = pd.Timestamp(cfg["in_sample"]["end"])
    oos_start = pd.Timestamp(cfg["out_of_sample"]["start"])
    oos_end = pd.Timestamp(cfg["out_of_sample"]["end"])
    if not is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Sample dates must be ordered with no IS/OOS overlap.")
    opens, closes = load_prices(tickers, pd.Timestamp(cfg["data_start"]), oos_end)
    predictions, trailing = forecast_risk(opens[risk], closes[risk], cfg["model"], is_start)
    if predictions["up"].empty:
        raise RuntimeError("No weekly rebalance dates in the configured samples.")
    threshold = cfg["allocation"]["downside_share_threshold"]
    reduced = cfg["allocation"]["reduced_sleeve_fraction"]
    total = predictions["up"] + predictions["down"]
    downside_gate = predictions["down"].div(total) > threshold
    trailing_gate = trailing["down"].div(trailing["up"] + trailing["down"]) > threshold
    gates = {
        "Forecast downside share": downside_gate,
        "Constant equal weight": pd.DataFrame(False, index=total.index, columns=risk),
        "Trailing downside share": trailing_gate,
        "Matched total variance": matched_total_gate(total, downside_gate),
    }
    results = {}
    for name, gate in gates.items():
        path = simulate(opens, target_weights(gate, reserve, reduced), cfg["cost_bps"] / 10_000, reserve)
        results[name] = period_metrics(path, cfg)
    benchmark = results["Constant equal weight"]
    output = {
        "strategy": cfg["strategy"],
        "data_range": f"{total.index[0]:%Y-%m-%d} to {opens.index[-1]:%Y-%m-%d}",
        **results["Forecast downside share"],
        "benchmark": {
            "name": "Constant equal weight",
            **benchmark["full"],
            **benchmark,
            "comparisons": {name: results[name] for name in ("Trailing downside share", "Matched total variance")},
        },
    }
    print(f"{cfg['strategy']} | {output['data_range']} | {cfg['cost_bps']:g} bps per traded dollar")
    print_table(results)
    print("RESULTS_JSON: " + json.dumps(output, allow_nan=False))


if __name__ == "__main__":
    main()
