# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "103_breaking_bad_trends"\n\nuniverse:\n  risk_assets: [SPY, EFA, EEM, IEF, TLT, GLD, DBC, USO, UUP]\n  reserve_asset: BIL\n\ndata_start: "2007-11-01"\nin_sample:\n  start: "2009-01-01"\n  end: "2023-12-31"\nout_of_sample:\n  start: "2024-01-01"\n  end: "2025-12-31"\n\nsignal:\n  fast_sessions: 21\n  slow_sessions: 252\n  rebalance_week_ending: "W-FRI"\n  sleeve_fraction:\n    Bull: 1.0\n    Correction: 0.0\n    Bear: 0.0\n    Rebound: 0.5\n\ntransaction_cost_bps: 5.0\nslippage_bps: 5.0\nsessions_per_year: 252\n'
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
"""Backtest the C04 weekly fast/slow trend strategy."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
STATES = ("Bull", "Correction", "Bear", "Rebound")
PRICE_ADJUSTMENT = "Yahoo Finance via yfinance auto_adjust=True (split- and distribution-adjusted Open and Close)"
PARAM_COMBOS_TRIED = 1


def load_prices(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    key = hashlib.sha256(",".join(tickers).encode()).hexdigest()[:12]
    cache = cache_dir / f"yahoo_adjusted_{start:%Y%m%d}_{end:%Y%m%d}_{key}.csv.gz"
    columns = pd.MultiIndex.from_product([["Open", "Close"], tickers])
    cached = cache.exists()
    if cached:
        with cache.open("rb") as snapshot:
            # Legacy snapshots recorded their write time in the gzip header only.
            header = snapshot.read(10)
            saved_at = int.from_bytes(header[4:8], "little")
            fetched_at = datetime.fromtimestamp(saved_at, timezone.utc) if saved_at else None
            snapshot.seek(0)
            prices = pd.read_csv(snapshot, compression="gzip", header=[0, 1], index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        yf.set_tz_cache_location(str(cache_dir / "yfinance"))
        raw = yf.download(
            tickers, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True, progress=False, keepna=True, timeout=15,
        )
        fetched_at = datetime.now(timezone.utc).replace(microsecond=0)
        prices = raw.reindex(columns=columns).sort_index()

    if (
        prices.empty or not prices.columns.equals(columns)
        or prices.index.has_duplicates or not prices.index.is_monotonic_increasing
        or not start <= prices.index[0] <= start + pd.Timedelta(days=4)
        or prices.index[-1] > end
        or (prices.index[-1] < end - pd.Timedelta(days=4))
        or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any()
    ):
        raise ValueError("Yahoo Finance data are incomplete or invalid for the requested ETFs and dates.")

    if not cached:
        with NamedTemporaryFile(dir=cache_dir, suffix=".tmp", delete=False) as fh:
            temp = Path(fh.name)
        try:
            prices.to_csv(temp, compression={"method": "gzip", "mtime": int(fetched_at.timestamp())})
            temp.replace(cache)
        finally:
            temp.unlink(missing_ok=True)
    return prices["Open"], prices["Close"], fetched_at, cached


def make_targets(closes, risk_assets, reserve, signal):
    risk_closes = closes[risk_assets]
    fast = risk_closes.pct_change(signal["fast_sessions"], fill_method=None)
    slow = risk_closes.pct_change(signal["slow_sessions"], fill_method=None)
    # Group by the Friday-ending exchange week. A holiday can move the signal to Thursday.
    week = closes.index.to_period(signal["rebalance_week_ending"])
    signal_dates = closes.index[~week.duplicated(keep="last")][:-1]
    signal_dates = signal_dates[slow.loc[signal_dates].notna().all(axis=1)]
    execution_dates = closes.index[closes.index.get_indexer(signal_dates) + 1]
    fast_up = fast.loc[signal_dates].to_numpy() >= 0
    slow_up = slow.loc[signal_dates].to_numpy() >= 0
    states = pd.DataFrame(
        np.where(slow_up, np.where(fast_up, "Bull", "Correction"),
                 np.where(fast_up, "Rebound", "Bear")),
        index=execution_dates, columns=risk_assets,
    )

    def allocate(fractions):
        weights = fractions / len(risk_assets)
        weights[reserve] = (1 - weights.sum(axis=1)).clip(lower=0)
        return weights.reindex(columns=closes.columns, fill_value=0.0)

    regime_weights = states.apply(lambda column: column.map(signal["sleeve_fraction"]))
    slow_weights = pd.DataFrame(slow_up.astype(float), index=execution_dates, columns=risk_assets)
    return allocate(regime_weights), allocate(slow_weights), states


def schedule_target_changes(targets, first):
    """Establish the latest available target, then trade only when the target changes."""
    changed = targets.ne(targets.shift()).any(axis=1)
    scheduled = targets.loc[changed & (targets.index > first)].copy()
    scheduled.loc[first] = targets.loc[:first].iloc[-1]
    return scheduled


def simulate(opens, closes, targets, fee_rate):
    """Mark prior holdings to the open, trade at the open, then mark to the close."""
    overnight = (opens / closes.shift()).to_numpy()
    intraday = (closes / opens).to_numpy()
    rebalances = dict(zip(targets.index, targets.to_numpy(), strict=True))
    holdings = np.zeros(len(opens.columns))
    uninvested_cash = 1.0
    previous_nav = 1.0
    daily_returns = np.empty(len(opens))
    turnover = np.zeros(len(opens))
    trades = np.zeros(len(opens), dtype=int)

    for i, date in enumerate(opens.index):
        if i > 0:
            holdings *= overnight[i]
        open_nav = holdings.sum() + uninvested_cash
        if date in rebalances:
            target = rebalances[date]
            if fee_rate == 0:
                post_fee_nav = open_nav
            else:
                # Expenses are paid from NAV while preserving the post-fee target weights.
                def balance(nav):
                    return nav + fee_rate * np.abs(nav * target - holdings).sum() - open_nav

                post_fee_nav = brentq(balance, 0, open_nav)
            post_trade = target * post_fee_nav
            traded = np.abs(post_trade - holdings) / open_nav
            turnover[i] = traded.sum()
            trades[i] = np.count_nonzero(traded > 1e-12)
            holdings = post_trade
            uninvested_cash = 0.0

        holdings *= intraday[i]
        close_nav = holdings.sum() + uninvested_cash
        daily_returns[i] = close_nav / previous_nav - 1
        previous_nav = close_nav

    return pd.DataFrame({"return": daily_returns, "turnover": turnover, "n_trades": trades}, index=opens.index)


def metrics(daily, cash_returns, sessions_per_year):
    returns = daily["return"]
    excess = returns - cash_returns.reindex(returns.index)
    equity = (1 + returns).cumprod()
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    drawdown = equity / equity.cummax().clip(lower=1) - 1
    result = {
        "cagr": equity.iloc[-1] ** (sessions_per_year / len(returns)) - 1,
        "ann_vol": volatility * np.sqrt(sessions_per_year),
        "sharpe": returns.mean() / volatility * np.sqrt(sessions_per_year)
        if volatility > 0 else None,
        "sharpe_excess": excess.mean() / excess_volatility * np.sqrt(sessions_per_year)
        if excess_volatility > 0 else None,
        "max_drawdown": drawdown.min(),
        "total_return": equity.iloc[-1] - 1,
        "turnover": daily["turnover"].mean() * sessions_per_year,
    }
    result = {key: float(value) if value is not None and np.isfinite(value) else None
              for key, value in result.items()}
    result["n_trades"] = int(daily["n_trades"].sum())
    return result


def state_stats(opens, states, periods):
    weekly_open = opens.loc[states.index, states.columns]
    forward_returns = weekly_open.shift(-1) / weekly_open - 1
    next_execution = states.index.to_series().shift(-1)
    results = {}
    for period, (start, end) in periods.items():
        complete = (states.index >= start) & (next_execution <= end)
        results[period] = {}
        for state in STATES:
            returns = forward_returns.loc[complete].where(states.loc[complete] == state)
            returns = returns.stack().dropna()
            results[period][state] = {
                "asset_weeks": int(len(returns)),
                "mean_asset_return": float(returns.mean()) if len(returns) else None,
                "positive_fraction": float((returns > 0).mean()) if len(returns) else None,
            }
    return results


def printable(value, percent=True):
    if value is None:
        return "n/a"
    return f"{value:.2%}" if percent else f"{value:.2f}"


def print_tables(results, reserve):
    fields = (
        ("cagr", "CAGR"), ("ann_vol", "Ann. vol"), ("sharpe", "SR (0%)"),
        ("sharpe_excess", f"SR vs {reserve}"), ("max_drawdown", "Max DD"),
        ("total_return", "Total ret"), ("turnover", "Ann. turn"), ("n_trades", "Trades"),
    )
    print(f"\n{'Portfolio':<27} {'Period':<6} " + " ".join(f"{label:>9}" for _, label in fields))
    for name, portfolios in (
        ("Fast/slow trend", results),
        ("Slow-only trend", results["slow_trend"]),
        (results["benchmark"]["name"], results["benchmark"]),
        (results["balanced_benchmark"]["name"], results["balanced_benchmark"]),
    ):
        for period in ("is", "oos", "full"):
            row = portfolios[period]
            values = [str(row[key]) if key == "n_trades" else printable(row[key], not key.startswith("sharpe"))
                      for key, _ in fields]
            print(f"{name:<27} {period.upper():<6} " + " ".join(f"{value:>9}" for value in values))

    print("\nState diagnostics (gross underlying ETF returns between consecutive rebalance opens)")
    print(f"{'Period':<6} {'State':<12} {'Asset-weeks':>11} {'Mean return':>12} {'Positive':>10}")
    for period, rows in results["states"].items():
        for state, row in rows.items():
            mean = printable(row["mean_asset_return"])
            positive = printable(row["positive_fraction"])
            print(f"{period.upper():<6} {state:<12} {row['asset_weeks']:>11} {mean:>12} {positive:>10}")


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    in_sample, out_of_sample = (
        tuple(pd.Timestamp(config[key][endpoint]) for endpoint in ("start", "end"))
        for key in ("in_sample", "out_of_sample")
    )
    data_start = pd.Timestamp(config["data_start"])
    if not data_start < in_sample[0] <= in_sample[1] < out_of_sample[0] <= out_of_sample[1]:
        raise ValueError("Use ordered, non-overlapping date ranges with a prior warm-up.")

    universe = config["universe"]
    risk_assets = universe["risk_assets"]
    reserve = universe["reserve_asset"]
    if not risk_assets or len(set(risk_assets)) != len(risk_assets) or reserve in risk_assets:
        raise ValueError("Specify unique risk assets and a separate reserve asset.")
    tickers = [*risk_assets, reserve]
    signal = config["signal"]
    fractions = signal["sleeve_fraction"]
    if not 0 < signal["fast_sessions"] < signal["slow_sessions"]:
        raise ValueError("Use positive lookbacks with the fast lookback shorter than the slow lookback.")
    if set(fractions) != set(STATES) or not all(0 <= value <= 1 for value in fractions.values()):
        raise ValueError("Specify a sleeve fraction between 0 and 1 for each state.")
    costs = config["transaction_cost_bps"]
    slippage = config["slippage_bps"]
    fee_rate = (costs + slippage) / 10_000
    sessions_per_year = config["sessions_per_year"]
    if not (costs >= 0 and slippage >= 0 and fee_rate < 0.5 and sessions_per_year > 0):
        raise ValueError("Invalid transaction cost or annualization setting.")

    full = (in_sample[0], out_of_sample[1])
    periods = {"is": in_sample, "oos": out_of_sample, "full": full}
    opens, closes, fetched_at, cached = load_prices(tickers, data_start, full[1])
    regime, slow, states = make_targets(closes, risk_assets, reserve, signal)
    study_opens, study_closes = opens.loc[full[0]:full[1]], closes.loc[full[0]:full[1]]
    if any(study_opens.loc[start:end].empty for start, end in periods.values()):
        raise ValueError("Each evaluation period must contain at least one trading session.")
    first = study_opens.index[0]
    if regime.loc[:first].empty:
        raise ValueError("Start the data earlier to warm up the slow signal before the study.")

    cash_returns = study_closes[reserve].pct_change(fill_method=None)
    cash_returns.loc[first] = study_closes.at[first, reserve] / study_opens.at[first, reserve] - 1
    hold = pd.DataFrame(0.0, index=[first], columns=opens.columns)
    hold.loc[first, risk_assets] = 1.0 / len(risk_assets)
    balanced = hold * 0.5
    balanced.loc[first, reserve] = 0.5
    summaries = []
    for scheduled in (schedule_target_changes(regime, first), schedule_target_changes(slow, first), hold, balanced):
        daily = simulate(study_opens, study_closes, scheduled, fee_rate)
        summaries.append({period: metrics(daily.loc[start:end], cash_returns, sessions_per_year)
                          for period, (start, end) in periods.items()})

    strategy_metrics, slow_metrics, hold_metrics, balanced_metrics = summaries
    results = {
        "strategy": config["strategy"],
        "data_range": f"{first:%Y-%m-%d} to {study_opens.index[-1]:%Y-%m-%d}",
        **strategy_metrics,
        "benchmark": {"name": "Risk ETFs equal-weight hold", **hold_metrics},
        "balanced_benchmark": {"name": f"50% risk ETFs / 50% {reserve}", **balanced_metrics},
        "slow_trend": slow_metrics,
        "states": state_stats(opens, states, periods),
    }
    print(f"{results['strategy']} | Yahoo Finance | {results['data_range']}")
    print(f"PARAM_COMBOS_TRIED: {PARAM_COMBOS_TRIED}")
    print(f"DATA_FETCHED_AT: {fetched_at.isoformat() if fetched_at else 'unknown'}")
    print("Data provenance: " + ("cache gzip timestamp (snapshot write time)" if cached else "live Yahoo Finance download"))
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"IS {in_sample[0]:%Y-%m-%d} to {in_sample[1]:%Y-%m-%d} | "
          f"OOS {out_of_sample[0]:%Y-%m-%d} to {out_of_sample[1]:%Y-%m-%d} | "
          f"{costs:g} bps costs + {slippage:g} bps slippage per buy or sell")
    print_tables(results, reserve)
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
