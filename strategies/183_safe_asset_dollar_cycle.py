"""183_safe_asset_dollar_cycle
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/183_safe_asset_dollar_cycle.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 2b23c5bbf706aea933c0cba5f5805efb450ee206 on 2026-11-26 14:26 Europe/Oslo, exit 0, output sha256 1a21ba0292e55cf66d96fad7fb76cb2af2b6ff88e53416bc37734496446adaed
Data source: Free Yahoo Finance daily adjusted Open and Close via yfinance. UUP and IEF supply the signal, and EEM, VWO and BIL are the allocation instruments.
Data fetched: 2026-11-26 14:26 Europe/Oslo (run time)
Sample period: The recorded backtest covers ETF sessions from 2008-01-02 through 2025-12-31. The configured price request starts on 2007-09-01 for signal warm-up.   data_range (from output): 2008-01-02 to 2025-12-31
IS/OOS split: IS 2008-01-01 through 2016-12-31 and OOS 2017-01-01 through 2025-12-31, with inclusive calendar bounds. The main strategy and equal-weight benchmark carry holdings across the split. The OOS-only static comparison uses mean daily IS target weight and starts at the first OOS open. OOS is a historical holdout that includes dates before the paper’s publication.
Costs and slippage: 5 bps per side on absolute buy and sell notional at each open rebalance, including initial purchases. The simulator solves for post-fee equity to fund the target holdings and fees. Slippage is not modelled separately. The fixed fee is the only execution-cost deduction and is described as an approximation of execution frictions.
Signal lag: Compute trailing ETF returns at the last session close of each Friday-ending week. Execute at the next ETF session’s open and mark to market at that session’s close. At sample entry, use the latest target already available for that opening.
Price adjustment: Yahoo Finance supplies adjusted Open and Close through yfinance with auto_adjust=True on download. Adjusted closes feed the signal and daily valuation, and adjusted opens feed simulated execution. The cache stores the adjusted bars. Historical values may change with distributions or data revisions.
Survivorship bias: The backtest uses fixed ETFs and their historical fund prices, avoiding a backfill of current equity constituents. Ex post selection of surviving funds leaves fund-level selection bias, and ETF listing dates constrain the sample.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.04807430944494462, ann_vol=0.2837247119625683, sharpe=-0.031187285334044242, max_drawdown=-0.6749303331866706, total_return=-0.3580335410301625, turnover=121.6445113645475, n_trades=1054
    oos: cagr=0.06034368999785067, ann_vol=0.1866004983089248, sharpe=0.4077822795109247, max_drawdown=-0.40664828459460156, total_return=0.6920532784875653, turnover=78.56216621602675, n_trades=1001
    full: cagr=0.004613464128480738, ann_vol=0.24016530795362268, sharpe=0.1397995385765699, max_drawdown=-0.6749303331866706, total_return=0.08624145157897067, turnover=200.20667758057425, n_trades=2055
    benchmark: cagr=0.02833252578351364, ann_vol=0.26879172597895107, sharpe=0.23786873488402185, max_drawdown=-0.6478506506736429, total_return=0.6522176721905795, turnover=2.4322077532565745, n_trades=1880
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 183_safe_asset_dollar_cycle\nuniverse:\n  dollar: UUP\n  treasuries: IEF\n  emerging_markets: [EEM, VWO]\n  cash_proxy: BIL\ndata_start: '2007-09-01'\nin_sample:\n  start: '2008-01-01'\n  end: '2016-12-31'\nout_of_sample:\n  start: '2017-01-01'\n  end: '2025-12-31'\nsignal:\n  lookback_sessions: 10\n  dollar_threshold: 0.01\n  normal_em_weight: 1.0\n  defensive_em_weight: 0.25\ncost_bps_per_side: 5.0\nsessions_per_year: 252\n"
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
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent


def load_prices(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"yahoo_adjusted_{'-'.join(sorted(tickers))}_{start.date()}_{end.date()}.csv"
    columns = pd.MultiIndex.from_product([["Open", "Close"], tickers])

    def validate(prices):
        prices = prices.reindex(columns=columns).sort_index()
        values = prices.to_numpy(dtype=float)
        if prices.empty or not prices.index.is_unique or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Yahoo data must contain unique sessions and all positive adjusted ETF prices")
        if prices.index[0] - start > pd.Timedelta(days=7) or end - prices.index[-1] > pd.Timedelta(days=7):
            raise ValueError("Yahoo data does not cover the requested date range")
        return prices

    if cache.is_file():
        return validate(pd.read_csv(cache, header=[0, 1], index_col=0, parse_dates=True))

    import yfinance as yf

    with TemporaryDirectory(dir=cache_dir, prefix=".yfinance-") as temp:
        yf.set_tz_cache_location(temp)
        raw = yf.download(
            tickers, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True, keepna=True, ignore_tz=True, progress=False, timeout=20,
        )
        prices = validate(raw)
        staging = Path(temp) / "prices.csv"
        prices.to_csv(staging)
        staging.replace(cache)
    return prices


def execution_schedule(close, universe, signal, start, end):
    momentum = close.pct_change(periods=signal["lookback_sessions"], fill_method=None)
    weekly = momentum.groupby(pd.Grouper(freq="W-FRI")).tail(1).dropna()
    dollar_up = weekly[universe["dollar"]] > signal["dollar_threshold"]
    bond_return = weekly[universe["treasuries"]]
    flags = pd.DataFrame({
        "strategy": dollar_up & (bond_return > 0),
        "uup_only": dollar_up,
        "uup_up_ief_down": dollar_up & (bond_return < 0),
    })
    at_close = flags.replace({True: signal["defensive_em_weight"], False: signal["normal_em_weight"]})
    at_close["benchmark"] = 1.0
    # Map a signal dated at the close to the next ETF session's open.
    at_open = at_close.reindex(close.index).shift(1)
    schedule = at_open.loc[start:end].copy()
    if schedule.empty:
        raise ValueError("No ETF sessions in the configured backtest dates")
    schedule.iloc[0] = at_open.ffill().loc[schedule.index[0]]
    if schedule.iloc[0].isna().any():
        raise ValueError("Insufficient ETF prices before the backtest to warm up the weekly signal")
    return schedule


def simulate(open_price, close_price, em_schedule, fee_rate):
    """Rebalance at the open and mark to market at the close, in USD."""
    opens = open_price.loc[em_schedule.index].to_numpy()
    closes = close_price.loc[em_schedule.index].to_numpy()
    overnight = opens[1:] / closes[:-1]
    intraday = closes / opens
    holdings = np.zeros(opens.shape[1])
    em_count = len(holdings) - 1  # The final column is the T-bill sleeve.
    previous_equity = 1.0
    rows = []

    for day, em_weight in enumerate(em_schedule):
        turnover = 0.0
        trades = 0
        if day > 0:
            holdings *= overnight[day - 1]
        if pd.notna(em_weight):
            equity = holdings.sum() if day > 0 else previous_equity
            current = holdings / equity
            target = np.full(len(holdings), em_weight / em_count)
            target[-1] = 1.0 - em_weight

            # Solve for post-fee equity so purchases, sales and fees are self-financing.
            def funding_error(remaining):
                return remaining + fee_rate * np.abs(remaining * target - current).sum() - 1.0

            remaining = brentq(funding_error, 0.0, 1.0) if fee_rate else 1.0
            changed = np.abs(remaining * target - current)
            turnover = float(changed.sum())
            trades = int((changed > 1e-12).sum())
            holdings = remaining * equity * target

        holdings *= intraday[day]
        equity = holdings.sum()
        rows.append((equity / previous_equity - 1.0, turnover, trades))
        previous_equity = equity

    return pd.DataFrame(rows, index=em_schedule.index, columns=["return", "turnover", "n_trades"])


def metrics(path, periods_per_year):
    daily = path["return"]
    equity = np.r_[1.0, (1.0 + daily).cumprod().to_numpy()]
    std = daily.std(ddof=1)
    values = {
        "cagr": equity[-1] ** (periods_per_year / len(daily)) - 1.0 if len(daily) else np.nan,
        "ann_vol": std * np.sqrt(periods_per_year),
        "sharpe": daily.mean() * np.sqrt(periods_per_year) / std if std > 0 else np.nan,
        "max_drawdown": (equity / np.maximum.accumulate(equity) - 1.0).min() if len(daily) else np.nan,
        "total_return": equity[-1] - 1.0 if len(daily) else np.nan,
        "turnover": path["turnover"].sum(),
    }
    cleaned = {key: float(value) if np.isfinite(value) else None for key, value in values.items()}
    return {**cleaned, "n_trades": int(path["n_trades"].sum())}


def report(rows):
    print(f"{'Portfolio':<24} {'Period':<6} {'CAGR':>10} {'Ann. vol.':>10} {'Sharpe':>8} "
          f"{'Max DD':>10} {'Total ret.':>10} {'Turnover':>10} {'Trades':>8}")
    for label, spans in rows.items():
        for period, row in spans.items():
            cagr, vol, dd, total, turnover = (
                f"{row[key]:.2%}" if row[key] is not None else "N/A"
                for key in ("cagr", "ann_vol", "max_drawdown", "total_return", "turnover")
            )
            sharpe = f"{row['sharpe']:.3f}" if row["sharpe"] is not None else "N/A"
            print(f"{label:<24} {period.upper():<6} {cagr:>10} {vol:>10} {sharpe:>8} "
                  f"{dd:>10} {total:>10} {turnover:>10} {row['n_trades']:>8}")


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    signal = config["signal"]
    em = universe["emerging_markets"]
    held = [*em, universe["cash_proxy"]]
    tickers = [universe["dollar"], universe["treasuries"], *held]
    data_start = pd.Timestamp(config["data_start"])
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    fee = config["cost_bps_per_side"] / 10_000
    annualization = config["sessions_per_year"]
    if not (data_start < is_start <= is_end < oos_start <= oos_end and
            is_end + pd.Timedelta(days=1) == oos_start):
        raise ValueError("Require warm-up data and consecutive, chronological IS and OOS dates")
    if (not em or len(set(tickers)) != len(tickers) or not 0 <= fee < 1 or annualization <= 0 or
            signal["lookback_sessions"] < 1 or
            not 0 <= signal["defensive_em_weight"] <= signal["normal_em_weight"] <= 1):
        raise ValueError("Check tickers, weights, transaction costs and session counts in config.yaml")

    prices = load_prices(tickers, data_start, oos_end)
    schedule = execution_schedule(prices["Close"], universe, signal, is_start, oos_end)
    if schedule.loc[is_start:is_end].empty or schedule.loc[oos_start:oos_end].empty:
        raise ValueError("Each evaluation period must contain ETF sessions")
    opening, closing = (prices[field][held] for field in ("Open", "Close"))
    periods = {"is": slice(is_start, is_end), "oos": slice(oos_start, oos_end), "full": slice(None)}
    results = {}
    for label, targets in schedule.items():
        path = simulate(opening, closing, targets, fee)
        results[label] = {period: metrics(path.loc[span], annualization) for period, span in periods.items()}

    matched_weight = float(schedule["strategy"].ffill().loc[is_start:is_end].mean())
    matched_schedule = schedule.loc[oos_start:oos_end, "strategy"].copy()
    matched_schedule.iloc[0] = matched_weight
    matched_schedule = matched_schedule.where(matched_schedule.isna(), matched_weight)
    matched = metrics(simulate(opening, closing, matched_schedule, fee), annualization)
    results["is_matched_static"] = {"oos": matched}

    data_range = f"{schedule.index[0].date()} to {schedule.index[-1].date()}"
    names = {
        "strategy": "Joint UUP / IEF",
        "uup_only": "UUP only",
        "uup_up_ief_down": "UUP up / IEF down",
        "benchmark": "Equal-weight EM",
        "is_matched_static": "IS-matched static",
    }
    print(f"{config['strategy']} | {data_range} | {config['cost_bps_per_side']:g} bps per side")
    print(f"Static OOS benchmark EM weight from IS targets = {matched_weight:.2%}\n")
    report({names[key]: rows for key, rows in results.items()})
    output = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **results["strategy"],
        "benchmark": results["benchmark"]["full"],
        "benchmark_by_period": {key: results["benchmark"][key] for key in ("is", "oos")},
        "comparisons": {
            "uup_only": results["uup_only"],
            "uup_up_ief_down": results["uup_up_ief_down"],
            "is_matched_static": {"em_weight": matched_weight, "oos": matched},
        },
    }
    print("RESULTS_JSON: " + json.dumps(output, allow_nan=False))


if __name__ == "__main__":
    main()
