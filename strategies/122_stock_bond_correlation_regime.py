"""122_stock_bond_correlation_regime
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/122_stock_bond_correlation_regime.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit ad594150a0142009ae8f6ffab0995de08b2f8304 on 2026-11-26 13:49 Europe/Oslo, exit 0, output sha256 6d8e95203ed486e0654d58e8213782ee8811db17c718fca65b3baea3822c5547
Data source: Yahoo Finance daily ETF prices via yfinance for SPY, IEF, GLD and BIL. The loader uses the matching local CSV on a cache hit and downloads on a miss.
Data fetched: 2026-11-26 13:49 Europe/Oslo (run time)
Sample period: Requested price window 2008-01-01 through 2025-12-31, with data before the 2009-01-01 scoring start used for warm-up. The recorded scored sessions run from 2009-01-02 through 2025-12-31.   data_range (from output): 2009-01-02 to 2025-12-31
IS/OOS split: Configured IS is 2009-01-01 through 2020-12-31 and OOS is 2021-01-01 through 2025-12-31. The recorded session ranges are IS 2009-01-02 through 2020-12-31 and OOS 2021-01-04 through 2025-12-31. The regime and static benchmark carry holdings across the split. Only the second benchmark uses mean IS entry and rebalance target weights, frozen for OOS. The OOS split is a retrospective chronological holdout and includes dates before the cited paper's publication.
Costs and slippage: 5 bps per side on the dollar value of each buy or sell, including initial entry. At each simulated opening trade, the solver deducts the fee on the sum of absolute position changes from available NAV before setting target holdings. Slippage and market impact are not separately modelled, and the simulator applies no separate taxes.
Signal lag: At month-end session t, the rule uses the rolling Pearson correlation of 126 daily SPY and IEF adjusted close-to-close returns through close t. A finite signal sets the target for the open of the next session, the first session of the new month. Existing holdings earn the overnight move and post-trade holdings earn the open-to-close move. If the correlation is unavailable, the monthly target is omitted and holdings carry forward.
Price adjustment: The download sets auto_adjust=False. The loader uses Yahoo's Adj Close for closes and raw Open multiplied by that session's Adj Close / raw Close for adjusted opens. It caches those two adjusted fields.
Survivorship bias: The strategy uses four fixed ETFs. It does not select historical securities from today's index constituents. Retrospectively selecting ETFs that remained in existence can still introduce selection bias.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.10755861509088471, ann_vol=0.09941406720946781, sharpe=1.0775814228193914, max_drawdown=-0.18855155250564437, total_return=2.4030823589813366, turnover=4.607190133792889, n_trades=296
    oos: cagr=0.09152764563665872, ann_vol=0.10833385503976842, sharpe=0.8626297080896611, max_drawdown=-0.20060790887330637, total_return=0.5467461414623578, turnover=4.1206749673575, n_trades=188
    full: cagr=0.10282929099397098, ann_vol=0.10210076028504925, sharpe=1.0099163972519734, max_drawdown=-0.20060790887330593, total_return=4.263704507833003, turnover=8.727865101150389, n_trades=484
    benchmark: cagr=0.0993358948799905, ann_vol=0.10222447459288463, sharpe=0.9777660270222724, max_drawdown=-0.2120087675966812, total_return=3.9878251165404066, turnover=4.75155622924156, n_trades=408
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "122_stock_bond_correlation_regime"\ncategory: macro\nuniverse: [SPY, IEF, GLD, BIL]\ndata_start: "2008-01-01"\nin_sample:\n  start: "2009-01-01"\n  end: "2020-12-31"\nout_of_sample:\n  start: "2021-01-01"\n  end: "2025-12-31"\nsignal:\n  equity: SPY\n  treasury: IEF\n  lookback_sessions: 126\n  threshold: 0.0\nweights:\n  at_or_below_threshold: {SPY: 0.60, IEF: 0.40, GLD: 0.00, BIL: 0.00}\n  above_threshold: {SPY: 0.60, IEF: 0.20, GLD: 0.10, BIL: 0.10}\n  static_benchmark: {SPY: 0.60, IEF: 0.40, GLD: 0.00, BIL: 0.00}\ntransaction_cost_bps: 5.0  # Charged on every dollar bought or sold.\n'
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
from scipy.optimize import brentq
import yaml


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252


def load_prices(tickers, start, end):
    """Load matched daily adjusted opens and closes for the configured window."""
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    symbol_key = "_".join(sorted(tickers))
    cache = cache_dir / f"yahoo_{symbol_key}_{start.date()}_{end.date()}.csv"
    fields = ["Open", "Close"]
    expected_columns = pd.MultiIndex.from_product([fields, tickers])

    if cache.is_file():
        prices = pd.read_csv(cache, header=[0, 1], index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        # Keep yfinance's own ancillary cache inside this strategy directory.
        with TemporaryDirectory(dir=cache_dir) as provider_cache:
            yf.set_tz_cache_location(provider_cache)
            raw = yf.download(
                tickers, start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=False, group_by="column", ignore_tz=True,
                progress=False, timeout=15,
            )
        if raw.empty or not {"Open", "Close", "Adj Close"}.issubset(raw.columns.get_level_values(0)):
            raise RuntimeError("Yahoo Finance did not return daily open, close and adjusted close.")
        adjusted_open = raw["Open"] * (raw["Adj Close"] / raw["Close"])
        prices = pd.concat({"Open": adjusted_open, "Close": raw["Adj Close"]}, axis=1)

    prices = prices.sort_index().reindex(columns=expected_columns)
    values = prices.to_numpy(dtype=float)
    if (
        prices.empty
        or prices.index.has_duplicates
        or not np.isfinite(values).all()
        or (values <= 0).any()
    ):
        raise RuntimeError(f"Incomplete or invalid daily ETF bars for {', '.join(tickers)}.")
    if not (
        start <= prices.index[0] <= start + pd.Timedelta(days=7)
        and end - pd.Timedelta(days=7) <= prices.index[-1] <= end
    ):
        raise RuntimeError("Daily ETF bars do not cover the requested date window.")

    if not cache.is_file():
        with TemporaryDirectory(dir=cache_dir) as cache_staging:
            temp_path = Path(cache_staging) / cache.name
            prices.to_csv(temp_path)
            temp_path.replace(cache)
    return prices["Open"], prices["Close"]


def read_weights(config_weights, tickers):
    if set(config_weights) != set(tickers):
        raise ValueError("Each allocation must specify exactly the universe tickers.")
    weights = pd.Series(config_weights, index=tickers, dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Weights must be finite, nonnegative and sum to 1.")
    return weights / weights.sum()


def signal_targets(close, signal, weights):
    lookback = signal["lookback_sessions"]
    if type(lookback) is not int or lookback < 2 or not -1 <= signal["threshold"] <= 1:
        raise ValueError("Set an integer lookback of at least two and a threshold in [-1, 1].")
    if signal["equity"] == signal["treasury"] or not {signal["equity"], signal["treasury"]} <= set(close.columns):
        raise ValueError("The signal must use two distinct tickers in the universe.")
    returns = close.pct_change(fill_method=None)
    correlation = returns[signal["equity"]].rolling(lookback).corr(
        returns[signal["treasury"]]
    )
    months = close.index.to_series().dt.to_period("M")
    first_of_month = months.ne(months.shift())
    # The previous session's close is the month-end observation for this open.
    observed = correlation.shift().loc[first_of_month]
    observed = observed.loc[np.isfinite(observed)]
    targets = pd.DataFrame(
        np.where(
            observed.to_numpy()[:, None] > signal["threshold"],
            weights["above_threshold"].to_numpy(),
            weights["at_or_below_threshold"].to_numpy(),
        ),
        index=observed.index, columns=close.columns,
    )
    return targets, first_of_month


def constant_targets(weights, rebalance_dates):
    return pd.DataFrame(
        [weights.to_numpy()] * len(rebalance_dates),
        index=rebalance_dates, columns=weights.index,
    )


def simulate(opens, closes, targets, start, end, fee_rate):
    """Mark the old holdings to the open, trade, then mark the new holdings to close."""
    sessions = closes.loc[start:end].index
    if sessions.empty:
        raise ValueError(f"No ETF sessions in {start.date()} through {end.date()}.")
    first = sessions[0]
    # A mid-month simulation start uses the most recent available allocation.
    initial_target = targets.loc[:first].iloc[-1]
    actions = targets.reindex(sessions)
    actions.loc[first] = initial_target

    overnight = (opens / closes.shift()).loc[sessions].to_numpy()
    intraday = (closes / opens).loc[sessions].to_numpy()
    positions = np.zeros(len(closes.columns))
    rows = []
    for day, target in enumerate(actions.to_numpy()):
        # Positions and trade amounts use units of wealth at the previous close.
        open_holdings = positions * overnight[day] if day else positions
        open_nav = open_holdings.sum() if day else 1.0
        turnover, n_trades = 0.0, 0
        if np.isfinite(target[0]):
            # Solve for the investable NAV after fees on the actual filled notional.
            def fee_balance(invested_nav):
                notional = np.abs(invested_nav * target - open_holdings).sum()
                return invested_nav + fee_rate * notional - open_nav

            invested_nav = brentq(fee_balance, 0, open_nav) if fee_rate else open_nav
            new_holdings = invested_nav * target
            trade_sizes = np.abs(new_holdings - open_holdings)
            turnover = trade_sizes.sum() / open_nav
            n_trades = int(np.count_nonzero(trade_sizes > 1e-12))
            open_holdings = new_holdings

        closing_holdings = open_holdings * intraday[day]
        growth = closing_holdings.sum()
        positions = closing_holdings / growth
        rows.append((growth - 1, turnover, n_trades))

    return pd.DataFrame(rows, index=sessions, columns=["return", "turnover", "n_trades"])


def metrics(daily):
    returns = daily["return"]
    equity = (1 + returns).cumprod()
    volatility = returns.std(ddof=1)
    result = {
        "cagr": equity.iloc[-1] ** (SESSIONS_PER_YEAR / len(equity)) - 1,
        "ann_vol": volatility * np.sqrt(SESSIONS_PER_YEAR),
        "sharpe": returns.mean() / volatility * np.sqrt(SESSIONS_PER_YEAR) if volatility > 0 else None,
        "max_drawdown": (equity / equity.cummax().clip(lower=1) - 1).min(),
        "total_return": equity.iloc[-1] - 1,
        "turnover": daily["turnover"].sum(),
        "n_trades": int(daily["n_trades"].sum()),
    }
    return {
        key: (float(value) if value is not None and np.isfinite(value) else None)
        for key, value in result.items()
        if key != "n_trades"
    } | {"n_trades": result["n_trades"]}


def split_metrics(daily, in_sample, out_of_sample):
    return {
        "is": metrics(daily.loc[in_sample[0]:in_sample[1]]),
        "oos": metrics(daily.loc[out_of_sample[0]:out_of_sample[1]]),
        "full": metrics(daily),
    }


def print_table(strategy_stats, static_stats, average_stats):
    labels = ("Portfolio / sample", "CAGR", "Ann. vol", "Sharpe", "Max DD", "Total return", "Turnover", "Trades")
    layout = "{:<28}" + "{:>14}" * (len(labels) - 1)
    print(layout.format(*labels))
    for name, results in (
        ("Regime", strategy_stats),
        ("Static allocation", static_stats),
        ("IS average weights", {"oos": average_stats}),
    ):
        for period, row in results.items():
            values = []
            for key, value in row.items():
                if value is None:
                    values.append("n/a")
                elif key == "n_trades":
                    values.append(str(value))
                elif key == "sharpe":
                    values.append(f"{value:.2f}")
                elif key == "turnover":
                    values.append(f"{value:.2f}x")
                else:
                    values.append(f"{value:.2%}")
            print(layout.format(f"{name} / {period.upper()}", *values))


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    tickers = config["universe"]
    if not tickers or len(tickers) != len(set(tickers)):
        raise ValueError("The universe must contain unique tickers.")
    weights = {key: read_weights(value, tickers) for key, value in config["weights"].items()}
    fee_rate = float(config["transaction_cost_bps"]) / 10_000
    if not 0 <= fee_rate < 1:
        raise ValueError("Transaction cost must be nonnegative and below 10,000 bps.")
    in_sample, out_of_sample = (
        tuple(pd.Timestamp(config[sample][boundary]) for boundary in ("start", "end"))
        for sample in ("in_sample", "out_of_sample")
    )
    start, end = in_sample[0], out_of_sample[1]
    data_start = pd.Timestamp(config["data_start"])
    if not data_start < start <= in_sample[1] < out_of_sample[0] <= end:
        raise ValueError("Set a warm-up start and two ordered, non-overlapping sample ranges.")

    opens, closes = load_prices(tickers, data_start, end)
    targets, first_of_month = signal_targets(closes, config["signal"], weights)
    sessions = closes.loc[start:end].index
    in_sample_sessions = closes.loc[in_sample[0]:in_sample[1]].index
    out_of_sample_sessions = closes.loc[out_of_sample[0]:out_of_sample[1]].index
    if in_sample_sessions.empty or out_of_sample_sessions.empty:
        raise ValueError("Both samples must contain ETF sessions.")
    first = sessions[0]
    if targets.loc[:first].empty:
        raise ValueError("Insufficient warm-up data for the first simulated open.")

    strategy_path = simulate(opens, closes, targets, start, end, fee_rate)
    rebalance_dates = first_of_month.loc[start:end]
    rebalance_dates = rebalance_dates[rebalance_dates].index
    static_dates = rebalance_dates.insert(0, first).unique()
    static_targets = constant_targets(weights["static_benchmark"], static_dates)
    static_path = simulate(opens, closes, static_targets, start, end, fee_rate)

    # Freeze the mean of the allocations used at IS entry and IS rebalances.
    training_targets = targets.loc[in_sample[0]:in_sample[1]].copy()
    training_targets.loc[first] = targets.loc[:first].iloc[-1]
    average_weights = training_targets.mean()
    average_dates = rebalance_dates.intersection(out_of_sample_sessions)
    average_dates = average_dates.insert(0, out_of_sample_sessions[0]).unique()
    average_targets = constant_targets(average_weights, average_dates)
    average_path = simulate(opens, closes, average_targets, *out_of_sample, fee_rate)

    strategy_stats = split_metrics(strategy_path, in_sample, out_of_sample)
    static_stats = split_metrics(static_path, in_sample, out_of_sample)
    average_stats = metrics(average_path)
    data_range = f"{sessions[0].date()} to {sessions[-1].date()}"
    print(f"{config['strategy']} | Scored sessions {data_range} | {fee_rate * 10_000:g} bps per side")
    print(f"IS {in_sample_sessions[0].date()} to {in_sample_sessions[-1].date()} | "
          f"OOS {out_of_sample_sessions[0].date()} to {out_of_sample_sessions[-1].date()}")
    print_table(strategy_stats, static_stats, average_stats)
    results = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **strategy_stats,
        "benchmark": {
            "name": "static_allocation",
            "weights": weights["static_benchmark"].to_dict(),
            **static_stats["full"],
            "is": static_stats["is"],
            "oos": static_stats["oos"],
            "training_average": {"weights": average_weights.to_dict(), "oos": average_stats},
        },
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
