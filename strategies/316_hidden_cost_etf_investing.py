"""316_hidden_cost_etf_investing
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/316_hidden_cost_etf_investing.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 2b23c5bbf706aea933c0cba5f5805efb450ee206 on 2026-11-26 14:27 Europe/Oslo, exit 0, output sha256 f96965e758b1970e0a83aee278401927f2feb93c4c1a48398fec14b978f23baf
Data source: Free Yahoo Finance daily ETF open and close prices via yfinance for SPY (hedge and benchmark), IVV, VOO, RSP, and VTI. The recorded run used the local Yahoo price cache.
Data fetched: 2026-11-26 14:27 Europe/Oslo (run time)
Sample period: Requested price range 2011-01-01 through 2025-12-31 inclusive, with pre-reporting observations used for warmup. The run reports trading sessions from 2012-01-03 through 2025-12-31.   data_range (from output): 2012-01-03 to 2025-12-31
IS/OOS split: Inclusive calendar bounds: IS 2012-01-01 through 2018-12-31; OOS 2019-01-01 through 2025-12-31. The code simulates a continuous daily_results and slices it for reporting. The rolling estimation and positions can carry across the boundary. OOS is a retrospective historical holdout.
Costs and slippage: 2 bps per side on the sum of absolute dollar position changes across ETF legs at each simulated open or close fill, including entries, exits, and rebalances. The fee is deducted from equity and target positions are sized using after-fee equity. Slippage is not modeled separately. The simulation uses the daily open and close as execution price proxies and has no bid-ask spread or market-impact model.
Signal lag: The new monthly signal uses 126 sessions through the previous month's final close. The overnight strategy first enters the new target at the next trading session's close and exits at the following session's open. The intraday comparison first uses its new target at the next trading session's open and exits at that session's close. The continuous comparison first uses the new day target at that open and the new night target at that close.
Price adjustment: The download uses yfinance with auto_adjust=True and keeps adjusted Open and Close. Cache hits load those columns from the Yahoo adjusted price CSV. Overnight returns use adjusted current open divided by adjusted prior close minus one, and intraday returns use adjusted close divided by adjusted same-day open minus one.
Survivorship bias: The ETF universe is a fixed, retrospectively selected set of surviving funds. The README identifies fund selection and survivorship bias and states that the strategy does not use current index constituents.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.0728480161367765, ann_vol=0.006851912881616202, sharpe=-11.033846439165666, max_drawdown=-0.4114142792191402, total_return=-0.4103722194988455, turnover=3519.1872668774013, n_trades=10557
    oos: cagr=-0.0827389489420226, ann_vol=0.014223765390166973, sharpe=-6.063599495200993, max_drawdown=-0.4535843442184826, total_return=-0.452926070149676, turnover=3519.210060246991, n_trades=10557
    full: cagr=-0.07780674302569868, ann_vol=0.0111673238577444, sharpe=-7.246596107165238, max_drawdown=-0.6775822705738499, total_return=-0.6774300129723079, turnover=7038.397327124392, n_trades=21114
    benchmark: cagr=0.14747108090161576, ann_vol=0.16667145025357671, sharpe=0.9090915546094717, max_drawdown=-0.3371726874596017, total_return=5.831057222045796, turnover=1.9923910554309021, n_trades=2
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 316_hidden_cost_etf_investing\nhedge_ticker: SPY\npair_tickers: [IVV, VOO, RSP, VTI]\n\n# Date bounds are inclusive. The download starts before the reporting window.\ndata_start: 2011-01-01\nin_sample:\n  start: 2012-01-01\n  end: 2018-12-31\nout_of_sample:\n  start: 2019-01-01\n  end: 2025-12-31\n\nlookback_sessions: 126\ngross_cap: 1.0\ntransaction_cost_bps: 2.0  # Per side, on the dollar amount traded in each ETF.\nsessions_per_year: 252\n'
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
"""Backtest the C107 ETF relative-value spread by trading session."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / f"yahoo_adjusted_{'-'.join(tickers)}_{start.date()}_{end.date()}.csv"
    columns = pd.MultiIndex.from_product([["Open", "Close"], tickers])
    if cache_file.exists():
        prices = pd.read_csv(cache_file, header=[0, 1], index_col=0, parse_dates=True,
                             float_precision="round_trip")
    else:
        with TemporaryDirectory(dir=CACHE, prefix="yahoo_metadata_") as metadata:
            yf.set_tz_cache_location(metadata)
            for attempt in range(2):
                data = yf.download(
                    tickers, start=start.date().isoformat(),
                    end=(end + pd.Timedelta(days=1)).date().isoformat(),
                    auto_adjust=True, threads=False, progress=False, timeout=15,
                    multi_level_index=True,
                )
                prices = data.reindex(columns=columns)
                if not prices.empty and prices.notna().all().all():
                    break
                if attempt == 0:
                    print("Retrying the public Yahoo Finance download after incomplete data.", flush=True)

    prices = prices.reindex(columns=columns).sort_index()
    if prices.empty or not np.isfinite(prices).all().all() or (prices <= 0).any().any():
        bad = prices.columns[(~np.isfinite(prices) | (prices <= 0)).any()]
        missing = ", ".join(f"{ticker} {field}" for field, ticker in bad)
        raise RuntimeError(f"Incomplete Yahoo Finance price panel: {missing or 'no prices'}")
    if not prices.index.is_unique or prices.index.hasnans:
        raise RuntimeError("Price panel has duplicate or invalid dates")
    if not (start <= prices.index[0] <= start + pd.Timedelta(days=7)
            and end - pd.Timedelta(days=7) <= prices.index[-1] <= end):
        raise RuntimeError("Yahoo Finance prices do not cover the requested date range")
    if not cache_file.exists():
        with NamedTemporaryFile(mode="w", dir=CACHE, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
        try:
            prices.to_csv(temporary)
            temporary.replace(cache_file)
        finally:
            temporary.unlink(missing_ok=True)
    return prices


def fit_session(returns):
    market = returns[:, 0]
    candidates = returns[:, 1:]
    centered_market = market - market.mean()
    sum_squares = centered_market @ centered_market
    if sum_squares <= 0:
        raise ValueError("Cannot estimate a session beta with zero SPY variance")
    beta = centered_market @ (candidates - candidates.mean(axis=0)) / sum_squares
    # The mean of y - beta*x is the fitted intercept. Subtracting the intercept
    # as well would force every sample mean residual to zero.
    premium = (candidates - market[:, None] * beta).mean(axis=0)
    return beta, premium


def monthly_targets(overnight, intraday, dates, lookback, gross_cap):
    night_targets = np.zeros_like(overnight)
    day_targets = np.zeros_like(intraday)
    months = dates.to_period("M")
    month_starts = np.flatnonzero(months[1:] != months[:-1]) + 1
    for start, end in zip(month_starts, np.r_[month_starts[1:], len(dates)]):
        if start <= lookback:  # The first close-to-open return needs a previous close.
            continue
        history = slice(start - lookback, start)
        night_beta, night_premium = fit_session(overnight[history])
        day_beta, day_premium = fit_session(intraday[history])
        ranking = np.argsort(night_premium - day_premium, kind="stable")
        short, long = ranking[[0, -1]]
        for targets, beta in ((night_targets, night_beta), (day_targets, day_beta)):
            weights = np.zeros(overnight.shape[1])
            weights[long + 1] = 1.0
            weights[short + 1] = -1.0
            weights[0] = beta[short] - beta[long]
            targets[start:end] = weights * (gross_cap / np.abs(weights).sum())
    return night_targets, day_targets


def rebalance(positions, equity, target, cost_rate):
    # The target uses after-fee equity. Iterate on the tiny fee adjustment so
    # the gross exposure at the fill is sized off the resulting equity.
    fee = 0.0
    for _ in range(16):
        new_positions = (equity - fee) * target
        traded = np.abs(new_positions - positions)
        notional = traded.sum()
        new_fee = cost_rate * notional
        if abs(new_fee - fee) <= 1e-13 * equity:
            orders = np.count_nonzero(traded > 1e-12 * equity)
            return new_positions, equity - new_fee, notional, orders
        fee = new_fee
    raise RuntimeError("Trade cost calculation did not converge")


def simulate(dates, overnight, intraday, close_targets, open_targets, cost_rate):
    equity = 1.0
    positions = np.zeros(overnight.shape[1])
    results = []
    for date, night, day, close_target, open_target in zip(
        dates, overnight, intraday, close_targets, open_targets, strict=True
    ):
        previous_equity = equity
        equity += positions @ night
        positions *= 1.0 + night
        positions, equity, open_notional, open_trades = rebalance(
            positions, equity, open_target, cost_rate
        )
        equity += positions @ day
        positions *= 1.0 + day
        positions, equity, close_notional, close_trades = rebalance(
            positions, equity, close_target, cost_rate
        )
        if not np.isfinite(equity) or equity <= 0:
            raise RuntimeError(f"Invalid portfolio equity on {date.date()}")
        results.append((equity / previous_equity - 1.0,
                        (open_notional + close_notional) / previous_equity,
                        open_trades + close_trades))
    return pd.DataFrame(results, index=dates, columns=["return", "turnover", "n_trades"])


def finite_float(value):
    return float(value) if np.isfinite(value) else None


def metrics(daily_results, annual_sessions):
    returns = daily_results["return"].to_numpy()
    equity = np.r_[1.0, np.cumprod(1.0 + returns)]
    std = returns.std(ddof=1) if len(returns) > 1 else np.nan
    return {
        "cagr": finite_float(equity[-1] ** (annual_sessions / len(returns)) - 1.0),
        "ann_vol": finite_float(std * np.sqrt(annual_sessions)),
        "sharpe": finite_float(returns.mean() / std * np.sqrt(annual_sessions)) if std > 0 else None,
        "max_drawdown": finite_float((equity / np.maximum.accumulate(equity) - 1.0).min()),
        "total_return": finite_float(equity[-1] - 1.0),
        "n_trades": int(daily_results["n_trades"].sum()),
        "turnover": finite_float(daily_results["turnover"].sum()),
    }


def summarize(daily_results, periods, annual_sessions):
    result = {}
    for name, (start, end) in periods.items():
        period = daily_results.loc[start:end]
        if period.empty:
            raise ValueError(f"No sessions in reporting period {name}")
        result[name] = metrics(period, annual_sessions)
    return result


def print_table(reports):
    columns = (
        ("cagr", "CAGR", 9, ".2%"), ("ann_vol", "Ann vol", 9, ".2%"),
        ("sharpe", "Sharpe", 8, ".2f"), ("max_drawdown", "Max DD", 9, ".2%"),
        ("total_return", "Total", 9, ".2%"), ("n_trades", "Trades", 9, "d"),
        ("turnover", "Turnover", 12, ".2f"),
    )
    headings = " ".join(label.rjust(width) for _, label, width, _ in columns)
    print(f"{'Portfolio':<26} {'Period':<6} {headings}")
    for portfolio, report in reports.items():
        for period, row in report.items():
            values = [(format(row[key], spec) if row[key] is not None else "-").rjust(width)
                      for key, _, width, spec in columns]
            print(f"{portfolio:<26} {period.upper():<6} {' '.join(values)}")


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    hedge = config["hedge_ticker"]
    tickers = [hedge, *config["pair_tickers"]]
    lookback = config["lookback_sessions"]
    gross_cap = config["gross_cap"]
    cost_rate = config["transaction_cost_bps"] / 10_000
    annual_sessions = config["sessions_per_year"]
    data_start = pd.Timestamp(config["data_start"])
    periods = {}
    for label, section in (("is", "in_sample"), ("oos", "out_of_sample")):
        periods[label] = tuple(pd.Timestamp(config[section][key]) for key in ("start", "end"))
    is_start, is_end = periods["is"]
    oos_start, oos_end = periods["oos"]
    if not data_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Check the date ordering of the warmup, IS and OOS windows")
    if len(tickers) < 3 or len(set(tickers)) != len(tickers):
        raise ValueError("Configure a unique hedge ETF and at least two unique pair candidates")
    if lookback < 2 or not 0 < gross_cap <= 1 or not 0 <= cost_rate <= 0.05 or annual_sessions <= 0:
        raise ValueError("Invalid lookback, gross cap, transaction cost or annualization setting")
    periods["full"] = (is_start, oos_end)

    prices = load_prices(tickers, data_start, oos_end)
    opens, closes = prices["Open"], prices["Close"]
    overnight = (opens / closes.shift(1) - 1.0).to_numpy()
    intraday = (closes / opens - 1.0).to_numpy()
    night, day = monthly_targets(overnight, intraday, prices.index, lookback, gross_cap)
    sample = (prices.index >= is_start) & (prices.index <= oos_end)
    dates = prices.index[sample]
    night, day = night[sample], day[sample]
    if not len(dates) or not night[0].any():
        raise ValueError("Download more warmup data to fit the first month's signal")
    overnight, intraday = overnight[sample], intraday[sample]
    flat = np.zeros_like(night)
    hold = flat.copy()
    hold[:, 0] = 1.0
    hold_until_end = hold.copy()
    hold_until_end[-1] = 0.0
    night[-1] = 0.0  # End at the last close, without starting another overnight trade.

    benchmark_name = f"{hedge} buy and hold"
    portfolios = {
        "Overnight spread": (night, flat),
        "Intraday spread": (flat, day),
        "Continuous spread": (night, day),
        benchmark_name: (hold_until_end, hold),
    }
    reports = {}
    for name, (close_targets, open_targets) in portfolios.items():
        daily_results = simulate(dates, overnight, intraday, close_targets, open_targets, cost_rate)
        reports[name] = summarize(daily_results, periods, annual_sessions)

    strategy_report = reports["Overnight spread"]
    benchmark = reports[benchmark_name]
    print(f"{config['strategy']} | {dates[0].date()} to {dates[-1].date()} | "
          f"{config['transaction_cost_bps']:g} bps per side | turnover in NAV multiples")
    print_table(reports)
    result = {
        "strategy": config["strategy"],
        "data_range": f"{dates[0].date()} to {dates[-1].date()}",
        **strategy_report,
        "benchmark": {"name": benchmark_name, **benchmark["full"],
                      "is": benchmark["is"], "oos": benchmark["oos"]},
        "comparisons": {
            "intraday": reports["Intraday spread"],
            "continuous": reports["Continuous spread"],
        },
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
