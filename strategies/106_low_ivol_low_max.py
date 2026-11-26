"""106_low_ivol_low_max
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/106_low_ivol_low_max.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit ad594150a0142009ae8f6ffab0995de08b2f8304 on 2026-11-26 13:49 Europe/Oslo, exit 0, output sha256 8c1bc39084f5c56eef55c0b0b3e847e8c042619b3f85450319eceae399b1bd6d
Data source: Yahoo Finance daily Open and Close via yfinance for the fixed US24 universe (AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, BAC, GS, MS, XOM, CVX, COP, SLB, WMT, COST, PG, KO, PEP, JNJ, PFE, MRK, HD, LOW) and the SPY ETF market proxy. The signal uses SPY Close. The loader uses the local price cache if present.
Data fetched: 2026-11-26 13:49 Europe/Oslo (run time)
Sample period: Price data requested from 2012-06-01 for warmup through 2025-12-31 inclusive. The recorded evaluation covers trading sessions from 2013-01-02 through 2025-12-31.   data_range (from output): 2013-01-02 to 2025-12-31
IS/OOS split: Chronological inclusive date ranges: IS 2013-01-01 through 2019-12-31 and OOS 2020-01-01 through 2025-12-31. Each portfolio is simulated continuously and its daily results are sliced for reporting, so positions carry across the boundary. The README states that neither interval was used for parameter tuning.
Costs and slippage: Baseline 5 bps per dollar traded, applied at each opening rebalance to the sum of absolute buy and sell notionals using opening prices. The simulator solves for post-fee NAV and sizes positions from that NAV. The same rule applies to the equal-weight benchmark. Sensitivity runs use 0 and 15 bps. Slippage is not modelled. Fills use the adjusted opening quote with no spread, price impact, or opening execution deviation model.
Signal lag: The signal uses close-to-close returns through the last session of the month, t. Targets are dated to the next session in the price index, t+1, and filled at that session's adjusted open. The fill-day opening quote is also used to check fill availability.
Price adjustment: The Yahoo Finance download explicitly sets yfinance auto_adjust=True and keeps adjusted Open and Close. The cache supplies those same fields when present. Simulated fills use adjusted Open, valuation uses adjusted Close, and stock and SPY signal returns use adjusted Close.
Survivorship bias: The fixed 24-name US large-cap basket was selected retrospectively from surviving names. The README identifies survivorship and selection bias in both IS and OOS. The same configured ticker list is used throughout, with monthly eligibility based on complete return history in the lookback window.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.13354604267031078, ann_vol=0.11944797905739339, sharpe=1.1094783378903714, max_drawdown=-0.16092030437462734, total_return=1.4023821862265917, turnover=10.168120199465026, n_trades=710
    oos: cagr=0.09576398227079963, ann_vol=0.17355173080188777, sharpe=0.6137576849836357, max_drawdown=-0.22875745948413362, total_return=0.7285092087351968, turnover=9.526209180808367, n_trades=599
    full: cagr=0.11596319232313723, ann_vol=0.14687418905305982, sharpe=0.8206469484361287, max_drawdown=-0.22875745948413362, total_return=3.1525397317940254, turnover=9.872095179240485, n_trades=1309
    benchmark  full: cagr=0.18997075411164976, ann_vol=0.16917210760235027, sharpe=1.1133397390273228, max_drawdown=-0.34266084292636223, total_return=8.553736509704272, turnover=0.6104076586634933, n_trades=3744
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 106_low_ivol_low_max\nuniverse:\n  name: US24\n  tickers: [AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, BAC, GS, MS, XOM, CVX,\n            COP, SLB, WMT, COST, PG, KO, PEP, JNJ, PFE, MRK, HD, LOW]\nmarket_proxy: SPY\ndata_start: 2012-06-01\nin_sample:\n  start: 2013-01-01\n  end: 2019-12-31\nout_of_sample:\n  start: 2020-01-01\n  end: 2025-12-31\nsignals:\n  ivol_lookback_sessions: 63\n  max_lookback_sessions: 21\n  portfolio_size: 6\ncosts:\n  bps_per_dollar_traded: 5\n  sensitivity_bps: [0, 15]\nannualization_sessions: 252\n'
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
"""Monthly US24 low-IVOL, low-MAX and combined backtest."""

import hashlib
from itertools import combinations
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
BOOKS = ("low_ivol", "low_max", "combined")
BENCHMARK = "equal_weight"
FIELDS = ("Open", "Close")


def download_prices(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_key = hashlib.sha256(
        f"{start.date()}:{end.date()}:{','.join(sorted(tickers))}".encode()
    ).hexdigest()[:16]
    cache = cache_dir / f"yfinance_adjusted_{cache_key}.csv.gz"
    columns = pd.MultiIndex.from_product([FIELDS, tickers])
    if cache.exists():
        prices = pd.read_csv(cache, header=[0, 1], index_col=0, parse_dates=True)
        return prices.reindex(columns=columns)

    frames = []
    pending = list(tickers)
    with TemporaryDirectory(prefix="yf_session_", dir=cache_dir) as session_cache:
        yf.set_tz_cache_location(session_cache)
        for _ in range(3):
            bars = yf.download(
                pending, start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=True, progress=False, threads=8, timeout=10,
                multi_level_index=True,
            )
            bars = bars.reindex(columns=pd.MultiIndex.from_product([FIELDS, pending]))
            bars = bars.where(np.isfinite(bars) & bars.gt(0))
            usable = (bars["Open"].notna() & bars["Close"].notna()).any()
            received = usable.index[usable].tolist()
            if received:
                frames.append(bars.loc[:, bars.columns.get_level_values(1).isin(received)])
            pending = [ticker for ticker in pending if ticker not in received]
            if not pending:
                break
    if pending:
        raise RuntimeError(f"Yahoo Finance returned no usable bars for {', '.join(pending)}")

    prices = pd.concat(frames, axis=1).reindex(columns=columns).sort_index()
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    if prices.index[0] > start + pd.Timedelta(days=7) or prices.index[-1] < end - pd.Timedelta(days=7):
        raise RuntimeError("Yahoo Finance data does not cover the requested range")
    with NamedTemporaryFile(dir=cache_dir, suffix=".csv.gz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        prices.to_csv(temporary)
        temporary.replace(cache)
    finally:
        temporary.unlink(missing_ok=True)
    return prices


def build_targets(opens, closes, market_close, first_day, last_day, signals):
    ivol_window = signals["ivol_lookback_sessions"]
    max_window = signals["max_lookback_sessions"]
    size = signals["portfolio_size"]
    lookback = max(ivol_window, max_window)
    stock_returns = closes.pct_change(fill_method=None)
    market_returns = market_close.pct_change(fill_method=None)
    days = closes.index

    # A month changes on the exchange calendar after the signal close.
    months = days.to_period("M")
    signal_indices = np.flatnonzero(months[:-1] != months[1:])
    signal_indices = [
        i for i in signal_indices if i >= lookback and first_day <= days[i + 1] <= last_day
    ]
    if not signal_indices:
        raise ValueError("No monthly rebalance has sufficient signal history in the requested dates")
    fill_days = days[np.asarray(signal_indices) + 1]
    targets = {name: pd.DataFrame(0.0, index=fill_days, columns=closes.columns)
               for name in (*BOOKS, BENCHMARK)}

    for i, fill_day in zip(signal_indices, fill_days):
        stock_window = stock_returns.iloc[i - lookback + 1:i + 1].dropna(axis=1)
        market_window = market_returns.iloc[i - ivol_window + 1:i + 1].to_numpy()
        eligible = stock_window.columns
        if len(eligible) >= size:
            targets[BENCHMARK].loc[fill_day, eligible] = 1 / len(eligible)
        if len(eligible) < size or not np.isfinite(market_window).all() or np.ptp(market_window) == 0:
            continue
        design = np.column_stack((np.ones(ivol_window), market_window))
        stock_ivol_window = stock_window.iloc[-ivol_window:].to_numpy()
        coefficients = np.linalg.lstsq(design, stock_ivol_window, rcond=None)[0]
        ivol = pd.Series((stock_ivol_window - design @ coefficients).std(axis=0, ddof=1), index=eligible)
        max_return = stock_window.iloc[-max_window:].max()
        scores = {
            "low_ivol": ivol,
            "low_max": max_return,
            "combined": (ivol.rank(pct=True) + max_return.rank(pct=True)) / 2,
        }
        for name, score in scores.items():
            selected = score.sort_index().nsmallest(size).index
            targets[name].loc[fill_day, selected] = 1 / size

    # If a selected stock has no opening quote, leave its slot in cash.
    available_at_fill = opens.loc[fill_days].notna()
    return {name: weights.where(available_at_fill, 0.0) for name, weights in targets.items()}


def simulate(opens, closes, targets, cost_bps):
    """Keep shares between rebalances and mark the portfolio at each close."""
    opening_prices = opens.to_numpy()
    closing_prices = closes.to_numpy()
    target_rows = targets.reindex(closes.index).to_numpy()
    shares = np.zeros(closes.shape[1])
    cash = previous_nav = 1.0
    cost_rate = cost_bps / 10_000
    daily = np.zeros((len(closes), 3))

    for i, date in enumerate(closes.index):
        target = target_rows[i]
        if np.isfinite(target[0]):
            opening = opening_prices[i]
            held = shares > 0
            if not np.isfinite(opening[held]).all():
                missing = closes.columns[held & ~np.isfinite(opening)].tolist()
                raise ValueError(f"Missing held-stock opening quote on {date.date()} for {missing}")
            current = np.zeros_like(shares)
            current[held] = shares[held] * opening[held]
            open_nav = cash + current.sum()

            # Solve net NAV = open NAV minus fees on the actual dollars exchanged.
            net_nav = open_nav
            for _ in range(20):
                traded = np.abs(net_nav * target - current).sum()
                updated_nav = open_nav - cost_rate * traded
                if abs(updated_nav - net_nav) <= open_nav * 1e-13:
                    net_nav = updated_nav
                    break
                net_nav = updated_nav
            else:
                raise RuntimeError("Rebalance cost calculation did not converge")

            desired = net_nav * target
            traded_by_stock = np.abs(desired - current)
            daily[i, 1] = traded_by_stock.sum() / open_nav
            daily[i, 2] = np.count_nonzero(traded_by_stock > open_nav * 1e-12)
            selected = target > 0
            shares = np.zeros_like(shares)
            shares[selected] = desired[selected] / opening[selected]
            cash = net_nav * (1 - target.sum())

        closing = closing_prices[i]
        held = shares > 0
        if not np.isfinite(closing[held]).all():
            missing = closes.columns[held & ~np.isfinite(closing)].tolist()
            raise ValueError(f"Missing held-stock closing quote on {date.date()} for {missing}")
        nav = cash + shares[held] @ closing[held]
        daily[i, 0] = nav / previous_nav - 1
        previous_nav = nav

    return pd.DataFrame(daily, index=closes.index, columns=["return", "turnover", "n_trades"])


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def metrics(daily, market_returns, sessions):
    returns = daily["return"]
    if returns.empty:
        raise ValueError("An evaluation period has no market sessions")
    equity = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    std = returns.std()
    paired = pd.concat([returns, market_returns.reindex(returns.index)], axis=1).dropna()
    market_var = paired.iloc[:, 1].var()
    beta = paired.iloc[:, 0].cov(paired.iloc[:, 1]) / market_var if market_var > 0 else np.nan
    return {
        "cagr": finite_or_none(equity[-1] ** (sessions / len(returns)) - 1),
        "ann_vol": finite_or_none(std * np.sqrt(sessions)),
        "sharpe": finite_or_none(returns.mean() / std * np.sqrt(sessions) if std > 0 else np.nan),
        "max_drawdown": finite_or_none(np.min(equity / np.maximum.accumulate(equity) - 1)),
        "total_return": finite_or_none(equity[-1] - 1),
        "turnover": finite_or_none(daily["turnover"].mean() * sessions),
        "n_trades": int(daily["n_trades"].sum()),
        "spy_beta": finite_or_none(beta),
    }


def portfolio_overlap(targets, periods):
    overlap = {}
    for name, (start, end) in periods.items():
        overlap[name] = {}
        for left, right in combinations(BOOKS, 2):
            a = targets[left].loc[start:end].gt(0)
            b = targets[right].loc[start:end].gt(0)
            both_invested = a.any(axis=1) & b.any(axis=1)
            shared = (a & b).sum(axis=1)[both_invested]
            overlap[name][f"{left}_vs_{right}"] = {
                "mean_shared_names": finite_or_none(shared.mean()),
                "n_rebalances": len(shared),
            }
    return overlap


def print_report(result, reports, baseline_bps):
    def display(value, percent=False):
        if value is None:
            return "n/a"
        return f"{value:.2%}" if percent else f"{value:.2f}"

    print(f"{result['strategy']} | {result['data_range']} | {baseline_bps:g} bps per dollar traded")
    print("Portfolios and benchmark (net of costs)")
    rows = []
    for book, periods in reports[baseline_bps].items():
        for period, values in periods.items():
            rows.append({
                "Portfolio": book, "Period": period.upper(),
                "CAGR": display(values["cagr"], True), "Ann. vol": display(values["ann_vol"], True),
                "Sharpe": display(values["sharpe"]), "Max DD": display(values["max_drawdown"], True),
                "Total return": display(values["total_return"], True),
                "Ann. turnover": display(values["turnover"], True),
                "Trades": values["n_trades"], "SPY beta": display(values["spy_beta"]),
            })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nMean shared holdings per rebalance")
    overlap = {period.upper(): {pair: display(values["mean_shared_names"])
                               for pair, values in pairs.items()}
               for period, pairs in result["overlap"].items()}
    print(pd.DataFrame(overlap).to_string())

    print("\nCost sensitivity (CAGR by period, 0 bps is gross)")
    cost_rows = []
    for cost, books in reports.items():
        for book, periods in books.items():
            cost_rows.append({"Bps": cost, "Portfolio": book, **{
                period.upper(): display(values["cagr"], True) for period, values in periods.items()
            }})
    print(pd.DataFrame(cost_rows).to_string(index=False))
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False, separators=(",", ":")))


def main():
    with (ROOT / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    universe = config["universe"]
    tickers = sorted(universe["tickers"])
    market = config["market_proxy"]
    start = pd.Timestamp(config["data_start"])
    in_sample, out_of_sample = (
        tuple(pd.Timestamp(config[period][bound]) for bound in ("start", "end"))
        for period in ("in_sample", "out_of_sample")
    )
    first_day, last_day = in_sample[0], out_of_sample[1]
    if not start < first_day <= in_sample[1] < out_of_sample[0] <= last_day:
        raise ValueError("Expected a warmup followed by chronological, non-overlapping IS and OOS ranges")
    signals = config["signals"]
    if (len(set(tickers)) != len(tickers) or market in tickers
            or not 1 <= signals["portfolio_size"] <= len(tickers)
            or signals["ivol_lookback_sessions"] < 3 or signals["max_lookback_sessions"] < 1):
        raise ValueError("Invalid universe, portfolio size or lookback")
    costs = config["costs"]
    baseline_bps = costs["bps_per_dollar_traded"]
    cost_levels = sorted({baseline_bps, *costs["sensitivity_bps"]})
    if any(not 0 <= cost < 1000 for cost in cost_levels):
        raise ValueError("Transaction costs must be nonnegative and below 1000 bps")

    prices = download_prices([*tickers, market], start, last_day)
    market_close = prices[("Close", market)].dropna()
    if (market_close.empty or market_close.index[0] > start + pd.Timedelta(days=7)
            or market_close.index[-1] < last_day - pd.Timedelta(days=7)):
        raise ValueError("Market proxy does not cover the requested price range")
    if not prices.index.equals(market_close.index):
        raise ValueError("The market proxy is missing closes on one or more universe sessions")
    opens = prices["Open"][tickers]
    closes = prices["Close"][tickers]
    targets = build_targets(opens, closes, market_close, first_day, last_day, signals)
    opens = opens.loc[first_day:last_day]
    closes = closes.loc[first_day:last_day]
    market_returns = market_close.pct_change(fill_method=None).rename("market")
    periods = {"is": in_sample, "oos": out_of_sample, "full": (first_day, last_day)}

    reports = {}
    for cost in cost_levels:
        reports[cost] = {}
        for book, weights in targets.items():
            daily = simulate(opens, closes, weights, cost)
            reports[cost][book] = {
                period: metrics(daily.loc[left:right], market_returns, config["annualization_sessions"])
                for period, (left, right) in periods.items()
            }

    net = reports[baseline_bps]
    result = {
        "strategy": config["strategy"],
        "data_range": f"{closes.index[0].date()} to {closes.index[-1].date()}",
        **net["combined"],
        "benchmark": {"name": f"{universe['name']}_equal_weight", **net[BENCHMARK]},
        "books": {book: net[book] for book in BOOKS if book != "combined"},
        "overlap": portfolio_overlap(targets, periods),
        "cost_bps_per_dollar_traded": baseline_bps,
        "cost_sensitivity": {f"{cost:g}": report for cost, report in reports.items() if cost != baseline_bps},
    }
    print_report(result, reports, baseline_bps)


if __name__ == "__main__":
    main()
