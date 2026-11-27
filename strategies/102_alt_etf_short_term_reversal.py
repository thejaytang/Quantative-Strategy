"""102_alt_etf_short_term_reversal
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/102_alt_etf_short_term_reversal.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 3140bbcffa77faae04956fc55adbc5be3b33f896 on 2026-11-26 16:06 Europe/Oslo, exit 0, output sha256 3a5a624fc62a0fc0a9e9943677e16112b5b69cc1ad9bdef2c7db266ca9f9685a
Data source: Public Yahoo Finance Chart API daily ETF bars, fetched directly over HTTPS via requests. The strategy ETFs are DBMF, MNA, PBP, and WTMF. BIL supplies the cash-return proxy. Valid per-ticker CSV caches are reused.
Data fetched: 2026-11-26 16:06 Europe/Oslo (run time)
Sample period: Evaluated sessions: 2020-01-02 through 2026-11-24, inclusive. The configured price request starts on 2019-11-01 for signal warm-up. data_range (from output): 2020-01-02 to 2026-11-24
IS/OOS split: IS: 2020-01-02 through 2026-02-23. OOS: 2026-02-24 through 2026-11-24. Both ranges are inclusive. The README describes OOS as a historical holdout beginning after the cited article's publication date. The code generates targets across the full period and slices the resulting daily_results for reporting.
Costs and slippage: 10 bps one way in the base case, with 25 and 50 bps one-way stress scenarios. Each daily position pays on its opening and closing notionals, including consecutive signal days. The charged amount as a fraction of starting NAV is the one-way fee times sum(abs(weight) × (2 + open-to-close return)). Shorts also pay 300 bps annualized divided by 252 per held session, applied to opening short weight. Slippage is not modelled. The simulator uses daily bar open and close as fill proxies. The buy-and-hold benchmark pays the base fee once on entry.
Signal lag: The adjusted close on session t is compared with the preceding 5 or 10 session closes. The resulting position is shifted by one session, enters at the next session's open, and exits at that session's close.
Price adjustment: Signals and BIL cash returns use Yahoo Finance adjusted closes. Each strategy ETF's adjusted open equals Yahoo's open multiplied by adjusted close divided by Yahoo's close for the same session. That factor cancels from the strategy's open-to-close return. The benchmark uses the first adjusted open and subsequent adjusted closes for valuation.
Survivorship bias: The four strategy ETFs are the fixed, retrospectively selected universe from the cited research. The README flags selection and survivorship risk from missing comparable liquidated or delisted funds.
Parameter combinations tried: 6
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.17333568791312226, ann_vol=0.04236025473654765, sharpe=-4.470903222637993, max_drawdown=-0.6919454115456477, total_return=-0.6882508251133215, turnover=0.9022837304539946, n_trades=2784
    oos: cagr=-0.25877167018359715, ann_vol=0.0335004926123123, sharpe=-8.916542913206, max_drawdown=-0.20330959251825953, total_return=-0.2030485219931647, turnover=0.8875840764411027, n_trades=339
    full: cagr=-0.1832096548612029, ann_vol=0.04152492300839152, sharpe=-4.850826308066646, max_drawdown=-0.7531062094823953, total_return=-0.7515510343066504, turnover=0.9006645644122054, n_trades=3123
    benchmark full: cagr=0.059312715810007965, ann_vol=0.07487129429664191, sharpe=0.8072281741588824, max_drawdown=-0.13600005720217412, total_return=0.48658665166792714, turnover=0.0005761251435991921, n_trades=4
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 102_alt_etf_short_term_reversal\n\nuniverse: [DBMF, MNA, PBP, WTMF]\ncash_ticker: BIL\ndata_start: 2019-11-01  # Signal warm-up only before the in-sample start.\nin_sample:\n  start: 2020-01-02\n  end: 2026-02-23\nout_of_sample:\n  start: 2026-02-24\n  end: 2026-11-24\n\nlookback: 5\nsensitivity_lookback: 10\nmax_abs_weight: 0.25\ncosts:\n  one_way_bps: 10\n  stress_one_way_bps: [25, 50]\n  short_borrow_annual_bps: 300\nsessions_per_year: 252\n'
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
"""Daily alternative-ETF reversal backtest using public Yahoo Finance prices."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import requests
import yaml


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_FIELDS = ["open", "close", "adj_close"]


def download_prices(session, ticker, start, end):
    params = {
        "period1": int(start.tz_localize("UTC").timestamp()),
        "period2": int((end + pd.Timedelta(days=1)).tz_localize("UTC").timestamp()),
        "interval": "1d",
    }
    errors = []
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        try:
            response = session.get(
                f"https://{host}/v8/finance/chart/{ticker}",
                params=params,
                timeout=(5, 15),
                allow_redirects=False,
            )
            response.raise_for_status()
            result = response.json()["chart"]["result"][0]
            timezone = result["meta"]["exchangeTimezoneName"]
            dates = pd.to_datetime(result["timestamp"], unit="s", utc=True)
            dates = dates.tz_convert(timezone).tz_localize(None).normalize()
            indicators = result["indicators"]
            quote = indicators["quote"][0]
            frame = pd.DataFrame(
                {
                    "open": quote["open"],
                    "close": quote["close"],
                    "adj_close": indicators["adjclose"][0]["adjclose"],
                },
                index=dates.rename("date"),
            )
            return validate_prices(frame, ticker, start, end)
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            errors.append(f"{host}: {exc}")
    raise RuntimeError(f"Yahoo Finance download failed for {ticker}: {'; '.join(errors)}")


def validate_prices(frame, ticker, start, end):
    frame = frame.loc[:, PRICE_FIELDS].sort_index().loc[start:end]
    buffer = pd.Timedelta(days=7)
    if frame.empty or frame.index[0] > start + buffer or frame.index[-1] < end - buffer:
        raise ValueError(f"{ticker} has incomplete coverage for {start.date()} to {end.date()}")
    values = frame.to_numpy(dtype=float)
    if not frame.index.is_unique or not (np.isfinite(values) & (values > 0)).all():
        raise ValueError(f"{ticker} has duplicate dates or invalid OHLC prices")
    return frame


def load_prices(session, ticker, start, end):
    path = CACHE / f"yahoo_{ticker}_{start.date()}_{end.date()}.csv"
    if path.is_file():
        try:
            frame = pd.read_csv(
                path, index_col="date", parse_dates=["date"], converters=dict.fromkeys(PRICE_FIELDS, float)
            )
            return validate_prices(frame, ticker, start, end)
        except (KeyError, TypeError, ValueError) as exc:
            print(f"Refreshing invalid cache for {ticker}: {exc}")

    frame = download_prices(session, ticker, start, end)
    with NamedTemporaryFile(mode="w", dir=CACHE, suffix=".tmp", delete=False) as handle:
        pending = Path(handle.name)
        try:
            frame.to_csv(handle)
            handle.close()
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
    return frame


def reversal_positions(adjusted_close, lookback, max_weight):
    prior = adjusted_close.shift(1).rolling(lookback)
    at_low = adjusted_close.lt(prior.min())
    at_high = adjusted_close.gt(prior.max())
    close_signal = (at_low.astype(float) - at_high.astype(float)) * max_weight
    return close_signal.shift(1, fill_value=0.0)


def simulate(positions, intraday, cash_returns, fee, daily_borrow):
    # Each position starts at the open and closes that session, even on repeated signals.
    gross = positions.abs()
    short_weight = -positions.clip(upper=0).sum(axis=1)
    turnover = (gross * (2 + intraday)).sum(axis=1)
    return pd.DataFrame(
        {
            "return": (positions * intraday).sum(axis=1)
            + (1 - gross.sum(axis=1)) * cash_returns
            - fee * turnover
            - daily_borrow * short_weight,
            "turnover": turnover,
            "n_trades": positions.ne(0).sum(axis=1),
        }
    )


def buy_and_hold(adjusted_open, adjusted_close, fee):
    # Equal opening allocations, with one entry fee and weights drifting thereafter.
    equity = adjusted_close.div(adjusted_open.iloc[0]).mean(axis=1) / (1 + fee)
    returns = equity.pct_change(fill_method=None)
    returns.iloc[0] = equity.iloc[0] - 1
    benchmark = pd.DataFrame({"return": returns, "turnover": 0.0, "n_trades": 0})
    benchmark.loc[equity.index[0], ["turnover", "n_trades"]] = [
        1 / (1 + fee),
        adjusted_close.shape[1],
    ]
    return benchmark


def metrics(daily_results, annualization):
    returns = daily_results["return"]
    if returns.empty or not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("Performance requires a nonempty period with finite, positive equity")

    wealth = (1 + returns).cumprod()
    vol = returns.std(ddof=1)
    values = {
        "cagr": wealth.iloc[-1] ** (annualization / len(returns)) - 1,
        "ann_vol": vol * np.sqrt(annualization),
        "sharpe": returns.mean() / vol * np.sqrt(annualization) if vol > 0 else np.nan,
        "max_drawdown": (wealth / wealth.cummax().clip(lower=1) - 1).min(),
        "total_return": wealth.iloc[-1] - 1,
        "turnover": daily_results["turnover"].mean(),
    }
    return {key: float(value) if np.isfinite(value) else None for key, value in values.items()} | {
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def summarize(daily_results, periods, annualization):
    return {name: metrics(daily_results.loc[start:end], annualization) for name, (start, end) in periods.items()}


def print_report(data_range, scenarios, benchmark, legs, lookbacks):
    def number(value, percent=False):
        if value is None:
            return "n/a"
        return f"{value:.2%}" if percent else f"{value:.2f}"

    print(f"\nAlternative ETF short-term reversal | {data_range}")
    print("Net returns | Costs in one-way bps | Turnover is the mean daily traded notional / NAV")
    print(f"{'Portfolio':<23} {'Split':<5} {'CAGR':>9} {'Ann. vol':>9} {'Sharpe':>7}"
          f" {'Max DD':>9} {'Total':>9} {'Turnover':>9} {'Entries':>8}")
    for label, summary in {**scenarios, "Equal-weight buy & hold": benchmark}.items():
        for period, row in summary.items():
            print(f"{label:<23} {period.upper():<5} {number(row['cagr'], True):>9}"
                  f" {number(row['ann_vol'], True):>9} {number(row['sharpe']):>7}"
                  f" {number(row['max_drawdown'], True):>9} {number(row['total_return'], True):>9}"
                  f" {number(row['turnover'], True):>9} {row['n_trades']:>8}")

    print("\nStandalone ETF legs | Base costs | OOS CAGR (all splits in JSON)")
    print(f"{'ETF':<8} {'Side':<8} {'Daily':>9} {str(lookbacks[0]) + 'd':>9} {str(lookbacks[1]) + 'd':>9}")
    for ticker, sides in legs.items():
        for side, variants in sides.items():
            values = [number(variants[name]["oos"]["cagr"], True)
                      for name in ("daily", *(f"{lookback}d" for lookback in lookbacks))]
            print(f"{ticker:<8} {side:<8} " + " ".join(f"{value:>9}" for value in values))


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    universe, cash = config["universe"], config["cash_ticker"]
    lookbacks = [config["lookback"], config["sensitivity_lookback"]]
    costs = config["costs"]
    base_cost = costs["one_way_bps"]
    max_weight = config["max_abs_weight"]
    annualization = config["sessions_per_year"]
    daily_borrow = costs["short_borrow_annual_bps"] / 10_000 / annualization
    periods = {
        name: tuple(pd.Timestamp(config[key][limit]) for limit in ("start", "end"))
        for name, key in (("is", "in_sample"), ("oos", "out_of_sample"))
    }
    start, end = periods["is"][0], periods["oos"][1]
    periods["full"] = (start, end)
    data_start = pd.Timestamp(config["data_start"])
    if not data_start < start <= periods["is"][1] < periods["oos"][0] <= end:
        raise ValueError("Use ordered, non-overlapping samples after the warm-up start")
    if not 0 < len(universe) * max_weight <= 1 or len(set([*universe, cash])) != len(universe) + 1:
        raise ValueError("Use distinct tickers and a positive sleeve weight with gross exposure at most 100%")

    CACHE.mkdir(exist_ok=True)
    with requests.Session() as session:
        session.headers["User-Agent"] = "Mozilla/5.0"
        session.auth = lambda request: request  # Anonymous access, bypassing netrc lookup.
        prices = {
            ticker: load_prices(session, ticker, data_start, end)
            for ticker in [*universe, cash]
        }
    if any(not prices[ticker].index.equals(prices[cash].index) for ticker in universe):
        raise ValueError("ETF session dates differ from the BIL session dates")

    adjusted = pd.DataFrame({ticker: prices[ticker]["adj_close"] for ticker in universe})
    if (adjusted.index < start).sum() < max(lookbacks) + 1:
        raise ValueError("Insufficient pre-sample sessions for the longest lookback")
    adjusted_open = pd.DataFrame({
        ticker: prices[ticker]["open"] * adjusted[ticker] / prices[ticker]["close"]
        for ticker in universe
    })
    # The same adjustment on both bars leaves the realized open-to-close return unchanged.
    intraday = (adjusted / adjusted_open - 1).loc[start:end]
    cash_returns = prices[cash]["adj_close"].pct_change(fill_method=None).loc[start:end]
    positions = {
        lookback: reversal_positions(adjusted, lookback, max_weight).loc[start:end]
        for lookback in lookbacks
    }
    scenarios = {}
    for lookback, targets in positions.items():
        for cost in [base_cost, *costs["stress_one_way_bps"]]:
            daily_results = simulate(targets, intraday, cash_returns, cost / 10_000, daily_borrow)
            scenarios[f"{lookback}d / {cost:g} bps"] = summarize(daily_results, periods, annualization)

    primary = scenarios[f"{lookbacks[0]}d / {base_cost:g} bps"]
    benchmark = summarize(
        buy_and_hold(adjusted_open.loc[start:end], adjusted.loc[start:end], base_cost / 10_000),
        periods,
        annualization,
    )
    legs = {}
    for ticker in universe:
        legs[ticker] = {}
        for side, sign in (("long", 1), ("short", -1)):
            variants = {"daily": pd.DataFrame(sign * max_weight, index=intraday.index, columns=[ticker])}
            for lookback, targets in positions.items():
                variants[f"{lookback}d"] = targets[[ticker]].where(targets[[ticker]] * sign > 0, 0)
            legs[ticker][side] = {
                label: summarize(
                    simulate(target, intraday[[ticker]], cash_returns, base_cost / 10_000, daily_borrow),
                    periods,
                    annualization,
                )
                for label, target in variants.items()
            }

    data_range = f"{intraday.index[0].date()} to {intraday.index[-1].date()}"
    print_report(data_range, scenarios, benchmark, legs, lookbacks)
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **primary,
        "benchmark": {"name": "Equal-weight alternative ETF buy and hold", **benchmark},
        "sensitivity": scenarios,
        "legs": legs,
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
