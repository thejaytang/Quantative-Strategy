"""341_monitoring_attention_inflation_news
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/341_monitoring_attention_inflation_news.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 3140bbcffa77faae04956fc55adbc5be3b33f896 on 2026-11-26 16:06 Europe/Oslo, exit 0, output sha256 bc62c7298cbd8d58a7ee65b9d4c02d0b63ce6e107d9240209235a989ea49aae4
Data source: Yahoo Finance via yfinance daily adjusted closes for the U.S. Treasury ETFs TIP and IEF. Federal Reserve Bank of St. Louis ALFRED historical vintages for monthly EMVMACROINFLATION (the U.S. inflation-news attention proxy) and daily T10YIE (10-year breakeven inflation). strategy.py:17-18,37-53,59-98,111-127,229-233; config.yaml:2-7; README.md:9-11.
Data fetched: 2026-11-26 16:06 Europe/Oslo (run time)
Sample period: 2019-11-01 to 2025-12-31, inclusive, for the evaluated ETF sessions. The ETF download begins earlier to provide the signal lookback. config.yaml:8-13; strategy.py:218-223,229-238,245-246; logs/run_20261125_151955.log:1.   data_range (from output): 2019-11-01 to 2025-12-31
IS/OOS split: IS 2019-11-01 to 2022-12-31 and OOS 2023-01-01 to 2025-12-31, both inclusive calendar bounds. IS and OOS are chronological slices of the same continuous simulation, with holdings and NAV carried across the split. The README records that neither period was used for parameter fitting or selection and that some OOS dates predate the ECB article. config.yaml:8-13; strategy.py:197-199,218-223,237-243; README.md:7,23.
Costs and slippage: 2 bps per traded side on the absolute dollar amount of each simulated buy or sell, including entry and rebalances. Fees are deducted from portfolio NAV at the execution close, with target weights funded from net NAV. Separate slippage is not modelled. The fixed fee is the approximation for spreads and trading friction. config.yaml:24; strategy.py:157-190,225,241-246; README.md:18,37; logs/run_20261125_151955.log:1.
Signal lag: The signal is dated at the final ETF session close of each month. The simulator fills the target at the next ETF session’s close, after marking existing holdings to that close. The new holdings earn returns from that close onward. If evaluation starts mid-month, the first evaluation close uses the last available target. strategy.py:101-109,129-153,157-187,237-243; README.md:15-18.
Price adjustment: Uses Yahoo Finance’s Adj Close field with yfinance auto_adjust=False. The README describes the adjusted series as accounting for fund distributions and splits. The simulator uses ratios of consecutive adjusted closes. strategy.py:37-53,157-166; README.md:11.
Survivorship bias: The fixed universe contains the two existing Treasury ETFs TIP and IEF. The README identifies possible ex-post fund selection bias and states that stock-constituent survivorship bias does not apply to the fixed pair. config.yaml:2-4; strategy.py:229-233; README.md:11,37.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.00874021854347129, ann_vol=0.07100421952352663, sharpe=-0.08819512583702388, max_drawdown=-0.1601653105885643, total_return=-0.02738230274751663, turnover=6.659008885414778, n_trades=76
    oos: cagr=0.03926316290962051, ann_vol=0.062024602947546165, sharpe=0.6519153063573007, max_drawdown=-0.08059866950032335, total_return=0.12178884422930825, turnover=3.1063027878982634, n_trades=72
    full: cagr=0.014280753041128635, ann_vol=0.06679056638637257, sharpe=0.24566318826798883, max_drawdown=-0.16498553686288175, total_return=0.09107168247783326, turnover=9.76531167331304, n_trades=148
    benchmark static_50_50 full: cagr=0.011755234970205919, ann_vol=0.06702154298174401, sharpe=0.20784782158784681, max_drawdown=-0.170615724046432, total_return=0.07447908804418346, turnover=1.378698858246723, n_trades=148
    benchmark breakevens_only full: cagr=0.017767719059401, ann_vol=0.06589222709096611, sharpe=0.3001984664262651, max_drawdown=-0.16444499845633243, total_return=0.11433315330160299, turnover=17.56752696225911, n_trades=148
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 341_monitoring_attention_inflation_news\nuniverse:\n  inflation_linked: TIP\n  nominal: IEF\nseries:\n  attention: EMVMACROINFLATION\n  breakeven: T10YIE\nin_sample:\n  start: 2019-11-01\n  end: 2022-12-31\nout_of_sample:\n  start: 2023-01-01\n  end: 2025-12-31\nsignal:\n  attention_lag_months: 2\n  attention_window_months: 60\n  attention_quantile: 0.80\n  breakeven_lookback_sessions: 20\n  breakeven_lag_sessions: 1\n  breakeven_max_quote_age_days: 7\nweights:\n  neutral_tip: 0.50\n  tilted_tip: 0.75\ntransaction_cost_bps: 2.0\n'
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
from hashlib import sha256
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
import json
import math
import time

import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
ALFRED_CSV = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
DAY = pd.Timedelta(days=1)
SESSIONS_PER_YEAR = 252


def save_csv(frame, path):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        frame.to_csv(temp_path)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def read_csv(source):
    return pd.read_csv(source, index_col=0, parse_dates=True, na_values=["."])


def load_prices(tickers, start, end):
    cache = CACHE / f"yahoo_{'-'.join(tickers)}_{start.date()}_{end.date()}.csv"
    if cache.exists():
        return read_csv(cache).loc[:, tickers]

    yf.set_tz_cache_location(str(CACHE / "yfinance"))
    for attempt in range(3):
        raw = yf.download(
            tickers, start=str(start.date()), end=str((end + DAY).date()),
            auto_adjust=False, progress=False, timeout=15,
        )
        prices = raw.get("Adj Close", pd.DataFrame()).reindex(columns=tickers)
        if not prices.empty and prices.notna().all().all():
            prices.index = prices.index.tz_localize(None)
            if (prices.index[0] - start).days <= 7 and (end - prices.index[-1]).days <= 7:
                save_csv(prices, cache)
                return prices
        if attempt < 2:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Yahoo Finance did not return complete adjusted closes for {tickers}.")


def load_vintages(session, series, vintage_dates, last_dates, first_dates=None):
    """Pair each ALFRED snapshot with its own observation range."""
    frames = []
    batch_size = 12
    for offset in range(0, len(vintage_dates), batch_size):
        batch = slice(offset, offset + batch_size)
        dates = vintage_dates[batch]
        params = {
            "id": ",".join([series] * len(dates)),
            "vintage_date": ",".join(dates.strftime("%Y-%m-%d")),
            "coed": ",".join(last_dates[batch].strftime("%Y-%m-%d")),
        }
        if first_dates is not None:
            params["cosd"] = ",".join(first_dates[batch].strftime("%Y-%m-%d"))
        digest = sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        path = CACHE / f"alfred_{series}_{digest}.csv"
        columns = [f"{series}_{date:%Y%m%d}" for date in dates]

        if path.exists():
            frame = read_csv(path)
        else:
            for attempt in range(3):
                try:
                    response = session.get(ALFRED_CSV, params=params, timeout=(5, 15))
                    response.raise_for_status()
                    frame = read_csv(StringIO(response.text))
                    break
                except (requests.RequestException, pd.errors.ParserError) as error:
                    if attempt == 2:
                        raise RuntimeError(f"ALFRED download failed for {series}, {dates[0]:%Y-%m-%d}.") from error
                    time.sleep(2 ** attempt)

        if frame.empty or not isinstance(frame.index, pd.DatetimeIndex) or list(frame.columns) != columns:
            raise ValueError(f"Unexpected ALFRED data for {series}, {dates[0]:%Y-%m-%d}.")
        frame = frame.apply(pd.to_numeric, errors="raise").sort_index()
        if not path.exists():
            save_csv(frame, path)
        frame.columns = dates
        frames.append(frame)
    return pd.concat(frames, axis=1).sort_index()


def make_targets(sessions, evaluation_start, config):
    """Create closing targets indexed by execution date, one session after the signal."""
    months = sessions.to_period("M")
    month_end_positions = np.flatnonzero(months[:-1] != months[1:])
    execution_dates = sessions[month_end_positions + 1]
    first = max(0, execution_dates.searchsorted(evaluation_start, side="right") - 1)
    month_end_positions = month_end_positions[first:]
    execution_dates = execution_dates[first:]
    decisions = sessions[month_end_positions]

    rules = config["signal"]
    vintage_dates = decisions - DAY
    attention_cutoffs = (decisions.to_period("M") - rules["attention_lag_months"]).to_timestamp()
    recent_positions = month_end_positions - rules["breakeven_lag_sessions"]
    prior_positions = recent_positions - rules["breakeven_lookback_sessions"]
    if not len(prior_positions) or prior_positions[0] < 0:
        raise ValueError("Insufficient ETF sessions to construct the monthly signals.")
    quote_dates = sessions[recent_positions]
    prior_dates = sessions[prior_positions]
    quote_age = pd.Timedelta(days=rules["breakeven_max_quote_age_days"])

    with requests.Session() as session:
        session.trust_env = False
        attention = load_vintages(session, config["series"]["attention"], vintage_dates, attention_cutoffs)
        breakeven = load_vintages(
            session, config["series"]["breakeven"], vintage_dates, quote_dates, prior_dates - quote_age,
        )

    neutral = config["weights"]["neutral_tip"]
    tilt = config["weights"]["tilted_tip"]
    targets = pd.DataFrame({
        "strategy": neutral, "static_50_50": 0.50, "breakevens_only": neutral,
    }, index=execution_dates)
    for execution, vintage, cutoff, recent_date, prior_date in zip(
        execution_dates, vintage_dates, attention_cutoffs, quote_dates, prior_dates,
    ):
        coverage = attention[vintage].loc[:cutoff].dropna()
        attention_high = False
        if not coverage.empty:
            window = pd.date_range(end=coverage.index[-1], periods=rules["attention_window_months"], freq="MS")
            history = coverage.reindex(window)
            attention_high = history.notna().all() and history.iloc[-1] > history.quantile(rules["attention_quantile"])

        quotes = breakeven[vintage].dropna().reindex(
            [prior_date, recent_date], method="ffill", tolerance=quote_age,
        )
        if quotes.iloc[1] > quotes.iloc[0]:
            targets.loc[execution, "breakevens_only"] = tilt
            if attention_high:
                targets.loc[execution, "strategy"] = tilt

    # If the sample opens mid-month, enter with the last available target.
    targets.index = targets.index.where(targets.index >= evaluation_start, evaluation_start)
    return targets


def simulate(prices, tip_targets, fee_rate):
    """Mark old holdings to each close and then fund any new targets net of fees."""
    relative_prices = prices.div(prices.shift())
    relative_prices.iloc[0] = 1.0
    holdings = np.zeros(prices.shape[1])
    nav = 1.0
    records = []

    for relatives, target in zip(relative_prices.to_numpy(), tip_targets.reindex(prices.index)):
        holdings *= relatives
        gross_nav = holdings.sum() if holdings.any() else nav
        turnover, trades = 0.0, 0
        if pd.notna(target):
            weights = np.array([target, 1.0 - target])
            # Solve net NAV + fees(net NAV) = gross NAV for self-financing trades.
            low, high = 0.0, gross_nav
            for _ in range(52):
                net_nav = (low + high) / 2.0
                traded = np.abs(weights * net_nav - holdings).sum()
                if net_nav + fee_rate * traded > gross_nav:
                    high = net_nav
                else:
                    low = net_nav
            net_nav = (low + high) / 2.0
            orders = np.abs(weights * net_nav - holdings)
            turnover = orders.sum() / gross_nav
            trades = np.count_nonzero(orders > 1e-12 * gross_nav)
            holdings = weights * net_nav
        else:
            net_nav = gross_nav
        records.append((net_nav / nav - 1.0, turnover, trades))
        nav = net_nav

    return pd.DataFrame(records, index=prices.index, columns=["return", "turnover", "n_trades"])


def finite(value):
    return float(value) if math.isfinite(value) else None


def metrics(daily_results, start, end):
    segment = daily_results.loc[start:end]
    daily = segment["return"]
    if daily.empty:
        raise ValueError(f"No return observations from {start.date()} to {end.date()}.")
    wealth = (1.0 + daily).cumprod()
    drawdown = wealth / wealth.cummax().clip(lower=1.0) - 1.0
    vol = daily.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": finite(wealth.iloc[-1] ** (SESSIONS_PER_YEAR / len(daily)) - 1.0),
        "ann_vol": finite(vol),
        "sharpe": finite(daily.mean() * SESSIONS_PER_YEAR / vol) if vol > 0 else None,
        "max_drawdown": finite(drawdown.min()),
        "total_return": finite(wealth.iloc[-1] - 1.0),
        "n_trades": int(segment["n_trades"].sum()),
        "turnover": finite(segment["turnover"].sum()),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    periods = {key: tuple(pd.Timestamp(config[field][bound]) for bound in ("start", "end"))
               for key, field in (("is", "in_sample"), ("oos", "out_of_sample"))}
    start, end = periods["is"][0], periods["oos"][1]
    if not start <= periods["is"][1] < periods["oos"][0] <= end:
        raise ValueError("IS and OOS must be in chronological order and must not overlap.")
    periods["full"] = start, end

    fee_rate = float(config["transaction_cost_bps"]) / 10_000
    neutral, tilted = (config["weights"][key] for key in ("neutral_tip", "tilted_tip"))
    if not 0 <= fee_rate < 1 or not 0 <= neutral < tilted <= 1:
        raise ValueError("Invalid costs or TIP weights.")
    tickers = [config["universe"][role] for role in ("inflation_linked", "nominal")]
    lookback = config["signal"]["breakeven_lookback_sessions"] + config["signal"]["breakeven_lag_sessions"]
    warmup = (start.to_period("M") - (lookback // 15 + 3)).start_time
    CACHE.mkdir(exist_ok=True)
    prices = load_prices(tickers, warmup, end)
    prices = prices.sort_index()
    if not prices.index.is_unique or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Invalid adjusted close history.")
    evaluation = prices.loc[start:end]
    targets = make_targets(prices.index, evaluation.index[0], config)

    reports = {}
    for model in targets:
        daily_results = simulate(evaluation, targets[model], fee_rate)
        reports[model] = {period: metrics(daily_results, *bounds) for period, bounds in periods.items()}

    data_range = f"{evaluation.index[0]:%Y-%m-%d} to {evaluation.index[-1]:%Y-%m-%d}"
    print(f"{config['strategy']} | {data_range} | {config['transaction_cost_bps']:g} bps per traded side")
    table = pd.DataFrame.from_dict(
        {(model, period.upper()): values for model, report in reports.items() for period, values in report.items()},
        orient="index",
    )
    table.index.names = ["portfolio", "period"]
    print(table.to_string(na_rep="n/a", formatters={
        "cagr": "{:.2%}".format, "ann_vol": "{:.2%}".format, "sharpe": "{:.2f}".format,
        "max_drawdown": "{:.2%}".format, "total_return": "{:.2%}".format, "turnover": "{:.2f}x".format,
    }))

    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **reports["strategy"],
        "benchmark": {model: reports[model] for model in ("static_50_50", "breakevens_only")},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()