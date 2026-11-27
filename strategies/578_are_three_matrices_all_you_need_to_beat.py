"""578_are_three_matrices_all_you_need_to_beat
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/578_are_three_matrices_all_you_need_to_beat.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 25a9179813de3addbbaee65a8c08579eb6e27d08 on 2026-11-27 19:08 Europe/Oslo, exit 0, output sha256 6dc13c97b5ae2ada9f9f740eaa5e3b8f2b8415f243f200561bc3144d11d0aba9
Data source: Public Yahoo Finance daily ETF quotes and corporate actions via yfinance. The fixed sector universe is XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV and XLY. SPY is the benchmark and market-regime input. BIL is the defensive allocation and excess-return reference.
Data fetched: 2026-11-27T18:08:37+00:00
Sample period: The configured data request starts on 2008-01-01 for warm-up. The reported backtest runs from 2013-01-02 through 2026-07-31, inclusive.   data_range (from output): {'start': '2013-01-02', 'end': '2026-07-31'}
IS/OOS split: IS is 2013-01-02 through 2021-12-31. OOS is 2022-01-03 through 2026-07-31. The official default simulation carries positions across the split, with transition estimates updated monthly using the trailing 36 completed transitions. The five sensitivity variants run only on IS. The script checks exact equality of the default IS-only run and the IS slice of the full default run.
Costs and slippage: The default models 3 bps commission and 2 bps flat slippage on the absolute opening notional of each buy and sell. Both charges are reserved during sizing and deducted from cash at the combined 5 bps rate. Fill prices use the quoted open. The IS cost sensitivity uses 6 bps commission and 4 bps slippage. The SPY opening purchase uses the default charge.
Signal lag: The strategy forms its signal from data through month-end close t and fills at the next observed trading session’s open under the configured one-session lag. The session index identifies month boundaries. The SPY benchmark is a fixed buy-and-hold allocation entered at the first backtest open.
Price adjustment: Yahoo Finance split-basis OHLC and ex-date distribution adjustment
Survivorship bias: The universe is a fixed set of nine Select Sector SPDR funds launched before the backtest. The design assumes continued availability of those funds, omits later sector funds, and uses a historical vendor snapshot subject to revision. The strategy does not construct a historical stock-constituent universe.
Parameter combinations tried: 6
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.1197949815894499, ann_vol=0.12831036949526273, sharpe=0.9463502998103694, sharpe_excess=0.9071360770866078, max_drawdown=-0.18476042058531994, total_return=1.767270781469319, turnover=7.101604882117022, n_trades=559
    oos: cagr=0.08290415814269325, ann_vol=0.1103605575146084, sharpe=0.7769995104336773, sharpe_excess=0.43063161268333255, max_drawdown=-0.11143703991805909, total_return=0.4373970983154216, turnover=6.871016069633281, n_trades=297
    full: cagr=0.10725547565748572, ann_vol=0.12255826984325069, sharpe=0.8929092810701442, sharpe_excess=0.7607125003781847, max_drawdown=-0.18476042058531994, total_return=2.9776669915370184, turnover=7.024089228608577, n_trades=856
    benchmark SPY full: cagr=0.14784703703007773, ann_vol=0.16803398917727722, sharpe=0.9050371167314135, sharpe_excess=0.8083830606821156, max_drawdown=-0.33717263923504326, total_return=5.479231407733765, turnover=0.07375521609619762, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 578_are_three_matrices_all_you_need_to_beat\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nbenchmark: SPY\ndefensive_etf: BIL\n\ndates:\n  data_start: 2008-01-01\n  start: 2013-01-02\n  is_end: 2021-12-31\n  end: 2026-07-31\n\nexecution:\n  lag_sessions: 1\n  cost_bps: 3.0\n  slippage_bps: 2.0\n\nmodel:\n  return_days: 126\n  volatility_days: 21\n  longer_volatility_days: 63\n  distance_days: 60\n  market_regime_days: 63\n  transition_months: 36\n  rank_buckets: 3\n  pooled_prior: 12.0\n  sector_count: 4\n  down_volatility_weight: 0.75\n  down_equity_weight: 0.40\n  distance_weight: 0.50\n  swap_threshold: 0.08\n\nrobustness:\n  - name: return_105_sessions\n    model: {return_days: 105}\n  - name: return_147_sessions\n    model: {return_days: 147}\n  - name: down_volatility_weight_0.60\n    model: {down_volatility_weight: 0.60}\n  - name: down_volatility_weight_0.90\n    model: {down_volatility_weight: 0.90}\n  - name: double_cost_and_slippage\n    execution: {cost_bps: 6.0, slippage_bps: 4.0}\n'
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
#!/usr/bin/env python3
"""C183 sector-ETF adaptation of Igor Halperin's OMD-Portfolio (2026).

Paper: https://arxiv.org/abs/2607.27461
Author's portfolio repository (private): https://github.com/ighalp/omd_portfolio
Companion OMD-Stocks research (ideas only, no code license stated):
https://github.com/ighalp/omd_finance
Comparison with hidden-state allocation (ideas only, no code license stated):
https://github.com/Newtype07/Portfolio-Allocation-Regime-Switching
Other search references: https://pypi.org/project/PyDTMC/
https://cdn.quantconnect.com/docs/i/Quantconnect-Research-Environment-Python.pdf
https://github.com/quantopian/research_public/blob/master/research/ml_algo.py
Independent implementation using public Yahoo Finance ETF data.
"""

import gzip
import hashlib
import json
import signal
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
ANNUAL_SESSIONS = 252
QUOTE_FIELDS = ("Open", "Close", "Adj Close", "Dividends", "Capital Gains", "Stock Splits")
MAX_RUNTIME_SECONDS = 840


def load_quotes(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_key = repr((tickers, start, end, QUOTE_FIELDS)).encode()
    cache_path = cache_dir / f"yahoo_{hashlib.sha256(cache_key).hexdigest()[:16]}.csv.gz"
    columns = pd.MultiIndex.from_product([QUOTE_FIELDS, tickers])

    def validate(quotes):
        if quotes is None or quotes.empty:
            raise ValueError("Yahoo Finance returned no daily quotes.")
        quotes = quotes.reindex(columns=columns)
        values = quotes.to_numpy(dtype=float)
        if (
            quotes.index.has_duplicates
            or not quotes.index.is_monotonic_increasing
            or not np.isfinite(values).all()
            or (quotes[["Open", "Close", "Adj Close"]] <= 0).any().any()
            or (quotes[["Dividends", "Capital Gains", "Stock Splits"]] < 0).any().any()
            or not start <= quotes.index[0] <= start + pd.Timedelta(days=7)
            or not end - pd.Timedelta(days=7) <= quotes.index[-1] <= end
        ):
            raise ValueError("Incomplete or invalid Yahoo Finance panel for the configured universe.")
        return quotes

    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
            fetched_at = stream.readline().strip()
            datetime.fromisoformat(fetched_at)
            quotes = pd.read_csv(stream, header=[0, 1], index_col=0, parse_dates=True)
        return validate(quotes), fetched_at

    with tempfile.TemporaryDirectory(dir=cache_dir, prefix="yfinance-") as download_cache:
        yf.set_tz_cache_location(download_cache)
        for attempt in range(2):
            quotes = yf.download(
                list(tickers), start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(), interval="1d",
                actions=True, auto_adjust=False, repair=False, progress=False, timeout=12,
            )
            try:
                quotes = validate(quotes)
            except ValueError:
                if attempt == 1:
                    raise
            else:
                break

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".csv.gz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8") as stream:
            stream.write(fetched_at + "\n")
            quotes.to_csv(stream)
        temporary_path.replace(cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return quotes, fetched_at


def price_factors(quotes):
    # Yahoo OHLC and distributions share the same split basis. Credit the
    # ex-date adjustment to the prior position before sizing at today's open.
    opening, closing = quotes["Open"], quotes["Close"]
    distribution = quotes["Dividends"] + quotes["Capital Gains"]
    ex_reference = closing.shift(1) - distribution
    if (ex_reference.iloc[1:] <= 0).any().any():
        raise ValueError("Non-positive ex-distribution reference price.")
    overnight = opening / ex_reference
    intraday = closing / opening
    close_factor = closing / ex_reference
    vendor_factor = quotes["Adj Close"].pct_change(fill_method=None) + 1
    adjustment_difference = float((vendor_factor - close_factor).abs().iloc[1:].to_numpy().max())
    if adjustment_difference > 0.0001:
        raise ValueError(f"Distribution or split adjustment disagrees with Yahoo: {adjustment_difference:g}")
    return overnight, intraday, close_factor, adjustment_difference


def rank_buckets(values, count, ascending):
    ranks = values.rank(axis=1, method="first", ascending=ascending)
    return ((ranks - 1) * count // values.shape[1]).to_numpy(dtype=int)


def transition_matrix(states, conditions, buckets, prior):
    # The condition belongs to the origin month of each completed transition.
    cells = (conditions[:-1] * buckets + states[:-1]) * buckets + states[1:]
    counts = np.bincount(cells.ravel(), minlength=buckets**3).reshape((buckets,) * 3)
    pooled = counts.sum(axis=0) + 0.5
    pooled /= pooled.sum(axis=1, keepdims=True)
    return (counts + prior * pooled) / (counts.sum(axis=2, keepdims=True) + prior)


def angular_distances(window):
    centered = window - window.mean(axis=0)
    market = centered.mean(axis=1)
    market_variance = market @ market
    if market_variance <= 0:
        raise ValueError("The distance window has no variance in its market factor.")
    residual = centered - np.outer(market, (market @ centered) / market_variance)
    full = np.arccos(np.clip(np.corrcoef(centered, rowvar=False), -1.0, 1.0))
    residual_distance = np.arccos(np.clip(np.corrcoef(residual, rowvar=False), -1.0, 1.0))
    if not np.isfinite(full).all() or not np.isfinite(residual_distance).all():
        raise ValueError("The distance window contains a constant return series.")
    np.fill_diagonal(full, 0)
    np.fill_diagonal(residual_distance, 0)
    return full, residual_distance


def select_sectors(scores, momentum, distance, previous, model):
    # Sorting by six-month return breaks ties within identical conditional states.
    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], -momentum[i], i))
    selected = list(previous)
    tilt = model["distance_weight"] / np.pi
    target_count = model["sector_count"]
    while len(selected) < target_count:
        remaining = [i for i in ranked if i not in selected]
        if selected:
            gains = scores[remaining] + tilt * distance[np.ix_(remaining, selected)].mean(axis=1)
            selected.append(remaining[int(gains.argmax())])
        else:
            selected.append(remaining[0])

    while True:
        remaining = [i for i in ranked if i not in selected]
        if not remaining:
            break
        connections = distance[np.ix_(remaining, selected)]
        added = connections.sum(axis=1, keepdims=True) - connections
        removed = distance[np.ix_(selected, selected)].sum(axis=1)
        gains = scores[remaining, None] - scores[selected]
        gains += tilt * (added - removed) / (target_count - 1)
        enter, leave = np.unravel_index(gains.argmax(), gains.shape)
        if gains[enter, leave] <= model["swap_threshold"]:
            break
        selected[leave] = remaining[enter]
    return selected


def target_schedule(close_factor, sectors, benchmark, start, lag, model):
    # The first observed session of a new month identifies the preceding
    # exchange month-end. There is no signal on a partial final month.
    sessions = close_factor.index
    months = sessions.to_period("M")
    month_ends = np.flatnonzero(months[1:] != months[:-1])
    first_feature = max(model[key] for key in (
        "return_days", "volatility_days", "longer_volatility_days",
        "distance_days", "market_regime_days",
    ))
    month_ends = month_ends[month_ends >= first_feature]

    returns = np.log(close_factor)
    sector_returns = returns.loc[:, sectors]
    momentum = sector_returns.rolling(model["return_days"]).sum().iloc[month_ends]
    volatility = sector_returns.rolling(model["volatility_days"]).std().iloc[month_ends]
    longer_vol = sector_returns.rolling(model["longer_volatility_days"]).std().iloc[month_ends]
    bull_market = returns[benchmark].rolling(model["market_regime_days"]).sum().iloc[month_ends] >= 0

    centralities = []
    residual_distances = []
    daily_sectors = sector_returns.to_numpy()
    for end_position in month_ends:
        left = end_position + 1 - model["distance_days"]
        full, residual = angular_distances(daily_sectors[left:end_position + 1])
        centralities.append(full.sum(axis=1) / (len(sectors) - 1))
        residual_distances.append(residual)

    buckets = model["rank_buckets"]
    return_states = rank_buckets(momentum, buckets, ascending=False)
    vol_states = rank_buckets(volatility, buckets, ascending=True)
    return_conditions = rank_buckets(longer_vol, buckets, ascending=True)
    vol_conditions = rank_buckets(pd.DataFrame(centralities), buckets, ascending=True)
    schedule = {}
    selected = []
    for month, signal_position in enumerate(month_ends):
        fill_position = signal_position + lag
        if fill_position >= len(sessions) or sessions[fill_position] < start:
            continue
        trailing_months = model["transition_months"]
        if month < trailing_months:
            raise ValueError("Use an earlier data_start to warm up the transition matrices.")
        history = slice(month - trailing_months, month + 1)
        return_chain = transition_matrix(
            return_states[history], return_conditions[history], buckets, model["pooled_prior"]
        )
        vol_chain = transition_matrix(
            vol_states[history], vol_conditions[history], buckets, model["pooled_prior"]
        )
        up = bool(bull_market.iloc[month])
        vol_weight = 0.0 if up else model["down_volatility_weight"]
        equity_weight = 1.0 if up else model["down_equity_weight"]
        return_score = return_chain[return_conditions[month], return_states[month], 0]
        vol_score = vol_chain[vol_conditions[month], vol_states[month], 0]
        scores = (1 - vol_weight) * return_score + vol_weight * vol_score
        selected = select_sectors(
            scores, momentum.iloc[month].to_numpy(), residual_distances[month], selected, model
        )
        weights = np.zeros(len(sectors) + 1)
        weights[selected] = equity_weight / len(selected)
        weights[-1] = 1 - equity_weight
        schedule[sessions[fill_position]] = (sessions[signal_position], weights)
    return schedule


def trade_at_open(open_values, cash, opening, target, rate):
    equity_before = cash + open_values.sum()
    # Solve for target value after paying costs on both buys and sells.
    low, high = 0.0, equity_before
    for _ in range(52):
        middle = (low + high) / 2
        required = middle + rate * np.abs(target * middle - open_values).sum()
        if required <= equity_before:
            low = middle
        else:
            high = middle
    # Leave a floating-point reserve so roundoff cannot spend more cash.
    investable = low * (1 - 16 * np.finfo(float).eps)
    fill_quantities = (target * investable - open_values) / opening
    traded_values = fill_quantities * opening
    notional = np.abs(traded_values).sum()
    next_cash = cash - traded_values.sum() - rate * notional
    if next_cash < 0:
        raise RuntimeError("Opening trades exceeded available cash.")
    n_trades = int(np.count_nonzero(np.abs(traded_values) > 1e-10 * equity_before))
    return target * investable, next_cash, notional / equity_before, n_trades


def simulate(opening, overnight, intraday, tickers, schedule, start, end, execution):
    session_prices = opening.loc[start:end, tickers]
    overnight_factors = overnight.loc[session_prices.index, tickers].to_numpy()
    intraday_factors = intraday.loc[session_prices.index, tickers].to_numpy()
    rate = (execution["cost_bps"] + execution["slippage_bps"]) / 10_000
    if not 0 <= rate < 1 or min(execution["cost_bps"], execution["slippage_bps"]) < 0:
        raise ValueError("Execution costs and slippage must be nonnegative and total less than 10,000 bps.")

    close_values = np.zeros(len(tickers))
    cash, prior_equity = 1.0, 1.0
    rows = []
    for day, (date, open_prices) in enumerate(zip(session_prices.index, session_prices.to_numpy())):
        open_values = close_values * overnight_factors[day]
        turnover, n_trades = 0.0, 0
        if date in schedule:
            _, target = schedule[date]
            open_values, cash, turnover, n_trades = trade_at_open(open_values, cash, open_prices, target, rate)
        close_values = open_values * intraday_factors[day]
        equity = cash + close_values.sum()
        rows.append((equity / prior_equity - 1, turnover, n_trades, cash))
        prior_equity = equity
    return pd.DataFrame(rows, columns=["return", "turnover", "n_trades", "cash"], index=session_prices.index)


def metrics(daily_results, defensive_returns):
    returns = daily_results["return"]
    excess = returns - defensive_returns.reindex(returns.index)
    if len(returns) < 2 or excess.isna().any():
        raise ValueError("A reported period lacks daily returns or its risk-free proxy.")
    growth = (1 + returns).cumprod()
    growth_with_start = np.r_[1.0, growth.to_numpy()]
    drawdown = growth_with_start / np.maximum.accumulate(growth_with_start) - 1
    root_year = np.sqrt(ANNUAL_SESSIONS)
    vol, excess_vol = returns.std(), excess.std()
    return {
        "cagr": float(growth.iloc[-1] ** (ANNUAL_SESSIONS / len(returns)) - 1),
        "ann_vol": float(vol * root_year),
        "sharpe": float(returns.mean() * root_year / vol) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() * root_year / excess_vol) if excess_vol > 0 else None,
        "max_drawdown": float(drawdown.min()),
        "total_return": float(growth.iloc[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * ANNUAL_SESSIONS),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def validate_model(sector_count, model, execution):
    if not 2 <= model["sector_count"] <= sector_count:
        raise ValueError("sector_count must be between 2 and the universe size.")
    if not 2 <= model["rank_buckets"] <= sector_count:
        raise ValueError("rank_buckets must be between 2 and the universe size.")
    for field in ("down_volatility_weight", "down_equity_weight"):
        if not 0 <= model[field] <= 1:
            raise ValueError(f"{field} must be between zero and one.")
    for field in ("volatility_days", "longer_volatility_days", "distance_days"):
        if model[field] < 3:
            raise ValueError(f"{field} must be at least three.")
    if min(model["return_days"], model["market_regime_days"], model["transition_months"]) < 1:
        raise ValueError("Return, market regime and transition windows must be positive.")
    if min(model["distance_weight"], model["swap_threshold"]) < 0 or model["pooled_prior"] <= 0:
        raise ValueError("Distance and swap settings must be nonnegative and the chain prior positive.")
    if execution["lag_sessions"] < 1 or not isinstance(execution["lag_sessions"], int):
        raise ValueError("lag_sessions must be a positive integer.")


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    sectors = sorted(config["universe"])
    benchmark, defensive = config["benchmark"], config["defensive_etf"]
    tickers = sectors + [defensive, benchmark]
    if len(set(tickers)) != len(tickers):
        raise ValueError("Universe, defensive ETF and benchmark must have distinct symbols.")
    data_start, start, is_end, end = (
        pd.Timestamp(config["dates"][key]) for key in ("data_start", "start", "is_end", "end")
    )
    if not data_start < start < is_end < end:
        raise ValueError("Expected data_start < start < is_end < end.")

    specifications = [("default", config["model"], config["execution"])]
    for variant in config["robustness"]:
        specifications.append((
            variant["name"],
            {**config["model"], **variant.get("model", {})},
            {**config["execution"], **variant.get("execution", {})},
        ))
    signatures = {json.dumps([model, execution], sort_keys=True) for _, model, execution in specifications}
    if len(signatures) != len(specifications):
        raise ValueError("Each robustness specification must differ from the other specifications.")
    for _, model, execution in specifications:
        validate_model(len(sectors), model, execution)

    quotes, fetched_at = load_quotes(tickers, data_start, end)
    opening = quotes["Open"]
    overnight, intraday, close_factor, adjustment_difference = price_factors(quotes)
    dates = opening.loc[start:end].index
    defensive_returns = close_factor.loc[dates, defensive] - 1
    defensive_returns.iloc[0] = intraday.loc[dates[0], defensive] - 1
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: Yahoo Finance split-basis OHLC and ex-date distribution adjustment")
    print(f"ADJUSTMENT_MAX_DAILY_DIFFERENCE_VS_YAHOO: {adjustment_difference:.8g}")
    print("ROBUSTNESS (IS only)")
    for name, model, execution in specifications:
        schedule = target_schedule(
            close_factor.loc[:is_end], sectors, benchmark, start, execution["lag_sessions"], model
        )
        daily_results = simulate(
            opening, overnight, intraday, sectors + [defensive], schedule, start, is_end, execution
        )
        report = metrics(daily_results, defensive_returns)
        print(f"  {name}: {json.dumps(report, allow_nan=False)}")
        if name == "default":
            default_is = daily_results

    _, model, execution = specifications[0]
    schedule = target_schedule(close_factor, sectors, benchmark, start, execution["lag_sessions"], model)
    official = simulate(opening, overnight, intraday, sectors + [defensive], schedule, start, end, execution)
    pd.testing.assert_frame_equal(official.loc[:is_end], default_is, check_exact=True)
    lag = execution["lag_sessions"]
    if not schedule or any(
        close_factor.index.get_loc(fill) - close_factor.index.get_loc(origin) != lag
        for fill, (origin, _) in schedule.items()
    ):
        raise RuntimeError("The simulated schedule failed its signal-to-fill lag check.")
    first_fill = min(schedule)

    benchmark_daily = simulate(
        opening, overnight, intraday, [benchmark], {dates[0]: (None, np.ones(1))},
        start, end, execution,
    )
    benchmark_results = {"ticker": benchmark}
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": dates[0].date().isoformat(), "end": dates[-1].date().isoformat()},
    }
    periods = {"is": dates <= is_end, "oos": dates > is_end, "full": slice(None)}
    for period, mask in periods.items():
        results[period] = metrics(official.loc[mask], defensive_returns)
        benchmark_results[period] = metrics(benchmark_daily.loc[mask], defensive_returns)
        print(f"{period.upper()}: {json.dumps(results[period], allow_nan=False)}")
        print(f"BENCHMARK {period.upper()} ({benchmark}): {json.dumps(benchmark_results[period], allow_nan=False)}")
    results["benchmark"] = benchmark_results
    if min(results[part]["n_trades"] for part in periods) <= 0:
        raise RuntimeError("The official specification must generate trades in each reported period.")
    print(
        f"LOOK_AHEAD_SELF_CHECK: {lag}-session open lag, close bars only through signal, "
        "ex-date distributions and split-basis prices, exact IS prefix match"
    )
    print(f"EXECUTION_CHECK: first_fill={first_fill.date()}, minimum_cash={official['cash'].min():.12g}")
    print("已回测 | 确有交易 | 待审")
    print(f"PARAM_COMBOS_TRIED: {len(specifications)}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False), flush=True)


def time_limit(_signum, _frame):
    raise TimeoutError("Backtest exceeded its 840-second runtime limit.")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, time_limit)
    signal.alarm(MAX_RUNTIME_SECONDS)
    try:
        main()
    finally:
        signal.alarm(0)
