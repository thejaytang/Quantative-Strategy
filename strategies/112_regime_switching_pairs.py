"""112_regime_switching_pairs
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/112_regime_switching_pairs.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit ad594150a0142009ae8f6ffab0995de08b2f8304 on 2026-11-26 13:49 Europe/Oslo, exit 0, output sha256 eff9b32ee86bf33a9a9f21d9299312696e66f864f4933c4f53ebc56ae3ee8a96
Data source: Yahoo Finance 公开日频复权开盘价和收盘价，通过 yfinance 下载或从同口径 CSV 缓存读取。股票配对为 XOM/CVX、KO/PEP、HD/LOW、JPM/BAC，基准为 SPY ETF。
Data fetched: 2026-11-26 13:49 Europe/Oslo (run time)
Sample period: 请求的数据区间为 2008-01-01 至 2025-12-31。日志中的首个预热交易日为 2008-01-02，回测报告区间为 2010-01-04 至 2025-12-31。 data_range (from output): 2010-01-04 to 2025-12-31
IS/OOS split: 配置的 IS 为 2010-01-01 至 2018-12-31，OOS 为 2019-01-01 至 2025-12-31。实际交易日分别为 2010-01-04 至 2018-12-31 和 2019-01-02 至 2025-12-31。组合自 IS 首日空仓启动，连续模拟至 OOS 结束，再按区间切片统计。对冲比率和半衰期在 OOS 中仍按当时的历史窗口更新。README 记载两段均未用于调参。
Costs and slippage: 每腿每边按实际成交绝对名义金额收取 5 bps，并在开盘成交时扣减现金。文档将该费用作为佣金和滑点的合并估计。滑点未单独建模，配对两腿按当日复权开盘价模拟成交。借券费年化 50 bps，以上一交易日收盘空头市值乘以相邻交易日之间的自然日数并按 365 天折算。SPY 买入持有基准在首日开盘计一次 5 bps 买入成本。
Signal lag: 交易日 t 收盘计算的进出场信号最早在下一交易日开盘执行。每月首个交易日开盘平掉旧仓，并取消上月尚未成交的入场信号。SPY 基准首日开盘买入，无收盘信号。
Price adjustment: yfinance 下载设置 auto_adjust=True，使用拆股和股息复权的 Open 和 Close。CSV 缓存存储相同的复权字段。成交使用复权开盘价，信号和日终估值使用复权收盘价。
Survivorship bias: 固定四对股票由事后已知仍存续的标的组成，README 明示存在生存者偏差和配对选择偏差。代码按配置中的固定清单取数。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.012485446688506041, ann_vol=0.023672319425127573, sharpe=-0.5189047543657448, max_drawdown=-0.11657362205416566, total_return=-0.10673945217602732, turnover=106.11861120656947, n_trades=213
    oos: cagr=0.0034355844554012638, ann_vol=0.024304851680974023, sharpe=0.1532359902339233, max_drawdown=-0.06823567538692588, total_return=0.02424262251041509, turnover=75.77618337845905, n_trades=150
    full: cagr=-0.005553293487243871, ann_vol=0.023953257742309983, sharpe=-0.2205187803621582, max_drawdown=-0.131314437943842, total_return=-0.08508447391168472, turnover=181.8947945850285, n_trades=363
    benchmark full: cagr=-0.003031142626246597, ann_vol=0.028101991170201387, sharpe=-0.09399361558063295, max_drawdown=-0.08334559850200107, total_return=-0.04731937804610442, turnover=228.9200655462535, n_trades=457
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 112_regime_switching_pairs\n\n# The first ticker is the dependent leg of each monthly log-price regression.\npairs:\n  - [XOM, CVX]\n  - [KO, PEP]\n  - [HD, LOW]\n  - [JPM, BAC]\nbenchmark_ticker: SPY\n\ndata_start: 2008-01-01  # Warm-up only until the first in-sample session.\nin_sample:\n  start: 2010-01-01\n  end: 2018-12-31\nout_of_sample:\n  start: 2019-01-01\n  end: 2025-12-31\n\nhedge_window: 252\nresidual_window: 63\nentry_z: 2.0\nexit_z: 0.25\nmin_half_life: 2.0\nmax_half_life: 20.0\nmax_holding_sessions: 20\nportfolio_gross_target: 1.0  # Divided equally among all pairs at entry.\ntransaction_cost_bps: 5.0  # Per leg, per side, applied to traded notional.\nshort_borrow_bps_annual: 50.0  # Applied to short notional using calendar days.\nsessions_per_year: 252\n'
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
"""Backtest C13's regime-gated pairs strategy and its benchmarks."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent


def load_prices(tickers, start, end):
    """Return aligned split- and dividend-adjusted Yahoo open and close prices."""
    cache = ROOT / "data_cache"
    cache.mkdir(exist_ok=True)
    path = cache / f"yahoo_adjusted_{start.date()}_{end.date()}_{'-'.join(tickers)}.csv"
    cached = path.exists()
    if cached:
        prices = pd.read_csv(path, header=[0, 1], index_col=0)
    else:
        with TemporaryDirectory(dir=cache, prefix=".yfinance-") as yahoo_cache:
            yf.set_tz_cache_location(yahoo_cache)
            prices = yf.download(
                tickers,
                start=str(start.date()),
                end=str((end + pd.Timedelta(days=1)).date()),
                auto_adjust=True,
                progress=False,
                threads=True,
                timeout=15,
            )

    if prices.empty:
        raise RuntimeError("Yahoo Finance returned no daily prices.")
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    prices = prices.sort_index().loc[start:end]
    columns = pd.MultiIndex.from_product([["Open", "Close"], tickers])
    prices = prices.reindex(columns=columns)
    if (
        prices.empty
        or prices.index.has_duplicates
        or prices.index[0] - start > pd.Timedelta(days=7)
        or end - prices.index[-1] > pd.Timedelta(days=7)
    ):
        raise RuntimeError("Daily prices do not cover the requested date range cleanly.")
    values = prices.to_numpy(dtype=float)
    invalid = np.argwhere(~np.isfinite(values) | (values <= 0))
    if invalid.size:
        row, col = invalid[0]
        field, ticker = prices.columns[col]
        raise RuntimeError(f"Missing or invalid {field} for {ticker} on {prices.index[row].date()}.")

    if not cached:
        with NamedTemporaryFile(dir=cache, suffix=".csv.tmp", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            prices.to_csv(temporary, index_label="Date")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return prices["Open"], prices["Close"]


def ols_slope(x, y):
    """Slope of a one-predictor regression with an intercept."""
    centered_x = x - x.mean()
    variance = centered_x @ centered_x
    if variance <= np.finfo(float).eps:
        return np.nan
    return (centered_x @ (y - y.mean())) / variance


def spread_signals(close, pairs, month_starts, config):
    """Fit before each month and score each close against earlier residuals."""
    shape = (len(close), len(pairs))
    hedge = np.full(shape, np.nan)
    score = np.full(shape, np.nan)
    fast_reversion = np.zeros(shape, dtype=bool)
    hedge_window = config["hedge_window"]
    residual_window = config["residual_window"]
    month_ends = [*month_starts[1:], len(close)]

    for pair_number, (first, second) in enumerate(pairs):
        y, x = np.log(close[[first, second]].to_numpy(dtype=float)).T
        for start, end in zip(month_starts, month_ends):
            if start < hedge_window:
                continue
            history = slice(start - hedge_window, start)
            beta = ols_slope(x[history], y[history])
            if not np.isfinite(beta) or beta <= 0:
                continue
            intercept = y[history].mean() - beta * x[history].mean()
            hedge[start:end, pair_number] = beta
            # Recompute all comparison residuals with this month's frozen parameters.
            residual_start = max(0, start - residual_window)
            residual = y[residual_start:end] - intercept - beta * x[residual_start:end]
            for day in range(max(start, residual_window), end):
                location = day - residual_start
                prior = residual[location - residual_window:location]
                stdev = prior.std(ddof=1)
                if stdev <= np.finfo(float).eps:
                    continue
                score[day, pair_number] = (residual[location] - prior.mean()) / stdev
                phi = ols_slope(prior[:-1], prior[1:])
                if 0 < phi < 1:
                    half_life = -np.log(2.0) / np.log(phi)
                    fast_reversion[day, pair_number] = (
                        config["min_half_life"] <= half_life <= config["max_half_life"]
                    )
    return hedge, score, fast_reversion


def simulate_pairs(opens, closes, hedge, score, entry_allowed, month_roll, config):
    """Book close-generated signals at the following open and mark to each close."""
    pairs = config["pairs"]
    pair_columns = [ticker for pair in pairs for ticker in pair]
    open_prices = opens[pair_columns].to_numpy(dtype=float).reshape(len(opens), len(pairs), 2)
    close_prices = closes[pair_columns].to_numpy(dtype=float).reshape(open_prices.shape)
    calendar_days = np.diff(opens.index.to_numpy(dtype="datetime64[D]")).astype(int)
    cost = config["transaction_cost_bps"] / 10_000
    daily_borrow = config["short_borrow_bps_annual"] / (10_000 * 365)
    gross_per_pair = config["portfolio_gross_target"] / len(pairs)

    cash = previous_equity = 1.0
    shares = np.zeros((len(pairs), 2))
    side = np.zeros(len(pairs), dtype=int)
    next_side = side.copy()
    entry_day = np.zeros(len(pairs), dtype=int)
    returns = np.empty(len(opens))
    entries = np.zeros(len(opens), dtype=int)
    turnover = np.zeros(len(opens))

    for day, (open_price, close_price) in enumerate(zip(open_prices, close_prices)):
        if day > 0:
            short_value = (-np.minimum(shares, 0) * close_prices[day - 1]).sum()
            cash -= short_value * daily_borrow * calendar_days[day - 1]
        equity_at_open = cash + (shares * open_price).sum()
        if equity_at_open <= 0:
            raise RuntimeError(f"Pairs portfolio equity exhausted at the open on {opens.index[day].date()}.")

        if month_roll[day]:
            # The old hedge expires at the first open of each calendar month.
            next_side.fill(0)
        for pair_number in np.flatnonzero(next_side != side):
            price = open_price[pair_number]
            if next_side[pair_number] == 0:
                target = np.zeros(2)
            else:
                beta = hedge[day - 1, pair_number]
                gross = equity_at_open * gross_per_pair
                dollars = next_side[pair_number] * gross * np.array([1, -beta]) / (1 + beta)
                target = dollars / price
                entry_day[pair_number] = day
                entries[day] += 1
            traded = (target - shares[pair_number]) * price
            traded_gross = np.abs(traded).sum()
            cash -= traded.sum() + cost * traded_gross
            turnover[day] += traded_gross / equity_at_open
            shares[pair_number] = target
            side[pair_number] = next_side[pair_number]

        equity = cash + (shares * close_price).sum()
        if equity <= 0:
            raise RuntimeError(f"Pairs portfolio equity exhausted at the close on {opens.index[day].date()}.")
        returns[day] = equity / previous_equity - 1
        previous_equity = equity

        for pair_number, z in enumerate(score[day]):
            if side[pair_number] != 0:
                held = day - entry_day[pair_number] + 1
                if abs(z) <= config["exit_z"] or held >= config["max_holding_sessions"]:
                    next_side[pair_number] = 0
            elif abs(z) > config["entry_z"] and entry_allowed[day, pair_number]:
                next_side[pair_number] = -1 if z > 0 else 1

    return pd.DataFrame({"return": returns, "entries": entries, "turnover": turnover}, index=opens.index)


def buy_and_hold(opens, closes, cost_bps):
    cost = cost_bps / 10_000
    results = pd.DataFrame({"return": closes.pct_change(), "entries": 0, "turnover": 0.0})
    results.iloc[0] = [closes.iloc[0] / (opens.iloc[0] * (1 + cost)) - 1, 1, 1 / (1 + cost)]
    return results


def metrics(results, sessions_per_year):
    returns = results["return"].to_numpy(dtype=float)
    equity = np.r_[1.0, np.cumprod(1 + returns)]
    annual_vol = returns.std(ddof=1) * np.sqrt(sessions_per_year) if len(returns) > 1 else None
    values = {
        "cagr": equity[-1] ** (sessions_per_year / len(returns)) - 1,
        "ann_vol": annual_vol,
        "sharpe": returns.mean() * sessions_per_year / annual_vol if annual_vol else None,
        "max_drawdown": (equity / np.maximum.accumulate(equity) - 1).min(),
        "total_return": equity[-1] - 1,
        "n_trades": int(results["entries"].sum()),
        "turnover": results["turnover"].sum(),
    }
    for key, value in values.items():
        if isinstance(value, (float, np.floating)):
            values[key] = float(value) if np.isfinite(value) else None
    return values


def print_table(series):
    row_format = "{:<23} {:<6} {:>10} {:>10} {:>9} {:>10} {:>10} {:>8} {:>10}"
    columns = [
        ("CAGR", "cagr", "{:.2%}"),
        ("Ann. vol", "ann_vol", "{:.2%}"),
        ("Sharpe", "sharpe", "{:.2f}"),
        ("Max DD", "max_drawdown", "{:.2%}"),
        ("Total ret", "total_return", "{:.2%}"),
        ("Entries", "n_trades", "{:d}"),
        ("Turnover", "turnover", "{:.2f}x"),
    ]
    print(row_format.format("Portfolio", "Period", *(label for label, _, _ in columns)))
    for name, periods in series.items():
        for period, values in periods.items():
            cells = [spec.format(values[key]) if values[key] is not None else "n/a"
                     for _, key, spec in columns]
            print(row_format.format(name, period.upper(), *cells))


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    data_start = pd.Timestamp(config["data_start"])
    if not data_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Warm-up, in-sample and out-of-sample dates must be chronological.")
    pairs = config["pairs"]
    benchmark = config["benchmark_ticker"]
    tickers = sorted({benchmark, *(ticker for pair in pairs for ticker in pair)})
    opens, closes = load_prices(tickers, data_start, oos_end)
    dates = closes.index
    months = dates.to_period("M")
    month_roll = np.r_[True, months[1:] != months[:-1]]
    hedge, score, fast_reversion = spread_signals(closes, pairs, np.flatnonzero(month_roll), config)

    live = (dates >= is_start) & (dates <= oos_end)
    opens, closes = opens.loc[live], closes.loc[live]
    periods = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    for name, (start, end) in periods.items():
        if closes.loc[start:end].empty:
            raise RuntimeError(f"No daily prices in the {name} evaluation period.")

    gated = simulate_pairs(opens, closes, hedge[live], score[live], fast_reversion[live],
                           month_roll[live], config)
    ungated = simulate_pairs(opens, closes, hedge[live], score[live], np.ones_like(fast_reversion[live]),
                             month_roll[live], config)
    buy_hold = buy_and_hold(opens[benchmark], closes[benchmark], config["transaction_cost_bps"])

    named_backtests = {"Regime-gated pairs": gated, "Ungated pairs": ungated, f"{benchmark} buy and hold": buy_hold}
    summary = {
        name: {period: metrics(backtest.loc[start:end], config["sessions_per_year"])
               for period, (start, end) in periods.items()}
        for name, backtest in named_backtests.items()
    }
    data_range = f"{closes.index[0].date()} to {closes.index[-1].date()}"
    print(f"{config['strategy']} | {data_range} | price warm-up starts {dates[0].date()}")
    for period, (start, end) in periods.items():
        sessions = closes.loc[start:end].index
        print(f"{period.upper():<5} {sessions[0].date()} to {sessions[-1].date()}")
    print_table(summary)
    payload = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **summary["Regime-gated pairs"],
        "benchmark": {
            "name": "Ungated pairs",
            **summary["Ungated pairs"],
            "buy_and_hold": {"ticker": benchmark, **summary[f"{benchmark} buy and hold"]},
        },
    }
    print("RESULTS_JSON: " + json.dumps(payload, allow_nan=False))


if __name__ == "__main__":
    main()
