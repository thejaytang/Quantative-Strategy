"""124_dollar_safe_haven_reversal
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/124_dollar_safe_haven_reversal.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 7f58470906099142d2404cd2e681f0aaa17db946 on 2026-11-26 16:09 Europe/Oslo, exit 0, output sha256 075f64b600dbd6afaaa29404d5b07dc6f72f6ed0475b5cef29ab9c98a960220f
Data source: Yahoo Finance 免费日线，经 yfinance 获取美国上市 ETF 的开盘价和收盘价。SPY 与 UUP 生成信号，UUP 也用于匹配窗口对照，FXE 与 FXY 构成反转组合，BIL 用作防御持仓和买入持有基准。本次日志记录的数据获取日期为 2026-11-26（Europe/Oslo）。[config.yaml:4-8, strategy.py:19-41, strategy.py:248-258, strategy.py:271-276, README.md:30, logs/codex_fix_d2_run_20261126_131116.log:1-2]
Data fetched: 2026-11-26
Sample period: 实际回测交易日为 2010-06-07 至 2026-11-24，共 4,144 个交易日。代码请求从样本起点提前 14 个日历日的数据，计算信号所需的前收盘价。[config.yaml:18-24, strategy.py:240-253, strategy.py:256-266, logs/codex_fix_d2_run_20261126_131116.log:1, logs/codex_fix_d2_run_20261126_131116.log:39]   data_range (from output): 2010-06-07 to 2026-11-24
IS/OOS split: 含边界的连续日历划分：IS 为 2010-06-07 至 2023-12-31（3,416 个交易日），OOS 为 2024-01-01 至 2026-11-24（728 个交易日），首个 OOS 交易日为 2024-01-02。组合在全样本连续模拟后按日期切分报告，持仓可跨边界。README 说明 OOS 与研究假设所依据的论文研究期部分重叠。[config.yaml:18-24, strategy.py:240-245, strategy.py:270-288, README.md:22-24, logs/codex_fix_d2_run_20261126_131116.log:39]
Costs and slippage: 每笔买入和卖出均按各自成交名义金额计 2.5 bps 交易费用和 2.5 bps 固定滑点，每边合计 5 bps。令 c = 0.0005，首次买入 BIL 的因子为 1/(1+c)，进入外汇组合和切回 BIL 时各乘 (1-c)/(1+c)，先扣卖出成本再用净额买入。滑点已作为固定成本建模。[config.yaml:13-15, strategy.py:228-238, strategy.py:81-104, README.md:24, README.md:34, logs/codex_fix_d2_run_20261126_131116.log:5]
Signal lag: 交易日 t 收盘后，用 SPY 和 UUP 的复权收盘至收盘收益判断 SPY < −1% 且 UUP > 0%。信号在下一交易日 t+1 开盘执行，模拟卖出 BIL 并等额买入 FXE 和 FXY。按预先确定的持有期，在第五个持有交易日收盘切回 BIL。该退出日的新信号仍在下一交易日开盘执行。[config.yaml:4-13, strategy.py:60-78, strategy.py:81-104, strategy.py:256-275, README.md:9-12]
Price adjustment: Yahoo Finance split- and distribution-adjusted Open and Close (yfinance auto_adjust=True)
Survivorship bias: 固定使用事后可见的存续 ETF 作为代理，存在代理选择及存续偏差的可能。代码从固定标的列表取数。[config.yaml:4-8, strategy.py:220-226, strategy.py:248-254, README.md:34]
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.020911261966735983, ann_vol=0.037598100655656344, sharpe=-0.5432533431445117, sharpe_excess=-0.7524738465926697, max_drawdown=-0.373857531426668, total_return=-0.2490900511562555, turnover=688.6556721639181, n_trades=345
    oos: cagr=0.03196788075520307, ann_vol=0.0326260761001361, sharpe=0.9807657012344821, sharpe_excess=-0.3300132777494527, max_drawdown=-0.02896334407945189, total_return=0.09516631341243587, turnover=115.94202898550725, n_trades=58
    full: cagr=-0.011821928575988072, ann_vol=0.036791124636116494, sharpe=-0.3048480355441712, sharpe_excess=-0.6857678045172146, max_drawdown=-0.373857531426668, total_return=-0.17762871962007643, turnover=804.5977011494253, n_trades=403
    benchmark BIL full: cagr=0.014055655325276195, ann_vol=0.0026432041182881346, sharpe=5.282100306017677, sharpe_excess=None, max_drawdown=-0.004597658194419263, total_return=0.25800623131584244, turnover=0.9995002498750625, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 124_dollar_safe_haven_reversal\ncategory: macro\n\nuniverse:\n  equity_signal: SPY\n  dollar_signal_and_comparison: UUP\n  reversal_sleeve: [FXE, FXY]\n  defensive_benchmark: BIL\n\nparameters:\n  equity_stress_return: -0.01\n  dollar_confirmation_return: 0.0\n  holding_sessions: 5\n  transaction_fee_bps: 2.5  # Per side, on traded notional.\n  slippage_bps: 2.5  # Per side, on traded notional.\n  annualization_sessions: 252\n\n# Inclusive calendar dates. The OOS label denotes a chronological holdout.\nin_sample:\n  start: 2010-06-07\n  end: 2023-12-31\nout_of_sample:\n  start: 2024-01-01\n  end: 2026-11-24\n\nregimes:\n  pre_2024:\n    start: 2010-06-07\n    end: 2023-12-31\n  "2024_2025":\n    start: 2024-01-01\n    end: 2025-12-31\n  "2026":\n    start: 2026-01-01\n    end: 2026-11-24\n'
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
"""Backtest the C25 dollar reversal rule with daily US-listed ETF prices."""

import json
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"


def load_prices(ticker, start, end):
    cache_file = CACHE / f"yahoo_adjusted_{ticker}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    cached = cache_file.exists()
    if cached:
        prices = pd.read_csv(cache_file, index_col="Date", parse_dates=["Date"])
        dates = prices.pop("FetchedAt").unique()
        if len(dates) != 1 or pd.isna(dates[0]):
            raise ValueError(f"Missing or inconsistent fetch date for {ticker}.")
        fetched_at = dates[0]
    else:
        # Yahoo uses an exclusive end date. Auto-adjust both sides of the bar.
        history = yf.Ticker(ticker).history(
            start=start.date(),
            end=(end + pd.Timedelta(days=1)).date(),
            auto_adjust=True,
            actions=False,
            timeout=15,
        )
        if history.empty:
            raise RuntimeError(f"Yahoo Finance returned no daily data for {ticker}.")
        prices = history[["Open", "Close"]].copy()
        prices.index = prices.index.tz_localize(None).normalize()
        fetched_at = datetime.now(ZoneInfo("Europe/Oslo")).date().isoformat()
    prices = prices.sort_index().loc[start:end]
    if prices.empty or not prices.index.is_unique:
        raise ValueError(f"Missing or duplicate price dates for {ticker}.")
    values = prices.to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"Invalid adjusted prices for {ticker}.")
    if not cached:
        with NamedTemporaryFile(mode="w", dir=CACHE, delete=False) as temp:
            temp_path = Path(temp.name)
            try:
                prices.assign(FetchedAt=fetched_at).to_csv(temp, index_label="Date")
                temp.close()
                temp_path.replace(cache_file)
            finally:
                temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def event_windows(triggers, holding_sessions):
    """Return (entry open, exit close) row offsets for non-overlapping events."""
    windows = []
    next_signal = 0
    for signal in np.flatnonzero(triggers.to_numpy()[:-1]):
        if signal >= next_signal:
            entry = signal + 1
            exit_session = entry + holding_sessions - 1
            windows.append((entry, exit_session))
            # The exit-close signal can be acted on at the following open.
            next_signal = exit_session
    return windows


def sleeve_path(opens, closes, sleeve, entry, exit_session):
    """Value of equal-dollar positions bought at the entry open, before fees."""
    return closes.iloc[entry : exit_session + 1][sleeve].div(
        opens.iloc[entry][sleeve]
    ).mean(axis=1)


def simulate(opens, closes, defensive, sleeve, windows, buy_factor, switch_factor):
    defensive_close = closes[defensive]
    defensive_open = opens[defensive]
    previous_close = defensive_close.shift()
    gross = defensive_close.div(previous_close)
    gross.iloc[0] = defensive_close.iloc[0] / defensive_open.iloc[0] * buy_factor
    turnover = pd.Series(0.0, index=closes.index)
    turnover.iloc[0] = buy_factor
    trades = pd.Series(0, index=closes.index, dtype=int)
    trades.iloc[0] = 1

    for entry, exit_session in windows:
        path = sleeve_path(opens, closes, sleeve, entry, exit_session)
        daily = path.div(path.shift(fill_value=1.0))
        gross.loc[path.index] = daily
        gross.iloc[entry] *= defensive_open.iloc[entry] / previous_close.iloc[entry]
        gross.iloc[entry] *= switch_factor
        turnover.iloc[entry] += 1 + switch_factor
        trades.iloc[entry] += 1
        if exit_session < len(closes):
            gross.iloc[exit_session] *= switch_factor
            turnover.iloc[exit_session] += 1 + switch_factor
            trades.iloc[exit_session] += 1
    return pd.DataFrame({"return": gross - 1, "turnover": turnover, "n_trades": trades})


def metrics(daily, cash_returns, annualization):
    def finite(value):
        return float(value) if np.isfinite(value) else None

    def sharpe(returns):
        vol = returns.std(ddof=1)
        return finite(returns.mean() / vol * np.sqrt(annualization)) if vol > 0 else None

    n = len(daily)
    if n == 0:
        return dict.fromkeys(
            ("cagr", "ann_vol", "sharpe", "sharpe_excess", "max_drawdown", "total_return", "turnover")
        ) | {"n_trades": 0, "n_sessions": 0}
    returns = daily["return"]
    equity = (1 + returns).cumprod()
    vol = returns.std(ddof=1) * np.sqrt(annualization)
    return {
        "cagr": finite(equity.iloc[-1] ** (annualization / n) - 1),
        "ann_vol": finite(vol),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(returns - cash_returns),
        "max_drawdown": finite((equity / equity.cummax().clip(lower=1) - 1).min()),
        "total_return": finite(equity.iloc[-1] - 1),
        "turnover": finite(daily["turnover"].sum()),
        "n_trades": int(daily["n_trades"].sum()),
        "n_sessions": n,
    }


def summarize_events(opens, closes, reversal, dollar, defensive, windows, periods, switch_factor):
    rows = []
    for entry, exit_session in windows:
        if exit_session >= len(closes):
            break
        fx_return = sleeve_path(opens, closes, reversal, entry, exit_session).iloc[-1] - 1
        usd_return = closes[dollar].iloc[exit_session] / opens[dollar].iloc[entry] - 1
        cash_return = closes[defensive].iloc[exit_session] / opens[defensive].iloc[entry] - 1
        rows.append((closes.index[entry], closes.index[exit_session], fx_return, usd_return, cash_return))
    events = pd.DataFrame(rows, columns=["entry", "exit", "reversal", "dollar", "cash"])
    events["reversal_minus_dollar"] = events["reversal"] - events["dollar"]
    events["reversal_minus_cash"] = events["reversal"] - events["cash"]
    events["net_reversal_minus_cash"] = (1 + events["reversal"]) * switch_factor**2 - (1 + events["cash"])
    summaries = {}
    for name, (start, end) in periods.items():
        selected = events.loc[(events["entry"] >= start) & (events["exit"] <= end)]
        n_events = len(selected)
        cash_excess = selected["reversal_minus_cash"]
        std = cash_excess.std(ddof=1)
        summaries[name] = {
            "n_events": n_events,
            "mean_reversal_return": float(selected["reversal"].mean()) if n_events else None,
            "mean_dollar_return": float(selected["dollar"].mean()) if n_events else None,
            "mean_return_difference": float(selected["reversal_minus_dollar"].mean()) if n_events else None,
            "mean_cash_return": float(selected["cash"].mean()) if n_events else None,
            "mean_reversal_minus_cash": float(cash_excess.mean()) if n_events else None,
            "t_stat_reversal_minus_cash": float(cash_excess.mean() / std * np.sqrt(n_events)) if std > 0 else None,
            "mean_net_reversal_minus_cash": float(selected["net_reversal_minus_cash"].mean()) if n_events else None,
        }
    return summaries


def date_pair(settings):
    start, end = (pd.Timestamp(settings[key]) for key in ("start", "end"))
    if start > end:
        raise ValueError("Each reporting period must run from start to end.")
    return start, end


def display_reports(reports, regimes, events):
    columns = ["CAGR", "Ann. vol", "Sharpe (rf=0)", "Sharpe vs cash", "Max DD", "Total", "Turnover", "Trades", "Sessions"]

    def fmt(value, spec):
        return format(value, spec) if value is not None else "—"

    def table(by_portfolio):
        rows = {}
        for portfolio, periods in by_portfolio.items():
            for name, result in periods.items():
                rows[(name, portfolio)] = [
                    fmt(result["cagr"], ".2%"),
                    fmt(result["ann_vol"], ".2%"),
                    fmt(result["sharpe"], ".3f"),
                    fmt(result["sharpe_excess"], ".3f"),
                    fmt(result["max_drawdown"], ".2%"),
                    fmt(result["total_return"], ".2%"),
                    fmt(result["turnover"], ".2f") + "x" if result["turnover"] is not None else "—",
                    result["n_trades"],
                    result["n_sessions"],
                ]
        return pd.DataFrame.from_dict(rows, orient="index", columns=columns)

    print("\nPortfolio performance after costs")
    print(table(reports).to_string())
    print("\nCalendar regimes after costs")
    print(table(regimes).to_string())
    print("\nMatched complete windows, mean entry-open to exit-close returns and gross cash-excess t-stat")
    event_table = pd.DataFrame.from_dict(events, orient="index")
    for column in event_table.columns.drop("n_events"):
        spec = ".2f" if column == "t_stat_reversal_minus_cash" else ".2%"
        event_table[column] = event_table[column].map(lambda value: "—" if pd.isna(value) else format(value, spec))
    print(event_table.rename(columns={
        "n_events": "Events",
        "mean_reversal_return": "Reversal gross",
        "mean_dollar_return": "Dollar gross",
        "mean_return_difference": "Gross vs dollar",
        "mean_cash_return": "Cash hold",
        "mean_reversal_minus_cash": "Gross vs cash",
        "t_stat_reversal_minus_cash": "Gross vs cash t",
        "mean_net_reversal_minus_cash": "Net vs cash",
    }).to_string())


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    universe = config["universe"]
    params = config["parameters"]
    equity = universe["equity_signal"]
    dollar = universe["dollar_signal_and_comparison"]
    reversal = universe["reversal_sleeve"]
    defensive = universe["defensive_benchmark"]
    holding = params["holding_sessions"]
    fee_bps = params["transaction_fee_bps"]
    slippage_bps = params["slippage_bps"]
    cost = (fee_bps + slippage_bps) / 10_000
    annualization = params["annualization_sessions"]
    if not isinstance(holding, int) or holding < 1 or not 0 <= cost < 1 or annualization <= 0:
        raise ValueError("Invalid holding period, transaction cost, or annualization factor.")
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("Fees and slippage must be nonnegative.")
    buy_factor = 1 / (1 + cost)
    # Sell the old position and buy with the proceeds after sell and buy costs.
    switch_factor = (1 - cost) * buy_factor

    in_sample = date_pair(config["in_sample"])
    out_of_sample = date_pair(config["out_of_sample"])
    if in_sample[1] + pd.Timedelta(days=1) != out_of_sample[0]:
        raise ValueError("The IS and OOS calendar ranges must be contiguous and non-overlapping.")
    start, end = in_sample[0], out_of_sample[1]
    periods = {"is": in_sample, "oos": out_of_sample, "full": (start, end)}
    regime_dates = {str(name): date_pair(settings) for name, settings in config["regimes"].items()}

    CACHE.mkdir(exist_ok=True)
    warmup = start - pd.Timedelta(days=14)
    tickers = dict.fromkeys([equity, dollar, *reversal, defensive])
    with TemporaryDirectory(dir=CACHE, prefix=".yfinance-") as yahoo_cache:
        yf.set_tz_cache_location(yahoo_cache)
        snapshots = {ticker: load_prices(ticker, warmup, end) for ticker in tickers}
    prices = {ticker: frame for ticker, (frame, _) in snapshots.items()}
    fetched_at = {ticker: stamp for ticker, (_, stamp) in snapshots.items()}
    calendar = prices[equity].index
    opens = pd.DataFrame({ticker: frame["Open"].reindex(calendar) for ticker, frame in prices.items()})
    closes = pd.DataFrame({ticker: frame["Close"].reindex(calendar) for ticker, frame in prices.items()})
    missing = closes.loc[start:end].isna()
    if missing.to_numpy().any():
        raise ValueError(f"Missing ETF prices on {equity} sessions: {missing.sum().to_dict()}")
    signal_returns = closes[[equity, dollar]].pct_change(fill_method=None).loc[start:end]
    if signal_returns.empty or signal_returns.isna().to_numpy().any():
        raise ValueError("The signal needs a previous close for both ETFs on every reported session.")
    opens = opens.loc[start:end]
    closes = closes.loc[start:end]
    triggers = (signal_returns[equity] < params["equity_stress_return"]) & (
        signal_returns[dollar] > params["dollar_confirmation_return"]
    )
    windows = event_windows(triggers, holding)
    benchmark_name = f"{defensive} buy and hold"
    portfolios = {
        "Strategy": simulate(opens, closes, defensive, reversal, windows, buy_factor, switch_factor),
        benchmark_name: simulate(opens, closes, defensive, [], [], buy_factor, switch_factor),
        f"Matched {dollar}": simulate(opens, closes, defensive, [dollar], windows, buy_factor, switch_factor),
    }
    cash_returns = portfolios[benchmark_name]["return"]

    def report(ranges):
        return {
            name: {
                period: metrics(daily.loc[lo:hi], cash_returns.loc[lo:hi], annualization)
                for period, (lo, hi) in ranges.items()
            }
            for name, daily in portfolios.items()
        }

    reports = report(periods)
    strategy = reports["Strategy"]
    if not strategy["is"]["n_sessions"] or not strategy["oos"]["n_sessions"]:
        raise ValueError("Yahoo Finance data must cover both the IS and OOS periods.")
    regimes = report(regime_dates)
    events = summarize_events(opens, closes, reversal, dollar, defensive, windows, periods | regime_dates, switch_factor)
    actual_range = f"{closes.index[0]:%Y-%m-%d} to {closes.index[-1]:%Y-%m-%d}"
    print(f"{config['strategy']} | Yahoo Finance adjusted daily bars | {actual_range}")
    if closes.index[0] > start or closes.index[-1] < end:
        print(f"Configured inclusive bounds {start.date()} to {end.date()}; reporting observed sessions.")
    fetch_label = next(iter(fetched_at.values())) if len(set(fetched_at.values())) == 1 else json.dumps(fetched_at)
    print(f"DATA_FETCHED_AT: {fetch_label}")
    print("PRICE_ADJUSTMENT: Yahoo Finance split- and distribution-adjusted Open and Close (yfinance auto_adjust=True)")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"Transaction cost: {fee_bps + slippage_bps:g} bps per side ({fee_bps:g} fees + {slippage_bps:g} slippage)")
    display_reports(reports, regimes, events)
    result = {
        "strategy": config["strategy"],
        "data_range": actual_range,
        **strategy,
        "benchmark": {"ticker": defensive, **reports[benchmark_name]},
        "matched_dollar": {"ticker": dollar, **reports[f"Matched {dollar}"]},
        "regimes": regimes,
        "event_windows": events,
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()