"""187_midterm_cycle
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/187_midterm_cycle.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 92e75bea82090bd8b6723b610a02dabec59e659b on 2026-11-27 07:29 Europe/Oslo, exit 0, output sha256 abafc7430b781b87f3708f935f52eaf97477661f9147ee4dc14b332e1d3e196d
Data source: 免费的 Yahoo Finance 日频 ETF 历史行情，通过 yfinance 获取 SPY 和 BIL 的复权开盘价（config.yaml:4-6，README.md:20，strategy.py:36-53）。记录中的运行使用随附快照，数据获取时间为 2026-11-25T12:18:05+01:00（config.yaml:22-25，logs/codex_fix_d2_run_20261126_180740.log:3-4）。
Data fetched: 2026-11-27T07:29:07+01:00
Sample period: 配置区间为 2008-01-01 至 2025-12-31（config.yaml:8-13）。实际持仓起始交易日为 2008-01-02 至 2025-12-31，最后估值为 2026-01-02 开盘（logs/codex_fix_d2_run_20261126_180740.log:1-2）。   data_range (from output): 2008-01-02 to 2025-12-31
IS/OOS split: IS 为 2008-01-01 至 2015-12-31，OOS 为 2016-01-01 至 2025-12-31（config.yaml:8-13）。两段按持仓起始日从同一次连续模拟切出，使用同一固定规则。OOS 为回顾性留出区间，所引研究发表于 2026 年 5 月并覆盖该区间（strategy.py:185-205，README.md:16）。
Costs and slippage: 每条买卖腿按成交额扣除 5 bps 交易费和 1 bp 固定滑点，合计 6 bps，包含 BIL 腿。从现金建仓扣买入腿成本，换仓扣买卖两腿成本。模拟以扣费后的目标持仓和开盘交易前漂移权重之差计算成交额。期末在开盘估值并保留持仓（config.yaml:18-20，strategy.py:65-84，strategy.py:175-177，README.md:14，logs/codex_fix_d2_run_20261126_180740.log:1-2）。
Signal lag: 交易日 t 收盘时，仅按下一交易日所属日历年生成目标。下一交易日开盘执行，年份模 4 为 3 时持有 SPY，其余时间持有 BIL。该仓位对应当日开盘至再下一交易日开盘的收益（config.yaml:4-6，config.yaml:15-16，strategy.py:185-196，README.md:5，README.md:13）。
Price adjustment: yfinance auto_adjust=True, split- and distribution-adjusted opens
Survivorship bias: 直接使用固定的 SPY 和 BIL ETF 历史行情，未用现有指数成分股回填历史。事后选择仍存续的 ETF，仍有产品选择偏差（config.yaml:4-6，strategy.py:39-48，README.md:30）。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.003935866386340914, ann_vol=0.09600191538910473, sharpe=0.08918020955787137, sharpe_excess=0.07083219401116359, max_drawdown=-0.20149909859278192, total_return=0.03194697413511616, turnover=6.995802518488906, n_trades=7
    oos: cagr=0.06623918086876812, ann_vol=0.05613529468470241, sharpe=1.1729967589655312, sharpe_excess=0.8118262541856878, max_drawdown=-0.09585921497012362, total_return=0.8986184583043175, turnover=9.994003597841296, n_trades=10
    full: cagr=0.03806831979746761, ann_vol=0.07649456591338988, sharpe=0.5276164995401547, sharpe_excess=0.3715895182712607, max_drawdown=-0.20149909859278192, total_return=0.9592735730842343, turnover=16.9898061163302, n_trades=17
    benchmark: cagr=0.038892480214626435, ann_vol=0.04811305023622558, sharpe=0.8184622706136078, sharpe_excess=0.5605163432180224, max_drawdown=-0.14581816333288178, total_return=0.9874657741661725, turnover=14.678471859268825, n_trades=9058
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 187_midterm_cycle\ncategory: seasonality\n\nuniverse:\n  equity: SPY\n  cash_proxy: BIL\n\nin_sample:\n  start: 2008-01-01\n  end: 2015-12-31\nout_of_sample:\n  start: 2016-01-01\n  end: 2025-12-31\n\n# Year modulo 4 is 3 in the calendar year following a U.S. midterm.\npost_midterm_year_mod_4: 3\nstatic_equity_weight: 0.25\n# Charged on each dollar bought or sold, including the BIL leg.\ntransaction_cost_bps: 5.0\nslippage_bps: 1.0\n\nprice_snapshot:\n  sha256: 26b5614e1ac5ff754e4440067087aa082f73c556ddf5c548d4fdf93efefae98e\n  # Original snapshot write time, immediately after the yfinance download (Europe/Oslo).\n  fetched_at: "2026-11-25T12:18:05+01:00"\n'
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
"""Backtest the U.S. post-midterm calendar rule using adjusted ETF opens."""

import hashlib
import json
import math
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
YEAR_TYPES = ("Presidential election", "First year", "Midterm election", "Post-midterm")


def load_opens(tickers, start, end, snapshot):
    """Load aligned adjusted opens, including the two valuation boundary sessions."""
    padding = pd.Timedelta(days=10)
    fetch_start, fetch_end = start - padding, end + padding
    cache_file = CACHE / f"adjusted_open_{fetch_start.date()}_{fetch_end.date()}.csv"
    if cache_file.is_file():
        contents = cache_file.read_bytes()
        if hashlib.sha256(contents).hexdigest() == snapshot["sha256"]:
            prices = pd.read_csv(BytesIO(contents), index_col="date", parse_dates=["date"])
            if tuple(prices.columns) == tickers:
                validate_opens(prices, start, end)
                return prices, snapshot["fetched_at"], "bundled snapshot"

    bars = []
    with TemporaryDirectory(dir=ROOT, prefix=".yfinance-") as cache_dir:
        yf.set_tz_cache_location(cache_dir)
        for ticker in tickers:
            history = yf.Ticker(ticker).history(
                start=fetch_start.date().isoformat(), end=fetch_end.date().isoformat(),
                auto_adjust=True, timeout=20,
            )
            if history.empty:
                raise RuntimeError(f"Yahoo Finance returned no daily bars for {ticker}.")
            opens = history["Open"].rename(ticker)
            opens.index = opens.index.tz_localize(None).normalize()
            bars.append(opens)
        fetched_at = datetime.now(ZoneInfo("Europe/Oslo")).isoformat(timespec="seconds")

    prices = pd.concat(bars, axis=1).sort_index()
    validate_opens(prices, start, end)
    return prices, fetched_at, "fresh download"


def validate_opens(prices, start, end):
    if (
        prices.empty or prices.index.has_duplicates or not prices.index.is_monotonic_increasing
        or not np.isfinite(prices).to_numpy().all() or (prices <= 0).to_numpy().any()
        or prices.index[0] >= start or prices.index[-1] <= end
    ):
        raise ValueError("Need complete, aligned, positive adjusted opens and both boundary sessions.")


def simulate(equity_weight, returns, cost_rate):
    targets = pd.DataFrame(
        {returns.columns[0]: equity_weight, returns.columns[1]: 1.0 - equity_weight},
        index=returns.index,
    )
    drifted = targets.shift(1) * (1.0 + returns.shift(1))
    pretrade = drifted.div(drifted.sum(axis=1), axis=0).fillna(0.0)

    # For two fully invested assets a rebalance has one buy and one sell.
    # Solve fee = rate * sum(abs(target * (1 - fee) - pretrade)), also from initial cash.
    change_before_fee = targets - pretrade
    signed_target = (np.sign(change_before_fee) * targets).sum(axis=1)
    fees = cost_rate * change_before_fee.abs().sum(axis=1) / (1.0 + cost_rate * signed_target)
    traded = (targets.mul(1.0 - fees, axis=0) - pretrade).abs()
    return pd.DataFrame({
        "return": (1.0 - fees) * (1.0 + (targets * returns).sum(axis=1)) - 1.0,
        "turnover": traded.sum(axis=1),
        "n_trades": (traded > 1e-10).sum(axis=1),
        "equity_weight": targets.iloc[:, 0],
    })


def annualized_sharpe(returns):
    std = returns.std(ddof=1)
    return returns.mean() / std * math.sqrt(252) if std > 0 else None


def metrics(portfolio, next_open, cash_returns):
    returns = portfolio["return"]
    nav = np.r_[1.0, (1.0 + returns).cumprod().to_numpy()]
    years = (next_open.loc[returns.index[-1]] - returns.index[0]).days / 365.2425
    std = returns.std(ddof=1)
    values = {
        "cagr": nav[-1] ** (1.0 / years) - 1.0,
        "ann_vol": std * math.sqrt(252),
        "sharpe": annualized_sharpe(returns),
        "sharpe_excess": annualized_sharpe(returns - cash_returns),
        "max_drawdown": (nav / np.maximum.accumulate(nav) - 1.0).min(),
        "total_return": nav[-1] - 1.0,
        "turnover": portfolio["turnover"].sum(),
        "avg_equity_weight": portfolio["equity_weight"].mean(),
    }
    finite = {key: float(value) if value is not None and math.isfinite(value) else None
              for key, value in values.items()}
    return {**finite, "n_trades": int(portfolio["n_trades"].sum())}


def fmt(value, spec):
    return "—" if value is None else format(value, spec)


def print_metrics(report):
    print(f"{'Portfolio':<22} {'Split':<5} {'CAGR':>9} {'Ann. vol':>9} "
          f"{'Excess SR':>9} {'Zero SR':>8} "
          f"{'Max DD':>9} {'Total':>10} {'Trades':>7} {'Turnover':>9} {'Equity':>8}")
    for name, splits in report.items():
        for split, row in splits.items():
            print(f"{name:<22} {split.upper():<5} {fmt(row['cagr'], '.2%'):>9} "
                  f"{fmt(row['ann_vol'], '.2%'):>9} {fmt(row['sharpe_excess'], '.2f'):>9} "
                  f"{fmt(row['sharpe'], '.2f'):>8} "
                  f"{fmt(row['max_drawdown'], '.2%'):>9} {fmt(row['total_return'], '.2%'):>10} "
                  f"{row['n_trades']:>7d} {fmt(row['turnover'], '.2f'):>8}x "
                  f"{fmt(row['avg_equity_weight'], '.2%'):>8}")


def print_diagnostics(periods, portfolios, equity_returns, post_midterm):
    start, end = periods["full"]
    annual = (1.0 + equity_returns).groupby(equity_returns.index.year).prod() - 1.0
    annual = annual.loc[
        start.year + int((start.month, start.day) != (1, 1)):
        end.year - int((end.month, end.day) != (12, 31))
    ]
    print("\nSPY adjusted first-open to first-open returns by year type, before costs")
    print(f"{'Year type':<23} {'Years':>6} {'Mean':>9} {'Median':>9}")
    for phase, label in enumerate(YEAR_TYPES):
        group = annual.loc[annual.index % 4 == phase]
        mean = group.mean() if len(group) else None
        median = group.median() if len(group) else None
        print(f"{label:<23} {len(group):>6d} {fmt(mean, '.2%'):>9} {fmt(median, '.2%'):>9}")

    print("\nObserved election cycles, net cumulative returns")
    print(f"{'Years':<11} {'Coverage':<8} {'Strategy':>10} {'Static mix':>12} {'SPY hold':>10}")
    complete_cycles = []
    for year in range(start.year - start.year % 4, end.year + 1, 4):
        first, last = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 3, 12, 31)
        complete = start <= first and last <= end
        log_returns = [np.log1p(p.loc[first:last, "return"]).sum() for p in portfolios]
        strategy, static, buy_hold = np.expm1(log_returns)
        print(f"{year}-{year + 3}   {'Full' if complete else 'Partial':<8} "
              f"{strategy:>10.2%} {static:>12.2%} {buy_hold:>10.2%}")
        if complete:
            complete_cycles.append((first, last))

    print("\nCycle coverage (descriptive; too few complete cycles for reliable inference)")
    for split, (first, last) in periods.items():
        count = sum(first <= a and b <= last for a, b in complete_cycles)
        holding_years = [str(year) for year in equity_returns.loc[first:last].index.year.unique()
                         if year % 4 == post_midterm]
        print(f"{split.upper():<5} {count} complete cycles | SPY holding years "
              f"{', '.join(holding_years) or 'none'}")


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not (is_start <= is_end and is_end + pd.Timedelta(days=1) == oos_start <= oos_end):
        raise ValueError("Use ordered, contiguous IS and OOS date ranges.")
    periods = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    tickers = (config["universe"]["equity"], config["universe"]["cash_proxy"])
    fee_bps = config["transaction_cost_bps"]
    slippage_bps = config["slippage_bps"]
    cost = (fee_bps + slippage_bps) / 10000.0
    equity_weight = config["static_equity_weight"]
    post_midterm = config["post_midterm_year_mod_4"]
    if not (fee_bps >= 0 and slippage_bps >= 0 and cost < 1):
        raise ValueError("Use nonnegative transaction costs and slippage with total below 100%.")
    if not (0 <= equity_weight <= 1 and post_midterm in range(4) and tickers[0] != tickers[1]):
        raise ValueError("Use a valid equity weight and election-cycle year, and distinct tickers.")

    opens, fetched_at, price_source = load_opens(tickers, is_start, oos_end, config["price_snapshot"])
    next_open = opens.index.to_series().shift(-1)
    returns = (opens.shift(-1) / opens - 1.0).loc[is_start:oos_end]
    if any(returns.loc[first:last].empty for first, last in periods.values()):
        raise ValueError("Each split must contain at least one trading session.")

    # Today's close signal uses only the known calendar of tomorrow's session.
    # Shifting the signal executes it at that next session's open.
    signal_at_close = next_open.dt.year.mod(4).eq(post_midterm).astype(float)
    equity_at_open = signal_at_close.shift(1).loc[returns.index]
    portfolios = {
        "Midterm cycle": simulate(equity_at_open, returns, cost),
        f"Static {equity_weight:.0%} {tickers[0]}": simulate(equity_weight, returns, cost),
        f"{tickers[0]} buy & hold": simulate(1.0, returns, cost),
    }
    cash_returns = returns[tickers[1]]
    report = {
        name: {split: metrics(portfolio.loc[first:last], next_open, cash_returns.loc[first:last])
               for split, (first, last) in periods.items()}
        for name, portfolio in portfolios.items()
    }
    actual_start, last_session = returns.index[[0, -1]]
    data_range = f"{actual_start.date()} to {last_session.date()}"
    print(f"{config['strategy']} | holding sessions {data_range} | "
          f"{fee_bps:g} bps fees + {slippage_bps:g} bp slippage per leg")
    print(f"Final valuation open {next_open.loc[last_session].date()}")
    print(f"DATA_SOURCE: Yahoo Finance via yfinance ({price_source})")
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, split- and distribution-adjusted opens")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"Excess SR uses daily net portfolio returns minus {tickers[1]} gross open-to-open returns.")
    print_metrics(report)
    print_diagnostics(periods, list(portfolios.values()), returns[tickers[0]], post_midterm)

    strategy, static, _ = report.values()
    output = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **strategy,
        "benchmark": {
            "ticker": tickers[0],
            "equity_weight": equity_weight,
            "cash_proxy": tickers[1],
            **static["full"],
            **static,
        },
    }
    print("RESULTS_JSON: " + json.dumps(output, allow_nan=False))


if __name__ == "__main__":
    main()
