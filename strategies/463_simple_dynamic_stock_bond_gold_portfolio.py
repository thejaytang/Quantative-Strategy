"""463_simple_dynamic_stock_bond_gold_portfolio
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/463_simple_dynamic_stock_bond_gold_portfolio.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit e0683ac343a175cb2458afc860f8d1a1066c8e3a on 2026-11-27 08:55 Europe/Oslo, exit 0, output sha256 37755cbcde4e958e0668d4c7aef66a38182ef541ead95671719bdd0fac3a7efb
Data source: Yahoo Finance 免费日线，经 yfinance 获取 SPY、AGG、GLD 和现金代理 BIL 的调整后收盘价。此份行情的下载时间记录为 2026-11-27T07:04:34+00:00。（config.yaml:3-8，strategy.py:38-61、151-160，README.md:15，logs/run_20261127_080647.log:1）
Data fetched: 2026-11-27T07:55:47+00:00
Sample period: 实际回测交易日为 2007-07-02 至 2025-12-31（含两端）。行情从 2007-06-01 起请求，结束日 2026-01-01 不含在内。回测配置起始日为 2007-07-01，此前行情用于波动率预热。（config.yaml:10-14，strategy.py:182-190、199-208，README.md:16，backtest_results.md:24，logs/run_20261127_080647.log:10）   data_range (from output): {'start': '2007-07-02', 'end': '2025-12-31'}
IS/OOS split: 以 2018-01-01 为界。IS 为 2007-07-02 至 2017-12-29，OOS 为 2018-01-02 至 2025-12-31。两段从同一次连续模拟中切出，跨界时持仓与净值延续。README 记载参数取自论文，OOS 区间与论文样本重叠。（config.yaml:10-14，strategy.py:139-147、199-203，README.md:16、19）
Costs and slippage: 每笔 ETF 买入或卖出按成交金额扣除 2.5 bps 交易成本和 1.0 bps 固定滑点，合计 3.5 bps。滑点作为固定费率从净值扣减，模拟按扣费后的净值确定目标成交额。策略与基准均对首日建仓和后续交易收费，适用于 BIL 的买卖。（config.yaml:21-23，strategy.py:80-94、97-115、176-180、191-201，README.md:17-18）
Signal lag: 月末最后一个交易日收盘 t 形成信号，默认在下一交易日 t+1 收盘成交。模拟先用原持仓计入 t 至 t+1 的收盘收益，再扣除调仓成本并更换权重。首日从美元现金建仓，动态策略使用前一交易日的信号。固定权重基准首日建仓，之后在年末收盘后的下一交易日收盘调仓。（config.yaml:16-19，strategy.py:73-77、97-115、182-201，README.md:17）
Price adjustment: yfinance auto_adjust total-return
Survivorship bias: README 记载此固定标的组为论文事后指定且仍存续的美国上市基金。样本使用各基金上市后的自身历史报价，未纳入已清算基金或其他代理标的，存在基金选择与样本依赖。（config.yaml:3-8，strategy.py:151-160、182-183，README.md:11、24）
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.05858755500376622, ann_vol=0.06928342093446645, sharpe=0.8565273102358415, sharpe_excess=0.7889101121748024, max_drawdown=-0.15771081529713005, total_return=0.8177456334169935, turnover=1.585841227967942, n_trades=464
    oos: cagr=0.09232732321716974, ann_vol=0.07811833784805332, sharpe=1.1697302620666046, sharpe_excess=0.859304455438021, max_drawdown=-0.12915806199983615, total_return=1.0233052519247727, turnover=1.5299978935500322, n_trades=360
    full: cagr=0.07303051945707328, ann_vol=0.07322894231592593, sharpe=0.9993212410854401, sharpe_excess=0.820082529308135, max_drawdown=-0.15771081529713005, total_return=2.677854286755904, turnover=1.5617216090859798, n_trades=824
    benchmark  full: cagr=0.08859220960517811, ann_vol=0.10451542438435779, sharpe=0.8645879165868351, sharpe_excess=0.73787620810064, max_drawdown=-0.2728431625363106, total_return=3.798758161999647, turnover=0.06640241041690788, n_trades=57
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 463_simple_dynamic_stock_bond_gold_portfolio\n\nuniverse:\n  risky_weights:\n    SPY: 0.50\n    AGG: 0.30\n    GLD: 0.20\n  cash_proxy: BIL\n\ndates:\n  data_start: 2007-06-01\n  start: 2007-07-01\n  oos_start: 2018-01-01\n  end_exclusive: 2026-01-01\n\nsignal:\n  lookback_sessions: 11\n  target_annual_vol: 0.07\n  execution_lag_sessions: 1\n\nexecution:\n  cost_bps: 2.5\n  slippage_bps: 1.0\n'
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
"""Research backtest for the paper's 50/30/20 volatility-control portfolio."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
TRADE_TOLERANCE = 1e-10


def validate_prices(prices, symbols, start, end):
    prices = prices.reindex(columns=symbols).sort_index()
    prices.index = prices.index.as_unit("s")
    prices.columns.name = None
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("Missing or duplicate quote dates.")
    starts_in_range = start <= prices.index[0] < start + pd.Timedelta(days=7)
    ends_in_range = end - pd.Timedelta(days=7) <= prices.index[-1] < end
    if not (starts_in_range and ends_in_range):
        raise ValueError("Yahoo Finance quotes do not cover the requested date range.")
    invalid = ~np.isfinite(prices) | (prices <= 0)
    if invalid.any().any():
        bad_dates = prices.index[invalid.any(axis=1)].strftime("%Y-%m-%d")
        raise ValueError(f"Missing or invalid closes on {', '.join(bad_dates[:5])}.")
    return prices


def load_prices(symbols, start, end):
    CACHE.mkdir(exist_ok=True)
    request = json.dumps(["yfinance_auto_adjust_close", symbols, str(start), str(end)])
    key = hashlib.sha256(request.encode()).hexdigest()[:16]
    path = CACHE / f"adjusted_close_{key}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as cached:
            fetched_at = cached.readline().strip().removeprefix("# fetched_at=")
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(cached, index_col="Date", parse_dates=["Date"])
        return validate_prices(prices, symbols, start, end), fetched_at

    with TemporaryDirectory(dir=CACHE, prefix="yfinance_") as quote_cache:
        yf.set_tz_cache_location(quote_cache)
        quotes = yf.download(
            symbols, start=start.date().isoformat(), end=end.date().isoformat(),
            interval="1d", auto_adjust=True, progress=False, timeout=15,
        )
    if quotes.empty or "Close" not in quotes.columns.get_level_values(0):
        raise ValueError("Yahoo Finance returned no adjusted close data.")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prices = quotes["Close"]
    prices.index = prices.index.tz_localize(None).normalize().rename("Date")
    prices = validate_prices(prices, symbols, start, end)

    # Keep prices and their fetch timestamp together, including on concurrent runs.
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE,
                            delete_on_close=False) as pending:
        pending.write(f"# fetched_at={fetched_at}\n")
        prices.to_csv(pending)
        pending.close()
        Path(pending.name).replace(path)
    return prices, fetched_at


def fill_sessions(dates, frequency, lag, first):
    # At the first session of a new period, the previous close is the period end.
    periods = dates.to_period(frequency)
    fills = np.flatnonzero(periods[1:] != periods[:-1]) + lag
    return np.r_[first, fills[(fills > first) & (fills < len(dates))]]


def execution(drifted, target, fee):
    # Post-trade NAV = pre-trade NAV minus fees on all ETF buy and sell notionals.
    # Solve in pre-trade NAV units so that the final weights sum to one after fees.
    if fee == 0:
        return 1.0, np.abs(target - drifted)
    lower, upper = 0.0, 1.0
    for _ in range(48):
        nav = (lower + upper) / 2
        traded = np.abs(nav * target - drifted).sum()
        if nav + fee * traded > 1:
            upper = nav
        else:
            lower = nav
    nav = (lower + upper) / 2
    return nav, np.abs(nav * target - drifted)


def simulate(returns, targets, fee):
    # Start in uninvested USD. Subsequent closes mark existing holdings first.
    weights = np.zeros(returns.shape[1])
    rows = []
    for day, asset_returns in zip(returns.index, returns.to_numpy(), strict=True):
        marked = weights * (1 + asset_returns)
        gross = 1 - weights.sum() + marked.sum()
        target = targets.get(day)
        turnover = 0.0
        trades = 0
        if target is None:
            weights = marked / gross
        else:
            nav_ratio, traded = execution(marked / gross, target, fee)
            turnover = traded.sum() / 2
            trades = int(np.count_nonzero(traded > TRADE_TOLERANCE))
            gross *= nav_ratio
            weights = target
        rows.append((gross - 1, turnover, trades))
    return pd.DataFrame(rows, index=returns.index, columns=["return", "turnover", "trades"])


def metrics(portfolio, cash_returns):
    daily = portfolio["return"]
    excess = daily - cash_returns.loc[daily.index]
    annualize = np.sqrt(SESSIONS_PER_YEAR)
    wealth = np.r_[1.0, (1 + daily).cumprod().to_numpy()]
    volatility = daily.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    return {
        "cagr": float(wealth[-1] ** (SESSIONS_PER_YEAR / len(daily)) - 1),
        "ann_vol": float(volatility * annualize),
        "sharpe": float(annualize * daily.mean() / volatility) if volatility > 0 else None,
        "sharpe_excess": (float(annualize * excess.mean() / excess_volatility)
                          if excess_volatility > 0 else None),
        "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
        "total_return": float(wealth[-1] - 1),
        "turnover": float(portfolio["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(portfolio["trades"].sum()),
    }


def summarize(portfolio, cash_returns, split):
    before_split = portfolio.index < split
    if min(before_split.sum(), (~before_split).sum()) < 2:
        raise ValueError("IS and OOS must each contain at least two portfolio dates.")
    return {
        "is": metrics(portfolio.loc[before_split], cash_returns),
        "oos": metrics(portfolio.loc[~before_split], cash_returns),
        "full": metrics(portfolio, cash_returns),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    risky = pd.Series(universe["risky_weights"], dtype=float)
    cash = universe["cash_proxy"]
    if cash in risky or not np.isfinite(risky).all() or (risky <= 0).any() or not np.isclose(
        risky.sum(), 1.0, rtol=0, atol=1e-12
    ):
        raise ValueError("Risky weights must be positive, sum to one and exclude the cash proxy.")
    risky /= risky.sum()
    symbols = [*risky.index, cash]

    dates = {key: pd.Timestamp(value) for key, value in config["dates"].items()}
    data_start, start, split, end = (dates[key] for key in (
        "data_start", "start", "oos_start", "end_exclusive",
    ))
    if not data_start < start < split < end:
        raise ValueError("Expected data_start < start < oos_start < end_exclusive.")
    signal = config["signal"]
    lag = signal["execution_lag_sessions"]
    lookback = signal["lookback_sessions"]
    target_vol = signal["target_annual_vol"]
    if type(lag) is not int or lag < 1 or type(lookback) is not int or lookback < 2:
        raise ValueError("Execution lag must be at least one session and lookback at least two.")
    if not np.isfinite(target_vol) or target_vol <= 0:
        raise ValueError("Target volatility must be finite and positive.")
    cost_bps = config["execution"]["cost_bps"]
    slippage_bps = config["execution"]["slippage_bps"]
    if not (cost_bps >= 0 and slippage_bps >= 0 and cost_bps + slippage_bps < 10_000):
        raise ValueError("Costs and slippage must be nonnegative with a total below 10,000 bps.")
    fee = (cost_bps + slippage_bps) / 10_000

    prices, fetched_at = load_prices(symbols, data_start, end)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    first = returns.index.searchsorted(start)
    basket_return = returns[risky.index] @ risky
    vol = basket_return.rolling(lookback).std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)
    scale = (target_vol / vol).clip(upper=1).shift(lag)
    monthly = fill_sessions(returns.index, "M", lag, first)
    if scale.iloc[monthly].isna().any():
        raise ValueError("Insufficient price history before the first lagged rebalance.")
    strategy_targets = {
        day: np.r_[risky.to_numpy() * weight, 1 - weight]
        for day, weight in scale.iloc[monthly].items()
    }

    annual = fill_sessions(returns.index, "Y", lag, first)
    fixed = np.r_[risky.to_numpy(), 0.0]
    benchmark_targets = dict.fromkeys(returns.index[annual], fixed)
    evaluation = returns.iloc[first:]
    portfolio = simulate(evaluation, strategy_targets, fee)
    benchmark = simulate(evaluation, benchmark_targets, fee)
    portfolio_metrics = summarize(portfolio, evaluation[cash], split)
    benchmark_metrics = summarize(benchmark, evaluation[cash], split)

    results = {
        "strategy": config["strategy"],
        "data_range": {"start": evaluation.index[0].date().isoformat(),
                       "end": evaluation.index[-1].date().isoformat()},
        **portfolio_metrics,
        "benchmark": {"name": "Fixed-weight annual rebalance", **benchmark_metrics},
    }
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return")
    print("PARAM_COMBOS_TRIED: 1")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(portfolio_metrics[period], allow_nan=False)}")
        print(f"BENCHMARK {period.upper()}: {json.dumps(benchmark_metrics[period], allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
