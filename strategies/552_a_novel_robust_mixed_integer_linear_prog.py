"""552_a_novel_robust_mixed_integer_linear_prog
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/552_a_novel_robust_mixed_integer_linear_prog.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 69a306c0280e31031e1c1674f053aa22730544ac on 2026-11-27 09:47 Europe/Oslo, exit 0, output sha256 39ae3de8e71bddd78a11b4eeda1c2a5d32696a1c90ec659d8810076314a92dca
Data source: 通过 yfinance 获取 Yahoo Finance 免费日线，字段为 Adj Close、Close 和 Volume。数据覆盖固定名单内的 76 只美国大盘股、跟踪基准 ETF SPY，以及计算每日超额收益所用的短期国债 ETF BIL。行情优先从 data_cache/yahoo_daily_<digest>.csv.gz 读取。该次运行记录的数据抓取时间为 2026-11-27T08:22:14+00:00。（config.yaml:7-85；strategy.py:18-36、46-51、273-276、293-296；README.md:19、28；logs/run_20261127_092612.log:1）
Data fetched: 2026-11-27T08:48:01+00:00
Sample period: 配置从 2017-01-01 开始请求行情用于预热，默认信号回看 252 个交易日。实际回测为 2018-07-02 至 2025-12-31，含首尾交易日。（config.yaml:2-5、87；strategy.py:24-26、46-50、110-117、282-287；logs/run_20261127_092612.log:3、22）   data_range (from output): {'start': '2018-07-02', 'end': '2025-12-31'}
IS/OOS split: 样本内为 2018-07-02 至 2022-06-30，样本外为 2022-07-01 至 2025-12-31。替代参数只在样本内运行。默认参数在全期连续模拟，分界处延续原组合，并按既定日程用滞后的历史日线更新权重。（config.yaml:2-5；strategy.py:143-157、287-289、301-322；README.md:21-22、27；logs/run_20261127_092612.log:3-4）
Costs and slippage: 默认买入和卖出各按实际成交额扣 5 bps 交易成本和 5 bps 滑点，单边合计 10 bps。滑点作为固定费率建模，随交易成本从调仓日组合净值扣除，费率乘以买卖成交额绝对值之和。SPY 买入持有基准的首次建仓用相同费率。样本内另测成本和滑点各 10 bps 的联合情景。税费和市场冲击未单独计入。（config.yaml:96-97、103-104；strategy.py:175-194、238-239、315-316；README.md:23、29；logs/run_20261127_092612.log:10）
Signal lag: 以 SPY 日线建立交易日历。默认在回测首日和每年 7 月首个交易日收盘换权重。若信号使用截至 t 日收盘的数据，则在下一交易日 t+1 的收盘模拟成交。旧持仓承担截至执行日收盘的收益，新持仓从该收盘之后开始承担收益。（config.yaml:94-95；strategy.py:143-157、161-194、282-287；README.md:22）
Price adjustment: yfinance Adj Close, dividend- and split-adjusted total-return proxy
Survivorship bias: 候选池在研究时点选定，为仍有完整公开行情的固定大盘股名单，存在幸存者偏差及候选范围的事后选择。逐期成交额筛选局限于该名单。未重建历史指数成分、权重、成分变更或退市股票收益。（config.yaml:9-85；strategy.py:110-124、273-276；README.md:15、28）
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.12281623793818297, ann_vol=0.22676107012579438, sharpe=0.6248847481867573, sharpe_excess=0.5871159075804567, max_drawdown=-0.32648016841394234, total_return=0.5886751453453372, turnover=1.3872351321699732, n_trades=64
    oos: cagr=0.22364479845787488, ann_vol=0.16950718800292952, sharpe=1.2756109950371826, sharpe_excess=1.0172067756834668, max_drawdown=-0.1745657821418527, total_return=1.0218562420710984, turnover=1.6310788002633327, n_trades=67
    full: cagr=0.16873135286990282, ann_vol=0.20206723335804852, sharpe=0.8731421056414895, sharpe_excess=0.7496604642348567, max_drawdown=-0.32648016841394234, total_return=2.212072759239675, turnover=1.5008823136408442, n_trades=131
    benchmark: cagr=0.14832168586750694, ann_vol=0.1964057566667045, sharpe=0.8027830959090245, sharpe_excess=0.6757490427989172, max_drawdown=-0.337172391004658, total_return=1.8152914324640967, turnover=0.1334826361337496, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: '552_ a_novel_robust_mixed_integer_linear_prog'\ndata_start: '2017-01-01'\nbacktest_start: '2018-07-02'\nis_end: '2022-06-30'\nbacktest_end: '2025-12-31'\n\nbenchmark: SPY\nrisk_free_proxy: BIL\nuniverse:\n  - AAPL\n  - ABBV\n  - ABT\n  - ACN\n  - AMD\n  - AMGN\n  - AMZN\n  - AVGO\n  - BA\n  - BAC\n  - BKNG\n  - BRK-B\n  - C\n  - CAT\n  - CMCSA\n  - COP\n  - COST\n  - CRM\n  - CSCO\n  - CVX\n  - DHR\n  - DIS\n  - DUK\n  - F\n  - GE\n  - GM\n  - GOOGL\n  - GS\n  - HD\n  - HON\n  - IBM\n  - INTC\n  - ISRG\n  - JNJ\n  - JPM\n  - KO\n  - LIN\n  - LLY\n  - LMT\n  - LOW\n  - MA\n  - MCD\n  - MDLZ\n  - META\n  - MMM\n  - MRK\n  - MS\n  - MSFT\n  - MU\n  - NEE\n  - NFLX\n  - NKE\n  - NVDA\n  - ORCL\n  - PEP\n  - PFE\n  - PG\n  - PLD\n  - PM\n  - QCOM\n  - RTX\n  - SBUX\n  - SCHW\n  - SO\n  - SPGI\n  - T\n  - TMO\n  - TXN\n  - UNH\n  - UNP\n  - UPS\n  - V\n  - VZ\n  - WFC\n  - WMT\n  - XOM\n\nlookback_sessions: 252\nliquidity_top_n: 50\ncorrelation_block_sessions: 50\ncorrelation_penalty: 0.75\nportfolio_size: 10\nmin_weight: 0.01\nmax_weight: 0.90\nrebalance_month: 7\nsignal_lag_sessions: 1\ncost_bps: 5\nslippage_bps: 5\nsessions_per_year: 252\n\nrobustness:\n  portfolio_sizes: [8, 12]\n  correlation_penalties: [0.50, 0.90]\n  higher_cost_bps: 10\n  higher_slippage_bps: 10\n  buy_and_hold: true\n"
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
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq, linprog
import scipy.sparse as sparse
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_FIELDS = ["Adj Close", "Close", "Volume"]
MAX_SECONDS = 14 * 60


def price_panel(settings, tickers):
    start = pd.Timestamp(settings["data_start"])
    end = pd.Timestamp(settings["backtest_end"])
    key = [1, str(start.date()), str(end.date()), tickers]
    digest = sha256(json.dumps(key).encode()).hexdigest()[:20]
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"yahoo_daily_{digest}.csv.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            fetched_at = stream.readline().strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(stream, header=[0, 1], index_col=0, parse_dates=True)
        return prices, fetched_at

    columns = pd.MultiIndex.from_product([PRICE_FIELDS, tickers])
    prices = pd.DataFrame(columns=columns, dtype=float)
    pending = tickers
    with TemporaryDirectory(dir=CACHE) as session_cache:
        yf.set_tz_cache_location(session_cache)
        for attempt in range(3):
            if attempt:
                time.sleep(2 * attempt)
            batch = yf.download(
                pending, start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d", auto_adjust=False, ignore_tz=True,
                threads=6, timeout=10, progress=False,
            )
            fetched = batch.reindex(columns=columns)
            prices = prices.combine_first(fetched)
            pending = [ticker for ticker in pending
                       if not prices.loc[:, (PRICE_FIELDS, ticker)].notna().any().all()]
            if not pending:
                break
    if pending:
        raise RuntimeError(f"Yahoo Finance download incomplete for {', '.join(pending)}.")

    prices = prices.sort_index().reindex(columns=columns)
    reference = prices["Adj Close"][[settings["benchmark"], settings["risk_free_proxy"]]].dropna()
    if (reference.empty or reference.index[0] > start + pd.Timedelta(days=7)
            or reference.index[-1] < end - pd.Timedelta(days=7)):
        raise RuntimeError("Yahoo Finance reference series do not cover the requested date range.")

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(dir=CACHE, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8") as stream:
            stream.write(fetched_at + "\n")
            prices.to_csv(stream, index_label="Date")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def representative_stocks(returns, settings):
    block_size = settings["correlation_block_sessions"]
    block_count = len(returns) // block_size
    blocks = returns[-block_count * block_size:].reshape(block_count, block_size, returns.shape[1])
    block_correlations = np.stack([np.corrcoef(block, rowvar=False) for block in blocks])
    dispersion = block_correlations.std(axis=0, ddof=1)
    similarity = np.corrcoef(returns, rowvar=False) - settings["correlation_penalty"] * dispersion
    np.fill_diagonal(similarity, 1.0)

    chosen = []
    coverage = np.full(returns.shape[1], -np.inf)
    for _ in range(settings["portfolio_size"]):
        candidate_coverage = np.maximum(coverage[:, None], similarity)
        scores = candidate_coverage.mean(axis=0)
        scores[chosen] = -np.inf
        best = int(np.argmax(scores))
        chosen.append(best)
        coverage = candidate_coverage[:, best]
    return chosen


def tracking_constraints(returns, benchmark):
    sessions, count = returns.shape
    # Absolute deviations are u + d with Rw + u - d = SPY returns.
    identity = sparse.eye(sessions, format="csc")
    tracking = sparse.hstack([sparse.csc_matrix(returns), identity, -identity], format="csc")
    budget = sparse.csc_matrix(np.r_[np.ones(count), np.zeros(2 * sessions)][None, :])
    return sparse.vstack([tracking, budget], format="csc"), np.r_[benchmark, 1.0]


def target_weights(asset_returns, dollar_volume, benchmark_returns, settings):
    lookback = settings["lookback_sessions"]
    block_size = settings["correlation_block_sessions"]
    if len(asset_returns) < lookback:
        raise RuntimeError("More price history is required before the first signal.")
    window = asset_returns.iloc[-lookback:]
    benchmark = benchmark_returns.loc[window.index].to_numpy()
    liquidity = dollar_volume.loc[window.index]
    eligible = window.columns[window.notna().all() & liquidity.notna().all()]
    tail_length = (lookback // block_size) * block_size
    for block_start in range(lookback - tail_length, lookback, block_size):
        block = window[eligible].iloc[block_start:block_start + block_size]
        eligible = eligible[block.nunique().gt(1)]
    ranked = liquidity[eligible].median().sort_values(ascending=False, kind="stable")
    screened = sorted(ranked.index[:settings["liquidity_top_n"]])
    if len(screened) < settings["portfolio_size"] or not np.isfinite(benchmark).all():
        raise RuntimeError(f"Insufficient complete signal data through {window.index[-1].date()}.")

    daily = window[screened].to_numpy()
    chosen = representative_stocks(daily, settings)
    assets = [screened[i] for i in chosen]
    constraints, targets = tracking_constraints(daily[:, chosen] * 100.0, benchmark * 100.0)
    bounds = [(settings["min_weight"], settings["max_weight"])] * len(chosen)
    solution = linprog(
        np.r_[np.zeros(len(chosen)), np.ones(2 * lookback)], A_eq=constraints, b_eq=targets,
        bounds=bounds + [(0, None)] * (2 * lookback), method="highs",
        options={"time_limit": 10.0},
    )
    if not solution.success:
        raise RuntimeError(f"Weight fit failed for {window.index[-1].date()}: {solution.message}")
    return pd.Series(solution.x[:len(chosen)], index=assets).reindex(window.columns, fill_value=0.0)


def make_schedule(asset_returns, dollar_volume, benchmark_returns, dates, settings):
    annual_dates = dates[dates.month == settings["rebalance_month"]]
    annual_dates = annual_dates[~annual_dates.year.duplicated()]
    execution_dates = dates[:1]
    if not settings.get("buy_and_hold", False):
        execution_dates = execution_dates.union(annual_dates)
    schedule = {}
    for execution_date in execution_dates:
        execution_index = asset_returns.index.get_loc(execution_date)
        signal_index = execution_index - settings["signal_lag_sessions"]
        if signal_index < 0:
            raise RuntimeError("Missing sessions between the first signal and execution.")
        weights = target_weights(asset_returns.iloc[:signal_index + 1], dollar_volume,
                                 benchmark_returns, settings)
        schedule[execution_date] = weights.to_numpy()
    return schedule


def simulate(close, returns, schedule, fee_rate):
    weights = np.zeros(close.shape[1])
    daily_results = []
    for date, price, change in zip(close.index, close.to_numpy(), returns.to_numpy(), strict=True):
        invested = weights > 0
        if not np.isfinite(change[invested]).all():
            missing = close.columns[invested & ~np.isfinite(change)]
            raise RuntimeError(f"Missing session return on {date.date()} for {', '.join(missing)}.")
        drifted = weights.copy()
        drifted[invested] *= 1.0 + change[invested]
        market_change = drifted.sum() - weights.sum()
        trade_notional = 0.0
        trade_count = 0
        retained = 1.0
        if date in schedule:
            desired = schedule[date]
            if not np.isfinite(price[desired > 0]).all():
                raise RuntimeError(f"Missing execution close on {date.date()}.")
            before_trade = drifted / (1.0 + market_change)

            # Size the post-cost weights so the actual buy and sell amounts fund their fees.
            def cost_equation(fraction):
                traded = np.abs(fraction * desired - before_trade).sum()
                return fraction + fee_rate * traded - 1.0

            retained = brentq(cost_equation, 0.0, 1.0, xtol=1e-14)
            traded = np.abs(retained * desired - before_trade)
            trade_notional = float(traded.sum())
            trade_count = int((traded > 1e-10).sum())
            weights = desired
        else:
            weights = drifted / (1.0 + market_change)
        net_return = (1.0 + market_change) * retained - 1.0
        daily_results.append((net_return, trade_notional, trade_count))
    return pd.DataFrame(daily_results, index=close.index, columns=["return", "notional", "trades"])


def metrics(daily_results, risk_free_returns, annual_sessions):
    returns = daily_results["return"].to_numpy()
    excess = returns - risk_free_returns.loc[daily_results.index].to_numpy()
    cumulative = np.r_[1.0, np.cumprod(1.0 + returns)]
    years = len(returns) / annual_sessions
    annual_scale = np.sqrt(annual_sessions)
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    return {
        "cagr": float(cumulative[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(volatility * annual_scale),
        "sharpe": float(returns.mean() * annual_scale / volatility) if volatility > 0 else None,
        "sharpe_excess": (float(excess.mean() * annual_scale / excess_volatility)
                          if excess_volatility > 0 else None),
        "max_drawdown": float(np.min(cumulative / np.maximum.accumulate(cumulative) - 1.0)),
        "total_return": float(cumulative[-1] - 1.0),
        "turnover": float(daily_results["notional"].sum() / years),
        "n_trades": int(daily_results["trades"].sum()),
    }


def tracking_metrics(daily_results, benchmark_returns, annual_sessions):
    active = daily_results["return"] - benchmark_returns.loc[daily_results.index]
    return {
        "tracking_error_annualized": float(active.std(ddof=1) * np.sqrt(annual_sessions)),
        "tracking_mad_daily": float(active.abs().mean()),
    }


def robustness_cases(settings):
    config = settings["robustness"]
    cases = [{"portfolio_size": value} for value in config["portfolio_sizes"]]
    cases.extend({"correlation_penalty": value} for value in config["correlation_penalties"])
    cases.append({"cost_bps": config["higher_cost_bps"],
                  "slippage_bps": config["higher_slippage_bps"]})
    if config["buy_and_hold"]:
        cases.append({"buy_and_hold": True})
    return cases


def fee_rate(settings):
    return (settings["cost_bps"] + settings["slippage_bps"]) / 10_000.0


def validate_settings(settings):
    start, first, split, last = (pd.Timestamp(settings[key]) for key in
                                 ("data_start", "backtest_start", "is_end", "backtest_end"))
    if not start < first <= split < last:
        raise ValueError("Dates must place a warm-up before the two evaluation periods.")
    tickers = settings["universe"] + [settings["benchmark"], settings["risk_free_proxy"]]
    if len(tickers) != len(set(tickers)):
        raise ValueError("The universe and reference series must use distinct tickers.")
    count = settings["portfolio_size"]
    minimum = settings["min_weight"]
    maximum = settings["max_weight"]
    if (not 2 <= count <= settings["liquidity_top_n"] <= len(settings["universe"])
            or not 0 < minimum <= maximum <= 1
            or not count * minimum <= 1.0 <= count * maximum):
        raise ValueError("Infeasible portfolio size, liquidity screen or weight bounds.")
    block_size = settings["correlation_block_sessions"]
    if not 2 <= block_size <= settings["lookback_sessions"] // 2:
        raise ValueError("Correlation dispersion needs at least two blocks of two sessions or more.")
    if (settings["signal_lag_sessions"] < 1 or not 1 <= settings["rebalance_month"] <= 12
            or not np.isfinite(settings["correlation_penalty"]) or settings["correlation_penalty"] < 0
            or settings["sessions_per_year"] < 1
            or not 0 <= fee_rate(settings) < 1.0
            or min(settings["cost_bps"], settings["slippage_bps"]) < 0):
        raise ValueError("Invalid lag, rebalance month, correlation penalty, fee or annualization.")


def main():
    settings = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cases = robustness_cases(settings)
    for changes in [{}, *cases]:
        validate_settings(settings | changes)
    universe = sorted(settings["universe"])
    benchmark, risk_free = settings["benchmark"], settings["risk_free_proxy"]
    annual_sessions = settings["sessions_per_year"]
    prices, fetched_at = price_panel(settings, sorted(universe + [benchmark, risk_free]))
    close = prices["Adj Close"].where(np.isfinite(prices["Adj Close"]) & (prices["Adj Close"] > 0))
    reference_gap = close[benchmark].notna() != close[risk_free].notna()
    both_missing = close[universe].notna().any(axis=1) & close[benchmark].isna()
    if (reference_gap | both_missing).any():
        raise RuntimeError(f"{benchmark} or {risk_free} has a missing close on a market session.")
    calendar = close[benchmark].dropna().index
    close = close.loc[calendar]
    dollar_volume = prices.loc[calendar, "Close"][universe] * prices.loc[calendar, "Volume"][universe]
    dollar_volume = dollar_volume.where(np.isfinite(dollar_volume) & (dollar_volume > 0))
    returns = close.pct_change(fill_method=None)
    dates = calendar[(calendar >= settings["backtest_start"]) & (calendar <= settings["backtest_end"])]
    split = pd.Timestamp(settings["is_end"])
    is_dates, oos_dates = dates[dates <= split], dates[dates > split]
    if min(len(is_dates), len(oos_dates)) < 2:
        raise RuntimeError("At least two sessions are required in each evaluation period.")

    tracking_reference = returns.loc[dates, benchmark].copy()
    risk_free_returns = returns.loc[dates, risk_free].copy()
    tracking_reference.iloc[0] = risk_free_returns.iloc[0] = 0.0
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance Adj Close, dividend- and split-adjusted total-return proxy")
    print(f"IS: {is_dates[0].date()} to {is_dates[-1].date()} | "
          f"OOS: {oos_dates[0].date()} to {oos_dates[-1].date()} | Sharpe excess vs {risk_free}")

    print(f"ROBUSTNESS: IS only | tracking diagnostics vs gross {benchmark}")
    for changes in [{}, *cases]:
        params = settings | changes
        schedule = make_schedule(returns.loc[:split, universe], dollar_volume.loc[:split],
                                 returns.loc[:split, benchmark], is_dates, params)
        daily = simulate(close.loc[is_dates, universe], returns.loc[is_dates, universe],
                         schedule, fee_rate(params))
        stats = metrics(daily, risk_free_returns, annual_sessions)
        tracking = tracking_metrics(daily, tracking_reference, annual_sessions)
        print(f"  {json.dumps(changes) if changes else 'default'}: "
              f"{json.dumps(stats | tracking, allow_nan=False)}")

    schedule = make_schedule(returns[universe], dollar_volume, returns[benchmark], dates, settings)
    daily = simulate(close.loc[dates, universe], returns.loc[dates, universe], schedule, fee_rate(settings))
    benchmark_daily = simulate(close.loc[dates, [benchmark]], returns.loc[dates, [benchmark]],
                               {dates[0]: np.ones(1)}, fee_rate(settings))
    periods = {"is": is_dates, "oos": oos_dates, "full": dates}
    strategy_stats = {}
    benchmark_stats = {}
    for label, period in periods.items():
        strategy_stats[label] = metrics(daily.loc[period], risk_free_returns, annual_sessions)
        benchmark_stats[label] = metrics(benchmark_daily.loc[period], risk_free_returns, annual_sessions)
        tracking = tracking_metrics(daily.loc[period], tracking_reference, annual_sessions)
        print(f"STRATEGY_{label.upper()}: {json.dumps(strategy_stats[label], allow_nan=False)}")
        print(f"BENCHMARK_{label.upper()} ({benchmark}): "
              f"{json.dumps(benchmark_stats[label], allow_nan=False)}")
        print(f"TRACKING_{label.upper()} (vs gross {benchmark}): {json.dumps(tracking, allow_nan=False)}")

    data_range = {"start": dates[0].date().isoformat(), "end": dates[-1].date().isoformat()}
    result = {"strategy": settings["strategy"], "data_range": data_range, **strategy_stats,
              "benchmark": benchmark_stats["full"]}
    print(f"PARAM_COMBOS_TRIED: {1 + len(cases)}")
    print(f"RESULTS_JSON: {json.dumps(result, allow_nan=False)}")


def stop_on_timeout(_signum, _frame):
    raise TimeoutError(f"Backtest exceeded its {MAX_SECONDS}-second runtime limit.")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_on_timeout)
    signal.alarm(MAX_SECONDS)
    try:
        main()
    finally:
        signal.alarm(0)