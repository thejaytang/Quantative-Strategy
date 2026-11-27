"""451_a_declining_cvar_glidepath_framework_for
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/451_a_declining_cvar_glidepath_framework_for.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 6df04a4d89c55b7a1b162dbdcff0dafc1e2bd682 on 2026-11-27 08:11 Europe/Oslo, exit 0, output sha256 0c50b887873b6e37af92d826bb454af8e571e940cd3d36d5181dbb73bf114a91
Data source: Yahoo Finance 日频 ETF 收盘价，通过 yfinance 下载或读取本地 CSV 缓存。SPY 为股票代理，IEF 为国债代理，BIL 为国库券及无风险收益代理。
Data fetched: 2026-11-27T07:11:29+00:00
Sample period: 价格请求区间为 2007-06-01 至 2025-12-31，2013 年前的价格用于预热。回测配置区间为 2013-01-01 至 2025-12-31，日志中的实际交易日区间为 2013-01-02 至 2025-12-31。 data_range (from output): {'start': '2013-01-02', 'end': '2025-12-31'}
IS/OOS split: IS 为 2013-01-01 至 2019-12-31，OOS 为 2020-01-01 至 2025-12-31，均按实际交易日取样。策略和基准各自先生成连续账本，再按日期拆分报告。OOS 延续期末持仓与调仓日历，信号继续使用滚动 60 个月窗口。
Costs and slippage: 每笔 ETF 买卖按成交名义金额计提 2 bps 费用及 3 bps 固定滑点，合计 5 bps。调仓时将各 ETF 买卖金额的绝对值加总计费，并求解扣费后可分配的净值。策略与基准均使用相同费率，初始建仓同样计费。
Signal lag: 月末最后一个有价格的交易日 t 收盘后，用截至 t 的月度收益情景计算目标，在下一价格交易日收盘成交。成交日的收盘到收盘收益先计入原持仓，新权重从成交收盘至下一收盘的区间起计收益。
Price adjustment: yfinance auto_adjust total-return closes
Survivorship bias: 固定使用 SPY、IEF、BIL 三只存续且具有较长历史的 ETF 代理。历史上清盘或合并的基金未纳入，存在幸存者偏差风险。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.09565953208021205, ann_vol=0.07128784235203181, sharpe=1.3174059712193307, sharpe_excess=1.2317920621778127, max_drawdown=-0.10556921495201077, total_return=0.8941504164026068, turnover=2.4011662124426105, n_trades=178
    oos: cagr=0.07623563998202965, ann_vol=0.10482085168230691, sharpe=0.753529088585022, sharpe_excess=0.5039402989496111, max_drawdown=-0.191011467110058, total_return=0.5521645221503395, turnover=2.0997263999457694, n_trades=190
    full: cagr=0.0866587923905131, ann_vol=0.08833470431651917, sharpe=0.9852314012909932, sharpe_excess=0.8113979244131939, max_drawdown=-0.19101146711005845, total_return=1.9400330759564262, turnover=4.50089261238838, n_trades=368
    benchmark: cagr=0.09446957702504388, ann_vol=0.0988115687037949, sharpe=0.963165007579257, sharpe_excess=0.8077248356069113, max_drawdown=-0.21193611256356581, total_return=2.226373192628762, turnover=3.7517822292275445, n_trades=312
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: '451_ a_declining_cvar_glidepath_framework_for'\nuniverse:\n  equity: SPY\n  bond: IEF\n  bills: BIL\ndates:\n  data_start: '2007-06-01'\n  backtest_start: '2013-01-01'\n  is_end: '2019-12-31'\n  backtest_end: '2025-12-31'\nglidepath:\n  transition_date: '2018-01-01'\n  target_date: '2025-01-01'\n  initial_cvar_limit: 0.06\n  terminal_cvar_limit: 0.03\n  confidence: 0.90\n  lookback_months: 60\nbenchmark:\n  equity_weight: 0.60\nexecution:\n  lag_sessions: 1\n  cost_bps: 2.0\n  slippage_bps: 3.0\n"
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
"""Monthly portfolio CVaR glidepath using public ETF closing prices."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import json

import numpy as np
import pandas as pd
from scipy.optimize import brentq, linprog
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
TRADE_TOLERANCE = 1e-10


def load_prices(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"yf_adjusted_{'_'.join(tickers)}_{start.date()}_{end.date()}.csv"

    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as cached:
            fetched_at = cached.readline().strip().removeprefix("# fetched_at=")
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(cached, index_col="Date", parse_dates=["Date"])
    else:
        with TemporaryDirectory(dir=cache_dir, prefix="yfinance-") as session_cache:
            yf.set_tz_cache_location(session_cache)
            history = yf.download(
                list(tickers),
                start=start.date(),
                end=(end + pd.Timedelta(days=1)).date(),
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=15,
            )
        if history.empty or "Close" not in history:
            raise RuntimeError("Yahoo Finance returned no adjusted closes.")
        prices = history["Close"].reindex(columns=tickers)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    prices = prices.loc[:, list(tickers)].sort_index()
    if (
        prices.empty
        or not prices.index.is_unique
        or prices.index[0] < start
        or prices.index[-1] > end
        or prices.index[-1] < end - pd.Timedelta(days=4)
        or not np.isfinite(prices.to_numpy()).all()
        or (prices <= 0).any().any()
    ):
        raise RuntimeError("Adjusted ETF closes are missing, invalid, or incomplete.")

    if not cache_path.exists():
        with NamedTemporaryFile(mode="w", dir=cache_dir, encoding="utf-8", delete=False) as cache:
            temp_path = Path(cache.name)
            try:
                cache.write(f"# fetched_at={fetched_at}\n")
                prices.to_csv(cache, index_label="Date")
                cache.flush()
                temp_path.replace(cache_path)
            finally:
                temp_path.unlink(missing_ok=True)

    return prices, fetched_at


def cvar_limit(glidepath, signal_date):
    transition = pd.Period(glidepath["transition_date"], freq="M").ordinal
    target = pd.Period(glidepath["target_date"], freq="M").ordinal
    # A month-end signal sets the risk budget for the following holding month.
    holding_month = signal_date.to_period("M").ordinal + 1
    fraction = np.clip((holding_month - transition) / (target - transition), 0.0, 1.0)
    initial = glidepath["initial_cvar_limit"]
    return initial + fraction * (glidepath["terminal_cvar_limit"] - initial)


def maximum_sleeve_weight(sleeve_returns, bill_returns, limit, confidence):
    """Maximize sleeve weight subject to the expected shortfall of empirical losses."""
    n = len(sleeve_returns)
    # Variables are sleeve weight, the loss threshold, and nonnegative tail losses.
    constraints = np.column_stack((bill_returns - sleeve_returns, -np.ones(n), -np.eye(n)))
    tail_mean = np.r_[0.0, 1.0, np.full(n, 1.0 / (n * (1.0 - confidence)))]
    result = linprog(
        c=np.r_[-1.0, np.zeros(n + 1)],
        A_ub=np.vstack((constraints, tail_mean)),
        b_ub=np.r_[bill_returns, limit],
        bounds=[(0.0, 1.0), (None, None)] + [(0.0, None)] * n,
        method="highs",
    )
    if result.status == 2:
        return None
    if not result.success:
        raise RuntimeError(f"CVaR optimization failed: {result.message}")
    return float(np.clip(result.x[0], 0.0, 1.0))


def build_targets(prices, start, glidepath, lag, benchmark_weights):
    # Each change of calendar month identifies the previous exchange session.
    months = prices.index.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])
    monthly_returns = prices.iloc[month_ends].pct_change(fill_method=None)
    lookback = glidepath["lookback_months"]
    strategy_targets = {}
    benchmark_targets = {}
    for signal_number, signal_pos in enumerate(month_ends):
        fill_pos = signal_pos + lag
        if fill_pos >= len(prices) or prices.index[fill_pos] < start:
            continue
        if signal_number < lookback:
            raise ValueError("The data start does not allow the full warm-up before the first fill.")

        window = monthly_returns.iloc[signal_number - lookback + 1 : signal_number + 1]
        if not np.isfinite(window.to_numpy()).all():
            raise RuntimeError("A monthly return is missing from the CVaR window.")
        exposure = maximum_sleeve_weight(
            window.to_numpy() @ benchmark_weights,
            window.iloc[:, 2].to_numpy(),
            cvar_limit(glidepath, prices.index[signal_pos]),
            glidepath["confidence"],
        )
        fill_date = prices.index[fill_pos]
        if exposure is None:
            strategy_targets[fill_date] = np.zeros(3)
            print(f"CVAR_FALLBACK_CASH: {fill_date.date()}")
        else:
            target = exposure * benchmark_weights
            target[2] = 1.0 - exposure
            strategy_targets[fill_date] = target
        benchmark_targets[fill_date] = benchmark_weights

    if not strategy_targets:
        raise ValueError("No scheduled fills fall inside the backtest.")
    return strategy_targets, benchmark_targets


def backtest(daily_returns, targets, fee_rate):
    weights = np.zeros(daily_returns.shape[1])
    rows = []
    for date, returns in zip(daily_returns.index, daily_returns.to_numpy()):
        gross_growth = 1.0 + weights @ returns
        weights = weights * (1.0 + returns) / gross_growth
        retained_nav = 1.0
        turnover = 0.0
        trades = 0

        if date in targets:
            target = targets[date]

            def funding_gap(nav):
                return (nav - 1.0) + fee_rate * np.abs(nav * target - weights).sum()

            if fee_rate:
                retained_nav = brentq(funding_gap, 0.0, 1.0)
            traded = np.abs(retained_nav * target - weights)
            turnover = traded.sum()
            trades = int(np.count_nonzero(traded > TRADE_TOLERANCE))
            weights = target

        rows.append((gross_growth * retained_nav - 1.0, turnover, trades))
    return pd.DataFrame(rows, index=daily_returns.index, columns=["return", "turnover", "n_trades"])


def metrics(daily_results, bill_returns):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise ValueError("Each reporting period needs at least two daily returns.")
    nav = (1.0 + returns).cumprod().to_numpy()
    peak = np.maximum.accumulate(np.r_[1.0, nav])[1:]
    vol = returns.std() * np.sqrt(SESSIONS_PER_YEAR)
    excess = returns - bill_returns.reindex(returns.index)
    excess_vol = excess.std() * np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(nav[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1.0),
        "ann_vol": float(vol),
        "sharpe": float(returns.mean() * SESSIONS_PER_YEAR / vol) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() * SESSIONS_PER_YEAR / excess_vol) if excess_vol > 0 else None,
        "max_drawdown": float((nav / peak - 1.0).min()),
        "total_return": float(nav[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    tickers = tuple(config["universe"][role] for role in ("equity", "bond", "bills"))
    if len(set(tickers)) != len(tickers) or any(not ticker.isalpha() for ticker in tickers):
        raise ValueError("Supply three distinct alphabetic ETF tickers.")
    data_start, start, is_end, end = (
        pd.Timestamp(config["dates"][key])
        for key in ("data_start", "backtest_start", "is_end", "backtest_end")
    )
    if not data_start < start < is_end < end < pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None):
        raise ValueError("The dates must be ordered and the data interval must be historical.")

    glidepath = config["glidepath"]
    if pd.Period(glidepath["transition_date"], freq="M") >= pd.Period(glidepath["target_date"], freq="M"):
        raise ValueError("The glidepath transition month must precede the target month.")
    if not 0 <= glidepath["terminal_cvar_limit"] < glidepath["initial_cvar_limit"] < 1:
        raise ValueError("CVaR limits must decline from the initial loss cap to the terminal cap.")
    if not 0 < glidepath["confidence"] < 1 or glidepath["lookback_months"] < 1:
        raise ValueError("Supply a valid CVaR confidence and a positive lookback.")

    execution = config["execution"]
    lag = execution["lag_sessions"]
    cost, slippage = execution["cost_bps"], execution["slippage_bps"]
    fee_rate = (cost + slippage) / 10_000.0
    if not isinstance(lag, int) or lag < 1 or min(cost, slippage) < 0 or not 0 <= fee_rate < 1:
        raise ValueError("Supply a positive session lag and valid cost and slippage rates.")
    equity_weight = config["benchmark"]["equity_weight"]
    if not 0 <= equity_weight <= 1:
        raise ValueError("The benchmark equity weight must be between zero and one.")
    benchmark_weights = np.array([equity_weight, 1.0 - equity_weight, 0.0])

    prices, fetched_at = load_prices(tickers, data_start, end)
    daily_returns = prices.pct_change(fill_method=None).loc[start:end]
    bill_returns = daily_returns.iloc[:, 2]
    strategy_targets, benchmark_targets = build_targets(prices, start, glidepath, lag, benchmark_weights)
    strategy_daily_results = backtest(daily_returns, strategy_targets, fee_rate)
    benchmark_daily_results = backtest(daily_returns, benchmark_targets, fee_rate)
    sessions = {
        "is": daily_returns.index <= is_end,
        "oos": daily_returns.index > is_end,
        "full": slice(None),
    }
    strategy_metrics = {name: metrics(strategy_daily_results.loc[mask], bill_returns) for name, mask in sessions.items()}
    benchmark_metrics = {name: metrics(benchmark_daily_results.loc[mask], bill_returns) for name, mask in sessions.items()}
    results = {
        "strategy": config["strategy"],
        "data_range": {
            "start": daily_returns.index[0].date().isoformat(),
            "end": daily_returns.index[-1].date().isoformat(),
        },
        **strategy_metrics,
        "benchmark": benchmark_metrics["full"],
    }
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return closes")
    print("PARAM_COMBOS_TRIED: 1")
    for name in sessions:
        print(f"{name.upper()}: {json.dumps(strategy_metrics[name], allow_nan=False)}")
        print(f"BENCHMARK {name.upper()}: {json.dumps(benchmark_metrics[name], allow_nan=False)}")
    print(f"RESULTS_JSON: {json.dumps(results, allow_nan=False)}")


if __name__ == "__main__":
    main()