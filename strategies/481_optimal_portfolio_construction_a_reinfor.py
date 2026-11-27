"""481_optimal_portfolio_construction_a_reinfor
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/481_optimal_portfolio_construction_a_reinfor.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit d2f4ee010416fa7342bf8a9f3404c2adc27e3f7d on 2026-11-27 09:18 Europe/Oslo, exit 0, output sha256 6972caad5beb669e6ef9953d09e3a7c570f5c648d320a1904401277c356edd61
Data source: 通过 yfinance 获取 Yahoo Finance 免费日线。组合使用 11 只美国 SPDR 行业 ETF（XLB、XLY、XLF、XLRE、XLC、XLE、XLI、XLK、XLP、XLV、XLU）。BIL 的日收益仅用于样本内参数选择及超额夏普计算。
Data fetched: 2026-11-27T08:18:12.417975+00:00
Sample period: 配置的数据请求区间为 2018-06-19 至 2025-08-13（含结束日），2019-08-01 前的数据用于预热。日志记录的实际回测区间为 2019-08-01 至 2025-08-13。 data_range (from output): 2019-08-01 to 2025-08-13
IS/OOS split: 按交易日顺序划分。样本内为 2019-08-01 至 2021-12-31，样本外为 2022-01-01 至 2025-08-13（区间内实际交易日）。仅按样本内扣费后的 BIL 超额夏普选择回看窗口，并列时选短窗口。本次选中 252 个交易日。样本外沿用该窗口滚动估计协方差，并承接样本内期末仓位。
Costs and slippage: 每笔买卖按实际成交名义金额计入 5 bps 费用和 2 bps 滑点，共 7 bps，首次建仓也扣除。滑点建模为固定比例扣款。回测在成交收盘从组合价值扣除这笔成本，再按扣费后净值配置目标权重。等权基准使用相同费率。
Signal lag: 以月末最后一条日线的收盘 t 为信号时点，使用截至 t 收盘的日收益。配置为在下一条日线（下一交易日）收盘成交。旧权重承担 t 至成交收盘的收益，新权重从成交收盘之后生效。
Price adjustment: yfinance auto_adjust total-return close (splits and distributions)
Survivorship bias: 行业 ETF 名单固定为当前留存的产品，未纳入退市或关闭基金，也未按历史时点重建行业分类，存在存续偏差。
Parameter combinations tried: 3
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.1977457824946769, ann_vol=0.23140466015374145, sharpe=0.8965517923892831, sharpe_excess=0.8786491222654823, max_drawdown=-0.34702573854434904, total_return=0.5488279593021037, turnover=2.0268241752917415, n_trades=319
    oos: cagr=0.06728406703342515, ann_vol=0.1498080975295015, sharpe=0.509567825513133, sharpe_excess=0.25270804116908646, max_drawdown=-0.1792413979211791, total_return=0.26378567164672706, turnover=1.2822491904440128, n_trades=484
    full: cagr=0.11802764213058414, ann_vol=0.18697650210758346, sharpe=0.6907391137683717, sharpe_excess=0.5587829182778951, max_drawdown=-0.34702573854434904, total_return=0.9573865828118413, turnover=3.3090733657357547, n_trades=803
    benchmark: cagr=0.12761358799829914, ann_vol=0.1977643902127808, sharpe=0.7068310637904088, sharpe_excess=0.582070882112418, max_drawdown=-0.36176255218303355, total_return=1.060614398218536, turnover=2.9880752659488206, n_trades=803
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 481_optimal_portfolio_construction_a_reinfor\n\nuniverse:\n  cyclical: [XLB, XLY, XLF, XLRE]\n  sensitive: [XLC, XLE, XLI, XLK]\n  defensive: [XLP, XLV, XLU]\nrisk_free_ticker: BIL\n\ndates:\n  data_start: '2018-06-19'\n  sample_start: '2019-08-01'\n  is_end: '2021-12-31'\n  sample_end: '2025-08-13'\n\nlookback_days: [63, 126, 252]\nsignal_lag_sessions: 1\ncost_bps: 5\nslippage_bps: 2\n"
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
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq
from sklearn.covariance import LedoitWolf


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252


def load_prices(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    query = f"daily_auto_adjust|{start.date()}|{end.date()}|{','.join(tickers)}"
    cache_path = cache_dir / f"prices_{hashlib.sha256(query.encode()).hexdigest()[:16]}.csv"

    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as file:
            fetched_at = datetime.fromisoformat(file.readline().strip())
            prices = pd.read_csv(file, index_col=0, parse_dates=True, converters=dict.fromkeys(tickers, float))
        return check_prices(prices, tickers), fetched_at

    with TemporaryDirectory(prefix="yfinance_", dir=cache_dir) as session_dir:
        yf.set_tz_cache_location(session_dir)
        raw = yf.download(
            tickers, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True, ignore_tz=True, progress=False, timeout=15,
        )
    if raw is None or raw.empty or "Close" not in raw:
        raise RuntimeError("Yahoo Finance returned no adjusted closing prices")
    prices = check_prices(raw["Close"], tickers)
    fetched_at = datetime.now(timezone.utc)

    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_dir, delete=False) as file:
        temp_path = Path(file.name)
        try:
            print(fetched_at.isoformat(), file=file)
            prices.to_csv(file, index_label="date")
            file.flush()
            temp_path.replace(cache_path)
        finally:
            temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def check_prices(prices, tickers):
    prices = prices.reindex(columns=tickers).sort_index()
    values = prices.to_numpy()
    if len(prices) < 2 or prices.index.has_duplicates:
        raise ValueError("Expected unique daily prices on multiple sessions")
    invalid = ~(np.isfinite(values) & (values > 0))
    if invalid.any():
        row, col = np.argwhere(invalid)[0]
        raise ValueError(f"Missing or invalid price for {tickers[col]} on {prices.index[row].date()}")
    return prices


def equal_risk_weights(covariance):
    size = len(covariance)
    covariance = covariance / np.diag(covariance).mean()
    diagonal = np.diag(covariance)
    budget = 1.0 / size
    weights = np.sqrt(budget / diagonal)
    # Coordinate updates minimize x' C x / 2 - sum(budget * log(x)).
    for _ in range(1000):
        for i in range(size):
            cross = covariance[i] @ weights - diagonal[i] * weights[i]
            weights[i] = (math.sqrt(cross**2 + 4 * diagonal[i] * budget) - cross) / (2 * diagonal[i])
        contributions = weights * (covariance @ weights)
        if np.allclose(contributions, budget, rtol=1e-9, atol=0):
            return weights / weights.sum()
    raise RuntimeError("Equal risk contribution solver did not converge")


def hierarchical_weights(observations, groups):
    covariance = LedoitWolf().fit(observations).covariance_
    within = np.zeros((covariance.shape[0], len(groups)))
    for col, members in enumerate(groups):
        within[members, col] = equal_risk_weights(covariance[np.ix_(members, members)])
    group_covariance = within.T @ covariance @ within
    return within @ equal_risk_weights(group_covariance)


def monthly_targets(returns, sample_start, lag, groups, lookback=None):
    # A new month makes the previous session's close the month-end signal.
    months = returns.index.to_period("M")
    first_rows = np.flatnonzero(months[1:] != months[:-1]) + 1
    targets = {}
    for first_row in first_rows:
        signal_row = first_row - 1
        fill_row = signal_row + lag
        if fill_row >= len(returns) or returns.index[fill_row] < sample_start:
            continue
        if lookback is None:
            weights = np.full(returns.shape[1], 1.0 / returns.shape[1])
        else:
            start_row = signal_row + 1 - lookback
            if start_row < 0:
                raise ValueError(f"Insufficient warm-up for {lookback} daily returns")
            observations = returns.iloc[start_row:signal_row + 1].to_numpy()
            weights = hierarchical_weights(observations, groups)
        targets[returns.index[fill_row]] = weights
    return targets


def simulate(returns, targets, fee_rate):
    weights = np.zeros(returns.shape[1])
    rows = []
    for date, asset_returns in zip(returns.index, returns.to_numpy()):
        # The incoming close-to-close return accrues to the previous close's weights.
        marked = weights * (1.0 + asset_returns)
        growth = 1.0 + weights @ asset_returns
        weights = marked / growth
        fraction, turnover, trades = 1.0, 0.0, 0
        if date in targets:
            target = targets[date]

            def funding_gap(fraction):
                traded = np.abs(fraction * target - weights).sum()
                return fraction + fee_rate * traded - 1.0

            fraction = brentq(funding_gap, 0.0, 1.0)
            changes = np.abs(fraction * target - weights)
            turnover = float(changes.sum())
            trades = int(np.count_nonzero(changes > 1e-10))
            weights = target
        rows.append((growth * fraction - 1.0, turnover, trades))
    return pd.DataFrame(rows, index=returns.index, columns=["return", "turnover", "n_trades"])


def annual_sharpe(returns):
    volatility = returns.std(ddof=1)
    if not np.isfinite(volatility) or volatility == 0:
        return None
    return float(math.sqrt(SESSIONS_PER_YEAR) * returns.mean() / volatility)


def metrics(daily_results, cash_proxy):
    returns = daily_results["return"]
    value_path = np.r_[1.0, (1.0 + returns.to_numpy()).cumprod()]
    drawdown = value_path / np.maximum.accumulate(value_path) - 1.0
    return {
        "cagr": float(value_path[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1.0),
        "ann_vol": float(returns.std(ddof=1) * math.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": annual_sharpe(returns),
        "sharpe_excess": annual_sharpe(returns - cash_proxy.loc[returns.index]),
        "max_drawdown": float(drawdown.min()),
        "total_return": float(value_path[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, cash_proxy, is_end):
    in_sample = daily_results.index <= is_end
    return {
        "is": metrics(daily_results.loc[in_sample], cash_proxy),
        "oos": metrics(daily_results.loc[~in_sample], cash_proxy),
        "full": metrics(daily_results, cash_proxy),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    dates = config["dates"]
    data_start, sample_start, is_end, sample_end = (
        pd.Timestamp(dates[key]) for key in ("data_start", "sample_start", "is_end", "sample_end")
    )
    if not data_start < sample_start <= is_end < sample_end:
        raise ValueError("Expected data_start < sample_start <= is_end < sample_end")

    groups, tickers = [], []
    for members in config["universe"].values():
        groups.append(list(range(len(tickers), len(tickers) + len(members))))
        tickers.extend(members)
    cash_ticker = config["risk_free_ticker"]
    if not groups or not all(groups) or len(set(tickers + [cash_ticker])) != len(tickers) + 1:
        raise ValueError("Expected nonempty groups with unique tickers and a distinct cash proxy")
    lag = config["signal_lag_sessions"]
    windows = config["lookback_days"]
    costs = [config["cost_bps"], config["slippage_bps"]]
    if type(lag) is not int or lag < 1:
        raise ValueError("signal_lag_sessions must be a positive integer")
    if not windows or any(type(days) is not int or days < 2 for days in windows):
        raise ValueError("lookback_days must contain integers of at least 2")
    if len(set(windows)) != len(windows):
        raise ValueError("lookback_days must be distinct")
    if any(not np.isfinite(cost) or cost < 0 for cost in costs) or sum(costs) >= 10_000:
        raise ValueError("Expected nonnegative costs totaling less than 10000 bps")
    fee_rate = sum(costs) / 10_000

    prices, fetched_at = load_prices(tickers + [cash_ticker], data_start, sample_end)
    all_returns = prices.pct_change(fill_method=None).iloc[1:]
    sector_returns = all_returns[tickers]
    cash_proxy = all_returns[cash_ticker]
    sample_returns = sector_returns.loc[sample_start:sample_end]
    is_history = sector_returns.loc[:is_end]
    is_returns = sample_returns.loc[:is_end]
    if min(len(is_returns), len(sample_returns) - len(is_returns)) < 2:
        raise ValueError("Both evaluation periods need at least two sessions")

    scores = {}
    for window in windows:
        targets = monthly_targets(is_history, sample_start, lag, groups, window)
        if not targets:
            raise ValueError("Expected at least one in-sample rebalance")
        is_daily = simulate(is_returns, targets, fee_rate)
        score = annual_sharpe(is_daily["return"] - cash_proxy.loc[is_returns.index])
        if score is None:
            raise ValueError(f"In-sample excess Sharpe is undefined for {window} days")
        scores[window] = score
    selected = max(scores, key=lambda window: (scores[window], -window))

    strategy_targets = monthly_targets(sector_returns, sample_start, lag, groups, selected)
    benchmark_targets = monthly_targets(sector_returns, sample_start, lag, groups)
    strategy_daily = simulate(sample_returns, strategy_targets, fee_rate)
    benchmark_daily = simulate(sample_returns, benchmark_targets, fee_rate)
    strategy_report = period_metrics(strategy_daily, cash_proxy, is_end)
    benchmark_report = period_metrics(benchmark_daily, cash_proxy, is_end)
    data_range = f"{sample_returns.index[0].date()} to {sample_returns.index[-1].date()}"

    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return close (splits and distributions)")
    print(f"PARAM_COMBOS_TRIED: {len(scores)}")
    print(f"SELECTED_LOOKBACK_DAYS: {selected}")
    print(f"BENCHMARK: monthly equal-weight sector ETFs | SHARPE_EXCESS_PROXY: {cash_ticker}")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()} STRATEGY: {json.dumps(strategy_report[period], allow_nan=False)}")
        print(f"{period.upper()} BENCHMARK: {json.dumps(benchmark_report[period], allow_nan=False)}")
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **strategy_report,
        "benchmark": {"name": "monthly_equal_weight_sector_etfs", **benchmark_report["full"], **benchmark_report},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()