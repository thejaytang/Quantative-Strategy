"""701_dependency_network_based_portfolio_desig
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/701_dependency_network_based_portfolio_desig.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 9532cd9f815cc2d43ceb143b2f1416acf34e9864 on 2026-11-27 20:01 Europe/Oslo, exit 0, output sha256 a94e091b3a375eb73860e992bd2ffd57d4c2656a2136b2c11ef4a0e20e144f08
Data source: 通过 yfinance 获取 Yahoo Finance 免费日频开盘价、收盘价和现金分派（股息及供应商提供的资本利得分派），读取或写入 data_cache 下的 NPZ 缓存。股票池为 MSFT、AAPL、AMZN、META、BRK-B、JPM、GOOG、JNJ、PG、XOM、V、T、HD、VZ、BAC、MA、DIS、INTC、CVX、MRK。SPY 为买入持有基准，BIL 日总回报用于计算超额夏普。
Data fetched: 2026-11-27T19:01:26+00:00
Sample period: 预热行情请求起点为 2019-09-03。记录中的回测交易日范围为 2021-01-04 至 2026-10-30，含首末日。   data_range (from output): ['2021-01-04', '2026-10-30']
IS/OOS split: 样本内（IS）为 2021-01-04 至 2025-07-31，样本外（OOS）为 2025-08-01 至 2026-10-30。默认组合连续模拟，OOS 延续 IS 期末仓位与现金，滚动信号可使用已发生的历史行情。稳健性变体仅在 IS 模拟。
Costs and slippage: 每笔买卖均模拟相对开盘价 5 bps 的不利滑点，并按滑点后的成交金额收取 5 bps 费用。以开盘名义金额计，买入扣款乘以 (1 + 0.0005) × (1 + 0.0005)，卖出收入乘以 (1 − 0.0005) × (1 − 0.0005)。这些系数计入可用资金和成交金额求解。IS 的双倍摩擦情景同时将费用与滑点各提高到 10 bps。
Signal lag: 默认信号窗口为截至前一交易日 t 收盘的 252 个日总回报。若 t 的下一交易日为当月首个交易日，则使用该目标权重在下一交易日开盘模拟成交。
Price adjustment: Yahoo Finance split-adjusted OHLC (auto_adjust=False, repair=False) with ex-date distributions modeled as reinvested at the open
Survivorship bias: 股票池固定为 SPY 于 2019-09-30 持仓中按列示市值排名最高的 20 家发行人，申报于 2019-11-27 公开。同一发行人保留列示市值最高的类别，FB 使用更名后的 META 连续历史。该做法减少使用现今成分股带来的幸存者偏差，仍受固定大盘股覆盖、公司行为及 Yahoo 历史数据覆盖限制。下载路径遇到无效开收盘价或股息数据时会报错。
Parameter combinations tried: 5
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.180109088218394, ann_vol=0.16231059879808513, sharpe=1.1016448681131725, sharpe_excess=0.9179301977059411, max_drawdown=-0.15055270383285213, total_return=1.1277998695732054, turnover=6.02387974706432, n_trades=337
    oos: cagr=0.12460145429459679, ann_vol=0.14576865194060687, sharpe=0.8784777165978142, sharpe_excess=0.6235590105012794, max_drawdown=-0.156753085040234, total_return=0.1581059715766191, turnover=6.672326805199812, n_trades=93
    full: cagr=0.16793902316396747, ann_vol=0.1588542080239822, sharpe=1.056869584514902, sharpe_excess=0.8592243093271341, max_drawdown=-0.15675308504023477, total_return=1.4642177352726753, turnover=6.163402167359866, n_trades=430
    benchmark: cagr=0.14284899294073772, ann_vol=0.165927103419055, sharpe=0.8876782441618803, sharpe_excess=0.6983672974622112, max_drawdown=-0.24492340105043264, total_return=1.17213895555982, turnover=0.17195914540579144, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 701_dependency_network_based_portfolio_desig\n# Largest 20 issuers by the SPY filing's 2019-09-30 position values.\n# For duplicate classes, use the larger class. META is the former FB series.\nuniverse: [MSFT, AAPL, AMZN, META, BRK-B, JPM, GOOG, JNJ, PG, XOM, V, T, HD, VZ, BAC, MA, DIS, INTC, CVX, MRK]\nbenchmark: SPY\nrisk_free: BIL\n\nhistory_start: 2019-09-03\nstart: 2021-01-04\noos_start: 2025-08-01\nend: 2026-10-30\n\nlookback_sessions: 252\nvar_lags: 1\nfevd_horizon: 10\ntop_k: 5\nvar_tail_probability: 0.05\nvar_floor: 0.001\nrebalance: monthly\nsignal_lag_sessions: 1\ncost_bps: 5.0\nslippage_bps: 5.0\n\nrobustness:\n  is_lookback_sessions: [210, 294, 120]\n  is_friction_multiplier: 2.0\n"
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
"""C269 long-only dependency-network research backtest.

Paper at https://arxiv.org/abs/2507.20039
Generalized FEVD and rolling connectedness references (GPL-3.0-or-later)
https://pypi.org/project/pyconnectedness/
https://github.com/zstoi/PyConnectedness/blob/main/src/pyconnectedness/connectedness/decomposition.py
https://github.com/zstoi/PyConnectedness/blob/main/src/pyconnectedness/connectedness/dynamic.py
ConnectednessApproach FEVD reference (GPL-3.0)
https://github.com/GabauerDavid/ConnectednessApproach/blob/main/R/FEVD.R
Network portfolio tutorial (MIT)
https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/17_network_portfolio.html
https://github.com/ramtin-asadi/Quantitative-Finance-Lab/blob/main/LICENSE
Paper replication search at https://www.catalyzex.com/paper/dependency-network-based-portfolio-design
These references informed the design. The implementation is original.
"""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from statsmodels.tsa.api import VAR
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
PRICE_ADJUSTMENT = (
    "Yahoo Finance split-adjusted OHLC (auto_adjust=False, repair=False) with "
    "ex-date distributions modeled as reinvested at the open"
)


def load_prices(tickers, history_start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_spec = f"yf-daily-open-close-actions-v2|{','.join(tickers)}|{history_start}|{end}"
    cache_path = cache_dir / f"yahoo_{sha256(cache_spec.encode()).hexdigest()[:16]}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            dates = pd.DatetimeIndex(saved["dates"], name="Date")
            opens, closes, distributions = (
                pd.DataFrame(saved[key], index=dates, columns=tickers)
                for key in ("opens", "closes", "distributions")
            )
            fetched_at = saved["fetched_at"].item()
        return opens, closes, distributions, fetched_at

    with TemporaryDirectory(dir=cache_dir, prefix="yf_session_") as session_dir:
        yf.set_tz_cache_location(session_dir)
        quotes = yf.download(
            tickers,
            start=str(history_start),
            end=str(end + timedelta(days=1)),
            interval="1d",
            auto_adjust=False,
            actions=True,
            repair=False,
            progress=False,
            threads=4,
            timeout=12,
        )
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if quotes.empty:
        raise RuntimeError("Yahoo Finance returned no daily bars.")

    dates = pd.DatetimeIndex(quotes.index).tz_localize(None).normalize()
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Yahoo Finance returned invalid daily bar dates.")
    quotes.index = dates
    opens = quotes["Open"].reindex(index=dates, columns=tickers)
    closes = quotes["Close"].reindex(index=dates, columns=tickers)
    dividends = quotes["Dividends"].reindex(index=dates, columns=tickers)
    bad_quotes = ~(np.isfinite(opens) & np.isfinite(closes)) | (opens <= 0) | (closes <= 0)
    bad_actions = ~np.isfinite(dividends) | (dividends < 0)
    if (bad_quotes | bad_actions).any().any():
        missing = (bad_quotes | bad_actions).sum()
        raise ValueError(f"Incomplete price or dividend data by ticker: {missing[missing > 0].to_dict()}")

    distributions = dividends
    if "Capital Gains" in quotes.columns.get_level_values(0):
        gains = quotes["Capital Gains"].reindex(index=dates, columns=tickers).fillna(0.0)
        distributions = dividends + gains
    if not np.isfinite(distributions).all().all() or (distributions < 0).any().any():
        raise ValueError("Yahoo Finance returned invalid cash distributions.")
    if dates[-1] != pd.offsets.BDay().rollback(pd.Timestamp(end)):
        raise ValueError(f"Yahoo Finance data end on {dates[-1].date()} before the configured end.")

    # A single atomic replacement keeps the data and its retrieval time together.
    with NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            np.savez_compressed(
                temporary,
                dates=dates.to_numpy(dtype="datetime64[D]"),
                opens=opens.to_numpy(),
                closes=closes.to_numpy(),
                distributions=distributions.to_numpy(),
                fetched_at=fetched_at,
            )
            temporary.close()
            temporary_path.replace(cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return opens, closes, distributions, fetched_at


def price_relatives(opens, closes, distributions):
    # Only positions from the preceding close earn the ex-date distribution.
    overnight = (opens + distributions) / closes.shift(1)
    intraday = closes / opens
    return overnight, intraday


def network_weights(window, config):
    n_assets = window.shape[1]
    pairs = list(combinations(range(n_assets), 2))
    values = window.to_numpy()
    fits = [VAR(values[:, pair]).fit(config["var_lags"], trend="c") for pair in pairs]

    covariance = np.stack([fit.sigma_u for fit in fits])
    shock_scale = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    if not np.all(shock_scale > 0):
        raise ValueError(f"Degenerate VAR residuals at {window.index[-1].date()}.")

    # A generalized shock scales a covariance column by its own residual SD.
    # Normalizing across sources cancels the shared forecast variance per row.
    shocks = covariance / shock_scale[:, None, :]
    responses = np.stack([fit.ma_rep(maxn=config["fevd_horizon"] - 1) for fit in fits])
    energy = np.square(responses @ shocks[:, None, :, :]).sum(axis=1)
    shares = energy / energy.sum(axis=2, keepdims=True)
    strength = np.maximum(shares[:, 0, 1], shares[:, 1, 0])
    edge_cost = np.maximum(1.0 - strength, np.finfo(float).eps)
    pair_start, pair_end = np.asarray(pairs).T
    graph = csr_matrix((edge_cost, (pair_start, pair_end)), shape=(n_assets, n_assets))
    tree = minimum_spanning_tree(graph).tocoo()
    if tree.nnz != n_assets - 1:
        raise ValueError(f"Disconnected dependency graph at {window.index[-1].date()}.")

    endpoints = np.concatenate([tree.row, tree.col])
    degree = np.bincount(endpoints, minlength=n_assets)
    incident_strength = np.bincount(endpoints, weights=np.tile(1.0 - tree.data, 2), minlength=n_assets)
    ranked = np.lexsort((window.columns.to_numpy(), -incident_strength, -degree))
    selected = ranked[:config["top_k"]]
    simple_returns = np.expm1(values[:, selected])
    historical_var = -np.quantile(simple_returns, config["var_tail_probability"], axis=0)
    inverse_var = 1.0 / np.maximum(historical_var, config["var_floor"])
    weights = np.zeros(n_assets)
    weights[selected] = inverse_var / inverse_var.sum()
    return weights


def monthly_targets(log_returns, dates, lookback, config):
    months = log_returns.index.to_period("M")
    first_sessions = log_returns.index[~months.duplicated()]
    fill_dates = first_sessions.intersection(dates)
    targets = {}
    for fill_date in fill_dates:
        signal_position = log_returns.index.get_loc(fill_date) - config["signal_lag_sessions"]
        window_start = signal_position + 1 - lookback
        if window_start < 1:
            raise ValueError(f"Insufficient warm-up for the {lookback}-session window on {fill_date.date()}.")
        window = log_returns.iloc[window_start:signal_position + 1]
        if window.index[-1] >= fill_date or not np.isfinite(window.to_numpy()).all():
            raise ValueError(f"Invalid signal window for {fill_date.date()}.")
        targets[fill_date] = network_weights(window, config)
    return targets


def open_rebalance(positions, cash, weights, buy_multiplier, sell_multiplier):
    opening_value = float(positions.sum()) + cash
    # Choose the greatest fully funded target value, including two-sided friction.
    low, high = 0.0, opening_value
    for _ in range(50):
        trial_value = (low + high) / 2.0
        change = weights * trial_value - positions
        debits = np.maximum(change, 0).sum() * buy_multiplier
        credits = np.maximum(-change, 0).sum() * sell_multiplier
        if debits <= cash + credits:
            low = trial_value
        else:
            high = trial_value
    new_positions = weights * low
    changes = new_positions - positions
    cash += np.maximum(-changes, 0).sum() * sell_multiplier
    cash -= np.maximum(changes, 0).sum() * buy_multiplier
    if cash < 0.0:
        raise ArithmeticError("Opening rebalance produced negative cash.")
    turnover = np.abs(changes).sum() / opening_value
    trades = int(np.count_nonzero(np.abs(changes) > 1e-10 * opening_value))
    return new_positions, cash, turnover, trades


def simulate(overnight, intraday, targets, cost_bps, slippage_bps):
    fee = cost_bps / 10_000.0
    slip = slippage_bps / 10_000.0
    buy_multiplier = (1.0 + slip) * (1.0 + fee)
    sell_multiplier = (1.0 - slip) * (1.0 - fee)
    positions = np.zeros(overnight.shape[1])
    cash = previous_close_value = 1.0
    daily_results = []
    for date, gap, session_move in zip(overnight.index, overnight.to_numpy(), intraday.to_numpy()):
        positions *= gap
        turnover, trades = 0.0, 0
        if date in targets:
            positions, cash, turnover, trades = open_rebalance(
                positions, cash, targets[date], buy_multiplier, sell_multiplier
            )
        positions *= session_move
        close_value = positions.sum() + cash
        daily_results.append((close_value / previous_close_value - 1.0, turnover, trades))
        previous_close_value = close_value
    return pd.DataFrame(daily_results, index=overnight.index, columns=["return", "turnover", "n_trades"])


def metrics(daily_results, risk_free):
    returns = daily_results["return"].to_numpy()
    excess = returns - risk_free.loc[daily_results.index].to_numpy()
    growth = np.r_[1.0, np.cumprod(1.0 + returns)]
    volatility = float(np.std(returns, ddof=1))
    excess_volatility = float(np.std(excess, ddof=1))
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1.0),
        "ann_vol": float(volatility * annualizer),
        "sharpe": float(annualizer * returns.mean() / volatility) if volatility > 0 else None,
        "sharpe_excess": float(annualizer * excess.mean() / excess_volatility) if excess_volatility > 0 else None,
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1.0)),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, risk_free, oos_start):
    is_rows = daily_results.index < oos_start
    return {
        "is": metrics(daily_results.loc[is_rows], risk_free),
        "oos": metrics(daily_results.loc[~is_rows], risk_free),
        "full": metrics(daily_results, risk_free),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    universe = sorted(config["universe"])
    benchmark, risk_free_ticker = config["benchmark"], config["risk_free"]
    tickers = [*universe, benchmark, risk_free_ticker]
    history_start, start, oos_start, end = (
        pd.Timestamp(config[key]) for key in ("history_start", "start", "oos_start", "end")
    )
    if not history_start < start < oos_start <= end:
        raise ValueError("Expected history_start < start < oos_start <= end.")
    if len(tickers) != len(set(tickers)) or not 1 <= config["top_k"] < len(universe):
        raise ValueError("Tickers must be distinct and top_k must be smaller than the universe.")
    if config["signal_lag_sessions"] < 1 or config["rebalance"] != "monthly":
        raise ValueError("The schedule requires a positive session lag and monthly rebalancing.")
    robustness = config["robustness"]
    lookback = config["lookback_sessions"]
    alternative_lookbacks = robustness["is_lookback_sessions"]
    friction_multiplier = robustness["is_friction_multiplier"]
    if len({lookback, *alternative_lookbacks}) != 1 + len(alternative_lookbacks):
        raise ValueError("Specify distinct official and robustness lookbacks.")
    fee, slip = config["cost_bps"], config["slippage_bps"]
    if friction_multiplier <= 1 or not all(0 <= rate * friction_multiplier < 10_000 for rate in (fee, slip)):
        raise ValueError("Invalid trading cost, slippage, or friction multiplier.")

    opens, closes, distributions, fetched_at = load_prices(tickers, history_start.date(), end.date())
    overnight, intraday = price_relatives(opens, closes, distributions)
    dates = opens.loc[start:end].index
    is_dates = dates[dates < oos_start]
    if min(len(is_dates), len(dates) - len(is_dates)) < 2:
        raise ValueError("Two or more daily observations are needed in each evaluation period.")
    if not np.isfinite(overnight.loc[dates].to_numpy()).all():
        raise ValueError("Missing overnight prices in the evaluation period.")
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}", flush=True)

    close_relatives = overnight * intraday
    log_returns = np.log(close_relatives[universe])
    risk_free = close_relatives[risk_free_ticker] - 1.0
    is_log_returns = log_returns.loc[:is_dates[-1]]
    is_overnight = overnight.loc[is_dates, universe]
    is_intraday = intraday.loc[is_dates, universe]
    is_risk_free = risk_free.loc[is_dates]
    is_targets = monthly_targets(is_log_returns, is_dates, lookback, config)
    print(f"ROBUSTNESS_IS ({is_dates[0].date()} to {is_dates[-1].date()})", flush=True)
    for alternative in alternative_lookbacks:
        targets = monthly_targets(is_log_returns, is_dates, alternative, config)
        stats = metrics(simulate(is_overnight, is_intraday, targets, fee, slip), is_risk_free)
        print(f"ROBUSTNESS_IS_LOOKBACK_{alternative}: {json.dumps(stats, allow_nan=False)}", flush=True)
    friction_stats = metrics(
        simulate(is_overnight, is_intraday, is_targets, fee * friction_multiplier, slip * friction_multiplier),
        is_risk_free,
    )
    print(f"ROBUSTNESS_IS_FRICTION_{friction_multiplier:g}X: {json.dumps(friction_stats, allow_nan=False)}", flush=True)

    oos_dates = dates[dates >= oos_start]
    targets = is_targets | monthly_targets(log_returns, oos_dates, lookback, config)
    daily_results = simulate(overnight.loc[dates, universe], intraday.loc[dates, universe], targets, fee, slip)
    benchmark_results = simulate(
        overnight.loc[dates, [benchmark]], intraday.loc[dates, [benchmark]], {dates[0]: np.ones(1)}, fee, slip
    )
    results = period_metrics(daily_results, risk_free, oos_start)
    benchmark_periods = period_metrics(benchmark_results, risk_free, oos_start)
    if any(stats["n_trades"] == 0 for stats in results.values()):
        raise RuntimeError("Official strategy requires fills in both evaluation periods.")
    print(f"PARAM_COMBOS_TRIED: {2 + len(alternative_lookbacks)}")
    for period, stats in results.items():
        print(f"STRATEGY_{period.upper()}: {json.dumps(stats, allow_nan=False)}")
    for period, stats in benchmark_periods.items():
        print(f"BENCHMARK_{period.upper()} ({benchmark}): {json.dumps(stats, allow_nan=False)}")
    print("已回测 | pending review / new")
    payload = {
        "strategy": config["strategy"],
        "data_range": [str(dates[0].date()), str(dates[-1].date())],
        **results,
        "benchmark": benchmark_periods["full"],
    }
    print(f"RESULTS_JSON: {json.dumps(payload, allow_nan=False)}")


if __name__ == "__main__":
    signal.alarm(14 * 60)
    main()
