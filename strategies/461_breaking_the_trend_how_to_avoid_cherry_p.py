"""461_breaking_the_trend_how_to_avoid_cherry_p
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/461_breaking_the_trend_how_to_avoid_cherry_p.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit f8add1176d5b1caebb66b01079720314ca29fcc4 on 2026-11-27 08:40 Europe/Oslo, exit 0, output sha256 7beacee321c60401c2996e2a7ec62dc06460eefd55658ace92bb1a8b216fd63f
Data source: yfinance 免费日线 ETF 收盘价。风险资产为 SPY、EFA、IEF、TLT、DBC、GLD、UUP、FXE，BIL 用于计算现金收益。数据优先从本地价格缓存读取，缓存缺失时通过 yfinance 下载。该运行记录的数据获取时间为 2026-11-27T07:03:23+00:00。
Data fetched: 2026-11-27T07:40:01+00:00
Sample period: 配置的数据及指标预热起点为 2008-01-02。日志所载完整评价样本为 2012-01-03 至 2025-12-31，含首尾交易日。 data_range (from output): {'start': '2012-01-03', 'end': '2025-12-31'}
IS/OOS split: 样本内为 2012-01-03 至 2019-12-31，含边界日。样本外为 2019-12-31 之后至 2025-12-31 的实际交易日。信号估计与持仓连续计算，随后按日期切分同一次模拟的结果。
Costs and slippage: 每次收盘调仓按买卖绝对名义金额合计扣除单边 2 bps 交易成本和 3 bps 滑点，滑点以固定扣费建模，两者合计 5 bps。名义金额为扣费后净值对应的目标持仓减去按收盘价变动后的原持仓。另按调仓前空头敞口扣除年化 50 bps 融券费，乘以两次收盘间的日历天数并除以 365。
Signal lag: 交易日 t 收盘价用于计算目标权重，目标顺延 1 个交易日，在 t+1 收盘模拟调仓。至 t+1 收盘的区间收益归原持仓，新持仓从随后一段收盘到收盘的区间开始计价。样本首次入场在首个评价日收盘。
Price adjustment: yfinance auto_adjust total-return adjusted closes
Survivorship bias: 固定 ETF 名单为事后从仍有公开历史数据的产品中选定。README 明确记录幸存者偏差和代理选择偏差。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.03167050393839688, ann_vol=0.043443702502825775, sharpe=0.7394671561597753, sharpe_excess=0.6165085679629827, max_drawdown=-0.07846018788139253, total_return=0.2826647957065067, turnover=4.784907378444121, n_trades=16096
    oos: cagr=0.04154964153955554, ann_vol=0.049412050009074865, sharpe=0.8486932252565359, sharpe_excess=0.3188784061835749, max_drawdown=-0.08676269147426674, total_return=0.2758487962904421, turnover=3.7083588881909724, n_trades=12064
    full: cagr=0.035891291312835305, ann_vol=0.046089530635760934, sharpe=0.7882069933264355, sharpe_excess=0.4786213642367812, max_drawdown=-0.08676269147426696, total_return=0.6364863356462758, turnover=4.323704218415215, n_trades=28160
    benchmark: cagr=0.059085313059533906, ann_vol=0.07433962491708757, sharpe=0.8094822516477763, sharpe_excess=0.6178770762566226, max_drawdown=-0.16906580278509553, total_return=1.2296760965940812, turnover=0.07155513152514205, n_trades=8
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: '461_ breaking_the_trend_how_to_avoid_cherry_p'\nuniverse:\n  - SPY\n  - EFA\n  - IEF\n  - TLT\n  - DBC\n  - GLD\n  - UUP\n  - FXE\nrisk_free_ticker: BIL\ndata_start: '2008-01-02'\nstart: '2012-01-03'\nis_end: '2019-12-31'\nend: '2025-12-31'\nsignal_lag_sessions: 1\nema_relaxation_sessions: 120\nrisk_vol_relaxation_sessions: 40\ncorrelation_relaxation_sessions: 750\nweight_relaxation_sessions: 20\nannual_vol_target: 0.10\ngross_exposure_limit: 1.0\ncost_bps: 2.0\nslippage_bps: 3.0\nshort_borrow_bps_annual: 50.0\n"
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
PRICE_ADJUSTMENT = "yfinance auto_adjust total-return adjusted closes"


def validate_prices(prices, tickers, start, end):
    if prices.empty or list(prices.columns) != tickers:
        raise ValueError("Adjusted close data missing for the configured tickers")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and sorted")
    if not (start <= prices.index[0] <= start + pd.Timedelta(days=7)
            and end - pd.Timedelta(days=7) <= prices.index[-1] <= end):
        raise ValueError("Price history does not cover the configured dates")
    bad = ~np.isfinite(prices) | (prices <= 0)
    if bad.to_numpy().any():
        row, column = np.argwhere(bad.to_numpy())[0]
        raise ValueError(f"Invalid price for {tickers[column]} on {prices.index[row].date()}")


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    key = f"adjusted-close-v1:{','.join(tickers)}:{start.date()}:{end.date()}"
    path = CACHE / f"prices_{hashlib.sha256(key.encode()).hexdigest()[:16]}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as source:
            label, fetched_at = source.readline().strip().split(": ", 1)
            if label != "DATA_FETCHED_AT" or datetime.fromisoformat(fetched_at).tzinfo is None:
                raise ValueError(f"Invalid fetch timestamp in {path}")
            prices = pd.read_csv(source, index_col="Date", parse_dates=["Date"])
        validate_prices(prices, tickers, start, end)
        return prices, fetched_at

    with TemporaryDirectory(dir=CACHE) as session_cache:
        yf.set_tz_cache_location(session_cache)
        download = yf.download(
            tickers, start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
            interval="1d", auto_adjust=True, ignore_tz=True, progress=False,
            threads=4, timeout=12,
        )
    if download is None or download.empty:
        raise RuntimeError("yfinance returned no daily price history")
    prices = download["Close"].reindex(columns=tickers)
    validate_prices(prices, tickers, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as output:
        temp_path = Path(output.name)
        try:
            output.write(f"DATA_FETCHED_AT: {fetched_at}\n")
            prices.to_csv(output, index_label="Date")
            output.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def trend_targets(prices, config):
    log_prices = np.log(prices)
    log_returns = log_prices.diff()
    relaxation = config["ema_relaxation_sessions"]
    alpha = 1.0 / relaxation

    # Equations (8) and (9). The current return uses the previous day's variance.
    variance = log_returns.pow(2).ewm(alpha=alpha, adjust=False, min_periods=relaxation).mean()
    standardized_returns = log_returns.div(np.sqrt(variance.shift()).where(lambda x: x > 0))
    signal = standardized_returns.fillna(0).ewm(alpha=alpha, adjust=False).mean() / np.sqrt(alpha)

    vol_days = config["risk_vol_relaxation_sessions"]
    vols = log_returns.ewm(alpha=1 / vol_days, adjust=False, min_periods=vol_days).std(bias=True)

    # A weekly estimate first becomes usable at its Friday close or the next session.
    weekly_returns = log_prices.resample("W-FRI").last().diff().dropna().loc[:prices.index[-1]]
    corr_weeks = config["correlation_relaxation_sessions"] / 5
    weekly_corr = weekly_returns.ewm(alpha=1 / corr_weeks, adjust=False,
                                     min_periods=int(np.ceil(corr_weeks))).corr()
    asset_count = len(prices.columns)
    matrices = weekly_corr.to_numpy().reshape(len(weekly_returns), asset_count, asset_count)
    usable = np.isfinite(matrices).all(axis=(1, 2))
    if not usable.any():
        raise ValueError("Insufficient weekly history for a correlation estimate")
    matrices = matrices[usable]
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    if (eigenvalues <= 0).any():
        raise ValueError("Risk assets must have a positive definite correlation matrix")
    inverse_roots = (eigenvectors / np.sqrt(eigenvalues[:, None, :])) @ eigenvectors.swapaxes(1, 2)

    week_indices = weekly_returns.index[usable].get_indexer(prices.index, method="pad")
    days = prices.index[week_indices >= 0]
    week_indices = week_indices[week_indices >= 0]
    daily_vols = vols.loc[days].to_numpy()
    whitened = np.einsum("tij,tj->ti", inverse_roots[week_indices], signal.loc[days].to_numpy())
    raw_weights = pd.DataFrame(whitened / daily_vols, index=days, columns=prices.columns)
    weights = raw_weights.ewm(alpha=1 / config["weight_relaxation_sessions"], adjust=False).mean()

    risk_loadings = weights.to_numpy() * daily_vols
    daily_risk = np.sqrt(np.einsum("ti,tij,tj->t", risk_loadings, matrices[week_indices], risk_loadings))
    target_daily_risk = config["annual_vol_target"] / np.sqrt(SESSIONS_PER_YEAR)
    gross = weights.abs().sum(axis=1).to_numpy()
    relative_size = np.maximum(daily_risk / target_daily_risk, gross / config["gross_exposure_limit"])
    scale = 1 / np.where(relative_size == 0, 1.0, relative_size)
    targets = weights.mul(scale, axis=0).reindex(prices.index)
    return targets.shift(config["signal_lag_sessions"])


def simulate(asset_returns, cash_returns, targets, cost_rate, borrow_rate):
    """Mark existing holdings to the close before applying the new close allocation."""
    weights = np.zeros(len(asset_returns.columns))
    records = []
    calendar_days = asset_returns.index.to_series().diff().dt.days.fillna(0)
    for returns, cash_return, target, days in zip(
        asset_returns.to_numpy(), cash_returns.to_numpy(), targets.to_numpy(), calendar_days
    ):
        drifted_values = weights * (1 + returns)
        free_cash = max(0.0, 1 - np.abs(weights).sum())
        short_exposure = -np.minimum(weights, 0).sum()
        before_cost = 1 + weights @ returns + free_cash * cash_return
        before_cost -= short_exposure * borrow_rate * days / 365
        if before_cost <= 0:
            raise ValueError("Portfolio equity fell to zero")

        after_cost = before_cost
        turnover = 0.0
        trades = 0
        if np.isnan(target).all():
            weights = drifted_values / before_cost
        else:
            # Solve for post-cost NAV so the actual holdings match the target weights.
            for _ in range(20):
                order_values = target * after_cost - drifted_values
                next_nav = before_cost - cost_rate * np.abs(order_values).sum()
                if abs(after_cost - next_nav) < 1e-12:
                    break
                after_cost = next_nav
            else:
                raise ValueError("Could not size post-cost holdings")
            after_cost = next_nav
            turnover = np.abs(order_values).sum() / before_cost
            trades = int((np.abs(order_values) > 1e-10 * before_cost).sum())
            weights = target
        records.append((after_cost - 1, turnover, trades))
    return pd.DataFrame(records, index=asset_returns.index, columns=["return", "turnover", "n_trades"])


def metrics(performance, cash_returns):
    returns = performance["return"]
    if len(returns) < 2:
        raise ValueError("Each evaluation period needs at least two sessions")
    wealth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    annualization = np.sqrt(SESSIONS_PER_YEAR)
    vol = returns.std()
    excess = returns - cash_returns.reindex(returns.index)
    excess_vol = excess.std()
    years = len(returns) / SESSIONS_PER_YEAR
    return {
        "cagr": float(wealth[-1] ** (1 / years) - 1),
        "ann_vol": float(vol * annualization),
        "sharpe": float(returns.mean() / vol * annualization) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * annualization) if excess_vol > 0 else None,
        "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
        "total_return": float(wealth[-1] - 1),
        "turnover": float(performance["turnover"].sum() / years),
        "n_trades": int(performance["n_trades"].sum()),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    data_start, start, is_end, end = (pd.Timestamp(config[key]) for key in ("data_start", "start", "is_end", "end"))
    if not data_start < start < is_end < end or config["signal_lag_sessions"] < 1:
        raise ValueError("Check the configured dates and use a signal lag of at least one session")
    if not 0 < config["gross_exposure_limit"] <= 1 or config["annual_vol_target"] <= 0:
        raise ValueError("Use a positive volatility target and a gross limit in (0, 1]")
    cost_rate = (config["cost_bps"] + config["slippage_bps"]) / 10_000
    borrow_rate = config["short_borrow_bps_annual"] / 10_000
    if min(config["cost_bps"], config["slippage_bps"], borrow_rate) < 0 or cost_rate >= 0.1:
        raise ValueError("Check the cost, slippage and short borrow rates")

    universe = config["universe"]
    cash_ticker = config["risk_free_ticker"]
    prices, fetched_at = load_prices([*universe, cash_ticker], data_start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print("PARAM_COMBOS_TRIED: 1")

    targets = trend_targets(prices[universe], config).loc[start:end]
    if targets.empty or not np.isfinite(targets.to_numpy()).all():
        raise ValueError("Incomplete targets in the sample; check signal warm-up and price history")
    returns = prices.pct_change(fill_method=None).loc[targets.index].copy()
    returns.iloc[0] = 0.0  # Initial entry occurs at the first sample close.
    cash_returns = returns[cash_ticker]
    performance = simulate(returns[universe], cash_returns, targets, cost_rate, borrow_rate)

    benchmark_targets = pd.DataFrame(np.nan, index=targets.index, columns=universe)
    benchmark_targets.iloc[0] = 1 / len(universe)
    benchmark = simulate(returns[universe], cash_returns, benchmark_targets, cost_rate, borrow_rate)

    results = {
        "strategy": config["strategy"],
        "data_range": {"start": str(targets.index[0].date()), "end": str(targets.index[-1].date())},
        "is": metrics(performance.loc[:is_end], cash_returns),
        "oos": metrics(performance.loc[performance.index > is_end], cash_returns),
        "full": metrics(performance, cash_returns),
        "benchmark": metrics(benchmark, cash_returns),
    }
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(results[period], allow_nan=False)}")
    print(f"BENCHMARK (equal-weight entry, buy and hold): {json.dumps(results['benchmark'], allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()