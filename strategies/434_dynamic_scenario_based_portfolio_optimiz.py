"""434_dynamic_scenario_based_portfolio_optimiz
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/434_dynamic_scenario_based_portfolio_optimiz.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 9102661980b3a465a33199af26a9f4e7a35fe278 on 2026-11-27 08:03 Europe/Oslo, exit 0, output sha256 f785b951c9da64250df2928cb63fa2766676c9e0de9de17a2e48c4b9583c6212
Data source: Yahoo Finance 日频 ETF 收盘价，经 yfinance 获取。策略使用 VT、VTI、ACWI、ACWX、SUSA、ROUS、IYG、BND、FPE、BYLD、NTSX、GAL、MDIV、AOM、AOR、INKM、HNDL、ALTY、IYLD、CVY、PDBC、QAI、HDG、MNA、RAAX、RLY、SRET、YYY、GAA、GMOM。另取 BIL 的收益作为事后统计的无风险代理（config.yaml:2-6，README.md:16、20，strategy.py:60-80、143-144、178-186）。
Data fetched: 2026-11-27T07:03:25+00:00
Sample period: 配置的价格样本为 2018-12-19 至 2026-09-30（两端包含），回测评估前的历史价格用于预热。日志记录的完整评估区间为 2022-12-21 至 2026-09-30（config.yaml:7-9、12，README.md:22，strategy.py:46-49、74-80、133-139，logs/run_20261127_073038.log:7）。   data_range (from output): {'start': '2022-12-21', 'end': '2026-09-30'}
IS/OOS split: 样本内为 2022-12-21 至 2024-12-31。配置的样本外起始日为 2025-01-01，日志中实际样本外区间为 2025-01-02 至 2026-09-30。代码先连续模拟，再按日期切分；样本外沿用此前的持仓和目标状态，协方差继续滚动更新（config.yaml:9-13，README.md:22，strategy.py:93-94、155-175、204-210，logs/run_20261127_073038.log:5-6）。
Costs and slippage: 交易成本 10 bps，另建模固定滑点 5 bps，合计 15 bps 乘以单边换手率，从成交日按原持仓计价后的组合价值扣除。常规调仓以目标权重与收盘价格变动后的实际权重之差的半 L1 距离计换手率。策略和等权买入持有基准首次用现金建仓均按 100% 换手扣费，基准此后无调仓费用。固定模型未涵盖市场冲击或变化的买卖价差（config.yaml:16-17，strategy.py:146-148、161-175，README.md:26-28、32）。
Signal lag: 用截至 t 日收盘的 1,008 个交易日 ETF 收益计算目标权重，在 t+1 交易日收盘模拟成交。t 至 t+1 的收盘到收盘收益归原持仓，回测首次成交前持有现金；新权重从成交后的下一段收盘到收盘区间起计收益（config.yaml:12-13，strategy.py:133-139、155-175，README.md:26-28）。
Price adjustment: yfinance auto_adjust total-return adjusted close
Survivorship bias: 使用 README 所述的论文附录 A 固定 ETF 名单。缺少历史时点成分记录和已退市基金样本，存在幸存者偏差与资产选择偏差（README.md:5、32，config.yaml:2-5，strategy.py:60-63）。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.10123538753245742, ann_vol=0.07882022941589692, sharpe=1.2630215849871045, sharpe_excess=0.6329351404499473, max_drawdown=-0.07286289980006089, total_return=0.21504195153869055, turnover=0.908607157928778, n_trades=15007
    oos: cagr=0.12175157127067315, ann_vol=0.08330458371304064, sharpe=1.4210044391755157, sharpe_excess=0.955509028574482, max_drawdown=-0.09161870233939973, total_return=0.22047023019721923, turnover=0.414982715985171, n_trades=12089
    full: cagr=0.11066568439098967, ann_vol=0.08088161674045737, sharpe=1.3383455296814277, sharpe_excess=0.7865803223376714, max_drawdown=-0.09161870233939984, total_return=0.48292253029370436, turnover=0.6805797994410865, n_trades=27096
    benchmark: cagr=0.11860672989214449, ann_vol=0.08984936941642806, sharpe=1.2926050641820854, sharpe_excess=0.7959261572853656, max_drawdown=-0.10704492831497336, total_return=0.5231178886782155, turnover=0.266384778012685, n_trades=30
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 434_dynamic_scenario_based_portfolio_optimiz\nuniverse:\n  [VT, VTI, ACWI, ACWX, SUSA, ROUS, IYG, BND, FPE, BYLD, NTSX, GAL, MDIV,\n   AOM, AOR, INKM, HNDL, ALTY, IYLD, CVY, PDBC, QAI, HDG, MNA, RAAX, RLY,\n   SRET, YYY, GAA, GMOM]\nrisk_free_ticker: BIL\ndata_start: '2018-12-19'\ndata_end: '2026-09-30'\nbacktest_start: '2022-12-21'\nis_end: '2024-12-31'\noos_start: '2025-01-01'\nlookback_sessions: 1008\nexecution_lag_sessions: 1\nsessions_per_year: 252\ntarget_turnover_cap: 0.00013227513227513228  # 1 / (30 × 252)\ncost_bps: 10.0\nslippage_bps: 5.0\n"
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
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize
from threadpoolctl import threadpool_limits
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
TRADE_TOLERANCE = 1e-8


def load_config():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key in ("data_start", "data_end", "backtest_start", "is_end", "oos_start"):
        config[key] = pd.Timestamp(config[key]).normalize()
    if not (config["data_start"] < config["backtest_start"] <= config["is_end"]
            < config["oos_start"] <= config["data_end"]):
        raise ValueError("Invalid data or IS/OOS dates")
    if len(set(config["universe"])) != len(config["universe"]) or len(config["universe"]) < 2:
        raise ValueError("The universe must contain at least two distinct ETFs")
    if config["risk_free_ticker"] != "BIL":
        raise ValueError("The risk-free proxy must be BIL for ETF total-return accounting")
    if config["risk_free_ticker"] in config["universe"]:
        raise ValueError("The risk-free proxy must be outside the strategy universe")
    for key, minimum in (("lookback_sessions", 2), ("execution_lag_sessions", 1),
                         ("sessions_per_year", 1)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer of at least {minimum}")
    if not 0 < config["target_turnover_cap"] <= 1:
        raise ValueError("target_turnover_cap must be in (0, 1]")
    rates = [config["cost_bps"], config["slippage_bps"]]
    if any(not np.isfinite(rate) or rate < 0 for rate in rates) or sum(rates) >= 10_000:
        raise ValueError("Invalid trading cost or slippage")
    return config


def check_prices(prices, symbols, start, end):
    prices = prices.reindex(columns=symbols).sort_index()
    prices = prices.loc[start:end]
    if len(prices) < 2 or prices.index.has_duplicates:
        raise ValueError("Price history is empty, too short, or has duplicate dates")
    invalid = (~np.isfinite(prices)) | (prices <= 0)
    if invalid.any().any():
        missing = invalid.sum().loc[lambda counts: counts > 0].to_dict()
        raise ValueError(f"Missing or invalid daily closing prices: {missing}")
    if prices.index[0] - start > pd.Timedelta(days=7) or end - prices.index[-1] > pd.Timedelta(days=7):
        raise ValueError("Price history does not cover the requested date boundaries")
    return prices


def load_prices(config):
    symbols = sorted([*config["universe"], config["risk_free_ticker"]])
    start, end = config["data_start"], config["data_end"]
    key = sha256(f"yf-adjusted-close-v1|{start.date()}|{end.date()}|{symbols}".encode()).hexdigest()[:16]
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"prices_{key}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            fetched_at = datetime.fromisoformat(handle.readline().strip()).isoformat()
            prices = pd.read_csv(handle, index_col="Date", parse_dates=["Date"])
        return check_prices(prices, symbols, start, end), fetched_at

    with TemporaryDirectory(dir=CACHE) as session_cache:
        yf.set_tz_cache_location(session_cache)
        raw = yf.download(
            symbols, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True, actions=False, progress=False, timeout=12, threads=6,
        )
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prices = check_prices(raw["Close"], symbols, start, end)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as handle:
        temp_path = Path(handle.name)
        try:
            handle.write(f"{fetched_at}\n")
            prices.to_csv(handle, index_label="Date")
            handle.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def minimum_variance(scenarios, previous, cap, constraints):
    covariance = np.cov(scenarios * 100.0, rowvar=False)
    previous_covariance = covariance @ previous
    n_assets = len(previous)

    # Split target changes into purchases and sales measured in units of the cap.
    def objective(changes):
        direction = changes[:n_assets] - changes[n_assets:]
        curvature = cap * (covariance @ direction)
        value = direction @ (2 * previous_covariance + curvature)
        gradient = 2 * (previous_covariance + curvature)
        return value, np.concatenate((gradient, -gradient))

    bounds = [(0, upper) for upper in np.minimum(1.0, np.r_[1 - previous, previous] / cap)]
    result = minimize(
        objective, np.zeros(2 * n_assets), jac=True, method="SLSQP",
        bounds=bounds, constraints=constraints, options={"ftol": 1e-12, "maxiter": 150},
    )
    if not result.success or not np.isfinite(result.x).all():
        return previous, False

    target = previous + cap * (result.x[:n_assets] - result.x[n_assets:])
    if abs(target.sum() - 1) > 1e-8 or target.min() < -1e-8 or target.max() > 1 + 1e-8:
        return previous, False
    target = np.clip(target, 0, 1)
    target /= target.sum()
    turnover = 0.5 * np.abs(target - previous).sum()
    if turnover > cap + 1e-8:
        return previous, False
    if turnover > cap:
        target = previous + (cap / turnover) * (target - previous)
    return target, True


def drift(weights, returns):
    positions = weights * (1 + returns)
    gross_growth = float(positions.sum())
    return positions / gross_growth, gross_growth


def simulate(prices, config):
    returns = prices.pct_change(fill_method=None).iloc[1:]
    evaluation_start = returns.index.searchsorted(config["backtest_start"])
    index = returns.index[evaluation_start:]
    window, lag = config["lookback_sessions"], config["execution_lag_sessions"]
    if evaluation_start - lag + 1 < window:
        raise ValueError("Insufficient price history for the configured rolling window and fill lag")
    if len(index[index <= config["is_end"]]) < 2 or len(index[index >= config["oos_start"]]) < 2:
        raise ValueError("IS and OOS both need at least two observations")

    asset_returns = returns[config["universe"]].to_numpy()
    n_assets = asset_returns.shape[1]
    equal = np.full(n_assets, 1 / n_assets)
    target = holdings = benchmark_holdings = equal
    cap = config["target_turnover_cap"]
    cost_rate = (config["cost_bps"] + config["slippage_bps"]) / 10_000
    constraints = [
        LinearConstraint(np.r_[np.ones(n_assets), -np.ones(n_assets)], 0, 0),
        LinearConstraint(np.ones(2 * n_assets), 0, 2),
    ]
    strategy, benchmark = [], []
    solver_fallbacks = 0
    for row in range(evaluation_start, len(returns)):
        decision = row - lag
        scenarios = asset_returns[decision - window + 1:decision + 1]
        target, success = minimum_variance(scenarios, target, cap, constraints)
        solver_fallbacks += int(not success)

        if row == evaluation_start:
            # Both portfolios fund their initial positions at this close from cash.
            pre_trade, gross_growth = np.zeros(n_assets), 1.0
            turnover = 1.0
            benchmark.append((-cost_rate, 1.0, n_assets))
        else:
            # Today's close-to-close move accrues to yesterday's holdings.
            pre_trade, gross_growth = drift(holdings, asset_returns[row])
            turnover = 0.5 * np.abs(target - pre_trade).sum()
            benchmark_holdings, benchmark_growth = drift(benchmark_holdings, asset_returns[row])
            benchmark.append((benchmark_growth - 1, 0.0, 0))
        n_trades = np.count_nonzero(np.abs(target - pre_trade) > TRADE_TOLERANCE)
        net_return = gross_growth * (1 - cost_rate * turnover) - 1
        strategy.append((net_return, turnover, n_trades))
        holdings = target

    columns = ["net_return", "turnover", "n_trades"]
    risk_free = returns.loc[index, config["risk_free_ticker"]].copy()
    risk_free.iloc[0] = 0.0  # The evaluation starts at the initial execution close.
    return (pd.DataFrame(strategy, index=index, columns=columns),
            pd.DataFrame(benchmark, index=index, columns=columns), risk_free, solver_fallbacks)


def metrics(period, risk_free, sessions_per_year):
    net = period["net_return"].to_numpy()
    excess = net - risk_free.loc[period.index].to_numpy()
    nav = np.r_[1.0, np.cumprod(1 + net)]
    years = len(net) / sessions_per_year
    vol = net.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    root_sessions = np.sqrt(sessions_per_year)
    return {
        "cagr": float(nav[-1] ** (1 / years) - 1),
        "ann_vol": float(vol * root_sessions),
        "sharpe": float(net.mean() * root_sessions / vol) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() * root_sessions / excess_vol) if excess_vol > 0 else None,
        "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1).min()),
        "total_return": float(nav[-1] - 1),
        "turnover": float(period["turnover"].sum() / years),
        "n_trades": int(period["n_trades"].sum()),
    }


def main():
    config = load_config()
    prices, fetched_at = load_prices(config)
    with threadpool_limits(limits=1, user_api="blas"):
        strategy, benchmark, risk_free, solver_fallbacks = simulate(prices, config)
    is_sample = strategy.loc[:config["is_end"]]
    oos_sample = strategy.loc[config["oos_start"]:]
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": str(strategy.index[0].date()), "end": str(strategy.index[-1].date())},
    }
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return adjusted close")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"SOLVER_FALLBACKS: {solver_fallbacks}")
    for key, period in (("is", is_sample), ("oos", oos_sample), ("full", strategy), ("benchmark", benchmark)):
        result[key] = metrics(period, risk_free, config["sessions_per_year"])
        label = "BENCHMARK (equal-weight buy and hold)" if key == "benchmark" else key.upper()
        print(f"{label} {period.index[0].date()} to {period.index[-1].date()}: {json.dumps(result[key])}")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()