"""434_dynamic_scenario_based_portfolio_optimiz
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/434_dynamic_scenario_based_portfolio_optimiz.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit da9754ee51e8d74f175ab50e14ef28c47c764ad1 on 2026-11-27 12:04 Europe/Oslo, exit 0, output sha256 7837842db62a5c103ed4550f2bdca39a291ee078e303d8edc02a36e01f0d6f16
Data source: Yahoo Finance 日频 ETF 收盘价，经 yfinance 获取。策略使用 VT、VTI、ACWI、ACWX、SUSA、ROUS、IYG、BND、FPE、BYLD、NTSX、GAL、MDIV、AOM、AOR、INKM、HNDL、ALTY、IYLD、CVY、PDBC、QAI、HDG、MNA、RAAX、RLY、SRET、YYY、GAA、GMOM。另取 BIL 的收益作为事后统计的无风险代理（config.yaml:2-6，README.md:16、20，strategy.py:60-80、143-144、178-186）。
Data fetched: 2026-11-27T11:04:30+00:00
Sample period: 配置的价格样本为 2018-12-19 至 2026-09-30（两端包含），回测评估前的历史价格用于预热。日志记录的完整评估区间为 2022-12-21 至 2026-09-30（config.yaml:7-9、12，README.md:22，strategy.py:46-49、74-80、133-139，logs/run_20261127_073038.log:7）。   data_range (from output): {'start': '2022-12-21', 'end': '2026-09-30'}
IS/OOS split: 样本内为 2022-12-21 至 2024-12-31。配置的样本外起始日为 2025-01-01，日志中实际样本外区间为 2025-01-02 至 2026-09-30。代码先连续模拟，再按日期切分；样本外沿用此前的持仓和目标状态，协方差继续滚动更新（config.yaml:9-13，README.md:22，strategy.py:93-94、155-175、204-210，logs/run_20261127_073038.log:5-6）。
Costs and slippage: 交易成本 10 bps，另建模固定滑点 5 bps，合计 15 bps 乘以单边换手率，从成交日按原持仓计价后的组合价值扣除。常规调仓以目标权重与收盘价格变动后的实际权重之差的半 L1 距离计换手率。策略和等权买入持有基准首次用现金建仓均按 100% 换手扣费，基准此后无调仓费用。固定模型未涵盖市场冲击或变化的买卖价差（config.yaml:16-17，strategy.py:146-148、161-175，README.md:26-28、32）。
Signal lag: 用截至 t 日收盘的 1,008 个交易日 ETF 收益计算目标权重，在 t+1 交易日收盘模拟成交。t 至 t+1 的收盘到收盘收益归原持仓，回测首次成交前持有现金；新权重从成交后的下一段收盘到收盘区间起计收益（config.yaml:12-13，strategy.py:133-139、155-175，README.md:26-28）。
Price adjustment: yfinance auto_adjust total-return adjusted close
Survivorship bias: 使用 README 所述的论文附录 A 固定 ETF 名单。缺少历史时点成分记录和已退市基金样本，存在幸存者偏差与资产选择偏差（README.md:5、32，config.yaml:2-5，strategy.py:60-63）。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.09509740675323997, ann_vol=0.07486259002847453, sharpe=1.2510479936156924, sharpe_excess=0.5876739925039375, max_drawdown=-0.06948767556996316, total_return=0.20140184270819472, turnover=1.264371798247597, n_trades=15008
    oos: cagr=0.11443704092387996, ann_vol=0.07903864738488306, sharpe=1.4105278187050023, sharpe_excess=0.919999377930032, max_drawdown=-0.08708509758469718, total_return=0.2067026847433573, turnover=0.7873321907863746, n_trades=12090
    full: cagr=0.10398918931818768, ann_vol=0.07678096762415192, sharpe=1.3270599913964574, sharpe_excess=0.7458868149787758, max_drawdown=-0.08708509758469729, total_return=0.4497348290515908, turnover=1.0440057216508167, n_trades=27098
    benchmark: cagr=0.11471918998783637, ann_vol=0.08700226145074544, sharpe=1.2919839023865243, sharpe_excess=0.779069378765289, max_drawdown=-0.1039106276279329, total_return=0.503341705027943, turnover=0.2555476531977772, n_trades=30
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 434_dynamic_scenario_based_portfolio_optimiz\nuniverse:\n  [VT, VTI, ACWI, ACWX, SUSA, ROUS, IYG, BND, FPE, BYLD, NTSX, GAL, MDIV,\n   AOM, AOR, INKM, HNDL, ALTY, IYLD, CVY, PDBC, QAI, HDG, MNA, RAAX, RLY,\n   SRET, YYY, GAA, GMOM]\nrisk_free_ticker: BIL\ndata_start: '2018-12-19'\ndata_end: '2026-09-30'\nbacktest_start: '2022-12-21'\nis_end: '2024-12-31'\noos_start: '2025-01-02'\nlookback_sessions: 1008\nexecution_lag_sessions: 1\nsessions_per_year: 252\ntarget_turnover_cap: 0.00013227513227513228  # 1 / (30 × 252)\n# Fraction of prior-close portfolio value kept outside the ETF orders when sizing.\ncash_buffer_fraction: 0.05\n# Each rate applies to gross buy and sell fill notional.\ncost_bps: 10.0\nslippage_bps: 5.0\n"
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
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)
from scipy.optimize import LinearConstraint, minimize
from threadpoolctl import threadpool_limits
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
STARTING_CASH = 1_000_000.0
TRADE_TOLERANCE = 1e-8


class NyseHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


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
        raise ValueError("The risk-free proxy must be BIL")
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
    if not 0 < config["cash_buffer_fraction"] < 1:
        raise ValueError("cash_buffer_fraction must be in (0, 1)")
    return config


def nyse_sessions(start, end):
    if start.year < 2018 or end.year > 2026:
        raise ValueError("NYSE calendar coverage is limited to 2018 through 2026")
    holidays = NyseHolidayCalendar().holidays(start, end)
    mourning_closures = pd.to_datetime(["2018-12-05", "2025-01-09"])
    return pd.bdate_range(start, end).difference(holidays.union(mourning_closures))


def check_prices(prices, symbols, start, end):
    prices = prices.reindex(columns=symbols).sort_index()
    prices = prices.loc[start:end]
    if len(prices) < 2 or prices.index.has_duplicates:
        raise ValueError("Price history is empty, too short, or has duplicate dates")
    expected = nyse_sessions(start, end)
    missing_dates = expected.difference(prices.index)
    extra_dates = prices.index.difference(expected)
    if len(missing_dates) or len(extra_dates):
        raise ValueError(f"NYSE session mismatch. Missing: {missing_dates[:5].date.tolist()}. "
                         f"Unexpected: {extra_dates[:5].date.tolist()}.")
    invalid = (~np.isfinite(prices)) | (prices <= 0)
    if invalid.any().any():
        missing = invalid.sum().loc[lambda counts: counts > 0].to_dict()
        raise ValueError(f"Missing or invalid daily closing prices: {missing}")
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


def size_trades(positions, target, reference_prices, previous_value, cash_buffer_fraction):
    target_positions = target * (previous_value * (1 - cash_buffer_fraction)) / reference_prices
    trade_units = target_positions - positions
    meaningful = np.abs(trade_units) * reference_prices > TRADE_TOLERANCE * previous_value
    return np.where(meaningful, trade_units, 0.0)


def simulate_fills(asset_prices, index, targets, config):
    positions = np.zeros(asset_prices.shape[1])
    cash = STARTING_CASH
    cost_rate = (config["cost_bps"] + config["slippage_bps"]) / 10_000
    daily_results = []
    for row, date in enumerate(index):
        reference_prices = asset_prices[row]
        previous_value = float(positions @ reference_prices) + cash
        target = targets.get(date)
        trade_units = np.zeros_like(positions) if target is None else size_trades(
            positions, target, reference_prices, previous_value, config["cash_buffer_fraction"],
        )

        # The fill close marks and settles the quantities already fixed above.
        fill_prices = asset_prices[row + 1]
        pre_fill_value = float(positions @ fill_prices) + cash
        signed_notionals = trade_units * fill_prices
        gross_notional = float(np.abs(signed_notionals).sum())
        cash -= float(signed_notionals.sum()) + gross_notional * cost_rate
        if cash < 0:
            raise ValueError(f"Cash would be negative on {date.date()} after filling fixed quantities: {cash:.2f}")
        positions += trade_units
        closing_value = float(positions @ fill_prices) + cash
        daily_results.append((closing_value / previous_value - 1,
                              gross_notional / pre_fill_value, np.count_nonzero(trade_units)))

    return pd.DataFrame(daily_results, index=index, columns=["net_return", "turnover", "n_trades"])


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
    target = equal
    cap = config["target_turnover_cap"]
    constraints = [
        LinearConstraint(np.r_[np.ones(n_assets), -np.ones(n_assets)], 0, 0),
        LinearConstraint(np.ones(2 * n_assets), 0, 2),
    ]
    targets = {}
    solver_fallbacks = 0
    for row in range(evaluation_start, len(returns)):
        decision = row - lag
        scenarios = asset_returns[decision - window + 1:decision + 1]
        target, success = minimum_variance(scenarios, target, cap, constraints)
        solver_fallbacks += int(not success)
        targets[returns.index[row]] = target

    # Each return row corresponds to the price row after it. Include the previous close for sizing.
    asset_prices = prices[config["universe"]].to_numpy()[evaluation_start:]
    strategy = simulate_fills(asset_prices, index, targets, config)
    benchmark = simulate_fills(asset_prices, index, {index[0]: equal}, config)
    risk_free = returns.loc[index, config["risk_free_ticker"]].copy()
    risk_free.iloc[0] = 0.0  # The evaluation starts at the initial execution close.
    return strategy, benchmark, risk_free, solver_fallbacks


def metrics(period, risk_free, sessions_per_year):
    net = period["net_return"].to_numpy()
    excess = net - risk_free.loc[period.index].to_numpy()
    growth = np.r_[1.0, np.cumprod(1 + net)]
    years = len(net) / sessions_per_year
    vol = net.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    root_sessions = np.sqrt(sessions_per_year)
    return {
        "cagr": float(growth[-1] ** (1 / years) - 1),
        "ann_vol": float(vol * root_sessions),
        "sharpe": float(net.mean() * root_sessions / vol) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() * root_sessions / excess_vol) if excess_vol > 0 else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(period["turnover"].sum() / years),
        "n_trades": int(period["n_trades"].sum()),
    }


def main():
    config = load_config()
    prices, fetched_at = load_prices(config)
    with threadpool_limits(limits=1, user_api="blas"):
        strategy, benchmark, risk_free, solver_fallbacks = simulate(prices, config)
    is_end, oos_start = config["is_end"], config["oos_start"]
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": str(strategy.index[0].date()), "end": str(strategy.index[-1].date())},
    }
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return adjusted close")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"SOLVER_FALLBACKS: {solver_fallbacks}")
    periods = (
        ("is", "IS", strategy.loc[:is_end]),
        ("oos", "OOS", strategy.loc[oos_start:]),
        ("full", "FULL", strategy),
        ("benchmark", "BENCHMARK (equal-weight buy and hold)", benchmark),
        (None, "BENCHMARK IS", benchmark.loc[:is_end]),
        (None, "BENCHMARK OOS", benchmark.loc[oos_start:]),
    )
    for key, label, period in periods:
        stats = metrics(period, risk_free, config["sessions_per_year"])
        if key is not None:
            result[key] = stats
        print(f"{label} {period.index[0].date()} to {period.index[-1].date()}: {json.dumps(stats)}")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
