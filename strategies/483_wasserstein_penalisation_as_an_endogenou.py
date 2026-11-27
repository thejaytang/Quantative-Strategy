"""483_wasserstein_penalisation_as_an_endogenou
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/483_wasserstein_penalisation_as_an_endogenou.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit cc50485fb68601d5f9b01f48262e3a778be06982 on 2026-11-27 09:02 Europe/Oslo, exit 0, output sha256 1052a65cbd2fd77c02a45f51010578e908e819b2b2853f53b99c9cf04a5fb0b3
Data source: yfinance 公开日频 ETF 收盘价。投资池为 XLB、XLE、XLF、XLI、XLK、XLP、XLU、XLV、XLY。BIL 的同期日收益仅用于超额夏普计算。该次日志记录的取数时间为 2026-11-27T07:09:52+00:00。依据 config.yaml:2-3、strategy.py:62-90、strategy.py:183-197、strategy.py:205-225、README.md:15、logs/run_20261127_081135.log:1。
Data fetched: 2026-11-27T08:02:54+00:00
Sample period: 预热数据从 2009-01-01 起请求。配置评估区间和该次日志记录的实际评估区间均为 2010-02-02 至 2025-12-31（含首尾）。依据 config.yaml:4-7、strategy.py:62-64、strategy.py:81-85、strategy.py:211-221、logs/run_20261127_081135.log:8。   data_range (from output): {'start': '2010-02-02', 'end': '2025-12-31'}
IS/OOS split: 样本内为 2010-02-02 至 2018-12-31，样本外为 2019-01-01 至 2025-12-31，均按实际交易日计入。代码先连续模拟，再按日期拆分指标。两段共用固定超参数和滚动估计规则，分界处延续组合权重。依据 config.yaml:5-11、strategy.py:134-151、strategy.py:205-224、README.md:11、README.md:16。
Costs and slippage: 交易成本 5 bps，滑点以固定费率 5 bps 建模。两者合计 10 bps，按每次调仓实际买入和卖出金额之和扣减组合净值。费用从组合内支付，首次建仓也计费。月度等权基准采用相同费率。依据 config.yaml:12-13、strategy.py:157-178、strategy.py:213-216、README.md:22。
Signal lag: 每月首个交易日收盘 t，用截至该收盘的最近 252 个交易日日收益生成目标权重，默认在下一交易日收盘 t+1 成交。t 至 t+1 的收盘到收盘收益由原持仓承担，新权重从 t+1 至 t+2 的区间开始承担收益。依据 config.yaml:8-9、strategy.py:134-151、strategy.py:157-178、README.md:21。
Price adjustment: yfinance auto_adjust total-return close
Survivorship bias: 使用固定的九只行业 ETF。README 说明这些基金在样本开始前已上市，编写时仍存续。固定名单避免把后续新设基金放回早期样本，仍有事后筛选及存续偏差，可能遗漏同期已清盘的基金。依据 config.yaml:2、strategy.py:62-64、strategy.py:208-211、README.md:17。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.10917524105703413, ann_vol=0.13258073188832425, sharpe=0.8481261163177968, sharpe_excess=0.8291732669915215, max_drawdown=-0.1735528850824305, total_return=1.516029740286044, turnover=1.2887418280836356, n_trades=914
    oos: cagr=0.12467397941997582, ann_vol=0.1737421050383139, sharpe=0.7635974925694565, sharpe_excess=0.6179057865631654, max_drawdown=-0.33023014069759293, total_return=1.2718339475300948, turnover=1.5621117220335066, n_trades=707
    full: cagr=0.11596139012990725, ann_vol=0.15203343682477102, sharpe=0.7980796208659574, sharpe_excess=0.7157225107616875, max_drawdown=-0.33023014069759304, total_return=4.716001776977165, turnover=1.4089044188308317, n_trades=1621
    benchmark: cagr=0.12669782576162292, ann_vol=0.16566777441341313, sharpe=0.8033555499888284, sharpe_excess=0.7277817673561406, max_drawdown=-0.3672401500748913, total_return=5.655227653054552, turnover=0.3401036116282401, n_trades=1719
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "483_ wasserstein_penalisation_as_an_endogenou"\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nrisk_free_ticker: BIL\ndata_start: 2009-01-01\nstart: 2010-02-02\nis_end: 2018-12-31\nend: 2025-12-31\nlookback_sessions: 252\nexecution_lag_sessions: 1\nrisk_aversion: 10\nwasserstein_radius: 0.00001\ncost_bps: 5\nslippage_bps: 5\n'
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
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, minimize


ROOT = Path(__file__).resolve().parent
TRADING_DAYS = 252
PRICE_ADJUSTMENT = "yfinance auto_adjust total-return close"


def load_config():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    for key in ("data_start", "start", "is_end", "end"):
        config[key] = pd.Timestamp(config[key]).normalize()
    if not config["data_start"] < config["start"] <= config["is_end"] < config["end"]:
        raise ValueError("Expected data_start < start <= is_end < end")
    universe = config["universe"]
    if len(set(universe)) != len(universe) or len(universe) < 2:
        raise ValueError("Universe must contain at least two distinct symbols")
    if config["risk_free_ticker"] != "BIL" or "BIL" in universe:
        raise ValueError("Use BIL as the separate excess-return reference")
    for key, minimum in (("lookback_sessions", 2), ("execution_lag_sessions", 1)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer of at least {minimum}")
    for key in ("risk_aversion", "wasserstein_radius", "cost_bps", "slippage_bps"):
        value = float(config[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        config[key] = value
    if config["risk_aversion"] == 0 or config["wasserstein_radius"] == 0:
        raise ValueError("Risk aversion and Wasserstein radius must be positive")
    if config["cost_bps"] + config["slippage_bps"] >= 10_000:
        raise ValueError("Combined cost and slippage must be below 10,000 bps")
    return config


def check_prices(prices, symbols, start, end):
    prices = prices.reindex(columns=symbols).sort_index().loc[start:end]
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("Expected a nonempty price panel with unique dates")
    if prices.index[0] - start > pd.Timedelta(days=7):
        raise ValueError("Price panel does not cover the requested warm-up start")
    if end - prices.index[-1] > pd.Timedelta(days=7):
        raise ValueError("Price panel does not cover the requested end")
    invalid = ~np.isfinite(prices) | (prices <= 0)
    if invalid.to_numpy().any():
        symbols_with_gaps = ", ".join(invalid.columns[invalid.any(axis=0)])
        raise ValueError(f"Missing or invalid adjusted closes for {symbols_with_gaps}")
    return prices


def load_prices(config):
    symbols = [*config["universe"], config["risk_free_ticker"]]
    start, end = config["data_start"], config["end"]
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    identity = json.dumps([PRICE_ADJUSTMENT, sorted(symbols), str(start.date()), str(end.date())])
    cache_key = hashlib.sha256(identity.encode()).hexdigest()[:16]
    cache_file = cache_dir / f"yfinance-{cache_key}.csv"
    if cache_file.exists():
        with cache_file.open(encoding="utf-8") as source:
            label, fetched_at = source.readline().rstrip().split(": ", 1)
            if label != "DATA_FETCHED_AT" or datetime.fromisoformat(fetched_at).tzinfo is None:
                raise ValueError(f"Invalid fetch timestamp in {cache_file.name}")
            prices = pd.read_csv(source, index_col=0, parse_dates=True)
        return check_prices(prices, symbols, start, end), fetched_at

    # Keep the library's transient cache within this strategy folder.
    with TemporaryDirectory(dir=cache_dir) as library_cache:
        yf.set_tz_cache_location(library_cache)
        raw = yf.download(
            symbols, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=True, keepna=True, ignore_tz=True,
            group_by="column", multi_level_index=True, progress=False, timeout=12,
        )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance did not return prices; see download diagnostics")
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    prices = check_prices(raw["Close"], symbols, start, end)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_dir, delete=False) as output:
        temporary_path = Path(output.name)
        try:
            output.write(f"DATA_FETCHED_AT: {fetched_at}\n")
            prices.to_csv(output, index_label="date", float_format="%.17g")
            output.close()
            temporary_path.replace(cache_file)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def penalised_weights(window, risk_aversion, radius):
    mean = window.mean(axis=0)
    risk_matrix = risk_aversion * np.cov(window, rowvar=False, ddof=1)
    penalty = math.sqrt(radius)
    equal_weights = np.full(len(mean), 1 / len(mean))

    # Express the daily objective in basis points for the numerical solver.
    def objective(weights):
        risk = risk_matrix @ weights
        norm = np.linalg.norm(weights)
        value = weights @ (risk - mean) + penalty * norm
        gradient = 2 * risk - mean + penalty * weights / norm
        return 10_000 * value, 10_000 * gradient

    solution = minimize(
        objective, equal_weights, jac=True, method="SLSQP",
        bounds=[(0.0, 1.0)] * len(mean),
        constraints={
            "type": "eq", "fun": lambda weights: weights.sum() - 1,
            "jac": lambda weights: np.ones_like(weights),
        },
        options={"ftol": 1e-10, "maxiter": 200},
    )
    weights = solution.x
    if (not solution.success or not np.isfinite(weights).all()
            or abs(weights.sum() - 1) > 1e-8 or weights.min() < -1e-8):
        raise RuntimeError(f"Portfolio optimisation failed: {solution.message}")
    weights = np.maximum(weights, 0)
    return weights / weights.sum()


def monthly_targets(asset_returns, config):
    dates = asset_returns.index
    months = dates.to_period("M")
    signal_rows = np.flatnonzero(months[1:] != months[:-1]) + 1
    lookback = config["lookback_sessions"]
    lag = config["execution_lag_sessions"]
    returns = asset_returns.to_numpy()
    targets = {}
    for signal_row in signal_rows:
        execution_row = signal_row + lag
        if execution_row >= len(dates) or dates[execution_row] < config["start"]:
            continue
        if signal_row < lookback:
            raise ValueError(f"Insufficient history for signal dated {dates[signal_row].date()}")
        window = returns[signal_row - lookback + 1:signal_row + 1]
        targets[dates[execution_row]] = penalised_weights(
            window, config["risk_aversion"], config["wasserstein_radius"],
        )
    if not targets:
        raise ValueError("No monthly rebalances fall within the requested sample")
    return targets


def simulate(asset_returns, targets, fee_rate):
    # Zero risky weights represent cash before the initial close execution.
    weights = np.zeros(asset_returns.shape[1])
    daily_results = []
    for date, returns in zip(asset_returns.index, asset_returns.to_numpy(), strict=True):
        gross_factor = 1 + weights @ returns
        weights = weights * (1 + returns) / gross_factor
        turnover = 0.0
        n_trades = 0
        target = targets.get(date)
        net_fraction = 1.0
        if target is not None:
            # Fees depend on actual dollars bought and sold after funding the fees.
            def funding_gap(fraction):
                return fraction + fee_rate * np.abs(fraction * target - weights).sum() - 1

            net_fraction = brentq(funding_gap, 0.0, 1.0, xtol=1e-14)
            changes = np.abs(net_fraction * target - weights)
            turnover = changes.sum()
            n_trades = np.count_nonzero(changes > 1e-8)
            weights = target
        daily_results.append((gross_factor * net_fraction - 1, turnover, n_trades))
    return pd.DataFrame(daily_results, index=asset_returns.index,
                        columns=["net_return", "turnover", "n_trades"])


def metrics(daily_results, risk_free_returns):
    returns = daily_results["net_return"].to_numpy()
    if len(returns) < 2:
        raise ValueError("Each evaluation period needs at least two sessions")
    excess_returns = returns - risk_free_returns.loc[daily_results.index].to_numpy()
    growth = np.r_[1.0, np.cumprod(1 + returns)]
    scale = math.sqrt(TRADING_DAYS)
    volatility = returns.std(ddof=1)
    excess_volatility = excess_returns.std(ddof=1)
    return {
        "cagr": float(growth[-1] ** (TRADING_DAYS / len(returns)) - 1),
        "ann_vol": float(volatility * scale),
        "sharpe": float(returns.mean() * scale / volatility) if volatility > 0 else None,
        "sharpe_excess": (float(excess_returns.mean() * scale / excess_volatility)
                          if excess_volatility > 0 else None),
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * TRADING_DAYS),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def main():
    config = load_config()
    prices, fetched_at = load_prices(config)
    all_returns = prices.pct_change(fill_method=None)
    asset_returns = all_returns[config["universe"]]
    risk_free_returns = all_returns[config["risk_free_ticker"]]
    targets = monthly_targets(asset_returns, config)
    evaluation_returns = asset_returns.loc[min(targets):]
    fee_rate = (config["cost_bps"] + config["slippage_bps"]) / 10_000
    daily_results = simulate(evaluation_returns, targets, fee_rate)
    equal_weights = np.full(len(config["universe"]), 1 / len(config["universe"]))
    benchmark_results = simulate(evaluation_returns, dict.fromkeys(targets, equal_weights), fee_rate)
    is_mask = daily_results.index <= config["is_end"]
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": daily_results.index[0].date().isoformat(),
                       "end": daily_results.index[-1].date().isoformat()},
        "is": metrics(daily_results.loc[is_mask], risk_free_returns),
        "oos": metrics(daily_results.loc[~is_mask], risk_free_returns),
        "full": metrics(daily_results, risk_free_returns),
        "benchmark": metrics(benchmark_results, risk_free_returns),
    }
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print("PARAM_COMBOS_TRIED: 1")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(results[period], allow_nan=False)}")
    print(f"BENCHMARK (monthly equal weight): {json.dumps(results['benchmark'], allow_nan=False)}")
    print(f"RESULTS_JSON: {json.dumps(results, allow_nan=False)}")


if __name__ == "__main__":
    main()