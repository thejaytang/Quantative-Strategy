"""534_portfolio_diversification_and_concentrat
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/534_portfolio_diversification_and_concentrat.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 69a306c0280e31031e1c1674f053aa22730544ac on 2026-11-27 09:47 Europe/Oslo, exit 0, output sha256 cdc5ec100dae0b2172dbf43e7d50a845c6fef749a0d3406d7a8ea301ad971482
Data source: Yahoo Finance 公开 ETF 日线，经 yfinance 下载。配置的投资标的为 SPY、TLT、GLD、VNQ、EFA，另取 BIL 日收益作为超额收益计算的无风险代理。下载数据可从 data_cache/closes_<hash>.json 复用。（config.yaml:2-3，strategy.py:45-73，strategy.py:187-191，strategy.py:233-242，README.md:13，README.md:17）
Data fetched: 2026-11-27T08:47:34.138659+00:00
Sample period: 请求的行情区间为 2007-06-01 至 2025-12-31，评价前行情用于预热。日志记录的实际评价区间为 2010-01-04 至 2025-12-31。（config.yaml:4-7，strategy.py:55-59，strategy.py:239-243，README.md:14，logs/run_20261127_091719.log:16）   data_range (from output): ['2010-01-04', '2025-12-31']
IS/OOS split: 样本内为 2010-01-04 至 2017-12-31，样本外为 2018-01-01 至 2025-12-31，按实际行情交易日切分。默认策略先连续模拟，再按日期分段计量，样本外起点继承原有持仓。7 组敏感性测试仅运行样本内。README 记载该分割是论文发表后设计的历史评价。（config.yaml:5-7，strategy.py:247-265，README.md:14，README.md:21）
Costs and slippage: 默认按每只 ETF 的每笔模拟买入或卖出金额收取成本 5 bps 和滑点 5 bps，共 10 bps。滑点按固定比例计入，与成本合并从组合净值扣除。每次调整先将旧权重按当日收益更新，再求解扣除摩擦后可用于目标配置的资金，费基为买卖金额绝对值之和，含首次建仓。样本内高摩擦情形将两项分别调为 10 bps，共 20 bps。（config.yaml:14-15，config.yaml:21-23，strategy.py:40-42，strategy.py:162-184，strategy.py:211-218，strategy.py:247-252，README.md:16）
Signal lag: 用截至每月最后一条交易日收盘的滚动日收益计算信号 t，默认在下一条交易日收盘 t+1 模拟成交。t+1 当天的收盘到收盘收益使用旧持仓，新持仓从成交后的下一段收盘到收盘区间开始计收益。（config.yaml:8，strategy.py:141-156，strategy.py:162-183，strategy.py:230-232，README.md:15）
Price adjustment: yfinance auto_adjust=True, split- and dividend-adjusted close total return
Survivorship bias: 代码使用配置中的固定 ETF 名单。README 明确记载，资产池事后选自目前仍存续且有长历史的 ETF，存在幸存者偏差和标的选择偏差。（config.yaml:2，strategy.py:233-242，README.md:23）
Parameter combinations tried: 8
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.11105023786021362, ann_vol=0.07688780529820026, sharpe=1.4083310227791432, sharpe_excess=1.4018860014024064, max_drawdown=-0.0815415886422276, total_return=1.3191301643794375, turnover=5.979214733564817, n_trades=262
    oos: cagr=0.06653612477153037, ann_vol=0.11229653380617459, sharpe=0.6298484956127925, sharpe_excess=0.4136739061216934, max_drawdown=-0.3231870393806132, total_return=0.6720508838798762, turnover=12.655513355456325, n_trades=319
    full: cagr=0.08857673016005974, ann_vol=0.09622138704022093, sharpe=0.9303126607942586, sharpe_excess=0.8016755222966933, max_drawdown=-0.32318703938061166, total_return=2.8777036411831283, turnover=18.63472808902114, n_trades=581
    benchmark: cagr=0.08748840317992723, ann_vol=0.10862820357639907, sharpe=0.8266208571591914, sharpe_excess=0.7121995493118695, max_drawdown=-0.2535979941195221, total_return=2.816258747286204, turnover=6.097575940197556, n_trades=960
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 534_portfolio_diversification_and_concentrat\nuniverse: [SPY, TLT, GLD, VNQ, EFA]\nrisk_free_proxy: BIL\ndata_start: 2007-06-01\nsample_start: 2010-01-04\nis_end: 2017-12-31\nsample_end: 2025-12-31\nexecution_lag_sessions: 1\n\nmodel:\n  lookback_sessions: 504\n  es_level: 0.95\n  worst_case_weight: 0.50\n  cost_bps: 5.0\n  slippage_bps: 5.0\n\nrobustness:\n  lookback_sessions: [420, 588]\n  es_level: [0.925, 0.975]\n  worst_case_weight: [0.25, 0.75]\n  higher_friction:\n    cost_bps: 10.0\n    slippage_bps: 10.0\n'
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
"""Monthly long-only allocation using the weighted ES objective in Liu and Liu (2026)."""

import hashlib
import json
import signal
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, linprog
from scipy.sparse import csr_matrix, eye, hstack


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
PRICE_ADJUSTMENT = "yfinance auto_adjust=True, split- and dividend-adjusted close total return"


@dataclass(frozen=True)
class Parameters:
    lookback_sessions: int
    es_level: float
    worst_case_weight: float
    cost_bps: float
    slippage_bps: float

    def __post_init__(self):
        if not isinstance(self.lookback_sessions, int) or self.lookback_sessions < 2:
            raise ValueError("lookback_sessions must be an integer of at least two")
        if not (0 < self.es_level < 1 and 0 <= self.worst_case_weight <= 1):
            raise ValueError("ES level must be in (0, 1) and worst-case weight in [0, 1]")
        if not (self.cost_bps >= 0 and self.slippage_bps >= 0 and self.friction < 1):
            raise ValueError("Cost and slippage must be nonnegative and sum to less than 10000 bps")

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10000


def load_closes(symbols, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    key = json.dumps(["yfinance-adjusted-close", sorted(symbols), str(start.date()), str(end.date())])
    cache_file = cache_dir / f"closes_{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"
    downloaded = not cache_file.exists()
    if downloaded:
        # Keep yfinance's transient connection metadata in the strategy directory.
        with TemporaryDirectory(dir=cache_dir) as metadata_dir:
            yf.set_tz_cache_location(metadata_dir)
            quotes = yf.download(
                symbols,
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d",
                auto_adjust=True,
                ignore_tz=True,
                progress=False,
                threads=False,
                timeout=10,
            )
        fetched_at = datetime.now(timezone.utc).isoformat()
        if quotes is None or quotes.empty:
            raise RuntimeError("yfinance returned no daily prices")
        closes = quotes["Close"]
    else:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        closes = pd.DataFrame(**payload["closes"])
        fetched_at = payload["fetched_at"]

    closes.index = pd.to_datetime(closes.index)
    closes = closes.reindex(columns=symbols).sort_index().loc[start:end].astype(float)
    if (
        closes.empty
        or not closes.index.is_unique
        or closes.index[0] > start + pd.Timedelta(days=7)
        or closes.index[-1] < end - pd.Timedelta(days=7)
        or not np.isfinite(closes.to_numpy()).all()
        or (closes <= 0).any().any()
    ):
        raise ValueError("Incomplete or invalid adjusted close history; check symbols and data dates")

    if downloaded:
        payload = {
            "fetched_at": fetched_at,
            "closes": {
                "index": closes.index.strftime("%Y-%m-%d").tolist(),
                "columns": symbols,
                "data": closes.to_numpy().tolist(),
            },
        }
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_dir, delete=False) as temp:
            temporary_path = Path(temp.name)
            try:
                json.dump(payload, temp, allow_nan=False)
                temp.flush()
                temporary_path.replace(cache_file)
            finally:
                temporary_path.unlink(missing_ok=True)
    return closes, fetched_at


def optimal_weights(returns, params):
    # Use percentage-point losses for numerical conditioning in the linear program.
    losses = -100 * returns
    n_rows, n_assets = losses.shape
    tail_size = n_rows * (1 - params.es_level)
    tail_mass = np.clip(tail_size - np.arange(n_rows), 0, 1)
    marginal_es = tail_mass @ np.sort(losses, axis=0)[::-1] / tail_size
    robust_share = params.worst_case_weight
    if robust_share == 1:
        return np.eye(n_assets)[np.argmin(marginal_es)]

    # Section 8, Eq. (19). The nominal ES is the threshold plus scaled excess losses.
    objective = np.r_[
        robust_share * marginal_es,
        1 - robust_share,
        np.full(n_rows, (1 - robust_share) / tail_size),
    ]
    loss_threshold = csr_matrix(np.column_stack((losses, -np.ones(n_rows))))
    result = linprog(
        objective,
        A_ub=hstack((loss_threshold, -eye(n_rows)), format="csr"),
        b_ub=np.zeros(n_rows),
        A_eq=csr_matrix(np.r_[np.ones(n_assets), np.zeros(n_rows + 1)][None, :]),
        b_eq=[1.0],
        bounds=[(0, 1)] * n_assets + [(None, None)] + [(0, None)] * n_rows,
        method="highs",
        options={"time_limit": 3.0},
    )
    if not result.success:
        raise RuntimeError(f"Expected-shortfall optimization failed: {result.message}")
    weights = np.maximum(result.x[:n_assets], 0)
    return weights / weights.sum()


def monthly_targets(returns, sample_start, lag, params):
    dates = returns.index
    months = dates.to_period("M")
    targets = {}
    for signal_row in np.flatnonzero(months[:-1] != months[1:]):
        fill_row = signal_row + lag
        if fill_row >= len(dates):
            break
        fill_date = dates[fill_row]
        if fill_date < sample_start:
            continue
        window_start = signal_row + 1 - params.lookback_sessions
        if window_start < 0:
            raise ValueError(f"Insufficient return history for the signal dated {dates[signal_row].date()}")
        window = returns.iloc[window_start : signal_row + 1].to_numpy()
        targets[fill_date] = optimal_weights(window, params)
    if not targets:
        raise ValueError("No scheduled rebalances in the evaluation period")
    return targets


def simulate(returns, targets, friction):
    weights = np.zeros(returns.shape[1])
    rows = []
    for date, asset_returns in zip(returns.index, returns.to_numpy(), strict=True):
        gross_factor = 1 + weights @ asset_returns
        weights = weights * (1 + asset_returns) / gross_factor
        remaining = 1.0
        turnover = 0.0
        n_trades = 0
        if date in targets:
            target = targets[date]

            def funding_gap(scale):
                return scale + friction * np.abs(scale * target - weights).sum() - 1

            # Costs are paid from the portfolio. New weights start earning returns after this close.
            remaining = brentq(funding_gap, 0.0, 1.0)
            traded = np.abs(remaining * target - weights)
            turnover = traded.sum()
            n_trades = np.count_nonzero(traded > 1e-8)
            weights = target
        rows.append((gross_factor * remaining - 1, turnover, n_trades))
    return pd.DataFrame(rows, index=returns.index, columns=["net_return", "turnover", "n_trades"])


def period_metrics(daily_results, cash_returns):
    net = daily_results["net_return"]
    if len(net) < 2:
        raise ValueError("Each evaluation period needs at least two daily returns")
    excess = net - cash_returns.loc[net.index]
    annual_factor = np.sqrt(SESSIONS_PER_YEAR)

    def sharpe(series):
        volatility = series.std()
        return float(annual_factor * series.mean() / volatility) if volatility > 0 else None

    growth = np.r_[1.0, (1 + net).cumprod().to_numpy()]
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(net)) - 1),
        "ann_vol": float(net.std() * annual_factor),
        "sharpe": sharpe(net),
        "sharpe_excess": sharpe(excess),
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1)),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def sensitivity_cases(default, robustness):
    cases = []
    for field in ("lookback_sessions", "es_level", "worst_case_weight"):
        for value in robustness[field]:
            cases.append((f"{field}={value}", replace(default, **{field: value})))
    higher = replace(default, **robustness["higher_friction"])
    cases.append((f"cost_bps={higher.cost_bps}, slippage_bps={higher.slippage_bps}", higher))
    return cases


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    default = Parameters(**config["model"])
    cases = sensitivity_cases(default, config["robustness"])
    data_start, sample_start, is_end, sample_end = (
        pd.Timestamp(config[key]) for key in ("data_start", "sample_start", "is_end", "sample_end")
    )
    if not data_start < sample_start <= is_end < sample_end:
        raise ValueError("Dates must satisfy data_start < sample_start <= is_end < sample_end")
    lag = config["execution_lag_sessions"]
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("Execution requires an integer lag of at least one trading session")
    universe = config["universe"]
    cash_symbol = config["risk_free_proxy"]
    symbols = [*universe, cash_symbol]
    if not universe or len(set(symbols)) != len(symbols) or cash_symbol != "BIL":
        raise ValueError("Use a nonempty, unique ETF universe and a separate BIL series")

    closes, fetched_at = load_closes(symbols, data_start, sample_end)
    returns = closes.pct_change(fill_method=None).iloc[1:]
    cash_returns = returns[cash_symbol]
    asset_returns = returns[universe]
    evaluation_returns = asset_returns.loc[sample_start:]
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")

    print("ROBUSTNESS (IS only)")
    is_history = asset_returns.loc[:is_end]
    for label, params in cases:
        targets = monthly_targets(is_history, sample_start, lag, params)
        daily_results = simulate(is_history.loc[sample_start:], targets, params.friction)
        print(f"  {label}: {json.dumps(period_metrics(daily_results, cash_returns), allow_nan=False)}")
    targets = monthly_targets(asset_returns, sample_start, lag, default)
    daily_results = simulate(evaluation_returns, targets, default.friction)
    equal = np.full(len(universe), 1 / len(universe))
    benchmark = simulate(evaluation_returns, dict.fromkeys(targets, equal), default.friction)
    results = {
        "strategy": config["strategy"],
        "data_range": [daily_results.index[0].date().isoformat(), daily_results.index[-1].date().isoformat()],
        "is": period_metrics(daily_results.loc[:is_end], cash_returns),
        "oos": period_metrics(daily_results.loc[daily_results.index > is_end], cash_returns),
        "full": period_metrics(daily_results, cash_returns),
        "benchmark": period_metrics(benchmark, cash_returns),
    }
    print(f"PARAM_COMBOS_TRIED: {1 + len(cases)}")
    for key, label in (("is", "IS"), ("oos", "OOS"), ("full", "FULL"), ("benchmark", "BENCHMARK (monthly equal weight, full)")):
        print(f"{label}: {json.dumps(results[key], allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    main()