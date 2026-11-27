"""432_revisiting_equity_bond_diversification_u
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/432_revisiting_equity_bond_diversification_u.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 4d91396b470c7410b1364756fad54a1bec5cce1f on 2026-11-27 07:46 Europe/Oslo, exit 0, output sha256 547603f6ea5ecf667de3fed9326e27230773d53475f459507969dd937c0f0bd9
Data source: Yahoo Finance，经 yfinance 获取美元计价的美国上市 ETF 日收盘行情。AOA、AOR、AOM、AOK、GAL、RLY 用于策略和同基金池等权基准。BIL 仅用于超额夏普的现金代理。见 config.yaml:2-3、strategy.py:37-64、strategy.py:191-206、README.md:16-20。
Data fetched: 2026-11-27T06:46:49.789948+00:00
Sample period: 行情请求区间为 2017-12-15 至 2024-12-31（含），2020-01-02 前的行情用于滚动窗口预热。运行日志所示评估区间为 2020-01-02 至 2024-12-31。见 config.yaml:5-8、strategy.py:51-55、strategy.py:191-194、README.md:21、logs/run_20261127_072244.log:11。   data_range (from output): 2020-01-02 to 2024-12-31
IS/OOS split: IS 为 2020-01-02 至 2021-12-31。OOS 为 2022-01-03 至 2024-12-31。以 2021-12-31（含）为 IS 截止日，连续模拟后按日期切分统计，跨界延续仓位和滚动窗口。见 config.yaml:6-8、strategy.py:99-118、strategy.py:191-206、README.md:22、logs/run_20261127_072244.log:5、logs/run_20261127_072244.log:8。
Costs and slippage: 每笔买卖按成交金额收取 1 bp 佣金和 2 bps 固定滑点，合计单边 3 bps，初始建仓同样计费。成交金额为各 ETF 收盘漂移后持仓与扣费后目标持仓之间的绝对差额之和。费用从调仓前净值扣除，滑点以固定费用形式建模。策略和等权基准应用相同费率。见 config.yaml:14-15、strategy.py:121-144、strategy.py:187-201、README.md:28。
Signal lag: 每日调仓目标由截至 t 日收盘的最近 504 个交易日 ETF 简单收益确定，于下一交易日 t+1 收盘模拟成交。成交日的收盘到收盘收益先按旧权重计入，新权重从 t+1 收盘后的区间开始承担价格变化。见 config.yaml:10-13、strategy.py:99-118、strategy.py:121-144、strategy.py:191-200、README.md:27。
Price adjustment: yfinance auto_adjust total-return close
Survivorship bias: 六只美国上市 ETF 从论文表 1 事后选取，回测使用固定基金池，可能遗漏已清盘或退市的基金，存在存续偏差。见 config.yaml:2、strategy.py:172-174、strategy.py:198-201、README.md:16、README.md:34。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.06800589882333763, ann_vol=0.08579811404089044, sharpe=0.8100554423215995, sharpe_excess=0.7920982130839342, max_drawdown=-0.14542600958111573, total_return=0.14093444120012721, turnover=0.9997000899730081, n_trades=1
    oos: cagr=0.004736434710668869, ann_vol=0.07741191157139196, sharpe=0.09968603811378203, sharpe_excess=-0.3889616387188324, max_drawdown=-0.18550334352022713, total_return=0.014219657313246481, turnover=0.0, n_trades=0
    full: cagr=0.029671517239883816, ann_vol=0.08087194179608696, sharpe=0.40210518871874146, sharpe_excess=0.11516751304130556, max_drawdown=-0.1893355132017548, total_return=0.1571581379708733, turnover=0.9997000899730081, n_trades=1
    benchmark: cagr=0.055970485246966906, ann_vol=0.11990197506170759, sharpe=0.5145235836084645, sharpe_excess=0.32092638815129526, max_drawdown=-0.23647829248441266, total_return=0.31241500132833866, turnover=3.6793175789725083, n_trades=7548
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "432_ revisiting_equity_bond_diversification_u"\nuniverse: [AOA, AOR, AOM, AOK, GAL, RLY]\nexcess_return_ticker: BIL\n\ndata_start: 2017-12-15\nevaluation_start: 2020-01-02\nis_end: 2021-12-31\nevaluation_end: 2024-12-31\n\nlookback_sessions: 504\ncvar_confidence: 0.95\nrebalance_every_sessions: 1\nexecution_lag_sessions: 1\ncommission_bps: 1.0\nslippage_bps: 2.0\n'
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
"""Walk-forward long-only historical CVaR research backtest."""

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy import sparse
from scipy.optimize import brentq, linprog


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252


def check_prices(prices, tickers, start, end):
    if prices.empty or not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("Price history is empty or its dates are invalid")
    prices = prices.reindex(columns=tickers)
    if prices.index[0] < start or prices.index[-1] > end:
        raise ValueError("Price history contains dates outside the request")
    if prices.index[0] - start > pd.Timedelta(days=7) or end - prices.index[-1] > pd.Timedelta(days=7):
        raise ValueError("Returned price history does not cover the requested date range")
    valid = np.isfinite(prices) & (prices > 0)
    if not valid.to_numpy().all():
        missing = ", ".join(f"{ticker} ({count})" for ticker, count in (~valid).sum().items() if count)
        raise ValueError(f"Missing or invalid ETF close observations: {missing}")
    return prices


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    request = ["yfinance-adjusted-close-v1", tickers, start.date().isoformat(), end.date().isoformat()]
    key = hashlib.sha256(json.dumps(request).encode()).hexdigest()[:16]
    path = CACHE / f"prices_{key}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as file:
            fetched_at = file.readline().removeprefix("# fetched_at=").strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(file, index_col="Date", parse_dates=["Date"])
        return check_prices(prices, tickers, start, end), fetched_at

    with tempfile.TemporaryDirectory(dir=CACHE, prefix="yfinance_") as yf_cache:
        yf.set_tz_cache_location(yf_cache)
        history = yf.download(
            tickers,
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True,
            ignore_tz=True,
            progress=False,
            threads=False,
            timeout=10,
        )
    fetched_at = datetime.now(timezone.utc).isoformat()
    if history.empty:
        raise RuntimeError("yfinance returned no ETF history")
    prices = check_prices(history["Close"], tickers, start, end)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as file:
        temporary_path = Path(file.name)
        file.write(f"# fetched_at={fetched_at}\n")
        prices.to_csv(file, index_label="Date")
    temporary_path.replace(path)
    return prices, fetched_at


def min_cvar(scenarios, confidence):
    observations, assets = scenarios.shape
    # Variables: ETF weights, the free VaR threshold, and one nonnegative
    # excess-loss variable per historical scenario.
    objective = np.r_[np.zeros(assets), 1.0, np.full(observations, 1 / ((1 - confidence) * observations))]
    loss_constraints = sparse.hstack(
        [sparse.csc_matrix(-scenarios), sparse.csc_matrix(-np.ones((observations, 1))),
         -sparse.eye(observations, format="csc")],
        format="csc",
    )
    budget = np.r_[np.ones(assets), np.zeros(observations + 1)].reshape(1, -1)
    result = linprog(
        objective,
        A_ub=loss_constraints,
        b_ub=np.zeros(observations),
        A_eq=budget,
        b_eq=[1.0],
        bounds=[(0.0, 1.0)] * assets + [(None, None)] + [(0.0, None)] * observations,
        method="highs",
    )
    if not result.success:
        return None
    weights = np.clip(result.x[:assets], 0.0, 1.0)
    return weights / weights.sum()


def target_schedule(returns, first_fill, config):
    lookback = config["lookback_sessions"]
    lag = config["execution_lag_sessions"]
    if first_fill - lag < lookback:
        raise ValueError("Start the price history earlier to allow a full window before the first fill")
    assets = returns.shape[1]
    equal_target = np.full(assets, 1.0 / assets)
    strategy_targets = {}
    benchmark_targets = {}
    cash_fallbacks = 0
    for fill in range(first_fill, len(returns), config["rebalance_every_sessions"]):
        signal = fill - lag
        scenarios = returns[signal - lookback + 1 : signal + 1]
        target = min_cvar(scenarios, config["cvar_confidence"])
        if target is None:
            target = np.zeros(assets)
            cash_fallbacks += 1
        strategy_targets[fill - first_fill] = target
        benchmark_targets[fill - first_fill] = equal_target
    return strategy_targets, benchmark_targets, cash_fallbacks


def simulate(returns, targets, cost_rate):
    # ETF weights at the prior close, with the residual earning zero as cash.
    weights = np.zeros(returns.shape[1])
    daily = []
    for day, asset_returns in enumerate(returns.to_numpy()):
        marked = weights * (1 + asset_returns)
        gross = 1 + weights @ asset_returns
        weights = marked / gross
        fee = turnover = 0.0
        trades = 0
        if day in targets:
            target = targets[day]
            # Fees are funded from pre-fill NAV. Post-fee holdings are a
            # fraction of the remaining NAV, so fee and notional are linked.
            def fee_balance(fraction):
                return fraction - cost_rate * np.abs((1 - fraction) * target - weights).sum()

            fee = brentq(fee_balance, 0.0, 1.0)
            changes = np.abs((1 - fee) * target - weights)
            turnover = changes.sum()
            trades = np.count_nonzero(changes > 1e-8)
            weights = target
        daily.append((gross * (1 - fee) - 1, turnover, trades))
    return pd.DataFrame(daily, index=returns.index, columns=["return", "turnover", "n_trades"])


def annualized_sharpe(returns):
    vol = returns.std(ddof=1)
    if pd.isna(vol) or vol <= 0:
        return None
    return float(np.sqrt(SESSIONS_PER_YEAR) * returns.mean() / vol)


def metrics(daily, cash_proxy):
    returns = daily["return"]
    equity = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    return {
        "cagr": float(equity[-1] ** (SESSIONS_PER_YEAR / len(daily)) - 1),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": annualized_sharpe(returns),
        "sharpe_excess": annualized_sharpe(returns - cash_proxy),
        "max_drawdown": float(np.min(equity / np.maximum.accumulate(equity) - 1)),
        "total_return": float(equity[-1] - 1),
        "turnover": float(daily["turnover"].sum()),
        "n_trades": int(daily["n_trades"].sum()),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    universe = config["universe"]
    excess_ticker = config["excess_return_ticker"]
    tickers = [*universe, excess_ticker]
    if len(universe) < 2 or len(tickers) != len(set(tickers)) or excess_ticker != "BIL":
        raise ValueError("Use distinct ETF tickers and reserve BIL for the excess-return calculation")
    data_start, evaluation_start, is_end, end = (
        pd.Timestamp(config[key]) for key in ("data_start", "evaluation_start", "is_end", "evaluation_end")
    )
    if not data_start < evaluation_start <= is_end < end:
        raise ValueError("Invalid warm-up, evaluation, or split dates")
    for key in ("lookback_sessions", "execution_lag_sessions", "rebalance_every_sessions"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 < config["cvar_confidence"] < 1:
        raise ValueError("CVaR confidence must be between zero and one")
    commission, slippage = config["commission_bps"], config["slippage_bps"]
    if not (commission >= 0 and slippage >= 0 and commission + slippage < 10_000):
        raise ValueError("Both costs must be nonnegative and sum to less than 10,000 bps")

    prices, fetched_at = load_prices(tickers, data_start, end)
    returns = prices.pct_change(fill_method=None)
    first_fill = returns.index.searchsorted(evaluation_start)
    evaluation = returns.iloc[first_fill:]
    in_sample = evaluation.index <= is_end
    if min(in_sample.sum(), (~in_sample).sum()) < 2:
        raise ValueError("IS and OOS must each have at least two sessions")
    targets, equal_targets, cash_fallbacks = target_schedule(returns[universe].to_numpy(), first_fill, config)
    rate = (commission + slippage) / 10_000
    strategy_daily = simulate(evaluation[universe], targets, rate)
    benchmark_daily = simulate(evaluation[universe], equal_targets, rate)
    masks = {"is": in_sample, "oos": ~in_sample, "full": slice(None)}
    strategy = {key: metrics(strategy_daily.loc[mask], evaluation.loc[mask, excess_ticker])
                for key, mask in masks.items()}
    benchmark = {key: metrics(benchmark_daily.loc[mask], evaluation.loc[mask, excess_ticker])
                 for key, mask in masks.items()}

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return close")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"CASH_FALLBACK_REBALANCES: {cash_fallbacks}")
    for key in masks:
        dates = evaluation.loc[masks[key]].index
        print(f"{key.upper()} {dates[0].date()} to {dates[-1].date()}")
        print(f"  Long-only CVaR: {json.dumps(strategy[key], allow_nan=False)}")
        print(f"  Equal-weight ETF benchmark: {json.dumps(benchmark[key], allow_nan=False)}")
    results = {
        "strategy": config["strategy"],
        "data_range": f"{evaluation.index[0].date()} to {evaluation.index[-1].date()}",
        **strategy,
        "benchmark": {"name": "equal_weight_etfs", **benchmark["full"], **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
