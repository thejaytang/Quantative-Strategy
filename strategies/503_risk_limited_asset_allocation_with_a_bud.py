"""503_risk_limited_asset_allocation_with_a_bud
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/503_risk_limited_asset_allocation_with_a_bud.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c0a8f76240a0431b019ab49c34040a95a765860a on 2026-11-27 10:03 Europe/Oslo, exit 0, output sha256 784530342e8a7538315a2b409514cd8b037983e3154df70614e81b735ccae1fc
Data source: Yahoo Finance 免费日频 ETF 价格，由 yfinance 下载。风险资产为 SPY、EFA、EEM、IEF、TLT、LQD、GLD、VNQ，现金代理为 BIL，基准为 SPY。（config.yaml:2-4，strategy.py:74-79、strategy.py:90-107，README.md:13）
Data fetched: 2026-11-27T09:03:48+00:00
Sample period: 行情请求区间为 2008-01-01 至 2025-12-31，2010 年前的价格用于信号预热。日志记录的实际模拟区间为 2010-01-05 至 2025-12-31，首日收盘执行首笔模拟交易。（config.yaml:6-9，README.md:14，logs/run_20261127_085803.log:14）   data_range (from output): 2010-01-05 to 2025-12-31
IS/OOS split: 样本内为 2010-01-05 至 2019-12-31（含）。样本外从 2020 年首个交易日起至 2025-12-31。默认策略连续模拟，分界处延续持仓，模拟结果按交易日切分。三组敏感性情景只使用样本内数据。（config.yaml:7-9，README.md:14、README.md:21，strategy.py:215-225、strategy.py:239-254，logs/run_20261127_085803.log:14）
Costs and slippage: 每笔买卖按成交金额计 2 bps 交易费及 3 bps 固定滑点，合计 5 bps。执行收盘先按价格更新原权重，再以漂移后的权重与扣费后目标头寸之间的绝对差额计算双边成交额，并从组合净值扣除费用及滑点。BIL 买卖及 SPY 基准的初始买入适用同一费率。样本内双倍摩擦情景为 4 bps 交易费及 6 bps 固定滑点。（config.yaml:15-20，strategy.py:156-188、strategy.py:235-248、strategy.py:255-260，README.md:16）
Signal lag: 每月首个交易日 t 收盘后，用当日及之前的首个交易日收盘价计算信号。配置将成交日顺延一个价格序列中的交易日，在 t+1 收盘模拟成交。原持仓获得 t 收盘至 t+1 收盘的收益，新权重从 t+1 收盘后生效。（config.yaml:14，strategy.py:120-136、strategy.py:171-188，README.md:15）
Price adjustment: yfinance daily auto_adjust total-return close
Survivorship bias: 固定资产池由编写时仍存续且历史较长的 ETF 构成。README 明确记录了事后选样与生存偏差。（config.yaml:2-4，README.md:23）
Parameter combinations tried: 4
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.031789411558305725, ann_vol=0.06408118585074485, sharpe=0.5204947922687424, sharpe_excess=0.4553993423395451, max_drawdown=-0.10321368778538409, total_return=0.36659870031960584, turnover=4.6139715897542475, n_trades=367
    oos: cagr=0.06103038469238231, ann_vol=0.0656654067668638, sharpe=0.935106556122568, sharpe_excess=0.5372567240133063, max_drawdown=-0.128087367995526, total_return=0.4254715292502713, turnover=2.992368862866614, n_trades=159
    full: cagr=0.04265460107331043, ann_vol=0.06467708700956099, sharpe=0.6782680478935853, sharpe_excess=0.4865673295437108, max_drawdown=-0.13318800109935525, total_return=0.9480475392160292, turnover=4.006122494018093, n_trades=526
    benchmark: cagr=0.13907318337129948, ann_vol=0.17209682893042028, sharpe=0.843092693885925, sharpe_excess=0.7706262715482035, max_drawdown=-0.3371727036119294, total_return=6.994776747296026, turnover=0.06260851677069752, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 503_risk_limited_asset_allocation_with_a_bud\nuniverse: [SPY, EFA, EEM, IEF, TLT, LQD, GLD, VNQ]\ncash_proxy: BIL\nbenchmark: SPY\n\nprices_start: "2008-01-01"\nstart: "2010-01-01"\nis_end: "2019-12-31"\nend: "2025-12-31"\n\nlookback_months: 12\nrisk_threshold: 0.40\nmax_asset_weight: 0.25\nexecution_lag_sessions: 1\ntrading_cost_bps: 2.0\nslippage_bps: 3.0\n\nrobustness:\n  risk_thresholds: [0.35, 0.45]\n  friction_multiplier: 2.0\n'
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
"""Long-only ETF allocation using Giller's budget-threshold risk cost."""

import hashlib
import json
import math
import signal
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
MIN_TRADE = 1e-10


def load_config():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    for key in ("prices_start", "start", "is_end", "end"):
        config[key] = pd.Timestamp(config[key])
    if not config["prices_start"] < config["start"] <= config["is_end"] < config["end"]:
        raise ValueError("Dates must include price history, an IS period and a later OOS period.")
    universe = config["universe"]
    if not universe or len(set(universe)) != len(universe) or config["cash_proxy"] in universe:
        raise ValueError("Use a nonempty universe of distinct assets and a separate cash proxy.")
    if config["cash_proxy"] != "BIL" or config["benchmark"] == "BIL":
        raise ValueError("Use BIL as the cash proxy and a separate benchmark.")
    if type(config["execution_lag_sessions"]) is not int or config["execution_lag_sessions"] < 1:
        raise ValueError("Execution lag must be at least one trading session.")
    if type(config["lookback_months"]) is not int or config["lookback_months"] < 2:
        raise ValueError("The lookback must include at least two monthly returns.")
    if not 0 < config["max_asset_weight"] <= 1:
        raise ValueError("The per-asset cap must be in (0, 1].")

    robustness = config["robustness"]
    thresholds = [config["risk_threshold"], *robustness["risk_thresholds"]]
    if any(not math.isfinite(k) or k < 0 for k in thresholds):
        raise ValueError("Risk thresholds must be finite and nonnegative.")
    if len(thresholds) < 3 or len(set(thresholds)) != len(thresholds):
        raise ValueError("Specify at least two distinct alternative thresholds.")
    costs = [config["trading_cost_bps"], config["slippage_bps"]]
    if any(not math.isfinite(cost) or cost < 0 for cost in costs):
        raise ValueError("Cost and slippage inputs must be finite and nonnegative.")
    multiplier = robustness["friction_multiplier"]
    if not math.isfinite(multiplier) or multiplier <= 1 or not 0 < sum(costs) * multiplier < 10_000:
        raise ValueError("The friction scenario needs a higher, positive cost below 10,000 bps.")
    return config


def check_prices(close, symbols, start, end):
    close = close.loc[start:end, symbols]
    if (
        close.empty
        or not close.index.is_unique
        or not close.index.is_monotonic_increasing
        or close.index[0] > start + pd.Timedelta(days=7)
        or close.index[-1] < end - pd.Timedelta(days=7)
    ):
        raise ValueError("Price dates are incomplete, duplicated or out of order.")
    invalid = (~np.isfinite(close) | (close <= 0)).sum()
    if invalid.any():
        detail = ", ".join(f"{symbol} ({count})" for symbol, count in invalid.items() if count)
        raise ValueError(f"Missing or invalid adjusted closes for {detail}.")
    return close


def load_prices(config):
    symbols = sorted(set([*config["universe"], config["cash_proxy"], config["benchmark"]]))
    start, end = config["prices_start"], config["end"]
    request = json.dumps([symbols, str(start.date()), str(end.date())]).encode()
    cache_key = hashlib.sha256(request).hexdigest()[:20]
    cache_path = CACHE_DIR / f"adjusted_close_{cache_key}.csv"
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as source:
            header = source.readline().strip()
            prefix = "# fetched_at="
            if not header.startswith(prefix):
                raise ValueError(f"Missing download timestamp in {cache_path}.")
            fetched_at = datetime.fromisoformat(header[len(prefix):]).isoformat()
            close = pd.read_csv(source, index_col="Date", parse_dates=["Date"])
        return check_prices(close, symbols, start, end), fetched_at

    CACHE_DIR.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=CACHE_DIR, prefix=".yfinance-") as scratch:
        yf.set_tz_cache_location(scratch)
        downloaded = yf.download(
            symbols,
            start=str(start.date()),
            end=str((end + pd.Timedelta(days=1)).date()),
            interval="1d",
            auto_adjust=True,
            actions=False,
            keepna=True,
            progress=False,
            timeout=12,
        )
        fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    if downloaded is None or downloaded.empty or "Close" not in downloaded:
        raise RuntimeError("Yahoo Finance returned no adjusted closes for the requested period.")
    close = check_prices(downloaded["Close"], symbols, start, end)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE_DIR, delete=False) as sink:
        cache_temp = Path(sink.name)
        try:
            sink.write(f"# fetched_at={fetched_at}\n")
            close.to_csv(sink, index_label="Date")
            sink.close()
            cache_temp.replace(cache_path)
        finally:
            cache_temp.unlink(missing_ok=True)
    return close, fetched_at


def signal_inputs(close, config):
    # First-session signals use that session's close and earlier first-session closes.
    months = close.index.to_period("M")
    signal_rows = np.flatnonzero(~months.duplicated())
    monthly = close.iloc[signal_rows].pct_change(fill_method=None)
    excess = monthly[config["universe"]].sub(monthly[config["cash_proxy"]], axis=0)
    trailing = excess.rolling(config["lookback_months"])

    fill_rows = signal_rows + config["execution_lag_sessions"]
    in_range = fill_rows < len(close)
    fill_dates = close.index[fill_rows[in_range]]
    alpha = trailing.mean().iloc[in_range].set_axis(fill_dates).loc[config["start"]:]
    risk = trailing.std(ddof=1).iloc[in_range].set_axis(fill_dates).loc[config["start"]:]
    if alpha.empty or alpha.isna().any().any() or risk.isna().any().any():
        raise ValueError("The chosen dates need enough first-session closes to warm up the signals.")
    daily = close.pct_change(fill_method=None).loc[alpha.index[0]:]
    return daily, alpha, risk


def make_targets(alpha, risk, symbols, config, risk_threshold):
    # Each positive utility slope receives up to the cap in descending order.
    scores = alpha - risk_threshold * risk
    targets = pd.DataFrame(0.0, index=scores.index, columns=symbols)
    cap = config["max_asset_weight"]
    for date, row in scores.iterrows():
        remaining = 1.0
        for symbol, score in row.sort_values(ascending=False, kind="stable").items():
            if score <= 0 or remaining <= 0:
                break
            weight = min(cap, remaining)
            targets.at[date, symbol] = weight
            remaining -= weight
        targets.at[date, config["cash_proxy"]] = remaining
    return targets


def trade(drifted, target, friction):
    # Solve for the post-cost value fraction so charges match the actual traded amounts.
    lower, upper = 0.0, 1.0
    for _ in range(48):
        fraction = (lower + upper) / 2
        traded = np.abs(fraction * target - drifted).sum()
        if fraction + friction * traded > 1:
            upper = fraction
        else:
            lower = fraction
    fraction = (lower + upper) / 2
    orders = np.abs(fraction * target - drifted)
    return 1.0 - friction * orders.sum(), orders


def simulate(daily, targets, friction):
    weights = np.zeros(len(daily.columns))
    fills = dict(zip(targets.index, targets[daily.columns].to_numpy(), strict=True))
    daily_results = []
    for date, asset_returns in zip(daily.index, daily.to_numpy(), strict=True):
        # The old weights earn the return up to the execution close.
        change = float(weights @ asset_returns)
        value_factor = 1.0 + change
        weights = weights * (1.0 + asset_returns) / value_factor
        turnover, n_trades = 0.0, 0
        if date in fills:
            target = fills[date]
            after_cost, orders = trade(weights, target, friction)
            value_factor *= after_cost
            turnover = float(orders.sum())
            n_trades = int(np.count_nonzero(orders > MIN_TRADE))
            weights = target
        daily_results.append((value_factor - 1.0, turnover, n_trades))
    return pd.DataFrame(daily_results, index=daily.index, columns=["return", "turnover", "n_trades"])


def metrics(daily_results, cash_returns):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise ValueError("Each reported period must include at least two sessions.")
    excess = returns - cash_returns.loc[returns.index]
    vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    curve = np.r_[1.0, (1.0 + returns).cumprod().to_numpy()]
    annualizer = math.sqrt(SESSIONS_PER_YEAR)
    years = len(returns) / SESSIONS_PER_YEAR
    sharpe_excess = float(excess.mean() / excess_vol * annualizer) if excess_vol > 1e-12 else None
    return {
        "cagr": float(curve[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(vol * annualizer),
        "sharpe": float(returns.mean() / vol * annualizer) if vol > 1e-12 else None,
        "sharpe_excess": sharpe_excess,
        "max_drawdown": float(np.min(curve / np.maximum.accumulate(curve) - 1.0)),
        "total_return": float(curve[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def report(daily_results, cash_returns, is_end, label):
    is_mask = daily_results.index <= is_end
    periods = {
        "is": daily_results.loc[is_mask],
        "oos": daily_results.loc[~is_mask],
        "full": daily_results,
    }
    result = {}
    for period, subset in periods.items():
        result[period] = metrics(subset, cash_returns)
        print(f"{label} {period.upper()}: {json.dumps(result[period], allow_nan=False)}")
    return result


def main():
    config = load_config()
    close, fetched_at = load_prices(config)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance daily auto_adjust total-return close")
    is_end = config["is_end"]
    threshold = config["risk_threshold"]
    friction = (config["trading_cost_bps"] + config["slippage_bps"]) / 10_000
    cash_proxy = config["cash_proxy"]

    robust = config["robustness"]
    cases = [(f"K={k:g}", k, friction) for k in robust["risk_thresholds"]]
    multiplier = robust["friction_multiplier"]
    cases.append((f"friction x{multiplier:g}", threshold, friction * multiplier))
    print("ROBUSTNESS (IS only)")
    is_daily, is_alpha, is_risk = signal_inputs(close.loc[:is_end], config)
    for name, k, case_friction in cases:
        targets = make_targets(is_alpha, is_risk, is_daily.columns, config, k)
        result = metrics(simulate(is_daily, targets, case_friction), is_daily[cash_proxy])
        print(f"{name}: {json.dumps(result, allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {1 + len(cases)}")

    daily, alpha, risk = signal_inputs(close, config)
    cash_returns = daily[cash_proxy]
    targets = make_targets(alpha, risk, daily.columns, config, threshold)
    result = report(simulate(daily, targets, friction), cash_returns, is_end, "STRATEGY")
    benchmark = config["benchmark"]
    benchmark_target = pd.DataFrame(0.0, index=daily.index[:1], columns=daily.columns)
    benchmark_target.loc[:, benchmark] = 1.0
    benchmark_result = report(
        simulate(daily, benchmark_target, friction), cash_returns, is_end, f"BENCHMARK {benchmark}"
    )
    output = {
        "strategy": config["strategy"],
        "data_range": f"{daily.index[0].date()} to {daily.index[-1].date()}",
        **result,
        "benchmark": {"name": benchmark, **benchmark_result["full"], **benchmark_result},
    }
    print("RESULTS_JSON: " + json.dumps(output, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)