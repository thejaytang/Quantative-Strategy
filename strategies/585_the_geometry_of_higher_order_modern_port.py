"""585_the_geometry_of_higher_order_modern_port
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/585_the_geometry_of_higher_order_modern_port.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 25a9179813de3addbbaee65a8c08579eb6e27d08 on 2026-11-27 19:09 Europe/Oslo, exit 0, output sha256 3135c1667990590d3a45d48134b08ef69b1c198b79ab402e9745b66cee6efd90
Data source: yfinance 免费 ETF 日线的复权开盘价和收盘价。投资池为 SPY、QQQ、IWM、EFA、IEF、TLT、GLD。SPY 另作买入持有基准，BIL 的收盘到收盘收益仅用于计算超额 Sharpe。
Data fetched: 2026-11-27T18:09:02+00:00
Sample period: 行情请求区间为 2007-06-01 至 2025-12-31，包含预热期。日志记录的实际回测区间为 2008-07-01 至 2025-12-31，含首末交易日。   data_range (from output): {'start': '2008-07-01', 'end': '2025-12-31', 'is_end': '2016-12-30', 'oos_start': '2017-01-03'}
IS/OOS split: 按时间划分。IS 为 2008-07-01 至 2016-12-30，OOS 为 2017-01-03 至 2025-12-31。配置以 2016-12-31 为包含式 IS 截止日。四个稳健性变体只在 IS 运行，官方规格使用固定参数贯穿全期，跨边界延续原仓位和现金，OOS 仍用当时已有的行情滚动估计。
Costs and slippage: 默认每边 5 bps 成本加 5 bps 固定滑点，共 10 bps，按模拟开盘买卖名义金额分别计提，初始建仓同样计提。卖出按成交额扣减摩擦后入账，买入按成交额加计摩擦后扣款。滑点以比例扣费建模，模拟成交价取开盘价。IS 双倍摩擦变体为每边 20 bps。
Signal lag: 季末交易日收盘 t 用截至该收盘的 252 个日收益估计目标权重，默认在下一交易日开盘成交。首次建仓也用成交前一交易日收盘的信号。
Price adjustment: yfinance auto_adjust total-return Open/Close
Survivorship bias: 全期使用固定 ETF 池，选自仍存续且具备长历史的基金。README 明示生存者偏差与事后选池风险。
Parameter combinations tried: 5
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.06968244063126217, ann_vol=0.07332115816797842, sharpe=0.9554646100555614, sharpe_excess=0.9467224953809281, max_drawdown=-0.1071683300625842, total_return=0.7728270392278458, turnover=1.2386384136827022, n_trades=172
    oos: cagr=0.0784325256269871, ann_vol=0.08758559307971171, sharpe=0.9060588330683033, sharpe_excess=0.6510531177208886, max_drawdown=-0.22157471352753944, total_return=0.9695001334775692, turnover=1.2850471699914132, n_trades=183
    full: cagr=0.07416778974399296, ann_vol=0.08095366449568815, sharpe=0.9243987380358087, sharpe_excess=0.7789564094418676, max_drawdown=-0.22157471352753877, total_return=2.491583090391931, turnover=1.2624750637213726, n_trades=355
    benchmark: cagr=0.12173330685690464, ann_vol=0.19944530061198287, sharpe=0.6758861740976237, sharpe_excess=0.6149336205294805, max_drawdown=-0.4716522800502114, total_return=6.445269441015689, turnover=0.05716354490196453, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 585_the_geometry_of_higher_order_modern_port\nuniverse: [SPY, QQQ, IWM, EFA, IEF, TLT, GLD]\nbenchmark: SPY\nrisk_free: BIL\ndata_start: 2007-06-01\nstart: 2008-07-01\nis_end: 2016-12-31\nend: 2025-12-31\n\nlookback_sessions: 252\nrebalance: quarterly\nlag_sessions: 1\nmax_weight: 0.35\nrisk_aversion: 60.0\ncost_bps: 5.0\nslippage_bps: 5.0\n\nrobustness:\n  risk_aversion: [45.0, 75.0]\n  friction_multiplier: 2.0\n'
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
"""585: rolling cross-asset allocations using four portfolio cumulants.

Paper: https://arxiv.org/abs/2511.20674
Research references (ideas only, independently implemented):
  highOrderPortfolios (GPL-3.0): https://github.com/dppalomar/highOrderPortfolios/blob/master/R/MVSK.R
  Riskfolio-Lib (BSD-3-Clause): https://github.com/dcajasn/Riskfolio-Lib/blob/master/examples/Tutorial%2057%20-%20Mean%20Variance%20Skewness%20Kurtosis%20Optimization.ipynb
  Riskfolio-Lib package: https://pypi.org/project/riskfolio-lib/
"""

import hashlib
import json
import signal
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, minimize


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
RETURN_UNIT = 0.01


def load_config():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    universe = config["universe"]
    if len(universe) < 2 or len(set(universe)) != len(universe):
        raise ValueError("Use at least two distinct universe tickers.")
    if not config["data_start"] < config["start"] < config["is_end"] < config["end"]:
        raise ValueError("Require data_start < start < is_end < end.")
    if config["rebalance"] != "quarterly" or config["lag_sessions"] < 1:
        raise ValueError("Require quarterly rebalancing and a positive fill lag.")
    if config["lookback_sessions"] < 4:
        raise ValueError("Use at least four daily returns per estimation window.")
    if not 1 / len(universe) <= config["max_weight"] <= 1:
        raise ValueError("The maximum weight must permit a fully invested long-only target.")
    risk_values = [config["risk_aversion"], *config["robustness"]["risk_aversion"]]
    if any(not np.isfinite(value) or value <= 0 for value in risk_values):
        raise ValueError("Risk aversion must be finite and positive.")
    if len(risk_values) < 3 or len(set(risk_values)) != len(risk_values):
        raise ValueError("Specify at least two distinct sensitivity values besides the default.")
    costs = [config["cost_bps"], config["slippage_bps"]]
    multiple = config["robustness"]["friction_multiplier"]
    if any(not np.isfinite(cost) or cost <= 0 for cost in costs) or not 1 < multiple:
        raise ValueError("Set positive cost and slippage and a friction multiplier above one.")
    if sum(costs) * multiple >= 10_000:
        raise ValueError("Total friction per side must be below 100%.")
    return config


def validate_prices(prices, columns, start, end):
    if prices.empty or not columns.isin(prices.columns).all():
        raise ValueError("Adjusted opening or closing prices are missing.")
    prices = prices.loc[:, columns].sort_index()
    if not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_unique:
        raise ValueError("Expected distinct daily price dates.")
    if not (start <= prices.index[0].date() <= start + timedelta(days=7)
            and end - timedelta(days=7) <= prices.index[-1].date() <= end):
        raise ValueError("Price history does not cover the configured start and end.")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Price panel has invalid or missing values on at least one session.")
    return prices


def adjusted_prices(config):
    tickers = sorted({*config["universe"], config["benchmark"], config["risk_free"]})
    start, end = config["data_start"], config["end"]
    columns = pd.MultiIndex.from_product([("Open", "Close"), tickers], names=["Price", "Ticker"])
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    request_key = f"yf-adjusted-daily-v1/{','.join(tickers)}/{start}/{end}"
    cache_path = cache_dir / f"prices_{hashlib.sha256(request_key.encode()).hexdigest()[:16]}.csv"
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as source:
            fetched_at = source.readline().strip().removeprefix("# ")
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True)
        return validate_prices(prices, columns, start, end), fetched_at

    # Keep yfinance's temporary provider session files out of the retained data snapshot.
    with tempfile.TemporaryDirectory(dir=cache_dir) as provider_cache:
        yf.set_tz_cache_location(provider_cache)
        for attempt in range(3):
            try:
                prices = yf.download(
                    tickers, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                    interval="1d", auto_adjust=True, actions=False, progress=False,
                    threads=4, timeout=12, multi_level_index=True,
                )
                prices = validate_prices(prices, columns, start, end)
                break
            except (OSError, ValueError) as error:
                if attempt == 2:
                    raise RuntimeError("Could not obtain a complete yfinance price panel.") from error
                time.sleep(attempt + 1)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with tempfile.NamedTemporaryFile(mode="w", dir=cache_dir, encoding="utf-8", delete=False) as target:
        temporary_path = Path(target.name)
        try:
            target.write(f"# {fetched_at}\n")
            prices.to_csv(target)
            target.flush()
            temporary_path.replace(cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def optimize_weights(returns, risk_aversion, max_weight, use_higher_cumulants):
    """Minimize the negative fourth-order exponential-utility certainty equivalent."""
    n_rows, n_assets = returns.shape
    equal = np.full(n_assets, 1.0 / n_assets)
    if max_weight == 1.0 / n_assets:
        return equal

    normalized = returns / RETURN_UNIT
    mean = normalized.mean(axis=0)
    centered = normalized - mean
    tilt = risk_aversion * RETURN_UNIT
    second = tilt / 2
    third, fourth = (tilt**2 / 6, tilt**3 / 24) if use_higher_cumulants else (0.0, 0.0)

    def objective(weights):
        projected = centered @ weights
        squared = projected**2
        variance = squared.mean()
        skew_term = (projected * squared).mean()
        fourth_cumulant = (squared**2).mean() - 3 * variance**2
        value = -(mean @ weights) + second * variance - third * skew_term + fourth * fourth_cumulant
        slope = 2 * second * projected - 3 * third * squared
        slope += 4 * fourth * projected * (squared - 3 * variance)
        return value, -mean + (centered.T @ slope) / n_rows

    # Deterministic feasible starts for the potentially nonconvex objective.
    capped = np.clip(1 - max_weight * np.arange(n_assets), 0, max_weight)
    favor_return = np.zeros(n_assets)
    favor_return[np.argsort(-mean, kind="stable")] = capped
    favor_low_variance = np.zeros(n_assets)
    favor_low_variance[np.argsort((centered**2).mean(axis=0), kind="stable")] = capped
    budget = {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)}
    candidates = []
    for initial in (equal, favor_return, favor_low_variance):
        result = minimize(
            objective, initial, jac=True, method="SLSQP", constraints=budget,
            bounds=[(0.0, max_weight)] * n_assets, options={"ftol": 1e-11, "maxiter": 100},
        )
        if result.success and np.isfinite(result.fun) and abs(result.x.sum() - 1) <= 1e-9:
            target = np.clip(result.x, 0, max_weight)
            residual = 1 - target.sum()
            index = int(np.argmax(max_weight - target if residual >= 0 else target))
            target[index] += residual
            if (target >= 0).all() and (target <= max_weight).all():
                candidates.append((objective(target)[0], target))
    if not candidates:
        raise RuntimeError("No moment optimizer start returned a feasible solution.")
    return min(candidates, key=lambda candidate: candidate[0])[1]


def signal_targets(close, first, config, risk_aversion, use_higher_cumulants=True):
    close_returns = close.pct_change(fill_method=None).to_numpy()
    quarters = close.index.to_period("Q")
    quarter_starts = np.flatnonzero(quarters[1:] != quarters[:-1]) + 1
    rebalance_indices = [first, *quarter_starts[quarter_starts > first]]
    lag = config["lag_sessions"]
    window = config["lookback_sessions"]
    targets = {}
    for next_open in rebalance_indices:
        signal_close = next_open - 1
        fill_open = signal_close + lag
        if fill_open >= len(close):
            break
        if signal_close < window or close.index[signal_close] >= close.index[fill_open]:
            raise ValueError("Signal must have a full trailing window and precede its fill.")
        trailing = close_returns[signal_close - window + 1:signal_close + 1]
        targets[fill_open] = optimize_weights(
            trailing, risk_aversion, config["max_weight"], use_higher_cumulants,
        )
    return targets


def rebalance_at_open(positions, cash, weights, friction):
    value_open = positions.sum() + cash
    opening_weights = positions / value_open
    # Solve for the investable fraction after charging friction on every buy and sell.
    investable = brentq(
        lambda fraction: fraction + friction * np.abs(fraction * weights - opening_weights).sum() - 1,
        0, 1, xtol=1e-14,
    )
    changes = investable * value_open * weights - positions
    changes[np.abs(changes) <= 1e-10 * value_open] = 0
    sales = -np.minimum(changes, 0)
    buys = np.maximum(changes, 0)
    available = cash + sales.sum() * (1 - friction)
    buy_value = buys.sum()
    budget = available / (1 + friction)
    if buy_value > budget:
        buys *= budget / buy_value
    cash_left = available - buys.sum() * (1 + friction)
    if cash_left < -1e-12 * value_open:
        raise RuntimeError("Opening trades would exceed available cash.")
    gross = sales.sum() + buys.sum()
    positions = np.maximum(positions - sales, 0) + buys
    trades = np.count_nonzero(sales) + np.count_nonzero(buys)
    return positions, max(0.0, cash_left), gross / value_open, trades


def simulate(opens, closes, first, targets, friction):
    opening_prices, closing_prices = opens.to_numpy(), closes.to_numpy()
    positions = np.zeros(closes.shape[1])
    cash = previous_value = 1.0
    rows = []
    for index in range(first, len(closes)):
        open_prices = opening_prices[index]
        positions *= open_prices / closing_prices[index - 1]
        turnover = trades = 0
        if index in targets:
            positions, cash, turnover, trades = rebalance_at_open(positions, cash, targets[index], friction)
        positions *= closing_prices[index] / open_prices
        closing_value = positions.sum() + cash
        rows.append((closing_value / previous_value - 1, turnover, trades))
        previous_value = closing_value
    return pd.DataFrame(rows, index=closes.index[first:], columns=["return", "turnover", "n_trades"])


def metrics(daily_results, bill_returns):
    returns = daily_results["return"]
    excess = returns - bill_returns.loc[returns.index]
    growth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    annualizer = SESSIONS_PER_YEAR / len(returns)
    daily_vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    return {
        "cagr": float(growth[-1]**annualizer - 1),
        "ann_vol": float(daily_vol * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": float(returns.mean() / daily_vol * np.sqrt(SESSIONS_PER_YEAR)) if daily_vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * np.sqrt(SESSIONS_PER_YEAR)) if excess_vol > 0 else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].sum() * annualizer),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, bill_returns, is_end):
    return {
        "is": metrics(daily_results.loc[:is_end], bill_returns),
        "oos": metrics(daily_results.loc[daily_results.index > is_end], bill_returns),
        "full": metrics(daily_results, bill_returns),
    }


def main():
    started = time.monotonic()
    config = load_config()
    prices, fetched_at = adjusted_prices(config)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return Open/Close")
    opens = prices["Open"][config["universe"]]
    closes = prices["Close"][config["universe"]]
    first = int(closes.index.searchsorted(pd.Timestamp(config["start"])))
    is_end = pd.Timestamp(config["is_end"])
    last_is = int(closes.index.searchsorted(is_end, side="right"))
    if first <= config["lookback_sessions"] or last_is - first < 2 or len(closes) - last_is < 2:
        raise ValueError("Require complete warm-up and at least two sessions in each period.")
    friction = (config["cost_bps"] + config["slippage_bps"]) / 10_000
    bill_returns = prices["Close"][config["risk_free"]].pct_change(fill_method=None)
    risk_aversion = config["risk_aversion"]

    print("ROBUSTNESS (IS ONLY)")
    only_is_opens, only_is_closes = opens.iloc[:last_is], closes.iloc[:last_is]
    only_is_bills = bill_returns.iloc[:last_is]
    variants = [(f"risk_aversion={risk:g}", risk, True, friction)
                for risk in config["robustness"]["risk_aversion"]]
    multiple = config["robustness"]["friction_multiplier"]
    variants += [("mean_variance_only", risk_aversion, False, friction),
                 (f"friction_x{multiple:g}", risk_aversion, True, friction * multiple)]
    trial_count = 0
    for label, risk, higher_cumulants, variant_friction in variants:
        targets = signal_targets(only_is_closes, first, config, risk, higher_cumulants)
        result = simulate(only_is_opens, only_is_closes, first, targets, variant_friction)
        trial_count += 1
        print(f"IS {label} {json.dumps(metrics(result, only_is_bills), allow_nan=False)}")

    targets = signal_targets(closes, first, config, risk_aversion)
    daily_results = simulate(opens, closes, first, targets, friction)
    trial_count += 1
    official = period_metrics(daily_results, bill_returns, is_end)
    if any(period["n_trades"] <= 0 for period in official.values()):
        raise RuntimeError("The official strategy must produce trades in IS and OOS.")

    benchmark = config["benchmark"]
    benchmark_results = simulate(
        prices["Open"][[benchmark]], prices["Close"][[benchmark]], first,
        {min(targets): np.array([1.0])}, friction,
    )
    benchmark_periods = period_metrics(benchmark_results, bill_returns, is_end)
    for label in ("is", "oos", "full"):
        print(f"OFFICIAL {label.upper()} {json.dumps(official[label], allow_nan=False)}")
        print(f"BENCHMARK {benchmark} {label.upper()} {json.dumps(benchmark_periods[label], allow_nan=False)}")
    print("LOOKAHEAD_CHECK signal_lag: trailing returns end on the signal close and fills use a later open.")
    print("LOOKAHEAD_CHECK availability: the model uses only published ETF closes; BIL is evaluation-only.")
    print("LOOKAHEAD_CHECK adjustment: adjacent price ratios use consistent split/dividend-adjusted fields.")
    print("LOOKAHEAD_CHECK IS/OOS: alternatives use the IS slice; official coefficients stay fixed and windows roll forward.")
    print(f"PARAM_COMBOS_TRIED: {trial_count}")
    print(f"ELAPSED_SECONDS: {time.monotonic() - started:.2f}")
    print("STATUS: 已回测 / pending review")
    summary = {
        "strategy": config["strategy"],
        "data_range": {
            "start": daily_results.index[0].date().isoformat(),
            "end": daily_results.index[-1].date().isoformat(),
            "is_end": daily_results.index[last_is - first - 1].date().isoformat(),
            "oos_start": daily_results.index[last_is - first].date().isoformat(),
        },
        **official,
        "benchmark": {"name": f"{benchmark} buy and hold", **benchmark_periods["full"], **benchmark_periods},
    }
    print("RESULTS_JSON: " + json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(840)
    main()
    signal.alarm(0)
