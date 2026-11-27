"""718_an_analytic_solution_for_asset_allocatio
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/718_an_analytic_solution_for_asset_allocatio.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 9532cd9f815cc2d43ceb143b2f1416acf34e9864 on 2026-11-27 20:03 Europe/Oslo, exit 0, output sha256 65400228517614745c343ceccf68363b38184178de39c90200bcb7bf09edc3a7
Data source: 通过 yfinance 从 Yahoo Finance 免费获取 SPY、TLT、GLD、IWM、EEM 和 BIL 的日线开盘价、收盘价及公司行动，BIL 为储备资产。已有本地行情快照时优先读取快照。
Data fetched: 2026-11-27T19:03:07.500003+00:00
Sample period: 计入回测结果的交易日为 2010-01-04 至 2025-12-31，含首尾。预热行情的请求起点为 2008-01-01。   data_range (from output): {'start': '2010-01-04', 'end': '2025-12-31'}
IS/OOS split: 样本内 2010-01-04 至 2017-12-29，样本外 2018-01-02 至 2025-12-31。单次连续模拟后按交易日切分，跨界延续持仓。默认策略参数固定，样本外仍按月滚动估计。六个稳健性变体仅使用截至样本内末日的行情。
Costs and slippage: 默认每笔买入和卖出均按未计摩擦的开盘成交额计 2 bps 费用和 3 bps 固定滑点。卖出回款扣减 5 bps，买入现金支出增加 5 bps。先卖后买，现金不足时按同一比例缩减买入量。样本内双倍摩擦变体为每边 4 bps 费用和 6 bps 滑点。
Signal lag: 每月首个交易日开盘模拟调仓，目标权重根据截至上一交易日 t 收盘的滚动窗口计算。信号与成交相隔一个交易日索引。
Price adjustment: yfinance split-adjusted Open/Close; forward ex-date total-return factors
Survivorship bias: 使用事后选定且仍存续的固定 ETF 名单，存在标的选择和幸存者偏差。
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.08023295012670606, ann_vol=0.09190316343621148, sharpe=0.8859168941527023, sharpe_excess=0.8800450229006901, max_drawdown=-0.1667597186630162, total_return=0.8524237729329169, turnover=4.943066385965336, n_trades=299
    oos: cagr=0.09750823274716214, ann_vol=0.11002623752945437, sharpe=0.9009247588644633, sharpe_excess=0.6806865008483273, max_drawdown=-0.1510193873406549, total_return=1.101166905544702, turnover=4.7979151951893835, n_trades=293
    full: cagr=0.08883203814884078, ann_vol=0.10135496704008665, sharpe=0.8906081828390754, sharpe_excess=0.7685938699649704, max_drawdown=-0.1667597186630162, total_return=2.8922515267308695, turnover=4.870526861946837, n_trades=592
    benchmark: cagr=0.08653393363012762, ann_vol=0.11599389722225623, sharpe=0.773693987351333, sharpe_excess=0.6665420617756532, max_drawdown=-0.25702038591174337, total_return=2.7631235578045366, turnover=0.39673020233609546, n_trades=960
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: '718_ an_analytic_solution_for_asset_allocatio'\nuniverse: [SPY, TLT, GLD, IWM, EEM]\nreserve_symbol: BIL\nprice_start: 2008-01-01\nsample_start: 2010-01-04\nis_end: 2017-12-29\nsample_end: 2025-12-31\nsignal_lag_sessions: 1\nhorizon_sessions: 21\nannualization_sessions: 252\nlookback_sessions: 252\nrisk_aversion: 15.0\ncost_bps: 2.0\nslippage_bps: 3.0\nrobustness:\n  - {name: lookback_210, lookback_sessions: 210}\n  - {name: lookback_294, lookback_sessions: 294}\n  - {name: risk_aversion_12, risk_aversion: 12.0}\n  - {name: risk_aversion_18, risk_aversion: 18.0}\n  - {name: gaussian_utility, distribution: gaussian}\n  - {name: double_execution_cost, cost_bps: 4.0, slippage_bps: 6.0}\n"
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
"""Monthly cross-asset allocation under Giller's multivariate Laplace utility.

Paper: https://arxiv.org/abs/2411.08967 (Graham L. Giller, 2024), §§1, 4.
Research references. No source code from these projects was copied.
https://github.com/cvxgrp/exp_util_gm_portfolio_opt (license unspecified)
https://raw.githubusercontent.com/cvxgrp/exp_util_gm_portfolio_opt/main/gm_evar_portfolio.py
https://github.com/skfolio/skfolio (BSD-3-Clause)
https://raw.githubusercontent.com/skfolio/skfolio/main/src/skfolio/optimization/convex/_mean_risk.py
https://skfolio.org/generated/skfolio.optimization.MeanRisk.html
https://pypi.org/project/skfolio/
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
FIELDS = ["Open", "Close", "Dividends", "Capital Gains", "Stock Splits"]


@dataclass(frozen=True)
class Specification:
    lookback_sessions: int
    risk_aversion: float
    cost_bps: float
    slippage_bps: float
    distribution: str = "laplace"

    def validate(self, dimension):
        if not isinstance(self.lookback_sessions, int) or self.lookback_sessions <= dimension:
            raise ValueError("The lookback must exceed the number of risky assets.")
        if not np.isfinite(self.risk_aversion) or self.risk_aversion <= 0:
            raise ValueError("Risk aversion must be positive and finite.")
        charges = np.array([self.cost_bps, self.slippage_bps])
        if not np.isfinite(charges).all() or min(charges) < 0 or sum(charges) >= 10_000:
            raise ValueError("Invalid execution charges.")
        if self.distribution not in {"laplace", "gaussian"}:
            raise ValueError("Unknown return distribution.")


def get_quotes(symbols, price_start, sample_end):
    CACHE.mkdir(exist_ok=True)
    end_exclusive = sample_end + pd.Timedelta(days=1)
    key = hashlib.sha256("|".join([*symbols, str(price_start), str(sample_end)]).encode())
    snapshot = CACHE / f"yahoo_prices_actions_v1_{key.hexdigest()[:16]}.csv"
    if snapshot.exists():
        with snapshot.open(encoding="utf-8") as saved:
            fetched_at = datetime.fromisoformat(saved.readline().strip())
            quotes = pd.read_csv(saved, header=[0, 1], index_col=0, parse_dates=True)
        return quotes, fetched_at

    frames = {}
    with TemporaryDirectory(prefix=".yfinance-", dir=CACHE) as vendor_cache:
        yf.set_tz_cache_location(vendor_cache)
        for symbol in symbols:
            quotes = yf.Ticker(symbol).history(
                start=price_start.date().isoformat(), end=end_exclusive.date().isoformat(),
                interval="1d", actions=True, auto_adjust=False, repair=False,
                keepna=True, timeout=10,
            )
            if quotes.empty:
                raise RuntimeError(f"Yahoo Finance did not return daily prices for {symbol}.")
            if "Capital Gains" not in quotes:
                quotes["Capital Gains"] = 0.0
            quotes.index = quotes.index.tz_localize(None).normalize()
            frames[symbol] = quotes[FIELDS]

    quotes = pd.concat(frames, axis=1).sort_index()
    validate_quotes(quotes, sample_end)
    fetched_at = datetime.now(timezone.utc)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as saved:
        temp_path = Path(saved.name)
        try:
            saved.write(fetched_at.isoformat() + "\n")
            quotes.to_csv(saved)
            saved.close()
            temp_path.replace(snapshot)
        finally:
            temp_path.unlink(missing_ok=True)
    return quotes, fetched_at


def validate_quotes(quotes, sample_end):
    if quotes.empty or not quotes.index.is_unique or not quotes.index.is_monotonic_increasing:
        raise ValueError("Yahoo Finance dates are missing, duplicated or unordered.")
    if quotes.index[-1] != sample_end:
        raise ValueError(f"Yahoo Finance prices must reach the configured last session {sample_end.date()}.")
    if not np.isfinite(quotes.to_numpy()).all():
        raise ValueError("All symbols must have complete quotes on their common trading sessions.")


def total_return_prices(quotes):
    """Apply ex-date adjustments forward in Yahoo's split-adjusted share units."""
    opens = quotes.xs("Open", axis=1, level=1)
    closes = quotes.xs("Close", axis=1, level=1)
    divs = quotes.xs("Dividends", axis=1, level=1) + quotes.xs("Capital Gains", axis=1, level=1)
    if (opens <= 0).any().any() or (closes <= 0).any().any() or (divs < 0).any().any():
        raise ValueError("Invalid Yahoo Finance price or distribution.")

    # The distribution at an ex-date open takes effect using the preceding close.
    # Yahoo has already applied split ratios to its Open, Close and distributions.
    ex_date_multiplier = 1.0 - divs / closes.shift(1)
    ex_date_multiplier.iloc[0] = 1.0
    if (ex_date_multiplier <= 0).any().any():
        raise ValueError("Distribution exceeds the preceding quote.")
    forward_factor = ex_date_multiplier.cumprod().rdiv(1.0)
    return opens * forward_factor, closes * forward_factor


def risky_weights(mean, covariance, spec):
    """Solve the long-only CARA problem with residual exposure available in BIL."""
    risk_aversion = spec.risk_aversion
    dimension = len(mean)
    shape = (dimension + 1) / 2
    # The division by risk_aversion leaves the minimizer unchanged.
    # s = λ² wᵀ Vw / (n + 1) is the squared argument in the transform domain.
    domain_buffer = 0.99

    def objective(weights):
        risk_gradient = risk_aversion * (covariance @ weights)
        normal_penalty = 0.5 * (weights @ risk_gradient)
        expected_loss = -(mean @ weights)
        if spec.distribution == "gaussian":
            return expected_loss + normal_penalty, -mean + risk_gradient

        transform_arg = risk_aversion * normal_penalty / shape
        # A tangent continuation lets line searches cross the log domain safely.
        # The final solution is required to lie in the unmodified region.
        bounded_arg = min(transform_arg, domain_buffer)
        penalty = shape / risk_aversion * (
            -np.log1p(-bounded_arg) + (transform_arg - bounded_arg) / (1.0 - bounded_arg)
        )
        return expected_loss + penalty, -mean + risk_gradient / (1.0 - bounded_arg)

    solution = minimize(
        objective, np.zeros(dimension), jac=True, method="SLSQP",
        bounds=[(0.0, 1.0)] * dimension,
        constraints=[{
            "type": "ineq", "fun": lambda w: 1.0 - w.sum(), "jac": lambda w: -np.ones_like(w),
        }],
        options={"ftol": 1e-13, "maxiter": 150},
    )
    if not solution.success or np.any(solution.x < -1e-10) or solution.x.sum() > 1.0 + 1e-10:
        raise RuntimeError(f"Portfolio optimization failed: {solution.message}")
    weights = np.maximum(solution.x, 0.0)
    weights /= max(1.0, weights.sum())
    transform_arg = risk_aversion**2 * (weights @ covariance @ weights) / (dimension + 1)
    if spec.distribution == "laplace" and not 0.0 <= transform_arg < domain_buffer:
        raise RuntimeError("Laplace optimum lies outside the supported transform domain.")
    return weights


def rebalance(positions, cash, open_prices, weights, spec):
    """Size at the open, process sales first, then scale purchases to available cash."""
    opening_value = cash + positions @ open_prices
    trade_values = opening_value * weights - positions * open_prices
    trade_values[np.abs(trade_values) < opening_value * 1e-10] = 0.0
    sales = np.maximum(-trade_values, 0.0)
    buys = np.maximum(trade_values, 0.0)
    cost = spec.cost_bps / 10_000
    slippage = spec.slippage_bps / 10_000
    spendable = cash + sales.sum() * (1.0 - slippage - cost)
    required = buys.sum() * (1.0 + slippage + cost)
    paid = min(spendable, required)
    if required > 0:
        buys *= paid / required
    positions = np.maximum(positions - sales / open_prices, 0.0) + buys / open_prices
    n_trades = np.count_nonzero(sales) + np.count_nonzero(buys)
    return positions, spendable - paid, (sales.sum() + buys.sum()) / opening_value, n_trades


def backtest(opens, closes, config, spec, *, equal_weight=False):
    dates = closes.index
    n_risky = len(config["universe"])
    lag = config["signal_lag_sessions"]
    horizon = config["horizon_sessions"]
    daily_returns = closes.pct_change(fill_method=None)
    reserve = daily_returns.iloc[:, -1]
    excess = daily_returns.iloc[:, :-1].sub(reserve, axis=0).to_numpy()
    is_month_open = ~dates.to_period("M").duplicated()
    price_opens = opens.to_numpy()
    price_closes = closes.to_numpy()

    cash = 1.0
    positions = np.zeros(n_risky + 1)
    previous_value = cash
    rows = []
    start_index = dates.get_loc(pd.Timestamp(config["sample_start"]))
    for fill_index in range(start_index, len(dates)):
        turnover, n_trades = 0.0, 0
        if is_month_open[fill_index]:
            signal_index = fill_index - lag
            first_return = signal_index - spec.lookback_sessions + 1
            if first_return < 1:
                raise ValueError("Price history is too short for the first signal.")
            if equal_weight:
                weights = np.r_[np.full(n_risky, 1 / n_risky), 0.0]
            else:
                window = excess[first_return:signal_index + 1]
                mean = horizon * window.mean(axis=0)
                covariance = horizon * np.cov(window, rowvar=False, ddof=1)
                risky = risky_weights(mean, covariance, spec)
                weights = np.r_[risky, max(0.0, 1.0 - risky.sum())]
            positions, cash, turnover, n_trades = rebalance(
                positions, cash, price_opens[fill_index], weights, spec,
            )

        close_value = cash + positions @ price_closes[fill_index]
        rows.append((close_value / previous_value - 1.0, turnover, n_trades, cash))
        previous_value = close_value

    daily_results = pd.DataFrame(
        rows, index=dates[start_index:], columns=["return", "turnover", "n_trades", "cash"],
    )
    daily_results["reserve_return"] = reserve
    if (daily_results["cash"] < 0).any() or not np.isfinite(daily_results.to_numpy()).all():
        raise RuntimeError("Invalid simulated cash or daily results.")
    return daily_results


def metrics(daily_results, annualization):
    returns = daily_results["return"].to_numpy()
    excess = returns - daily_results["reserve_return"].to_numpy()
    growth = np.r_[1.0, np.cumprod(1.0 + returns)]
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    root_year = np.sqrt(annualization)
    return {
        "cagr": float(growth[-1] ** (annualization / len(returns)) - 1.0),
        "ann_vol": float(volatility * root_year),
        "sharpe": float(root_year * returns.mean() / volatility) if volatility > 0 else 0.0,
        "sharpe_excess": float(root_year * excess.mean() / excess_volatility) if excess_volatility > 0 else 0.0,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1.0).min()),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].mean() * annualization),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def summarize(daily_results, is_end, annualization):
    in_sample = daily_results.index <= is_end
    return {
        "is": metrics(daily_results.loc[in_sample], annualization),
        "oos": metrics(daily_results.loc[~in_sample], annualization),
        "full": metrics(daily_results, annualization),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    price_start, sample_start, is_end, sample_end = (
        pd.Timestamp(config[key]) for key in ("price_start", "sample_start", "is_end", "sample_end")
    )
    if not price_start < sample_start < is_end < sample_end:
        raise ValueError("Invalid warm-up, sample or split dates.")
    annualization = config["annualization_sessions"]
    for value in (config["signal_lag_sessions"], config["horizon_sessions"], annualization):
        if not isinstance(value, int) or value < 1:
            raise ValueError("Lag, model horizon and annualization must be positive whole sessions.")
    symbols = [*config["universe"], config["reserve_symbol"]]
    if len(set(symbols)) != len(symbols) or len(symbols) < 3:
        raise ValueError("Specify distinct symbols and at least two risky assets.")

    default = Specification(**{key: config[key] for key in (
        "lookback_sessions", "risk_aversion", "cost_bps", "slippage_bps",
    )})
    variants = {
        entry["name"]: replace(default, **{k: v for k, v in entry.items() if k != "name"})
        for entry in config["robustness"]
    }
    specifications = {default, *variants.values()}
    if len(specifications) != len(config["robustness"]) + 1:
        raise ValueError("Robustness specifications must be distinct.")
    for spec in specifications:
        spec.validate(len(config["universe"]))

    quotes, fetched_at = get_quotes(symbols, price_start, sample_end)
    validate_quotes(quotes, sample_end)
    opens, closes = total_return_prices(quotes)
    if is_end not in closes.index or sample_start not in closes.index:
        raise ValueError("The configured score start and IS end must be trading sessions.")
    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}")
    print("PRICE_ADJUSTMENT: yfinance split-adjusted Open/Close; forward ex-date total-return factors")

    official = backtest(opens, closes, config, default)
    baseline = backtest(opens, closes, config, default, equal_weight=True)
    official_metrics = summarize(official, is_end, annualization)
    baseline_metrics = summarize(baseline, is_end, annualization)
    if not official_metrics["is"]["n_trades"] or not official_metrics["oos"]["n_trades"]:
        raise RuntimeError("Official backtest must execute trades in both periods.")

    print(f"SAMPLE: {official.index[0].date()} to {official.index[-1].date()}; "
          f"IS through {is_end.date()}; OOS from {official.loc[official.index > is_end].index[0].date()}")
    for period in ("is", "oos", "full"):
        print(f"STRATEGY {period.upper()}: {json.dumps(official_metrics[period], allow_nan=False)}")
        print(f"EQUAL-WEIGHT BENCHMARK {period.upper()}: {json.dumps(baseline_metrics[period], allow_nan=False)}")

    print("ROBUSTNESS: in-sample only; official configuration stays fixed")
    print("default IS: " + json.dumps(official_metrics["is"], allow_nan=False))
    for name, spec in variants.items():
        sensitivity = backtest(opens.loc[:is_end], closes.loc[:is_end], config, spec)
        print(f"{name} IS: {json.dumps(metrics(sensitivity, annualization), allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {len(specifications)}")
    print(f"LOOK_AHEAD_CHECK: lag={config['signal_lag_sessions']} sessions; open sizing; completed bars; "
          "forward ex-date factors; trailing windows; robustness restricted to IS")
    print("RESULTS_JSON: " + json.dumps({
        "strategy": config["strategy"],
        "data_range": {"start": sample_start.date().isoformat(), "end": sample_end.date().isoformat()},
        **official_metrics,
        "benchmark": baseline_metrics["full"],
    }, allow_nan=False))


if __name__ == "__main__":
    main()
