"""617_path_portfolio_optimization_defect_lift
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/617_path_portfolio_optimization_defect_lift.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 25a9179813de3addbbaee65a8c08579eb6e27d08 on 2026-11-27 19:10 Europe/Oslo, exit 0, output sha256 1bc8ad9b3534a9fff947dff8fb8ea852957278293b1958348f673f0db1a76e73
Data source: Yahoo Finance 免费日线，经 yfinance 获取 Open、Close、Dividends、Capital Gains 和 Stock Splits。配置的 ETF 池为 SPY、QQQ、EFA、EEM、IEF、TLT、LQD、HYG、GLD、VNQ，BIL 的日总回报只用于计算超额 Sharpe。
Data fetched: 2026-11-27T18:10:30+00:00
Sample period: 行情请求区间为 2008-01-02 至 2026-10-30（含），2013 年之前的行情用于预热和历史训练。日志中回测区间为 2013-01-02 至 2026-10-30。   data_range (from output): {'start': '2013-01-02', 'end': '2026-10-30'}
IS/OOS split: 以 2022-01-01 为界。IS 为 2013-01-02 至 2021-12-31，OOS 为 2022-01-03 至 2026-10-30。正式策略全程连续模拟，持仓和现金跨界延续。稳健性变体仅用 IS 切片，正式全期结果的 IS 前缀与独立 IS 运行逐日精确比对。OOS 仍按时点滚动训练，可用届时已经完成的 OOS 历史标签。
Costs and slippage: 每笔买卖均建模 5 bps 滑点，买入价上调、卖出价下调，再按滑点后的成交额收取 5 bps 费用。以开盘名义金额为基准，买入现金支出乘 (1+0.0005)×(1+0.0005)，卖出现金收入乘 (1−0.0005)×(1−0.0005)。先卖后买，资金不足时同比缩减买入额。
Signal lag: 每月首个交易日 t 收盘计算目标权重，配置滞后一个交易日，于 t+1 开盘模拟成交。代码要求 lag_sessions 为正整数。
Price adjustment: Yahoo split-adjusted Open/Close with forward total return from ex-date reinvestment at Open
Survivorship bias: 固定 ETF 池由当前仍存续的基金回顾性构造，README 明确记录选样和基金幸存者偏差。
Parameter combinations tried: 5
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.08926620580092148, ann_vol=0.10740265607560007, sharpe=0.8500562367147968, sharpe_excess=0.8029413365834326, max_drawdown=-0.18965406733869328, total_return=1.1580371290802614, turnover=11.16320797936066, n_trades=647
    oos: cagr=-0.00784414302391645, ann_vol=0.13831369297656151, sharpe=0.012028330992638002, sharpe_excess=-0.26332766122955015, max_drawdown=-0.31242016812620976, total_return=-0.0371670373235673, turnover=10.574216162049982, n_trades=350
    full: cagr=0.05440133172151418, ann_vol=0.11909604097885257, sharpe=0.5043975540415649, sharpe_excess=0.3651612371535599, max_drawdown=-0.3245745692552606, total_return=1.077829282558087, turnover=10.958017383620348, n_trades=997
    benchmark: cagr=0.07026811135446165, ann_vol=0.09741143679824506, sharpe=0.7460300022337045, sharpe_excess=0.5758667592359443, max_drawdown=-0.2454336291428253, total_return=1.553650933443548, turnover=0.3423540008795481, n_trades=1660
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 617_path_portfolio_optimization_defect_lift\nuniverse: [SPY, QQQ, EFA, EEM, IEF, TLT, LQD, HYG, GLD, VNQ]\nrisk_free: BIL\ndata_start: 2008-01-02\nstart: 2013-01-02\noos_start: 2022-01-01\nend: 2026-10-30\nlag_sessions: 1\ncost_bps: 5\nslippage_bps: 5\nmodel:\n  signature_order: 2\n  path_sessions: 90\n  fit_sessions: 1008\n  fit_stride: 5\n  horizon_sessions: 21\n  risk_sessions: 126\n  risk_aversion: 3.0\n  max_weight: 0.25\nrobustness:\n  path_sessions: [60, 120]\n  alternative_signature_order: 1\n  cost_multiplier: 2.0\n'
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
"""Monthly ETF allocation using the symmetric signature block and shrinkage.

Source paper: Miquel Noguer i Alonso (2026), https://arxiv.org/abs/2608.02355
Method references (independent implementation):
https://github.com/owenfutter/signature_trading (license unspecified)
https://github.com/janka-moeller/Sig-SPT (license unspecified)
https://pypi.org/project/iisignature/ and https://github.com/bottler/iisignature (MIT)
"""

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_FIELDS = ("Open", "Close", "Dividends", "Capital Gains", "Stock Splits")
SESSIONS_PER_YEAR = 252


def validate_config(config):
    dates = [pd.Timestamp(config[key]) for key in ("data_start", "start", "oos_start", "end")]
    if any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("Expected data_start < start < oos_start < end")
    if type(config["lag_sessions"]) is not int or config["lag_sessions"] < 1:
        raise ValueError("lag_sessions must be a positive integer")
    universe = config["universe"]
    if len(set(universe)) != len(universe) or len(universe) < 2 or config["risk_free"] in universe:
        raise ValueError("Use a distinct ETF universe and BIL as a separate reference")
    if config["risk_free"] != "BIL":
        raise ValueError("The risk-free reference must be BIL")
    model = config["model"]
    for key in ("path_sessions", "fit_sessions", "fit_stride", "horizon_sessions", "risk_sessions"):
        if type(model[key]) is not int or model[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if model["fit_sessions"] <= model["fit_stride"] or model["risk_sessions"] < 2:
        raise ValueError("The fit and risk windows must each supply multiple observations")
    if not 1 / len(universe) <= model["max_weight"] <= 1 or not model["risk_aversion"] > 0:
        raise ValueError("Invalid portfolio constraints")


def validate_bars(bars, request):
    expected = pd.MultiIndex.from_product([PRICE_FIELDS, request["tickers"]], names=["Price", "Ticker"])
    if not bars.columns.equals(expected) or not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError("Unexpected Yahoo Finance bar schema")
    start, end = (pd.Timestamp(request[key]) for key in ("start", "end"))
    if (bars.empty or bars.index[0] < start or bars.index[-1] > end
            or bars.index[0] > start + pd.Timedelta(days=7)
            or bars.index[-1] < end - pd.Timedelta(days=7)):
        raise ValueError("Yahoo Finance snapshot does not cover the requested dates")
    if not np.isfinite(bars.to_numpy()).all() or (bars[["Open", "Close"]] <= 0).any().any():
        raise ValueError("Missing or invalid ETF bars. The backtest requires a complete common grid.")
    if (bars[["Dividends", "Capital Gains", "Stock Splits"]] < 0).any().any():
        raise ValueError("Unexpected negative corporate-action value")


def download_bars(request):
    tickers = request["tickers"]
    end_exclusive = (pd.Timestamp(request["end"]) + pd.Timedelta(days=1)).date().isoformat()
    # Keep the downloader's transient session cache apart from the data snapshot.
    with TemporaryDirectory(dir=CACHE) as transient:
        yf.set_tz_cache_location(transient)
        for attempt in range(2):
            raw = yf.download(
                tickers, start=request["start"], end=end_exclusive,
                auto_adjust=False, actions=True, repair=False, ignore_tz=True,
                keepna=False, threads=False, progress=False, timeout=12,
            )
            try:
                if raw is None or raw.empty:
                    raise ValueError("Yahoo Finance returned no bars")
                bars = raw.reindex(columns=pd.MultiIndex.from_product(
                    [PRICE_FIELDS, tickers], names=["Price", "Ticker"],
                )).copy()
                for ticker in tickers:
                    if ("Capital Gains", ticker) not in raw.columns:
                        bars["Capital Gains", ticker] = 0.0
                validate_bars(bars, request)
                return bars
            except ValueError:
                if attempt == 1:
                    raise


def load_bars(config):
    request = {
        "schema": 1,
        "start": str(config["data_start"]),
        "end": str(config["end"]),
        "tickers": [*config["universe"], config["risk_free"]],
    }
    key = sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]
    CACHE.mkdir(exist_ok=True)
    snapshot = CACHE / f"yahoo_daily_{key}.zip"
    if not snapshot.exists():
        bars = download_bars(request)
        metadata = {"request": request, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with NamedTemporaryFile(dir=CACHE, suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with ZipFile(temporary_path, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr("metadata.json", json.dumps(metadata))
                archive.writestr("bars.csv", bars.to_csv())
            temporary_path.replace(snapshot)
        finally:
            temporary_path.unlink(missing_ok=True)
    with ZipFile(snapshot) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        csv_bytes = archive.read("bars.csv")
    if metadata["request"] != request:
        raise ValueError("The cached snapshot does not match the configured data request")
    bars = pd.read_csv(BytesIO(csv_bytes), header=[0, 1], index_col=0, parse_dates=True)
    validate_bars(bars, request)
    return bars, metadata["fetched_at"], sha256(csv_bytes).hexdigest()


def total_return_prices(bars):
    """Use a forward scale with ex-date distributions reinvested at that day's open."""
    opening = bars["Open"]
    distributions = bars["Dividends"] + bars["Capital Gains"]
    # Yahoo Open and Close already share a split-adjusted share basis.
    scale = (1 + distributions / opening).cumprod()
    return opening * scale, bars["Close"] * scale


def signature_features(closing, path_sessions, order):
    first = np.log(closing).diff(path_sessions).to_numpy()
    if order == 1:
        return first
    if order != 2:
        raise ValueError("signature_order must be 1 or 2")
    row, column = np.triu_indices(closing.shape[1])
    # The symmetric geometric second level is half the terminal increment outer product.
    second = 0.5 * first[:, row] * first[:, column]
    return np.column_stack([first, second])


def portfolio_weights(forecast, covariance, model):
    equal = np.full(len(forecast), 1 / len(forecast))
    # Optimize in percentage points to keep the gradient scale near one.
    expected = 100 * forecast
    penalty = 100 * model["risk_aversion"] * covariance

    def loss(weights):
        active = weights - equal
        return 0.5 * active @ penalty @ active - expected @ active

    def gradient(weights):
        return penalty @ (weights - equal) - expected

    result = minimize(
        loss, equal, jac=gradient, method="SLSQP",
        bounds=[(0, model["max_weight"])] * len(equal),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
        options={"ftol": 1e-10, "maxiter": 100},
    )
    if not result.success or not np.isclose(result.x.sum(), 1, rtol=0, atol=1e-8):
        raise RuntimeError(f"Portfolio optimization failed ({result.message})")
    return result.x


def estimate_weights(day, opening, daily_returns, features, model, lag):
    horizon = model["horizon_sessions"]
    last_start = day - lag - horizon
    first_start = last_start - model["fit_sessions"] + 1
    risk_start = day - model["risk_sessions"] + 1
    if first_start < model["path_sessions"] or risk_start < 1:
        raise ValueError("The configured warm-up is too short for the first signal")
    starts = np.arange(last_start, first_start - 1, -model["fit_stride"])[::-1]
    label_open = starts + lag
    targets = opening[label_open + horizon] / opening[label_open] - 1

    training = features[starts]
    center = training.mean(axis=0)
    scale = training.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (training - center) / scale
    shrinkage = LedoitWolf(assume_centered=True).fit(standardized)
    defect = shrinkage.covariance_ + 1e-10 * np.eye(training.shape[1])
    cross = standardized.T @ (targets - targets.mean(axis=0)) / len(starts)
    coefficients = np.linalg.solve(defect, (1 - shrinkage.shrinkage_) * cross)
    forecast = ((features[day] - center) / scale) @ coefficients

    asset_risk = LedoitWolf().fit(daily_returns[risk_start:day + 1]).covariance_ * horizon
    return portfolio_weights(forecast, asset_risk, model)


def monthly_targets(opening, closing, config, model):
    dates = closing.index
    months = dates.to_period("M")
    signal_days = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])
    start = pd.Timestamp(config["start"])
    lag = config["lag_sessions"]
    if model is None:
        equal = np.full(closing.shape[1], 1 / closing.shape[1])
    else:
        features = signature_features(closing, model["path_sessions"], model["signature_order"])
        opening_values = opening.to_numpy()
        daily_returns = closing.pct_change(fill_method=None).to_numpy()
    targets = {}
    for day in signal_days:
        fill = day + lag
        if dates[day] < start or fill >= len(dates):
            continue
        weights = equal if model is None else estimate_weights(day, opening_values, daily_returns, features, model, lag)
        targets[fill] = weights
    return targets


def rebalance(positions, cash, opening, weights, buy_multiplier, sell_multiplier):
    open_values = positions * opening
    before = cash + open_values.sum()
    changes = before * weights - open_values
    changes[np.abs(changes) < before * 1e-12] = 0
    sells = np.maximum(-changes, 0)
    buys = np.maximum(changes, 0)
    cash += sells.sum() * sell_multiplier
    needed = buys.sum() * buy_multiplier
    if needed > cash:
        buys *= cash / needed
    cash -= buys.sum() * buy_multiplier
    if cash < -before * 1e-12:
        raise RuntimeError("Open execution exceeded the available cash")
    notionals = buys + sells
    return (open_values - sells + buys) / opening, max(cash, 0.0), notionals.sum() / before, np.count_nonzero(notionals)


def simulate(opening, closing, config, model):
    targets = monthly_targets(opening, closing, config, model)
    first_day = closing.index.searchsorted(pd.Timestamp(config["start"]))
    cost = config["cost_bps"] / 10_000
    slippage = config["slippage_bps"] / 10_000
    if not 0 <= cost < 1 or not 0 <= slippage < 1:
        raise ValueError("Each friction rate must lie in [0, 10000) bps")
    buy_multiplier = (1 + slippage) * (1 + cost)
    sell_multiplier = (1 - slippage) * (1 - cost)
    positions = np.zeros(closing.shape[1])
    cash = previous_value = 1.0
    opening_values, closing_values = opening.to_numpy(), closing.to_numpy()
    records = []
    for day in range(first_day, len(closing)):
        turnover, trades = 0.0, 0
        if day in targets:
            positions, cash, turnover, trades = rebalance(
                positions, cash, opening_values[day], targets[day], buy_multiplier, sell_multiplier,
            )
        value = cash + positions @ closing_values[day]
        records.append((value / previous_value - 1, turnover, trades, cash))
        previous_value = value
    return pd.DataFrame(records, index=closing.index[first_day:], columns=["return", "turnover", "trades", "cash"])


def performance(daily_results, risk_free_returns):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise ValueError("Each reporting interval needs at least two sessions")
    excess = returns - risk_free_returns.loc[returns.index]
    compounded = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    periods_per_year = SESSIONS_PER_YEAR / len(returns)
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    annual_scale = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(compounded[-1] ** periods_per_year - 1),
        "ann_vol": float(volatility * annual_scale),
        "sharpe": float(returns.mean() / volatility * annual_scale) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_volatility * annual_scale) if excess_volatility > 0 else None,
        "max_drawdown": float((compounded / np.maximum.accumulate(compounded) - 1).min()),
        "total_return": float(compounded[-1] - 1),
        "turnover": float(daily_results["turnover"].sum() * periods_per_year),
        "n_trades": int(daily_results["trades"].sum()),
    }


def periods(daily_results, risk_free_returns, split):
    before = daily_results.index < split
    return {
        "is": performance(daily_results.loc[before], risk_free_returns),
        "oos": performance(daily_results.loc[~before], risk_free_returns),
        "full": performance(daily_results, risk_free_returns),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    validate_config(config)
    bars, fetched_at, digest = load_bars(config)
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print(f"DATA_SHA256: {digest}")
    print("PRICE_ADJUSTMENT: Yahoo split-adjusted Open/Close with forward total return from ex-date reinvestment at Open")
    all_opening, all_closing = total_return_prices(bars)
    opening, closing = all_opening[config["universe"]], all_closing[config["universe"]]
    risk_free_returns = all_closing[config["risk_free"]].pct_change(fill_method=None)
    split = pd.Timestamp(config["oos_start"])
    is_mask = opening.index < split
    is_opening, is_closing = opening.loc[is_mask], closing.loc[is_mask]
    model = config["model"]
    robustness = config["robustness"]

    official_is = simulate(is_opening, is_closing, config, model)
    print("ROBUSTNESS: IS only, official parameters fixed before evaluation")
    print(f"ROBUSTNESS_OFFICIAL_IS: {json.dumps(performance(official_is, risk_free_returns), allow_nan=False)}")
    cases = [
        (f"path_sessions={window}", config, {**model, "path_sessions": window})
        for window in robustness["path_sessions"]
    ]
    alternative = robustness["alternative_signature_order"]
    cases.append((f"signature_order={alternative}", config, {**model, "signature_order": alternative}))
    cost_factor = robustness["cost_multiplier"]
    high_cost = {**config, "cost_bps": config["cost_bps"] * cost_factor,
                 "slippage_bps": config["slippage_bps"] * cost_factor}
    cases.append((f"cost_and_slippage_multiplier={cost_factor:g}", high_cost, model))
    tried = {(tuple(sorted(model.items())), config["cost_bps"], config["slippage_bps"])}
    for name, case_config, case_model in cases:
        trial = (tuple(sorted(case_model.items())), case_config["cost_bps"], case_config["slippage_bps"])
        if trial in tried:
            continue
        trial_results = simulate(is_opening, is_closing, case_config, case_model)
        tried.add(trial)
        metrics = performance(trial_results, risk_free_returns)
        print(f"ROBUSTNESS_IS {name}: {json.dumps(metrics, allow_nan=False)}")

    daily_results = simulate(opening, closing, config, model)
    pd.testing.assert_frame_equal(daily_results.loc[daily_results.index < split], official_is, check_exact=True)
    benchmark_results = simulate(opening, closing, config, None)
    metrics = periods(daily_results, risk_free_returns, split)
    benchmark = periods(benchmark_results, risk_free_returns, split)
    if metrics["is"]["n_trades"] == 0 or metrics["oos"]["n_trades"] == 0:
        raise RuntimeError("The official strategy must have fills in both periods")

    print(f"PARAM_COMBOS_TRIED: {len(tried)}")
    print("LOOKAHEAD_SELF_CHECK: later-open fills, completed labels, split-adjusted units with forward ex-date distributions, identical IS prefix")
    print("STATUS: 已回测，待审查")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(metrics[period], allow_nan=False)}")
        print(f"BENCHMARK_{period.upper()}: {json.dumps(benchmark[period], allow_nan=False)}")
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": daily_results.index[0].date().isoformat(), "end": daily_results.index[-1].date().isoformat()},
        **metrics,
        "benchmark": {"name": "monthly_equal_weight_etfs", **benchmark["full"]},
    }
    print(f"RESULTS_JSON: {json.dumps(result, allow_nan=False)}")


if __name__ == "__main__":
    signal.alarm(840)
    with threadpool_limits(limits=1, user_api="blas"):
        main()
