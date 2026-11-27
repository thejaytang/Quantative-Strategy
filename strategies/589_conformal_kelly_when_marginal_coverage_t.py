"""589_conformal_kelly_when_marginal_coverage_t
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/589_conformal_kelly_when_marginal_coverage_t.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 25a9179813de3addbbaee65a8c08579eb6e27d08 on 2026-11-27 19:09 Europe/Oslo, exit 0, output sha256 634cbddf17b1dbd2579ba2c86799fbc5a227e1b26d6d8aeaa9ef1f69c9ce084a
Data source: 通过 yfinance 获取 Yahoo Finance 免费公开日线开盘价、收盘价和公司行动数据。策略资产为 SPY、QQQ、DIA、MDY、GLD、SLV、USO、DBC。BIL 日总回报仅用于计算超额夏普。
Data fetched: 2026-11-27T18:09:14+00:00
Sample period: 请求行情自 2006-05-01 至 2024-09-20，含末日。2015-12-31 及以前用于预热。日志记录的实际回测交易日为 2016-01-04 至 2024-09-20。   data_range (from output): {'start': '2016-01-04', 'end': '2024-09-20', 'history_start': '2006-05-01', 'is_end': '2021-12-31', 'oos_start': '2022-01-03'}
IS/OOS split: 实际交易日 IS 为 2016-01-04 至 2021-12-31（含当日），OOS 为 2022-01-03 至 2024-09-20。OOS 承接 IS 末日持仓，并继续用固定超参数更新预测及校准。敏感性变体只在截断至 IS 末日的数据上运行。
Costs and slippage: 默认成本 5 bps，滑点按固定 2 bps 建模。开盘再平衡时，按各资产买卖名义金额的绝对值合计扣 7 bps，目标市值按扣费后的组合价值求解。滑点以费用计入，模拟成交价采用开盘价。IS 高成本变体为成本 20 bps 加滑点 2 bps，共 22 bps。
Signal lag: 交易日 t 收盘后生成目标权重，向后移一条交易日记录，在下一交易日开盘模拟成交。
Price adjustment: yfinance split-adjusted OHLC, cash distributions reinvested at ex-date open
Survivorship bias: 八只基金为论文选定且现有历史数据可得的事后固定池，存在存续偏差及资产范围选择偏差，缺少包含退市基金的历史可选池。Yahoo 历史行情可能修订，且缺少历史版本数据。
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.16682261756643224, ann_vol=0.15468113778826573, sharpe=1.0756143149699449, sharpe_excess=1.023227714733401, max_drawdown=-0.29352543847400747, total_return=1.5221048708819684, turnover=5.657790090776324, n_trades=10460
    oos: cagr=0.09963122030141469, ann_vol=0.15257314728034868, sharpe=0.6987659279347408, sharpe_excess=0.4563013350262997, max_drawdown=-0.19889255539506667, total_return=0.2935780314119667, turnover=6.295744869351077, n_trades=4896
    full: cagr=0.1454769414811905, ann_vol=0.1540031159414119, sharpe=0.9595416105646994, sharpe_excess=0.8485148026708655, max_drawdown=-0.29352543847400747, total_return=2.2625394538900254, turnover=5.856387681371837, n_trades=15356
    benchmark: cagr=0.11654045184225192, ann_vol=0.15380490592381102, sharpe=0.7943755399026347, sharpe_excess=0.6831303335418966, max_drawdown=-0.32757519256945455, total_return=1.6110295348379693, turnover=1.9329261824359898, n_trades=17552
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 589_conformal_kelly_when_marginal_coverage_t\nuniverse: [SPY, QQQ, DIA, MDY, GLD, SLV, USO, DBC]\nrisk_free_ticker: BIL\nhistory_start: '2006-05-01'\nis_start: '2016-01-01'\nis_end: '2021-12-31'\nend: '2024-09-20'\nexecution_lag: 1\ncost_bps: 5.0\nslippage_bps: 2.0\n\nmodel:\n  horizons: [12, 16, 21, 27, 34]\n  reference_horizon: 21\n  momentum_windows: [21, 63, 252]\n  volatility_span: 20\n  ridge_penalty: 10.0\n  first_fit_after: 750\n  refit_sessions: 21\n  calibration_window: 500\n  alpha: 0.25\n  anchor_weight: 0.30\n  minimum_half_width: 0.000001\n  scale_mode: conformal\n  kelly_fraction: 0.15\n  per_asset_cap: 0.75\n  gross_cap: 1.0\n  stagger_sessions: 5\n\nrobustness:\n  calibration_windows: [400, 600]\n  alphas: [0.20, 0.30]\n  higher_cost_bps: 20.0\n  alternative_scale: rolling_volatility\n"
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
"""Strategy 589: conformal interval widths for cash-funded fractional Kelly sizing.

Paper and private replication details: https://arxiv.org/abs/2608.01494
https://arxiv.org/html/2608.01494
Open-source research references (ideas only, independent implementation):
CQR (MIT): https://github.com/yromano/cqr/blob/master/nonconformist/nc.py
MEYNIMF (MIT):
https://github.com/dungnotnull/everything-you-need-in-modern-finance-agent/blob/main/portfolio/risk/conformal_sizing.py
https://github.com/dungnotnull/everything-you-need-in-modern-finance-agent/blob/main/openbb_extensions/meynimf_risk/rules.py
MAPIE package and tutorial: https://pypi.org/project/MAPIE/
https://mapie.readthedocs.io/en/v0.9.1/examples_regression/4-tutorials/plot_cqr_tutorial.html
QuantConnect: https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/position-sizing
Price history: https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
from statistics import NormalDist
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
ANNUAL_SESSIONS = 252


def price_panels(raw, config):
    """Align the strategy universe independently of the BIL inception date."""
    symbols = config["universe"]
    risk_free = config["risk_free_ticker"]
    closes = raw["Close"]
    if not closes.index.is_unique or not closes.index.is_monotonic_increasing:
        raise ValueError("Yahoo returned duplicate or unordered sessions")
    exchange_dates = closes.loc[closes[symbols].notna().any(axis=1)].index
    if exchange_dates.empty or exchange_dates[-1] != pd.Timestamp(config["end"]):
        raise ValueError("The requested final trading session is missing")
    if exchange_dates[0] > pd.Timestamp(config["history_start"]) + pd.offsets.BDay(2):
        raise ValueError("The requested history is incomplete")

    def total_return_prices(columns, dates):
        opens = raw["Open"].loc[dates, columns]
        ends = closes.loc[dates, columns]
        if not all(np.isfinite(prices).all().all() and (prices > 0).all().all()
                   for prices in (opens, ends)):
            raise ValueError(f"Missing or invalid daily quotes for {columns}")
        distributions = raw["Dividends"].loc[dates, columns].fillna(0.0)
        if "Capital Gains" in raw.columns.get_level_values(0):
            gains = raw["Capital Gains"].reindex(index=dates, columns=columns).fillna(0.0)
            distributions = distributions + gains
        if not np.isfinite(distributions).all().all() or (distributions < 0).any().any():
            raise ValueError(f"Invalid cash distributions for {columns}")
        # Yahoo OHLC already share a split-adjusted price unit. Reinvest each
        # distribution in that same unit at the ex-date open.
        factor = (1.0 + distributions / opens).cumprod()
        return opens * factor, ends * factor

    opens, ends = total_return_prices(symbols, exchange_dates)
    # Include the preceding close to measure the first IS session's BIL return.
    first_session = exchange_dates.searchsorted(pd.Timestamp(config["is_start"]))
    if not 0 < first_session < len(exchange_dates):
        raise ValueError("IS must start after the price warm-up and before the end date")
    _, bil = total_return_prices([risk_free], exchange_dates[first_session - 1:])
    return opens, ends, bil[risk_free].pct_change().iloc[1:]


def load_prices(config):
    CACHE_DIR.mkdir(exist_ok=True)
    tickers = [*config["universe"], config["risk_free_ticker"]]
    start = config["history_start"]
    end = (pd.Timestamp(config["end"]) + pd.Timedelta(days=1)).date().isoformat()
    key = sha256(json.dumps(["yfinance_daily_actions_v1", tickers, start, end]).encode())
    cache_path = CACHE_DIR / f"prices_{key.hexdigest()[:16]}.csv"
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as handle:
            fetched_at = datetime.fromisoformat(handle.readline().strip()).isoformat()
            raw = pd.read_csv(handle, header=[0, 1], index_col=0, parse_dates=True)
        return raw, price_panels(raw, config), fetched_at

    with TemporaryDirectory(prefix=".yf-", dir=CACHE_DIR) as work_dir:
        yf.set_tz_cache_location(work_dir)
        raw = yf.download(tickers, start=start, end=end, interval="1d", timeout=15,
                          threads=True, progress=False, ignore_tz=True,
                          auto_adjust=False, actions=True, repair=False)
        if raw is None or raw.empty:
            raise RuntimeError("Yahoo returned no daily data")
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        panels = price_panels(raw, config)
        pending = Path(work_dir) / "prices.csv"
        with pending.open("w", encoding="utf-8") as handle:
            handle.write(fetched_at + "\n")
            raw.to_csv(handle)
        os.replace(pending, cache_path)
    return raw, panels, fetched_at


def ridge_forecasts(closes, model):
    """Make historical forecasts at each close with an expanding training set."""
    returns = closes.pct_change()
    features = [closes.pct_change(window) for window in model["momentum_windows"]]
    features.append(returns.ewm(span=model["volatility_span"], adjust=False,
                                min_periods=model["volatility_span"]).std())
    features = np.stack(features, axis=-1)
    complete_rows = np.isfinite(features).all(axis=(1, 2))
    if not complete_rows.any():
        raise ValueError("Insufficient data for ridge features")
    first_features = int(np.flatnonzero(complete_rows)[0])
    first_fit = model["first_fit_after"]
    refit = model["refit_sessions"]
    horizons = model["horizons"]
    if first_fit - max(horizons) <= first_features or not complete_rows[first_features:].all():
        raise ValueError("Ridge fitting requires a complete feature history and mature labels")
    penalty = model["ridge_penalty"] * np.eye(features.shape[-1])
    forecasts = {}
    for horizon in horizons:
        landed = returns.rolling(horizon).sum().to_numpy()
        prediction = np.full(closes.shape, np.nan)
        for row in range(first_fit, len(closes), refit):
            # Feature row j pairs with the return sum ending at close j + H.
            # The newest training label here ends at the current close.
            fit_features = features[first_features:row - horizon + 1]
            fit_labels = landed[first_features + horizon:row + 1]
            feature_mean = fit_features.mean(axis=0)
            label_mean = fit_labels.mean(axis=0)
            centered = fit_features - feature_mean
            covariance = np.einsum("tai,taj->aij", centered, centered)
            response = np.einsum("tai,ta->ai", centered, fit_labels - label_mean)
            coefficients = np.linalg.solve(covariance + penalty, response[..., None])[..., 0]
            batch = slice(row, row + refit)
            prediction[batch] = label_mean + np.einsum(
                "tai,ai->ta", features[batch] - feature_mean, coefficients
            )
        forecasts[horizon] = pd.DataFrame(prediction, index=closes.index, columns=closes.columns)
    reference_target = returns.rolling(model["reference_horizon"]).sum()
    return forecasts, reference_target


def interval_half_width(forecast, landed_target, model):
    # A score enters calibration when its label lands, even if the forecast
    # comes from an older fit than the one used on the landing date.
    scores = (landed_target - forecast.shift(model["reference_horizon"])).abs()
    quantile = 1.0 - model["alpha"]
    rolling = scores.rolling(model["calibration_window"]).quantile(quantile)
    expanding = scores.expanding().quantile(quantile)
    anchors = expanding.iloc[model["first_fit_after"]::model["refit_sessions"]]
    anchor = anchors.reindex(scores.index).ffill()
    anchor_weight = model["anchor_weight"]
    width = rolling.pow(1.0 - anchor_weight) * anchor.pow(anchor_weight)
    return width.clip(lower=model["minimum_half_width"])


def target_weights(forecasts, landed_target, model):
    reference = model["reference_horizon"]
    numerator = 0.0
    denominator = 0.0
    for horizon, prediction in forecasts.items():
        scaled = prediction * (reference / horizon)
        width = interval_half_width(scaled, landed_target, model)
        precision = 1.0 / width.pow(2)
        numerator = numerator + precision * scaled
        denominator = denominator + precision
    ensemble = numerator / denominator
    width = interval_half_width(ensemble, landed_target, model)
    if model["scale_mode"] == "conformal":
        scale = width / NormalDist().inv_cdf(1.0 - model["alpha"] / 2.0)
    elif model["scale_mode"] == "rolling_volatility":
        scale = landed_target.rolling(model["calibration_window"]).std()
        scale = scale.clip(lower=model["minimum_half_width"]).where(width.notna())
    else:
        raise ValueError("Unknown scale_mode")
    requested = (model["kelly_fraction"] * ensemble / scale.pow(2)).clip(
        lower=0.0, upper=model["per_asset_cap"]
    )
    staggered = requested.rolling(model["stagger_sessions"]).mean()
    gross = staggered.sum(axis=1, min_count=len(staggered.columns))
    scaled_weights = staggered.div((gross / model["gross_cap"]).clip(lower=1.0), axis=0)
    return scaled_weights, ensemble, width, gross.gt(model["gross_cap"])


def rebalance_open(open_values, value_before, weights, friction):
    """Solve target dollar values using portfolio value after open execution costs."""
    # Solve v + friction * sum(abs(w*v - old_values)) = value_before.
    # The left hand side is increasing because friction * sum(w) < 1.
    value_after = value_before
    for _ in range(16):
        desired = weights * value_after
        trade_values = np.abs(desired - open_values)
        next_value = value_before - friction * trade_values.sum()
        if abs(next_value - value_after) <= 1e-13 * value_before:
            cash = next_value - desired.sum()
            if cash < -1e-12 * value_before:
                raise RuntimeError("Opening rebalance overspent cash")
            return desired, max(cash, 0.0), trade_values
        value_after = next_value
    raise RuntimeError("Opening rebalance did not converge")


def simulate(opens, closes, targets, config):
    opens = opens.loc[config["is_start"]:]
    closes = closes.loc[opens.index]
    lagged = targets.shift(config["execution_lag"]).loc[opens.index]
    if (not np.isfinite(lagged).all().all() or (lagged < 0).any().any()
            or (lagged.sum(axis=1) > 1.0 + 1e-12).any()):
        raise ValueError("Every opening session requires a valid, cash-funded target")
    friction = (config["cost_bps"] + config["slippage_bps"]) / 10_000.0
    positions = np.zeros(len(opens.columns))
    cash = value_at_close = 1.0
    daily_results = []
    for open_price, close_price, weights in zip(opens.to_numpy(), closes.to_numpy(),
                                                lagged.to_numpy(), strict=True):
        open_values = positions * open_price
        value_before = open_values.sum() + cash
        desired, cash, trade_values = rebalance_open(open_values, value_before, weights, friction)
        positions = desired / open_price
        next_close_value = positions @ close_price + cash
        open_gross = desired.sum() / (desired.sum() + cash)
        daily_results.append((next_close_value / value_at_close - 1.0,
                              trade_values.sum() / value_before,
                              int(np.count_nonzero(trade_values > 1e-12 * value_before)),
                              cash, open_gross))
        value_at_close = next_close_value
    return pd.DataFrame(daily_results, index=opens.index,
                        columns=["return", "turnover", "n_trades", "cash", "open_gross"])


def metrics(daily_results, risk_free):
    returns = daily_results["return"].to_numpy()
    excess = returns - risk_free.loc[daily_results.index].to_numpy()
    wealth = np.r_[1.0, np.cumprod(1.0 + returns)]
    volatility = float(returns.std(ddof=1))
    excess_vol = float(excess.std(ddof=1))
    annualizer = np.sqrt(ANNUAL_SESSIONS)
    return {
        "cagr": float(wealth[-1] ** (ANNUAL_SESSIONS / len(returns)) - 1.0),
        "ann_vol": float(annualizer * volatility),
        "sharpe": float(annualizer * returns.mean() / volatility) if volatility > 0 else None,
        "sharpe_excess": float(annualizer * excess.mean() / excess_vol) if excess_vol > 0 else None,
        "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0)),
        "total_return": float(wealth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].mean() * ANNUAL_SESSIONS),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def return_periods(daily_results, split):
    return {"is": daily_results.loc[:split], "oos": daily_results.loc[daily_results.index > split],
            "full": daily_results}


def coverage(ensemble, width, landed_target, dates, horizon):
    # Count only forecasts whose origin and outcome are both inside the period.
    origins = ensemble.index.to_series().shift(horizon)
    matured = dates[origins.loc[dates] >= dates[0]]
    error = (landed_target - ensemble.shift(horizon)).loc[matured].abs()
    scales = width.shift(horizon).loc[matured]
    valid = scales.notna() & error.notna()
    inside = error.le(scales) & valid
    return {"coverage": float(inside.to_numpy().sum() / valid.to_numpy().sum()),
            "n_asset_outcomes": int(valid.to_numpy().sum())}


def emit(label, values):
    print(f"{label}: {json.dumps(values, ensure_ascii=False, allow_nan=False)}")


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    model = config["model"]
    split = pd.Timestamp(config["is_end"])
    if not (pd.Timestamp(config["is_start"]) < split < pd.Timestamp(config["end"])
            and isinstance(config["execution_lag"], int) and config["execution_lag"] >= 1
            and 0.0 < model["per_asset_cap"] <= 1.0 and 0.0 < model["gross_cap"] <= 1.0):
        raise ValueError("Invalid evaluation dates, lag, or cash-funded exposure caps")
    controls = config["robustness"]
    if not (0.0 <= config["cost_bps"] <= controls["higher_cost_bps"] <= 500.0
            and 0.0 <= config["slippage_bps"] <= 500.0):
        raise ValueError("Cost or slippage outside the supported basis-point range")
    raw, (opens, closes, risk_free), fetched_at = load_prices(config)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance split-adjusted OHLC, cash distributions reinvested at ex-date open")
    print("ROBUSTNESS: IS only, fixed variants")

    is_config = {**config, "end": split.date().isoformat()}
    is_opens, is_closes, is_rf = price_panels(raw.loc[:split], is_config)
    is_forecasts, is_target = ridge_forecasts(is_closes, model)
    is_weights, is_ensemble, is_width, _ = target_weights(is_forecasts, is_target, model)
    is_results = simulate(is_opens, is_closes, is_weights, config)
    variants = [
        (f"calibration_window_{window}", {"calibration_window": window}, {})
        for window in controls["calibration_windows"]
    ]
    variants.extend((f"alpha_{alpha}", {"alpha": alpha}, {}) for alpha in controls["alphas"])
    variants.extend([
        ("higher_cost", {}, {"cost_bps": controls["higher_cost_bps"]}),
        ("rolling_volatility", {"scale_mode": controls["alternative_scale"]}, {}),
    ])
    emit("ROBUSTNESS_DEFAULT", metrics(is_results, is_rf))
    for name, model_change, execution_change in variants:
        weights = is_weights
        if model_change:
            weights, _, _, _ = target_weights(is_forecasts, is_target, {**model, **model_change})
        trial = simulate(is_opens, is_closes, weights, {**config, **execution_change})
        emit(f"ROBUSTNESS_{name}", metrics(trial, is_rf))

    forecasts, landed_target = ridge_forecasts(closes, model)
    weights, ensemble, width, capped = target_weights(forecasts, landed_target, model)
    daily_results = simulate(opens, closes, weights, config)
    prefix_checks = [(is_opens, opens), (is_closes, closes), (is_rf, risk_free),
                     (is_ensemble, ensemble), (is_width, width),
                     (is_weights, weights), (is_results, daily_results)]
    prefix_checks.extend((is_forecasts[horizon], forecasts[horizon]) for horizon in forecasts)
    for truncated, full_series in prefix_checks:
        np.testing.assert_allclose(truncated, full_series.loc[truncated.index],
                                   rtol=1e-11, atol=1e-12, equal_nan=True)
    print("LOOK_AHEAD_SELF_CHECK: IS prefix matches independently truncated prices, forecasts, weights and fills")

    equal_weight = pd.DataFrame(1.0 / len(closes.columns), index=closes.index, columns=closes.columns)
    benchmark_results = simulate(opens, closes, equal_weight, config)
    periods = return_periods(daily_results, split)
    strategy_metrics = {label: metrics(frame, risk_free) for label, frame in periods.items()}
    benchmark_metrics = {label: metrics(frame, risk_free)
                         for label, frame in return_periods(benchmark_results, split).items()}
    if min(strategy_metrics["is"]["n_trades"], strategy_metrics["oos"]["n_trades"]) == 0:
        raise RuntimeError("The default strategy needs actual simulated trades in both periods")

    for label, frame in periods.items():
        details = coverage(ensemble, width, landed_target, frame.index, model["reference_horizon"])
        details["gross_cap_hit_rate"] = float(capped.shift(config["execution_lag"]).loc[frame.index].mean())
        emit(f"DIAGNOSTICS_{label.upper()}", details)
        emit(label.upper(), strategy_metrics[label])
        emit(f"BENCHMARK_{label.upper()}", benchmark_metrics[label])
    for year, frame in periods["oos"].groupby(periods["oos"].index.year):
        emit(f"OOS_COVERAGE_{year}", coverage(ensemble, width, landed_target, frame.index,
                                             model["reference_horizon"]))
    print(f"CASH_AND_EXPOSURE: min_cash={daily_results['cash'].min():.12g}, "
          f"max_open_gross={daily_results['open_gross'].max():.12g}")
    print(f"PARAM_COMBOS_TRIED: {1 + len(variants)}")
    print("STATUS: 已回测")
    emit("RESULTS_JSON", {
        "strategy": config["strategy"],
        "data_range": {"start": daily_results.index[0].date().isoformat(),
                       "end": daily_results.index[-1].date().isoformat(),
                       "history_start": closes.index[0].date().isoformat(),
                       "is_end": periods["is"].index[-1].date().isoformat(),
                       "oos_start": periods["oos"].index[0].date().isoformat()},
        **strategy_metrics,
        "benchmark": {"name": "Daily equal-weight ETF universe", **benchmark_metrics["full"],
                      **benchmark_metrics},
    })


if __name__ == "__main__":
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)
