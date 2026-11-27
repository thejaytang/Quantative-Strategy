"""549_drawdown_risk_beyond_brownian_motion_a_m
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/549_drawdown_risk_beyond_brownian_motion_a_m.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c0a8f76240a0431b019ab49c34040a95a765860a on 2026-11-27 10:04 Europe/Oslo, exit 0, output sha256 8095425c9fa90e2fdf7110c58fa20eb88cd1b5531201e66c8f19cf7344a87d42
Data source: Yahoo Finance（yfinance）的 SPY、BIL ETF 日收盘价，分别用于权益仓位和现金代理。优先读取 data_cache/yfinance_SPY_BIL_2008-01-01_2025-12-31.json，缓存缺失时下载。config.yaml:2-9，strategy.py:63-91，README.md:21。
Data fetched: 2026-11-27T09:04:52+00:00
Sample period: 实际回测评估期为 2010-01-04 至 2025-12-31。请求的数据起始日为 2008-01-01，评估前的数据用于信号预热。config.yaml:5-9，strategy.py:276-285，README.md:22，logs/run_20261127_092340.log:17。   data_range (from output): {'start': '2010-01-04', 'is_end': '2017-12-29', 'oos_start': '2018-01-02', 'end': '2025-12-31'}
IS/OOS split: 样本内 2010-01-04 至 2017-12-29，样本外 2018-01-02 至 2025-12-31，以 2017-12-31 为配置切分点。默认参数下先连续模拟，再分段统计，跨期保留仓位；六组稳健性对照仅使用样本内和预热数据。config.yaml:5-9，strategy.py:216-224，strategy.py:279-295，README.md:22，README.md:30，logs/run_20261127_092340.log:17。
Costs and slippage: 每条买卖腿计 1 bp 成本和 2 bp 固定滑点，合并为每腿 3 bp。首次买入仅计买入腿，ETF 切换先从卖出所得扣除卖出费用，再用剩余资金支付买入金额及买入费用，持仓不变时不收费。SPY 买入持有基准采用同一费用模型。config.yaml:10-13，strategy.py:168-188，strategy.py:269-274，strategy.py:285-289，README.md:24。
Signal lag: 每周首个有价格记录的交易日收盘 t 用截至该收盘的数据计算信号，在下一交易日收盘执行。执行当日截至收盘的收益归原仓位，新仓位自执行收盘后开始计收益。首个评估日收盘按当时已生效的滞后目标建仓。config.yaml:10-11，strategy.py:137-176，strategy.py:285-286，README.md:23。
Price adjustment: yfinance auto_adjust total-return close
Survivorship bias: SPY 和 BIL 是事后选定且持续存续的两只 ETF，存在资产选择与存续偏差。回测使用 ETF 历史价格，未用当前指数成分股名单重建历史。config.yaml:2-4，strategy.py:78-93，strategy.py:253-257，README.md:26。
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.08123564274010064, ann_vol=0.13529425056055372, sharpe=0.6452106695609724, sharpe_excess=0.6406203098051639, max_drawdown=-0.24275444114299882, total_return=0.8662035700025075, turnover=7.634073414339336, n_trades=61
    oos: cagr=0.08361651114511881, ann_vol=0.139940858931897, sharpe=0.6443375483436078, sharpe_excess=0.47086967812803443, max_drawdown=-0.27605521782373477, total_return=0.8980730980825375, turnover=5.512021182307666, n_trades=44
    full: cagr=0.0824248306657227, ann_vol=0.13761893221178903, sharpe=0.6447550919964501, sharpe_excess=0.5543922136076288, max_drawdown=-0.2760552178237349, total_return=2.5421907917673563, turnover=6.573574647287724, n_trades=105
    benchmark  full: cagr=0.1392392082804994, ann_vol=0.17207623498878946, sharpe=0.8440204110848967, sharpe_excess=0.7715614794331113, max_drawdown=-0.3371725559557691, total_return=7.017546249945816, turnover=0.06260547283131163, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 549_drawdown_risk_beyond_brownian_motion_a_m\nuniverse:\n  equity: SPY\n  cash_proxy: BIL\ndates:\n  data_start: "2008-01-01"\n  evaluation_start: "2010-01-01"\n  is_end: "2017-12-31"\n  end: "2025-12-31"\nexecution:\n  lag_sessions: 1\n  cost_bps: 1.0\n  slippage_bps: 2.0\nsignal:\n  lookback_days: 252\n  horizon_days: 63\n  quantile: 0.90\n  paths: 2048\n  seed: 164\n  hurst_scales: [1, 2, 4, 8, 16]\n  hurst_weight: 0.50\n  hurst_min: 0.40\n  hurst_max: 0.70\nrobustness:\n  lookbacks: [210, 294]\n  quantiles: [0.85, 0.95]\n  high_cost_multiplier: 2.0\n'
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
"""Weekly SPY/BIL momentum with a simulated drawdown threshold."""

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


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
TRADING_DAYS = 252


@dataclass(frozen=True)
class SignalSpec:
    lookback_days: int
    horizon_days: int
    quantile: float
    paths: int
    seed: int
    hurst_scales: tuple[int, ...]
    hurst_weight: float
    hurst_min: float
    hurst_max: float

    def validate(self):
        if (
            len(self.hurst_scales) < 2
            or tuple(sorted(set(self.hurst_scales))) != self.hurst_scales
            or self.hurst_scales[0] < 1
            or self.lookback_days <= self.hurst_scales[-1]
        ):
            raise ValueError("Hurst scales must be increasing and shorter than the lookback.")
        if not 0 <= self.hurst_weight <= 1 or not (
            0 < self.hurst_min <= 0.5 <= self.hurst_max < 1
        ):
            raise ValueError("Invalid Hurst bounds or estimation weight.")
        if self.horizon_days < 2 or self.paths < 2 or not 0 < self.quantile < 1:
            raise ValueError("Invalid drawdown horizon, path count, or quantile.")


def check_prices(prices, data_start, end):
    index = prices.index
    if prices.empty or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Yahoo prices are empty or their dates are invalid.")
    if not (
        data_start <= index[0] <= data_start + pd.Timedelta(days=7)
        and end - pd.Timedelta(days=7) <= index[-1] <= end
    ):
        raise ValueError("Yahoo prices do not cover the requested date range.")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Yahoo prices contain an incomplete or invalid trading session.")


def load_prices(tickers, data_start, end):
    CACHE.mkdir(exist_ok=True)
    start_date = data_start.date().isoformat()
    end_date = end.date().isoformat()
    cache_file = CACHE / f"yfinance_{'_'.join(tickers)}_{start_date}_{end_date}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        dates = pd.DatetimeIndex(cached["dates"], dtype="datetime64[ns]", name="date")
        prices = pd.DataFrame(cached["closes"], index=dates)
        prices = prices.loc[:, list(tickers)]
        check_prices(prices, data_start, end)
        return prices, cached["fetched_at"]

    with TemporaryDirectory(dir=CACHE, prefix=".yfinance-") as download_cache:
        yf.set_tz_cache_location(download_cache)
        daily = yf.download(
            list(tickers),
            start=start_date,
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d",
            auto_adjust=True,
            ignore_tz=True,
            threads=False,
            progress=False,
            timeout=15,
        )
    if daily is None or daily.empty:
        raise RuntimeError("yfinance returned no daily prices for " + ", ".join(tickers))
    prices = daily["Close"].loc[:, list(tickers)].rename_axis(index="date", columns=None)
    prices.index = prices.index.astype("datetime64[ns]")
    check_prices(prices, data_start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "fetched_at": fetched_at,
        "dates": prices.index.strftime("%Y-%m-%d").tolist(),
        "closes": prices.to_dict(orient="list"),
    }
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            json.dump(payload, temp, allow_nan=False)
            temp.close()
            temp_path.replace(cache_file)
        finally:
            temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def estimate_hurst(window, spec):
    if spec.hurst_weight == 0:
        return 0.5
    path = np.r_[0.0, np.cumsum(window - window.mean())]
    scales = spec.hurst_scales
    dispersions = np.array(
        [np.std(path[width:] - path[:-width], ddof=1) for width in scales]
    )
    if (dispersions <= 0).any():
        return 0.5
    raw_hurst = np.polyfit(np.log(scales), np.log(dispersions), 1)[0]
    shrunk = 0.5 + spec.hurst_weight * (raw_hurst - 0.5)
    return float(np.clip(shrunk, spec.hurst_min, spec.hurst_max))


def drawdown_limit(window, hurst, resample_indices, spec):
    # Use common uniform ranks, mapped to the current empirical return distribution.
    drift = window.mean()
    shocks = np.sort(window)[resample_indices] - drift
    scale = spec.horizon_days ** (hurst - 0.5)
    paths = np.cumsum(drift + shocks * scale, axis=1)
    # Include the zero log-return at the path's starting close.
    peaks = np.maximum(0.0, np.maximum.accumulate(paths, axis=1))
    return float(np.quantile((peaks - paths).max(axis=1), spec.quantile))


def execution_targets(prices, equity, cash_proxy, spec, lag):
    log_prices = np.log(prices)
    log_returns = log_prices.diff()
    equity_returns = log_returns[equity].to_numpy()
    excess = (log_returns[equity] - log_returns[cash_proxy]).to_numpy()
    equity_log_prices = log_prices[equity].to_numpy()

    # Week membership and its first observed session are known by that close.
    weeks = prices.index.to_period("W-SUN")
    first_sessions = np.flatnonzero(np.r_[True, weeks[1:] != weeks[:-1]])
    picks = pd.Series(pd.NA, index=prices.index, dtype="string")
    rng = np.random.default_rng(spec.seed)
    uniforms = rng.random((spec.paths, spec.horizon_days))
    resample_indices = (uniforms * spec.lookback_days).astype(np.intp)
    warmup = max(spec.lookback_days, spec.horizon_days)
    for day in first_sessions[first_sessions >= warmup]:
        past = slice(day + 1 - spec.lookback_days, day + 1)
        excess_window = excess[past]
        # A positive mean with nonzero dispersion is a positive trailing Sharpe.
        chosen = cash_proxy
        if excess_window.mean() > 0 and excess_window.std(ddof=1) > 0:
            window = equity_returns[past]
            hurst = estimate_hurst(window, spec)
            limit = drawdown_limit(window, hurst, resample_indices, spec)
            peak = equity_log_prices[day - spec.horizon_days : day + 1].max()
            if peak - equity_log_prices[day] < limit:
                chosen = equity
        picks.iloc[day] = chosen
    return picks.shift(lag).ffill()


def simulate(asset_returns, targets, cash_proxy, fee):
    if targets.isna().any():
        raise ValueError("Missing execution targets. Extend the signal warm-up.")
    # The old position earns the return through the execution close.
    prior = targets.shift()
    market_return = np.zeros(len(targets))
    for ticker in asset_returns.columns:
        in_asset = prior.eq(ticker).fillna(False).to_numpy(dtype=bool)
        market_return[in_asset] = asset_returns[ticker].to_numpy()[in_asset]

    retained = targets.eq(prior).fillna(False).to_numpy(dtype=bool)
    buy = ~retained
    sell = buy & prior.notna().to_numpy()
    # Size the purchase from available proceeds after both sides' charges.
    purchase_fraction = (1.0 - fee * sell) / (1.0 + fee)
    fill_factor = np.where(buy, purchase_fraction, 1.0)
    risk_free = asset_returns[cash_proxy].to_numpy(copy=True)
    risk_free[0] = 0.0
    return pd.DataFrame(
        {
            "return": (1.0 + market_return) * fill_factor - 1.0,
            "risk_free": risk_free,
            "turnover": sell + buy * purchase_fraction,
            "n_trades": buy.astype(int) + sell.astype(int),
        },
        index=targets.index,
    )


def metrics(daily_results):
    returns = daily_results["return"].to_numpy()
    excess = returns - daily_results["risk_free"].to_numpy()
    compounded = np.r_[1.0, np.cumprod(1.0 + returns)]
    vol = np.std(returns, ddof=1)
    excess_vol = np.std(excess, ddof=1)
    annualizer = np.sqrt(TRADING_DAYS)
    return {
        "cagr": float(compounded[-1] ** (TRADING_DAYS / len(returns)) - 1.0),
        "ann_vol": float(annualizer * vol),
        "sharpe": float(annualizer * returns.mean() / vol) if vol else None,
        "sharpe_excess": float(annualizer * excess.mean() / excess_vol) if excess_vol else None,
        "max_drawdown": float((compounded / np.maximum.accumulate(compounded) - 1.0).min()),
        "total_return": float(compounded[-1] - 1.0),
        "turnover": float(TRADING_DAYS * daily_results["turnover"].mean()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def by_period(daily_results, is_end):
    is_rows = daily_results.index <= is_end
    if min(int(is_rows.sum()), int((~is_rows).sum())) < 2:
        raise ValueError("Both evaluation periods must contain at least two sessions.")
    return {
        "is": metrics(daily_results.loc[is_rows]),
        "oos": metrics(daily_results.loc[~is_rows]),
        "full": metrics(daily_results),
    }


def run_robustness(prices, is_returns, tickers, spec, lag, cost, slippage, settings):
    print("ROBUSTNESS (IS only)")
    cases = {
        (replace(spec, lookback_days=lookback), cost, slippage): f"lookback={lookback}"
        for lookback in settings["lookbacks"]
    }
    cases.update({
        (replace(spec, quantile=quantile), cost, slippage): f"quantile={quantile}"
        for quantile in settings["quantiles"]
    })
    cases[(replace(spec, hurst_weight=0.0), cost, slippage)] = "H=0.5; empirical shocks"
    multiplier = settings["high_cost_multiplier"]
    cases[(spec, cost * multiplier, slippage * multiplier)] = f"cost and slippage x{multiplier:g}"
    cases.pop((spec, cost, slippage), None)
    for (scenario, case_cost, case_slippage), name in cases.items():
        scenario.validate()
        fee = (case_cost + case_slippage) * 1e-4
        if not 0 <= fee < 1:
            raise ValueError("Invalid robustness cost and slippage.")
        targets = execution_targets(prices, *tickers, scenario, lag)
        runs = simulate(is_returns, targets.reindex(is_returns.index), tickers[1], fee)
        print(f"  {name}  {json.dumps(metrics(runs), allow_nan=False)}")
    return len(cases)


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    tickers = config["universe"]["equity"], config["universe"]["cash_proxy"]
    if tickers != ("SPY", "BIL"):
        raise ValueError("The research universe is SPY and BIL.")
    equity, cash_proxy = tickers
    dates = config["dates"]
    data_start, start, is_end, end = (
        pd.Timestamp(dates[key])
        for key in ("data_start", "evaluation_start", "is_end", "end")
    )
    if not data_start < start < is_end < end:
        raise ValueError("Invalid warm-up, evaluation or split dates.")
    signal_settings = config["signal"].copy()
    signal_settings["hurst_scales"] = tuple(signal_settings["hurst_scales"])
    spec = SignalSpec(**signal_settings)
    spec.validate()
    execution = config["execution"]
    lag = execution["lag_sessions"]
    cost, slippage = execution["cost_bps"], execution["slippage_bps"]
    fee = (cost + slippage) * 1e-4
    if not isinstance(lag, int) or lag < 1 or not (0 <= fee < 1 and cost >= 0 and slippage >= 0):
        raise ValueError("Use a positive whole-session lag and nonnegative execution charges.")

    prices, fetched_at = load_prices(tickers, data_start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return close")
    returns = prices.pct_change(fill_method=None).loc[start:end]
    is_returns = returns.loc[:is_end]
    cases = run_robustness(
        prices.loc[:is_end], is_returns, tickers, spec, lag, cost, slippage, config["robustness"]
    )

    targets = execution_targets(prices, *tickers, spec, lag).reindex(returns.index)
    daily_results = simulate(returns, targets, cash_proxy, fee)
    strategy_periods = by_period(daily_results, is_end)
    benchmark_targets = pd.Series(equity, index=returns.index, dtype="string")
    benchmark = by_period(simulate(returns, benchmark_targets, cash_proxy, fee), is_end)
    data_range = {
        "start": returns.index[0].date().isoformat(),
        "is_end": is_returns.index[-1].date().isoformat(),
        "oos_start": returns.index[len(is_returns)].date().isoformat(),
        "end": returns.index[-1].date().isoformat(),
    }
    for label in ("is", "oos", "full"):
        print(f"STRATEGY {label.upper()} {json.dumps(strategy_periods[label], allow_nan=False)}")
        print(f"BENCHMARK {label.upper()} {json.dumps(benchmark[label], allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {1 + cases}")
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **strategy_periods,
        "benchmark": {"name": f"{equity} buy and hold", **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    main()
