"""448_continuous_cash_overlay_filters_for_a_st
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/448_continuous_cash_overlay_filters_for_a_st.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 6df04a4d89c55b7a1b162dbdcff0dafc1e2bd682 on 2026-11-27 08:11 Europe/Oslo, exit 0, output sha256 bc248716743e62b63290f676c6bb2777b3acd7381bbeff210a501d2ab806c93e
Data source: yfinance 日频收盘数据，ETF 为 QQQ、XLK、VGT、SPYG、VUG、SCHD、VYM、VTV、FDVV、COWZ、BIL、HYG、SHY，另有 ^VIX 指数。FRED 公开 CSV 提供 DGS3MO、DGS10、DAAA、DBAA。（config.yaml:3-14，strategy.py:44-71、74-88、293-299，README.md:9、15-17）
Data fetched: 2026-11-27T07:11:13+00:00
Sample period: 2020-01-02 至 2026-04-30（含首尾）。数据从 2013-01-01 起请求用于预热，ETF 收益从各所需 ETF 都有价格后的首个可计算日开始。（config.yaml:16-20，strategy.py:177-193、298-306，README.md:16，logs/run_20261127_073353.log:12）   data_range (from output): {'start': '2020-01-02', 'end': '2026-04-30'}
IS/OOS split: 固定切分。IS 从 2020-01-02 起，日历截止日为 2022-12-31。OOS 从 2023-01-03 至 2026-04-30。两个分支仅用 IS 评估参数，OOS 沿用选定参数并按日更新信号。策略最后一个 IS 收盘归零现金覆盖并按换手扣费，选定参数的信号最早在首个 OOS 收盘成交。（config.yaml:16-20，strategy.py:222-235、258-279、304-332，README.md:16、24）
Costs and slippage: 每次收盘调整按单边换手收取 10 bps 成本和 2 bps 滑点。单边换手为目标现金权重与收盘成交前漂移后现金权重之差的绝对值，成交前权益乘以 1 − 0.0012 × 单边换手。滑点按固定费率建模。风险篮子内部每日等权再平衡的成本未纳入。（config.yaml:22-27，strategy.py:185-188、230-236，README.md:18）
Signal lag: ETF 收盘 t 计算的现金覆盖信号在下一 ETF 交易日 t+1 的收盘成交，调整后的权重从该收盘至再下一收盘的收益区间开始生效。VIX 输入滞后 1 个 ETF 交易日，FRED 观测值对齐 BIL 交易日历后滞后 2 个交易日。（config.yaml:22-25，strategy.py:91-94、165-175、222-235，README.md:17）
Price adjustment: yfinance auto_adjust total-return for ETFs; VIX index level
Survivorship bias: 固定的增长和防御篮子采用事后已知且存续的 ETF，存在存续偏差和成分选择偏差。实现从全部所需 ETF 均有数据时开始计算，并对共同历史的价格缺口报错。（config.yaml:3-9，strategy.py:177-188，README.md:28）
Parameter combinations tried: 433
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.18178944003838438, ann_vol=0.18829900765048005, sharpe=0.9815801631500097, sharpe_excess=0.9517457003931556, max_drawdown=-0.18583238550959247, total_return=0.6505181900769017, turnover=3.189607868148608, n_trades=306
    oos: cagr=0.1890399049160485, ann_vol=0.133601608162248, sharpe=1.3635145468434402, sharpe_excess=1.021565856083687, max_drawdown=-0.18273436789360387, total_return=0.773635511817313, turnover=3.87803927216172, n_trades=388
    full: cagr=0.18558698378061722, ann_vol=0.16192926571269078, sharpe=1.1328022628669068, sharpe_excess=0.9683634282206043, max_drawdown=-0.18583238550959247, total_return=1.927417674820827, turnover=3.5507096234611466, n_trades=694
    benchmark  full: cagr=0.16985522111059415, ann_vol=0.21007111697526423, sharpe=0.8524156886053043, sharpe_excess=0.7255813263131303, max_drawdown=-0.33578568427478206, total_return=1.690798758687699, turnover=0.0, n_trades=0
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 448_continuous_cash_overlay_filters_for_a_st\n\nuniverse:\n  growth: [QQQ, XLK, VGT, SPYG, VUG]\n  defensive: [SCHD, VYM, VTV, FDVV, COWZ]\n  cash: BIL\n  vix: ^VIX\n  high_yield: HYG\n  treasury_etf: SHY\n  fred:\n    cash_yield: DGS3MO\n    ten_year_yield: DGS10\n    aaa_yield: DAAA\n    baa_yield: DBAA\n\ndates:\n  data_start: 2013-01-01\n  sample_start: 2020-01-02\n  is_end: 2022-12-31\n  sample_end: 2026-04-30\n\nexecution:\n  signal_lag_sessions: 1\n  vix_lag_sessions: 1\n  fred_lag_sessions: 2\n  cost_bps: 10\n  slippage_bps: 2\n\nmodel:\n  trading_days_per_year: 252\n  vix_percentile_window: 756\n  slow_min_history: 252\n  brake_min_history: 2\n  slow_clip_quantile: 0.01\n  material_cash: 0.30\n  exit_speed: 0.25\n  slow_cash_floor: 0.01\n\n  slow_grid:\n    core_mix: [0.50, 0.67]\n    rate_weight: [0.25, 0.50]\n    interaction_weight: [0.25, 0.50]\n    max_cash: [0.50, 0.75, 1.00]\n    scale: [0.75, 1.00, 1.25]\n    power: [2, 3, 4]\n  brake_grid:\n    core_mix: [0.50, 0.67]\n    rate_weight: [0.25, 0.50]\n    interaction_weight: [0.00, 0.25]\n    max_cash: [0.50, 0.75, 1.00]\n    scale: [0.50, 0.75, 1.00]\n    power: [3, 4, 5]\n\n  slow_selection_weights:\n    cagr: 0.30\n    max_drawdown: 0.25\n    calmar: 0.20\n    sharpe: 0.10\n    low_turnover: 0.10\n    low_cash: 0.05\n  brake_selection_weights:\n    cagr: 0.15\n    max_drawdown: 0.25\n    calmar: 0.15\n    low_turnover: 0.10\n    low_cash: 0.05\n'
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
"""Research backtest of the slow-tail and V-shape maximum cash overlay."""

import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
GRID_KEYS = ("core_mix", "rate_weight", "interaction_weight", "max_cash", "scale", "power")


def cache_path(source, identifiers, start, end):
    key = json.dumps([sorted(identifiers), str(start.date()), str(end.date())])
    return CACHE / f"{source}_{hashlib.sha256(key.encode()).hexdigest()[:16]}.csv"


def save_cache(frame, path):
    with NamedTemporaryFile(dir=CACHE, mode="w", suffix=".tmp", delete=False) as file:
        temporary_path = Path(file.name)
        try:
            frame.to_csv(file, index_label="date")
            file.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def read_cache(path):
    return pd.read_csv(path, index_col="date", parse_dates=["date"]).sort_index()


def load_prices(tickers, start, end):
    path = cache_path("yahoo_adjusted_close", tickers, start, end)
    if path.exists():
        return read_cache(path), path

    missing = set(tickers)
    prices = pd.DataFrame()
    with TemporaryDirectory(dir=CACHE, prefix="yahoo_session_") as session_cache:
        yf.set_tz_cache_location(session_cache)
        for _ in range(2):
            downloaded = yf.download(
                sorted(missing), start=start.date(), end=(end + pd.Timedelta(days=1)).date(),
                auto_adjust=True, progress=False, timeout=15,
            )
            if "Close" in downloaded:
                closes = downloaded["Close"].dropna(axis=1, how="all")
                prices = prices.join(closes, how="outer")
                missing.difference_update(closes.columns)
            if not missing:
                break
    if missing:
        raise RuntimeError(f"Missing Yahoo Finance data for {', '.join(sorted(missing))}")
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    prices = prices.sort_index().loc[start:end]
    if prices.empty or prices.index[-1] < end - pd.Timedelta(days=5):
        raise RuntimeError("Yahoo Finance data do not cover the requested end date")
    save_cache(prices, path)
    return prices, path


def load_fred(series_ids, start, end):
    path = cache_path("fred", series_ids, start, end)
    if path.exists():
        return read_cache(path), path

    query = urlencode({"id": ",".join(series_ids), "cosd": start.date(), "coed": end.date()})
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{query}"
    with urlopen(url, timeout=25) as response:
        data = pd.read_csv(BytesIO(response.read()), index_col=0, parse_dates=[0], na_values=".")
    data = data.sort_index().loc[start:end]
    data = data[series_ids].apply(pd.to_numeric, errors="raise")
    if data.empty or data.apply(pd.Series.last_valid_index).isna().any():
        raise RuntimeError("FRED did not return all required yield series")
    save_cache(data, path)
    return data, path


def lag_fred(fred, sessions, lag):
    # Align observation dates to the market calendar before applying the publication buffer.
    history = fred.reindex(fred.index.union(sessions)).sort_index().ffill(limit=7)
    return history.reindex(sessions).shift(lag)


def expanding_z(series, history, prior_only):
    expanding = series.expanding(min_periods=history)
    mean = expanding.mean()
    std = expanding.std(ddof=0)
    if prior_only:
        mean = mean.shift(1)
        std = std.shift(1)
    return (series - mean) / std.where(std > 1e-12)


def drawdown(returns):
    wealth = (1.0 + returns).cumprod()
    return wealth / wealth.cummax().clip(lower=1.0) - 1.0


def slow_features(risky, vix, vix_rank, credit_relative, yields, model):
    days = model["trading_days_per_year"]
    wealth = (1.0 + risky).cumprod()
    raw = pd.DataFrame({
        "cash_level": yields["cash_yield"],
        "cash_change": yields["cash_yield"].diff(21),
        "rate_change": yields["ten_year_yield"].diff(21),
        "drawdown": -drawdown(risky),
        "vix_rank": vix_rank,
        "vix_change": vix.pct_change(21, fill_method=None),
        "credit_riskoff": -credit_relative,
        "r_63": wealth.pct_change(63, fill_method=None),
        "r_126": wealth.pct_change(126, fill_method=None),
        "low_vol": -risky.rolling(63).std() * np.sqrt(days),
    })
    # The archived slow filter clips extremes. Estimate both cutoffs from past data.
    q = model["slow_clip_quantile"]
    history = model["slow_min_history"]
    prior = raw.expanding(min_periods=history)
    raw = raw.clip(prior.quantile(q).shift(1), prior.quantile(1.0 - q).shift(1))
    z = lambda series: expanding_z(series, history, prior_only=True)
    strength = z(0.5 * (raw["r_63"] + raw["r_126"]))
    calm_vol = z(raw["low_vol"])
    calm_vix = z(-raw["vix_rank"])
    cash_high = z(raw["cash_level"])
    headwind = z(raw["rate_change"])
    riskoff = z(raw["credit_riskoff"])
    depth = z(raw["drawdown"])
    compression = (
        z(strength * calm_vol) + z(strength * calm_vol * calm_vix)
        + z(strength * calm_vix) + z(cash_high * strength * calm_vix)
        + z(cash_high * strength * calm_vol)
    ) / 5.0
    path_stress = (
        z(headwind * z(raw["vix_change"])) + z(headwind * riskoff)
        + z(headwind * -strength) + z(depth * headwind * riskoff)
    ) / 4.0
    return z(z(raw["cash_change"])), compression, z(headwind), path_stress


def brake_features(risky, vix, vix_rank, credit_relative, yields, model):
    z = lambda series: expanding_z(series, model["brake_min_history"], prior_only=False)
    wealth = (1.0 + risky).cumprod()
    level = z(vix_rank)
    spike = z(vix.pct_change(10, fill_method=None))
    rate_relief = z(-yields["ten_year_yield"].diff(21))
    risky_loss = z(-wealth.pct_change(10, fill_method=None))
    riskoff = z(-credit_relative)
    spread_widening = z((yields["baa_yield"] - yields["aaa_yield"]).diff(21))
    rate_panic = 0.5 * (z(rate_relief * risky_loss * level) + z(rate_relief * riskoff))
    return z(level), z(spike), rate_panic, z(spike * spread_widening)


def build_panel(prices, fred, config):
    universe = config["universe"]
    execution = config["execution"]
    model = config["model"]
    cash = universe["cash"]
    hyg, shy = universe["high_yield"], universe["treasury_etf"]
    sessions = prices[cash].dropna().index
    prices = prices.reindex(sessions)
    yields = lag_fred(fred, sessions, execution["fred_lag_sessions"])
    vix = prices[universe["vix"]].shift(execution["vix_lag_sessions"])
    vix_rank = vix.rolling(model["vix_percentile_window"]).rank(pct=True)

    assets = universe["growth"] + universe["defensive"] + [cash, hyg, shy]
    first = max(prices[asset].first_valid_index() for asset in assets)
    asset_prices = prices.loc[first:, assets]
    if asset_prices.isna().any().any() or (asset_prices <= 0).any().any():
        raise RuntimeError("ETF prices contain a gap or invalid level in the common history")
    if prices[universe["vix"]].isna().any() or yields.loc[first:].isna().any().any():
        raise RuntimeError("VIX or lagged FRED values contain gaps in the required history")

    returns = asset_prices.pct_change(fill_method=None)
    returns["risky"] = 0.5 * (
        returns[universe["growth"]].mean(axis=1) + returns[universe["defensive"]].mean(axis=1)
    )
    yields = yields.rename(columns={code: name for name, code in universe["fred"].items()})
    credit_relative = prices[hyg].pct_change(21, fill_method=None) - prices[shy].pct_change(21, fill_method=None)
    args = (returns["risky"], vix.loc[first:], vix_rank.loc[first:], credit_relative.loc[first:], yields.loc[first:], model)
    features = {"slow": slow_features(*args), "brake": brake_features(*args)}
    return returns[["risky", cash]].rename(columns={cash: "cash"}).iloc[1:], features


def cash_signal(features, parameters, model, branch):
    first, second, rate, interaction = features
    mix, rate_weight, interaction_weight, cap, scale, power = parameters
    score = mix * first + (1.0 - mix) * second + rate_weight * rate
    if interaction_weight:
        score = score + interaction_weight * interaction
    slow = branch == "slow"
    history = model["slow_min_history"] if slow else model["brake_min_history"]
    z = expanding_z(score, history, prior_only=slow)
    raw = cap * (1.0 + np.exp(-(z / scale).clip(-40, 40))) ** -power
    gate, exit_speed = model["material_cash"], model["exit_speed"]
    floor = model["slow_cash_floor"] if slow else 0.0
    weights = np.zeros(len(raw))
    previous = 0.0
    for i, weight in enumerate(raw.to_numpy()):
        target = weight if weight >= gate else 0.0
        if target >= previous or (slow and target > 0.0):
            previous = target
        else:
            previous += exit_speed * (target - previous)
            if previous < floor:
                previous = 0.0
        weights[i] = previous
    return pd.Series(weights, index=raw.index)


def simulate(returns, signal, execution, reset_at=None):
    # The order fills at close t + lag. Its new weight earns returns from that close onward.
    lag = execution["signal_lag_sessions"]
    fill = signal.reindex(returns.index).shift(lag, fill_value=0.0)
    if reset_at is not None:
        # Start OOS in the benchmark until a signal using the fitted pair can first fill.
        reset_position = fill.index.get_loc(reset_at)
        fill.iloc[reset_position:reset_position + lag] = 0.0
    held = fill.shift(1, fill_value=0.0)
    gross = (1.0 - held) * returns["risky"] + held * returns["cash"]
    cash_at_fill = held * (1.0 + returns["cash"]) / (1.0 + gross)
    turnover = (fill - cash_at_fill).abs()
    friction = (execution["cost_bps"] + execution["slippage_bps"]) / 10_000.0
    net = (1.0 + gross) * (1.0 - friction * turnover) - 1.0
    return pd.DataFrame({"net": net, "turnover": turnover, "cash_weight": held, "bil": returns["cash"]})


def metrics(frame, annual_days):
    returns = frame["net"]
    excess = returns - frame["bil"]
    total = (1.0 + returns).prod() - 1.0
    volatility = returns.std(ddof=0)
    excess_vol = excess.std(ddof=0)
    annual_scale = np.sqrt(annual_days)
    return {
        "cagr": float((1.0 + total) ** (annual_days / len(frame)) - 1.0),
        "ann_vol": float(volatility * annual_scale),
        "sharpe": float(returns.mean() / volatility * annual_scale) if volatility else None,
        "sharpe_excess": float(excess.mean() / excess_vol * annual_scale) if excess_vol else None,
        "max_drawdown": float(drawdown(returns).min()),
        "total_return": float(total),
        "turnover": float(frame["turnover"].mean() * annual_days),
        "n_trades": int((frame["turnover"] > 1e-10).sum()),
    }


def select_parameters(features, returns, config, start, is_end, branch):
    model = config["model"]
    training_features = tuple(series.loc[:is_end] for series in features)
    training_returns = returns.loc[:is_end]
    grid = model[f"{branch}_grid"]
    combinations = list(product(*(grid[key] for key in GRID_KEYS)))
    rows = []
    for parameters in combinations:
        signal = cash_signal(training_features, parameters, model, branch)
        result = simulate(training_returns, signal, config["execution"], training_returns.index[-1]).loc[start:]
        stats = metrics(result, model["trading_days_per_year"])
        drawdown_size = -stats["max_drawdown"]
        rows.append({
            **stats,
            "calmar": stats["cagr"] / drawdown_size if drawdown_size else np.nan,
            "low_turnover": -stats["turnover"],
            "low_cash": -result["cash_weight"].mean(),
        })
    ranks = pd.DataFrame(rows).rank(pct=True, na_option="bottom")
    selection = pd.Series(model[f"{branch}_selection_weights"])
    winner = ranks[selection.index].dot(selection).idxmax()
    return combinations[winner], len(combinations)


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    dates = {name: pd.Timestamp(value) for name, value in config["dates"].items()}
    if not dates["data_start"] < dates["sample_start"] <= dates["is_end"] < dates["sample_end"]:
        raise ValueError("Data start, sample start, IS end and sample end must be ordered")
    execution = config["execution"]
    if execution["signal_lag_sessions"] < 1 or execution["vix_lag_sessions"] < 1 or execution["fred_lag_sessions"] < 2:
        raise ValueError("Configure at least one session for execution and VIX, and two for FRED")
    if min(execution["cost_bps"], execution["slippage_bps"]) < 0:
        raise ValueError("Cost and slippage must be nonnegative")

    universe = config["universe"]
    tickers = sorted(set(universe["growth"] + universe["defensive"] + [
        universe["cash"], universe["vix"], universe["high_yield"], universe["treasury_etf"],
    ]))
    CACHE.mkdir(exist_ok=True)
    prices, yahoo_path = load_prices(tickers, dates["data_start"], dates["sample_end"])
    fred, fred_path = load_fred(list(universe["fred"].values()), dates["data_start"], dates["sample_end"])
    fetched_at = datetime.fromtimestamp(max(path.stat().st_mtime for path in (yahoo_path, fred_path)), timezone.utc)
    print(f"DATA_FETCHED_AT: {fetched_at.isoformat(timespec='seconds')}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return for ETFs; VIX index level")

    returns, features = build_panel(prices, fred, config)
    start, split, end = dates["sample_start"], dates["is_end"], dates["sample_end"]
    sample = returns.loc[start:end]
    if len(sample.loc[:split]) < 2 or len(sample.loc[sample.index > split]) < 2:
        raise RuntimeError("Both IS and OOS need at least two daily returns")

    signals = {}
    trials = 0
    for branch, inputs in features.items():
        parameters, count = select_parameters(inputs, returns, config, start, split, branch)
        trials += count
        signals[branch] = cash_signal(inputs, parameters, config["model"], branch)
        print(f"{branch.upper()}_IS_SELECTION: {json.dumps(dict(zip(GRID_KEYS, parameters)))}")

    combined = pd.DataFrame(signals).max(axis=1)
    strategy = simulate(returns, combined, execution, sample.loc[:split].index[-1]).loc[start:end]
    benchmark = simulate(returns, pd.Series(0.0, index=returns.index), execution).loc[start:end]
    annual_days = config["model"]["trading_days_per_year"]
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": str(sample.index[0].date()), "end": str(sample.index[-1].date())},
    }
    benchmark_results = {"name": "100% static 50/50 growth-defensive sleeve"}
    for period, index in (
        ("is", sample.index[sample.index <= split]),
        ("oos", sample.index[sample.index > split]),
        ("full", sample.index),
    ):
        result[period] = metrics(strategy.loc[index], annual_days)
        benchmark_results[period] = metrics(benchmark.loc[index], annual_days)
        print(f"{period.upper()}: {json.dumps(result[period], allow_nan=False)}")
        print(f"BENCHMARK_{period.upper()}: {json.dumps(benchmark_results[period], allow_nan=False)}")
    result["benchmark"] = benchmark_results
    print(f"PARAM_COMBOS_TRIED: {trials + 1}")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()