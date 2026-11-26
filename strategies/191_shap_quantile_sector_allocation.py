"""191_shap_quantile_sector_allocation
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/191_shap_quantile_sector_allocation.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 2b23c5bbf706aea933c0cba5f5805efb450ee206 on 2026-11-26 14:26 Europe/Oslo, exit 0, output sha256 7ba22ee4a3c7dc849d699d0a092925ce39abad56e5224f3eab0e3e548773a3c3
Data source: Yahoo Finance 日频 ETF 开盘价和收盘价，通过 yfinance 下载或读取已有价格 CSV 缓存。行业 ETF 为 XLB、XLE、XLF、XLI、XLK、XLP、XLU、XLV、XLY，SPY 用于市场相关性，BIL 用于现金配置及超额收益标签。见 config.yaml:3-6、strategy.py:18-38、strategy.py:80-94、strategy.py:210-215、strategy.py:263-268。
Data fetched: 2026-11-26 14:26 Europe/Oslo (run time)
Sample period: 配置请求的数据区间为 2009-01-01 至 2025-06-30，2014-01-01 之前为预热期。日志记录的实际开盘估值区间为 2014-01-06 至 2025-06-30。见 config.yaml:8-16、strategy.py:18-33、strategy.py:257-260、strategy.py:268-275、logs/run_20261125_122619.log:1。   data_range (from output): 2014-01-06 to 2025-06-30
IS/OOS split: 配置的样本内为 2014-01-01 至 2019-12-31，样本外为 2020-01-01 至 2025-06-30。代码连续模拟持仓，再按开盘估值日切分指标。每半年在首个可处理的周度信号日重训，使用最近最多 260 个已到期的周度训练截面，至少需要 104 个。样本外重训可使用此前已到期的样本外标签。见 config.yaml:11-16、config.yaml:28-30、strategy.py:114-141、strategy.py:273-288、README.md:20-22。
Costs and slippage: 主情景每单位买入或卖出名义金额收取 5 bps，另测 0 和 15 bps。每次再平衡先按当天开盘价估值，再按所有 ETF 的绝对名义仓位变化之和扣费，目标权重以扣费后的净值计算。未建模滑点、开盘价差或市场冲击。见 config.yaml:37-40、strategy.py:191-226、strategy.py:279-287、README.md:23、README.md:28。
Signal lag: 用每周最后一个交易日 t 的收盘价及此前价格计算特征和目标仓位。周五休市时使用该周内周五之前的最后一个交易日。模拟在 t 之后的首个交易日按该日开盘价再平衡。见 strategy.py:62-89、strategy.py:143-158、strategy.py:199-225、README.md:9。
Price adjustment: 下载路径设置 yfinance auto_adjust=True，并保留 Open 和 Close。缓存命中时直接读取同名 adjusted CSV，代码检查价格有效性和覆盖情况，未独立核验缓存的复权来源或历史版本。特征使用复权收盘价，标签、成交与估值使用复权开盘价。见 strategy.py:21-53、strategy.py:72-94、strategy.py:191-226、strategy.py:268-274、README.md:11、README.md:28。
Survivorship bias: 固定选择九只目前仍存在的行业 ETF，存在产品选择与存续偏差。实现省略房地产和通信服务，基金成分及行业分类可能随时间变化。见 config.yaml:3-6、strategy.py:263-268、README.md:15、README.md:28。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.04423014655788293, ann_vol=0.1027113809489663, sharpe=0.47265155129226427, max_drawdown=-0.12726151636035687, total_return=0.29562407417485614, turnover=31.823863369747908
    oos: cagr=0.07540595937397288, ann_vol=0.17948182334749618, sharpe=0.4959680733257117, max_drawdown=-0.32924221566781064, total_return=0.4890068258780427, turnover=23.090229350110807
    full: cagr=0.05901278586658676, ann_vol=0.14455322652994004, sharpe=0.4696202287092069, max_drawdown=-0.3292422156678104, total_return=0.9291930902182794, turnover=27.650589496098608
    benchmark equal_weight full: cagr=0.11519908055559913, ann_vol=0.16342063591554412, sharpe=0.7495216925439285, max_drawdown=-0.34568592765912987, total_return=2.4887798663233145, turnover=0.7313002204093221
    benchmark median_only full: cagr=0.0881833400503711, ann_vol=0.20190392906231966, sharpe=0.5202592848217296, max_drawdown=-0.40083341040267784, total_return=1.6340099049792398, turnover=42.73790658941202
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "191_shap_quantile_sector_allocation"\n\nuniverse:\n  sectors: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\n  market: SPY\n  cash: BIL\n\ndata:\n  start: "2009-01-01"\n\nin_sample:\n  start: "2014-01-01"\n  end: "2019-12-31"\nout_of_sample:\n  start: "2020-01-01"\n  end: "2025-06-30"\n\nfeatures:\n  short_return_sessions: 5\n  medium_return_sessions: 21\n  downside_sessions: 21\n  market_correlation_sessions: 63\n\nmodel:\n  lower_quantile: 0.20\n  median_quantile: 0.50\n  lower_quantile_weight: 0.50\n  max_training_weeks: 260\n  min_training_weeks: 104\n  refit_months: 6\n  n_estimators: 60\n  max_depth: 2\n  learning_rate: 0.10\n  random_seed: 191\n  permutation_repeats: 10\n\nportfolio:\n  max_sectors: 3\n  transaction_cost_bps: 5\n  sensitivity_bps: [0, 15]\n'
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
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq
from sklearn.ensemble import GradientBoostingRegressor


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
STRATEGIES = ("quantile", "median_only", "equal_weight")


def load_prices(config, tickers, market):
    start = pd.Timestamp(config["data"]["start"])
    end = pd.Timestamp(config["out_of_sample"]["end"])
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"adjusted_{start.date()}_{end.date()}_{'-'.join(tickers)}.csv"

    if cache_file.is_file():
        prices = pd.read_csv(cache_file, header=[0, 1], index_col=0, parse_dates=True)
    else:
        with TemporaryDirectory(dir=cache_dir) as yf_cache:
            yf.set_tz_cache_location(yf_cache)
            for _ in range(2):
                prices = yf.download(
                    tickers, start=start.date(), end=(end + pd.Timedelta(days=1)).date(),
                    auto_adjust=True, progress=False, threads=True, timeout=12,
                )
                if not prices.empty:
                    prices = prices[["Open", "Close"]]
                    if prices.notna().any().all():
                        break
            else:
                raise RuntimeError("Yahoo Finance returned no history for one or more requested ETFs")

    prices = prices.sort_index().where(lambda frame: frame.gt(0) & np.isfinite(frame))
    opens = prices["Open"].reindex(columns=tickers)
    closes = prices["Close"].reindex(columns=tickers)
    sessions = opens.index[opens[market].notna() | closes[market].notna()]
    if (sessions.empty or sessions[0] > start + pd.Timedelta(days=7)
            or sessions[-1] < end - pd.Timedelta(days=7)):
        raise RuntimeError("Yahoo Finance history does not cover the requested dates")
    opens, closes = opens.loc[sessions], closes.loc[sessions]
    if not (opens.notna().any().all() and closes.notna().any().all()):
        raise RuntimeError("Missing adjusted Open or Close history for an ETF")
    if opens[market].isna().any() or closes[market].isna().any():
        raise RuntimeError(f"Incomplete session prices for market reference {market}")
    if not cache_file.is_file():
        with TemporaryDirectory(dir=cache_dir) as temp_dir:
            temporary_file = Path(temp_dir) / cache_file.name
            prices.to_csv(temporary_file)
            temporary_file.replace(cache_file)
    return opens, closes


def weekly_data(opens, closes, sectors, market, cash, feature_config, data_end):
    dates = pd.Series(closes.index, index=closes.index)
    # A Friday holiday uses the preceding session's close in the same week.
    week_closes = dates.resample("W-FRI").last().dropna()
    week_closes = week_closes.loc[:data_end]
    signal_dates = pd.DatetimeIndex(week_closes, name="signal")
    # Shift on the session calendar before subsetting to weekly signal dates.
    entry = dates.shift(-1).reindex(signal_dates)
    schedule = pd.DataFrame({"entry": entry, "exit": entry.shift(-1)})

    returns = closes.pct_change(fill_method=None)
    sector_closes = closes[sectors]
    sector_returns = returns[sectors]
    feature_frames = {
        "short_return": sector_closes.pct_change(
            feature_config["short_return_sessions"], fill_method=None),
        "medium_return": sector_closes.pct_change(
            feature_config["medium_return_sessions"], fill_method=None),
        "downside_semivariance": sector_returns.clip(upper=0).pow(2).rolling(
            feature_config["downside_sessions"]).mean(),
        "market_correlation": sector_returns.rolling(
            feature_config["market_correlation_sessions"]).corr(returns[market]),
    }
    snapshot_index = pd.MultiIndex.from_product([signal_dates, sectors], names=["signal", "sector"])
    features = pd.concat(
        {name: frame.reindex(signal_dates).rename_axis(columns="sector").stack()
         for name, frame in feature_frames.items()}, axis=1,
    ).reindex(snapshot_index).replace([np.inf, -np.inf], np.nan)

    entry_prices = opens.reindex(schedule["entry"]).set_axis(signal_dates)
    holding_returns = entry_prices.shift(-1).div(entry_prices).sub(1)
    labels = holding_returns[sectors].sub(holding_returns[cash], axis=0)
    labels = labels.rename_axis(columns="sector").stack().rename("target")
    return schedule, features, labels


def sector_weights(scores, max_sectors):
    ranked = scores[scores > 0].sort_values(ascending=False, kind="stable")
    return pd.Series(1 / max_sectors, index=ranked.index[:max_sectors])


def weekly_targets(schedule, features, labels, closes, config, start, oos_start):
    model_config = config["model"]
    max_sectors = config["portfolio"]["max_sectors"]
    sectors = config["universe"]["sectors"]
    levels = [model_config["lower_quantile"], model_config["median_quantile"]]
    panel = features.join(labels)
    feature_names = features.columns.tolist()
    weights = {name: {} for name in STRATEGIES}
    evaluations = []
    fit_period = None

    for signal, week in schedule.dropna(subset=["entry"]).iterrows():
        entry = week["entry"]
        if entry < start or entry >= closes.index[-1]:
            continue
        months = model_config["refit_months"]
        period = (signal.year * 12 + signal.month - 1) // months
        if period != fit_period:
            # The exit open of every fitting target must already have occurred.
            mature = schedule.index[schedule["exit"].le(signal)]
            train = panel.loc[panel.index.get_level_values("signal").isin(mature)].dropna()
            train_dates = train.index.get_level_values("signal").unique()
            train_dates = train_dates[-model_config["max_training_weeks"]:]
            if len(train_dates) < model_config["min_training_weeks"]:
                if fit_period is None:
                    continue
                raise RuntimeError(f"Insufficient training snapshots at {signal.date()}")
            train = train.loc[train_dates]
            models = [
                GradientBoostingRegressor(
                    loss="quantile", alpha=level,
                    n_estimators=model_config["n_estimators"],
                    max_depth=model_config["max_depth"],
                    learning_rate=model_config["learning_rate"],
                    random_state=model_config["random_seed"],
                ).fit(train[feature_names].to_numpy(), train["target"])
                for level in levels
            ]
            fit_period = period

        snapshot = features.loc[signal].dropna().reindex(sectors).dropna()
        if snapshot.empty:
            lower = median = pd.Series(dtype=float)
        else:
            predictions = [model.predict(snapshot.to_numpy()) for model in models]
            lower, median = [pd.Series(values, index=snapshot.index) for values in predictions]
            if entry >= oos_start and pd.notna(week["exit"]):
                realized = panel.loc[signal, "target"].reindex(snapshot.index).dropna()
                if len(realized) > 1:
                    evaluations.append((models, snapshot.loc[realized.index], realized))

        score = median + model_config["lower_quantile_weight"] * lower
        weights["quantile"][entry] = sector_weights(score, max_sectors)
        weights["median_only"][entry] = sector_weights(median, max_sectors)
        available = closes.loc[signal, sectors].dropna().index
        weights["equal_weight"][entry] = pd.Series(1 / len(sectors), index=available)

    if not weights["quantile"]:
        raise RuntimeError("No evaluation trades scheduled; check the warm-up and sample dates")
    return weights, evaluations, levels


def permutation_diagnostic(evaluations, levels, feature_names, config):
    repeats = config["permutation_repeats"]
    generator = np.random.default_rng(config["random_seed"])
    loss_change = np.zeros((len(levels), len(feature_names)))
    observations = 0
    for models, snapshot, realized in evaluations:
        features = snapshot.to_numpy()
        n_rows, n_features = features.shape
        shuffled = np.broadcast_to(features, (n_features, repeats, *features.shape)).copy()
        # Shuffle each feature across sectors within the same held-out week.
        for feature in range(n_features):
            for repeat in range(repeats):
                shuffled[feature, repeat, :, feature] = generator.permutation(features[:, feature])
        for number, (model, level) in enumerate(zip(models, levels)):
            base_error = realized.to_numpy() - model.predict(features)
            base_loss = np.maximum(level * base_error, (level - 1) * base_error).sum()
            predicted = model.predict(shuffled.reshape(-1, n_features)).reshape(
                n_features, repeats, n_rows)
            error = realized.to_numpy() - predicted
            losses = np.maximum(level * error, (level - 1) * error)
            loss_change[number] += losses.sum(axis=2).mean(axis=1) - base_loss
        observations += n_rows
    return (pd.DataFrame(loss_change / observations, index=levels, columns=feature_names).T
            if observations else pd.DataFrame())


def simulate(opens, targets, cash, cost_bps):
    fee = cost_bps / 10_000
    columns = opens.columns
    invested_cash = columns.get_loc(cash)
    shares = np.zeros(len(columns))
    uninvested_cash = previous_nav = 1.0
    records = []

    for date, opening in zip(opens.index, opens.to_numpy()):
        available = np.isfinite(opening)
        if ((shares != 0) & ~available).any():
            missing = columns[(shares != 0) & ~available].tolist()
            raise RuntimeError(f"Cannot value held ETFs at the {date.date()} open: {missing}")
        prices = np.where(available, opening, 1.0)
        holdings = shares * prices
        nav_before = holdings.sum() + uninvested_cash
        nav = nav_before
        turnover = 0.0

        if date in targets:
            target = targets[date].reindex(columns, fill_value=0.0).to_numpy(copy=True)
            target[~available] = 0
            if available[invested_cash]:
                target[invested_cash] = 1.0 - target.sum()

            # Target weights apply to post-fee NAV. Solve the self-financing fill.
            opening_weights = holdings / nav_before
            if fee:
                nav_fraction = brentq(
                    lambda fraction: fraction + fee * np.abs(target * fraction - opening_weights).sum() - 1,
                    0.0, 1.0,
                )
                nav = nav_before * nav_fraction
            turnover = np.abs(target * nav - holdings).sum() / nav_before
            shares = target * nav / prices
            uninvested_cash = nav * (1.0 - target.sum())

        records.append((nav / previous_nav - 1.0, turnover))
        previous_nav = nav

    return pd.DataFrame(records, index=opens.index, columns=["return", "turnover"])


def metrics(history):
    returns = history["return"]
    if returns.empty:
        return dict.fromkeys(
            ["cagr", "ann_vol", "sharpe", "max_drawdown", "total_return", "turnover"])
    wealth = (1 + returns).cumprod()
    ending_wealth = wealth.iloc[-1]
    daily_vol = returns.std()
    result = {
        "cagr": ending_wealth ** (SESSIONS_PER_YEAR / len(returns)) - 1,
        "ann_vol": daily_vol * np.sqrt(SESSIONS_PER_YEAR),
        "sharpe": (returns.mean() / daily_vol * np.sqrt(SESSIONS_PER_YEAR)
                   if daily_vol > 0 else np.nan),
        "max_drawdown": (wealth / wealth.cummax().clip(lower=1.0) - 1.0).min(),
        "total_return": ending_wealth - 1,
        "turnover": history["turnover"].sum() * SESSIONS_PER_YEAR / len(returns),
    }
    return {name: float(value) if np.isfinite(value) else None for name, value in result.items()}


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not pd.Timestamp(config["data"]["start"]) < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Expected a warm-up followed by ordered, disjoint IS and OOS periods")
    if is_end + pd.Timedelta(days=1) != oos_start:
        raise ValueError("IS and OOS periods must be adjacent calendar date ranges")
    sectors = config["universe"]["sectors"]
    market, cash = (config["universe"][key] for key in ("market", "cash"))
    if not 1 <= config["portfolio"]["max_sectors"] <= len(sectors):
        raise ValueError("max_sectors must be between one and the number of sector ETFs")

    opens, closes = load_prices(config, sectors + [market, cash], market)
    schedule, features, labels = weekly_data(
        opens, closes, sectors, market, cash, config["features"], oos_end)
    weights, evaluations, levels = weekly_targets(
        schedule, features, labels, closes, config, is_start, oos_start)
    opens = opens.loc[min(weights["quantile"]):oos_end, sectors + [cash]]
    periods = {"is": opens.loc[is_start:is_end].index,
               "oos": opens.loc[oos_start:oos_end].index, "full": opens.index}
    if any(days.empty for days in periods.values()):
        raise RuntimeError("Both IS and OOS require evaluable market sessions")

    primary_bps = config["portfolio"]["transaction_cost_bps"]
    cost_levels = list(dict.fromkeys([primary_bps, *config["portfolio"]["sensitivity_bps"]]))
    primary = {}
    table = []
    for cost_bps in cost_levels:
        if not 0 <= cost_bps < 10_000:
            raise ValueError("Transaction cost must be nonnegative and below 10,000 bps")
        for name in STRATEGIES:
            performance = simulate(opens, weights[name], cash, cost_bps)
            results = {period: metrics(performance.loc[days]) for period, days in periods.items()}
            for period, result in results.items():
                table.append({"cost_bps": cost_bps, "portfolio": name, "period": period, **result})
            if cost_bps == primary_bps:
                primary[name] = results

    data_range = f"{opens.index[0].date()} to {opens.index[-1].date()}"
    print(f"{config['strategy']} | opening valuation dates {data_range}")
    print("Turnover is annualized buys plus sells / pre-trade NAV. Sharpe uses a zero rate.")
    print(pd.DataFrame(table).to_string(
        index=False, na_rep="n/a", float_format=lambda value: f"{value:.4f}"))
    importance = permutation_diagnostic(evaluations, levels, features.columns, config["model"])
    if not importance.empty:
        print("\nOOS cross-sector permutation importance (change in mean pinball loss in bps)")
        print((10_000 * importance).to_string(float_format=lambda value: f"{value:.4f}"))

    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **primary["quantile"],
        "benchmark": {name: primary[name] for name in ("equal_weight", "median_only")},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
