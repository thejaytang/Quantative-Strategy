"""126_hmm_regime_allocation
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/126_hmm_regime_allocation.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit ad594150a0142009ae8f6ffab0995de08b2f8304 on 2026-11-26 13:49 Europe/Oslo, exit 0, output sha256 0a0e9fe28414b358d82f962c2469bb10cfa56cab2e38950616107e9cf162730c
Data source: Yahoo Finance 日度 ETF 开盘价和收盘价，经 yfinance 下载或从本地 CSV 缓存读取。标的为 SPY、TLT、GLD，信号特征仅使用 SPY 收盘价。
Data fetched: 2026-11-26 13:49 Europe/Oslo (run time)
Sample period: 数据请求区间为 2005-01-03 至 2025-12-31，含首尾日期。样本内开始前的数据用于预热和早期训练。日志记录的回测区间为 2010-01-04 至 2025-12-31。； data_range (from output): 2010-01-04 to 2025-12-31
IS/OOS split: 配置的样本内区间为 2010-01-01 至 2019-12-31，样本外为 2020-01-01 至 2025-12-31，区间均含首尾日期，按实际交易日切片。样本内按半年重估。样本外周度决策使用截至样本内最后一个交易日的最终拟合，固定特征缩放、HMM 参数和状态配置，仅继续用新观测更新过滤概率。两段指标取自同一条连续模拟路径。
Costs and slippage: 每买入或卖出一美元收取 5 bps。开盘再平衡按各 ETF 买卖金额的绝对值求和计费，费用从组合净值扣除，包括首次建仓，等权基准沿用相同费率和再平衡时点。未单独建模滑点、开盘冲击、流动性或税费，模拟以复权开盘价成交。
Signal lag: 每周以周五为周界，在下一周首个实际交易日开盘调仓，信号使用该开盘前一交易日的收盘信息。初次建仓在回测首个交易日开盘，使用前一交易日的收盘信号。已有持仓先计入上一收盘至该开盘的变动，再执行调仓。
Price adjustment: 下载设置 auto_adjust=True，策略读取 Yahoo Finance 复权开盘价和收盘价，缓存文件存在时直接使用其中的 Open 和 Close。以复权收盘价构造信号，并以复权开盘价模拟成交。
Survivorship bias: SPY、TLT、GLD 为事后选定的固定 ETF 组合，存在资产选择偏差。回测未使用现今指数成分股名单。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.09860539725931905, ann_vol=0.0886344559585291, sharpe=1.1055567438178378, max_drawdown=-0.15857439079577373, total_return=1.5572253538998706, turnover=19.736329382205927, n_trades=1566
    oos: cagr=0.12142479713736187, ann_vol=0.14906167437844522, sharpe=0.8435612031872329, max_drawdown=-0.2425922141280834, total_return=0.9853218754033275, turnover=16.156861073693214, n_trades=939
    full: cagr=0.10710208722331305, ann_vol=0.11504465245747789, sharpe=0.9421612030950982, max_drawdown=-0.24259221412808296, total_return=4.076915435433465, turnover=35.89319045589914, n_trades=2505
    benchmark full: cagr=0.09052152484681719, ann_vol=0.09134339992193448, sharpe=0.9945308426569265, max_drawdown=-0.22882857783662758, total_return=2.989818471752803, turnover=11.276953889678538, n_trades=2505
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 126_hmm_regime_allocation\ntickers: [SPY, TLT, GLD]\nsignal_ticker: SPY\ndata_start: 2005-01-03\n\n# Inclusive dates. The data before in_sample.start are used for warm-up.\nin_sample:\n  start: 2010-01-01\n  end: 2019-12-31\nout_of_sample:\n  start: 2020-01-01\n  end: 2025-12-31\n\ncost_bps: 5.0  # Per dollar bought or sold.\nsessions_per_year: 252\n\nhmm:\n  states: 3\n  vol_window: 21\n  training_sessions: 1260\n  refit_months: 6  # During IS. One final fit at the IS cutoff is frozen for OOS.\n  seeds: [126, 127]\n  max_iterations: 40\n  variance_floor: 0.0001  # In standardized feature units.\n  tolerance: 0.001  # Absolute difference in training log likelihood.\n\npolicy:\n  min_training_weeks: 20\n  variance_penalty: 2.0\n  # Weights in ticker order. Equal weight is included automatically.\n  allocations:\n    - [0.7, 0.2, 0.1]\n    - [0.2, 0.4, 0.4]\n'
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
"""C27 ETF allocation backtest with a filtered Gaussian HMM."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from numba import njit
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent


def load_prices(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    universe = "_".join(tickers)
    path = cache_dir / f"yahoo_adjusted_{universe}_{start.date()}_{end.date()}.csv"
    if path.exists():
        prices = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    else:
        yf.set_tz_cache_location(str(cache_dir / "yfinance"))
        prices = yf.download(
            tickers, start=start, end=end + pd.Timedelta(days=1),
            auto_adjust=True, group_by="column", multi_level_index=True,
            progress=False, threads=False, timeout=15,
        )
    columns = pd.MultiIndex.from_product([["Open", "Close"], tickers])
    prices = prices.reindex(columns=columns).sort_index().loc[start:end]
    if (
        prices.empty or prices.index.has_duplicates
        or (prices.index[0] - start).days > 7 or (end - prices.index[-1]).days > 7
        or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any()
    ):
        raise ValueError(f"Incomplete adjusted Yahoo price history for {universe}, {start.date()} to {end.date()}")
    if not path.exists():
        with NamedTemporaryFile(mode="w", dir=cache_dir, delete=False) as temporary:
            prices.to_csv(temporary)
        Path(temporary.name).replace(path)
    return prices["Open"], prices["Close"]


def gaussian_likelihoods(values, means, variances):
    residual = values[:, None, :] - means
    log_density = -0.5 * (np.log(2 * np.pi * variances) + residual**2 / variances).sum(axis=2)
    offsets = log_density.max(axis=1)
    return np.exp(log_density - offsets[:, None]), offsets


@njit
def forward_filter(likelihoods, prior, transition):
    filtered = np.empty_like(likelihoods)
    scales = np.empty(len(likelihoods))
    for row, likelihood in enumerate(likelihoods):
        posterior = prior * likelihood
        scales[row] = posterior.sum()
        posterior /= scales[row]
        filtered[row] = posterior
        prior = posterior @ transition
    return filtered, scales


@njit
def backward_pass(likelihoods, transition, scales):
    backward = np.ones_like(likelihoods)
    for row in range(len(likelihoods) - 2, -1, -1):
        backward[row] = transition @ (likelihoods[row + 1] * backward[row + 1]) / scales[row + 1]
    return backward


def fit_hmm(values, settings):
    """Estimate diagonal Gaussian emissions and transitions by Baum-Welch."""
    n_states = settings["states"]
    floor = settings["variance_floor"]
    groups = np.array_split(values[np.argsort(values[:, 1])], n_states)
    centers = np.stack([group.mean(axis=0) for group in groups])
    best_log_likelihood = -np.inf
    for seed in settings["seeds"]:
        rng = np.random.default_rng(seed)
        means = centers + rng.normal(0, 0.1, size=centers.shape)
        variances = np.stack([group.var(axis=0).clip(min=floor) for group in groups])
        initial = np.full(n_states, 1 / n_states)
        transition = 0.95 * np.eye(n_states) + np.full((n_states, n_states), 0.05 / n_states)
        previous_log_likelihood = -np.inf

        for _ in range(settings["max_iterations"]):
            likelihoods, offsets = gaussian_likelihoods(values, means, variances)
            filtered, scales = forward_filter(likelihoods, initial, transition)
            log_likelihood = (np.log(scales) + offsets).sum()
            if abs(log_likelihood - previous_log_likelihood) < settings["tolerance"]:
                break
            previous_log_likelihood = log_likelihood
            backward = backward_pass(likelihoods, transition, scales)
            smoothed = filtered * backward
            smoothed /= smoothed.sum(axis=1, keepdims=True)

            # Expected state pairs, summed over adjacent training observations.
            next_evidence = likelihoods[1:] * backward[1:] / scales[1:, None]
            counts = transition * (filtered[:-1].T @ next_evidence) + 1e-8
            transition = counts / counts.sum(axis=1, keepdims=True)
            initial = smoothed[0] + 1e-8
            initial /= initial.sum()
            weights = smoothed.T / smoothed.sum(axis=0)[:, None]
            means = weights @ values
            residual = values[None, :, :] - means[:, None, :]
            variances = (weights[:, :, None] * residual**2).sum(axis=1).clip(min=floor)

        likelihoods, offsets = gaussian_likelihoods(values, means, variances)
        filtered, scales = forward_filter(likelihoods, initial, transition)
        log_likelihood = (np.log(scales) + offsets).sum()
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            order = np.argsort(means[:, 1])
            model = transition[np.ix_(order, order)], means[order], variances[order]
            best_filtered = filtered[:, order]
    return model, best_filtered


def learn_policy(states, weekly_returns, allocations, n_states, settings):
    policy = np.tile(allocations[0], (n_states, 1))
    for state in range(n_states):
        returns = weekly_returns[states == state] @ allocations.T
        if len(returns) >= settings["min_training_weeks"]:
            scores = returns.mean(axis=0) - settings["variance_penalty"] * returns.var(axis=0)
            policy[state] = allocations[scores.argmax()]
    return policy


def weekly_targets(opens, closes, start_row, last_is_row, config, allocations):
    settings = config["hmm"]
    log_returns = np.log(closes[config["signal_ticker"]]).diff()
    volatility = log_returns.rolling(settings["vol_window"]).std()
    features = np.column_stack((log_returns, volatility))

    dates = opens.index
    week_opens = np.flatnonzero(~dates.to_period("W-FRI").duplicated())
    entries = week_opens[:-1]
    exits = week_opens[1:]
    open_values = opens.to_numpy()
    weekly_returns = open_values[exits] / open_values[entries] - 1
    execution_rows = np.union1d([start_row], week_opens[week_opens >= start_row])

    targets = {}
    previous_fit = None
    for execution in execution_rows:
        signal_row = execution - 1
        fit_row = min(signal_row, last_is_row)
        refit_date = dates[min(execution, last_is_row)]
        refit_bucket = (refit_date.year * 12 + refit_date.month - 1) // settings["refit_months"]
        fit_key = refit_bucket, fit_row == last_is_row
        if fit_key != previous_fit:
            first_train_row = max(settings["vol_window"], fit_row - settings["training_sessions"] + 1)
            train = features[first_train_row:fit_row + 1]
            if len(train) < settings["states"] * 2:
                raise ValueError("Need more daily price history before the first execution")
            center = train.mean(axis=0)
            scale = train.std(axis=0).clip(min=1e-12)
            (transition, means, variances), filtered = fit_hmm((train - center) / scale, settings)

            # A policy can learn from a week only once its exit open is known.
            eligible = (entries - 1 >= first_train_row) & (exits <= fit_row)
            states = filtered[entries[eligible] - 1 - first_train_row].argmax(axis=1)
            policy = learn_policy(states, weekly_returns[eligible], allocations, settings["states"], config["policy"])
            posterior = filtered[-1]
            filtered_through = fit_row
            previous_fit = fit_key

        if signal_row > filtered_through:
            recent = (features[filtered_through + 1:signal_row + 1] - center) / scale
            likelihoods, _ = gaussian_likelihoods(recent, means, variances)
            filtered, _ = forward_filter(likelihoods, posterior @ transition, transition)
            posterior = filtered[-1]
            filtered_through = signal_row
        targets[execution] = policy[posterior.argmax()]
    return targets


def simulate(opens, closes, targets, cost_rate):
    """Hold shares between rebalances and book the close-to-open move first."""
    open_values = opens.to_numpy()
    close_values = closes.to_numpy()
    first_row = min(targets)
    holdings = np.zeros(opens.shape[1])
    previous_nav = 1.0
    results = []
    for row in range(first_row, len(opens)):
        if row > first_row:
            holdings *= open_values[row] / close_values[row - 1]
        turnover = 0.0
        n_trades = 0
        if row in targets:
            open_nav = 1.0 if row == first_row else holdings.sum()
            pre_trade_weights = holdings / open_nav
            target = targets[row]

            # Pay fees from the same account so the post-trade portfolio is fully invested.
            def fee_balance(net_fraction):
                traded = np.abs(net_fraction * target - pre_trade_weights).sum()
                return net_fraction + cost_rate * traded - 1

            net_fraction = brentq(fee_balance, 0, 1)
            new_weights = net_fraction * target
            traded = np.abs(new_weights - pre_trade_weights)
            turnover = traded.sum()
            n_trades = np.count_nonzero(traded > 1e-10)
            holdings = open_nav * new_weights
        holdings *= close_values[row] / open_values[row]
        nav = holdings.sum()
        results.append((nav / previous_nav - 1, turnover, n_trades))
        previous_nav = nav
    return pd.DataFrame(results, index=opens.index[first_row:], columns=["return", "turnover", "n_trades"])


def metrics(daily, sessions_per_year):
    returns = daily["return"]
    wealth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    volatility = returns.std() * np.sqrt(sessions_per_year)
    result = {
        "cagr": wealth[-1] ** (sessions_per_year / len(returns)) - 1,
        "ann_vol": volatility,
        "sharpe": returns.mean() * sessions_per_year / volatility if volatility > 0 else np.nan,
        "max_drawdown": (wealth / np.maximum.accumulate(wealth) - 1).min(),
        "total_return": wealth[-1] - 1,
        "n_trades": int(daily["n_trades"].sum()),
        "turnover": daily["turnover"].sum(),
    }
    return {key: value if key == "n_trades" else float(value) if np.isfinite(value) else None
            for key, value in result.items()}


def print_metrics(results):
    columns = ["cagr", "ann_vol", "sharpe", "max_drawdown", "total_return", "n_trades", "turnover"]
    formats = [".2%", ".2%", ".2f", ".2%", ".2%", "d", ".2f"]
    print(f"{results['strategy']} | {results['data_range']}")
    print(f"Benchmark: {results['benchmark']['name']}")
    headings = ["CAGR", "Ann. vol", "Sharpe", "Max drawdown", "Total return", "Trades", "Turnover (x)"]
    print(f"{'Portfolio / period':22}" + "".join(f"{heading:>14}" for heading in headings))
    for name, portfolio in [("Strategy", results), ("Benchmark", results["benchmark"])]:
        for period in ("is", "oos", "full"):
            row = portfolio[period]
            values = [format(row[key], spec) if row[key] is not None else "n/a" for key, spec in zip(columns, formats)]
            print(f"{name + ' / ' + period.upper():22}" + "".join(f"{value:>14}" for value in values))
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


def main():
    with (ROOT / "config.yaml").open() as file:
        config = yaml.safe_load(file)
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    data_start = pd.Timestamp(config["data_start"])
    if not (data_start < is_start <= is_end < oos_start <= oos_end) or oos_start - is_end != pd.Timedelta(days=1):
        raise ValueError("Use chronological, contiguous IS and OOS ranges with earlier warm-up data")
    cost_rate = config["cost_bps"] / 10_000
    if not 0 <= cost_rate < 1:
        raise ValueError("Transaction cost must be between 0 and 10,000 bps (exclusive upper bound)")

    tickers = config["tickers"]
    equal_weight = np.full(len(tickers), 1 / len(tickers))
    allocations = np.array([equal_weight, *config["policy"]["allocations"]], dtype=float)
    if (allocations < 0).any() or not np.allclose(allocations.sum(axis=1), 1):
        raise ValueError("Every allocation must be long-only and sum to one")
    opens, closes = load_prices(tickers, data_start, oos_end)
    start_row = opens.index.searchsorted(is_start)
    first_oos_row = opens.index.searchsorted(oos_start)
    if not 0 < start_row < first_oos_row < len(opens):
        raise ValueError("Price history must cover warm-up, IS and OOS sessions")
    targets = weekly_targets(opens, closes, start_row, first_oos_row - 1, config, allocations)
    strategy_daily = simulate(opens, closes, targets, cost_rate)
    benchmark_daily = simulate(opens, closes, dict.fromkeys(targets, equal_weight), cost_rate)
    results = {
        "strategy": config["strategy"],
        "data_range": f"{strategy_daily.index[0].date()} to {strategy_daily.index[-1].date()}",
    }
    benchmark = {"name": "Weekly equal-weight " + "/".join(tickers)}
    periods = {"is": slice(is_start, is_end), "oos": slice(oos_start, oos_end), "full": slice(None)}
    for period, date_range in periods.items():
        results[period] = metrics(strategy_daily.loc[date_range], config["sessions_per_year"])
        benchmark[period] = metrics(benchmark_daily.loc[date_range], config["sessions_per_year"])
    results["benchmark"] = benchmark
    print_metrics(results)


if __name__ == "__main__":
    main()
