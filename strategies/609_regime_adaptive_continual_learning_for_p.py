"""609_regime_adaptive_continual_learning_for_p
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/609_regime_adaptive_continual_learning_for_p.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 25a9179813de3addbbaee65a8c08579eb6e27d08 on 2026-11-27 19:09 Europe/Oslo, exit 0, output sha256 e06a6a8d551b81af4df8e30f473c99cc476550adb34e86b23d80f02d04e80cb6
Data source: Yahoo Finance 免费 ETF 日线，由 yfinance 获取 SPY、TLT、GLD、BIL 的开盘和收盘价。宏观数据来自免费 FRED 和 ALFRED 历史版本 CSV，序列为 DGS10、DGS2、DGS5、DFII5，用于构造 10 年减 2 年期限利差和 5 年名义减实际收益率之差。
Data fetched: 2026-11-27T18:10:23+00:00
Sample period: 行情请求自 2007-06-01 起用于预热，报告的回测区间为 2013-01-02 至 2025-04-29。   data_range (from output): {'start': '2013-01-02', 'end': '2025-04-29'}
IS/OOS split: IS 为 2013-01-02 至 2020-04-30，OOS 为 2020-05-01 至 2025-04-29。默认规格全期连续模拟，跨界保留持仓并继续更新状态先验和收益矩。五个稳健性变体仅运行至 IS 末日。
Costs and slippage: 默认每笔买卖单边手续费 5 bps、滑点 5 bps，均按净买卖成交额计提，合计为成交额的 10 bps，含初次建仓。滑点以固定比例从资金扣减，成交金额以开盘价计算。调仓按调仓前开盘组合价值和总买卖额求解扣费后的目标敞口。higher_cost 变体将手续费和滑点各调至 10 bps。
Signal lag: 月末最后一个有行情的交易日 t 收盘以当日特征生成目标权重，默认于下一个交易日开盘调仓。原持仓先承担前收盘至成交日开盘的变动，调仓后的持仓随后承担成交日开盘至收盘的变动。
Price adjustment: yfinance auto_adjust total-return open and close
Survivorship bias: 固定使用现今存续的 SPY、TLT、GLD、BIL，存在事后资产选择和基金幸存者偏差。未用当前指数成分股名单回溯选股。
Parameter combinations tried: 6
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.07826408575880062, ann_vol=0.09202456652466456, sharpe=0.8650068222783381, sharpe_excess=0.7958717334808668, max_drawdown=-0.14305075381829147, total_return=0.736180109721521, turnover=30.335969669787033, n_trades=294
    oos: cagr=0.11061583189581525, ann_vol=0.11684719526290752, sharpe=0.956528180221364, sharpe_excess=0.7420689389277382, max_drawdown=-0.18367408868265467, total_return=0.6862239581252929, turnover=21.959847724906968, n_trades=189
    full: cagr=0.09124621658030296, ann_vol=0.10278505717321412, sharpe=0.9011409249949658, sharpe_excess=0.7656322958193631, max_drawdown=-0.18367408868265478, total_return=1.9275884966330104, turnover=52.295817394694, n_trades=483
    benchmark  full: cagr=0.05528653362578728, ann_vol=0.06960126239629086, sharpe=0.808035837865999, sharpe_excess=0.6077841341547617, max_drawdown=-0.1757578642609996, total_return=0.9386212509768312, turnover=4.366809818858483, n_trades=592
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 609_regime_adaptive_continual_learning_for_p\nuniverse: [SPY, TLT, GLD, BIL]\ncash_proxy: BIL\ndata_start: '2007-06-01'\nstart: '2013-01-02'\nis_end: '2020-04-30'\nend: '2025-04-29'\ncommission_bps: 5\nslippage_bps: 5\nsignal_lag_sessions: 1\nmacro_lag_calendar_days: 7\nmodel:\n  n_regimes: 3\n  bootstrap_sessions: 1008\n  feature_sessions: 63\n  policy_half_life_sessions: 252\n  gate_half_life_sessions: 63\n  gate_baseline_share: 0.20\n  shrinkage_days: 63\n  covariance_diagonal_weight: 0.20\n  risk_aversion: 6.0\n  max_risky_weight: 0.65\n  seed: 609\nrobustness:\n  - {name: two_regimes, model: {n_regimes: 2}}\n  - {name: four_regimes, model: {n_regimes: 4}}\n  - {name: shorter_policy_memory, model: {policy_half_life_sessions: 189}}\n  - {name: longer_policy_memory, model: {policy_half_life_sessions: 315}}\n  - {name: higher_cost, commission_bps: 10, slippage_bps: 10}\n"
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
"""Monthly multi-asset regime allocation inspired by ReCAP.

Paper: https://arxiv.org/abs/2606.00143
Research references (ideas only, no copied code):
https://github.com/Dumail/ReCAP
https://github.com/mgupta97/regime-aware-portfolio-optimizer
https://www.quantconnect.com/docs/v2/research-environment/machine-learning/popular-libraries/hmmlearn
https://pypi.org/project/hmmlearn/
https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
Price adjustment: https://github.com/ranaroussi/yfinance/blob/main/yfinance/utils.py
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import signal
import tempfile
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_ARCHIVE = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
TREASURY_SERIES = ("DGS10", "DGS2", "DGS5", "DFII5")
MACRO_FEATURES = ["term_spread", "breakeven"]
SESSIONS_PER_YEAR = 252


def cached_csv(filename, fetch):
    path = CACHE / filename
    if not path.exists():
        data = fetch()
        with tempfile.NamedTemporaryFile(mode="w", dir=CACHE, delete=False) as output:
            temp_path = Path(output.name)
            try:
                data.to_csv(output, index_label="date")
                output.flush()
                temp_path.replace(path)
            finally:
                temp_path.unlink(missing_ok=True)
    fetched_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    if data.empty or not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError(f"Missing or unordered observations in {path}")
    return data, fetched_at


def download_prices(ticker, start, end):
    end_exclusive = end + pd.Timedelta(days=1)
    for _ in range(2):
        history = yf.Ticker(ticker).history(
            start=start.date().isoformat(), end=end_exclusive.date().isoformat(),
            auto_adjust=True, actions=False, timeout=15,
        )
        if not history.empty:
            prices = history[["Open", "Close"]].copy()
            prices.index = prices.index.tz_localize(None).normalize()
            return prices
    raise RuntimeError(f"Yahoo Finance did not return {ticker} history")


def fred_csv(endpoint, query):
    url = f"{endpoint}?{urlencode(query)}"
    for attempt in range(3):
        try:
            with urlopen(url, timeout=12) as response:
                payload = response.read()
            return pd.read_csv(BytesIO(payload), index_col=0, parse_dates=True, na_values=".")
        except (URLError, TimeoutError):
            if attempt == 2:
                raise


def quarterly_macro(start, end, lag_days):
    quarters = pd.period_range(start, end, freq="Q").start_time
    publication_lag = pd.Timedelta(days=lag_days)
    observation_window = pd.Timedelta(days=14)
    latest = fred_csv(FRED, {
        "id": ",".join(TREASURY_SERIES),
        "cosd": (quarters[0] - publication_lag - observation_window).date().isoformat(),
        "coed": end.date().isoformat(),
    })

    def archive_batch(batch):
        # The graph CSV supports at most 12 series-vintage pairs per call.
        cutoffs = batch - publication_lag
        archived = fred_csv(FRED_ARCHIVE, {
            "id": ",".join(TREASURY_SERIES * len(batch)),
            "vintage_date": ",".join(
                cutoff.date().isoformat() for cutoff in cutoffs for _ in TREASURY_SERIES
            ),
            "cosd": ",".join(
                (cutoff - observation_window).date().isoformat()
                for cutoff in cutoffs for _ in TREASURY_SERIES
            ),
            "coed": ",".join(
                cutoff.date().isoformat() for cutoff in cutoffs for _ in TREASURY_SERIES
            ),
        })
        rows = []
        for quarter, cutoff in zip(batch, cutoffs, strict=True):
            columns = [f"{series}_{cutoff:%Y%m%d}" for series in TREASURY_SERIES]
            published = archived.loc[cutoff - observation_window:cutoff, columns].dropna()
            if published.empty:
                raise ValueError(f"No common FRED observation in the vintage dated {cutoff.date()}")
            observation = published.index[-1]
            past = published.iloc[-1].to_numpy(dtype=float)
            restated = latest.reindex([observation]).loc[:, list(TREASURY_SERIES)].iloc[0]
            unchanged = np.isclose(past, restated, atol=1e-12, rtol=0)
            ten_year, two_year, five_year, five_year_real = np.where(unchanged, restated, past)
            rows.append({
                "date": quarter,
                "term_spread": ten_year - two_year,
                "breakeven": five_year - five_year_real,
                "observation_date": observation.date().isoformat(),
                "vintage_date": cutoff.date().isoformat(),
                "archive_replacements": int((~unchanged).sum()),
            })
        return rows

    batches = [quarters[offset:offset + 3] for offset in range(0, len(quarters), 3)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [row for batch in pool.map(archive_batch, batches) for row in batch]
    return pd.DataFrame(rows).set_index("date")


def load_inputs(config):
    start, end = pd.Timestamp(config["data_start"]), pd.Timestamp(config["end"])
    CACHE.mkdir(exist_ok=True)
    frames = {}
    fetched = []
    with tempfile.TemporaryDirectory(prefix="yahoo_session_", dir=CACHE) as session_cache:
        yf.set_tz_cache_location(session_cache)
        for ticker in config["universe"]:
            frame, stamp = cached_csv(
                f"yahoo_adjusted_{ticker}_{start.date()}_{end.date()}.csv",
                lambda: download_prices(ticker, start, end),
            )
            frames[ticker] = frame
            fetched.append(stamp)
    prices = pd.concat(frames, axis=1).sort_index().loc[start:end]
    values = prices.to_numpy(dtype=float)
    if prices.empty or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("The configured ETF universe needs complete, positive daily open and close prices")
    if prices.index[-1] != end:
        raise ValueError(f"Missing prices on the configured final session {end.date()}")

    macro_lag = config["macro_lag_calendar_days"]
    macro, stamp = cached_csv(
        f"fred_quarterly_pit_{start.date()}_{end.date()}_lag{macro_lag}.csv",
        lambda: quarterly_macro(start, end, macro_lag),
    )
    vintages = pd.DatetimeIndex(pd.to_datetime(macro["vintage_date"]))
    observations = pd.DatetimeIndex(pd.to_datetime(macro["observation_date"]))
    if not (observations <= vintages).all() or not (
        vintages <= macro.index - pd.Timedelta(days=macro_lag)
    ).all():
        raise ValueError("Macro cache violates the point-in-time constraints")
    expected = pd.period_range(start, end, freq="Q").start_time
    if not macro.index.equals(expected) or not np.isfinite(macro[MACRO_FEATURES]).all().all():
        raise ValueError("Incomplete quarterly macro data")
    fetched.append(stamp)
    opens = prices.xs("Open", axis=1, level=1)
    closes = prices.xs("Close", axis=1, level=1)
    print(f"DATA_FETCHED_AT: {max(fetched).isoformat(timespec='seconds')}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return open and close")
    print(f"FRED_ARCHIVE_REPLACEMENTS: {int(macro['archive_replacements'].sum())}")
    return opens, closes, macro[MACRO_FEATURES]


def market_features(closes, macro, window):
    returns = closes.pct_change()
    daily_macro = macro.reindex(closes.index, method="ffill")
    return daily_macro.assign(
        equity_return=closes["SPY"].pct_change(window),
        equity_volatility=returns["SPY"].rolling(window).std() * np.sqrt(SESSIONS_PER_YEAR),
        equity_bond_correlation=returns["SPY"].rolling(window).corr(returns["TLT"]),
    )


def policy_weights(mean, covariance, upper_bounds, model):
    diagonal_share = model["covariance_diagonal_weight"]
    diagonal = np.diag(np.diag(covariance))
    annual_covariance = SESSIONS_PER_YEAR * (
        (1 - diagonal_share) * covariance + diagonal_share * diagonal
    )
    annual_mean = SESSIONS_PER_YEAR * mean
    penalty = model["risk_aversion"] * annual_covariance
    initial = upper_bounds / upper_bounds.sum()
    result = minimize(
        lambda weights: 0.5 * weights @ penalty @ weights - annual_mean @ weights,
        initial,
        jac=lambda weights: penalty @ weights - annual_mean,
        bounds=list(zip(np.zeros_like(upper_bounds), upper_bounds, strict=True)),
        constraints={
            "type": "eq", "fun": lambda weights: weights.sum() - 1,
            "jac": lambda weights: np.ones_like(weights),
        },
        method="SLSQP", options={"ftol": 1e-10, "maxiter": 100},
    )
    if not result.success or abs(result.x.sum() - 1) > 1e-9:
        raise RuntimeError(f"Mean-variance optimization failed: {result.message}")
    return result.x / result.x.sum()


def initialize_gate(train, model):
    scale = StandardScaler().fit(train)
    mixture = GaussianMixture(
        n_components=model["n_regimes"], covariance_type="diag", n_init=3,
        max_iter=250, reg_covar=1e-5, random_state=model["seed"],
    ).fit(scale.transform(train))
    if not mixture.converged_:
        raise RuntimeError("Warm-up Gaussian mixture did not converge")
    return scale, mixture


def blend_policies(gate, mass, first_moment, second_moment, base_mean, base_covariance, caps, model):
    policies = []
    for evidence, weighted_return, weighted_cross_product in zip(
        mass, first_moment, second_moment, strict=True,
    ):
        mean = weighted_return / evidence
        covariance = weighted_cross_product / evidence - np.outer(mean, mean)
        # Recent support determines how much to trust the state relative to the warm-up pool.
        share = evidence / (evidence + model["shrinkage_days"])
        difference = mean - base_mean
        shrink_mean = share * mean + (1 - share) * base_mean
        shrink_covariance = (
            share * covariance + (1 - share) * base_covariance
            + share * (1 - share) * np.outer(difference, difference)
        )
        policies.append(policy_weights(shrink_mean, shrink_covariance, caps, model))
    return gate @ np.vstack(policies)


def monthly_targets(opens, closes, macro, config):
    model = config["model"]
    lag = config["signal_lag_sessions"]
    start = pd.Timestamp(config["start"])
    features = market_features(closes, macro, model["feature_sessions"]).to_numpy()
    months = opens.index.to_period("M")
    fills = {
        close_row: opens.index[close_row + lag]
        for close_row in np.flatnonzero(months[1:] != months[:-1])
        if close_row + lag < len(opens) and opens.index[close_row + lag] >= start
    }
    if not fills:
        raise ValueError("No monthly rebalances in the backtest window")
    upper_bounds = np.full(opens.shape[1], model["max_risky_weight"])
    upper_bounds[opens.columns.get_loc(config["cash_proxy"])] = 1.0

    first_signal = min(fills)
    # A label ends at open(i + lag + 1), which must already have occurred at this close.
    bootstrap_stop = first_signal - lag
    bootstrap_start = bootstrap_stop - model["bootstrap_sessions"]
    if bootstrap_start < model["feature_sessions"]:
        raise ValueError(f"Insufficient warm-up history before {opens.index[first_signal].date()}")
    bootstrap_features = features[bootstrap_start:bootstrap_stop]
    realized = opens.div(opens.shift(1)).sub(1).to_numpy()
    bootstrap_returns = realized[bootstrap_start + lag + 1:first_signal + 1]
    scale, mixture = initialize_gate(bootstrap_features, model)
    base_mean = bootstrap_returns.mean(axis=0)
    base_covariance = np.cov(bootstrap_returns, rowvar=False, bias=True)
    policy_decay = np.exp2(-1 / model["policy_half_life_sessions"])
    gate_decay = np.exp2(-1 / model["gate_half_life_sessions"])
    ages = np.arange(len(bootstrap_features) - 1, -1, -1)
    memberships = mixture.predict_proba(scale.transform(bootstrap_features))
    importance = (policy_decay ** ages)[:, None] * memberships
    mass = importance.sum(axis=0)
    first_moment = importance.T @ bootstrap_returns
    second_moment = np.einsum("ik,ij,il->kjl", importance, bootstrap_returns, bootstrap_returns)
    base_prior = mixture.weights_
    live_prior = base_prior.copy()
    anchor = model["gate_baseline_share"]

    targets = {}
    pending = {}
    for close_row in range(first_signal, max(fills) + 1):
        if close_row > first_signal:
            mass *= policy_decay
            first_moment *= policy_decay
            second_moment *= policy_decay
        completed_row = close_row - lag - 1
        if completed_row in pending:
            membership = pending.pop(completed_row)
            outcome = realized[close_row]
            mass += membership
            first_moment += membership[:, None] * outcome
            second_moment += membership[:, None, None] * np.outer(outcome, outcome)

        initial_posterior = mixture.predict_proba(scale.transform(features[[close_row]]))[0]
        prior = (1 - anchor) * live_prior + anchor * base_prior
        gate = initial_posterior * (prior / base_prior)
        gate /= gate.sum()
        pending[close_row] = gate
        live_prior = gate_decay * live_prior + (1 - gate_decay) * gate
        if close_row in fills:
            targets[fills[close_row]] = blend_policies(
                gate, mass, first_moment, second_moment, base_mean, base_covariance, upper_bounds, model,
            )
    return targets


def rebalance(open_positions, cash, target, cost_rate):
    if not np.isfinite(target).all() or (target < 0).any() or abs(target.sum() - 1) > 1e-9:
        raise ValueError("Target weights must be long-only and fully allocated")
    target = target / target.sum()
    opening_value = open_positions.sum() + cash
    old_weights = open_positions / opening_value

    # Solve for the fraction of opening value that can buy target exposures after costs.
    # The lower endpoint always leaves a sufficient reserve for all buys and sells.
    lower, upper = 0.0, 1.0
    for _ in range(48):
        fraction = (lower + upper) / 2
        gross_turnover = np.abs(fraction * target - old_weights).sum()
        if fraction + gross_turnover * cost_rate <= 1:
            lower = fraction
        else:
            upper = fraction
    new_weights = lower * target
    orders = np.abs(new_weights - old_weights)
    gross_turnover = float(orders.sum())
    remaining_cash = (1 - (lower + gross_turnover * cost_rate)) * opening_value
    if remaining_cash < 0:
        raise RuntimeError("Rebalance exceeded available funds")
    return new_weights * opening_value, remaining_cash, gross_turnover, int((orders > 1e-10).sum())


def simulate(opens, closes, targets, config):
    start = pd.Timestamp(config["start"])
    cost_rate = (config["commission_bps"] + config["slippage_bps"]) / 10_000
    overnight = opens.div(closes.shift(1)).loc[start:]
    intraday = closes.div(opens).loc[start:]
    positions = np.zeros(opens.shape[1])
    cash = previous_value = 1.0
    daily_results = []
    for date, gap, session in zip(overnight.index, overnight.to_numpy(), intraday.to_numpy(), strict=True):
        positions *= gap
        turnover, trades = 0.0, 0
        if date in targets:
            positions, cash, turnover, trades = rebalance(positions, cash, targets[date], cost_rate)
        positions *= session
        value = positions.sum() + cash
        daily_results.append((value / previous_value - 1, turnover, trades))
        previous_value = value
    return pd.DataFrame(daily_results, index=overnight.index, columns=["return", "turnover", "n_trades"])


def metrics(results, cash_returns):
    daily = results["return"]
    excess = daily - cash_returns.reindex(daily.index)
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    equity = np.r_[1.0, (1 + daily).cumprod().to_numpy()]
    # The initial value also counts as a high-water mark in each measurement period.
    drawdowns = equity / np.maximum.accumulate(equity) - 1
    return {
        "cagr": float(equity[-1] ** (SESSIONS_PER_YEAR / len(daily)) - 1),
        "ann_vol": float(daily.std() * annualizer),
        "sharpe": float(daily.mean() / daily.std() * annualizer) if daily.std() else 0.0,
        "sharpe_excess": float(excess.mean() / excess.std() * annualizer) if excess.std() else 0.0,
        "max_drawdown": float(drawdowns.min()),
        "total_return": float(equity[-1] - 1),
        "turnover": float(results["turnover"].sum()),
        "n_trades": int(results["n_trades"].sum()),
    }


def period_metrics(results, cash_returns, split):
    return {
        "is": metrics(results.loc[results.index <= split], cash_returns),
        "oos": metrics(results.loc[results.index > split], cash_returns),
        "full": metrics(results, cash_returns),
    }


def main():
    with (ROOT / "config.yaml").open() as file:
        config = yaml.safe_load(file)
    data_start, start, split, end = map(pd.Timestamp, (
        config["data_start"], config["start"], config["is_end"], config["end"],
    ))
    if not data_start < start < split < end:
        raise ValueError("Configure warm-up, start, IS end and end in chronological order")
    if config["signal_lag_sessions"] < 1 or config["macro_lag_calendar_days"] < 1:
        raise ValueError("Signal and macro release lags must be positive")
    if min(config["commission_bps"], config["slippage_bps"]) < 0 or (
        config["commission_bps"] + config["slippage_bps"] >= 10_000
    ):
        raise ValueError("Total transaction friction must be in [0, 10000) bps")

    opens, closes, macro = load_inputs(config)
    cash_proxy = config["cash_proxy"]
    cash_returns = closes[cash_proxy].pct_change().loc[start:]
    first = cash_returns.index[0]
    cash_returns.loc[first] = closes.at[first, cash_proxy] / opens.at[first, cash_proxy] - 1

    trials = 0
    print("ROBUSTNESS (IS only)", flush=True)
    in_sample_open, in_sample_close, in_sample_macro = opens.loc[:split], closes.loc[:split], macro.loc[:split]
    for case in config["robustness"]:
        overrides = {key: value for key, value in case.items() if key not in {"name", "model"}}
        variant = dict(config, **overrides, model={**config["model"], **case.get("model", {})})
        targets = monthly_targets(in_sample_open, in_sample_close, in_sample_macro, variant)
        results = simulate(in_sample_open, in_sample_close, targets, variant)
        trials += 1
        print(f"  {case['name']}: {json.dumps(metrics(results, cash_returns), allow_nan=False)}", flush=True)

    targets = monthly_targets(opens, closes, macro, config)
    results = simulate(opens, closes, targets, config)
    trials += 1
    equal_weight = np.full(opens.shape[1], 1 / opens.shape[1])
    benchmark = simulate(opens, closes, dict.fromkeys(targets, equal_weight), config)
    strategy_periods = period_metrics(results, cash_returns, split)
    benchmark_periods = period_metrics(benchmark, cash_returns, split)
    if any(strategy_periods[period]["n_trades"] == 0 for period in ("is", "oos")):
        raise RuntimeError("The default specification must execute trades in both periods")

    report = {
        "strategy": config["strategy"],
        "data_range": {"start": str(results.index[0].date()), "end": str(results.index[-1].date())},
        **strategy_periods,
        "benchmark": {"name": "Monthly equal-weight: " + "/".join(opens.columns), **benchmark_periods},
    }
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(strategy_periods[period], allow_nan=False)}")
        print(f"BENCHMARK {period.upper()}: {json.dumps(benchmark_periods[period], allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {trials}")
    print("RESULTS_JSON: " + json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(13 * 60)
    with threadpool_limits(limits=1):
        main()
