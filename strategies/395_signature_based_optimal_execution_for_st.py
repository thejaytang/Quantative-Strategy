"""395_signature_based_optimal_execution_for_st
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/395_signature_based_optimal_execution_for_st.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 4d91396b470c7410b1364756fad54a1bec5cce1f on 2026-11-27 07:45 Europe/Oslo, exit 0, output sha256 a0603951913a091d0a20972f7aa6e5e7b7db0a4902f8b8f3d3405e8b4a3cdf82
Data source: Yahoo Finance via yfinance daily closing prices for SHEL (Shell) and BP, with BIL as the cash proxy for excess-return metrics, including validation. The loader uses a local price cache or downloads the three symbols from Yahoo Finance.
Data fetched: 2026-11-27T06:45:20+00:00
Sample period: Requested price history is 2022-02-01 inclusive through 2026-01-01 exclusive. The logged reporting period is 2023-01-03 through 2025-12-31 inclusive.   data_range (from output): 2023-01-03 to 2025-12-31
IS/OOS split: Initial calibration uses February–June 2022. Lookback selection uses July–December 2022 validation by excess Sharpe versus BIL. Reported IS spans 2023–2024 with expanding-history recalibration at each year boundary. Reported OOS spans 2025 with one policy fit using history through the final 2024 session and the previously selected lookback. Each fit uses complete historical episodes through the close before the segment begins.
Costs and slippage: The account applies 0.5 bps transaction cost plus 2.0 bps modelled slippage, for 2.5 bps one way on each leg's absolute traded notional at entries, rebalances and exits. Each close first marks the old positions and calculates their drifted weights. The sum of absolute target-minus-drifted weights is multiplied by 0.00025 and deducted from the pre-fill NAV. The threshold benchmark uses the same account function. The policy fit separately uses quadratic temporary-impact coefficients of 0.1 and 0.01. Cash earns zero. Borrow fees, financing fees, taxes and trade-size-dependent execution impact are outside the backtest.
Signal lag: A signal using close t maps to a target at the next trading session's close. The fill follows marking the existing holdings at that close, so the new holdings first accrue price returns over the following close-to-close interval. Scheduled liquidation overrides any signal arriving at the episode's final close. Both the signature strategy and the threshold benchmark use this timing.
Price adjustment: yfinance auto_adjust=True total-return adjusted Close
Survivorship bias: SHEL and BP are a fixed pair drawn from the paper's known surviving sample. The README identifies hindsight in pair selection and survivorship bias.
Parameter combinations tried: 3
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.028356826163000948, ann_vol=0.03775142501331932, sharpe=-0.7430701466877387, sharpe_excess=-2.057906516496046, max_drawdown=-0.0961874490383906, total_return=-0.05569397593124081, turnover=132.4941886100986, n_trades=1000
    oos: cagr=-0.07128765794544856, ann_vol=0.059833931486702976, sharpe=-1.205832690604267, sharpe_excess=-1.8887201068945385, max_drawdown=-0.07628468494952223, total_return=-0.07074238631565533, turnover=138.44135441688815, n_trades=496
    full: cagr=-0.04284485041633779, ann_vol=0.04625441691972641, sharpe=-0.9234169888488118, sharpe_excess=-1.933544072156635, max_drawdown=-0.16370763042513825, total_return=-0.12249643748611327, turnover=134.47130490224936, n_trades=1496
    benchmark: cagr=-0.011786753907241931, ann_vol=0.030376877624429817, sharpe=-0.37511620212539665, sharpe_excess=-1.906947844371158, max_drawdown=-0.05112819890834608, total_return=-0.034763473522819055, turnover=45.29650735545673, n_trades=450
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 395_signature_based_optimal_execution_for_st\nuniverse: [SHEL, BP]\ncash_proxy: BIL\n\ndata_start: '2022-02-01'\nvalidation_start: '2022-07-01'\nbacktest_start: '2023-01-01'\noos_start: '2025-01-01'\nend_exclusive: '2026-01-01'\n\nsignal_lag_sessions: 1\ncost_bps: 0.5\nslippage_bps: 2.0\ngross_limit: 1.0\n\nepisode_sessions: 4\nz_lookback_candidates: [5, 10, 20]\nbenchmark_entry_z: 2.0\nalpha_scale: 1.0\ntemporary_impact: [0.1, 0.01]\nterminal_inventory_penalty: 1.0\nridge: 1.0e-8\n"
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
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
Z_SIGNATURE_COORDINATE = 4
PRICE_ADJUSTMENT = "yfinance auto_adjust=True total-return adjusted Close"


def load_prices(config):
    tickers = [*config["universe"], config["cash_proxy"]]
    cache = ROOT / "data_cache"
    cache.mkdir(exist_ok=True)
    start, end = config["data_start"], config["end_exclusive"]
    path = cache / f"adjusted-close_{'_'.join(tickers)}_{start}_{end}.csv"
    if path.exists():
        with path.open() as stream:
            fetched_at = stream.readline().strip().removeprefix("# fetched_at=")
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(stream, index_col=0, parse_dates=True)[tickers]
    else:
        yf.set_tz_cache_location(str(cache / "py-yfinance"))
        quotes = yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False, timeout=15
        )
        prices = quotes["Close"].reindex(columns=tickers)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    prices = prices.sort_index()
    if (
        prices.empty
        or prices.index.has_duplicates
        or not np.isfinite(prices.to_numpy()).all()
        or (prices <= 0).any().any()
        or prices.index[0] < pd.Timestamp(start)
        or prices.index[0] - pd.Timestamp(start) > pd.Timedelta(days=7)
        or prices.index[-1] >= pd.Timestamp(end)
        or pd.Timestamp(end) - prices.index[-1] > pd.Timedelta(days=7)
    ):
        raise ValueError(f"Incomplete or invalid Yahoo Finance price history for {tickers}")

    if not path.exists():
        with NamedTemporaryFile(mode="w", dir=cache, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(f"# fetched_at={fetched_at}\n")
                prices.to_csv(stream)
                stream.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    return prices, fetched_at


def market_path(prices, lookback):
    log_prices = np.log(prices)
    spread = log_prices.iloc[:, 0] - log_prices.iloc[:, 1]
    # Normalize the current close against prior closes so spikes are not capped by the window size.
    rolling = spread.shift(1).rolling(lookback)
    scale = rolling.std(ddof=0).clip(lower=1e-12)
    return log_prices.assign(z=(spread - rolling.mean()) / scale).dropna()


def signature_prefixes(window, sessions):
    """Level 0, 1 and 2 signatures of successive piecewise linear prefixes."""
    time = np.arange(len(window)) / sessions
    path = np.column_stack((time, window))
    origin = path[0].copy()
    # A same-time initial segment makes the level-one z coordinate equal z itself.
    origin[-1] = 0.0
    levels_one = np.zeros(path.shape[1])
    levels_two = np.zeros((path.shape[1], path.shape[1]))
    features = []
    for increment in np.diff(path, axis=0, prepend=origin[np.newaxis, :]):
        levels_two += np.outer(levels_one, increment) + 0.5 * np.outer(increment, increment)
        levels_one += increment
        features.append(np.r_[1.0, levels_one, levels_two.ravel()])
    return np.asarray(features)


def episode_starts(observations, sessions):
    # The last close of an episode is also the first observation of the next one.
    return range(0, observations - sessions, sessions)


def fit_policy(history, config):
    sessions = config["episode_sessions"]
    lag = config["signal_lag_sessions"]
    dt = 1.0 / sessions
    episodes = [
        signature_prefixes(history.iloc[start : start + sessions].to_numpy(), sessions)
        for start in episode_starts(len(history), sessions)
    ]
    if not episodes:
        raise ValueError("No complete historical episode before calibration cutoff")

    features = np.stack(episodes)
    speeds = np.zeros_like(features)
    speeds[:, lag:] = features[:, :-lag]
    inventory = np.cumsum(speeds, axis=1) * dt
    # Project both legs onto the dollar-neutral direction (+1, -1).
    alpha = -2.0 * config["alpha_scale"] * features[:, :, Z_SIGNATURE_COORDINATE]
    impact = sum(config["temporary_impact"])
    terminal_penalty = 2.0 * config["terminal_inventory_penalty"]
    reward = dt * np.einsum("et,etf->f", alpha, inventory) / len(episodes)
    penalty = (
        dt * impact * np.einsum("etf,etg->fg", speeds, speeds)
        + terminal_penalty * (inventory[:, -1].T @ inventory[:, -1])
    ) / len(episodes)
    penalty += config["ridge"] * np.eye(len(reward))
    return np.linalg.solve(2.0 * penalty, reward)


def threshold_targets(z_scores, entry, limit):
    position = 0.0
    targets = []
    for z in z_scores:
        if position == 0.0:
            if abs(z) > entry:
                position = float(-np.sign(z) * limit)
        elif position * z >= 0.0:
            position = 0.0
        targets.append(position)
    return targets


def target_weights(path, dates, policy, config):
    sessions = config["episode_sessions"]
    lag = config["signal_lag_sessions"]
    limit = config["gross_limit"] / 2.0
    weights = np.zeros(len(dates))
    observations = path.reindex(dates).to_numpy()
    if not np.isfinite(observations).all():
        raise ValueError("Signal warmup must finish before the first evaluation close")
    for start in episode_starts(len(dates), sessions):
        # Orders arriving at the liquidation close are superseded by the scheduled exit.
        signal_window = observations[start : start + sessions - lag]
        if policy is None:
            targets = threshold_targets(signal_window[:, -1], config["benchmark_entry_z"], limit)
        else:
            speeds = signature_prefixes(signal_window, sessions) @ policy
            targets = []
            position = 0.0
            for speed in speeds:
                position = np.clip(position + speed / sessions, -limit, limit)
                targets.append(position)
        weights[start + lag : start + sessions] = targets
    return np.column_stack((weights, -weights))


def account(prices, targets, config):
    friction = (config["cost_bps"] + config["slippage_bps"]) / 10_000.0
    asset_returns = prices.pct_change(fill_method=None).iloc[1:]
    held = np.zeros(2)
    rows = []
    for returns, target in zip(asset_returns.to_numpy(), targets[1:], strict=True):
        # The current close marks the old positions before executing the delayed target.
        growth = 1.0 + held @ returns
        if growth <= 0:
            raise ValueError("Portfolio NAV exhausted before execution")
        drifted = held * (1.0 + returns) / growth
        changes = np.abs(target - drifted)
        turnover = changes.sum()
        net_of_cost = 1.0 - friction * turnover
        if net_of_cost <= 0:
            raise ValueError("Portfolio NAV exhausted by execution costs")
        held = target / net_of_cost
        rows.append((growth * net_of_cost - 1.0, turnover, np.count_nonzero(changes > 1e-12)))
    return pd.DataFrame(rows, index=asset_returns.index, columns=["return", "turnover", "trades"])


def evaluate(prices, path, begin, end, config, *, include_benchmark=False):
    first = prices.index.searchsorted(begin)
    stop = prices.index.searchsorted(end)
    if first == 0 or stop - first < config["episode_sessions"]:
        raise ValueError(f"Insufficient observations for evaluation from {begin} to {end}")
    window = prices.iloc[first - 1 : stop]
    policy = fit_policy(path.loc[: window.index[0]], config)
    history = account(window, target_weights(path, window.index, policy, config), config)
    if not include_benchmark:
        return history
    benchmark = account(window, target_weights(path, window.index, None, config), config)
    return history, benchmark


def metrics(history, cash_returns):
    returns = history["return"].to_numpy()
    excess = returns - cash_returns.loc[history.index].to_numpy()
    wealth = np.r_[1.0, np.cumprod(1.0 + returns)]
    scale = np.sqrt(SESSIONS_PER_YEAR)

    def sharpe(series):
        vol = series.std(ddof=1)
        return float(series.mean() * scale / vol) if vol > 0 else None

    return {
        "cagr": float(wealth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1.0),
        "ann_vol": float(returns.std(ddof=1) * scale),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(excess),
        "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1.0).min()),
        "total_return": float(wealth[-1] - 1.0),
        "turnover": float(history["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(history["trades"].sum()),
    }


def validate(config):
    dates = [
        pd.Timestamp(config[key])
        for key in ("data_start", "validation_start", "backtest_start", "oos_start", "end_exclusive")
    ]
    if any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("Require data start, validation start, backtest start, OOS start and end in order")
    if len(set([*config["universe"], config["cash_proxy"]])) != 3 or len(config["universe"]) != 2:
        raise ValueError("Require two distinct equities and a distinct cash proxy")
    if not 1 <= config["signal_lag_sessions"] < config["episode_sessions"]:
        raise ValueError("Signal lag must be positive and shorter than the execution episode")
    if not config["z_lookback_candidates"] or min(config["z_lookback_candidates"]) < 2:
        raise ValueError("Require at least one z-score lookback of two or more sessions")
    impact = config["temporary_impact"]
    if len(impact) != 2 or min(impact) <= 0 or config["ridge"] <= 0:
        raise ValueError("Require two positive impact coefficients and a positive ridge penalty")
    if (
        min(config["cost_bps"], config["slippage_bps"], config["terminal_inventory_penalty"]) < 0
        or config["gross_limit"] <= 0
        or config["benchmark_entry_z"] <= 0
    ):
        raise ValueError("Require nonnegative costs and terminal penalty, positive gross cap and entry threshold")


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    validate(config)
    prices, fetched_at = load_prices(config)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    pair = prices[config["universe"]]
    cash_returns = prices[config["cash_proxy"]].pct_change(fill_method=None)

    validation_start = pd.Timestamp(config["validation_start"])
    is_start = pd.Timestamp(config["backtest_start"])
    oos_start = pd.Timestamp(config["oos_start"])
    end = pd.Timestamp(config["end_exclusive"])
    training_prices = pair.loc[pair.index < is_start]
    scores = {}
    for lookback in dict.fromkeys(config["z_lookback_candidates"]):
        path = market_path(training_prices, lookback)
        history = evaluate(training_prices, path, validation_start, is_start, config)
        scores[lookback] = metrics(history, cash_returns)["sharpe_excess"]
    print(f"PARAM_COMBOS_TRIED: {len(scores)}")
    ranked = [(score, -lookback) for lookback, score in scores.items() if score is not None]
    if not ranked:
        raise ValueError("No candidate produced a defined excess Sharpe in validation")
    lookback = -max(ranked)[1]
    print(f"SELECTED_Z_LOOKBACK: {lookback} sessions (pre-reporting IS validation)")

    path = market_path(pair, lookback)
    boundaries = [is_start, *pd.date_range(is_start, oos_start, freq="YS", inclusive="neither"), oos_start, end]
    strategy_segments, benchmark_segments = [], []
    for start, stop in zip(boundaries, boundaries[1:]):
        strategy, benchmark = evaluate(pair, path, start, stop, config, include_benchmark=True)
        strategy_segments.append(strategy)
        benchmark_segments.append(benchmark)

    result = {"strategy": config["strategy"]}
    for label, segments in (("signature", strategy_segments), ("z-score threshold", benchmark_segments)):
        history = pd.concat(segments)
        periods = {
            "is": history.loc[history.index < oos_start],
            "oos": history.loc[history.index >= oos_start],
            "full": history,
        }
        reports = {period: metrics(segment, cash_returns) for period, segment in periods.items()}
        for period, report in reports.items():
            print(f"{label} {period.upper()} (excess Sharpe vs {config['cash_proxy']}) {json.dumps(report)}")
        if label == "signature":
            result["data_range"] = f"{history.index[0].date()} to {history.index[-1].date()}"
            result.update(reports)
        else:
            result["benchmark"] = reports["full"]
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()