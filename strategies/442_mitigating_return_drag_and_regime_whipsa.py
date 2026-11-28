"""442_mitigating_return_drag_and_regime_whipsa
WARNING: known defects (see DEEP_REVIEW_FINDINGS.md); numbers invalid.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/442_mitigating_return_drag_and_regime_whipsa.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 6df04a4d89c55b7a1b162dbdcff0dafc1e2bd682 on 2026-11-27 08:09 Europe/Oslo, exit 0, output sha256 18269de031c2fbbc5d14600fd6a8b535c88b433de1d97ca2449dc046647550e0
Data source: Yahoo Finance via yfinance, daily closes for sector ETFs XLK, XLF, XLV, XLE, XLI, XLY, XLP, XLU, XLB, XLRE, XLC and benchmark SPY, plus ^VIX index levels and ^IRX annualized 13-week U.S. Treasury bill yield quotes.
Data fetched: 2026-11-27T07:09:42+00:00
Sample period: Daily price history 2018-07-30 through 2025-12-31, with monthly returns from 2018-08 through 2020-07 used for warm-up. The reported evaluation begins on the first fill, 2020-08-03, and ends 2025-12-31.   data_range (from output): 2020-08-03 to 2025-12-31
IS/OOS split: IS 2020-08-03 through 2022-12-30 and OOS 2023-01-03 through 2025-12-31. The calendar split is 2023-01-01. The same fixed parameter configuration is used throughout. The model state and holdings carry across the boundary, and the code splits the simulated history for reporting. The README describes the OOS interval as retrospective and overlapping the paper's sample.
Costs and slippage: Each simulated buy or sell incurs 5 bps transaction cost plus 5 bps fixed slippage, charged together at 10 bps on absolute traded notional. Existing holdings drift through the fill-day close before rebalancing. The simulator solves post-cost NAV and traded notional jointly. Slippage is modelled as a fixed NAV deduction. The same rate applies to the benchmark's initial buy.
Signal lag: The signal uses month-end ETF and SPY closes on session t. Target weights are filled at the next session's close (t+1). The close-to-close return ending on the fill day accrues to existing holdings before the fill. VIX and IRX are shifted by one trading session before use, and the monthly hurdle uses the previous month-end's lagged IRX.
Price adjustment: yfinance auto_adjust total-return ETF closes, VIX and IRX index levels
Survivorship bias: The fixed universe consists of 11 existing sector ETFs, and the sample starts within their common history. The README identifies possible ex-post selection bias from the fixed fund universe and says historical classification and constituent changes are reflected in fund returns.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.07944429546267373, ann_vol=0.14406697534278398, sharpe=0.6029186299241285, sharpe_excess=0.5442516687927351, max_drawdown=-0.17359677088744718, total_return=0.20291212535245995, turnover=0.3084857199461089, n_trades=115
    oos: cagr=0.13569605518085814, ann_vol=0.191795013615633, sharpe=0.7593796682531926, sharpe_excess=0.5151006962675895, max_drawdown=-0.23908072812265646, total_return=0.4618714191598525, turnover=0.17832558452446848, n_trades=86
    full: cagr=0.11017180962771933, ann_vol=0.17203285726808604, sharpe=0.6937116087752424, sharpe_excess=0.521277675764718, max_drawdown=-0.23908072812265713, total_return=0.7585028558135938, turnover=0.23639702955873876, n_trades=201
    benchmark: cagr=0.16085858719137214, ann_vol=0.17102660062116049, sharpe=0.957719413157952, sharpe_excess=0.7843088993056458, max_drawdown=-0.24496371769512282, total_return=1.2379981907765645, turnover=0.01536924613847691, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 442_mitigating_return_drag_and_regime_whipsa\n\nuniverse:\n  sectors: [XLK, XLF, XLV, XLE, XLI, XLY, XLP, XLU, XLB, XLRE, XLC]\n  benchmark: SPY\n  volatility_index: ^VIX\n  risk_free_yield: ^IRX\n\ndates:\n  fetch_start: 2018-07-30\n  evaluation_start: 2020-08-01\n  oos_start: 2023-01-01\n  end: 2025-12-31\n\nexecution:\n  lag_sessions: 1\n  cost_bps: 5.0\n  slippage_bps: 5.0\n\npublication_lag_sessions:\n  volatility_index: 1\n  risk_free_yield: 1\n\nmodel:\n  lookback_months: 24\n  clusters: 3\n  sticky_kappa: 0.30\n  confidence_threshold: 0.65\n  bull_return_weight: 1.2\n  turnover_penalty: 0.015\n  semicovariance_floor: 0.000001\n  hmm_initial_self_transition: 0.85\n  hmm_variance_floor: 0.001\n  hmm_max_iterations: 60\n  hmm_tolerance: 0.01\n'
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
"""Walk-forward, downside-risk nested clustered sector allocation."""

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.optimize import LinearConstraint, brentq, minimize
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
PARAM_COMBOS_TRIED = 1


def checked_closes(closes, tickers, benchmark, start, end):
    closes = closes.reindex(columns=tickers).sort_index()
    closes = closes.loc[start:end]
    if closes.empty or not closes.index.is_unique:
        raise ValueError("Yahoo Finance returned empty or duplicate daily dates")
    absent = closes.columns[closes.isna().all()].tolist()
    if absent:
        raise ValueError(f"Missing Yahoo Finance series: {absent}")
    # SPY supplies the ETF trading calendar. A hole within that calendar is an error.
    if closes[benchmark].isna().any():
        raise ValueError(f"Missing {benchmark} session close")
    if not np.isfinite(closes.to_numpy()).all():
        raise ValueError(f"Missing or nonfinite daily observations: {closes.isna().sum().to_dict()}")
    if (closes.index[0] - start).days > 4 or (end - closes.index[-1]).days > 4:
        raise ValueError("Yahoo Finance daily data does not cover the requested boundaries")
    if (closes.drop(columns=tickers[-1]) <= 0).any().any():
        raise ValueError("ETF prices and volatility levels must be positive")
    return closes


def load_closes(tickers, benchmark, start, end):
    cache = ROOT / "data_cache"
    cache.mkdir(exist_ok=True)
    key = sha256(repr((tickers, str(start.date()), str(end.date()), "adjusted-close-v1")).encode())
    path = cache / f"yahoo_{key.hexdigest()[:16]}.csv"
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            fetched_at = handle.readline().strip().removeprefix("# fetched_at=")
            datetime.fromisoformat(fetched_at)
            closes = pd.read_csv(handle, index_col=0, parse_dates=True)
        return checked_closes(closes, tickers, benchmark, start, end), fetched_at

    # Keep yfinance's transient transport cache within the strategy directory.
    with TemporaryDirectory(dir=cache) as transport_cache:
        yf.set_tz_cache_location(transport_cache)
        history = yf.download(
            tickers, start=start.date().isoformat(), end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=True, threads=4, timeout=12, progress=False,
            ignore_tz=True,
        )
    if history is None or history.empty or "Close" not in history:
        raise RuntimeError("No Yahoo Finance daily closes were downloaded")
    closes = checked_closes(history["Close"], tickers, benchmark, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache, delete=False) as handle:
        temporary_path = Path(handle.name)
        try:
            handle.write(f"# fetched_at={fetched_at}\n")
            closes.to_csv(handle)
            handle.flush()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return closes, fetched_at


def bull_probability(observations, model):
    """Fit a diagonal Gaussian HMM on the supplied history and filter its final state."""
    floor = model["hmm_variance_floor"]
    observations = np.asarray(observations)
    scale = np.maximum(observations.std(axis=0), np.sqrt(np.finfo(float).eps))
    values = (observations - observations.mean(axis=0)) / scale
    by_vix = values[np.argsort(values[:, 1], kind="stable")]
    halves = np.array_split(by_vix, 2)
    means = np.stack([half.mean(axis=0) for half in halves])
    variances = np.maximum(np.stack([half.var(axis=0) for half in halves]), floor)
    diagonal = model["hmm_initial_self_transition"]
    transition = np.array([[diagonal, 1 - diagonal], [1 - diagonal, diagonal]])
    initial = np.full(2, 0.5)
    kappa = model["sticky_kappa"]
    sticky = kappa * np.eye(2)
    n = len(values)

    def forward():
        residuals = values[:, None, :] - means
        log_density = -0.5 * (residuals**2 / variances + np.log(variances)).sum(axis=2)
        offsets = log_density.max(axis=1)
        density = np.exp(np.maximum(log_density - offsets[:, None], -700))
        filtered = np.empty((n, 2))
        scales = np.empty(n)
        predicted = initial
        for t in range(n):
            mass = predicted * density[t]
            scales[t] = mass.sum()
            filtered[t] = mass / scales[t]
            predicted = filtered[t] @ transition
        return filtered, density, scales, np.sum(np.log(scales) + offsets)

    last_likelihood = -np.inf
    for _ in range(model["hmm_max_iterations"]):
        filtered, density, scales, likelihood = forward()
        if abs(likelihood - last_likelihood) < model["hmm_tolerance"]:
            break
        last_likelihood = likelihood
        backward = np.ones((n, 2))
        for t in range(n - 2, -1, -1):
            backward[t] = transition @ (density[t + 1] * backward[t + 1]) / scales[t + 1]
        posterior = filtered * backward
        posterior /= posterior.sum(axis=1, keepdims=True)
        next_state = density[1:] * backward[1:] / scales[1:, None]
        moves = transition * (filtered[:-1].T @ next_state)
        moves = np.maximum(moves, 1e-12)
        transition = (moves / moves.sum(axis=1, keepdims=True) + sticky) / (1 + kappa)
        initial = np.maximum(posterior[0], 1e-12)
        initial /= initial.sum()
        state_mass = posterior.sum(axis=0)
        if np.any(state_mass < 1e-12):
            raise RuntimeError("A Gaussian HMM state has no observations")
        means = (posterior.T @ values) / state_mass[:, None]
        residuals = values[:, None, :] - means
        variances = np.maximum(
            (posterior[:, :, None] * residuals**2).sum(axis=0) / state_mass[:, None], floor
        )

    filtered, _, _, _ = forward()
    return float(filtered[-1, np.argmax(means[:, 0])])


def downside_semicovariance(returns, monthly_target, floor):
    shortfall = np.minimum(returns - monthly_target[:, None], 0.0)
    semicovariance = shortfall.T @ shortfall / len(shortfall)
    ridge = max(0.0, floor - np.linalg.eigvalsh(semicovariance)[0])
    return semicovariance + ridge * np.eye(semicovariance.shape[0])


def optimize_weights(covariance, expected_returns, is_bull, gamma, previous, penalty):
    size = len(expected_returns)
    if size == 1:
        return np.ones(1)

    # Auxiliary variables bound absolute changes to keep the L1 objective differentiable.
    dimensions = 2 * size if penalty else size
    simplex = np.zeros(dimensions)
    simplex[:size] = 1.0
    constraints = [LinearConstraint(simplex, 1.0, 1.0)]
    initial = previous
    if penalty:
        eye = np.eye(size)
        absolute_change = np.block([[eye, -eye], [-eye, -eye]])
        constraints.append(LinearConstraint(absolute_change, -np.inf, np.r_[previous, -previous]))
        initial = np.r_[previous, np.zeros(size)]

    quadratic = covariance if is_bull else 2 * covariance
    reward = gamma * expected_returns if is_bull else np.zeros(size)

    def objective(variables):
        weights = variables[:size]
        risk = quadratic @ weights
        value = 0.5 * weights @ risk - weights @ reward + penalty * variables[size:].sum()
        gradient = np.r_[risk - reward, np.full(dimensions - size, penalty)]
        return value, gradient

    solved = minimize(
        objective, initial, jac=True, method="SLSQP", bounds=[(0.0, 1.0)] * dimensions,
        constraints=constraints, options={"ftol": 1e-11, "maxiter": 200},
    )
    if not solved.success:
        raise RuntimeError(f"Nested cluster optimization failed: {solved.message}")
    weights = np.maximum(solved.x[:size], 0.0)
    return weights / weights.sum()


def nested_weights(returns, target, is_bull, previous, model):
    floor = model["semicovariance_floor"]
    gamma = model["bull_return_weight"]
    covariance = downside_semicovariance(returns, target, floor)
    deviations = np.sqrt(np.diag(covariance))
    correlation = np.clip(covariance / np.outer(deviations, deviations), -1.0, 1.0)
    distance = np.sqrt(0.5 * (1.0 - correlation))
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="ward")
    n_clusters = model["clusters"]
    labels = cut_tree(tree, n_clusters=n_clusters).ravel()
    within = np.zeros((len(previous), n_clusters))
    expected_returns = returns.mean(axis=0)
    for cluster in range(n_clusters):
        members = np.flatnonzero(labels == cluster)
        reference = previous[members]
        cluster_weight = reference.sum()
        reference = reference / cluster_weight if cluster_weight > 1e-10 else np.full(
            len(members), 1.0 / len(members)
        )
        within[members, cluster] = optimize_weights(
            covariance[np.ix_(members, members)], expected_returns[members], is_bull,
            gamma, reference, model["turnover_penalty"],
        )

    cluster_returns = returns @ within
    across = optimize_weights(
        downside_semicovariance(cluster_returns, target, floor), cluster_returns.mean(axis=0),
        is_bull, gamma, np.full(n_clusters, 1.0 / n_clusters), 0.0,
    )
    return within @ across


def build_targets(closes, config):
    universe = config["universe"]
    sectors = universe["sectors"]
    model = config["model"]
    index = closes.index
    lagged_vix = closes[universe["volatility_index"]].shift(
        config["publication_lag_sessions"]["volatility_index"]
    )
    lagged_yield = closes[universe["risk_free_yield"]].shift(
        config["publication_lag_sessions"]["risk_free_yield"]
    ) / 100.0
    month = index.to_period("M")
    month_ends = index[~month.duplicated(keep="last")]
    if len(month_ends) != len(pd.period_range(month[0], month[-1], freq="M")):
        raise ValueError("At least one calendar month has no daily observations")
    monthly = closes.loc[month_ends, [*sectors, universe["benchmark"]]].pct_change(fill_method=None)
    # A month's downside hurdle uses the yield observable at its opening boundary.
    hurdle = lagged_yield.loc[month_ends].shift(1) / 12.0
    features = pd.concat([monthly[universe["benchmark"]], lagged_vix.loc[month_ends]], axis=1)
    eligible = monthly.notna().all(axis=1) & hurdle.notna() & features.notna().all(axis=1)
    monthly, hurdle, features = monthly[eligible], hurdle[eligible], features[eligible]
    lookback = model["lookback_months"]
    dates = config["dates"]
    lag = config["execution"]["lag_sessions"]
    previous = np.full(len(sectors), 1.0 / len(sectors))
    is_bull = True
    fills = {}
    for row, signal_date in enumerate(monthly.index):
        fill_position = index.get_loc(signal_date) + lag
        if row + 1 < lookback or fill_position >= len(index):
            continue
        fill_date = index[fill_position]
        if fill_date < pd.Timestamp(dates["evaluation_start"]):
            continue

        probability = bull_probability(features.iloc[: row + 1].to_numpy(), model)
        confidence = model["confidence_threshold"]
        if is_bull and 1.0 - probability >= confidence:
            is_bull = False
        elif not is_bull and probability >= confidence:
            is_bull = True
        window = slice(row + 1 - lookback, row + 1)
        previous = nested_weights(
            monthly[sectors].iloc[window].to_numpy(), hurdle.iloc[window].to_numpy(),
            is_bull, previous, model,
        )
        fills[fill_date] = previous
    if not fills:
        raise ValueError("No month-end signal with enough history and an executable date")
    return fills, lagged_yield / SESSIONS_PER_YEAR


def simulate(daily_returns, fills, friction):
    positions = np.zeros(daily_returns.shape[1])
    cash = 1.0
    rows = []
    for date, returns in zip(daily_returns.index, daily_returns.to_numpy()):
        # Existing holdings earn today's close-to-close move before today's fill.
        positions *= 1.0 + returns
        gross = positions.sum() + cash
        positions /= gross
        cash /= gross
        turnover = 0.0
        trades = 0
        net = 1.0
        if date in fills:
            target = fills[date]
            # Solve post-fee NAV and actual traded notional on the same capital base.
            def balance(nav):
                notional = np.abs(nav * target - positions).sum()
                return nav + friction * notional - 1.0

            net = brentq(balance, 0.0, 1.0) if friction else 1.0
            changes = np.abs(net * target - positions)
            turnover = changes.sum()
            trades = np.count_nonzero(changes > 1e-8)
            positions = target.copy()
            cash = 0.0

        rows.append((gross * net - 1.0, turnover, trades))
    return pd.DataFrame(rows, index=daily_returns.index, columns=["return", "turnover", "n_trades"])


def metrics(period, daily_risk_free):
    returns = period["return"].to_numpy()
    if len(returns) < 2:
        raise ValueError("Each evaluation period needs at least two daily observations")
    excess = returns - daily_risk_free.loc[period.index].to_numpy()
    wealth = np.r_[1.0, np.cumprod(1.0 + returns)]
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    return {
        "cagr": float(wealth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1.0),
        "ann_vol": float(vol * annualizer),
        "sharpe": float(returns.mean() * annualizer / vol) if vol else None,
        "sharpe_excess": float(excess.mean() * annualizer / excess_vol) if excess_vol else None,
        "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0)),
        "total_return": float(wealth[-1] - 1.0),
        "turnover": float(period["turnover"].groupby(period.index.to_period("M")).sum().mean()),
        "n_trades": int(period["n_trades"].sum()),
    }


def period_metrics(history, daily_risk_free, oos_start):
    return {
        "is": metrics(history.loc[history.index < oos_start], daily_risk_free),
        "oos": metrics(history.loc[history.index >= oos_start], daily_risk_free),
        "full": metrics(history, daily_risk_free),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    universe = config["universe"]
    sectors, benchmark = universe["sectors"], universe["benchmark"]
    tickers = [*sectors, benchmark, universe["volatility_index"], universe["risk_free_yield"]]
    dates = config["dates"]
    start, evaluation_start, oos_start, end = (
        pd.Timestamp(dates[field]) for field in ("fetch_start", "evaluation_start", "oos_start", "end")
    )
    execution = config["execution"]
    bps = [execution["cost_bps"], execution["slippage_bps"]]
    if not start < evaluation_start < oos_start <= end:
        raise ValueError("Expected fetch_start < evaluation_start < oos_start <= end")
    if execution["lag_sessions"] < 1 or min(config["publication_lag_sessions"].values()) < 0:
        raise ValueError("Execution needs at least one session of lag and publication lags must be nonnegative")
    if min(bps) < 0 or sum(bps) >= 10000:
        raise ValueError("Expected nonnegative costs and slippage with a sum below 10000 bps")
    if not 1 <= config["model"]["clusters"] <= len(sectors) or len(sectors) < 2:
        raise ValueError("Expected at least two sectors and between one and N clusters")
    if len(tickers) != len(set(tickers)) or config["model"]["lookback_months"] < 3:
        raise ValueError("Expected distinct tickers and at least three months of model history")

    closes, fetched_at = load_closes(tickers, benchmark, start, end)
    fills, daily_risk_free = build_targets(closes, config)
    first_fill = next(iter(fills))
    daily = closes[[*sectors, benchmark]].pct_change(fill_method=None).loc[first_fill:]
    friction = sum(bps) / 10000.0
    strategy_returns = simulate(daily[sectors], fills, friction)
    benchmark_returns = simulate(daily[[benchmark]], {first_fill: np.ones(1)}, friction)
    strategy_periods = period_metrics(strategy_returns, daily_risk_free, oos_start)
    benchmark_periods = period_metrics(benchmark_returns, daily_risk_free, oos_start)
    data_range = f"{daily.index[0].date()} to {daily.index[-1].date()}"
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **strategy_periods,
        "benchmark": {"ticker": benchmark, **benchmark_periods["full"], **benchmark_periods},
    }

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return ETF closes, VIX and IRX index levels")
    print(f"PARAM_COMBOS_TRIED: {PARAM_COMBOS_TRIED}")
    print(f"PRICE_HISTORY: {closes.index[0].date()} to {closes.index[-1].date()}")
    print(f"EVALUATION: {data_range} (OOS from {oos_start.date()})")
    table = pd.DataFrame({
        **{f"Strategy {key.upper()}": value for key, value in strategy_periods.items()},
        **{f"{benchmark} {key.upper()}": value for key, value in benchmark_periods.items()},
    }).T.astype({"n_trades": int})
    print(table.to_string(float_format=lambda number: f"{number:.6f}"))
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()