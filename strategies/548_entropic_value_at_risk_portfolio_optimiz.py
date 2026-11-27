"""548_entropic_value_at_risk_portfolio_optimiz
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/548_entropic_value_at_risk_portfolio_optimiz.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c0a8f76240a0431b019ab49c34040a95a765860a on 2026-11-27 10:04 Europe/Oslo, exit 0, output sha256 7a8a8cf1fff887763fa8edf88eb4577c63a7c56e5f6e5d3ad768708f9c6b2140
Data source: Public Yahoo Finance daily data via yfinance. Select Sector SPDR ETF closes for XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV and XLY, plus the ^IRX 91-day T-bill discount quote for excess-return evaluation.
Data fetched: 2026-11-27T09:04:49+00:00
Sample period: Price request starts 1998-12-22 for warm-up. The logged evaluation runs 2001-07-02 through 2026-03-31 inclusive.   data_range (from output): {'start': '2001-07-02', 'end': '2026-03-31'}
IS/OOS split: IS is 2001-07-02 through 2014-12-31 inclusive. OOS starts on the first trading session of 2015 and ends 2026-03-31. The retrospective split slices one continuous default simulation, carrying IS closing weights into OOS. Alternative specifications are evaluated in IS only.
Costs and slippage: Default costs are 5 bps plus 2 bps of modeled slippage per dollar bought or sold, including initial entry. The simulator deducts the combined rate at the execution close on actual ETF trade notional, solving for post-cost portfolio value so the target weights are fully funded. The monthly equal-weight benchmark uses the same default costs. The IS stress case uses 20 bps plus 5 bps of slippage per side.
Signal lag: The signal uses the return window through the final trading session's close in each month. It fills at the next trading session's close under the configured one-session lag. Pre-rebalance holdings earn the fill day's close-to-close return. The target weights earn returns starting with the following close-to-close interval.
Price adjustment: yfinance auto_adjust=True total-return close (sector ETFs)
Survivorship bias: The fixed list uses currently surviving sector ETFs, so survivorship bias is possible. XLRE and XLC have later first-close dates and qualify only after a complete finite return window. Sector classifications, fund exposures and index rules can vary over time.
Parameter combinations tried: 6
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.07872060721614993, ann_vol=0.1471814076304864, sharpe=0.58852061141004, sharpe_excess=0.4900005267040017, max_drawdown=-0.3675921076457643, total_return=1.776435697424448, turnover=0.8027505007565849, n_trades=530
    oos: cagr=0.08361801410340688, ann_vol=0.14451492317013548, sharpe=0.6281899936190294, sharpe_excess=0.4872299248565395, max_drawdown=-0.2925093363128783, total_return=1.4617857315413356, turnover=0.9417729379877461, n_trades=430
    full: cagr=0.08094266383947574, ann_vol=0.14596447453754782, sharpe=0.6063853148546147, sharpe_excess=0.4887732554485077, max_drawdown=-0.3675921076457643, total_return=5.834989784461553, turnover=0.8659059611539003, n_trades=960
    benchmark  full: cagr=0.09065407478578284, ann_vol=0.1819581388634223, sharpe=0.5681821752030523, sharpe_excess=0.47384834939293247, max_drawdown=-0.5329383239019848, total_return=7.524328475874429, turnover=0.19509046115916592, n_trades=2843
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 548_entropic_value_at_risk_portfolio_optimiz\nuniverse: [XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY]\nrisk_free: "^IRX"\nprice_start: 1998-12-22\nstart: 2001-07-02\nis_end: 2014-12-31\nend: 2026-03-31\nexecution_lag_sessions: 1\n\ndefault:\n  lookback_sessions: 504\n  confidence: 0.95\n  cost_bps: 5\n  slippage_bps: 2\n\nrobustness_is:\n  lower_confidence: {confidence: 0.925}\n  higher_confidence: {confidence: 0.975}\n  shorter_lookback: {lookback_sessions: 462}\n  longer_lookback: {lookback_sessions: 546}\n  higher_cost: {cost_bps: 20, slippage_bps: 5}\n'
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
"""Monthly long-only allocation by empirical Entropic Value-at-Risk."""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
FIRST_CLOSE = dict.fromkeys("XLB XLE XLF XLI XLK XLP XLU XLV XLY".split(), "1998-12-22")
FIRST_CLOSE.update(XLRE="2015-10-08", XLC="2018-06-19")
RISK_FREE = "^IRX"


@dataclass(frozen=True)
class Specification:
    lookback_sessions: int
    confidence: float
    cost_bps: float
    slippage_bps: float

    def __post_init__(self):
        if type(self.lookback_sessions) is not int or self.lookback_sessions < 2:
            raise ValueError("lookback_sessions must be an integer of at least two")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between zero and one")
        if not (0 <= self.cost_bps and 0 <= self.slippage_bps and self.drag < 1):
            raise ValueError("Costs and slippage must be nonnegative and sum to less than 10000 bps")

    @property
    def drag(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def load_config():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for key in ("price_start", "start", "is_end", "end"):
        config[key] = pd.Timestamp(config[key])
    if not config["price_start"] < config["start"] <= config["is_end"] < config["end"]:
        raise ValueError("Dates must satisfy price_start < start <= is_end < end")
    universe = config["universe"]
    if not universe or not set(universe) <= FIRST_CLOSE.keys() or len(universe) != len(set(universe)):
        raise ValueError("universe must contain unique Select Sector SPDR tickers")
    if config["risk_free"] != RISK_FREE:
        raise ValueError(f"risk_free must be {RISK_FREE} for the discount-rate conversion")
    lag = config["execution_lag_sessions"]
    if type(lag) is not int or lag < 1:
        raise ValueError("execution_lag_sessions must be an integer of at least one")
    default = Specification(**config["default"])
    alternatives = {
        name: replace(default, **changes)
        for name, changes in config["robustness_is"].items()
    }
    if "default" in alternatives:
        raise ValueError("default is reserved for the official specification")
    if len({default, *alternatives.values()}) != 1 + len(alternatives):
        raise ValueError("Strategy specifications must be distinct")
    return config, default, alternatives


def check_prices(close, sectors, price_start, end):
    if close.empty or close.index.has_duplicates or not close.index.is_monotonic_increasing:
        raise ValueError("Price snapshot has no rows or has invalid dates")
    sector_close = close[sectors].dropna(how="all")
    if sector_close.empty or sector_close.index[-1] < end - pd.Timedelta(days=7):
        raise ValueError(f"Missing sector price coverage near {end.date()}")
    for ticker in sectors:
        prices = sector_close[ticker]
        first = prices.first_valid_index()
        expected_first = max(price_start, pd.Timestamp(FIRST_CLOSE[ticker]))
        if first is None or first > expected_first + pd.Timedelta(days=7):
            raise ValueError(f"Missing initial price history for {ticker}")
        if not (np.isfinite(prices.loc[first:]) & (prices.loc[first:] > 0)).all():
            raise ValueError(f"{ticker} has missing or invalid prices after its first close")
    if not np.isfinite(close[RISK_FREE]).any():
        raise ValueError(f"Missing price history for {RISK_FREE}")
    return sector_close


def load_prices(config):
    price_start, end = config["price_start"], config["end"]
    sectors = [t for t in config["universe"] if pd.Timestamp(FIRST_CLOSE[t]) <= end]
    if not sectors:
        raise ValueError("No sectors had launched by the configured end date")
    tickers = [*sectors, RISK_FREE]
    universe_key = sha256(",".join(sorted(tickers)).encode()).hexdigest()[:12]
    CACHE_DIR.mkdir(exist_ok=True)
    snapshot = CACHE_DIR / f"yahoo_adjusted_{price_start.date()}_{end.date()}_{universe_key}.csv"
    if snapshot.exists():
        with snapshot.open(encoding="utf-8") as source:
            fetched_at = source.readline().strip().removeprefix("# fetched_at=")
            datetime.fromisoformat(fetched_at)
            close = pd.read_csv(
                source, index_col=0, parse_dates=True,
                converters=dict.fromkeys(tickers, lambda value: float(value or "nan")),
            )
        return check_prices(close, sectors, price_start, end), close[RISK_FREE], fetched_at

    with TemporaryDirectory(dir=CACHE_DIR) as vendor_cache:
        yf.set_tz_cache_location(vendor_cache)
        prices = yf.download(
            tickers, start=price_start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True, interval="1d", threads=4, progress=False, timeout=10,
        )
    if prices is None or prices.empty:
        raise RuntimeError("Yahoo Finance returned no daily prices")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    close = prices["Close"].reindex(columns=tickers)
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    sector_close = check_prices(close, sectors, price_start, end)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            tmp.write(f"# fetched_at={fetched_at}\n")
            close.to_csv(tmp)
            tmp.flush()
            tmp_path.replace(snapshot)
        finally:
            tmp_path.unlink(missing_ok=True)
    return sector_close, close[RISK_FREE], fetched_at


def risk_free_returns(quotes, sessions):
    # Use the latest published close strictly before this session.
    quotes = quotes[np.isfinite(quotes)]
    prior = quotes.reindex(
        sessions - pd.Timedelta(days=1), method="ffill", tolerance=pd.Timedelta(days=10)
    )
    bill_price = 1.0 - (prior.to_numpy() / 100) * 91 / 360
    return pd.Series(np.expm1(-np.log(bill_price) * 365.25 / (91 * SESSIONS_PER_YEAR)), index=sessions)


def minimum_evar_weights(losses, confidence):
    # In percentage points, v = 1/u gives a joint convex perspective.
    size = losses.shape[1]
    uniform = np.full(size, 1.0 / size)
    if size == 1:
        return uniform
    tail_penalty = -np.log1p(-confidence)
    min_v = 1e-6
    start_v = max(10 * min_v, np.std(losses @ uniform) / np.sqrt(2 * tail_penalty))

    def objective(choice):
        weights, v = choice[:-1], choice[-1]
        portfolio_losses = losses @ weights
        maximum = portfolio_losses.max()
        shifted = (portfolio_losses - maximum) / v
        mass = np.exp(shifted)
        mass_sum = mass.sum()
        probabilities = mass / mass_sum
        log_term = np.log(mass_sum / len(losses)) + tail_penalty
        value = maximum + v * log_term
        gradient = np.r_[probabilities @ losses, log_term - probabilities @ shifted]
        return value, gradient

    budget_jacobian = np.r_[np.ones(size), 0.0]
    solution = minimize(
        objective, np.r_[uniform, start_v], jac=True, method="SLSQP",
        bounds=[(0, 1)] * size + [(min_v, None)],
        constraints=[{
            "type": "eq",
            "fun": lambda choice: budget_jacobian @ choice - 1.0,
            "jac": lambda _choice: budget_jacobian,
        }],
        options={"ftol": 1e-10, "maxiter": 200},
    )
    if not solution.success or not np.isfinite(solution.fun):
        raise RuntimeError(f"EVaR optimization failed: {solution.message}")
    weights = np.clip(solution.x[:-1], 0, 1)
    if abs(weights.sum() - 1.0) > 1e-8:
        raise RuntimeError("EVaR optimization missed the budget constraint")
    return weights / weights.sum()


def build_targets(returns, start, lag, spec, include_benchmark=False):
    sessions = returns.index
    months = sessions.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])
    series = returns.to_numpy()
    targets, equal_weights = {}, {}
    for signal_index in month_ends:
        fill_index = signal_index + lag
        if fill_index >= len(sessions) or sessions[fill_index] < start:
            continue
        window_start = signal_index + 1 - spec.lookback_sessions
        if window_start < 0:
            continue
        window = series[window_start:signal_index + 1]
        eligible = np.isfinite(window).all(axis=0)
        if not eligible.any():
            continue
        fill_date = sessions[fill_index]
        weights = np.zeros(len(returns.columns))
        try:
            weights[eligible] = minimum_evar_weights(-100.0 * window[:, eligible], spec.confidence)
        except RuntimeError as error:
            raise RuntimeError(f"Signal on {sessions[signal_index].date()}: {error}") from error
        targets[fill_date] = weights
        if include_benchmark:
            equal_weights[fill_date] = eligible.astype(float) / eligible.sum()
    if not targets:
        raise ValueError("No monthly signals have sufficient history and an execution date")
    return targets, equal_weights


def simulate(daily_returns, targets, spec, risk_free):
    # The only missing ETF returns are before listing or on the first available close.
    asset_returns = daily_returns.fillna(0.0).to_numpy()
    weights = np.zeros(len(daily_returns.columns))
    daily_results = []
    for date, returns in zip(daily_returns.index, asset_returns, strict=True):
        gross_change = 1.0 + weights @ returns
        drifted = weights * (1.0 + returns) / gross_change
        target = targets.get(date)
        turnover, trades = 0.0, 0
        weights = drifted
        if target is not None:
            turnover = 0.5 * (np.abs(target - drifted).sum() + abs(1.0 - drifted.sum()))

            # Solve for the post-cost fraction whose target weights are fully funded.
            def funding_gap(fraction):
                return fraction + spec.drag * np.abs(fraction * target - drifted).sum() - 1.0

            fraction = brentq(funding_gap, 0.0, 1.0)
            trades = np.count_nonzero(np.abs(fraction * target - drifted) > 1e-8)
            gross_change *= fraction
            weights = target
        daily_results.append((gross_change - 1.0, turnover, trades))
    result = pd.DataFrame(
        daily_results, index=daily_returns.index, columns=["return", "turnover", "n_trades"]
    )
    result["risk_free"] = risk_free.reindex(result.index)
    if not np.isfinite(result["risk_free"]).all():
        raise ValueError(f"Missing a recent prior {RISK_FREE} close during the evaluation period")
    return result


def metrics(period):
    if len(period) < 2:
        raise ValueError("Each evaluation period needs at least two sessions")
    returns = period["return"]
    excess = returns - period["risk_free"]
    vol, excess_vol = float(returns.std()), float(excess.std())
    path = np.r_[0.0, np.log1p(returns.to_numpy()).cumsum()]
    years = len(period) / SESSIONS_PER_YEAR
    return {
        "cagr": float(np.expm1(path[-1] / years)),
        "ann_vol": vol * np.sqrt(SESSIONS_PER_YEAR),
        "sharpe": float(returns.mean()) * np.sqrt(SESSIONS_PER_YEAR) / vol if vol else None,
        "sharpe_excess": float(excess.mean()) * np.sqrt(SESSIONS_PER_YEAR) / excess_vol if excess_vol else None,
        "max_drawdown": float(np.expm1(path - np.maximum.accumulate(path)).min()),
        "total_return": float(np.expm1(path[-1])),
        "turnover": float(period["turnover"].sum()) / years,
        "n_trades": int(period["n_trades"].sum()),
    }


def split_metrics(daily_results, is_end):
    return {
        "is": metrics(daily_results.loc[:is_end]),
        "oos": metrics(daily_results.loc[daily_results.index > is_end]),
        "full": metrics(daily_results),
    }


def main():
    config, default, alternatives = load_config()
    close, quotes, fetched_at = load_prices(config)
    start, is_end, lag = config["start"], config["is_end"], config["execution_lag_sessions"]
    returns = close.pct_change(fill_method=None)
    scored = returns.loc[start:]
    risk_free = risk_free_returns(quotes, scored.index)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True total-return close (sector ETFs)")

    targets, equal_weights = build_targets(returns, start, lag, default, include_benchmark=True)
    daily_results = simulate(scored, targets, default, risk_free)
    official = split_metrics(daily_results, is_end)
    benchmark = split_metrics(simulate(scored, equal_weights, default, risk_free), is_end)
    for name, report in (("STRATEGY", official), ("BENCHMARK monthly equal-weight sectors", benchmark)):
        for period, values in report.items():
            print(f"{name} {period.upper()}: {json.dumps(values, allow_nan=False)}")

    print("ROBUSTNESS (IS only)")
    print(f"default IS {default}: {json.dumps(official['is'], allow_nan=False)}")
    is_returns = returns.loc[:is_end]
    base_is_targets = {date: weights for date, weights in targets.items() if date <= is_end}
    for name, spec in alternatives.items():
        same_signal = (spec.lookback_sessions, spec.confidence) == (default.lookback_sessions, default.confidence)
        if same_signal:
            is_targets = base_is_targets
        else:
            is_targets, _ = build_targets(is_returns, start, lag, spec)
        values = metrics(simulate(scored.loc[:is_end], is_targets, spec, risk_free.loc[:is_end]))
        print(f"{name} IS {spec}: {json.dumps(values, allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {1 + len(alternatives)}")

    result = {
        "strategy": config["strategy"],
        "data_range": {"start": str(scored.index[0].date()), "end": str(scored.index[-1].date())},
        **official,
        "benchmark": {"name": "monthly_equal_weight_sectors", **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    main()