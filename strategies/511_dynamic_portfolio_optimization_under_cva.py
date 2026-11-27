"""511_dynamic_portfolio_optimization_under_cva
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/511_dynamic_portfolio_optimization_under_cva.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c0a8f76240a0431b019ab49c34040a95a765860a on 2026-11-27 10:04 Europe/Oslo, exit 0, output sha256 80485668f0bb2a3c816df6b93fc34b873fa018bb96e996667ac7f94c03407bda
Data source: Public yfinance daily adjusted closes for SPY and BIL, read from a local price snapshot when valid and downloaded from yfinance otherwise. BIL also supplies the cash-proxy return.
Data fetched: 2026-11-27T09:04:12+00:00
Sample period: Requested price warmup starts 2007-10-01. The recorded evaluation runs from 2009-01-02 through 2025-12-31 on common trading sessions.   data_range (from output): {'start': '2009-01-02', 'end': '2025-12-31'}
IS/OOS split: IS includes sessions through 2017-12-31. OOS includes sessions after that date through 2025-12-31. The official simulation runs continuously across the split and carries forward portfolio state. The five alternative settings are evaluated on IS prices only.
Costs and slippage: The default charges 2 bps in fees and models slippage as a fixed 3 bps charge per ETF buy or sell leg. The combined 5 bps is applied to each leg's absolute traded notional, with trade sizing solved net of the charge. This includes the initial BIL allocation and the benchmark's initial SPY purchase.
Signal lag: A signal computed at the first common session's close each month fills at the next common session's close. The existing weights earn the fill day's close-to-close return, and the new weights earn returns starting with the following close-to-close interval. The inception-close BIL allocation is fixed in advance.
Price adjustment: yfinance auto_adjust=True, adjusted Close with distributions and splits
Survivorship bias: The fixed SPY/BIL universe avoids backfilling historical stock membership with today's constituent list. The two surviving ETFs were selected retrospectively, so ETF selection and survivorship bias remain.
Parameter combinations tried: 6
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.05565306803976333, ann_vol=0.08121606191022353, sharpe=0.7076397706412602, sharpe_excess=0.6989994426926684, max_drawdown=-0.13889481209004317, total_return=0.6271274795421922, turnover=2.34785470623866, n_trades=161
    oos: cagr=0.053049348687101316, ann_vol=0.08678435293894238, sharpe=0.641100242486871, sharpe_excess=0.3613992574433916, max_drawdown=-0.15543789767256655, total_return=0.5125732083219408, turnover=2.8558857873805756, n_trades=148
    full: cagr=0.05442570415963144, ann_vol=0.08387101148415133, sharpe=0.6749539544689964, sharpe_excess=0.5344194284157883, max_drawdown=-0.15543789767256622, total_return=1.4611494320799219, turnover=2.587178515044679, n_trades=309
    benchmark  full: cagr=0.1450916912079374, ann_vol=0.17887800435260817, sharpe=0.8483905539629366, sharpe_excess=0.7818750116957577, max_drawdown=-0.33717265401236207, total_return=8.998677147358219, turnover=0.05881423715401845, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 511_dynamic_portfolio_optimization_under_cva\n\nuniverse:\n  risky: SPY\n  defensive: BIL\n\ndates:\n  price_start: 2007-10-01\n  start: 2009-01-02\n  is_end: 2017-12-31\n  end: 2025-12-31\n\nmodel:\n  lookback_sessions: 252\n  risk_horizon_sessions: 21\n  drawdown_lookback_sessions: 252\n  alpha: 0.95\n  risk_aversion: 5.0\n  cvar_limit: 0.05\n  cvar_floor: 0.01\n  drawdown_slope: 0.50\n\nexecution:\n  lag_sessions: 1\n  cost_bps: 2.0\n  slippage_bps: 3.0\n\nrobustness:\n  lookback_offsets: [-21, 21]\n  cvar_limit_offsets: [-0.01, 0.01]\n  friction_multiplier: 2.0\n'
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
"""Monthly SPY/BIL allocation with a drawdown-sensitive empirical CVaR limit."""

from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_ADJUSTMENT = "yfinance auto_adjust=True, adjusted Close with distributions and splits"
TRADE_TOLERANCE = 1e-12
SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class Settings:
    lookback_sessions: int
    risk_horizon_sessions: int
    drawdown_lookback_sessions: int
    alpha: float
    risk_aversion: float
    cvar_limit: float
    cvar_floor: float
    drawdown_slope: float
    lag_sessions: int
    cost_bps: float
    slippage_bps: float

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000

    def validate(self):
        counts = (
            self.lookback_sessions, self.risk_horizon_sessions,
            self.drawdown_lookback_sessions, self.lag_sessions,
        )
        if any(type(count) is not int or count < 1 for count in counts):
            raise ValueError("Session counts and execution lag must be positive integers.")
        if self.lookback_sessions <= self.risk_horizon_sessions:
            raise ValueError("Lookback must exceed the scenario horizon.")
        if not 0 < self.alpha < 1 or not 0 < self.cvar_floor <= self.cvar_limit < 1:
            raise ValueError("Invalid CVaR confidence or limits.")
        if not np.isfinite(self.risk_aversion) or self.risk_aversion <= 0:
            raise ValueError("Risk aversion must be positive and finite.")
        if not np.isfinite(self.drawdown_slope) or self.drawdown_slope < 0:
            raise ValueError("Drawdown slope must be nonnegative and finite.")
        if not (self.cost_bps >= 0 and self.slippage_bps >= 0 and self.friction < 1):
            raise ValueError("Invalid costs or slippage in basis points.")


def validate_prices(prices, symbols, start, end):
    if list(prices.columns) != list(symbols) or len(prices) < 2:
        raise ValueError("Adjusted-close series for SPY and BIL are required.")
    dates = prices.index
    if (not isinstance(dates, pd.DatetimeIndex) or dates.tz is not None or dates.hasnans
            or not dates.is_monotonic_increasing or dates.has_duplicates):
        raise ValueError("Price dates must be unique and chronological.")
    one_week = pd.Timedelta(days=7)
    if not (start <= dates[0] <= start + one_week and end - one_week <= dates[-1] <= end):
        raise ValueError("Price coverage does not reach the requested date boundaries.")
    if (dates[1:] - dates[:-1]).max() > one_week:
        raise ValueError("Price calendar contains a gap of more than a week.")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("A common session has a missing or invalid adjusted close.")


def load_prices(symbols, start, end):
    CACHE.mkdir(exist_ok=True)
    cache_path = CACHE / f"{'_'.join(symbols)}_{start.date()}_{end.date()}_adjusted.csv"
    if cache_path.exists():
        try:
            with cache_path.open(encoding="utf-8") as snapshot:
                header = snapshot.readline().strip()
                if not header.startswith("# fetched_at="):
                    raise ValueError("Missing fetch timestamp.")
                fetched_at = header.removeprefix("# fetched_at=")
                if datetime.fromisoformat(fetched_at).tzinfo is None:
                    raise ValueError("Fetch timestamp has no time zone.")
                prices = pd.read_csv(snapshot, index_col="Date", parse_dates=["Date"],
                                     converters=dict.fromkeys(symbols, float))
            validate_prices(prices, symbols, start, end)
            return prices, fetched_at
        except (OSError, ValueError) as error:
            print(f"Refreshing incomplete price cache ({error})", file=sys.stderr)

    with TemporaryDirectory(dir=CACHE, prefix=".yfinance-") as download_cache:
        yf.set_tz_cache_location(download_cache)
        for attempt in range(2):
            raw = yf.download(
                list(symbols), start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
                interval="1d", auto_adjust=True, ignore_tz=True, multi_level_index=True,
                progress=False, threads=False, timeout=15,
            )
            try:
                prices = raw["Close"].reindex(columns=symbols).astype(float).sort_index()
                validate_prices(prices, symbols, start, end)
            except (KeyError, ValueError) as error:
                if attempt == 1:
                    raise RuntimeError(f"Could not obtain the requested yfinance closes: {error}") from error
            else:
                break

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as snapshot:
        temporary_path = Path(snapshot.name)
        try:
            snapshot.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(snapshot, index_label="Date")
            snapshot.flush()
            temporary_path.replace(cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def choose_weight(scenarios, drawdown, settings):
    """Maximize mean minus a variance penalty over the convex feasible interval."""
    defensive = scenarios[:, 1]
    spread = scenarios[:, 0] - defensive
    centered_spread = spread - spread.mean()
    variance = np.mean(centered_spread**2)
    covariance = np.mean(centered_spread * (defensive - defensive.mean()))
    slope = spread.mean() - settings.risk_aversion * covariance
    if variance == 0:
        preferred = float(slope > 0)
    else:
        preferred = float(np.clip(slope / (settings.risk_aversion * variance), 0, 1))

    budget = max(settings.cvar_floor, settings.cvar_limit - settings.drawdown_slope * drawdown)
    tail_mass = (1 - settings.alpha) * len(scenarios)
    whole = int(tail_mass)
    fraction = tail_mass - whole

    def residual(weight):
        # Fractionally include the boundary scenario for an exact empirical tail mean.
        losses = np.sort(-(defensive + weight * spread))[::-1]
        tail_mean = (losses[:whole].sum() + fraction * losses[whole:whole + 1].sum()) / tail_mass
        return tail_mean - budget

    if residual(preferred) <= 0:
        return preferred

    anchor = 0.0
    if residual(anchor) > 0:
        minimum = minimize_scalar(residual, bounds=(0.0, 1.0), method="bounded",
                                  options={"xatol": 1e-12})
        anchor = min((0.0, float(minimum.x), 1.0), key=residual)
        if residual(anchor) >= 0:
            return anchor

    lower, upper = sorted((anchor, preferred))
    return float(brentq(residual, lower, upper, xtol=1e-12))


def rebalance(weights, target, friction):
    """Size both ETF legs after fees, with dollars scaled by pre-fill value."""
    if np.max(np.abs(target - weights)) <= TRADE_TOLERANCE:
        return 1.0, 0.0, 0

    def spent(multiplier):
        return np.abs(multiplier * target - weights).sum()

    multiplier = brentq(lambda scale: scale + friction * spent(scale) - 1.0, 0.0, 1.0)
    traded = np.abs(multiplier * target - weights)
    return multiplier, float(traded.sum()), int(np.count_nonzero(traded > TRADE_TOLERANCE))


def simulate(prices, start, settings):
    settings.validate()
    dates = prices.index
    first = dates.searchsorted(start)
    if first < settings.lookback_sessions or first >= len(prices) - 1:
        raise ValueError("Prices must cover the lookback and at least two evaluation sessions.")

    factors = (prices / prices.shift(1)).to_numpy()
    scenarios = (prices / prices.shift(settings.risk_horizon_sessions) - 1).to_numpy()
    weights = np.zeros(2)
    growth = 1.0
    recent_growth = deque([growth], maxlen=settings.drawdown_lookback_sessions + 1)
    pending = {}
    rows = []

    for i in range(first, len(prices)):
        target = pending.pop(i, None)
        if i == first:
            # Start from cash with a fixed BIL allocation at the inception close.
            gross_factor = 1.0
            proxy_return = 0.0
            target = np.array([0.0, 1.0])
        else:
            marked = weights * factors[i]
            gross_factor = marked.sum()
            weights = marked / gross_factor
            proxy_return = factors[i, 1] - 1.0

        multiplier, turnover, trades = 1.0, 0.0, 0
        if target is not None:
            multiplier, turnover, trades = rebalance(weights, target, settings.friction)
            if turnover:
                weights = target

        daily_factor = gross_factor * multiplier
        growth *= daily_factor
        recent_growth.append(growth)
        rows.append((daily_factor - 1.0, proxy_return, turnover, trades))

        if dates[i].month != dates[i - 1].month:
            drawdown = 1.0 - growth / max(recent_growth)
            first_scenario = i - settings.lookback_sessions + settings.risk_horizon_sessions
            risky_weight = choose_weight(scenarios[first_scenario:i + 1], drawdown, settings)
            fill = i + settings.lag_sessions
            pending[fill] = np.array([risky_weight, 1.0 - risky_weight])

    return pd.DataFrame(rows, index=dates[first:],
                        columns=["return", "proxy_return", "turnover", "n_trades"])


def buy_and_hold(prices, daily_results, friction):
    baseline = daily_results.copy()
    multiplier, turnover, trades = rebalance(np.zeros(2), np.array([1.0, 0.0]), friction)
    baseline["return"] = prices.iloc[:, 0].pct_change().loc[baseline.index]
    baseline["turnover"] = 0.0
    baseline["n_trades"] = 0
    first = baseline.index[0]
    baseline.loc[first, ["return", "turnover", "n_trades"]] = multiplier - 1.0, turnover, trades
    return baseline


def metrics(daily, starting_close):
    years = (daily.index[-1] - starting_close).days / 365.2425 if len(daily) else 0
    if len(daily) < 2 or years <= 0:
        raise ValueError("Each period needs at least two sessions and a positive duration.")

    def sharpe(returns):
        deviation = returns.std(ddof=1)
        return float(np.sqrt(SESSIONS_PER_YEAR) * returns.mean() / deviation) if deviation else None

    returns = daily["return"]
    curve = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    return {
        "cagr": float(curve[-1] ** (1 / years) - 1),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(returns - daily["proxy_return"]),
        "max_drawdown": float(np.min(curve / np.maximum.accumulate(curve) - 1)),
        "total_return": float(curve[-1] - 1),
        "turnover": float(daily["turnover"].sum() / years),
        "n_trades": int(daily["n_trades"].sum()),
    }


def period_metrics(daily_results, split):
    insample = daily_results.loc[:split]
    outsample = daily_results.loc[daily_results.index > split]
    inception = daily_results.index[0]
    return {
        "is": metrics(insample, inception),
        "oos": metrics(outsample, insample.index[-1]),
        "full": metrics(daily_results, inception),
    }


def robustness_cases(official, options):
    for offset in options["lookback_offsets"]:
        yield replace(official, lookback_sessions=official.lookback_sessions + offset)
    for offset in options["cvar_limit_offsets"]:
        yield replace(official, cvar_limit=round(official.cvar_limit + offset, 12))
    scale = options["friction_multiplier"]
    yield replace(official, cost_bps=official.cost_bps * scale,
                  slippage_bps=official.slippage_bps * scale)


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    symbols = (config["universe"]["risky"], config["universe"]["defensive"])
    if symbols != ("SPY", "BIL"):
        raise ValueError("This specification uses SPY and BIL.")
    dates = config["dates"]
    price_start, start, split, end = (
        pd.Timestamp(dates[key]) for key in ("price_start", "start", "is_end", "end")
    )
    if not price_start < start <= split < end:
        raise ValueError("Invalid warmup, evaluation, or split dates.")

    official = Settings(**config["model"], **config["execution"])
    official.validate()
    prices, fetched_at = load_prices(symbols, price_start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")

    # The original parameters and alternatives are fixed before either sample is scored.
    cases = dict.fromkeys(robustness_cases(official, config["robustness"]))
    cases.pop(official, None)
    insample_prices = prices.loc[:split]
    print("ROBUSTNESS (IS only)")
    for case in cases:
        daily_results = simulate(insample_prices, start, case)
        parameters = (f"lookback={case.lookback_sessions}, cvar_limit={case.cvar_limit:g}, "
                      f"cost_bps={case.cost_bps:g}, slippage_bps={case.slippage_bps:g}")
        print(f"  {parameters} " + json.dumps(metrics(daily_results, daily_results.index[0]),
                                               allow_nan=False))

    daily_results = simulate(prices, start, official)
    performance = period_metrics(daily_results, split)
    benchmark = buy_and_hold(prices, daily_results, official.friction)
    benchmark_performance = period_metrics(benchmark, split)
    data_range = {"start": str(daily_results.index[0].date()),
                  "end": str(daily_results.index[-1].date())}
    print(f"DATE RANGE: {data_range['start']} to {data_range['end']}")
    print(f"IS through {split.date()} | OOS after {split.date()} | Excess Sharpe vs BIL")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: " + json.dumps(performance[period], allow_nan=False))
        print(f"SPY BUY-AND-HOLD {period.upper()}: "
              + json.dumps(benchmark_performance[period], allow_nan=False))

    print(f"PARAM_COMBOS_TRIED: {1 + len(cases)}")
    results = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **performance,
        "benchmark": {"name": "SPY buy-and-hold", **benchmark_performance},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(840)
    main()
