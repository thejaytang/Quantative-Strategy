"""582_the_geometry_of_higher_order_modern_port
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/582_the_geometry_of_higher_order_modern_port.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 25a9179813de3addbbaee65a8c08579eb6e27d08 on 2026-11-27 19:08 Europe/Oslo, exit 0, output sha256 979dca4c41538dcc148be4c68a02e46a75cc19e351f0c171fc998f4ab0526889
Data source: Free Yahoo Finance daily ETF opens, closes and corporate actions via yfinance for SPY, QQQ, IWM, EFA, IEF, TLT and GLD. BIL comes from the same feed and serves only as an ex-post cash benchmark.
Data fetched: 2026-11-27T18:08:48.730412+00:00
Sample period: Backtest sessions run from 2009-01-02 through 2025-12-31, with the first simulated fill on 2009-01-02. The data request starts on 2007-07-01 to supply signal warm-up history.   data_range (from output): 2009-01-02 to 2025-12-31
IS/OOS split: IS is 2009-01-02 through 2018-12-31. OOS is 2019-01-02 through 2025-12-31. The configured default runs continuously through the retrospective split, carrying positions, cash and receivables and re-estimating targets from rolling history. The five robustness variants run only through the IS cutoff.
Costs and slippage: Each buy and sell incurs 5 bps transaction cost plus 3 bps modeled slippage on its opening-price trade notional. The combined 8 bps applies to the sum of absolute trade notionals and is deducted from cash. The solver includes the charge when sizing the batch, and simulated fills use the opening prices.
Signal lag: After the last trading session of each quarter closes, the default model computes target weights using the trailing 252 daily returns through that close. The target is sized and filled at the next trading session's open.
Price adjustment: Yahoo split-adjusted Open/Close; ex-date cash added for signal returns and accrued separately for portfolio valuation
Survivorship bias: The ETF universe was fixed at the research date. All selected ETFs had data before the backtest began and continued to exist. Historical candidate lists and exited funds were excluded, leaving survivorship and retrospective selection bias.
Parameter combinations tried: 6
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.05738545128141248, ann_vol=0.08560680887226037, sharpe=0.6947156609030024, sharpe_excess=0.6661934057872119, max_drawdown=-0.13316316884458124, total_return=0.7456157424796652, turnover=2.499912214519875, n_trades=191
    oos: cagr=0.0921041982407973, ann_vol=0.1066504263908576, sharpe=0.8796799205899042, sharpe_excess=0.6429203814476173, max_drawdown=-0.24314082339169862, total_return=0.8502958297087548, turnover=1.8931063179496677, n_trades=122
    full: cagr=0.0715399821062821, ann_vol=0.09482949578412338, sharpe=0.776226091933369, sharpe_excess=0.6515733511718139, max_drawdown=-0.24314082339169918, total_return=2.2299055285840486, turnover=2.250151134547105, n_trades=313
    benchmark: cagr=0.10353942791606818, ann_vol=0.10805600235810019, sharpe=0.9660060119500856, sharpe_excess=0.855733315214343, max_drawdown=-0.2507735721575748, total_return=4.3215136691437, turnover=0.25570096753174015, n_trades=476
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 582_the_geometry_of_higher_order_modern_port\nuniverse: [SPY, QQQ, IWM, EFA, IEF, TLT, GLD]\ncash_proxy: BIL\ndata_start: 2007-07-01\nsample_start: 2009-01-01\nis_end: 2018-12-31\nsample_end: 2025-12-31\nsignal_lag_sessions: 1\ndistribution_delay_days: 60\n\nmodel:\n  lookback_sessions: 252\n  risk_aversion: 25.0\n  cumulant_order: 4\n  max_weight: 0.45\n  cost_bps: 5.0\n  slippage_bps: 3.0\n\nrobustness:\n  lookback_210: {lookback_sessions: 210}\n  lookback_294: {lookback_sessions: 294}\n  risk_aversion_20: {risk_aversion: 20.0}\n  risk_aversion_30: {risk_aversion: 30.0}\n  mean_variance_only: {cumulant_order: 2}\n'
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
"""Quarterly cross-asset ETF allocation with four sample cumulants.

Paper: Emil Horobet, The geometry of higher order modern portfolio theory (2025).
https://arxiv.org/abs/2511.20674

Research references (ideas only; independently written):
https://github.com/dppalomar/highOrderPortfolios (GPL-3.0)
https://github.com/dppalomar/highOrderPortfolios/blob/master/R/MVSK.R
https://arxiv.org/abs/2008.00863
https://github.com/dthinkr/yand-mvsk (MIT, Ben Wu)
https://github.com/dthinkr/yand-mvsk/blob/main/src/yand_mvsk/_core.py
https://pypi.org/project/yand-mvsk/
https://portfoliooptimizationbook.com/book/9.2-high-order-moments.html
https://github.com/dcajasn/Riskfolio-Lib
https://portfoliooptimization.org/riskfoliolib/portfolio.html
"""

from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class Model:
    lookback_sessions: int
    risk_aversion: float
    cumulant_order: int
    max_weight: float
    cost_bps: float
    slippage_bps: float

    @property
    def fill_rate(self):
        return (self.cost_bps + self.slippage_bps) / 10_000

    def check(self, n_assets):
        if type(self.lookback_sessions) is not int or self.lookback_sessions < 4:
            raise ValueError("The lookback must be an integer of at least four sessions")
        if self.cumulant_order not in (2, 4) or not 0 < self.risk_aversion < np.inf:
            raise ValueError("Set cumulant order to 2 or 4 and use positive finite risk aversion")
        if not 1 / n_assets <= self.max_weight <= 1:
            raise ValueError("The weight cap must permit full deployment")
        if not (self.cost_bps >= 0 and self.slippage_bps >= 0 and self.fill_rate < 1):
            raise ValueError("Costs and slippage must be nonnegative with a combined rate below one")


def check_quotes(quotes, symbols, start, end):
    dates = quotes.index
    if (quotes.empty or not dates.is_unique or not dates.is_monotonic_increasing
            or dates[0] < start or dates[-1] > end
            or (dates[0] - start).days > 7 or (end - dates[-1]).days > 7):
        raise ValueError("Yahoo snapshot has invalid session dates or incomplete endpoint coverage")
    for field in ("Open", "Close"):
        prices = quotes[field].reindex(columns=symbols)
        valid = np.isfinite(prices) & (prices > 0)
        if not valid.to_numpy().all():
            date_row, symbol_col = np.argwhere(~valid.to_numpy())[0]
            raise ValueError(f"Missing or invalid {field} for {symbols[symbol_col]} on {dates[date_row].date()}")


def load_quotes(symbols, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    key = json.dumps(["yahoo_ohlc_actions_v1", symbols, str(start.date()), str(end.date())])
    cache = cache_dir / f"yahoo_{hashlib.sha256(key.encode()).hexdigest()[:16]}.csv"
    if cache.exists():
        with cache.open(encoding="utf-8") as stream:
            fetched_at = datetime.fromisoformat(stream.readline().strip())
            quotes = pd.read_csv(stream, index_col=0, header=[0, 1], parse_dates=True)
        check_quotes(quotes, symbols, start, end)
        return quotes, fetched_at

    with TemporaryDirectory(dir=cache_dir) as transient_cache:
        yf.set_tz_cache_location(transient_cache)
        quotes = yf.download(
            symbols, start=start.date().isoformat(), end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=False, actions=True, repair=False,
            threads=False, progress=False, timeout=12,
        )
    check_quotes(quotes, symbols, start, end)
    fetched_at = datetime.now(timezone.utc)
    # Publish one complete snapshot, including its original fetch time.
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_dir, delete=False) as stream:
        staging = Path(stream.name)
        try:
            stream.write(fetched_at.isoformat() + "\n")
            quotes.to_csv(stream)
            stream.close()
            staging.replace(cache)
        finally:
            staging.unlink(missing_ok=True)
    return quotes, fetched_at


def cash_distributions(quotes, symbols):
    actions = quotes.reindex(columns=pd.MultiIndex.from_product([
        ["Dividends", "Capital Gains"], symbols,
    ])).fillna(0.0)
    if not (np.isfinite(actions) & (actions >= 0)).to_numpy().all():
        raise ValueError("Invalid Yahoo cash distribution")
    return actions["Dividends"] + actions["Capital Gains"]


def sample_objective(sample, model):
    mean = sample.mean(axis=0)
    centered = sample - mean
    count = len(sample)
    gamma = model.risk_aversion
    variance_weight = gamma / 2
    skew_weight = gamma**2 / 6 if model.cumulant_order == 4 else 0.0
    tail_weight = gamma**3 / 24 if model.cumulant_order == 4 else 0.0

    def loss_and_gradient(weights):
        deviations = centered @ weights
        squared = deviations**2
        variance = float(squared.mean())
        skew_cumulant = float(np.mean(squared * deviations))
        fourth_cumulant = float(np.mean(squared**2)) - 3 * variance**2
        loss = -mean @ weights + variance_weight * variance - skew_weight * skew_cumulant
        loss += tail_weight * fourth_cumulant

        # The derivative of m4 - 3*m2**2 includes -6*m2*grad(m2).
        derivative = (2 * (variance_weight - 6 * tail_weight * variance) * deviations
                      - 3 * skew_weight * squared + 4 * tail_weight * squared * deviations)
        gradient = -mean + centered.T @ derivative / count
        return 10_000 * loss, 10_000 * gradient

    return loss_and_gradient


def optimize_weights(sample, model):
    n_assets = sample.shape[1]
    equal = np.full(n_assets, 1 / n_assets)
    if model.max_weight == equal[0]:
        return equal

    objective = sample_objective(sample, model)
    other_weight = (1 - model.max_weight) / (n_assets - 1)
    tilted = other_weight + (model.max_weight - other_weight) * np.eye(n_assets)
    budget = {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)}
    candidates = []
    for seed in (equal, *tilted):
        result = minimize(
            objective, seed, jac=True, method="SLSQP", constraints=budget,
            bounds=[(0.0, model.max_weight)] * n_assets,
            options={"maxiter": 200, "ftol": 1e-10},
        )
        if result.success and np.isfinite(result.fun) and abs(result.x.sum() - 1) < 1e-8:
            weights = np.clip(result.x, 0, model.max_weight)
            candidates.append(weights / weights.sum())
    if not candidates:
        raise RuntimeError("No SLSQP start returned a feasible allocation")
    return min(candidates, key=lambda weights: objective(weights)[0])


def quarterly_targets(returns, start, lag, model):
    dates = returns.index
    quarters = dates.to_period("Q")
    samples = returns.to_numpy()
    targets = {}
    for signal_row in np.flatnonzero(quarters[1:] != quarters[:-1]):
        fill_row = signal_row + lag
        if fill_row >= len(dates) or dates[fill_row] < start:
            continue
        sample_start = signal_row - model.lookback_sessions + 1
        if sample_start < 1:
            raise ValueError(f"Insufficient warm-up for the {dates[signal_row].date()} signal")
        sample = samples[sample_start:signal_row + 1]
        targets[fill_row] = optimize_weights(sample, model)
    if not targets:
        raise ValueError("No quarterly target falls in the requested sample")
    return targets


def fill_at_open(positions, cash, open_prices, weights, fill_rate):
    opening_values = positions * open_prices
    available = float(opening_values.sum() + cash)
    # Solve target value + costs = available, retaining the feasible side.
    low, high = 0.0, available
    for _ in range(48):
        target_value = (low + high) / 2
        desired = target_value * weights
        gross = np.abs(desired - opening_values).sum()
        if desired.sum() + fill_rate * gross <= available:
            low = target_value
        else:
            high = target_value
    desired = low * weights
    deltas = np.abs(desired - opening_values)
    gross = float(deltas.sum())
    remaining = available - (float(desired.sum()) + fill_rate * gross)
    if not remaining >= 0:
        raise RuntimeError("Opening fill exceeded the available cash and sale proceeds")
    count = int(np.count_nonzero(deltas > available * 1e-10))
    return desired / open_prices, remaining, gross, count


def simulate(opens, closes, distributions, start, targets, rate, distribution_delay):
    dates = closes.index
    opening_prices = opens.to_numpy()
    closing_prices = closes.to_numpy()
    payouts = distributions.to_numpy()
    first_row = dates.searchsorted(start)
    positions = np.zeros(closes.shape[1])
    unpaid = deque()
    receivables = 0.0
    cash = previous_value = 1.0
    rows = []
    for row in range(first_row, len(dates)):
        # Ex-date cash belongs to the position carried into the session.
        prior_positions = positions
        turnover = 0.0
        trades = 0
        if row in targets:
            open_prices = opening_prices[row]
            opening_value = float(positions @ open_prices + cash + receivables)
            positions, cash, gross, trades = fill_at_open(positions, cash, open_prices, targets[row], rate)
            turnover = gross / opening_value

        distribution = float(prior_positions @ payouts[row])
        date = dates[row]
        if distribution:
            unpaid.append((date + distribution_delay, distribution))
            receivables += distribution
        while unpaid and unpaid[0][0] <= date:
            _, paid = unpaid.popleft()
            cash += paid
            receivables -= paid

        value = float(positions @ closing_prices[row] + cash + receivables)
        if not np.isfinite(value) or value <= 0:
            raise RuntimeError(f"Invalid closing portfolio value on {date.date()}")
        rows.append((value / previous_value - 1, turnover, trades))
        previous_value = value
    return pd.DataFrame(rows, index=dates[first_row:], columns=["return", "turnover", "n_trades"])


def metrics(daily_results, cash_returns):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise ValueError("At least two sessions are required per reporting period")
    excess = returns - cash_returns.loc[returns.index]
    growth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    scale = np.sqrt(SESSIONS_PER_YEAR)
    deviation, excess_deviation = returns.std(), excess.std()
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(deviation * scale),
        "sharpe": float(returns.mean() * scale / deviation) if deviation else None,
        "sharpe_excess": float(excess.mean() * scale / excess_deviation) if excess_deviation else None,
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1)),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def periods(daily_results, cash_returns, is_end):
    return {
        "is": metrics(daily_results.loc[:is_end], cash_returns),
        "oos": metrics(daily_results.loc[daily_results.index > is_end], cash_returns),
        "full": metrics(daily_results, cash_returns),
    }


def report(label, result):
    values = " | ".join(f"{name}={value:.6f}" if isinstance(value, float) else f"{name}={value}"
                        for name, value in result.items())
    print(f"{label} | {values}")


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    cash_proxy = config["cash_proxy"]
    if len(universe) < 2 or len(set(universe)) != len(universe) or cash_proxy in universe:
        raise ValueError("Supply a distinct ETF universe and a separate cash proxy")
    data_start, sample_start, is_end, sample_end = (
        pd.Timestamp(config[key]) for key in ("data_start", "sample_start", "is_end", "sample_end")
    )
    if not data_start < sample_start < is_end < sample_end:
        raise ValueError("Expected data_start < sample_start < is_end < sample_end")
    lag = config["signal_lag_sessions"]
    delay_days = config["distribution_delay_days"]
    if type(lag) is not int or lag < 1 or type(delay_days) is not int or delay_days < 1:
        raise ValueError("Signal lag and distribution delay must be positive integers")
    distribution_delay = pd.Timedelta(days=delay_days)
    official = Model(**config["model"])
    variants = {name: replace(official, **overrides) for name, overrides in config["robustness"].items()}
    if len({official, *variants.values()}) != 1 + len(variants):
        raise ValueError("Each robustness configuration must be distinct")
    for model in (official, *variants.values()):
        model.check(len(universe))

    symbols = [*universe, cash_proxy]
    quotes, fetched_at = load_quotes(symbols, data_start, sample_end)
    distributions = cash_distributions(quotes, symbols)
    closes = quotes["Close"][symbols]
    returns = (closes + distributions).div(closes.shift(1)) - 1
    cash_returns = returns[cash_proxy]
    opens = quotes["Open"][universe]
    closes = closes[universe]
    distributions = distributions[universe]
    returns = returns[universe]

    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}")
    print("PRICE_ADJUSTMENT: Yahoo split-adjusted Open/Close; ex-date cash added for signal returns "
          "and accrued separately for portfolio valuation")
    print(f"EXECUTION: lag={lag} session(s) | sizing=open | fill=open | "
          f"cost_bps={official.cost_bps:g} | slippage_bps={official.slippage_bps:g}")

    is_robustness = {}
    is_returns = returns.loc[:is_end]
    for name, model in variants.items():
        targets = quarterly_targets(is_returns, sample_start, lag, model)
        daily_results = simulate(opens.loc[:is_end], closes.loc[:is_end], distributions.loc[:is_end],
                                 sample_start, targets, model.fill_rate, distribution_delay)
        is_robustness[name] = metrics(daily_results, cash_returns.loc[:is_end])

    targets = quarterly_targets(returns, sample_start, lag, official)
    daily_results = simulate(opens, closes, distributions, sample_start, targets,
                             official.fill_rate, distribution_delay)
    equal_weights = np.full(len(universe), 1 / len(universe))
    benchmark_results = simulate(opens, closes, distributions, sample_start,
                                 dict.fromkeys(targets, equal_weights), official.fill_rate, distribution_delay)
    summary = periods(daily_results, cash_returns, is_end)
    benchmark_summary = periods(benchmark_results, cash_returns, is_end)
    if summary["full"]["n_trades"] == 0:
        raise RuntimeError("The default configuration produced no fills")

    dates = daily_results.index
    data_range = f"{dates[0].date()} to {dates[-1].date()}"
    first_fill = daily_results.index[daily_results["n_trades"] > 0][0].date()
    print(f"SAMPLE: {data_range} | IS through {is_end.date()} | first fill {first_fill}")
    print("LOOKAHEAD_SELF_CHECK: lagged close windows; no macro or intraday release inputs; "
          "split-normalized prices and ex-date distributions; IS-only variants and continuous OOS")
    print(f"已回测 | 研究用途，待审 | excess Sharpe vs {cash_proxy}")
    for period in ("is", "oos", "full"):
        report(f"Strategy {period.upper()}", summary[period])
        report(f"Equal-weight {period.upper()}", benchmark_summary[period])
    print("ROBUSTNESS | IS only | default fixed in config.yaml")
    report("default", summary["is"])
    for name, result in is_robustness.items():
        report(name, result)
    print(f"PARAM_COMBOS_TRIED: {1 + len(is_robustness)}")
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **summary,
        "benchmark": {"name": "quarterly_equal_weight", **benchmark_summary["full"], **benchmark_summary},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(840)
    main()
