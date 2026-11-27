"""692_dynamic_cvar_portfolio_construction_with
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/692_dynamic_cvar_portfolio_construction_with.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 9532cd9f815cc2d43ceb143b2f1416acf34e9864 on 2026-11-27 20:00 Europe/Oslo, exit 0, output sha256 c02ed7da5353221918d9ba5277eb188d600dbe5b30713e76004b941e2677614a
Data source: Yahoo Finance via yfinance daily ETF open and close prices, dividends, and capital gains distributions, with a local CSV cache. The strategy universe is XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, and XLY. SPY is the benchmark, and BIL daily total returns are used only for excess Sharpe.
Data fetched: 2026-11-27T19:01:00.479559+00:00
Sample period: Data requested from 2007-10-01 through 2025-12-31 inclusive, with pre-backtest data used for warm-up. The configured backtest start is 2009-01-01. The logged backtest covers 2009-01-02 through 2025-12-31 on actual sessions.   data_range (from output): {'start': '2009-01-02', 'end': '2025-12-31'}
IS/OOS split: IS is 2009-01-02 through 2017-12-29. OOS is 2018-01-02 through 2025-12-31. The code splits the continuous default and benchmark paths at the inclusive calendar cutoff of 2017-12-31 and carries positions across the boundary. Sensitivity variants run on data ending at the IS cutoff. The default OOS path uses the fixed configuration and refits PCA on rolling history available through each signal close.
Costs and slippage: The default charges 2 bps commission and 3 bps modeled slippage, totaling 5 bps on each simulated buy or sell notional. At the open, the rebalance reserves and deducts this cost on the sum of absolute position-value changes. Slippage is a fixed cash charge. The IS higher-cost variant charges 6 bps commission and 9 bps slippage. The SPY benchmark pays default friction on entry.
Signal lag: The monthly signal uses data through the last session close of the previous month. With a one-session lag, its target is executed at the next session's open, which is the first session of the new month.
Price adjustment: yfinance split-adjusted Open/Close with cash distributions reinvested at the ex-date open
Survivorship bias: The universe is fixed to the original nine 1998 Select Sector SPDR sector ETFs. The README identifies survivor and fund-family selection bias and explains that the strategy avoids backfilling today's stock constituent list.
Parameter combinations tried: 12
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.1293195153241009, ann_vol=0.12247549443347648, sharpe=1.0545121872686372, sharpe_excess=1.0460344369213315, max_drawdown=-0.21713762503237266, total_return=1.9834772178073576, turnover=3.0374579663202055, n_trades=516
    oos: cagr=0.09405689641294401, ann_vol=0.1566272180816258, sharpe=0.6524309050366298, sharpe_excess=0.497379843307092, max_drawdown=-0.30194606414667235, total_return=1.049012761049402, turnover=3.7939545157691974, n_trades=562
    full: cagr=0.11259615378759813, ann_vol=0.13956752212948456, sharpe=0.8345136485930057, sharpe_excess=0.7491309340552179, max_drawdown=-0.30194606414667213, total_return=5.113182891587409, turnover=3.3932377981588218, n_trades=1078
    benchmark  full: cagr=0.1471616273694496, ann_vol=0.17899525218648338, sharpe=0.8568882489056618, sharpe_excess=0.7903588282505031, max_drawdown=-0.33722737548691084, total_return=9.27366851276752, turnover=0.05890413072229086, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 692_dynamic_cvar_portfolio_construction_with\n\n# Original 1998 Select Sector SPDR line-up, fixed over the simulation.\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nbenchmark: SPY\nrisk_free: BIL\n\ndata_start: 2007-10-01\nbacktest_start: 2009-01-01\nis_end: 2017-12-31\nbacktest_end: 2025-12-31  # Inclusive.\n\nsignal_lag_sessions: 1\ncommission_bps: 2.0\nslippage_bps: 3.0\n\nlookback_sessions: 252\npca_components: 3\nvolatility_half_life: 21\nvolatility_init_sessions: 21\nhorizon_sessions: 21\nscenarios: 1024\nconfidence: 0.95\nmax_weight: 0.35\nseed: 692260\n\n# Six combinations ran on the earlier block model before the overlap with 689 was found.\nprior_parameter_combos: 6\n\n# One change at a time, run on the in-sample period only.\nrobustness:\n  - {name: faster_volatility, volatility_half_life: 15}\n  - {name: slower_volatility, volatility_half_life: 30}\n  - {name: lower_confidence, confidence: 0.925}\n  - {name: higher_confidence, confidence: 0.975}\n  - {name: higher_cost, commission_bps: 6.0, slippage_bps: 9.0}\n'
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
"""Research baseline for C260, dynamic CVaR with rolling factor scenarios.

Paper: Chuting Sun, Qi Wu, Xing Yan, arXiv (2023), https://arxiv.org/abs/2301.07318
Original data source: https://arxiv.org/html/2301.07318v1#S5.SS1
PyPortfolioOpt EfficientCVaR (MIT), scenario-loss formulation reference:
https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/pypfopt/efficient_frontier/efficient_cvar.py
https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/LICENSE
https://pypi.org/project/pyportfolioopt/
Riskfolio-Lib PCR tutorial (BSD-3-Clause), factor and scenario modeling reference:
https://github.com/dcajasn/Riskfolio-Lib/blob/master/examples/Tutorial%208%20-%20Portfolio%20Optimization%20using%20Risk%20Factors%20and%20Principal%20Components%20Regression%20(PCR).ipynb
https://github.com/dcajasn/Riskfolio-Lib/blob/master/LICENSE.txt
https://pypi.org/project/riskfolio-lib/
arch (NCSA), recursive volatility bootstrap reference:
https://arch.readthedocs.io/en/stable/univariate/forecasting.html#bootstrap-forecasts
https://github.com/bashtage/arch/blob/main/arch/univariate/volatility.py
https://github.com/bashtage/arch/blob/main/LICENSE.md
https://pypi.org/project/arch/
These projects informed the design. This script uses independent code.
"""

import hashlib
import json
import re
import signal
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy import sparse
from scipy.optimize import linprog
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
FIELDS = ("Open", "Close", "Dividends", "Capital Gains")
SESSIONS_PER_YEAR = 252


def price_panel(symbols, start, end):
    cache_key = json.dumps(["yf-split-ohlc-distributions-v1", symbols, str(start), str(end)])
    cache_path = CACHE / f"prices_{hashlib.sha256(cache_key.encode()).hexdigest()[:20]}.csv"
    expected_columns = pd.MultiIndex.from_product([FIELDS, symbols])
    if cache_path.exists():
        with cache_path.open() as source:
            fetched_at = datetime.fromisoformat(source.readline().strip())
            prices = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True)
    else:
        CACHE.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=CACHE) as session_dir:
            yf.set_tz_cache_location(session_dir)
            history = yf.download(
                symbols,
                start=str(start.date()),
                end=str((end + pd.Timedelta(days=1)).date()),
                auto_adjust=False,
                actions=True,
                repair=False,
                progress=False,
                threads=False,
                ignore_tz=True,
                timeout=10,
            )
        fetched_at = datetime.now(timezone.utc)
        prices = history.loc[:, [key for key in history.columns if key[0] in FIELDS]]
        # A missing capital-gains field is normal for funds with no such actions.
        if not all(("Dividends", symbol) in prices for symbol in symbols):
            raise ValueError("Yahoo did not return corporate-action fields for every symbol")
        prices = prices.reindex(columns=expected_columns, fill_value=0.0)
        check_prices(prices, symbols, start, end)
        temp_path = None
        try:
            with NamedTemporaryFile(mode="w", dir=CACHE, delete=False) as destination:
                temp_path = Path(destination.name)
                destination.write(fetched_at.isoformat() + "\n")
                prices.to_csv(destination)
            temp_path.replace(cache_path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    if not prices.columns.equals(expected_columns):
        raise ValueError("Unexpected price cache columns")
    check_prices(prices, symbols, start, end)
    return prices, fetched_at


def check_prices(prices, symbols, start, end):
    dates = prices.index
    if not isinstance(dates, pd.DatetimeIndex) or dates.hasnans or not dates.is_unique:
        raise ValueError("Missing or duplicate Yahoo session dates")
    if not dates.is_monotonic_increasing or prices.empty:
        raise ValueError("Empty or unsorted Yahoo history")
    if not (start <= dates[0] <= start + pd.Timedelta(days=7)) or not (
        end - pd.Timedelta(days=7) <= dates[-1] <= end
    ):
        raise ValueError(f"Yahoo history does not cover the configured range: {dates[0]} to {dates[-1]}")
    for symbol in symbols:
        quotes = prices.loc[:, [("Open", symbol), ("Close", symbol)]].to_numpy()
        payouts = prices.loc[:, [("Dividends", symbol), ("Capital Gains", symbol)]].to_numpy()
        if not np.isfinite(quotes).all() or (quotes <= 0).any():
            raise ValueError(f"Missing or invalid daily open/close for {symbol}")
        if not np.isfinite(payouts).all() or (payouts < 0).any():
            raise ValueError(f"Missing or invalid distributions for {symbol}")


def session_moves(prices):
    opens, closes = prices["Open"], prices["Close"]
    distributions = prices["Dividends"] + prices["Capital Gains"]
    # Yahoo quotes and distributions already use the same split-adjusted units.
    # Existing positions earn the ex-date distribution at the open in total-return units.
    overnight = (opens + distributions) / closes.shift(1)
    intraday = closes / opens
    return overnight, intraday


def forecast_scenarios(history, cfg, signal_date):
    asset_scale = history.std(axis=0)
    if (asset_scale <= 0).any() or not np.isfinite(history).all():
        raise ValueError(f"PCA window has an invalid or constant return series at {signal_date}")
    scaled = history / asset_scale
    pca = PCA(n_components=cfg["pca_components"], svd_solver="full")
    factors = pca.fit_transform(scaled)
    residuals = scaled - pca.inverse_transform(factors)

    # The variance assigned to a historical shock is based on earlier shocks.
    initial_days = cfg["volatility_init_sessions"]
    persistence = np.exp2(-1.0 / cfg["volatility_half_life"])
    common_power = np.square(factors).sum(axis=1)
    prior_variance = np.empty(len(history) - initial_days + 1)
    prior_variance[0] = common_power[:initial_days].mean()
    for row, power in enumerate(common_power[initial_days:]):
        prior_variance[row + 1] = persistence * prior_variance[row] + (1 - persistence) * power

    # Draw factor and residual innovations from the same historical day.
    # The prior common variance determines which empirical pool is available.
    observed_variance = prior_variance[:-1]
    threshold = np.median(observed_variance)
    standardized_factors = factors[initial_days:] / np.sqrt(observed_variance)[:, None]
    quiet = observed_variance <= threshold
    pools = []
    for regime in (quiet, ~quiet):
        factor_pool = standardized_factors[regime]
        residual_pool = residuals[initial_days:][regime]
        if len(factor_pool) < 2:
            raise ValueError(f"Insufficient observations in a volatility regime at {signal_date}")
        pools.append((factor_pool - factor_pool.mean(axis=0), residual_pool - residual_pool.mean(axis=0)))

    rng = np.random.default_rng([cfg["seed"], signal_date.toordinal()])
    state = np.full(cfg["scenarios"], prior_variance[-1])
    cumulative_moves = np.zeros((len(state), history.shape[1]))
    for _ in range(cfg["horizon_sessions"]):
        quiet = state <= threshold
        for paths, (factor_pool, residual_pool) in zip((quiet, ~quiet), pools, strict=True):
            draws = rng.integers(len(factor_pool), size=paths.sum())
            shocks = factor_pool[draws] * np.sqrt(state[paths, None])
            cumulative_moves[paths] += shocks @ pca.components_ + residual_pool[draws]
            state[paths] = persistence * state[paths] + (1 - persistence) * np.square(shocks).sum(axis=1)

    return np.expm1(cumulative_moves * asset_scale)


def minimum_cvar(scenarios, confidence, max_weight, signal_date):
    n_scenarios, n_assets = scenarios.shape
    # Loss is -scenario @ weights. The auxiliary variable for each loss is
    # constrained to exceed both zero and the loss minus the VaR threshold.
    loss_limits = sparse.hstack(
        [sparse.csc_matrix(-scenarios), sparse.csc_matrix(-np.ones((n_scenarios, 1))), -sparse.eye(n_scenarios)],
        format="csc",
    )
    objective = np.r_[np.zeros(n_assets), 1.0, np.full(n_scenarios, 1 / (n_scenarios * (1 - confidence)))]
    budget = np.r_[np.ones(n_assets), np.zeros(n_scenarios + 1)][None, :]
    bounds = [(0, max_weight)] * n_assets + [(None, None)] + [(0, None)] * n_scenarios
    result = linprog(
        objective, A_ub=loss_limits, b_ub=np.zeros(n_scenarios),
        A_eq=budget, b_eq=[1.0], bounds=bounds, method="highs-ipm", options={"time_limit": 4.0},
    )
    if not result.success:
        raise RuntimeError(f"CVaR solve failed at {signal_date.date()}: {result.message}")
    weights = result.x[:n_assets]
    if abs(weights.sum() - 1) > 1e-8 or weights.min() < -1e-8 or weights.max() > max_weight + 1e-8:
        raise RuntimeError(f"Invalid CVaR weights at {signal_date.date()}")
    weights = np.clip(weights, 0, max_weight)
    return weights / weights.sum()


def monthly_targets(overnight, intraday, cfg):
    dates = overnight.index
    history = (np.log(overnight) + np.log(intraday)).to_numpy()
    months = dates.to_period("M")
    first_sessions = np.flatnonzero(months[1:] != months[:-1]) + 1
    window = cfg["lookback_sessions"]
    for first in first_sessions:
        signal_row = first - 1
        fill = signal_row + cfg["signal_lag_sessions"]
        if fill >= len(dates) or dates[fill] < cfg["backtest_start"]:
            continue
        if fill <= signal_row or signal_row < window:
            raise ValueError("Signal lag must be positive and warm-up must cover the PCA window")
        signal_date = dates[signal_row]
        scenarios = forecast_scenarios(history[signal_row - window + 1:signal_row + 1], cfg, signal_date)
        yield dates[fill], minimum_cvar(scenarios, cfg["confidence"], cfg["max_weight"], signal_date)


def rebalance(positions_at_open, cash, weights, friction):
    pretrade_value = float(positions_at_open.sum()) + cash
    # Solve for the invested value after paying both sides of the rebalance.
    # Keep the lower feasible bound so the final target fits available cash.
    low, high = 0.0, pretrade_value
    for _ in range(48):
        candidate = (low + high) / 2
        invested = candidate * weights
        required = float(invested.sum() + friction * np.abs(invested - positions_at_open).sum())
        if required <= pretrade_value:
            low = candidate
        else:
            high = candidate
    positions = low * weights
    changes = np.abs(positions - positions_at_open)
    traded = float(changes.sum())
    cash = pretrade_value - float(positions.sum() + friction * traded)
    if cash < 0:
        raise RuntimeError("Rebalance would spend more than the cash available")
    return positions, cash, traded / pretrade_value, int((changes > 1e-10 * pretrade_value).sum())


def simulate_path(overnight, intraday, start, targets, friction):
    overnight, intraday = overnight.loc[start:], intraday.loc[start:]
    positions = np.zeros(overnight.shape[1])
    cash, prev_value = 1.0, 1.0
    rows = []
    for day, gap, day_move in zip(overnight.index, overnight.to_numpy(), intraday.to_numpy(), strict=True):
        positions *= gap
        turnover, trades = 0.0, 0
        if day in targets:
            positions, cash, turnover, trades = rebalance(positions, cash, targets[day], friction)
        positions *= day_move
        value = float(positions.sum()) + cash
        rows.append((value / prev_value - 1, turnover, trades))
        prev_value = value
    return pd.DataFrame(rows, index=overnight.index, columns=["return", "turnover", "n_trades"])


def metrics(daily_results, risk_free):
    returns = daily_results["return"]
    excess = returns - risk_free.loc[returns.index]
    growth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(vol * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": float(returns.mean() / vol * np.sqrt(SESSIONS_PER_YEAR)) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * np.sqrt(SESSIONS_PER_YEAR)) if excess_vol > 0 else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def summarize(daily_results, risk_free, is_end):
    in_sample = daily_results.index <= is_end
    return {
        "is": metrics(daily_results.loc[in_sample], risk_free),
        "oos": metrics(daily_results.loc[~in_sample], risk_free),
        "full": metrics(daily_results, risk_free),
    }


def trade_friction(cfg):
    commission, slippage = cfg["commission_bps"], cfg["slippage_bps"]
    if min(commission, slippage) < 0 or not commission + slippage < 10000:
        raise ValueError("Commission and slippage must be nonnegative and sum to less than 10000 bps")
    return (commission + slippage) / 10000


def run_strategy(overnight, intraday, cfg):
    friction = trade_friction(cfg)
    if not 1 <= cfg["pca_components"] <= overnight.shape[1]:
        raise ValueError("PCA rank must fit the universe")
    if not 1 <= cfg["volatility_init_sessions"] <= cfg["lookback_sessions"] - 4:
        raise ValueError("Volatility initialization must leave observations for both regimes")
    if min(cfg["volatility_half_life"], cfg["horizon_sessions"], cfg["signal_lag_sessions"]) <= 0:
        raise ValueError("Half-life, scenario horizon and execution lag must be positive")
    if not 0 < cfg["confidence"] < 1 or cfg["scenarios"] * (1 - cfg["confidence"]) < 1:
        raise ValueError("CVaR needs a valid confidence and at least one scenario in the tail")
    if not 1 / overnight.shape[1] <= cfg["max_weight"] <= 1:
        raise ValueError("Weight cap must permit a fully invested portfolio")
    targets = dict(monthly_targets(overnight, intraday, cfg))
    return simulate_path(overnight, intraday, cfg["backtest_start"], targets, friction)


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    for key in ("data_start", "backtest_start", "is_end", "backtest_end"):
        cfg[key] = pd.Timestamp(cfg[key])
    if not cfg["data_start"] < cfg["backtest_start"] < cfg["is_end"] < cfg["backtest_end"]:
        raise ValueError("Expected warm-up, in-sample and out-of-sample periods in chronological order")
    symbols = [*cfg["universe"], cfg["benchmark"], cfg["risk_free"]]
    if len(set(symbols)) != len(symbols) or any(not re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z])?", s) for s in symbols):
        raise ValueError("Use distinct US-listed equity or ETF symbols")
    prices, fetched_at = price_panel(symbols, cfg["data_start"], cfg["backtest_end"])
    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance split-adjusted Open/Close with cash distributions reinvested at the ex-date open", flush=True)
    overnight, intraday = session_moves(prices)
    universe = cfg["universe"]
    gaps, day_moves = overnight[universe], intraday[universe]
    risk_free = overnight[cfg["risk_free"]] * intraday[cfg["risk_free"]] - 1

    print("ROBUSTNESS: in-sample only", flush=True)
    variants_tried = 0
    for variant in cfg["robustness"]:
        name = variant["name"]
        alternative = {**cfg, **{key: value for key, value in variant.items() if key != "name"}}
        is_results = run_strategy(gaps.loc[:cfg["is_end"]], day_moves.loc[:cfg["is_end"]], alternative)
        print(f"ROBUSTNESS {name}: {json.dumps(metrics(is_results, risk_free), allow_nan=False)}", flush=True)
        variants_tried += 1

    daily_results = run_strategy(gaps, day_moves, cfg)
    results = summarize(daily_results, risk_free, cfg["is_end"])
    if results["is"]["n_trades"] == 0 or results["oos"]["n_trades"] == 0:
        raise RuntimeError("Default strategy must have simulated fills in both periods")
    benchmark = cfg["benchmark"]
    benchmark_results = simulate_path(
        overnight[[benchmark]], intraday[[benchmark]], cfg["backtest_start"],
        {daily_results.index[0]: np.ones(1)}, trade_friction(cfg),
    )
    benchmark_periods = summarize(benchmark_results, risk_free, cfg["is_end"])
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(results[period], allow_nan=False)}")
        print(f"BENCHMARK {period.upper()}: {json.dumps(benchmark_periods[period], allow_nan=False)}")
    report = {
        "strategy": cfg["strategy"],
        "data_range": {"start": str(daily_results.index[0].date()), "end": str(daily_results.index[-1].date())},
        **results,
        "benchmark": {"name": benchmark, **benchmark_periods},
    }
    print("STATUS: 已回测（待审）")
    run_combos = 1 + variants_tried
    print(f"RUN_PARAM_COMBOS_TRIED: {run_combos}")
    print(f"PARAM_COMBOS_TRIED: {cfg['prior_parameter_combos'] + run_combos}")
    print("RESULTS_JSON: " + json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    with threadpool_limits(limits=1):
        main()
