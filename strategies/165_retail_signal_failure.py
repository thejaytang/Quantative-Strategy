"""165_retail_signal_failure
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/165_retail_signal_failure.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 2b23c5bbf706aea933c0cba5f5805efb450ee206 on 2026-11-26 14:26 Europe/Oslo, exit 0, output sha256 11005ab85b924cbf822a747afb576d0fb917399e2e3634f03aec8208531581ef
Data source: Yahoo Finance via yfinance: daily adjusted SPY Open for both the strategy and benchmark, and daily unadjusted ^IRX Close (annualized 13-week Treasury bill bank-discount yield) for the cash proxy used in excess Sharpe. Reads series from data_cache/ on cache hits and downloads them on misses.
Data fetched: 2026-11-26 14:26 Europe/Oslo (run time)
Sample period: 2000-01-03 through 2025-12-31, inclusive, for the observed SPY sessions in the recorded run. The configured window starts on 2000-01-01.   data_range (from output): 2000-01-03 to 2025-12-31
IS/OOS split: IS 2000-01-01 through 2018-12-31 and OOS 2019-01-01 through 2025-12-31, inclusive. Reports slice one continuous portfolio path with positions carried across the split. The headline Sell-in-May rule is fixed. After an eight-year warm-up, the gated diagnostic trains before each IS deployment year and freezes its OOS choice using data through IS end. The OOS window is a historical time holdout and predates the cited paper.
Costs and slippage: 5 bps per side on traded notional, booked on the execution date at the open. After marking the prior holding to that open, the model divides equity by 1 + 0.0005 on entry to fund the fee and multiplies equity by 1 - 0.0005 on exit. The buy-and-hold benchmark pays the opening entry fee. The final open is marked to market. Slippage is not modelled.
Signal lag: There is no close-t price signal. Calendar targets depend on the next scheduled session's month, weekday, or ordinal trading day within the month. Position changes are simulated at that session's open. The position established at an open earns the following open-to-open return, recorded at the ending open.
Price adjustment: SPY uses the Yahoo Finance Open returned with auto_adjust=True, and open-to-open returns are computed from that series. ^IRX uses the Close returned with auto_adjust=False and is converted from a bank-discount quote into cash-proxy accrual. The README notes that the vendor may revise adjusted data.
Survivorship bias: The strategy and benchmark use one fixed ETF, SPY, so they do not reconstruct returns from today's constituents. The README flags product-selection bias from using that ETF.
Parameter combinations tried: 8
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.042232581273145886, ann_vol=0.1287723155968708, sharpe=0.3855839693870387, sharpe_excess=0.25909856892808225, max_drawdown=-0.3610089269254664, total_return=1.1912234690329955, turnover=2.055970131694981, n_trades=39
    oos: cagr=0.0742503414104867, ann_vol=0.14908524079390076, sharpe=0.555465170659737, sharpe_excess=0.37471358185613596, max_drawdown=-0.32048956186554733, total_return=0.6490915431996691, turnover=2.0040445686247783, n_trades=14
    full: cagr=0.05075523692537032, ann_vol=0.1345330227532017, sharpe=0.43541324492441674, sharpe_excess=0.29301364761558174, max_drawdown=-0.3610089269254664, total_return=2.6135280920429693, turnover=2.0419941428582233, n_trades=53
    benchmark SPY full: cagr=0.07992718405998378, ann_vol=0.19060600016847706, sharpe=0.4990103209001279, sharpe_excess=0.3984843322659264, max_drawdown=-0.5542290108664004, total_return=6.354168533802761, turnover=0.03851874338102397, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 165_retail_signal_failure\nticker: SPY\nbenchmark: SPY\nheadline_rule: sell_in_may\n\n# Inclusive dates. The headline rule is fixed. Only the gated diagnostic is calibrated on IS.\nin_sample: {start: '2000-01-01', end: '2018-12-31'}\nout_of_sample: {start: '2019-01-01', end: '2025-12-31'}\nmin_training_years: 8\ntrading_days_per_year: 252\ncost_bps_per_side: 5\n\n# Monday = 0. Windows apply to exposure starting at the session open.\ncalendar_rules:\n  sell_in_may: {months: [11, 12, 1, 2, 3, 4]}\n  january: {months: [1]}\n  month_start: {first_sessions: 3}\n  monday: {weekdays: [0]}\n  tuesday: {weekdays: [1]}\n  wednesday: {weekdays: [2]}\n  thursday: {weekdays: [3]}\n  friday: {weekdays: [4]}\n\ngates:\n  min_sharpe_gap: 0.20\n  min_cagr_gap: 0.01\n  alpha: 0.05\n  bootstrap_paths: 4000\n  mean_block_sessions: 20\n  random_seed: 165\n  survival_sessions: 63\n  stress_leverage: 2.0\n  maintenance_margin: 0.25\n  max_liquidation_probability: 0.10\n"
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
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.stats import beta, false_discovery_control
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
CASH_TICKER = "^IRX"


def load_yahoo_series(ticker, start, end, field, *, adjusted):
    symbol = ticker.lstrip("^").lower()
    label = f"{'adjusted_' if adjusted else ''}{field.lower()}"
    path = CACHE / f"{symbol}_{start:%Y%m%d}_{end:%Y%m%d}_{label}.csv"
    if path.exists():
        series = pd.read_csv(path, index_col="Date", parse_dates=True)[field]
    else:
        end_exclusive = end + pd.Timedelta(days=1)
        data = yf.download(
            ticker, start=start.date(), end=end_exclusive.date(), auto_adjust=adjusted,
            progress=False, threads=False, timeout=20, multi_level_index=False,
        )
        if data.empty or field not in data:
            raise RuntimeError(f"Yahoo Finance returned no daily {field} for {ticker}")
        series = data[field].rename_axis("Date")
        series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()

    if ticker == CASH_TICKER:
        series = series.dropna()

    boundary_slack = pd.Timedelta(days=7)
    if (
        series.empty or not series.index.is_monotonic_increasing or not series.index.is_unique
        or series.index[0] < start or series.index[-1] > end
        or series.index[0] > start + boundary_slack or series.index[-1] < end - boundary_slack
        or not np.isfinite(series).all() or (adjusted and (series <= 0).any())
    ):
        raise RuntimeError(f"Incomplete or invalid {ticker} {field} for {start.date()} to {end.date()}")

    if not path.exists():
        with NamedTemporaryFile(mode="w", dir=CACHE, encoding="utf-8", delete=False) as file:
            temporary_path = Path(file.name)
            try:
                series.to_csv(file)
                file.close()
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)
    return series


def cash_proxy_returns(sessions, discount_yields):
    # Use only closes strictly before the opening date of each open-to-open interval.
    opening_dates = sessions[:-1]
    available = discount_yields.reindex(
        opening_dates - pd.Timedelta(days=1), method="ffill", tolerance=pd.Timedelta(days=7),
    )
    if available.isna().any():
        missing_date = available.index[available.isna()][0] + pd.Timedelta(days=1)
        raise RuntimeError(f"No recent {CASH_TICKER} close known before {missing_date.date()}")
    # Convert the annual bank-discount quote to a return on a 91-day bill's purchase price.
    discount = available.to_numpy() / 100 * (91 / 360)
    elapsed_days = (sessions[1:] - opening_dates).days.to_numpy()
    accrual = np.expm1(-np.log1p(-discount) * elapsed_days / 91)
    if not np.isfinite(accrual).all():
        raise RuntimeError(f"Invalid {CASH_TICKER} discount yield")
    return pd.Series(np.r_[0.0, accrual], index=sessions, name="cash_proxy")


def calendar_targets(sessions, rules):
    # These are orders for the next scheduled open, conditional only on its calendar date.
    session_number = pd.Series(1, index=sessions).groupby(sessions.to_period("M")).cumsum()
    targets = pd.DataFrame(index=sessions)
    for name, window in rules.items():
        if window.keys() == {"months"}:
            targets[name] = sessions.month.isin(window["months"])
        elif window.keys() == {"first_sessions"}:
            targets[name] = session_number <= window["first_sessions"]
        elif window.keys() == {"weekdays"}:
            targets[name] = sessions.day_of_week.isin(window["weekdays"])
        else:
            raise ValueError(f"Invalid calendar window {name}: {window}")
    return targets


def backtest(asset_returns, targets, cost):
    # Return dated t accrues on yesterday's position. Trade at t's open afterward.
    held = targets.shift(1, fill_value=False)
    entries = targets & ~held
    exits = held & ~targets
    gross = 1 + held.mul(asset_returns, axis=0)
    net = gross * (1 - cost * exits) / (1 + cost * entries) - 1
    # Buy notional excludes the entry fee. The full pre-trade equity is sold on exit.
    turnover = entries / (1 + cost) + exits
    return net, entries | exits, turnover, held


def stationary_indices(rng, observations, paths, horizon, mean_block):
    steps = np.arange(horizon, dtype=np.int32)
    restart = rng.random((paths, horizon)) < 1 / mean_block
    restart[:, 0] = True
    block_start_steps = np.maximum.accumulate(np.where(restart, steps, 0), axis=1)
    starts = rng.integers(observations, size=restart.shape, dtype=np.int32)
    block_start_indices = np.take_along_axis(starts, block_start_steps, axis=1)
    return (block_start_indices + steps - block_start_steps) % observations


def comparison_gaps(sums, observations, annual_sessions):
    returns, squared_returns, log_growth = np.split(sums, 3, axis=-1)
    sample_variance = (squared_returns - returns**2 / observations) / (observations - 1)
    volatility = np.sqrt(np.maximum(sample_variance, 0))
    sharpe = np.divide(
        returns * np.sqrt(annual_sessions) / observations, volatility,
        out=np.full_like(volatility, np.nan), where=volatility > 0,
    )
    cagr = np.expm1(log_growth * annual_sessions / observations)
    rule_sharpe, matched_sharpe = np.split(sharpe, 2, axis=-1)
    rule_cagr, matched_cagr = np.split(cagr, 2, axis=-1)
    return rule_sharpe - matched_sharpe, rule_cagr - matched_cagr


def comparison_bootstrap(net, matched, annual_sessions, gates, rng):
    both = np.column_stack((net, matched))
    components = np.concatenate((both, both**2, np.log1p(both)), axis=1)
    n = len(both)
    observed = comparison_gaps(components.sum(axis=0), n, annual_sessions)

    sharpe_draws, cagr_draws = [], []
    for offset in range(0, gates["bootstrap_paths"], 128):
        batch = min(128, gates["bootstrap_paths"] - offset)
        indices = stationary_indices(rng, n, batch, n, gates["mean_block_sessions"])
        # Paired samples use the same date counts for every rule and benchmark.
        counts = np.stack([np.bincount(path, minlength=n) for path in indices])
        sums = np.einsum("bi,ij->bj", counts, components)
        sharpe, cagr = comparison_gaps(sums, n, annual_sessions)
        sharpe_draws.append(sharpe)
        cagr_draws.append(cagr)
    return observed, np.concatenate(sharpe_draws), np.concatenate(cagr_draws)


def liquidation_upper_bound(returns, in_market, cost, gates):
    paths = returns.shape[0]
    leverage = gates["stress_leverage"]
    maintenance = gates["maintenance_margin"]
    equity = np.ones(paths)
    position = np.zeros(paths)
    debt = np.zeros(paths)
    liquidated = np.zeros(paths, dtype=bool)
    previously_active = np.zeros(paths, dtype=bool)

    for daily_return, exposure in zip(returns.T, in_market.T):
        active = exposure & ~liquidated
        entry = active & ~previously_active
        exit_trade = previously_active & ~active & ~liquidated

        equity[exit_trade] = position[exit_trade] * (1 - cost) - debt[exit_trade]
        position[entry] = equity[entry] * leverage / (1 + cost * leverage)
        debt[entry] = position[entry] * (1 - 1 / leverage)
        position[active] *= 1 + daily_return[active]
        equity[active] = position[active] - debt[active]

        liquidated |= (active & (equity <= maintenance * position)) | (equity <= 0)
        previously_active = active

    failures = int(liquidated.sum())
    if failures == paths:
        return 1.0
    return float(beta.ppf(1 - gates["alpha"], failures + 1, paths - failures))


def calibrate(net, held, asset_returns, cost, annual_sessions, gates):
    rng = np.random.default_rng(gates["random_seed"])
    matched = asset_returns.to_numpy()[:, None] * held.mean().to_numpy()
    (sharpe_gap, cagr_gap), sharpe_draws, cagr_draws = comparison_bootstrap(
        net.to_numpy(), matched, annual_sessions, gates, rng,
    )
    # Center at the point estimate to test the boundary of the material-edge null.
    valid_sharpe = np.isfinite(sharpe_gap) & np.isfinite(sharpe_draws).all(axis=0)
    null_exceedances = (sharpe_draws - sharpe_gap) >= (sharpe_gap - gates["min_sharpe_gap"])
    p_values = (1 + null_exceedances.sum(axis=0)) / (1 + len(sharpe_draws))
    adjusted_p = false_discovery_control(np.where(valid_sharpe, p_values, 1.0), method="by")
    cagr_lower = np.quantile(cagr_draws, gates["alpha"] / net.shape[1], axis=0)

    paths = stationary_indices(
        rng, len(net), gates["bootstrap_paths"], gates["survival_sessions"], gates["mean_block_sessions"],
    )
    sampled_returns = asset_returns.to_numpy()[paths]
    ruin_upper = [
        liquidation_upper_bound(sampled_returns, held[name].to_numpy()[paths], cost, gates)
        for name in net.columns
    ]
    results = pd.DataFrame({
        "sharpe_gap": sharpe_gap,
        "by_p": adjusted_p,
        "cagr_gap": cagr_gap,
        "cagr_lower": cagr_lower,
        "liquidation_upper": ruin_upper,
    }, index=net.columns)
    results["passes"] = (
        (results["sharpe_gap"] > gates["min_sharpe_gap"])
        & (results["by_p"] < gates["alpha"])
        & (results["cagr_lower"] > gates["min_cagr_gap"])
        & (results["liquidation_upper"] < gates["max_liquidation_probability"])
    )
    passing = results.loc[results["passes"], "cagr_gap"]
    return (None if passing.empty else passing.idxmax()), results


def gated_targets(targets, rule_net, rule_held, asset_returns, is_range, oos_range, cfg):
    sessions = targets.index
    is_start, is_end = is_range
    oos_start, oos_end = oos_range
    cost = cfg["cost_bps_per_side"] / 10_000
    annual_sessions = cfg["trading_days_per_year"]
    chosen_targets = pd.Series(False, index=sessions, name="gated")
    first_eligible = is_start + pd.DateOffset(years=cfg["min_training_years"])

    in_sample_sessions = sessions[(sessions >= is_start) & (sessions <= is_end)]
    for year in in_sample_sessions.year.unique():
        year_sessions = in_sample_sessions[in_sample_sessions.year == year]
        deployment_start = year_sessions[0]
        if deployment_start < first_eligible:
            continue
        fit = sessions < deployment_start
        selected, _ = calibrate(
            rule_net.loc[fit], rule_held.loc[fit], asset_returns.loc[fit],
            cost, annual_sessions, cfg["gates"],
        )
        if selected is not None:
            chosen_targets.loc[year_sessions] = targets.loc[year_sessions, selected]
        print(f"Gated IS deployment {year}: {selected or 'cash'}", flush=True)

    fit = sessions <= is_end
    selected, final_gates = calibrate(
        rule_net.loc[fit], rule_held.loc[fit], asset_returns.loc[fit],
        cost, annual_sessions, cfg["gates"],
    )
    if selected is not None:
        chosen_targets.loc[oos_start:oos_end] = targets.loc[oos_start:oos_end, selected]
    print(f"\nGated OOS deployment (frozen on IS): {selected or 'cash'}")
    print(f"Final IS calibration against cost-free, exposure-matched {cfg['ticker']}:")
    print(final_gates.to_string(float_format=lambda value: f"{value:.4f}"))
    return chosen_targets


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def annualized_sharpe(returns, annual_sessions):
    volatility = returns.std(ddof=1)
    return finite_or_none(np.sqrt(annual_sessions) * returns.mean() / volatility) if volatility > 0 else None


def metrics(returns, trades, turnover, cash_returns, annual_sessions):
    if returns.empty:
        raise ValueError("A configured report window contains no price sessions")
    growth = np.concatenate(([1.0], (1 + returns.to_numpy()).cumprod()))
    return {
        "cagr": finite_or_none(growth[-1] ** (annual_sessions / len(returns)) - 1),
        "ann_vol": finite_or_none(returns.std(ddof=1) * np.sqrt(annual_sessions)),
        "sharpe": annualized_sharpe(returns, annual_sessions),
        "sharpe_excess": annualized_sharpe(returns - cash_returns, annual_sessions),
        "max_drawdown": finite_or_none((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": finite_or_none(growth[-1] - 1),
        "turnover": finite_or_none(turnover.mean() * annual_sessions),
        "n_trades": int(trades.sum()),
    }


def print_metrics(portfolios):
    print("\nPortfolio performance")
    print(f"{'Portfolio':<26} {'Period':<5} {'CAGR':>10} {'Ann. vol':>10} {'Sharpe':>9} "
          f"{'Ex. Sharpe':>10} {'Max DD':>10} {'Total ret.':>11} {'Ann. TO':>10} {'Trades':>7}")
    for label, periods in portfolios.items():
        for period, row in periods.items():
            percent = [f"{row[key]:.2%}" if row[key] is not None else "n/a"
                       for key in ("cagr", "ann_vol", "max_drawdown", "total_return", "turnover")]
            sharpe, excess = (f"{row[key]:.3f}" if row[key] is not None else "n/a"
                              for key in ("sharpe", "sharpe_excess"))
            print(f"{label:<26} {period.upper():<5} {percent[0]:>10} {percent[1]:>10} {sharpe:>9} "
                  f"{excess:>10} {percent[2]:>10} {percent[3]:>11} {percent[4]:>10} {row['n_trades']:>7}")


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    is_start, is_end = (pd.Timestamp(cfg["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(cfg["out_of_sample"][key]) for key in ("start", "end"))
    if not (is_start < is_end and is_end + pd.Timedelta(days=1) == oos_start < oos_end):
        raise ValueError("Use ordered, adjacent IS and OOS windows")
    if is_start + pd.DateOffset(years=cfg["min_training_years"]) > oos_start:
        raise ValueError("The IS window is too short for the configured minimum training years")
    gates = cfg["gates"]
    if not 0 <= gates["maintenance_margin"] < 1 / gates["stress_leverage"] <= 1:
        raise ValueError("The survival scenario must start above maintenance margin at leverage >= 1")
    headline = cfg["headline_rule"]
    if headline not in cfg["calendar_rules"]:
        raise ValueError(f"Unknown headline rule {headline}")

    ticker, benchmark_ticker = cfg["ticker"], cfg["benchmark"]
    CACHE.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="yfinance-", dir=CACHE) as provider_cache:
        yf.set_tz_cache_location(provider_cache)
        price_history = load_yahoo_series(ticker, is_start.replace(day=1), oos_end, "Open", adjusted=True)
        prices = price_history.loc[is_start:]
        benchmark_prices = prices
        if benchmark_ticker != ticker:
            benchmark_prices = load_yahoo_series(
                benchmark_ticker, is_start, oos_end, "Open", adjusted=True,
            ).reindex(prices.index)
        if benchmark_prices.isna().any():
            raise RuntimeError("Benchmark lacks one or more strategy trading sessions")
        discount_yields = load_yahoo_series(
            CASH_TICKER, is_start - pd.Timedelta(days=14), oos_end, "Close", adjusted=False,
        )

    asset_returns = prices.pct_change().fillna(0.0)
    benchmark_returns = benchmark_prices.pct_change().fillna(0.0)
    cash_returns = cash_proxy_returns(prices.index, discount_yields)
    targets = calendar_targets(price_history.index, cfg["calendar_rules"]).loc[prices.index]
    cost = cfg["cost_bps_per_side"] / 10_000
    net, trades, turnover, held = backtest(asset_returns, targets, cost)
    chosen_targets = gated_targets(
        targets, net, held, asset_returns, (is_start, is_end), (oos_start, oos_end), cfg,
    )
    buy_and_hold = pd.Series(True, index=prices.index, name="buy_and_hold")

    windows = {"is": slice(is_start, is_end), "oos": slice(oos_start, oos_end), "full": slice(None)}
    annual_sessions = cfg["trading_days_per_year"]

    def report(returns, trade_flags, traded_fraction):
        return {
            name: metrics(
                returns.loc[span], trade_flags.loc[span], traded_fraction.loc[span],
                cash_returns.loc[span], annual_sessions,
            )
            for name, span in windows.items()
        }

    sleeves = {name: report(net[name], trades[name], turnover[name]) for name in targets.columns}
    benchmark = report(*backtest(benchmark_returns, buy_and_hold, cost)[:3])
    gated = report(*backtest(asset_returns, chosen_targets, cost)[:3])
    results = {
        "strategy": cfg["strategy"],
        "data_range": f"{prices.index[0].date()} to {prices.index[-1].date()}",
        **sleeves[headline],
        "benchmark": {"ticker": benchmark_ticker, **benchmark},
    }
    print(f"\nHeadline fixed research rule: {headline}")
    print(f"Excess Sharpe uses lagged {CASH_TICKER}; idle portfolio cash earns zero.")
    print_metrics({
        f"{headline} (headline)": sleeves[headline],
        **{name: periods for name, periods in sleeves.items() if name != headline},
        "Original gated deployment": gated,
        f"{benchmark_ticker} buy & hold": benchmark,
    })
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
