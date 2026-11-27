"""199_pca_stat_arb
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/199_pca_stat_arb.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 92e75bea82090bd8b6723b610a02dabec59e659b on 2026-11-27 07:29 Europe/Oslo, exit 0, output sha256 24080a4cea1bea183ef06fd6d81b3a9fedb1cf536d56e4ed42c526b63367b8bb
Data source: Free Yahoo Finance daily open and close prices via yfinance for the fixed 24-stock US equity universe, SPY as the benchmark and single-factor comparison input, and BIL as the cash return proxy. The loader uses a local CSV cache when present.
Data fetched: 2026-11-27T06:29:33+00:00
Sample period: Logged price history covers 2013-01-02 through 2025-12-31, with 3,270 complete sessions. The requested data start is 2013-01-01. The evaluated period is 2015-01-02 through 2025-12-31, with earlier data used for formation and signal history.   data_range (from output): 2015-01-02 to 2025-12-31
IS/OOS split: By trading session, inclusive: IS 2015-01-02 through 2019-12-31 and OOS 2020-01-02 through 2025-12-31. The configured calendar starts are 2015-01-01 and 2020-01-01. Each portfolio runs continuously and daily results are sliced for reporting, so positions carry across the boundary. Monthly fits use past returns. The OOS period had appeared in earlier backtests and lacks independent holdout status.
Costs and slippage: The base simulation charges 1 bp commission and models 4 bp slippage as a flat cash cost, for 5 bp per dollar of absolute one-way executed notional on buys and sells at each opening rebalance. It applies the charge to drifted positions and solves for after-fee equity while retaining the bar open as the execution price. Short borrow is 100 bp annualized on the beginning short market value of each interval, prorated over 365 calendar days with 6.5 hours allocated to the regular session and the remaining calendar time to the overnight interval. Sensitivities use 0 or 15 bp total transaction cost at base borrow, or 5 bp total transaction cost at 0 bp borrow.
Signal lag: On an eligible Friday, the signal uses closing prices through session t and is simulated at the next available session's open. The old portfolio carries the overnight return through that open and the new weights take effect before the open-to-close return. Fridays without bars and sessions with fewer than five residual observations in the current month yield no signal.
Price adjustment: Yahoo Finance via yfinance (auto_adjust=True), split- and dividend-adjusted open and close
Survivorship bias: Fixed 24-stock US large-cap basket. The README identifies survivorship and retrospective selection bias and says delisted names are excluded. The code reuses the configured tickers throughout and raises an error for any missing or invalid open or close on a panel date.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.025845061529083502, ann_vol=0.05383262426839303, sharpe=0.5009662947172744, sharpe_excess=0.3375168069773928, max_drawdown=-0.08080266355269161, total_return=0.13584981025379594, turnover=64.24992674266458, n_trades=5112
    oos: cagr=0.004885730704877078, ann_vol=0.07148766412996045, sharpe=0.1038251258756123, sharpe_excess=-0.26165618509268523, max_drawdown=-0.16726726825336247, total_return=0.029595125663159916, turnover=65.14926837308519, n_trades=6216
    full: cagr=0.014364586479493235, ann_vol=0.06405599760418902, sharpe=0.2546514486659549, sharpe_excess=-0.03049914764219899, max_drawdown=-0.16726726825336335, total_return=0.16946542812273502, turnover=64.74024025628509, n_trades=11328
    benchmark: cagr=0.13400518464240663, ann_vol=0.17790164185237436, sharpe=0.7962290798453442, sharpe_excess=0.6934118236102376, max_drawdown=-0.3371727036119293, total_return=2.976171274755677, turnover=0.09104086494610444, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 199_pca_stat_arb\nuniverse:\n  - AAPL\n  - MSFT\n  - AMZN\n  - GOOGL\n  - META\n  - NVDA\n  - JPM\n  - BAC\n  - GS\n  - MS\n  - XOM\n  - CVX\n  - COP\n  - SLB\n  - WMT\n  - COST\n  - PG\n  - KO\n  - PEP\n  - JNJ\n  - PFE\n  - MRK\n  - HD\n  - LOW\nbenchmark_ticker: SPY\ncash_ticker: BIL\ndata_start: 2013-01-01\nin_sample:\n  start: 2015-01-01\n  end: 2019-12-31\nout_of_sample:\n  start: 2020-01-01\n  end: 2025-12-31\n\nformation_sessions: 252\nn_components: 3\nresidual_sessions: 5\nnames_per_side: 4\nrebalance_weekday: 4  # Monday = 0\ngross_exposure: 1.0\nannualization_sessions: 252\n\ncommission_bps: 1.0  # Per dollar of executed notional, one way\nslippage_bps: 4.0  # Per dollar of executed notional, one way\ntransaction_cost_sensitivity_bps: [0.0, 15.0]\nshort_borrow_bps_annual: 100.0\nshort_borrow_sensitivity_bps_annual: 0.0\n'
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
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRIMARY = "PCA projected"
UNPROJECTED = "PCA unprojected"
SPY_RESIDUAL = "SPY residual"
PRICE_ADJUSTMENT = "Yahoo Finance via yfinance (auto_adjust=True), split- and dividend-adjusted open and close"


def load_prices(tickers, start, end):
    """Fetch dividend- and split-adjusted opens and closes for a fixed panel."""
    CACHE.mkdir(exist_ok=True)
    key = hashlib.sha256(" ".join(tickers).encode()).hexdigest()[:12]
    path = CACHE / f"yahoo_adjusted_{start.date()}_{end.date()}_{key}.csv"
    columns = pd.MultiIndex.from_product([["Open", "Close"], tickers])
    if path.is_file():
        prices = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
        fetched_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")
        return prices.reindex(columns=columns).sort_index(), fetched_at

    yf.set_tz_cache_location(str(CACHE / "yfinance"))
    prices = pd.DataFrame(columns=columns, dtype=float)
    missing = tickers
    for _ in range(3):
        downloaded = yf.download(
            missing,
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True,
            ignore_tz=True,
            multi_level_index=True,
            threads=8,
            timeout=12,
            progress=False,
        )
        # Preserve successful series and retry symbols for which Yahoo gave no bars.
        prices = prices.combine_first(downloaded.reindex(columns=columns))
        missing = [ticker for ticker in tickers if not prices.xs(ticker, axis=1, level=1).notna().any().all()]
        if not missing:
            break
    if missing:
        raise RuntimeError(f"Yahoo Finance returned no adjusted open/close history for {', '.join(missing)}")

    prices = prices.reindex(columns=columns).sort_index()
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=CACHE, mode="w", suffix=".csv.tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            prices.to_csv(temporary)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    fetched_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")
    return prices, fetched_at


def ranked_weights(scores, names_per_side, gross):
    order = np.argsort(scores, kind="stable")
    weights = np.zeros(len(scores))
    weights[order[:names_per_side]] = gross / (2 * names_per_side)
    weights[order[-names_per_side:]] = -gross / (2 * names_per_side)
    return weights


def neutral_basis(loadings, sigma):
    # Standardized-return exposures are w * sigma. The inverse-sigma
    # constraint makes the projected exposures sum to zero in dollar units.
    dollar_constraint = 1.0 / sigma
    dollar_constraint /= np.linalg.norm(dollar_constraint)
    constraints = np.column_stack((loadings, dollar_constraint))
    basis, singular_values, _ = np.linalg.svd(constraints, full_matrices=False)
    tolerance = singular_values[0] * max(constraints.shape) * np.finfo(float).eps
    return basis[:, singular_values > tolerance]


def build_targets(closes, config):
    stocks = config["universe"]
    benchmark = config["benchmark_ticker"]
    fit_sessions = config["formation_sessions"]
    residual_sessions = config["residual_sessions"]
    components = config["n_components"]
    names_per_side = config["names_per_side"]
    gross = config["gross_exposure"]
    returns = closes.pct_change(fill_method=None)
    targets = {
        name: pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
        for name in (PRIMARY, UNPROJECTED, SPY_RESIDUAL)
    }

    for _, month in returns.groupby(returns.index.to_period("M")):
        start = returns.index.get_loc(month.index[0])
        if start <= fit_sessions:
            continue
        formation = returns.iloc[start - fit_sessions:start]
        stock_formation = formation[stocks].to_numpy()
        mean = stock_formation.mean(axis=0)
        sigma = stock_formation.std(axis=0, ddof=1)
        market = formation[benchmark].to_numpy()
        market_mean = market.mean()
        market = market - market_mean
        market_squares = market @ market
        if np.any(sigma <= 0) or market_squares <= 0:
            continue

        standardized = (stock_formation - mean) / sigma
        _, vectors = np.linalg.eigh(standardized.T @ standardized)
        loadings = vectors[:, -components:]
        basis = neutral_basis(loadings, sigma)
        market_beta = market @ standardized / market_squares

        stock_month = (month[stocks].to_numpy() - mean) / sigma
        pca_residuals = stock_month - (stock_month @ loadings) @ loadings.T
        market_residuals = stock_month - np.outer(month[benchmark] - market_mean, market_beta)

        for offset, close_date in enumerate(month.index):
            first = offset + 1 - residual_sessions
            fill = start + offset + 1
            if first < 0 or close_date.weekday() != config["rebalance_weekday"] or fill >= len(closes):
                continue
            ranks = pca_residuals[first:offset + 1].sum(axis=0)
            raw = ranked_weights(ranks, names_per_side, gross)
            exposures = raw * sigma
            projected = (exposures - basis @ (basis.T @ exposures)) / sigma
            projected_gross = np.abs(projected).sum()
            if projected_gross <= gross * 1e-10:
                continue

            spy_ranks = market_residuals[first:offset + 1].sum(axis=0)
            fill_date = closes.index[fill]
            new_weights = {
                PRIMARY: gross * projected / projected_gross,
                UNPROJECTED: raw,
                SPY_RESIDUAL: ranked_weights(spy_ranks, names_per_side, gross),
            }
            for name, weights in new_weights.items():
                targets[name].loc[fill_date] = 0.0
                targets[name].loc[fill_date, stocks] = weights
    return targets


def rebalance(open_weights, target, rate):
    """Solve for after-fee equity so targets are fractions of investable NAV."""
    nav = 1.0
    for _ in range(20):
        trades = np.abs(nav * target - open_weights)
        after_fees = 1.0 - rate * trades.sum()
        if abs(after_fees - nav) < 1e-12:
            return after_fees, trades
        nav = after_fees
    raise RuntimeError("Transaction cost calculation failed to converge")


def marked_nav(weights, price_ratio, cash_return, borrow_rate):
    # Current short market value is segregated at zero rebate. Only own
    # cash left after the longs earns the cash proxy (or pays it if negative).
    free_cash = 1.0 - np.maximum(weights, 0.0).sum()
    short_value = -np.minimum(weights, 0.0).sum()
    return 1.0 + weights @ (price_ratio - 1.0) + free_cash * cash_return - short_value * borrow_rate


def simulate(opens, closes, targets, cash_marks, cost_bps, borrow_bps):
    trading_rate = cost_bps / 10_000
    daily_borrow_rate = borrow_bps / (10_000 * 365)
    session_days = 6.5 / 24
    opens_array = opens.to_numpy()
    closes_array = closes.to_numpy()
    prev_close = opens_array[0]
    weights = np.zeros(len(opens.columns))
    daily = np.zeros((len(opens), 3))

    for i, (opening, closing, target, cash) in enumerate(
        zip(opens_array, closes_array, targets.to_numpy(), cash_marks.to_numpy())
    ):
        cash_overnight, cash_intraday, calendar_days = cash
        # Overnight P&L belongs to the old book up to this morning's fill.
        overnight = opening / prev_close
        open_nav = marked_nav(weights, overnight, cash_overnight, daily_borrow_rate * (calendar_days - session_days))
        if open_nav <= 0:
            raise RuntimeError(f"Nonpositive opening equity on {opens.index[i].date()}")
        weights = weights * overnight / open_nav
        after_fees = 1.0
        if not np.isnan(target[0]):
            after_fees, trades = rebalance(weights, target, trading_rate)
            daily[i, 1] = trades.sum() * open_nav
            daily[i, 2] = np.count_nonzero(trades > 1e-12)
            weights = target

        intraday = closing / opening
        close_nav = marked_nav(weights, intraday, cash_intraday, daily_borrow_rate * session_days)
        if close_nav <= 0:
            raise RuntimeError(f"Nonpositive closing equity on {opens.index[i].date()}")
        daily[i, 0] = open_nav * after_fees * close_nav - 1.0
        weights = weights * intraday / close_nav
        prev_close = closing

    daily_results = pd.DataFrame(daily, index=opens.index, columns=["return", "traded", "n_trades"])
    daily_results["cash_return"] = (1.0 + cash_marks["overnight"]) * (1.0 + cash_marks["intraday"]) - 1.0
    return daily_results


def metrics(daily, annualization):
    returns = daily["return"].to_numpy()
    excess = returns - daily["cash_return"].to_numpy()
    periods = len(returns)
    equity = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    volatility = returns.std(ddof=1) if periods > 1 else np.nan
    excess_volatility = excess.std(ddof=1) if periods > 1 else np.nan
    values = {
        "cagr": equity[-1] ** (annualization / periods) - 1.0,
        "ann_vol": volatility * np.sqrt(annualization),
        "sharpe": returns.mean() * np.sqrt(annualization) / volatility if volatility > 0 else np.nan,
        "sharpe_excess": excess.mean() * np.sqrt(annualization) / excess_volatility if excess_volatility > 0 else np.nan,
        "max_drawdown": (equity / np.maximum.accumulate(equity) - 1.0).min(),
        "total_return": equity[-1] - 1.0,
        "turnover": daily["traded"].mean() * annualization,
    }
    return {key: float(value) if np.isfinite(value) else None for key, value in values.items()} | {
        "n_trades": int(daily["n_trades"].sum()),
    }


def print_table(rows):
    def format_value(value, pattern):
        return "n/a" if value is None else format(value, pattern)

    print(f"{'Portfolio':<32} {'Split':<5} {'CAGR':>9} {'Ann vol':>9} {'Sharpe':>8} {'Excess':>8} "
          f"{'Max DD':>9} {'Total':>9} {'Turn/yr':>9} {'Trades':>7}")
    for label, splits in rows:
        for period, values in splits.items():
            cagr, vol, sharpe, excess, drawdown, total, turnover = (
                format_value(values[key], pattern) for key, pattern in (
                    ("cagr", ".2%"), ("ann_vol", ".2%"), ("sharpe", ".2f"), ("sharpe_excess", ".2f"),
                    ("max_drawdown", ".2%"), ("total_return", ".2%"), ("turnover", ".2f"),
                )
            )
            print(f"{label:<32} {period.upper():<5} {cagr:>9} {vol:>9} {sharpe:>8} {excess:>8} "
                  f"{drawdown:>9} {total:>9} {turnover:>9} {values['n_trades']:>7}")


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    data_start = pd.Timestamp(config["data_start"])
    if not data_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Expected ordered warm-up, in-sample and out-of-sample dates")
    stocks = config["universe"]
    benchmark = config["benchmark_ticker"]
    cash = config["cash_ticker"]
    trade_tickers = [*stocks, benchmark]
    tickers = [*trade_tickers, cash]
    if (len(tickers) != len(set(tickers)) or not 1 <= config["n_components"] < len(stocks) - 1
            or not 1 <= config["names_per_side"] <= len(stocks) // 2):
        raise ValueError("Check that tickers are unique and PCA and rank sizes fit the universe")

    prices, fetched_at = load_prices(tickers, data_start, oos_end)
    prices = prices.loc[data_start:oos_end]
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("Expected a nonempty price panel with one row per session")
    prices = prices.where(np.isfinite(prices) & prices.gt(0))
    incomplete = prices.index[prices.isna().any(axis=1)]
    if len(incomplete):
        raise ValueError(f"Incomplete fixed-universe price panel on {incomplete[0].date()} ({len(incomplete)} sessions)")
    opens = prices["Open"][trade_tickers]
    closes = prices["Close"][trade_tickers]
    cash_closes = prices["Close"][cash]
    cash_opens = prices["Open"][cash]
    cash_marks = pd.DataFrame({
        "overnight": cash_opens / cash_closes.shift(1) - 1.0,
        "intraday": cash_closes / cash_opens - 1.0,
        "calendar_days": closes.index.to_series().diff().dt.days,
    })
    print(f"Yahoo Finance adjusted bars: {len(prices)} complete sessions, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}.")

    masks = {
        "is": opens.loc[is_start:is_end].index,
        "oos": opens.loc[oos_start:oos_end].index,
        "full": opens.loc[is_start:oos_end].index,
    }
    if not len(masks["is"]) or not len(masks["oos"]) or len(masks["full"]) != len(masks["is"]) + len(masks["oos"]):
        raise ValueError("IS and OOS must both contain sessions and cover the full period without a session gap")
    if len(opens.loc[opens.index < is_start]) <= config["formation_sessions"]:
        raise ValueError("Extend data_start to provide a full formation window before IS")
    targets = {label: target.loc[masks["full"]].copy() for label, target in build_targets(closes, config).items()}
    opens, closes = opens.loc[masks["full"]], closes.loc[masks["full"]]
    first_date = opens.index[0]
    cash_marks = cash_marks.loc[masks["full"]]
    passive = pd.DataFrame(np.nan, index=opens.index, columns=trade_tickers)
    passive.loc[first_date] = 0.0
    passive.loc[first_date, benchmark] = 1.0

    annualization = config["annualization_sessions"]
    commission = float(config["commission_bps"])
    slippage = float(config["slippage_bps"])
    base_cost = commission + slippage
    base_borrow = float(config["short_borrow_bps_annual"])

    def evaluate(target, cost=base_cost, borrow=base_borrow):
        daily = simulate(opens, closes, target, cash_marks, cost, borrow)
        return {period: metrics(daily.loc[dates], annualization) for period, dates in masks.items()}

    main_results = evaluate(targets[PRIMARY])
    benchmark_results = evaluate(passive, borrow=0.0)
    comparisons = {"pca_unprojected": evaluate(targets[UNPROJECTED]), "spy_residual": evaluate(targets[SPY_RESIDUAL])}
    table = [(PRIMARY, main_results), (f"{benchmark} buy and hold", benchmark_results),
             (UNPROJECTED, comparisons["pca_unprojected"]), (SPY_RESIDUAL, comparisons["spy_residual"])]
    cost_sensitivity = []
    scenarios = [(float(cost), base_borrow) for cost in config["transaction_cost_sensitivity_bps"]]
    scenarios.append((base_cost, float(config["short_borrow_sensitivity_bps_annual"])))
    for cost, borrow in scenarios:
        splits = evaluate(targets[PRIMARY], cost=cost, borrow=borrow)
        cost_sensitivity.append({"transaction_cost_bps": cost, "short_borrow_bps_annual": borrow, **splits})
        table.append((f"PCA {cost:g} bp cost, {borrow:g} bp borrow", splits))

    date_ranges = {
        period: f"{dates[0].date()} to {dates[-1].date()}" for period, dates in masks.items()
    }
    print(f"IS {date_ranges['is']} | OOS {date_ranges['oos']} | Full {date_ranges['full']}")
    print(f"Base costs: {commission:g} bp commission + {slippage:g} bp slippage per dollar traded, "
          f"{base_borrow:g} bp annual short borrow.")
    print(f"Excess Sharpe cash reference: {cash} adjusted close-to-close total return.")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print_table(table)
    result = {
        "strategy": config["strategy"],
        "data_range": date_ranges["full"],
        "date_ranges": date_ranges,
        **main_results,
        "benchmark": {"ticker": benchmark, **benchmark_results["full"], **benchmark_results},
        "comparisons": comparisons,
        "cost_sensitivity": cost_sensitivity,
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
