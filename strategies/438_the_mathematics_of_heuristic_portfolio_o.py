"""438_the_mathematics_of_heuristic_portfolio_o
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/438_the_mathematics_of_heuristic_portfolio_o.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 6df04a4d89c55b7a1b162dbdcff0dafc1e2bd682 on 2026-11-27 08:09 Europe/Oslo, exit 0, output sha256 d79ee5e99dda7ed9f766fc0b88fa7d05ad23dbcea619ee84f38a481416975c04
Data source: Yahoo Finance 日线，经 yfinance 获取。配置资产为 SPY、EFA、EEM、IEF、TLT、GLD、VNQ、DBC。BIL 日收益用作现金代理及超额收益参照，配置权重信号只使用资产池收益。已有价格缓存时读取 data_cache/prices_<缓存键>.csv。config.yaml:2-3，strategy.py:59-81，strategy.py:202-205，strategy.py:227-234，README.md:13
Data fetched: 2026-11-27T07:09:41+00:00
Sample period: 回测覆盖实际交易日 2011-01-03 至 2025-12-31。价格请求从 2009-12-01 开始，样本前数据用于建立首个 252 个交易日的估计窗口。config.yaml:5-10，README.md:15，logs/run_20261127_073114.log:16   data_range (from output): {'start': '2011-01-03', 'end': '2025-12-31'}
IS/OOS split: 按日期切分。样本内 2011-01-03 至 2020-12-31，样本外 2021-01-04 至 2025-12-31，均含首尾交易日。每个规则连续模拟一次，再分段汇报，跨界延续持仓。README.md:15-16 记录两段数据均早于论文发表，且未按两段表现修改规则。config.yaml:6-8，strategy.py:228-248
Costs and slippage: 每次调仓（含首次建仓）对资产池 ETF 买卖两侧分别按实际成交名义额收取 2 bps 交易成本和 3 bps 固定滑点，合计每侧 5 bps。代码以成交前净值为基准求解扣费后的净值比例和买卖名义额之和，在成交日一次性扣减净值。config.yaml:13-14，strategy.py:179-199，strategy.py:241-243，README.md:21
Signal lag: 月末交易日 t 收盘后，使用截至 t 的 252 个日收益率计算目标，默认在下一交易日 t+1 收盘成交。t 至 t+1 的收益按旧持仓（首次建仓前按 BIL 现金代理）计算，成交费用在 t+1 扣除，新权重适用于 t+1 至 t+2 的收盘到收盘收益。首笔在样本首个交易日收盘执行，使用前一交易日信号。config.yaml:10-12，strategy.py:153-175，strategy.py:179-199，strategy.py:232-243，README.md:20
Price adjustment: yfinance auto_adjust=True daily Close, total-return basis
Survivorship bias: 固定资产池采用事后仍存续且有充分历史的 ETF，选择可能受存续偏差影响。代码全期使用同一组配置标的并要求对齐价格有效。config.yaml:2-3，strategy.py:45-60，strategy.py:81-83，README.md:26
Parameter combinations tried: 4
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.04126428528209658, ann_vol=0.0573460467134883, sharpe=0.7348348320550061, sharpe_excess=0.6550821249763297, max_drawdown=-0.11951897001011591, total_return=0.4984331497152643, turnover=0.8958418239078443, n_trades=960
    oos: cagr=0.022373123882158952, ann_vol=0.07998313633214012, sharpe=0.3176643792482349, sharpe_excess=-0.0652739127419278, max_drawdown=-0.20100930057431043, total_return=0.11697005547885553, turnover=1.0423469835965764, n_trades=480
    full: cagr=0.03492991426966907, ann_vol=0.06573975149270864, sharpe=0.5563279807720216, sharpe_excess=0.35500141221611226, max_drawdown=-0.20144238272629778, total_return=0.6737049583688182, turnover=0.944667964004874, n_trades=1440
    benchmark: cagr=0.06015547192911108, ann_vol=0.09888949334385949, sharpe=0.6416286766995711, sharpe_excess=0.5075019303313238, max_drawdown=-0.2059042945998486, total_return=1.4019754715866561, turnover=0.3887659748944498, n_trades=1440
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "438_ the_mathematics_of_heuristic_portfolio_o"\nuniverse: [SPY, EFA, EEM, IEF, TLT, GLD, VNQ, DBC]\nrisk_free_ticker: BIL\n\ndata_start: \'2009-12-01\'\nstart: \'2011-01-01\'\nis_end: \'2020-12-31\'\nend: \'2025-12-31\'\n\nlookback_sessions: 252\nrebalance_months: 1\nexecution_lag_sessions: 1\ncost_bps: 2.0\nslippage_bps: 3.0\n'
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
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import brentq
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_ADJUSTMENT = "yfinance auto_adjust=True daily Close, total-return basis"
ANNUAL_SESSIONS = 252
TRADE_TOLERANCE = 1e-10
RULES = ("hrp", "equal_weight", "inverse_volatility", "equal_risk_contribution")


def load_config():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for key in ("data_start", "start", "is_end", "end"):
        config[key] = pd.Timestamp(config[key])
    if not config["data_start"] < config["start"] <= config["is_end"] < config["end"]:
        raise ValueError("Dates must satisfy data_start < start <= is_end < end.")
    universe = config["universe"]
    if len(universe) < 2 or len(set(universe)) != len(universe) or "BIL" in universe:
        raise ValueError("The universe needs at least two distinct symbols outside BIL.")
    if config["risk_free_ticker"] != "BIL":
        raise ValueError("The cash return series must be BIL.")
    for key, minimum in (("lookback_sessions", 2), ("rebalance_months", 1),
                         ("execution_lag_sessions", 1)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer of at least {minimum}.")
    cost, slippage = config["cost_bps"], config["slippage_bps"]
    if not (0 <= cost and 0 <= slippage and cost + slippage < 10_000):
        raise ValueError("Costs and slippage must be nonnegative and sum to less than 10,000 bps.")
    return config


def check_prices(prices, tickers, start, end):
    if list(prices.columns) != tickers or prices.empty:
        raise ValueError("The adjusted-close panel is missing requested assets.")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("Price dates must be sorted and unique.")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        missing = prices.columns[prices.isna().any()].tolist()
        raise ValueError(f"Missing or invalid adjusted close in the aligned panel. Missing: {missing}")
    tolerance = pd.Timedelta(days=7)
    if not (start <= prices.index[0] <= start + tolerance and
            end - tolerance <= prices.index[-1] <= end):
        raise ValueError("Adjusted close history does not cover the requested dates.")


def load_prices(config):
    tickers = [*config["universe"], config["risk_free_ticker"]]
    start, end = config["data_start"], config["end"]
    request = f"{','.join(tickers)}|{start.date()}|{end.date()}|adjusted-close-v1"
    cache_key = hashlib.sha256(request.encode()).hexdigest()[:16]
    path = CACHE / f"prices_{cache_key}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as cached:
            fetched_at = cached.readline().strip().removeprefix("# fetched_at=")
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(cached, index_col="Date", parse_dates=["Date"])
        check_prices(prices, tickers, start, end)
        return prices, fetched_at

    CACHE.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=CACHE, prefix="yf_") as download_cache:
        yf.set_tz_cache_location(download_cache)
        for attempt in range(2):
            try:
                raw = yf.download(tickers, start=str(start.date()),
                                  end=str((end + pd.Timedelta(days=1)).date()),
                                  interval="1d", auto_adjust=True, progress=False, timeout=15)
                prices = raw["Close"].reindex(columns=tickers) if not raw.empty else pd.DataFrame()
                prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
                check_prices(prices, tickers, start, end)
                break
            except (KeyError, ValueError) as exc:
                if attempt == 1:
                    raise RuntimeError(f"Yahoo Finance data could not be loaded: {exc}") from exc

    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(temporary, index_label="Date")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def hrp_weights(covariance):
    volatility = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(volatility, volatility)
    distance = np.sqrt(np.clip((1.0 - correlation) / 2.0, 0.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="single"))
    weights = np.empty(len(order))

    def cluster_variance(cluster):
        block = covariance[np.ix_(cluster, cluster)]
        inverse_variance = 1.0 / np.diag(block)
        local = inverse_variance / inverse_variance.sum()
        return local @ block @ local

    def split(cluster, mass):
        if len(cluster) == 1:
            weights[cluster[0]] = mass
            return
        mid = len(cluster) // 2
        left, right = cluster[:mid], cluster[mid:]
        left_variance, right_variance = cluster_variance(left), cluster_variance(right)
        total_variance = left_variance + right_variance
        if total_variance <= 0 or min(left_variance, right_variance) < 0:
            raise ValueError("Degenerate cluster variance in HRP.")
        split(left, mass * right_variance / total_variance)
        split(right, mass * left_variance / total_variance)

    split(order, 1.0)
    return weights


def equal_risk_weights(covariance):
    # Solve the positive log-barrier problem in Section 6 by coordinate descent.
    covariance = covariance / np.diag(covariance).max()
    diagonal = np.diag(covariance)
    budget = 1.0 / len(diagonal)
    weights = np.sqrt(budget / diagonal)
    for _ in range(10_000):
        for i, variance in enumerate(diagonal):
            cross = covariance[i] @ weights - variance * weights[i]
            root = np.hypot(cross, 2.0 * np.sqrt(variance * budget))
            if cross >= 0:
                weights[i] = 2.0 * budget / (root + cross)
            else:
                weights[i] = (root - cross) / (2.0 * variance)
        if np.max(np.abs(weights * (covariance @ weights) - budget)) < 1e-10:
            return weights / weights.sum()
    raise ValueError("Equal-risk-contribution solver did not converge.")


def allocation_targets(asset_returns, first_session, config):
    lookback = config["lookback_sessions"]
    lag = config["execution_lag_sessions"]
    first_signal = first_session - lag
    if first_signal + 1 < lookback:
        raise ValueError("Insufficient pre-sample history for the first lagged signal.")

    month = asset_returns.index.to_period("M").asi8
    month_ends = np.flatnonzero(month[:-1] != month[1:])
    scheduled = month_ends[(month[month_ends] + 1) % config["rebalance_months"] == 0]
    signals = [first_signal, *(i for i in scheduled if first_signal < i < len(month) - lag)]
    targets = {rule: {} for rule in RULES}
    for signal in signals:
        trailing = asset_returns.iloc[signal + 1 - lookback:signal + 1]
        covariance = trailing.cov().to_numpy()
        if (np.diag(covariance) <= 0).any():
            raise ValueError(f"Nonpositive asset variance at {trailing.index[-1].date()}.")
        inverse_volatility = 1.0 / np.sqrt(np.diag(covariance))
        fill = asset_returns.index[signal + lag]
        targets["hrp"][fill] = hrp_weights(covariance)
        targets["equal_weight"][fill] = np.full(len(covariance), 1.0 / len(covariance))
        targets["inverse_volatility"][fill] = inverse_volatility / inverse_volatility.sum()
        targets["equal_risk_contribution"][fill] = equal_risk_weights(covariance)
    return targets


def simulate(asset_returns, cash, targets, cost_rate):
    weights = np.zeros(asset_returns.shape[1])
    rows = []
    for date, returns, cash_return in zip(asset_returns.index, asset_returns.to_numpy(), cash, strict=True):
        holdings = weights * (1.0 + returns)
        gross = holdings.sum() + (1.0 - weights.sum()) * (1.0 + cash_return)
        weights = holdings / gross
        nav_fraction, turnover, trades = 1.0, 0.0, 0
        target = targets.get(date)
        if target is not None:
            # Solve for the remaining NAV so both fees use the actual filled notional.
            def turnover_at(fraction):
                return np.abs(fraction * target - weights).sum()

            nav_fraction = brentq(lambda fraction: fraction + cost_rate * turnover_at(fraction) - 1.0,
                                  0.0, 1.0)
            turnover = turnover_at(nav_fraction)
            trades = np.count_nonzero(np.abs(nav_fraction * target - weights) > TRADE_TOLERANCE)
            weights = target
        rows.append((gross * nav_fraction - 1.0, turnover, trades))
    return pd.DataFrame(rows, index=asset_returns.index, columns=["return", "turnover", "trades"])


def metrics(period, cash, opening_close):
    returns = period["return"].to_numpy()
    excess = returns - cash.loc[period.index].to_numpy()
    years = (period.index[-1] - opening_close).days / 365.2425
    wealth = np.r_[1.0, np.cumprod(1.0 + returns)]

    def sharpe(values):
        sigma = values.std(ddof=1)
        return float(np.sqrt(ANNUAL_SESSIONS) * values.mean() / sigma) if sigma > 0 else None

    return {
        "cagr": float(wealth[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(ANNUAL_SESSIONS)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(excess),
        "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0)),
        "total_return": float(wealth[-1] - 1.0),
        "turnover": float(period["turnover"].sum() / years),
        "n_trades": int(period["trades"].sum()),
    }


def main():
    config = load_config()
    prices, fetched_at = load_prices(config)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    first_session = returns.index.searchsorted(config["start"])
    split = returns.index.searchsorted(config["is_end"], side="right")
    if split - first_session < 2 or len(returns) - split < 2:
        raise ValueError("Both sample periods need at least two daily returns.")
    cash = returns[config["risk_free_ticker"]]
    asset_returns = returns[config["universe"]]
    targets = allocation_targets(asset_returns, first_session, config)
    periods = {"is": slice(first_session, split), "oos": slice(split, len(returns)),
               "full": slice(first_session, len(returns))}
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"PARAM_COMBOS_TRIED: {len(targets)}")
    reports = {}
    cost_rate = (config["cost_bps"] + config["slippage_bps"]) / 10_000.0
    for rule, rebalances in targets.items():
        daily = simulate(asset_returns.iloc[first_session:], cash.iloc[first_session:], rebalances, cost_rate)
        reports[rule] = {}
        for name, period in periods.items():
            start = period.start - first_session
            end = period.stop - first_session
            result = metrics(daily.iloc[start:end], cash, prices.index[period.start])
            reports[rule][name] = result
            print(f"{rule} {name.upper()}: {json.dumps(result, allow_nan=False)}")

    benchmark = reports["equal_weight"]
    output = {
        "strategy": config["strategy"],
        "data_range": {"start": str(returns.index[first_session].date()),
                       "end": str(returns.index[-1].date())},
        **reports["hrp"],
        "benchmark": {"name": "equal_weight", **benchmark["full"], **benchmark},
    }
    print(f"RESULTS_JSON: {json.dumps(output, allow_nan=False)}")


if __name__ == "__main__":
    main()