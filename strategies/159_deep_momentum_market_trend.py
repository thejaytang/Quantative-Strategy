"""159_deep_momentum_market_trend
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/159_deep_momentum_market_trend.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 498f8135d2067ed389c0a52a60055b65c4b14836 on 2026-11-27 06:36 Europe/Oslo, exit 0, output sha256 adc713e8dd0a12bba86b72e9f907fb95603cb07f3590581fdfda6c5a8d3287c8
Data source: Yahoo Finance 免费公开 Chart 接口日线。ETF：SPY、EFA、EEM、IEF、TLT、GLD、DBC、VNQ、UUP，基准为 SPY。现金参考为 Yahoo Finance ^IRX 日收盘利率。优先读取本地 CSV 快照，缓存缺失时从 Yahoo 下载。（config.yaml:3-5，strategy.py:26-56、100-116、439-448，README.md:13-16）
Data fetched: {"SPY": "2026-11-27T05:36:37+00:00", "EFA": "2026-11-27T05:36:37+00:00", "EEM": "2026-11-27T05:36:37+00:00", "IEF": "2026-11-27T05:36:37+00:00", "TLT": "2026-11-27T05:36:37+00:00", "GLD": "2026-11-27T05:36:37+00:00", "DBC": "2026-11-27T05:36:37+00:00", "VNQ": "2026-11-27T05:36:37+00:00", "UUP": "2026-11-27T05:36:37+00:00", "^IRX": "2026-11-27T05:36:37+00:00"}
Sample period: ETF 配置起点为 2008-01-01，实际报价覆盖 2008-01-02 至 2025-12-31，2008 年用于指标预热。^IRX 请求区间为 2009-01-01 至 2025-12-31。实际回测为 2015-01-05 至 2025-12-31。（config.yaml:2、7-12，README.md:24-25，strategy.py:439-448，logs/codex_fix_d2_run_20261126_175108.log:13）   data_range (from output): 2015-01-05 to 2025-12-31
IS/OOS split: 配置的 IS 训练区间为 2009-01-01 至 2021-12-31，实际 IS 回测为 2015-01-05 至 2021-12-31。配置的 OOS 从 2022-01-01 开始，实际 OOS 回测为 2022-01-03 至 2025-12-31。IS 逐年走步拟合，最后一次拟合在 2021-12-31。OOS 模型参数冻结，特征和仓位随行情更新，持仓和净值跨切分连续。README 记录该 OOS 区间此前已用于回测和评审。（config.yaml:7-12，strategy.py:330-370、455-483，README.md:24-25、38，logs/codex_fix_d2_run_20261126_175108.log:5-13）
Costs and slippage: 每次开盘换仓按各标的绝对成交名义额之和从现金扣除单边交易成本 5 bps 和固定滑点 2 bps，合计 7 bps。滑点以现金费用建模，账本成交价为复权开盘价。SPY 基准首次开盘建仓采用同一费率。以前一收盘的空头名义额按 ACT/360 扣年化 50 bps 借券费，前一收盘的正负现金余额按滞后的 ^IRX 换算利率计息。组合与基准期末均按持仓市值计价。（config.yaml:15-17，strategy.py:119-129、373-407、458-467，README.md:26、30-32，logs/codex_fix_d2_run_20261126_175108.log:13）
Signal lag: 交易日 t 收盘数据生成目标仓位，目标向后移一个交易日，在 t+1 复权开盘价成交，当日收盘计价。LSTM 训练目标为 t+1 开盘至 t+2 开盘的毛收益。（strategy.py:132-159、319-323、355-370、380-405，README.md:26）
Price adjustment: Yahoo dividend- and split-adjusted ETF close; open multiplied by adjusted close / quoted close; raw ^IRX close in percent
Survivorship bias: 固定使用九只现存且有足够长历史的 ETF，回溯选样存在存续和选择偏差。股指 ETF 历史报价反映各指数随时间变化的成分。（config.yaml:3-4，strategy.py:100-116，README.md:13、38）
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.008722701327142035, ann_vol=0.009904941842780898, sharpe=0.8817887932245972, sharpe_excess=0.05860412299869036, max_drawdown=-0.019509534549935093, total_return=0.06260689187639223, turnover=41.30196467351861, n_trades=15858
    oos: cagr=0.03818959332401861, ann_vol=0.010501040660460436, sharpe=3.5745306639599006, sharpe_excess=-0.36084492552071395, max_drawdown=-0.0049850704288952885, total_return=0.16087039342566656, turnover=22.29348434188947, n_trades=9027
    full: cagr=0.019313841737686577, ann_vol=0.01016087042870555, sharpe=1.8878338181989713, sharpe_excess=-0.09846194474697785, max_drawdown=-0.019509534549935093, total_return=0.23354888062937262, turnover=63.59544901540808, n_trades=24885
    benchmark  full: cagr=0.1351720279033517, ann_vol=0.1778810794470133, sharpe=0.8020847359071032, sharpe_excess=0.6886626470100211, max_drawdown=-0.33717268745960094, total_return=3.019286829180989, turnover=0.99930048965724, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 159_deep_momentum_market_trend\ndata_start: 2008-01-01\ntickers: [SPY, EFA, EEM, IEF, TLT, GLD, DBC, VNQ, UUP]\nbenchmark: SPY\ncash_reference: ^IRX\n\nin_sample:\n  start: 2009-01-01\n  end: 2021-12-31\nout_of_sample:\n  start: 2022-01-01\n  end: 2025-12-31\n\nannualization_sessions: 252\ntransaction_cost_bps: 5.0\nslippage_bps: 2.0\nannual_short_borrow_bps: 50.0\nfast_weeks: [4, 8, 20]\nslow_weeks: 52\nreturn_sessions: [1, 21, 63, 126, 252]\nmacd_spans: [[8, 24], [16, 48], [32, 96]]\nvol_span: 60\nannual_vol_floor: 0.05\nasset_vol_target: 0.15\nmax_gross_exposure: 1.0\n\nmodel:\n  seed: 159\n  min_regime_train_sessions: 504\n  regime_iterations: 60\n  regime_leaf_nodes: 15\n  min_lstm_train_sessions: 504\n  sequence_length: 20\n  hidden_size: 10\n  epochs: 8\n  batch_dates: 128\n  learning_rate: 0.003\n'
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
"""Causal ETF backtest of a market-transition-conditioned momentum network."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
# States encode the signs of slow and fast momentum in two bits.
# A one-bit transition has its own class; all other observations are class 0.
TRANSITIONS = np.array([[0, 1, 2, 0], [3, 0, 0, 4], [5, 0, 0, 6], [0, 7, 8, 0]])
N_REGIME_CLASSES = 9


def load_history(ticker, start, end, *, adjusted=True):
    symbol = quote(ticker, safe="")
    cache = ROOT / "data_cache" / f"{symbol}_{start.date()}_{end.date()}.csv"
    if cache.exists():
        with cache.open(encoding="utf-8") as file:
            header = file.readline()
            if header.startswith("# fetched_at="):
                fetched_at = header.removeprefix("# fetched_at=").strip()
            else:
                # The inherited ETF snapshots predate embedded fetch timestamps.
                fetched_at = datetime.fromtimestamp(cache.stat().st_mtime, timezone.utc).isoformat(
                    timespec="seconds"
                )
                file.seek(0)
            frame = pd.read_csv(file, index_col="date", parse_dates=["date"])
        return frame, fetched_at

    query = urlencode({
        "period1": int(start.tz_localize("UTC").timestamp()),
        "period2": int((end + pd.Timedelta(days=1)).tz_localize("UTC").timestamp()),
        "interval": "1d",
    })
    failures = []
    for host in ("query2.finance.yahoo.com", "query1.finance.yahoo.com"):
        request = Request(
            f"https://{host}/v8/finance/chart/{symbol}?{query}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        try:
            with urlopen(request, timeout=20) as response:
                chart = json.load(response)["chart"]
            fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            if chart["error"]:
                raise ValueError(chart["error"])
            row = chart["result"][0]
            prices = row["indicators"]["quote"][0]
            dates = pd.to_datetime(row["timestamp"], unit="s", utc=True)
            dates = dates.tz_convert(row["meta"]["exchangeTimezoneName"])
            dates = dates.tz_localize(None).normalize().rename("date")
            close = np.asarray(prices["close"], dtype=float)
            if adjusted:
                adjusted_close = np.asarray(
                    row["indicators"]["adjclose"][0]["adjclose"], dtype=float
                )
                adjusted_open = np.asarray(prices["open"], dtype=float) * adjusted_close / close
                frame = pd.DataFrame({"open": adjusted_open, "close": adjusted_close}, index=dates)
                frame = frame.where(frame > 0)
            else:
                frame = pd.DataFrame({"close": close}, index=dates)
            frame = frame.where(np.isfinite(frame)).dropna().sort_index()
            frame = frame.loc[start:end]
            if (
                frame.empty
                or not frame.index.is_unique
                or frame.index[0] > start + pd.Timedelta(days=10)
                or frame.index[-1] < end - pd.Timedelta(days=7)
            ):
                raise ValueError("price history does not cover the requested range")
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            failures.append(f"{host}: {exc}")
            continue

        temporary = cache.with_suffix(f".{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as file:
                file.write(f"# fetched_at={fetched_at}\n")
                frame.to_csv(file)
            temporary.replace(cache)
        finally:
            temporary.unlink(missing_ok=True)
        return frame, fetched_at
    raise RuntimeError(f"Yahoo Finance download failed for {ticker}: {'; '.join(failures)}")


def load_universe(config, end):
    tickers = list(dict.fromkeys([*config["tickers"], config["benchmark"]]))
    start = pd.Timestamp(config["data_start"])
    (ROOT / "data_cache").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        histories = list(pool.map(lambda ticker: load_history(ticker, start, end), tickers))
    prices = {ticker: frame for ticker, (frame, _) in zip(tickers, histories)}
    fetched_at = {ticker: timestamp for ticker, (_, timestamp) in zip(tickers, histories)}
    benchmark_dates = prices[config["benchmark"]].index
    for ticker, frame in prices.items():
        if not frame.index.equals(benchmark_dates):
            raise ValueError(f"{ticker} quote dates differ from the benchmark's session calendar")
        if not np.isfinite(frame.to_numpy()).all() or (frame <= 0).any().any():
            raise ValueError(f"{ticker} has invalid adjusted prices")
    quotes = pd.concat(prices, axis=1)
    print(f"Yahoo Finance adjusted prices | {len(quotes)} aligned ETF sessions", flush=True)
    return quotes.xs("open", axis=1, level=1), quotes.xs("close", axis=1, level=1), fetched_at


def cash_carry(rates, dates):
    """Use only IRX closes dated before the ETF session receiving the accrual."""
    prior_dates = dates - pd.Timedelta(days=1)
    quotes = rates.reindex(prior_dates, method="ffill", tolerance=pd.Timedelta(days=6))
    if not np.isfinite(quotes.to_numpy()).all():
        raise ValueError("Missing ^IRX close within the seven calendar days before an ETF session")
    days = dates.to_series().diff().dt.days.fillna(0)
    discount = quotes.to_numpy() / 100
    # IRX is a bank-discount quote; convert a 91-day bill to simple ACT/360 carry.
    money_market_yield = discount / (1 - discount * 91 / 360)
    return pd.DataFrame({"cash_return": money_market_yield * days / 360, "days": days})


def make_features(closes, config):
    log_price = np.log(closes)
    volatility = log_price.diff().ewm(
        span=config["vol_span"], min_periods=config["vol_span"], adjust=False
    ).std()
    root_year = np.sqrt(config["annualization_sessions"])
    annual_vol = (volatility * root_year).clip(lower=config["annual_vol_floor"])
    daily_vol = annual_vol / root_year
    scale = config["asset_vol_target"] / (closes.shape[1] * annual_vol)
    scale = scale.div((scale.sum(axis=1) / config["max_gross_exposure"]).clip(lower=1), axis=0)

    slow_days = 5 * config["slow_weeks"]
    fast_days = [5 * weeks for weeks in config["fast_weeks"]]
    horizons = dict.fromkeys([*config["return_sessions"], *fast_days, slow_days])
    momentum = {
        days: log_price.diff(days) / (np.sqrt(days) * daily_vol) for days in horizons
    }
    macds = [
        (log_price.ewm(span=fast, adjust=False).mean()
         - log_price.ewm(span=slow, adjust=False).mean()) / (daily_vol * np.sqrt(slow))
        for fast, slow in config["macd_spans"]
    ]
    features = np.stack(
        [frame.to_numpy() for frame in [*momentum.values(), *macds]], axis=2
    ).clip(-8, 8).astype(np.float32)
    anchors = np.tanh(np.stack(
        [momentum[days].to_numpy() for days in fast_days], axis=2
    ) / 2).astype(np.float32)

    labels = np.empty(anchors.shape, dtype=np.int8)
    for column, days in enumerate(fast_days):
        state = (2 * (momentum[slow_days].to_numpy() > 0)
                 + (momentum[days].to_numpy() > 0))
        labels[:-1, :, column] = TRANSITIONS[state[:-1], state[1:]]
    labels[-1] = -1
    return features, anchors, labels, scale


def regime_probabilities(train_x, train_y, predict_x, config):
    n_dates, n_assets, n_features = predict_x.shape
    scores = np.zeros((n_dates * n_assets, train_y.shape[2], N_REGIME_CLASSES),
                      dtype=np.float32)
    for column in range(train_y.shape[2]):
        model = HistGradientBoostingClassifier(
            max_iter=config["regime_iterations"],
            max_leaf_nodes=config["regime_leaf_nodes"],
            early_stopping=False,
            random_state=config["seed"],
        )
        labels = train_y[:, :, column].ravel()
        classes = np.unique(labels)
        if len(classes) == 1:
            scores[:, column, classes[0]] = 1
        else:
            model.fit(train_x.reshape(-1, n_features), labels)
            scores[:, column, model.classes_] = model.predict_proba(
                predict_x.reshape(-1, n_features)
            )
    return scores.reshape(n_dates, n_assets, -1)


class MomentumLSTM:
    """An LSTM with a softmax over momentum horizons and a sigmoid exposure gate."""

    def __init__(self, n_features, n_horizons, config):
        self.config = config
        self.rng = np.random.default_rng(config["seed"])
        size = config["hidden_size"]
        self.n_horizons = n_horizons
        self.params = [
            self.rng.normal(0, 1 / np.sqrt(n_features), (n_features, 4 * size)),
            self.rng.normal(0, 1 / np.sqrt(size), (size, 4 * size)),
            np.r_[np.zeros(size), np.ones(size), np.zeros(2 * size)],
            self.rng.normal(0, 0.1, (size, n_horizons + 1)),
            np.zeros(n_horizons + 1),
        ]
        self.params = [value.astype(np.float32) for value in self.params]

    def forward(self, sequences, anchors):
        input_w, recurrent_w, bias, output_w, output_bias = self.params
        steps, count, _ = sequences.shape
        size = recurrent_w.shape[0]
        projected = sequences @ input_w + bias
        hidden = np.zeros((steps + 1, count, size), dtype=np.float32)
        cell = np.zeros_like(hidden)
        gates = np.empty_like(projected)
        for step in range(steps):
            combined = projected[step] + hidden[step] @ recurrent_w
            gates[step, :, :3 * size] = expit(combined[:, :3 * size])
            gates[step, :, 3 * size:] = np.tanh(combined[:, 3 * size:])
            i, f, o, g = np.split(gates[step], 4, axis=1)
            cell[step + 1] = f * cell[step] + i * g
            hidden[step + 1] = o * np.tanh(cell[step + 1])

        logits = hidden[-1] @ output_w + output_bias
        mix = softmax(logits[:, :self.n_horizons], axis=1)
        strength = expit(logits[:, -1])
        trend = (mix * anchors).sum(axis=1)
        return strength * trend, (hidden, cell, gates, mix, strength, trend)

    def gradients(self, sequences, anchors, returns):
        positions, (hidden, cell, gates, mix, strength, trend) = self.forward(
            sequences, anchors
        )
        # Each batch date contains all assets. Differentiate negative portfolio Sharpe.
        pnl = (positions.reshape(returns.shape) * returns).sum(axis=1)
        centered = pnl - pnl.mean()
        std = np.sqrt(np.mean(centered ** 2) + 1e-12)
        d_pnl = (pnl.mean() * centered / std ** 3 - 1 / std) / len(pnl)
        d_position = (d_pnl[:, None] * returns).ravel()
        d_mix_logits = (d_position * strength)[:, None] * mix * (anchors - trend[:, None])
        d_strength_logits = d_position * trend * strength * (1 - strength)
        d_logits = np.column_stack([d_mix_logits, d_strength_logits])
        d_hidden = d_logits @ self.params[3].T
        d_cell = np.zeros_like(d_hidden)
        d_gates = np.empty_like(gates)
        for step in range(len(gates) - 1, -1, -1):
            i, f, o, g = np.split(gates[step], 4, axis=1)
            tanh_cell = np.tanh(cell[step + 1])
            d_cell += d_hidden * o * (1 - tanh_cell ** 2)
            d_gates[step] = np.concatenate([
                d_cell * g * i * (1 - i),
                d_cell * cell[step] * f * (1 - f),
                d_hidden * tanh_cell * o * (1 - o),
                d_cell * i * (1 - g ** 2),
            ], axis=1)
            d_hidden = d_gates[step] @ self.params[1].T
            d_cell *= f

        flat_gates = d_gates.reshape(-1, d_gates.shape[-1])
        return [
            sequences.reshape(-1, sequences.shape[-1]).T @ flat_gates,
            hidden[:-1].reshape(-1, hidden.shape[-1]).T @ flat_gates,
            flat_gates.sum(axis=0),
            hidden[-1].T @ d_logits,
            d_logits.sum(axis=0),
        ]

    @staticmethod
    def batch(windows, anchors, dates):
        selected = windows[dates]
        sequences = selected.transpose(3, 0, 1, 2).reshape(
            selected.shape[3], -1, selected.shape[2]
        )
        return np.ascontiguousarray(sequences), anchors[dates].reshape(-1, anchors.shape[2])

    def fit(self, windows, anchors, returns, dates):
        averages = [np.zeros_like(value) for value in self.params]
        squares = [np.zeros_like(value) for value in self.params]
        step = 0
        batch_dates = self.config["batch_dates"]
        for _ in range(self.config["epochs"]):
            order = self.rng.permutation(dates)
            for start in range(0, len(dates), batch_dates):
                batch = order[start:start + batch_dates]
                sequences, trends = self.batch(windows, anchors, batch)
                grads = self.gradients(sequences, trends, returns[batch])
                step += 1
                for param, average, square, grad in zip(self.params, averages, squares, grads):
                    average *= 0.9
                    average += 0.1 * grad
                    square *= 0.999
                    square += 0.001 * grad ** 2
                    corrected_mean = average / (1 - 0.9 ** step)
                    corrected_std = np.sqrt(square / (1 - 0.999 ** step))
                    param -= self.config["learning_rate"] * corrected_mean / (corrected_std + 1e-8)

    def predict(self, windows, anchors, dates):
        batch_dates = self.config["batch_dates"]
        outputs = []
        for start in range(0, len(dates), batch_dates):
            batch = dates[start:start + batch_dates]
            sequences, trends = self.batch(windows, anchors, batch)
            position, _ = self.forward(sequences, trends)
            outputs.append(position.reshape(len(batch), anchors.shape[1]))
        return np.concatenate(outputs)


def walk_forward(opens, closes, config):
    dates = closes.index
    settings = config["model"]
    start_is = dates.searchsorted(pd.Timestamp(config["in_sample"]["start"]))
    end_is = dates.searchsorted(pd.Timestamp(config["in_sample"]["end"]), side="right")
    start_oos = dates.searchsorted(pd.Timestamp(config["out_of_sample"]["start"]))
    if not start_is < end_is == start_oos < len(dates):
        raise ValueError("IS and OOS must be nonempty, consecutive session ranges")

    features, anchors, labels, scale = make_features(closes, config)
    # A decision at close t is executed at open t+1 and held to open t+2.
    training_returns = ((opens.shift(-2) / opens.shift(-1) - 1) * scale).to_numpy(
        dtype=np.float32
    )
    valid = np.isfinite(features).all(axis=(1, 2))
    scores = np.full((*features.shape[:2], anchors.shape[2] * N_REGIME_CLASSES),
                     np.nan, dtype=np.float32)
    targets = np.full(closes.shape, np.nan)
    first_score = None
    sequence_length = settings["sequence_length"]
    first_session_of_year = np.flatnonzero(dates.year[1:] != dates.year[:-1]) + 1
    # The last IS close also gets a refit so the first OOS open uses the frozen model.
    final_fit = start_oos - 1
    boundaries = [i for i in first_session_of_year if start_is < i < final_fit]
    boundaries.extend([final_fit, len(dates)])

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        # Regime labels at t use close t+1. Both training stages stop before
        # the first signal date of the new block.
        regime_train = np.arange(start_is, start - 1)
        regime_train = regime_train[valid[regime_train]]
        if len(regime_train) < settings["min_regime_train_sessions"]:
            continue
        if first_score is None:
            first_score = start
        scores[start:end] = regime_probabilities(
            features[regime_train], labels[regime_train], features[start:end], settings
        )

        first_sequence_end = first_score + sequence_length - 1
        # LSTM targets at t use open t+2, which must also be inside the training block.
        train_dates = np.arange(first_sequence_end, start - 2)
        if len(train_dates) < settings["min_lstm_train_sessions"]:
            continue

        inputs = np.concatenate([features, scores], axis=2)
        # Leading padding keeps window indices aligned to their last session.
        padded = np.pad(inputs, ((sequence_length - 1, 0), (0, 0), (0, 0)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, sequence_length, axis=0)
        model = MomentumLSTM(inputs.shape[2], anchors.shape[2], settings)
        model.fit(windows, anchors, training_returns, train_dates)
        positions = model.predict(windows, anchors, np.arange(start, end))
        targets[start:end] = positions * scale.iloc[start:end].to_numpy()
        if not np.isfinite(targets[start:end]).all():
            raise ValueError(f"Nonfinite strategy weights for block starting {dates[start].date()}")
        print(f"Fit {dates[start].date()} | {len(regime_train)} regime / "
              f"{len(train_dates)} LSTM training sessions", flush=True)
    if not np.isfinite(targets[final_fit]).all():
        raise ValueError("The IS range is too short to fit both models for OOS")
    # Shift each decision to the next open. The first rows are model warm-up.
    return pd.DataFrame(targets, index=dates, columns=closes.columns).shift(1)


def simulate(targets, opens, closes, carry, cost_rate, borrow_rate):
    """Hold fractional shares and cash, with rebalancing at the adjusted open."""
    cash = 1.0
    prior_close_nav = 1.0
    prior_short_notional = 0.0
    shares = np.zeros(opens.shape[1])
    records = []
    for target, open_px, close_px, (cash_return, days) in zip(
        targets.to_numpy(), opens.to_numpy(), closes.to_numpy(), carry.to_numpy()
    ):
        # Post close-to-close carry on prior-close balances after today's open trade.
        interest = cash * cash_return - prior_short_notional * borrow_rate * days / 360
        holdings = shares * open_px
        open_nav = cash + holdings.sum()
        if open_nav <= 0:
            raise ValueError("Portfolio NAV is nonpositive at the open")
        turnover = 0.0
        n_trades = 0
        if np.isfinite(target).all():
            trade = open_nav * target - holdings
            executed = np.abs(trade) > open_nav * 1e-12
            trade[~executed] = 0
            n_trades = int(executed.sum())
            notional = np.abs(trade).sum()
            cash -= trade.sum() + cost_rate * notional
            shares += trade / open_px
            turnover = notional / open_nav
        cash += interest
        close_holdings = shares * close_px
        close_nav = cash + close_holdings.sum()
        if close_nav <= 0:
            raise ValueError("Portfolio NAV is nonpositive at the close")
        records.append((close_nav / prior_close_nav - 1, cash_return, turnover, n_trades))
        prior_close_nav = close_nav
        prior_short_notional = -np.minimum(close_holdings, 0).sum()
    return pd.DataFrame(
        records, index=opens.index, columns=["return", "cash_return", "turnover", "n_trades"]
    )


def annualized_sharpe(returns, sessions):
    std = returns.std(ddof=1) if len(returns) > 1 else np.nan
    # A cash-only daily_results can differ from its reference by floating-point rounding.
    return returns.mean() / std * np.sqrt(sessions) if std > 1e-15 else np.nan


def metrics(frame, sessions):
    if frame.empty:
        raise ValueError("A report period has no simulated trading sessions")
    returns = frame["return"].to_numpy()
    equity = np.r_[1.0, np.cumprod(1 + returns)]
    std = returns.std(ddof=1) if len(returns) > 1 else np.nan
    summary = {
        "cagr": equity[-1] ** (sessions / len(returns)) - 1,
        "ann_vol": std * np.sqrt(sessions),
        "sharpe": annualized_sharpe(returns, sessions),
        "sharpe_excess": annualized_sharpe(returns - frame["cash_return"].to_numpy(), sessions),
        "max_drawdown": (equity / np.maximum.accumulate(equity) - 1).min(),
        "total_return": equity[-1] - 1,
        "turnover": frame["turnover"].sum(),
    }
    result = {key: float(value) if np.isfinite(value) else None for key, value in summary.items()}
    result["n_trades"] = int(frame["n_trades"].sum())
    return result


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    is_split, oos_split = config["in_sample"], config["out_of_sample"]
    end = pd.Timestamp(oos_split["end"])
    opens, closes, fetched_at = load_universe(config, end)
    cash_symbol = config["cash_reference"]
    rates, fetched_at[cash_symbol] = load_history(
        cash_symbol, pd.Timestamp(is_split["start"]), end, adjusted=False
    )
    print("PARAM_COMBOS_TRIED: 1", flush=True)
    print("DATA_FETCHED_AT: " + json.dumps(fetched_at), flush=True)
    print("PRICE_ADJUSTMENT: Yahoo dividend- and split-adjusted ETF close; "
          "open multiplied by adjusted close / quoted close; raw ^IRX close in percent",
          flush=True)
    tickers = config["tickers"]
    with threadpool_limits(limits=1):
        targets = walk_forward(opens[tickers], closes[tickers], config).dropna()
    trade_dates = targets.index
    carry = cash_carry(rates["close"], trade_dates)
    cost = (config["transaction_cost_bps"] + config["slippage_bps"]) / 10_000
    borrow_rate = config["annual_short_borrow_bps"] / 10_000
    strategy = simulate(targets, opens.loc[trade_dates, tickers],
                        closes.loc[trade_dates, tickers], carry, cost, borrow_rate)
    symbol = config["benchmark"]
    benchmark_target = pd.DataFrame(np.nan, index=trade_dates, columns=[symbol])
    benchmark_target.iloc[0, 0] = 1 / (1 + cost)
    benchmark = simulate(benchmark_target, opens.loc[trade_dates, [symbol]],
                         closes.loc[trade_dates, [symbol]], carry, cost, borrow_rate)

    result = {
        "strategy": config["strategy"],
        "data_range": f"{trade_dates[0].date()} to {trade_dates[-1].date()}",
        "benchmark": {"name": f"{symbol} buy and hold"},
    }
    splits = {"is": is_split, "oos": oos_split,
              "full": {"start": is_split["start"], "end": oos_split["end"]}}
    rows = {}
    for name, dates in splits.items():
        period = slice(str(dates["start"]), str(dates["end"]))
        rows["Strategy", name.upper()] = result[name] = metrics(
            strategy.loc[period], config["annualization_sessions"]
        )
        rows[symbol, name.upper()] = result["benchmark"][name] = metrics(
            benchmark.loc[period], config["annualization_sessions"]
        )
    print(f"Backtest {result['data_range']} | {config['transaction_cost_bps']:g} bps costs + "
          f"{config['slippage_bps']:g} bps slippage per dollar traded | "
          f"{config['annual_short_borrow_bps']:g} bps annual short borrow | "
          f"cash reference {cash_symbol} | turnover is cumulative as a multiple of NAV")
    percentages = ["cagr", "ann_vol", "max_drawdown", "total_return"]
    print(pd.DataFrame.from_dict(rows, orient="index").to_string(
        formatters={metric: "{:.2%}".format for metric in percentages},
        float_format=lambda value: f"{value:.3f}", na_rep="N/A"
    ))
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
