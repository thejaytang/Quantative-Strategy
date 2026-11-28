# r5 line-number correction — 452_simple_dynamic_stock_bond_gold_portfolio

This file only maps citations. It does not change `REVIEW_codex_self_r5.md`, official `strategy.py`, or the packed publication. A reported review file is never edited in place. r5 remains SHA-256 `1fde96efe5026a0c9026340a250395d1dda04b8e3e896f8b23dbbae619b6d60e` (65 lines).

## What r5 actually reviewed

- r5 date: 2026-11-27 18:05:52 Europe/Oslo. Conclusion: PASS on 成交数量检查. r5 states that the pre-entry review was author (Codex) self-review, not an independent third-party review.
- The line numbers in r5 refer to the review-run copy `/workspace/codex_runs/quant_burn/review_runs/452_simple_dynamic_stock_bond_gold_portfolio_self/strategy.py`, SHA-256 `1d183c67b333d1d5babd513a7b34695b9fcbeaa3e6f23774f8cdb832563c4eac` (348 lines).
- That copy is not the later pre-embed official file (SHA-256 `95d804289cfc84c04294eed2a8cbaa31d814734c0e2adf22b07cb68fd54ca3a8`, 358 lines) and is not the current official embed.
- Current official `strategy.py` SHA-256 `064db2281b208a5ec241eee0fcb2698eeeb6174fc7048f8d4cb7a01808a3f768` (357 lines). A later snapshot embed replaced `cached_frame` / `fetch_closes` with `load_config` / `load_prices`. Some `rebalance_targets` lines in the 348-line copy also differ from the current official function.

## strategy.py citations in r5 → current official 064db228

Ranges below use the r5 form `strategy.py:start`–`end` (backtick-separated numbers in the r5 text). "Exact" means that whole r5 range is still present, identically, at the official lines given.

| r5 citation (348-line copy) | Current official 064db228 | Notes |
|---|---|---|
| 32–56 | 33–57 | Exact. `EquityHolidays` / `equity_sessions`. |
| 59–93 | no counterpart | `cached_frame` and `fetch_closes`. Replaced by `load_config` at 61–64 and `load_prices` at 67–100. |
| 82–109 | 103–116 for the `split_closes` portion (r5 96–109) | r5 82–93 is `fetch_closes` and has no official counterpart. |
| 96–109 | 103–116 | Exact. `split_closes`. |
| 110–130 | 117–137 | Exact. End of `split_closes` and `cash_proxy_returns`. |
| 117–158 | 124–170, partial | `cash_proxy_returns` and the start of `rebalance_targets` still match. In the 348-line copy, the benchmark-target block around r5 153–156 differs from official 160–169. |
| 134–157, 134–178, 134–185 | 141–169 (`rebalance_targets`); 173–184 (`size_position_changes`); 187–204 (`fill_position_changes`) | Same benchmark-construction difference as above in the early part of these ranges. |
| 161–171 | 173–184 | `size_position_changes`, identical. |
| 161–185 | 173–204 | Sizing plus settlement. Official `fill_position_changes` is 187–204. |
| 174–190 | 187–204 | `fill_position_changes`. Identical; one shared `)` line is not unique under a line-for-line search. |
| 199–206 | 213–220 | Exact. First fill is locked from starting cash and the close before the first evaluation session. |
| 211–223 | 225–237 | Exact. Fill-loop settlement of already-locked quantities. |
| 211–232 | 225–246 | Exact. Settlement, then sizing of the next session's fill at this close. |
| 226–232 | 240–246 | Exact. After settlement, size only the next session's fill at this close. |
| 237–253 | 251–267 | Exact. `metrics`. |
| 253–254 | 267–268 | Exact. Turnover and `n_trades` fields. |
| 263–265 | 276–278 | Exact. Universe load in `main`. |
| 272–274 | 285–287 | Exact. Cost, slippage and `cash_buffer` in `main`. |
| 281–288 | 294–301 | Exact. Lag and cash-quote buffer checks. |
| 294–318 | 307–326, partial | `CACHE.mkdir`, `download_prices` and the `cached_frame(...)` call have no official counterpart. Official loading is `load_prices` at 339–340. Trailing identical setup lines map into 307–326. |
| 303–308 | 313–315 for the identical trailing lines | The `cached_frame` call itself has no official counterpart; official uses `load_prices` at 339–340 and `split_closes` at 309. |
| 309–325 | 317–333 | Exact. Evaluation masks and the start of the result block. |
| 317–318 | 325–326 | Exact. Strategy and benchmark both call `simulate`. |
| 337 | 345 | Exact. `IRX_MAX_QUOTE_AGE_DAYS`. |

r5 成交数量检查 citations in current official terms: shared `simulate` at 325–326; first-fill lock at 213–220; quantity formula at 173–184 and next-fill sizing at 240–246; settlement of locked quantities, with abort rather than same-close resize, at 187–204 and 225–237.

## Other r5 citations

- `config.yaml` lines 2–19, including 2–5, 2–7, 2–12, 7, 8–15, 16 and 17–19, still refer to the same pre-`data:` prefix in the current official config.yaml (that prefix was kept byte-for-byte when the snapshot was embedded).
- `README.md` citations (9–15, 14–15, 20, 21–23, 27, 27–29, 33, 41, 43–45) referred to the README in the review copy at the time of r5. The ORIG README was later updated only in the defect-table progress cell. Those r5 line numbers should not be read against a later README without this mapping.
- `backtest_results.md:6` and `backtest_results.md:17`–`29` referred to the then-current results sidecar. That sidecar was later replaced by the official-embed write-up. r5 was not edited. The RESULTS themselves still match the 18:03 official log.
- `codex_rerun_logs/rerun_20261127_180214.log` lines 3, 4 and 7–12 remain in that review-run log and are not part of official `strategy.py`.
