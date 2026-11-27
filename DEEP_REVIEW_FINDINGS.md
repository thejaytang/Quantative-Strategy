# Deep-review findings (already published)

Already-published strategies are not removed from git history. If a later deep review finds a real problem (look-ahead, wrong data, or wrong numbers), the file is kept as last published on origin/main and listed here in a new commit. Fixes, if any, also land in a new commit.

Generated 2026-11-27 10:05 Europe/Oslo.

| File | Latest self-review | Category | Stage note |
|---|---|---|---|
| strategies/382_statistical_arbitrage_in_polish_equities.py | 待修 | 成交日最终收盘价参与确定开仓股数; 股票池以事后可用数据构造，样本外仍继承筛选偏差 | kept_from_origin: latest self-review is not PASS |
| strategies/438_the_mathematics_of_heuristic_portfolio_o.py | 待修 | 成交规模依赖成交日最终收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/463_simple_dynamic_stock_bond_gold_portfolio.py | 待修 | 成交金额依赖成交当日最终收盘 | kept_from_origin: latest self-review is not PASS |
