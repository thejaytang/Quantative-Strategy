# Quantative-Strategy

**研究用途，非投资建议（Research only, not investment advice）。** 本仓库任何策略进入投资系统前须另经 Gatekeeper 审查。不含下单或券商接口代码，只用免费公开数据，不含第三方原始价格文件。

每个策略一个文件：`strategies/<编号>_<名称>.py`，一条命令运行：`python strategies/<文件>.py`（依赖见 requirements.txt）。文件头写明数据源与抓取时间、样本期、IS/OOS 划分、成本与滑点、信号滞后、复权口径、幸存者偏差说明、尝试的参数组合数，以及在记录的 commit 上实际运行得到的关键数字（backtest reproduced）。收录条件：真实运行、确有交易、无同收盘成交前视、非作者审查 PASS、单文件复跑与审查时结果一致。

本表由文件实际内容生成（2026-11-26 13:10 Europe/Oslo），共 3 个。

| 文件 | 作者 | 代码 commit | 全样本 Sharpe | 全样本 CAGR |
|---|---|---|---|---|
| strategies/100_etf_asset_momentum_corr_hedge.py | Codex | e45e81ae3a | 0.5880769053765496 | 0.07506872233162931 |
| strategies/105_factor_momentum_style_etf.py | Codex | e45e81ae3a | 0.7670643508346912 | 0.11463689309321756 |
| strategies/107_illiquidity_premium_amihud.py | Codex | f6870af4ff | 0.7804337271623667 | 0.1739239491585871 |
