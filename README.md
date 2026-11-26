# Quantative-Strategy

**研究用途，非投资建议（Research only, not investment advice）。** 本仓库任何策略进入投资系统前须另经 Gatekeeper 审查。不含下单或券商接口代码，只用免费公开数据，不含第三方原始价格文件。

每个策略一个文件：`strategies/<编号>_<名称>.py`，一条命令运行：`python strategies/<文件>.py`（依赖见 requirements.txt）。文件头写明数据源与抓取时间、样本期、IS/OOS 划分、成本与滑点、信号滞后、复权口径、幸存者偏差说明、尝试的参数组合数，以及在记录的 commit 上实际运行得到的关键数字（backtest reproduced）。收录条件：真实运行、确有交易、无同收盘成交前视、非作者审查 PASS、单文件复跑与审查时结果一致。

本表由文件实际内容生成（2026-11-26 14:57 Europe/Oslo），共 32 个。

| 文件 | 作者 | 代码 commit | 全样本 Sharpe | 全样本 CAGR |
|---|---|---|---|---|
| strategies/100_etf_asset_momentum_corr_hedge.py | Codex | e45e81ae3a | 0.5880769053765496 | 0.07506872233162931 |
| strategies/105_factor_momentum_style_etf.py | Codex | e45e81ae3a | 0.7670643508346912 | 0.11463689309321756 |
| strategies/106_low_ivol_low_max.py | Codex | ad594150a0 | 0.8206469484361287 | 0.11596319232313723 |
| strategies/107_illiquidity_premium_amihud.py | Codex | f6870af4ff | 0.7804337271623667 | 0.1739239491585871 |
| strategies/109_factor_etf_vol_matched.py | Codex | ad594150a0 | 0.719138747893456 | 0.09708749816782114 |
| strategies/110_cointegration_etf_pairs.py | Codex | ad594150a0 | -1.9086957510381422 | -0.0066972690980858784 |
| strategies/112_regime_switching_pairs.py | Codex | ad594150a0 | -0.2205187803621582 | -0.005553293487243871 |
| strategies/113_kalman_garch_pairs.py | Codex | ad594150a0 | -0.4358413397896265 | -0.011697193932705185 |
| strategies/122_stock_bond_correlation_regime.py | Codex | ad594150a0 | 1.0099163972519734 | 0.10282929099397098 |
| strategies/125_liquidity_risk_indicator_timing.py | Codex | ad594150a0 | 0.9392801869667401 | 0.092108548038075 |
| strategies/126_hmm_regime_allocation.py | Codex | ad594150a0 | 0.9421612030950982 | 0.10710208722331305 |
| strategies/129_ml_etf_trend_vs_oscillation.py | Codex | ad594150a0 | -0.1633159490129353 | -0.00856614180659876 |
| strategies/130_ml_risk_budget_allocation.py | Codex | ad594150a0 | 0.8004610178300348 | 0.05922868384241897 |
| strategies/148_decomposing_momentum.py | Codex | 2b23c5bbf7 | 0.8720364930091413 | 0.20916162811949102 |
| strategies/156_cta_replication_trend_factors.py | Codex | 2b23c5bbf7 | 0.44395267327406285 | 0.033093560604958716 |
| strategies/165_retail_signal_failure.py | Codex | 2b23c5bbf7 | 0.43541324492441674 | 0.05075523692537032 |
| strategies/167_bayesian_cointegration_lowvol.py | Codex | 2b23c5bbf7 | -0.19791642317160424 | -0.01419780064634768 |
| strategies/179_crude_oil_news_sentiment.py | Codex | 2b23c5bbf7 | -0.3389139440305118 | -0.039653231117312226 |
| strategies/183_safe_asset_dollar_cycle.py | Codex | 2b23c5bbf7 | 0.1397995385765699 | 0.004613464128480738 |
| strategies/185_gold_month_seasonality.py | Codex | 2b23c5bbf7 | 0.7807274675730697 | 0.062351172766873075 |
| strategies/191_shap_quantile_sector_allocation.py | Codex | 2b23c5bbf7 | 0.4696202287092069 | 0.05901278586658676 |
| strategies/192_downside_upside_rv_forecast.py | Codex | 2b23c5bbf7 | 0.5860331533877041 | 0.04621676334252989 |
| strategies/303_timing_moving_average_distance.py | Codex | 2b23c5bbf7 | 0.45867381657213385 | 0.07523504995979335 |
| strategies/304_unintended_consequences_rebalancing.py | Codex | 2b23c5bbf7 | 0.8171238877819083 | 0.09014626117652913 |
| strategies/309_intangible_value_sixth_factor.py | Codex | 2b23c5bbf7 | 0.6812919092091535 | 0.10776069989766413 |
| strategies/310_factor_beta_overnight_intraday.py | Codex | 2b23c5bbf7 | -1.614616855678312 | -0.027935906863817195 |
| strategies/312_cost_access_understanding_price.py | Codex | 2b23c5bbf7 | -0.9523822221196712 | -0.043607644709638405 |
| strategies/314_statistical_proxy_based_meanreverting.py | Codex | 2b23c5bbf7 | -0.4590101262621272 | -0.023263609116357276 |
| strategies/315_leveraged_funds_robust_replication.py | Codex | 2b23c5bbf7 | -3.893135170656959 | -0.011175684674659903 |
| strategies/316_hidden_cost_etf_investing.py | Codex | 2b23c5bbf7 | -7.246596107165238 | -0.07780674302569868 |
| strategies/317_trendbreaks_persistence_closedend_fund.py | Codex | 2b23c5bbf7 | -0.014750860045009091 | -0.0003613816756397048 |
| strategies/323_information_transmission_across_tail.py | Codex | b43d901ca9 | 0.16070605776219252 | -0.006924251937111547 |
