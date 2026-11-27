# Quantative-Strategy

**研究用途，非投资建议（Research only, not investment advice）。** 本仓库任何策略进入投资系统前须另经 Gatekeeper 审查。不含下单或券商接口代码，只用免费公开数据，不含第三方原始价格文件。自 2026-11-27 起新收录策略在入库前的审查为作者（Codex）自审，不是独立第三方审查；此前已收录策略曾经过非作者审查。

每个策略一个文件：`strategies/<编号>_<名称>.py`，一条命令运行：`python strategies/<文件>.py`（依赖见 requirements.txt）。文件头写明数据源与抓取时间、样本期、IS/OOS 划分、成本与滑点、信号滞后、复权口径、幸存者偏差说明、尝试的参数组合数，以及在记录的 commit 上实际运行得到的关键数字（backtest reproduced）。收录条件：真实运行、确有交易、无同收盘成交前视、Codex 自审 PASS（进入本仓库前的审查为作者自审，不是独立第三方审查）、单文件复跑与审查时结果一致。

本表由文件实际内容生成（2026-11-27 07:47 Europe/Oslo），共 63 个。

| 文件 | 作者 | 代码 commit | 全样本 Sharpe | 全样本 CAGR |
|---|---|---|---|---|
| strategies/022_sector_residual_statarb.py | Gemini | ed11995234 | -0.03513344696253851 | -0.006229499073650224 |
| strategies/100_etf_asset_momentum_corr_hedge.py | Codex | e45e81ae3a | 0.5880769053765496 | 0.07506872233162931 |
| strategies/101_correlated_stress_reversal.py | Codex | 3140bbcffa | -0.3568289041254561 | -0.0331016237951367 |
| strategies/102_alt_etf_short_term_reversal.py | Codex | 3140bbcffa | -4.850826308066646 | -0.1832096548612029 |
| strategies/103_breaking_bad_trends.py | Codex | ed11995234 | 0.24156559048879273 | 0.011219751158929592 |
| strategies/105_factor_momentum_style_etf.py | Codex | e45e81ae3a | 0.7670643508346912 | 0.11463689309321756 |
| strategies/106_low_ivol_low_max.py | Codex | ad594150a0 | 0.8206469484361287 | 0.11596319232313723 |
| strategies/107_illiquidity_premium_amihud.py | Codex | f6870af4ff | 0.7804337271623667 | 0.1739239491585871 |
| strategies/109_factor_etf_vol_matched.py | Codex | ad594150a0 | 0.719138747893456 | 0.09708749816782114 |
| strategies/110_cointegration_etf_pairs.py | Codex | ad594150a0 | -1.9086957510381422 | -0.0066972690980858784 |
| strategies/112_regime_switching_pairs.py | Codex | ad594150a0 | -0.2205187803621582 | -0.005553293487243871 |
| strategies/113_kalman_garch_pairs.py | Codex | ad594150a0 | -0.4358413397896265 | -0.011697193932705185 |
| strategies/122_stock_bond_correlation_regime.py | Codex | ad594150a0 | 1.0099163972519734 | 0.10282929099397098 |
| strategies/124_dollar_safe_haven_reversal.py | Codex | 7f58470906 | -0.3048480355441712 | -0.011821928575988072 |
| strategies/125_liquidity_risk_indicator_timing.py | Codex | ad594150a0 | 0.9392801869667401 | 0.092108548038075 |
| strategies/126_hmm_regime_allocation.py | Codex | ad594150a0 | 0.9421612030950982 | 0.10710208722331305 |
| strategies/129_ml_etf_trend_vs_oscillation.py | Codex | ad594150a0 | -0.1633159490129353 | -0.00856614180659876 |
| strategies/130_ml_risk_budget_allocation.py | Codex | ad594150a0 | 0.8004610178300348 | 0.05922868384241897 |
| strategies/148_decomposing_momentum.py | Codex | 2b23c5bbf7 | 0.8720364930091413 | 0.20916162811949102 |
| strategies/156_cta_replication_trend_factors.py | Codex | 2b23c5bbf7 | 0.44395267327406285 | 0.033093560604958716 |
| strategies/157_skew_path_dependent_vol.py | Codex | 498f8135d2 | 0.697460500588218 | 0.09150872373465474 |
| strategies/159_deep_momentum_market_trend.py | Codex | 498f8135d2 | 1.8878338181989713 | 0.019313841737686577 |
| strategies/165_retail_signal_failure.py | Codex | 2b23c5bbf7 | 0.43541324492441674 | 0.05075523692537032 |
| strategies/167_bayesian_cointegration_lowvol.py | Codex | 2b23c5bbf7 | -0.19791642317160424 | -0.01419780064634768 |
| strategies/179_crude_oil_news_sentiment.py | Codex | 2b23c5bbf7 | -0.3389139440305118 | -0.039653231117312226 |
| strategies/183_safe_asset_dollar_cycle.py | Codex | 2b23c5bbf7 | 0.1397995385765699 | 0.004613464128480738 |
| strategies/185_gold_month_seasonality.py | Codex | 2b23c5bbf7 | 0.7807274675730697 | 0.062351172766873075 |
| strategies/186_federal_holiday_effect.py | Codex | 92e75bea82 | -0.25031295695704375 | 0.0013228055598666355 |
| strategies/187_midterm_cycle.py | Codex | 92e75bea82 | 0.5276164995401547 | 0.03806831979746761 |
| strategies/188_opex_week.py | Codex | 92e75bea82 | 0.1560391651359337 | 0.009701616162450577 |
| strategies/191_shap_quantile_sector_allocation.py | Codex | 2b23c5bbf7 | 0.4696202287092069 | 0.05901278586658676 |
| strategies/192_downside_upside_rv_forecast.py | Codex | 2b23c5bbf7 | 0.5860331533877041 | 0.04621676334252989 |
| strategies/194_breaking_beta_defensive.py | Codex | 92e75bea82 | 0.768731258142307 | 0.10473819474909929 |
| strategies/195_century_reversals_volatility.py | Codex | 7f58470906 | 0.22717879797260454 | 0.015693851583082008 |
| strategies/197_tsmom_boundaries.py | Codex | 92e75bea82 | 0.8837194303501501 | 0.1331516279106011 |
| strategies/199_pca_stat_arb.py | Codex | 92e75bea82 | 0.2546514486659549 | 0.014364586479493235 |
| strategies/303_timing_moving_average_distance.py | Codex | 14eecadd0b | 0.4586740593305042 | 0.07523509455182698 |
| strategies/304_unintended_consequences_rebalancing.py | Codex | 2b23c5bbf7 | 0.8171238877819083 | 0.09014626117652913 |
| strategies/305_price_discovery_or_overreaction.py | Codex | 7f58470906 | -1.0653912895778344 | -0.027053797736966922 |
| strategies/307_common_idiosyncratic_quantile_factors.py | Codex | dfc503a96c | 0.09606686676127542 | 0.004632679142801965 |
| strategies/309_intangible_value_sixth_factor.py | Codex | 14eecadd0b | 0.6812915302713196 | 0.1077606874915078 |
| strategies/310_factor_beta_overnight_intraday.py | Codex | 2b23c5bbf7 | -1.614616855678312 | -0.027935906863817195 |
| strategies/312_cost_access_understanding_price.py | Codex | 2b23c5bbf7 | -0.9523822221196712 | -0.043607644709638405 |
| strategies/314_statistical_proxy_based_meanreverting.py | Codex | 2b23c5bbf7 | -0.4590101262621272 | -0.023263609116357276 |
| strategies/315_leveraged_funds_robust_replication.py | Codex | 2b23c5bbf7 | -3.893135170656959 | -0.011175684674659903 |
| strategies/316_hidden_cost_etf_investing.py | Codex | 14eecadd0b | -7.246553110896458 | -0.07780648632706832 |
| strategies/317_trendbreaks_persistence_closedend_fund.py | Codex | 2b23c5bbf7 | -0.014750860045009091 | -0.0003613816756397048 |
| strategies/323_information_transmission_across_tail.py | Codex | b43d901ca9 | 0.16070605776219252 | -0.006924251937111547 |
| strategies/324_volatility_conditions_weekend_effect.py | Codex | 3140bbcffa | -2.773981843436604 | -0.024112281949736802 |
| strategies/328_are_investors_really_moonstruck.py | Codex | b9dcd3d64e | 0.11378870437459518 | 0.006493183456579565 |
| strategies/336_saoftrl_novel_adaptive_algorithmic.py | Codex | 3140bbcffa | 0.9829686247958807 | 0.11517988658790612 |
| strategies/337_adaptive_eventdriven_labeling_multiscale.py | Codex | c5088ecf80 | 0.24586500117894552 | 0.022129935252036992 |
| strategies/338_optimal_portfolio_using_factor.py | Codex | 3140bbcffa | 0.666182564775382 | 0.0959349329372805 |
| strategies/339_dancing_dynamic_rebalancing_mechanism.py | Codex | 3140bbcffa | 0.8542589301354799 | 0.09623018976702165 |
| strategies/341_monitoring_attention_inflation_news.py | Codex | 3140bbcffa | 0.24566318826798883 | 0.014280753041128635 |
| strategies/350_continuous_timing_growth_defensive.py | Codex | 92e75bea82 | 0.9164036107251631 | 0.17281151630967062 |
| strategies/353_hold_the_dip.py | Codex | 92e75bea82 | 0.6959605518103854 | 0.09267657072001478 |
| strategies/354_strategy.py | Codex | 92e75bea82 | 0.17788436987988948 | 0.011436229915955831 |
| strategies/382_statistical_arbitrage_in_polish_equities.py | Codex | 4d91396b47 | 0.497781741038713 | 0.02277320791556292 |
| strategies/391_evolutionary_factor_searching_for_sparse.py | Codex | 4d91396b47 | 0.9944754757893223 | 0.28389911046975636 |
| strategies/395_signature_based_optimal_execution_for_st.py | Codex | 4d91396b47 | -0.9234169888488118 | -0.04284485041633779 |
| strategies/400_noxtrader_lstm_based_stock_return_moment.py | Codex | 4d91396b47 | -0.4419440960750732 | -0.1977945613653347 |
| strategies/432_revisiting_equity_bond_diversification_u.py | Codex | 4d91396b47 | 0.40210518871874146 | 0.029671517239883816 |
