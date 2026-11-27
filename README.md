# Quantative-Strategy

**研究用途，非投资建议（Research only, not investment advice）。** 本仓库任何策略进入投资系统前须另经 Gatekeeper 审查。不含下单或券商接口代码，只用免费公开数据，不含第三方原始价格文件。自 2026-11-27 起新收录策略在入库前的审查为作者（Codex）自审，不是独立第三方审查；此前已收录策略曾经过非作者审查。

每个策略一个文件：`strategies/<编号>_<名称>.py`，一条命令运行：`python strategies/<文件>.py`（依赖见 requirements.txt）。文件头写明数据源与抓取时间、样本期、IS/OOS 划分、成本与滑点、信号滞后、复权口径、幸存者偏差说明、尝试的参数组合数，以及在记录的 commit 上实际运行得到的关键数字（backtest reproduced）。收录条件：真实运行、确有交易、无同收盘成交前视、Codex 自审 PASS（进入本仓库前的审查为作者自审，不是独立第三方审查）、单文件复跑与审查时结果一致。

本表由文件实际内容生成（2026-11-27 11:35 Europe/Oslo），本表由文件实际内容生成（2026-11-27 11:35 Europe/Oslo），共 46 个。
。

| 文件 | 作者 | 代码 commit | 全样本 Sharpe | 全样本 CAGR |
|---|---|---|---|---|
| strategies/101_correlated_stress_reversal.py | Codex | 10f8238200 | -0.4528647975895032 | -0.040200837249976584 |
| strategies/102_alt_etf_short_term_reversal.py | Codex | 3140bbcffa | -4.850826308066646 | -0.1832096548612029 |
| strategies/103_breaking_bad_trends.py | Codex | ed11995234 | 0.24156559048879273 | 0.011219751158929592 |
| strategies/105_factor_momentum_style_etf.py | Codex | e45e81ae3a | 0.7670643508346912 | 0.11463689309321756 |
| strategies/106_low_ivol_low_max.py | Codex | 10f8238200 | 0.8198976625975141 | 0.1152196820035738 |
| strategies/110_cointegration_etf_pairs.py | Codex | ad594150a0 | -1.9086957510381422 | -0.0066972690980858784 |
| strategies/112_regime_switching_pairs.py | Codex | ad594150a0 | -0.2205187803621582 | -0.005553293487243871 |
| strategies/122_stock_bond_correlation_regime.py | Codex | ad594150a0 | 1.0099163972519734 | 0.10282929099397098 |
| strategies/124_dollar_safe_haven_reversal.py | Codex | 10f8238200 | -0.278935429402221 | -0.010816713521787924 |
| strategies/125_liquidity_risk_indicator_timing.py | Codex | ad594150a0 | 0.9392801869667401 | 0.092108548038075 |
| strategies/126_hmm_regime_allocation.py | Codex | ad594150a0 | 0.9421612030950982 | 0.10710208722331305 |
| strategies/129_ml_etf_trend_vs_oscillation.py | Codex | ad594150a0 | -0.1633159490129353 | -0.00856614180659876 |
| strategies/148_decomposing_momentum.py | Codex | 2b23c5bbf7 | 0.8720364930091413 | 0.20916162811949102 |
| strategies/157_skew_path_dependent_vol.py | Codex | 498f8135d2 | 0.697460500588218 | 0.09150872373465474 |
| strategies/165_retail_signal_failure.py | Codex | 2b23c5bbf7 | 0.43541324492441674 | 0.05075523692537032 |
| strategies/167_bayesian_cointegration_lowvol.py | Codex | 2b23c5bbf7 | -0.19791642317160424 | -0.01419780064634768 |
| strategies/179_crude_oil_news_sentiment.py | Codex | 10f8238200 | -0.2457468465858118 | -0.030724431650478 |
| strategies/183_safe_asset_dollar_cycle.py | Codex | 2b23c5bbf7 | 0.1397995385765699 | 0.004613464128480738 |
| strategies/185_gold_month_seasonality.py | Codex | 2b23c5bbf7 | 0.7807274675730697 | 0.062351172766873075 |
| strategies/186_federal_holiday_effect.py | Codex | e0984d15ab | -0.2039764296266909 | 0.003356979677578531 |
| strategies/187_midterm_cycle.py | Codex | 92e75bea82 | 0.5276164995401547 | 0.03806831979746761 |
| strategies/191_shap_quantile_sector_allocation.py | Codex | 2b23c5bbf7 | 0.4696202287092069 | 0.05901278586658676 |
| strategies/192_downside_upside_rv_forecast.py | Codex | 2b23c5bbf7 | 0.5860331533877041 | 0.04621676334252989 |
| strategies/194_breaking_beta_defensive.py | Codex | 10f8238200 | 0.7668802709220355 | 0.10356713253459682 |
| strategies/195_century_reversals_volatility.py | Codex | 10f8238200 | 0.26694943743596733 | 0.018967890521761177 |
| strategies/199_pca_stat_arb.py | Codex | 92e75bea82 | 0.2546514486659549 | 0.014364586479493235 |
| strategies/303_timing_moving_average_distance.py | Codex | 14eecadd0b | 0.4586740593305042 | 0.07523509455182698 |
| strategies/304_unintended_consequences_rebalancing.py | Codex | 2b23c5bbf7 | 0.8171238877819083 | 0.09014626117652913 |
| strategies/305_price_discovery_or_overreaction.py | Codex | 7f58470906 | -1.0653912895778344 | -0.027053797736966922 |
| strategies/307_common_idiosyncratic_quantile_factors.py | Codex | dfc503a96c | 0.09606686676127542 | 0.004632679142801965 |
| strategies/312_cost_access_understanding_price.py | Codex | 2b23c5bbf7 | -0.9523822221196712 | -0.043607644709638405 |
| strategies/314_statistical_proxy_based_meanreverting.py | Codex | 2b23c5bbf7 | -0.4590101262621272 | -0.023263609116357276 |
| strategies/317_trendbreaks_persistence_closedend_fund.py | Codex | 2b23c5bbf7 | -0.014750860045009091 | -0.0003613816756397048 |
| strategies/323_information_transmission_across_tail.py | Codex | 2585fac6f6 | 0.18281176874662353 | 0.0015919214787900593 |
| strategies/328_are_investors_really_moonstruck.py | Codex | b9dcd3d64e | 0.11378870437459518 | 0.006493183456579565 |
| strategies/336_saoftrl_novel_adaptive_algorithmic.py | Codex | 3140bbcffa | 0.9829686247958807 | 0.11517988658790612 |
| strategies/337_adaptive_eventdriven_labeling_multiscale.py | Codex | c5088ecf80 | 0.24586500117894552 | 0.022129935252036992 |
| strategies/338_optimal_portfolio_using_factor.py | Codex | 10f8238200 | 0.6617808751579951 | 0.09403463821519598 |
| strategies/339_dancing_dynamic_rebalancing_mechanism.py | Codex | 3140bbcffa | 0.8542589301354799 | 0.09623018976702165 |
| strategies/350_continuous_timing_growth_defensive.py | Codex | 9102661980 | 0.9164034821773692 | 0.1728115110163706 |
| strategies/382_statistical_arbitrage_in_polish_equities.py | Codex | ea97bbcbc9 | 0.500628613686733 | 0.023135888068235744 |
| strategies/438_the_mathematics_of_heuristic_portfolio_o.py | Codex | f9ce9ed5a4 | 0.573210240682426 | 0.0343462522184903 |
| strategies/463_simple_dynamic_stock_bond_gold_portfolio.py | Codex | f9ce9ed5a4 | 0.9859305218392249 | 0.0686041912862605 |
| strategies/478_portfolio_allocation_under_heterogeneous.py | Codex | e0984d15ab | 0.9428916934150752 | 0.10526210296784755 |
| strategies/497_interpretable_allocation_and_model_predi.py | Codex | e0984d15ab | 0.7229355302682273 | 0.10847871566869949 |
| strategies/560_drawdown_risk_beyond_brownian_motion_a_m.py | Codex | dabba95798 | 0.6719090607841077 | 0.08084054202667357 |

## 已收录后的深审发现

已推送到 origin/main 的策略不会改写历史提交。若之后的深审发现前视、数据或数字问题，文件保留为当时已发布的版本，并在新提交中列入 [DEEP_REVIEW_FINDINGS.md](DEEP_REVIEW_FINDINGS.md)。修复（如有）也只出现在新提交中。

| 文件 | 最新自审 | 问题类别 |
|---|---|---|
| strategies/546_generalizing_markowitz_portfolio_optimiz.py | FAIL | see review |
| strategies/547_generalizing_markowitz_portfolio_optimiz.py | FAIL | 最终买入数量可由成交日收盘价决定 |
| strategies/544_special_markowitz_thermodynamic_formalis.py | FAIL | 成交日收盘价决定部分实际买入份额 |
| strategies/501_nested_clustered_optimization_is_one_end.py | FAIL | 收盘价决定实际买入份额 |
| strategies/451_a_declining_cvar_glidepath_framework_for.py | FAIL | 资金不足时，最终买入份额由成交日收盘价反算 |
| strategies/510_some_general_results_on_risk_budgeting_p.py | FAIL | 实际买入数量由成交日收盘价决定 |
| strategies/452_simple_dynamic_stock_bond_gold_portfolio.py | FAIL | 现金不足时，成交日收盘价决定实际买入份额 |
| strategies/494_wasserstein_penalisation_as_an_endogenou.py | FAIL | look-ahead / same-close fill |
| strategies/533_special_markowitz_thermodynamic_formalis.py | FAIL | see review |
| strategies/514_risk_limited_asset_allocation_with_a_bud.py | FAIL | see review |
| strategies/484_nonlinear_shrinkage_for_higher_moments_e.py | FAIL | 实际买入份额可由成交日收盘价决定; 股票池是事后固定的幸存者样本 |
| strategies/476_conditional_value_at_risk_under_reward_p.py | FAIL | 成交日收盘价参与决定实际买入数量 |
| strategies/432_revisiting_equity_bond_diversification_u.py | FAIL | 共用模拟器在成交收盘重新确定买入份额; 收盘模型使用事后复权快照与简化费用 |
| strategies/505_competitive_optimal_portfolio_selection.py | FAIL | 成交日收盘价决定部分实际买入份额 |
| strategies/539_schur_complementary_allocation_a_unifica.py | FAIL | 现金不足时用成交日收盘价重算买入份额 |
| strategies/449_the_mathematics_of_heuristic_portfolio_o.py | FAIL | see review |
| strategies/483_wasserstein_penalisation_as_an_endogenou.py | FAIL | 最终买入份额由成交日收盘价决定 |
| strategies/503_risk_limited_asset_allocation_with_a_bud.py | FAIL | 收盘后重定买入份额 |
| strategies/431_mitigating_return_drag_and_regime_whipsa.py | FAIL | 实际买入份额会按成交收盘价缩减 |
| strategies/532_robust_enhanced_index_tracking_portfolio.py | FAIL | 资金不足时用成交日收盘价改写买入股数; 事后固定选池造成幸存者偏差和成分选择偏差 |
| strategies/534_portfolio_diversification_and_concentrat.py | FAIL | 最终买入数量依赖成交日收盘价 |
| strategies/509_beyond_de_prado_and_cotton_hierarchical.py | FAIL | see review |
| strategies/520_beyond_de_prado_and_cotton_hierarchical.py | FAIL | 实际买入份额依赖成交日收盘价 |
| strategies/548_entropic_value_at_risk_portfolio_optimiz.py | FAIL | see review |
| strategies/395_signature_based_optimal_execution_for_st.py | FAIL | 资金约束在同一收盘价重定成交股数 |
| strategies/450_a_declining_cvar_glidepath_framework_for.py | FAIL | 成交日收盘价改变实际买入份额 |
| strategies/551_squeezed_covariance_matrix_estimation_an.py | FAIL | 成交时用收盘价缩减买量; 事后固定股票池 |
| strategies/481_optimal_portfolio_construction_a_reinfor.py | FAIL | 现金不足时以成交日收盘价决定实际买量 |
| strategies/500_dynamic_portfolio_optimization_under_cva.py | FAIL | 收盘价参与确定同一收盘的实际买量 |
| strategies/541_entropy_regularization_under_no_shorting.py | FAIL | 最终买入份额使用成交日收盘价 |
| strategies/459_entropic_value_at_risk_parity_for_temper.py | FAIL | 实际买入份额由成交日收盘价决定 |
| strategies/442_mitigating_return_drag_and_regime_whipsa.py | FAIL | look-ahead / same-close fill |
| strategies/022_sector_residual_statarb.py | FAIL | 结算数量仍可由当日收盘价决定 |
| strategies/465_semivariance_and_semivolatility_in_portf.py | FAIL | 复权快照限制成交与数据时点的可追溯性 |
| strategies/113_kalman_garch_pairs.py | FAIL | see review |
| strategies/315_leveraged_funds_robust_replication.py | FAIL | 资金不足时用成交日收盘价改变实际成交份额 |
| strategies/480_topological_risk_parity.py | FAIL | 实际买入份额可由成交日收盘价决定 |
| strategies/341_monitoring_attention_inflation_news.py | FAIL | 现金不足时，实际买入股数取决于成交日收盘价 |
| strategies/545_portfolio_diversification_and_concentrat.py | FAIL | 实际买入份额依赖成交日收盘价 |
| strategies/552_a_novel_robust_mixed_integer_linear_prog.py | FAIL | 现金不足时按成交日收盘改写实际买入股数; 固定候选池存在幸存者偏差及事后选样; 模拟成交与实际市场价格、容量和公司行动有差距 |
| strategies/543_robust_enhanced_index_tracking_portfolio.py | FAIL | 最终买入量依赖成交日收盘价; 固定股票池存在事后选股与幸存者偏差 |
| strategies/461_breaking_the_trend_how_to_avoid_cherry_p.py | FAIL | 成交日收盘价决定了实际成交数量; 资金与成交模型只覆盖日频 |
| strategies/487_conditional_value_at_risk_under_reward_p.py | FAIL | see review |
| strategies/538_mirror_descent_algorithms_for_risk_budge.py | FAIL | see review |
| strategies/472_breaking_the_trend_how_to_avoid_cherry_p.py | FAIL | see review |
| strategies/467_an_entropic_factor_model_for_robust_port.py | FAIL | 实际买入量依赖成交收盘价; 固定股票池存在幸存者与事后筛选偏差; 复权成交与资金约束是简化模型 |
| strategies/491_topological_risk_parity.py | FAIL | 成交数量依赖成交日收盘价 |
| strategies/188_opex_week.py | FAIL | 周末收盘买入的实际份额由当日收盘价决定; 固定成本和日线成交假设限制执行结论 |
| strategies/443_revisiting_equity_bond_diversification_u.py | FAIL | 最终买入量可由成交日收盘价决定; 动态配置缺少实际成交证据 |
| strategies/159_deep_momentum_market_trend.py | FAIL | 开盘定量输入依赖成交日收盘 |
| strategies/109_factor_etf_vol_matched.py | FAIL | see review |
| strategies/490_noise_proofing_universal_portfolio_shrin.py | FAIL | 成交日收盘价决定实际买入量 |
| strategies/456_an_entropic_factor_model_for_robust_port.py | FAIL | 成交数量使用执行日收盘价 |
| strategies/537_hierarchical_minimum_variance_portfolios.py | FAIL | see review |
| strategies/511_dynamic_portfolio_optimization_under_cva.py | FAIL | 成交日收盘价参与确定同一收盘的交易量 |
| strategies/400_noxtrader_lstm_based_stock_return_moment.py | FAIL | see review |
| strategies/504_exploratory_mean_variance_with_jumps_an.py | FAIL | 成交数量使用成交日收盘价 |
| strategies/458_two_sides_of_schur_damping_high_dimensio.py | FAIL | see review |
| strategies/454_semivariance_and_semivolatility_in_portf.py | FAIL | 成交日收盘后倒算成交量 |
| strategies/492_optimal_portfolio_construction_a_reinfor.py | FAIL | 成交数量依赖成交日收盘价 |
| strategies/549_drawdown_risk_beyond_brownian_motion_a_m.py | FAIL | 换仓买入金额由成交日收盘值决定 |
| strategies/391_evolutionary_factor_searching_for_sparse.py | FAIL | 成交数量依赖成交日收盘价; 股票池有事后成分选择和幸存者偏差 |
| strategies/107_illiquidity_premium_amihud.py | FAIL | 成交数量规则不通过; 股票池存在幸存者和事后选样偏差 |
| strategies/309_intangible_value_sixth_factor.py | FAIL | 以成交日收盘价倒算成交量; 成本有披露，成交摩擦仅由固定费率近似 |
| strategies/130_ml_risk_budget_allocation.py | PASS | see review |
| strategies/448_continuous_cash_overlay_filters_for_a_st.py | FAIL | see review |
| strategies/468_cleaning_correlation_matrices_with_exter.py | FAIL | 成交数量依赖成交日收盘 |
| strategies/324_volatility_conditions_weekend_effect.py | FAIL | 开仓数量隐含使用成交日收盘价 |
| strategies/466_hierarchical_risk_parity_for_portfolio_a.py | FAIL | 成交量依赖成交日收盘价 |
| strategies/455_hierarchical_risk_parity_for_portfolio_a.py | FAIL | see review |
| strategies/460_understanding_the_long_only_minimum_vari.py | FAIL | 成交定量使用执行日收盘信息 |
| strategies/469_two_sides_of_schur_damping_high_dimensio.py | FAIL | 成交数量依赖成交日收盘价 |
| strategies/474_multiperiod_bond_portfolio_optimization.py | FAIL | see review |
| strategies/512_nested_clustered_optimization_is_one_end.py | FAIL | 用成交日收盘价反推同一收盘的成交规模; 全市场缺行可能改变信号及成交日期 |
| strategies/197_tsmom_boundaries.py | FAIL | 开盘换仓金额依赖当日收盘字段 |
| strategies/316_hidden_cost_etf_investing.py | FAIL | 收盘建仓和调仓数量依赖成交日收盘价 |
| strategies/535_shrinkage_estimators_for_mean_and_covari.py | FAIL | 成交日收盘参与定量 |
| strategies/553_large_scale_portfolio_optimization_probl.py | FAIL | see review |
| strategies/310_factor_beta_overnight_intraday.py | FAIL | 收盘成交使用同一收盘价确定股数; 成本已计入，拍卖成交与做空摩擦仍是简化假设 |
| strategies/353_hold_the_dip.py | FAIL | see review |
| strategies/445_dynamic_scenario_based_portfolio_optimiz.py | FAIL | 成交数量使用成交日收盘信息 |
| strategies/457_cleaning_correlation_matrices_with_exter.py | FAIL | 成交数量使用成交日收盘信息 |
| strategies/482_dynamic_inclusion_and_bounded_multi_fact.py | FAIL | 调仓用成交日收盘价倒算数量 |
| strategies/434_dynamic_scenario_based_portfolio_optimiz.py | FAIL | 调仓数量使用成交日收盘价; 固定 ETF 名单由事后公开的论文提供 |
| strategies/528_schur_complementary_allocation_a_unifica.py | FAIL | look-ahead / same-close fill |
| strategies/354_strategy.py | FAIL | 买入规模依赖成交收盘价 |
| strategies/470_entropic_value_at_risk_parity_for_temper.py | FAIL | look-ahead / same-close fill |
| strategies/522_asset_allocation_using_a_markov_process.py | FAIL | 实际买入份额取决于成交日收盘价 |
| strategies/479_noise_proofing_universal_portfolio_shrin.py | FAIL | 成交日收盘价反推成交数量和名义金额 |
| strategies/156_cta_replication_trend_factors.py | FAIL | see review |
