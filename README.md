# Quantative-Strategy

**研究用途，非投资建议（Research only, not investment advice）。** 本仓库任何策略进入投资系统前须另经 Gatekeeper 审查。不含下单或券商接口代码，只用免费公开数据，不含第三方原始价格文件。自 2026-11-27 起新收录策略在入库前的审查为作者（Codex）自审，不是独立第三方审查；此前已收录策略曾经过非作者审查。

每个策略一个文件：`strategies/<编号>_<名称>.py`，一条命令运行：`python strategies/<文件>.py`（依赖见 requirements.txt）。文件头写明数据源与抓取时间、样本期、IS/OOS 划分、成本与滑点、信号滞后、复权口径、幸存者偏差说明、尝试的参数组合数，以及在记录的 commit 上实际运行得到的关键数字（backtest reproduced）。收录条件：真实运行、确有交易、无同收盘成交前视、Codex 自审 PASS（进入本仓库前的审查为作者自审，不是独立第三方审查）、单文件复跑与审查时结果一致。

本表由文件实际内容生成（2026-11-27 21:29 Europe/Oslo），见下表（accepted + origin accepted kept）。

| 文件 | 作者 | 代码 commit | 全样本 Sharpe | 全样本 CAGR |
|---|---|---|---|---|
| strategies/514_risk_limited_asset_allocation_with_a_bud.py | Codex | 5df80ea472 | 0.6608472010901224 | 0.052573156425622214 |
| strategies/538_mirror_descent_algorithms_for_risk_budge.py | Codex | 5df80ea472 | 1.0791687893517397 | 0.09927088301658604 |
| strategies/455_hierarchical_risk_parity_for_portfolio_a.py | Codex | fc4032f1dc | 0.25327656512931895 | 0.03190383664914931 |
| strategies/457_cleaning_correlation_matrices_with_exter.py | Codex | fc4032f1dc | 0.9099819827776527 | 0.02626628471672099 |
| strategies/472_breaking_the_trend_how_to_avoid_cherry_p.py | Codex | fc4032f1dc | 0.8201357043332782 | 0.0416970442905884 |
| strategies/480_topological_risk_parity.py | Codex | fc4032f1dc | 0.8096455143944183 | 0.142227290925115 |
| strategies/535_shrinkage_estimators_for_mean_and_covari.py | Codex | fc4032f1dc | 0.7532230751823523 | 0.09410913670840992 |
| strategies/543_robust_enhanced_index_tracking_portfolio.py | Codex | fc4032f1dc | 1.2684719847129582 | 0.23921713918638687 |
| strategies/544_special_markowitz_thermodynamic_formalis.py | Codex | fc4032f1dc | 0.7640949051797069 | 0.12152354741167992 |
| strategies/546_generalizing_markowitz_portfolio_optimiz.py | Codex | fc4032f1dc | 0.8113127385091321 | 0.15251232621073885 |
| strategies/547_generalizing_markowitz_portfolio_optimiz.py | Codex | fc4032f1dc | 0.8075847136548383 | 0.14060266781900777 |
| strategies/548_entropic_value_at_risk_portfolio_optimiz.py | Codex | fc4032f1dc | 0.6055526110249723 | 0.07710121425323506 |
| strategies/100_etf_asset_momentum_corr_hedge.py | Codex | c71364ad99 | 0.590749534227566 | 0.07218221248997536 |
| strategies/101_correlated_stress_reversal.py | Codex | 10f8238200 | -0.4528647975895032 | -0.040200837249976584 |
| strategies/102_alt_etf_short_term_reversal.py | Codex | 3140bbcffa | -4.850826308066646 | -0.1832096548612029 |
| strategies/103_breaking_bad_trends.py | Codex | ed11995234 | 0.24156559048879273 | 0.011219751158929592 |
| strategies/105_factor_momentum_style_etf.py | Codex | e45e81ae3a | 0.7670643508346912 | 0.11463689309321756 |
| strategies/106_low_ivol_low_max.py | Codex | 10f8238200 | 0.8198976625975141 | 0.1152196820035738 |
| strategies/107_illiquidity_premium_amihud.py | Codex | da9754ee51 | 0.7839042842398569 | 0.15778681683311424 |
| strategies/110_cointegration_etf_pairs.py | Codex | ad594150a0 | -1.9086957510381422 | -0.0066972690980858784 |
| strategies/112_regime_switching_pairs.py | Codex | ad594150a0 | -0.2205187803621582 | -0.005553293487243871 |
| strategies/122_stock_bond_correlation_regime.py | Codex | ad594150a0 | 1.0099163972519734 | 0.10282929099397098 |
| strategies/124_dollar_safe_haven_reversal.py | Codex | 10f8238200 | -0.278935429402221 | -0.010816713521787924 |
| strategies/125_liquidity_risk_indicator_timing.py | Codex | ad594150a0 | 0.9392801869667401 | 0.092108548038075 |
| strategies/126_hmm_regime_allocation.py | Codex | ad594150a0 | 0.9421612030950982 | 0.10710208722331305 |
| strategies/129_ml_etf_trend_vs_oscillation.py | Codex | ad594150a0 | -0.1633159490129353 | -0.00856614180659876 |
| strategies/130_ml_risk_budget_allocation.py | Codex | ad594150a0 | 0.8004610178300348 | 0.05922868384241897 |
| strategies/148_decomposing_momentum.py | Codex | 2b23c5bbf7 | 0.8720364930091413 | 0.20916162811949102 |
| strategies/157_skew_path_dependent_vol.py | Codex | 498f8135d2 | 0.697460500588218 | 0.09150872373465474 |
| strategies/165_retail_signal_failure.py | Codex | 2b23c5bbf7 | 0.43541324492441674 | 0.05075523692537032 |
| strategies/167_bayesian_cointegration_lowvol.py | Codex | 2b23c5bbf7 | -0.19791642317160424 | -0.01419780064634768 |
| strategies/179_crude_oil_news_sentiment.py | Codex | 10f8238200 | -0.2457468465858118 | -0.030724431650478 |
| strategies/183_safe_asset_dollar_cycle.py | Codex | 2b23c5bbf7 | 0.1397995385765699 | 0.004613464128480738 |
| strategies/185_gold_month_seasonality.py | Codex | 2b23c5bbf7 | 0.7807274675730697 | 0.062351172766873075 |
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
| strategies/309_intangible_value_sixth_factor.py | Codex | da9754ee51 | 0.6804882022326363 | 0.10265342924479492 |
| strategies/312_cost_access_understanding_price.py | Codex | 2b23c5bbf7 | -0.9523822221196712 | -0.043607644709638405 |
| strategies/314_statistical_proxy_based_meanreverting.py | Codex | 2b23c5bbf7 | -0.4590101262621272 | -0.023263609116357276 |
| strategies/316_hidden_cost_etf_investing.py | Codex | da9754ee51 | -11.78031511310225 | -0.11162230404518114 |
| strategies/317_trendbreaks_persistence_closedend_fund.py | Codex | 2b23c5bbf7 | -0.014750860045009091 | -0.0003613816756397048 |
| strategies/323_information_transmission_across_tail.py | Codex | 2585fac6f6 | 0.18281176874662353 | 0.0015919214787900593 |
| strategies/328_are_investors_really_moonstruck.py | Codex | b9dcd3d64e | 0.11378870437459518 | 0.006493183456579565 |
| strategies/336_saoftrl_novel_adaptive_algorithmic.py | Codex | 3140bbcffa | 0.9829686247958807 | 0.11517988658790612 |
| strategies/337_adaptive_eventdriven_labeling_multiscale.py | Codex | c5088ecf80 | 0.24586500117894552 | 0.022129935252036992 |
| strategies/338_optimal_portfolio_using_factor.py | Codex | 10f8238200 | 0.6617808751579951 | 0.09403463821519598 |
| strategies/339_dancing_dynamic_rebalancing_mechanism.py | Codex | 3140bbcffa | 0.8542589301354799 | 0.09623018976702165 |
| strategies/341_monitoring_attention_inflation_news.py | Codex | 0ad743fb19 | 0.24769001759345974 | 0.014321528863530775 |
| strategies/350_continuous_timing_growth_defensive.py | Codex | 9102661980 | 0.9164034821773692 | 0.1728115110163706 |
| strategies/354_strategy.py | Codex | 4dbc1df0d2 | 0.1758253313432972 | 0.010438970679190307 |
| strategies/382_statistical_arbitrage_in_polish_equities.py | Codex | ea97bbcbc9 | 0.500628613686733 | 0.023135888068235744 |
| strategies/391_evolutionary_factor_searching_for_sparse.py | Codex | da9754ee51 | 0.9903968721257259 | 0.25587424645442103 |
| strategies/438_the_mathematics_of_heuristic_portfolio_o.py | Codex | f9ce9ed5a4 | 0.573210240682426 | 0.0343462522184903 |
| strategies/463_simple_dynamic_stock_bond_gold_portfolio.py | Codex | f9ce9ed5a4 | 0.9859305218392249 | 0.0686041912862605 |
| strategies/466_hierarchical_risk_parity_for_portfolio_a.py | Codex | da9754ee51 | 0.25202646099041437 | 0.03140765257653144 |
| strategies/492_optimal_portfolio_construction_a_reinfor.py | Codex | da9754ee51 | 0.89927754240382 | 0.1296232752782156 |
| strategies/504_exploratory_mean_variance_with_jumps_an.py | Codex | da9754ee51 | 0.7431772376620975 | 0.05658096575636318 |
| strategies/511_dynamic_portfolio_optimization_under_cva.py | Codex | 4ea0dd4e23 | 0.678115570051477 | 0.05417596316252293 |
| strategies/549_drawdown_risk_beyond_brownian_motion_a_m.py | Codex | f0c72714f0 | 0.6380887390535539 | 0.07247489420330977 |
| strategies/560_drawdown_risk_beyond_brownian_motion_a_m.py | Codex | dabba95798 | 0.6719090607841077 | 0.08084054202667357 |


Note on 511: the self-defined backtest helper previously named `execute_order()` was renamed to `apply_fill` (official re-run + Codex self-review). `broker_check` is now False. It was never a live or broker API. The approximately 1e-7 change in reported full-sample Sharpe (0.6781159099371227 → 0.678115570051477) comes from a different Yahoo Finance data snapshot (fetched 2026-11-27T11:05:24+00:00 vs 2026-11-27T08:00:04+00:00), not from the rename.

Note on 354: Gatekeeper passed the look-ahead fix at a code-level spot-check, so this file is in the accepted table. The code commit is 4dbc1df0d2. META flags were not hand-edited. False-positive reasons cite the official fix file sha256 566c9d806e59a7f085e28541d72875418e3193facb5dcd6f5c57cd0d3f371a11 (314 lines) and are recorded in DEEP_REVIEW_FINDINGS.md, together with a known limitation on the sources of ANNOUNCED_CLOSURES announcement dates (including when Juneteenth was first announced).


## 已收录后的深审发现

已推送到 origin/main 的策略不会改写历史提交。若之后的深审发现前视、数据或数字问题，文件保留为当时已发布的版本，并在新提交中列入 [DEEP_REVIEW_FINDINGS.md](DEEP_REVIEW_FINDINGS.md)。修复（如有）也只出现在新提交中。

| 文件 | 最新自审 | 问题类别 |
|---|---|---|
| strategies/501_nested_clustered_optimization_is_one_end.py | INVALID | 无独立非作者审查; 样本外为回顾性切分; 依赖完整历史与事后复权快照，无法还原逐日发布版本 |
| strategies/451_a_declining_cvar_glidepath_framework_for.py | INVALID | 无独立非作者审查; OOS 独立性有限（事后切分）; 固定代理存在幸存者选择偏差; 快照无法还原逐日发布与复权版本 |
| strategies/510_some_general_results_on_risk_budgeting_p.py | INVALID | 无独立非作者审查; 固定存续 ETF 池（幸存者偏差）; 样本外为事后切分; 固定数量结算依赖现金余量; 快照无法还原逐日发布版本 |
| strategies/452_simple_dynamic_stock_bond_gold_portfolio.py | INVALID | 无独立非作者审查; 固定存续 ETF 池（幸存者偏差）; 样本外为事后切分; 快照无法还原 ^IRX 逐次发布时间; PARAM_COMBOS_TRIED=1，缺乏邻域稳健性证据 |
| strategies/494_wasserstein_penalisation_as_an_endogenou.py | INVALID | 无独立非作者审查; 固定存续的九只 ETF 池（事后选择、幸存者偏差）; 样本外为事后切分; 快照无法还原逐日发布版本; PARAM_COMBOS_TRIED=7（含六个样本内变体）; 固定成交假设与现金约束 |
| strategies/533_special_markowitz_thermodynamic_formalis.py | INVALID | 无独立非作者审查; 固定九只行业 ETF 池与完整历史条件（幸存者及事后选池偏差）; 样本外为事后日期留出; 快照无法还原逐日到达或修订时间; 资金缓冲有失效边界，成交成本为固定假设 |
| strategies/484_nonlinear_shrinkage_for_higher_moments_e.py | INVALID | 无独立非作者审查; 事后固定 50 只大盘股池（幸存者及事后选池偏差，早期 GOOGL 类别映射缺口）; 仅复用论文目标函数，未实现论文的高阶矩非线性收缩与历史成分/退市处理; 样本外为事后日期留出; 快照无法还原逐日到达或修订时间; 结算对现金和成交假设敏感（固定 10 bps 摩擦，资金不足则整批中止） |
| strategies/476_conditional_value_at_risk_under_reward_p.py | INVALID | 无独立非作者审查; 样本外为事后日期留出（阈值亦取自论文对整段训练期的估计）; 固定存续 ETF 池存在选择与幸存者偏差; 快照无法还原逐日到达或修订时间; 收盘成交与现金可用性采用简化模型（固定 5 bps 摩擦，资金不足则整批中止） |
| strategies/432_revisiting_equity_bond_diversification_u.py | INVALID | 无独立非作者审查; 事后选池及存续偏差（六只基金取自回测后发布的论文）; 动态配置缺乏成交证据（1258 次信号均选 AOK 100%，全期仅 1 笔建仓，结果等同 AOK 买入持有）; 样本外为事后日期留出; 快照无法还原逐日到达或修订时间; 成交及资金约束是简化模型（固定 3 bps 摩擦，资金不足则整批中止） |
| strategies/505_competitive_optimal_portfolio_selection.py | INVALID | 无独立非作者审查; 固定九只 ETF 池依赖存续与完整行情（无退市处置，偏差方向和幅度未知）; 样本外为历史日期留出（参数首次锁定与首次查看样本外结果的时间无记录）; PARAM_COMBOS_TRIED=7（含六个样本内变体）; 快照无法还原逐日到达或修订时间; 成交与资金约束采用简化模型（固定 8 bps 摩擦，同一收盘结算，资金不足则整批中止） |
| strategies/539_schur_complementary_allocation_a_unifica.py | INVALID | 无独立非作者审查; 固定资产池存在幸存者与事后选池偏差; 样本外为事后留出段; 历史快照缺少逐时点版本; 执行对成交价跳变和现金余量敏感 |
| strategies/449_the_mathematics_of_heuristic_portfolio_o.py | INVALID | 无独立非作者审查; 基金池存在幸存者偏差与事后选样（八只全期存续基金，要求全期完整行情）; 主样本外结果继承事后选定窗口的样本内模拟持仓（在样本内结束时择窗，整个样本早于论文首次提交，属回顾性研究）; 复权行情缺少逐时点数据版本，结算模型简化（分数复权份额、按收盘价结算、固定滑点）; 现金不足会终止整次模拟（据作者自审 REVIEW_codex_self_r5.md sha256 dde616b3ee809dfbe6da340f285c7fc1875ed692fb5016ed422777bd23309444，非独立审查） |
| strategies/483_wasserstein_penalisation_as_an_endogenou.py | INVALID | 无独立非作者审查; 历史留出的事前冻结证据不足（持仓和现金跨 IS/OOS 分界连续）; 固定池和复权快照限制外推（全期必须有报价，缺价则整次回测中止，无清盘退出处理）; 成交模型简化（成本与滑点各 5 bp，各边合计 10 bp，含首次建仓，现金结余零收益，同次卖出净款可用）（据作者自审 REVIEW_codex_self_r4.md sha256 a8649314de76536f588fe4c4fa0d01d9c78f99f5fd610da5b60479da26942cff，非独立审查） |
| strategies/503_risk_limited_asset_allocation_with_a_bud.py | INVALID | 无独立非作者审查; 收盘结算的资金与流动性假设 |
| strategies/431_mitigating_return_drag_and_regime_whipsa.py | INVALID | 无独立非作者审查; 模拟执行依赖收盘成交及资金假设 |
| strategies/532_robust_enhanced_index_tracking_portfolio.py | INVALID | 无独立非作者审查; 事后固定股票池及幸存者偏差 |
| strategies/534_portfolio_diversification_and_concentrat.py | INVALID | 无独立非作者审查; 固定份额的资金与成交假设 |
| strategies/509_beyond_de_prado_and_cotton_hierarchical.py | INVALID | 无独立非作者审查; 成交和资金模型对价差与流动性变化覆盖有限 |
| strategies/520_beyond_de_prado_and_cotton_hierarchical.py | INVALID | 无独立非作者审查; 成交模型依赖现金缓冲和固定摩擦 |
| strategies/395_signature_based_optimal_execution_for_st.py | INVALID | 无独立非作者审查; 事后选对与幸存者偏差; 成交成本与资金约束有明确的模拟边界 |
| strategies/450_a_declining_cvar_glidepath_framework_for.py | INVALID | 无独立非作者审查; 收盘成交与资金模型 |
| strategies/551_squeezed_covariance_matrix_estimation_an.py | INVALID | 无独立非作者审查; 事后固定股票池（幸存者偏差）; 论文表 3 权重晚于样本终点; 收盘结算和容量假设有限 |
| strategies/481_optimal_portfolio_construction_a_reinfor.py | INVALID | 无独立非作者审查; 固定现金缓冲与全额收盘成交假设限制执行稳健性 |
| strategies/500_dynamic_portfolio_optimization_under_cva.py | INVALID | 无独立非作者审查; 固定 ETF 池的幸存者及事后选取偏差; 历史留出缺少规格冻结证据（论文晚于回测样本）; 复权快照与现金结算近似 |
| strategies/541_entropy_regularization_under_no_shorting.py | INVALID | 无独立非作者审查; 历史行情与成交假设限制外推 |
| strategies/459_entropic_value_at_risk_parity_for_temper.py | INVALID | 无独立非作者审查; 固定资产池存在幸存者偏差和事后选择偏差; 固定成本和收盘结算限制了结果的执行含义 |
| strategies/497_interpretable_allocation_and_model_predi.py | INVALID | 无独立非作者审查; 固定研究池存在幸存者偏差 |
| strategies/442_mitigating_return_drag_and_regime_whipsa.py | INVALID | 无独立非作者审查; 历史样本外与论文参数评估期重叠（独立锁参留出证据缺失）; 全期完整报价造成事后条件（固定池来自论文所列存续基金，缺少退市、清算、替换和停牌处理）; 执行模型对资金缺口和交易摩擦作了简化（同批净额结算，跳空致资金不足则整次回测中止，无成交量、价差或冲击模型，剩余现金收益为零）（据作者自审 REVIEW_codex_self_r5.md sha256 35ba5fccaf3d3a8b7e86a33aa0255991b5d9950e1b910be6ecb23b91143bc109，非独立审查） |
| strategies/022_sector_residual_statarb.py | INVALID | 无独立非作者审查; 事后选池与退市处理缺口; 样本外为事后切分; 成交及资金模型简化 |
| strategies/465_semivariance_and_semivolatility_in_portf.py | INVALID | 无独立非作者审查; 成交和现金取决于简化假设 |
| strategies/113_kalman_garch_pairs.py | INVALID | 无独立非作者审查; 固定存续配对限制结果的外推; 成交与资金约束为简化模型 |
| strategies/315_leveraged_funds_robust_replication.py | INVALID | 无独立非作者审查; 事后选池限制结论的适用范围 |
| strategies/545_portfolio_diversification_and_concentrat.py | INVALID | 无独立非作者审查; 固定资产池有幸存者偏差和事后选样; 复权行情与成交模型限制（事后抓取的复权快照无法重建每个信号时点可见的行情版本，按可分割复权单位和收盘价成交）; 样本外表现弱于同池等权（OOS 年化约 7.51% 对 9.56%，最大回撤约 -29.46% 对 -23.13%）（据作者自审 REVIEW_codex_self_r4.md sha256 616ba2c4a05c3ab290d56399c55dd35a069b3a326c2ae4258d06a3e05ce722f5，非独立审查） |
| strategies/552_a_novel_robust_mixed_integer_linear_prog.py | INVALID | 无独立非作者审查; 固定股票池存在幸存者偏差; 行情可得时点与 OOS 解释有限; 成交与异常行情采用简化规则 |
| strategies/461_breaking_the_trend_how_to_avoid_cherry_p.py | INVALID | 无独立非作者审查; 固定基金名单带来幸存者偏差 |
| strategies/487_conditional_value_at_risk_under_reward_p.py | INVALID | 无独立非作者审查; 事后选股与幸存者偏差; 资金约束与成交假设 |
| strategies/467_an_entropic_factor_model_for_robust_port.py | INVALID | 无独立非作者审查; 股票池为晚于样本发表的论文所列幸存者名单，且要求全样本行情完整，带来样本内外的幸存者/选择偏差，偏差大小与方向未确定; 复权收盘价快照缺少逐时点的行情与公司行动版本（据作者自审 REVIEW_codex_self_r10.md sha256 f404d55666459338cda84f04e8d16b8f2dcda996c45a016d6b9e0aa5b55fe1bb；原执行时点缺陷标签已删除（原文见 DEEP_REVIEW_FINDINGS.md 阶段说明栏）：Codex xhigh 对当前代码的复核结论 VERDICT FIXED / SIZING PASS，信号在 t 日收盘、t+1 日收盘成交，数量按前一日收盘冻结；复核记录 VERIFY.md sha256 45b3931e1abecfbbb76f31e00e68ab92a2b1a60b5da90aac361eb5a579c779ed；均为作者方复核，非独立审查） |
| strategies/478_portfolio_allocation_under_heterogeneous.py | INVALID | 无独立非作者审查; 缺少官方行情快照; 开盘成交假设及资金约束 |
| strategies/491_topological_risk_parity.py | INVALID | 无独立非作者审查; 事后固定投资池; 成交与资金模型的适用范围 |
| strategies/188_opex_week.py | INVALID | 无独立非作者审查; 开收盘成交及资金假设偏强 |
| strategies/443_revisiting_equity_bond_diversification_u.py | INVALID | 无独立非作者审查; 动态配置缺少成交证据 |
| strategies/159_deep_momentum_market_trend.py | INVALID | 无独立非作者审查; 固定选取九只现存且长历史的基金，回测可运行取决于每只基金的整段行情（高；缺少回测起点冻结名单的历史记录）; 2022–2025 年 OOS 已被此前回测和评审查看，且大部分日期早于论文正式发表，属回溯评估; README 记录的本次行情快照未随审查副本交付，只能核对日志副本; 除息日即将分配计入可用现金，可能提前动用实际支付日前的应收股息或提前扣减空头应付分配（据作者自审 REVIEW_codex_self_r5.md sha256 572d1dc1ad458c17d4a7ad1a6a0f20ca5d17179d88809241a796495c545e7ee0，非独立审查） |
| strategies/109_factor_etf_vol_matched.py | INVALID | 无独立非作者审查; 基金选择与幸存者偏差（事后选定存续 ETF，要求全期报价完整，缺少选入记录及退市、清盘、替换规则）; OOS 的事前性无法追溯（基金名单、参数与切分缺少带日期的冻结记录）; 执行和现金缓冲的外推有限（按下一收盘参考价模拟复权小数份额，同次卖出净款可买入，现金不足则整次回测报错）（据作者自审 REVIEW_codex_self_r4.md sha256 73e264f3044e5c6c4996b5c6a8c39f5d41969d782ee98b2d6bdcbe0cf57a426b，非独立审查） |
| strategies/186_federal_holiday_effect.py | INVALID | 无独立非作者审查; 开盘换仓依赖成交与资金假设 |
| strategies/490_noise_proofing_universal_portfolio_shrin.py | INVALID | 无独立非作者审查; 固定存续 ETF 池产生幸存者与选择偏差 |
| strategies/456_an_entropic_factor_model_for_robust_port.py | INVALID | 无独立非作者审查; 成交与成本模型简化 |
| strategies/537_hierarchical_minimum_variance_portfolios.py | INVALID | 无独立非作者审查; 执行模型简化收盘流动性与现金约束 |
| strategies/400_noxtrader_lstm_based_stock_return_moment.py | INVALID | 无独立非作者审查; 固定股票池引入幸存者和事后选样偏差 |
| strategies/458_two_sides_of_schur_damping_high_dimensio.py | INVALID | 无独立非作者审查; 固定标的池存在幸存者偏差（两种配置共用事后选定的完整历史名单，比较只适用于这组标的）; 复权快照缺少历史可得性记录（仅记录整批抓取时间，无法重建逐日收盘价和公司行动调整的首次可得时间及后续修订）（据作者自审 REVIEW_codex_self_r5.md sha256 e3b0436f48e06cf0685564bc60ebbfba5f7069ed55d682334f95bc63b39cebd6，非独立审查） |
| strategies/454_semivariance_and_semivolatility_in_portf.py | INVALID | 无独立非作者审查; 资金与成交模型简化 |
| strategies/448_continuous_cash_overlay_filters_for_a_st.py | INVALID | 无独立非作者审查; 日线版本与成交模型有限制 |
| strategies/468_cleaning_correlation_matrices_with_exter.py | INVALID | 无独立非作者审查; 事后选定的存续标的池; 摩擦已披露，卖空及成交约束仍简化 |
| strategies/324_volatility_conditions_weekend_effect.py | INVALID | 无独立非作者审查; 审查副本缺少本次行情快照（README 所链接的缓存文件在审查副本中缺失）; 固定代理与样本外冻结证据不足（固定使用存续 ETF，缺少当时可选名单、退市处理规则）; ETF 代理实验与论文的检验口径有差异（论文用 Kenneth French 因子，本实现用 ETF 代理）（据作者自审 REVIEW_codex_self_r4.md sha256 196c76c02123cbe9e8eddfbd831f703592054c4a6691e9b9aac7a3705c33a1ff，非独立审查） |
| strategies/460_understanding_the_long_only_minimum_vari.py | INVALID | 无独立非作者审查; 成交模型与资金缓冲有现实限制 |
| strategies/469_two_sides_of_schur_damping_high_dimensio.py | INVALID | 无独立非作者审查; 固定基金池有幸存者偏差与事后选样偏差（高；全区间要求完整报价）; 历史留出的证据有限（IS 为 2010-01-04 至 2017-12-29，OOS 为 2018-01-02 至 2025-12-31，同一模拟完成后切分，持仓和滚动窗口跨界）; 结算模型在现金不足时直接中止（固定缓冲用于吸收执行日价格变动及费用）（据作者自审 REVIEW_codex_self_r4.md sha256 1a9aa270a10ce272624d79506fc76ef4d81072a3e31bd27de746fdc174ee7ca0，非独立审查） |
| strategies/474_multiperiod_bond_portfolio_optimization.py | INVALID | 无独立非作者审查; 规划成本与实际成交近似不同 |
| strategies/512_nested_clustered_optimization_is_one_end.py | INVALID | 无独立非作者审查; 固定资产池的幸存者与选择偏差（事后固定九只原始行业基金，各标的全期行情齐备，无纳入、退出或清算处理）; 样本外为事后切分的历史留出（论文提交日期晚于样本终点，仅在样本内评估六个替代参数）; 结算模型适用范围有限（可分割复权份额，所有份额假定按当日收盘价全量成交，同批卖出款可抵买入款）（据作者自审 REVIEW_codex_self_r4.md sha256 8e5217108ce3adf1115ac949ea9467d0e2f1a9353781f4462f24718ed0ad25d3，非独立审查） |
| strategies/197_tsmom_boundaries.py | INVALID | 无独立非作者审查; 开盘成交和股息现金的近似 |
| strategies/553_large_scale_portfolio_optimization_probl.py | INVALID | 无独立非作者审查; 候选池有幸存者和事后选样偏差; 同一代码跨股权类别; 样外仅为历史时间切分; 成交与资金模型简化; 快照无法还原逐日发布版本 |
| strategies/310_factor_beta_overnight_intraday.py | INVALID | 无独立非作者审查; OOS 的独立性不足; 费用与成交模型简化 |
| strategies/353_hold_the_dip.py | INVALID | 无独立非作者审查; 样本外区间零交易，切换规则在样本外未被检验; 事后选定 SPY 与 BIL，存在幸存者/市场选择偏差; 成交模型假设两腿按收盘价全额成交、固定滑点、卖出款立即可用（据作者自审 REVIEW_codex_self_r5.md sha256 61722bcf906b9de25f5d237aa3fbc3dfdc87f5cadbb6487c7c328aababd775f5；原执行时点缺陷标签已删除（原文见 DEEP_REVIEW_FINDINGS.md 阶段说明栏）：Codex xhigh 对当前代码的复核结论 VERDICT FIXED / SIZING PASS，信号在 t 日收盘、t+1 日收盘成交，数量按前一日收盘冻结；复核记录 VERIFY.md sha256 379609b00c79460dc6b8e088c26addee9698e6c28ad522a7ee08ccf2d010852e；均为作者方复核，非独立审查） |
| strategies/445_dynamic_scenario_based_portfolio_optimiz.py | INVALID | 无独立非作者审查; ETF 池有幸存者和事后选择偏差; 复权成交与资金处理是模型假设 |
| strategies/482_dynamic_inclusion_and_bounded_multi_fact.py | INVALID | 无独立非作者审查; 固定候选池有幸存者和事后选择偏差 |
| strategies/434_dynamic_scenario_based_portfolio_optimiz.py | INVALID | 无独立非作者审查; 事后资产池与回顾性样本切分 |
| strategies/528_schur_complementary_allocation_a_unifica.py | INVALID | 无独立非作者审查; 固定基金池存在幸存者偏差与资产选择偏差（事后选定五只基金，要求全期行情齐备，历史准入、退市清算和替换无规则）; 历史 OOS 的规格冻结缺少证据（2017-12-29 及以前为 IS，2018-01-02 起为 OOS，替代组仅运行到 IS 截止日）; 执行模型使用复权份额和简化摩擦（收盘价全额成交、可分割份额、固定比例滑点）（据作者自审 REVIEW_codex_self_r4.md sha256 b569e74bc7a001d17261515e41fd693233b4c893fa0119c1fc9f90dc10650ac5，非独立审查） |
| strategies/470_entropic_value_at_risk_parity_for_temper.py | INVALID | 无独立非作者审查; 固定基金池有幸存者偏差及事后选池偏差（高；完整样本报价是回测正常完成的前提）; 历史快照缺少逐日报价的发布时间、修订版本和逐时点复权记录; 现金不足会中止整个回测（用前收组合价值预留 10% 缓冲，无部分成交或延后）（据作者自审 REVIEW_codex_self_r4.md sha256 50298295177d17c81ea777422f5ca9627ab2b624abb71ab7dddc00727d998ea2，非独立审查） |
| strategies/522_asset_allocation_using_a_markov_process.py | INVALID | 无独立非作者审查; 收盘执行依赖固定滑点和现金缓冲 |
| strategies/479_noise_proofing_universal_portfolio_shrin.py | INVALID | 无独立非作者审查; 基金池存在事后选择和幸存者偏差; 模拟成交依赖复权日线和固定摩擦 |
| strategies/156_cta_replication_trend_factors.py | INVALID | 无独立非作者审查; 固定存续 ETF 池（十只事后选定、全期有报价的 ETF，缺少历史候选池、选入规则、退市和清算处理）; 样本外为回溯分段（缺少分界前冻结 ETF 池、参数及执行规则的记录）; 成交与融资模型简化（单边 3 bp 手续费、2 bp 滑点、年化 50 bp 借券费、零闲置现金利息，多头分配于除息日计入价值，空头分配于除息日扣现金，实际付款日现金流缺失）（据作者自审 REVIEW_codex_self_r4.md sha256 50b5a38da6192c0927b8071f58112314d1a7b4c5739b8a947704812eb792f727，非独立审查） |

## 待审新策略 / pending review (new)

These strategies are new research drafts. They have **not** been reviewed by Gatekeeper and are **not** in the accepted table. They must not be treated as passing. Any look-ahead remains INVALID with no metrics here.

| 文件 | 作者 | 说明 |
|---|---|---|
| strategies/574_objective_and_subjective_entropy_measure.py | Codex | not reviewed (pending) |
| strategies/577_objective_and_subjective_entropy_measure.py | Codex | not reviewed (pending) |
| strategies/578_are_three_matrices_all_you_need_to_beat.py | Codex | not reviewed (pending) |
| strategies/579_are_three_matrices_all_you_need_to_beat.py | Codex | not reviewed (pending) |
| strategies/580_anticipatory_portfolio_optimization.py | Codex | not reviewed (pending) |
| strategies/581_asymmetric_nonlinear_return_extrapolatio.py | Codex | not reviewed (pending) |
| strategies/582_the_geometry_of_higher_order_modern_port.py | Codex | not reviewed (pending) |
| strategies/583_anticipatory_portfolio_optimization.py | Codex | not reviewed (pending) |
| strategies/584_asymmetric_nonlinear_return_extrapolatio.py | Codex | not reviewed (pending) |
| strategies/585_the_geometry_of_higher_order_modern_port.py | Codex | not reviewed (pending) |
| strategies/586_bayesian_portfolio_optimization_by_predi.py | Codex | not reviewed (pending) |
| strategies/587_bayesian_portfolio_optimization_by_predi.py | Codex | not reviewed (pending) |
| strategies/588_signed_network_models_for_portfolio_opti.py | Codex | not reviewed (pending) |
| strategies/589_conformal_kelly_when_marginal_coverage_t.py | Codex | not reviewed (pending) |
| strategies/590_deeparies_adaptive_rebalancing_interval.py | Codex | not reviewed (pending) |
| strategies/609_regime_adaptive_continual_learning_for_p.py | Codex | not reviewed (pending) |
| strategies/617_path_portfolio_optimization_defect_lift.py | Codex | not reviewed (pending) |
| strategies/623_dynamic_causal_portfolio_choice_hedging.py | Codex | not reviewed (pending) |
| strategies/625_portfolio_choice_with_competing_precauti.py | Codex | not reviewed (pending) |
| strategies/649_dynamic_factor_models_with_forward_looki.py | Codex | not reviewed (pending) |
| strategies/653_risk_budgeting_portfolios_from_simulatio.py | Codex | not reviewed (pending) |
| strategies/656_risk_budgeting_portfolios_from_simulatio.py | Codex | not reviewed (pending) |
| strategies/658_market_adaptive_ratio_for_portfolio_mana.py | Codex | not reviewed (pending) |
| strategies/659_market_adaptive_ratio_for_portfolio_mana.py | Codex | not reviewed (pending) |
| strategies/667_dynamic_factor_analysis_of_price_movemen.py | Codex | not reviewed (pending) |
| strategies/668_dynamic_mean_variance_asset_allocation_i.py | Codex | not reviewed (pending) |
| strategies/671_high_dimensional_covariance_matrix_estim.py | Codex | not reviewed (pending) |
| strategies/672_randomized_signature_methods_in_optimal.py | Codex | not reviewed (pending) |
| strategies/673_a_new_behavioral_model_for_portfolio_sel.py | Codex | not reviewed (pending) |
| strategies/674_optimal_pair_trading_consumption_investm.py | Codex | not reviewed (pending) |
| strategies/675_randomized_signature_methods_in_optimal.py | Codex | not reviewed (pending) |
| strategies/676_a_new_behavioral_model_for_portfolio_sel.py | Codex | not reviewed (pending) |
| strategies/677_optimal_pair_trading_consumption_investm.py | Codex | not reviewed (pending) |
| strategies/681_dynamic_portfolio_selection_for_nonlinea.py | Codex | not reviewed (pending) |
| strategies/682_optimal_portfolio_with_ratio_type_period.py | Codex | not reviewed (pending) |
| strategies/692_dynamic_cvar_portfolio_construction_with.py | Codex | not reviewed (pending) |
| strategies/701_dependency_network_based_portfolio_desig.py | Codex | not reviewed (pending) |
| strategies/706_sparse_index_tracking_simultaneous_asset.py | Codex | not reviewed (pending) |
| strategies/708_doubly_robust_mean_cvar_portfolio.py | Codex | not reviewed (pending) |
| strategies/713_behavioral_probability_weighting_and_por.py | Codex | not reviewed (pending) |
| strategies/716_behavioral_probability_weighting_and_por.py | Codex | not reviewed (pending) |
| strategies/718_an_analytic_solution_for_asset_allocatio.py | Codex | not reviewed (pending) |
| strategies/719_an_analytic_solution_for_asset_allocatio.py | Codex | not reviewed (pending) |
| strategies/720_kendall_correlation_coefficients_for_por.py | Codex | not reviewed (pending) |
| strategies/721_conformal_predictive_portfolio_selection.py | Codex | not reviewed (pending) |
| strategies/802_downside_risk_reduction_using_regime_swi.py | Codex | not reviewed (pending); first defect: 无独立非作者审查; loader changed because of a checker result (see DEEP_REVIEW_FINDINGS.md) |
| strategies/804_at_risk_transformation_for_u_s_recession.py | Codex | not reviewed (pending); first defect: 无独立非作者审查; see DEEP_REVIEW_FINDINGS.md |
| strategies/803_cast_a_cross_asset_state_space_trading_s.py | Codex | not reviewed (pending); first defect: 无独立非作者审查; broker_check True is a base64 false positive (see DEEP_REVIEW_FINDINGS.md) |
| strategies/800_kellyboost_growth_optimal_portfolio_cons.py | Codex | not reviewed (pending); first defect: 无独立非作者审查; loader deviations from the fixed template listed in DEEP_REVIEW_FINDINGS.md |
| strategies/807_rethinking_portfolio_risk_forecasting_vo.py | Codex | not reviewed (pending); first defect: 无独立非作者审查; loader deviations from the fixed template listed in DEEP_REVIEW_FINDINGS.md |
| strategies/808_cost_sensitive_online_window_size_select.py | Codex | not reviewed (pending); first defect: 无独立非作者审查; loader deviations from the fixed template listed in DEEP_REVIEW_FINDINGS.md |
| strategies/811_deep_declarative_risk_budgeting_portfoli.py | Codex | not reviewed (pending); first defect: 无独立非作者审查; loader deviations from the fixed template listed in DEEP_REVIEW_FINDINGS.md |
| strategies/813_rachev_ratio_portfolio_optimization_with.py | Codex | not reviewed (pending); first defect: 无独立非作者审查; loader deviations from the fixed template listed in DEEP_REVIEW_FINDINGS.md |
| strategies/814_high_order_universal_portfolios.py | Codex | not reviewed (pending); first defect: 无独立非作者审查; loader deviations from the fixed template listed in DEEP_REVIEW_FINDINGS.md |
| strategies/816_mean_variance_portfolio_selection_in_lon.py | Codex | not reviewed (pending); first defect: 无独立非作者审查; loader deviations from the fixed template listed in DEEP_REVIEW_FINDINGS.md |
