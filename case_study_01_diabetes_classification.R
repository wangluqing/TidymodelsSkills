# 案例篇
# tidymodels 包
# 健康科技

rm(list = ls())
# R包
library(tidyverse)
library(tidymodels)
library(themis)
library(doParallel)
library(gtsummary) 
library(gt)
library(bonsai) 
library(discrim)
library(finetune)
library(patchwork)
library(vip)
library(DALEXtra) 

# 1 动机 ----
# 为什么做这个事情？
# 1）糖尿病的影响，死亡和疾病，国家，年龄和性别
# 2）糖尿病的类型划分，占大多数的Type2类型，病程早期进行发现和管理，很大程度上是可以预防的
# 甚至是可逆的

# 做个什么事情？
# Type2 糖尿病 早期的诊断和识别，具有重大意义和科研价值

# 诊断程序在糖尿病保健管理中起着重要作用
# 以便于可靠地预防疾病
# 我们要兼顾模型的可解释性和模型的精度两个方面
# 我们需要解决的问题点
# 第一点：我们要从数据中了解风险因素与糖尿病的关系
# 第二点：我们要对新的数据做样本外预测和分析

# association between available features (risk factors) and diabetes
# tidymodels 框架和 使用不同的ML算法
# 针对kaggle的糖尿病数据集

# 2 数据和模型策略 ----
# 数据概览
# 10万个样本
# 1个目标和8个潜在风险因素
# 数据拆分 75%做训练集


# 2.1 数据集概览 ----

diabetes_df <- read_csv('data/diabetes_prediction_dataset.csv') %>%
  mutate(diabetes = as.factor(diabetes)) %>%
  mutate(bmi=as.numeric(case_when(bmi!='N/A' ~ bmi,
                                  TRUE ~ NA_real_))) %>%
  mutate_if(is.character, as.factor) 


# 研究可重复性
set.seed(1005)

# 数据集拆分 
diabetes_split <- diabetes_df %>%
  initial_split(prop = 0.75, strata=diabetes)

diabetes_train_df <- training(diabetes_split)
diabetes_test_df  <-  testing(diabetes_split)


# 创建汇总统计
hd_tab1 <- diabetes_train_df %>% 
  tbl_summary(by=diabetes,
              statistic = list(all_continuous() ~ "{mean} ({sd})",
                               all_categorical() ~ "{n} ({p}%)"),
              digits = all_continuous() ~ 2) %>%
  add_p(test=list(all_continuous() ~ "t.test", 
                  all_categorical() ~ "chisq.test.no.correct")) %>%
  add_overall() %>%
  modify_spanning_header(c("stat_1", "stat_2") ~ "**diabetes**") %>%
  modify_caption("**Table 1: Descriptive Statistics Training Data**")
hd_tab1

# 保存汇总统计结果
hd_tab1 %>%
  as_gt() %>%
  gtsave("figs/diabetes_tab1.png", expand = 10)

# 结论
# 1）p<0.001，所有解释变量都与糖尿病相关
# 2）符合预期和领域知识

# 2.2 数据准备和特征工程 ----
diabetes_recipe <-
  recipe(diabetes ~ ., data=diabetes_train_df) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_downsample(diabetes) 

# 2.3 模型设计和构建 ----
# XGB模型
# NB模型
# SVM模型
# DT模型
# LR模型

# LR模型做基准模型，简单，不要调参
# DT模型 内在可解释性模型，决策工具

# 不同的模型设计和方法
lr_mod <- logistic_reg() %>%
  set_engine("glm") %>% 
  set_mode("classification")

svm_mod <- svm_linear(cost = tune(), margin = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("classification")

xgb_mod <- boost_tree(tree_depth = tune(), learn_rate = tune(), loss_reduction = tune(), 
                      min_n = tune(), sample_size = tune(), trees = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

nb_mod <- naive_Bayes(smoothness = tune(), Laplace = tune()) %>% 
  set_engine("naivebayes") %>% 
  set_mode("classification")

cit_mod <- decision_tree(tree_depth=tune(), min_n=tune()) %>%
  set_engine(engine = "partykit") %>%
  set_mode(mode = "classification") 


# 2.4 超参数调优
# 权衡偏差和方差
# 超参数调优策略
# 空间填充网格搜索策略
# 交叉验证技术

# 度量指标
# 使用ROC-AUC
# 说明：
# AUC与阈值无关，因此在结果变量相对不平衡的情况下通常是一个很好的选择。
# 同时，采用并行化技术加快运算

# 准备交叉验证技术
set.seed(1001)
diabetes_train_folds <- vfold_cv(diabetes_train_df, v=8, strata = diabetes)

# 准备工作流
wf_set <- workflow_set(
  preproc = list(mod = diabetes_recipe), 
  models = list(log_reg=lr_mod, svm_linear = svm_mod, xgboost=xgb_mod, naiveBayes=nb_mod, tree=cit_mod)) 

# 准备网格搜索
grid_ctrl <-
  control_grid(
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = TRUE , 
    event_level = "second"
  )


# 准备并行处理 
cores <- parallel::detectCores(logical = TRUE)


# Create a cluster object and then register: 
cl <- makePSOCKcluster(cores) 
registerDoParallel(cl)

# 设计超参数调优
train_results <- wf_set %>%
  workflow_map(
    fn = 'tune_grid', 
    metrics = metric_set(roc_auc), 
    seed = 1503,
    resamples = diabetes_train_folds, 
    grid = 25, 
    control = grid_ctrl 
  )

stopCluster(cl)

# 超参数调优结果可视化
p1_diab <- train_results %>%
  autoplot() +
  theme_minimal() +
  labs(title='Figure 1: Results Hyperparameter Tuning')
p1_diab


ggsave(p1_diab, file="figs/p1_diab.png")

# 2.4 进一步优化 ----
# 最佳模型
# 模拟退火算法
# 进化算法

xgb_results <- train_results %>% 
  extract_workflow_set_result("mod_xgboost") 

xgb_wf <- train_results %>% 
  extract_workflow("mod_xgboost")


cl <- makePSOCKcluster(cores) 
registerDoParallel(cl)

set.seed(1005)
xgb_sa <- xgb_wf %>%
  tune_sim_anneal(
    resamples =diabetes_train_folds,
    metrics = metric_set(roc_auc), 
    initial = xgb_results,
    iter = 40, 
    control = control_sim_anneal(verbose = TRUE, 
                                 no_improve = 10L, event_level = "second", cooling_coef = 0.1))

stopCluster(cl)


# save max auc:
auc_out <- xgb_sa  %>% 
  collect_metrics() %>% 
  slice_max(mean) %>%
  pull(mean)


# visualize sim annealing:
p2_diab <- autoplot(xgb_sa, type = "performance", metric = 'roc_auc') +
  geom_hline(yintercept=auc_out, linetype="dashed", color = 'red') +
  labs(title='Figure 2: Performance Improvement by Simulated Annealing ') +
  theme_minimal()

ggsave(p2_diab, file="figs/p2_diab.png")


## 2.5 Interpretable Machine Learning （IML） ----

# extract model fit after simulated annealing: 
xgb_fit <- xgb_sa %>% 
  extract_workflow() %>%
  finalize_workflow(xgb_sa %>% select_best())  %>%
  fit(data = diabetes_train_df) %>%
  extract_fit_parsnip()

# Variable importance plot:
p3_diab <- xgb_fit %>%
  vip() +
  theme_minimal() +
  labs(title="Figure 3: Variable Importance")

ggsave(p3_diab, file="figs/p3_diab.png")

#  partial dependence plots 
#prepare training data for pdp:
xgb_df <- xgb_sa %>% 
  extract_workflow() %>%
  finalize_workflow(xgb_sa %>% select_best())  %>%
  fit(data = diabetes_train_df) %>%
  extract_recipe() %>%
  bake(new_data=diabetes_train_df)

explain_xgb <- explain_tidymodels(
  model=xgb_fit, 
  data = (xgb_df %>% dplyr::select(-diabetes)), #data without target column
  y = xgb_df$diabetes,
  label = "xgboost",
  verbose = FALSE
)

# create model profile:
pdp_diab  <- model_profile(explain_xgb, N=1000, variables = "HbA1c_level", groups='hypertension') 

#Create ggplot manually for HbA1c, grouped by hypertension:
p4_diab <- pdp_diab$agr_profiles %>% 
  as_tibble() %>%
  mutate(RiskFactor=paste0('hypertension=', ifelse(stringr::str_sub(`_label_`, 9, 9)=='-', '0', '1'))) %>%
  ggplot(aes(x=`_x_`, y=`_yhat_`, color=RiskFactor)) +
  geom_line(linewidth=2) +
  labs(y='Diabetes Risk Score', x='HbA1c_level', title='Figure 4: Partial Dependence Plot') +
  theme_minimal() 

ggsave(p4_diab, file="figs/p4_diab.png")

# 测试分析
# Fit new best model once on test data at the final end of the process: 
test_results <- xgb_sa %>%
  extract_workflow() %>%
  finalize_workflow(xgb_sa %>% select_best()) %>% 
  last_fit(split = diabetes_split)

# create predictions:
test_p <- collect_predictions(test_results)

# confusion matrix:
conf_mat <- conf_mat(test_p, diabetes, .pred_class) 

p5a_diab <- conf_mat %>%
  autoplot(type = "heatmap") +
  theme(legend.position = "none") +
  labs(title='Confusion Matrix')

#AUC overall
auc <- test_p %>%
  roc_auc(diabetes, .pred_1, event_level = "second") %>%
  mutate(.estimate=round(.estimate, 3)) %>%
  pull(.estimate)

#ROC-curve
roc_curve <- roc_curve(test_p, diabetes, .pred_1, event_level = "second") 

p5b_diab <- roc_curve %>%
  autoplot() +
  annotate('text', x = 0.3, y = 0.75, label = auc) +
  theme_minimal() +
  labs(title='ROC-AUC')

#combine both plots:
p5_diab <- p5a_diab + p5b_diab + plot_annotation('Figure 5: Evaluation on Test Data')

ggsave(p5_diab, file="figs/p5_diab.png")

lalonde <- haven::read_dta("http://www.nber.org/~rdehejia/data/nsw.dta") 


# 参考资料：
# 1 https://boiled-data.github.io/ClassificationDiabetes.html#