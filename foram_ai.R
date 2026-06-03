# -------------------- 环境准备 --------------------
required_packages <- c("ggplot2", "cowplot", "dplyr", "tidyr", "patchwork",
                       "viridis", "ggsci", "RColorBrewer", "scales", "grid",
                       "gtable", "corrplot",  "stringr")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

save_figures <- TRUE
output_dir <- "figures_R/three_targets"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# 辅助函数
save_plot <- function(filename_base, plot, width, height, dpi = 300) {
  ggsave(paste0(filename_base, ".tif"), plot, width = width, height = height, dpi = dpi, compression = "lzw")
  ggsave(paste0(filename_base, ".pdf"), plot, width = width, height = height)
}

clean_feature_names <- function(x) {
  if (is.character(x)) {
    for (i in 0:9) x <- gsub(intToUtf8(0x2080 + i), as.character(i), x, fixed = TRUE)
    x <- gsub("\\[([0-9]+)\\]", "\\1", x)
  }
  x
}

theme_pub <- function(base_size = 15, base_family = "sans") {
  theme_bw(base_size = base_size, base_family = base_family) %+replace%
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = rel(1.3)),
      axis.title = element_text(face = "bold", size = rel(1)),
      axis.text = element_text(size = rel(0.9), colour = "black"),
      axis.line = element_line(colour = "black"),
      legend.title = element_text(face = "bold", size = rel(0.9)),
      legend.text = element_text(size = rel(0.8)),
      legend.position = "top",
      strip.background = element_rect(fill = "grey95", colour = NA),
      strip.text = element_text(face = "bold", size = rel(0.9)),
      panel.grid.major = element_line(colour = "grey90", size = 0.3),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(colour = "black", fill = NA)
    )
}

# 定义三个目标
target_subset <- c("reads", "ZOTUs", "Monothalamida")
suffix <- "three_targets"

# 读取最佳模型表并过滤
if (!file.exists("output_group/Table3_Best_Model_per_Target.csv")) {
  stop("请先运行Python脚本生成 output_group/Table3_Best_Model_per_Target.csv")
}
best_all <- read.csv("output_group/Table3_Best_Model_per_Target.csv")
best_sub <- best_all %>% filter(Target %in% target_subset) %>%
  mutate(Target_label = paste0(Target, " (", Model, ")"))

# ==================== 主图1：预测-观测散点 ====================
if (file.exists("output_group/scatter_data_all.csv")) {
  df_scatter <- read.csv("output_group/scatter_data_all.csv") %>%
    filter(Target %in% target_subset)
  df_scatter$Model <- factor(df_scatter$Model,
                             levels = c("Linear", "Ridge", "RF", "XGBoost", "LightGBM", "GAM"))
  df_r2 <- df_scatter %>%
    group_by(Target, Model) %>%
    summarise(r2 = round(cor(Observed, Predicted)^2, 3), .groups = 'drop')

  p1 <- ggplot(df_scatter, aes(x = Observed, y = Predicted)) +
    geom_point(alpha = 0.55, size = 1.1, colour = "#0072B2") +
    geom_abline(slope = 1, intercept = 0, colour = "#D55E00", linetype = "dashed", size = 0.7) +
    facet_grid(Target ~ Model, scales = "free") +
    geom_text(data = df_r2, aes(x = -Inf, y = Inf, label = paste0("italic(R) ^ 2 == ", r2)),
              parse = TRUE, hjust = -0.2, vjust = 2, size = 4.5, colour = "gray20", inherit.aes = FALSE) +
    labs(title = "Predicted vs Observed Values",
         x = "Observed", y = "Predicted") +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  print(p1)
  if (save_figures) save_plot(paste0(output_dir, "/Figure1_scatter_", suffix), p1, width = 16, height = 10)
  message("✓ 图1 已生成")
}

# ==================== 主图2：模型性能对比（R² ± SD） ====================
if (file.exists("output_group/model_performance_summary.csv")) {
  df_perf <- read.csv("output_group/model_performance_summary.csv") %>%
    filter(Target %in% target_subset)
  df_r2 <- df_perf %>%
    group_by(Target, Model) %>%
    summarise(Mean = mean(R2, na.rm = TRUE), SD = sd(R2, na.rm = TRUE), .groups = 'drop')

  model_order <- df_r2 %>% group_by(Model) %>% summarise(avg_R2 = mean(Mean)) %>%
    arrange(desc(avg_R2)) %>% pull(Model)
  df_r2$Model <- factor(df_r2$Model, levels = model_order)

  p2 <- ggplot(df_r2, aes(x = Model, y = Mean, fill = Target)) +
    geom_bar(stat = "identity", position = position_dodge(0.8), colour = "white", width = 0.7, size = 0.2) +
    geom_errorbar(aes(ymin = Mean - SD, ymax = Mean + SD), position = position_dodge(0.8), width = 0.2, size = 0.6, colour = "gray30") +
    scale_fill_npg(name = "Target") +
    labs(title = "Spatial cross-validated R²", y = "R² ± SD", x = "") +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "right") +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "gray50")
  print(p2)
  if (save_figures) save_plot(paste0(output_dir, "/Figure2_R2_comparison_", suffix), p2, width = 10, height = 6)
  message("✓ 图2 已生成")
}

# ==================== 主图3：SHAP热图（最佳模型） ====================
if (file.exists("output_group/feature_shap_importance_by_model.csv")) {
  df_shap_all <- read.csv("output_group/feature_shap_importance_by_model.csv") %>%
    filter(Target %in% target_subset)
  df_shap_all$Feature <- clean_feature_names(df_shap_all$Feature)
  df_shap_best <- df_shap_all %>% inner_join(best_sub, by = c("Target", "Model"))

  all_features <- unique(df_shap_best$Feature)
  targets <- unique(df_shap_best$Target)
  heat_plots <- list()

  for (i in seq_along(targets)) {
    target_name <- targets[i]
    model_name <- best_sub$Model[best_sub$Target == target_name]
    df_target <- df_shap_best %>%
      filter(Target == target_name, Feature %in% all_features) %>%
      arrange(MeanAbsSHAP) %>%
      mutate(Feature = factor(Feature, levels = unique(Feature)))
    shap_range <- range(df_target$MeanAbsSHAP, na.rm = TRUE)
    p <- ggplot(df_target, aes(x = Target, y = Feature, fill = MeanAbsSHAP)) +
      geom_tile(colour = "white", size = 0.2) +
      scale_fill_viridis_c(name = "Mean |SHAP|", option = "plasma", limits = shap_range, oob = scales::squish,
                           guide = guide_colorbar(barwidth = 0.8, barheight = 10)) +
      labs(title = paste0(target_name, " (", model_name, ")"), x = "", y = "") +
      theme_minimal(base_size = 14) +
      theme(axis.text.x = element_blank(), axis.text.y = element_text(size = 12), legend.position = "right",
            legend.title = element_text(face = "bold", size = 12), legend.text = element_text(size = 10),
            plot.title = element_text(face = "bold", size = 14, hjust = 0.5), panel.grid = element_blank())
    heat_plots[[i]] <- p
  }
  n_col <- length(targets)  # 3个目标，一行放3个
  p3 <- wrap_plots(heat_plots, ncol = n_col) +
    plot_annotation(title = "SHAP Feature Importance (Best-performing Models)",
                    theme = theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 18)))
  print(p3)
  if (save_figures) save_plot(paste0(output_dir, "/Figure3_SHAP_heatmap_", suffix), p3, width = 15, height = 8)
  message("✓ 图3 已生成")
}

# ==================== 主图4：SHAP条形图（分组颜色，统一图例到底部） ====================
if (file.exists("output_group/feature_shap_importance_by_model.csv")) {
  df_shap_feat <- read.csv("output_group/feature_shap_importance_by_model.csv") %>%
    filter(Target %in% target_subset)
  df_shap_feat$Feature <- clean_feature_names(df_shap_feat$Feature)

  if (!"Group" %in% colnames(df_shap_feat)) {
    group_map <- c(
      "Sediment_PC1" = "Sediment Gradient (PC1)", "Sediment_PC2" = "Sediment Gradient (PC2)",
      "Major_PC1" = "Major Element Gradient (PC1)", "Major_PC2" = "Major Element Gradient (PC2)",
      "Trace_PC1" = "Trace Element Gradient (PC1)", "Trace_PC2" = "Trace Element Gradient (PC2)",
      "Organic_PC1" = "Organic-Depth Gradient (PC1)", "Organic_PC2" = "Organic-Depth Gradient (PC2)"
    )
    df_shap_feat <- df_shap_feat %>% mutate(Group = group_map[Feature])
  }

  df_shap_feat$Group_wrapped <- str_wrap(df_shap_feat$Group, width = 18)
  all_groups <- sort(unique(df_shap_feat$Group_wrapped))

  df_shap_best <- df_shap_feat %>% inner_join(best_sub, by = c("Target", "Model"))

  targets <- unique(df_shap_best$Target)
  shap_plots <- list()

  for (i in seq_along(targets)) {
    target_name <- targets[i]
    model_name <- best_sub$Model[best_sub$Target == target_name]
    df_target <- df_shap_best %>%
      filter(Target == target_name) %>%
      arrange(MeanAbsSHAP) %>%
      mutate(Feature = factor(Feature, levels = unique(Feature)))

    p <- ggplot(df_target, aes(x = Feature, y = MeanAbsSHAP, fill = Group_wrapped)) +
      geom_col(width = 0.7, alpha = 0.9) +
      coord_flip() +
      scale_fill_lancet(drop = FALSE, breaks = all_groups) +
      labs(title = paste0(target_name, " (", model_name, ")"),
           x = "", y = "Mean |SHAP|") +
      theme_bw(base_size = 14) +
      theme(legend.position = "none",
            panel.grid.major.y = element_blank(),
            axis.text.y = element_text(size = 12),
            axis.text.x = element_text(size = 11),
            axis.title.x = element_text(size = 12),
            plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
    shap_plots[[i]] <- p
  }

  # 3个目标，排列为3列1行
  p4 <- wrap_plots(shap_plots, ncol = 3, nrow = 1) +
    plot_annotation(title = "Feature Importance by Target (Best-performing Models)",
                    theme = theme(plot.title = element_text(face = "bold", size = 18, hjust = 0.5))) +
    plot_layout(guides = "collect") &
    theme(legend.position = "bottom",
          legend.justification = "center",
          legend.box = "horizontal",
          legend.title = element_text(face = "bold", size = 12),
          legend.text = element_text(size = 10),
          legend.key.size = unit(1, "cm"),
          legend.margin = margin(t = 5, b = 5, l = 10, r = 10)) &
    guides(fill = guide_legend(nrow = 1, byrow = TRUE,
                               title.position = "left",
                               title.hjust = 0.5))

  print(p4)
  if (save_figures) save_plot(paste0(output_dir, "/Figure4_SHAP_all_features_", suffix), p4, width = 18, height = 8)
  message("✓ 图4 已生成")
}

# ==================== 主图5：SHAP分组贡献比例 ====================
if (file.exists("output_group/shap_group_contributions_by_model.csv")) {
  df_shap_group <- read.csv("output_group/shap_group_contributions_by_model.csv") %>%
    filter(Target %in% target_subset) %>%
    inner_join(best_sub, by = c("Target", "Model"))
  df_shap_group$Target_label <- paste0(df_shap_group$Target, " (", df_shap_group$Model, ")")

  wrap_x_labels <- function(x) gsub("(.+?)\\s\\((.+)\\)", "\\1\n(\\2)", x)
  p5 <- ggplot(df_shap_group, aes(x = Group, y = Contribution, fill = Target_label)) +
    geom_bar(stat = "identity", position = position_dodge(0.8), colour = "white", width = 0.7, size = 0.2) +
    scale_fill_jco(name = "Target (Best-performing Model)") +
    labs(title = "SHAP Feature Group Contribution",
         x = "", y = "Proportion of |SHAP|") +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 11, margin = margin(t = 10, b = 5)),
          legend.text = element_text(size = 10), legend.key.size = unit(0.8, "cm"),
          legend.title = element_text(face = "bold", size = 11),
          legend.position = "bottom",
          legend.justification = "center") +
    scale_x_discrete(labels = wrap_x_labels) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1), expand = expansion(mult = c(0, 0.05))) +
    guides(fill = guide_legend(nrow = 1))
  print(p5)
  if (save_figures) save_plot(paste0(output_dir, "/Figure5_SHAP_group_contribution_", suffix), p5, width = 10, height = 7)
  message("✓ 图5 已生成")
}

# ==================== 主图6：残差诊断 ====================
if (file.exists("output_group/oof_with_residuals.csv")) {
  df_resid <- read.csv("output_group/oof_with_residuals.csv") %>%
    filter(Target %in% target_subset) %>%
    inner_join(best_sub, by = c("Target", "Model")) %>%
    mutate(Target_label = paste0(Target, " (", Model, ")"))
  p6a <- ggplot(df_resid, aes(x = Predicted, y = Residual)) +
    geom_point(alpha = 0.6, colour = "#0072B2", size = 2) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "#D55E00", size = 0.8) +
    facet_wrap(~ Target_label, scales = "free", ncol = 3) +
    labs(title = "Residuals vs Fitted", x = "Predicted", y = "Residual") +
    theme_pub() + theme(strip.text = element_text(size = 13))
  p6b <- ggplot(df_resid, aes(sample = Residual)) +
    stat_qq(alpha = 0.6, colour = "#0072B2", size = 2) +
    stat_qq_line(colour = "#D55E00", linetype = "dashed", size = 0.8) +
    facet_wrap(~ Target_label, scales = "free", ncol = 3) +
    labs(title = "Q-Q Plot", x = "Theoretical Quantiles", y = "Sample Quantiles") +
    theme_pub() + theme(strip.text = element_text(size = 13))
  p6 <- p6a / p6b + plot_annotation(title = "Residual Diagnostics for Best-performing Models") &
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 18))
  print(p6)
  if (save_figures) save_plot(paste0(output_dir, "/Figure6_residual_diagnostic_", suffix), p6, width = 14, height = 12)
  message("✓ 图6 已生成")
}

# ==================== 补充图S1：PCA载荷热图 ====================
if (file.exists("output_group/pca_loadings_all.csv")) {
  df_load <- read.csv("output_group/pca_loadings_all.csv") %>% filter(Target %in% target_subset)
  df_load_mean <- df_load %>%
    group_by(Target, Group, PC, Variable) %>%
    summarise(Loading = mean(Loading), .groups = 'drop')
  var_order <- df_load_mean %>%
    group_by(Variable) %>%
    summarise(avg_loading = mean(abs(Loading))) %>%
    arrange(desc(avg_loading)) %>% pull(Variable)
  df_load_mean$Variable <- factor(df_load_mean$Variable, levels = var_order)

  base_size_large <- 16

  df_pc1 <- df_load_mean %>% filter(PC == "PC1")
  p_pc1 <- ggplot(df_pc1, aes(x = Target, y = Variable, fill = Loading)) +
    geom_tile(colour = "white", size = 0.2) +
    scale_fill_gradient2(low = "#313695", mid = "white", high = "#A50026", name = "Loading", midpoint = 0) +
    facet_grid(Group ~ ., scales = "free_y", space = "free_y") +
    labs(title = "A", x = "", y = "") +
    theme_minimal(base_size = base_size_large) +
    theme(strip.text.y = element_text(angle = 0, face = "bold", size = 14),
          axis.text.x = element_text(angle = 45, hjust = 1, size = 13),
          axis.text.y = element_text(size = 12),
          legend.position = "right",
          legend.title = element_text(size = 14, face = "bold"),
          legend.text = element_text(size = 12),
          plot.title = element_text(face = "bold", size = 18, hjust = -0.1),
          panel.grid = element_blank())

  df_pc2 <- df_load_mean %>% filter(PC == "PC2")
  p_pc2 <- ggplot(df_pc2, aes(x = Target, y = Variable, fill = Loading)) +
    geom_tile(colour = "white", size = 0.2) +
    scale_fill_gradient2(low = "#313695", mid = "white", high = "#A50026", name = "Loading", midpoint = 0) +
    facet_grid(Group ~ ., scales = "free_y", space = "free_y") +
    labs(title = "B", x = "", y = "") +
    theme_minimal(base_size = base_size_large) +
    theme(strip.text.y = element_text(angle = 0, face = "bold", size = 14),
          axis.text.x = element_text(angle = 45, hjust = 1, size = 13),
          axis.text.y = element_blank(),
          legend.position = "right",
          legend.title = element_text(size = 14, face = "bold"),
          legend.text = element_text(size = 12),
          plot.title = element_text(face = "bold", size = 18, hjust = -0.1),
          panel.grid = element_blank())

  pS1 <- (p_pc1 + p_pc2) +
    plot_annotation(title = "PCA Loadings (PC1 vs PC2)",
                    theme = theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 20)))
  print(pS1)
  if (save_figures) save_plot(paste0(output_dir, "/FigureS1_PCA_loadings_", suffix), pS1, width = 18, height = 20)
  message("✓ 附图S1 已生成")
}

# ==================== 附图S2：PCA方差解释率 ====================
if (file.exists("output_group/pca_variances_all.csv")) {
  df_var <- read.csv("output_group/pca_variances_all.csv") %>%
    filter(Target %in% target_subset)
  pS2 <- ggplot(df_var, aes(x = Target, y = ExplainedVarianceRatio, fill = PC)) +
    geom_boxplot(alpha = 0.7, outlier.size = 2) +
    scale_fill_manual(values = c("PC1" = "#4DBBD5", "PC2" = "#E64B35"), name = "Principal Component") +
    labs(title = "Variance Explained by PC1 and PC2", y = "Explained Variance Ratio", x = "") +
    theme_pub() + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12), legend.position = "right")
  print(pS2)
  if (save_figures) save_plot(paste0(output_dir, "/FigureS2_PCA_variance_", suffix), pS2, width = 10, height = 7)
  message("✓ 附图S2 已生成")
}

# ==================== 附图S3：VIF对比 ====================
if (file.exists("output_group/vif_original_features.csv") && file.exists("output_group/vif_pc_features.csv")) {
  vif_orig <- read.csv("output_group/vif_original_features.csv"); vif_orig$Type <- "Original Features"
  vif_pc <- read.csv("output_group/vif_pc_features.csv"); vif_pc$Type <- "PC Features"
  vif_all <- bind_rows(vif_orig, vif_pc)
  pS3 <- ggplot(vif_all, aes(x = reorder(Feature, VIF), y = VIF, fill = Type)) +
    geom_col(position = position_dodge(0.8), width = 0.7, alpha = 0.8) +
    geom_hline(yintercept = c(5, 10), linetype = "dashed", colour = "gray40", size = 0.5) +
    scale_fill_jco(name = "") + coord_flip() +
    labs(title = "Variance Inflation Factor (VIF) Comparison", x = "", y = "VIF") +
    theme_pub(base_size = 14) + theme(legend.position = "top", axis.text.y = element_text(size = 11))
  print(pS3)
  if (save_figures) save_plot(paste0(output_dir, "/FigureS3_VIF_comparison_", suffix), pS3, width = 10, height = 12)
  message("✓ 附图S3 已生成")
}

# ==================== 附图S4：SHAP稳定性热图 ====================
if (file.exists("output_group/feature_importance_variation.csv")) {
  df_cv <- read.csv("output_group/feature_importance_variation.csv")
  cv_col <- if ("Importance_CV" %in% names(df_cv)) "Importance_CV" else "CV"
  df_cv <- df_cv %>%
    filter(Target %in% target_subset) %>%
    inner_join(best_sub, by = c("Target", "Model")) %>%
    mutate(Feature = clean_feature_names(Feature), Target_label = paste0(Target, " (", Model, ")"))
  df_cv$CV_value <- df_cv[[cv_col]]
  pS4 <- ggplot(df_cv, aes(x = Target_label, y = Feature, fill = CV_value)) +
    geom_tile(colour = "white", size = 0.2) +
    scale_fill_viridis_c(option = "magma", name = "CV", direction = -1) +
    labs(title = "SHAP Importance Variability", x = "", y = "") +
    theme_minimal(base_size = 14) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12), axis.text.y = element_text(size = 11),
          legend.title = element_text(face = "bold", size = 12), plot.title = element_text(hjust = 0.5, face = "bold", size = 16))
  print(pS4)
  if (save_figures) save_plot(paste0(output_dir, "/FigureS4_SHAP_CV_heatmap_", suffix), pS4, width = 12, height = 8)
  message("✓ 附图S4 已生成")
}

# ==================== 附图S5：SHAP排名稳定性 ====================
if (file.exists("output_group/shap_rank_spearman_stability.csv")) {
  df_spear <- read.csv("output_group/shap_rank_spearman_stability.csv") %>%
    filter(Target %in% target_subset)
  pS5 <- ggplot(df_spear, aes(x = Model, y = SpearmanR, fill = Model)) +
    geom_boxplot(alpha = 0.7, outlier.size = 2) +
    facet_wrap(~ Target, ncol = 3) +
    scale_fill_lancet() +
    labs(title = "Cross-fold Stability of SHAP Feature Rankings (Spearman Correlation)",
         y = "Spearman's ρ", x = "") +
    theme_pub() + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 11), legend.position = "none")
  print(pS5)
  if (save_figures) save_plot(paste0(output_dir, "/FigureS5_SHAP_rank_spearman_", suffix), pS5, width = 14, height = 9)
  message("✓ 附图S5 已生成")
}

# ==================== 附图S6：训练-测试Gap ====================
if (file.exists("output_group/train_test_gap_analysis.csv")) {
  df_gap <- read.csv("output_group/train_test_gap_analysis.csv") %>%
    filter(Target %in% target_subset)
  pS6 <- ggplot(df_gap, aes(x = Model, y = R2_Gap, fill = Target)) +
    geom_bar(stat = "identity", position = position_dodge(0.8), colour = "white", width = 0.7, alpha = 0.8) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "gray30") +
    scale_fill_jco() +
    labs(title = "Train-Test R² Gap (Overfitting Indicator)", y = "R²_train - R²_test", x = "") +
    theme_pub() + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 11))
  print(pS6)
  if (save_figures) save_plot(paste0(output_dir, "/FigureS6_train_test_gap_", suffix), pS6, width = 12, height = 7)
  message("✓ 附图S6 已生成")
}

# ==================== 附图S7：超参数稳定性 ====================
if (file.exists("output_group/best_hyperparameters_all_folds.csv")) {
  df_hp_raw <- read.csv("output_group/best_hyperparameters_all_folds.csv") %>%
    filter(Target %in% target_subset)
  if (nrow(df_hp_raw) > 0) {
    hp_long <- df_hp_raw %>%
      pivot_longer(cols = c(n_estimators, max_depth, learning_rate, alpha),
                   names_to = "Parameter", values_to = "Value") %>%
      filter(!is.na(Value))
    pS7 <- ggplot(hp_long, aes(x = Model, y = Value, fill = Model)) +
      geom_boxplot(alpha = 0.7, outlier.size = 2) +
      facet_wrap(~ Parameter, scales = "free_y", ncol = 2) +
      scale_fill_lancet() +
      labs(title = "Hyperparameter Distribution across Folds", y = "Value", x = "") +
      theme_pub() + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 11), legend.position = "none")
    print(pS7)
    if (save_figures) save_plot(paste0(output_dir, "/FigureS7_hyperparameter_stability_", suffix), pS7, width = 12, height = 9)
    message("✓ 附图S7 已生成")
  }
}

# ==================== 附图S8：GAM参数稳定性 ====================
if (file.exists("output_group/gam_lam_records.csv")) {
  df_gam_raw <- read.csv("output_group/gam_lam_records.csv") %>%
    filter(Target %in% target_subset)
  pS8a <- ggplot(df_gam_raw, aes(x = Target, y = BestLam)) +
    geom_boxplot(alpha = 0.7, fill = "#4DBBD5", outlier.size = 2) + scale_y_log10() +
    labs(title = "GAM: Optimal Lambda", y = "Lambda (log scale)", x = "") +
    theme_pub() + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 11))
  pS8b <- ggplot(df_gam_raw, aes(x = Target, y = EDOF)) +
    geom_boxplot(alpha = 0.7, fill = "#E64B35", outlier.size = 2) +
    labs(title = "GAM: Effective Degrees of Freedom", y = "EDOF", x = "") +
    theme_pub() + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 11))
  pS8c <- ggplot(df_gam_raw, aes(x = Target, y = CV_R2)) +
    geom_boxplot(alpha = 0.7, fill = "#00A087", outlier.size = 2) +
    labs(title = "GAM: Cross-validated R²", y = "CV R²", x = "") +
    theme_pub() + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 11))
  pS8 <- (pS8a | pS8b) / pS8c +
    plot_annotation(title = "GAM Hyperparameter Stability",
                    theme = theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 18)))
  print(pS8)
  if (save_figures) save_plot(paste0(output_dir, "/FigureS8_GAM_stability_", suffix), pS8, width = 12, height = 10)
  message("✓ 附图S8 已生成")
}

# ==================== 附图S9：特征重要性变异系数 ====================
if (file.exists("output_group/feature_importance_variation.csv")) {
  df_var <- read.csv("output_group/feature_importance_variation.csv")
  cv_col <- if ("Importance_CV" %in% names(df_var)) "Importance_CV" else "CV"
  df_var <- df_var %>%
    filter(Target %in% target_subset) %>%
    inner_join(best_sub, by = c("Target", "Model")) %>%
    mutate(Feature = clean_feature_names(Feature), Target_label = paste0(Target, " (", Model, ")"))
  df_var$CV_value <- df_var[[cv_col]]
  pS9 <- ggplot(df_var, aes(x = reorder(Feature, CV_value), y = CV_value, fill = Target_label)) +
    geom_col(alpha = 0.8, width = 0.7) + coord_flip() +
    facet_wrap(~ Target_label, scales = "free_y", ncol = 3) +
    scale_fill_jco() +
    labs(title = "Feature Importance Variability", x = "", y = "Coefficient of Variation") +
    theme_pub(base_size = 14) + theme(legend.position = "none", strip.text = element_text(size = 12), axis.text.y = element_text(size = 11))
  print(pS9)
  if (save_figures) save_plot(paste0(output_dir, "/FigureS9_feature_variability_", suffix), pS9, width = 14, height = 9)
  message("✓ 附图S9 已生成")
}

# ==================== 附图S10：特征出现频率 ====================
if (file.exists("output_group/shap_top_frequency.csv")) {
  df_freq <- read.csv("output_group/shap_top_frequency.csv") %>%
    filter(Target %in% target_subset) %>%
    inner_join(best_sub, by = c("Target", "Model")) %>%
    mutate(Feature = clean_feature_names(Feature), Target_label = paste0(Target, " (", Model, ")"))
  pS10 <- ggplot(df_freq, aes(x = reorder(Feature, -Top_Frequency), y = Top_Frequency, fill = Target_label)) +
    geom_col(alpha = 0.8, width = 0.7) + coord_flip() +
    facet_wrap(~ Target_label, scales = "free_y", ncol = 3) +
    scale_fill_jco() +
    labs(title = "Top-2 Feature Frequency", x = "", y = "Count (out of 3 folds)") +
    theme_pub(base_size = 14) + theme(legend.position = "none", strip.text = element_text(size = 12), axis.text.y = element_text(size = 11))
  print(pS10)
  if (save_figures) save_plot(paste0(output_dir, "/FigureS10_top_frequency_", suffix), pS10, width = 14, height = 9)
  message("✓ 附图S10 已生成")
}

#组合图
# ==================== 补充图S1 + S3：组合为三个子图 A/B/C（标签内嵌，主图占比更大） ====================
# 注意：需要提前加载 patchwork 包，若无则安装：install.packages("patchwork")
library(patchwork)
library(ggplot2)
library(dplyr)
library(ggpubr)   # 若缺少，可注释并使用 theme_minimal 替代
library(ggsci)    # 为 scale_fill_jco

# -------------------- 子图 A: PC1 载荷热图 --------------------
if (file.exists("output_group/pca_loadings_all.csv")) {
  df_load <- read.csv("output_group/pca_loadings_all.csv") %>% filter(Target %in% target_subset)
  df_load_mean <- df_load %>%
    group_by(Target, Group, PC, Variable) %>%
    summarise(Loading = mean(Loading), .groups = 'drop')
  var_order <- df_load_mean %>%
    group_by(Variable) %>%
    summarise(avg_loading = mean(abs(Loading))) %>%
    arrange(desc(avg_loading)) %>% pull(Variable)
  df_load_mean$Variable <- factor(df_load_mean$Variable, levels = var_order)
  
  base_size_large <- 16   # 增大基础字体（原为14）
  
  # 子图 A: PC1
  df_pc1 <- df_load_mean %>% filter(PC == "PC1")
  p_pc1 <- ggplot(df_pc1, aes(x = Target, y = Variable, fill = Loading)) +
    geom_tile(colour = "white", size = 0.2) +
    scale_fill_gradient2(low = "#313695", mid = "white", high = "#A50026", name = "Loading", midpoint = 0) +
    facet_grid(Group ~ ., scales = "free_y", space = "free_y") +
    labs(tag = "A", x = "", y = "") +
    theme_minimal(base_size = base_size_large) +
    theme(strip.text.y = element_text(angle = 0, face = "bold", size = 12),   # 增大分面标题
          axis.text.x = element_text(angle = 45, hjust = 1, size = 11),        # X轴标签增大
          axis.text.y = element_text(size = 11),                               # Y轴标签增大
          legend.position = "right",
          legend.title = element_text(size = 12, face = "bold"),               # 图例标题增大
          legend.text = element_text(size = 10),                               # 图例文字增大
          plot.tag = element_text(face = "bold", size = 18,                    # 标签A/B/C字体增大
                                  margin = margin(t = -2, r = -2, b = -2, l = -2)),  # 负边距，更贴近角落
          plot.tag.position = c(0.01, 0.98),                                   # 调整至左上角，避免遮挡
          panel.grid = element_blank())
  
  # 子图 B: PC2 (隐藏 Y 轴文字)
  df_pc2 <- df_load_mean %>% filter(PC == "PC2")
  p_pc2 <- ggplot(df_pc2, aes(x = Target, y = Variable, fill = Loading)) +
    geom_tile(colour = "white", size = 0.2) +
    scale_fill_gradient2(low = "#313695", mid = "white", high = "#A50026", name = "Loading", midpoint = 0) +
    facet_grid(Group ~ ., scales = "free_y", space = "free_y") +
    labs(tag = "B", x = "", y = "") +
    theme_minimal(base_size = base_size_large) +
    theme(strip.text.y = element_text(angle = 0, face = "bold", size = 12),
          axis.text.x = element_text(angle = 45, hjust = 1, size = 11),
          axis.text.y = element_blank(),                                     # 隐藏Y轴
          legend.position = "right",
          legend.title = element_text(size = 12, face = "bold"),
          legend.text = element_text(size = 10),
          plot.tag = element_text(face = "bold", size = 18,
                                  margin = margin(t = -2, r = -2, b = -2, l = -2)),
          plot.tag.position = c(0.01, 0.98),
          panel.grid = element_blank())
  
  p_s1_part <- TRUE
} else {
  p_s1_part <- FALSE
  message("警告：未找到 pca_loadings_all.csv，跳过 PC1/PC2 热图")
}

# -------------------- 子图 C: VIF 对比图 --------------------
if(file.exists("output_group/vif_original_features.csv") && file.exists("output_group/vif_pc_features.csv")) {
  vif_orig <- read.csv("output_group/vif_original_features.csv"); vif_orig$Type <- "Original Features"
  vif_pc <- read.csv("output_group/vif_pc_features.csv"); vif_pc$Type <- "PC Features"
  vif_all <- bind_rows(vif_orig, vif_pc)
  
  p_pc3 <- ggplot(vif_all, aes(x = reorder(Feature, VIF), y = VIF, fill = Type)) +
    geom_col(position = position_dodge(0.8), width = 0.7, alpha = 0.8) +
    geom_hline(yintercept = c(5, 10), linetype = "dashed", colour = "gray40", size = 0.5) +
    scale_fill_jco(name = "") + 
    coord_flip() +
    labs(tag = "C", x = "", y = "",title = "VIF") +
    theme_pub(base_size = 14) +
    theme(
      # 将图例放置在绘图区内部右下角，避免超出边界
      legend.position = c(0,-0.2),
      legend.justification = c(0, 0),
      # legend.background = element_rect(fill = "white", colour = "black", size = 0.3),
      legend.title = element_blank(),
      legend.text = element_text(size = 10),
      # 调整 Y 轴标签（特征名）与轴的距离，减小左边距
      axis.text.y = element_text(size = 11, margin = margin(r = 0)),
      axis.title.x = element_text(size = 12),
      # 减小整个图形的边距，使标签更紧凑
      plot.margin = margin(t = 5, r = 10, b = 0, l = 0, unit = "pt"),
      plot.tag = element_text(face = "bold", size = 18,
                              margin = margin(t = -2, r = -2, b = -2, l = -2)),
      plot.tag.position = c(0.01, 0.98)
    )
  
  p_s3_part <- TRUE
} else {
  p_s3_part <- FALSE
  message("警告：未找到 VIF 数据文件，跳过 VIF 图")
}


# -------------------- 最终组合：三个子图并排 (A | B | C) 并调整宽度比例 --------------------
if (p_s1_part && p_s3_part) {
  p_combined <- p_pc1 + p_pc2 + p_pc3 +
    plot_layout(widths = c(1, 1, 1)) +
    plot_annotation(
                    theme = theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 20)))
  
  print(p_combined)
  if (save_figures) {
    save_plot(paste0(output_dir, "/FigureS1_S3_three_panels_", suffix), 
              p_combined, width = 24, height = 10)
  }
  message("✓ 组合图（三个子图并排，字体增大，标签不遮挡）已生成")
} else {
  message("✗ 无法生成组合图：缺少必要数据")
}



message("\n🎉 三个目标完整分析完成！所有图片保存在 ", output_dir, "/")