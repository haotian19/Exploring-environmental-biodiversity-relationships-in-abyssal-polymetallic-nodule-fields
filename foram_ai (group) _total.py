import numpy as np
import pandas as pd
import os
import warnings
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from pygam import LinearGAM, s
import shap
from functools import reduce
import operator
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
np.random.seed(22)
os.makedirs("output_group", exist_ok=True)


# =========================================================
# 1. 数据加载（新文件、新表名）
# =========================================================
df = pd.read_excel("data/foram_env_data_dy79.xlsx", sheet_name="summary")
df.columns = df.columns.str.strip()   # 移除列名两端空格

import unicodedata

def normalize_column_names(col):
    # 将常见下标数字转换为普通数字
    subscript_map = {
        '₂': '2', '₃': '3', '₄': '4', '₅': '5',
        '₆': '6', '₇': '7', '₈': '8', '₉': '9'
    }
    for sub, digit in subscript_map.items():
        col = col.replace(sub, digit)
    return col

df.columns = [normalize_column_names(col) for col in df.columns]

# 分组变量（用于GroupKFold）
groups = df["Area"]

# 目标变量：9个新指标（确保列名与Excel一致）
target_cols = ["reads", "zotu", "shannon", "simpson",
               "Rotaliida", "Textulariida", "Monothalamida", "Miliolida", "Other"]
Y = df[target_cols].copy()
target_names = Y.columns.tolist()

# 原始特征列（全部用于分组PCA）
feature_cols = [
    "Grain Size", "Sorting", "Skewness", "Kurtosis",
    "Clay Content", "Silt Content", "Sand Content",
    "Chlorophyll a", "N", "C", "S1", "TOC",
    "Na2O", "MgO", "Al2O3", "SiO2", "P", "S2", "Cl",
    "K2O", "CaO", "Fe2O3",
    "Sc", "Ti", "V", "Cr", "Mn", "Co", "Ni", "Cu", "Zn", "Ga", "As", "Rb",
    "Depth"
]


# 外部交叉验证：按区域分组，保证同区域样本不跨折
outer_cv = GroupKFold(n_splits=3)
# 内部交叉验证：用于超参数调优，随机打乱
inner_cv = KFold(n_splits=4, shuffle=True, random_state=22)

# =========================================================
# 2. PCA函数（在每个外部折的训练集上拟合，转换训练集和验证集）
#    同时将载荷和方差解释率追加到传入的列表中，最后统一保存
# =========================================================

def build_pca(train_df, test_df, fold, target, all_loadings, all_variances):
    """
    对四组特征分别进行PCA降维，提取前两个主成分（PC1, PC2）。
    将载荷和方差解释率追加到列表中。
    返回训练集和验证集的PC1/PC2特征数据框。
    """
    groups_dict = {
        "Sediment": ["Grain Size","Sorting","Skewness","Kurtosis",
                     "Clay Content","Silt Content","Sand Content"],
        "Major": ["Na2O","MgO","Al2O3","SiO2","K2O","CaO","Fe2O3"],
        "Trace": ["Sc","Ti","V","Cr","Mn","Co","Ni","Cu","Zn","Ga","As","Rb"],
        "Organic": ["Chlorophyll a","Depth","N","C","S1","P","S2","Cl"]
    }

    X_tr_final = pd.DataFrame()
    X_va_final = pd.DataFrame()

    for gname, vars_list in groups_dict.items():
        # 标准化
        scaler_local = StandardScaler()
        tr_scaled = scaler_local.fit_transform(train_df[vars_list])
        va_scaled = scaler_local.transform(test_df[vars_list])

        # PCA降维到2个主成分
        pca = PCA(n_components=2)
        tr_pc = pca.fit_transform(tr_scaled)
        va_pc = pca.transform(va_scaled)

        # 保存PC1和PC2
        X_tr_final[f"{gname}_PC1"] = tr_pc[:, 0]
        X_tr_final[f"{gname}_PC2"] = tr_pc[:, 1]
        X_va_final[f"{gname}_PC1"] = va_pc[:, 0]
        X_va_final[f"{gname}_PC2"] = va_pc[:, 1]

        # 记录每个变量的载荷（PC1和PC2分开）
        for comp_idx, comp_name in enumerate([f"PC1", f"PC2"]):
            for var, loading in zip(vars_list, pca.components_[comp_idx]):
                all_loadings.append({
                    "Target": target,
                    "Fold": fold,
                    "Group": gname,
                    "PC": comp_name,
                    "Variable": var,
                    "Loading": loading
                })

        # 记录方差解释率（每个主成分单独）
        for comp_idx, var_ratio in enumerate(pca.explained_variance_ratio_):
            all_variances.append({
                "Target": target,
                "Fold": fold,
                "Group": gname,
                "PC": f"PC{comp_idx+1}",
                "ExplainedVarianceRatio": var_ratio
            })

    return X_tr_final, X_va_final

# 最终建模使用的特征名称（4组 × 2个PC = 8个）
feature_names = [
    "Sediment_PC1", "Sediment_PC2",
    "Major_PC1", "Major_PC2",
    "Trace_PC1", "Trace_PC2",
    "Organic_PC1", "Organic_PC2"
]

# =========================================================
# 3. 各模型的超参数网格
# =========================================================

rf_grid = {
    'n_estimators':[200,500],
    'max_depth':[3,5],
    'min_samples_split':[2,5],
    'min_samples_leaf':[1,2]
}

xgb_grid = {
    'n_estimators':[200,500],
    'max_depth':[3,5],
    'learning_rate':[0.05,0.1]
}

lgb_grid = {
    'n_estimators':[100,300],
    'max_depth':[3,5],
    'learning_rate':[0.05,0.1]
}

ridge_grid = {'alpha': [0.1, 1, 10, 100]}   # 岭回归调优网格

# =========================================================
# 4. 主循环：对每个目标变量和每个外部折进行建模
# =========================================================

all_perf = []          # 存储所有模型的性能指标（R2, RMSE, MAE）
all_pred = []          # 存储所有模型的预测值与真实值
all_shap = []          # 存储三个树模型（RF, XGB, LGB）的SHAP值
all_best_params = []   # 存储每次网格搜索得到的最佳超参数
gam_lam_records = []   # 存储GAM模型选择的lambda和有效自由度
all_pca_loadings = []  # 合并所有fold的PCA载荷
all_pca_variances = [] # 合并所有fold的PCA方差解释率


for target in target_names:
    y = Y[target]
    print(f"\n========== 目标变量: {target} ==========")

    for fold, (train_idx, val_idx) in enumerate(
            outer_cv.split(df, y, groups)):

        # 划分训练集和验证集（外部折）
        df_tr = df.iloc[train_idx]
        df_va = df.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_va = y.iloc[val_idx]

        # 在当前外部折内进行PCA，并收集载荷和方差
        X_tr, X_va = build_pca(df_tr, df_va, fold, target,
                                all_pca_loadings, all_pca_variances)

        # ---------------- 线性回归（无超参数）
        lin = LinearRegression().fit(X_tr, y_tr)
        lin_pred_val = lin.predict(X_va)
        lin_pred_train = lin.predict(X_tr)
        # 添加合并的性能记录（验证 + 训练）
        all_perf.append({
            "Target": target, "Fold": fold, "Model": "Linear",
            "R2": r2_score(y_va, lin_pred_val),
            "R2_train": r2_score(y_tr, lin_pred_train),
            "RMSE": np.sqrt(mean_squared_error(y_va, lin_pred_val)),
            "RMSE_train": np.sqrt(mean_squared_error(y_tr, lin_pred_train)),
            "MAE": mean_absolute_error(y_va, lin_pred_val),
            "MAE_train": mean_absolute_error(y_tr, lin_pred_train)
        })

        # ---------------- 岭回归（带超参数调优）
        ridge_cv = GridSearchCV(
            Ridge(),
            ridge_grid,
            cv=inner_cv,
            scoring='neg_root_mean_squared_error'
        )
        ridge_cv.fit(X_tr, y_tr)
        ridge_best = ridge_cv.best_estimator_
        ridge_pred_val = ridge_best.predict(X_va)
        ridge_pred_train = ridge_best.predict(X_tr)
        all_perf.append({
            "Target": target, "Fold": fold, "Model": "Ridge",
            "R2": r2_score(y_va, ridge_pred_val),
            "R2_train": r2_score(y_tr, ridge_pred_train),
            "RMSE": np.sqrt(mean_squared_error(y_va, ridge_pred_val)),
            "RMSE_train": np.sqrt(mean_squared_error(y_tr, ridge_pred_train)),
            "MAE": mean_absolute_error(y_va, ridge_pred_val),
            "MAE_train": mean_absolute_error(y_tr, ridge_pred_train)
        })

        bp = ridge_cv.best_params_
        bp.update({"Model":"Ridge","Target":target,"Fold":fold})
        all_best_params.append(bp)

        # ---------------- 随机森林
        rf_cv = GridSearchCV(
            RandomForestRegressor(random_state=22),
            rf_grid, cv=inner_cv,
            scoring='neg_root_mean_squared_error'
        )
        rf_cv.fit(X_tr, y_tr)
        rf_best = rf_cv.best_estimator_
        rf_pred_val = rf_best.predict(X_va)
        rf_pred_train = rf_best.predict(X_tr)
        all_perf.append({
            "Target": target, "Fold": fold, "Model": "RF",
            "R2": r2_score(y_va, rf_pred_val),
            "R2_train": r2_score(y_tr, rf_pred_train),
            "RMSE": np.sqrt(mean_squared_error(y_va, rf_pred_val)),
            "RMSE_train": np.sqrt(mean_squared_error(y_tr, rf_pred_train)),
            "MAE": mean_absolute_error(y_va, rf_pred_val),
            "MAE_train": mean_absolute_error(y_tr, rf_pred_train)
        })

        bp = rf_cv.best_params_
        bp.update({"Model":"RF","Target":target,"Fold":fold})
        all_best_params.append(bp)

        # ---------------- XGBoost
        xgb_cv = GridSearchCV(
            xgb.XGBRegressor(random_state=22,
                             use_label_encoder=False,
                             eval_metric='rmse'),
            xgb_grid, cv=inner_cv,
            scoring='neg_root_mean_squared_error'
        )
        xgb_cv.fit(X_tr, y_tr)
        xgb_best = xgb_cv.best_estimator_
        xgb_pred_val = xgb_best.predict(X_va)
        xgb_pred_train = xgb_best.predict(X_tr)
        all_perf.append({
            "Target": target, "Fold": fold, "Model": "XGBoost",
            "R2": r2_score(y_va, xgb_pred_val),
            "R2_train": r2_score(y_tr, xgb_pred_train),
            "RMSE": np.sqrt(mean_squared_error(y_va, xgb_pred_val)),
            "RMSE_train": np.sqrt(mean_squared_error(y_tr, xgb_pred_train)),
            "MAE": mean_absolute_error(y_va, xgb_pred_val),
            "MAE_train": mean_absolute_error(y_tr, xgb_pred_train)
        })

        bp = xgb_cv.best_params_
        bp.update({"Model":"XGBoost","Target":target,"Fold":fold})
        all_best_params.append(bp)

        # ---------------- LightGBM
        lgb_cv = GridSearchCV(
            lgb.LGBMRegressor(random_state=22, verbose=-1),
            lgb_grid, cv=inner_cv,
            scoring='neg_root_mean_squared_error'
        )
        lgb_cv.fit(X_tr, y_tr)
        lgb_best = lgb_cv.best_estimator_
        lgb_pred_val = lgb_best.predict(X_va)
        lgb_pred_train = lgb_best.predict(X_tr)
        all_perf.append({
            "Target": target, "Fold": fold, "Model": "LightGBM",
            "R2": r2_score(y_va, lgb_pred_val),
            "R2_train": r2_score(y_tr, lgb_pred_train),
            "RMSE": np.sqrt(mean_squared_error(y_va, lgb_pred_val)),
            "RMSE_train": np.sqrt(mean_squared_error(y_tr, lgb_pred_train)),
            "MAE": mean_absolute_error(y_va, lgb_pred_val),
            "MAE_train": mean_absolute_error(y_tr, lgb_pred_train)
        })

        bp = lgb_cv.best_params_
        bp.update({"Model":"LightGBM","Target":target,"Fold":fold})
        all_best_params.append(bp)

        # ---------------- GAM
        lam_candidates = [0.1, 0.6, 1, 5]
        best_lam = None
        best_cv_score = -np.inf

        for lam in lam_candidates:
            terms = [s(i, n_splines=5, lam=lam) for i in range(X_tr.shape[1])]
            gam_tmp = LinearGAM(reduce(operator.add, terms))
            scores = cross_val_score(gam_tmp, X_tr.values, y_tr.values,
                                     cv=KFold(5, shuffle=True, random_state=22),
                                     scoring='r2')
            mean_score = np.mean(scores)
            if mean_score > best_cv_score:
                best_cv_score = mean_score
                best_lam = lam

        terms = [s(i, n_splines=5, lam=best_lam) for i in range(X_tr.shape[1])]
        gam = LinearGAM(reduce(operator.add, terms)).fit(X_tr.values, y_tr.values)
        gam_pred_val = gam.predict(X_va.values)
        gam_pred_train = gam.predict(X_tr.values)
        all_perf.append({
            "Target": target, "Fold": fold, "Model": "GAM",
            "R2": r2_score(y_va, gam_pred_val),
            "R2_train": r2_score(y_tr, gam_pred_train),
            "RMSE": np.sqrt(mean_squared_error(y_va, gam_pred_val)),
            "RMSE_train": np.sqrt(mean_squared_error(y_tr, gam_pred_train)),
            "MAE": mean_absolute_error(y_va, gam_pred_val),
            "MAE_train": mean_absolute_error(y_tr, gam_pred_train)
        })

        gam_lam_records.append({
            "Target": target,
            "Fold": fold,
            "BestLam": best_lam,
            "CV_R2": best_cv_score,
            "EDOF": gam.statistics_["edof"]
        })

        # ---------------- 收集所有模型的预测值和真实值（用于散点图）
        model_preds = {
            "Linear": lin_pred_val,
            "Ridge": ridge_pred_val,
            "RF": rf_pred_val,
            "XGBoost": xgb_pred_val,
            "LightGBM": lgb_pred_val,
            "GAM": gam_pred_val
        }

        for m, pred in model_preds.items():
            all_pred.append(pd.DataFrame({
                "Target": target,
                "Fold": fold,
                "Model": m,
                "Observed": y_va.values,
                "Predicted": pred
            }))

        # ---------------- 对三个树模型计算SHAP值（使用验证集）
        # XGBoost
        explainer_xgb = shap.TreeExplainer(xgb_best)
        shap_vals_xgb = explainer_xgb.shap_values(X_va)
        shap_mean_xgb = np.abs(shap_vals_xgb).mean(axis=0)
        for fname, val in zip(feature_names, shap_mean_xgb):
            all_shap.append({
                "Target": target,
                "Fold": fold,
                "Model": "XGBoost",
                "Feature": fname,
                "MeanAbsSHAP": val
            })

        # Random Forest
        explainer_rf = shap.TreeExplainer(rf_best)
        shap_vals_rf = explainer_rf.shap_values(X_va)
        shap_mean_rf = np.abs(shap_vals_rf).mean(axis=0)
        for fname, val in zip(feature_names, shap_mean_rf):
            all_shap.append({
                "Target": target,
                "Fold": fold,
                "Model": "RF",
                "Feature": fname,
                "MeanAbsSHAP": val
            })

        # LightGBM
        explainer_lgb = shap.TreeExplainer(lgb_best)
        shap_vals_lgb = explainer_lgb.shap_values(X_va)
        shap_mean_lgb = np.abs(shap_vals_lgb).mean(axis=0)
        for fname, val in zip(feature_names, shap_mean_lgb):
            all_shap.append({
                "Target": target,
                "Fold": fold,
                "Model": "LightGBM",
                "Feature": fname,
                "MeanAbsSHAP": val
            })

# =========================================================
# 5. 保存合并后的输出文件
# =========================================================

pd.DataFrame(all_perf).to_csv("output_group/model_performance_summary.csv", index=False)
pd.concat(all_pred).to_csv("output_group/scatter_data_all.csv", index=False)
pd.DataFrame(all_shap).to_csv("output_group/shap_all_models.csv", index=False)
pd.DataFrame(all_best_params).to_csv("output_group/best_hyperparameters_all_folds.csv", index=False)
pd.DataFrame(gam_lam_records).to_csv("output_group/gam_lam_records.csv", index=False)

# 保存合并的PCA载荷和方差解释率
pd.DataFrame(all_pca_loadings).to_csv("output_group/pca_loadings_all.csv", index=False)
pd.DataFrame(all_pca_variances).to_csv("output_group/pca_variances_all.csv", index=False)

# =========================================================
# 6. 全局SHAP重要性（按目标和模型分组）
# =========================================================

df_shap = pd.read_csv("output_group/shap_all_models.csv")
df_global_shap = df_shap.groupby(["Target", "Model", "Feature"])["MeanAbsSHAP"].mean().reset_index()
df_global_shap.to_csv("output_group/feature_shap_importance_by_model.csv", index=False)

# 将特征映射到原始分组名称（增加PC2）
group_map = {
    "Sediment_PC1": "Sediment Gradient (PC1)",
    "Sediment_PC2": "Sediment Gradient (PC2)",
    "Major_PC1": "Major Element Gradient (PC1)",
    "Major_PC2": "Major Element Gradient (PC2)",
    "Trace_PC1": "Trace Element Gradient (PC1)",
    "Trace_PC2": "Trace Element Gradient (PC2)",
    "Organic_PC1": "Organic-Depth Gradient (PC1)",
    "Organic_PC2": "Organic-Depth Gradient (PC2)"
}

df_global_shap["Group"] = df_global_shap["Feature"].map(group_map)
df_group_shap = df_global_shap.groupby(["Target", "Model", "Group"])["MeanAbsSHAP"].sum().reset_index()
df_group_shap["Contribution"] = df_group_shap.groupby(["Target", "Model"])["MeanAbsSHAP"].transform(lambda x: x / x.sum())
df_group_shap.to_csv("output_group/shap_group_contributions_by_model.csv", index=False)

# =========================================================
# 7a. 对原始特征（标准化后）计算方差膨胀因子（VIF）
# =========================================================
# 标准化所有原始特征
scaler_all = StandardScaler()
X_original_scaled = scaler_all.fit_transform(df[feature_cols])
X_original_scaled = pd.DataFrame(X_original_scaled, columns=feature_cols)

# 计算每个原始特征的VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = feature_cols
vif_data["VIF"] = [variance_inflation_factor(X_original_scaled.values, i)
                   for i in range(X_original_scaled.shape[1])]
vif_data.to_csv("output_group/vif_original_features.csv", index=False)

# =========================================================
# 7b. 对处理后的特征（PC1）计算方差膨胀因子（VIF）
# =========================================================
# 在整个数据集上构建PC1特征（不涉及交叉验证，仅用于VIF分析）
pca_loadings_temp = []  # 临时存储，不需要保存
pca_variances_temp = [] # 临时存储，不需要保存
# 修正：使用 fold='global' 避免与交叉验证的折数混淆
X_pc_full, _ = build_pca(df, df, fold='global', target="FULL",
                          all_loadings=pca_loadings_temp,
                          all_variances=pca_variances_temp)
# X_pc_full 包含了四个PC1特征，直接计算VIF
vif_pc_data = pd.DataFrame()
vif_pc_data["Feature"] = X_pc_full.columns
vif_pc_data["VIF"] = [variance_inflation_factor(X_pc_full.values, i)
                      for i in range(X_pc_full.shape[1])]
vif_pc_data.to_csv("output_group/vif_pc_features.csv", index=False)

# =========================================================
# 8. 自动生成论文表格（英文命名）
# =========================================================

# 表1：各区域样本数量分布
df["Area"].value_counts().reset_index().rename(
    columns={"index": "Area", "Area": "Count"}
).to_csv("output_group/Table1_Area_Distribution.csv", index=False)

# 表2：各目标变量下各模型的平均性能（跨fold取均值）
perf_mean = pd.DataFrame(all_perf).groupby(["Target", "Model"]).mean().reset_index()
perf_mean.to_csv("output_group/Table2_Model_Performance_Summary.csv", index=False)

# 表3：每个目标变量的最佳模型（基于平均R2最高）
best_model = perf_mean.loc[perf_mean.groupby("Target")["R2"].idxmax()]
best_model.to_csv("output_group/Table3_Best_Model_per_Target.csv", index=False)

# 表4：每个目标变量下SHAP重要性排名前5的特征（以XGBoost为例）
df_shap_xgb = df_global_shap[df_global_shap["Model"] == "XGBoost"]
top5_shap = df_shap_xgb.sort_values(["Target", "MeanAbsSHAP"], ascending=[True, False]) \
                       .groupby("Target").head(5)
top5_shap.to_csv("output_group/Table4_SHAP_Top5_Features_XGBoost.csv", index=False)

# =========================================================
# 9. SHAP稳定性分析
# =========================================================

df_shap_full = pd.read_csv("output_group/shap_all_models.csv")

# 计算Mean ± SD
shap_stability = df_shap_full.groupby(
    ["Target","Model","Feature"]
)["MeanAbsSHAP"].agg(["mean","std"]).reset_index()

shap_stability["CV"] = shap_stability["std"] / shap_stability["mean"]

shap_stability.to_csv("output_group/shap_stability_mean_sd_cv.csv", index=False)

# Top变量出现频率
top_freq_records = []

for (target, model), sub in df_shap_full.groupby(["Target","Model"]):
    for fold in sub["Fold"].unique():
        fold_data = sub[sub["Fold"] == fold]
        top_features = fold_data.sort_values(
            "MeanAbsSHAP", ascending=False
        )["Feature"].head(2).tolist()

        for f in top_features:
            top_freq_records.append({
                "Target": target,
                "Model": model,
                "Feature": f
            })

df_top_freq = pd.DataFrame(top_freq_records)
freq_table = df_top_freq.groupby(
    ["Target","Model","Feature"]
).size().reset_index(name="Top_Frequency")

freq_table.to_csv("output_group/shap_top_frequency.csv", index=False)


# =========================================================
# 10. 超参数稳定性统计
# =========================================================

df_params = pd.read_csv("output_group/best_hyperparameters_all_folds.csv")

param_summary = df_params.groupby(["Model","Target"]).agg(
    ["mean","std"]
)

param_summary.to_csv("output_group/hyperparameter_stability_summary.csv")


# =========================================================
# 11. PCA稳定性统计（加载量稳定性，区分PC1/PC2）
# =========================================================

df_loadings = pd.read_csv("output_group/pca_loadings_all.csv")

pca_stability = df_loadings.groupby(
    ["Target", "Group", "PC", "Variable"]
)["Loading"].agg(["mean", "std"]).reset_index()

pca_stability["CV"] = abs(pca_stability["std"] / pca_stability["mean"])

pca_stability.to_csv("output_group/pca_loading_stability.csv", index=False)

# =========================================================
# 12. GAM稳定性分析
# =========================================================

df_gam = pd.read_csv("output_group/gam_lam_records.csv")

gam_stability = df_gam.groupby("Target").agg(
    {"BestLam":["mean","std"],
     "EDOF":["mean","std"],
     "CV_R2":["mean","std"]}
)

gam_stability.to_csv("output_group/gam_stability_summary.csv")


# =========================================================
# 13. OOF预测整合（最佳模型）
# =========================================================

df_perf = pd.read_csv("output_group/model_performance_summary.csv")
df_pred = pd.read_csv("output_group/scatter_data_all.csv")

best_models = perf_mean.loc[
    perf_mean.groupby("Target")["R2"].idxmax()
][["Target","Model"]]

oof_records = []

for _, row in best_models.iterrows():
    t = row["Target"]
    m = row["Model"]

    subset = df_pred[
        (df_pred["Target"]==t) &
        (df_pred["Model"]==m)
    ]

    oof_records.append(subset)

pd.concat(oof_records).to_csv(
    "output_group/oof_predictions_best_models.csv",
    index=False
)


# =========================================================
# 14. 残差分布数据
# =========================================================

oof_df = pd.read_csv("output_group/oof_predictions_best_models.csv")
oof_df["Residual"] = oof_df["Observed"] - oof_df["Predicted"]

oof_df.to_csv("output_group/oof_with_residuals.csv", index=False)


# =========================================================
# 15. 模型过拟合Gap分析（仅验证集）
# =========================================================

gap_records = []

for target in target_names:
    sub = pd.DataFrame(all_perf)
    sub = sub[sub["Target"]==target]

    for model in sub["Model"].unique():
        model_sub = sub[sub["Model"]==model]

        gap_records.append({
            "Target": target,
            "Model": model,
            "R2_Mean": model_sub["R2"].mean(),
            "R2_SD": model_sub["R2"].std()
        })

pd.DataFrame(gap_records).to_csv(
    "output_group/model_gap_analysis.csv",
    index=False
)


# =========================================================
# 16. 变量重要性变异系数（稳健性）
# =========================================================

importance_cv = shap_stability.copy()
importance_cv.rename(columns={"CV":"Importance_CV"}, inplace=True)

importance_cv.to_csv(
    "output_group/feature_importance_variation.csv",
    index=False
)

# =========================================================
# 17. SHAP 排名一致性（Spearman）
# =========================================================

df_shap_full = pd.read_csv("output_group/shap_all_models.csv")

rank_records = []

for (target, model), sub in df_shap_full.groupby(["Target","Model"]):
    pivot = sub.pivot_table(
        index="Feature",
        columns="Fold",
        values="MeanAbsSHAP"
    )

    folds = pivot.columns.tolist()

    for i in range(len(folds)):
        for j in range(i+1, len(folds)):
            f1, f2 = folds[i], folds[j]
            corr, _ = spearmanr(pivot[f1], pivot[f2])
            rank_records.append({
                "Target": target,
                "Model": model,
                "Fold1": f1,
                "Fold2": f2,
                "SpearmanR": corr
            })

df_rank_stability = pd.DataFrame(rank_records)
df_rank_stability.to_csv("output_group/shap_rank_spearman_stability.csv", index=False)


# =========================================================
# 18. Train-Test Gap 汇总（基于已保存的训练和验证指标）
# =========================================================

df_perf_full = pd.read_csv("output_group/model_performance_summary.csv")

gap_records = []

for (target, model), sub in df_perf_full.groupby(["Target","Model"]):
    if "R2_train" in sub.columns:
        gap = sub["R2_train"].mean() - sub["R2"].mean()
        gap_records.append({
            "Target": target,
            "Model": model,
            "Train_R2_mean": sub["R2_train"].mean(),
            "Test_R2_mean": sub["R2"].mean(),
            "R2_Gap": gap
        })

pd.DataFrame(gap_records).to_csv(
    "output_group/train_test_gap_analysis.csv",
    index=False
)

# =========================================================
# 19. Cross-validation稳定性统计
# =========================================================

df_perf = pd.read_csv("output_group/model_performance_summary.csv")

cv_summary = df_perf.groupby(
    ["Target","Model"]
).agg(
    R2_mean=("R2","mean"),
    R2_sd=("R2","std"),
    RMSE_mean=("RMSE","mean"),
    RMSE_sd=("RMSE","std"),
    MAE_mean=("MAE","mean"),
    MAE_sd=("MAE","std")
).reset_index()

cv_summary.to_csv(
    "output_group/model_performance_cv_summary.csv",
    index=False
)

# =========================================================
# 20. 最佳模型预测散点图数据
# =========================================================

df_best = pd.read_csv("output_group/Table3_Best_Model_per_Target.csv")
df_pred = pd.read_csv("output_group/scatter_data_all.csv")

scatter_records = []

for _, row in df_best.iterrows():
    t = row["Target"]
    m = row["Model"]

    sub = df_pred[
        (df_pred["Target"]==t) &
        (df_pred["Model"]==m)
    ]

    scatter_records.append(sub)

best_scatter = pd.concat(scatter_records)

best_scatter.to_csv(
    "output_group/best_model_scatter_data.csv",
    index=False
)

# =========================================================
# 21. 残差统计汇总
# =========================================================

df_res = pd.read_csv("output_group/oof_with_residuals.csv")

residual_summary = df_res.groupby("Target").agg(
    Residual_mean=("Residual","mean"),
    Residual_sd=("Residual","std"),
    Residual_abs_mean=("Residual", lambda x: np.mean(np.abs(x)))
).reset_index()

residual_summary.to_csv(
    "output_group/residual_distribution_summary.csv",
    index=False
)

print("所有分析已完成")