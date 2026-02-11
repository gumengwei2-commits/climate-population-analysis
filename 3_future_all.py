# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# -------------------------------
# 1. 数据路径（按你环境修改）
# -------------------------------
pop_path = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\4-未来预测\1-历史人口生育力数据.xlsx"
climate_path = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\4-未来预测\2-历史气候数据.xlsx"
ssp126_folder = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\4-未来预测\3-未来气候数据\ssp126"
ssp585_folder = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\4-未来预测\3-未来气候数据\ssp585"
output_path = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\2-结果\10.20 最终结果\未来预测结果.xlsx"

# -------------------------------
# 2. 加载历史数据并合并（假设包含 'ID','country','year','TFR','CBR' 等）
# -------------------------------
pop_df = pd.read_excel(pop_path)
climate_df = pd.read_excel(climate_path)
data = pd.merge(pop_df, climate_df, on=["ID", "country", "year"])
print("历史数据：", data.shape)

# -------------------------------
# 3. 加载未来数据（函数）
# -------------------------------
def load_future(folder, scenario):
    all_files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.endswith(".xlsx") or f.endswith(".csv")]
    print(f"{scenario} 找到文件数: {len(all_files)}")
    if not all_files:
        raise FileNotFoundError(f"⚠️ 没有找到任何 .xlsx 或 .csv 文件，请检查路径: {folder}")

    df_list = []
    for f in all_files:
        print(f"读取: {f}")
        if f.endswith(".xlsx"):
            df = pd.read_excel(f)
        else:
            df = pd.read_csv(f)
        df["scenario"] = scenario
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

future_ssp126 = load_future(ssp126_folder, "SSP1-2.6")
future_ssp585 = load_future(ssp585_folder, "SSP5-8.5")
print("未来 SSP1-2.6:", future_ssp126.shape)
print("未来 SSP5-8.5:", future_ssp585.shape)

# -------------------------------
# 4. 定义特征与目标（根据你的表列名调整）
# -------------------------------
# 假设数据中有 'TFR' 和 'CBR' 两个目标列；去除非特征列
exclude_cols = ["ID", "country", "year", "TFR", "CBR", "scenario"]  # 根据实际调整
features = [c for c in data.columns if c not in exclude_cols]
print("使用特征数:", len(features), features[:10])

# -------------------------------
# 5. 建模函数：XGBoost + GridSearchCV
# -------------------------------
def train_xgb_model(X, y, param_grid=None, cv=10, random_state=42):
    if param_grid is None:
        param_grid = {
            'n_estimators':[100,200,300], 'max_depth':[3,5], 'learning_rate':[0.05,0.1],
            'subsample':[0.7,0.9], 'colsample_bytree':[0.7,0.9], 'reg_alpha':[0,0.1,0.5], 'reg_lambda':[1,1.5]
        }
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state, n_jobs=-1)
    grid_search = GridSearchCV(xgb_model, param_grid, scoring='r2', cv=cv, n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    best = grid_search.best_estimator_
    # 再用全部训练数据拟合一次（可选）
    best.fit(X, y)
    return best, grid_search

# -------------------------------
# 6. 为 TFR 和 CBR 分别训练模型（先划分训练/测试）
# -------------------------------
# 注意：若你想按国家/年份分层采样请修改 split 策略
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
X_train = train_df[features]; X_test = test_df[features]
y_train_tfr = train_df["TFR"]; y_test_tfr = test_df["TFR"]
y_train_cbr = train_df["CBR"]; y_test_cbr = test_df["CBR"]

print("训练集样本:", X_train.shape, "测试集样本:", X_test.shape)

# 训练 TFR 模型
tfr_model, tfr_grid = train_xgb_model(X_train, y_train_tfr)
# 训练 CBR 模型
cbr_model, cbr_grid = train_xgb_model(X_train, y_train_cbr)

# ---------------------- 模型评估 ----------------------
def eval_model(model, X_tr, y_tr, X_te, y_te):
    y_tr_pred = model.predict(X_tr)
    y_te_pred = model.predict(X_te)
    r2_tr = r2_score(y_tr, y_tr_pred)
    r2_te = r2_score(y_te, y_te_pred)
    rmse_te = np.sqrt(mean_squared_error(y_te, y_te_pred))
    mae_te = mean_absolute_error(y_te, y_te_pred)
    mse_te = mean_squared_error(y_te, y_te_pred)
    return {"r2_train": r2_tr, "r2_test": r2_te, "rmse_test": rmse_te, "mae_test": mae_te, "mse_test": mse_te}

tfr_perf = eval_model(tfr_model, X_train, y_train_tfr, X_test, y_test_tfr)
cbr_perf = eval_model(cbr_model, X_train, y_train_cbr, X_test, y_test_cbr)
print("TFR perf:", tfr_perf)
print("CBR perf:", cbr_perf)

# -------------------------------
# 7. SHAP 分析（对未来情景）
# -------------------------------
def shap_future(model, future_df, features, target_name, scenario):
    X_future = future_df[features].copy()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_future)
    # 兼容 shap 不同版本：如果返回 list（比如多输出），取第一个元素
    if isinstance(shap_values, list):
        shap_arr = shap_values[0]
    else:
        shap_arr = shap_values
    shap_df = pd.DataFrame(shap_arr, columns=features)
    shap_df["year"] = future_df["year"].values
    shap_df["指标"] = target_name
    shap_df["scenario"] = scenario
    return shap_df

summary_tfr_126 = shap_future(tfr_model, future_ssp126, features, "TFR", "SSP1-2.6")
summary_cbr_126 = shap_future(cbr_model, future_ssp126, features, "CBR", "SSP1-2.6")
summary_tfr_585 = shap_future(tfr_model, future_ssp585, features, "TFR", "SSP5-8.5")
summary_cbr_585 = shap_future(cbr_model, future_ssp585, features, "CBR", "SSP5-8.5")

summary_all = pd.concat(
    [summary_tfr_126, summary_cbr_126, summary_tfr_585, summary_cbr_585],
    ignore_index=True
)

# -------------------------------
# 8. 按 period 平均（注意 bins 与 labels 的含义）
# -------------------------------
# 这里 bins 对应 (2025,2035], (2035,2055], (2055,2095]
bins = [2025, 2035, 2055, 2095]
periods = ["近期(2026-2035)", "中期(2036-2055)", "远期(2056-2095)"]
summary_all["period"] = pd.cut(summary_all["year"], bins=bins, labels=periods)

# 按 period + 指标 + scenario 平均
summary_pivot = summary_all.groupby(["scenario", "指标", "period"])[features].mean().reset_index()

# 只保留正负符号，保留三位小数（字符串）
for f in features:
    summary_pivot[f] = summary_pivot[f].apply(lambda x: f"{x:+.3f}")

# -------------------------------
# 9. 保存 Excel（把所有 sheet 写在同一个 writer 块内）
# -------------------------------
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    summary_pivot.to_excel(writer, sheet_name="未来预测_SHAP", index=False)
    # 模型性能与最优参数
    perf_df = pd.DataFrame([
        ["TFR", tfr_grid.best_params_, round(tfr_perf["mse_test"], 3), round(tfr_perf["r2_test"], 3)],
        ["CBR", cbr_grid.best_params_, round(cbr_perf["mse_test"], 3), round(cbr_perf["r2_test"], 3)]
    ], columns=["指标", "最佳参数", "MSE_test", "R²_test"])
    perf_df.to_excel(writer, sheet_name="模型性能", index=False)
    # 可选：保存 grid search 最佳参数详情
    pd.DataFrame([tfr_grid.best_params_]).to_excel(writer, sheet_name="TFR_BestParams", index=False)
    pd.DataFrame([cbr_grid.best_params_]).to_excel(writer, sheet_name="CBR_BestParams", index=False)

print(f"\n✅ 表格已生成：{output_path}")





