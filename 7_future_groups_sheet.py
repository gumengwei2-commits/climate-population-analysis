# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# -------------------------------
# 1. 数据路径
# -------------------------------
pop_path = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\4-未来预测\1-历史人口生育力数据.xlsx"
climate_path = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\4-未来预测\2-历史气候数据.xlsx"
ssp126_folder = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\4-未来预测\3-未来气候数据\ssp126"
ssp585_folder = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\4-未来预测\3-未来气候数据\ssp585"

output_dir = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\2-结果\10.20 最终结果"
os.makedirs(output_dir, exist_ok=True)

output_path_geo = os.path.join(output_dir, "未来预测_SHAP_geo.xlsx")
output_path_income = os.path.join(output_dir, "未来预测_SHAP_income.xlsx")
output_table_geo = output_path_geo.replace(".xlsx", "_table.xlsx")
output_table_income = output_path_income.replace(".xlsx", "_table.xlsx")

# -------------------------------
# 2. 加载历史数据
# -------------------------------
pop_df = pd.read_excel(pop_path)
climate_df = pd.read_excel(climate_path)
data = pd.merge(pop_df, climate_df, on=["ID", "country", "year"])
print("✅ 历史数据加载完成：", data.shape)

# 定义特征
features = [c for c in data.columns if c not in ["ID", "country", "year", "TFR", "CBR"]]

# -------------------------------
# 3. 加载未来数据
# -------------------------------
def load_future(folder, scenario):
    all_files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.endswith(".xlsx") or f.endswith(".csv")]
    df_list = []
    for f in all_files:
        df = pd.read_excel(f) if f.endswith(".xlsx") else pd.read_csv(f)
        df["scenario"] = scenario
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

future_ssp126 = load_future(ssp126_folder, "SSP1-2.6")
future_ssp585 = load_future(ssp585_folder, "SSP5-8.5")
print("✅ 未来气候数据加载完成")

# -------------------------------
# 4. 模型训练函数
# -------------------------------
param_grid = {
    'n_estimators': [200],
    'max_depth': [3, 5],
    'learning_rate': [0.05],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1]
}

def train_model(target_name):
    X = data[features]
    y = data[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring='r2', cv=5, n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)

    report = pd.DataFrame({
        "数据集": ["CV(5折)", "训练集", "测试集"],
        "R²": [grid_search.best_score_, r2_train, r2_test],
        "RMSE": [np.nan, np.nan, rmse_test],
        "MAE": [np.nan, np.nan, mae_test]
    })
    report_path = os.path.join(output_dir, f"{target_name}_XGB_Model_Performance.xlsx")
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        report.to_excel(writer, sheet_name="Performance", index=False)
        pd.DataFrame([grid_search.best_params_]).to_excel(writer, sheet_name="Best_Params", index=False)
    print(f"✅ {target_name} 模型训练完成，性能表保存：{report_path}")
    return best_model

tfr_model = train_model("TFR")
cbr_model = train_model("CBR")

# -------------------------------
# 5. SHAP 分析函数
# -------------------------------
def shap_future(model, future_df, target_name, scenario):
    X_future = future_df[features].copy()
    explainer = shap.Explainer(model, X_future)
    shap_values = explainer(X_future).values
    shap_df = pd.DataFrame(shap_values, columns=features)
    shap_df["year"] = future_df["year"].values
    shap_df["指标"] = target_name
    shap_df["scenario"] = scenario
    shap_df["country"] = future_df["country"].values
    return shap_df

summary_tfr_126 = shap_future(tfr_model, future_ssp126, "TFR", "SSP1-2.6")
summary_cbr_126 = shap_future(cbr_model, future_ssp126, "CBR", "SSP1-2.6")
summary_tfr_585 = shap_future(tfr_model, future_ssp585, "TFR", "SSP5-8.5")
summary_cbr_585 = shap_future(cbr_model, future_ssp585, "CBR", "SSP5-8.5")

summary_all = pd.concat(
    [summary_tfr_126, summary_cbr_126, summary_tfr_585, summary_cbr_585],
    ignore_index=True
)

# -------------------------------
# 6. 地理分组（mainland / island ）
# -------------------------------
mainland = ['Myanmar', 'Thailand', 'Laos', 'Cambodia', 'Vietnam']
island = ['Indonesia', 'Philippines', 'Malaysia', 'Brunei', 'Singapore', 'Timor-Leste']

summary_all['region_type'] = summary_all['country'].apply(
    lambda x: 'mainland' if x in mainland else ('island' if x in island else 'other')
)

# -------------------------------
# 7. 收入分组（high / upper_middle / lower_middle）
# -------------------------------
high_income = ['Brunei', 'Singapore']
upper_middle_income = ['Indonesia', 'Malaysia', 'Thailand']
lower_middle_income = ['Myanmar', 'Cambodia', 'Vietnam', 'Laos', 'Philippines', 'Timor-Leste']

def assign_income(country):
    if country in high_income:
        return 'high_income'
    elif country in upper_middle_income:
        return 'upper_middle_income'
    elif country in lower_middle_income:
        return 'lower_middle_income'
    else:
        return 'other'

summary_all['income_group'] = summary_all['country'].apply(assign_income)

# -------------------------------
# 8. 宗教分组（buddhist / non_buddhist）
# -------------------------------
buddhist = ['Thailand','Cambodia','Laos','Myanmar']
non_buddhist = ['Indonesia','Brunei','Malaysia','Philippines','Timor-Leste','Singapore','Vietnam']

summary_all['religion_group'] = summary_all['country'].apply(
    lambda x: 'buddhist' if x in buddhist else ('non_buddhist' if x in non_buddhist else 'other')
)

# -------------------------------
# 10. 按时期分组
# -------------------------------
bins = [2025, 2035, 2055, 2095]
periods = ["近期(2026-2035)", "中期(2036-2055)", "远期(2056-2095)"]
summary_all["period"] = pd.cut(summary_all["year"], bins=bins, labels=periods)

# -------------------------------
# 11. SHAP 汇总输出
# -------------------------------
def summarize_shap(group_col, save_path_excel):
    os.makedirs(os.path.dirname(save_path_excel), exist_ok=True)
    summary_group = summary_all.groupby(
        ["scenario", "指标", "period", group_col], observed=True
    )[features].mean().reset_index()
    summary_long = summary_group.melt(
        id_vars=["scenario", "指标", "period", group_col],
        value_vars=features,
        var_name="Climate_Factor",
        value_name="SHAP_Value"
    )
    summary_long["SHAP_Value"] = summary_long["SHAP_Value"].astype(float)
    summary_long.to_excel(save_path_excel, index=False)
    print(f"✅ {group_col} SHAP 汇总表已保存：{save_path_excel}")
    return summary_long

shap_geo_long = summarize_shap("region_type", output_path_geo)
shap_income_long = summarize_shap("income_group", output_path_income)

# -------------------------------
# 12. 生成对比表
# -------------------------------
def shap_to_table(summary_long, group_col, save_path_excel):
    table = summary_long.groupby(
        ["scenario", "指标", "period", group_col, "Climate_Factor"],
        observed=True
    )["SHAP_Value"].mean().reset_index()

    table_pivot = table.pivot_table(
        index=["scenario", "指标", "period", group_col],
        columns="Climate_Factor",
        values="SHAP_Value"
    ).reset_index()

    ordered_features = ["TXx", "TX90p", "HW", "Rx1day", "R10", "CWD", "CDD", "SPI", "CDWE", "CWWE"]
    valid_cols = [f for f in ordered_features if f in table_pivot.columns]
    table_pivot = table_pivot[["scenario", "指标", "period", group_col] + valid_cols]

    table_pivot.to_excel(save_path_excel, index=False)
    print(f"✅ {group_col} SHAP 对比表已保存（顺序已固定）：{save_path_excel}")
    return table_pivot

shap_to_table(shap_geo_long, "region_type", output_table_geo)
shap_to_table(shap_income_long, "income_group", output_table_income)

print("\n🎉 全流程运行完成！所有结果已保存。")



