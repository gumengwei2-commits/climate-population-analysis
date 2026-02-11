# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------- 全局字体 ----------------
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 18

# --------------------------- User settings ---------------------------
input_path = r"D:/0_@博士/1-论文/3-小论文——气候变化与出生率/1-数据/4-整体变量/整体变量（英文代码版）.xlsx"
table_dir = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\2-结果\10.20 最终结果\异质性\收入异质性"
fig_dir = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\1-论文图\0-10.20 论文最终图\收入异质性\交互因子"

os.makedirs(table_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# ---------------- 变量设置 ----------------
climate_vars = ['TXx','TX90p','HW','Rx1day','R10','CWD','CDD','SPI','CDWE','CWWE']
control_vars = ['URB','GDP','IMR','UNE','FLFP','NMI','FPI','FEY']
target_column = 'TFR'
country_column = 'country'
year_column = 'year'
all_vars = climate_vars + control_vars

# 收入分组
high_income = ['Brunei','Singapore']
upper_middle_income = ['Indonesia','Malaysia','Thailand']
lower_middle_income = ['Myanmar','Cambodia','Vietnam','Laos','Philippines','Timor-Leste']

income_groups = {
    'High Income': high_income,
    'Upper Middle Income': upper_middle_income,
    'Lower Middle Income': lower_middle_income
}

# ---------------- 读取数据 ----------------
df = pd.read_excel(input_path)

# ---------------- 模型训练与评估函数 ----------------
def train_xgb_model(X, y, group_name):
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 超参数网格
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9]
    }

    # 模型 + 十折交叉验证
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='r2',
        cv=10,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # 最优模型
    model = grid_search.best_estimator_
    model.fit(X_train, y_train)

    # 性能评估
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)

    print(f"✅ {group_name}: Best Params = {grid_search.best_params_}")
    print(f"Group={group_name} | CV R²={grid_search.best_score_:.3f} | Train R²={r2_train:.3f} | "
          f"Test R²={r2_test:.3f} | RMSE={rmse_test:.3f} | MAE={mae_test:.3f}")

    # 保存性能结果
    perf_df = pd.DataFrame({
        "数据集": ["十折交叉验证(CV)", "训练集", "测试集"],
        "R²": [grid_search.best_score_, r2_train, r2_test],
        "RMSE": [np.nan, np.nan, rmse_test],
        "MAE": [np.nan, np.nan, mae_test]
    })
    perf_path = os.path.join(table_dir, f"TFR_XGB_performance_{group_name.replace(' ','_')}.csv")
    perf_df.to_csv(perf_path, index=False, encoding='utf-8-sig')

    return model

# ---------------- 主模型训练（全样本） ----------------
X = df[all_vars]
y = df[target_column]
full_model = train_xgb_model(X, y, "All Countries")

# ---------------- SHAP 交互分析：按收入分组 ----------------
income_interactions = {}

for group_name, countries in income_groups.items():
    df_group = df[df[country_column].isin(countries)]
    explainer = shap.TreeExplainer(full_model)
    shap_inter = explainer.shap_interaction_values(df_group[all_vars])

    inter_df = pd.DataFrame(np.abs(shap_inter).mean(0), index=all_vars, columns=all_vars)
    inter_ce = inter_df.loc[climate_vars, control_vars]
    income_interactions[group_name] = inter_ce * 1000  # 放大值便于可视化

# ---------------- 热力图绘制（三组并排，标题与标签加粗） ----------------
fig, axes = plt.subplots(1, 3, figsize=(30, 10))

vmin = min(income_interactions[g].min().min() for g in income_interactions)
vmax = max(income_interactions[g].max().max() for g in income_interactions)

for ax, (group_name, inter_matrix) in zip(axes, income_interactions.items()):
    sns.heatmap(
        inter_matrix, cmap="Blues", annot=True, fmt=".1f",
        cbar=True, vmin=vmin, vmax=vmax, ax=ax,
        annot_kws={"fontsize": 18, "fontweight": "bold"}
    )

    # 加粗标题
    ax.set_title(f"{group_name}", fontsize=22, fontweight='bold', pad=10)


    # 加粗坐标刻度文字
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=18, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=18, fontweight='bold')

plt.tight_layout()

# 保存整张图
output_path = os.path.join(fig_dir, "TFR_income_group_heatmaps.png")
plt.savefig(output_path, dpi=300)
plt.show()

print("✅ 三组交互热力图（标题与坐标标签加粗）已保存：", output_path)

