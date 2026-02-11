# -*- coding: utf-8 -*-
"""
XGBoost + SHAP Interaction Analysis
Mainland vs Island | TFR and Climate–Socioeconomic Factors
Ensured Times New Roman font display even under Chinese path
Author: [Your Name]
Updated: 2025-10-20
"""

import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from pathlib import Path

# --------------------- 全局设置 ---------------------
warnings.filterwarnings('ignore')

# ✅ 强制加载 Times New Roman（即使路径中有中文也能生效）
font_path = r"C:\Windows\Fonts\times.ttf"  # 也可以换成 timesi.ttf (斜体)
if not os.path.exists(font_path):
    raise FileNotFoundError("❌ Times New Roman 字体文件未找到，请检查路径 C:\\Windows\\Fonts\\times.ttf")

font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
font_name = prop.get_name()

mpl.rcParams['font.family'] = font_name
mpl.rcParams['font.sans-serif'] = [font_name]
mpl.rcParams['font.serif'] = [font_name]
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 18
sns.set_style("whitegrid", {"font.family": [font_name]})

# ✅ 字体验证
print(f"✅ 当前使用字体: {font_name}")

# --------------------- 用户设置 ---------------------
input_path = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\1-数据\4-整体变量\整体变量（英文代码版）.xlsx"
output_dir = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\1-论文图\0-10.20 论文最终图\地理异质性\交互因子"
os.makedirs(output_dir, exist_ok=True)

climate_vars = ['TXx','TX90p','HW','Rx1day','R10','CWD','CDD','SPI','CDWE','CWWE']
control_vars = ['URB','GDP','IMR','UNE','FLFP','NMI','FPI','FEY']
target_column = 'TFR'
country_column = 'country'
year_column = 'year'
all_vars = climate_vars + control_vars

mainland = ['Myanmar','Thailand','Laos','Cambodia','Vietnam']
island = ['Indonesia','Philippines','Malaysia','Brunei','Singapore','Timor-Leste']

# --------------------- 读取数据 ---------------------
if not Path(input_path).exists():
    raise FileNotFoundError(f"❌ 找不到输入文件: {input_path}")

df = pd.read_excel(input_path)
df = df.dropna(subset=[target_column])
df = df[~df[all_vars].isna().all(axis=1)].copy()

# --------------------- 模型参数 ---------------------
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 1.5]
}
RANDOM_SEED = 42

# --------------------- 训练函数 ---------------------
def train_xgb(X, y):
    """训练 XGBoost 模型并返回最佳模型"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    model = XGBRegressor(objective='reg:squarederror', random_state=RANDOM_SEED)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='r2',
        cv=10,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"✅ Model trained | R²={r2:.3f} | RMSE={rmse:.3f}")
    return best_model

# --------------------- 按地理组训练 ---------------------
print("\n🚀 Training model for Mainland...")
model_main = train_xgb(df[df[country_column].isin(mainland)][all_vars],
                       df[df[country_column].isin(mainland)][target_column])

print("\n🚀 Training model for Island...")
model_island = train_xgb(df[df[country_column].isin(island)][all_vars],
                         df[df[country_column].isin(island)][target_column])

# --------------------- SHAP Interaction ---------------------
print("\n📊 Calculating SHAP interaction values...")

explainer_main = shap.TreeExplainer(model_main)
explainer_island = shap.TreeExplainer(model_island)

df_mainland = df[df[country_column].isin(mainland)]
df_island = df[df[country_column].isin(island)]

shap_inter_main = explainer_main.shap_interaction_values(df_mainland[all_vars])
shap_inter_island = explainer_island.shap_interaction_values(df_island[all_vars])

inter_main_df = pd.DataFrame(np.abs(shap_inter_main).mean(0), index=all_vars, columns=all_vars)
inter_island_df = pd.DataFrame(np.abs(shap_inter_island).mean(0), index=all_vars, columns=all_vars)

inter_main_ce = inter_main_df.loc[climate_vars, control_vars]
inter_island_ce = inter_island_df.loc[climate_vars, control_vars]

# --------------------- 热力图绘制 ---------------------
scale_factor = 1000
inter_diff_scaled = (inter_main_ce - inter_island_ce) * scale_factor

# 确保 Times New Roman 字体加载
prop = font_manager.FontProperties(family='Times New Roman', weight='bold')

plt.figure(figsize=(12, 8))
ax = sns.heatmap(inter_diff_scaled, cmap="coolwarm", center=0, annot=True, fmt=".1f", cbar=True)
plt.title("Factors Interaction Differences on TFR by Country type (Mainland-Island)",
          fontproperties=prop, fontsize=16, weight='bold', pad=20)
plt.xticks(fontproperties=prop, rotation=30, ha='right')
plt.yticks(fontproperties=prop, rotation=0)
plt.tight_layout()

save_path = os.path.join(output_dir, "TFR.tiff")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"\n✅ All done!\n📂 Output: {output_dir}\n🖼 Heatmap saved to: {save_path}")

