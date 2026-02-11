# -*- coding: utf-8 -*-
"""
XGBoost + SHAP Interaction Analysis
Buddhist vs Non-Buddhist | TFR and Climate–Socioeconomic Factors
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

# Force load Times New Roman
font_path = r"C:\Windows\Fonts\times.ttf"
if not os.path.exists(font_path):
    raise FileNotFoundError("❌ Times New Roman font not found at C:\\Windows\\Fonts\\times.ttf")

font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
font_name = prop.get_name()

mpl.rcParams['font.family'] = font_name
mpl.rcParams['font.sans-serif'] = [font_name]
mpl.rcParams['font.serif'] = [font_name]
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 14
sns.set_style("whitegrid", {"font.family": [font_name]})

print(f"✅ Using font: {font_name}")

# --------------------- 用户设置 ---------------------
input_path = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\1-数据\4-整体变量\整体变量（英文代码版）.xlsx"
output_dir = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\1-论文图\0-10.20 论文最终图\宗教异质性\交互因子"
os.makedirs(output_dir, exist_ok=True)

climate_vars = ['TXx','TX90p','HW','Rx1day','R10','CWD','CDD','SPI','CDWE','CWWE']
control_vars = ['URB','GDP','IMR','UNE','FLFP','NMI','FPI','FEY']
target_column = 'CBR'
country_column = 'country'
year_column = 'year'
all_vars = climate_vars + control_vars

# --------------------- Religion Groups ---------------------
buddhist = ['Thailand','Cambodia','Laos','Myanmar']
non_buddhist = ['Indonesia','Brunei','Malaysia','Philippines','Timor-Leste','Singapore','Vietnam']

# --------------------- 读取数据 ---------------------
if not Path(input_path).exists():
    raise FileNotFoundError(f"❌ Input file not found: {input_path}")

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

# --------------------- XGBoost 训练函数 ---------------------
def train_xgb(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
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

# --------------------- 按宗教组训练 ---------------------
print("\n🚀 Training model for Buddhist region...")
model_bud = train_xgb(df[df[country_column].isin(buddhist)][all_vars],
                      df[df[country_column].isin(buddhist)][target_column])

print("\n🚀 Training model for Non-Buddhist region...")
model_non = train_xgb(df[df[country_column].isin(non_buddhist)][all_vars],
                      df[df[country_column].isin(non_buddhist)][target_column])

# --------------------- SHAP Interaction ---------------------
print("\n📊 Calculating SHAP interaction values...")

explainer_bud = shap.TreeExplainer(model_bud)
explainer_non = shap.TreeExplainer(model_non)

df_bud = df[df[country_column].isin(buddhist)]
df_non = df[df[country_column].isin(non_buddhist)]

shap_inter_bud = explainer_bud.shap_interaction_values(df_bud[all_vars])
shap_inter_non = explainer_non.shap_interaction_values(df_non[all_vars])

inter_bud_df = pd.DataFrame(np.abs(shap_inter_bud).mean(0), index=all_vars, columns=all_vars)
inter_non_df = pd.DataFrame(np.abs(shap_inter_non).mean(0), index=all_vars, columns=all_vars)

inter_bud_ce = inter_bud_df.loc[climate_vars, control_vars]
inter_non_ce = inter_non_df.loc[climate_vars, control_vars]

# --------------------- 差异热力图 ---------------------
scale_factor = 1000
inter_diff_scaled = (inter_bud_ce - inter_non_ce) * scale_factor

prop = font_manager.FontProperties(family='Times New Roman', weight='bold')

plt.figure(figsize=(12, 8))
ax = sns.heatmap(inter_diff_scaled, cmap="coolwarm", center=0, annot=True, fmt=".1f", cbar=True)

plt.title(
    "Interaction Differences on CBR by Religion Type (Buddhist - Non-Buddhist)",
    fontproperties=prop,
    fontsize=16,
    weight='bold',
    pad=20
)
plt.xticks(fontproperties=prop, rotation=30, ha='right')
plt.yticks(fontproperties=prop, rotation=0)
plt.tight_layout()

save_path = os.path.join(output_dir, "CBR.tiff")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"\n✅ All done!\n📂 Output folder: {output_dir}\n🖼 Heatmap saved to: {save_path}")
