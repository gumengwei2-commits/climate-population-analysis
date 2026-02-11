import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# ---------------- 全局字体 ----------------
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 20

# ---------------- 数据路径与变量 ----------------
input_path = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\1-数据\4-整体变量\整体变量（英文代码版）.xlsx"
table_dir = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\2-结果\10.20 最终结果"
fig_dir = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\1-论文图\0-10.20 论文最终图"

output_dir = fig_dir
os.makedirs(output_dir, exist_ok=True)

climate_vars = ['TXx','TX90p','HW','Rx1day','R10','CWD','CDD','SPI','CDWE','CWWE']
control_vars = ['URB','GDP','IMR','UNE','FLFP','NMI','FPI','FEY']
all_vars = climate_vars + control_vars
target_column = 'CBR'

# ---------------- 读取数据 ----------------
df = pd.read_excel(input_path)
X = df[all_vars]
y = df[target_column]

# ---------------- 数据分割 ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------- XGBoost 超参数搜索（十折CV） ----------------------
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9]
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='r2',
    cv=10,
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# 获取最优模型
model = grid_search.best_estimator_
model.fit(X_train, y_train)

# ---------------------- 模型评估 ----------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)

# ---------------------- 输出性能表格 ----------------------
report_table = pd.DataFrame({
    "数据集": ["十折交叉验证(CV)", "训练集", "测试集"],
    "R²": [grid_search.best_score_, r2_train, r2_test],
    "RMSE": [np.nan, np.nan, rmse_test],
    "MAE": [np.nan, np.nan, mae_test]
})

excel_path = os.path.join(table_dir, "CBR_XGB_Model_Performance_Report_interation.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    report_table.to_excel(writer, sheet_name="Performance", index=False)
    pd.DataFrame([grid_search.best_params_]).to_excel(writer, sheet_name="Best_Params", index=False)

print(f"✅ 模型性能表格已保存: {excel_path}")

# ---------------- SHAP 值计算 ----------------
explainer = shap.TreeExplainer(model)
try:
    shap_values_array = explainer.shap_values(X_test)
    shap_interaction_values = explainer.shap_interaction_values(X_test)
except:
    shap_values_array = explainer(X_test).values
    shap_interaction_values = explainer.shap_interaction_values(X_test)

# ---------------- 气候 × 社会经济交互 ----------------
interaction_df = pd.DataFrame(
    np.abs(shap_interaction_values).mean(0),
    index=all_vars,
    columns=all_vars
)
interaction_ce = interaction_df.loc[climate_vars, control_vars]

# ---------------- 放大交互值 ----------------
scale_factor = 1000
interaction_ce_scaled = interaction_ce * scale_factor
shap_values_scaled = shap_values_array * scale_factor

# ---------------- Top 6 气候 × 社会经济交互 ----------------
n_top = 6
interaction_flat = interaction_ce_scaled.abs().unstack().sort_values(ascending=False)
top_pairs = interaction_flat.head(n_top).index.tolist()

# ---------------- 整体布局 ----------------
fig = plt.figure(figsize=(26,14))

# 左侧热力图，占两列
ax1 = plt.subplot2grid((3, 4), (0, 0), rowspan=3, colspan=2)
sns.heatmap(
    interaction_ce_scaled, 
    annot=True, fmt=".3f", cmap="YlOrBr",
    linewidths=0.5, linecolor='white', ax=ax1,
    annot_kws={"fontsize": 20}
)
ax1.set_title(f"Climate × Socioeconomic SHAP interactions (×{scale_factor})", fontsize=24)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=22)
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=22)

# 右侧 6 个散点图 (3行2列)
for i, (clim, econ) in enumerate(top_pairs):
    row = i // 2
    col = i % 2 + 2
    ax = plt.subplot2grid((3, 4), (row, col))
    
    y_vals = shap_values_scaled[:, all_vars.index(clim)]
    x_vals = X_test[clim].values
    color_vals = X_test[econ].values
    
    y_abs = np.abs(y_vals)
    size_vals = 50 * (y_abs - y_abs.min()) / (y_abs.max() - y_abs.min()) + 10
    
    scatter = ax.scatter(x_vals, y_vals, c=color_vals, cmap='plasma', s=size_vals, alpha=0.7)
    ax.set_xlabel(clim, fontsize=22)
    ax.set_ylabel(f"SHAP({clim})", fontsize=22)
    ax.set_title(f"{clim} × {econ}", fontsize=22)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(econ, fontsize=20)
    cbar.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.4)

output_path = os.path.join(output_dir, f"Climate_Econ_Interaction_{target_column}.tiff")
plt.savefig(output_path, dpi=300, format='tiff')
plt.show()
print(f"✅ 交互整体图已保存到: {output_path}")






