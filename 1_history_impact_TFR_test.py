import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# ----------------------
# 绘图函数
# ----------------------
def plot_shap_group(X, shap_values, y, features, title, output_file):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D
    import numpy as np
    import pandas as pd

    plt.rcParams['font.family'] = 'Times New Roman'

    aesthetic_params = {
        'suptitle_size': 24,
        'ax_label_size': 22,
        'tick_label_size': 22,
        'legend_size': 16,
        'cbar_label_size': 14,
        'summary_cbar_width': 0.012,
        'summary_cbar_height_shrink': 1.0,
        'summary_cbar_pad': 0.02,
        'dep_cbar_width': 0.006,
        'dep_cbar_height_shrink': 1.0,
        'dep_cbar_pad': 0.005,
        'dep_cbar_tick_length': 2,
        'grid_wspace': 0.6,
        'grid_hspace': 0.5
    }

    # ================== SHAP 重要性计算 ==================
    mean_abs_shaps = np.abs(shap_values.values).mean(axis=0)
    mean_shaps = shap_values.values.mean(axis=0)

    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': mean_abs_shaps,
        'effect_direction': ['Positive' if v > 0 else 'Negative' for v in mean_shaps]
    }).sort_values('importance', ascending=True)

    feature_importance_df = feature_importance_df[
        feature_importance_df['feature'].isin(features)
    ]

    colors = ['#FFC0CB' if d == "Positive" else '#ADD8E6'
              for d in feature_importance_df['effect_direction']]

    # ================== 画布 ==================
    fig = plt.figure(figsize=(30, 18))
    gs = gridspec.GridSpec(
        3, 5, figure=fig,
        wspace=aesthetic_params['grid_wspace'],
        hspace=aesthetic_params['grid_hspace']
    )

    # ================== 左侧 Summary ==================
    ax_main = fig.add_subplot(gs[:, :2])
    ax_top = ax_main.twiny()

    ax_top.barh(
        range(len(feature_importance_df)),
        feature_importance_df['importance'],
        color=colors, alpha=0.6, height=0.7
    )

    # ⭐ Mean SHAP Value 标题抬高
    ax_top.set_xlabel(
        title,
        fontsize=aesthetic_params['ax_label_size'],
        labelpad=25
    )
    ax_top.tick_params(axis='x', labelsize=aesthetic_params['tick_label_size'])
    ax_top.grid(False)

    ax_main.set_yticks(range(len(feature_importance_df)))
    ax_main.set_yticklabels(
        feature_importance_df['feature'],
        fontsize=aesthetic_params['tick_label_size']
    )

    # ================== SHAP Beeswarm ==================
    cmap = plt.get_cmap("viridis")

    for i, feature_name in enumerate(feature_importance_df['feature']):
        idx = X.columns.get_loc(feature_name)
        shap_vals = shap_values.values[:, idx]
        feature_vals = X.iloc[:, idx]
        y_jitter = np.random.normal(0, 0.08, size=len(shap_vals))

        ax_main.scatter(
            shap_vals, i + y_jitter,
            c=feature_vals, cmap=cmap,
            s=20, alpha=0.9, zorder=2
        )

    ax_main.tick_params(axis='x', labelsize=aesthetic_params['tick_label_size'])
    ax_main.grid(True, axis='x', linestyle='--', alpha=0.6)

    # ================== 左侧 Colorbar ==================
    fig.canvas.draw()
    ax_pos = ax_main.get_position()

    cax = fig.add_axes([
        ax_pos.x1 + aesthetic_params['summary_cbar_pad'],
        ax_pos.y0,
        aesthetic_params['summary_cbar_width'],
        ax_pos.height
    ])

    norm = plt.Normalize(vmin=X.values.min(), vmax=X.values.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)
    # ❌ 去掉左侧 colorbar 标签文字，只保留颜色条
    cbar.outline.set_visible(False)
    cbar.set_ticks([])

    # High / Low 标注
    cbar.ax.text(0.6, 1.02, 'High', transform=cbar.ax.transAxes,
                 ha='center', va='top',
                 fontsize=aesthetic_params['tick_label_size'])
    cbar.ax.text(0.6, -0.02, 'Low', transform=cbar.ax.transAxes,
                 ha='center', va='bottom',
                 fontsize=aesthetic_params['tick_label_size'])

    # ================== 右侧 Dependence Plots ==================
    top_features = feature_importance_df['feature'].tail(6).iloc[::-1].tolist()
    axes_scatter = []

    for i in range(3):
        for j in range(2):
            idx_feature = i * 2 + j
            if idx_feature >= len(top_features):
                continue
            axes_scatter.append(fig.add_subplot(gs[i, j + 2]))

    for i, feature in enumerate(top_features):
        ax = axes_scatter[i]
        idx = X.columns.get_loc(feature)

        scatter = ax.scatter(
            X[feature],
            shap_values.values[:, idx],
            c=y, cmap=cmap, s=30, alpha=0.8
        )

        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        # ❌ 右侧依赖图不显示 y 轴标签
        ax.set_ylabel("")
        ax.set_xlabel(feature, fontsize=aesthetic_params['ax_label_size'])
        ax.tick_params(axis='both', labelsize=aesthetic_params['tick_label_size'])

        # 中位数 & 阈值
        median_val = X[feature].median()
        threshold_val = np.percentile(X[feature], 75)

        ax.axvline(median_val, color='black', linestyle='--', linewidth=1)
        ax.axvline(threshold_val, color='red', linestyle=':', linewidth=1.2)

        handles = [
            Line2D([0], [0], color='black', linestyle='--',
                   label=f'Median: {median_val:.2f}'),
            Line2D([0], [0], color='red', linestyle=':',
                   label=f'Threshold: {threshold_val:.2f}')
        ]
        ax.legend(handles=handles, fontsize=aesthetic_params['legend_size'])

        # 右侧 colorbar，保留标题 TFR
        fig.canvas.draw()
        ax_pos = ax.get_position()
        cax_dep = fig.add_axes([
            ax_pos.x1 + aesthetic_params['dep_cbar_pad'],
            ax_pos.y0,
            aesthetic_params['dep_cbar_width'],
            ax_pos.height
        ])
        cbar = fig.colorbar(scatter, cax=cax_dep)
        cbar.outline.set_visible(False)
        # ✅ 保留标题
        cbar.ax.set_title("TFR", fontsize=14)
        cbar.ax.tick_params(
            axis='y',
            length=aesthetic_params['dep_cbar_tick_length'],
            labelsize=aesthetic_params['tick_label_size']
        )

    # ================== 保存 ==================
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='tiff')
    plt.close()
    print(f"✅ Saved: {output_file}")








# ---------------------- 数据路径 ----------------------
input_path = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\1-数据\4-整体变量\整体变量（英文代码版）.xlsx"
table_dir = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\2-结果\10.20 最终结果"
fig_dir = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\1-论文图\0-2026.1.23"
os.makedirs(table_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

climate_vars = ['TXx','TX90p','HW','Rx1day','R10','CWD','CDD','SPI','CDWE','CWWE']
control_vars = ['URB','GDP','IMR','UNE','FLFP','NMI','FPI','FEY']
all_vars = climate_vars + control_vars

# ---------------------- 读取数据 ----------------------
df = pd.read_excel(input_path)
X = df[all_vars]
y = df['TFR']

# ---------------------- 数据分割 ----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------- XGBoost 超参数搜索（十折CV） ----------------------
param_grid = {
    'n_estimators':[100,200,300], 'max_depth':[3,5], 'learning_rate':[0.05,0.1],
    'subsample':[0.7,0.9], 'colsample_bytree':[0.7,0.9], 'reg_alpha':[0,0.1,0.5], 'reg_lambda':[1,1.5]
}
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(xgb_model, param_grid, scoring='r2', cv=10, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

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
    "数据集":["十折交叉验证(CV)","训练集","测试集"],
    "R²":[grid_search.best_score_, r2_train, r2_test],
    "RMSE":[np.nan, np.nan, rmse_test],
    "MAE":[np.nan, np.nan, mae_test]
})
excel_path = os.path.join(table_dir, "TFR_XGB_Model_Performance_Report.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    report_table.to_excel(writer, sheet_name="Performance", index=False)
    pd.DataFrame([grid_search.best_params_]).to_excel(writer, sheet_name="Best_Params", index=False)
print(f"✅ 模型性能表格已保存: {excel_path}")

# ---------------------- SHAP 分析 ----------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

shap_importance = pd.DataFrame({
    "Variable": all_vars,
    "Mean_Abs_SHAP": np.abs(shap_values.values).mean(axis=0),
    "Effect_Direction": ["正向" if x>0 else "负向" for x in shap_values.values.mean(axis=0)]
}).sort_values(by="Mean_Abs_SHAP", ascending=False)

shap_path = os.path.join(table_dir, "TFR_SHAP_results_xgb.csv")
shap_importance.to_csv(shap_path, index=False, encoding='utf-8-sig')
print(f"✅ SHAP 表格已保存: {shap_path}")

# ---------------------- 绘制 SHAP 图 ----------------------
plot_shap_group(X_test, shap_values, y_test, climate_vars, "Mean SHAP Value (Climate)",
                os.path.join(fig_dir, "TFR_SHAP_Climate.tiff"))
plot_shap_group(X_test, shap_values, y_test, control_vars, "Mean SHAP Value (Socioeconomic)",
                os.path.join(fig_dir, "TFR_SHAP_Socioeconomic.tiff"))

# ---------------------- 测试集预测效果可视化 ----------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual TFR")
plt.ylabel("Predicted TFR")
plt.title(f"XGBoost Test Set Prediction\nR² = {r2_test:.3f}")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "TFR_XGB_R2_Test.tiff"), dpi=300)
plt.show()