import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

warnings.filterwarnings('ignore')
mpl.rcParams['font.family'] = 'Times New Roman'

# --------------------------- User settings ---------------------------
input_path = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\1-数据\4-整体变量\整体变量（英文代码版）.xlsx"
table_dir = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\2-结果\10.20 最终结果"
fig_dir = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\1-论文图\0-10.20 论文最终图\宗教异质性"
os.makedirs(table_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

climate_vars = ['TXx','TX90p','HW','Rx1day','R10','CWD','CDD','SPI','CDWE','CWWE']
target_column = 'CBR'
country_column = 'country'

# --------------------------- Religion groups ---------------------------
buddhist = ['Thailand','Cambodia','Laos','Myanmar']
non_buddhist = ['Indonesia','Brunei','Malaysia','Philippines','Timor-Leste','Singapore','Vietnam']

param_grid = {
    'n_estimators':[100,200,300],
    'max_depth':[3,5],
    'learning_rate':[0.05,0.1],
    'subsample':[0.7,0.9],
    'colsample_bytree':[0.7,0.9],
    'reg_alpha':[0,0.1,0.5],
    'reg_lambda':[1,1.5]
}

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --------------------------- Load data ---------------------------
if not Path(input_path).exists():
    raise FileNotFoundError(f"Input file not found: {input_path}")

df = pd.read_excel(input_path)
df = df.dropna(subset=[target_column])
df = df[~df[climate_vars].isna().all(axis=1)].copy()

# --------------------------- Helper functions ---------------------------
def train_group_XGB(df_group, group_name):
    X = df_group[climate_vars].copy()
    y = df_group[target_column].copy()
    non_missing = X.dropna().index
    X = X.loc[non_missing]
    y = y.loc[non_missing].values

    if X.shape[0] < 10:
        print(f"Warning: group {group_name} has only {X.shape[0]} observations. Skipping.")
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # ---------------------- XGBoost 10-fold CV ----------------------
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=RANDOM_SEED)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='r2',
        cv=10,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"Best params for group {group_name}: {grid_search.best_params_}")

    model = grid_search.best_estimator_
    model.fit(X_train, y_train)

    # ---------------------- Performance ----------------------
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)
    print(f"Group={group_name} | Train R²={r2_train:.3f} | Test R²={r2_test:.3f} | Test RMSE={rmse_test:.3f} | Test MAE={mae_test:.3f}")

    # ---------------------- Save performance table ----------------------
    performance_df = pd.DataFrame({
        "Dataset": ["Train","Test"],
        "R²": [r2_train, r2_test],
        "RMSE": [np.nan, rmse_test],
        "MAE": [np.nan, mae_test]
    })
    performance_df.to_csv(os.path.join(table_dir, f"CBR_XGB_performance_{group_name}.csv"), index=False, encoding='utf-8-sig')

    # ---------------------- SHAP ----------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_df.to_csv(os.path.join(table_dir, f'CBR_shap_values_{group_name}.csv'), index=False)

    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'CBR_shap_summary_{group_name}.tiff'), dpi=300)
    plt.close()

    return X, shap_values, X.columns

def make_group_df(country_list):
    return df[df[country_column].isin(country_list)].copy()

# --------------------------- Train for groups ---------------------------
shap_data = {}
for gname, glist in zip(['buddhist','non_buddhist'], [buddhist, non_buddhist]):
    gdf = make_group_df(glist)
    X, shap_vals, cols = train_group_XGB(gdf, gname)
    if shap_vals is not None:
        shap_data[gname] = (X, shap_vals, cols)

# --------------------------- SHAP comparison plot ---------------------------
if all(k in shap_data for k in ['buddhist','non_buddhist']):
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    X_b, shap_b, cols_b = shap_data['buddhist']
    X_n, shap_n, cols_n = shap_data['non_buddhist']

    plt.sca(axes[0])
    shap.summary_plot(shap_b, X_b, show=False, plot_type='dot', max_display=10)
    axes[0].set_title("Buddhist region")

    plt.sca(axes[1])
    shap.summary_plot(shap_n, X_n, show=False, plot_type='dot', max_display=10)
    axes[1].set_title("Non-Buddhist region")

    axes[0].tick_params(axis='y', labelsize=10)
    axes[1].tick_params(axis='y', labelsize=10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(os.path.join(fig_dir, 'CBR_religion_comparison.tiff'), dpi=300)
    plt.close()

print("All done. Tables saved to:", table_dir)
print("Figures saved to:", fig_dir)

