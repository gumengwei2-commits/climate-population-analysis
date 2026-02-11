import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import shap
import matplotlib
matplotlib.use('Agg')  # 非 GUI 后端
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings('ignore')

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'

# --------------------------- 用户设置 ---------------------------
input_path = r"D:/0_@博士/1-论文/3-小论文——气候变化与出生率/1-数据/4-整体变量/整体变量（英文代码版）.xlsx"
table_dir = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\2-结果\10.20 最终结果\异质性\收入异质性"
fig_dir = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\1-论文图\0-10.20 论文最终图\收入异质性"

climate_vars = ['TXx','TX90p','HW','Rx1day','R10','CWD','CDD','SPI','CDWE','CWWE']
target_column = 'CBR'
country_column = 'country'

# 收入分组
high_income = ['Brunei','Singapore']
upper_middle_income = ['Indonesia','Malaysia','Thailand']
lower_middle_income = ['Myanmar','Cambodia','Vietnam','Laos','Philippines','Timor-Leste']

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

os.makedirs(table_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# --------------------------- 读取数据 ---------------------------
if not Path(input_path).exists():
    raise FileNotFoundError(f"Input file not found: {input_path}")

df = pd.read_excel(input_path)
required_cols = [country_column, target_column] + climate_vars
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in input file: {missing}")

df = df.dropna(subset=[target_column])
df = df[~df[climate_vars].isna().all(axis=1)].copy()

# --------------------------- 辅助函数 ---------------------------
def train_shap_for_group(df_group, group_name, table_dir, fig_dir):
    X = df_group[climate_vars].copy()
    y = df_group[target_column].copy()
    non_missing = X.dropna().index
    X = X.loc[non_missing]
    y = y.loc[non_missing].values

    if X.shape[0] < 10:
        print(f"Warning: group {group_name} has only {X.shape[0]} observations. Skipping.")
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # ---------------------- XGBoost 十折 CV ----------------------
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

    print(f"Best params for {group_name}: {grid_search.best_params_}")

    model = grid_search.best_estimator_
    model.fit(X_train, y_train)

    # ---------------------- 模型评估 ----------------------
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)
    print(f"Group={group_name} | Train R²={r2_train:.3f} | Test R²={r2_test:.3f} | Test RMSE={rmse_test:.3f} | Test MAE={mae_test:.3f}")

    # ---------------------- 保存性能表 ----------------------
    performance_df = pd.DataFrame({
        "数据集": ["十折交叉验证(CV)","训练集","测试集"],
        "R²": [grid_search.best_score_, r2_train, r2_test],
        "RMSE": [np.nan, np.nan, rmse_test],
        "MAE": [np.nan, np.nan, mae_test]
    })
    performance_df.to_csv(os.path.join(table_dir, f"CBR_XGB_performance_{group_name.replace(' ','_')}.csv"), 
                          index=False, encoding='utf-8-sig')

    # ---------------------- SHAP ----------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_df.to_csv(os.path.join(table_dir, f'CBR_shap_values_{group_name.replace(" ","_")}.csv'), index=False)

    # 保存 SHAP summary TIFF
    plt.figure(figsize=(6,6))
    shap.summary_plot(shap_values, X, show=False, plot_type='dot', max_display=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'CBR_shap_summary_{group_name.replace(" ","_")}.tiff'), dpi=300)
    plt.close()

    return X, shap_values, X.columns

def make_group_df(country_list):
    return df[df[country_column].isin(country_list)].copy()

# --------------------------- 训练各收入组 ---------------------------
shap_data = {}
group_dict = {
    'High Income': high_income,
    'Upper Middle Income': upper_middle_income,
    'Lower Middle Income': lower_middle_income
}

for gname, glist in group_dict.items():
    gdf = make_group_df(glist)
    Xg, shap_g, cols_g = train_shap_for_group(gdf, gname, table_dir, fig_dir)
    if shap_g is not None:
        shap_data[gname] = (Xg, shap_g, cols_g)

# --------------------------- 拼接 SHAP 图像，添加标题 ---------------------------
titles = ['(a) High Income', '(b) Upper Middle Income', '(c) Lower Middle Income']
saved_images = [os.path.join(fig_dir, f'CBR_shap_summary_{g.replace(" ","_")}.tiff') for g in shap_data.keys()]
images = [Image.open(f) for f in saved_images]

# 统一高度
target_height = 1200
resized_images = [im.resize((int(im.width * target_height / im.height), target_height), Image.LANCZOS) for im in images]

# 设置标题和空白条
titles = ['High Income', 'Upper Middle Income', 'Lower Middle Income']
title_bar_height = 120
font_size = 60

# 加载 Times New Roman 字体
try:
    font_path = r"C:\Windows\Fonts\times.ttf"  # 或者 timesnewroman.ttf
    font = ImageFont.truetype(font_path, font_size)
except:
    print("Warning: Times New Roman not found. Using default font.")
    font = ImageFont.load_default()

# 假设 resized_images 已经生成
titled_images = []
for im, title in zip(resized_images, titles):
    new_height = im.height + title_bar_height
    im_with_title = Image.new('RGB', (im.width, new_height), (255, 255, 255))
    im_with_title.paste(im, (0, title_bar_height))

    draw = ImageDraw.Draw(im_with_title)
    # 获取文字尺寸（兼容新版 Pillow）
    try:
        bbox = draw.textbbox((0,0), title, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        text_width, text_height = font.getsize(title)

    # 居中绘制标题
    x_text = (im.width - text_width) // 2
    y_text = (title_bar_height - text_height) // 2
    draw.text((x_text, y_text), title, fill=(0,0,0), font=font)

    titled_images.append(im_with_title)

# 横向拼接
total_width = sum(im.width for im in titled_images)
final_im = Image.new('RGB', (total_width, titled_images[0].height), (255,255,255))
x_offset = 0
for im in titled_images:
    final_im.paste(im, (x_offset, 0))
    x_offset += im.width

# 保存最终 TIFF
final_im.save(os.path.join(fig_dir, 'CBR_shap_summary_income_comparison.tiff'), dpi=(300,300))

print("All done. Tables saved to:", table_dir)
print("Figures saved to:", fig_dir)

