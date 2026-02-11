import pandas as pd
import numpy as np
import pymannkendall as mk
from scipy.stats import theilslopes
from statsmodels.tsa.stattools import acf
import os
import matplotlib.pyplot as plt

# --------------------------
# 路径设置
# --------------------------
input_file = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\1-数据\4-整体变量\整体变量（英文代码版）.xlsx"
output_folder = r"D:\0_@博士\1-论文\3-小论文——气候变化与出生率\4-实证结果\1-论文图\3-人口生育力的时空分布特征图"
os.makedirs(output_folder, exist_ok=True)

# --------------------------
# 读取数据并清理
# --------------------------
df = pd.read_excel(input_file)
df['country'] = df['country'].str.strip()
for col in ['TFR', 'CBR']:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(',', '').str.strip(),
        errors='coerce'
    )

# --------------------------
# 填充完整年份并插值
# --------------------------
years_all = np.arange(2004, 2023)
countries = df['country'].unique()
df_full = pd.DataFrame([(c, y) for c in countries for y in years_all], columns=['country', 'year'])
df_full = pd.merge(df_full, df[['country', 'year', 'TFR', 'CBR']], on=['country', 'year'], how='left')
df_full['TFR'] = df_full.groupby('country')['TFR'].transform(lambda x: x.interpolate())
df_full['CBR'] = df_full.groupby('country')['CBR'].transform(lambda x: x.interpolate())

# --------------------------
# 指标列表
# --------------------------
indices = ['TFR', 'CBR']
mean_df = df_full.groupby("year")[indices].mean().reset_index()

# --------------------------
# 图形设置
# --------------------------
plt.rcParams['font.family'] = 'Times New Roman'
fig2 = plt.figure(figsize=(30, 10))
gs2 = fig2.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.15)
axes2 = [fig2.add_subplot(gs2[0]), fig2.add_subplot(gs2[1])]

results = []

# --------------------------
# 绘图与统计
# --------------------------
for i, ind in enumerate(indices):
    ax = axes2[i]

    series = mean_df[ind].values
    years = mean_df["year"].values

    # 一阶自相关
    r1 = acf(series, nlags=1, fft=False)[1]

    # Hamed & Rao 修正 MK
    mk_result = mk.hamed_rao_modification_test(series)

    # Theil–Sen 斜率
    slope, intercept, lo_slope, hi_slope = theilslopes(series, years, 0.95)
    mean_val = np.nanmean(series)
    pct_per_decade = (slope * 10 / mean_val) * 100.0

    # 数据与趋势线
    ax.plot(years, series, marker='o', color='skyblue', linewidth=2.5, label="SE Asia mean")
    linestyle = '-' if mk_result.p < 0.05 else '--'
    ax.plot(years, intercept + slope * years, color='black', linewidth=2, linestyle=linestyle, label="Theil–Sen trend")

    # 居中子图标题：TFR / CBR
    ax.set_title(ind, fontsize=32, pad=18)

    # 右上角统计信息
    mk_sign = "significant" if mk_result.p < 0.05 else "not significant"
    ax.text(
        0.65, 0.95,
        f"{pct_per_decade:.1f}% / decade\nMK {mk_sign}\nr₁={r1:.2f}",
        transform=ax.transAxes,
        fontsize=30,
        va='top',
        ha='left'
    )

    # 坐标轴格式
    ax.tick_params(direction="in", axis='x', labelsize=30)
    ax.tick_params(direction="in", axis='y', labelsize=30)
    ax.set_xticks(np.arange(2004, 2023, 2))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    if ind == "TFR":
        ax.set_yticks([2.0, 2.2, 2.4, 2.6])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
    ax.grid(True, linestyle="--", alpha=0.5)

    # 保存统计结果
    results.append({
        "Indicator": ind,
        "Mean": mean_val,
        "TheilSen_slope_per_year": slope,
        "Pct_per_decade (%)": pct_per_decade,
        "r1_autocorr": r1,
        "MK_p_adj_HamedRao": mk_result.p,
        "MK_significant": mk_result.p < 0.05,
        "Trend_direction": "Increasing" if slope > 0 else "Decreasing"
    })

# --------------------------
# Figure-level panel labels (A, B) —— 图框外
# --------------------------
panel_positions = [
    (0.02, 0.94),  # A：左图
    (0.52, 0.94)   # B：右图
]
for i, (x, y) in enumerate(panel_positions):
    fig2.text(
        x, y,
        chr(65 + i),
        fontsize=32,
        fontweight='bold',
        ha='left',
        va='top'
    )

# --------------------------
# 保存结果
# --------------------------
plt.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.12)
fig2.tight_layout(rect=[0, 0.05, 1, 1])

fig2_path = os.path.join(output_folder, "Temporal trends in fertility rates.tiff")
plt.savefig(fig2_path, dpi=300)
plt.close()
print("Figure saved to:", fig2_path)

results_df = pd.DataFrame(results)
excel_path = os.path.join(output_folder, "SEA_trend_percent_decade_HamedRao.xlsx")
results_df.to_excel(excel_path, index=False)
print("Excel saved to:", excel_path)
print(results_df)
