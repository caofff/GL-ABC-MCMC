import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
# 指定列名
column_names = ['theta1', 'theta2']
# MCMC = pd.read_csv('glmcmc_results.csv', names=column_names)
MCMC = pd.read_csv('easyabc_Marjoram.csv', names=column_names)
# 选择需要的数据
data_subset = MCMC.iloc[30000:40000]  # 注意 Python 的索引从 0 开始

# 计算每个 (theta1, theta2) 对的计数
count_series = data_subset.groupby(['theta1', 'theta2']).size()
count_df = count_series.reset_index(name='count')

# 将计数合并回原始数据集
data_with_counts = pd.merge(data_subset, count_df, on=['theta1', 'theta2'])

# 创建散点图
plt.figure(figsize=(4.3, 4))
sns.scatterplot(data=data_with_counts, x='theta1', y='theta2',
                size='count', sizes=(20, 200),  # 设置点的大小范围
                color='red', alpha=0.5)

# 添加路径线
plt.plot(data_with_counts['theta1'], data_with_counts['theta2'], color='lightgrey', linewidth=0.5)

# 设置图形主题和标签
plt.title('Trace Plot of θ', fontsize=14)
plt.xlabel(r'$\theta_1$', fontsize=12)
plt.ylabel(r'$\theta_2$', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 隐藏图例
plt.legend().set_visible(False)
plt.savefig('traceplot_GLMCMC.pdf', dpi=300, bbox_inches='tight')
#plt.savefig('traceplot_easyabc_Marjoram.pdf', dpi=300, bbox_inches='tight')
# 展示图形
plt.show()


fig, ax = plt.subplots(figsize=(5,4))

# 绘制密度等高线图，使用蓝色系列线条
contour = sns.kdeplot(data=MCMC, x='theta1', y='theta2', cmap='Blues', fill=True, levels=8, linewidths=1, ax=ax)

# 设置图形主题和标签
ax.set_title('Density Contour Plot of θ', fontsize=14)
ax.set_xlabel(r'$\theta_1$', fontsize=12)
ax.set_ylabel(r'$\theta_2$', fontsize=12)

# 创建自定义颜色条
norm = Normalize(vmin=contour.collections[0].get_array().min(), vmax=contour.collections[0].get_array().max())
sm = cm.ScalarMappable(cmap='Blues', norm=norm)
sm.set_array([])

# 添加颜色条 (colorbar)
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label(label='Density', rotation=270, labelpad=15)  # 设置颜色条标签

# 显示图形
plt.grid(False)
plt.savefig('posteriorGLMCMC_fill.pdf', dpi=300, bbox_inches='tight')
# plt.savefig('posterior_easyabc_Marjoram_fill.pdf', dpi=300, bbox_inches='tight')
plt.show()
