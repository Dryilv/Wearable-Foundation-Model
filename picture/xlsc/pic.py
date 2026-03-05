import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体，确保图表中的中文正常显示 (Windows通常为'SimHei'，Mac为'Arial Unicode MS')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# === 1. 数据准备 ===
# 类别名称
class_names = ['窦性心律', '室性早搏', '房性早搏', '室性心动过速', '室上性心动过速', '心房颤动']

# 模型 A / 基准模型 (moment2024)
model_a_name = "moment2024"
model_a_metrics = [0.7808, 0.9303, 0.6458]
model_a_class_f1 = [0.9052, 0.6512, 0.3135, 0.5373, 0.6036, 0.8643]

# 模型 B / 你的最新模型 (从最后一张图提取的数据)
model_b_name = "our model (未改进)"
model_b_metrics = [0.8500, 0.9600, 0.7400]
model_b_class_f1 = [0.9800, 0.7800, 0.5300, 0.5900, 0.6800, 0.8900]

model_c_name = "our model (改进后)"
model_c_metrics = [0.8739, 0.9703, 0.7728]
model_c_class_f1 = [0.9880, 0.8187, 0.6087, 0.6245, 0.6833, 0.9139]

# 左图指标名称
metric_labels = ['Accuracy', 'AUC', 'Macro F1']

# === 2. 绘图参数设置 ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

color_a = '#8ab6d6'
color_b = '#00447c'
color_c = '#f18f01'
bar_width = 0.25

# === 3. 绘制左图：整体性能对比 ===
x_metrics = np.arange(len(metric_labels))

rects1 = ax1.bar(x_metrics - bar_width, model_a_metrics, bar_width, label=model_a_name, color=color_a, edgecolor='white')
rects2 = ax1.bar(x_metrics, model_b_metrics, bar_width, label=model_b_name, color=color_b, edgecolor='white')
rects3 = ax1.bar(x_metrics + bar_width, model_c_metrics, bar_width, label=model_c_name, color=color_c, edgecolor='white')

ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('整体模型性能对比 (Overall Performance)', fontsize=14, pad=15)
ax1.set_xticks(x_metrics)
ax1.set_xticklabels(metric_labels, fontsize=11)
ax1.set_ylim(0, 1.15) # 稍微调高一点 Y 轴，防止数字和图例重叠
ax1.legend(loc='upper left', frameon=True)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# === 4. 绘制右图：各类别 F1 分数对比 ===
x_classes = np.arange(len(class_names))

rects4 = ax2.bar(x_classes - bar_width, model_a_class_f1, bar_width, label=model_a_name, color=color_a, edgecolor='white')
rects5 = ax2.bar(x_classes, model_b_class_f1, bar_width, label=model_b_name, color=color_b, edgecolor='white')
rects6 = ax2.bar(x_classes + bar_width, model_c_class_f1, bar_width, label=model_c_name, color=color_c, edgecolor='white')

ax2.set_ylabel('F1 Score', fontsize=12)
ax2.set_title('各类别 F1 分数对比 (Impact on Classes)', fontsize=14, pad=15)
ax2.set_xticks(x_classes)
ax2.set_xticklabels(class_names, fontsize=10, rotation=15)
ax2.set_ylim(0, 1.15)
ax2.legend(loc='upper right', frameon=True)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# === 5. 辅助函数：在柱子上添加数值标签 ===
def autolabel(rects, ax):
    """在每个柱状图顶部显示数值，保留两位小数"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1, ax1)
autolabel(rects2, ax1)
autolabel(rects3, ax1)
autolabel(rects4, ax2)
autolabel(rects5, ax2)
autolabel(rects6, ax2)

# 调整布局并显示
plt.tight_layout()
plt.show()
plt.savefig('xlsc.png', dpi=300, bbox_inches='tight')
