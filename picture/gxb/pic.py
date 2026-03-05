import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class_names = ['正常', '生病']
metric_labels = ['Accuracy', 'AUC', 'Macro F1']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
models = [
    {
        'name': 'PPG+ECG',
        'metrics': [0.7244, 0.7858, 0.7180],
        'class_f1': [0.7721, 0.6514]
    },
    {
        'name': 'PPG',
        'metrics': [0.6991, 0.7590, 0.6886],
        'class_f1': [0.7456, 0.6316]
    },
    {
        'name': 'ECG',
        'metrics': [0.7054, 0.7626, 0.6905],
        'class_f1': [0.7585, 0.6225]
    }
]
colors = ['#4C78A8', '#72B7B2', '#F58518']
bar_width = 0.24
x_metrics = np.arange(len(metric_labels))
offsets = np.linspace(-bar_width, bar_width, len(models))
rects_metrics = []
for i, model in enumerate(models):
    rects = ax1.bar(
        x_metrics + offsets[i],
        model['metrics'],
        bar_width,
        label=model['name'],
        color=colors[i],
        edgecolor='white'
    )
    rects_metrics.append(rects)

ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('整体模型性能对比 (Overall Performance)', fontsize=14, pad=15)
ax1.set_xticks(x_metrics)
ax1.set_xticklabels(metric_labels, fontsize=11)
ax1.set_ylim(0, 1.1)
ax1.legend(loc='upper left', frameon=True)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

x_classes = np.arange(len(class_names))
rects_class = []
for i, model in enumerate(models):
    rects = ax2.bar(
        x_classes + offsets[i],
        model['class_f1'],
        bar_width,
        label=model['name'],
        color=colors[i],
        edgecolor='white'
    )
    rects_class.append(rects)

ax2.set_ylabel('F1 Score', fontsize=12)
ax2.set_title('各类别 F1 分数对比 (Impact on Classes)', fontsize=14, pad=15)
ax2.set_xticks(x_classes)
ax2.set_xticklabels(class_names, fontsize=11)
ax2.set_ylim(0, 1.1)
ax2.legend(loc='upper right', frameon=True)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f'{height:.3f}',
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

for rects in rects_metrics:
    autolabel(rects, ax1)
for rects in rects_class:
    autolabel(rects, ax2)

plt.tight_layout()
plt.show()
plt.savefig('gxb.png', dpi=300, bbox_inches='tight')