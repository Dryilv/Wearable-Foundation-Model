import os
import json
import pickle
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

# ================= 配置路径 =================
DATA_DIR = "/home/bml/storage/mnt/v-044d0fb740b04ad3/org/WFM/wearable_FM/downstream_data/ppgguanxinbing/sample_for_downstream"
SPLIT_FILE = "/home/bml/storage/mnt/v-044d0fb740b04ad3/org/WFM/wearable_FM/downstream_data/ppgguanxinbing/train_test_split.json" # 你的 split 文件路径

def load_data_and_group(split_file, data_dir):
    """
    加载数据并将信号按类别分组 (0: Healthy, 1: CHD)
    假设 label 列表中的第一个字典包含分类标签 {"class": x}
    """
    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    # 合并 train 和 test 文件列表，我们需要看整体数据的区分度
    all_files = splits['train'] + splits['test']
    
    group_0_signals = [] # 负样本 (例如健康)
    group_1_signals = [] # 正样本 (例如冠心病)
    
    print(f"开始加载 {len(all_files)} 个样本...")
    
    for file_name in all_files:
        file_path = os.path.join(data_dir, file_name)
        
        try:
            with open(file_path, 'rb') as f:
                sample = pickle.load(f)
                
            # 1. 获取信号数据
            # 假设 shape 是 [num_channels, sequence_length]
            # 我们取第一个通道 (通常是 PPG)，并转为 float32 以便计算
            raw_signal = sample['data'][0, :].astype(np.float32)
            
            # 2. 获取标签
            # 你的格式是List: [{"class": label}, {"reg": label}]
            # 我们假设第一个元素是分类标签
            label_dict = sample['label'][0] 
            if 'class' in label_dict:
                label = label_dict['class']
            else:
                continue # 如果没有 class 标签则跳过
            
            # 3. 简单的特征提取 (为了 K-S 检验)
            # 直接比较原始信号点量太大且受相位影响，
            # 建议比较信号的统计特征，例如：均值、标准差、或者信号幅值的分布
            # 这里我们将该样本的所有采样点放入对应的组，观察整体幅值分布
            if label == 0:
                group_0_signals.append(raw_signal)
            elif label == 1:
                group_1_signals.append(raw_signal)
                
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    # 将列表展平为一维数组 (注意内存占用，如果数据量太大需采样)
    print("正在合并数组...")
    data_class_0 = np.concatenate(group_0_signals)
    data_class_1 = np.concatenate(group_1_signals)
    
    return data_class_0, data_class_1

def perform_ks_test(data1, data2, label1="Class 0", label2="Class 1"):
    """
    执行 K-S 检验并打印结果
    """
    print(f"\n正在执行 K-S 检验: {label1} vs {label2}")
    
    # 为了加快计算速度，如果数据量超过 100万点，可以进行随机降采样
    if len(data1) > 1000000:
        data1 = np.random.choice(data1, 1000000, replace=False)
    if len(data2) > 1000000:
        data2 = np.random.choice(data2, 1000000, replace=False)

    # 计算 K-S 统计量
    statistic, p_value = ks_2samp(data1, data2)
    
    print(f"K-S Statistic: {statistic:.4f}")
    print(f"P-value:       {p_value:.4e}")
    
    # 结果解读
    if p_value < 0.05:
        print("结论: 拒绝原假设，两个分布存在显著差异。")
    else:
        print("结论: 无法拒绝原假设，两个分布可能相同。")
        
    return statistic

def plot_cdf(data1, data2, label1="Class 0", label2="Class 1"):
    """
    绘制累积分布函数 (CDF) 图，直观展示 K-S 距离
    """
    print("正在绘制 CDF 图...")
    # 排序
    sorted_data1 = np.sort(data1)
    sorted_data2 = np.sort(data2)
    
    # 计算 y 轴 (0 到 1)
    y1 = np.arange(1, len(sorted_data1) + 1) / len(sorted_data1)
    y2 = np.arange(1, len(sorted_data2) + 1) / len(sorted_data2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_data1, y1, label=label1, linewidth=1.5)
    plt.plot(sorted_data2, y2, label=label2, linewidth=1.5)
    plt.title('Cumulative Distribution Function (CDF) Comparison')
    plt.xlabel('Signal Amplitude / Feature Value')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 加载数据
    # 注意：请确保 DATA_DIR 和 SPLIT_FILE 路径正确
    # 如果没有真实文件，这段代码会报错。
    if os.path.exists(SPLIT_FILE):
        c0_data, c1_data = load_data_and_group(SPLIT_FILE, DATA_DIR)
        
        # 2. 执行检验
        ks_stat = perform_ks_test(c0_data, c1_data, "Healthy", "CHD")
        
        # 3. 绘图
        plot_cdf(c0_data, c1_data, "Healthy", "CHD")
    else:
        print("未找到 split 文件，请检查路径。")