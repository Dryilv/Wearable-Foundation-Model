import os
import json
import pickle
import numpy as np
import scipy.signal as signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from tqdm import tqdm

# ==========================================
# 1. 信号处理与特征提取 (对应论文方法)
# ==========================================

class SDPTGFeatureExtractor:
    def __init__(self):
        pass

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        """带通滤波，去除基线漂移和高频噪声"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        y = signal.filtfilt(b, a, data)
        return y

    def get_sdptg(self, ppg_signal):
        """
        计算二阶导数 (SDPTG)
        论文引用: "The second derivative of the PPG (SDPTG) can be used to reflect arterial characteristics."
        """
        # 一阶导数
        first_deriv = np.gradient(ppg_signal)
        # 二阶导数
        second_deriv = np.gradient(first_deriv)
        return second_deriv

    def extract_beat_features(self, sdptg_beat):
        """
        从单个心跳的SDPTG中提取 a, b, c, d, e 波特征
        注意：在实际含噪信号中，c, d, e往往难以精确区分，这里采用鲁棒的极值法。
        """
        # a波: 初始阶段的最大正峰值
        a_idx = np.argmax(sdptg_beat)
        a_val = sdptg_beat[a_idx]
        
        # b波: a波之后的第一个深谷（通常是全局最小值）
        # 为了简化，我们在a波之后寻找最小点作为b
        if a_idx < len(sdptg_beat) - 1:
            b_idx = a_idx + np.argmin(sdptg_beat[a_idx:])
            b_val = sdptg_beat[b_idx]
        else:
            b_val = 0 # 异常处理

        # c, d, e 波通常位于 b波之后。
        # 论文公式: Aging Index = (b - c - d - e) / a
        # 由于 c,d,e 识别困难，我们提取 b波之后的统计特征作为替代，
        # 或者尝试寻找 b波之后的局部极值。
        
        # 这里为了Baseline的稳定性，我们提取 b/a 比值 (论文重点提到的指标)
        b_a_ratio = b_val / a_val if a_val != 0 else 0
        
        # 提取SDPTG波形的整体统计特征，捕捉 c,d,e 的潜在信息
        sdptg_std = np.std(sdptg_beat)
        sdptg_skew = np.mean((sdptg_beat - np.mean(sdptg_beat))**3)
        
        # 模拟 Aging Index 的简化版：(Min_Value - Mean_Rest) / Max_Value
        # 这不是严格的论文公式，但在无法精确分割cde时是有效的近似
        approx_ai = (b_val - np.mean(sdptg_beat[a_idx:])) / a_val if a_val != 0 else 0

        return [b_a_ratio, approx_ai, sdptg_std, sdptg_skew]

    def process_sample(self, ppg_raw, fs):
        """处理单个样本：滤波 -> 分割心跳 -> 提取特征 -> 平均"""
        # 1. 滤波 (0.5-8Hz是PPG的典型频段)
        filtered_ppg = self.butter_bandpass_filter(ppg_raw, 0.5, 8.0, fs)
        
        # 2. 计算SDPTG
        sdptg_signal = self.get_sdptg(filtered_ppg)
        
        # 3. 心跳分割 (使用原始PPG找峰值)
        # distance=fs*0.6 假设心率不超过100bpm，避免检测过密
        peaks, _ = signal.find_peaks(filtered_ppg, distance=int(fs*0.6))
        
        beat_features_list = []
        
        # 遍历每个心跳周期
        for i in range(len(peaks) - 1):
            start = peaks[i]
            end = peaks[i+1]
            
            # 截取该心跳的 SDPTG 片段
            beat_sdptg = sdptg_signal[start:end]
            
            if len(beat_sdptg) < 10: continue # 忽略过短片段
            
            feats = self.extract_beat_features(beat_sdptg)
            beat_features_list.append(feats)
            
        if not beat_features_list:
            return np.zeros(4) # 如果没检测到心跳，返回0向量
            
        # 对所有心跳的特征取平均，代表该患者的整体状态
        avg_features = np.mean(beat_features_list, axis=0)
        return avg_features

# ==========================================
# 2. 数据加载与训练流程
# ==========================================

def load_dataset(split_file, data_dir, mode='train'):
    """加载数据并转换为特征矩阵 X 和 标签 y"""
    
    with open(split_file, 'r') as f:
        splits = json.load(f)
        
    file_list = splits[mode]
    extractor = SDPTGFeatureExtractor()
    
    X = []
    y = []
    
    print(f"Processing {mode} data...")
    for file_name in tqdm(file_list):
        file_path = os.path.join(data_dir, file_name)
        
        try:
            with open(file_path, 'rb') as f:
                sample = pickle.load(f)
                
            # 解析数据结构
            # 假设 data shape 是 [channels, seq_len]，取第一个通道
            raw_data = sample['data']
            if raw_data.ndim > 1:
                ppg_signal = raw_data[0, :] 
            else:
                ppg_signal = raw_data
                
            fs = sample['sampling_rate']
            
            # 解析标签
            # 结构: [{"class": label}, {"reg": label}]
            label = None
            for l in sample['label']:
                if 'class' in l:
                    label = l['class']
                    break
            
            if label is None:
                continue # 跳过无分类标签的数据
                
            # 特征提取
            features = extractor.process_sample(ppg_signal, fs)
            
            X.append(features)
            y.append(label)
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
            
    return np.array(X), np.array(y)

def run_baseline(data_dir, split_file):
    # 1. 加载数据
    X_train, y_train = load_dataset(split_file, data_dir, mode='train')
    X_test, y_test = load_dataset(split_file, data_dir, mode='test')
    
    # 2. 数据清洗 (处理可能产生的NaN)
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    # 3. 定义模型
    # 使用随机森林，因为它对特征的非线性关系（如比率）处理较好
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    # 4. 训练
    print("\nTraining Random Forest Classifier...")
    clf.fit(X_train, y_train)
    
    # 5. 预测与评估
    print("\nEvaluating...")
    y_pred = clf.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 输出特征重要性，验证是否符合论文结论（b/a ratio是否重要）
    feature_names = ['b/a ratio', 'Approx AI', 'SDPTG Std', 'SDPTG Skew']
    importances = clf.feature_importances_
    print("\nFeature Importances:")
    for name, imp in zip(feature_names, importances):
        print(f"{name}: {imp:.4f}")

# ==========================================
# 执行入口
# ==========================================
if __name__ == "__main__":
    # 请修改为您的实际路径
    DATA_DIR = "./data_folder" 
    SPLIT_FILE = "./train_test_split.json"
    
    # 确保文件存在再运行
    if os.path.exists(SPLIT_FILE):
        run_baseline(DATA_DIR, SPLIT_FILE)
    else:
        print("请配置正确的路径")