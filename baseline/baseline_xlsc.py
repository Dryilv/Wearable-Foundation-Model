import os
import json
import pickle
import numpy as np
import scipy.signal as signal
from tqdm import tqdm
import warnings

# Scikit-learn 和 imbalanced-learn 相关库
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# 忽略特定的 RuntimeWarning
warnings.filterwarnings("ignore", message="overflow encountered in multiply")
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# ==========================================
# 1. 信号处理与特征提取 (针对心率失常优化)
# ==========================================
class ArrhythmiaFeatureExtractor:
    def __init__(self):
        pass

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        y = signal.filtfilt(b, a, data.astype(np.float64))
        return y

    def get_sdptg(self, ppg_signal):
        """计算二阶导数 (加速度波)"""
        first_deriv = np.gradient(ppg_signal)
        second_deriv = np.gradient(first_deriv)
        return second_deriv

    def extract_hrv_features(self, peaks, fs):
        """
        提取心率变异性(HRV)时域特征 - 这是心率失常检测的核心
        """
        if len(peaks) < 2:
            return [np.nan] * 5

        # 计算 RR 间期 (这里是 PP 间期，单位：毫秒)
        rr_intervals = np.diff(peaks) / fs * 1000
        
        # 简单的异常值过滤 (去除 <300ms 或 >2000ms 的生理不可能值)
        valid_rr = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
        
        if len(valid_rr) < 2:
            return [np.nan] * 5

        # 1. Mean RR (平均心跳周期)
        mean_rr = np.mean(valid_rr)
        # 2. Heart Rate (心率 BPM)
        mean_hr = 60000 / mean_rr if mean_rr > 0 else 0
        # 3. SDNN (RR间期的标准差 - 整体变异性)
        sdnn = np.std(valid_rr, ddof=1)
        # 4. RMSSD (相邻RR间期差值的均方根 - 反映副交感神经活性，对房颤等敏感)
        diff_rr = np.diff(valid_rr)
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
        # 5. pNN50 (相邻RR间期差值超过50ms的百分比)
        nn50 = np.sum(np.abs(diff_rr) > 50)
        pnn50 = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0

        return [mean_rr, mean_hr, sdnn, rmssd, pnn50]

    def extract_morphology_features(self, sdptg_beat):
        """提取单心拍的形态学特征 (保留之前的逻辑)"""
        if len(sdptg_beat) < 5: return [0, 0, 0, 0]

        a_idx = np.argmax(sdptg_beat)
        a_val = sdptg_beat[a_idx]
        
        if a_idx < len(sdptg_beat) - 1:
            b_idx = a_idx + np.argmin(sdptg_beat[a_idx:])
            b_val = sdptg_beat[b_idx]
        else:
            b_val = 0

        b_a_ratio = b_val / a_val if a_val != 0 else 0
        sdptg_std = np.std(sdptg_beat)
        sdptg_skew = np.mean((sdptg_beat - np.mean(sdptg_beat))**3) if sdptg_std > 1e-6 else 0
        approx_ai = (b_val - np.mean(sdptg_beat[a_idx:])) / a_val if a_val != 0 and a_idx < len(sdptg_beat) -1 else 0

        return [b_a_ratio, approx_ai, sdptg_std, sdptg_skew]

    def process_sample(self, ppg_raw, fs):
        # 1. 预处理
        filtered_ppg = self.butter_bandpass_filter(ppg_raw, 0.5, 8.0, fs)
        sdptg_signal = self.get_sdptg(filtered_ppg)
        
        # 2. 寻峰 (对于心率失常，寻峰的准确性至关重要)
        # distance=fs*0.3 (300ms) 防止检测到过近的波峰
        peaks, _ = signal.find_peaks(filtered_ppg, distance=int(fs*0.3), height=np.std(filtered_ppg)*0.5)
        
        if len(peaks) < 2:
            # 返回全NaN向量 (5个HRV特征 + 4个形态特征 = 9个特征)
            return np.full(9, np.nan)

        # 3. 提取 HRV 特征 (Rhythm)
        hrv_feats = self.extract_hrv_features(peaks, fs)

        # 4. 提取形态特征 (Morphology) - 对所有心拍取平均
        beat_morph_list = []
        for i in range(len(peaks) - 1):
            start, end = peaks[i], peaks[i+1]
            # 稍微扩宽一点窗口以包含完整的波形
            beat_sdptg = sdptg_signal[start:end]
            if len(beat_sdptg) < 10: continue
            feats = self.extract_morphology_features(beat_sdptg)
            beat_morph_list.append(feats)
            
        if not beat_morph_list:
            avg_morph_feats = [np.nan] * 4
        else:
            avg_morph_feats = np.nanmean(beat_morph_list, axis=0).tolist()

        # 5. 合并特征向量
        final_features = np.array(hrv_feats + avg_morph_feats)
        return final_features

# ==========================================
# 2. 数据加载 (通用逻辑)
# ==========================================
def load_dataset(split_file, data_dir, mode='train'):
    with open(split_file, 'r') as f:
        splits = json.load(f)
    file_list = splits[mode]
    extractor = ArrhythmiaFeatureExtractor()
    X, y = [], []
    
    print(f"正在处理 {mode} 数据 (心率失常任务)...")
    for file_name in tqdm(file_list):
        file_path = os.path.join(data_dir, file_name)
        try:
            with open(file_path, 'rb') as f:
                sample = pickle.load(f)
            
            # 数据读取逻辑 (根据你的数据格式可能需要微调)
            raw_data = sample['data']
            ppg_signal = raw_data[0, :] if raw_data.ndim > 1 else raw_data
            ppg_signal = ppg_signal.astype(np.float64)
            fs = sample['sampling_rate']
            
            # 标签读取逻辑
            label = next((l['class'] for l in sample['label'] if 'class' in l), None)
            
            if label is None: continue
            
            # 特征提取
            features = extractor.process_sample(ppg_signal, fs)
            
            # 只有当特征不全为NaN时才添加
            if not np.all(np.isnan(features)):
                X.append(features)
                y.append(label)
                
        except Exception as e:
            # print(f"处理文件 {file_name} 时出错: {e}") # 调试时可打开
            continue
            
    return np.array(X), np.array(y)

# ==========================================
# 3. 模型训练与超参数搜索
# ==========================================
def run_arrhythmia_baseline(data_dir, split_file):
    """
    心率失常检测 Baseline
    """
    # 1. 加载数据
    X_train, y_train = load_dataset(split_file, data_dir, mode='train')
    X_test, y_test = load_dataset(split_file, data_dir, mode='test')

    print(f"\n训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

    # 2. 构建 Pipeline
    # 心率失常数据通常极度不平衡（正常样本多，异常样本少），SMOTE 很有必要
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # 使用中位数填充NaN，对异常值更鲁棒
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 3. 定义超参数网格
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__class_weight': ['balanced', None]
    }

    # 4. 运行 GridSearchCV
    # 使用 f1_macro 或 f1_weighted，因为多分类或不平衡二分类中 accuracy 具有误导性
    grid_search = GridSearchCV(estimator=pipeline, 
                               param_grid=param_grid, 
                               scoring='f1_macro', 
                               cv=3, 
                               n_jobs=-1, 
                               verbose=2)
    
    print("\n开始训练与超参数搜索...")
    grid_search.fit(X_train, y_train)

    # 5. 结果输出
    print("\n搜索完成！")
    print(f"最佳验证集 F1-Macro: {grid_search.best_score_:.4f}")
    print("最佳参数:", grid_search.best_params_)

    # 6. 最终评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\n=== 最终测试集评估报告 ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

# ==========================================
# 执行入口
# ==========================================
if __name__ == "__main__":
    # 请修改为心率失常数据的实际路径
    # 假设数据结构与冠心病数据一致
    DATA_DIR = "/home/bml/storage/mnt/v-044d0fb740b04ad3/org/WFM/processed_dataset/data" 
    SPLIT_FILE = "/home/bml/storage/mnt/v-044d0fb740b04ad3/org/WFM/processed_dataset/split.json"
    
    if os.path.exists(SPLIT_FILE) and os.path.exists(DATA_DIR):
        run_arrhythmia_baseline(DATA_DIR, SPLIT_FILE)
    else:
        print(f"错误: 路径不存在。\nData: {DATA_DIR}\nSplit: {SPLIT_FILE}")