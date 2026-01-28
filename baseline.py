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

# (可选) 如果您想尝试XGBoost，请取消下面的注释
# import xgboost as xgb

# 忽略特定的 RuntimeWarning
warnings.filterwarnings("ignore", message="overflow encountered in multiply")

# ==========================================
# 1. 信号处理与特征提取 (这部分与之前相同)
# ==========================================
class SDPTGFeatureExtractor:
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
        first_deriv = np.gradient(ppg_signal)
        second_deriv = np.gradient(first_deriv)
        return second_deriv

    def extract_beat_features(self, sdptg_beat):
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
        filtered_ppg = self.butter_bandpass_filter(ppg_raw, 0.5, 8.0, fs)
        sdptg_signal = self.get_sdptg(filtered_ppg)
        peaks, _ = signal.find_peaks(filtered_ppg, distance=int(fs*0.6), height=np.std(filtered_ppg)*0.5)
        
        if len(peaks) < 2:
            return np.full(4, np.nan)
            
        beat_features_list = []
        for i in range(len(peaks) - 1):
            start, end = peaks[i], peaks[i+1]
            beat_sdptg = sdptg_signal[start:end]
            if len(beat_sdptg) < 10: continue
            feats = self.extract_beat_features(beat_sdptg)
            beat_features_list.append(feats)
            
        if not beat_features_list:
            return np.full(4, np.nan)
            
        return np.nanmean(beat_features_list, axis=0)

# ==========================================
# 2. 数据加载 (这部分与之前相同)
# ==========================================
def load_dataset(split_file, data_dir, mode='train'):
    with open(split_file, 'r') as f:
        splits = json.load(f)
    file_list = splits[mode]
    extractor = SDPTGFeatureExtractor()
    X, y = [], []
    
    print(f"正在处理 {mode} 数据...")
    for file_name in tqdm(file_list):
        file_path = os.path.join(data_dir, file_name)
        try:
            with open(file_path, 'rb') as f:
                sample = pickle.load(f)
            raw_data = sample['data']
            ppg_signal = raw_data[0, :] if raw_data.ndim > 1 else raw_data
            ppg_signal = ppg_signal.astype(np.float64)
            fs = sample['sampling_rate']
            label = next((l['class'] for l in sample['label'] if 'class' in l), None)
            if label is None: continue
            features = extractor.process_sample(ppg_signal, fs)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")
            continue
    return np.array(X), np.array(y)

# ==========================================
# 3. 模型训练与超参数搜索 (核心优化部分)
# ==========================================
def run_hyperparameter_optimization(data_dir, split_file):
    """
    使用GridSearchCV进行模型训练和超参数优化
    """
    # 1. 加载数据
    X_train, y_train = load_dataset(split_file, data_dir, mode='train')
    X_test, y_test = load_dataset(split_file, data_dir, mode='test')

    # 2. 创建一个集成了“缺失值处理”、“过采样”和“分类”的流水线（Pipeline）
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 3. 定义要搜索的超参数网格
    # 注意：参数名称的格式是 '步骤名__参数名'
    # 这是一个示例网格，您可以根据需要扩展它。网格越大，搜索时间越长。
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__class_weight': [None, 'balanced'] # 即使有SMOTE，有时balanced也有用
    }

    # --- (可选) 如果您想尝试XGBoost，请注释掉上面的RandomForest部分，使用下面的 ---
    # pipeline = Pipeline([
    #     ('imputer', SimpleImputer(strategy='mean')),
    #     ('smote', SMOTE(random_state=42)),
    #     ('classifier', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
    # ])
    # param_grid = {
    #     'classifier__n_estimators': [100, 200],
    #     'classifier__max_depth': [5, 7, 10],
    #     'classifier__learning_rate': [0.05, 0.1]
    # }
    # --------------------------------------------------------------------------

    # 4. 配置并运行GridSearchCV
    # scoring='f1_macro'：在不均衡数据集上是比'accuracy'更好的评估指标
    # cv=3：使用3折交叉验证。可以增加到5以获得更稳健的结果，但会更慢
    # n_jobs=-1：使用所有可用的CPU核心来加速搜索
    # verbose=2：打印详细的搜索过程日志
    grid_search = GridSearchCV(estimator=pipeline, 
                               param_grid=param_grid, 
                               scoring='f1_macro', 
                               cv=3, 
                               n_jobs=-1, 
                               verbose=2)
    
    print("\n开始进行超参数搜索...")
    grid_search.fit(X_train, y_train)

    # 5. 打印搜索结果
    print("\n超参数搜索完成！")
    print(f"最佳交叉验证得分 (f1_macro): {grid_search.best_score_:.4f}")
    print("找到的最佳超参数组合:")
    print(grid_search.best_params_)

    # 6. 使用找到的最佳模型在测试集上进行最终评估
    print("\n在测试集上评估最佳模型...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\n最终测试集准确率:", accuracy_score(y_test, y_pred))
    print("\n最终测试集分类报告:")
    print(classification_report(y_test, y_pred, zero_division=0))


# ==========================================
# 执行入口
# ==========================================
if __name__ == "__main__":
    # 请修改为您的实际路径
    DATA_DIR = "/home/bml/storage/mnt/v-044d0fb740b04ad3/org/WFM/wearable_FM/downstream_data/ppgguanxinbing/sample_for_downstream" 
    SPLIT_FILE = "/home/bml/storage/mnt/v-044d0fb740b04ad3/org/WFM/wearable_FM/downstream_data/ppgguanxinbing/train_test_split.json"
    
    if os.path.exists(SPLIT_FILE) and os.path.exists(DATA_DIR):
        run_hyperparameter_optimization(DATA_DIR, SPLIT_FILE)
    else:
        print(f"错误: 请检查数据目录 ('{DATA_DIR}') 和切分文件 ('{SPLIT_FILE}') 是否存在。")