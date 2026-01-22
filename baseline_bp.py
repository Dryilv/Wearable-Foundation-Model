import os
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import skew, kurtosis, entropy
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 信号处理与特征提取工具
# ==============================================================================
def butter_bandpass_filter(data, lowcut=0.5, highcut=8.0, fs=100, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def extract_features(signal, fs=100):
    """
    提取 PPG 的形态学特征 (基于单周期分析)
    """
    features = {}
    
    # 1. 寻找波峰和波谷
    # distance=40 对应心率 150bpm，防止误检
    peaks, properties = find_peaks(signal, distance=fs*0.4, prominence=0.5)
    valleys, _ = find_peaks(-signal, distance=fs*0.4, prominence=0.5)
    
    if len(peaks) < 2 or len(valleys) < 2:
        return {k: 0 for k in ['sys_time', 'dia_time', 'pulse_width_50', 'peak_valley_ratio']}

    # 确保第一个是波谷，最后一个是波峰，方便配对
    if valleys[0] > peaks[0]: peaks = peaks[1:]
    if len(peaks) == 0: return {k: 0 for k in ['sys_time', 'dia_time', 'pulse_width_50', 'peak_valley_ratio']}
    if valleys[-1] < peaks[-1]: peaks = peaks[:-1]
    
    # 截取完整周期进行分析
    num_cycles = min(len(peaks), len(valleys)-1)
    sys_times = []
    dia_times = []
    widths = []
    
    for i in range(num_cycles):
        # 关键点索引
        idx_start = valleys[i]
        idx_peak = peaks[i]
        idx_end = valleys[i+1]
        
        # A. 收缩期上升时间 (Systolic Time)
        t_sys = (idx_peak - idx_start) / fs
        sys_times.append(t_sys)
        
        # B. 舒张期下降时间 (Diastolic Time)
        t_dia = (idx_end - idx_peak) / fs
        dia_times.append(t_dia)
        
        # C. 半高宽 (Width at 50% height)
        # 简单的线性插值估算
        cycle_wave = signal[idx_start:idx_end]
        cycle_peak_val = signal[idx_peak]
        cycle_min_val = min(signal[idx_start], signal[idx_end])
        half_height = (cycle_peak_val + cycle_min_val) / 2
        
        # 找左右交点
        # (这里简化处理，实际可用 scipy.signal.peak_widths)
        w = 0
        try:
            # 左侧交点
            left_cross = np.where(signal[idx_start:idx_peak] > half_height)[0][0] + idx_start
            # 右侧交点
            right_cross = np.where(signal[idx_peak:idx_end] < half_height)[0][0] + idx_peak
            w = (right_cross - left_cross) / fs
        except:
            w = 0.1 # fallback
        widths.append(w)

    features['sys_time_mean'] = np.mean(sys_times)
    features['dia_time_mean'] = np.mean(dia_times)
    features['sys_dia_ratio'] = np.mean(sys_times) / (np.mean(dia_times) + 1e-6)
    features['pulse_width_mean'] = np.mean(widths)
    
    return features

# 在 extract_features 主函数中调用它并合并字典

# ==============================================================================
# 2. 数据加载与构建
# ==============================================================================
def load_dataset(signal_dir, excel_path, fs=100, target_len=1400):
    print("Loading Excel metadata...")
    df_meta = pd.read_excel(excel_path)
    
    # 建立 ID 映射
    meta_dict = {}
    for _, row in df_meta.iterrows():
        sid = int(row['subject_ID'])
        meta_dict[sid] = {
            'Age': row['Age(year)'],
            'Height': row['Height(cm)'],
            'Weight': row['Weight(kg)'],
            'BMI': row['BMI(kg/m^2)'],
            'Sex': 1 if str(row['Sex(M/F)']).upper().strip() == 'F' else 0,
            'HR_static': row['Heart Rate(b/m)'],
            'SBP': row['Systolic Blood Pressure(mmHg)'],
            'DBP': row['Diastolic Blood Pressure(mmHg)']
        }
    
    print("Processing signals and extracting features...")
    data_rows = []
    
    files = [f for f in os.listdir(signal_dir) if f.endswith('.txt')]
    
    for fname in files:
        try:
            sid = int(fname.split('_')[0])
            if sid not in meta_dict: continue
            
            # 读取信号
            path = os.path.join(signal_dir, fname)
            df_sig = pd.read_csv(path, sep='\t', header=None, engine='python')
            raw_signal = df_sig.values.flatten()
            raw_signal = raw_signal[~np.isnan(raw_signal)]
            
            # 降采样 (1000Hz -> 100Hz)
            factor = 10
            if len(raw_signal) >= factor:
                limit = (len(raw_signal) // factor) * factor
                signal = raw_signal[:limit].reshape(-1, factor).mean(axis=1)
            else:
                signal = raw_signal
                
            # 截断
            if len(signal) > target_len:
                signal = signal[:target_len]
            
            # --- 核心：提取特征 ---
            feats = extract_features(signal, fs=fs)
            
            # 合并人口学特征
            row_data = feats.copy()
            row_data.update(meta_dict[sid]) # 加入 Age, BMI, SBP, DBP 等
            row_data['subject_id'] = sid    # 记录 ID 用于划分
            
            data_rows.append(row_data)
            
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue
            
    return pd.DataFrame(data_rows)

# ==============================================================================
# 3. 训练与评估 (XGBoost)
# ==============================================================================
def train_baseline(df):
    # 准备数据
    # 排除 ID 和 目标值，其他都是特征
    feature_cols = [c for c in df.columns if c not in ['subject_id', 'SBP', 'DBP']]
    target_cols = ['SBP', 'DBP']
    
    X = df[feature_cols]
    y = df[target_cols]
    groups = df['subject_id'] # 用于 GroupKFold
    
    print(f"\nFeature Count: {len(feature_cols)}")
    print(f"Features: {feature_cols}")
    
    # 使用 GroupKFold 进行交叉验证 (模拟 Subject-wise Split)
    # 这样可以保证验证集的病人是训练集没见过的
    gkf = GroupKFold(n_splits=5)
    
    mae_sbp_list = []
    mae_dbp_list = []
    r2_sbp_list = []
    
    fold = 1
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 标准化 (对 XGBoost 不是必须的，但对某些特征有帮助)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # --- 模型 1: SBP ---
        model_sbp = xgb.XGBRegressor(
            n_estimators=500, 
            learning_rate=0.05, 
            max_depth=6, 
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )
        model_sbp.fit(X_train_scaled, y_train['SBP'])
        pred_sbp = model_sbp.predict(X_val_scaled)
        
        # --- 模型 2: DBP ---
        model_dbp = xgb.XGBRegressor(
            n_estimators=500, 
            learning_rate=0.05, 
            max_depth=6, 
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )
        model_dbp.fit(X_train_scaled, y_train['DBP'])
        pred_dbp = model_dbp.predict(X_val_scaled)
        
        # 计算指标
        mae_sbp = mean_absolute_error(y_val['SBP'], pred_sbp)
        mae_dbp = mean_absolute_error(y_val['DBP'], pred_dbp)
        r2_sbp = r2_score(y_val['SBP'], pred_sbp)
        
        mae_sbp_list.append(mae_sbp)
        mae_dbp_list.append(mae_dbp)
        r2_sbp_list.append(r2_sbp)
        
        print(f"Fold {fold} | MAE SBP: {mae_sbp:.2f} | MAE DBP: {mae_dbp:.2f} | R2 SBP: {r2_sbp:.4f}")
        fold += 1
        
    print("\n" + "="*30)
    print(f"Average Baseline Results (5-Fold CV):")
    print(f"MAE SBP: {np.mean(mae_sbp_list):.2f} ± {np.std(mae_sbp_list):.2f}")
    print(f"MAE DBP: {np.mean(mae_dbp_list):.2f} ± {np.std(mae_dbp_list):.2f}")
    print(f"R2 SBP : {np.mean(r2_sbp_list):.4f}")
    print("="*30)
    
    # 输出特征重要性 (以 SBP 模型为例，取最后一次 fold)
    print("\nTop 10 Important Features for SBP:")
    importances = model_sbp.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(10):
        print(f"{i+1}. {feature_cols[indices[i]]}: {importances[indices[i]]:.4f}")

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    # 配置路径
    SIGNAL_DIR = r'C:\Users\Administrator\Desktop\ppgbp\PPG-BP_Database (1)\ppg_bp\data'  # 修改为你的路径
    EXCEL_PATH = r'C:\Users\Administrator\Desktop\ppgbp\PPG-BP_Database (1)\ppg_bp\dataset.xlsx'  # 修改为你的路径
    
    # 1. 加载并提取特征
    if os.path.exists('baseline_features.csv'):
        print("Loading cached features from baseline_features.csv...")
        df = pd.read_csv('baseline_features.csv')
    else:
        df = load_dataset(SIGNAL_DIR, EXCEL_PATH)
        df.to_csv('baseline_features.csv', index=False)
        print("Features saved to baseline_features.csv")
    
    # 2. 训练 Baseline
    if len(df) > 0:
        train_baseline(df)
    else:
        print("No data loaded.")