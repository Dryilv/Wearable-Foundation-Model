import os
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm
import argparse
import shutil

def check_file(file_path, signal_len):
    """
    根据 DownstreamClassificationDataset 的逻辑检查文件是否有效。
    返回 (is_valid, reason)
    """
    if not os.path.exists(file_path):
        return False, "文件不存在"
    
    try:
        with open(file_path, 'rb') as f:
            content = pickle.load(f)
        
        # --- 1. 加载数据 ---
        if isinstance(content, dict) and 'data' in content:
            raw_data = content['data']
        else:
            raw_data = content
            
        if isinstance(raw_data, list):
            raw_data = np.array(raw_data)

        if raw_data is None or (isinstance(raw_data, np.ndarray) and raw_data.size == 0):
            return False, "数据为空"

        if raw_data.ndim == 0:
            return False, "数据维度为 0"

        if raw_data.ndim == 1:
            raw_data = raw_data[np.newaxis, :] # (1, L)
        
        raw_signal = raw_data.astype(np.float32) # (M, L_raw)
        
        if raw_signal.shape[1] == 0:
            return False, "信号长度为 0"

        # --- 2. 检查数值有效性 ---
        if np.any(np.isnan(raw_signal)) or np.any(np.isinf(raw_signal)):
            return False, "包含 NaN 或 Inf"

        # --- 3. 检查 Robust Normalization 是否会产生无效值 ---
        # 模仿 _robust_norm 的核心逻辑
        q25 = np.percentile(raw_signal, 25, axis=1, keepdims=True)
        q75 = np.percentile(raw_signal, 75, axis=1, keepdims=True)
        iqr = q75 - q25
        # iqr 处理逻辑已在 dataset 中通过 np.where(iqr < 1e-6, 1.0, iqr) 保证鲁棒
        
        return True, None
    except Exception as e:
        return False, f"加载失败: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="清洗下游任务数据集：移除损坏的 pkl 文件并同步更新 split JSON。")
    parser.add_argument('--data_root', type=str, required=True, help="数据集根目录 (包含 pkl 文件的目录)")
    parser.add_argument('--split_file', type=str, required=True, help="split JSON 文件路径")
    parser.add_argument('--signal_len', type=int, default=3000, help="目标信号长度 (默认 3000)")
    parser.add_argument('--dry_run', action='store_true', help="只显示将要删除的文件，不执行实际删除和更新")
    parser.add_argument('--backup', action='store_true', help="更新前备份原始 JSON 文件")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.split_file):
        print(f"错误: 找不到 split 文件: {args.split_file}")
        return

    print(f"正在加载 split 文件: {args.split_file}")
    with open(args.split_file, 'r') as f:
        try:
            splits = json.load(f)
        except Exception as e:
            print(f"错误: 无法解析 JSON 文件: {e}")
            return
    
    new_splits = {}
    total_files_checked = 0
    total_removed = 0
    files_to_delete = []

    for mode, file_list in splits.items():
        if not isinstance(file_list, list):
            print(f"跳过非列表项: {mode}")
            new_splits[mode] = file_list
            continue
            
        print(f"\n正在检查 [{mode}] 集合 ({len(file_list)} 个文件)...")
        new_file_list = []
        for filename in tqdm(file_list):
            total_files_checked += 1
            file_path = os.path.join(args.data_root, filename)
            is_valid, reason = check_file(file_path, args.signal_len)
            
            if is_valid:
                new_file_list.append(filename)
            else:
                total_removed += 1
                files_to_delete.append((filename, file_path, reason))
        
        new_splits[mode] = new_file_list
        print(f"  [{mode}] 结果: {len(file_list)} -> {len(new_file_list)} (移除 {len(file_list) - len(new_file_list)} 个)")

    if total_removed == 0:
        print("\n未发现需要清洗的文件。")
        return

    print(f"\n清洗统计:")
    print(f"  总检查文件数: {total_files_checked}")
    print(f"  待移除文件总数: {total_removed}")

    if args.dry_run:
        print("\n[Dry Run] 模式：不会修改任何文件。")
        print("待移除文件示例 (前 10 个):")
        for name, path, reason in files_to_delete[:10]:
            print(f"  - {name}: {reason}")
    else:
        # 1. 备份 JSON
        if args.backup:
            backup_path = args.split_file + ".bak"
            shutil.copy2(args.split_file, backup_path)
            print(f"已备份原始 JSON 到: {backup_path}")
        
        # 2. 更新 JSON
        with open(args.split_file, 'w') as f:
            json.dump(new_splits, f, indent=4)
        print(f"已同步更新 split JSON 文件: {args.split_file}")
        
        # 3. 物理删除文件
        print(f"正在物理删除 {len(files_to_delete)} 个原始文件...")
        deleted_count = 0
        for name, fpath, reason in tqdm(files_to_delete):
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    deleted_count += 1
                except Exception as e:
                    print(f"  [错误] 无法删除 {fpath}: {e}")
            else:
                # 如果文件本来就不存在，我们也算它“移除”成功
                deleted_count += 1
        
        print(f"\n清洗完成。")
        print(f"  - JSON 中移除条目: {total_removed}")
        print(f"  - 物理删除文件: {deleted_count}")

if __name__ == "__main__":
    main()
