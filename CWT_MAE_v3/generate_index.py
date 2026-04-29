import os
import json
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm

def process_directory(data_dir, output_file, min_length=1000):
    """
    扫描目录下的所有 pkl 文件，提取信息并生成训练索引。
    """
    index_data = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist.")
        return
        
    pkl_files = list(data_path.rglob('*.pkl'))
    print(f"Found {len(pkl_files)} pickle files in {data_dir}")
    
    valid_count = 0
    invalid_count = 0
    
    for pkl_file in tqdm(pkl_files, desc="Processing files"):
        try:
            with open(pkl_file, 'rb') as f:
                content = pickle.load(f)
                
            if 'data' not in content:
                invalid_count += 1
                continue
                
            data = content['data']
            
            # 检查维度 (M, L)
            if len(data.shape) == 1:
                length = data.shape[0]
            elif len(data.shape) == 2:
                length = data.shape[1]
            else:
                invalid_count += 1
                continue
                
            # 过滤掉过短的信号
            if length < min_length:
                invalid_count += 1
                continue
                
            # 提取可能的 label
            label = content.get('label', 0)
            
            # 存入索引列表
            index_data.append({
                'path': str(pkl_file.absolute()),
                'len': length,
                'label': label
            })
            valid_count += 1
            
        except Exception as e:
            # print(f"Error processing {pkl_file}: {e}")
            invalid_count += 1
            
    print(f"\nSummary:")
    print(f"  Valid samples: {valid_count}")
    print(f"  Invalid/Skipped samples: {invalid_count}")
    print(f"  Total samples: {len(pkl_files)}")
    
    # 写入 JSON
    with open(output_file, 'w') as f:
        json.dump(index_data, f, indent=2)
        
    print(f"Index successfully saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate dataset index for CWT_MAE_v3")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the .pkl files")
    parser.add_argument('--output', type=str, default='train_index.json', help="Output JSON file path")
    parser.add_argument('--min_len', type=int, default=1000, help="Minimum signal length to keep")
    
    args = parser.parse_args()
    process_directory(args.data_dir, args.output, args.min_len)
