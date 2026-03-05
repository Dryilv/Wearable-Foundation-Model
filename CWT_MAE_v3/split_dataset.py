import json
import os
import pickle
import numpy as np
import argparse
import shutil

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs):
        return iterable

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle_file(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def split_dataset(input_json, data_root, output_dir_ecg, output_dir_ppg, output_json_ecg, output_json_ppg):
    # 1. Load Input JSON
    if not os.path.exists(input_json):
        print(f"Error: Input JSON file '{input_json}' not found.")
        return

    print(f"Loading dataset index from {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        dataset_index = json.load(f)

    # 2. Create Output Directories
    os.makedirs(output_dir_ecg, exist_ok=True)
    os.makedirs(output_dir_ppg, exist_ok=True)
    print(f"Created output directories: {output_dir_ecg}, {output_dir_ppg}")

    # 3. Determine Format (Pretrain vs Finetune)
    is_finetune_format = isinstance(dataset_index, dict)
    
    files_to_process = [] # List of (source_path, relative_path)

    if is_finetune_format:
        print("Detected Finetune/Downstream format (Dict of lists).")
        # Structure: {'train': ['f1.pkl', ...], 'val': ['f2.pkl', ...]}
        # We need to process all unique files.
        seen_files = set()
        for split_name, file_list in dataset_index.items():
            if isinstance(file_list, list):
                for fname in file_list:
                    if fname not in seen_files:
                        if data_root:
                            src = os.path.join(data_root, fname)
                        else:
                            src = fname # Assume absolute or relative to CWD if data_root not provided
                        files_to_process.append((src, fname))
                        seen_files.add(fname)
    else:
        print("Detected Pretrain format (List of dicts).")
        # Structure: [{'path': '...', 'label': ...}, ...]
        for item in dataset_index:
            path = item.get('path')
            if path:
                # Use absolute path if available, or combine with data_root
                if data_root and not os.path.isabs(path):
                    src = os.path.join(data_root, path)
                else:
                    src = path
                
                # For output filename, we use basename or relative path if possible
                # But pretrain format usually has absolute paths.
                # We will use basename for the new file in output dir.
                rel_path = os.path.basename(src)
                files_to_process.append((src, rel_path))

    # 4. Process Files
    print(f"Processing {len(files_to_process)} unique files...")
    
    success_count = 0
    
    for src_path, rel_path in tqdm(files_to_process):
        if not os.path.exists(src_path):
            print(f"Warning: File not found {src_path}, skipping.")
            continue

        try:
            content = load_pickle_file(src_path)
            
            # Handle different content structures
            # dataset_finetune.py handles: dict with 'data', or direct data
            raw_signal = None
            if isinstance(content, dict) and 'data' in content:
                raw_signal = content['data']
            elif isinstance(content, (np.ndarray, list)):
                raw_signal = content
            else:
                # Maybe content is the data itself if it's not a dict?
                raw_signal = content
            
            if isinstance(raw_signal, list):
                raw_signal = np.array(raw_signal)

            # Extract ECG and PPG
            # Logic: 
            # - If 2 channels: 0:ECG, 1:PPG (User specified)
            # - If >=5 channels: 0:ECG, 4:PPG (Legacy dataset.py compatibility)
            
            ecg_signal = None
            ppg_signal = None

            if isinstance(raw_signal, np.ndarray):
                if raw_signal.ndim == 2:
                    if raw_signal.shape[0] == 2:
                        # User specified: Dual channel (ECG, PPG)
                        ecg_signal = raw_signal[0, :]
                        ppg_signal = raw_signal[1, :]
                    elif raw_signal.shape[0] >= 5:
                        # Legacy format
                        ecg_signal = raw_signal[0, :] # Shape (L,)
                        ppg_signal = raw_signal[4, :] # Shape (L,)
                        if success_count == 0:
                            print(f"Note: Detected >=5 channels in {rel_path}. Using index 0 (ECG) and 4 (PPG).")
                    else:
                         print(f"Warning: Unexpected channel count {raw_signal.shape[0]} in {src_path}, skipping.")
                         continue
                elif raw_signal.ndim == 1:
                    # Single channel data - cannot split
                    print(f"Warning: Single channel data found in {src_path}, skipping.")
                    continue
            
            if ecg_signal is None or ppg_signal is None:
                 print(f"Warning: Could not split channels for {src_path} (Shape: {np.shape(raw_signal)}), skipping.")
                 continue

            # Prepare new content
            # If original was dict, keep other keys. If not, create dict or keep as array?
            # dataset_finetune.py handles both. Let's stick to dict for safety/metadata.
            
            content_ecg = {'data': ecg_signal}
            content_ppg = {'data': ppg_signal}
            
            if isinstance(content, dict):
                for k, v in content.items():
                    if k != 'data':
                        content_ecg[k] = v
                        content_ppg[k] = v
            else:
                # Original was just array, so we wrap it.
                # But wait, if original was just array, where is the label?
                # dataset_finetune handles this.
                pass

            # Define new paths
            # Maintain directory structure if rel_path has subdirs?
            # User said "json file can be directly copied", which implies filenames don't change.
            # So we should save to output_dir/rel_path.
            
            new_path_ecg = os.path.join(output_dir_ecg, rel_path)
            new_path_ppg = os.path.join(output_dir_ppg, rel_path)
            
            # Ensure subdirs exist
            os.makedirs(os.path.dirname(new_path_ecg), exist_ok=True)
            os.makedirs(os.path.dirname(new_path_ppg), exist_ok=True)
            
            save_pickle_file(content_ecg, new_path_ecg)
            save_pickle_file(content_ppg, new_path_ppg)
            
            success_count += 1

        except Exception as e:
            print(f"Error processing {src_path}: {e}")
            continue

    print(f"Successfully processed {success_count}/{len(files_to_process)} files.")

    # 5. Save New JSON Files
    if is_finetune_format:
        # For finetune format, we just copy the JSON because filenames (relative paths) are preserved.
        # We assume the user will run finetune with --data_root pointing to the new dirs.
        print(f"Copying index files (structure preserved)...")
        with open(output_json_ecg, 'w', encoding='utf-8') as f:
            json.dump(dataset_index, f, indent=2)
        
        with open(output_json_ppg, 'w', encoding='utf-8') as f:
            json.dump(dataset_index, f, indent=2)
            
    else:
        # For pretrain format, we need to update paths to absolute paths of new files.
        new_index_ecg = []
        new_index_ppg = []
        
        for item in dataset_index:
            path = item.get('path')
            if not path: continue
            
            # Reconstruct logic to find where we saved it
            if data_root and not os.path.isabs(path):
                src = os.path.join(data_root, path)
            else:
                src = path
            rel_path = os.path.basename(src)
            
            new_path_ecg = os.path.abspath(os.path.join(output_dir_ecg, rel_path))
            new_path_ppg = os.path.abspath(os.path.join(output_dir_ppg, rel_path))
            
            item_ecg = item.copy()
            item_ecg['path'] = new_path_ecg
            new_index_ecg.append(item_ecg)
            
            item_ppg = item.copy()
            item_ppg['path'] = new_path_ppg
            new_index_ppg.append(item_ppg)
            
        print(f"Saving new index files (paths updated)...")
        with open(output_json_ecg, 'w', encoding='utf-8') as f:
            json.dump(new_index_ecg, f, indent=2)
        with open(output_json_ppg, 'w', encoding='utf-8') as f:
            json.dump(new_index_ppg, f, indent=2)

    print(f"Done. Saved to {output_json_ecg} and {output_json_ppg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into ECG and PPG datasets (Supports Pretrain & Finetune formats).")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON index file.")
    parser.add_argument("--data_root", type=str, default=None, help="Root directory for data files (required if JSON has relative paths).")
    parser.add_argument("--output_dir_ecg", type=str, default="../data_ecg", help="Directory to save ECG data.")
    parser.add_argument("--output_dir_ppg", type=str, default="../data_ppg", help="Directory to save PPG data.")
    parser.add_argument("--output_json_ecg", type=str, default="train_index_ecg.json", help="Output JSON file for ECG dataset.")
    parser.add_argument("--output_json_ppg", type=str, default="train_index_ppg.json", help="Output JSON file for PPG dataset.")
    
    args = parser.parse_args()
    
    split_dataset(args.input_json, args.data_root, args.output_dir_ecg, args.output_dir_ppg, args.output_json_ecg, args.output_json_ppg)
