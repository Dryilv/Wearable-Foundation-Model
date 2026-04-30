"""
将 numpy 2.x 保存的 pkl 文件转换为 numpy 1.x 兼容格式。

本地运行（需要 numpy >= 2.0）:
    python reserialize_pkl_local.py --data_dir /path/to/pkl/files
    python reserialize_pkl_local.py --data_dir /path/to/pkl/files --split_file split.json --dry-run
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ============================================================
# 将 numpy 2.x 的对象转换为 numpy 1.x 兼容格式
# ============================================================
def convert_for_numpy1x(obj):
    """
    将 numpy 对象转换为 numpy 1.x 兼容的 pickle 格式。
    核心思路：用 np.save/np.load 的格式保存 array，避免 numpy._core.* 引用。
    """
    if isinstance(obj, np.ndarray):
        # 方法：先转为 bytes，再重建为干净的 array
        # 这样 pickle 时不会引用 numpy._core 内部模块
        dtype = obj.dtype
        if dtype == object:
            return np.array(obj, dtype=object)
        # 用 tobytes + frombuffer 重建
        return np.frombuffer(obj.tobytes(), dtype=dtype).reshape(obj.shape)
    elif isinstance(obj, dict):
        return {k: convert_for_numpy1x(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        converted = [convert_for_numpy1x(item) for item in obj]
        return tuple(converted) if isinstance(obj, tuple) else converted
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


class Numpy1xPickler:
    """
    自定义 Pickler，拦截所有 numpy._core.* 的引用，替换为 numpy.*。
    """
    def __init__(self, protocol=None):
        self.protocol = protocol or pickle.HIGHEST_PROTOCOL

    def dump(self, obj, file):
        # 先转换对象
        clean_obj = convert_for_numpy1x(obj)
        pickle.dump(clean_obj, file, protocol=self.protocol)

    def dumps(self, obj):
        import io
        clean_obj = convert_for_numpy1x(obj)
        buf = io.BytesIO()
        pickle.dump(clean_obj, buf, protocol=self.protocol)
        return buf.getvalue()


def safe_load_pkl(file_path):
    """安全加载 pkl 文件。"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f), None
    except Exception as e:
        return None, str(e)


def reserialize_file(src_path, dst_path, dry_run=False):
    """重新序列化单个文件。"""
    data, err = safe_load_pkl(str(src_path))
    if data is None:
        return False, f'load_failed: {err}'

    if dry_run:
        return True, 'ok'

    # 用 numpy 1.x 兼容格式保存
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dst_path, 'wb') as f:
        pickler = Numpy1xPickler(protocol=4)  # protocol 4 兼容性更好
        pickler.dump(data, f)

    return True, 'ok'


def get_all_pkl_files(data_dir, split_file=None):
    """获取所有需要处理的 pkl 文件。"""
    filenames = []

    if split_file and os.path.exists(split_file):
        with open(split_file, 'r') as f:
            splits = json.load(f)
        for split_entries in splits.values():
            if isinstance(split_entries, list):
                for entry in split_entries:
                    if isinstance(entry, str):
                        filenames.append(entry)
                    elif isinstance(entry, dict):
                        p = entry.get('path') or entry.get('file') or entry.get('filename')
                        if p:
                            filenames.append(p)
        filenames = list(set(filenames))
    else:
        for p in Path(data_dir).rglob('*.pkl'):
            filenames.append(str(p))

    return filenames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--split_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--backup', action='store_true', default=True)
    parser.add_argument('--sample-errors', type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f'ERROR: {data_dir} not found')
        sys.exit(1)

    filenames = get_all_pkl_files(data_dir, args.split_file)
    if not filenames:
        print('No pkl files found.')
        return

    resolved = []
    for fname in filenames:
        if os.path.isabs(fname):
            resolved.append(Path(fname))
        else:
            resolved.append(data_dir / fname)

    print(f'Found {len(resolved)} files')
    if args.dry_run:
        print('*** DRY RUN ***')

    success = 0
    failed = 0
    errors = []

    for src in tqdm(resolved, desc='Converting'):
        if not src.exists():
            failed += 1
            continue

        dst = Path(args.output_dir) / src.name if args.output_dir else src

        if args.backup and not args.dry_run and dst == src:
            backup = src.with_suffix(src.suffix + '.bak')
            if not backup.exists():
                import shutil
                shutil.copy2(src, backup)

        ok, msg = reserialize_file(src, dst, dry_run=args.dry_run)
        if ok:
            success += 1
        else:
            failed += 1
            if len(errors) < args.sample_errors:
                errors.append((src.name, msg))

    print(f'\nDone! Success: {success}, Failed: {failed}')
    if errors:
        print(f'\nError samples:')
        for name, msg in errors:
            print(f'  {name}: {msg}')


if __name__ == '__main__':
    main()
