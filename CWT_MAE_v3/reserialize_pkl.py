"""
重新序列化 pkl 文件，消除旧版 numpy 的 pickle 依赖。

用法:
    python reserialize_pkl.py --data_dir /path/to/pkl/files
    python reserialize_pkl.py --data_dir /path/to/pkl/files --split_file /path/to/split.json
    python reserialize_pkl.py --data_dir /path/to/pkl/files --dry-run
"""

import argparse
import json
import os
import pickle
import pickletools
import sys
import traceback
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ============================================================
# 核心：自定义 Unpickler，拦截所有 numpy 相关引用
# ============================================================
class NumpyCompatUnpickler(pickle.Unpickler):
    """
    兼容旧版 numpy 的 Unpickler。
    将 numpy._core.* / numpy.core.* 等旧模块路径映射到当前 numpy。
    同时对 numpy 内部函数做安全回退。
    """

    # numpy 内部函数到当前版本的映射
    NUMPY_REMAP = {
        ('numpy._core.numeric', '_frombuffer'): ('numpy', '_frombuffer'),
        ('numpy._core.multiarray', '_reconstruct'): ('numpy', '_reconstruct'),
        ('numpy._core.multiarray', '_frombuffer'): ('numpy', '_frombuffer'),
        ('numpy.core.numeric', '_frombuffer'): ('numpy', '_frombuffer'),
        ('numpy.core.multiarray', '_reconstruct'): ('numpy', '_reconstruct'),
    }

    def find_class(self, module, name):
        # 尝试重映射
        key = (module, name)
        if key in self.NUMPY_REMAP:
            module, name = self.NUMPY_REMAP[key]

        # 通用 numpy 子模块 -> numpy 映射
        if module and (module.startswith('numpy._core') or module.startswith('numpy.core')):
            module = 'numpy'

        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # 如果找不到，尝试从 numpy 根模块找
            if module != 'numpy' and module and 'numpy' in module:
                try:
                    return super().find_class('numpy', name)
                except Exception:
                    pass
            # 最后的回退
            raise


def safe_load_pickle(file_path):
    """
    安全加载 pickle 文件。
    返回 (data, error_string_or_None)。
    """
    # 方法 1: 自定义兼容 Unpickler
    try:
        with open(file_path, 'rb') as f:
            data = NumpyCompatUnpickler(f).load()
        return data, None
    except Exception as e1:
        err1 = str(e1)

    # 方法 2: encoding='latin1'
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data, None
    except Exception as e2:
        pass

    # 方法 3: encoding='bytes'
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        return data, None
    except Exception:
        pass

    # 方法 4: 标准加载
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data, None
    except Exception:
        pass

    return None, err1


def convert_to_native(obj):
    """将 numpy 对象转换为干净的 numpy 对象（用当前版本重建）。"""
    if isinstance(obj, np.ndarray):
        # 重新创建，剥离旧版内部绑定
        return np.array(obj, dtype=obj.dtype)
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        converted = [convert_to_native(item) for item in obj]
        return tuple(converted) if isinstance(obj, tuple) else converted
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.complexfloating):
        return complex(obj)
    else:
        return obj


def reserialize_file(src_path, dst_path, dry_run=False):
    """重新序列化单个 pkl 文件。返回 (success, message)。"""
    data, err = safe_load_pickle(str(src_path))

    if data is None:
        return False, f'load_failed: {err}'

    # 转换为干净的对象
    try:
        clean_data = convert_to_native(data)
    except Exception as e:
        return False, f'convert_failed: {e}'

    if dry_run:
        return True, 'ok'

    # 保存
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dst_path, 'wb') as f:
        pickle.dump(clean_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return True, 'ok'


def get_all_pkl_files(data_dir, split_file=None):
    """获取所有需要处理的 pkl 文件路径。"""
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
    parser = argparse.ArgumentParser(description='重新序列化 pkl 文件以解决 numpy 版本不兼容')
    parser.add_argument('--data_dir', type=str, required=True, help='pkl 文件所在目录')
    parser.add_argument('--split_file', type=str, default=None, help='可选: split JSON 路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录（默认覆盖原文件）')
    parser.add_argument('--dry-run', action='store_true', help='只检查不写入')
    parser.add_argument('--backup', action='store_true', default=True, help='覆盖前备份为 .bak')
    parser.add_argument('--sample_errors', type=int, default=5, help='打印详细错误的文件数')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f'ERROR: data_dir not found: {data_dir}')
        sys.exit(1)

    filenames = get_all_pkl_files(data_dir, args.split_file)

    if not filenames:
        print('No pkl files found.')
        return

    # 如果给了 split_file，文件名需要拼接 data_dir
    resolved_paths = []
    for fname in filenames:
        if os.path.isabs(fname):
            resolved_paths.append(Path(fname))
        else:
            resolved_paths.append(data_dir / fname)

    print(f'Found {len(resolved_paths)} files to process')
    if args.dry_run:
        print('*** DRY RUN MODE ***')

    success = 0
    failed = 0
    error_samples = []

    for src in tqdm(resolved_paths, desc='Reserializing'):
        if not src.exists():
            failed += 1
            continue

        if args.output_dir:
            dst = Path(args.output_dir) / src.name
        else:
            dst = src
            if args.backup and not args.dry_run:
                backup = src.with_suffix(src.suffix + '.bak')
                if not backup.exists():
                    import shutil
                    shutil.copy2(src, backup)

        ok, msg = reserialize_file(src, dst, dry_run=args.dry_run)

        if ok:
            success += 1
        else:
            failed += 1
            if len(error_samples) < args.sample_errors:
                error_samples.append((str(src.name), msg))

    print(f'\nDone! Success: {success}, Failed: {failed}')

    if error_samples:
        print(f'\nError samples (first {len(error_samples)}):')
        for fname, msg in error_samples:
            print(f'  {fname}: {msg}')


if __name__ == '__main__':
    main()
