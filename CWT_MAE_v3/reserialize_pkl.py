"""
重新序列化 pkl 文件，消除旧版 numpy 的 pickle 依赖。

用法:
    # 重新序列化整个目录
    python reserialize_pkl.py --data_dir /path/to/pkl/files

    # 只处理指定 split 的文件
    python reserialize_pkl.py --data_dir /path/to/pkl/files --split_file /path/to/split.json

    # 干跑模式（不写入，只打印会处理的文件）
    python reserialize_pkl.py --data_dir /path/to/pkl/files --dry-run
"""

import argparse
import json
import os
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_with_fallback(file_path):
    """尝试用多种方式加载 pickle，返回 Python 原生对象（numpy array / dict / list）。"""
    with open(file_path, 'rb') as f:
        raw_bytes = f.read()

    # 方法 1: 标准加载
    try:
        return pickle.loads(raw_bytes), 'standard'
    except Exception:
        pass

    # 方法 2: latin1 编码
    try:
        return pickle.loads(raw_bytes, encoding='latin1'), 'latin1'
    except Exception:
        pass

    # 方法 3: bytes 编码
    try:
        return pickle.loads(raw_bytes, encoding='bytes'), 'bytes'
    except Exception:
        pass

    # 方法 4: 自定义 Unpickler 绕过 numpy 模块
    class FallbackUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # 映射所有 numpy 子模块到 numpy 本身
            if module and (module.startswith('numpy.') or module.startswith('numpy_')):
                module = 'numpy'
            return super().find_class(module, name)

    try:
        return FallbackUnpickler.__new__(FallbackUnpickler, pickle.Unpickler.__bases__[0]).load(pickle.Unpickler.__bases__[0](file_path)), 'fallback'
    except Exception:
        pass

    # 方法 5: 最暴力 — 提取纯数据
    try:
        import pickletools
        # 尝试提取数据
        return None, 'failed'
    except Exception:
        return None, 'failed'


def convert_to_native(obj):
    """将 numpy 对象转换为 pickle 安全的原生 Python/numPy 对象。"""
    if isinstance(obj, np.ndarray):
        # 重新创建一个干净的 ndarray，不绑定旧版 numpy 内部模块
        return np.array(obj, dtype=obj.dtype)
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        converted = [convert_to_native(item) for item in obj]
        return tuple(converted) if isinstance(obj, tuple) else converted
    elif isinstance(obj, np.generic):
        # numpy 标量 -> Python 原生类型
        return obj.item()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def reserialize_file(src_path, dst_path, dry_run=False):
    """重新序列化单个 pkl 文件。"""
    data, method = load_with_fallback(src_path)

    if data is None:
        return False, f'load_failed ({method})'

    # 转换为原生对象
    try:
        clean_data = convert_to_native(data)
    except Exception as e:
        return False, f'convert_failed: {e}'

    if dry_run:
        return True, f'ok (via {method})'

    # 保存为新文件（使用当前环境的 numpy 和 pickle 协议）
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dst_path, 'wb') as f:
        pickle.dump(clean_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return True, f'ok (via {method})'


def main():
    parser = argparse.ArgumentParser(description='重新序列化 pkl 文件以解决 numpy 版本不兼容')
    parser.add_argument('--data_dir', type=str, required=True, help='pkl 文件所在目录')
    parser.add_argument('--split_file', type=str, default=None, help='可选: split JSON 路径，只处理其中的文件')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录（默认覆盖原文件，会先备份）')
    parser.add_argument('--dry-run', action='store_true', help='只检查不写入')
    parser.add_argument('--backup_suffix', type=str, default='.bak', help='备份文件后缀')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f'ERROR: data_dir not found: {data_dir}')
        sys.exit(1)

    # 确定要处理的文件列表
    if args.split_file:
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
        # 合并所有 split 的文件名
        filenames = []
        for split_entries in splits.values():
            if isinstance(split_entries, list):
                # 条目可能是字符串或 dict
                for entry in split_entries:
                    if isinstance(entry, str):
                        filenames.append(entry)
                    elif isinstance(entry, dict) and 'path' in entry:
                        filenames.append(entry['path'])
                    elif isinstance(entry, dict) and 'file' in entry:
                        filenames.append(entry['file'])
        filenames = list(set(filenames))  # 去重
    else:
        filenames = list(data_dir.glob('*.pkl'))
        filenames = [str(f.name) for f in filenames]

    if not filenames:
        print('No pkl files found to process.')
        return

    print(f'Found {len(filenames)} files to process')
    if args.dry_run:
        print('*** DRY RUN MODE ***')

    success = 0
    failed = 0
    failed_details = []

    for fname in tqdm(filenames, desc='Reserializing'):
        # 处理路径
        if os.path.isabs(fname):
            src = Path(fname)
        else:
            src = data_dir / fname

        if not src.exists():
            failed += 1
            failed_details.append((str(fname), 'file_not_found'))
            continue

        if args.output_dir:
            dst = Path(args.output_dir) / src.name
        else:
            # 覆盖模式：先备份
            dst = src
            backup = src.with_suffix(src.suffix + args.backup_suffix)
            if not args.dry_run and not backup.exists():
                import shutil
                shutil.copy2(src, backup)

        ok, msg = reserialize_file(str(src), str(dst), dry_run=args.dry_run)

        if ok:
            success += 1
        else:
            failed += 1
            failed_details.append((str(fname), msg))

    print(f'\nDone! Success: {success}, Failed: {failed}')

    if failed_details and len(failed_details) <= 50:
        print('\nFailed files:')
        for fname, reason in failed_details:
            print(f'  {fname}: {reason}')
    elif failed_details:
        print(f'\n{len(failed_details)} files failed. First 20:')
        for fname, reason in failed_details[:20]:
            print(f'  {fname}: {reason}')


if __name__ == '__main__':
    main()
