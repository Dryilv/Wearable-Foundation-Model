import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def get_entry_path(entry):
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and "path" in entry:
        return entry["path"]
    return None


def extract_data(content):
    if isinstance(content, dict):
        data = content.get("data", None)
    else:
        data = content
    if data is None:
        return None
    return np.asarray(data)


def sample_is_bad(file_path):
    if not os.path.exists(file_path):
        return True, "missing_file"
    try:
        content = load_pickle(file_path)
    except Exception:
        return True, "read_error"
    data = extract_data(content)
    if data is None:
        return True, "missing_data"
    try:
        if np.isnan(data).any():
            return True, "nan_in_data"
        if np.isinf(data).any():
            return True, "inf_in_data"
    except Exception:
        return True, "invalid_data_array"
    return False, "ok"


def clean_index_json(json_data, data_root):
    removed_details = []
    kept = []
    for idx, entry in enumerate(tqdm(json_data, desc="clean pretrain index", leave=False)):
        rel_path = get_entry_path(entry)
        if rel_path is None:
            removed_details.append((idx, "invalid_entry"))
            continue
        abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(data_root, rel_path)
        bad, reason = sample_is_bad(abs_path)
        if bad:
            removed_details.append((idx, reason))
            continue
        kept.append(entry)
    return kept, removed_details


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_summary(total_count, kept_count, removed_details):
    reason_count = {}
    for _, reason in removed_details:
        reason_count[reason] = reason_count.get(reason, 0) + 1
    print(f"input total:  {total_count}")
    print(f"kept total:   {kept_count}")
    print(f"removed total:{len(removed_details)}")
    for reason, count in sorted(reason_count.items(), key=lambda x: x[0]):
        print(f"  {reason}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--inplace", action="store_true")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"json file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    if not isinstance(json_data, list):
        raise ValueError("pretrain index json root must be a list")

    cleaned, removed_details = clean_index_json(json_data, args.data_root)

    if args.inplace:
        out_path = str(json_path)
    else:
        if args.output_json is None:
            out_path = str(json_path.with_name(f"{json_path.stem}_cleaned{json_path.suffix}"))
        else:
            out_path = args.output_json

    write_json(out_path, cleaned)
    print(f"input json:  {json_path}")
    print(f"output json: {out_path}")
    print_summary(len(json_data), len(cleaned), removed_details)


if __name__ == "__main__":
    main()
