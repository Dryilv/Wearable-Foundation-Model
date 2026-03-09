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


def extract_data_and_label(content):
    if isinstance(content, dict):
        data = content.get("data", None)
        label = content.get("label", None)
    else:
        data = content
        label = None
    if data is None:
        return None, label
    data = np.asarray(data)
    return data, label


def label_has_nan_or_inf(label):
    if label is None:
        return False
    if isinstance(label, dict):
        values = list(label.values())
    elif isinstance(label, (list, tuple)):
        values = []
        for item in label:
            if isinstance(item, dict) and "class" in item:
                values.append(item["class"])
            elif isinstance(item, dict):
                values.extend(item.values())
            else:
                values.append(item)
    else:
        values = [label]
    for value in values:
        try:
            v = float(value)
        except Exception:
            continue
        if np.isnan(v) or np.isinf(v):
            return True
    return False


def sample_is_bad(file_path):
    if not os.path.exists(file_path):
        return True, "missing_file"
    try:
        content = load_pickle(file_path)
    except Exception:
        return True, "read_error"
    data, label = extract_data_and_label(content)
    if data is None:
        return True, "missing_data"
    if np.isnan(data).any() or np.isinf(data).any():
        return True, "nan_or_inf_in_data"
    if label_has_nan_or_inf(label):
        return True, "nan_or_inf_in_label"
    return False, "ok"


def get_entry_path(entry):
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and "path" in entry:
        return entry["path"]
    return None


def clean_split_json(json_data, data_root):
    removed_details = []
    cleaned = {}
    for split_name, entries in json_data.items():
        if not isinstance(entries, list):
            cleaned[split_name] = entries
            continue
        kept = []
        for idx, entry in enumerate(tqdm(entries, desc=f"clean {split_name}", leave=False)):
            rel_path = get_entry_path(entry)
            if rel_path is None:
                removed_details.append((split_name, idx, "invalid_entry"))
                continue
            abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(data_root, rel_path)
            bad, reason = sample_is_bad(abs_path)
            if bad:
                removed_details.append((split_name, idx, reason))
                continue
            kept.append(entry)
        cleaned[split_name] = kept
    return cleaned, removed_details


def clean_index_json(json_data, data_root):
    removed_details = []
    kept = []
    for idx, entry in enumerate(tqdm(json_data, desc="clean index", leave=False)):
        rel_path = get_entry_path(entry)
        if rel_path is None:
            removed_details.append(("index", idx, "invalid_entry"))
            continue
        abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(data_root, rel_path)
        bad, reason = sample_is_bad(abs_path)
        if bad:
            removed_details.append(("index", idx, reason))
            continue
        kept.append(entry)
    return kept, removed_details


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_summary(removed_details):
    reason_count = {}
    for _, _, reason in removed_details:
        reason_count[reason] = reason_count.get(reason, 0) + 1
    print(f"removed total: {len(removed_details)}")
    for reason, count in sorted(reason_count.items(), key=lambda x: x[0]):
        print(f"  {reason}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--inplace", action="store_true")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"json file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    if isinstance(json_data, dict):
        cleaned, removed_details = clean_split_json(json_data, args.data_root)
    elif isinstance(json_data, list):
        cleaned, removed_details = clean_index_json(json_data, args.data_root)
    else:
        raise ValueError("json root must be dict(split) or list(index)")

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
    print_summary(removed_details)


if __name__ == "__main__":
    main()
