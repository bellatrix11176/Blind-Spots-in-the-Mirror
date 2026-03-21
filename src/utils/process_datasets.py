# ============================================================
# process_datasets.py
# Converts raw JSONL files from data/raw/ to processed JSONL
# files in data/processed/ ready for benchmark evaluation.
#
# Fixes applied:
#   - is_answerable cast to bool (Task 1)
#   - expected_confidence_range parsed to list (Task 2)
#   - Drops columns not needed by the SDK task functions
#   - Validates required columns are present before writing
#
# Run from project root:
#   python src/utils/process_datasets.py
# ============================================================

import ast
import json
import os
import sys

import pandas as pd


# ── Paths ─────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")


def ensure_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_jsonl(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(records)


def save_jsonl(df: pd.DataFrame, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
    print(f"  Saved: {os.path.relpath(path, ROOT)}")


def validate_columns(df: pd.DataFrame, required: list, task_name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ERROR: {task_name} is missing columns: {missing}")
        sys.exit(1)


# ── Task 1: Metacognitive Knowledge Boundary ─────────────────
def process_task1():
    print("\nProcessing Task 1 — Knowledge Boundary...")

    raw_path = os.path.join(RAW_DIR, "task1_knowledge_raw.jsonl")
    out_path = os.path.join(PROCESSED_DIR, "task1_knowledge_processed.jsonl")

    df = load_jsonl(raw_path)
    print(f"  Loaded {len(df)} items")

    # Required columns for the SDK task function
    required = ["question", "is_answerable", "correct_answer", "expected_model_behavior"]
    validate_columns(df, required, "Task 1")

    # Fix: cast is_answerable to bool
    df["is_answerable"] = df["is_answerable"].astype(bool)

    # Fix: fill null correct_answer for unanswerable items
    df["correct_answer"] = df["correct_answer"].fillna("")

    # Keep only columns the SDK task function needs
    keep = ["question", "is_answerable", "correct_answer", "expected_model_behavior"]
    df_out = df[keep].copy()

    # Validation report
    answerable = df_out["is_answerable"].sum()
    unanswerable = (~df_out["is_answerable"]).sum()
    print(f"  Answerable items:   {answerable}")
    print(f"  Unanswerable items: {unanswerable}")

    save_jsonl(df_out, out_path)


# ── Task 2: Metacognitive Monitoring ─────────────────────────
def process_task2():
    print("\nProcessing Task 2: Metacognitive Monitoring...")

    raw_path = os.path.join(RAW_DIR, "task2_monitoring_raw.jsonl")
    out_path = os.path.join(PROCESSED_DIR, "task2_monitoring_processed.jsonl")

    df = load_jsonl(raw_path)
    print(f"  Loaded {len(df)} items")

    required = ["question", "correct_answer", "epistemic_status", "expected_confidence_range"]
    validate_columns(df, required, "Task 2")

    # Fix: parse expected_confidence_range from string to list if needed
    def parse_range(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            return ast.literal_eval(val)
        return [0, 100]

    df["expected_confidence_range"] = df["expected_confidence_range"].apply(parse_range)

    # Validate all ranges are [low, high] with 0 <= low < high <= 100
    def validate_range(r):
        return (
            isinstance(r, list)
            and len(r) == 2
            and 0 <= r[0] < r[1] <= 100
        )

    bad_ranges = df[~df["expected_confidence_range"].apply(validate_range)]
    if len(bad_ranges) > 0:
        print(f"  WARNING: {len(bad_ranges)} items have invalid confidence ranges:")
        print(bad_ranges[["id", "expected_confidence_range"]].to_string())

    # Keep only columns the SDK task function needs
    keep = ["question", "correct_answer", "epistemic_status", "expected_confidence_range"]
    df_out = df[keep].copy()

    # Validation report
    status_counts = df_out["epistemic_status"].value_counts()
    print("  Epistemic status breakdown:")
    for status, count in status_counts.items():
        print(f"    {status}: {count}")

    save_jsonl(df_out, out_path)


# ── Task 3: Metacognitive Control ────────────────────────────
def process_task3():
    print("\nProcessing Task 3: Metacognitive Control...")

    raw_path = os.path.join(RAW_DIR, "task3_control_raw.jsonl")
    out_path = os.path.join(PROCESSED_DIR, "task3_control_processed.jsonl")

    df = load_jsonl(raw_path)
    print(f"  Loaded {len(df)} items")

    required = [
        "control_type",
        "turn_1_prompt",
        "turn_2_prompt",
        "turn_2_correct_behavior",
    ]
    validate_columns(df, required, "Task 3")

    # Validate control_type values
    valid_types = {
        "sycophancy_resistance",
        "error_detection",
        "self_correction",
        "calibrated_abstention",
        "confidence_revision_up",
        "confidence_revision_down",
    }
    bad_types = df[~df["control_type"].isin(valid_types)]
    if len(bad_types) > 0:
        print(f"  WARNING: {len(bad_types)} items have unrecognised control_type values:")
        print(bad_types[["id", "control_type"]].to_string())

    # Keep only columns the SDK task function needs
    keep = ["control_type", "turn_1_prompt", "turn_2_prompt", "turn_2_correct_behavior"]
    df_out = df[keep].copy()

    # Validation report
    type_counts = df_out["control_type"].value_counts()
    print("  Control type breakdown:")
    for ctype, count in type_counts.items():
        print(f"    {ctype}: {count}")

    save_jsonl(df_out, out_path)


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Dataset Processing — Blind Spots in the Mirror")
    print("  Aulabaugh (2026b) Metacognition Benchmark")
    print("=" * 55)

    ensure_dirs()
    process_task1()
    process_task2()
    process_task3()

    print("\n" + "=" * 55)
    expected_files = [
        os.path.join(PROCESSED_DIR, "task1_knowledge_processed.jsonl"),
        os.path.join(PROCESSED_DIR, "task2_monitoring_processed.jsonl"),
        os.path.join(PROCESSED_DIR, "task3_control_processed.jsonl"),
    ]
    all_present = True
    for fpath in expected_files:
        fname = os.path.basename(fpath)
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
            print(f"  VERIFIED: {fname}")
        else:
            print(f"  MISSING or EMPTY: {fname}")
            all_present = False

    if all_present:
        print("  Status: all output files confirmed on disk.")
    else:
        print("  Status: one or more output files missing - check errors above.")
        import sys
        sys.exit(1)
    print("=" * 55)
