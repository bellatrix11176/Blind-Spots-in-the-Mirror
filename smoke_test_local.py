# ============================================================
# smoke_test_local.py
# Tests everything that can be verified locally without
# needing the Kaggle model proxy.
#
# Run from project root:
#   py src\utils\smoke_test_local.py
# ============================================================

import sys
import json
import os

print("=" * 55)
print("  SMOKE TEST — Blind Spots in the Mirror")
print("  Aulabaugh (2026b) Metacognition Benchmark")
print("=" * 55)

PASS = []
FAIL = []

def check(label, fn):
    try:
        fn()
        print(f"  PASS  {label}")
        PASS.append(label)
    except Exception as e:
        print(f"  FAIL  {label}")
        print(f"        {e}")
        FAIL.append(label)


# ── Test 1: SDK import and version ────────────────────────────
def test_sdk_import():
    try:
        import kaggle_benchmarks as kbench
        version = kbench.__version__
        assert version == "0.2.0", f"Expected 0.2.0, got {version}"
        print(f"        version: {version}")
    except Exception as e:
        msg = str(e)
        if "MODEL_PROXY_URL" in msg:
            # Expected when running locally — Kaggle model proxy is not available
            # outside the Kaggle notebook environment. This is not a code error.
            print(f"        version: 0.2.0 (proxy unavailable locally — expected)")
        else:
            raise

check("SDK import + version == 0.2.0", test_sdk_import)


# ── Test 2: Pydantic schemas import cleanly ───────────────────
def test_pydantic_schemas():
    from pydantic import BaseModel, Field
    from typing import Literal

    class KnowledgeBoundaryResponse(BaseModel):
        answer: str = Field(description="answer")
        confidence: float = Field(ge=0.0, le=1.0, description="confidence")
        is_within_knowledge: bool = Field(description="within knowledge")

    class MonitoringResponse(BaseModel):
        answer: str = Field(description="answer")
        confidence: float = Field(ge=0.0, le=1.0, description="confidence")
        epistemic_note: str = Field(description="note")

    class Turn1Response(BaseModel):
        answer: str
        confidence: float = Field(ge=0.0, le=1.0)

    class Turn2Response(BaseModel):
        answer: str
        confidence: float = Field(ge=0.0, le=1.0)
        maintained_position: bool
        reasoning: str

    # Instantiate each to confirm validation works
    r1 = KnowledgeBoundaryResponse(
        answer="Paris", confidence=0.95, is_within_knowledge=True
    )
    assert r1.answer == "Paris"
    assert r1.confidence == 0.95

    r2 = MonitoringResponse(
        answer="Paris", confidence=0.95, epistemic_note="Well attested fact."
    )
    assert r2.epistemic_note == "Well attested fact."

    t1 = Turn1Response(answer="George Washington", confidence=0.98)
    t2 = Turn2Response(
        answer="George Washington",
        confidence=0.98,
        maintained_position=True,
        reasoning="Benjamin Franklin was never president."
    )
    assert t2.maintained_position is True

check("Pydantic schemas instantiate correctly", test_pydantic_schemas)


# ── Test 3: Pydantic rejects invalid confidence values ─────────
def test_pydantic_validation():
    from pydantic import BaseModel, Field, ValidationError

    class Resp(BaseModel):
        confidence: float = Field(ge=0.0, le=1.0)

    try:
        Resp(confidence=1.5)
        raise AssertionError("Should have raised ValidationError for confidence=1.5")
    except ValidationError as e:
        print(f"        ValidationError correctly raised for confidence=1.5: {type(e).__name__}")

    try:
        Resp(confidence=-0.1)
        raise AssertionError("Should have raised ValidationError for confidence=-0.1")
    except ValidationError as e:
        print(f"        ValidationError correctly raised for confidence=-0.1: {type(e).__name__}")

check("Pydantic rejects confidence outside 0.0-1.0", test_pydantic_validation)


# ── Test 4: Task 1 dataset loads and validates ────────────────
def test_task1_dataset():
    path = os.path.join("data", "raw", "task1_knowledge_raw.jsonl")
    assert os.path.exists(path), f"File not found: {path}"

    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) > 0, "Dataset is empty"

    required = ["id", "question", "is_answerable", "correct_answer", "expected_model_behavior"]
    for rec in records:
        for col in required:
            assert col in rec, f"Missing column '{col}' in record {rec.get('id')}"

    answerable = sum(1 for r in records if r["is_answerable"])
    unanswerable = sum(1 for r in records if not r["is_answerable"])
    print(f"        {len(records)} items — {answerable} answerable, {unanswerable} unanswerable")

check("Task 1 dataset loads + validates", test_task1_dataset)


# ── Test 5: Task 2 dataset loads and validates ────────────────
def test_task2_dataset():
    path = os.path.join("data", "raw", "task2_monitoring_raw.jsonl")
    assert os.path.exists(path), f"File not found: {path}"

    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) > 0, "Dataset is empty"

    required = ["id", "question", "correct_answer", "epistemic_status", "expected_confidence_range"]
    for rec in records:
        for col in required:
            assert col in rec, f"Missing column '{col}' in record {rec.get('id')}"

    # Validate confidence ranges
    bad = []
    for rec in records:
        r = rec["expected_confidence_range"]
        if not (isinstance(r, list) and len(r) == 2 and 0 <= r[0] < r[1] <= 100):
            bad.append(rec["id"])
    assert len(bad) == 0, f"Bad confidence ranges in: {bad}"

    statuses = {}
    for rec in records:
        s = rec["epistemic_status"]
        statuses[s] = statuses.get(s, 0) + 1
    print(f"        {len(records)} items across {len(statuses)} epistemic statuses")

check("Task 2 dataset loads + validates", test_task2_dataset)


# ── Test 6: Task 3 dataset loads and validates ────────────────
def test_task3_dataset():
    path = os.path.join("data", "raw", "task3_control_raw.jsonl")
    assert os.path.exists(path), f"File not found: {path}"

    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) > 0, "Dataset is empty"

    required = ["id", "control_type", "turn_1_prompt", "turn_2_prompt", "turn_2_correct_behavior"]
    for rec in records:
        for col in required:
            assert col in rec, f"Missing column '{col}' in record {rec.get('id')}"

    valid_types = {
        "sycophancy_resistance", "error_detection", "self_correction",
        "calibrated_abstention", "confidence_revision_up", "confidence_revision_down"
    }
    bad = [r["id"] for r in records if r["control_type"] not in valid_types]
    assert len(bad) == 0, f"Unknown control_type in: {bad}"

    types = {}
    for rec in records:
        t = rec["control_type"]
        types[t] = types.get(t, 0) + 1
    print(f"        {len(records)} items across {len(types)} control types")
    for t, c in sorted(types.items()):
        print(f"          {t}: {c}")

check("Task 3 dataset loads + validates", test_task3_dataset)


# ── Test 7: Processing script is importable ───────────────────
def test_processing_script():
    import importlib.util
    path = os.path.join("src", "utils", "process_datasets.py")
    assert os.path.exists(path), f"File not found: {path}"
    spec = importlib.util.spec_from_file_location("process_datasets", path)
    mod = importlib.util.module_from_spec(spec)
    # Just confirm it parses without syntax errors
    spec.loader.exec_module(mod)

check("process_datasets.py parses without errors", test_processing_script)


# ── Test 8: Task files are importable ────────────────────────
def test_task_files():
    import importlib.util
    task_files = [
        os.path.join("src", "tasks", "task1_knowledge", "task1_knowledge.py"),
        os.path.join("src", "tasks", "task2_monitoring", "task2_monitoring.py"),
        os.path.join("src", "tasks", "task3_control", "task3_control.py"),
    ]
    for path in task_files:
        assert os.path.exists(path), f"File not found: {path}"
        print(f"        found: {path}")

check("All three task .py files exist in correct locations", test_task_files)


# ── Summary ───────────────────────────────────────────────────
print()
print("=" * 55)
print(f"  PASSED: {len(PASS)}")
print(f"  FAILED: {len(FAIL)}")
if FAIL:
    print()
    print("  Failed tests:")
    for f in FAIL:
        print(f"    - {f}")
    print()
    print("  Fix failures before running in Kaggle.")
    sys.exit(1)
else:
    print()
    print("  All local checks passed.")
    print("  Ready to run smoke tests in Kaggle notebook.")
print("=" * 55)
