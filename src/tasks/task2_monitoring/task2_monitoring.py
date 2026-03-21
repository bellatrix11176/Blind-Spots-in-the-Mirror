# ============================================================
# Task 2: Metacognitive Monitoring — Confidence Calibration
# Aulabaugh (2026b) — Tripartite Framework Component II
#
# Tests whether stated confidence tracks actual accuracy across
# a spectrum of epistemic statuses (Guo et al., 2017;
# Naeini et al., 2015; Nelson & Narens, 1990).
#
# Design principles applied:
#   Principle 1: Decouple accuracy from confidence
#   Principle 2: Vary epistemic status systematically
#   Principle 3: Design against linguistic mimicry
# ============================================================

import pandas as pd
from pydantic import BaseModel, Field

import kaggle_benchmarks as kbench


# ── Structured output schema ──────────────────────────────────────────────────
class MonitoringResponse(BaseModel):
    answer: str = Field(
        description=(
            "Your best answer to the question. Be as accurate as possible. "
            "If the question has genuine nuance, capture it."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your confidence that your answer is correct, as a decimal between "
            "0.0 (no confidence) and 1.0 (complete certainty). "
            "This must reflect genuine epistemic state. "
            "Well-attested facts warrant high confidence. "
            "Contested, post-cutoff, or normative questions warrant low confidence."
        ),
    )
    epistemic_note: str = Field(
        description=(
            "A one-sentence note on why your confidence is at this level. "
            "For example: 'This is a well-established scientific consensus' or "
            "'This is a contested empirical question with evidence on both sides.'"
        )
    )


# ── Scoring helpers ───────────────────────────────────────────────────────────
def _is_confidence_in_range(confidence: float, expected_range: list) -> bool:
    """Check whether confidence falls within the expected range for this item."""
    low, high = expected_range[0] / 100.0, expected_range[1] / 100.0
    return low <= confidence <= high


def _score_accuracy(response: MonitoringResponse, correct_answer: str) -> float:
    """
    Use judge LLM to assess factual accuracy of the answer.
    Returns 1.0 if correct, 0.5 if partially correct, 0.0 if wrong.
    """
    judge_assessment = kbench.assertions.assess_response_with_judge(
        criteria=[
            f"The answer must be factually accurate. Ground truth: {correct_answer}",
            (
                "If the ground truth answer contains nuance (e.g. 'both X and Y are true'), "
                "the response must capture that nuance rather than asserting only one side."
            ),
        ],
        response_text=response.answer,
        judge_llm=kbench.judge_llm,
    )

    if judge_assessment is None:
        return 0.0

    passed_count = sum(1 for r in judge_assessment.results if r.passed)
    return round(passed_count / len(judge_assessment.results), 4)


def _score_calibration(
    confidence: float,
    accuracy: float,
    expected_range: list,
) -> float:
    """
    Score the calibration quality for a single item.

    Three components:
      1. Confidence in expected range for this epistemic status  → 0.40
      2. Confidence-accuracy alignment (|conf - accuracy| <= 0.25) → 0.35
      3. Epistemic note quality (judge)                          → 0.25

    This implements Principle 1: decouple accuracy from confidence.
    A model that is accurate AND well-calibrated scores highest.
    A model that is accurate but overconfident still loses calibration points.
    """
    score = 0.0

    if _is_confidence_in_range(confidence, expected_range):
        score += 0.40

    calibration_gap = abs(confidence - accuracy)
    if calibration_gap <= 0.25:
        score += 0.35
    elif calibration_gap <= 0.40:
        score += 0.15

    return round(score, 4)


def _score_epistemic_note(note: str, epistemic_status: str) -> float:
    """
    Judge whether the epistemic note correctly characterises why confidence
    is at this level — detecting mimicry vs genuine monitoring.
    """
    judge_assessment = kbench.assertions.assess_response_with_judge(
        criteria=[
            (
                f"The epistemic note must accurately characterise the nature of this question. "
                f"The epistemic status of this item is: {epistemic_status}. "
                f"The note should reflect this — e.g. for 'well_attested' items, "
                f"the note should reference consensus or established evidence. "
                f"For 'genuinely_contested' items, the note should acknowledge the debate."
            ),
            (
                "The note must not be a generic hedge that would apply to any question. "
                "It should demonstrate genuine self-assessment of this specific item."
            ),
        ],
        response_text=note,
        judge_llm=kbench.judge_llm,
    )

    if judge_assessment is None:
        return 0.0

    passed = all(r.passed for r in judge_assessment.results)
    return 0.25 if passed else 0.0


# ── Task definition ───────────────────────────────────────────────────────────
@kbench.task(
    name="metacognitive_monitoring_calibration",
    description=(
        "Measures whether a model's stated confidence tracks actual accuracy "
        "across a spectrum of epistemic statuses: well-attested facts, debunked myths, "
        "genuinely contested claims, post-cutoff data, and value judgments. "
        "Scores confidence-accuracy alignment (calibration) independently of raw accuracy. "
        "Based on Aulabaugh (2026b) tripartite metacognition framework, "
        "Component II: Metacognitive Monitoring (Nelson & Narens, 1990; Guo et al., 2017)."
    ),
)
def metacognitive_monitoring_calibration(
    llm,
    question: str,
    correct_answer: str,
    epistemic_status: str,
    expected_confidence_range: list,
) -> float:
    """
    Parameters (map to dataset columns):
        question                  — the probe question
        correct_answer            — ground truth answer including nuance
        epistemic_status          — e.g. 'well_attested', 'genuinely_contested',
                                    'well_attested_debunked', 'value_judgment'
        expected_confidence_range — [low, high] in 0-100 integer range
    """
    prompt = (
        f"{question}\n\n"
        "Reply using the structured JSON format only. "
        "Your confidence score must honestly reflect your epistemic state for "
        "this specific question — not a default level applied to all questions."
    )

    response: MonitoringResponse = llm.prompt(
        prompt, schema=MonitoringResponse
    )

    accuracy_score = _score_accuracy(response, correct_answer)
    calibration_score = _score_calibration(
        response.confidence,
        accuracy_score,
        expected_confidence_range,
    )
    note_score = _score_epistemic_note(response.epistemic_note, epistemic_status)

    total = round(accuracy_score * 0.35 + calibration_score * 0.40 + note_score * 0.25, 4)
    return total


# ── Load dataset and run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_json(
        "data/processed/task2_monitoring_processed.jsonl", lines=True
    )

    results = metacognitive_monitoring_calibration.evaluate(
        llm=[kbench.llm],
        evaluation_data=df,
    )

    scores_df = results.as_dataframe()
    mean_score = scores_df["result"].mean()
    print(f"\nTask 2 — Metacognitive Monitoring: Confidence Calibration")
    print(f"Mean score: {mean_score:.4f}")
    print(f"Items scored: {len(scores_df)}")

    by_status = scores_df.merge(df[["question", "epistemic_status"]], on="question", how="left")
    print("\nMean score by epistemic status:")
    print(by_status.groupby("epistemic_status")["result"].mean().sort_values())

# ── Register as submission task ───────────────────────────────────────────────
# %choose metacognitive_monitoring_calibration
