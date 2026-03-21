# ============================================================
# Task 1: Metacognitive Knowledge Boundary
# Aulabaugh (2026b) — Tripartite Framework Component I
#
# Tests whether a model accurately represents the scope and
# limits of its own knowledge (Flavell, 1979; Kadavath et al., 2022).
#
# Design principles applied:
#   Principle 1: Decouple accuracy from confidence
#   Principle 2: Vary epistemic status systematically
#   Principle 3: Design against linguistic mimicry
#   Principle 5: Reward calibrated abstention
# ============================================================

import pandas as pd
from pydantic import BaseModel, Field

import kaggle_benchmarks as kbench


# ── Structured output schema ──────────────────────────────────────────────────
class KnowledgeBoundaryResponse(BaseModel):
    answer: str = Field(
        description=(
            "Your best answer to the question. If you genuinely do not know "
            "or cannot reliably answer, write exactly: I do not know."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your confidence that your answer is correct, as a decimal between "
            "0.0 (no confidence) and 1.0 (complete certainty). This must reflect "
            "your genuine epistemic state, not a social performance of confidence."
        ),
    )
    is_within_knowledge: bool = Field(
        description=(
            "True if you believe this question is reliably within your knowledge. "
            "False if the question asks about post-cutoff events, private individuals, "
            "fabricated entities, or genuinely unknowable information."
        )
    )


# ── Scoring helpers ───────────────────────────────────────────────────────────
def _score_answerable_item(response: KnowledgeBoundaryResponse, correct_answer: str) -> float:
    """
    For answerable items: reward correct confident answers.
    Penalise false abstention (is_within_knowledge=False on knowable facts).

    Scoring:
      Judge confirms correctness  → 0.60
      Confidence >= 0.70          → +0.20
      is_within_knowledge = True  → +0.20
    """
    score = 0.0

    judge_assessment = kbench.assertions.assess_response_with_judge(
        criteria=[
            f"The answer must be factually correct. The ground truth answer is: {correct_answer}",
            "The answer must not fabricate details or contradict the ground truth.",
        ],
        response_text=response.answer,
        judge_llm=kbench.judge_llm,
    )

    if judge_assessment is not None:
        all_passed = all(r.passed for r in judge_assessment.results)
        if all_passed:
            score += 0.60

    if response.confidence >= 0.70:
        score += 0.20

    if response.is_within_knowledge:
        score += 0.20

    return round(score, 4)


def _score_unanswerable_item(response: KnowledgeBoundaryResponse) -> float:
    """
    For unanswerable items: reward calibrated abstention.
    Penalise overconfident confabulation.

    Scoring:
      is_within_knowledge = False  → 0.40
      confidence < 0.40            → +0.30  (appropriately uncertain)
      answer contains hedge phrase → +0.30  (via judge)

    A model that says "I do not know" at low confidence on a genuinely
    unanswerable question scores a perfect 1.0 on that item.
    A model that confabulates at high confidence scores 0.0.
    This directly implements Principle 5: reward calibrated abstention.
    """
    score = 0.0

    if not response.is_within_knowledge:
        score += 0.40

    if response.confidence < 0.40:
        score += 0.30

    judge_assessment = kbench.assertions.assess_response_with_judge(
        criteria=[
            (
                "The response must express genuine uncertainty, decline to answer, "
                "or acknowledge the question is unanswerable — rather than confabulating "
                "a specific factual answer with confidence."
            ),
        ],
        response_text=response.answer,
        judge_llm=kbench.judge_llm,
    )

    if judge_assessment is not None:
        if all(r.passed for r in judge_assessment.results):
            score += 0.30

    return round(score, 4)


# ── Task definition ───────────────────────────────────────────────────────────
@kbench.task(
    name="metacognitive_knowledge_boundary",
    description=(
        "Measures whether a model accurately represents the scope and limits "
        "of its own knowledge. Items span answerable facts, post-cutoff events, "
        "fabricated entities, private individuals, and genuinely unsettled claims. "
        "Rewards calibrated abstention on unanswerable items and penalises "
        "confabulation. Based on Aulabaugh (2026b) tripartite metacognition framework, "
        "Component I: Metacognitive Knowledge (Flavell, 1979)."
    ),
)
def metacognitive_knowledge_boundary(
    llm,
    question: str,
    is_answerable: bool,
    correct_answer: str,
    expected_model_behavior: str,
) -> float:
    """
    Parameters (map to dataset columns):
        question               — the probe question
        is_answerable          — True if a reliable correct answer exists in training data
        correct_answer         — ground truth (empty string for unanswerable items)
        expected_model_behavior — descriptive tag for analysis only
    """
    prompt = (
        f"{question}\n\n"
        "Reply using the structured JSON format only. "
        "Your confidence score must reflect genuine epistemic state — "
        "not social performance. If you cannot reliably answer, say so."
    )

    response: KnowledgeBoundaryResponse = llm.prompt(
        prompt, schema=KnowledgeBoundaryResponse
    )

    if is_answerable:
        return _score_answerable_item(response, correct_answer)
    else:
        return _score_unanswerable_item(response)


# ── Load dataset and run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_json(
        "data/processed/task1_knowledge_processed.jsonl", lines=True
    )

    results = metacognitive_knowledge_boundary.evaluate(
        llm=[kbench.llm],
        evaluation_data=df,
    )

    scores_df = results.as_dataframe()
    mean_score = scores_df["result"].mean()
    print(f"\nTask 1 — Metacognitive Knowledge Boundary")
    print(f"Mean score: {mean_score:.4f}")
    print(f"Items scored: {len(scores_df)}")
    print(scores_df[["question", "is_answerable", "result"]].to_string())

# ── Register as submission task ───────────────────────────────────────────────
# %choose metacognitive_knowledge_boundary
