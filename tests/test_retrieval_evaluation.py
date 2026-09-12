from dataclasses import dataclass

import pytest

from app.retrieval.retrieval_evaluator import (
    RetrievalEvaluationCase,
    RetrievalEvaluator,
)


@dataclass
class FakeMemory:
    id: int


def make_result(memory_id: int) -> dict:
    return {
        "memory": FakeMemory(id=memory_id),
        "score": 0.9,
        "semantic_score": 0.9,
    }


def fake_search_factory(rankings):
    def fake_search(query, top_k=5):
        ids = rankings.get(query, [])
        return [
            make_result(memory_id)
            for memory_id in ids[:top_k]
        ]

    return fake_search


def test_precision_at_k():
    ranked_ids = [1, 2, 3, 4]
    relevant_ids = {1, 3}

    score = RetrievalEvaluator.precision_at_k(
        ranked_ids,
        relevant_ids,
        4,
    )

    assert score == 0.5


def test_recall_at_k():
    ranked_ids = [1, 2, 3, 4]
    relevant_ids = {1, 3, 5}

    score = RetrievalEvaluator.recall_at_k(
        ranked_ids,
        relevant_ids,
        4,
    )

    assert score == pytest.approx(2 / 3)


def test_hit_rate_at_k():
    ranked_ids = [1, 2, 3]

    assert (
        RetrievalEvaluator.hit_rate_at_k(
            ranked_ids,
            {3},
            3,
        )
        == 1.0
    )

    assert (
        RetrievalEvaluator.hit_rate_at_k(
            ranked_ids,
            {5},
            3,
        )
        == 0.0
    )


def test_reciprocal_rank():
    ranked_ids = [8, 4, 2, 9]
    relevant_ids = {2}

    score = RetrievalEvaluator.reciprocal_rank(
        ranked_ids,
        relevant_ids,
    )

    assert score == pytest.approx(1 / 3)


def test_reciprocal_rank_returns_zero_when_missing():
    ranked_ids = [8, 4, 2]

    score = RetrievalEvaluator.reciprocal_rank(
        ranked_ids,
        {99},
    )

    assert score == 0.0


def test_ndcg_at_k_perfect_ranking():
    ranked_ids = [1, 2, 3]
    relevant_ids = {1, 2}

    score = RetrievalEvaluator.ndcg_at_k(
        ranked_ids,
        relevant_ids,
        3,
    )

    assert score == pytest.approx(1.0)


def test_ndcg_at_k_penalizes_bad_ranking():
    ranked_ids = [3, 1, 2]
    relevant_ids = {1, 2}

    score = RetrievalEvaluator.ndcg_at_k(
        ranked_ids,
        relevant_ids,
        3,
    )

    assert 0.0 < score < 1.0


def test_duplicate_memory_ids_are_removed():
    results = [
        make_result(1),
        make_result(1),
        make_result(2),
    ]

    ranked_ids = RetrievalEvaluator._ranked_ids(results)

    assert ranked_ids == [1, 2]


def test_invalid_results_are_ignored():
    results = [
        make_result(1),
        {},
        {"memory": None},
        {"memory": FakeMemory(id=None)},
        make_result(2),
    ]

    ranked_ids = RetrievalEvaluator._ranked_ids(results)

    assert ranked_ids == [1, 2]


def test_evaluate_single_case():
    rankings = {
        "project query": [1, 2, 3],
    }

    evaluator = RetrievalEvaluator(
        fake_search_factory(rankings)
    )

    case = RetrievalEvaluationCase(
        query="project query",
        relevant_memory_ids=frozenset({1, 3}),
        top_k=3,
    )

    result = evaluator.evaluate_case(case)

    assert result["query"] == "project query"
    assert result["retrieved_ids"] == [1, 2, 3]
    assert result["relevant_ids"] == [1, 3]

    assert result["precision_at_k"] == pytest.approx(2 / 3)
    assert result["recall_at_k"] == 1.0
    assert result["hit_rate_at_k"] == 1.0
    assert result["mrr"] == 1.0
    assert result["ndcg_at_k"] > 0.0


def test_evaluate_multiple_cases():
    rankings = {
        "projects": [1, 2, 3],
        "location": [4, 5, 6],
        "skills": [7, 8, 9],
    }

    evaluator = RetrievalEvaluator(
        fake_search_factory(rankings)
    )

    cases = [
        RetrievalEvaluationCase(
            query="projects",
            relevant_memory_ids=frozenset({1, 2}),
            top_k=3,
        ),
        RetrievalEvaluationCase(
            query="location",
            relevant_memory_ids=frozenset({4}),
            top_k=3,
        ),
        RetrievalEvaluationCase(
            query="skills",
            relevant_memory_ids=frozenset({8}),
            top_k=3,
        ),
    ]

    result = evaluator.evaluate(cases)

    assert len(result["cases"]) == 3

    summary = result["summary"]

    assert summary["precision_at_k"] > 0.0
    assert summary["recall_at_k"] > 0.0
    assert summary["hit_rate_at_k"] == 1.0
    assert summary["mrr"] > 0.0
    assert summary["ndcg_at_k"] > 0.0


def test_evaluate_empty_dataset():
    evaluator = RetrievalEvaluator(
        fake_search_factory({})
    )

    result = evaluator.evaluate([])

    assert result["cases"] == []

    for metric in RetrievalEvaluator.METRICS:
        assert result["summary"][metric] == 0.0


def test_no_relevant_memory_is_safe():
    rankings = {
        "unknown": [1, 2, 3],
    }

    evaluator = RetrievalEvaluator(
        fake_search_factory(rankings)
    )

    case = RetrievalEvaluationCase(
        query="unknown",
        relevant_memory_ids=frozenset(),
        top_k=3,
    )

    result = evaluator.evaluate_case(case)

    assert result["precision_at_k"] == 0.0
    assert result["recall_at_k"] == 0.0
    assert result["hit_rate_at_k"] == 0.0
    assert result["mrr"] == 0.0
    assert result["ndcg_at_k"] == 0.0


def test_thresholds_pass_when_quality_is_good():
    summary = {
        "precision_at_k": 0.80,
        "recall_at_k": 0.90,
        "hit_rate_at_k": 1.00,
        "mrr": 0.85,
        "ndcg_at_k": 0.88,
    }

    RetrievalEvaluator.assert_thresholds(
        summary,
        {
            "precision_at_k": 0.70,
            "hit_rate_at_k": 0.90,
            "mrr": 0.80,
        },
    )


def test_thresholds_fail_on_quality_regression():
    summary = {
        "precision_at_k": 0.50,
        "recall_at_k": 0.60,
        "hit_rate_at_k": 0.70,
        "mrr": 0.40,
        "ndcg_at_k": 0.50,
    }

    with pytest.raises(
        AssertionError,
        match="Retrieval quality regression",
    ):
        RetrievalEvaluator.assert_thresholds(
            summary,
            {
                "hit_rate_at_k": 0.80,
                "mrr": 0.60,
            },
        )


def test_evaluation_case_rejects_empty_query():
    with pytest.raises(
        ValueError,
        match="Evaluation query cannot be empty",
    ):
        RetrievalEvaluationCase(
            query="   ",
            relevant_memory_ids=frozenset({1}),
        )


def test_evaluation_case_rejects_invalid_top_k():
    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        RetrievalEvaluationCase(
            query="test",
            relevant_memory_ids=frozenset({1}),
            top_k=0,
        )