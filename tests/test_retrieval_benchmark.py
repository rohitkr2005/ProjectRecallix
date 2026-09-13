from dataclasses import dataclass

import pytest

from app.retrieval.retrieval_benchmark import RetrievalBenchmark
from app.retrieval.retrieval_benchmark_runner import RetrievalBenchmarkRunner


@dataclass
class FakeMemory:
    id: int


def make_result(memory_id: int) -> dict:
    return {
        "memory": FakeMemory(id=memory_id),
        "score": 0.9,
        "semantic_score": 0.9,
    }


def perfect_search_factory(benchmark):
    memory_map = benchmark.get_memory_map()

    def search(query, top_k=5):
        case = next(case for case in benchmark.get_cases() if case.query == query)
        relevant = [memory_map[memory_id] for memory_id in sorted(case.relevant_memory_ids)]
        return [
            make_result(memory.id)
            for memory in relevant[:top_k]
        ]

    return search


def perfect_intent(query, benchmark):
    case = next(case for case in benchmark.get_cases() if case.query == query)
    return {"intent": case.expected_intent}


def test_benchmark_loads_expected_dataset():
    benchmark = RetrievalBenchmark()

    stats = benchmark.validate()

    assert stats["memory_count"] == 18
    assert stats["case_count"] == 16
    assert stats["intents"] == [
        "EDUCATION",
        "GOAL",
        "LOCATION",
        "PREFERENCE",
        "PROJECT",
        "SKILL",
    ]


def test_benchmark_memory_ids_are_unique():
    benchmark = RetrievalBenchmark()

    ids = [memory.id for memory in benchmark.memories]

    assert len(ids) == len(set(ids))


def test_benchmark_cases_have_known_relevant_memories():
    benchmark = RetrievalBenchmark()
    memory_ids = set(benchmark.get_memory_map())

    for case in benchmark.get_cases():
        assert case.query.strip()
        assert case.expected_intent in {
            "PROJECT",
            "EDUCATION",
            "SKILL",
            "GOAL",
            "LOCATION",
            "PREFERENCE",
        }
        assert case.relevant_memory_ids
        assert case.relevant_memory_ids <= memory_ids


def test_benchmark_case_ids_are_unique():
    benchmark = RetrievalBenchmark()

    case_ids = [case.id for case in benchmark.get_cases()]

    assert len(case_ids) == len(set(case_ids))


def test_benchmark_cases_cover_all_supported_intents():
    benchmark = RetrievalBenchmark()

    intents = {
        case.expected_intent
        for case in benchmark.get_cases()
    }

    assert intents == {
        "PROJECT",
        "EDUCATION",
        "SKILL",
        "GOAL",
        "LOCATION",
        "PREFERENCE",
    }


def test_benchmark_runner_produces_perfect_synthetic_baseline():
    benchmark = RetrievalBenchmark()
    search = perfect_search_factory(benchmark)

    runner = RetrievalBenchmarkRunner(
        search_fn=search,
        intent_fn=lambda query: perfect_intent(query, benchmark),
        benchmark=benchmark,
    )

    result = runner.run()

    assert result["benchmark"]["case_count"] == 16
    assert result["intent"]["accuracy"] == 1.0
    assert result["retrieval"]["summary"]["precision_at_k"] == 1.0
    assert result["retrieval"]["summary"]["recall_at_k"] == 1.0
    assert result["retrieval"]["summary"]["hit_rate_at_k"] == 1.0
    assert result["retrieval"]["summary"]["mrr"] == 1.0
    assert result["retrieval"]["summary"]["ndcg_at_k"] == 1.0


def test_benchmark_runner_detects_intent_regression():
    benchmark = RetrievalBenchmark()
    search = perfect_search_factory(benchmark)

    runner = RetrievalBenchmarkRunner(
        search_fn=search,
        intent_fn=lambda query: {"intent": "PROJECT"},
        benchmark=benchmark,
    )

    result = runner.run()

    assert result["intent"]["accuracy"] < 1.0
    assert result["intent"]["correct"] < result["intent"]["total"]


def test_benchmark_runner_validates_callable_arguments():
    with pytest.raises(TypeError, match="search_fn must be callable"):
        RetrievalBenchmarkRunner(
            search_fn=None,
            intent_fn=lambda query: {"intent": "PROJECT"},
        )

    with pytest.raises(TypeError, match="intent_fn must be callable"):
        RetrievalBenchmarkRunner(
            search_fn=lambda query, top_k=5: [],
            intent_fn=None,
        )


def test_benchmark_runner_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        RetrievalBenchmarkRunner(
            search_fn=lambda query, top_k=5: [],
            intent_fn=lambda query: {"intent": "PROJECT"},
            top_k=0,
        )
