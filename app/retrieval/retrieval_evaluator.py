"""
Retrieval evaluation and quality benchmarking utilities.

This module evaluates retrieval results using standard information
retrieval metrics without changing the retrieval/ranking logic itself.
"""

from dataclasses import dataclass
from math import log2
from statistics import mean
from typing import Callable, Iterable


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    """
    Defines one retrieval evaluation case.

    Attributes:
        query: Query to send to the retrieval system.
        relevant_memory_ids: Memory IDs considered correct for the query.
        top_k: Number of results to evaluate.
    """

    query: str
    relevant_memory_ids: frozenset[int]
    top_k: int = 5

    def __post_init__(self):
        if not self.query or not self.query.strip():
            raise ValueError("Evaluation query cannot be empty.")

        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0.")


class RetrievalEvaluator:
    """
    Evaluates a retrieval system using standard ranking metrics.

    The evaluator receives a search function instead of depending directly
    on RetrievalEngine. This keeps evaluation independent, reusable, and
    easy to test.
    """

    METRICS = (
        "precision_at_k",
        "recall_at_k",
        "hit_rate_at_k",
        "mrr",
        "ndcg_at_k",
    )

    def __init__(self, search_fn: Callable):
        if not callable(search_fn):
            raise TypeError("search_fn must be callable.")

        self.search_fn = search_fn

    @staticmethod
    def _ranked_ids(results) -> list[int]:
        """
        Extract memory IDs from retrieval results.

        Duplicate memory IDs are removed while preserving their first
        occurrence in the ranking.
        """

        ranked_ids = []
        seen = set()

        for item in results or []:
            if not isinstance(item, dict):
                continue

            memory = item.get("memory")
            memory_id = getattr(memory, "id", None)

            if memory_id is None:
                continue

            if memory_id in seen:
                continue

            seen.add(memory_id)
            ranked_ids.append(memory_id)

        return ranked_ids

    @staticmethod
    def precision_at_k(
        ranked_ids: list[int],
        relevant_ids: set[int] | frozenset[int],
        k: int,
    ) -> float:
        """
        Precision@K.

        Measures the proportion of the top-K retrieved memories
        that are relevant.
        """

        if k <= 0:
            return 0.0

        top_results = ranked_ids[:k]

        if not top_results:
            return 0.0

        relevant_count = sum(
            memory_id in relevant_ids
            for memory_id in top_results
        )

        return relevant_count / len(top_results)

    @staticmethod
    def recall_at_k(
        ranked_ids: list[int],
        relevant_ids: set[int] | frozenset[int],
        k: int,
    ) -> float:
        """
        Recall@K.

        Measures how many of the known relevant memories were
        successfully retrieved within the top-K results.
        """

        if not relevant_ids or k <= 0:
            return 0.0

        top_results = ranked_ids[:k]

        relevant_count = sum(
            memory_id in relevant_ids
            for memory_id in top_results
        )

        return relevant_count / len(relevant_ids)

    @staticmethod
    def hit_rate_at_k(
        ranked_ids: list[int],
        relevant_ids: set[int] | frozenset[int],
        k: int,
    ) -> float:
        """
        Hit Rate@K.

        Returns 1 if at least one relevant memory appears
        in the top-K results, otherwise 0.
        """

        if k <= 0 or not relevant_ids:
            return 0.0

        top_results = ranked_ids[:k]

        return float(
            any(memory_id in relevant_ids for memory_id in top_results)
        )

    @staticmethod
    def reciprocal_rank(
        ranked_ids: list[int],
        relevant_ids: set[int] | frozenset[int],
    ) -> float:
        """
        Reciprocal Rank.

        Returns 1/rank for the first relevant result.
        Returns 0 if no relevant result is found.
        """

        for rank, memory_id in enumerate(ranked_ids, start=1):
            if memory_id in relevant_ids:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def ndcg_at_k(
        ranked_ids: list[int],
        relevant_ids: set[int] | frozenset[int],
        k: int,
    ) -> float:
        """
        NDCG@K using binary relevance.

        Relevant memories receive relevance 1.
        Non-relevant memories receive relevance 0.
        """

        if k <= 0 or not relevant_ids:
            return 0.0

        top_results = ranked_ids[:k]

        dcg = 0.0

        for rank, memory_id in enumerate(top_results, start=1):
            if memory_id in relevant_ids:
                dcg += 1.0 / log2(rank + 1)

        ideal_hits = min(len(relevant_ids), k)

        if ideal_hits == 0:
            return 0.0

        idcg = sum(
            1.0 / log2(rank + 1)
            for rank in range(1, ideal_hits + 1)
        )

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def evaluate_case(
        self,
        case: RetrievalEvaluationCase,
    ) -> dict:
        """
        Evaluate one benchmark case.
        """

        results = self.search_fn(
            case.query,
            top_k=case.top_k,
        )

        ranked_ids = self._ranked_ids(results)

        relevant_ids = case.relevant_memory_ids

        return {
            "query": case.query,
            "top_k": case.top_k,
            "retrieved_ids": ranked_ids[:case.top_k],
            "relevant_ids": sorted(relevant_ids),
            "precision_at_k": self.precision_at_k(
                ranked_ids,
                relevant_ids,
                case.top_k,
            ),
            "recall_at_k": self.recall_at_k(
                ranked_ids,
                relevant_ids,
                case.top_k,
            ),
            "hit_rate_at_k": self.hit_rate_at_k(
                ranked_ids,
                relevant_ids,
                case.top_k,
            ),
            "mrr": self.reciprocal_rank(
                ranked_ids,
                relevant_ids,
            ),
            "ndcg_at_k": self.ndcg_at_k(
                ranked_ids,
                relevant_ids,
                case.top_k,
            ),
        }

    def evaluate(
        self,
        cases: Iterable[RetrievalEvaluationCase],
    ) -> dict:
        """
        Evaluate an entire benchmark dataset.

        Returns both per-case results and aggregate averages.
        """

        cases = list(cases)

        case_results = [
            self.evaluate_case(case)
            for case in cases
        ]

        if not case_results:
            return {
                "cases": [],
                "summary": {
                    metric: 0.0
                    for metric in self.METRICS
                },
            }

        summary = {
            metric: mean(
                result[metric]
                for result in case_results
            )
            for metric in self.METRICS
        }

        return {
            "cases": case_results,
            "summary": summary,
        }

    @staticmethod
    def assert_thresholds(
        summary: dict,
        thresholds: dict[str, float],
    ) -> None:
        """
        Fail if benchmark metrics fall below configured thresholds.

        Example:

            evaluator.assert_thresholds(
                result["summary"],
                {
                    "hit_rate_at_k": 0.80,
                    "mrr": 0.60,
                },
            )
        """

        failures = []

        for metric, minimum in thresholds.items():
            actual = summary.get(metric)

            if actual is None:
                raise KeyError(
                    f"Unknown evaluation metric: {metric}"
                )

            if actual < minimum:
                failures.append(
                    f"{metric}={actual:.3f} "
                    f"< required {minimum:.3f}"
                )

        if failures:
            raise AssertionError(
                "Retrieval quality regression: "
                + "; ".join(failures)
            )