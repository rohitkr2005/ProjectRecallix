"""
Utilities for loading and validating the curated Recallix retrieval benchmark.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkMemory:
    id: int
    subject: str
    relation: str
    value: str
    category: str
    importance: int


@dataclass(frozen=True)
class RetrievalBenchmarkCase:
    id: str
    query: str
    expected_intent: str
    relevant_memory_ids: frozenset[int]


class RetrievalBenchmark:
    """Load deterministic benchmark memories and labeled query cases."""

    def __init__(self, path=None):
        self.path = Path(path) if path else (
            Path(__file__).resolve().parents[2]
            / "data"
            / "retrieval_benchmark.json"
        )
        self.memories, self.cases = self._load()

    def _load(self):
        with self.path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        memories = tuple(
            BenchmarkMemory(**memory)
            for memory in payload["memories"]
        )

        memory_ids = {memory.id for memory in memories}

        cases = []
        seen_case_ids = set()

        for case in payload["cases"]:
            if case["id"] in seen_case_ids:
                raise ValueError(
                    f"Duplicate benchmark case id: {case['id']}"
                )

            seen_case_ids.add(case["id"])
            relevant_ids = frozenset(case["relevant_memory_ids"])

            if not relevant_ids:
                raise ValueError(
                    f"Benchmark case has no relevant memories: {case['id']}"
                )

            unknown_ids = relevant_ids - memory_ids
            if unknown_ids:
                raise ValueError(
                    f"Benchmark case {case['id']} references unknown "
                    f"memory IDs: {sorted(unknown_ids)}"
                )

            cases.append(
                RetrievalBenchmarkCase(
                    id=case["id"],
                    query=case["query"],
                    expected_intent=case["expected_intent"],
                    relevant_memory_ids=relevant_ids,
                )
            )

        return memories, tuple(cases)

    def get_cases(self):
        return self.cases

    def get_memory_map(self):
        return {memory.id: memory for memory in self.memories}

    def validate(self):
        """Validate the loaded benchmark and return basic dataset statistics."""
        intents = sorted({case.expected_intent for case in self.cases})
        return {
            "memory_count": len(self.memories),
            "case_count": len(self.cases),
            "intents": intents,
        }
