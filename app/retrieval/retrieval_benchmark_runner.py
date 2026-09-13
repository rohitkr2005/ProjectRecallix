"""Run the curated retrieval benchmark against a retrieval/intent implementation."""

from types import SimpleNamespace

from app.retrieval.retrieval_benchmark import RetrievalBenchmark
from app.retrieval.retrieval_evaluator import (
    RetrievalEvaluationCase,
    RetrievalEvaluator,
)


class RetrievalBenchmarkRunner:
    """Connect the labeled benchmark dataset to Recallix retrieval code."""

    def __init__(self, search_fn, intent_fn, benchmark=None, top_k=5):
        if not callable(search_fn):
            raise TypeError("search_fn must be callable.")
        if not callable(intent_fn):
            raise TypeError("intent_fn must be callable.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        self.search_fn = search_fn
        self.intent_fn = intent_fn
        self.benchmark = benchmark or RetrievalBenchmark()
        self.top_k = top_k
        self.evaluator = RetrievalEvaluator(search_fn)

    @classmethod
    def from_retrieval_engine(cls, engine, benchmark=None, top_k=5):
        """Create a runner that evaluates the real RetrievalEngine.

        Results are mapped to stable benchmark IDs using relation + value,
        so database primary keys do not become part of the benchmark.
        """
        benchmark = benchmark or RetrievalBenchmark()
        memory_map = {
            (memory.relation, memory.value): memory.id
            for memory in benchmark.memories
        }

        def search_fn(query, top_k=5):
            results = engine.search(query, top_k=top_k)
            mapped = []

            for item in results:
                memory = item.get("memory") if isinstance(item, dict) else None
                if memory is None:
                    continue

                benchmark_id = memory_map.get(
                    (memory.relation, memory.value)
                )
                if benchmark_id is None:
                    continue

                mapped_item = dict(item)
                mapped_item["memory"] = SimpleNamespace(id=benchmark_id)
                mapped.append(mapped_item)

            return mapped

        def intent_fn(query):
            return engine.analyze_query_intent(query)

        return cls(
            search_fn=search_fn,
            intent_fn=intent_fn,
            benchmark=benchmark,
            top_k=top_k,
        )

    def run(self):
        cases = self.benchmark.get_cases()

        retrieval_cases = [
            RetrievalEvaluationCase(
                query=case.query,
                relevant_memory_ids=case.relevant_memory_ids,
                top_k=self.top_k,
            )
            for case in cases
        ]

        retrieval_result = self.evaluator.evaluate(retrieval_cases)

        intent_results = []
        correct_intents = 0

        for case in cases:
            analysis = self.intent_fn(case.query)
            predicted_intent = (
                analysis.get("intent")
                if isinstance(analysis, dict)
                else analysis
            )
            correct = predicted_intent == case.expected_intent
            correct_intents += int(correct)
            intent_results.append({
                "case_id": case.id,
                "query": case.query,
                "expected_intent": case.expected_intent,
                "predicted_intent": predicted_intent,
                "correct": correct,
            })

        intent_accuracy = (
            correct_intents / len(cases)
            if cases
            else 0.0
        )

        return {
            "benchmark": self.benchmark.validate(),
            "retrieval": retrieval_result,
            "intent": {
                "accuracy": intent_accuracy,
                "correct": correct_intents,
                "total": len(cases),
                "cases": intent_results,
            },
        }
