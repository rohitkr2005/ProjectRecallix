"""Run the curated retrieval benchmark against Recallix retrieval code."""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.database import Base
from app.database.models import Memory
from app.embeddings.embedding_engine import EmbeddingEngine
from app.memory.memory_store import MemoryStore
from app.retrieval.retrieval_benchmark import RetrievalBenchmark
from app.retrieval.retrieval_engine import RetrievalEngine
from app.retrieval.retrieval_evaluator import (
    RetrievalEvaluationCase,
    RetrievalEvaluator,
)


DEFAULT_REGRESSION_THRESHOLDS = {
    "intent_accuracy": 0.90,
    "recall_at_k": 0.90,
    "hit_rate_at_k": 0.95,
    "mrr": 0.90,
    "ndcg_at_k": 0.90,
}


class BenchmarkMemoryStore(MemoryStore):
    """MemoryStore connected to an isolated benchmark database."""

    def __init__(self, session_factory):
        self.session = session_factory()


class RetrievalBenchmarkRunner:
    """Connect the labeled benchmark dataset to Recallix retrieval code."""

    def __init__(
        self,
        search_fn,
        intent_fn,
        benchmark=None,
        top_k=5,
    ):
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
    def from_retrieval_engine(
        cls,
        engine,
        benchmark=None,
        top_k=5,
    ):
        """Create a runner that evaluates the real RetrievalEngine.

        Results are mapped to stable benchmark IDs using relation + value,
        so database primary keys do not become part of the benchmark.
        """
        benchmark = benchmark or RetrievalBenchmark()

        memory_map = {
            (
                memory.relation,
                memory.value.lower().strip(),
            ): memory.id
            for memory in benchmark.memories
        }

        def search_fn(query, top_k=5):
            results = engine.search(
                query,
                top_k=top_k,
            )

            mapped = []

            for item in results:
                memory = (
                    item.get("memory")
                    if isinstance(item, dict)
                    else None
                )

                if memory is None:
                    continue

                key = (
                    memory.relation,
                    str(memory.value).lower().strip(),
                )

                benchmark_id = memory_map.get(key)

                if benchmark_id is None:
                    continue

                mapped_item = dict(item)

                mapped_item["memory"] = SimpleNamespace(
                    id=benchmark_id
                )

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
        """Run retrieval and intent evaluation."""
        cases = self.benchmark.get_cases()

        retrieval_cases = [
            RetrievalEvaluationCase(
                query=case.query,
                relevant_memory_ids=case.relevant_memory_ids,
                top_k=self.top_k,
            )
            for case in cases
        ]

        retrieval_result = self.evaluator.evaluate(
            retrieval_cases
        )

        intent_results = []
        correct_intents = 0

        for case in cases:
            analysis = self.intent_fn(case.query)

            predicted_intent = (
                analysis.get("intent")
                if isinstance(analysis, dict)
                else analysis
            )

            correct = (
                predicted_intent == case.expected_intent
            )

            correct_intents += int(correct)

            intent_results.append(
                {
                    "case_id": case.id,
                    "query": case.query,
                    "expected_intent": case.expected_intent,
                    "predicted_intent": predicted_intent,
                    "correct": correct,
                }
            )

        intent_accuracy = (
            correct_intents / len(cases)
            if cases
            else 0.0
        )

        benchmark_cases = [
            {
                "case_id": case.id,
                "query": case.query,
                "expected_intent": case.expected_intent,
                "relevant_memory_ids": sorted(
                    case.relevant_memory_ids
                ),
            }
            for case in cases
        ]

        return {
            "benchmark": self.benchmark.validate(),
            "benchmark_cases": benchmark_cases,
            "retrieval": retrieval_result,
            "intent": {
                "accuracy": intent_accuracy,
                "correct": correct_intents,
                "total": len(cases),
                "cases": intent_results,
            },
        }

    @staticmethod
    def assert_regression_thresholds(
        result,
        thresholds=None,
    ):
        """Fail when benchmark quality drops below protected thresholds."""

        thresholds = (
            thresholds
            if thresholds is not None
            else DEFAULT_REGRESSION_THRESHOLDS
        )

        intent_accuracy = result["intent"]["accuracy"]

        retrieval_summary = result["retrieval"]["summary"]

        actual_metrics = {
            "intent_accuracy": intent_accuracy,
            "recall_at_k": retrieval_summary["recall_at_k"],
            "hit_rate_at_k": retrieval_summary["hit_rate_at_k"],
            "mrr": retrieval_summary["mrr"],
            "ndcg_at_k": retrieval_summary["ndcg_at_k"],
        }

        failures = []

        for metric, minimum in thresholds.items():
            actual = actual_metrics.get(metric)

            if actual is None:
                raise KeyError(
                    f"Unknown benchmark regression metric: {metric}"
                )

            if actual < minimum:
                failures.append(
                    f"{metric}={actual:.4f} "
                    f"< required {minimum:.4f}"
                )

        if failures:
            raise AssertionError(
                "Benchmark regression detected: "
                + "; ".join(failures)
            )


def _create_benchmark_database():
    """Create an isolated temporary SQLite database."""
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".db",
        delete=False,
    )

    database_path = Path(temp_file.name)

    temp_file.close()

    database_url = f"sqlite:///{database_path}"

    engine = create_engine(
        database_url,
        connect_args={
            "check_same_thread": False
        },
    )

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    Base.metadata.create_all(engine)

    return (
        database_path,
        engine,
        session_factory,
    )


def _populate_benchmark_database(
    session_factory,
    benchmark,
    embedding_engine,
):
    """Populate the isolated database with benchmark memories."""
    session = session_factory()

    try:
        for benchmark_memory in benchmark.memories:
            text = (
                f"The user "
                f"{benchmark_memory.relation.replace('_', ' ')} "
                f"{benchmark_memory.value}."
            )

            embedding = (
                embedding_engine.generate_embedding(text)
            )

            memory = Memory(
                subject=benchmark_memory.subject,
                relation=benchmark_memory.relation,
                value=benchmark_memory.value,
                category=benchmark_memory.category,
                importance=benchmark_memory.importance,
                active=True,
                embedding=embedding.tobytes(),
            )

            session.add(memory)

        session.commit()

    finally:
        session.close()


def _print_metric(label, value):
    """Print a benchmark metric consistently."""
    if isinstance(value, float):
        print(
            f"{label:<18}: {value:.4f}"
        )
    else:
        print(
            f"{label:<18}: {value}"
        )


def _print_benchmark_report(result):
    """Print a human-readable benchmark report."""
    benchmark_info = result["benchmark"]
    retrieval = result["retrieval"]
    intent = result["intent"]

    print()
    print("=" * 64)
    print(
        "        PROJECT RECALLIX RETRIEVAL BENCHMARK"
    )
    print("=" * 64)

    print()
    print("Dataset")
    print("-" * 64)

    _print_metric(
        "Benchmark memories",
        benchmark_info.get(
            "memory_count",
            0,
        ),
    )

    _print_metric(
        "Benchmark queries",
        benchmark_info.get(
            "case_count",
            0,
        ),
    )

    _print_metric(
        "Top-K",
        result.get(
            "top_k",
            5,
        ),
    )

    print()
    print("Intent Understanding")
    print("-" * 64)

    _print_metric(
        "Intent accuracy",
        intent["accuracy"],
    )

    _print_metric(
        "Correct",
        intent["correct"],
    )

    _print_metric(
        "Total",
        intent["total"],
    )

    print()
    print("Retrieval Quality")
    print("-" * 64)

    summary = retrieval.get(
        "summary",
        retrieval,
    )

    metric_names = {
        "precision_at_k": "Precision@K",
        "recall_at_k": "Recall@K",
        "hit_rate_at_k": "Hit Rate@K",
        "mrr": "MRR",
        "ndcg_at_k": "NDCG@K",
    }

    for key, label in metric_names.items():
        if key in summary:
            _print_metric(
                label,
                summary[key],
            )

    print()
    print("Per-Query Results")
    print("-" * 64)

    retrieval_cases = retrieval.get(
        "cases",
        [],
    )

    intent_cases = {
        case["case_id"]: case
        for case in intent["cases"]
    }

    benchmark_cases = result.get(
        "benchmark_cases",
        [],
    )

    for index, retrieval_case in enumerate(
        retrieval_cases
    ):
        if index < len(benchmark_cases):
            benchmark_case = benchmark_cases[index]

            case_id = benchmark_case[
                "case_id"
            ]

            query = benchmark_case[
                "query"
            ]
        else:
            case_id = (
                f"case_{index + 1:02d}"
            )

            query = retrieval_case.get(
                "query",
                "",
            )

        print()
        print(
            f"[{case_id}] {query}"
        )

        intent_case = intent_cases.get(
            case_id
        )

        if intent_case is not None:
            expected_intent = (
                intent_case[
                    "expected_intent"
                ]
            )

            predicted_intent = (
                intent_case[
                    "predicted_intent"
                ]
            )

            correct = intent_case[
                "correct"
            ]

            print(
                f"  Expected intent   : "
                f"{expected_intent}"
            )

            print(
                f"  Predicted intent  : "
                f"{predicted_intent}"
            )

            print(
                f"  Intent correct    : "
                f"{'YES' if correct else 'NO'}"
            )

        if "precision_at_k" in retrieval_case:
            print(
                f"  Precision@K       : "
                f"{retrieval_case['precision_at_k']:.4f}"
            )

        if "recall_at_k" in retrieval_case:
            print(
                f"  Recall@K          : "
                f"{retrieval_case['recall_at_k']:.4f}"
            )

        if "hit_rate_at_k" in retrieval_case:
            print(
                f"  Hit Rate@K        : "
                f"{retrieval_case['hit_rate_at_k']:.4f}"
            )

        if "mrr" in retrieval_case:
            print(
                f"  MRR               : "
                f"{retrieval_case['mrr']:.4f}"
            )

        if "ndcg_at_k" in retrieval_case:
            print(
                f"  NDCG@K            : "
                f"{retrieval_case['ndcg_at_k']:.4f}"
            )

    print()
    print("=" * 64)


def main():
    """Run the benchmark against the real RetrievalEngine."""
    benchmark = RetrievalBenchmark()

    database_path = None
    database_engine = None
    memory_store = None
    embedding_engine = None
    retrieval_engine = None

    top_k = 5

    try:
        print()
        print("Loading Recallix benchmark...")
        print("Creating isolated benchmark database...")

        (
            database_path,
            database_engine,
            session_factory,
        ) = _create_benchmark_database()

        print(
            f"Benchmark database: "
            f"{database_path}"
        )

        print("Loading embedding model...")

        embedding_engine = EmbeddingEngine()

        print(
            "Populating benchmark memories..."
        )

        _populate_benchmark_database(
            session_factory=session_factory,
            benchmark=benchmark,
            embedding_engine=embedding_engine,
        )

        memory_store = BenchmarkMemoryStore(
            session_factory=session_factory
        )

        retrieval_engine = RetrievalEngine(
            memory_store=memory_store,
            embedding_engine=embedding_engine,
        )

        print(
            "Running real RetrievalEngine "
            "benchmark..."
        )

        runner = (
            RetrievalBenchmarkRunner
            .from_retrieval_engine(
                engine=retrieval_engine,
                benchmark=benchmark,
                top_k=top_k,
            )
        )

        result = runner.run()

        result["top_k"] = top_k

        _print_benchmark_report(
            result
        )

        print()
        print("Regression Protection")
        print("-" * 64)

        try:
            runner.assert_regression_thresholds(
                result
            )

            print(
                "Status            : PASS"
            )

            print(
                "All protected benchmark "
                "thresholds are satisfied."
            )

        except AssertionError as exc:
            print(
                "Status            : FAIL"
            )

            print(
                str(exc)
            )

            raise

        print()
        print("=" * 64)
        print(
            "Benchmark execution completed."
        )
        print("=" * 64)
        print()

    finally:
        if retrieval_engine is not None:
            retrieval_engine.close()

        elif memory_store is not None:
            memory_store.close()

        if database_engine is not None:
            database_engine.dispose()

        if database_path is not None:
            try:
                os.remove(
                    database_path
                )
            except OSError:
                pass


if __name__ == "__main__":
    main()