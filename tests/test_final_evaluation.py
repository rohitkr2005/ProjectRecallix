import datetime
import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app.assistant.assistant_engine import AssistantEngine, InputType
from app.config import settings
from app.database.models import Memory
from app.memory.forgetting import MemoryForgetter
from app.memory.importance_learning import ImportanceLearner
from app.memory.lifecycle import evaluate_update_semantics
from app.memory.temporal import TemporalState
from app.utils.exceptions import (
    DatabaseError,
    EmbeddingError,
    InvalidQueryError,
    LLMError,
    MemoryValidationError,
    RecallixError,
)
from app.utils.metrics import LatencyTracker
from app.utils.validators import (
    sanitize_text,
    validate_embedding_vector,
    validate_memory_content,
    validate_user_query,
)


# ---------------------------------------------------------------------------
# 13.1 End-to-End Multi-Turn Conversational Tests
# ---------------------------------------------------------------------------
class TestEndToEndConversationalWorkflows:
    """Validate full end-to-end multi-turn conversation and reasoning flows."""

    @pytest.fixture
    def assistant_pipeline(self):
        """Build an integrated AssistantEngine with mock persistence and search."""
        mock_retrieval = MagicMock()
        mock_llm = MagicMock()
        mock_store = MagicMock()
        mock_extractor = MagicMock()
        mock_learner = MagicMock()
        mock_graph = MagicMock()
        mock_forgetter = MagicMock()

        # In-memory memory bank for realistic multi-turn simulation
        stored_memories = []

        def mock_extract(text):
            text_lower = text.lower()
            if "live in mumbai" in text_lower or "lives in mumbai" in text_lower or "moved to mumbai" in text_lower:
                return [MagicMock(subject="User", relation="lives_in", value="Mumbai", category="LOCATION", importance=8, temporal_state="PRESENT")]
            elif "lived in delhi" in text_lower or "previously lived in delhi" in text_lower or "lives in delhi" in text_lower:
                return [MagicMock(subject="User", relation="lives_in", value="Delhi", category="LOCATION", importance=7, temporal_state="PAST")]
            elif "building project recallix" in text_lower or "work on recallix" in text_lower:
                return [MagicMock(subject="User", relation="works_on", value="Project Recallix", category="PROJECT", importance=9, temporal_state="PRESENT")]
            return []

        def mock_save(subject, relation, value, category="GENERAL", importance=5, embedding=None, temporal_state="PRESENT"):
            mem = MagicMock(
                id=len(stored_memories) + 1,
                subject=subject,
                relation=relation,
                value=value,
                category=category,
                importance=importance,
                active=True,
                temporal_state=temporal_state,
                created_at=datetime.datetime.utcnow(),
                updated_at=datetime.datetime.utcnow(),
            )
            stored_memories.append(mem)
            return mem, "created", {}

        def mock_search(query, top_k=5, min_score=0.0, semantic_threshold=0.0):
            query_lower = query.lower()
            results = []
            for m in stored_memories:
                if not m.active:
                    continue
                score = 0.5
                if "where" in query_lower and m.category == "LOCATION":
                    if "before" in query_lower or "previously" in query_lower:
                        if m.temporal_state == "PAST":
                            score = 0.95
                    else:
                        if m.temporal_state == "PRESENT":
                            score = 0.95
                elif "project" in query_lower and m.category == "PROJECT":
                    score = 0.95

                if score >= min_score:
                    results.append({
                        "memory": m,
                        "score": score,
                        "explanation": {"reasons": ["Semantic & temporal match"]},
                    })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        def mock_generate(user_message, memories, temperature=0.2, max_tokens=500):
            if not memories:
                return AssistantEngine.UNSUPPORTED_RESPONSE
            val = memories[0]["memory"].value
            if "live" in user_message.lower():
                return f"You live in {val}."
            elif "project" in user_message.lower():
                return f"You are working on {val}."
            return f"Based on your memory: {val}."

        def mock_grounding(response, memories):
            if not memories or AssistantEngine.UNSUPPORTED_RESPONSE in response:
                return {"grounded": True, "supported_values": [], "grounding_score": 1.0, "details": "Declined"}
            sup = [m["memory"].value for m in memories if m["memory"].value in response]
            return {"grounded": bool(sup), "supported_values": sup, "grounding_score": 1.0 if sup else 0.0, "details": "Verified"}

        mock_extractor.extract.side_effect = mock_extract
        mock_store.save_memory_with_semantics.side_effect = mock_save
        mock_store.get_all_memories.return_value = stored_memories
        mock_retrieval.search.side_effect = mock_search
        mock_retrieval.analyze_query_intent.return_value = {"intent": None, "confidence": 0.5, "strict": False, "normalized_query": ""}
        mock_llm.generate_with_memories.side_effect = mock_generate
        mock_llm.verify_answer_grounding.side_effect = mock_grounding
        mock_llm.is_available.return_value = True

        engine = AssistantEngine(
            retrieval_engine=mock_retrieval,
            llm_engine=mock_llm,
            memory_store=mock_store,
            memory_extractor=mock_extractor,
            importance_learner=mock_learner,
            memory_graph=mock_graph,
            memory_forgetter=mock_forgetter,
        )
        return engine, stored_memories

    def test_multi_turn_learning_and_grounded_recall(self, assistant_pipeline):
        assistant, memory_bank = assistant_pipeline

        # Turn 1: Learn a new project
        turn1 = assistant.respond("I am building Project Recallix.")
        assert turn1["input_type"] == InputType.STATEMENT
        assert len(turn1["extracted_memories"]) == 1
        assert turn1["extracted_memories"][0]["value"] == "Project Recallix"
        assert "performance_metrics" in turn1

        # Turn 2: Query the project
        turn2 = assistant.respond("What project am I working on?")
        assert turn2["input_type"] == InputType.QUESTION
        assert turn2["supported"] is True
        assert "Project Recallix" in turn2["response"]
        assert turn2["grounded"] is True

    def test_multi_turn_temporal_location_evolution(self, assistant_pipeline):
        assistant, memory_bank = assistant_pipeline

        # Turn 1: Past location
        assistant.respond("Previously lived in Delhi.")
        # Turn 2: Current location
        assistant.respond("Currently lives in Mumbai.")

        # Both facts should coexist with different temporal states
        assert len(memory_bank) == 2
        past_mem = [m for m in memory_bank if m.temporal_state == "PAST"][0]
        present_mem = [m for m in memory_bank if m.temporal_state == "PRESENT"][0]
        assert past_mem.value == "Delhi"
        assert present_mem.value == "Mumbai"

        # Turn 3: Query current location
        q_now = assistant.respond("Where do I currently live?")
        assert "Mumbai" in q_now["response"]

        # Turn 4: Query past location
        q_before = assistant.respond("Where did I live previously before?")
        assert "Delhi" in q_before["response"]

    def test_unsupported_question_zero_hallucination(self, assistant_pipeline):
        assistant, _ = assistant_pipeline

        # Ask about a fact not in memory
        result = assistant.respond("What is the name of my pet dog?")
        assert result["supported"] is False
        assert result["response"] == AssistantEngine.UNSUPPORTED_RESPONSE
        assert result["grounding_details"]["grounded"] is True
        assert result["llm_status"] == "skipped_unsupported"


# ---------------------------------------------------------------------------
# 13.2 Failure & Resilience Injection Tests
# ---------------------------------------------------------------------------
class TestFailureResilienceAndSafety:
    """Validate system robustness against unexpected runtime failures and malformed inputs."""

    def test_llm_runtime_failure_triggers_extractive_fallback(self):
        mock_retrieval = MagicMock()
        mock_store = MagicMock()
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = []

        # Return a memory
        mem = MagicMock(subject="User", relation="knows", value="Python", category="SKILL", active=True)
        mock_retrieval.search.return_value = [{"memory": mem, "score": 0.9, "explanation": {}}]
        mock_retrieval.analyze_query_intent.return_value = {"intent": "SKILL", "confidence": 0.9, "strict": True}

        mock_llm = MagicMock()
        mock_llm.generate_with_memories.side_effect = Exception("Ollama service unreachable on port 11434")
        mock_llm.build_memory_context.return_value = "- User knows Python (SKILL)"

        assistant = AssistantEngine(
            retrieval_engine=mock_retrieval,
            llm_engine=mock_llm,
            memory_store=mock_store,
            memory_extractor=mock_extractor,
        )

        # Calling respond with fallback enabled should NOT raise an exception
        result = assistant.respond("What skills do I know?", fallback_to_extractive=True)
        assert result["supported"] is True
        assert "unable to reach the local LLM runtime" in result["response"]
        assert "User knows Python" in result["response"]
        assert "fallback: Ollama service unreachable" in result["llm_status"]

    def test_llm_runtime_failure_without_fallback_raises_exception(self):
        mock_retrieval = MagicMock()
        mem = MagicMock(subject="User", relation="knows", value="Python", category="SKILL", active=True)
        mock_retrieval.search.return_value = [{"memory": mem, "score": 0.9}]
        mock_retrieval.analyze_query_intent.return_value = {"intent": "SKILL", "confidence": 0.9, "strict": True}

        mock_llm = MagicMock()
        mock_llm.generate_with_memories.side_effect = LLMError("Timeout waiting for LLM")

        assistant = AssistantEngine(
            retrieval_engine=mock_retrieval,
            llm_engine=mock_llm,
            memory_extractor=MagicMock(extract=MagicMock(return_value=[])),
        )

        with pytest.raises(LLMError, match="Timeout waiting for LLM"):
            assistant.respond("What skills do I know?", fallback_to_extractive=False)

    def test_corrupted_vector_validation(self):
        # Empty vector
        with pytest.raises(EmbeddingError, match="cannot be empty"):
            validate_embedding_vector(b"")

        # Wrong dimension (100 bytes instead of 1536)
        with pytest.raises(EmbeddingError, match="Corrupted embedding size"):
            validate_embedding_vector(b"\x00" * 100, expected_dim=384)

        # NaN vector
        nan_vec = np.ones(384, dtype=np.float32)
        nan_vec[42] = np.nan
        with pytest.raises(EmbeddingError, match="vector contains NaN values"):
            validate_embedding_vector(nan_vec.tobytes(), expected_dim=384)

        # Inf vector
        inf_vec = np.ones(384, dtype=np.float32)
        inf_vec[12] = np.inf
        with pytest.raises(EmbeddingError, match="vector contains infinite values"):
            validate_embedding_vector(inf_vec.tobytes(), expected_dim=384)

        # Zero magnitude vector
        zero_vec = np.zeros(384, dtype=np.float32)
        with pytest.raises(EmbeddingError, match="vector has zero magnitude"):
            validate_embedding_vector(zero_vec.tobytes(), expected_dim=384)

    def test_adversarial_and_malformed_input_sanitization(self):
        # SQL injection strings should be sanitized without crashing
        sql_input = "'; DROP TABLE memories; --"
        sanitized = sanitize_text(sql_input)
        assert sanitized == "'; DROP TABLE memories; --"

        # Control characters stripped
        dirty = "User\x00\x08 knows \x1fPython\x7f"
        assert sanitize_text(dirty) == "User knows Python"

        # Memory validation rejects empty fields
        with pytest.raises(MemoryValidationError):
            validate_memory_content(subject="", relation="knows", value="Python")

        with pytest.raises(MemoryValidationError):
            validate_memory_content(subject="User", relation="", value="Python")

        with pytest.raises(MemoryValidationError):
            validate_memory_content(subject="User", relation="knows", value="")

        # Memory validation rejects oversized fields
        with pytest.raises(MemoryValidationError, match="exceeds maximum allowed length"):
            validate_memory_content(subject="User", relation="knows", value="x" * 1000, max_length=200)

    def test_empty_and_whitespace_query_rejection(self):
        with pytest.raises(InvalidQueryError):
            validate_user_query("")

        with pytest.raises(InvalidQueryError):
            validate_user_query("   \t\n  ")

        with pytest.raises(InvalidQueryError, match="exceeds maximum allowed length"):
            validate_user_query("a" * 2000, max_length=500)


# ---------------------------------------------------------------------------
# 13.3 Memory Protection & Controlled Forgetting Under Stress
# ---------------------------------------------------------------------------
class TestMemoryForgettingAndProtection:
    """Validate memory forgetting policy, dormancy evaluation, and protection shields."""

    def test_protected_memory_is_not_forgotten(self):
        learner = ImportanceLearner()
        forgetter = MemoryForgetter(importance_learner=learner)

        # Case 1: High reinforcement (helpful_count >= 3)
        protected_mem_1 = MagicMock(
            id=1,
            subject="User",
            relation="works_on",
            value="Recallix",
            importance=5,
            helpful_count=4,
            access_count=5,
            reinforcement_score=2.0,
            active=True,
            last_accessed_at=datetime.datetime.utcnow() - datetime.timedelta(days=100),
        )
        should_forget, reason, details = forgetter.evaluate_forgetting(protected_mem_1)
        assert should_forget is False
        assert "protected" in reason.lower()

        # Case 2: High base/effective importance (effective_importance >= 6.0)
        protected_mem_2 = MagicMock(
            id=2,
            subject="User",
            relation="lives_in",
            value="Mumbai",
            importance=9,
            helpful_count=0,
            access_count=1,
            reinforcement_score=0.0,
            active=True,
            last_accessed_at=datetime.datetime.utcnow() - datetime.timedelta(days=100),
        )
        should_forget, reason, details = forgetter.evaluate_forgetting(protected_mem_2)
        assert should_forget is False
        assert "protected" in reason.lower()

    def test_dormant_unhelpful_memory_is_forgotten(self):
        learner = ImportanceLearner()
        forgetter = MemoryForgetter(importance_learner=learner)

        dormant_mem = MagicMock(
            id=3,
            subject="User",
            relation="bought",
            value="Notebook",
            importance=1,
            helpful_count=0,
            access_count=1,
            reinforcement_score=0.0,
            active=True,
            last_accessed_at=datetime.datetime.utcnow() - datetime.timedelta(days=45),
        )
        should_forget, reason, details = forgetter.evaluate_forgetting(dormant_mem, decay_days=30, min_importance=2)
        assert should_forget is True
        assert "dormant" in reason.lower()


# ---------------------------------------------------------------------------
# 13.4 Performance & Latency Profile Tests
# ---------------------------------------------------------------------------
class TestPerformanceAndLatencyProfiling:
    """Validate LatencyTracker performance profiling in AssistantEngine."""

    def test_latency_tracker_all_stages_and_formatting(self):
        tracker = LatencyTracker()

        with tracker.timer("memory_ms"):
            _ = sum(i for i in range(1000))

        with tracker.timer("retrieval_ms"):
            _ = sum(i for i in range(2000))

        with tracker.timer("llm_ms"):
            _ = sum(i for i in range(3000))

        metrics = tracker.get_metrics()
        assert "memory_ms" in metrics
        assert "retrieval_ms" in metrics
        assert "llm_ms" in metrics
        assert "total_ms" in metrics

        for val in metrics.values():
            assert isinstance(val, float)
            assert val >= 0.0

        formatted = tracker.formatted()
        assert "memory_ms=" in formatted
        assert "retrieval_ms=" in formatted
        assert "llm_ms=" in formatted
        assert "total_ms=" in formatted

    def test_assistant_attaches_performance_metrics(self):
        mock_retrieval = MagicMock()
        mock_retrieval.search.return_value = []
        mock_retrieval.analyze_query_intent.return_value = {"intent": None, "confidence": 0.0, "strict": False}

        assistant = AssistantEngine(
            retrieval_engine=mock_retrieval,
            llm_engine=MagicMock(is_available=MagicMock(return_value=True)),
            memory_extractor=MagicMock(extract=MagicMock(return_value=[])),
        )

        res = assistant.respond("Hello!")
        assert "performance_metrics" in res
        assert "total_ms" in res["performance_metrics"]
