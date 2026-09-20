import json
import logging
import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app.config import Settings, settings
from app.utils.exceptions import (
    DatabaseError,
    EmbeddingError,
    InvalidQueryError,
    LLMError,
    MemoryValidationError,
    RecallixError,
)
from app.utils.logging import (
    JSONFormatter,
    get_logger,
    log_error_event,
    log_llm_event,
    log_memory_event,
    log_retrieval_event,
)
from app.utils.metrics import LatencyTracker
from app.utils.validators import (
    sanitize_text,
    validate_embedding_vector,
    validate_memory_content,
    validate_user_query,
)
from app.assistant.assistant_engine import AssistantEngine, InputType


# ---------------------------------------------------------------------------
# 11.1 Error Handling & Reliability Tests
# ---------------------------------------------------------------------------
class TestExceptionHierarchy:
    """Validate the custom exception hierarchy in app/utils/exceptions.py."""

    def test_base_recallix_error(self):
        err = RecallixError("Base failure")
        assert isinstance(err, Exception)
        assert str(err) == "Base failure"

    def test_specialized_exceptions_inherit_from_recallix_error(self):
        assert issubclass(DatabaseError, RecallixError)
        assert issubclass(EmbeddingError, RecallixError)
        assert issubclass(LLMError, RecallixError)
        assert issubclass(MemoryValidationError, RecallixError)
        assert issubclass(InvalidQueryError, RecallixError)

    def test_validation_errors_backward_compatibility_with_value_error(self):
        """Ensure MemoryValidationError & InvalidQueryError inherit from ValueError."""
        assert issubclass(MemoryValidationError, ValueError)
        assert issubclass(InvalidQueryError, ValueError)

        with pytest.raises(ValueError):
            raise MemoryValidationError("Invalid memory")

        with pytest.raises(ValueError):
            raise InvalidQueryError("Empty query")


# ---------------------------------------------------------------------------
# 11.2 Structured Logging & Observability Tests
# ---------------------------------------------------------------------------
class TestStructuredLogging:
    """Validate structured logging helpers and JSONFormatter in app/utils/logging.py."""

    def test_json_formatter(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="User event occurred",
            args=(),
            exc_info=None,
        )
        record.event_type = "user_action"
        record.custom_attr = "custom_val"

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "User event occurred"
        assert data["event_type"] == "user_action"
        assert data["custom_attr"] == "custom_val"
        assert "timestamp" in data

    def test_get_logger(self):
        logger = get_logger("recallix.test_logger")
        assert logger.name == "recallix.test_logger"
        assert len(logger.handlers) > 0

    def test_log_memory_event(self):
        mock_logger = MagicMock()
        log_memory_event(
            mock_logger,
            action="saved",
            memory_id=42,
            subject="User",
            relation="works_on",
            status="created",
        )
        mock_logger.info.assert_called_once()
        args, kwargs = mock_logger.info.call_args
        assert "Memory event: saved" in args[0]
        extra = kwargs.get("extra", {})
        assert extra["event_type"] == "memory_event"
        assert extra["action"] == "saved"
        assert extra["memory_id"] == 42
        assert extra["subject"] == "User"

    def test_log_retrieval_event(self):
        mock_logger = MagicMock()
        log_retrieval_event(mock_logger, query="What is my project?", count=3, top_score=0.92)
        mock_logger.info.assert_called_once()
        args, kwargs = mock_logger.info.call_args
        assert "Retrieval event: 3 memories" in args[0]
        extra = kwargs.get("extra", {})
        assert extra["event_type"] == "retrieval_event"
        assert extra["count"] == 3
        assert extra["top_score"] == 0.92

    def test_log_llm_event(self):
        mock_logger = MagicMock()
        log_llm_event(mock_logger, prompt="Summarize", status="success", tokens=45)
        mock_logger.info.assert_called_once()
        extra = mock_logger.info.call_args[1].get("extra", {})
        assert extra["event_type"] == "llm_event"
        assert extra["status"] == "success"
        assert extra["tokens"] == 45

    def test_log_error_event(self):
        mock_logger = MagicMock()
        err = DatabaseError("Disk full")
        log_error_event(mock_logger, context="db_write", error=err)
        mock_logger.error.assert_called_once()
        extra = mock_logger.error.call_args[1].get("extra", {})
        assert extra["event_type"] == "error_event"
        assert extra["context"] == "db_write"
        assert extra["error_type"] == "DatabaseError"
        assert extra["error_message"] == "Disk full"


# ---------------------------------------------------------------------------
# 11.3 Centralized Configuration & Environment Management Tests
# ---------------------------------------------------------------------------
class TestCentralizedConfiguration:
    """Validate Settings dataclass and environment variable handling in app/config.py."""

    def test_default_settings(self):
        s = Settings()
        assert s.db_path.endswith("recallix.db")
        assert s.embedding_model == "all-MiniLM-L6-v2"
        assert s.ollama_model == "qwen2.5:3b"
        assert s.enable_metrics is True
        assert s.max_query_length == 1000
        assert s.max_memory_length == 500

    def test_settings_from_env_overrides(self):
        env_vars = {
            "RECALLIX_OLLAMA_MODEL": "mistral:latest",
            "RECALLIX_ENABLE_METRICS": "false",
            "RECALLIX_LOG_LEVEL": "DEBUG",
            "RECALLIX_MAX_QUERY_LENGTH": "500",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            s = Settings.from_env()
            assert s.ollama_model == "mistral:latest"
            assert s.enable_metrics is False
            assert s.log_level == "DEBUG"
            assert s.max_query_length == 500


# ---------------------------------------------------------------------------
# 11.4 Performance & Latency Metrics Tests
# ---------------------------------------------------------------------------
class TestLatencyTracker:
    """Validate LatencyTracker context manager and metric reporting in app/utils/metrics.py."""

    def test_latency_tracker_timing(self):
        tracker = LatencyTracker()
        with tracker.timer("stage_a"):
            _ = [i ** 2 for i in range(1000)]
        with tracker.timer("stage_b"):
            _ = [i ** 3 for i in range(1000)]

        metrics = tracker.get_metrics()
        assert "stage_a" in metrics
        assert "stage_b" in metrics
        assert "total_ms" in metrics
        assert abs(metrics["total_ms"] - (metrics["stage_a"] + metrics["stage_b"])) <= 0.05

    def test_formatted_metrics_string(self):
        tracker = LatencyTracker()
        with tracker.timer("retrieval_ms"):
            pass
        formatted = tracker.formatted()
        assert "retrieval_ms=" in formatted
        assert "total_ms=" in formatted


# ---------------------------------------------------------------------------
# 11.5 Memory Safety & Security Tests
# ---------------------------------------------------------------------------
class TestMemorySafetyAndValidators:
    """Validate input sanitization, memory validation, and embedding validation."""

    def test_sanitize_text(self):
        # Strip null bytes and control characters
        dirty = "Hello\x00 World\x1f\x7f!"
        assert sanitize_text(dirty) == "Hello World!"

        assert sanitize_text("   clean text   ") == "clean text"
        assert sanitize_text(None) == ""

    def test_validate_memory_content_valid(self):
        res = validate_memory_content(
            subject="User",
            relation="works_on",
            value="Project Recallix",
            category="PROJECT",
        )
        assert res["subject"] == "User"
        assert res["relation"] == "works_on"
        assert res["value"] == "Project Recallix"
        assert res["category"] == "PROJECT"

    def test_validate_memory_content_empty_fields(self):
        with pytest.raises(MemoryValidationError, match="Memory subject cannot be empty"):
            validate_memory_content(subject="", relation="works_on", value="Project")

        with pytest.raises(MemoryValidationError, match="Memory relation cannot be empty"):
            validate_memory_content(subject="User", relation="   ", value="Project")

        with pytest.raises(MemoryValidationError, match="Memory value cannot be empty"):
            validate_memory_content(subject="User", relation="works_on", value="")

    def test_validate_memory_content_oversized(self):
        with pytest.raises(MemoryValidationError, match="exceeds maximum allowed length"):
            validate_memory_content(
                subject="User",
                relation="works_on",
                value="x" * 2000,
                max_length=500,
            )

    def test_validate_user_query_valid(self):
        assert validate_user_query("  What is my name?  ") == "What is my name?"

    def test_validate_user_query_empty(self):
        with pytest.raises(InvalidQueryError, match="User message cannot be empty"):
            validate_user_query("")

        with pytest.raises(InvalidQueryError, match="User message cannot be empty"):
            validate_user_query("    \t\n  ")

    def test_validate_user_query_oversized(self):
        with pytest.raises(InvalidQueryError, match="exceeds maximum allowed length"):
            validate_user_query("a" * 3000, max_length=1000)

    def test_validate_embedding_vector_valid(self):
        arr = np.random.randn(384).astype(np.float32)
        valid_bytes = arr.tobytes()
        parsed = validate_embedding_vector(valid_bytes, expected_dim=384)
        assert parsed.shape == (384,)
        np.testing.assert_allclose(parsed, arr)

    def test_validate_embedding_vector_corrupted_size(self):
        short_bytes = b"\x00" * 100
        with pytest.raises(EmbeddingError, match="Corrupted embedding size"):
            validate_embedding_vector(short_bytes, expected_dim=384)

    def test_validate_embedding_vector_nan(self):
        arr = np.ones(384, dtype=np.float32)
        arr[10] = np.nan
        with pytest.raises(EmbeddingError, match="vector contains NaN values"):
            validate_embedding_vector(arr.tobytes(), expected_dim=384)

    def test_validate_embedding_vector_inf(self):
        arr = np.ones(384, dtype=np.float32)
        arr[5] = np.inf
        with pytest.raises(EmbeddingError, match="vector contains infinite values"):
            validate_embedding_vector(arr.tobytes(), expected_dim=384)

    def test_validate_embedding_vector_zero(self):
        arr = np.zeros(384, dtype=np.float32)
        with pytest.raises(EmbeddingError, match="vector has zero magnitude"):
            validate_embedding_vector(arr.tobytes(), expected_dim=384)


# ---------------------------------------------------------------------------
# 11.6 AssistantEngine Integration Tests
# ---------------------------------------------------------------------------
class TestAssistantProductionHardening:
    """Verify production hardening integration in AssistantEngine."""

    @pytest.fixture
    def mock_components(self):
        mock_retrieval = MagicMock()
        mock_retrieval.search.return_value = [
            {
                "memory": MagicMock(
                    id=1,
                    subject="User",
                    relation="works_on",
                    value="Recallix",
                    category="PROJECT",
                    active=True,
                ),
                "score": 0.95,
                "explanation": {"reasons": ["Semantic match"]},
            }
        ]
        mock_retrieval.analyze_query_intent.return_value = {
            "intent": "PROJECT",
            "confidence": 0.98,
            "strict": True,
            "normalized_query": "what project am i working on",
            "scores": {"PROJECT": 0.98},
        }

        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.generate_with_memories.return_value = "You are working on Project Recallix."
        mock_llm.verify_answer_grounding.return_value = {
            "grounded": True,
            "supported_values": ["Recallix"],
            "grounding_score": 1.0,
            "details": "Verified",
        }

        mock_store = MagicMock()
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = []

        return {
            "retrieval_engine": mock_retrieval,
            "llm_engine": mock_llm,
            "memory_store": mock_store,
            "memory_extractor": mock_extractor,
        }

    def test_respond_returns_performance_metrics_when_enabled(self, mock_components):
        assistant = AssistantEngine(**mock_components)
        result = assistant.respond("What project am I working on?")

        assert "performance_metrics" in result
        metrics = result["performance_metrics"]
        assert "retrieval_ms" in metrics
        assert "llm_ms" in metrics
        assert "total_ms" in metrics
        assert metrics["total_ms"] >= 0.0

    def test_respond_statement_returns_performance_metrics(self, mock_components):
        assistant = AssistantEngine(**mock_components)
        mock_mem = MagicMock(subject="User", relation="lives_in", value="Berlin", category="LOCATION", importance=5)
        mock_components["memory_extractor"].extract.return_value = [mock_mem]
        mock_components["memory_store"].save_memory_with_semantics.return_value = (
            mock_mem,
            "created",
            {},
        )

        result = assistant.respond("I live in Berlin.")
        assert result["input_type"] == InputType.STATEMENT
        assert "performance_metrics" in result
        assert "memory_ms" in result["performance_metrics"]

    def test_respond_empty_query_raises_invalid_query_error(self, mock_components):
        assistant = AssistantEngine(**mock_components)
        with pytest.raises(InvalidQueryError):
            assistant.respond("")

        with pytest.raises(ValueError):
            assistant.respond("   \t\n  ")
