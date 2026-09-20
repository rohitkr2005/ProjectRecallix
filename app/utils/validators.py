import re
from typing import Optional

import numpy as np

from app.config import settings
from app.utils.exceptions import EmbeddingError, InvalidQueryError, MemoryValidationError


CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_text(text: str) -> str:
    """Remove control characters and normalize whitespace."""
    if not text:
        return ""
    cleaned = CONTROL_CHARS_PATTERN.sub("", str(text))
    return cleaned.strip()


def validate_memory_content(
    subject: str,
    relation: str,
    value: str,
    category: Optional[str] = None,
    max_length: Optional[int] = None,
) -> dict:
    """
    11.5 Memory Safety: Validate memory fields and ensure no injection / malformed data.
    Raises MemoryValidationError if invalid.
    """
    limit = max_length or settings.max_memory_length

    clean_sub = sanitize_text(subject)
    clean_rel = sanitize_text(relation)
    clean_val = sanitize_text(value)
    clean_cat = sanitize_text(category or "GENERAL")

    if not clean_sub:
        raise MemoryValidationError("Memory subject cannot be empty.")
    if not clean_rel:
        raise MemoryValidationError("Memory relation cannot be empty.")
    if not clean_val:
        raise MemoryValidationError("Memory value cannot be empty.")

    if len(clean_val) > limit:
        raise MemoryValidationError(
            f"Memory value exceeds maximum allowed length of {limit} characters (got {len(clean_val)})."
        )

    return {
        "subject": clean_sub,
        "relation": clean_rel,
        "value": clean_val,
        "category": clean_cat,
    }


def validate_embedding_vector(
    vector_bytes: Optional[bytes],
    expected_dim: int = 384,
) -> np.ndarray:
    """
    11.5 Memory Safety: Verify embedding vector integrity.
    Prevents NaN, Inf, zero-vectors, and dimension mismatches.
    Raises EmbeddingError if corrupted.
    """
    if vector_bytes is None or len(vector_bytes) == 0:
        raise EmbeddingError("Embedding vector bytes cannot be empty.")

    expected_bytes = expected_dim * 4  # float32 = 4 bytes
    if len(vector_bytes) != expected_bytes:
        raise EmbeddingError(
            f"Corrupted embedding size: expected {expected_bytes} bytes ({expected_dim} dims), got {len(vector_bytes)} bytes."
        )

    try:
        vec = np.frombuffer(vector_bytes, dtype=np.float32)
    except Exception as e:
        raise EmbeddingError(f"Failed to parse embedding buffer: {str(e)}")

    if np.isnan(vec).any():
        raise EmbeddingError("Corrupted embedding: vector contains NaN values.")
    if np.isinf(vec).any():
        raise EmbeddingError("Corrupted embedding: vector contains infinite values.")

    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise EmbeddingError("Corrupted embedding: vector has zero magnitude.")

    return vec


def validate_user_query(
    query: str,
    max_length: Optional[int] = None,
) -> str:
    """
    Validate user query string.
    Raises InvalidQueryError if empty or oversized.
    """
    limit = max_length or settings.max_query_length

    if not query or not str(query).strip():
        raise InvalidQueryError("User message cannot be empty.")

    clean = sanitize_text(query)
    if not clean:
        raise InvalidQueryError("User message contains only invalid or control characters.")

    if len(clean) > limit:
        raise InvalidQueryError(
            f"User query exceeds maximum allowed length of {limit} characters (got {len(clean)})."
        )

    return clean
