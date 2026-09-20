"""
11.1 Error Handling & Reliability

Custom exception hierarchy for Project Recallix.
"""


class RecallixError(Exception):
    """Base exception for all Project Recallix errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DatabaseError(RecallixError):
    """Raised when a database query, transaction, or migration fails."""
    pass


class EmbeddingError(RecallixError):
    """Raised when embedding model loading or vector encoding fails."""
    pass


class LLMError(RecallixError):
    """Raised when LLM service communication, generation, or parsing fails."""
    pass


class MemoryValidationError(ValueError, RecallixError):
    """Raised when memory content or structure is malformed, oversized, or invalid."""
    pass


class InvalidQueryError(ValueError, RecallixError):
    """Raised when a user query is invalid, empty, or exceeds safety thresholds."""
    pass
