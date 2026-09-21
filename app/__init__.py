"""
Project Recallix — Long-term memory-augmented conversational assistant.
"""

from app.assistant.assistant_engine import AssistantEngine, InputType
from app.config import Settings, get_settings, settings
from app.memory.memory_store import MemoryStore
from app.retrieval.retrieval_engine import RetrievalEngine

__version__ = "1.0.0"
__author__ = "Rohit Kumar"

__all__ = [
    "AssistantEngine",
    "InputType",
    "MemoryStore",
    "RetrievalEngine",
    "Settings",
    "get_settings",
    "settings",
    "__version__",
    "__author__",
]
