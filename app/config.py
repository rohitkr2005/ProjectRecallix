from dataclasses import dataclass
import os
from typing import Optional


@dataclass
class Settings:
    """
    11.3 Configuration & Environment Management

    Centralized configuration management with environment variable overrides.
    """
    # Database
    database_url: str = os.getenv("RECALLIX_DATABASE_URL", "sqlite:///data/recallix.db")
    db_path: str = os.getenv("RECALLIX_DB_PATH", "data/recallix.db")

    # Embeddings
    embedding_model: str = os.getenv("RECALLIX_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # LLM
    llm_model: str = os.getenv("RECALLIX_LLM_MODEL", os.getenv("RECALLIX_OLLAMA_MODEL", "qwen2.5:3b"))
    ollama_model: str = os.getenv("RECALLIX_OLLAMA_MODEL", os.getenv("RECALLIX_LLM_MODEL", "qwen2.5:3b"))
    llm_base_url: str = os.getenv("RECALLIX_LLM_BASE_URL", "http://localhost:11434")
    llm_timeout: int = int(os.getenv("RECALLIX_LLM_TIMEOUT", "120"))
    llm_connect_timeout: int = int(os.getenv("RECALLIX_LLM_CONNECT_TIMEOUT", "3"))

    # Retrieval
    default_top_k: int = int(os.getenv("RECALLIX_TOP_K", "5"))
    min_relevance: float = float(os.getenv("RECALLIX_MIN_RELEVANCE", "0.25"))

    # Observability & Metrics
    log_level: str = os.getenv("RECALLIX_LOG_LEVEL", "INFO")
    enable_metrics: bool = os.getenv("RECALLIX_ENABLE_METRICS", "true").lower() in ("true", "1", "yes")

    # Safety limits
    max_memory_length: int = int(os.getenv("RECALLIX_MAX_MEMORY_LENGTH", "500"))
    max_query_length: int = int(os.getenv("RECALLIX_MAX_QUERY_LENGTH", "1000"))

    def reload(self):
        """Reload settings from environment variables."""
        self.database_url = os.getenv("RECALLIX_DATABASE_URL", "sqlite:///data/recallix.db")
        self.db_path = os.getenv("RECALLIX_DB_PATH", "data/recallix.db")
        self.embedding_model = os.getenv("RECALLIX_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.llm_model = os.getenv("RECALLIX_LLM_MODEL", os.getenv("RECALLIX_OLLAMA_MODEL", "qwen2.5:3b"))
        self.ollama_model = os.getenv("RECALLIX_OLLAMA_MODEL", os.getenv("RECALLIX_LLM_MODEL", "qwen2.5:3b"))
        self.llm_base_url = os.getenv("RECALLIX_LLM_BASE_URL", "http://localhost:11434")
        self.llm_timeout = int(os.getenv("RECALLIX_LLM_TIMEOUT", "120"))
        self.llm_connect_timeout = int(os.getenv("RECALLIX_LLM_CONNECT_TIMEOUT", "3"))
        self.default_top_k = int(os.getenv("RECALLIX_TOP_K", "5"))
        self.min_relevance = float(os.getenv("RECALLIX_MIN_RELEVANCE", "0.25"))
        self.log_level = os.getenv("RECALLIX_LOG_LEVEL", "INFO")
        self.enable_metrics = os.getenv("RECALLIX_ENABLE_METRICS", "true").lower() in ("true", "1", "yes")
        self.max_memory_length = int(os.getenv("RECALLIX_MAX_MEMORY_LENGTH", "500"))
        self.max_query_length = int(os.getenv("RECALLIX_MAX_QUERY_LENGTH", "1000"))
        return self

    @classmethod
    def from_env(cls) -> "Settings":
        """Factory method to construct fresh Settings from current environment."""
        s = cls()
        s.reload()
        return s


settings = Settings()


def get_settings() -> Settings:
    return settings
