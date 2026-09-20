import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.config import settings


_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class JSONFormatter(logging.Formatter):
    """Format log records as JSON strings with extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "message"
            } and not key.startswith("_"):
                data[key] = val
        return json.dumps(data)


def get_logger(name: str = "recallix") -> logging.Logger:
    """Return a configured logger with the log level specified in settings."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(_LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    level_str = getattr(settings, "log_level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)
    return logger


def log_memory_event(
    logger: logging.Logger,
    action: str,
    memory_id: Optional[int] = None,
    subject: Optional[str] = None,
    relation: Optional[str] = None,
    status: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """Log structured memory lifecycle event."""
    merged_details = dict(details or {})
    if subject:
        merged_details["subject"] = subject
    if relation:
        merged_details["relation"] = relation
    if status:
        merged_details["status"] = status
    merged_details.update(kwargs)

    data = {
        "event_type": "memory_event",
        "action": action,
        "memory_id": memory_id,
        "subject": subject,
        "details": merged_details,
    }
    logger.info(
        f"Memory event: {action} (id={memory_id})",
        extra=data,
    )


def log_retrieval_event(
    logger: logging.Logger,
    query: str,
    count: int = 0,
    top_score: float = 0.0,
    top_k: Optional[int] = None,
    latency_ms: float = 0.0,
    details: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """Log structured retrieval event."""
    data = {
        "event_type": "retrieval_event",
        "query": query,
        "count": count,
        "top_k": top_k or count,
        "top_score": round(top_score, 4),
        "latency_ms": round(latency_ms, 2),
        "details": details or {},
    }
    data.update(kwargs)
    logger.info(
        f"Retrieval event: {count} memories for '{query}'",
        extra=data,
    )


def log_llm_event(
    logger: logging.Logger,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
    status: str = "success",
    tokens: int = 0,
    latency_ms: float = 0.0,
    details: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """Log structured LLM inference event."""
    data = {
        "event_type": "llm_event",
        "prompt": prompt,
        "model": model or getattr(settings, "ollama_model", "llama3:latest"),
        "status": status,
        "tokens": tokens,
        "latency_ms": round(latency_ms, 2),
        "details": details or {},
    }
    data.update(kwargs)
    logger.info(
        f"LLM event: {status} (model={data['model']})",
        extra=data,
    )


def log_error_event(
    logger: logging.Logger,
    context: Optional[str] = None,
    error: Optional[Exception] = None,
    error_type: Optional[str] = None,
    message: Optional[str] = None,
    component: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """Log structured system error event."""
    err_cls = error_type or (type(error).__name__ if error else "RecallixError")
    err_msg = message or (str(error) if error else "Unknown error")
    data = {
        "event_type": "error_event",
        "context": context or component or "general",
        "error_type": err_cls,
        "error_message": err_msg,
        "details": details or {},
    }
    data.update(kwargs)
    logger.error(
        f"Error in {data['context']}: [{err_cls}] {err_msg}",
        extra=data,
    )
