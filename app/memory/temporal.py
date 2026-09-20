from enum import Enum
import re
from typing import Optional


class TemporalState(str, Enum):
    """
    10.5 Temporal Memory

    Distinguishes temporal state of user memories:
    - PAST: historical facts ("Previously lived in Delhi", "Used to work at Google")
    - PRESENT: current state ("Currently lives in Mumbai", "Works on Recallix")
    - FUTURE: aspirational / planned facts ("Plans to move to Bangalore", "Wants to learn Rust")
    """
    PAST = "PAST"
    PRESENT = "PRESENT"
    FUTURE = "FUTURE"


PAST_PATTERNS = [
    r"\b(?:previously|formerly|used to|in the past|earlier|back in)\b",
    r"\b(?:was living|was working|studied at .+ before|lived in .+ before)\b",
    r"\b(?:had a role|formerly worked|previously lived)\b",
    r"\b(?:did live|did work)\b",
]

FUTURE_PATTERNS = [
    r"\b(?:plans to|planning to|plan to|plans on|planning on)\b",
    r"\b(?:wants to|want to|aims to|aiming to|hoping to|hope to)\b",
    r"\b(?:will move|will live|will learn|will build|will work)\b",
    r"\b(?:going to move|going to learn|in the future)\b",
    r"\b(?:intends to|aspiring to)\b",
]

PRESENT_PATTERNS = [
    r"\b(?:currently|now|at present|today|at the moment)\b",
    r"\b(?:current city|current role|current project)\b",
    r"\b(?:is currently|am currently)\b",
]

QUERY_PAST_PATTERNS = [
    r"\b(?:did i live|did i work|did i study|where was i|what was my|where did i)\b",
    r"\b(?:before|previously|formerly|in the past|used to|use to)\b",
]

QUERY_FUTURE_PATTERNS = [
    r"\b(?:plan to|planning to|plans to|want to|wants to|will i|future)\b",
    r"\b(?:where do i plan|what do i plan|what do i want to learn)\b",
    r"\b(?:planning to move|planning to build)\b",
]

QUERY_PRESENT_PATTERNS = [
    r"\b(?:currently|now|at present|current)\b",
]


def detect_temporal_state(text: str) -> TemporalState:
    """Detect whether a factual statement represents a past, present, or future fact."""
    if not text or not text.strip():
        return TemporalState.PRESENT

    t = text.strip().lower()

    for pattern in PAST_PATTERNS:
        if re.search(pattern, t, re.IGNORECASE):
            return TemporalState.PAST

    for pattern in FUTURE_PATTERNS:
        if re.search(pattern, t, re.IGNORECASE):
            return TemporalState.FUTURE

    return TemporalState.PRESENT


def detect_query_temporal_intent(query: str) -> Optional[TemporalState]:
    """Detect if a user's question targets a specific temporal aspect."""
    if not query or not query.strip():
        return None

    q = query.strip().lower()

    for pattern in QUERY_PAST_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return TemporalState.PAST

    for pattern in QUERY_FUTURE_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return TemporalState.FUTURE

    for pattern in QUERY_PRESENT_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return TemporalState.PRESENT

    return None
