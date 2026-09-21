from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Auth Schemas
# ---------------------------------------------------------------------------
class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=6, max_length=100)


class UserLoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ---------------------------------------------------------------------------
# Chat Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    include_explanations: bool = False
    top_k: int = Field(default=5, ge=1, le=20)


class GroundingDetailsSchema(BaseModel):
    grounded: bool
    supported_values: List[str] = []
    grounding_score: float = 1.0
    details: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    input_type: str
    supported: bool
    grounded: bool
    grounding_details: Optional[GroundingDetailsSchema] = None
    performance_metrics: Optional[Dict[str, float]] = None
    memories_used: List[Dict[str, Any]] = []
    why_used: Optional[List[str]] = None
    llm_status: str


# ---------------------------------------------------------------------------
# Memory Schemas
# ---------------------------------------------------------------------------
class MemoryCreateRequest(BaseModel):
    subject: str = Field(default="User", max_length=100)
    relation: str = Field(..., min_length=1, max_length=100)
    value: str = Field(..., min_length=1, max_length=500)
    category: str = Field(default="GENERAL", max_length=50)
    importance: int = Field(default=5, ge=1, le=10)
    temporal_state: str = Field(default="PRESENT", pattern="^(PAST|PRESENT|FUTURE)$")


class MemoryUpdateRequest(BaseModel):
    value: Optional[str] = Field(None, min_length=1, max_length=500)
    category: Optional[str] = Field(None, max_length=50)
    importance: Optional[int] = Field(None, ge=1, le=10)
    temporal_state: Optional[str] = Field(None, pattern="^(PAST|PRESENT|FUTURE)$")


class MemoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    subject: str
    relation: str
    value: str
    category: str
    importance: int
    active: bool
    temporal_state: Optional[str] = "PRESENT"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class MemoryListResponse(BaseModel):
    total: int
    items: List[MemoryResponse]


# ---------------------------------------------------------------------------
# Search Schemas
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchResultItem(BaseModel):
    memory: MemoryResponse
    score: float
    explanation: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    query: str
    total: int
    results: List[SearchResultItem]


# ---------------------------------------------------------------------------
# Health & Status Schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    llm: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
