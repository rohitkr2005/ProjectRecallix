"""
15.1 - 15.3 FastAPI Application & REST API Architecture

Main FastAPI application exposing health, authentication, and core Recallix API endpoints.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List

import os
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import or_, text
from sqlalchemy.orm import Session

from app.api.auth import (
    create_access_token,
    get_current_user,
    get_db,
    hash_password,
    verify_password,
)
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    GroundingDetailsSchema,
    HealthResponse,
    MemoryCreateRequest,
    MemoryListResponse,
    MemoryResponse,
    MemoryUpdateRequest,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)
from app.assistant.assistant_engine import AssistantEngine
from app.config import get_settings
from app.database.models import Memory, User, init_db
from app.embeddings.embedding_engine import EmbeddingEngine
from app.memory.memory_store import MemoryStore
from app.retrieval.retrieval_engine import RetrievalEngine
from app.utils.exceptions import RecallixError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context: initialize database tables on startup."""
    init_db()
    yield


app = FastAPI(
    title="Project Recallix API",
    description="Personal AI Second Brain REST API — Memory, Retrieval, and Chat",
    version="1.0.0",
    lifespan=lifespan,
)

# 15.1 CORS Configuration for browser and client access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 15.6 Static Files & Web App Mounting
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))


# ---------------------------------------------------------------------------
# Global Exception Handlers
# ---------------------------------------------------------------------------
@app.exception_handler(RecallixError)
async def recallix_exception_handler(request: Request, exc: RecallixError):
    """Map internal Recallix errors to standardized JSON error responses."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": exc.message,
            "details": exc.details,
            "type": type(exc).__name__,
        },
    )


# ---------------------------------------------------------------------------
# 15.1 Health & Status Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check(db: Session = Depends(get_db)):
    """Check backend, database, and LLM configuration health."""
    settings = get_settings()
    db_status = "healthy"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    llm_status = "configured" if settings.llm_base_url else "not_configured"

    return HealthResponse(
        status="healthy" if db_status == "healthy" else "degraded",
        version="1.0.0",
        database=db_status,
        llm=llm_status,
        timestamp=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# 15.3 Authentication Endpoints
# ---------------------------------------------------------------------------
@app.post(
    "/api/v1/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Authentication"],
)
def register_user(
    req: UserRegisterRequest,
    db: Session = Depends(get_db),
):
    """Register a new user account with unique username and email."""
    # Check if username exists
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    # Check if email exists
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Hash password and persist new user
    hashed = hash_password(req.password)
    new_user = User(
        username=req.username,
        email=req.email,
        hashed_password=hashed,
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


@app.post(
    "/api/v1/auth/login",
    response_model=TokenResponse,
    tags=["Authentication"],
)
def login_user(
    req: UserLoginRequest,
    db: Session = Depends(get_db),
):
    """Authenticate user credentials and issue a JWT access token."""
    user = db.query(User).filter(User.username == req.username).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is deactivated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": str(user.id), "username": user.username})
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


@app.get(
    "/api/v1/auth/me",
    response_model=UserResponse,
    tags=["Authentication"],
)
def get_current_user_profile(
    current_user: User = Depends(get_current_user),
):
    """Retrieve the profile of the currently authenticated user."""
    return current_user


# ---------------------------------------------------------------------------
# 15.4 Chat Endpoint
# ---------------------------------------------------------------------------
@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    tags=["Chat"],
)
def chat_with_assistant(
    req: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Send a message to Recallix Assistant with memory retrieval,
    knowledge grounding, and user-scoped context.
    """
    memory_store = MemoryStore(session=db)
    retrieval_engine = RetrievalEngine(memory_store=memory_store)
    assistant = AssistantEngine(
        retrieval_engine=retrieval_engine,
        memory_store=memory_store,
    )

    result = assistant.respond(
        user_message=req.message,
        top_k=req.top_k,
        include_explanations=req.include_explanations,
        user_id=current_user.id,
    )

    grounding_details = None
    if result.get("grounding_details"):
        gd = result["grounding_details"]
        grounding_details = GroundingDetailsSchema(
            grounded=gd.get("grounded", True),
            supported_values=gd.get("supported_values", []),
            grounding_score=gd.get("grounding_score", 1.0),
            details=gd.get("details"),
        )

    memories_used = []
    for item in result.get("memories", []):
        if isinstance(item, dict):
            mem = item.get("memory")
            if mem:
                memories_used.append({
                    "id": getattr(mem, "id", None),
                    "subject": getattr(mem, "subject", ""),
                    "relation": getattr(mem, "relation", ""),
                    "value": getattr(mem, "value", ""),
                    "score": item.get("score", 0.0),
                })

    raw_input_type = result["input_type"]
    input_type_str = raw_input_type.value if hasattr(raw_input_type, "value") else str(raw_input_type)

    return ChatResponse(
        response=result["response"],
        input_type=input_type_str,
        supported=result["supported"],
        grounded=result["grounded"],
        grounding_details=grounding_details,
        performance_metrics=result.get("performance_metrics"),
        memories_used=memories_used,
        why_used=result.get("why_used"),
        llm_status=result.get("llm_status", "completed"),
    )


# ---------------------------------------------------------------------------
# 15.5 Memory CRUD Endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/api/v1/memories",
    response_model=MemoryListResponse,
    tags=["Memories"],
)
def list_memories(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    category: Optional[str] = Query(None, description="Filter by category"),
    include_archived: bool = Query(False, description="Include archived memories"),
    search: Optional[str] = Query(None, description="Filter memories by substring"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List memories belonging to the authenticated user with filtering and pagination."""
    query = db.query(Memory).filter(Memory.user_id == current_user.id)

    if not include_archived:
        query = query.filter(Memory.active.is_(True))

    if category:
        query = query.filter(Memory.category == category)

    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(
            or_(
                Memory.subject.ilike(pattern),
                Memory.relation.ilike(pattern),
                Memory.value.ilike(pattern),
            )
        )

    total = query.count()
    items = (
        query.order_by(Memory.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return MemoryListResponse(
        total=total,
        items=[MemoryResponse.model_validate(m) for m in items],
    )


@app.post(
    "/api/v1/memories",
    response_model=MemoryResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Memories"],
)
def create_memory(
    req: MemoryCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create and embed a new memory for the authenticated user."""
    embedding_engine = EmbeddingEngine()
    embedding = None
    try:
        embedding = embedding_engine.generate_memory_embedding(
            subject=req.subject,
            relation=req.relation,
            value=req.value,
            category=req.category,
        )
    except Exception:
        embedding = None

    store = MemoryStore(session=db)
    memory, status_text, _ = store.save_memory_with_semantics(
        subject=req.subject,
        relation=req.relation,
        value=req.value,
        category=req.category,
        importance=req.importance,
        embedding=embedding,
        temporal_state=req.temporal_state,
        user_id=current_user.id,
    )

    return MemoryResponse.model_validate(memory)


@app.get(
    "/api/v1/memories/{memory_id}",
    response_model=MemoryResponse,
    tags=["Memories"],
)
def get_memory_detail(
    memory_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve details of a single memory by ID."""
    memory = (
        db.query(Memory)
        .filter(Memory.id == memory_id, Memory.user_id == current_user.id)
        .first()
    )
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )
    return MemoryResponse.model_validate(memory)


@app.put(
    "/api/v1/memories/{memory_id}",
    response_model=MemoryResponse,
    tags=["Memories"],
)
def update_memory_detail(
    memory_id: int,
    req: MemoryUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update fields on an existing memory."""
    memory = (
        db.query(Memory)
        .filter(Memory.id == memory_id, Memory.user_id == current_user.id)
        .first()
    )
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )

    if req.value is not None:
        memory.value = req.value
        embedding_engine = EmbeddingEngine()
        try:
            memory.embedding = embedding_engine.generate_memory_embedding(
                subject=memory.subject,
                relation=memory.relation,
                value=req.value,
                category=memory.category,
            )
        except Exception:
            pass

    if req.category is not None:
        memory.category = req.category

    if req.importance is not None:
        memory.importance = req.importance

    if req.temporal_state is not None:
        memory.temporal_state = req.temporal_state

    memory.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(memory)

    return MemoryResponse.model_validate(memory)


@app.delete(
    "/api/v1/memories/{memory_id}",
    tags=["Memories"],
)
def archive_memory_endpoint(
    memory_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Archive (soft-delete) a memory."""
    memory = (
        db.query(Memory)
        .filter(Memory.id == memory_id, Memory.user_id == current_user.id)
        .first()
    )
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )

    store = MemoryStore(session=db)
    store.archive_memory(memory_id)
    return {"status": "archived", "id": memory_id}


@app.post(
    "/api/v1/memories/{memory_id}/restore",
    tags=["Memories"],
)
def restore_memory_endpoint(
    memory_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Restore an archived memory."""
    memory = (
        db.query(Memory)
        .filter(Memory.id == memory_id, Memory.user_id == current_user.id)
        .first()
    )
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )

    store = MemoryStore(session=db)
    success = store.restore_memory(memory_id, allow_conflict=True)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to restore memory due to conflicting active records",
        )

    db.refresh(memory)
    return {
        "status": "restored",
        "id": memory_id,
        "memory": MemoryResponse.model_validate(memory),
    }


# ---------------------------------------------------------------------------
# 15.5 Search Endpoint
# ---------------------------------------------------------------------------
@app.post(
    "/api/v1/search",
    response_model=SearchResponse,
    tags=["Search"],
)
def search_memories(
    req: SearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Perform intent-aware semantic and relational search over user memories."""
    store = MemoryStore(session=db)
    retrieval_engine = RetrievalEngine(memory_store=store)
    results = retrieval_engine.search(
        query=req.query,
        top_k=req.top_k,
        min_score=req.min_score,
        user_id=current_user.id,
    )

    items = []
    for r in results:
        items.append(
            SearchResultItem(
                memory=MemoryResponse.model_validate(r["memory"]),
                score=r["score"],
                explanation=r.get("explanation"),
            )
        )

    return SearchResponse(
        query=req.query,
        total=len(items),
        results=items,
    )

