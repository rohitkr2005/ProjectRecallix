"""
15.1 - 15.3 FastAPI Application & REST API Architecture

Main FastAPI application exposing health, authentication, and core Recallix API endpoints.
"""

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.auth import (
    create_access_token,
    get_current_user,
    get_db,
    hash_password,
    verify_password,
)
from app.api.schemas import (
    HealthResponse,
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)
from app.config import get_settings
from app.database.models import User, init_db
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
