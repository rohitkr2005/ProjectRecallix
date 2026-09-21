"""
Unit and integration tests for Recallix REST API and Authentication (Phase 15, Step 1).
"""

from datetime import timedelta
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.app import app
from app.api.auth import (
    create_access_token,
    decode_access_token,
    get_db,
    hash_password,
    verify_password,
)
from app.database.database import Base
from app.database.models import User, init_db
from app.utils.exceptions import RecallixError


@pytest.fixture
def db_session():
    """Create a clean in-memory SQLite database session for each test."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    init_db(target_engine=test_engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def client(db_session):
    """Create a FastAPI TestClient with database session dependency override."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 15.1 Health Check Tests
# ---------------------------------------------------------------------------
def test_health_endpoint(client):
    """Test that the /api/health endpoint returns 200 and expected status fields."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["version"] == "1.0.0"
    assert "database" in data
    assert "llm" in data
    assert "timestamp" in data


# ---------------------------------------------------------------------------
# 15.3 Password Hashing & JWT Security Unit Tests
# ---------------------------------------------------------------------------
def test_password_hashing_and_verification():
    """Test PBKDF2 password hashing and comparison."""
    password = "SuperSecretPassword123!"
    hashed = hash_password(password)

    assert hashed.startswith("pbkdf2_sha256$")
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False
    assert verify_password("", hashed) is False
    assert verify_password(password, "invalid_hash_string") is False


def test_jwt_token_generation_and_decoding():
    """Test creating and decoding JWT access tokens."""
    payload = {"sub": "42", "username": "alice"}
    token = create_access_token(payload, expires_delta=timedelta(minutes=15))

    decoded = decode_access_token(token)
    assert decoded["sub"] == "42"
    assert decoded["username"] == "alice"
    assert "exp" in decoded


def test_jwt_token_expired():
    """Test that expired JWT tokens raise HTTPException 401."""
    payload = {"sub": "42"}
    token = create_access_token(payload, expires_delta=timedelta(seconds=-10))

    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(token)
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


# ---------------------------------------------------------------------------
# 15.3 User Registration Endpoints
# ---------------------------------------------------------------------------
def test_register_user_success(client):
    """Test successful user registration returns 201 and UserResponse."""
    payload = {
        "username": "testuser",
        "email": "testuser@example.com",
        "password": "Password123",
    }
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "testuser"
    assert data["email"] == "testuser@example.com"
    assert "id" in data
    assert data["is_active"] is True
    assert "created_at" in data
    assert "password" not in data
    assert "hashed_password" not in data


def test_register_duplicate_username(client):
    """Test duplicate username registration returns 400 Bad Request."""
    payload = {
        "username": "duplicate_user",
        "email": "user1@example.com",
        "password": "Password123",
    }
    r1 = client.post("/api/v1/auth/register", json=payload)
    assert r1.status_code == 201

    payload2 = {
        "username": "duplicate_user",
        "email": "user2@example.com",
        "password": "Password123",
    }
    r2 = client.post("/api/v1/auth/register", json=payload2)
    assert r2.status_code == 400
    assert "Username already registered" in r2.json()["detail"]


def test_register_duplicate_email(client):
    """Test duplicate email registration returns 400 Bad Request."""
    payload = {
        "username": "user1",
        "email": "shared@example.com",
        "password": "Password123",
    }
    r1 = client.post("/api/v1/auth/register", json=payload)
    assert r1.status_code == 201

    payload2 = {
        "username": "user2",
        "email": "shared@example.com",
        "password": "Password123",
    }
    r2 = client.post("/api/v1/auth/register", json=payload2)
    assert r2.status_code == 400
    assert "Email already registered" in r2.json()["detail"]


# ---------------------------------------------------------------------------
# 15.3 User Login & Protected Routes
# ---------------------------------------------------------------------------
def test_login_user_success(client):
    """Test user login returns a valid JWT access token and user info."""
    # Register first
    client.post(
        "/api/v1/auth/register",
        json={"username": "loginuser", "email": "login@example.com", "password": "SecretPassword123"},
    )

    # Login
    login_response = client.post(
        "/api/v1/auth/login",
        json={"username": "loginuser", "password": "SecretPassword123"},
    )
    assert login_response.status_code == 200
    token_data = login_response.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    assert token_data["user"]["username"] == "loginuser"


def test_login_user_invalid_credentials(client):
    """Test login with wrong password returns 401."""
    client.post(
        "/api/v1/auth/register",
        json={"username": "user_wrong_pw", "email": "wrongpw@example.com", "password": "CorrectPassword1"},
    )

    # Wrong password
    r1 = client.post(
        "/api/v1/auth/login",
        json={"username": "user_wrong_pw", "password": "IncorrectPassword"},
    )
    assert r1.status_code == 401

    # Non-existent user
    r2 = client.post(
        "/api/v1/auth/login",
        json={"username": "non_existent", "password": "SomePassword"},
    )
    assert r2.status_code == 401


def test_protected_me_endpoint_flow(client):
    """Test accessing /api/v1/auth/me with valid, invalid, and missing tokens."""
    # 1. Register & Login
    client.post(
        "/api/v1/auth/register",
        json={"username": "profileuser", "email": "profile@example.com", "password": "SecretPassword123"},
    )
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "profileuser", "password": "SecretPassword123"},
    )
    token = login_resp.json()["access_token"]

    # 2. Access /me with valid token
    headers = {"Authorization": f"Bearer {token}"}
    me_resp = client.get("/api/v1/auth/me", headers=headers)
    assert me_resp.status_code == 200
    profile = me_resp.json()
    assert profile["username"] == "profileuser"
    assert profile["email"] == "profile@example.com"

    # 3. Access /me with missing token
    missing_resp = client.get("/api/v1/auth/me")
    assert missing_resp.status_code == 401

    # 4. Access /me with invalid token
    invalid_resp = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer invalid.jwt.token"})
    assert invalid_resp.status_code == 401


def test_recallix_error_handler(client):
    """Test that internal RecallixError triggers global exception handler returning 400."""
    # Temporary test endpoint that raises RecallixError
    @app.get("/api/test-error")
    def trigger_error():
        raise RecallixError("Test memory failure", details={"code": "ERR_TEST"})

    resp = client.get("/api/test-error")
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"] == "Test memory failure"
    assert data["details"]["code"] == "ERR_TEST"
    assert data["type"] == "RecallixError"
