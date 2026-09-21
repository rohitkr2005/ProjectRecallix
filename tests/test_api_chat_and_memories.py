"""
Unit and integration tests for Recallix Chat and Memory CRUD / Search APIs (Phase 15, Step 2: 15.4 - 15.5).
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.app import app
from app.api.auth import get_db
from app.database.database import Base
from app.database.models import Memory, User, init_db


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


@pytest.fixture
def auth_headers(client):
    """Register and login a primary test user, returning Bearer auth headers."""
    client.post(
        "/api/v1/auth/register",
        json={"username": "alice", "email": "alice@example.com", "password": "Password123!"},
    )
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "alice", "password": "Password123!"},
    )
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_user2(client):
    """Register and login a secondary test user for multi-user isolation tests."""
    client.post(
        "/api/v1/auth/register",
        json={"username": "bob", "email": "bob@example.com", "password": "Password456!"},
    )
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "bob", "password": "Password456!"},
    )
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# 15.5 Memory CRUD Tests
# ---------------------------------------------------------------------------
def test_create_and_get_memory(client, auth_headers):
    """Test creating a memory via POST /api/v1/memories and retrieving it."""
    payload = {
        "subject": "User",
        "relation": "works_on",
        "value": "Project Recallix",
        "category": "PROJECT",
        "importance": 8,
        "temporal_state": "PRESENT",
    }
    create_resp = client.post("/api/v1/memories", json=payload, headers=auth_headers)
    assert create_resp.status_code == 201
    created = create_resp.json()
    assert created["relation"] == "works_on"
    assert created["value"] == "Project Recallix"
    assert created["importance"] == 8
    assert created["active"] is True
    memory_id = created["id"]

    # Retrieve single memory
    get_resp = client.get(f"/api/v1/memories/{memory_id}", headers=auth_headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == memory_id


def test_list_memories_and_pagination(client, auth_headers):
    """Test listing memories with filtering and pagination."""
    # Add multiple memories
    for i in range(5):
        client.post(
            "/api/v1/memories",
            json={"subject": "User", "relation": "knows", "value": f"Skill_{i}", "category": "SKILL"},
            headers=auth_headers,
        )

    # List page 1 with page_size=3
    resp = client.get("/api/v1/memories?page=1&page_size=3", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5
    assert len(data["items"]) == 3

    # Filter by category
    resp_cat = client.get("/api/v1/memories?category=SKILL", headers=auth_headers)
    assert resp_cat.json()["total"] == 5

    # Filter by search substring
    resp_search = client.get("/api/v1/memories?search=Skill_2", headers=auth_headers)
    assert resp_search.json()["total"] == 1
    assert resp_search.json()["items"][0]["value"] == "Skill_2"


def test_update_memory(client, auth_headers):
    """Test updating an existing memory's value, category, and importance."""
    create_resp = client.post(
        "/api/v1/memories",
        json={"subject": "User", "relation": "likes", "value": "Football", "importance": 5},
        headers=auth_headers,
    )
    mem_id = create_resp.json()["id"]

    update_payload = {"value": "Basketball", "importance": 9, "category": "SPORTS"}
    put_resp = client.put(f"/api/v1/memories/{mem_id}", json=update_payload, headers=auth_headers)
    assert put_resp.status_code == 200
    updated = put_resp.json()
    assert updated["value"] == "Basketball"
    assert updated["importance"] == 9
    assert updated["category"] == "SPORTS"


def test_archive_and_restore_memory(client, auth_headers):
    """Test soft-deleting (archiving) and restoring a memory."""
    create_resp = client.post(
        "/api/v1/memories",
        json={"subject": "User", "relation": "lives_in", "value": "Delhi"},
        headers=auth_headers,
    )
    mem_id = create_resp.json()["id"]

    # Archive
    del_resp = client.delete(f"/api/v1/memories/{mem_id}", headers=auth_headers)
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "archived"

    # Default list should exclude archived
    list_active = client.get("/api/v1/memories", headers=auth_headers).json()
    assert list_active["total"] == 0

    # Including archived should show it
    list_all = client.get("/api/v1/memories?include_archived=true", headers=auth_headers).json()
    assert list_all["total"] == 1
    assert list_all["items"][0]["active"] is False

    # Restore
    restore_resp = client.post(f"/api/v1/memories/{mem_id}/restore", headers=auth_headers)
    assert restore_resp.status_code == 200
    assert restore_resp.json()["status"] == "restored"
    assert restore_resp.json()["memory"]["active"] is True


# ---------------------------------------------------------------------------
# 15.3 & 15.5 Multi-User Data Isolation Tests
# ---------------------------------------------------------------------------
def test_multi_user_memory_isolation(client, auth_headers, auth_headers_user2):
    """Ensure User 1's memories are invisible and inaccessible to User 2."""
    # User 1 creates a secret memory
    create_resp = client.post(
        "/api/v1/memories",
        json={"subject": "User", "relation": "secret_key", "value": "AliceSecret123"},
        headers=auth_headers,
    )
    alice_mem_id = create_resp.json()["id"]

    # User 2 lists memories -> should be empty
    u2_list = client.get("/api/v1/memories", headers=auth_headers_user2).json()
    assert u2_list["total"] == 0

    # User 2 tries to GET Alice's memory by ID -> 404
    u2_get = client.get(f"/api/v1/memories/{alice_mem_id}", headers=auth_headers_user2)
    assert u2_get.status_code == 404

    # User 2 tries to PUT Alice's memory -> 404
    u2_put = client.put(f"/api/v1/memories/{alice_mem_id}", json={"value": "Hacked"}, headers=auth_headers_user2)
    assert u2_put.status_code == 404

    # User 2 tries to DELETE Alice's memory -> 404
    u2_del = client.delete(f"/api/v1/memories/{alice_mem_id}", headers=auth_headers_user2)
    assert u2_del.status_code == 404


# ---------------------------------------------------------------------------
# 15.5 Search Endpoint Tests
# ---------------------------------------------------------------------------
def test_search_memories_endpoint(client, auth_headers):
    """Test POST /api/v1/search endpoint with intent and embedding ranking."""
    client.post(
        "/api/v1/memories",
        json={"subject": "User", "relation": "works_on", "value": "Quantum AI", "category": "PROJECT"},
        headers=auth_headers,
    )
    client.post(
        "/api/v1/memories",
        json={"subject": "User", "relation": "likes", "value": "Italian Pasta", "category": "PREFERENCE"},
        headers=auth_headers,
    )

    search_resp = client.post(
        "/api/v1/search",
        json={"query": "Which projects am I working on?", "top_k": 3},
        headers=auth_headers,
    )
    assert search_resp.status_code == 200
    data = search_resp.json()
    assert data["query"] == "Which projects am I working on?"
    assert len(data["results"]) >= 1
    top_result = data["results"][0]
    assert top_result["memory"]["value"] == "Quantum AI"
    assert "score" in top_result
    assert "explanation" in top_result


# ---------------------------------------------------------------------------
# 15.4 Chat Endpoint Tests
# ---------------------------------------------------------------------------
def test_chat_statement_and_retrieval_flow(client, auth_headers):
    """Test full conversational memory flow: save via statement, query via question."""
    # 1. Statement
    stmt_resp = client.post(
        "/api/v1/chat",
        json={"message": "I live in Berlin.", "include_explanations": True},
        headers=auth_headers,
    )
    assert stmt_resp.status_code == 200
    stmt_data = stmt_resp.json()
    assert "live in Berlin" in stmt_data["response"] or "Berlin" in stmt_data["response"]
    assert stmt_data["input_type"] == "STATEMENT"
    assert stmt_data["grounded"] is True

    # 2. Question
    q_resp = client.post(
        "/api/v1/chat",
        json={"message": "Where do I live?", "include_explanations": True},
        headers=auth_headers,
    )
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    assert q_data["input_type"] == "QUESTION"
    assert q_data["supported"] is True
    assert "Berlin" in q_data["response"]
    assert q_data["why_used"] is not None
    assert len(q_data["why_used"]) >= 1


def test_chat_multi_user_isolation(client, auth_headers, auth_headers_user2):
    """Ensure User 2 cannot retrieve or hallucinate User 1's facts."""
    # User 1 stores city
    client.post(
        "/api/v1/chat",
        json={"message": "I live in Tokyo."},
        headers=auth_headers,
    )

    # User 2 asks where they live -> should decline / unsupported
    u2_resp = client.post(
        "/api/v1/chat",
        json={"message": "Where do I live?"},
        headers=auth_headers_user2,
    )
    assert u2_resp.status_code == 200
    u2_data = u2_resp.json()
    assert u2_data["supported"] is False
    assert "Tokyo" not in u2_data["response"]


def test_frontend_index_served(client):
    """Verify that the frontend index.html is served at root GET /."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Recallix" in response.text
    assert "AI Second Brain" in response.text

