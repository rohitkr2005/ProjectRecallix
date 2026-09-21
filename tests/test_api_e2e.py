"""
15.12 End-to-End Integration Tests for Recallix Web Application

Simulates complete user lifecycles: registration, login, memory ingestion via chat,
grounded question-answering with explainability, memory CRUD operations, search sandbox,
multi-user data isolation, and frontend asset delivery.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.app import app
from app.api.auth import get_db
from app.database.database import Base
from app.database.models import init_db


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


def test_full_user_lifecycle_and_multi_user_isolation_e2e(client):
    """
    Complete end-to-end integration test:
    1. Health check
    2. User registration & authentication
    3. Chat memory ingestion & grounded recall
    4. Memory CRUD & search sandbox
    5. Multi-user security isolation
    6. Frontend static asset delivery
    """
    # -----------------------------------------------------------------------
    # 1. Health Check
    # -----------------------------------------------------------------------
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data["status"] == "healthy"
    assert health_data["version"] == "1.0.0"

    # -----------------------------------------------------------------------
    # 2. User 1 Registration & Authentication
    # -----------------------------------------------------------------------
    reg_payload = {
        "username": "alex",
        "email": "alex@recallix.ai",
        "password": "SecurePassword123!",
    }
    reg_resp = client.post("/api/v1/auth/register", json=reg_payload)
    assert reg_resp.status_code == 201
    user1_id = reg_resp.json()["id"]

    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username": "alex", "password": "SecurePassword123!"},
    )
    assert login_resp.status_code == 200
    user1_token = login_resp.json()["access_token"]
    user1_headers = {"Authorization": f"Bearer {user1_token}"}

    me_resp = client.get("/api/v1/auth/me", headers=user1_headers)
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == "alex"

    # -----------------------------------------------------------------------
    # 3. Chat Memory Ingestion & Grounded Recall
    # -----------------------------------------------------------------------
    # Ingest project memory via chat
    stmt1_resp = client.post(
        "/api/v1/chat",
        json={"message": "I am working on Project Recallix Web App.", "include_explanations": True},
        headers=user1_headers,
    )
    assert stmt1_resp.status_code == 200
    stmt1_data = stmt1_resp.json()
    assert stmt1_data["input_type"] == "STATEMENT"
    assert stmt1_data["grounded"] is True

    # Ingest location memory via chat
    stmt2_resp = client.post(
        "/api/v1/chat",
        json={"message": "I live in San Francisco.", "include_explanations": True},
        headers=user1_headers,
    )
    assert stmt2_resp.status_code == 200
    assert stmt2_resp.json()["input_type"] == "STATEMENT"

    # Query location via chat
    q1_resp = client.post(
        "/api/v1/chat",
        json={"message": "Where do I live?", "include_explanations": True},
        headers=user1_headers,
    )
    assert q1_resp.status_code == 200
    q1_data = q1_resp.json()
    assert q1_data["input_type"] == "QUESTION"
    assert q1_data["supported"] is True
    assert q1_data["grounded"] is True
    assert "San Francisco" in q1_data["response"]
    assert q1_data["why_used"] is not None
    assert len(q1_data["why_used"]) >= 1

    # Query project via chat
    q2_resp = client.post(
        "/api/v1/chat",
        json={"message": "What project am I building?", "include_explanations": True},
        headers=user1_headers,
    )
    assert q2_resp.status_code == 200
    q2_data = q2_resp.json()
    assert q2_data["supported"] is True
    assert "Recallix" in q2_data["response"]

    # -----------------------------------------------------------------------
    # 4. Memory CRUD Operations & Search Sandbox
    # -----------------------------------------------------------------------
    # Explicit memory creation
    create_mem_resp = client.post(
        "/api/v1/memories",
        json={
            "subject": "User",
            "relation": "knows",
            "value": "Reinforcement Learning",
            "category": "SKILL",
            "importance": 9,
            "temporal_state": "PRESENT",
        },
        headers=user1_headers,
    )
    assert create_mem_resp.status_code == 201
    created_mem_id = create_mem_resp.json()["id"]

    # List memories
    list_resp = client.get("/api/v1/memories", headers=user1_headers)
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total"] == 3  # project, location, skill

    # Search sandbox
    search_resp = client.post(
        "/api/v1/search",
        json={"query": "What machine learning skills do I have?", "top_k": 3},
        headers=user1_headers,
    )
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert len(search_data["results"]) >= 1
    assert search_data["results"][0]["memory"]["value"] == "Reinforcement Learning"

    # Update memory
    update_resp = client.put(
        f"/api/v1/memories/{created_mem_id}",
        json={"value": "Deep Reinforcement Learning", "importance": 10},
        headers=user1_headers,
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["value"] == "Deep Reinforcement Learning"
    assert update_resp.json()["importance"] == 10

    # Archive memory
    del_resp = client.delete(f"/api/v1/memories/{created_mem_id}", headers=user1_headers)
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "archived"

    # Verify active list has 2 memories
    list_active = client.get("/api/v1/memories", headers=user1_headers).json()
    assert list_active["total"] == 2

    # Restore memory
    restore_resp = client.post(f"/api/v1/memories/{created_mem_id}/restore", headers=user1_headers)
    assert restore_resp.status_code == 200
    assert restore_resp.json()["status"] == "restored"

    # Verify active list is back to 3
    list_restored = client.get("/api/v1/memories", headers=user1_headers).json()
    assert list_restored["total"] == 3

    # -----------------------------------------------------------------------
    # 5. Multi-User Security & Data Isolation
    # -----------------------------------------------------------------------
    # Register and login User 2 (Bob)
    client.post(
        "/api/v1/auth/register",
        json={"username": "bob", "email": "bob@recallix.ai", "password": "BobPassword456!"},
    )
    bob_login = client.post(
        "/api/v1/auth/login",
        json={"username": "bob", "password": "BobPassword456!"},
    )
    bob_headers = {"Authorization": f"Bearer {bob_login.json()['access_token']}"}

    # Bob asks where he lives -> unsupported, must NOT see Alex's "San Francisco"
    bob_chat = client.post(
        "/api/v1/chat",
        json={"message": "Where do I live?"},
        headers=bob_headers,
    )
    assert bob_chat.status_code == 200
    assert bob_chat.json()["supported"] is False
    assert "San Francisco" not in bob_chat.json()["response"]

    # Bob's memory list is completely empty
    bob_list = client.get("/api/v1/memories", headers=bob_headers).json()
    assert bob_list["total"] == 0

    # Bob cannot access Alex's memory
    bob_get_mem = client.get(f"/api/v1/memories/{created_mem_id}", headers=bob_headers)
    assert bob_get_mem.status_code == 404

    bob_put_mem = client.put(f"/api/v1/memories/{created_mem_id}", json={"value": "Hacked"}, headers=bob_headers)
    assert bob_put_mem.status_code == 404

    bob_del_mem = client.delete(f"/api/v1/memories/{created_mem_id}", headers=bob_headers)
    assert bob_del_mem.status_code == 404

    # -----------------------------------------------------------------------
    # 6. Frontend Static Asset Delivery
    # -----------------------------------------------------------------------
    # Main HTML page
    index_resp = client.get("/")
    assert index_resp.status_code == 200
    assert "<title>Recallix — AI Second Brain</title>" in index_resp.text
    assert "authView" in index_resp.text
    assert "appContainer" in index_resp.text

    # CSS static file
    css_resp = client.get("/static/css/styles.css")
    assert css_resp.status_code == 200
    assert "--bg-primary" in css_resp.text

    # JS static files
    api_js_resp = client.get("/static/js/api.js")
    assert api_js_resp.status_code == 200
    assert "class RecallixAPI" in api_js_resp.text

    app_js_resp = client.get("/static/js/app.js")
    assert app_js_resp.status_code == 200
    assert "checkAuthSession" in app_js_resp.text
