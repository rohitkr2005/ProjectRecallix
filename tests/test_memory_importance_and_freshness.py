import time
import uuid
from datetime import datetime, timedelta
import pytest
from app.memory.lifecycle import (
    calculate_freshness_score,
    validate_importance,
)
from app.memory.memory_store import MemoryStore


def test_validate_importance_clamping():
    assert validate_importance(1) == 1
    assert validate_importance(10) == 10
    assert validate_importance(5) == 5
    assert validate_importance(0) == 1
    assert validate_importance(-5) == 1
    assert validate_importance(15) == 10
    assert validate_importance(None) == 5
    assert validate_importance("invalid") == 5


def test_update_importance_in_place():
    store = MemoryStore()
    user_id = f"ImpUser_{uuid.uuid4()}"

    mem, _ = store.save_memory(
        subject=user_id,
        relation="likes",
        value="Python",
        category="PREFERENCE",
        importance=5,
    )
    assert mem.importance == 5

    time.sleep(0.01)
    original_updated_at = mem.updated_at

    success = store.update_importance(mem.id, 9)
    assert success is True
    assert mem.importance == 9
    assert mem.updated_at >= original_updated_at

    # Clamping during update
    store.update_importance(mem.id, 99)
    assert mem.importance == 10

    # Nonexistent memory returns False
    assert store.update_importance(9999999, 8) is False

    store.close()


def test_freshness_exponential_decay():
    class MockMemory:
        importance = 1  # Minimal floor
        created_at = datetime.utcnow()
        updated_at = datetime.utcnow()

    mem = MockMemory()
    now = datetime.utcnow()

    # 1. Immediate memory -> ~1.0
    score_0 = calculate_freshness_score(mem, reference_time=now, half_life_days=30.0, importance_damping=False)
    assert score_0 == pytest.approx(1.0, abs=0.01)

    # 2. 30 days old -> ~0.50
    mem.updated_at = now - timedelta(days=30)
    score_30 = calculate_freshness_score(mem, reference_time=now, half_life_days=30.0, importance_damping=False)
    assert score_30 == pytest.approx(0.50, abs=0.02)

    # 3. 60 days old -> ~0.25
    mem.updated_at = now - timedelta(days=60)
    score_60 = calculate_freshness_score(mem, reference_time=now, half_life_days=30.0, importance_damping=False)
    assert score_60 == pytest.approx(0.25, abs=0.02)


def test_freshness_high_importance_damping():
    now = datetime.utcnow()

    class CriticalMemory:
        importance = 10
        created_at = now - timedelta(days=500)
        updated_at = now - timedelta(days=500)

    class LowMemory:
        importance = 1
        created_at = now - timedelta(days=500)
        updated_at = now - timedelta(days=500)

    crit = CriticalMemory()
    low = LowMemory()

    # Without damping, both decay to nearly 0.0
    crit_raw = calculate_freshness_score(crit, reference_time=now, importance_damping=False)
    low_raw = calculate_freshness_score(low, reference_time=now, importance_damping=False)
    assert crit_raw < 0.001
    assert low_raw < 0.001

    # With damping, critical memory preserves its floor (0.30)
    crit_damped = calculate_freshness_score(crit, reference_time=now, importance_damping=True)
    low_damped = calculate_freshness_score(low, reference_time=now, importance_damping=True)

    assert crit_damped >= 0.30
    assert low_damped < 0.01
