from datetime import datetime, timedelta

from app.database.models import Memory
from app.retrieval.retrieval_engine import RetrievalEngine


class DummyMemoryStore:
    def close(self):
        pass


class DummyEmbeddingEngine:
    def generate_embedding(self, text):
        return None


def create_engine():
    return RetrievalEngine(
        memory_store=DummyMemoryStore(),
        embedding_engine=DummyEmbeddingEngine()
    )


def create_memory(
    relation="likes",
    category="PREFERENCE",
    importance=5,
    created_at=None,
    updated_at=None
):
    return Memory(
        subject="User",
        relation=relation,
        value="Python",
        category=category,
        importance=importance,
        active=True,
        created_at=created_at or datetime.utcnow(),
        updated_at=updated_at or datetime.utcnow()
    )


def test_importance_score():
    engine = create_engine()

    low = create_memory(importance=1)
    medium = create_memory(importance=5)
    high = create_memory(importance=10)

    assert engine._calculate_importance_score(low) == 0.0
    assert engine._calculate_importance_score(medium) == 4 / 9
    assert engine._calculate_importance_score(high) == 1.0


def test_importance_score_handles_missing_value():
    engine = create_engine()

    memory = create_memory(importance=None)

    score = engine._calculate_importance_score(memory)

    assert score == 4 / 9


def test_importance_score_clamps_values():
    engine = create_engine()

    too_low = create_memory(importance=-10)
    too_high = create_memory(importance=100)

    assert engine._calculate_importance_score(too_low) == 0.0
    assert engine._calculate_importance_score(too_high) == 1.0


def test_recency_score_recent_memory_is_higher():
    engine = create_engine()

    recent = create_memory(
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    old = create_memory(
        created_at=datetime.utcnow() - timedelta(days=60),
        updated_at=datetime.utcnow() - timedelta(days=60)
    )

    recent_score = engine._calculate_recency_score(recent)
    old_score = engine._calculate_recency_score(old)

    assert recent_score > old_score
    assert recent_score > 0.9
    assert old_score < 0.3


def test_recency_score_missing_timestamps():
    engine = create_engine()

    memory = create_memory()
    memory.created_at = None
    memory.updated_at = None

    assert engine._calculate_recency_score(memory) == 0.0


def test_relationship_score_direct_match():
    engine = create_engine()

    memory = create_memory(
        relation="works_on",
        category="PROJECT"
    )

    score = engine._calculate_relationship_score(
        memory,
        "PROJECT"
    )

    assert score == 1.0


def test_relationship_score_category_match():
    engine = create_engine()

    memory = create_memory(
        relation="likes",
        category="PROJECT"
    )

    score = engine._calculate_relationship_score(
        memory,
        "PROJECT"
    )

    assert score == 0.5


def test_relationship_score_no_match():
    engine = create_engine()

    memory = create_memory(
        relation="likes",
        category="PERSONAL"
    )

    score = engine._calculate_relationship_score(
        memory,
        "PROJECT"
    )

    assert score == 0.0


def test_relationship_score_without_intent():
    engine = create_engine()

    memory = create_memory(
        relation="works_on",
        category="PROJECT"
    )

    score = engine._calculate_relationship_score(
        memory,
        None
    )

    assert score == 0.0


def test_final_score_uses_all_ranking_signals():
    engine = create_engine()

    memory = create_memory(
        relation="works_on",
        category="PROJECT",
        importance=10,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    semantic_score = 0.5

    final_score = engine._calculate_final_score(
        semantic_score,
        memory,
        "PROJECT"
    )

    importance_score = 1.0
    relationship_score = 1.0

    recency_score = engine._calculate_recency_score(memory)

    expected_score = (
        semantic_score
        + (0.10 * importance_score)
        + (0.10 * recency_score)
        + (0.20 * relationship_score)
    )

    assert abs(final_score - expected_score) < 1e-6


def test_query_intent_detection():
    engine = create_engine()

    assert engine._detect_query_intent(
        "What projects am I working on?"
    ) == "PROJECT"

    assert engine._detect_query_intent(
        "What do I study?"
    ) == "EDUCATION"

    assert engine._detect_query_intent(
        "Where do I live?"
    ) == "LOCATION"

    assert engine._detect_query_intent(
        "What do I like?"
    ) == "PREFERENCE"

    assert engine._detect_query_intent(
        "What skills do I know?"
    ) == "SKILL"

    assert engine._detect_query_intent(
        "What do I want to learn?"
    ) == "GOAL"

    assert engine._detect_query_intent(
        "Tell me something random"
    ) is None
    
def test_higher_importance_wins_when_semantic_scores_are_equal():
    engine = create_engine()

    high_importance = create_memory(
        relation="likes",
        category="PREFERENCE",
        importance=10
    )

    low_importance = create_memory(
        relation="likes",
        category="PREFERENCE",
        importance=2
    )

    high_score = engine._calculate_final_score(
        0.80,
        high_importance,
        None
    )

    low_score = engine._calculate_final_score(
        0.80,
        low_importance,
        None
    )

    assert high_score > low_score


def test_recent_memory_wins_when_semantic_scores_are_equal():
    engine = create_engine()

    recent = create_memory(
        importance=5,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    old = create_memory(
        importance=5,
        created_at=datetime.utcnow() - timedelta(days=60),
        updated_at=datetime.utcnow() - timedelta(days=60)
    )

    recent_score = engine._calculate_final_score(
        0.80,
        recent,
        None
    )

    old_score = engine._calculate_final_score(
        0.80,
        old,
        None
    )

    assert recent_score > old_score


def test_relationship_relevance_wins_when_semantic_scores_are_equal():
    engine = create_engine()

    relevant = create_memory(
        relation="works_on",
        category="PROJECT",
        importance=5
    )

    irrelevant = create_memory(
        relation="likes",
        category="PERSONAL",
        importance=5
    )

    relevant_score = engine._calculate_final_score(
        0.80,
        relevant,
        "PROJECT"
    )

    irrelevant_score = engine._calculate_final_score(
        0.80,
        irrelevant,
        "PROJECT"
    )

    assert relevant_score > irrelevant_score


def test_all_ranking_signals_favor_relevant_memory():
    engine = create_engine()

    strong_memory = create_memory(
        relation="works_on",
        category="PROJECT",
        importance=10,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    weak_memory = create_memory(
        relation="likes",
        category="PERSONAL",
        importance=2,
        created_at=datetime.utcnow() - timedelta(days=60),
        updated_at=datetime.utcnow() - timedelta(days=60)
    )

    strong_score = engine._calculate_final_score(
        0.80,
        strong_memory,
        "PROJECT"
    )

    weak_score = engine._calculate_final_score(
        0.80,
        weak_memory,
        "PROJECT"
    )

    assert strong_score > weak_score