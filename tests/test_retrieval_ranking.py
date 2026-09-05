from datetime import datetime, timedelta

import numpy as np

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

    normalized_semantic_score = (
        engine._normalize_semantic_score(
            semantic_score
        )
    )

    expected_score = (
        normalized_semantic_score
        + (0.10 * importance_score)
        + (0.10 * recency_score)
        + (0.20 * relationship_score)
    ) / 1.40

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
    
def test_final_score_ranks_relevant_memory_higher():
    engine = create_engine()

    highly_relevant = create_memory(
        relation="works_on",
        category="PROJECT",
        importance=10,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    less_relevant = create_memory(
        relation="likes",
        category="PERSONAL",
        importance=3,
        created_at=datetime.utcnow() - timedelta(days=60),
        updated_at=datetime.utcnow() - timedelta(days=60)
    )

    semantic_score = 0.5

    high_score = engine._calculate_final_score(
        semantic_score,
        highly_relevant,
        "PROJECT"
    )

    low_score = engine._calculate_final_score(
        semantic_score,
        less_relevant,
        "PROJECT"
    )

    assert high_score > low_score


def test_semantic_similarity_affects_final_ranking():
    engine = create_engine()

    memory = create_memory(
        relation="works_on",
        category="PROJECT",
        importance=5,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    low_semantic_score = engine._calculate_final_score(
        0.2,
        memory,
        "PROJECT"
    )

    high_semantic_score = engine._calculate_final_score(
        0.9,
        memory,
        "PROJECT"
    )

    assert high_semantic_score > low_semantic_score


def test_relationship_relevance_can_outweigh_equal_semantic_score():
    engine = create_engine()

    project_memory = create_memory(
        relation="works_on",
        category="PROJECT",
        importance=5,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    unrelated_memory = create_memory(
        relation="likes",
        category="PERSONAL",
        importance=5,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    semantic_score = 0.5

    project_score = engine._calculate_final_score(
        semantic_score,
        project_memory,
        "PROJECT"
    )

    unrelated_score = engine._calculate_final_score(
        semantic_score,
        unrelated_memory,
        "PROJECT"
    )

    assert project_score > unrelated_score


def test_recent_memory_ranks_higher_than_old_memory():
    engine = create_engine()

    recent_memory = create_memory(
        importance=5,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    old_memory = create_memory(
        importance=5,
        created_at=datetime.utcnow() - timedelta(days=60),
        updated_at=datetime.utcnow() - timedelta(days=60)
    )

    semantic_score = 0.5

    recent_score = engine._calculate_final_score(
        semantic_score,
        recent_memory,
        None
    )

    old_score = engine._calculate_final_score(
        semantic_score,
        old_memory,
        None
    )

    assert recent_score > old_score
    
class DummySearchEmbeddingEngine:
    def __init__(self, query_embedding):
        self.query_embedding = query_embedding

    def generate_embedding(self, text):
        return self.query_embedding


class DummySearchMemoryStore:
    def __init__(self, memories):
        self.memories = memories

    def get_all_memories(self):
        return self.memories

    def get_embedding(self, memory):
        return memory.embedding

    def close(self):
        pass


def create_search_memory(
    relation,
    category,
    importance,
    embedding,
    created_at=None,
    updated_at=None
):
    memory = create_memory(
        relation=relation,
        category=category,
        importance=importance,
        created_at=created_at,
        updated_at=updated_at
    )

    memory.embedding = embedding

    return memory


def test_search_returns_results_in_ranking_order():
    query_embedding = np.zeros(384, dtype=np.float32)
    query_embedding[0] = 1.0

    highly_similar = np.zeros(384, dtype=np.float32)
    highly_similar[0] = 1.0

    weakly_similar = np.zeros(384, dtype=np.float32)
    weakly_similar[1] = 1.0

    memory_1 = create_search_memory(
        relation="works_on",
        category="PROJECT",
        importance=10,
        embedding=weakly_similar
    )

    memory_2 = create_search_memory(
        relation="works_on",
        category="PROJECT",
        importance=10,
        embedding=highly_similar
    )

    store = DummySearchMemoryStore(
        [memory_1, memory_2]
    )

    engine = RetrievalEngine(
        memory_store=store,
        embedding_engine=DummySearchEmbeddingEngine(
            query_embedding
        )
    )

    results = engine.search(
        "What projects am I working on?"
    )

    assert len(results) == 2

    assert results[0]["memory"] is memory_2
    assert results[1]["memory"] is memory_1

    assert results[0]["score"] > results[1]["score"]
    
def test_search_respects_top_k():
    query_embedding = np.zeros(384, dtype=np.float32)
    query_embedding[0] = 1.0

    memories = []

    for _ in range(5):
        embedding = np.zeros(384, dtype=np.float32)
        embedding[0] = 1.0

        memories.append(
            create_search_memory(
                relation="works_on",
                category="PROJECT",
                importance=5,
                embedding=embedding
            )
        )

    store = DummySearchMemoryStore(memories)

    engine = RetrievalEngine(
        memory_store=store,
        embedding_engine=DummySearchEmbeddingEngine(
            query_embedding
        )
    )

    results = engine.search(
        "What projects am I working on?",
        top_k=2
    )

    assert len(results) == 2
    
def test_search_respects_min_score():
    query_embedding = np.zeros(384, dtype=np.float32)
    query_embedding[0] = 1.0

    good_embedding = np.zeros(384, dtype=np.float32)
    good_embedding[0] = 1.0

    poor_embedding = np.zeros(384, dtype=np.float32)
    poor_embedding[1] = 1.0

    good_memory = create_search_memory(
        relation="works_on",
        category="PROJECT",
        importance=10,
        embedding=good_embedding
    )

    poor_memory = create_search_memory(
        relation="likes",
        category="PERSONAL",
        importance=1,
        embedding=poor_embedding
    )

    store = DummySearchMemoryStore(
        [good_memory, poor_memory]
    )

    engine = RetrievalEngine(
        memory_store=store,
        embedding_engine=DummySearchEmbeddingEngine(
            query_embedding
        )
    )

    results = engine.search(
        "What projects am I working on?",
        min_score=0.95
    )

    assert len(results) == 1
    assert results[0]["memory"] is good_memory
    assert results[0]["score"] >= 1.0
    
def test_search_empty_query_returns_empty_list():
    store = DummySearchMemoryStore([])

    engine = RetrievalEngine(
        memory_store=store,
        embedding_engine=DummySearchEmbeddingEngine(
            np.zeros(384, dtype=np.float32)
        )
    )

    assert engine.search("") == []
    assert engine.search("   ") == []
    
def test_search_invalid_top_k_returns_empty_list():
    store = DummySearchMemoryStore([])

    engine = RetrievalEngine(
        memory_store=store,
        embedding_engine=DummySearchEmbeddingEngine(
            np.zeros(384, dtype=np.float32)
        )
    )

    assert engine.search(
        "What do I like?",
        top_k=0
    ) == []

    assert engine.search(
        "What do I like?",
        top_k=-1
    ) == []
    
    
def test_context_can_break_a_close_semantic_match():
    engine = create_engine()

    relevant_memory = create_memory(
        relation="works_on",
        category="PROJECT",
        importance=10,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    unrelated_memory = create_memory(
        relation="likes",
        category="PERSONAL",
        importance=5,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    semantic_score_relevant = 0.80
    semantic_score_unrelated = 0.82

    relevant_score = engine._calculate_final_score(
        semantic_score_relevant,
        relevant_memory,
        "PROJECT"
    )

    unrelated_score = engine._calculate_final_score(
        semantic_score_unrelated,
        unrelated_memory,
        "PROJECT"
    )

    assert relevant_score > unrelated_score
    
def test_irrelevant_memory_does_not_change_relevant_ranking():
    query_embedding = np.zeros(384, dtype=np.float32)
    query_embedding[0] = 1.0

    first_embedding = np.zeros(384, dtype=np.float32)
    first_embedding[0] = 0.95
    first_embedding[1] = 0.05

    second_embedding = np.zeros(384, dtype=np.float32)
    second_embedding[0] = 0.80
    second_embedding[1] = 0.60

    irrelevant_embedding = np.zeros(384, dtype=np.float32)
    irrelevant_embedding[2] = 1.0

    first_memory = create_search_memory(
        relation="works_on",
        category="PROJECT",
        importance=10,
        embedding=first_embedding
    )

    second_memory = create_search_memory(
        relation="works_on",
        category="PROJECT",
        importance=8,
        embedding=second_embedding
    )

    irrelevant_memory = create_search_memory(
        relation="likes",
        category="PERSONAL",
        importance=1,
        embedding=irrelevant_embedding
    )

    engine_without_irrelevant = RetrievalEngine(
        memory_store=DummySearchMemoryStore(
            [first_memory, second_memory]
        ),
        embedding_engine=DummySearchEmbeddingEngine(
            query_embedding
        )
    )

    engine_with_irrelevant = RetrievalEngine(
        memory_store=DummySearchMemoryStore(
            [
                first_memory,
                second_memory,
                irrelevant_memory
            ]
        ),
        embedding_engine=DummySearchEmbeddingEngine(
            query_embedding
        )
    )

    results_without = engine_without_irrelevant.search(
        "What projects am I working on?"
    )

    results_with = engine_with_irrelevant.search(
        "What projects am I working on?"
    )

    assert results_without[0]["memory"] is first_memory
    assert results_without[1]["memory"] is second_memory

    assert results_with[0]["memory"] is first_memory
    assert results_with[1]["memory"] is second_memory
    
def test_equal_score_memories_preserve_input_order():
    query_embedding = np.zeros(384, dtype=np.float32)
    query_embedding[0] = 1.0

    embedding = np.zeros(384, dtype=np.float32)
    embedding[0] = 1.0

    same_timestamp = datetime.utcnow()

    first_memory = create_search_memory(
        relation="works_on",
        category="PROJECT",
        importance=5,
        embedding=embedding,
        created_at=same_timestamp,
        updated_at=same_timestamp
    )

    second_memory = create_search_memory(
        relation="works_on",
        category="PROJECT",
        importance=5,
        embedding=embedding,
        created_at=same_timestamp,
        updated_at=same_timestamp
    )

    store = DummySearchMemoryStore(
        [first_memory, second_memory]
    )

    engine = RetrievalEngine(
        memory_store=store,
        embedding_engine=DummySearchEmbeddingEngine(
            query_embedding
        )
    )

    results = engine.search(
        "What projects am I working on?"
    )

    assert results[0]["memory"] is first_memory
    assert results[1]["memory"] is second_memory
    
def test_search_respects_semantic_threshold():
    query_embedding = np.zeros(384, dtype=np.float32)
    query_embedding[0] = 1.0

    relevant_embedding = np.zeros(384, dtype=np.float32)
    relevant_embedding[0] = 1.0

    weak_embedding = np.zeros(384, dtype=np.float32)
    weak_embedding[1] = 1.0

    relevant_memory = create_search_memory(
        relation="works_on",
        category="PROJECT",
        importance=5,
        embedding=relevant_embedding
    )

    weak_memory = create_search_memory(
        relation="likes",
        category="PERSONAL",
        importance=5,
        embedding=weak_embedding
    )

    store = DummySearchMemoryStore(
        [relevant_memory, weak_memory]
    )

    engine = RetrievalEngine(
        memory_store=store,
        embedding_engine=DummySearchEmbeddingEngine(
            query_embedding
        )
    )

    results = engine.search(
        "What projects am I working on?",
        semantic_threshold=0.5
    )

    assert len(results) == 1
    assert results[0]["memory"] is relevant_memory


def test_search_semantic_threshold_does_not_replace_min_score():
    query_embedding = np.zeros(384, dtype=np.float32)
    query_embedding[0] = 1.0

    embedding = np.zeros(384, dtype=np.float32)
    embedding[0] = 1.0

    memory = create_search_memory(
        relation="works_on",
        category="PROJECT",
        importance=5,
        embedding=embedding
    )

    store = DummySearchMemoryStore(
        [memory]
    )

    engine = RetrievalEngine(
        memory_store=store,
        embedding_engine=DummySearchEmbeddingEngine(
            query_embedding
        )
    )

    results = engine.search(
        "What projects am I working on?",
        semantic_threshold=0.5,
        min_score=2.0
    )

    assert results == []
    
def test_search_top_k_returns_only_best_memory():
    query_embedding = np.zeros(384, dtype=np.float32)
    query_embedding[0] = 1.0

    best_embedding = np.zeros(384, dtype=np.float32)
    best_embedding[0] = 1.0

    weaker_embedding = np.zeros(384, dtype=np.float32)
    weaker_embedding[0] = 0.8
    weaker_embedding[1] = 0.6

    best_memory = create_search_memory(
        relation="works_on",
        category="PROJECT",
        importance=5,
        embedding=best_embedding
    )

    weaker_memory = create_search_memory(
        relation="likes",
        category="PREFERENCE",
        importance=5,
        embedding=weaker_embedding
    )

    store = DummySearchMemoryStore(
        [best_memory, weaker_memory]
    )

    engine = RetrievalEngine(
        memory_store=store,
        embedding_engine=DummySearchEmbeddingEngine(
            query_embedding
        )
    )

    results = engine.search(
        "What projects am I working on?",
        top_k=1
    )

    assert len(results) == 1
    assert results[0]["memory"] is best_memory


def test_search_top_k_returns_best_n_memories():
    query_embedding = np.zeros(384, dtype=np.float32)
    query_embedding[0] = 1.0

    embeddings = []

    for value in [1.0, 0.9, 0.8, 0.7]:
        embedding = np.zeros(384, dtype=np.float32)
        embedding[0] = value
        embedding[1] = np.sqrt(
            max(0.0, 1.0 - value ** 2)
        )
        embeddings.append(embedding)

    memories = [
        create_search_memory(
            relation="works_on",
            category="PROJECT",
            importance=5,
            embedding=embedding
        )
        for embedding in embeddings
    ]

    store = DummySearchMemoryStore(memories)

    engine = RetrievalEngine(
        memory_store=store,
        embedding_engine=DummySearchEmbeddingEngine(
            query_embedding
        )
    )

    results = engine.search(
        "What projects am I working on?",
        top_k=3
    )

    assert len(results) == 3

    assert results[0]["memory"] is memories[0]
    assert results[1]["memory"] is memories[1]
    assert results[2]["memory"] is memories[2]


def test_search_top_k_larger_than_available_returns_all():
    query_embedding = np.zeros(384, dtype=np.float32)
    query_embedding[0] = 1.0

    embedding = np.zeros(384, dtype=np.float32)
    embedding[0] = 1.0

    memory = create_search_memory(
        relation="works_on",
        category="PROJECT",
        importance=5,
        embedding=embedding
    )

    store = DummySearchMemoryStore(
        [memory]
    )

    engine = RetrievalEngine(
        memory_store=store,
        embedding_engine=DummySearchEmbeddingEngine(
            query_embedding
        )
    )

    results = engine.search(
        "What projects am I working on?",
        top_k=10
    )

    assert len(results) == 1
    assert results[0]["memory"] is memory
    
def test_search_top_k_zero_returns_empty():
    query_embedding = np.zeros(384, dtype=np.float32)
    query_embedding[0] = 1.0

    embedding = np.zeros(384, dtype=np.float32)
    embedding[0] = 1.0

    memory = create_search_memory(
        relation="works_on",
        category="PROJECT",
        importance=5,
        embedding=embedding
    )

    store = DummySearchMemoryStore([memory])

    engine = RetrievalEngine(
        memory_store=store,
        embedding_engine=DummySearchEmbeddingEngine(
            query_embedding
        )
    )

    results = engine.search(
        "What projects am I working on?",
        top_k=0
    )

    assert results == []


def test_search_negative_top_k_returns_empty():
    query_embedding = np.zeros(384, dtype=np.float32)
    query_embedding[0] = 1.0

    embedding = np.zeros(384, dtype=np.float32)
    embedding[0] = 1.0

    memory = create_search_memory(
        relation="works_on",
        category="PROJECT",
        importance=5,
        embedding=embedding
    )

    store = DummySearchMemoryStore([memory])

    engine = RetrievalEngine(
        memory_store=store,
        embedding_engine=DummySearchEmbeddingEngine(
            query_embedding
        )
    )

    results = engine.search(
        "What projects am I working on?",
        top_k=-1
    )

    assert results == []


def test_semantic_threshold_is_applied_before_top_k():
    query_embedding = np.zeros(384, dtype=np.float32)
    query_embedding[0] = 1.0

    strong_embedding = np.zeros(384, dtype=np.float32)
    strong_embedding[0] = 1.0

    weak_embedding = np.zeros(384, dtype=np.float32)
    weak_embedding[1] = 1.0

    strong_memory = create_search_memory(
        relation="works_on",
        category="PROJECT",
        importance=5,
        embedding=strong_embedding
    )

    weak_memory = create_search_memory(
        relation="likes",
        category="PREFERENCE",
        importance=10,
        embedding=weak_embedding
    )

    store = DummySearchMemoryStore(
        [strong_memory, weak_memory]
    )

    engine = RetrievalEngine(
        memory_store=store,
        embedding_engine=DummySearchEmbeddingEngine(
            query_embedding
        )
    )

    results = engine.search(
        "What projects am I working on?",
        top_k=1,
        semantic_threshold=0.5
    )

    assert len(results) == 1
    assert results[0]["memory"] is strong_memory


def test_min_score_is_applied_before_top_k():
    query_embedding = np.zeros(384, dtype=np.float32)
    query_embedding[0] = 1.0

    embedding = np.zeros(384, dtype=np.float32)
    embedding[0] = 1.0

    high_score_memory = create_search_memory(
        relation="works_on",
        category="PROJECT",
        importance=10,
        embedding=embedding
    )

    low_score_memory = create_search_memory(
        relation="likes",
        category="PERSONAL",
        importance=1,
        embedding=embedding
    )

    store = DummySearchMemoryStore(
        [high_score_memory, low_score_memory]
    )

    engine = RetrievalEngine(
        memory_store=store,
        embedding_engine=DummySearchEmbeddingEngine(
            query_embedding
        )
    )

    results = engine.search(
        "What projects am I working on?",
        top_k=1,
        min_score=0.85
    )

    assert len(results) == 1
    assert results[0]["memory"] is high_score_memory


def test_top_k_is_applied_after_score_sorting():
    query_embedding = np.zeros(384, dtype=np.float32)
    query_embedding[0] = 1.0

    embedding = np.zeros(384, dtype=np.float32)
    embedding[0] = 1.0

    weaker_memory = create_search_memory(
        relation="likes",
        category="PERSONAL",
        importance=1,
        embedding=embedding
    )

    stronger_memory = create_search_memory(
        relation="works_on",
        category="PROJECT",
        importance=10,
        embedding=embedding
    )

    store = DummySearchMemoryStore(
        [weaker_memory, stronger_memory]
    )

    engine = RetrievalEngine(
        memory_store=store,
        embedding_engine=DummySearchEmbeddingEngine(
            query_embedding
        )
    )

    results = engine.search(
        "What projects am I working on?",
        top_k=1
    )

    assert len(results) == 1
    assert results[0]["memory"] is stronger_memory
    
def test_semantic_score_normalization_maps_negative_one_to_zero():
    engine = create_engine()

    normalized = engine._normalize_semantic_score(-1.0)

    assert normalized == 0.0


def test_semantic_score_normalization_maps_zero_to_half():
    engine = create_engine()

    normalized = engine._normalize_semantic_score(0.0)

    assert normalized == 0.5


def test_semantic_score_normalization_maps_one_to_one():
    engine = create_engine()

    normalized = engine._normalize_semantic_score(1.0)

    assert normalized == 1.0


def test_semantic_score_normalization_clamps_out_of_range_values():
    engine = create_engine()

    too_low = engine._normalize_semantic_score(-2.0)
    too_high = engine._normalize_semantic_score(2.0)

    assert too_low == 0.0
    assert too_high == 1.0
    
def test_final_score_uses_normalized_semantic_score():
    engine = create_engine()

    memory = create_memory(
        importance=1
    )

    score = engine._calculate_final_score(
        semantic_score=-1.0,
        memory=memory,
        query_intent=None
    )

    assert 0.0 <= score <= 1.0


def test_final_score_with_perfect_semantic_similarity_is_highest():
    engine = create_engine()

    memory = create_memory(
        importance=10
    )

    score = engine._calculate_final_score(
        semantic_score=1.0,
        memory=memory,
        query_intent=None
    )

    assert score <= 1.0
    assert score > 0.7