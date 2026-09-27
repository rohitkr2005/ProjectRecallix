from enum import Enum
import re

from app.config import settings
from app.llm.llm_engine import LLMEngine, UNSUPPORTED_RESPONSE
from app.memory.forgetting import MemoryForgetter
from app.memory.graph import MemoryGraph
from app.memory.importance_learning import ImportanceLearner
from app.memory.memory_extractor import MemoryExtractor
from app.memory.memory_store import MemoryStore
from app.memory.temporal import detect_query_temporal_intent, detect_temporal_state
from app.retrieval.retrieval_engine import RetrievalEngine
from app.utils.logging import (
    get_logger,
    log_error_event,
    log_llm_event,
    log_memory_event,
    log_retrieval_event,
)
from app.utils.metrics import LatencyTracker
from app.utils.validators import validate_memory_content, validate_user_query


class InputType(str, Enum):
    STATEMENT = "STATEMENT"
    QUESTION = "QUESTION"
    BOTH = "BOTH"
    NEITHER = "NEITHER"


class AssistantEngine:
    """Main Recallix memory-aware assistant pipeline."""

    UNSUPPORTED_RESPONSE = UNSUPPORTED_RESPONSE
    InputType = InputType

    QUESTION_PATTERNS = [
        r"\?",
        r"^\s*(?:what|where|who|when|why|how|which)\b",
        r"^\s*(?:can you tell me|tell me|could you tell me)\b",
        r"^\s*(?:do i|am i|have i|did i|will i|was i)\b",
        r"^\s*(?:is there|are there|do you remember|do you know)\b",
        r"\b(?:what is|what are|where do|where am|who is|which project|which skill)\b",
    ]

    def __init__(
        self,
        retrieval_engine=None,
        llm_engine=None,
        memory_store=None,
        memory_extractor=None,
        importance_learner=None,
        memory_graph=None,
        memory_forgetter=None,
    ):
        self.logger = get_logger("recallix.assistant")
        self.retrieval_engine = retrieval_engine or RetrievalEngine(memory_store=memory_store)
        self.llm_engine = llm_engine or LLMEngine()
        self.memory_store = (
            memory_store
            or getattr(self.retrieval_engine, "memory_store", None)
            or MemoryStore()
        )
        self.memory_extractor = memory_extractor or MemoryExtractor()
        self.embedding_engine = getattr(self.retrieval_engine, "embedding_engine", None)
        self.importance_learner = importance_learner or ImportanceLearner(
            session=getattr(self.memory_store, "session", None)
        )
        self.memory_graph = memory_graph or MemoryGraph(
            session=getattr(self.memory_store, "session", None)
        )
        self.memory_forgetter = memory_forgetter or MemoryForgetter(
            session=getattr(self.memory_store, "session", None),
            importance_learner=self.importance_learner,
        )

    def is_available(self):
        return self.llm_engine.is_available()

    def classify_input(self, user_message: str) -> InputType:
        """
        Classify input message into:
        - STATEMENT: new information/memory asserted
        - QUESTION: inquiry/question asked
        - BOTH: both new information and question present
        - NEITHER: casual chit-chat or greeting
        """
        if not user_message or not user_message.strip():
            return InputType.NEITHER

        text = user_message.strip()
        has_question = False
        for pattern in self.QUESTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                has_question = True
                break

        extracted = []
        if callable(getattr(self.memory_extractor, "extract", None)):
            try:
                extracted = self.memory_extractor.extract(text)
            except Exception:
                extracted = []

        has_memories = bool(extracted)

        if has_memories and has_question:
            return InputType.BOTH
        elif has_memories and not has_question:
            return InputType.STATEMENT
        elif not has_memories and has_question:
            return InputType.QUESTION
        else:
            return InputType.NEITHER

    def save_extracted_memories(self, extracted_memories, temporal_state=None, user_id=None):
        """Save extracted memories into MemoryStore using lifecycle semantics."""
        saved = []
        if not extracted_memories:
            return saved

        for mem in extracted_memories:
            subject = getattr(mem, "subject", "User")
            relation = getattr(mem, "relation", "")
            value = getattr(mem, "value", "")
            category = getattr(mem, "category", "GENERAL")
            importance = getattr(mem, "importance", 5)
            temp_state = getattr(mem, "temporal_state", None) or temporal_state or "PRESENT"

            # 11.5 Memory Safety: Validate & sanitize memory content
            try:
                val = validate_memory_content(
                    subject=subject,
                    relation=relation,
                    value=value,
                    category=category,
                )
                subject, relation, value, category = val["subject"], val["relation"], val["value"], val["category"]
            except Exception as e:
                log_error_event(self.logger, "memory_validation_failed", error=e)
                continue

            embedding = None
            if self.embedding_engine and hasattr(self.embedding_engine, "generate_memory_embedding"):
                try:
                    embedding = self.embedding_engine.generate_memory_embedding(
                        subject=subject,
                        relation=relation,
                        value=value,
                        category=category,
                    )
                except Exception:
                    embedding = None

            save_kwargs = {
                "subject": subject,
                "relation": relation,
                "value": value,
                "category": category,
                "importance": importance,
                "embedding": embedding,
                "temporal_state": temp_state,
            }
            if user_id is not None:
                save_kwargs["user_id"] = user_id
            try:
                memory, status, metadata = self.memory_store.save_memory_with_semantics(**save_kwargs)
            except TypeError:
                save_kwargs.pop("user_id", None)
                memory, status, metadata = self.memory_store.save_memory_with_semantics(**save_kwargs)

            # Update memory graph
            if hasattr(self, "memory_graph") and self.memory_graph:
                self.memory_graph.add_edge(
                    source=subject,
                    relation=relation,
                    target=value,
                    category=category,
                    memory_id=getattr(memory, "id", None),
                )

            saved.append({
                "memory": memory,
                "status": status,
                "metadata": metadata,
                "subject": subject,
                "relation": relation,
                "value": value,
                "category": category,
                "importance": importance,
                "temporal_state": temp_state,
            })
        return saved

    def _analyze_intent(self, user_message):
        """Use richer query analysis when available, with a safe legacy fallback."""
        analyzer = getattr(self.retrieval_engine, "analyze_query_intent", None)
        if callable(analyzer):
            res = analyzer(user_message)
            if isinstance(res, dict) and "intent" in res:
                return res

        detector = getattr(self.retrieval_engine, "_detect_query_intent", None)
        intent = detector(user_message) if callable(detector) else None
        return {
            "intent": intent,
            "confidence": 1.0 if intent else 0.0,
            "strict": bool(intent),
            "normalized_query": user_message.strip().lower(),
            "scores": {},
        }

    def _detect_intent(self, user_message):
        return self._analyze_intent(user_message)["intent"]

    def _filter_memories(self, memories, intent, min_relevance=0.25, strict_intent=True):
        """Filter active, relevant memories while allowing semantic fallback."""
        if not memories:
            return []

        relevant_memories = []
        intent_relations = {
            "PROJECT": {"works_on"},
            "EDUCATION": {"studies", "studies_at"},
            "LOCATION": {"lives_in", "current_city"},
            "PREFERENCE": {"likes"},
            "SKILL": {"knows"},
            "GOAL": {"wants_to_learn", "wants_to_become", "wants_to_build"},
        }

        for item in memories:
            if not isinstance(item, dict):
                continue
            memory = item.get("memory")
            if memory is None or getattr(memory, "active", True) is not True:
                continue

            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            if score < min_relevance:
                continue

            if intent is None or not strict_intent:
                relevant_memories.append(item)
                continue

            category = getattr(memory, "category", None)
            relation = getattr(memory, "relation", None)
            if category == intent or relation in intent_relations.get(intent, set()):
                relevant_memories.append(item)

        relevant_memories.sort(
            key=lambda x: float(x.get("score", 0.0)),
            reverse=True,
        )

        return relevant_memories

    def _has_supported_memories(self, memories):
        if not memories:
            return False
        for item in memories:
            if not isinstance(item, dict):
                continue
            memory = item.get("memory")
            if memory is None or getattr(memory, "active", True) is not True:
                continue
            value = getattr(memory, "value", None)
            if value is not None and str(value).strip():
                return True
        return False

    def _build_retrieval_explanations(self, memories):
        """Return explainability metadata without exposing memory internals."""
        explanations = []
        for item in memories or []:
            if not isinstance(item, dict):
                continue
            explanation = item.get("explanation")
            if not isinstance(explanation, dict):
                continue
            explanations.append(dict(explanation))
        return explanations

    def _build_user_facing_explanations(self, memories):
        """
        Build human-readable explanations answering 'Why was this memory used?'.
        Does not leak database IDs or raw embedding vectors.
        """
        user_explanations = []
        for item in memories or []:
            if not isinstance(item, dict):
                continue
            memory = item.get("memory")
            if memory is None:
                continue
            subject = getattr(memory, "subject", "User")
            relation = getattr(memory, "relation", "").replace("_", " ")
            value = getattr(memory, "value", "")
            explanation = item.get("explanation", {})

            reasons = explanation.get("reasons", [])
            if reasons:
                reason_str = ", ".join(reasons)
            else:
                score = item.get("score", 0.0)
                reason_str = f"relevance score of {score:.2f}"

            user_explanations.append(
                f"Memory '{subject} {relation} {value}' was used because: {reason_str}."
            )
        return user_explanations

    def _is_assistant_self_query(self, text: str) -> bool:
        """Detect questions asking about Recallix itself, its purpose, or capabilities."""
        patterns = [
            r"\bwho\s+are\s+you\b",
            r"\bwhat\s+are\s+you\b",
            r"\bwhat\s+is\s+your\s+name\b",
            r"\bwhat\s+can\s+you\s+do\b",
            r"\bhow\s+do\s+you\s+work\b",
            r"\bwhat\s+is\s+recallix\b",
            r"\btell\s+me\s+about\s+yourself\b",
            r"\bintroduce\s+yourself\b",
        ]
        t = text.strip().lower()
        return any(re.search(p, t) for p in patterns)

    def _is_owner_identity_query(self, text: str) -> bool:
        """Detect queries asking about who the user is or whether the assistant knows them."""
        patterns = [
            r"\bwho\s+am\s+i\b",
            r"\bdo\s+you\s+know\s+me\b",
            r"\bdo\s+you\s+remember\s+me\b",
            r"\bwhat\s+is\s+my\s+name\b",
            r"\bwho\s+i\s+am\b",
        ]
        t = text.strip().lower()
        return any(re.search(p, t) for p in patterns)

    def _is_personal_memory_query(self, text: str) -> bool:
        """Detect queries specifically asking for personal/private facts about the user."""
        patterns = [
            r"\bmy\s+[a-zA-Z]",
            r"\bdo\s+i\b",
            r"\bam\s+i\b",
            r"\bdid\s+i\b",
            r"\bhave\s+i\b",
            r"\bwhere\s+do\s+i\b",
            r"\bwhat\s+do\s+i\b",
            r"\bwho\s+is\s+my\b",
            r"\btell\s+me\s+about\s+my\b",
            r"\bremember\s+about\s+me\b",
            r"\bwhat\s+are\s+my\b",
            r"\bwhich\s+is\s+my\b",
        ]
        t = text.strip().lower()
        return any(re.search(p, t) for p in patterns)

    def _build_owner_identity_response(self, user_id=None, newly_saved=None):
        """Construct warm, personalized response when user asks 'who am I' or 'do you know me'."""
        name = None
        if newly_saved:
            for s in newly_saved:
                if s.get("relation") == "name":
                    name = s.get("value")
                    break

        if not name:
            session = getattr(self.memory_store, "session", None)
            if session:
                try:
                    from app.database import Memory
                    q = session.query(Memory).filter(Memory.active.is_(True))
                    if user_id is not None:
                        q = q.filter(Memory.user_id == user_id)
                    all_mems = q.all()
                    for m in all_mems:
                        if getattr(m, "relation", "") == "name":
                            name = getattr(m, "value", None)
                            break
                except Exception:
                    pass

        if not name and self.memory_store:
            for fetcher_name in ("get_all_memories", "list_memories"):
                fetcher = getattr(self.memory_store, fetcher_name, None)
                if callable(fetcher):
                    try:
                        mems = fetcher(user_id=user_id) if "user_id" in fetcher.__code__.co_varnames else fetcher()
                        for m in mems or []:
                            if getattr(m, "relation", "") == "name" and getattr(m, "active", True):
                                name = getattr(m, "value", None)
                                break
                        if name:
                            break
                    except Exception:
                        pass

        if name:
            return (
                f"Yes, of course! You are {name}. I remember who you are. "
                "I'm keeping track of your projects, skills, preferences, and notes so you never lose them. "
                "How can I help you today?"
            )
        else:
            return (
                "Yes, I know you as the owner of this account! However, you haven't told me your name yet. "
                "What should I call you, and what are you working on?"
            )

    def respond(
        self,
        user_message,
        top_k=5,
        min_score=0.0,
        semantic_threshold=0.0,
        temperature=0.2,
        max_tokens=500,
        memory_relevance=0.25,
        include_explanations=False,
        fallback_to_extractive=True,
        user_id=None,
        conversation_history=None,
    ):
        user_message = validate_user_query(user_message)
        tracker = LatencyTracker()

        input_type = self.classify_input(user_message)
        saved_records = []

        temporal_state = detect_temporal_state(user_message)
        # 9.2 / 9.3: Handle statement extraction for STATEMENT or BOTH
        if input_type in (InputType.STATEMENT, InputType.BOTH):
            with tracker.timer("memory_ms"):
                try:
                    extracted = self.memory_extractor.extract(user_message)
                    if extracted:
                        saved_records = self.save_extracted_memories(
                            extracted,
                            temporal_state=temporal_state,
                            user_id=user_id,
                        )
                        for r in saved_records:
                            log_memory_event(
                                self.logger,
                                "saved",
                                memory_id=getattr(r.get("memory"), "id", None),
                                subject=r.get("subject"),
                                relation=r.get("relation"),
                                status=r.get("status"),
                            )
                except Exception as e:
                    log_error_event(self.logger, "memory_extraction_failed", error=e)
                    saved_records = []

        # 0. Check for Assistant Self-Awareness Query ("who are you", "what is recallix")
        if self._is_assistant_self_query(user_message):
            reply = (
                "I am Recallix, your personal AI Second Brain and dedicated assistant! "
                "I'm designed to help you remember everything about your life, projects, skills, "
                "preferences, ideas, and notes across all our conversations. "
                "Unlike other assistants that forget who you are, I store your personal context and recall it "
                "whenever you need it. You can chat with me, ask me questions, or tell me anything you'd like me to remember. "
                "How can I help you today?"
            )
            result = {
                "response": reply,
                "input_type": input_type,
                "extracted_memories": saved_records,
                "memories": [],
                "retrieved_memories": [],
                "intent": None,
                "intent_confidence": 1.0,
                "intent_strict": False,
                "supported": True,
                "grounded": True,
                "grounding_details": {
                    "grounded": True,
                    "supported_values": [],
                    "grounding_score": 1.0,
                    "details": "Assistant self-awareness response.",
                },
                "llm_status": "assistant_identity",
            }
            if include_explanations:
                result["retrieval_explanations"] = []
                result["explanations"] = []
                result["why_used"] = []
            if settings.enable_metrics:
                result["performance_metrics"] = tracker.get_metrics()
            return result

        # 0.5 Check for Owner Identity Query ("do you know me", "who am I")
        if self._is_owner_identity_query(user_message):
            saved_name = None
            if saved_records:
                for r in saved_records:
                    if r.get("relation") == "name":
                        saved_name = r.get("value")
                        break

            if saved_name:
                reply = (
                    f"Hello {saved_name}! Nice to meet you. I've recorded your name in my memory. "
                    "Since this is our first chat, I don't have other notes or projects saved for you yet, "
                    "but tell me what you're working on, what you like, or anything you'd like me to keep in mind, "
                    "and I'll remember everything!"
                )
            else:
                reply = self._build_owner_identity_response(user_id=user_id, newly_saved=saved_records)

            result = {
                "response": reply,
                "input_type": input_type,
                "extracted_memories": saved_records,
                "memories": [],
                "retrieved_memories": [],
                "intent": None,
                "intent_confidence": 1.0,
                "intent_strict": False,
                "supported": True,
                "grounded": True,
                "grounding_details": {
                    "grounded": True,
                    "supported_values": [saved_name] if saved_name else [],
                    "grounding_score": 1.0,
                    "details": "Owner identity recognition.",
                },
                "llm_status": "owner_identity",
            }
            if include_explanations:
                result["retrieval_explanations"] = []
                result["explanations"] = []
                result["why_used"] = []
            if settings.enable_metrics:
                result["performance_metrics"] = tracker.get_metrics()
            return result

        # 1. Pure STATEMENT handling
        if input_type == InputType.STATEMENT:
            if saved_records:
                name_rec = next((r for r in saved_records if r.get("relation") == "name"), None)
                if name_rec:
                    response_text = f"Nice to meet you, {name_rec['value']}! I've recorded your name in my memory. What are you working on today?"
                elif len(saved_records) == 1:
                    rec = saved_records[0]
                    rel = rec["relation"].replace("_", " ")
                    is_updated = (
                        rec.get("status") == "updated"
                        or bool(rec.get("metadata", {}).get("superseded_memory_id"))
                    )
                    verb = "updated" if is_updated else "remembered"
                    response_text = f"Got it. I've {verb} that you {rel} {rec['value']}."
                else:
                    updated_count = sum(
                        1 for r in saved_records
                        if r.get("status") == "updated" or bool(r.get("metadata", {}).get("superseded_memory_id"))
                    )
                    if updated_count == len(saved_records):
                        response_text = f"Got it. I've updated these {len(saved_records)} facts."
                    else:
                        response_text = f"Got it. I've recorded these {len(saved_records)} facts."
            else:
                response_text = "Got it. Thanks for sharing!"

            result = {
                "response": response_text,
                "input_type": input_type,
                "extracted_memories": saved_records,
                "memories": [],
                "retrieved_memories": [],
                "intent": None,
                "intent_confidence": 0.0,
                "intent_strict": False,
                "supported": True,
                "grounded": True,
                "grounding_details": {
                    "grounded": True,
                    "supported_values": [r["value"] for r in saved_records],
                    "grounding_score": 1.0,
                    "details": "Stored new memory statements.",
                },
                "llm_status": "skipped_statement",
            }
            if include_explanations:
                result["retrieval_explanations"] = []
                result["explanations"] = []
                result["why_used"] = []
            if settings.enable_metrics:
                result["performance_metrics"] = tracker.get_metrics()
            return result

        # 2. Pure NEITHER (chit-chat / greeting) handling
        if input_type == InputType.NEITHER:
            with tracker.timer("chit_chat_ms"):
                msg_lower = user_message.lower()
                if any(w in msg_lower for w in ["hi", "hello", "hey"]):
                    reply = "Hello! How can I help you today?"
                elif any(w in msg_lower for w in ["thank", "thanks"]):
                    reply = "You're welcome!"
                elif any(w in msg_lower for w in ["bye", "goodbye"]):
                    reply = "Goodbye! Let me know whenever you need anything."
                else:
                    reply = "I'm here to answer your questions and manage your memories."

            result = {
                "response": reply,
                "input_type": input_type,
                "extracted_memories": [],
                "memories": [],
                "retrieved_memories": [],
                "intent": None,
                "intent_confidence": 0.0,
                "intent_strict": False,
                "supported": True,
                "grounded": True,
                "grounding_details": {
                    "grounded": True,
                    "supported_values": [],
                    "grounding_score": 1.0,
                    "details": "Conversational reply.",
                },
                "llm_status": "skipped_chit_chat",
            }
            if include_explanations:
                result["retrieval_explanations"] = []
                result["explanations"] = []
                result["why_used"] = []
            if settings.enable_metrics:
                result["performance_metrics"] = tracker.get_metrics()
            return result

        # 3. QUESTION or BOTH handling
        with tracker.timer("retrieval_ms"):
            search_kwargs = {
                "query": user_message,
                "top_k": top_k,
                "min_score": min_score,
                "semantic_threshold": semantic_threshold,
            }
            if user_id is not None:
                search_kwargs["user_id"] = user_id
            try:
                retrieved_memories = self.retrieval_engine.search(**search_kwargs)
            except TypeError:
                search_kwargs.pop("user_id", None)
                retrieved_memories = self.retrieval_engine.search(**search_kwargs)

        top_score = (
            float(retrieved_memories[0].get("score", 0.0))
            if retrieved_memories and isinstance(retrieved_memories[0], dict)
            else 0.0
        )
        log_retrieval_event(
            self.logger,
            query=user_message,
            count=len(retrieved_memories),
            top_score=top_score,
        )

        intent_analysis = self._analyze_intent(user_message)
        intent = intent_analysis["intent"]
        strict_intent = intent_analysis["strict"]

        relevant_memories = self._filter_memories(
            memories=retrieved_memories,
            intent=intent,
            min_relevance=memory_relevance,
            strict_intent=strict_intent,
        )

        explanations = self._build_retrieval_explanations(retrieved_memories)
        user_explanations = self._build_user_facing_explanations(relevant_memories)

        is_personal_query = self._is_personal_memory_query(user_message)
        has_supported = self._has_supported_memories(relevant_memories)

        if not has_supported and is_personal_query:
            grounding = {
                "grounded": True,
                "supported_values": [],
                "grounding_score": 1.0,
                "details": "Response correctly declined unsupported question.",
            }
            result = {
                "response": self.UNSUPPORTED_RESPONSE,
                "input_type": input_type,
                "extracted_memories": saved_records,
                "memories": [],
                "retrieved_memories": retrieved_memories,
                "intent": intent,
                "intent_confidence": intent_analysis["confidence"],
                "intent_strict": strict_intent,
                "supported": False,
                "grounded": True,
                "grounding_details": grounding,
                "llm_status": "skipped_unsupported",
            }
            if include_explanations:
                result["retrieval_explanations"] = explanations
                result["explanations"] = user_explanations
                result["why_used"] = user_explanations
            if settings.enable_metrics:
                result["performance_metrics"] = tracker.get_metrics()
            return result

        llm_status = "success"
        with tracker.timer("llm_ms"):
            try:
                response = self.llm_engine.generate_with_memories(
                    user_message=user_message,
                    memories=relevant_memories,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    fallback_on_empty=is_personal_query,
                    history=conversation_history,
                )
            except Exception as error:
                log_error_event(self.logger, "llm_generation_failed", error=error)
                if not fallback_to_extractive:
                    raise
                context_text = self.llm_engine.build_memory_context(relevant_memories)
                if relevant_memories:
                    response = (
                        "I'm currently unable to reach the local LLM runtime. "
                        f"Based directly on your stored memory:\n{context_text}"
                    )
                else:
                    response = (
                        "I am currently having trouble connecting to my local LLM runtime to process your request. "
                        "Please verify Ollama is active."
                    )
                llm_status = f"fallback: {str(error)}"

        log_llm_event(self.logger, prompt=user_message, status=llm_status)

        grounding = getattr(
            self.llm_engine,
            "verify_answer_grounding",
            lambda r, m, **k: {"grounded": True, "supported_values": [], "grounding_score": 1.0, "details": "Unverified"},
        )(response, relevant_memories, is_general_query=not is_personal_query)

        # 10.2 / 10.3: Feedback & Reinforcement
        if relevant_memories and hasattr(self, "importance_learner") and self.importance_learner:
            try:
                sup_vals = grounding.get("supported_values", [])
                self.importance_learner.apply_retrieval_feedback(
                    retrieved_memories=relevant_memories,
                    supported_values=sup_vals,
                )
            except Exception:
                pass

        result = {
            "response": response,
            "input_type": input_type,
            "extracted_memories": saved_records,
            "memories": relevant_memories,
            "retrieved_memories": retrieved_memories,
            "intent": intent,
            "intent_confidence": intent_analysis["confidence"],
            "intent_strict": strict_intent,
            "supported": True,
            "grounded": grounding.get("grounded", True),
            "grounding_details": grounding,
            "llm_status": llm_status,
        }
        if include_explanations:
            result["retrieval_explanations"] = explanations
            result["explanations"] = user_explanations
            result["why_used"] = user_explanations
        if settings.enable_metrics:
            result["performance_metrics"] = tracker.get_metrics()
        return result

    # 9.1: Unified alias
    process = respond

    def close(self):
        if hasattr(self.retrieval_engine, "close") and callable(self.retrieval_engine.close):
            self.retrieval_engine.close()
        if hasattr(self.llm_engine, "close") and callable(self.llm_engine.close):
            self.llm_engine.close()
        if hasattr(self.memory_store, "close") and callable(self.memory_store.close):
            self.memory_store.close()
        if hasattr(self.memory_graph, "clear") and callable(self.memory_graph.clear):
            self.memory_graph.clear()


