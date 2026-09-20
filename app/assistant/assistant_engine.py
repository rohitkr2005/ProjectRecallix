from enum import Enum
import re

from app.llm.llm_engine import LLMEngine, UNSUPPORTED_RESPONSE
from app.memory.forgetting import MemoryForgetter
from app.memory.graph import MemoryGraph
from app.memory.importance_learning import ImportanceLearner
from app.memory.memory_extractor import MemoryExtractor
from app.memory.memory_store import MemoryStore
from app.memory.temporal import detect_query_temporal_intent, detect_temporal_state
from app.retrieval.retrieval_engine import RetrievalEngine


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

    def save_extracted_memories(self, extracted_memories, temporal_state=None):
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

            memory, status, metadata = self.memory_store.save_memory_with_semantics(
                subject=subject,
                relation=relation,
                value=value,
                category=category,
                importance=importance,
                embedding=embedding,
                temporal_state=temp_state,
            )

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
    ):
        if not user_message or not user_message.strip():
            raise ValueError("User message cannot be empty.")

        user_message = user_message.strip()
        input_type = self.classify_input(user_message)
        saved_records = []

        temporal_state = detect_temporal_state(user_message)
        # 9.2 / 9.3: Handle statement extraction for STATEMENT or BOTH
        if input_type in (InputType.STATEMENT, InputType.BOTH):
            try:
                extracted = self.memory_extractor.extract(user_message)
                if extracted:
                    saved_records = self.save_extracted_memories(extracted, temporal_state=temporal_state)
            except Exception:
                saved_records = []

        # 1. Pure STATEMENT handling
        if input_type == InputType.STATEMENT:
            if saved_records:
                if len(saved_records) == 1:
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
            return result

        # 2. Pure NEITHER (chit-chat / greeting) handling
        if input_type == InputType.NEITHER:
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
            return result

        # 3. QUESTION or BOTH handling
        retrieved_memories = self.retrieval_engine.search(
            query=user_message,
            top_k=top_k,
            min_score=min_score,
            semantic_threshold=semantic_threshold,
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

        if not self._has_supported_memories(relevant_memories):
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
            return result

        llm_status = "success"
        try:
            response = self.llm_engine.generate_with_memories(
                user_message=user_message,
                memories=relevant_memories,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as error:
            if not fallback_to_extractive:
                raise
            context_text = self.llm_engine.build_memory_context(relevant_memories)
            response = (
                "I'm currently unable to reach the local LLM runtime. "
                f"Based directly on your stored memory:\n{context_text}"
            )
            llm_status = f"fallback: {str(error)}"

        grounding = getattr(
            self.llm_engine,
            "verify_answer_grounding",
            lambda r, m: {"grounded": True, "supported_values": [], "grounding_score": 1.0, "details": "Unverified"},
        )(response, relevant_memories)

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


