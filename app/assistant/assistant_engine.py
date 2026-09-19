from app.llm.llm_engine import LLMEngine, UNSUPPORTED_RESPONSE
from app.retrieval.retrieval_engine import RetrievalEngine


class AssistantEngine:
    """Main Recallix memory-aware assistant pipeline."""

    UNSUPPORTED_RESPONSE = UNSUPPORTED_RESPONSE

    def __init__(self, retrieval_engine=None, llm_engine=None):
        self.retrieval_engine = retrieval_engine or RetrievalEngine()
        self.llm_engine = llm_engine or LLMEngine()

    def is_available(self):
        return self.llm_engine.is_available()

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

        if not self._has_supported_memories(relevant_memories):
            grounding = {
                "grounded": True,
                "supported_values": [],
                "grounding_score": 1.0,
                "details": "Response correctly declined unsupported question.",
            }
            result = {
                "response": self.UNSUPPORTED_RESPONSE,
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

        result = {
            "response": response,
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
        return result

    def close(self):
        self.retrieval_engine.close()
        self.llm_engine.close()

