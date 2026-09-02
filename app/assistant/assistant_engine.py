from app.llm.llm_engine import LLMEngine
from app.retrieval.retrieval_engine import RetrievalEngine
class AssistantEngine:
    """
    Main Recallix assistant pipeline.
    Connects semantic memory retrieval with the
    local Ollama LLM.
    """
    def __init__(
        self,
        retrieval_engine=None,
        llm_engine=None,
    ):
        self.retrieval_engine = (
            retrieval_engine
            or RetrievalEngine()
        )
        self.llm_engine = (
            llm_engine
            or LLMEngine()
        )
    def is_available(self):
        """
        Check whether the local LLM service is available.
        """
        return self.llm_engine.is_available()
    def _detect_intent(self, user_message):
        """
        Detect the same high-level intent used by
        the retrieval layer.
        """
        return self.retrieval_engine._detect_query_intent(
            user_message
        )
    def _filter_memories(
        self,
        memories,
        intent,
        min_relevance=0.25,
    ):
        """
        Keep memories that are relevant to the
        detected query intent.
        This prevents unrelated memories from being
        unnecessarily sent to the LLM.
        """
        if not memories:
            return []
        if intent is None:
            return [
                item
                for item in memories
                if item.get("score", 0.0) >= min_relevance
            ]
        filtered = []
        for item in memories:
            memory = item.get("memory")
            if memory is None:
                continue
            score = item.get("score", 0.0)
            if score < min_relevance:
                continue
            category = getattr(
                memory,
                "category",
                None,
            )
            relation = getattr(
                memory,
                "relation",
                None,
            )
            relevant = False
            if category == intent:
                relevant = True
            elif (
                intent == "PROJECT"
                and relation == "works_on"
            ):
                relevant = True
            elif (
                intent == "LOCATION"
                and relation == "lives_in"
            ):
                relevant = True
            if relevant:
                filtered.append(item)
        # If intent filtering produces no results,
        # fall back to sufficiently relevant semantic results.
        if not filtered:
            return [
                item
                for item in memories
                if item.get("score", 0.0) >= min_relevance
            ]
        return filtered
    def respond(
        self,
        user_message,
        top_k=5,
        min_score=0.0,
        temperature=0.2,
        max_tokens=500,
        memory_relevance=0.25,
    ):
        """
        Generate a memory-aware response.
        Pipeline:
        User message
            ↓
        Semantic retrieval
            ↓
        Intent-aware filtering
            ↓
        Relevant memory context
            ↓
        Ollama LLM
            ↓
        Final response
        """
        if not user_message or not user_message.strip():
            raise ValueError(
                "User message cannot be empty."
            )
        user_message = user_message.strip()
        memories = self.retrieval_engine.search(
            query=user_message,
            top_k=top_k,
            min_score=min_score,
        )
        intent = self._detect_intent(
            user_message
        )
        relevant_memories = self._filter_memories(
            memories=memories,
            intent=intent,
            min_relevance=memory_relevance,
        )
        response = self.llm_engine.generate_with_memories(
            user_message=user_message,
            memories=relevant_memories,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "response": response,
            "memories": relevant_memories,
            "retrieved_memories": memories,
            "intent": intent,
        }
    def close(self):
        """
        Close the underlying Recallix components.
        """
        self.retrieval_engine.close()
        self.llm_engine.close()
