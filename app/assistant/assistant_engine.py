from app.llm.llm_engine import LLMEngine
from app.retrieval.retrieval_engine import RetrievalEngine


class AssistantEngine:
    """
    Main Recallix assistant pipeline.

    Connects:
        User Query
            ↓
        Retrieval
            ↓
        Intent Detection
            ↓
        Context Filtering
            ↓
        LLM
            ↓
        Final Response
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
        Filter retrieved memories before they reach
        the LLM.

        Only active, sufficiently relevant memories
        are allowed into the final context.
        """

        if not memories:
            return []

        relevant_memories = []

        intent_relations = {
            "PROJECT": {
                "works_on",
            },
            "EDUCATION": {
                "studies",
                "studies_at",
            },
            "LOCATION": {
                "lives_in",
                "current_city",
            },
            "PREFERENCE": {
                "likes",
            },
            "SKILL": {
                "knows",
            },
            "GOAL": {
                "wants_to_learn",
                "wants_to_become",
                "wants_to_build",
            },
        }

        for item in memories:

            if not isinstance(item, dict):
                continue

            memory = item.get("memory")

            if memory is None:
                continue

            # Defensive conflict protection.
            # RetrievalEngine normally returns only active
            # memories, but the assistant should never pass
            # archived memories to the LLM.
            if getattr(memory, "active", True) is not True:
                continue

            score = item.get("score", 0.0)

            try:
                score = float(score)
            except (TypeError, ValueError):
                continue

            if score < min_relevance:
                continue

            # Unknown intent:
            # rely on semantic relevance.
            if intent is None:
                relevant_memories.append(item)
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

            if category == intent:
                relevant_memories.append(item)

                continue

            if relation in intent_relations.get(
                intent,
                set(),
            ):
                relevant_memories.append(item)

        return relevant_memories

    def _has_supported_memories(self, memories):
        """
        Determine whether the final context contains
        at least one usable memory.
        """

        if not memories:
            return False

        for item in memories:

            if not isinstance(item, dict):
                continue

            memory = item.get("memory")

            if memory is None:
                continue

            if getattr(memory, "active", True) is not True:
                continue

            value = getattr(
                memory,
                "value",
                None,
            )

            if value is None:
                continue

            if not str(value).strip():
                continue

            return True

        return False

    def respond(
        self,
        user_message,
        top_k=5,
        min_score=0.0,
        semantic_threshold=0.0,
        temperature=0.2,
        max_tokens=500,
        memory_relevance=0.25,
    ):
        """
        Execute the complete memory-aware Recallix pipeline.

        Pipeline:

            User message
                ↓
            Semantic retrieval
                ↓
            Intent detection
                ↓
            Intent-aware filtering
                ↓
            Conflict-aware active memory selection
                ↓
            LLM-ready context
                ↓
            Supported-answer check
                ↓
            LLM
                ↓
            Final response
        """

        if not user_message or not user_message.strip():
            raise ValueError(
                "User message cannot be empty."
            )

        user_message = user_message.strip()

        # -------------------------------------------------
        # 1. Retrieve memories
        # -------------------------------------------------

        retrieved_memories = (
            self.retrieval_engine.search(
                query=user_message,
                top_k=top_k,
                min_score=min_score,
                semantic_threshold=semantic_threshold,
            )
        )

        # -------------------------------------------------
        # 2. Detect query intent
        # -------------------------------------------------

        intent = self._detect_intent(
            user_message
        )

        # -------------------------------------------------
        # 3. Select relevant memories
        # -------------------------------------------------

        relevant_memories = self._filter_memories(
            memories=retrieved_memories,
            intent=intent,
            min_relevance=memory_relevance,
        )

        # -------------------------------------------------
        # 4. Prevent unsupported memory answers
        # -------------------------------------------------

        has_supported_memories = (
            self._has_supported_memories(
                relevant_memories
            )
        )

        if not has_supported_memories:

            return {
                "response": (
                    "I don't have enough information "
                    "in my memory to answer that."
                ),
                "memories": [],
                "retrieved_memories": retrieved_memories,
                "intent": intent,
                "supported": False,
            }

        # -------------------------------------------------
        # 5. Generate final answer
        # -------------------------------------------------

        response = (
            self.llm_engine.generate_with_memories(
                user_message=user_message,
                memories=relevant_memories,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

        return {
            "response": response,
            "memories": relevant_memories,
            "retrieved_memories": retrieved_memories,
            "intent": intent,
            "supported": True,
        }

    def close(self):
        """
        Close the underlying Recallix components.
        """
        self.retrieval_engine.close()
        self.llm_engine.close()