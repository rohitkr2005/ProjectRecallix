import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

UNSUPPORTED_RESPONSE = "I don't have enough information in my memory to answer that."

GROUNDED_SYSTEM_PROMPT = (
    "You are Recallix, an AI assistant with memory. "
    "Answer the user's question using only the explicitly "
    "provided memory context. "
    "Do not invent facts. "
    "Do not infer facts that are not explicitly supported. "
    "Do not treat general world knowledge as user memory. "
    "If the provided memories do not contain enough information "
    "to answer the question about the user, clearly say that "
    "you do not have enough information. "
    "Never claim an unsupported user fact. "
    "Do not mention memory categories, relevance scores, "
    "retrieval details, embeddings, or system metadata. "
    "Answer naturally, clearly, and directly."
)


class LLMEngine:
    """
    Local LLM interface for Recallix.

    Uses Ollama as the local LLM runtime.
    Default model: qwen2.5:3b
    """

    UNSUPPORTED_RESPONSE = UNSUPPORTED_RESPONSE
    GROUNDED_SYSTEM_PROMPT = GROUNDED_SYSTEM_PROMPT

    def __init__(
        self,
        model=None,
        base_url=None,
        timeout=None,
        connect_timeout=None,
    ):
        self.model = model or os.getenv(
            "RECALLIX_LLM_MODEL",
            "qwen2.5:3b",
        )

        self.base_url = (
            base_url
            or os.getenv(
                "RECALLIX_LLM_BASE_URL",
                "http://localhost:11434",
            )
        ).rstrip("/")

        try:
            self.timeout = float(
                timeout or os.getenv("RECALLIX_LLM_TIMEOUT", "120")
            )
        except (ValueError, TypeError):
            self.timeout = 120.0

        try:
            self.connect_timeout = float(
                connect_timeout or os.getenv("RECALLIX_LLM_CONNECT_TIMEOUT", "3")
            )
        except (ValueError, TypeError):
            self.connect_timeout = 3.0

    def is_available(self):
        """
        Check whether the Ollama server is reachable.
        """

        try:
            request = Request(
                f"{self.base_url}/api/tags",
                method="GET",
            )

            with urlopen(request, timeout=self.connect_timeout) as response:
                return response.status == 200

        except (URLError, HTTPError, OSError):
            return False

    def generate(
        self,
        prompt,
        system_prompt=None,
        temperature=0.2,
        max_tokens=500,
    ):
        """
        Generate a response using the configured Ollama model.
        """

        if not prompt or not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt.strip(),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt.strip(),
            }
        )

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(
                    response.read().decode("utf-8")
                )

        except HTTPError as error:

            try:
                error_body = (
                    error.read().decode("utf-8")
                )
            except Exception:
                error_body = str(error)

            raise RuntimeError(
                f"Ollama request failed: {error_body}"
            ) from error

        except (URLError, OSError) as error:

            raise RuntimeError(
                "Unable to connect to Ollama. "
                "Make sure Ollama is running on "
                f"{self.base_url}."
            ) from error

        message = result.get(
            "message",
            {}
        )

        content = message.get(
            "content"
        )

        if not content:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return content.strip()

    def generate_with_memories(
        self,
        user_message,
        memories,
        temperature=0.2,
        max_tokens=500,
        system_prompt=None,
        fallback_on_empty=True,
    ):
        """
        Generate a response using retrieved Recallix memories.

        When fallback_on_empty is True, if no supported active memories
        are provided, immediately returns UNSUPPORTED_RESPONSE without
        incurring LLM latency or hallucination risk.
        """

        if not user_message or not user_message.strip():
            raise ValueError(
                "User message cannot be empty."
            )

        if fallback_on_empty and not self._has_supported_memories(memories):
            return self.UNSUPPORTED_RESPONSE

        memory_context = self.build_memory_context(memories)

        effective_system_prompt = (
            system_prompt or self.GROUNDED_SYSTEM_PROMPT
        )

        prompt = (
            "Memory context:\n"
            f"{memory_context}\n\n"
            "User question:\n"
            f"{user_message.strip()}"
        )

        return self.generate(
            prompt=prompt,
            system_prompt=effective_system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def build_memory_context(self, memories, max_memories=None):
        """
        Convert memory objects into clean LLM-ready context.

        Internal metadata such as scores, categories, timestamps,
        embeddings, and IDs are intentionally excluded.
        """

        if not memories:
            return "No relevant memories found."

        lines = []

        for item in memories:

            if isinstance(item, dict):
                memory = item.get("memory")
            else:
                memory = item

            if memory is None:
                continue

            if getattr(
                memory,
                "active",
                True,
            ) is not True:
                continue

            subject = getattr(
                memory,
                "subject",
                "User",
            )

            relation = getattr(
                memory,
                "relation",
                "",
            )

            value = getattr(
                memory,
                "value",
                "",
            )

            if not str(value).strip():
                continue

            relation_phrases = {
                "likes": "likes",
                "lives_in": "lives in",
                "studies": "studies",
                "studies_at": "studies at",
                "works_at": "works at",
                "current_role": "has the current role",
                "current_city": "currently lives in",
                "works_on": "works on",
                "knows": "knows",
                "wants_to_learn": "wants to learn",
                "wants_to_become": "wants to become",
                "wants_to_build": "wants to build",
            }

            relation_text = relation_phrases.get(
                relation,
                relation.replace(
                    "_",
                    " ",
                ),
            )

            formatted_line = f"{subject} {relation_text} {value}."
            if formatted_line not in lines:
                lines.append(formatted_line)

            if max_memories is not None and len(lines) >= max_memories:
                break

        if not lines:
            return "No relevant memories found."

        return "\n".join(lines)

    def _build_memory_context(self, memories):
        """Backwards compatibility alias for build_memory_context."""
        return self.build_memory_context(memories)

    def prioritize_memories(
        self,
        memories,
        max_memories=5,
        min_relevance=0.25,
    ):
        """
        Prioritize and rank memories for context injection:
        1. Filter for active memories only.
        2. Filter out memories below min_relevance.
        3. Sort descending by relevance score.
        4. Cap at max_memories.
        """
        if not memories:
            return []

        filtered = []
        for item in memories:
            if not isinstance(item, dict):
                continue
            memory = item.get("memory")
            if memory is None or getattr(memory, "active", True) is not True:
                continue
            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0

            if score < min_relevance:
                continue

            filtered.append(item)

        filtered.sort(
            key=lambda x: float(x.get("score", 0.0)),
            reverse=True,
        )

        if max_memories is not None and max_memories > 0:
            return filtered[:max_memories]

        return filtered

    def verify_answer_grounding(self, response, memories):
        """
        Verify whether an answer is grounded in the provided memories.

        Returns:
            dict: {
                "grounded": bool,
                "supported_values": list[str],
                "grounding_score": float,
                "details": str,
            }
        """
        if not response or not response.strip():
            return {
                "grounded": False,
                "supported_values": [],
                "grounding_score": 0.0,
                "details": "Empty response cannot be grounded.",
            }

        response_clean = response.strip()

        # If the response explicitly states lack of information, it is grounded
        # in the lack of memory evidence.
        if (
            response_clean == self.UNSUPPORTED_RESPONSE
            or "don't have enough information" in response_clean.lower()
            or "do not have enough information" in response_clean.lower()
        ):
            return {
                "grounded": True,
                "supported_values": [],
                "grounding_score": 1.0,
                "details": "Response correctly declined to answer due to missing/insufficient memory.",
            }

        # Extract active values from memories
        memory_values = []
        for item in memories or []:
            if isinstance(item, dict):
                mem = item.get("memory")
            else:
                mem = item
            if mem is None or getattr(mem, "active", True) is not True:
                continue
            val = getattr(mem, "value", None)
            if val is not None and str(val).strip():
                memory_values.append(str(val).strip())

        if not memory_values:
            return {
                "grounded": False,
                "supported_values": [],
                "grounding_score": 0.0,
                "details": "Substantive answer provided without supporting memories (hallucination risk).",
            }

        matched_values = []
        response_lower = response_clean.lower()

        for val in memory_values:
            val_lower = val.lower()
            if val_lower in response_lower:
                matched_values.append(val)
            else:
                tokens = [t for t in val_lower.split() if len(t) > 3]
                if tokens and any(t in response_lower for t in tokens):
                    matched_values.append(val)

        unique_matched = list(dict.fromkeys(matched_values))
        grounding_score = round(
            len(unique_matched) / max(len(memory_values), 1),
            4,
        )

        is_grounded = len(unique_matched) > 0

        details = (
            f"Grounded: {len(unique_matched)}/{len(memory_values)} memory values supported."
            if is_grounded
            else "Ungrounded: Response does not reference any provided memory values."
        )

        return {
            "grounded": is_grounded,
            "supported_values": unique_matched,
            "grounding_score": grounding_score,
            "details": details,
        }

    def _has_supported_memories(self, memories):
        """Check whether there is at least one active memory with non-empty content."""
        if not memories:
            return False
        for item in memories:
            if isinstance(item, dict):
                memory = item.get("memory")
            else:
                memory = item
            if memory is None or getattr(memory, "active", True) is not True:
                continue
            value = getattr(memory, "value", None)
            if value is not None and str(value).strip():
                return True
        return False

    def close(self):
        """
        Ollama is managed as a separate local service,
        so there is no client connection to close.
        """
        pass