import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMEngine:
    """
    Local LLM interface for Recallix.

    Uses Ollama as the local LLM runtime.
    Default model: qwen2.5:3b
    """

    def __init__(
        self,
        model=None,
        base_url=None,
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

    def is_available(self):
        """
        Check whether the Ollama server is reachable.
        """

        try:
            request = Request(
                f"{self.base_url}/api/tags",
                method="GET",
            )

            with urlopen(request, timeout=3) as response:
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
            with urlopen(request, timeout=120) as response:
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
    ):
        """
        Generate a response using retrieved Recallix memories.
        """

        if not user_message or not user_message.strip():
            raise ValueError(
                "User message cannot be empty."
            )

        memory_context = (
            self._build_memory_context(
                memories
            )
        )

        system_prompt = (
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

        prompt = (
            "Memory context:\n"
            f"{memory_context}\n\n"
            "User question:\n"
            f"{user_message.strip()}"
        )

        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _build_memory_context(self, memories):
        """
        Convert memory objects into clean LLM-ready context.

        Internal metadata such as scores and categories are
        intentionally excluded.
        """

        if not memories:
            return "No relevant memories found."

        lines = []

        for item in memories:

            if not isinstance(item, dict):
                continue

            memory = item.get("memory")

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

            lines.append(
                f"{subject} {relation_text} {value}."
            )

        if not lines:
            return "No relevant memories found."

        return "\n".join(lines)

    def close(self):
        """
        Ollama is managed as a separate local service,
        so there is no client connection to close.
        """
        pass