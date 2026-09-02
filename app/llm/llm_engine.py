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
                error_body = error.read().decode("utf-8")
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

        message = result.get("message", {})
        content = message.get("content")

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

        memory_context = self._build_memory_context(
            memories
        )

        system_prompt = (
            "You are Recallix, an AI assistant with memory. "
            "Use the provided memories when they are relevant. "
            "Only state information directly supported by the provided memories. "
            "Do not invent, infer, compare, rank, or assume facts that are not explicitly present. "
            "If a memory is not relevant, ignore it. "
            "Do not mention internal memory categories, "
            "relevance scores, retrieval details, or system metadata. "
            "Answer naturally, clearly, and directly."
        )

        prompt = (
            f"Relevant memories:\n"
            f"{memory_context}\n\n"
            f"User message:\n"
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
        Convert retrieved memory results into readable context.
        """

        if not memories:
            return "No relevant memories found."

        lines = []

        for index, item in enumerate(
            memories,
            start=1,
        ):
            memory = item.get("memory")
            score = item.get("score")

            if memory is None:
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

            category = getattr(
                memory,
                "category",
                "",
            )

            if score is not None:
                lines.append(
                    f"{index}. "
                    f"{subject} {relation} {value} "
                    f"[{category}] "
                    f"(relevance: {score:.3f})"
                )
            else:
                lines.append(
                    f"{index}. "
                    f"{subject} {relation} {value} "
                    f"[{category}]"
                )

        if not lines:
            return "No relevant memories found."

        return "\n".join(lines)

    def close(self):
        """
        Kept for interface compatibility.

        Ollama is managed as a separate local service,
        so there is no client connection to close here.
        """
        pass