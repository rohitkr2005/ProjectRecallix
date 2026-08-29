import re
from dataclasses import dataclass
from typing import List


@dataclass
class ExtractedMemory:
    subject: str
    relation: str
    value: str
    category: str
    importance: int


class MemoryExtractor:
    """
    Extract structured long-term memories from natural-language messages.

    Step 3 responsibilities:
    - Detect useful personal information
    - Extract preferences
    - Extract locations
    - Extract education/learning information
    - Extract projects
    - Extract skills
    - Extract goals
    - Support multiple memories in one message
    - Ignore casual/irrelevant messages

    This class does NOT save anything to the database.
    Database persistence is handled separately by MemoryStore.
    """

    def __init__(self):
        self.subject = "User"

    def extract(self, message: str) -> List[ExtractedMemory]:
        """
        Extract all memories from a message.

        Returns:
            List[ExtractedMemory]
        """

        if not message or not message.strip():
            return []

        message = self._clean_message(message)

        if not message:
            return []

        memories = []

        # Split compound sentences so different memory types
        # can be extracted independently.
        sentences = self._split_sentences(message)

        for sentence in sentences:
            memories.extend(self._extract_from_sentence(sentence))

        return self._remove_duplicates(memories)

    # ---------------------------------------------------------
    # Core extraction
    # ---------------------------------------------------------

    def _extract_from_sentence(
        self,
        sentence: str
    ) -> List[ExtractedMemory]:

        memories = []

        sentence = sentence.strip()

        if not sentence:
            return memories

        # Preference / likes
        memories.extend(
            self._extract_preferences(sentence)
        )

        # Location
        memories.extend(
            self._extract_location(sentence)
        )

        # Education / learning
        memories.extend(
            self._extract_education(sentence)
        )

        # Projects
        memories.extend(
            self._extract_projects(sentence)
        )

        # Skills
        memories.extend(
            self._extract_skills(sentence)
        )

        # Goals
        memories.extend(
            self._extract_goals(sentence)
        )

        return memories

    # ---------------------------------------------------------
    # Preferences
    # ---------------------------------------------------------

    def _extract_preferences(
        self,
        sentence: str
    ) -> List[ExtractedMemory]:

        patterns = [
            r"\bI\s+like\s+(.+)",
            r"\bI\s+love\s+(.+)",
            r"\bI\s+enjoy\s+(.+)",
            r"\bI\s+prefer\s+(.+)",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                sentence,
                flags=re.IGNORECASE
            )

            if match:
                values = self._extract_values(
                    match.group(1)
                )

                return [
                    ExtractedMemory(
                        subject=self.subject,
                        relation="likes",
                        value=value,
                        category="PREFERENCE",
                        importance=7
                    )
                    for value in values
                ]

        return []

    # ---------------------------------------------------------
    # Location
    # ---------------------------------------------------------

    def _extract_location(
        self,
        sentence: str
    ) -> List[ExtractedMemory]:

        patterns = [
            r"\bI\s+live\s+in\s+(.+)",
            r"\bI\s+am\s+from\s+(.+)",
            r"\bI'm\s+from\s+(.+)",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                sentence,
                flags=re.IGNORECASE
            )

            if match:
                value = self._clean_value(
                    match.group(1)
                )

                if value:
                    return [
                        ExtractedMemory(
                            subject=self.subject,
                            relation="lives_in",
                            value=value,
                            category="LOCATION",
                            importance=8
                        )
                    ]

        return []

    # ---------------------------------------------------------
    # Education / Learning
    # ---------------------------------------------------------

    def _extract_education(
        self,
        sentence: str
    ) -> List[ExtractedMemory]:

        patterns = [
            r"\bI\s+study\s+(.+)",
            r"\bI\s+am\s+studying\s+(.+)",
            r"\bI'm\s+studying\s+(.+)",
            r"\bI\s+am\s+learning\s+(.+)",
            r"\bI'm\s+learning\s+(.+)",
            r"\bI\s+learn\s+(.+)",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                sentence,
                flags=re.IGNORECASE
            )

            if match:
                values = self._extract_values(
                    match.group(1)
                )

                return [
                    ExtractedMemory(
                        subject=self.subject,
                        relation="studies",
                        value=value,
                        category="EDUCATION",
                        importance=8
                    )
                    for value in values
                ]

        return []

    # ---------------------------------------------------------
    # Projects
    # ---------------------------------------------------------

    def _extract_projects(
        self,
        sentence: str
    ) -> List[ExtractedMemory]:

        patterns = [
            r"\bI\s+work\s+on\s+(.+)",
            r"\bI\s+am\s+working\s+on\s+(.+)",
            r"\bI'm\s+working\s+on\s+(.+)",
            r"\bI\s+build\s+(.+)",
            r"\bI\s+am\s+building\s+(.+)",
            r"\bI'm\s+building\s+(.+)",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                sentence,
                flags=re.IGNORECASE
            )

            if match:
                values = self._extract_values(
                    match.group(1)
                )

                return [
                    ExtractedMemory(
                        subject=self.subject,
                        relation="works_on",
                        value=value,
                        category="PROJECT",
                        importance=8
                    )
                    for value in values
                ]

        return []

    # ---------------------------------------------------------
    # Skills
    # ---------------------------------------------------------

    def _extract_skills(
        self,
        sentence: str
    ) -> List[ExtractedMemory]:

        patterns = [
            r"\bI\s+know\s+(.+)",
            r"\bI\s+am\s+skilled\s+in\s+(.+)",
            r"\bI'm\s+skilled\s+in\s+(.+)",
            r"\bI\s+use\s+(.+)",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                sentence,
                flags=re.IGNORECASE
            )

            if match:
                values = self._extract_values(
                    match.group(1)
                )

                return [
                    ExtractedMemory(
                        subject=self.subject,
                        relation="knows",
                        value=value,
                        category="SKILL",
                        importance=7
                    )
                    for value in values
                ]

        return []

    # ---------------------------------------------------------
    # Goals
    # ---------------------------------------------------------

    def _extract_goals(
        self,
        sentence: str
    ) -> List[ExtractedMemory]:

        patterns = [
            (
                r"\bI\s+want\s+to\s+learn\s+(.+)",
                "wants_to_learn"
            ),
            (
                r"\bI\s+want\s+to\s+become\s+(.+)",
                "wants_to_become"
            ),
            (
                r"\bI\s+want\s+to\s+build\s+(.+)",
                "wants_to_build"
            ),
            (
                r"\bMy\s+goal\s+is\s+to\s+learn\s+(.+)",
                "wants_to_learn"
            ),
            (
                r"\bMy\s+goal\s+is\s+to\s+build\s+(.+)",
                "wants_to_build"
            ),
            (
                r"\bI\s+want\s+to\s+(.+)",
                "wants_to"
            ),
        ]

        for pattern, relation in patterns:
            match = re.search(
                pattern,
                sentence,
                flags=re.IGNORECASE
            )

            if match:
                value = self._clean_value(
                    match.group(1)
                )

                if value:
                    return [
                        ExtractedMemory(
                            subject=self.subject,
                            relation=relation,
                            value=value,
                            category="GOAL",
                            importance=9
                        )
                    ]

        return []

    # ---------------------------------------------------------
    # Value handling
    # ---------------------------------------------------------

    def _extract_values(
        self,
        text: str
    ) -> List[str]:

        text = self._clean_value(text)

        if not text:
            return []

        # Convert "A, B and C" into individual values.
        text = re.sub(
            r"\s*,\s*and\s+",
            ",",
            text,
            flags=re.IGNORECASE
        )

        parts = re.split(
            r"\s*,\s*|\s+\band\b\s+",
            text,
            flags=re.IGNORECASE
        )

        values = []

        for part in parts:
            value = self._clean_value(part)

            if value:
                values.append(value)

        return values

    def _clean_value(
        self,
        value: str
    ) -> str:

        value = value.strip()

        # Remove common sentence-ending punctuation.
        value = re.sub(
            r"[.!?]+$",
            "",
            value
        )

        value = value.strip(" ,;:")

        # Remove duplicate whitespace.
        value = re.sub(
            r"\s+",
            " ",
            value
        )

        return value

    def _clean_message(
        self,
        message: str
    ) -> str:

        message = message.strip()

        # Normalize curly apostrophes.
        message = message.replace("’", "'")

        # Normalize whitespace.
        message = re.sub(
            r"\s+",
            " ",
            message
        )

        return message

    # ---------------------------------------------------------
    # Sentence splitting
    # ---------------------------------------------------------

    def _split_sentences(
        self,
        message: str
    ) -> List[str]:

        # First split on normal sentence punctuation.
        sentences = re.split(
            r"(?<=[.!?])\s+",
            message
        )

        result = []

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            # Handle compound statements such as:
            #
            # I like Python and I study Data Science.
            #
            # without breaking values such as:
            #
            # Python and SQL
            #
            pieces = re.split(
                r"\s+\band\s+(?=I\s+)",
                sentence,
                flags=re.IGNORECASE
            )

            for piece in pieces:
                piece = piece.strip()

                if piece:
                    result.append(piece)

        return result

    # ---------------------------------------------------------
    # Duplicate removal
    # ---------------------------------------------------------

    def _remove_duplicates(
        self,
        memories: List[ExtractedMemory]
    ) -> List[ExtractedMemory]:

        unique = []
        seen = set()

        for memory in memories:

            key = (
                memory.subject.lower(),
                memory.relation.lower(),
                memory.value.lower(),
                memory.category.lower()
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(memory)

        return unique