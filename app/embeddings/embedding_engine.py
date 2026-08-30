from sentence_transformers import SentenceTransformer


class EmbeddingEngine:

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        print(f"Loading embedding model: {self.MODEL_NAME}")

        self.model = SentenceTransformer(
            self.MODEL_NAME
        )

        print("Embedding model loaded successfully.")

    def generate_embedding(self, text):

        if not text or not text.strip():
            raise ValueError(
                "Cannot generate an embedding from empty text."
            )

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding

    def generate_memory_text(
        self,
        subject,
        relation,
        value,
        category
    ):

        return (
            f"{subject} {relation} {value} "
            f"{category}"
        )

    def generate_memory_embedding(
        self,
        subject,
        relation,
        value,
        category
    ):

        memory_text = self.generate_memory_text(
            subject=subject,
            relation=relation,
            value=value,
            category=category
        )

        return self.generate_embedding(
            memory_text
        )