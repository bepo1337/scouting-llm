from abc import ABC, abstractmethod
from langchain_community.embeddings import OllamaEmbeddings

class EmbeddingModel(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> list:
        pass


class MistralEmbedding(EmbeddingModel):
    def __init__(self):
        self.model = OllamaEmbeddings(model="mistral")

    def embed_query(self, text: str) -> list:
        return self.model.embed_query(text)

class MistralEmbedding(EmbeddingModel):
    def __init__(self):
        self.model = OllamaEmbeddings(model="mistral")

    def embed_query(self, text: str) -> list:
        return self.model.embed_query(text)


class NomicEmbedding(EmbeddingModel):
    def __init__(self):
        self.model = OllamaEmbeddings(model="nomic-embed-text")

    def embed_query(self, text: str) -> list:
        return self.model.embed_query(text)
