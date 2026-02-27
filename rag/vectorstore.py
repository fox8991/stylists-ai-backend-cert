"""Vector store creation and management."""

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from config import settings


def create_vector_store(
    chunks: list[Document],
    collection_name: str = "fashion_knowledge",
    location: str = ":memory:",
) -> QdrantVectorStore:
    """Embed chunks and store in Qdrant vector store.

    Args:
        chunks: List of document chunks to embed and store.
        collection_name: Name of the Qdrant collection.
        location: Qdrant storage location. Use \":memory:\" for in-memory,
                  a path like \"./qdrant_data\" for local disk, or a URL
                  for Qdrant Cloud.

    Returns:
        QdrantVectorStore ready for similarity search.
    """
    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        location=location,
    )
    return vector_store
