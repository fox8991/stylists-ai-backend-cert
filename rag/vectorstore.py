"""Vector store creation and management."""

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from config import settings
from rag.chunking import chunk_documents
from rag.loader import load_knowledge_files


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


def get_vector_store(
    collection_name: str = "fashion_knowledge",
    location: str = ":memory:",
) -> QdrantVectorStore:
    """Get a ready-to-use vector store.

    Currently loads knowledge files, chunks them, and builds an in-memory
    vector store. Later this will connect to a pre-built Qdrant Cloud
    collection instead.

    Args:
        collection_name: Name of the Qdrant collection.
        location: Qdrant storage location.

    Returns:
        QdrantVectorStore ready for retrieval.
    """
    docs = load_knowledge_files()
    chunks = chunk_documents(docs)
    return create_vector_store(chunks, collection_name=collection_name, location=location)
