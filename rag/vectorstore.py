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
    url: str | None = None,
    api_key: str | None = None,
) -> QdrantVectorStore:
    """Embed chunks and store in Qdrant vector store.

    Args:
        chunks: List of document chunks to embed and store.
        collection_name: Name of the Qdrant collection.
        location: Qdrant storage location for local mode.
        url: Qdrant Cloud URL. If provided, location is ignored.
        api_key: Qdrant Cloud API key.

    Returns:
        QdrantVectorStore ready for similarity search.
    """
    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)

    kwargs: dict = {
        "documents": chunks,
        "embedding": embeddings,
        "collection_name": collection_name,
    }
    if url:
        kwargs["url"] = url
        kwargs["api_key"] = api_key
    else:
        kwargs["location"] = location

    return QdrantVectorStore.from_documents(**kwargs)


def get_vector_store(
    collection_name: str = "fashion_knowledge",
) -> QdrantVectorStore:
    """Get a ready-to-use vector store.

    If QDRANT_URL is configured, connects to an existing Qdrant Cloud
    collection. Otherwise, builds an in-memory vector store from
    knowledge files (for local dev).

    Args:
        collection_name: Name of the Qdrant collection.

    Returns:
        QdrantVectorStore ready for retrieval.
    """
    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)

    if settings.QDRANT_URL:
        return QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=collection_name,
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )

    docs = load_knowledge_files()
    chunks = chunk_documents(docs)
    return create_vector_store(chunks, collection_name=collection_name)
