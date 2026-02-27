"""Retriever strategies for the RAG pipeline."""

from langchain_qdrant import QdrantVectorStore


def create_retriever(vector_store: QdrantVectorStore, k: int = 5):
    """Create a naive similarity retriever from a vector store.

    Args:
        vector_store: The vector store to retrieve from.
        k: Number of documents to retrieve.

    Returns:
        A LangChain retriever instance.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})
