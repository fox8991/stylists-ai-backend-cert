"""Shared RAG components initialized at app startup."""


class RAGRegistry:
    """Holds initialized RAG components for the app.

    Attributes:
        vector_store: The vector store backing retrieval.
        retriever: A LangChain retriever created from the vector store.
    """

    vector_store = None
    retriever = None


rag_registry = RAGRegistry()
