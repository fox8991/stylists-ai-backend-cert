"""Retriever strategies for the RAG pipeline.

Each factory function returns a LangChain Retriever with a standard
.invoke(query) interface, making it easy to swap strategies in both
the app tool and eval notebooks.
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


def create_naive_retriever(vector_store: VectorStore, k: int = 10):
    """Create a simple vector similarity retriever.

    Args:
        vector_store: Any LangChain VectorStore.
        k: Number of documents to retrieve.

    Returns:
        A VectorStoreRetriever using cosine similarity.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})


def create_bm25_retriever(documents: list[Document], k: int = 10):
    """Create a sparse keyword-based BM25 retriever.

    Args:
        documents: List of documents to index.
        k: Number of documents to retrieve.

    Returns:
        A BM25Retriever instance.
    """
    from langchain_community.retrievers import BM25Retriever

    return BM25Retriever.from_documents(documents, k=k)


def create_rerank_retriever(base_retriever, model: str = "rerank-v3.5", top_n: int = 5):
    """Retrieve then rerank with Cohere for higher precision.

    Args:
        base_retriever: A retriever to fetch initial candidates (use higher k).
        model: Cohere rerank model name.
        top_n: Number of top documents to keep after reranking.

    Returns:
        A ContextualCompressionRetriever with Cohere reranking.
    """
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain_cohere import CohereRerank

    compressor = CohereRerank(model=model, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )


def create_multi_query_retriever(base_retriever, llm):
    """Generate multiple query reformulations to improve recall.

    Args:
        base_retriever: A retriever to fetch documents for each query variant.
        llm: LLM to generate query reformulations.

    Returns:
        A MultiQueryRetriever instance.
    """
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever

    return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)


def create_parent_document_retriever(
    raw_docs: list[Document],
    embeddings,
    parent_chunk_size: int = 2000,
    parent_chunk_overlap: int = 200,
    child_chunk_size: int = 400,
    child_chunk_overlap: int = 50,
    collection_name: str = "parent_child",
    location: str = ":memory:",
):
    """Small-to-big: search child chunks, return parent chunks.

    Args:
        raw_docs: Raw (unsplit) documents.
        embeddings: Embedding model for the child vector store.
        parent_chunk_size: Max characters per parent chunk.
        parent_chunk_overlap: Overlap between parent chunks.
        child_chunk_size: Max characters per child chunk.
        child_chunk_overlap: Overlap between child chunks.
        collection_name: Qdrant collection name for child chunks.
        location: Qdrant storage location.

    Returns:
        A ParentDocumentRetriever instance.
    """
    from langchain_classic.retrievers import ParentDocumentRetriever
    from langchain_core.stores import InMemoryStore
    from langchain_qdrant import QdrantVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from qdrant_client import QdrantClient, models

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap
    )

    client = QdrantClient(location=location)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    child_vectorstore = QdrantVectorStore(
        collection_name=collection_name, embedding=embeddings, client=client
    )

    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=child_vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.add_documents(raw_docs, ids=None)
    return retriever


def create_ensemble_retriever(retrievers: list, weights: list[float] | None = None):
    """Combine multiple retrievers with reciprocal rank fusion.

    Args:
        retrievers: List of retrievers to combine.
        weights: Weight for each retriever. Defaults to equal weighting.

    Returns:
        An EnsembleRetriever instance.
    """
    from langchain_classic.retrievers import EnsembleRetriever

    if weights is None:
        weights = [1 / len(retrievers)] * len(retrievers)
    return EnsembleRetriever(retrievers=retrievers, weights=weights)
