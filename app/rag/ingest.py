# app/rag/ingest.py
"""Ingest fashion knowledge files: load, chunk, embed, store in Qdrant."""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from app.config import settings

KNOWLEDGE_DIR = Path(__file__).parent.parent.parent / "knowledge"

DOMAINS = [
    "body_shapes",
    "color_theory",
    "fundamentals",
    "occasion_dressing",
    "style_archetypes",
    "wardrobe_building",
]


def load_knowledge_files() -> list[Document]:
    """Load all markdown files from knowledge/ directory with domain metadata."""
    documents = []
    for domain in DOMAINS:
        domain_dir = KNOWLEDGE_DIR / domain
        if not domain_dir.exists():
            continue
        for md_file in sorted(domain_dir.glob("*.md")):
            loader = TextLoader(str(md_file), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["domain"] = domain
                doc.metadata["source"] = f"{domain}/{md_file.name}"
            documents.extend(docs)
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 30,
) -> list[Document]:
    """Split documents into chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def create_vector_store(
    chunks: list[Document],
    collection_name: str = "fashion_knowledge",
) -> QdrantVectorStore:
    """Embed chunks and store in Qdrant in-memory vector store."""
    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        location=":memory:",
    )
    return vector_store


def build_retriever(vector_store: QdrantVectorStore, k: int = 5):
    """Create a retriever from the vector store."""
    return vector_store.as_retriever(search_kwargs={"k": k})
