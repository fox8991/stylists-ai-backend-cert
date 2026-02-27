"""One-time script to ingest knowledge files into Qdrant Cloud.

Usage:
    uv run python scripts/ingest_to_qdrant.py
"""

from config import settings
from rag.chunking import chunk_documents
from rag.loader import load_knowledge_files
from rag.vectorstore import create_vector_store


def main() -> None:
    """Load, chunk, embed, and push to Qdrant Cloud."""
    if not settings.QDRANT_URL:
        raise RuntimeError("QDRANT_URL not set in .env")

    print("Loading knowledge files...")
    docs = load_knowledge_files()
    print(f"Loaded {len(docs)} documents")

    print("Chunking...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print(f"Embedding and pushing to {settings.QDRANT_URL}...")
    create_vector_store(
        chunks=chunks,
        collection_name="fashion_knowledge",
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )
    print("Done! Collection 'fashion_knowledge' is now on Qdrant Cloud.")


if __name__ == "__main__":
    main()
