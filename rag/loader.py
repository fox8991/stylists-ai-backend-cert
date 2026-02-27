"""Load knowledge files from disk with domain metadata."""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"

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
