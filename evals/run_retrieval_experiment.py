"""Run retrieval strategy experiments with RAGAS evaluation.

Uses the winning chunking config (1000/100) and tests 5 retrieval strategies:
  1. Naive dense (k=10)
  2. BM25 keyword (k=10)
  3. Rerank (Cohere rerank-v3.5, broad k=20 -> top 5)
  4. Parent-child (small embed k=10, return parent)
  5. Ensemble (naive + BM25, reciprocal rank fusion)

Usage:
    uv run python evals/run_retrieval_experiment.py
"""

import os
import sys
import time

import nest_asyncio

nest_asyncio.apply()

from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
load_dotenv(os.path.join(project_root, ".env"))

os.environ["LANGSMITH_TRACING"] = "false"

import pandas as pd
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas import EvaluationDataset, RunConfig
from ragas import evaluate as ragas_evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
)

from rag.chunking import chunk_documents
from rag.loader import load_knowledge_files
from rag.vectorstore import create_vector_store

EVALS_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(EVALS_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
TESTSET_PATH = os.path.join(EVALS_DIR, "synthetic_testset.csv")

# --- Load docs and test set ---
print("Loading documents...")
docs = load_knowledge_files()
print(f"  {len(docs)} documents loaded")

print("Loading cached test set...")
test_df = pd.read_csv(TESTSET_PATH)
print(f"  {len(test_df)} test samples")

# --- Chunk with winning config: 1000/100 ---
print("\nChunking with large (1000/100)...")
chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=100)
print(f"  {len(chunks)} chunks")

# --- Shared components ---
rag_llm = ChatOpenAI(model="gpt-4.1-mini")
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

RAG_PROMPT = """You are a personal AI stylist. Answer the question using ONLY the provided context.

### Question
{question}

### Context
{context}
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

METRICS = [
    LLMContextRecall(),
    Faithfulness(),
    FactualCorrectness(),
    ResponseRelevancy(),
    ContextEntityRecall(),
]


def run_experiment(retriever_fn, name: str) -> pd.DataFrame:
    """Run a retriever through all test queries and score with RAGAS."""
    cache_path = os.path.join(RESULTS_DIR, f"retrieval_{name}.csv")
    if os.path.exists(cache_path):
        print(f"\nLoading cached results for {name}")
        return pd.read_csv(cache_path)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")

    samples = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        retrieved = retriever_fn(row["user_input"])
        docs_content = "\n\n".join(doc.page_content for doc in retrieved)
        messages = rag_prompt.format_messages(
            question=row["user_input"], context=docs_content
        )
        response = rag_llm.invoke(messages).content
        samples.append(
            SingleTurnSample(
                user_input=row["user_input"],
                response=response,
                reference=row["reference"],
                retrieved_contexts=[doc.page_content for doc in retrieved],
            )
        )
        if (i + 1) % 7 == 0:
            print(f"  Generated {i + 1}/{len(test_df)} responses")

    print(f"  All {len(samples)} responses generated. Running RAGAS eval...")
    t0 = time.time()
    result = ragas_evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=METRICS,
        run_config=RunConfig(timeout=360),
    )
    elapsed = time.time() - t0
    print(f"  RAGAS eval done in {elapsed:.0f}s")

    df = result.to_pandas()
    df.to_csv(cache_path, index=False)
    print(f"  Saved to {cache_path}")
    return df


# =====================================================================
# Strategy 1: Naive Dense (k=10)
# =====================================================================
print("\n--- Building Naive Dense retriever ---")
vs = create_vector_store(chunks, collection_name="retrieval_naive")
naive_retriever = vs.as_retriever(search_kwargs={"k": 10})
naive_results = run_experiment(naive_retriever.invoke, "naive_dense")

# =====================================================================
# Strategy 2: BM25 Keyword (k=10)
# =====================================================================
print("\n--- Building BM25 retriever ---")
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(chunks, k=10)
bm25_results = run_experiment(bm25_retriever.invoke, "bm25")

# =====================================================================
# Strategy 3: Rerank (Cohere rerank-v3.5)
# Broad retrieval k=20, then rerank to top 5
# =====================================================================
print("\n--- Building Rerank retriever ---")
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_cohere import CohereRerank

wide_retriever = vs.as_retriever(search_kwargs={"k": 20})
cohere_reranker = CohereRerank(model="rerank-v3.5", top_n=5)
rerank_retriever = ContextualCompressionRetriever(
    base_compressor=cohere_reranker,
    base_retriever=wide_retriever,
)


def rerank_retrieve(query: str) -> list[Document]:
    """Retrieve k=20 via dense, then rerank to top 5 with Cohere."""
    # Cohere trial key: 10 calls/min, sleep to avoid rate limit
    time.sleep(7)
    return rerank_retriever.invoke(query)


rerank_results = run_experiment(rerank_retrieve, "rerank")

# =====================================================================
# Strategy 4: Parent-Child
# Embed small chunks (400), return parent chunks (2000)
# =====================================================================
print("\n--- Building Parent-Child retriever ---")
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

from langchain_qdrant import QdrantVectorStore

parent_child_vs = QdrantVectorStore.from_documents(
    documents=[],
    embedding=embeddings,
    collection_name="retrieval_parent_child",
    location=":memory:",
)
docstore = InMemoryStore()

parent_child_retriever = ParentDocumentRetriever(
    vectorstore=parent_child_vs,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
parent_child_retriever.add_documents(docs)
print(f"  Parent-child store built")

parent_child_results = run_experiment(parent_child_retriever.invoke, "parent_child")

# =====================================================================
# Strategy 5: Ensemble (all 4 retrievers, reciprocal rank fusion)
# =====================================================================
print("\n--- Building Ensemble retriever ---")
from langchain_classic.retrievers import EnsembleRetriever

retriever_list = [naive_retriever, bm25_retriever, rerank_retriever, parent_child_retriever]
equal_weighting = [1 / len(retriever_list)] * len(retriever_list)

ensemble_retriever = EnsembleRetriever(
    retrievers=retriever_list,
    weights=equal_weighting,
)


def ensemble_retrieve(query: str) -> list[Document]:
    """Ensemble retrieval with rate limit sleep for Cohere reranker."""
    time.sleep(7)
    return ensemble_retriever.invoke(query)


ensemble_results = run_experiment(ensemble_retrieve, "ensemble")

# =====================================================================
# Comparison
# =====================================================================
print("\n" + "=" * 80)
print("RETRIEVAL STRATEGY COMPARISON (chunking: 1000/100)")
print("=" * 80)

KEY_METRICS = [
    "context_recall",
    "faithfulness",
    "factual_correctness",
    "answer_relevancy",
    "context_entity_recall",
]

all_results = {
    "naive_dense": naive_results,
    "bm25": bm25_results,
    "rerank": rerank_results,
    "parent_child": parent_child_results,
    "ensemble": ensemble_results,
}

comparison_rows = []
for name, df in all_results.items():
    row: dict = {"strategy": name}
    for metric in KEY_METRICS:
        if metric in df.columns:
            valid = df[metric].dropna()
            row[metric] = float(valid.mean())
            row[f"{metric}_valid"] = int(len(valid))
    # Composite: avg of context_recall, faithfulness, answer_relevancy
    main_scores = [
        row.get("context_recall", 0.0),
        row.get("faithfulness", 0.0),
        row.get("answer_relevancy", 0.0),
    ]
    row["composite_avg"] = sum(main_scores) / len(main_scores)
    comparison_rows.append(row)

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(os.path.join(RESULTS_DIR, "retrieval_comparison.csv"), index=False)

print(f"\n{'Strategy':<16} {'ctx_recall':>12} {'faithful':>12} {'fact_corr':>12} {'relevancy':>12} {'entity_rec':>12} {'composite':>12}")
print("-" * 100)
for _, row in comparison_df.iterrows():
    print(f"{row['strategy']:<16}", end="")
    for m in KEY_METRICS:
        val = row.get(m, float("nan"))
        if pd.isna(val):
            print(f"{'NaN':>12}", end="")
        else:
            print(f"{val:>12.4f}", end="")
    print(f"{row['composite_avg']:>12.4f}")

best_idx = comparison_df["composite_avg"].idxmax()
best = comparison_df.loc[best_idx]
print(f"\nBest strategy: {best['strategy']} (composite avg: {best['composite_avg']:.4f})")
print(f"\nSaved comparison to evals/results/retrieval_comparison.csv")
