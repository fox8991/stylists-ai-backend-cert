"""Run chunking experiments with RAGAS evaluation.

Tests 3 chunking configurations with naive dense retrieval (k=10),
evaluates each with RAGAS, and picks the best one.

Usage:
    uv run python evals/run_chunking_experiment.py
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

# --- Shared components ---
rag_llm = ChatOpenAI(model="gpt-4.1-mini")
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
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
    return df


# --- Chunking configurations ---
chunking_configs = [
    {"name": "small_250_50", "chunk_size": 250, "chunk_overlap": 50},
    {"name": "medium_500_50", "chunk_size": 500, "chunk_overlap": 50},
    {"name": "large_1000_100", "chunk_size": 1000, "chunk_overlap": 100},
]

all_results = {}

for config in chunking_configs:
    name = config["name"]
    cache_path = os.path.join(RESULTS_DIR, f"chunking_{name}.csv")

    if os.path.exists(cache_path):
        print(f"\nLoading cached results for {name} from {cache_path}")
        all_results[name] = pd.read_csv(cache_path)
        continue

    chunks = chunk_documents(docs, config["chunk_size"], config["chunk_overlap"])
    print(f"\n  Chunking: {name} -> {len(chunks)} chunks")

    vs = create_vector_store(chunks, collection_name=f"chunking_{name}")
    retriever = vs.as_retriever(search_kwargs={"k": 10})

    df = run_experiment(retriever.invoke, name)
    df.to_csv(cache_path, index=False)
    print(f"  Saved to {cache_path}")
    all_results[name] = df

# --- Comparison ---
print("\n" + "=" * 70)
print("CHUNKING EXPERIMENT COMPARISON")
print("=" * 70)

KEY_METRICS = [
    "context_recall",
    "faithfulness",
    "factual_correctness",
    "answer_relevancy",
    "context_entity_recall",
    "noise_sensitivity",
]

comparison_rows = []
for name, df in all_results.items():
    row = {"config": name}
    for metric in KEY_METRICS:
        if metric in df.columns:
            valid = df[metric].dropna()
            row[metric] = valid.mean()
            row[f"{metric}_valid"] = len(valid)
    # Composite score (avg of 4 main metrics)
    main = [row.get(m, 0) for m in KEY_METRICS[:4]]
    row["composite_avg"] = sum(main) / len(main) if main else 0
    comparison_rows.append(row)

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(os.path.join(RESULTS_DIR, "chunking_comparison.csv"), index=False)

print(f"\n{'Config':<20}", end="")
for m in KEY_METRICS[:4]:
    print(f"{m:>20}", end="")
print(f"{'composite':>12}")
print("-" * 92)

for _, row in comparison_df.iterrows():
    print(f"{row['config']:<20}", end="")
    for m in KEY_METRICS[:4]:
        val = row.get(m, float("nan"))
        print(f"{val:>20.4f}", end="")
    print(f"{row['composite_avg']:>12.4f}")

best = comparison_df.loc[comparison_df["composite_avg"].idxmax()]
print(f"\nBest chunking: {best['config']} (composite avg: {best['composite_avg']:.4f})")
print(f"\nSaved comparison to evals/chunking_comparison.csv")
print("Individual results saved to evals/chunking_<name>.csv")
