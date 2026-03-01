"""Run RAGAS baseline evaluation with aggressive caching.

Caches at every stage so re-runs are cheap:
  1. knowledge_graph.json  — KG transforms (~$3 to build, $0 to reload)
  2. synthetic_testset.csv — SDG questions (~$0.20, $0 to reload)
  3. baseline_responses.csv — RAG answers (~$0.05, $0 to reload)
  4. baseline_per_query.csv — RAGAS scores (~$0.50, $0 to reload)

Usage:
    uv run python evals/run_baseline_eval.py           # use all caches
    uv run python evals/run_baseline_eval.py --fresh    # rebuild everything
"""

import os
import sys

import nest_asyncio
from dotenv import load_dotenv

# Fix event loop for Python 3.13 + RAGAS async internals
nest_asyncio.apply()

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

load_dotenv(os.path.join(project_root, ".env"))

# Disable LangSmith tracing for eval runs
os.environ["LANGSMITH_TRACING"] = "false"

EVALS_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(EVALS_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
FRESH = "--fresh" in sys.argv

# --- Paths for cached artifacts ---
KG_PATH = os.path.join(EVALS_DIR, "knowledge_graph.json")
TESTSET_PATH = os.path.join(EVALS_DIR, "synthetic_testset.csv")
RESPONSES_PATH = os.path.join(RESULTS_DIR, "baseline_responses.csv")
RESULTS_PATH = os.path.join(RESULTS_DIR, "baseline_per_query.csv")


def _cache_exists(path: str) -> bool:
    return os.path.exists(path) and not FRESH


# =====================================================================
# Steps 1-2: Knowledge Graph + Synthetic Test Set
# Skip KG entirely if testset is already cached (KG is only used for SDG)
# =====================================================================
import pandas as pd

if _cache_exists(TESTSET_PATH):
    print(f"Loading cached test set from {TESTSET_PATH}")
    test_df = pd.read_csv(TESTSET_PATH)
    print(f"  Loaded {len(test_df)} test samples (skipping KG build)")
else:
    from ragas.testset.graph import KnowledgeGraph
    from ragas.testset.transforms import default_transforms, apply_transforms
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import RunConfig
    from ragas.testset.persona import Persona
    from ragas.testset import TestsetGenerator

    # Use gpt-4.1-mini for SDG (5x cheaper than gpt-4.1, quality is sufficient)
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
    generator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    # Step 1: Build or load KnowledgeGraph
    if _cache_exists(KG_PATH):
        print(f"Loading cached knowledge graph from {KG_PATH}")
        kg = KnowledgeGraph.load(KG_PATH)
        print(f"  Loaded KG with {len(kg.nodes)} nodes")
    else:
        from rag.loader import load_knowledge_files

        docs = load_knowledge_files()
        print(f"Loaded {len(docs)} documents")

        print("Building knowledge graph (this is the expensive step)...")
        kg = KnowledgeGraph()
        kg.add(docs)
        transforms = default_transforms(
            documents=docs, llm=generator_llm, embedding_model=generator_embeddings
        )
        apply_transforms(kg, transforms, run_config=RunConfig(timeout=360, max_retries=5))
        kg.save(KG_PATH)
        print(f"  Saved KG ({len(kg.nodes)} nodes) to {KG_PATH}")

    # Step 2: Generate synthetic test set
    personas = [
        Persona(
            name="Deep Autumn Professional",
            role_description=(
                "A male professional with warm, dark skin tone and an inverted triangle body shape. "
                "He is a Deep Autumn color season and prefers classic, structured clothing. "
                "He works in a business casual office and wants personalized styling advice "
                "grounded in color theory and body shape principles."
            ),
        ),
        Persona(
            name="Style Beginner",
            role_description=(
                "Someone new to fashion who doesn't know their color season or body shape yet. "
                "They ask basic questions about how to dress well, build a capsule wardrobe, "
                "and understand dress codes. They need foundational styling education."
            ),
        ),
        Persona(
            name="Occasion Dresser",
            role_description=(
                "Someone who has specific events coming up — job interviews, dates, weddings, "
                "casual weekends — and wants occasion-specific outfit guidance. They care about "
                "dress codes, formality levels, and making the right impression."
            ),
        ),
    ]

    print(f"\nGenerating synthetic test set with {len(personas)} personas...")
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg,
        persona_list=personas,
    )
    dataset = generator.generate(testset_size=20)
    test_df = dataset.to_pandas()
    test_df.to_csv(TESTSET_PATH, index=False)
    print(f"  Generated {len(test_df)} samples, saved to {TESTSET_PATH}")

print("\nSample questions:")
for _, row in test_df[["user_input"]].head(5).iterrows():
    print(f"  - {row['user_input'][:100]}")


# =====================================================================
# Steps 3-4: Baseline Responses + RAGAS Evaluation
# Skip responses if final results are already cached
# =====================================================================
if _cache_exists(RESULTS_PATH):
    print(f"\nLoading cached evaluation results from {RESULTS_PATH}")
    results_df = pd.read_csv(RESULTS_PATH)
else:
    # --- Step 3: Run baseline RAG or load cached responses ---
    if _cache_exists(RESPONSES_PATH):
        print(f"\nLoading cached baseline responses from {RESPONSES_PATH}")
        responses_df = pd.read_csv(RESPONSES_PATH)
        print(f"  Loaded {len(responses_df)} responses")
    else:
        from rag.loader import load_knowledge_files
        from rag.chunking import chunk_documents
        from rag.vectorstore import create_vector_store
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langgraph.graph import START, StateGraph
        from typing_extensions import List, TypedDict
        from langchain_core.documents import Document

        # Build vector store (in-memory, cheap — only uses embeddings)
        docs = load_knowledge_files()
        chunks = chunk_documents(docs)
        vector_store = create_vector_store(chunks)
        baseline_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

        RAG_PROMPT = """You are a personal AI stylist. Answer the question using ONLY the provided context.

### Question
{question}

### Context
{context}
"""
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        rag_llm = ChatOpenAI(model="gpt-4.1-mini")

        class RAGState(TypedDict):
            question: str
            context: List[Document]
            response: str

        def retrieve_baseline(state):
            return {"context": baseline_retriever.invoke(state["question"])}

        def generate(state):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = rag_prompt.format_messages(
                question=state["question"], context=docs_content
            )
            return {"response": rag_llm.invoke(messages).content}

        baseline_graph = StateGraph(RAGState).add_sequence([retrieve_baseline, generate])
        baseline_graph.add_edge(START, "retrieve_baseline")
        baseline_graph = baseline_graph.compile()

        # Sanity check
        print("\nSanity check...")
        test_response = baseline_graph.invoke(
            {"question": "What colors suit a Deep Autumn?"}
        )
        print(f"  Response: {test_response['response'][:150]}...")
        print(f"  Retrieved {len(test_response['context'])} chunks")

        # Run all test queries
        print(f"\nRunning baseline on {len(test_df)} queries...")
        results = []
        for i, (_, row) in enumerate(test_df.iterrows()):
            response = baseline_graph.invoke({"question": row["user_input"]})
            results.append(
                {
                    "user_input": row["user_input"],
                    "reference": row["reference"],
                    "reference_contexts": row.get("reference_contexts", ""),
                    "response": response["response"],
                    "retrieved_contexts": "\n---\n".join(
                        ctx.page_content for ctx in response["context"]
                    ),
                }
            )
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(test_df)} queries")

        responses_df = pd.DataFrame(results)
        responses_df.to_csv(RESPONSES_PATH, index=False)
        print(f"  Saved {len(responses_df)} responses to {RESPONSES_PATH}")

    # --- Step 4: Evaluate with RAGAS ---
    from ragas import EvaluationDataset, evaluate, RunConfig
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    from ragas.metrics import (
        LLMContextRecall,
        Faithfulness,
        FactualCorrectness,
        ResponseRelevancy,
        ContextEntityRecall,
    )

    samples = []
    for _, row in responses_df.iterrows():
        retrieved = row["retrieved_contexts"]
        contexts = retrieved.split("\n---\n") if isinstance(retrieved, str) else []
        samples.append(
            SingleTurnSample(
                user_input=row["user_input"],
                response=row["response"],
                reference=row["reference"],
                retrieved_contexts=contexts,
            )
        )
    eval_dataset = EvaluationDataset(samples=samples)

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))

    print(f"\nEvaluating {len(samples)} samples with RAGAS...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            LLMContextRecall(),
            Faithfulness(),
            FactualCorrectness(),
            ResponseRelevancy(),
            ContextEntityRecall(),
        ],
        llm=evaluator_llm,
        run_config=RunConfig(timeout=360),
    )

    results_df = result.to_pandas()
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"  Saved results to {RESULTS_PATH}")

# =====================================================================
# Print results
# =====================================================================
print("\n" + "=" * 60)
print("BASELINE RESULTS (Naive Dense Retrieval, k=10)")
print("=" * 60)
for col in results_df.select_dtypes(include="number").columns:
    valid = results_df[col].dropna()
    print(f"  {col:40s} {valid.mean():.4f}  ({len(valid)}/{len(results_df)} valid)")

print(f"\nCached artifacts in {EVALS_DIR}/:")
for path in [KG_PATH, TESTSET_PATH, RESPONSES_PATH, RESULTS_PATH]:
    name = os.path.basename(path)
    exists = "✓" if os.path.exists(path) else "✗"
    print(f"  {exists} {name}")

print("\nDone! Re-run to use caches. Use --fresh to rebuild everything.")
