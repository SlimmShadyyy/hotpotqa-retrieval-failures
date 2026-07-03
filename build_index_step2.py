"""
Step 2: Embeddings + retrieval.
Builds on Step 1 (sample selection + chunking).
For each question: embed its 10 chunks + the question, retrieve top-k by cosine
similarity, check whether the gold chunk(s) were retrieved.
"""

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading HotpotQA (distractor, validation split)...")
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

N_BRIDGE = 150
N_COMPARISON = 50
bridge_examples = [ex for ex in ds if ex["type"] == "bridge"][:N_BRIDGE]
comparison_examples = [ex for ex in ds if ex["type"] == "comparison"][:N_COMPARISON]
sample = bridge_examples + comparison_examples
print(f"Sample size: {len(sample)}\n")


def chunk_example(example):
    chunks = []
    titles = example["context"]["title"]
    sentences_per_doc = example["context"]["sentences"]
    for i, (title, sents) in enumerate(zip(titles, sentences_per_doc)):
        text = " ".join(sents).strip()
        chunks.append({"chunk_id": f"{example['id']}_{i}", "title": title, "text": text})
    return chunks


def get_gold_titles(example):
    return set(example["supporting_facts"]["title"])


# --- Load embedding model ---
print("Loading embedding model (all-MiniLM-L6-v2)... this downloads once, ~90MB.")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.\n")


def cosine_sim(query_vec, chunk_vecs):
    """query_vec: (dim,) ; chunk_vecs: (n_chunks, dim) -> returns (n_chunks,) similarity scores"""
    query_norm = query_vec / np.linalg.norm(query_vec)
    chunk_norms = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    return chunk_norms @ query_norm


def retrieve_top_k(example, k=3):
    """
    Embed the question + this example's chunks, return top-k chunks by cosine similarity.
    Returns: list of chunk dicts (with an added 'score' field), sorted best-first.
    """
    chunks = chunk_example(example)
    chunk_texts = [c["text"] for c in chunks]

    question_vec = model.encode(example["question"])
    chunk_vecs = model.encode(chunk_texts)

    scores = cosine_sim(question_vec, chunk_vecs)

    # sort chunks by score, descending
    order = np.argsort(-scores)
    top_k_chunks = []
    for idx in order[:k]:
        c = dict(chunks[idx])
        c["score"] = float(scores[idx])
        top_k_chunks.append(c)
    return top_k_chunks


def gold_coverage(example, retrieved_chunks):
    """
    Returns (num_gold_retrieved, num_gold_total, retrieved_titles, gold_titles)
    """
    gold_titles = get_gold_titles(example)
    retrieved_titles = {c["title"] for c in retrieved_chunks}
    hit = gold_titles & retrieved_titles
    return len(hit), len(gold_titles), retrieved_titles, gold_titles


# --- Sanity check on example 0 (the Corliss Archer / Shirley Temple one) ---
ex0 = sample[0]
print("=" * 80)
print("RETRIEVAL CHECK — example 0")
print("=" * 80)
print(f"Question: {ex0['question']}\n")

top3 = retrieve_top_k(ex0, k=3)
print("Top-3 retrieved chunks (by cosine similarity):")
gold_titles = get_gold_titles(ex0)
for c in top3:
    is_gold = "  <-- GOLD" if c["title"] in gold_titles else ""
    print(f"  score={c['score']:.3f}  {c['title']}{is_gold}")

n_hit, n_total, retrieved_titles, gold = gold_coverage(ex0, top3)
print(f"\nGold coverage at k=3: {n_hit}/{n_total} gold chunks retrieved")

# --- Run over the whole sample at k=3, report aggregate stats ---
print("\n" + "=" * 80)
print("AGGREGATE RETRIEVAL QUALITY over full sample, k=3")
print("=" * 80)

coverage_counts = {0: 0, 1: 0, 2: 0}  # how many questions had 0, 1, or 2 gold chunks retrieved
for ex in sample:
    top3 = retrieve_top_k(ex, k=3)
    n_hit, n_total, _, _ = gold_coverage(ex, top3)
    # cap at 2 for the histogram (comparison questions always have exactly 2 gold;
    # bridge questions can vary but usually 2 as well)
    coverage_counts[min(n_hit, 2)] += 1

print(f"Questions with 0/N gold retrieved: {coverage_counts[0]}")
print(f"Questions with 1/N gold retrieved: {coverage_counts[1]}")
print(f"Questions with 2/N gold retrieved (full hit): {coverage_counts[2]}")
print(f"\n(This is your baseline retrieval quality number at k=3. Note it down.)")