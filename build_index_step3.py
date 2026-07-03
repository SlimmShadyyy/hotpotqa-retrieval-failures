
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


print("Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.\n")


def cosine_sim(query_vec, chunk_vecs):
    query_norm = query_vec / np.linalg.norm(query_vec)
    chunk_norms = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    return chunk_norms @ query_norm


def retrieve_top_k(example, k=3):
    chunks = chunk_example(example)
    chunk_texts = [c["text"] for c in chunks]
    question_vec = model.encode(example["question"])
    chunk_vecs = model.encode(chunk_texts)
    scores = cosine_sim(question_vec, chunk_vecs)
    order = np.argsort(-scores)
    top_k_chunks = []
    for idx in order[:k]:
        c = dict(chunks[idx])
        c["score"] = float(scores[idx])
        top_k_chunks.append(c)
    return top_k_chunks


def gold_coverage(example, retrieved_chunks):
    gold_titles = get_gold_titles(example)
    retrieved_titles = {c["title"] for c in retrieved_chunks}
    hit = gold_titles & retrieved_titles
    missed = gold_titles - retrieved_titles
    return len(hit), len(gold_titles), hit, missed


print("Running retrieval over full sample at k=3...")
results = []  # list of dicts: example, retrieved, hit, missed
for ex in sample:
    top3 = retrieve_top_k(ex, k=3)
    n_hit, n_total, hit_titles, missed_titles = gold_coverage(ex, top3)
    results.append({
        "example": ex,
        "type": ex["type"],
        "n_hit": n_hit,
        "n_total": n_total,
        "hit_titles": hit_titles,
        "missed_titles": missed_titles,
        "retrieved": top3,
    })

# --- Breakdown by question type ---
print("\n" + "=" * 80)
print("RETRIEVAL QUALITY BY QUESTION TYPE (k=3)")
print("=" * 80)

for qtype in ["bridge", "comparison"]:
    subset = [r for r in results if r["type"] == qtype]
    full_hit = sum(1 for r in subset if r["n_hit"] == r["n_total"])
    partial = sum(1 for r in subset if 0 < r["n_hit"] < r["n_total"])
    zero_hit = sum(1 for r in subset if r["n_hit"] == 0)
    print(f"\n{qtype.upper()} (n={len(subset)}):")
    print(f"  Full hit:    {full_hit} ({100*full_hit/len(subset):.0f}%)")
    print(f"  Partial hit: {partial} ({100*partial/len(subset):.0f}%)")
    print(f"  Zero hit:    {zero_hit} ({100*zero_hit/len(subset):.0f}%)")

print("\n" + "=" * 80)
print("BRIDGE QUESTIONS WITH PARTIAL HIT — which chunk got missed? (first 8 examples)")
print("=" * 80)

bridge_partial = [r for r in results if r["type"] == "bridge" and 0 < r["n_hit"] < r["n_total"]]
print(f"Total bridge partial-hit cases: {len(bridge_partial)}\n")

for r in bridge_partial[:8]:
    ex = r["example"]
    print(f"Q: {ex['question']}")
    print(f"   Retrieved (hit):    {r['hit_titles']}")
    print(f"   MISSED:             {r['missed_titles']}")
    all_chunks = chunk_example(ex)
    missed_title = list(r['missed_titles'])[0]
    missed_chunk = next((c for c in all_chunks if c["title"] == missed_title), None)
    if missed_chunk:
        q_vec = model.encode(ex["question"])
        m_vec = model.encode(missed_chunk["text"])
        score = float(cosine_sim(q_vec, m_vec.reshape(1, -1))[0])
        print(f"   (missed chunk's actual similarity score was {score:.3f} - compare to the k=3 cutoff)")
    print()
