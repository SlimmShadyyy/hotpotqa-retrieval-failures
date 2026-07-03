"""
Step 1: Sample selection + chunking.
Run this, look at the printed output, make sure it matches what you'd expect
before we move on to embeddings.
"""

from datasets import load_dataset

print("Loading HotpotQA (distractor, validation split)...")
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

# --- Sample selection ---
# HotpotQA questions come in two types: "bridge" (true multi-hop: fact A leads
# you to fact B) and "comparison" (compare two known entities). Bridge questions
# are the cleaner test of "does retrieval fail on multi-step questions", so we
# oversample bridge but keep some comparison too for contrast.
N_BRIDGE = 150
N_COMPARISON = 50

bridge_examples = [ex for ex in ds if ex["type"] == "bridge"][:N_BRIDGE]
comparison_examples = [ex for ex in ds if ex["type"] == "comparison"][:N_COMPARISON]

sample = bridge_examples + comparison_examples
print(f"Sample size: {len(sample)} ({len(bridge_examples)} bridge, {len(comparison_examples)} comparison)\n")


def chunk_example(example):
    """
    Turn one HotpotQA example's context into a flat list of chunk dicts.
    One chunk = one paragraph (matches HotpotQA's existing structure).

    Returns: list of {"chunk_id": str, "title": str, "text": str}
    """
    chunks = []
    titles = example["context"]["title"]
    sentences_per_doc = example["context"]["sentences"]

    for i, (title, sents) in enumerate(zip(titles, sentences_per_doc)):
        text = " ".join(sents).strip()
        chunks.append({
            "chunk_id": f"{example['id']}_{i}",
            "title": title,
            "text": text,
        })
    return chunks


def get_gold_titles(example):
    """The set of paragraph titles that are actually needed to answer the question."""
    return set(example["supporting_facts"]["title"])


# --- Sanity check: run chunking on the first example and inspect ---
ex0 = sample[0]
chunks0 = chunk_example(ex0)
gold0 = get_gold_titles(ex0)

print("=" * 80)
print(f"CHUNKING CHECK — example type: {ex0['type']}")
print("=" * 80)
print(f"Question: {ex0['question']}")
print(f"Answer:   {ex0['answer']}")
print(f"Gold titles: {gold0}\n")
print(f"Produced {len(chunks0)} chunks:")
for c in chunks0:
    is_gold = "  <-- GOLD" if c["title"] in gold0 else ""
    print(f"  [{c['chunk_id']}] {c['title']}{is_gold}")
    print(f"      {c['text'][:100]}...")

# --- Chunk the whole sample, report totals ---
all_chunks_by_example = [chunk_example(ex) for ex in sample]
total_chunks = sum(len(c) for c in all_chunks_by_example)
print(f"\nTotal chunks across full sample: {total_chunks}")
print(f"(should be ~10 * {len(sample)} = {10 * len(sample)}, since each question has 10 context paragraphs)")