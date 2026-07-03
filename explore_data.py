
import json
from datasets import load_dataset

print("Loading HotpotQA (distractor, validation split)...")
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
print(f"Loaded {len(ds)} examples.\n")

example = ds[0]
print("=" * 80)
print("FULL RAW EXAMPLE (example 0)")
print("=" * 80)
print(json.dumps(example, indent=2)[:3000]) 
print("...\n")

print("=" * 80)
print("DECODED VIEW")
print("=" * 80)

print(f"QUESTION: {example['question']}")
print(f"ANSWER:   {example['answer']}\n")

titles = example["context"]["title"]
sentences_per_doc = example["context"]["sentences"]

print(f"Number of context paragraphs (should be 10): {len(titles)}\n")

for i, (title, sents) in enumerate(zip(titles, sentences_per_doc)):
    joined = " ".join(sents)
    print(f"[{i}] TITLE: {title}")
    print(f"    TEXT: {joined[:150]}{'...' if len(joined) > 150 else ''}\n")

print("=" * 80)
print("SUPPORTING FACTS (the answer key — which paragraphs actually matter)")
print("=" * 80)
gold_titles = example["supporting_facts"]["title"]
gold_sent_ids = example["supporting_facts"]["sent_id"]

for t, sid in zip(gold_titles, gold_sent_ids):
    print(f"  Gold paragraph title: '{t}'  -> relevant sentence index: {sid}")

print(f"\nSo out of {len(titles)} paragraphs, only {len(set(gold_titles))} are gold.")
print("The rest are distractors: similar-sounding but irrelevant to the answer.")

print("\n" + "=" * 80)
print("TWO MORE EXAMPLES (compact view)")
print("=" * 80)
for idx in [1, 2]:
    ex = ds[idx]
    print(f"\n--- Example {idx} ---")
    print(f"Q: {ex['question']}")
    print(f"A: {ex['answer']}")
    print(f"All paragraph titles: {ex['context']['title']}")
    print(f"Gold titles (supporting_facts): {list(set(ex['supporting_facts']['title']))}")
