"""
Step 4: Generation + answer quality.
Builds on Steps 1-3 (chunking, retrieval, bridge/comparison breakdown).

New idea: retrieval quality is only useful insofar as it affects the final
answer. So for each question, we take the top-k retrieved chunks, concatenate
them into a context, run an extractive QA model over that context, and score
the predicted answer against the gold answer (EM / F1, the standard SQuAD
metrics). Then we check whether answer accuracy tracks gold coverage
(full hit / partial hit / zero hit) and question type (bridge / comparison).
"""

import re
import string
import collections

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
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
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")


def cosine_sim(query_vec, chunk_vecs):
    query_norm = query_vec / np.linalg.norm(query_vec)
    chunk_norms = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    return chunk_norms @ query_norm


def retrieve_top_k(example, k=3):
    chunks = chunk_example(example)
    chunk_texts = [c["text"] for c in chunks]
    question_vec = embed_model.encode(example["question"])
    chunk_vecs = embed_model.encode(chunk_texts)
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


# --- New: load a small extractive QA model ---
# distilbert-base-cased-distilled-squad: fine-tuned on SQuAD, runs fine on CPU.
# Extractive means it picks a span out of the given context - it can only be
# as right as the context you feed it, which is exactly what we want to test.
#
# Note: we load the tokenizer/model directly instead of using
# transformers.pipeline("question-answering", ...). As of transformers==5.3,
# HuggingFace removed the "question-answering" pipeline task shortcut (see
# https://github.com/huggingface/course/issues/1211). Calling the model
# directly does the same thing the pipeline did internally, and isn't
# affected by that change.
print("Loading QA model (distilbert-base-cased-distilled-squad)... this downloads once, ~260MB.")
qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
qa_model.eval()
print("QA model loaded.\n")


def answer_from_retrieved(example, retrieved_chunks):
    """
    Concatenate retrieved chunks into one context string, run extractive QA
    over it, return the predicted answer string (empty string if the model
    finds nothing / context is empty).
    """
    context = " ".join(c["text"] for c in retrieved_chunks).strip()
    if not context:
        return ""

    inputs = qa_tokenizer(
        example["question"],
        context,
        return_tensors="pt",
        truncation="only_second",  # never truncate the question, only the context
        max_length=384,
    )
    with torch.no_grad():
        outputs = qa_model(**inputs)

    start = int(torch.argmax(outputs.start_logits))
    end = int(torch.argmax(outputs.end_logits)) + 1
    if end <= start:
        return ""

    answer_ids = inputs["input_ids"][0][start:end]
    answer = qa_tokenizer.decode(answer_ids, skip_special_tokens=True)
    return answer.strip()


# --- SQuAD-style normalization + EM/F1 (standard, not something to tweak) ---
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return int(pred_tokens == gold_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# --- Sanity check on example 0 (Corliss Archer / Shirley Temple) ---
ex0 = sample[0]
top3_0 = retrieve_top_k(ex0, k=3)
n_hit0, n_total0, hit0, missed0 = gold_coverage(ex0, top3_0)
pred0 = answer_from_retrieved(ex0, top3_0)

print("=" * 80)
print("GENERATION CHECK — example 0")
print("=" * 80)
print(f"Question: {ex0['question']}")
print(f"Gold answer:      {ex0['answer']}")
print(f"Predicted answer: {pred0}")
print(f"Gold coverage: {n_hit0}/{n_total0}  (missed: {missed0 if missed0 else 'none'})")
print(f"EM: {exact_match(pred0, ex0['answer'])}   F1: {f1_score(pred0, ex0['answer']):.2f}\n")

# --- Run over the whole sample at k=3 ---
print("=" * 80)
print("Running retrieval + generation over full sample at k=3...")
print("(this is the slow part - embedding + QA inference per question)")
print("=" * 80)

results = []
for i, ex in enumerate(sample):
    top3 = retrieve_top_k(ex, k=3)
    n_hit, n_total, hit_titles, missed_titles = gold_coverage(ex, top3)
    pred = answer_from_retrieved(ex, top3)
    em = exact_match(pred, ex["answer"])
    f1 = f1_score(pred, ex["answer"])

    if n_hit == n_total:
        coverage_bucket = "full"
    elif n_hit == 0:
        coverage_bucket = "zero"
    else:
        coverage_bucket = "partial"

    results.append({
        "example": ex,
        "type": ex["type"],
        "coverage_bucket": coverage_bucket,
        "n_hit": n_hit,
        "n_total": n_total,
        "pred": pred,
        "em": em,
        "f1": f1,
    })

    if (i + 1) % 50 == 0:
        print(f"  ...{i + 1}/{len(sample)} done")


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


# --- Answer quality by gold coverage bucket (the headline result) ---
print("\n" + "=" * 80)
print("ANSWER QUALITY BY GOLD COVERAGE (k=3)")
print("=" * 80)
for bucket in ["full", "partial", "zero"]:
    subset = [r for r in results if r["coverage_bucket"] == bucket]
    if not subset:
        continue
    print(f"{bucket.upper():8s} (n={len(subset):3d}): "
          f"EM={avg([r['em'] for r in subset]):.2f}  "
          f"F1={avg([r['f1'] for r in subset]):.2f}")

# --- Answer quality by question type ---
print("\n" + "=" * 80)
print("ANSWER QUALITY BY QUESTION TYPE (k=3)")
print("=" * 80)
for qtype in ["bridge", "comparison"]:
    subset = [r for r in results if r["type"] == qtype]
    print(f"{qtype.upper():10s} (n={len(subset):3d}): "
          f"EM={avg([r['em'] for r in subset]):.2f}  "
          f"F1={avg([r['f1'] for r in subset]):.2f}")

# --- Cross-tab: bridge questions specifically, by coverage bucket ---
print("\n" + "=" * 80)
print("BRIDGE QUESTIONS ONLY: ANSWER QUALITY BY GOLD COVERAGE (k=3)")
print("=" * 80)
bridge_results = [r for r in results if r["type"] == "bridge"]
for bucket in ["full", "partial", "zero"]:
    subset = [r for r in bridge_results if r["coverage_bucket"] == bucket]
    if not subset:
        print(f"{bucket.upper():8s} (n=0): -")
        continue
    print(f"{bucket.upper():8s} (n={len(subset):3d}): "
          f"EM={avg([r['em'] for r in subset]):.2f}  "
          f"F1={avg([r['f1'] for r in subset]):.2f}")

# --- Overall ---
print("\n" + "=" * 80)
print("OVERALL")
print("=" * 80)
print(f"All questions (n={len(results)}): "
      f"EM={avg([r['em'] for r in results]):.2f}  "
      f"F1={avg([r['f1'] for r in results]):.2f}")

# --- A few concrete failure examples: partial/zero coverage bridge questions where EM=0 ---
print("\n" + "=" * 80)
print("CONCRETE FAILURE EXAMPLES — bridge, incomplete coverage, wrong answer (first 5)")
print("=" * 80)
failures = [
    r for r in bridge_results
    if r["coverage_bucket"] in ("partial", "zero") and r["em"] == 0
]
print(f"Total such cases: {len(failures)}\n")
for r in failures[:5]:
    ex = r["example"]
    print(f"Q: {ex['question']}")
    print(f"   Gold answer:      {ex['answer']}")
    print(f"   Predicted answer: {r['pred']}")
    print(f"   Coverage: {r['n_hit']}/{r['n_total']} ({r['coverage_bucket']})")
    print()