# Retrieval and Reasoning Failures in Multi-Hop QA
Track: NLP / GenAI — retrieval and reasoning-failure framing
Dataset: HotpotQA (distractor, validation split)

## Part A — Baseline

I built a simple retrieve-then-answer pipeline and ran it over a 200-question sample (150 bridge, 50
comparison). Chunking is paragraph-level — one chunk per HotpotQA context paragraph, with gold chunks
labeled via `supporting_facts`. Retrieval is dense: `all-MiniLM-L6-v2` embeddings, cosine similarity,
top-k=3. Answers come from extractive QA (`distilbert-base-cased-distilled-squad`) run over whatever got
retrieved, scored with standard SQuAD EM/F1.

## Part B — What I went looking for, and what I actually found

My hypothesis going in was that single-vector similarity retrieval would struggle on bridge questions
specifically. A bridge question only names the first entity in the chain — the second one, the one that
actually answers the question, only becomes findable once you've read the first document. It has no
reason to share vocabulary with the original question. Comparison questions don't have this problem;
both entities are named up front, so retrieval should be close to trivial by comparison.

That's basically what happened. Bridge questions hit full gold coverage only 39% of the time, versus 80%
for comparison, with a 7% zero-hit rate on bridge vs. 0% on comparison. Looking through the partial-hit
bridge cases by hand, the pattern was almost boringly consistent: the chunk that got retrieved always
shared a literal word with the question, and the chunk that got missed was always the "next fact you'd
only look for after reading hop one." The Corliss Archer example is a good illustration — the question
mentions "Kiss and Tell," so that chunk gets pulled in, but "Shirley Temple" is nowhere in the question
text and gets missed entirely.

Answer quality followed the same shape for bridge questions, which is what you'd hope to see: EM went
47% → 15% → 0% as gold coverage dropped from full to partial to zero. It's a clean chain from retrieval
failure to wrong answer, and the QA model's mistake is easy to read once you see the missing context —
without "Chief of Protocol" anywhere in front of it, it grabbed the only proper noun it had, "Shirley
Temple," and answered with that.

Here's the part I didn't expect: comparison questions retrieve well but still come out worse overall than
bridge questions (6% EM vs. 26%). So retrieval clearly isn't what's holding comparison questions back —
something else is. My guess, and I think it holds up, is that extractive QA is just the wrong tool for
this question type. It can only pull out a contiguous span of text; it has no way to actually compare two
things. "Who is older, Annie Morton or Terry Richardson?" almost certainly had both names sitting in
context, and the model still answered "American model," because there's no single span in the text that
says who's older. No amount of better retrieval fixes that — the bottleneck moved to the model itself.

One smaller thing worth flagging: at least one question labeled "bridge" ("Who is older, Annie Morton or
Terry Richardson?") is really structured like a comparison question — both names appear directly in the
question. That looks like label noise in HotpotQA's `type` field, and it's worth knowing about if anyone
uses `type` downstream for training splits or eval stratification.

Overall, the useful conclusion here isn't just "accuracy is low" — it's that there are two separate
failure modes doing the damage, and they need different fixes. Bridge questions fail at retrieval.
Comparison questions fail at generation. A single blended accuracy number would have hidden that.

## What I'd try next

1. **Query decomposition for bridge questions.** Use whatever gets retrieved for hop one to generate a
   follow-up query — pull "Shirley Temple" out of the retrieved passage, re-embed, retrieve again. Cheap
   to test and aimed directly at the failure mode I actually observed, rather than a generic fix.
2. **Swap in a generative model for the comparison subset** and see if EM improves without touching
   retrieval at all. That would confirm the retrieval/generation split directly instead of leaving it as
   an inference from the numbers.
3. **Re-run bridge questions at k=5–10** to see how much of that gap is just "the window was too small" vs.
   a real ceiling where more retrieved chunks can't help because the missing entity has no lexical
   overlap with the question to begin with.