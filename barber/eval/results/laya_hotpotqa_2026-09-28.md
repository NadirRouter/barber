# Local Laya passage-selection trial — 2026-09-28

On HotpotQA distractor validation `[400:700]`, native contexts, Barber kept the
same paragraph splitter, `keep=0.6`, pins, lead/tail rule, markers, and token
savings guard. BGE-small-en-v1.5 and Laya scored the same passages. Laya 0.3.21
used its English checkpoint on Apple MPS, one `noul` question per passage:
"Does this passage help answer the question, directly or as a bridge fact?"
Passages were batched eight at a time. The [per-question records](laya_hotpotqa_2026-09-28.jsonl)
include the selected token counts, supporting titles, and Laya usage/latency.

| Scorer | Prompt tokens saved | Supporting passages kept | Median selection latency |
| --- | ---: | ---: | ---: |
| BGE embedding | 19.20% | 588/600 | 71 ms |
| Laya English | 30.37% | 559/600 | 491 ms |

Laya missed 41 supporting passages, including five also missed by BGE. It
recovered seven of BGE's twelve missed supports, but added 36 new misses.
These are passage-retention results, not answer grades; keeping a supporting
passage is necessary but does not prove answer quality.

On the first 25 examples, rewording Laya's instruction to include
"intermediate or bridge facts" improved the English checkpoint from 45/50 to
46/50 supports and saved 24.14%. The `typed-decisions` checkpoint with that
wording kept 47/50 and saved 19.04%. BGE kept 50/50 on that slice. Prompt
changes therefore did not close the quality gap in this pilot.

For comparison, the Apache-licensed MS MARCO MiniLM-L6-v2 reranker kept
554/600 supports on the 300-example slice while saving 39.96% of prompt tokens.
Qwen3-Reranker-0.6B kept 49/50 on the first 25 examples, the same count as
MiniLM and below BGE's 50/50, at roughly 640 ms median scoring latency.
Neither qualified for a quality-preserving switch. Jina reranker v3.5 was
excluded because its weights are licensed for noncommercial use.

Kev-4B is another Apache-licensed, Jev-compatible decision model that can run
locally. See its [Barber trial](kev_hotpotqa_2026-09-28.md) for paired
passage-selection and targeted answer results.

Jev was not compared yet: no TypeSafe API key is configured. The Jev arm of
`barber.eval.jev_trial` is ready for a paired run once a key is available.
