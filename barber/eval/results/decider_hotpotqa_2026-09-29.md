# Decider-4B Q4_K_M passage-selection trial — 2026-09-29

[Mapika/decider-4b-GGUF](https://huggingface.co/Mapika/decider-4b-GGUF), file
`decider-4b-v2.1-Q4_K_M.gguf`, ran through `decider-ai` 1.6.0 and
`llama-cpp-python` 0.3.35 with Metal on an M5 Pro. The model reads a shared
question and numbered passages, then scores one `noul` question per passage
in a single `system_one(..., independent=False)` call. The `n_ctx` setting was
4096. Source revision: `23579f7a`; model snapshot: `b79f09d9`. This uses
Decider's typed probability output; ordinary Ollama text
generation would not score the model correctly.

The paired trial used HotpotQA distractor validation examples `[400:700]` with
their native contexts. All scorers used Barber's public transform with
`keep=0.6` and the same chunking, pins, markers, and no-negative-savings guard.
[Per-question records](decider_hotpotqa_2026-09-29.jsonl) contain token counts,
supporting titles, latency, and scorer input usage.

| Scorer | Prompt tokens saved | Supporting passages kept | Median selection latency |
| --- | ---: | ---: | ---: |
| BGE-small-en-v1.5 | 19.20% | 588/600 | 84 ms |
| Kev-4B BF16, MLX | 43.61% | 587/600 | 964 ms |
| Decider-4B Q4_K_M, Metal | 44.42% | 595/600 | 1440 ms |

Decider missed a supporting passage on five questions: 453, 537, 649, 663,
and 667. Four overlap Kev's 13 missed-support questions; 537 is new. It kept
the support passages on nine questions Kev missed. Relative to Kev, Decider
saved only 3,223 more downstream prompt tokens across 300 examples while
spending 145 more seconds on selection and reading 69,761 more scorer tokens.
Total Decider selection time was 434.5 seconds, versus 289.4 for Kev and 27.7
for BGE. No net dollar savings can be claimed without the downstream input
price and local GPU cost.

For an answer-level check, local `qwen3.6:27b-mlx` answered from Decider's
selected context on all 36 previously checked questions plus new miss 537.
Temperature was zero and thinking disabled. Full and BGE contexts were also
checked for 537. Decider's answer became clearly wrong relative to both full
context and BGE on 537 (Melbourne → insufficient information) and 667
(Wikimedia Foundation → insufficient information). It answered correctly on
the other three missed-support questions and on three of Kev's four previously
observed clear regressions (468, 514, 675). The remaining Kev regression,
667, persisted. [Per-question answers](decider_hotpotqa_answers_2026-09-29.jsonl)
are saved for inspection.

The targeted check found individual regressions, but cannot establish the
overall quality difference. We therefore generated [paired BGE and Decider
answers](decider_vs_bge_answers_2026-09-29.jsonl) for **all 300** examples with
the same local `qwen3.6:27b-mlx` model, temperature zero, and thinking disabled.
Using HotpotQA-style lowercasing, punctuation/article removal, and token F1
against the reference answer:

| Selected context | Exact answers | Mean answer F1 |
| --- | ---: | ---: |
| BGE | 147/300 | 0.6574 |
| Decider Q4_K_M | 154/300 | 0.6862 |

Decider's paired F1 advantage was 0.0288; a 10,000-resample paired bootstrap
gave a 95% interval of approximately +0.0018 to +0.0563. It won 34 questions,
lost 14, and tied 252 by F1. Token F1 is imperfect for verbose answers: some
large numeric differences were equivalent answers with different phrasing.
Manual review of large differences found real Decider wins (for example 469,
496, 564, 606) and losses (411, 537, 667).

On these native contexts, Decider reduced prompt tokens **and improved measured
aggregate answer quality** relative to BGE. It is now available as the
explicit `barber.embedders.decider()` option. The zero-dependency lexical
default remains appropriate for users who have not chosen to download a 2.7 GB
model and run local inference. Medium and large contexts remain untested.
Whether Decider reduces *total* dollars depends on the downstream input price
and local GPU cost: against BGE it saved 100,997 additional prompt tokens but
used 406.8 more seconds of scorer time over these 300 examples.
