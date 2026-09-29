# Local Kev-4B passage-selection trial — 2026-09-28

[Kev-4B](https://github.com/jaredpalmer/kev) ran through its local
`/v1/systemone` API on Apple Silicon (MLX, bfloat16). The 300 HotpotQA
distractor validation examples `[400:700]` used native contexts. Kev and
BGE-small-en-v1.5 scored the same passages under Barber's `keep=0.6` policy;
splitter, pins, lead/tail rule, markers, and token-savings guard were identical.
Kev received one `noul` question per passage, asking whether it was needed
directly or as a bridge fact. [Per-question selection records](kev_hotpotqa_2026-09-28.jsonl)
contain token counts, supporting titles, and Kev latency/usage.

| Scorer | Prompt tokens saved | Supporting passages kept | Median selection latency |
| --- | ---: | ---: | ---: |
| BGE embedding | 19.20% | 588/600 | 84 ms |
| Kev-4B | 43.61% | 587/600 | 964 ms |

Kev missed a supporting passage on 13 questions; BGE missed one on 12. Only
one question overlapped. Kev therefore recovered 11 of BGE's 12 missed
supporting passages but introduced 12 different misses. Combining both
selections kept 599/600 supports, but saved only about 17.6% of prompt tokens
before accounting for Kev inference, so that combination is not a cost win
over BGE.

For an answer-level check, local `qwen3.6:27b-mlx` answered using Kev's
selected context for all 13 Kev missed-support questions, all 12 BGE
missed-support questions, and the 12 retained-support controls from the prior
BGE evaluation. Temperature was zero and thinking disabled. On four Kev
missed-support questions (indices 468, 514, 667, 675), Kev's selected context
produced a clearly wrong answer while full context and BGE's selection
produced the gold answer. Kev's context produced the gold answer on all seven
previously observed BGE regressions. No new regression appeared in the 12
retained-support controls; three of those controls were wrong with full
context too. [Answers for all 36 distinct checked questions](kev_hotpotqa_answers_2026-09-28.jsonl)
are saved for inspection.

This is a targeted answer check, not a complete answer-level evaluation of
all 300 examples. Kev may have a better aggregate result than BGE while
still harming individual questions. It also took roughly 11 times as long to
select on this machine. Actual net dollar savings require the downstream
model's input price and Kev serving cost. Medium and large contexts have not
been tested. Do not switch Barber's default scorer from these results alone.

Jev remains untested because no TypeSafe API key is configured. The same
paired harness supports `--backend jev` once a key is available.
