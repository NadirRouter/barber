# Local HotpotQA evaluation — 2026-09-28

Cached HotpotQA distractor validation examples, with padding from validation
`[2000:2600]`. Selection used BGE-small-en-v1.5, `keep=0.6`, and the public
Barber transform. Tokens use `o200k_base`. Supporting passage retention checks
numbered passage headings, not mentions of a title elsewhere in the context.

| Context | Slice | Token saving | Supporting passages retained | Cost-guard activations |
| --- | ---: | ---: | ---: | ---: |
| Native (~1.3K tokens) | 400:700 | 19.16% | 588/600 | 0/300 |
| Medium (~4K tokens) | 400:600 | 31.39% | 400/400 | 0/200 |
| Large (~12K tokens) | 400:500 | 34.61% | 200/200 | 0/100 |

For answer quality, local `qwen3.6:27b-mlx` generated full-context and trimmed
answers at temperature 0 with thinking disabled. The sample included all 12
native examples where a supporting passage was dropped, plus 12 randomly chosen
retained-support examples (seed 13). Answers were checked against HotpotQA's
reference answers. Seven of the 12 dropped-support cases were clear regressions;
none of the 12 controls regressed. Thus at least 7 of the original 300 native
examples (2.33%) regressed for this generator. The other 288 were not all
answer-graded. [Per-question answers and grades](local_hotpotqa_2026-09-28.jsonl)
include one corrected local-judge error (index 608) and note the rerun of a
truncated answer (index 631).

On the same 300 native examples, `keep=0.8` saved 6.18% and retained 597/600
supporting passages. `keep=0.9` saved 0.52% and retained 600/600. An 8,000
character minimum saved 1.72% and retained 600/600 there, but a disjoint
600-example native slice retained only 1199/1200 at that minimum (1.97% saved).
It is not a validated quality-safe cutoff.

The cost guard was also exercised on four synthetic, marker-heavy contexts. It
avoided prompt expansion in all four; the local model answered correctly with
both the old selection and the guarded full context. Those checks do not prove
that retaining extra distractors is harmless on arbitrary tasks. The medium
and large slices above measure passage retention, not answer-level quality.
