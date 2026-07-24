# Benchmark methodology

How barber's defaults were locked, and how to reproduce every number in the
README. The full write-up is on the Nadir blog:
[Context selection, benchmarked with MiniMax as the judge](https://getnadir.com/blog/context-selection-benchmark-minimax.html).

## Method in one line

Answer each question twice, once with full context and once with trimmed
context, then have a blind LLM judge grade both answers against the
known-correct reference. The number that governs shipping is the regression
rate: how often trimming turned a right answer wrong.

## Dataset

HotpotQA, `distractor` config (HuggingFace `hotpotqa/hotpot_qa`, validation
split). Each example has 10 paragraphs (2 gold + 8 distractors), a short gold
answer, and supporting-fact labels. One dataset gives you regression rate and
gold-paragraph recall in the same run.

Native contexts are ~1.3K tokens. Real RAG traffic is bigger, so the harness
builds larger haystacks by padding with real Wikipedia paragraphs drawn from
other examples (true distractors), then shuffling so the gold paragraphs land
at random positions. This is the standard lost-in-the-middle protocol: the gold
answer and grading are unchanged, only the haystack grows.

| Size | Target padding | Measured average context |
|---|---|---|
| small | none (native) | ~1.3K tokens |
| medium | 4,000 tokens | ~6K tokens |
| large | 12,000 tokens | ~14K tokens |

## Roles

- **Generator:** MiniMax M3, temperature 0. Answers each question twice, once
  per context condition, with "answer using ONLY the context" instructions.
- **Judge:** MiniMax M3, blind. Sees the question, the reference answer, and
  the two answers labeled A and B in randomized order. It is never told which
  answer came from trimmed context. The harness records the mapping and
  un-blinds after grading.
- **Selector:** barber, `keep=0.6`, all guards on. Published runs used the
  `llm-semantic-router/mmbert-embed-32k-2d-matryoshka` embedder (chosen for its
  32K context window; `BAAI/bge-small-en-v1.5` tied with it on quality).

## Metrics

- **Regression rate = P(trimmed wrong | full right).** The only number that can
  hurt a user: trimming broke an answer that full context got right. Gate on
  this. Target: under 1 to 2%.
- **Full vs trimmed accuracy.** Should be within noise if trimming is safe.
- **Gold-paragraph recall.** Did selection keep the supporting paragraphs?
  Cheap upper bound; if recall drops, regression follows.
- **Tokens saved %.** The payoff, measured with tiktoken (`o200k_base`).
- **Materially-worse rate.** The blind judge's direct A/B call, as a
  cross-check on the regression number.

Read the results as a frontier: at each `keep` setting, plot tokens saved
against regression rate. Your safe operating point is the most aggressive
setting where regression stays under your bar.

## Published results

HotpotQA distractor config, about 350 judged pairs after filtering, MiniMax M3
as blind judge, `keep=0.6`:

| | Medium (~6K tok) | Large (~14K tok) |
|---|---|---|
| Tokens saved | 31.8% | 34.1% |
| Answer-paragraph retention | 100% | 100% |
| Full-context accuracy | 97.2% | 94.8% |
| Trimmed-context accuracy | 96.0% | 95.9% |

On large contexts the trimmed condition scored higher than full context:
selection removes the distractors models trip over. The entire eval, roughly
2,000 model calls, ran on well under $20 of MiniMax credit.

## Reproduce

```bash
pip install "barber-llm[eval]"
export JUDGE_API_KEY=sk-...        # MiniMax key (MINIMAX_API_KEY also honored)

# sanity-check the generator + judge first (half the cost of the A/B)
barber-eval --baseline --n 25 --size small

# the published large-context run
barber-eval --n 200 --keep 0.6 --size large
```

Judge and generator are configurable via env vars, so you can point either at
any OpenAI compatible endpoint:

| Variable | Default | Purpose |
|---|---|---|
| `JUDGE_API_KEY` | (required; `MINIMAX_API_KEY` honored) | judge credentials |
| `JUDGE_BASE_URL` | `https://api.minimax.io/v1` | judge endpoint |
| `JUDGE_MODEL` | `MiniMax-M3` | judge model |
| `GEN_API_KEY` | judge key | generator credentials |
| `GEN_BASE_URL` | judge base URL | generator endpoint |
| `GEN_MODEL` | `MiniMax-M3` | generator model |
| `ST_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | `--embedder st` model |
| `VLLM_EMBED_BASE_URL` | `http://localhost:8000/v1` | `--embedder vllm` endpoint |
| `VLLM_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | `--embedder vllm` model |

Determinism notes: the harness seeds Python's RNG (`random.seed(13)`), pads
each example with a per-example RNG derived from the question hash, and slices
the dataset with `--offset` so repeat runs and disjoint slices are cheap.
`--marker neutral` re-runs the drop-marker ablation against the shipped
assertive wording.

## Judge prompt

The exact judge prompts ship in the package:
[`barber/eval/JUDGE_PROMPT.md`](../barber/eval/JUDGE_PROMPT.md). Grading is
reference-based (not pairwise vibes), blind, and order-randomized, with a
`materially_worse` field that separates content regressions from style
differences.

## Running it on your own data

The harness is HotpotQA-specific, but the pieces are not. To gate barber on
your traffic: sample real requests, split each into (context, question), run
`barber.trim` on the context, generate both answers with your production
model, and grade with the judge prompt above (use the reference-free variant if
you have no gold answers). Gate the rollout on regression rate, not on tokens
saved.
