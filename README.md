<p align="center">
  <img src="https://raw.githubusercontent.com/NadirRouter/barber/main/assets/barber-readme-hero.png" alt="barber: your context could use a trim" width="100%">
</p>

# barber

Query-aware context trimming for LLM requests.

**Your context could use a trim.** *33% off the top.*

[![PyPI](https://img.shields.io/pypi/v/barber-llm)](https://pypi.org/project/barber-llm/)
[![npm](https://img.shields.io/npm/v/barber-llm)](https://www.npmjs.com/package/barber-llm)
[![CI](https://github.com/NadirRouter/barber/actions/workflows/ci.yml/badge.svg)](https://github.com/NadirRouter/barber/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/barber-llm/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/NadirRouter/barber/blob/main/LICENSE)
[![Deps](https://img.shields.io/badge/dependencies-zero-brightgreen)](https://github.com/NadirRouter/barber/blob/main/pyproject.toml)

## The 30-second pitch

Retrieved context is mostly irrelevant to any single question. Your RAG stack
fetches ten paragraphs, the question needs two, and you pay for all ten on
every request.

barber embeds the chunks and the question, keeps what matters, and drops the
rest. Nothing is rewritten or summarized: chunks survive verbatim or vanish.
No model calls at trim time, no required dependencies, deterministic output.

In the published benchmark that locked barber's defaults, that meant 31.8 to
34.1% of context tokens gone with answer quality within noise of full context
([numbers below](#the-benchmark)).

## Install

```bash
pip install barber-llm
```

The PyPI name is `barber-llm`; the import name is `barber`. Zero required
dependencies. Extras when you want them:

| Extra | Pulls in | Gives you |
|---|---|---|
| `barber-llm[semantic]` | `sentence-transformers` | semantic scoring, paraphrase-safe |
| `barber-llm[tokens]` | `tiktoken` | exact token counts in `TrimResult` |
| `barber-llm[eval]` | `datasets`, `openai`, `tiktoken` | the `barber-eval` benchmark harness |

JavaScript is a first-class citizen too. The [npm package](https://www.npmjs.com/package/barber-llm)
is a zero-dependency port of the same algorithm with the same defaults, kept
decision-identical to this package by golden fixtures regenerated from the
Python implementation and replayed in CI ([js/](js/)):

```bash
npm install barber-llm
```

or one-shot from the shell:

```bash
npx barber-llm --keep 0.6 < messages.json > trimmed.json
```

## Quickstart

Zero-dependency lexical mode, runnable as pasted:

```python
from barber import trim

context = "\n\n".join(f"Passage {i}: facts about topic {i}." for i in range(40))  # your RAG block
messages = [{"role": "user", "content": context},
            {"role": "user", "content": "What do the passages say about topic 7?"}]

result = trim(messages)   # keep=0.6, lexical fallback, zero deps
print(result.tokens_saved, result.chunks_dropped, result.changed)
# result.messages is the same conversation, fewer tokens; send it to your LLM
```

`tokens_saved` is signed. Each dropped run costs about 20 tokens of marker, so
on blocks with many small, scattered chunks the markers can cost more than the
drops save. A negative number is barber telling you this shape is not worth
trimming, so skip it or raise `keep`.

Semantic mode, the configuration the benchmark shipped with:

```python
from barber import trim, embedders

embed = embedders.sentence_transformers("llm-semantic-router/mmbert-embed-32k-2d-matryoshka")
result = trim(messages, keep=0.6, embedder=embed)
```

The lexical fallback is deterministic and dependency-free, but it matches
words, not meaning. For production traffic use the semantic embedder above
(32K context window) or `BAAI/bge-small-en-v1.5` as the lightweight
alternative; the two tied on quality in the published runs. There is also
`embedders.endpoint(base_url, model, api_key)` for any OpenAI compatible
`/v1/embeddings` server (vLLM, TEI), which needs the `openai` package.

Multi-turn pipelines use the transform form. Pass a shared cache and a block is
decided once, then replayed byte-identically on every later turn, so your
provider prompt cache stays warm:

```python
from barber import make_transform, Cache

cache = Cache()                          # bounded LRU, safe for a long-running process
name, fn = make_transform(embedder=embed, keep=0.6, cache=cache)
messages, changed = fn(messages)         # call this every turn
```

A plain dict works too, but it never evicts: one entry per distinct context
block for the life of the process. `Cache` bounds it and is thread-safe, so a
threaded server can share one. Two things to size against:

- An entry holds one whole trimmed block, so the bound is a count, not a byte
  budget. At `maxsize=4096` with large RAG blocks that is hundreds of megabytes
  per process. Lower it if your blocks are big.
- Keep `maxsize` above your live-conversation count. Evicting a block means the
  next turn decides it again against the then-current question, which is the
  prefix churn the cache exists to prevent.

## When barber does nothing

`trim()` returns `changed=False` and leaves your messages untouched unless some
message meets every one of these:

| Requirement | Why |
|---|---|
| Role is `user`, `tool`, or `function` | System and assistant messages are never touched. |
| Not the latest user message | That message is the question, and the question is never trimmed. |
| `content` is a string, or text/`tool_result` parts | Other parts (images, `tool_use` inputs) pass through untouched. |
| At least 800 characters | Anything shorter is not retrieved context. |
| At least 4 chunks | Below that there is nothing to choose between. |

The common surprise is the single-message shape. This is a no-op, because the
only message present is the question:

```python
messages = [{"role": "user", "content": f"Context:\n{docs}\n\nQuestion: {q}"}]
```

Give the context its own message and barber has something to work with:

```python
messages = [{"role": "user", "content": docs},
            {"role": "user", "content": q}]
```

Most APIs accept consecutive user messages. If yours does not, put the context
in a `tool` or `function` message, which is where retrieved context usually
arrives anyway.

## Coding agents

A coding-agent transcript is not a RAG block, and two of its habits defeat
plain `trim()`:

* **Its context lives in content parts.** A tool result is a `tool_result`
  block, not a string, and that is where the big text is. `trim()` now walks
  into text and `tool_result` parts (`tool_use` inputs and images are passed
  through untouched).
* **Its output has no blank lines.** A file read, a grep, an `ls`, a diff — the
  paragraph splitter finds one chunk and selection declines the block. barber
  now falls back to line structure when, and only when, nothing else found any:
  a line-numbered read is chunked on the blank lines of the underlying *file*,
  a diff on its hunks, flat output on line windows. A JSON body is never
  line-chunked, because half a JSON object does not parse.

Selection still only guesses relevance. In an agent transcript, most of the
dead weight does not need a guess at all — it is content the transcript itself
proves is no longer true:

```python
from barber import trim
from barber.agent import sweep

result = sweep(trim(messages).messages)
```

`sweep()` drops three things, each provable from the transcript, each marked
with the file path so the agent can just read it again:

| Dropped | Because |
|---|---|
| A file body written before the same file was **fully rewritten** | It is not the file any more. A later `Edit` does *not* count: the file still holds that body. |
| A `Read` result for a file **modified since** | It is not the file any more. |
| A byte-identical repeat of an earlier tool result | It says nothing new. |

Blocks are emptied, never removed: a request rejects a `tool_use` with no
matching `tool_result`, so the structure survives even when the content does
not.

**`sweep()` rewrites history, and `trim()` deliberately does not.** The
freeze-on-first-sight cache exists to keep the prefix byte-stable so the
provider prompt cache stays warm; `sweep()` edits messages in the middle of the
conversation, so everything after the earliest edit is re-primed once. That is
a good trade at a compaction point with many turns left to amortize it, and a
bad one on turn three — which is why it is a call you make, not something
`trim()` does behind your back.

Tell it how many turns are left and it will do that arithmetic for you:

```python
sweep(messages, remaining_turns=50)
```

An edit at position P costs re-writing every token after P at 1.25x instead of
reading it back at 0.1x, and buys those removed tokens never being re-read
again. With `T` tokens after the cut and `S` of them removed, it only pays once

    S/T > 1.15 / (1.15 + 0.1 * remaining_turns)

— 53% of the tail at 10 turns, 19% at 50, 10% at 100. Editing later costs less
and saves less, so `sweep()` scores every candidate cut point and keeps only
the edits at or after the best one; everything before it is left alone and
stays cached. When nothing clears the bar it changes nothing and says so. The
15.7% of tool tokens the sweep can find needs roughly 62 remaining turns to pay
for itself if you claim all of it from the front of the transcript.

Left unset, `remaining_turns` edits everything it finds — right only when the
cache is already cold.

Measured on six real Claude Code sessions (1.16M tokens of transcript), in
quota units that charge cache reads at 0.1x and cache writes at 1.25x, and
paying the sweep's re-prime cost in full:

| | Tokens removed | Quota saved |
|---|---|---|
| `trim()` before this (content-parts guard) | 0.0% | 0.0% |
| `trim()` | 11.5% | 14.9% |
| `trim()` + `sweep()` | 19.5% | 23.7% |

That is 1.31x the turns per unit of quota. Your mileage varies with what your
session does: the spread across those six sessions was 1.18x to 1.68x, and the
sessions that gain most are the ones that rewrite the same files repeatedly.
`sweep()` is Python-only for now; the JS port has the `trim()` half.

## The benchmark

<img src="https://raw.githubusercontent.com/NadirRouter/barber/main/assets/barber-benchmark-results.png" alt="benchmark stat card" align="right" width="300">

barber's defaults were locked by a paired A/B on HotpotQA (distractor config):
answer each question with full context and with trimmed context, then have a
blind judge grade both answers against the gold reference.

| | Medium (~6K tok) | Large (~14K tok) |
|---|---|---|
| Tokens saved | 31.8% | 34.1% |
| Answer-paragraph retention | 100% | 100% |
| Full-context accuracy | 97.2% | 94.8% |
| Trimmed-context accuracy | 96.0% | 95.9% |

About 350 judged pairs, MiniMax M3 as the blind judge, answers graded against
gold references, `keep=0.6`. On large contexts trimmed beat full: selection
removes the distractors models trip over.

Full write-up: [the benchmark post](https://getnadir.com/blog/context-selection-benchmark-minimax.html).
Full protocol: [docs/methodology.md](https://github.com/NadirRouter/barber/blob/main/docs/methodology.md). Reproduce it:

```bash
pip install "barber-llm[eval]" && barber-eval --n 200 --keep 0.6 --size large
```

<br clear="right">

## How it works

<p align="center">
  <img src="https://raw.githubusercontent.com/NadirRouter/barber/main/assets/barber-how-it-works.png" alt="how barber works" width="100%">
</p>

1. **Query.** The latest user message is the question. It is never trimmed.
2. **Candidate blocks.** Large user, tool, and function messages (800+ chars,
   4+ chunks). System and assistant messages are never candidates.
3. **Chunk.** Split each block on blank lines, `---` rules, headings, and
   `[n]` citation markers, the boundaries RAG concatenations already have.
   Falls back to sentences for a single wall of prose.
4. **Score.** Embed every chunk plus the question, cosine each chunk against
   the question, keep the top `keep` fraction.
5. **Guards.** Pinned chunks bypass the budget entirely, the first and last
   chunk always survive, and a relevance floor drops pure noise even when the
   budget would keep it. Details below.
6. **Marker.** Each dropped run collapses into one line, so the model knows
   the cut happened and doesn't go looking for missing text:

   ```
   [… 4 passage(s) omitted as not relevant to this question — the remaining context is sufficient …]
   ```

   The assertive wording is deliberate and benchmark-locked: it won our
   marker ablation. The working theory is that a neutral "lower-relevance
   passages omitted" invites the model to hedge, while this one tells it to
   proceed. Re-run the ablation yourself with `barber-eval --marker neutral`.

Decisions are memoized on the block hash alone, never the query. The first
turn to see a block decides it; every later turn replays the decision
byte-identically. Your history prefix stays stable and your provider prompt
cache keeps hitting.

## What barber does not do

- **No summarization.** Chunks survive verbatim or vanish. A summary is a
  rewrite, and rewrites can silently invent or lose facts.
- **No letter tricks.** Dropping vowels, truncating words, gzip-then-base64:
  in the same benchmark, every character-level scheme cost MORE tokens, not
  fewer. Letter removal measured 1.34x to 1.69x the tokens; classic
  compression 3.2x to 3.7x. Tokenizers are already compressors; fighting them
  backfires. [The numbers](https://getnadir.com/blog/context-selection-benchmark-minimax.html).
- **No history compaction.** Providers do that natively now, and re-writing
  old turns busts their prompt caches. barber targets fresh retrieved context
  only.
- **No model calls at trim time.** Scoring is an embedding pass (or pure
  lexical math). Nothing is sent anywhere unless you opt into the
  `endpoint()` embedder.

## Guards, or why your answers survive

The failure mode of context pruning is silent: drop the one chunk that held
the answer and nothing errors, the model just answers worse. barber ships with
every guard on:

- **Deontic and PII pinning.** Chunks with constraint language ("must",
  "never", "do not", "don't", "shall not", "prohibited", "required",
  "only if") or sensitive-data markers (PII, HIPAA, PCI, SSN, password,
  secret, API key) are never dropped.
- **Rare-query-entity pinning.** A query term that appears in only one or two
  chunks of a block is a strong "this chunk answers the question" signal.
  Those chunks are never dropped, which is what protects multi-hop questions.
- **Lead and tail keep.** The first and last chunk of every block survive:
  headers, conclusions, and the lost-in-the-middle mitigation.
- **Relevance floor (0.35).** A chunk scoring far below the block's top chunk
  is noise even if the budget has room for it.
- **Never touched at all:** the latest user message, system messages,
  assistant messages, non-string content, any message under 800 chars, any
  block under 4 chunks. See [when barber does nothing](#when-barber-does-nothing).

In the benchmark these guards held answer-paragraph retention at 100% in both
published sizes ([table above](#the-benchmark)).

## Evaluate it on your data

Published numbers are an existence proof, not a guarantee for your traffic.
The harness that produced them ships in the package:

```bash
pip install "barber-llm[eval]"
export JUDGE_API_KEY=sk-...     # any OpenAI compatible judge; MiniMax by default

barber-eval --baseline --n 25   # sanity-check generator + judge first
barber-eval --n 200 --keep 0.6 --size large
```

The number to gate on is the **regression rate**: how often trimming turned a
right answer wrong. Hold it under 1 to 2%, then take the most aggressive
`keep` that stays under your bar. Tokens saved is the payoff, not the gate.
The exact judge prompt ships in
[barber/eval/JUDGE_PROMPT.md](https://github.com/NadirRouter/barber/blob/main/barber/eval/JUDGE_PROMPT.md), including a
reference-free variant for chat data without gold answers. Judge and generator
are swappable via `JUDGE_*` and `GEN_*` env vars
([details](https://github.com/NadirRouter/barber/blob/main/docs/methodology.md)).

## Credits

- The **MiniMax team**: MiniMax M3 was the generator and the blind judge for
  the entire eval, about 2,000 model calls on well under $20 of credit.
- The **vLLM Semantic Router team**: their
  [mmbert-embed-32k-2d-matryoshka](https://huggingface.co/llm-semantic-router/mmbert-embed-32k-2d-matryoshka)
  is the recommended embedder, picked for its 32K window.

---

Want this tier-aware, quality-monitored in production, and billed only on
verified savings? That's [Nadir](https://getnadir.com).
