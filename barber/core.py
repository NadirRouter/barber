"""barber core — query-conditioned pruning of retrieved/tool context.

The ONE compaction operation that actually reduces tokens instead of fighting the
tokenizer: it drops whole chunks (the billing unit is the token, and a dropped
chunk removes whole tokens), rather than mangling characters inside words.

Ported verbatim from Nadir's validated ``context_selection.py``. The selection
logic is byte-for-byte the benchmarked algorithm; only the module docstring, the
benchmark-locked drop-marker default, and the embedder imports differ.

Design decisions (each one is load-bearing — read before changing):

1. TARGETS FRESH CONTEXT ONLY. Selection runs on the current request's retrieved
   chunks and large tool outputs — content that is full-priced and NOT yet a warm
   cached prefix. It deliberately does NOT re-prune the stable prefix (system
   prompt, old history) every turn, because query-conditioned pruning of the
   prefix would mutate it each turn and bust the provider prompt cache. Old
   history is left to middle-out / dedup. This sidesteps the prefix-stability
   trap that killed naive selection.

2. FREEZE-ON-FIRST-SIGHT. The keep/drop decision for a given block is a pure
   function of (block_hash, query_hash) and is memoized. If the same request
   shape recurs, it reproduces byte-identically. (Cache is process-local; a
   production build keys it in the request store.)

3. TIER-CONDITIONED BUDGET. A stronger routed model needs less context (it fills
   gaps from its weights); a weaker one needs more. Only a router knows the tier,
   so the keep-threshold is a function of it. This is the one router-native edge.

4. CONSTRAINT PINNING. Chunks matching safety/policy/negation/number patterns, or
   containing the query's own rare entities, are never dropped — dropping them is
   the silent-failure mode (multi-hop / lost-in-the-middle).

5. REVERSIBLE. Dropped runs leave a compact marker; wire a retrieve handle if you
   want the model to be able to pull a dropped block back (optional).

Integration: use ``barber.trim(messages)`` for one-shot trimming, or
``barber.make_transform(...)`` for the pipeline hook. The transform callable has
the message-transform signature of a Nadir-style optimizer pipeline:
    fn(messages: list[dict]) -> tuple[list[dict], bool]
"""
from __future__ import annotations
import hashlib, logging, re, math
from dataclasses import dataclass, field
from typing import Callable, Optional

from .embedders import lexical as make_lexical_embedder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SelectionConfig:
    # Only consider a message for selection if it is at least this many chars
    # (small messages aren't retrieved-context; leave them alone).
    min_message_chars: int = 800
    # Only split/select when the message has at least this many chunks.
    min_chunks: int = 4
    # Keep-ratio bounds (fraction of chunks retained) after tier conditioning.
    min_keep_ratio: float = 0.25
    max_keep_ratio: float = 1.00
    # A chunk scoring below (top_score * relative_floor) is a drop candidate even
    # if the budget would keep it — protects against keeping pure noise.
    relative_floor: float = 0.35
    # Always keep the first & last chunk of a block (lead/tail bias — headers,
    # conclusions, and the "lost in the middle" mitigation).
    keep_lead_tail: bool = True
    # Marker inserted where a run of chunks was dropped. The assertive wording is
    # benchmark-locked: it beat the neutral variant in the published A/B (the
    # neutral marker primes refusals; this one tells the model to proceed).
    drop_marker: str = "[… {n} passage(s) omitted as not relevant to this question — the remaining context is sufficient …]"
    # Roles whose large messages are selectable (never the latest user query).
    selectable_roles: tuple = ("user", "tool", "function")

# Tier -> target keep ratio, for router integrations that know task complexity.
# Strong models reason from less context. Tune these against your own
# shadow-eval; conservative defaults here. (barber's public API pins the ratio
# via trim(keep=...) instead; this table only applies when you pass tier=.)
TIER_KEEP_RATIO = {
    "simple":   0.45,   # cheap tier, short tasks: aggressive
    "mid":      0.55,
    "medium":   0.55,   # routers commonly emit "medium", not "mid". Both map
                        # here so the middle tier is a lookup, not a default
                        # that happens to hold the right value.
    "complex":  0.70,
    "reasoning":0.85,   # hard tasks: keep almost everything
}

# ---------------------------------------------------------------------------
# Pinning — never-drop patterns (the silent-failure guard)
# ---------------------------------------------------------------------------

_PIN_PATTERNS = [
    re.compile(r"\b(must|never|do not|don't|shall not|prohibited|required|only if)\b", re.I),  # deontic/safety
    re.compile(r"\b(?:PII|HIPAA|PCI|SSN|password|secret|api[_ ]?key)\b", re.I),
]
# NOTE: we deliberately do NOT pin on bare digits — most chunks contain a number,
# so digit-pinning neuters selection (the harness caught this). Numeric chunks
# that matter are kept by RELEVANCE scoring; safety/policy text is pinned above,
# and RARE query entities (below) protect the answer-bearing chunk for multi-hop.

def _is_pinned(chunk: str, rare_query_entities: set[str]) -> bool:
    for p in _PIN_PATTERNS:
        if p.search(chunk):
            return True
    # A query entity that is RARE across the block (appears in <=2 chunks) is a
    # strong "this chunk answers the query" signal — pin it so aggressive budgets
    # can't drop the one gold chunk (multi-hop protection).
    low = chunk.lower()
    return any(e in low for e in rare_query_entities)

# ---------------------------------------------------------------------------
# Chunking & scoring
# ---------------------------------------------------------------------------

_TOK = re.compile(r"[a-z0-9]+")

def _tokenize(s: str) -> list[str]:
    return _TOK.findall(s.lower())

# Line-oriented shapes an agent harness produces. Prose has blank lines between
# paragraphs; a file read, a grep, a diff, or an ls does not, so the paragraph
# splitter below returns one chunk and selection declines the block entirely.
_LINE_NO = re.compile(r"^\s*\d+\t")             # `cat -n` / Read line prefix
_HUNK = re.compile(r"^(?:diff --git |@@ )")     # unified diff

def _split_lines(text: str, min_chunks: int) -> list[str]:
    """Chunk line-oriented text. Returns [] when no shape applies.

    Only reached when the paragraph and sentence splitters both came up short,
    so this widens coverage without moving any decision they already made.
    """
    if text.lstrip()[:1] in "{[":
        # A JSON body is one value: dropping lines out of the middle yields
        # something that no longer parses. Leave it whole.
        return []
    lines = text.split("\n")
    if len(lines) < min_chunks:
        return []

    # Line-numbered file read: strip the prefix to find the blank lines of the
    # underlying file, then chunk on those — the file's own paragraph structure
    # (function and block boundaries), not an arbitrary window.
    head = lines[:40]
    if sum(1 for l in head if _LINE_NO.match(l)) >= len(head) * 0.8:
        groups = _group(lines, lambda l: not _LINE_NO.sub("", l).strip(), drop=True)
        if len(groups) >= min_chunks:
            return groups

    # Unified diff: one chunk per hunk.
    if any(_HUNK.match(l) for l in lines[:5]):
        groups = _group(lines, lambda l: _HUNK.match(l) is not None, drop=False)
        if len(groups) >= min_chunks:
            return groups

    # Anything else line-oriented (grep hits, file lists, logs): fixed windows.
    # ponytail: 12 windows is a knob, not a law — it is enough chunks for the
    # keep ratio to have somewhere to cut without slicing single log lines.
    step = max(1, len(lines) // 12)
    groups = ["\n".join(lines[i:i + step]) for i in range(0, len(lines), step)]
    return groups if len(groups) >= min_chunks else []

def _group(lines: list[str], is_boundary, *, drop: bool) -> list[str]:
    """Split lines at boundary lines. drop=True discards them (blank lines),
    drop=False starts a new chunk with them (diff hunk headers)."""
    out: list[str] = []
    cur: list[str] = []
    for l in lines:
        if is_boundary(l):
            if drop:
                if cur:
                    out.append("\n".join(cur)); cur = []
                continue
            if cur:
                out.append("\n".join(cur)); cur = []
        cur.append(l)
    if cur:
        out.append("\n".join(cur))
    return out

def split_chunks(text: str, min_chunks: int = 4) -> list[str]:
    """Split a block into selectable chunks: prefer blank-line / delimiter
    boundaries (RAG concatenations, tool rows), fall back to sentences, then to
    line structure for the line-oriented output an agent harness produces."""
    # explicit chunk delimiters common in RAG concatenation
    parts = re.split(r"\n\s*\n|\n---+\n|\n#{1,6}\s|\[\d+\]\s", text)
    parts = [p.strip() for p in parts if p and p.strip()]
    if len(parts) >= 2:
        # Delimiters found: this is a chunked block, and how many chunks it has
        # is the answer. Too few to bother cutting is a real signal, not a
        # failure to look harder — leave that decision where the benchmark set
        # it, and do not fall through.
        return parts
    # fall back to sentence-ish splitting for a single wall of prose
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    if len(sents) >= min_chunks:
        return sents
    # Nothing found any structure, so selection was about to decline the block
    # whole (see the min_chunks gate in _select_block). Line layout is the last
    # place to look; if that comes up empty too, nothing is lost.
    return _split_lines(text, min_chunks) or sents

def _cos(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    keys = a.keys() & b.keys()
    num = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return num / (na * nb) if na and nb else 0.0

def _cos_vec(a, b) -> float:
    """Cosine for dense vectors (numpy arrays / lists) when a real encoder is used."""
    import numpy as np
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0

# ---------------------------------------------------------------------------
# Core selection over a single block
# ---------------------------------------------------------------------------

@dataclass
class SelectionStats:
    blocks_processed: int = 0
    chunks_in: int = 0
    chunks_kept: int = 0
    chars_in: int = 0
    chars_out: int = 0

def _select_block(text: str, query: str, embed_fn, keep_ratio: float,
                  cfg: SelectionConfig, query_entities: set[str],
                  stats: SelectionStats) -> tuple[str, bool]:
    chunks = split_chunks(text, cfg.min_chunks)
    if len(chunks) < cfg.min_chunks:
        return text, False

    # rare query entities = query tokens appearing in <=2 chunks of THIS block
    q_ents = {w for w in _tokenize(query) if len(w) >= 4}
    df = {e: sum(1 for c in chunks if e in c.lower()) for e in q_ents}
    rare_query_entities = {e for e, d in df.items() if 1 <= d <= 2}

    vecs = embed_fn(chunks + [query])
    qv = vecs[-1]; cvs = vecs[:-1]
    sim = _cos if isinstance(qv, dict) else _cos_vec
    scores = [sim(cv, qv) for cv in cvs]
    # A *relative* floor needs something to be relative to. When no chunk shares
    # a token with the query -- a non-Latin-script query on the lexical
    # embedder, a block on a wholly different subject -- every score is 0 and
    # the ranking carries no signal at all. The old `max(scores) or 1e-9` turned
    # that into a floor of 3.5e-10 that every zero score failed, collapsing the
    # block to lead+tail even at keep=1.0, where the caller asked to drop
    # nothing. With no signal to discriminate on, the budget alone decides.
    top = max(scores)
    floor = top * cfg.relative_floor if top > 0 else 0.0

    n = len(chunks)
    keep_n = max(1, min(n, round(n * keep_ratio)))

    # rank chunks; always keep pinned + (optionally) lead/tail
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    keep = set(order[:keep_n])
    if cfg.keep_lead_tail:
        keep.add(0); keep.add(n - 1)
    for i in range(n):
        if _is_pinned(chunks[i], rare_query_entities):
            keep.add(i)
    # drop anything far below the top even if budget kept it
    keep = {i for i in keep if scores[i] >= floor or _is_pinned(chunks[i], rare_query_entities) or i in (0, n-1)}

    if len(keep) >= n:
        return text, False

    # rebuild preserving order, collapsing dropped runs into one marker
    out, run = [], 0
    for i in range(n):
        if i in keep:
            if run:
                out.append(cfg.drop_marker.format(n=run)); run = 0
            out.append(chunks[i])
        else:
            run += 1
    if run:
        out.append(cfg.drop_marker.format(n=run))

    stats.blocks_processed += 1
    stats.chunks_in += n
    stats.chunks_kept += len(keep)
    new_text = "\n\n".join(out)
    stats.chars_in += len(text); stats.chars_out += len(new_text)
    return new_text, True

# ---------------------------------------------------------------------------
# Build the pipeline transform
# ---------------------------------------------------------------------------

def make_selection_transform(
    embed_fn: Optional[Callable] = None,
    tier: str = "mid",
    complexity_score: Optional[float] = None,
    cfg: Optional[SelectionConfig] = None,
    decision_cache: Optional[dict] = None,
):
    """Return (name, fn) where fn(messages)->(messages, changed), matching
    the message-transform hook of a Nadir-style optimizer pipeline.

    embed_fn(list[str]) -> list[vector]  (dict bags OR dense vectors). If None,
    a deterministic lexical fallback is used (swap in your encoder in prod).
    tier / complexity_score set how aggressively to prune.
    """
    cfg = cfg or SelectionConfig()
    embed_fn = embed_fn or make_lexical_embedder()
    cache = decision_cache if decision_cache is not None else {}

    keep_ratio = TIER_KEEP_RATIO.get(tier, 0.55)
    if complexity_score is not None:  # optionally blend the router's continuous score
        keep_ratio = max(cfg.min_keep_ratio, min(cfg.max_keep_ratio,
                        0.30 + 0.6 * float(complexity_score)))
    keep_ratio = max(cfg.min_keep_ratio, min(cfg.max_keep_ratio, keep_ratio))

    def _query_of(messages: list[dict]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content")
                q = c if isinstance(c, str) else _text_of(c)
                # In an agent loop the last user message is usually a tool
                # result carrying no text of its own. That is not the question;
                # keep walking back to the last thing the human actually asked,
                # or the whole transform bails on an empty query.
                if q.strip():
                    return q
        return ""

    def _select_messages(messages: list[dict]) -> tuple[list[dict], bool]:
        query = _query_of(messages)
        if not query:
            return messages, False
        q_toks = _tokenize(query)
        # "rare" query tokens = load-bearing entities to pin (skip stopwords-ish short/common)
        query_entities = {w for w in q_toks if len(w) >= 4}

        stats = SelectionStats()
        changed_any = False
        out_msgs = []
        # the latest user message is the QUERY — never prune it
        last_user_idx = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)

        def select_text(text: str) -> tuple[str, bool]:
            """Run (or replay) the decision for one string block."""
            if len(text) < cfg.min_message_chars:
                return text, False
            # Key on the block ONLY -- not the query, and not keep_ratio.
            # A conversation asks a new question every turn AND may route to a
            # different tier every turn, so including either meant the history
            # block was re-selected and the prefix mutated, silently destroying
            # the provider prompt cache this memoization exists to protect.
            # Freeze-on-first-sight: the first turn to see a block decides it,
            # every later turn replays that decision verbatim.
            key = hashlib.md5(text.encode()).hexdigest()[:12]
            if key not in cache:
                cache[key] = _select_block(text, query, embed_fn, keep_ratio,
                                           cfg, query_entities, stats)
            return cache[key]

        for idx, m in enumerate(messages):
            content = m.get("content")
            if idx == last_user_idx or m.get("role") not in cfg.selectable_roles:
                out_msgs.append(m); continue

            # Content parts, not a plain string: this is the shape every agent
            # harness actually sends, because a tool result is a content block
            # (Anthropic tool_result, OpenAI-style text parts). The big blocks
            # in an agent session live in here, so skipping lists meant
            # selection never fired on an agent transcript at all.
            if isinstance(content, list):
                new_parts, ch_any = _select_parts(content, select_text)
                if ch_any:
                    out_msgs.append({**m, "content": new_parts}); changed_any = True
                else:
                    out_msgs.append(m)
                continue

            if not isinstance(content, str) or len(content) < cfg.min_message_chars:
                out_msgs.append(m); continue

            # Key on the block ONLY -- not the query, and not keep_ratio.
            # A conversation asks a new question every turn AND may route to a
            # different tier every turn, so including either meant the history
            # block was re-selected and the prefix mutated, silently destroying
            # the provider prompt cache this memoization exists to protect.
            # Freeze-on-first-sight: the first turn to see a block decides it,
            # every later turn replays that decision verbatim.
            key = hashlib.md5(content.encode()).hexdigest()[:12]
            if key in cache:                       # freeze-on-first-sight -> prefix stable
                new_content, ch = cache[key]
            else:
                new_content, ch = _select_block(content, query, embed_fn, keep_ratio,
                                                cfg, query_entities, stats)
                cache[key] = (new_content, ch)
            if ch:
                out_msgs.append({**m, "content": new_content}); changed_any = True
            else:
                out_msgs.append(m)

        transform.last_stats = stats
        return out_msgs, changed_any

    def transform(messages: list[dict]) -> tuple[list[dict], bool]:
        """Fail-open wrapper. A context optimizer must never be the reason a
        request dies: an injected embedder that raises (encoder endpoint down,
        numpy missing on the dense path, a model that returns the wrong arity)
        should cost the trim, not the conversation. The failure is logged, not
        swallowed silently — a permanently-broken encoder that quietly saves
        nothing is its own kind of outage."""
        try:
            return _select_messages(messages)
        except Exception:
            logger.warning("barber: selection failed, context passed through untrimmed",
                           exc_info=True)
            return messages, False

    transform.last_stats = SelectionStats()
    return ("context_selection", transform)


def _select_parts(parts: list, select_text) -> tuple[list, bool]:
    """Apply `select_text` to the trimmable string bodies inside a content list.

    Trimmable means text parts and tool-result bodies. Everything else (images,
    tool_use inputs, anything unrecognised) is copied through untouched: a
    tool_use input is arguments the provider will parse, not prose to prune.
    """
    out, changed = [], False
    for p in parts:
        if not isinstance(p, dict):
            out.append(p); continue
        kind = p.get("type")
        if kind == "text" and isinstance(p.get("text"), str):
            new, ch = select_text(p["text"])
            out.append({**p, "text": new} if ch else p); changed |= ch
        elif kind == "tool_result":
            inner = p.get("content")
            if isinstance(inner, str):
                new, ch = select_text(inner)
                out.append({**p, "content": new} if ch else p); changed |= ch
            elif isinstance(inner, list):
                new_inner, ch = _select_parts(inner, select_text)
                out.append({**p, "content": new_inner} if ch else p); changed |= ch
            else:
                out.append(p)
        else:
            out.append(p)
    return out, changed


def _text_of(content, tool_results: bool = False) -> str:
    """Flatten content to text. `tool_results=True` also walks tool-result
    bodies — right for token accounting, wrong for finding the user's question
    (a tool result is not something the human asked)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text":
                out.append(p.get("text", ""))
            elif tool_results and p.get("type") == "tool_result":
                out.append(_text_of(p.get("content"), True))
        return " ".join(out)
    return ""
