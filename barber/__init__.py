"""barber — query-aware context trimming for LLM requests.

Your context could use a trim.

    from barber import trim

    result = trim(messages, keep=0.6)
    send(result.messages)          # same conversation, fewer tokens

Chunks survive verbatim or vanish. Nothing is rewritten, nothing is summarized,
and no model is called at trim time. The selection algorithm is the validated,
benchmarked one from Nadir (see README for the numbers).
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Callable, Optional

from .core import SelectionConfig, SelectionStats, make_selection_transform
from .core import _text_of

__version__ = "0.1.0"

__all__ = ["trim", "TrimResult", "make_transform", "SelectionConfig", "__version__"]


@dataclass
class TrimResult:
    messages: list          # trimmed messages (originals are not mutated)
    tokens_saved: int       # tiktoken o200k count if installed, len//4 fallback
    chunks_dropped: int
    changed: bool


def _token_counter() -> Callable[[str], int]:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return lambda s: len(enc.encode(s))
    except Exception:
        return lambda s: len(s) // 4


def _count_tokens(messages: list, ntok: Callable[[str], int]) -> int:
    return sum(ntok(_text_of(m.get("content"))) for m in messages)


def make_transform(
    embedder: Optional[Callable] = None,
    keep: float = 0.6,
    *,
    cfg: Optional[SelectionConfig] = None,
    cache: Optional[dict] = None,
):
    """Return the ("barber", fn) pair for pipeline integration, where
    fn(messages) -> (messages, changed).

    Pass a shared `cache` dict to get freeze-on-first-sight memoization across
    turns: the first turn to see a block decides it, every later turn replays
    that decision byte-identically, keeping the provider prompt cache warm.
    `keep` pins the fraction of chunks retained (0.6 is the benchmark default).
    """
    cfg = replace(cfg or SelectionConfig(), min_keep_ratio=keep, max_keep_ratio=keep)
    _, fn = make_selection_transform(embed_fn=embedder, cfg=cfg, decision_cache=cache)
    return ("barber", fn)


def trim(
    messages: list,
    keep: float = 0.6,
    embedder: Optional[Callable] = None,
    *,
    cfg: Optional[SelectionConfig] = None,
    cache: Optional[dict] = None,
) -> TrimResult:
    """Trim query-irrelevant chunks out of an OpenAI-style message list.

    messages: list[dict] with "role" and "content" keys.
    keep:     fraction of chunks retained per block (benchmark default 0.6).
    embedder: None -> deterministic lexical fallback (zero deps), or one of
              barber.embedders.lexical / sentence_transformers / endpoint.

    The latest user message and system messages are never trimmed. Guards
    (lead/tail keep, deontic/PII pinning, rare-query-entity pinning, relevance
    floor) are always on. Deterministic: same input, same output.
    """
    _, fn = make_transform(embedder, keep, cfg=cfg, cache=cache)
    out, changed = fn(list(messages))
    stats: SelectionStats = fn.last_stats
    ntok = _token_counter()
    saved = _count_tokens(messages, ntok) - _count_tokens(out, ntok)
    return TrimResult(
        messages=out,
        tokens_saved=max(0, saved),
        chunks_dropped=stats.chunks_in - stats.chunks_kept,
        changed=changed,
    )
