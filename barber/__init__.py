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
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Callable, Optional

from .core import SelectionConfig, SelectionStats, make_selection_transform
from .core import _text_of

__version__ = "0.1.0"

__all__ = ["trim", "TrimResult", "make_transform", "Cache", "SelectionConfig", "__version__"]


class Cache(OrderedDict):
    """Bounded LRU cache to hand to `make_transform(cache=...)`.

    A plain dict works and never evicts: one entry per distinct context block,
    held for the life of the process. That is fine for a script and a leak in a
    server, so use this instead.

    Eviction re-opens a decision: if an evicted block reappears under a
    different question it is selected again, and the result can differ from the
    first time. Keep `maxsize` above your live-conversation count and that never
    happens.
    """

    def __init__(self, maxsize: int = 4096):
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key):
        self.move_to_end(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.maxsize:
            self.popitem(last=False)


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

    Pass a shared `cache` to get freeze-on-first-sight memoization across turns:
    the first turn to see a block decides it, every later turn replays that
    decision byte-identically, keeping the provider prompt cache warm. Use
    `barber.Cache()` in a long-running process; a plain dict never evicts.
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

    A message is only trimmed when ALL of these hold, so `changed` comes back
    False if none qualifies:
      - role is "user", "tool", or "function" (system and assistant are never
        touched)
      - it is NOT the latest user message (that one is the question)
      - content is a plain string, not a list of content parts
      - at least 800 characters and at least 4 chunks

    So context and question packed into ONE user message is a no-op: put the
    context in its own earlier message. See "When barber does nothing" in the
    README.

    Guards (lead/tail keep, deontic/PII pinning, rare-query-entity pinning,
    relevance floor) are always on. Deterministic: same input, same output.
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
