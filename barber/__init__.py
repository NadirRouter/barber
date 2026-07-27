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
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable, Optional

from .core import SelectionConfig, make_selection_transform
from .core import _text_of

__version__ = "0.3.0"

__all__ = ["trim", "TrimResult", "make_transform", "Cache", "SelectionConfig", "__version__"]


class Cache:
    """Bounded, thread-safe LRU cache to hand to `make_transform(cache=...)`.

    A plain dict works and never evicts: one entry per distinct context block,
    held for the life of the process. That is fine for a script and a leak in a
    server, so use this instead.

    An entry holds one whole trimmed block, so size `maxsize` against your block
    size, not just your conversation count: 4096 large RAG blocks is hundreds of
    megabytes per process.

    Eviction re-opens a decision: if an evicted block reappears under a
    different question it is selected again, and the result can differ from the
    first time. Keep `maxsize` above your live-conversation count and that never
    happens.

    This wraps an OrderedDict rather than subclassing one, deliberately. As a
    subclass, the C internals and the inherited public methods (pop, popitem,
    copy, |) re-enter the overridden __getitem__/__setitem__, which corrupts the
    cache on CPython 3.10 and hides an unsynchronized read-modify-write. Only
    the four operations barber actually performs are exposed, each under a lock.
    """

    def __init__(self, *, maxsize: int = 4096):
        # keyword-only on purpose: Cache(some_dict) would otherwise bind the
        # mapping to maxsize and fail later, deep inside the eviction loop.
        self.maxsize = maxsize
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def __contains__(self, key) -> bool:
        # Locked and LRU-refreshing: in this codebase a membership test is
        # always immediately followed by a read, so a hit counts as a use.
        with self._lock:
            if key not in self._d:
                return False
            self._d.move_to_end(key)
            return True

    def __getitem__(self, key):
        with self._lock:
            value = self._d[key]
            self._d.move_to_end(key)
            return value

    def __setitem__(self, key, value) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._d)

    def __iter__(self):
        with self._lock:
            return iter(list(self._d))

    def values(self):
        with self._lock:
            return list(self._d.values())

    def __repr__(self) -> str:
        return f"Cache(maxsize={self.maxsize}, entries={len(self)})"


@dataclass
class TrimResult:
    messages: list          # trimmed messages (originals are not mutated)
    tokens_saved: int       # tiktoken o200k count if installed, len//4 fallback.
                            # Signed: negative means trimming cost more than it
                            # saved, which happens when drops are scattered and
                            # each one emits its own marker.
    chunks_dropped: int
    changed: bool


@lru_cache(maxsize=1)
def _token_counter() -> Callable[[str], int]:
    # Memoized: tiktoken.get_encoding downloads its vocab on a cold cache, and
    # rebuilding the counter per trim() call re-attempted that download (with no
    # timeout) on every request from a host that could not reach it.
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return lambda s: len(enc.encode(s))
    except Exception:
        return lambda s: len(s) // 4


def _count_tokens(messages: list, ntok: Callable[[str], int]) -> int:
    return sum(ntok(_text_of(m.get("content"), True)) for m in messages)


def _marker_re(template: str) -> "re.Pattern":
    return re.compile(re.escape(template).replace(re.escape("{n}"), r"(\d+)"))


def _count_dropped(before: list, after: list, marker: str) -> int:
    """Count dropped chunks from the markers in the output.

    Not from SelectionStats: the stats are only written when a block is
    actually selected, so a cache hit (every turn after the first with a shared
    cache) leaves them at zero while the content is still trimmed.
    """
    rx = _marker_re(marker)
    total = 0
    for b, a in zip(before, after):
        if b is a:
            continue
        was = sum(int(n) for n in rx.findall(_text_of(b.get("content"), True)))
        now = sum(int(n) for n in rx.findall(_text_of(a.get("content"), True)))
        total += now - was
    return total


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
    `keep` pins the fraction of chunks retained (0.6 is the benchmark default),
    overriding min_keep_ratio/max_keep_ratio on any `cfg` you pass.
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
      - it is NOT the latest user message (that one is the question, and in an
        agent loop it is the tool result the agent is about to act on)
      - content is a string, or text / tool_result parts inside a content list
        (images and tool_use inputs pass through untouched)
      - at least 800 characters and at least 4 chunks

    So context and question packed into ONE user message is a no-op: put the
    context in its own earlier message. See "When barber does nothing" in the
    README.

    `tokens_saved` is signed. Negative means the markers cost more than the
    dropped chunks saved, which is worth acting on rather than hiding.

    PASS A `cache` IN A MULTI-TURN LOOP. Without one, every call starts a fresh
    decision cache and re-selects each block against that turn's question — so
    the same history block comes back with different bytes each turn, mutating
    the prefix and costing you the provider prompt cache that freeze-on-first-
    sight exists to protect (see core.py, design decisions 1 and 2). One-shot
    trimming is fine without it; a conversation is not. Use `barber.Cache()` in
    a long-running process, or a plain dict in a script.

    Guards (lead/tail keep, deontic/PII pinning, rare-query-entity pinning,
    relevance floor) are always on. Deterministic: same input, same output.
    """
    effective = replace(cfg or SelectionConfig(), min_keep_ratio=keep, max_keep_ratio=keep)
    _, fn = make_selection_transform(embed_fn=embedder, cfg=effective, decision_cache=cache)
    out, changed = fn(list(messages))
    if not changed:
        # Nothing was substituted, so every message in `out` is the input object:
        # the counts are 0 by construction, no need to tokenize the prompt twice.
        return TrimResult(messages=out, tokens_saved=0, chunks_dropped=0, changed=False)
    ntok = _token_counter()
    saved = _count_tokens(messages, ntok) - _count_tokens(out, ntok)
    return TrimResult(
        messages=out,
        tokens_saved=saved,
        chunks_dropped=_count_dropped(messages, out, effective.drop_marker),
        changed=True,
    )
