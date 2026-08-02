"""Document-set filtering: the one primitive every retriever integration needs.

A retriever hands its post-processor a query and N candidate documents and wants
a subset back. barber's public call takes a message list and returns a message
list, so this module is the map between the two:

    N document bodies  ->  one context block, blank-line separated
    barber.trim(block + query)
    trimmed block      ->  a keep/drop flag per document

That block shape is not an arbitrary choice. It is exactly what the published
benchmark trimmed: HotpotQA paragraphs joined with blank lines into a single
context block, scored against the question (``barber/eval/harness.py``,
``build_context``). One document is one chunk, so the decision barber makes here
is a decision about whole documents.

Framework-free on purpose: no langchain, no llama-index, nothing but barber. The
adapters import this; the tests can exercise it with the extras absent.
"""
from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

from .. import trim
from ..core import SelectionConfig

__all__ = ["select_documents"]

SEP = "\n\n"


def _marker_re(template: str) -> "re.Pattern":
    return re.compile(re.escape(template).replace(re.escape("{n}"), r"\d+"))


def select_documents(
    contents: Sequence[str],
    query: str,
    *,
    keep: float = 0.6,
    embedder: Optional[Callable] = None,
    cfg: Optional[SelectionConfig] = None,
) -> list[bool]:
    """Return one keep/drop flag per document body, in input order.

    Fails open in every ambiguous case: if the query is empty, if there is
    nothing to choose between, if barber declines the block, or if the trimmed
    text cannot be mapped back onto the inputs, every flag comes back ``True``
    and the caller's document list is unchanged. A context filter that loses
    documents when it is confused is worse than one that does nothing.

    barber's own gates apply unchanged, which is where "nothing happened" almost
    always comes from: the joined block must be at least
    ``cfg.min_message_chars`` characters (800) and split into at least
    ``cfg.min_chunks`` chunks (4). Four short documents are not enough context to
    be worth cutting, and barber says so by declining.
    """
    n = len(contents)
    flags = [True] * n
    if n < 2 or not query.strip():
        return flags

    # Char span of each document inside the joined block. Chunks come back as
    # contiguous substrings of what went in (barber never rewrites a chunk), so
    # a span is all it takes to map a survivor back to the document it came from.
    spans: list[tuple[int, int]] = []
    at = 0
    for body in contents:
        spans.append((at, at + len(body)))
        at += len(body) + len(SEP)
    block = SEP.join(contents)

    result = trim(
        [{"role": "user", "content": block},
         {"role": "user", "content": query}],
        keep=keep,
        embedder=embedder,
        cfg=cfg,
    )
    if not result.changed:
        return flags

    trimmed = result.messages[0]["content"]
    marker = _marker_re((cfg or SelectionConfig()).drop_marker)

    kept = [False] * n
    cursor = 0
    for seg in trimmed.split(SEP):
        if not seg.strip():
            continue
        # Surviving text is a substring of the block, and survivors keep their
        # order, so the search starts where the last one ended. Text that is NOT
        # findable is text barber inserted, i.e. a drop marker; anything else
        # means the mapping broke and the honest answer is to keep everything.
        found = block.find(seg, cursor)
        if found < 0:
            if marker.fullmatch(seg.strip()):
                continue
            return flags
        end = found + len(seg)
        cursor = end
        for i, (start, stop) in enumerate(spans):
            if start < end and found < stop:
                kept[i] = True

    for i, body in enumerate(contents):
        # A blank document has no text to score and never appears in the output.
        # Dropping it would be a decision barber never made.
        if not body.strip():
            kept[i] = True

    return kept if any(kept) else flags
