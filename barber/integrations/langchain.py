"""LangChain integration: barber as a `BaseDocumentCompressor`.

    pip install "barber-llm[langchain]"

`ContextualCompressionRetriever` hands its compressor a query and the documents
a base retriever just returned, and takes a filtered list back. That is the same
job barber's published benchmark measured — query plus candidate passages, keep
what answers the question — so it is the one LangChain slot where those numbers
describe the work being done rather than something adjacent to it.

    from langchain_classic.retrievers import ContextualCompressionRetriever
    from barber.integrations.langchain import BarberDocumentCompressor

    retriever = ContextualCompressionRetriever(
        base_compressor=BarberDocumentCompressor(keep=0.6),
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
    )
    docs = retriever.invoke("What is the refund policy?")

(`ContextualCompressionRetriever` moved to `langchain_classic.retrievers` in
LangChain 1.x; on 0.x it is `langchain.retrievers`. This module only imports
`langchain_core`, which both share.)

The alternative in that slot, `LLMChainExtractor`, spends one LLM call per
retrieved document per query. barber spends none: scoring is an embedding pass,
or pure lexical math with no dependencies at all.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import ConfigDict

from ..core import SelectionConfig
from ._documents import select_documents

__all__ = ["BarberDocumentCompressor"]


class BarberDocumentCompressor(BaseDocumentCompressor):
    """Drop the retrieved documents that do not bear on the query.

    Surviving documents are returned as the exact objects that came in — same
    `page_content`, same `metadata`, same order. Nothing is rewritten,
    summarized, or re-ranked, so anything downstream that keys off document
    identity or metadata (citations, source links, dedup) still works.

    That is a filter, not an extractor, and the difference is worth being
    explicit about: a long document that is half relevant comes back whole. See
    "What this cannot express" below.

    Args:
        keep: fraction of documents retained by the budget. 0.6 is the
            benchmark default. It is a budget, not a quota, and the guards move
            the real count in both directions: pinning and lead/tail keep
            documents the budget would have cut, and the relevance floor
            (`cfg.relative_floor`, 0.35) drops documents scoring far below the
            best one even when the budget had room. On a set where only one or
            two documents share anything with the query — the normal case for
            the lexical fallback, which matches words rather than meaning — the
            floor is what decides, and `keep` barely shows up in the result.
        embedder: `embed_fn(list[str]) -> list[vector]`, or None for barber's
            deterministic lexical fallback (zero dependencies, matches words
            rather than meaning). For retrieval you almost certainly already
            have an encoder — `barber.embedders.sentence_transformers()` or
            `barber.embedders.endpoint()` wrap one in the shape this wants.
        cfg: a `SelectionConfig` for everything else: `pin_patterns` for a
            non-English corpus, `min_chunks`, `relative_floor`. `keep` above
            overrides its keep-ratio bounds.

    When it does nothing (returns the input list unchanged):

    - fewer than `cfg.min_chunks` documents (4). Below that there is nothing to
      choose between.
    - the documents joined together come to under `cfg.min_message_chars`
      characters (800). That is barber's "this is not retrieved context" gate,
      and it is sized for Latin script: lower it for CJK.
    - an empty query.

    What this cannot express:

    - **Per-document trimming.** barber decides at document granularity here, so
      a kept document keeps all of its text. Trimming *inside* a large document
      is `barber.trim` on a message list, which is a different call with a
      different shape.
    - **Scores.** `compress_documents` returns a subset, not a ranking, and
      barber's cosine scores are internal to the selection call. Documents come
      back in retrieval order with no `relevance_score` written to metadata.
    - **Callbacks.** The `callbacks` argument is accepted and ignored. It exists
      to trace the LLM call a compressor might make; barber makes none.

    Async: `acompress_documents` is inherited. The base class runs the sync path
    in an executor, which is right here — selection is CPU work with no I/O to
    await, so a hand-written coroutine would only block the event loop.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    keep: float = 0.6
    embedder: Optional[Callable[[list[str]], list[Any]]] = None
    cfg: Optional[SelectionConfig] = None

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Return the subset of `documents` that barber kept for `query`."""
        docs = list(documents)
        if not docs:
            return docs
        flags = select_documents(
            [d.page_content for d in docs],
            query,
            keep=self.keep,
            embedder=self.embedder,
            cfg=self.cfg,
        )
        return [d for d, keep in zip(docs, flags) if keep]
