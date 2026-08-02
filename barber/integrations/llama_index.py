"""LlamaIndex integration: barber as a `BaseNodePostprocessor`.

    pip install "barber-llm[llamaindex]"

Same shape as the LangChain compressor, same underlying call: nodes plus a query
bundle in, a filtered list of nodes out.

    from llama_index.core import VectorStoreIndex
    from barber.integrations.llama_index import BarberNodePostprocessor

    engine = index.as_query_engine(
        similarity_top_k=20,
        node_postprocessors=[BarberNodePostprocessor(keep=0.6)],
    )
    engine.query("What is the refund policy?")

Retrieve wide, then let barber cut — that is the configuration the published
benchmark measured (a padded candidate set, one question), and it is where a
postprocessor earns anything at all. `similarity_top_k=3` leaves nothing to
select between, and barber will say so by returning all three.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from pydantic import ConfigDict, Field

from ..core import SelectionConfig
from ._documents import select_documents

__all__ = ["BarberNodePostprocessor"]


class BarberNodePostprocessor(BaseNodePostprocessor):
    """Drop the retrieved nodes that do not bear on the query.

    Survivors are the exact `NodeWithScore` objects that came in — same node,
    same `score`, same order. barber does not re-rank and does not write a score
    of its own, so a downstream `SimilarityPostprocessor` or a reranker still
    sees the retriever's numbers.

    Args:
        keep: fraction of nodes retained by the budget (benchmark default 0.6).
            A budget, not a quota: pinning and lead/tail keep nodes above it,
            and the relevance floor (`cfg.relative_floor`, 0.35) cuts below it.
        embedder: `embed_fn(list[str]) -> list[vector]`, or None for barber's
            deterministic lexical fallback. `barber.embedders` has wrappers for
            sentence-transformers and any OpenAI-compatible embeddings endpoint.
        cfg: a `SelectionConfig` for pinning patterns, chunk minimums, and the
            relevance floor. `keep` overrides its keep-ratio bounds.
        metadata_mode: which rendering of the node is scored. Defaults to
            `MetadataMode.LLM`, the text the LLM will actually be charged for,
            so metadata that is part of the prompt is part of the decision.
            `MetadataMode.NONE` scores the node text alone.

    It returns the node list untouched when there is no decision to make: fewer
    than `cfg.min_chunks` nodes (4), under `cfg.min_message_chars` characters
    (800) of text across all of them, or no query bundle.

    What this cannot express: barber decides whole nodes here, so a kept node
    keeps all of its text — this is a filter, not a compactor. Nodes are not
    rewritten and no relevance score is attached.

    Async `apostprocess_nodes` is inherited; the base class runs the sync path
    via `asyncio.to_thread`, which is right for CPU work with no I/O to await.
    It arrived in `llama-index-core` 0.13, which is why the extra floors there
    rather than at 0.11 (where the sync path alone works).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    keep: float = 0.6
    embedder: Optional[Callable[[list[str]], list[Any]]] = None
    cfg: Optional[SelectionConfig] = None
    metadata_mode: MetadataMode = Field(default=MetadataMode.LLM)

    @classmethod
    def class_name(cls) -> str:
        return "BarberNodePostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Return the subset of `nodes` that barber kept for the query."""
        if not nodes or query_bundle is None:
            return list(nodes)
        flags = select_documents(
            [n.get_content(metadata_mode=self.metadata_mode) for n in nodes],
            query_bundle.query_str,
            keep=self.keep,
            embedder=self.embedder,
            cfg=self.cfg,
        )
        return [n for n, keep in zip(nodes, flags) if keep]
