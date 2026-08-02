"""Framework adapters: the document-set mapping, and the LangChain/LlamaIndex
classes built on it.

The mapping tests run everywhere — `_documents` imports nothing but barber. The
adapter tests `importorskip` their framework, because CI installs barber without
extras and an integration that forces its dependency on everyone is the thing
this package exists not to be.
"""
import subprocess
import sys

import pytest

from barber.core import SelectionConfig
from barber.integrations._documents import select_documents

# 15 single-paragraph passages, one of which answers the query: the shape the
# published benchmark trims (HotpotQA paragraphs + one question).
TOPICS = [
    "The refund policy allows returns within 30 days of purchase for a full refund.",
    "Photosynthesis converts light energy into chemical energy inside plant cells.",
    "Quarterly revenue increased twelve percent in the enterprise segment last year.",
    "The Eiffel Tower was completed in 1889 for the World's Fair held in Paris.",
    "Customer retention improved across all major regions during the past year.",
    "Mitochondria are the powerhouse organelles found in most eukaryotic cells.",
    "Shipping is free on orders over fifty dollars anywhere within the country.",
    "The Amazon rainforest spans nine countries across the South American continent.",
    "Honeybees communicate the direction of food sources through a waggle dance.",
    "The stock market closed higher on strong earnings from the technology sector.",
    "Glaciers store roughly three quarters of the world's supply of fresh water.",
    "The printing press spread rapidly across Europe during the fifteenth century.",
    "Coral reefs support about a quarter of all known marine species on Earth.",
    "The marathon distance was standardized at 42.195 kilometers in the year 1921.",
    "Volcanic soil is unusually fertile and supports intensive agriculture worldwide.",
]

QUERY = "What is the refund policy?"


# --- the zero-dependency promise ---------------------------------------------

# `dependencies = []` in pyproject.toml only means anything if importing the
# package pulls in nothing third-party. Adding integrations is exactly how that
# stops being true, so this runs in a fresh interpreter and names every offender.
_PROBE = (
    "import sys, json; "
    "before = set(sys.modules); "
    "import barber; "
    "new = {m.split('.')[0] for m in set(sys.modules) - before}; "
    "print(json.dumps(sorted(n for n in new "
    "if not n.startswith('_') and n != 'barber' "
    "and n not in sys.stdlib_module_names)))"
)


def test_importing_barber_pulls_in_nothing_third_party():
    out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"import barber now pulls in {out.stdout.strip()}"


def test_integrations_package_imports_nothing_by_itself():
    probe = _PROBE.replace("import barber;", "import barber.integrations;")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"barber.integrations now pulls in {out.stdout.strip()}"


# --- the mapping -------------------------------------------------------------

def test_drops_documents_and_keeps_the_answer():
    flags = select_documents(TOPICS, QUERY)
    assert len(flags) == len(TOPICS)
    assert flags[0] is True, "the document answering the query must survive"
    assert flags.count(False) > 0, "irrelevant documents should be dropped"


def test_lead_and_tail_documents_always_survive():
    flags = select_documents(TOPICS, QUERY)
    assert flags[0] and flags[-1]


def test_higher_keep_never_drops_more():
    counts = [select_documents(TOPICS, QUERY, keep=k).count(True) for k in (0.3, 0.6, 1.0)]
    assert counts == sorted(counts), f"keep should be monotone, got {counts}"
    assert select_documents(TOPICS, QUERY, keep=1.0).count(True) >= counts[0]


def test_deterministic():
    assert select_documents(TOPICS, QUERY) == select_documents(TOPICS, QUERY)


def test_pinned_document_survives_an_aggressive_budget():
    docs = list(TOPICS)
    # deontic language is pinned (SelectionConfig.pin_patterns) even though this
    # sentence shares nothing with the query
    docs[7] = "Operators must never disclose the escrow account number to a caller."
    flags = select_documents(docs, QUERY, keep=0.25)
    assert flags[7] is True


def test_a_multi_paragraph_document_survives_if_any_part_does():
    docs = list(TOPICS)
    docs.insert(6, "A note on refunds and returns.\n\nUnrelated aside about glaciers.")
    flags = select_documents(docs, QUERY)
    assert flags[6] is True
    assert len(flags) == len(docs)


# --- failing open ------------------------------------------------------------

def test_empty_query_is_a_no_op():
    assert select_documents(TOPICS, "   ") == [True] * len(TOPICS)


def test_too_few_documents_is_a_no_op():
    # under SelectionConfig.min_chunks (4) there is nothing to choose between
    assert select_documents(TOPICS[:3], QUERY) == [True, True, True]


def test_short_document_set_is_a_no_op():
    # under min_message_chars (800) joined, this is not retrieved context
    short = ["alpha beta", "gamma delta", "epsilon zeta", "eta theta", "iota kappa"]
    assert select_documents(short, "what about gamma?") == [True] * 5


def test_single_document_is_a_no_op():
    assert select_documents([" ".join(TOPICS)], QUERY) == [True]


def test_blank_documents_are_never_dropped():
    docs = list(TOPICS)
    docs.insert(4, "")
    docs.insert(9, "   \n ")
    flags = select_documents(docs, QUERY)
    assert flags[4] is True and flags[9] is True


def test_something_always_survives():
    # a query with no lexical overlap at all: every score is zero and the
    # relative floor has nothing to be relative to. Dropping the whole set is
    # never the answer.
    flags = select_documents(TOPICS, "zzqqxx wwvvuu ttssrr")
    assert any(flags)


def test_custom_config_is_honored():
    # a config whose marker differs must still map back: the marker regex is
    # built from the config, not from the shipped default
    cfg = SelectionConfig(drop_marker="<<{n} cut>>")
    flags = select_documents(TOPICS, QUERY, cfg=cfg)
    assert flags[0] is True
    assert flags.count(False) > 0


def test_custom_embedder_is_used():
    seen = []

    def embedder(texts):
        seen.append(len(texts))
        # rank by position: last chunk wins, everything else scores lower
        return [{"t": float(i)} for i in range(len(texts) - 1)] + [{"t": 1.0}]

    select_documents(TOPICS, QUERY, embedder=embedder)
    assert seen, "the injected embedder must actually be called"


# --- LangChain ---------------------------------------------------------------

def test_langchain_compressor():
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document
    from langchain_core.documents.compressor import BaseDocumentCompressor
    from barber.integrations.langchain import BarberDocumentCompressor

    docs = [Document(page_content=t, metadata={"i": i}) for i, t in enumerate(TOPICS)]
    compressor = BarberDocumentCompressor(keep=0.6)
    assert isinstance(compressor, BaseDocumentCompressor)

    out = list(compressor.compress_documents(docs, QUERY))
    assert 0 < len(out) < len(docs)
    # survivors are the input objects, untouched: same identity, same metadata
    assert all(any(o is d for d in docs) for o in out)
    assert [o.metadata["i"] for o in out] == sorted(o.metadata["i"] for o in out)
    assert out[0].metadata["i"] == 0, "the answering document survives"


def test_langchain_compressor_async_matches_sync():
    pytest.importorskip("langchain_core")
    import asyncio
    from langchain_core.documents import Document
    from barber.integrations.langchain import BarberDocumentCompressor

    docs = [Document(page_content=t) for t in TOPICS]
    compressor = BarberDocumentCompressor()
    sync = [d.page_content for d in compressor.compress_documents(docs, QUERY)]
    a = asyncio.run(compressor.acompress_documents(docs, QUERY))
    assert [d.page_content for d in a] == sync


def test_langchain_compressor_edge_cases():
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document
    from barber.integrations.langchain import BarberDocumentCompressor

    compressor = BarberDocumentCompressor()
    assert list(compressor.compress_documents([], QUERY)) == []
    docs = [Document(page_content=t) for t in TOPICS]
    assert len(list(compressor.compress_documents(docs, "  "))) == len(docs)
    # callbacks are accepted and ignored — barber makes no model call to trace
    assert compressor.compress_documents(docs, QUERY, callbacks=None) is not None


def test_langchain_compressor_config_passthrough():
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document
    from barber.integrations.langchain import BarberDocumentCompressor

    docs = [Document(page_content=t) for t in TOPICS]
    cfg = SelectionConfig(min_chunks=99)   # nothing has enough chunks: decline
    assert len(list(BarberDocumentCompressor(cfg=cfg).compress_documents(docs, QUERY))) == len(docs)
    # the dataclass survives pydantic's validation as the same object
    assert BarberDocumentCompressor(cfg=cfg).cfg is cfg


def test_langchain_wires_into_contextual_compression_retriever():
    pytest.importorskip("langchain_core")
    retrievers = pytest.importorskip("langchain_classic.retrievers")
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from barber.integrations.langchain import BarberDocumentCompressor

    class Fixed(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager=None, **kwargs):
            return [Document(page_content=t) for t in TOPICS]

    retriever = retrievers.ContextualCompressionRetriever(
        base_compressor=BarberDocumentCompressor(keep=0.6),
        base_retriever=Fixed(),
    )
    out = retriever.invoke(QUERY)
    assert 0 < len(out) < len(TOPICS)
    assert out[0].page_content == TOPICS[0]


# --- LlamaIndex --------------------------------------------------------------

def test_llama_index_postprocessor():
    pytest.importorskip("llama_index.core")
    from llama_index.core.schema import NodeWithScore, TextNode
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core import QueryBundle
    from barber.integrations.llama_index import BarberNodePostprocessor

    nodes = [NodeWithScore(node=TextNode(text=t, id_=str(i)), score=1.0 - i / 100)
             for i, t in enumerate(TOPICS)]
    pp = BarberNodePostprocessor(keep=0.6)
    assert isinstance(pp, BaseNodePostprocessor)

    out = pp.postprocess_nodes(nodes, QueryBundle(query_str=QUERY))
    assert 0 < len(out) < len(nodes)
    assert all(any(o is n for n in nodes) for o in out)
    assert out[0].node.node_id == "0", "the answering node survives"
    assert out[0].score == nodes[0].score, "scores are passed through, not rewritten"


def test_llama_index_async_matches_sync():
    pytest.importorskip("llama_index.core")
    import asyncio
    from llama_index.core import QueryBundle
    from llama_index.core.schema import NodeWithScore, TextNode
    from barber.integrations.llama_index import BarberNodePostprocessor

    nodes = [NodeWithScore(node=TextNode(text=t)) for t in TOPICS]
    pp = BarberNodePostprocessor()
    bundle = QueryBundle(query_str=QUERY)
    sync = [n.get_content() for n in pp.postprocess_nodes(nodes, bundle)]
    a = asyncio.run(pp.apostprocess_nodes(nodes, bundle))
    assert [n.get_content() for n in a] == sync


def test_llama_index_accepts_query_str_and_no_query():
    pytest.importorskip("llama_index.core")
    from llama_index.core.schema import NodeWithScore, TextNode
    from barber.integrations.llama_index import BarberNodePostprocessor

    nodes = [NodeWithScore(node=TextNode(text=t)) for t in TOPICS]
    pp = BarberNodePostprocessor()
    by_str = pp.postprocess_nodes(nodes, query_str=QUERY)
    assert 0 < len(by_str) < len(nodes)
    # no query at all: nothing to select against, so nothing is dropped
    assert len(pp.postprocess_nodes(nodes)) == len(nodes)
