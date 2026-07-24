"""Embedders — the scorers barber uses to rank chunks against the query.

Each factory returns ``embed_fn(list[str]) -> list[vector]``. Vectors are either
term-weight dicts (the lexical fallback) or dense vectors (real encoders); the
core scorer picks the matching cosine automatically.

Ported verbatim from the validated ``context_selection.py`` embedder factories:
``lexical`` = ``make_lexical_embedder``, ``sentence_transformers`` =
``make_st_embedder``, ``endpoint`` = ``make_vllm_embedder``. Only the names and
docstrings changed; the bodies did not.
"""
from __future__ import annotations
import math
import re
from typing import Callable, Optional

_TOK = re.compile(r"[a-z0-9]+")


def _tokenize(s: str) -> list[str]:
    return _TOK.findall(s.lower())


def lexical(corpus_df: Optional[dict] = None) -> Callable[[list[str]], list[dict]]:
    """Deterministic BM25-lite scorer used as a *fallback* when no vector encoder
    is injected. Returns bag-of-words idf-weighted term vectors (as dicts).
    Zero dependencies, but lexical only: it matches words, not meaning. For real
    semantic relevance (paraphrase-safe), use ``sentence_transformers()`` or
    ``endpoint()`` — strictly better on paraphrase."""
    def embed(texts: list[str]) -> list[dict]:
        # idf over the batch (the request's own chunks + query)
        df = {}
        toks_list = [_tokenize(t) for t in texts]
        for toks in toks_list:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        n = len(texts)
        vecs = []
        for toks in toks_list:
            tf = {}
            for w in toks:
                tf[w] = tf.get(w, 0) + 1
            vec = {w: (c / len(toks)) * math.log((n + 1) / (df[w])) for w, c in tf.items()}
            vecs.append(vec)
        return vecs
    return embed


def sentence_transformers(model_name: str = "BAAI/bge-small-en-v1.5"):
    """Semantic embedder via sentence-transformers (``pip install barber-llm[semantic]``).
    Matches meaning, not words — fixes the paraphrase gold-drops the lexical
    fallback suffers. Works with any HF embedding checkpoint. Recommended:
    ``llm-semantic-router/mmbert-embed-32k-2d-matryoshka`` (32K context window,
    matryoshka); ``BAAI/bge-small-en-v1.5`` is the lightweight alternative."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, trust_remote_code=True)
    def embed(texts):
        return model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return embed


def endpoint(base_url: str = "http://localhost:8000/v1",
             model: str = "BAAI/bge-small-en-v1.5",
             api_key: str = "EMPTY", batch_size: int = 64):
    """Embedder backed by an OpenAI compatible /v1/embeddings endpoint, e.g. vLLM
    (``vllm serve <model> --task embed``). Same scores as loading the model
    locally, but GPU-batched and out-of-process — the production deployment of
    the encoder. Use a RETRIEVAL-trained model (bge/e5/gte/Qwen3-embedding), not
    an intent classifier. Requires the ``openai`` package."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)
    def embed(texts):
        texts = list(texts)
        out = []
        for i in range(0, len(texts), batch_size):
            r = client.embeddings.create(model=model, input=texts[i:i + batch_size])
            out.extend(d.embedding for d in r.data)
        return out   # dense vectors -> scored by _cos_vec
    return embed
