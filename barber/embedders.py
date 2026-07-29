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

# Unicode-aware. `[a-z0-9]+` (what this was) tokenized "café" as "caf" and any
# non-Latin script as nothing at all, so the zero-dependency path degraded to a
# no-op outside English without saying so. `[^\W_]+` is \w minus underscore, so
# ASCII behaviour is byte-identical to before while accented Latin, Cyrillic,
# Greek, Hebrew, Arabic and Hangul now tokenize as words.
#
# The leading alternative gives CJK one token per character: Chinese and
# Japanese are unspaced, so a word regex would swallow a whole sentence into one
# token that matches nothing. Character unigrams are a weak but real overlap
# signal there — the honest ceiling of a lexical fallback. For CJK relevance
# that actually works, inject `sentence_transformers()` or `endpoint()`.
# The JS port carries the same regex as `[\p{L}\p{N}]` (JS `\w` is ASCII-only
# even under /u); the golden fixtures pin the two together.
_TOK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]|[^\W_]+")


def _tokenize(s: str) -> list[str]:
    return _TOK.findall(s.lower())


def lexical(corpus_df: Optional[dict] = None) -> Callable[[list[str]], list[dict]]:
    """Deterministic BM25-lite scorer used as a *fallback* when no vector encoder
    is injected. Returns bag-of-words idf-weighted term vectors (as dicts).
    Zero dependencies, but lexical only: it matches words, not meaning. For real
    semantic relevance (paraphrase-safe), use ``sentence_transformers()`` or
    ``endpoint()`` — strictly better on paraphrase.

    Tokenization is Unicode-aware (see ``_TOK``), so any alphabetic script
    scores. It is still word-overlap: CJK falls back to character unigrams, and
    morphologically rich languages match on surface forms because there is no
    stemmer. Pinning (``SelectionConfig.pin_patterns``) ships English patterns
    only — set your own for another language."""
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


def sentence_transformers(model_name: str = "BAAI/bge-small-en-v1.5",
                          trust_remote_code: bool = False):
    """Semantic embedder via sentence-transformers (``pip install barber-llm[semantic]``).
    Matches meaning, not words — fixes the paraphrase gold-drops the lexical
    fallback suffers. Works with any HF embedding checkpoint. Recommended:
    ``llm-semantic-router/mmbert-embed-32k-2d-matryoshka`` (32K context window,
    matryoshka, needs ``trust_remote_code=True``); ``BAAI/bge-small-en-v1.5`` is
    the lightweight alternative and does not.

    ``trust_remote_code`` executes Python that ships with the HF repo, at load
    time, with your process's privileges. It used to be hardcoded True here, so
    any ``model_name`` — including one arriving from config — ran arbitrary code.
    Opt in per model, after you have looked at the repo."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
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
