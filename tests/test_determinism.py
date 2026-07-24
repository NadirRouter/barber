"""Determinism and the freeze-on-first-sight / prefix-stability property."""
import hashlib
import json

from barber import trim, make_transform, Cache

PASSAGES = [
    "The refund policy allows returns within 30 days of purchase for a full refund.",
    "Photosynthesis converts light energy into chemical energy inside plant cells.",
    "Quarterly revenue increased twelve percent in the enterprise segment last year.",
    "The Eiffel Tower was completed in 1889 for the World's Fair held in Paris.",
    "Customer retention improved across all major regions during the past year.",
    "Mitochondria are the powerhouse organelles found in most eukaryotic cells.",
    "Shipping is free on orders over fifty dollars anywhere within the country.",
    "The Amazon rainforest spans nine countries across the South American continent.",
    "Honeybees communicate the direction of food sources through a waggle dance.",
    "Glaciers store roughly three quarters of the world's supply of fresh water.",
    "The printing press spread rapidly across Europe during the fifteenth century.",
    "Coral reefs support about a quarter of all known marine species on Earth.",
]
BLOCK = "\n\n".join(PASSAGES)
assert len(BLOCK) >= 800, "fixture must clear the min_message_chars threshold"


def canon(messages):
    return json.dumps(messages, sort_keys=True, ensure_ascii=False)


def test_same_input_twice_is_byte_identical():
    messages = [{"role": "user", "content": BLOCK},
                {"role": "user", "content": "What is the refund policy?"}]
    first = trim(messages, keep=0.6)
    second = trim(messages, keep=0.6)
    assert first.changed and second.changed
    assert canon(first.messages) == canon(second.messages)
    assert first.tokens_saved == second.tokens_saved
    assert first.chunks_dropped == second.chunks_dropped


def test_prefix_stays_stable_across_turns_with_shared_cache():
    """Freeze-on-first-sight: a growing conversation must reproduce the selected
    history block byte-identically even though the query changes every turn —
    otherwise the provider prompt cache misses on every request."""
    cache = {}
    _, fn = make_transform(keep=0.6, cache=cache)

    turn1 = fn([{"role": "user", "content": BLOCK},
                {"role": "user", "content": "What is the refund policy?"}])[0]
    turn2 = fn([{"role": "user", "content": BLOCK},
                {"role": "assistant", "content": "Within 30 days."},
                {"role": "user", "content": "And what about shipping costs?"}])[0]
    turn3 = fn([{"role": "user", "content": BLOCK},
                {"role": "assistant", "content": "Within 30 days."},
                {"role": "user", "content": "And what about shipping costs?"},
                {"role": "assistant", "content": "Free over fifty dollars."},
                {"role": "user", "content": "Tell me about the Eiffel Tower."}])[0]

    assert turn1[0]["content"] == turn2[0]["content"], "history block changed on turn 2"
    assert turn1[0]["content"] == turn3[0]["content"], "history block changed on turn 3"
    assert "omitted" in turn1[0]["content"], "selection never fired; test is vacuous"
    assert len(cache) == 1, f"one block should mean one cache entry, got {len(cache)}"


def other_block(tag):
    """A distinct selectable block: over 800 chars and well over 4 chunks, or
    core skips it and any cache assertion below passes vacuously."""
    block = "\n\n".join(
        f"{tag} passage {j}: retrieved prose about subject {j} with enough words "
        f"in it to clear the minimum block size that selection requires."
        for j in range(8))
    assert len(block) >= 800, "fixture too small to be selectable"
    return block


def test_cache_evicts_and_stays_bounded():
    """A plain dict grows forever; Cache holds the bound and drops the oldest."""
    cache = Cache(maxsize=10)
    _, fn = make_transform(keep=0.6, cache=cache)
    for i in range(50):
        fn([{"role": "user", "content": other_block(f"Tenant {i}")},
            {"role": "user", "content": f"question {i}?"}])
    assert len(cache) == 10, f"cache should stay bounded at 10, got {len(cache)}"
    assert "Tenant 49" in list(cache.values())[-1][0], "newest entry should survive"

    plain = {}
    _, fn2 = make_transform(keep=0.6, cache=plain)
    for i in range(50):
        fn2([{"role": "user", "content": other_block(f"Tenant {i}")},
             {"role": "user", "content": f"question {i}?"}])
    assert len(plain) == 50, "a plain dict is expected to grow without bound"


def test_live_conversation_survives_eviction_pressure():
    """The promise: one conversation's history block stays byte-identical across
    turns even while other traffic churns the cache. Each turn asks a DIFFERENT
    question, so if the block is ever evicted it gets re-decided against the new
    question and the prefix mutates. LRU keeps the touched block; FIFO does not.
    """
    cache = Cache(maxsize=5)
    _, fn = make_transform(keep=0.6, cache=cache)
    first = fn([{"role": "user", "content": BLOCK},
                {"role": "user", "content": "What is the refund policy?"}])[0][0]["content"]
    assert "omitted" in first, "block was never selected; test would be vacuous"

    # questions that genuinely move the selection, so a re-decided block shows up
    follow_ups = ["Tell me about the Eiffel Tower.",
                  "How does photosynthesis work?",
                  "What happened to quarterly revenue?",
                  "Where is the Amazon rainforest?"]
    for i in range(20):
        fn([{"role": "user", "content": other_block(f"Other {i}")},
            {"role": "user", "content": f"q{i}?"}])
        turn = fn([{"role": "user", "content": BLOCK},
                   {"role": "user", "content": follow_ups[i % len(follow_ups)]}])[0][0]["content"]
        assert turn == first, f"history block mutated on turn {i} under cache pressure"

    assert len(cache) == 5, f"cache should stay bounded at 5, got {len(cache)}"
    hot_key = hashlib.md5(BLOCK.encode()).hexdigest()[:12]
    assert hot_key in cache, "the live conversation's block should still be resident"


def test_fresh_caches_agree_with_each_other():
    """Two independent trims (separate caches) of the same request must agree —
    the decision is a pure function of the block and the query."""
    messages = [{"role": "user", "content": BLOCK},
                {"role": "user", "content": "What is the refund policy?"}]
    a = trim(messages, keep=0.6, cache={})
    b = trim(messages, keep=0.6, cache={})
    assert canon(a.messages) == canon(b.messages)
