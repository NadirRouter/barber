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


def growing_conversation(turns):
    """A conversation where a NEW selectable block lands every turn, which is
    what an agent session actually looks like: tool results accumulate. The
    existing prefix test has its one block present from turn 1, so it never
    exercises whether a LATER arrival disturbs an EARLIER decision.

    The questions deliberately SHARE vocabulary with the blocks and each targets
    a different subject, so a given block really is scored differently from one
    turn to the next. Questions drawn from unrelated prose would score every
    chunk zero, ties would resolve by index identically every turn, and the
    prefix would come out stable even with the memoization torn out — a test
    that passes for a reason unrelated to what it claims to check.
    """
    msgs = []
    for t in range(turns):
        msgs.append({"role": "user", "content": chunked_block(f"turn{t}")})
        msgs.append({"role": "assistant", "content": f"noted {t}"})
        msgs.append({"role": "user", "content":
                     f"what does the note on subject {(t * 3) % 8} say?"})
    return msgs


def chunked_block(tag):
    """Selectable block whose chunks are distinguishable from one another, so
    which ones win depends on the question being asked."""
    block = "\n\n".join(
        f"{tag} note on subject {j}: the operator reviews subject {j} during the "
        f"weekly pass and records anything unusual about subject {j} in the log."
        for j in range(8))
    assert len(block) >= 800, "fixture too small to be selectable"
    return block


def test_appending_a_turn_never_rewrites_an_earlier_one():
    """Prefix-monotonicity, as a property over every prefix rather than three
    hand-written turns: for each k, trimming the first k messages must agree
    with trimming the first k+1 on every message they share.

    This is the invariant design decisions 1 and 2 exist to protect. If it
    fails, the provider prompt cache misses on every request and the memoization
    is not merely useless but actively expensive.
    """
    full = growing_conversation(6)
    cache = Cache()
    _, fn = make_transform(keep=0.6, cache=cache)

    # Compare at TURN BOUNDARIES only — prefixes ending on the user's question,
    # which are the only states ever sent to a provider. A prefix cut mid-turn
    # ends on the context block itself, and that block is then the latest user
    # message, which selection deliberately never prunes (core.py: the live
    # question is not history). Including those states would flag that
    # by-design transition as a prefix mutation.
    renderings = []
    for t in range(1, len(full) // 3 + 1):
        out, _ = fn([dict(m) for m in full[:t * 3]])
        renderings.append([m.get("content") for m in out])

    for k in range(len(renderings) - 1):
        shorter, longer = renderings[k], renderings[k + 1]
        assert shorter == longer[:len(shorter)], (
            f"trimming {k + 2} messages rewrote one of the first {k + 1}: "
            f"appending a turn mutated the prefix")

    # Guard against a vacuous pass: the property is trivially true if selection
    # never fired at all.
    assert any("omitted" in (c or "") for c in renderings[-1]), \
        "selection never fired; the property held for the wrong reason"


def test_without_a_shared_cache_the_prefix_is_not_stable():
    """The documented footgun, pinned. `trim(messages)` with no `cache=` starts
    a fresh decision cache per call and re-selects each block against that
    turn's question, so the same history block comes back with different bytes
    as the conversation moves on.

    This asserts the CURRENT behaviour, not a desirable one. If someone makes
    trim() cache globally by default, this test should fail and be deleted
    along with the warning in its docstring — that is the point of pinning it.
    """
    block = other_block("hist")
    queries = ["what does passage 2 say?", "anything about subject 7?",
               "summarize the retrieved prose"]
    seen = {
        trim([{"role": "user", "content": block},
              {"role": "user", "content": q}], keep=0.6).messages[0]["content"]
        for q in queries
    }
    assert len(seen) > 1, (
        "the un-cached prefix stayed stable; either trim() now shares a cache "
        "by default (good — delete this test and the docstring warning) or the "
        "fixture stopped discriminating between queries")
