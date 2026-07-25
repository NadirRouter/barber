"""The premise: chunks survive because they are RELEVANT, not just guarded.

Every other fixture in this suite is guard-saturated, so a chunk survives via
lead/tail keep, a pin, or the relevance floor. That means relevance ranking
itself can be broken (scores replaced with a constant, or the sort inverted so
the LEAST relevant chunks win) and the rest of the suite stays green.

The fixtures here are deliberately homogeneous: every chunk shares the query's
vocabulary, so no query term is rare enough to pin, and no chunk is unrelated
enough for the floor to do the work. What is left is ranking.
"""
from barber import trim

# Shared vocabulary everywhere ("refund", "policy", "section"), so rare-entity
# pinning cannot fire. No deontic words, so constraint pinning cannot fire.
FILLERS = [
    f"Section {i} of the handbook covers the refund policy for orders placed in "
    f"region {i}, where the refund policy is summarized for staff in section {i}."
    for i in range(10)
]
# Same words, much denser: this should win on cosine alone.
GOLD = ("The refund policy grants a refund within thirty days of purchase: the "
        "refund policy issues the refund to the original card, and the refund "
        "policy applies to every refund request under the policy.")
QUERY = "What does the refund policy say about a refund?"


def block_with_gold_in_the_middle():
    chunks = FILLERS[:5] + [GOLD] + FILLERS[5:]
    assert chunks.index(GOLD) not in (0, len(chunks) - 1), "gold must not be lead or tail"
    text = "\n\n".join(chunks)
    assert len(text) >= 800
    return chunks, text


def test_the_most_relevant_chunk_survives_a_tight_budget():
    """Kills the inverted-ranking and constant-score mutants."""
    chunks, text = block_with_gold_in_the_middle()
    result = trim([{"role": "user", "content": text},
                   {"role": "user", "content": QUERY}], keep=0.2)
    out = result.messages[0]["content"]
    kept = [c for c in chunks if c in out]
    assert result.changed, "budget this tight must drop something"
    assert len(kept) < len(chunks), "nothing was dropped; test would be vacuous"
    assert GOLD in out, "the chunk that best answers the query was dropped"


def test_ranking_prefers_the_dense_chunk_over_the_filler():
    """Gold must outrank filler, not merely survive alongside all of it."""
    chunks, text = block_with_gold_in_the_middle()
    result = trim([{"role": "user", "content": text},
                   {"role": "user", "content": QUERY}], keep=0.2)
    out = result.messages[0]["content"]
    middle_fillers_kept = [c for c in FILLERS if c in out]
    assert GOLD in out
    # lead and tail are FILLERS[0] and FILLERS[9], kept by the lead/tail guard;
    # anything beyond those two would mean ranking is not discriminating
    assert len(middle_fillers_kept) <= 2, (
        f"ranking kept {len(middle_fillers_kept)} fillers; gold should outrank them")


def test_keep_controls_how_much_survives():
    """Kills any mutant where `keep` stops reaching the budget: on a fixture with
    no floor saturation, more budget must mean more chunks."""
    chunks = [
        f"Section {i}: the deployment guide describes how the service handles "
        f"request routing and retries for cluster node number {i} in the fleet."
        for i in range(12)
    ]
    text = "\n\n".join(chunks)
    assert len(text) >= 800
    query = "How does the service handle request routing?"

    counts = []
    for keep in (0.1, 0.5, 0.9):
        out = trim([{"role": "user", "content": text},
                    {"role": "user", "content": query}], keep=keep).messages[0]["content"]
        counts.append(sum(1 for c in chunks if c in out))
    assert counts[0] < counts[1] < counts[2], (
        f"kept counts should rise with keep, got {counts}")
