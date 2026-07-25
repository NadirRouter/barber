"""trim() mechanics: what gets trimmed, what is never touched, marker accounting."""
import re

from barber import trim, Cache
from barber.core import SelectionConfig

# The locked default marker (benchmark-validated wording). If this assertion
# fails, a locked default was changed — see the README "locked defaults" note.
LOCKED_MARKER = "[… {n} passage(s) omitted as not relevant to this question — the remaining context is sufficient …]"
MARKER_RE = re.compile(re.escape(LOCKED_MARKER).replace(re.escape("{n}"), r"(\d+)"))

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


def block(chunks):
    return "\n\n".join(chunks)


def msgs(context, query=QUERY):
    return [{"role": "user", "content": context}, {"role": "user", "content": query}]


def test_default_marker_is_the_locked_wording():
    assert SelectionConfig().drop_marker == LOCKED_MARKER


def test_trims_a_15_chunk_block_at_default_keep():
    context = block(TOPICS)
    assert len(context) >= 800 and len(TOPICS) == 15
    result = trim(msgs(context), keep=0.6)
    out_text = result.messages[0]["content"]
    assert result.changed is True
    assert len(out_text) < len(context)
    assert result.chunks_dropped > 0
    assert result.tokens_saved > 0
    assert MARKER_RE.search(out_text), "dropped runs should leave a marker"
    # the answer chunk survives (rare query terms 'refund'/'policy' pin it)
    assert TOPICS[0] in out_text


def test_marker_appears_exactly_once_per_dropped_run():
    result = trim(msgs(block(TOPICS)), keep=0.6)
    out_text = result.messages[0]["content"]
    present = [t in out_text for t in TOPICS]
    dropped = present.count(False)
    assert dropped > 0
    # count maximal runs of consecutive dropped chunks
    runs = sum(1 for i, p in enumerate(present) if not p and (i == 0 or present[i - 1]))
    marker_ns = [int(n) for n in MARKER_RE.findall(out_text)]
    assert len(marker_ns) == runs, "one marker per dropped run"
    assert sum(marker_ns) == dropped, "marker counts must add up to dropped chunks"
    assert result.chunks_dropped == dropped


def test_latest_user_message_is_never_trimmed():
    # even a huge, chunky latest user message is the query — untouched
    big_query = block(TOPICS)
    result = trim([{"role": "user", "content": big_query}], keep=0.6)
    assert result.messages[0]["content"] == big_query
    assert result.changed is False
    # and in the two-message form, the trailing query survives verbatim
    result = trim(msgs(block(TOPICS)), keep=0.6)
    assert result.messages[-1]["content"] == QUERY


def test_the_question_itself_is_never_trimmed_even_when_selectable():
    """The earlier no-op case passes even without the guard, because those
    messages fail the size/chunk gates anyway. This one is fully selectable:
    long, chunky, repeated vocabulary so no chunk is rare-pinned. Only the
    latest-user-message guard keeps it intact."""
    rows = "\n\n".join(
        f"2026-07-24T10:0{i}:11 worker-{i} retry attempt {i} for request routing "
        f"in the retry queue, retry backoff applied to the routing worker."
        for i in range(12))
    pasted = f"Here is the log I pasted, please explain the retries:\n\n{rows}"
    assert len(pasted) >= 800
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": pasted}]
    result = trim(messages, keep=0.4)
    assert result.messages[-1]["content"] == pasted, "the user's own question was trimmed"
    assert result.changed is False
    assert result.chunks_dropped == 0


def test_tool_and_function_messages_are_selectable():
    """Large tool output is the headline use case and had no positive test:
    only the negative system/assistant cases were covered."""
    context = block(TOPICS)
    for role in ("tool", "function"):
        result = trim([{"role": role, "content": context},
                       {"role": "user", "content": QUERY}], keep=0.4)
        assert result.changed is True, f"{role} message was not selected"
        assert result.chunks_dropped > 0, f"{role} message lost no chunks"
        assert MARKER_RE.search(result.messages[0]["content"]), f"no marker in {role} output"


def test_system_message_is_never_trimmed():
    sys_text = block(TOPICS)
    messages = [{"role": "system", "content": sys_text}] + msgs(block(TOPICS))
    result = trim(messages, keep=0.6)
    assert result.messages[0]["content"] == sys_text
    assert result.changed is True  # the user block still gets trimmed


def test_assistant_messages_are_not_selectable():
    assistant_text = block(TOPICS)
    messages = [{"role": "assistant", "content": assistant_text},
                {"role": "user", "content": QUERY}]
    result = trim(messages, keep=0.6)
    assert result.messages[0]["content"] == assistant_text
    assert result.changed is False


def test_block_under_4_chunks_is_a_noop():
    three = ["This paragraph talks about glaciers. " * 10,
             "This paragraph talks about coral reefs. " * 10,
             "This paragraph talks about volcanoes. " * 10]
    context = block([t.strip() for t in three])
    assert len(context) >= 800
    result = trim(msgs(context), keep=0.6)
    assert result.messages[0]["content"] == context
    assert result.changed is False
    assert result.chunks_dropped == 0
    assert result.tokens_saved == 0


def test_block_under_800_chars_is_a_noop():
    context = block(TOPICS[1:6])  # 5 chunks, well under 800 chars
    assert len(context) < 800
    result = trim(msgs(context), keep=0.6)
    assert result.messages[0]["content"] == context
    assert result.changed is False
    assert result.chunks_dropped == 0


def test_chunks_dropped_survives_a_warm_cache():
    """chunks_dropped used to come from SelectionStats, which core only writes
    when it actually selects. On a cache hit it stayed zero while the content
    was still trimmed, so every turn after the first under-reported."""
    cache = Cache()
    messages = msgs(block(TOPICS))
    reports = [trim(messages, keep=0.6, cache=cache) for _ in range(3)]
    assert all(r.changed for r in reports)
    counts = [r.chunks_dropped for r in reports]
    assert counts[0] > 0
    assert len(set(counts)) == 1, f"chunks_dropped drifted across turns: {counts}"
    # and it still agrees with the markers actually present
    present = [t in reports[-1].messages[0]["content"] for t in TOPICS]
    assert counts[-1] == present.count(False)


def test_tokens_saved_is_signed_when_markers_cost_more():
    """Scattered drops emit one marker each, and a marker is ~20 tokens. When
    the chunks are small that is a net loss, and clamping it to 0 made a real
    regression indistinguishable from a clean no-op."""
    rows = "\n\n".join(f"row {i} alpha beta" if i % 2 else f"row {i} refund policy detail here"
                       for i in range(40))
    result = trim(msgs(rows, "refund policy?"), keep=0.5)
    assert result.changed is True
    assert result.chunks_dropped > 0
    assert result.tokens_saved < 0, (
        f"expected a net loss to surface as a negative, got {result.tokens_saved}")


def test_changed_flag_matches_output():
    trimmed = trim(msgs(block(TOPICS)), keep=0.6)
    assert trimmed.changed is True
    assert trimmed.messages != msgs(block(TOPICS))
    untouched = trim(msgs(block(TOPICS[1:6])), keep=0.6)
    assert untouched.changed is False
    assert untouched.messages == msgs(block(TOPICS[1:6]))
