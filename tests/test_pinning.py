"""Guards: pinned chunks survive even at an aggressive keep=0.1 budget."""
from barber import trim

# Filler chunks avoid pin patterns and every 4+ letter query token, so nothing
# here survives by accident. Each is long enough that the block clears 800 chars.
FILLER = [
    "Photosynthesis converts light energy into chemical energy inside green plant cells.",
    "Quarterly revenue increased twelve percent in the enterprise segment over the year.",
    "Mitochondria are the powerhouse organelles found in nearly all eukaryotic cells.",
    "The Amazon rainforest spans nine countries across the South American continent.",
    "Honeybees communicate the direction of food sources through an intricate waggle dance.",
    "Coral reefs support about a quarter of all known marine species on the planet.",
    "The printing press spread rapidly across Europe during the fifteenth century.",
]

QUERY = "Where is the zephyrite deposit that the survey team mapped?"

# scores top for the query (repeats its common terms), so it wins the tiny budget
DECOY = ("The survey team mapped every deposit in the region, and the survey "
         "team logged each deposit the survey covered in the mapped area.")
# the one chunk with the rare query entity: 'zephyrite' appears nowhere else
GOLD = "Field notes place the zephyrite seam under the eastern ridge, far from town."
# deontic wording triggers the constraint pin despite zero query overlap
DEONTIC = "Contractors must not pour concrete while the ground temperature is below freezing."

LEAD = FILLER[0]
TAIL = FILLER[1]


def build_messages():
    chunks = [LEAD, DECOY, GOLD, FILLER[2], DEONTIC, FILLER[3], FILLER[4], FILLER[5], FILLER[6], TAIL]
    context = "\n\n".join(chunks)
    assert len(context) >= 800
    return [{"role": "user", "content": context}, {"role": "user", "content": QUERY}], chunks


def test_deontic_chunk_survives_keep_010():
    messages, _ = build_messages()
    result = trim(messages, keep=0.1)
    assert result.changed is True, "budget this tight must drop something"
    assert DEONTIC in result.messages[0]["content"], "'must not' chunk was dropped"


def test_rare_query_entity_chunk_survives_keep_010():
    messages, _ = build_messages()
    result = trim(messages, keep=0.1)
    out = result.messages[0]["content"]
    assert GOLD in out, "the one chunk containing the rare query term was dropped"
    # sanity: the budget was actually tight — an unpinned filler chunk fell out
    assert any(f not in out for f in FILLER[2:]), "nothing was dropped; test is vacuous"


def test_lead_and_tail_survive_keep_010():
    messages, _ = build_messages()
    result = trim(messages, keep=0.1)
    out = result.messages[0]["content"]
    assert LEAD in out, "lead chunk was dropped"
    assert TAIL in out, "tail chunk was dropped"
