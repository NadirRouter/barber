"""The line-aware fallback must find chunks in agent output without moving any
decision the paragraph splitter already makes."""
from barber import trim
from barber.core import split_chunks


def read_output(*lines: str) -> str:
    """What a Read tool returns: every line carries a `N\\t` prefix, so the
    blank lines of the underlying file are not blank any more."""
    return "\n".join(f"{i + 1}\t{l}" for i, l in enumerate(lines))


def test_paragraphs_are_untouched():
    # Delimiters found: the paragraph splitter owns the answer, including when
    # it finds too few chunks to bother cutting. This is the benchmarked path.
    text = "\n\n".join(f"Paragraph {i} about a topic." for i in range(6))
    assert len(split_chunks(text)) == 6
    assert len(split_chunks("one para.\n\ntwo para.\n\nthree para.")) == 3


def test_prose_still_splits_on_sentences():
    text = " ".join(f"Sentence number {i} says a thing." for i in range(8))
    assert len(split_chunks(text)) == 8


def test_line_numbered_read_chunks_on_the_files_own_blank_lines():
    src = ["def a():", "    return 1", "", "def b():", "    return 2", "",
           "def c():", "    return 3", "", "def d():", "    return 4"]
    chunks = split_chunks(read_output(*src))
    assert len(chunks) == 4
    assert chunks[0].splitlines() == ["1\tdef a():", "2\t    return 1"]
    assert "\t" in chunks[-1], "line prefixes survive verbatim in the output"


def test_diff_chunks_per_hunk():
    diff = "\n".join(["diff --git a/x.py b/x.py", "--- a/x.py", "+++ b/x.py",
                      "@@ -1,2 +1,2 @@", "-old one", "+new one",
                      "@@ -9,2 +9,2 @@", "-old two", "+new two",
                      "@@ -20,2 +20,2 @@", "-old three", "+new three"])
    chunks = split_chunks(diff)
    assert len(chunks) == 4
    assert chunks[1].startswith("@@ -1,2")


def test_flat_line_output_falls_back_to_windows():
    grep = "\n".join(f"src/file{i}.py:{i}: match here" for i in range(60))
    chunks = split_chunks(grep)
    assert len(chunks) >= 4
    assert "\n".join(chunks) == grep, "windows must partition the input exactly"


def test_json_is_never_line_chunked():
    # Dropping lines out of a JSON body yields something that no longer parses.
    blob = "{\n" + "\n".join(f'  "key{i}": "value {i}",' for i in range(40)) + '\n  "last": 1\n}'
    assert len(split_chunks(blob)) < 4
    assert split_chunks(blob) == split_chunks(blob)


def test_short_text_is_left_alone():
    assert len(split_chunks("just three\nshort lines\nhere")) < 4


def test_a_read_result_now_actually_gets_trimmed():
    # The end-to-end shape: a tool result as an Anthropic content block. Before
    # the content-parts fix this was a guaranteed no-op.
    src = []
    for i in range(16):
        src += [f"def unrelated_helper_number_{i}(argument, other_argument):",
                f"    intermediate = compute_something_unrelated_{i}(argument)",
                f"    return intermediate + other_argument", ""]
    src += ["def parse_yaml_config(path):", "    return yaml.safe_load(path)"]
    msgs = [
        {"role": "user", "content": "read the config module"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/m.py"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": read_output(*src)}]},
        {"role": "user", "content": "what does parse_yaml_config do?"},
    ]
    r = trim(msgs, keep=0.6)
    assert r.changed and r.tokens_saved > 0 and r.chunks_dropped > 0
    kept = r.messages[2]["content"][0]["content"]
    assert "parse_yaml_config" in kept, "the chunk that answers the question survives"
    assert r.messages[3] == msgs[3], "the question itself is never trimmed"
