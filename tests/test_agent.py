"""The sweep must drop what is provably dead and nothing else."""
from barber.agent import DUPLICATE, STALE, SUPERSEDED, sweep

BODY = "def f():\n    return 1\n" + "# filler line\n" * 60
OTHER = "def g():\n    return 2\n" + "# other filler\n" * 60


def use(tid, name, path, content=None, **extra):
    inp = {"file_path": path, **extra}
    if content is not None:
        inp["content"] = content
    return {"role": "assistant",
            "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}]}


def result(tid, text):
    return {"role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}]}


def bodies(msgs):
    out = []
    for m in msgs:
        if isinstance(m.get("content"), list):
            for p in m["content"]:
                if p.get("type") == "tool_use":
                    out.append(p["input"].get("content"))
                elif p.get("type") == "tool_result":
                    out.append(p.get("content"))
    return out


def test_later_full_write_kills_the_earlier_body():
    msgs = [use("1", "Write", "/a.py", BODY), result("1", "ok"),
            use("2", "Write", "/a.py", OTHER), result("2", "ok")]
    r = sweep(msgs)
    assert r.changed and r.tokens_saved > 0
    assert bodies(r.messages)[0] == SUPERSEDED.format(path="/a.py")
    assert bodies(r.messages)[2] == OTHER, "the live body must survive"


def test_a_later_edit_does_not_kill_an_earlier_write_body():
    # The file still holds that body; an Edit changed part of it. Dropping the
    # Write body here would throw away live file content.
    msgs = [use("1", "Write", "/a.py", BODY), result("1", "ok"),
            use("2", "Edit", "/a.py", old_string="x", new_string="y"), result("2", "ok")]
    assert not sweep(msgs).changed
    assert bodies(sweep(msgs).messages)[0] == BODY


def test_read_is_stale_once_the_file_is_edited():
    msgs = [use("1", "Read", "/a.py"), result("1", BODY),
            use("2", "Edit", "/a.py", old_string="x", new_string="y"), result("2", "ok")]
    r = sweep(msgs)
    assert bodies(r.messages)[1] == STALE.format(path="/a.py")


def test_read_of_an_untouched_file_survives():
    msgs = [use("1", "Read", "/a.py"), result("1", BODY),
            use("2", "Edit", "/b.py", old_string="x", new_string="y"), result("2", "ok")]
    assert not sweep(msgs).changed


def test_identical_results_collapse_after_the_first():
    msgs = [use("1", "Read", "/a.py"), result("1", BODY),
            use("2", "Read", "/a.py"), result("2", BODY)]
    r = sweep(msgs)
    assert bodies(r.messages)[1] == BODY, "the first copy is the one to keep"
    assert bodies(r.messages)[3] == DUPLICATE


def test_small_payloads_are_left_alone():
    # Not worth a prompt-cache re-prime.
    msgs = [use("1", "Write", "/a.py", "x = 1"), result("1", "ok"),
            use("2", "Write", "/a.py", "x = 2"), result("2", "ok")]
    assert not sweep(msgs).changed


def test_structure_survives_so_the_request_stays_valid():
    msgs = [use("1", "Write", "/a.py", BODY), result("1", "ok"),
            use("2", "Write", "/a.py", OTHER), result("2", "ok")]
    r = sweep(msgs)
    ids = [p["id"] for m in r.messages for p in m["content"] if p.get("type") == "tool_use"]
    rids = [p["tool_use_id"] for m in r.messages for p in m["content"]
            if p.get("type") == "tool_result"]
    assert ids == ["1", "2"] and rids == ["1", "2"]
    assert all(p["input"]["file_path"] for m in r.messages for p in m["content"]
               if p.get("type") == "tool_use"), "file_path must survive the drop"


def test_input_is_not_mutated():
    msgs = [use("1", "Write", "/a.py", BODY), result("1", "ok"),
            use("2", "Write", "/a.py", OTHER), result("2", "ok")]
    sweep(msgs)
    assert msgs[0]["content"][0]["input"]["content"] == BODY


def test_a_short_horizon_declines_the_re_prime():
    # One dead body, ~half the transcript. Over 10 turns the cache re-prime
    # costs more than never re-reading it saves, so the honest move is nothing.
    msgs = [use("1", "Write", "/a.py", BODY), result("1", "ok"),
            use("2", "Write", "/a.py", OTHER), result("2", "ok")]
    assert sweep(msgs).changed, "the drop is real; only the economics are in doubt"
    assert not sweep(msgs, remaining_turns=10).changed
    assert sweep(msgs, remaining_turns=500).changed, "enough turns to amortize it"


def test_the_cut_point_walks_past_the_edits_that_do_not_pay():
    # Dead body up front, a live wall of context, then a dead body at the end.
    # Editing from the front re-primes the wall; editing only the tail doesn't.
    big = "# a much larger file\n" * 1000
    wall = [(use(str(i), "Read", f"/w{i}.py"), result(str(i), f"line {i}\n" * 400))
            for i in range(6)]
    msgs = ([use("a", "Write", "/a.py", BODY), result("a", "ok")]
            + [m for pair in wall for m in pair]
            + [use("b", "Write", "/b.py", big), result("b", "ok"),
               use("c", "Write", "/a.py", BODY), result("c", "ok"),
               use("d", "Write", "/b.py", big), result("d", "ok")])
    r = sweep(msgs, remaining_turns=30)
    assert r.changed
    kept, dropped = bodies(r.messages)[0], bodies(r.messages)[-6]
    assert kept == BODY, "the early body is cached; re-priming past it costs more"
    assert dropped == SUPERSEDED.format(path="/b.py"), "the tail edit pays"


def test_noop_reports_nothing():
    r = sweep([{"role": "user", "content": "hi"}])
    assert not r.changed and r.tokens_saved == 0 and r.chunks_dropped == 0
