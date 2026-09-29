"""The Jev trial must use live scores and fail if its remote scorer fails."""
import io
import json
import math
import sys
from types import SimpleNamespace

import pytest

for dependency in ("datasets", "openai", "tiktoken"):
    pytest.importorskip(dependency)

from barber.eval import jev_trial


def test_jev_trial_uses_scores_and_reports_remote_failure(monkeypatch, tmp_path):
    titles = ["Sky"] + [f"Other {i}" for i in range(9)]
    context = "\n\n".join(
        f"[{i+1}] {title}: " + ("blue sky detail " if i == 0 else "unrelated detail ") * 20
        for i, title in enumerate(titles)
    )
    example = {"question": "What color is the sky?", "answer": "blue",
               "supporting_facts": {"title": ["Sky"]}}
    monkeypatch.setattr(jev_trial, "_key", lambda: "test-key")
    monkeypatch.setattr(jev_trial, "load_dataset", lambda *a, **kw: [example])
    monkeypatch.setattr(jev_trial, "build_distractor_pool", lambda ds: [])
    monkeypatch.setattr(jev_trial, "build_context", lambda ex, pool, size: (context, titles))
    monkeypatch.setattr(jev_trial.embedders, "sentence_transformers",
                        lambda model: jev_trial.embedders.lexical())
    output = tmp_path / "trial.jsonl"
    monkeypatch.setattr(sys, "argv", ["jev_trial", "--n", "1", "--out", str(output)])

    def fake_urlopen(request, timeout):
        sent = json.loads(request.data)
        assert sent["state"]["question"] == example["question"]
        assert len(sent["questions"]) >= 4
        sky = next(i for i, passage in enumerate(sent["state"]["passages"])
                   if "[1] Sky:" in passage["text"])
        return io.BytesIO(json.dumps({
            "model": "jev-test",
            "answers": {name: {"type": "noul", "noul": .9 if name == f"p{sky}" else .1}
                        for name in sent["questions"]},
            "usage": {"input_tokens": 123, "output_tokens": 10},
        }).encode())

    monkeypatch.setattr(jev_trial.urllib.request, "urlopen", fake_urlopen)
    jev_trial.main()
    row = json.loads(output.read_text().splitlines()[0])
    assert row["jev_model"] == "jev-test"
    assert row["jev_usage"]["input_tokens"] == 123
    assert row["jev"]["gold_kept"] == ["Sky"]

    def failing_urlopen(request, timeout):
        raise OSError("simulated Jev outage")

    monkeypatch.setattr(jev_trial.urllib.request, "urlopen", failing_urlopen)
    with pytest.raises(RuntimeError, match="Jev request failed"):
        jev_trial.main()

    monkeypatch.setattr(jev_trial, "_key", lambda: pytest.fail("Kev needs no Jev key"))
    monkeypatch.setattr(sys, "argv", ["jev_trial", "--backend", "kev", "--n", "1",
                                       "--out", str(output)])

    def fake_kev(request, timeout):
        assert request.full_url == "http://127.0.0.1:8009/v1/systemone"
        assert json.loads(request.data)["model"] == "kev-latest"
        return fake_urlopen(request, timeout)

    monkeypatch.setattr(jev_trial.urllib.request, "urlopen", fake_kev)
    jev_trial.main()
    assert json.loads(output.read_text().splitlines()[0])["kev"]["gold_kept"] == ["Sky"]


def test_laya_trial_uses_local_scores_without_jev_key(monkeypatch, tmp_path):
    titles = ["Sky"] + [f"Other {i}" for i in range(9)]
    context = "\n\n".join(
        f"[{i+1}] {title}: " + ("blue sky detail " if i == 0 else "unrelated detail ") * 20
        for i, title in enumerate(titles)
    )
    example = {"question": "What color is the sky?", "answer": "blue",
               "supporting_facts": {"title": ["Sky"]}}
    monkeypatch.setattr(jev_trial, "_key", lambda: pytest.fail("Laya needs no Jev key"))
    monkeypatch.setattr(jev_trial, "load_dataset", lambda *a, **kw: [example])
    monkeypatch.setattr(jev_trial, "build_distractor_pool", lambda ds: [])
    monkeypatch.setattr(jev_trial, "build_context", lambda ex, pool, size: (context, titles))
    monkeypatch.setattr(jev_trial.embedders, "sentence_transformers",
                        lambda model: jev_trial.embedders.lexical())

    class FakeRouter:
        def __init__(self):
            pass

        def predict_batch(self, requests, batch_size):
            assert batch_size == 8
            return [{"answers": {"relevant": {"noul": .9 if "[1] Sky:" in r["state"] else .1}},
                     "usage": {"input_tokens": 10}} for r in requests]

    monkeypatch.setitem(sys.modules, "laya", SimpleNamespace(Router=FakeRouter))
    output = tmp_path / "laya.jsonl"
    monkeypatch.setattr(sys, "argv", ["jev_trial", "--backend", "laya", "--n", "1",
                                       "--out", str(output)])
    jev_trial.main()
    row = json.loads(output.read_text().splitlines()[0])
    assert row["laya"]["gold_kept"] == ["Sky"]
    assert row["laya_usage"]["input_tokens"] >= 10

    class FakeDecider:
        def __init__(self, *args, **kwargs):
            pass

        def system_one(self, state, questions, independent):
            assert independent is False
            assert len(state["passages"]) == len(questions)
            return {"model": "decider-test", "usage": {"input_tokens": 50, "output_tokens": 0},
                    "answers": {f"p{i}": {"noul": .9 if "[1] Sky:" in passage["text"] else .1}
                                for i, passage in enumerate(state["passages"])}}

    monkeypatch.setitem(sys.modules, "decider", SimpleNamespace(infer=SimpleNamespace(Decider=FakeDecider)))
    monkeypatch.setitem(sys.modules, "decider.infer", SimpleNamespace(Decider=FakeDecider))
    monkeypatch.setattr(sys, "argv", ["jev_trial", "--backend", "decider", "--n", "1",
                                       "--out", str(output)])
    jev_trial.main()
    row = json.loads(output.read_text().splitlines()[0])
    assert row["decider"]["gold_kept"] == ["Sky"]
    assert row["decider_usage"]["input_tokens"] == 50

    scorer = jev_trial.embedders.decider(gguf_options={"n_ctx": 4096})
    vectors = scorer(["[1] Sky: blue", "[2] Other: red", example["question"]])
    assert math.isclose(sum(a * b for a, b in zip(vectors[0], vectors[-1])), .9)
    assert math.isclose(sum(a * b for a, b in zip(vectors[1], vectors[-1])), .1)
    scorer.close()
    with pytest.raises(RuntimeError, match="closed"):
        scorer(["passage", example["question"]])
