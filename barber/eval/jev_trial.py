"""Paired Jev, Kev, Laya, or Decider vs BGE passage-selection trial on HotpotQA.

Run with a TypeSafe key in TYPESAFE_API_KEY (or the ignored repo .env):
    python -m barber.eval.jev_trial --n 25 --size small --out /tmp/jev-trial.jsonl
    python -m barber.eval.jev_trial --backend laya --n 25 --out /tmp/laya-trial.jsonl
    python -m barber.eval.jev_trial --backend kev --n 25 --out /tmp/kev-trial.jsonl
    python -m barber.eval.jev_trial --backend decider --n 25 --out /tmp/decider-trial.jsonl

The chosen backend supplies relevance scores; Barber's chunking, pins, budget, markers, and
no-negative-savings guard remain identical between arms. This checks supporting-passage
recall and prompt-token savings, not answer quality or net inference cost.
"""
import argparse
import json
import math
import os
import re
import time
import urllib.request
from pathlib import Path
from statistics import median

from datasets import load_dataset

from barber import embedders, make_transform
from barber.core import SelectionConfig
from barber.eval.harness import build_context, build_distractor_pool, ntok


def _key():
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "TYPESAFE_API_KEY":
                return value.strip().strip('"\'')
    raise SystemExit("Set TYPESAFE_API_KEY in the environment or Barber's ignored .env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["jev", "kev", "laya", "decider"], default="jev")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--offset", type=int, default=400)
    ap.add_argument("--size", choices=["small", "medium", "large"], default="small")
    ap.add_argument("--out", help="optional per-question JSONL output")
    args = ap.parse_args()
    key = _key() if args.backend == "jev" else "local"
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor",
                      split=f"validation[{args.offset}:{args.offset + args.n}]")
    pool = build_distractor_pool(load_dataset(
        "hotpotqa/hotpot_qa", "distractor", split="validation[2000:2600]"))
    bge = embedders.sentence_transformers("BAAI/bge-small-en-v1.5")
    totals = {name: {"selected": 0, "gold_kept": 0} for name in ("bge", args.backend)}
    select_latency = {"bge": [], args.backend: []}
    score_usage = {"input_tokens": 0, "output_tokens": 0}
    score_latency = []
    score_models = set()
    score_error = []

    if args.backend == "decider":
        from decider.infer import Decider
        model = Decider("Mapika/decider-4b-GGUF", gguf_file="decider-4b-v2.1-Q4_K_M.gguf",
                        gguf_options={"n_ctx": 4096})

        def score_embed(texts):
            chunks, query = texts[:-1], texts[-1]
            start = time.perf_counter()
            try:
                result = model.system_one(
                    {"question": query, "passages": [
                        {"id": i, "text": chunk} for i, chunk in enumerate(chunks)]},
                    {f"p{i}": {"type": "noul", "instructions":
                        f"Is passage {i} needed to answer the question correctly, directly or as a bridge fact?"}
                     for i in range(len(chunks))},
                    independent=False,
                )
                scores = [float(result["answers"][f"p{i}"]["noul"])
                          for i in range(len(chunks))]
                if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
                    raise ValueError("Decider returned a probability outside [0, 1]")
                for k in score_usage:
                    score_usage[k] += int(result["usage"][k])
                score_models.add(result["model"])
                score_latency.append(time.perf_counter() - start)
                return [[p, math.sqrt(1 - p * p)] for p in scores] + [[1.0, 0.0]]
            except Exception as exc:
                score_error.append(exc)
                raise
    elif args.backend == "laya":
        from laya import Router
        router = Router()

        def score_embed(texts):
            chunks, query = texts[:-1], texts[-1]
            question = {"relevant": {"type": "noul", "instructions":
                "Does this passage help answer the question, directly or as a bridge fact?"}}
            requests = [{"state": f"Question: {query}\nPassage: {chunk}",
                         "questions": question, "model": "english"} for chunk in chunks]
            start = time.perf_counter()
            try:
                results = router.predict_batch(requests, batch_size=8)
                scores = [float(result["answers"]["relevant"]["noul"]) for result in results]
                if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
                    raise ValueError("Laya returned a probability outside [0, 1]")
                score_usage["input_tokens"] += sum(int(r["usage"]["input_tokens"]) for r in results)
                score_models.add("english")
                score_latency.append(time.perf_counter() - start)
                return [[p, math.sqrt(1 - p * p)] for p in scores] + [[1.0, 0.0]]
            except Exception as exc:
                score_error.append(exc)
                raise
    else:

        def score_embed(texts):
            chunks, query = texts[:-1], texts[-1]
            payload = {
                "model": "kev-latest" if args.backend == "kev" else "jev-latest",
                "state": {"question": query, "passages": [
                    {"id": i, "text": text} for i, text in enumerate(chunks)]},
                "questions": {f"p{i}": {
                    "type": "noul",
                    "instructions": f"Is passage {i} needed to answer the question correctly, directly or as a bridge fact?",
                } for i in range(len(chunks))},
            }
            req = urllib.request.Request(
                ("http://127.0.0.1:8009/v1/systemone" if args.backend == "kev"
                 else "https://api.typesafe.ai/v1/systemone"),
                data=json.dumps(payload, ensure_ascii=False).encode(),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                method="POST",
            )
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    result = json.load(response)
                scores = [float(result["answers"][f"p{i}"]["noul"])
                          for i in range(len(chunks))]
                if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
                    raise ValueError("Jev returned a probability outside [0, 1]")
                for k in score_usage:
                    score_usage[k] += int(result["usage"][k])
                score_models.add(result["model"])
                score_latency.append(time.perf_counter() - start)
                return [[p, math.sqrt(1 - p * p)] for p in scores] + [[1.0, 0.0]]
            except Exception as exc:
                score_error.append(exc)  # Barber fails open; the evaluation must not.
                raise

    cfg = SelectionConfig(min_message_chars=200, min_chunks=4)
    _, bge_select = make_transform(embedder=bge, keep=0.6, cfg=cfg)
    _, score_select = make_transform(embedder=score_embed, keep=0.6, cfg=cfg)
    full_tokens = gold_total = 0
    out = open(args.out, "w", encoding="utf-8") if args.out else None
    try:
        for i, ex in enumerate(ds, args.offset):
            context, _ = build_context(ex, pool, args.size)
            question = ex["question"]
            messages = [{"role": "user", "content": "CONTEXT:\n\n" + context},
                        {"role": "user", "content": question}]
            full_tokens += ntok(context)
            titles = set(ex["supporting_facts"]["title"])
            gold_total += len(titles)
            row = {"index": i, "question": question, "gold": ex["answer"],
                   "gold_titles": sorted(titles), "tokens_full": ntok(context)}
            usage_before = dict(score_usage)
            if out:
                row["context_full"] = context
            for name, select in (("bge", bge_select), (args.backend, score_select)):
                start = time.perf_counter()
                selected, _ = select(messages)
                if score_error:
                    raise RuntimeError(f"{args.backend.capitalize()} request failed") from score_error[-1]
                elapsed = (time.perf_counter() - start) * 1000
                select_latency[name].append(elapsed)
                text = selected[0]["content"].removeprefix("CONTEXT:\n\n")
                kept = sorted(t for t in titles if re.search(
                    r"^\[\d+\] " + re.escape(t) + ":", text, re.M))
                count = ntok(text)
                totals[name]["selected"] += count
                totals[name]["gold_kept"] += len(kept)
                row[name] = {"tokens_selected": count, "gold_kept": kept}
                if out:
                    row[name].update({"context": text, "latency_ms": round(elapsed, 1)})
            if out:
                row[f"{args.backend}_usage"] = {k: score_usage[k] - usage_before[k] for k in score_usage}
                row[f"{args.backend}_model"] = sorted(score_models)[-1] if score_models else None
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
    finally:
        if out:
            out.close()

    if not full_tokens:
        raise SystemExit("No examples in the requested slice")
    for name, result in totals.items():
        print(f"{name}: saved={100 * (1 - result['selected'] / full_tokens):.2f}% "
              f"support={result['gold_kept']}/{gold_total} "
              f"median_selection_ms={median(select_latency[name]):.0f}")
    if score_latency:
        cost = (f"list_price_usd=${score_usage['input_tokens'] * .042 / 1_000_000:.6f} "
                if args.backend == "jev" else "")
        print(f"{args.backend}: models={','.join(sorted(score_models))} "
              f"input_tokens={score_usage['input_tokens']} {cost}"
              f"median_latency_ms={median(score_latency) * 1000:.0f}")
    else:
        print(f"{args.backend}: no eligible contexts scored")
    if args.backend == "decider":
        del model


if __name__ == "__main__":
    main()
