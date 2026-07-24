"""
barber-eval — context-selection quality harness. LLM judge, HotpotQA as data.

Measures whether TRIMMED context produces answers as good as FULL context, using
a judge model to grade both against the gold answer. Outputs the regression rate
(the number that governs whether it's safe to ship) plus tokens saved and gold
recall.

SETUP:
    pip install barber-llm[eval]
    export JUDGE_API_KEY=sk-...        # or MINIMAX_API_KEY (honored as fallback)
    barber-eval --n 100 --keep 0.6

Everything is overridable via env, no code edits:
    JUDGE_API_KEY / JUDGE_BASE_URL / JUDGE_MODEL   -> the judge
    GEN_API_KEY / GEN_BASE_URL / GEN_MODEL         -> the generator (defaults to judge creds)
e.g. run the generator on Groq/DeepSeek/OpenRouter/your Nadir proxy while
MiniMax judges, or point BOTH somewhere else. Defaults reproduce the published
benchmark: MiniMax-M3 as both generator and judge.

Ported from Nadir's validated selection_quality_test.py; the judge prompts,
context construction, and metrics are unchanged.
"""
import os, re, json, random, argparse, hashlib
random.seed(13)

try:
    import tiktoken
    from openai import OpenAI
    from datasets import load_dataset
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        f"barber-eval needs the eval extras ({e.name} missing). "
        "Install with: pip install barber-llm[eval]"
    )

from barber.core import make_selection_transform, SelectionConfig
from barber import embedders

ENC = tiktoken.get_encoding("o200k_base")
def ntok(s): return len(ENC.encode(s))

# --- Clients (OpenAI compatible), created in _init_clients() at run time so
#     `barber-eval --help` works without a key.
minimax = None
gen_client = None
JUDGE_MODEL = None
GEN_MODEL = None

def _init_clients():
    global minimax, gen_client, JUDGE_MODEL, GEN_MODEL
    judge_base = os.environ.get("JUDGE_BASE_URL",
                 os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1"))
    judge_key = os.environ.get("JUDGE_API_KEY", os.environ.get("MINIMAX_API_KEY"))
    if not judge_key:
        raise SystemExit("Set JUDGE_API_KEY (or MINIMAX_API_KEY) to run barber-eval. "
                         "Any OpenAI compatible judge endpoint works via JUDGE_BASE_URL/JUDGE_MODEL.")
    JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "MiniMax-M3")
    minimax = OpenAI(base_url=judge_base, api_key=judge_key)

    gen_base = os.environ.get("GEN_BASE_URL", judge_base)
    gen_key = os.environ.get("GEN_API_KEY", judge_key)
    GEN_MODEL = os.environ.get("GEN_MODEL", "MiniMax-M3")
    gen_client = OpenAI(base_url=gen_base, api_key=gen_key)   # >>> PROD: your routed endpoint

JUDGE_SYSTEM = """You are a strict, impartial grader. Your job is to measure whether reducing an AI assistant's context caused its answer to get worse.

You are given:
- QUESTION: the user's question.
- REFERENCE: the known-correct short answer.
- ANSWER A and ANSWER B: two assistant answers to the QUESTION. One was written with the full context and one with a reduced context — you are NOT told which is which. Judge only the text in front of you.

Grade EACH answer independently against the REFERENCE:
- "correct":   conveys the reference answer. Allow paraphrase, different wording, extra detail that is also correct.
- "partial":   gets part of it but omits a key element the question explicitly asks for.
- "incorrect": wrong, contradicts the reference, refuses, or says it lacks the information needed to answer.

Ignore differences in style, length, formatting, verbosity, or politeness. Grade ONLY factual correctness and completeness relative to the REFERENCE.

CRITICAL RULES:
- The REFERENCE is the complete ground truth. An answer that conveys the REFERENCE is "correct" even if it omits other details. NEVER penalize an answer for omitting information that is not in the REFERENCE.
- NEVER use your own world knowledge to dispute or reinterpret the REFERENCE. If the answer matches the REFERENCE, it is correct.

Output ONLY strict JSON, nothing else:
{"a_grade":"correct|partial|incorrect","b_grade":"correct|partial|incorrect","worse_answer":"A|B|neither","materially_worse":true|false,"reason":"<=25 words on what the worse answer got wrong or omitted; 'none' if equivalent"}

Set "materially_worse":true ONLY when one answer is clearly less correct or complete than the other on something the QUESTION needs — never for stylistic differences."""

GEN_SYSTEM = "Answer the QUESTION using ONLY the CONTEXT. Be concise. If the context does not contain the answer, say you don't have enough information."

JUDGE_SINGLE_SYSTEM = """You are a strict, impartial grader. Grade ANSWER against the known-correct REFERENCE:
- "correct":   conveys the reference answer (paraphrase and extra correct detail allowed).
- "partial":   gets part of it but omits a key element the question explicitly asks for.
- "incorrect": wrong, contradicts the reference, refuses, or says it lacks information.
Ignore style, length, formatting. The REFERENCE is complete ground truth: never penalize omission of details not in the REFERENCE, and never dispute the REFERENCE from your own knowledge. Output ONLY strict JSON: {"grade":"correct|partial|incorrect","reason":"<=15 words"}"""

def judge_single(question, gold, answer):
    user = f"QUESTION:\n{question}\n\nREFERENCE:\n{gold}\n\nANSWER:\n{answer}"
    v = parse_json(chat(minimax, JUDGE_MODEL, JUDGE_SINGLE_SYSTEM, user, max_tokens=4096))
    return v.get("grade"), v.get("reason", "")

def chat(client, model, system, user, temp=0.0, max_tokens=512, retries=4):
    import time
    from openai import RateLimitError, APIStatusError
    for attempt in range(retries + 1):
        try:
            r = client.chat.completions.create(model=model, temperature=temp, max_tokens=max_tokens,
                messages=[{"role":"system","content":system},{"role":"user","content":user}])
            ch = r.choices[0]
            if getattr(ch, "finish_reason", None) == "length":
                chat.truncated = getattr(chat, "truncated", 0) + 1
            return (ch.message.content or "").strip()
        except RateLimitError as e:
            msg = str(e)
            if "usage limit reached" in msg or "2056" in msg:
                raise SystemExit("\n*** Judge plan quota EXHAUSTED (not a per-minute limit). "
                                 "Top up / wait for reset, or point GEN_BASE_URL+GEN_API_KEY+GEN_MODEL "
                                 "at another OpenAI compatible provider. ***")
            if attempt == retries: raise
            wait = 2 ** attempt * 5
            print(f"  [429 rate-limit, retrying in {wait}s...]"); time.sleep(wait)
        except APIStatusError as e:
            if e.status_code >= 500 and attempt < retries:
                time.sleep(2 ** attempt * 3); continue
            raise

def parse_json(s):
    # Reasoning judges (M3) may emit JSON after a long thinking preamble, or an
    # early {...} may appear inside the reasoning. Our judge JSON is flat, so
    # scan ALL flat objects and return the LAST one that parses.
    out = {}
    for m in re.findall(r"\{[^{}]*\}", s or "", re.DOTALL):
        try: out = json.loads(m)
        except Exception: pass
    return out

# --- build a HotpotQA request: paragraphs -> one context block -----------------
# Context sizes: native HotpotQA is ~1.3K tokens (10 paragraphs). Real RAG traffic
# is 5-50K. We build bigger haystacks by padding with REAL Wikipedia paragraphs
# drawn from OTHER examples (true distractors), then shuffling — the standard
# lost-in-the-middle protocol. Gold answer & grading are unchanged; only the
# haystack grows.
SIZE_TARGET_TOKENS = {"small": 0, "medium": 4000, "large": 12000}  # 0 = native

def build_distractor_pool(ds, limit=3000):
    """Paragraphs from across the dataset, keyed by title, used as padding."""
    pool = []
    for ex in ds:
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            pool.append((title, " ".join(sents)))
            if len(pool) >= limit:
                return pool
    return pool

def build_context(ex, pool, size):
    paras = ex["context"]
    blocks = [(t, " ".join(s)) for t, s in zip(paras["title"], paras["sentences"])]
    own_titles = set(paras["title"])
    target = SIZE_TARGET_TOKENS[size]
    if target:
        # deterministic per-example RNG so runs are reproducible
        rng = random.Random(int(hashlib.md5(ex["question"].encode()).hexdigest()[:8], 16))
        cur = sum(ntok(t + ": " + x) for t, x in blocks)
        # walk the pool from a per-example offset, skip this example's own titles
        start = rng.randrange(len(pool))
        i = 0
        while cur < target and i < len(pool):
            title, text = pool[(start + i) % len(pool)]; i += 1
            if title in own_titles:
                continue
            blocks.append((title, text))
            cur += ntok(title + ": " + text)
        rng.shuffle(blocks)   # bury the gold paragraphs at random positions
    ctx = "\n\n".join(f"[{i+1}] {t}: {x}" for i, (t, x) in enumerate(blocks))
    return ctx, [t for t, _ in blocks]

def answer_with(context, question):
    # reasoning models (M3) need budget for thinking + final answer
    return chat(gen_client, GEN_MODEL, GEN_SYSTEM,
                f"CONTEXT:\n\n{context}\n\nQUESTION: {question}", max_tokens=3000)

def judge(question, gold, ans_ctrl, ans_treat):
    # randomize A/B so the judge can't infer control vs treatment
    swap = random.random() < 0.5
    a, b = (ans_treat, ans_ctrl) if swap else (ans_ctrl, ans_treat)
    user = f"QUESTION:\n{question}\n\nREFERENCE:\n{gold}\n\nANSWER A:\n{a}\n\nANSWER B:\n{b}"
    v = parse_json(chat(minimax, JUDGE_MODEL, JUDGE_SYSTEM, user, max_tokens=4096))
    if not v: return None
    # un-blind: map A/B grades back to control/treatment
    ctrl_grade = v.get("b_grade" if swap else "a_grade")
    treat_grade = v.get("a_grade" if swap else "b_grade")
    worse = v.get("worse_answer")
    treat_worse = v.get("materially_worse") and (
        (worse == "A" and swap) or (worse == "B" and not swap))
    return ctrl_grade, treat_grade, bool(treat_worse), v.get("reason","")

def main():
    ap = argparse.ArgumentParser(prog="barber-eval")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--keep", type=float, default=0.6, help="keep ratio for selection (benchmark default 0.6)")
    ap.add_argument("--size", choices=["small", "medium", "large"], default="small",
                    help="context size: small=native ~1.3K tok, medium ~4K, large ~12K")
    ap.add_argument("--offset", type=int, default=0, help="dataset slice offset (use disjoint slices per run)")
    ap.add_argument("--marker", choices=["neutral", "assertive"], default="assertive",
                    help="drop-marker wording: assertive = 'omitted as not relevant to this question' "
                         "(barber's shipped default; won the A/B); neutral = 'lower-relevance passages omitted'")
    ap.add_argument("--embedder", choices=["lexical", "st", "vllm"], default="st",
                    help="chunk scorer: st = local sentence-transformers, vllm = remote OpenAI compatible "
                         "/v1/embeddings (env: VLLM_EMBED_BASE_URL, VLLM_EMBED_MODEL), lexical = BM25-lite")
    ap.add_argument("--baseline", action="store_true",
                    help="BASELINE mode: full context only, no selection. Establishes "
                         "generator accuracy + judge sanity + token counts per size "
                         "(1 gen call + 1 judge call per example — half the A/B cost).")
    args = ap.parse_args()
    _init_clients()

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor",
                      split=f"validation[{args.offset}:{args.offset + args.n}]")
    pool = build_distractor_pool(
        load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation[2000:2600]"))  # padding pool, disjoint from eval slice
    if args.embedder == "vllm":
        embed = embedders.endpoint(
            base_url=os.environ.get("VLLM_EMBED_BASE_URL", "http://localhost:8000/v1"),
            model=os.environ.get("VLLM_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
        print(f"[embedder: endpoint {os.environ.get('VLLM_EMBED_MODEL','BAAI/bge-small-en-v1.5')} @ {os.environ.get('VLLM_EMBED_BASE_URL','http://localhost:8000/v1')}]")
    elif args.embedder == "st":
        try:
            st_model = os.environ.get("ST_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
            embed = embedders.sentence_transformers(st_model)
            print(f"[embedder: sentence-transformers {st_model} — semantic]")
        except Exception as e:
            print(f"[sentence-transformers unavailable ({e}); falling back to lexical]")
            embed = embedders.lexical()
    else:
        embed = embedders.lexical()
        print("[embedder: lexical BM25-lite (weak on paraphrase) — install barber-llm[semantic] and use --embedder st]")
    marker = ("[… {n} passage(s) omitted as not relevant to this question — the remaining context is sufficient …]"
              if args.marker == "assertive"
              else "[… {n} lower-relevance passage(s) omitted …]")
    cfg = SelectionConfig(min_message_chars=200, min_chunks=4,
                          max_keep_ratio=args.keep, min_keep_ratio=args.keep,
                          drop_marker=marker)

    tok_full = tok_sel = 0
    ctrl_correct = treat_correct = 0
    regressions = 0            # control right -> treatment wrong  (THE number)
    mat_worse = 0              # blind-judge cross-check
    gold_kept = gold_tot = 0
    judged = 0

    if args.baseline:
        grades = {"correct": 0, "partial": 0, "incorrect": 0}
        for i, ex in enumerate(ds):
            ctx_full, _ = build_context(ex, pool, args.size)
            q, gold = ex["question"], ex["answer"]
            tok_full += ntok(ctx_full)
            ans = answer_with(ctx_full, q)
            g, reason = judge_single(q, gold, ans)
            if g not in grades:
                dropped = getattr(main, "_dropped", 0) + 1; main._dropped = dropped
                continue
            judged += 1
            grades[g] += 1
            if g != "correct":
                print(f"  [{g.upper():9}] q={q[:70]!r} gold={gold!r} why={reason}")
        print("\n" + "=" * 60)
        print(f"BASELINE  size={args.size}  n_judged={judged}  judge_parse_dropped={getattr(main,'_dropped',0)}  gen_truncated={getattr(chat,'truncated',0)}  avg_ctx_tokens={tok_full//max(1,judged)}")
        for g, c in grades.items():
            print(f"  {g:10} {c:4}  ({c/max(1,judged)*100:5.1f}%)")
        print("=" * 60)
        print("This is the ceiling selection is measured against. If 'correct' is")
        print("<60% on small, inspect the printed failures before running the A/B —")
        print("a generator/judge bug looks identical to a selection regression later.")
        return

    for ex in ds:
        ctx_full, titles = build_context(ex, pool, args.size)
        q, gold = ex["question"], ex["answer"]
        gold_titles = set(ex["supporting_facts"]["title"])

        # SELECTED context
        msgs = [{"role":"user","content":f"CONTEXT:\n\n{ctx_full}"}, {"role":"user","content":q}]
        _, fn = make_selection_transform(embed_fn=embed, cfg=cfg)
        out, _ = fn(msgs)
        ctx_sel = out[0]["content"].replace("CONTEXT:\n\n","")

        # gold recall: did selection keep the supporting paragraphs?
        for t in gold_titles:
            gold_tot += 1; gold_kept += int(t in ctx_sel)

        tok_full += ntok(ctx_full); tok_sel += ntok(ctx_sel)

        ans_ctrl = answer_with(ctx_full, q)
        ans_treat = answer_with(ctx_sel, q)
        res = judge(q, gold, ans_ctrl, ans_treat)
        if not res:
            main._dropped = getattr(main, "_dropped", 0) + 1
            continue
        cg, tg, tw, reason = res
        judged += 1
        cok = cg == "correct"; tok_ok = tg == "correct"
        ctrl_correct += cok; treat_correct += tok_ok
        if cok and not tok_ok:
            regressions += 1
            print(f"  REGRESSION  q={q[:60]!r}  gold={gold!r}  why={reason}")
        if tok_ok and not cok:
            main._improved = getattr(main, "_improved", 0) + 1
        mat_worse += int(tw)

    print("\n" + "="*60)
    print(f"size={args.size}  keep_ratio={args.keep}  n_judged={judged}  judge_parse_dropped={getattr(main,'_dropped',0)}  gen_truncated={getattr(chat,'truncated',0)}  avg_ctx_tokens={tok_full//max(1,judged)}")
    print(f"tokens saved:            {(1-tok_sel/tok_full)*100:5.1f}%")
    print(f"gold-paragraph recall:   {gold_kept/gold_tot*100:5.1f}%")
    print(f"control accuracy:        {ctrl_correct/judged*100:5.1f}%")
    print(f"treatment accuracy:      {treat_correct/judged*100:5.1f}%")
    print(f"REGRESSION RATE:         {regressions/judged*100:5.1f}%   <- gate on this (<1-2%)")
    print(f"IMPROVEMENT RATE:        {getattr(main,'_improved',0)/judged*100:5.1f}%   <- selection fixed answers control got wrong")
    print(f"materially-worse (blind):{mat_worse/judged*100:5.1f}%   <- cross-check")
    print("="*60)
    print("Sweep --keep 0.6/0.4/0.3/0.2 to find the most aggressive setting where")
    print("REGRESSION RATE stays under your bar. That's your safe operating point.")

if __name__ == "__main__":
    main()
