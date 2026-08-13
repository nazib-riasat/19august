"""Phase 2.5 deliverable 1 — a thin Stage-A slice over the sampled turns.

**A slice, not the component** (GRAFT_PHASE2_5_BUILD.md §0): extraction +
span-grounding + counters, nothing else. No RollingSummary (sessions here are
short windows), no NLI verifier, no support gate, no event log — those are
Phase 5.

**Model: Qwen2.5-3B-Instruct, bf16 — a declared deviation from the
architecture's 7B-4bit extractor, at the user's instruction (13 Aug 2026),
sized to the 8 GB RTX 5050.** Consequences recorded in PHASE2_5_DECISIONS.md:
gap G5's original question (does the 7B fit?) stays unmeasured; what is
measured instead is the 3B bf16 fit, which doubles as a rehearsal of the
Phase-10 reader load (same model class and precision).

Span discipline: the model is asked for **exact quotes**, never for offsets —
LLM character offsets are unreliable — and offsets are recovered by exact
substring search, then a difflib fuzzy window (the SpanGrounder pattern of
architecture Phase 5). Failures are dropped and counted, per the plan.

Usage:  python scripts/phase2_5/extract.py [--limit N] [--device cuda|cpu]
Reads:  data/phase2_5/sample.json
Writes: data/phase2_5/extraction.jsonl, data/phase2_5/extraction_stats.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import time

from common import DATA, short_hash, write_jsonl

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
CONTEXT_TURNS = 10  # architecture Phase 5: previous m = 10 turns
MAX_NEW_TOKENS = 600

SYSTEM = """You extract structured memory from one conversation turn.
Given the conversation context and the CURRENT TURN, output ONLY a JSON object:
{
 "mentions": [{"text": "<exact substring of the current turn naming an entity>"}],
 "assertions": [{"kind": "claim|value|event|time",
                 "text_norm": "<the fact, one sentence, self-contained: resolve pronouns and relative dates from context>",
                 "quote": "<exact contiguous substring of the current turn that supports it>"}]
}
Rules:
- "quote" and mention "text" MUST be copied verbatim from the CURRENT TURN only.
- Extract facts about the user and their world (preferences, events, values, dates).
- kind: "value" for numbers/quantities, "time" for dates/durations, "event" for things that happened, "claim" otherwise.
- No commentary. JSON only. Empty lists are fine for turns with no extractable content."""


def ground(quote: str, text: str) -> tuple[int, int] | None:
    """Exact match first, then a fuzzy window — the SpanGrounder pattern."""
    ix = text.find(quote)
    if ix >= 0:
        return ix, ix + len(quote)
    # Fuzzy: slide a window of the quote's length, accept the best match >= 0.85.
    n, m = len(text), len(quote)
    if m == 0 or m > n:
        return None
    best, best_ix = 0.0, -1
    step = max(1, m // 4)
    for start in range(0, n - m + 1, step):
        ratio = difflib.SequenceMatcher(
            None, text[start : start + m], quote, autojunk=False
        ).ratio()
        if ratio > best:
            best, best_ix = ratio, start
    if best >= 0.85:
        return best_ix, best_ix + m
    return None


def parse_json_block(raw: str) -> dict | None:
    """The fenced or bare JSON object in a model reply, or None."""
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    obj.setdefault("mentions", [])
    obj.setdefault("assertions", [])
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="turns to process")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sample = json.loads((DATA / "sample.json").read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    load_s = time.perf_counter() - t0
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows: list[dict] = []
    stats = {
        "model": MODEL_ID,
        "dtype": "bfloat16",
        "device": args.device,
        "load_seconds": round(load_s, 2),
        "turns": 0,
        "turns_with_parse_failure": 0,
        "mentions": 0,
        "assertions": 0,
        "grounding_failures": 0,
        "generated_tokens": 0,
        "generation_seconds": 0.0,
    }

    todo = []
    for session in sample["sessions"]:
        for i, turn in enumerate(session["turns"]):
            todo.append((session, i, turn))
    if args.limit:
        todo = todo[: args.limit]

    for session, i, turn in todo:
        context = session["turns"][max(0, i - CONTEXT_TURNS) : i]
        ctx_text = "\n".join(f"[{t['role']}] {t['text']}" for t in context) or "(none)"
        user = (
            f"CONTEXT (earlier turns, {session['session_date']}):\n{ctx_text}\n\n"
            f"CURRENT TURN [{turn['role']}]:\n{turn['text']}"
        )
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        g0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        stats["generation_seconds"] += time.perf_counter() - g0
        stats["generated_tokens"] += int(gen.shape[0])
        reply = tok.decode(gen, skip_special_tokens=True)

        obj = parse_json_block(reply)
        if obj is None:
            stats["turns_with_parse_failure"] += 1
            obj = {"mentions": [], "assertions": []}

        mentions, assertions = [], []
        for m in obj["mentions"]:
            text = (m or {}).get("text", "") if isinstance(m, dict) else str(m)
            if not text:
                continue
            span = ground(text, turn["text"])
            if span is None:
                stats["grounding_failures"] += 1
                continue
            mentions.append(
                {
                    "mention_id": f"m_{short_hash(turn['turn_id'] + text + str(span[0]))}",
                    "text": turn["text"][span[0] : span[1]],
                    "start": span[0],
                    "end": span[1],
                }
            )
        for a in obj["assertions"]:
            if not isinstance(a, dict):
                continue
            quote = a.get("quote", "")
            span = ground(quote, turn["text"]) if quote else None
            if span is None:
                stats["grounding_failures"] += 1
                continue
            kind = a.get("kind", "claim")
            if kind not in ("claim", "value", "event", "time"):
                kind = "claim"
            assertions.append(
                {
                    "assertion_id": f"a_{short_hash(turn['turn_id'] + a.get('text_norm', '') + str(span[0]))}",
                    "kind": kind,
                    "text_norm": a.get("text_norm", ""),
                    "quote": turn["text"][span[0] : span[1]],
                    "start": span[0],
                    "end": span[1],
                    "grounded": True,
                }
            )
        stats["turns"] += 1
        stats["mentions"] += len(mentions)
        stats["assertions"] += len(assertions)
        rows.append(
            {
                "turn_id": turn["turn_id"],
                "question_id": session["question_id"],
                "question_type": session["question_type"],
                "session_id": session["session_id"],
                "session_date": session["session_date"],
                "role": turn["role"],
                "text": turn["text"],
                "mentions": mentions,
                "assertions": assertions,
            }
        )
        print(
            f"[{stats['turns']}/{len(todo)}] {turn['turn_id']}: "
            f"{len(mentions)} mentions, {len(assertions)} assertions",
            flush=True,
        )

    if args.device == "cuda":
        stats["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    if stats["generation_seconds"]:
        stats["tokens_per_s"] = round(
            stats["generated_tokens"] / stats["generation_seconds"], 1
        )

    write_jsonl(DATA / "extraction.jsonl", rows)
    (DATA / "extraction_stats.json").write_text(
        json.dumps(stats, indent=1), encoding="utf-8", newline="\n"
    )
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
