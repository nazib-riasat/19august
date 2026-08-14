"""Phase 2.5 — the timed annotation CLI, and Cohen's kappa for the re-pass.

Timing is **per item, wall-clock, from display to answer** (gap G2) — never
session-total divided by count, which hides that D2 items arrive in bursts
around one re-read. Labels land in Phase-6-loadable JSONL with provenance back
to turn ids and offsets (gap G6).

Annotate:     python scripts/phase2_5/annotate.py d1 --annotator you
Re-annotate:  python scripts/phase2_5/annotate.py d1 --annotator you --pass-2 --subset 20
Kappa:        python scripts/phase2_5/annotate.py kappa d1 --annotator you

`--pass-2` re-presents the first `--subset` items labelled in pass 1, shuffled,
and writes to a separate file; run it **at least two calendar days** after pass
1 (gap G4) — the gap is checked from the recorded timestamps and the kappa
report says whether it was honoured. Self-agreement is *weaker* than
inter-annotator agreement and every report labels it so.

D1 keys:  L<n> = LINK_EXISTING to candidate n · C = CREATE_NEW_ENTITY ·
          N = NON_ENTITY · D = DEFER      (?<note> flags an unclear item)
D2 keys:  I = INDEPENDENT · U = DUPLICATE · C = CONFLICT · S = SUPERSEDES
"""
from __future__ import annotations

import argparse
import datetime as dt
import random
import time

from common import DATA, LABELS, append_jsonl, read_jsonl

D2_KEYS = {"I": "INDEPENDENT", "U": "DUPLICATE", "C": "CONFLICT", "S": "SUPERSEDES"}


def _labels_path(
    decoder: str, annotator: str, pass2: bool, items: str = "", second: str = ""
) -> str:
    """Label file for one (decoder, annotator, pass) — and one *item set*.

    ``items`` names the batch when it is not the spike's default.  Without it a
    run over a different item set would write into the same file as the spike's,
    and two batches' labels would be indistinguishable afterwards — the kind of
    silent mixing that makes an annotation record unusable months later.
    """
    who = second or annotator
    # A *second annotator* writes under their own name, never as the first
    # annotator's "pass 2".  Provenance is the point: a file called
    # `d1_labels_Sabbir_pass2.jsonl` that a different person produced is a record
    # nobody can read correctly six weeks later, and the agreement figure
    # computed from it would be labelled self-agreement when it is not.
    suffix = "_pass2" if (pass2 and not second) else ""
    tag = f"_{items}" if items else ""
    return LABELS / f"{decoder}_labels_{who}{tag}{suffix}.jsonl"


def _show_d1(item: dict) -> None:
    print("\n" + "=" * 72)
    print(f"[{item['item_id']}]  turn {item['turn_id']}")
    print(f"TURN: {item['turn_text']}")
    print(f"\nMENTION: «{item['mention']}»  (chars {item['start']}–{item['end']})")
    if item["candidates"]:
        print("CANDIDATES:")
        for i, c in enumerate(item["candidates"]):
            print(f"  L{i}: {c['name']}   ({c['entity_id']})")
    else:
        print("CANDIDATES: (none yet in this conversation)")
    print("ACTIONS: L<n> link · C create · N non-entity · D defer")


def _show_d2(item: dict) -> None:
    print("\n" + "=" * 72)
    print(f"[{item['item_id']}]  {item['question_type']}"
          f"{'  (cross-session)' if item['cross_session'] else ''}")
    a, b = item["claim_a"], item["claim_b"]
    print(f"A ({a['session_date']}): {a['text']}")
    print(f"B ({b['session_date']}): {b['text']}")
    print("LABELS: I independent · U duplicate · C conflict · S supersedes (B supersedes A)")


def _read_answer(decoder: str, item: dict) -> tuple[str, str | None, float]:
    start = time.perf_counter()
    while True:
        raw = input("> ").strip()
        note = None
        if "?" in raw:
            raw, _, note = raw.partition("?")
            raw = raw.strip()
            note = note.strip() or None
        key = raw.upper()
        if decoder == "d1":
            if key.startswith("L") and key[1:].isdigit():
                n = int(key[1:])
                if n < len(item["candidates"]):
                    return (
                        f"LINK_EXISTING({item['candidates'][n]['entity_id']})",
                        note,
                        time.perf_counter() - start,
                    )
            elif key in ("C", "N", "D"):
                label = {"C": "CREATE_NEW_ENTITY", "N": "NON_ENTITY", "D": "DEFER"}[key]
                return label, note, time.perf_counter() - start
        else:
            if key in D2_KEYS:
                return D2_KEYS[key], note, time.perf_counter() - start
        print("  unrecognised — see the key legend above")


def annotate(
    decoder: str,
    annotator: str,
    pass2: bool,
    subset: int,
    items_tag: str = "",
    second: str = "",
) -> None:
    name = f"{decoder}_items_{items_tag}.jsonl" if items_tag else f"{decoder}_items.jsonl"
    items = read_jsonl(DATA / name)
    out_path = _labels_path(decoder, annotator, pass2, items_tag, second)
    done = {r["item_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()

    if pass2:
        first_pass = read_jsonl(_labels_path(decoder, annotator, False, items_tag))
        chosen_ids = [r["item_id"] for r in first_pass[:subset]]
        items = [i for i in items if i["item_id"] in chosen_ids]
        random.Random(20260817).shuffle(items)

    todo = [i for i in items if i["item_id"] not in done]
    print(f"{decoder.upper()}: {len(todo)} items to annotate "
          f"({len(done)} already done{' — resuming' if done else ''})")
    for item in todo:
        (_show_d1 if decoder == "d1" else _show_d2)(item)
        label, note, seconds = _read_answer(decoder, item)
        append_jsonl(
            out_path,
            {
                "item_id": item["item_id"],
                "label": label,
                "note": note,
                "seconds": round(seconds, 1),
                # The row's provenance is the person who PRODUCED it.  With
                # --second-annotator the file is already theirs (_labels_path);
                # stamping the first annotator's name inside it would invert
                # exactly the provenance the separate file exists to keep.
                "annotator": second or annotator,
                "machine_assisted": (second or annotator).startswith("claude"),
                # A second annotator's labels are their OWN pass 1, whatever
                # subset they were shown: `--pass-2` selects the κ items, but
                # "pass 2" as a row field means "the same person, again", and
                # the gold loader excludes pass 2 on that meaning.  Stamping a
                # different person's rows pass 2 would silently drop them from
                # gold and let the first annotator's label win every measured
                # disagreement — the exact silent overwrite the loader refuses.
                "pass": 2 if (pass2 and not second) else 1,
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            },
        )
    print(f"\nwrote {out_path}")


def _coarse(label: str) -> str:
    """LINK_EXISTING(id) collapses to its action for the 4-way kappa; the id
    match is reported separately, per v1.2 §6.4's three-number D1 rule."""
    return "LINK_EXISTING" if label.startswith("LINK_EXISTING") else label


def kappa(decoder: str, annotator: str, items_tag: str = "", second: str = "") -> None:
    """Agreement between two label sets.

    **Two different statistics wear the same arithmetic**, and which one this is
    depends entirely on who produced the second set:

    *Self-agreement* (one person, twice, >= 2 days apart) measures whether an
    annotator is consistent with themselves.  It is **weaker than IAA** and every
    report says so — the gap exists only to stop them remembering their own
    answers, which is why it is checked.

    *Inter-annotator agreement* (``--second-annotator``) is the real thing plan
    §7 item 7 asks for.  Independence comes from being different people, so the
    calendar gap is **not required and not checked** — reporting a gap warning
    there would be a warning about nothing.
    """
    p1 = {r["item_id"]: r for r in read_jsonl(_labels_path(decoder, annotator, False, items_tag))}
    p2 = {
        r["item_id"]: r
        for r in read_jsonl(_labels_path(decoder, annotator, True, items_tag, second))
    }
    shared = sorted(set(p1) & set(p2))
    if not shared:
        raise SystemExit("no overlapping items between pass 1 and pass 2")

    a = [_coarse(p1[i]["label"]) for i in shared]
    b = [_coarse(p2[i]["label"]) for i in shared]
    cats = sorted(set(a) | set(b))
    n = len(shared)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")

    exact = sum(p1[i]["label"] == p2[i]["label"] for i in shared) / n

    if second:
        print(f"{decoder.upper()} INTER-ANNOTATOR agreement, n = {n}")
        print(f"  annotators: {annotator} vs {second}")
    else:
        print(f"{decoder.upper()} SELF-agreement (weaker than IAA), n = {n}")
    print(f"  raw agreement (4-way action): {po:.3f}")
    print(f"  Cohen's kappa (4-way action): {k:.3f}")
    print(f"  exact agreement incl. entity id: {exact:.3f}")

    if second:
        # No gap requirement: independence comes from being two people.
        print("  calendar gap: not required (two annotators, independent by "
              "construction)")
    else:
        gap_days = (
            dt.datetime.fromisoformat(min(r["ts"] for r in p2.values()))
            - dt.datetime.fromisoformat(max(r["ts"] for r in p1.values()))
        ).total_seconds() / 86400
        print(f"  gap between passes: {gap_days:.1f} days "
              f"({'OK' if gap_days >= 2 else 'UNDER the required 2 days — G4 violated'})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["d1", "d2", "kappa"])
    ap.add_argument("decoder", nargs="?", choices=["d1", "d2"])
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--pass-2", action="store_true", dest="pass2")
    ap.add_argument("--subset", type=int, default=20)
    ap.add_argument(
        "--second-annotator",
        dest="second",
        default="",
        help="a DIFFERENT person doing the re-annotation pass. Their labels are "
        "written under their own name, the agreement is reported as "
        "inter-annotator rather than self-agreement, and the 2-day gap is not "
        "required — independence comes from being two people.",
    )
    ap.add_argument(
        "--items",
        default="",
        help="item-set tag, e.g. 'pilot' reads d1_items_pilot.jsonl and writes "
        "labels tagged with it; omit for the spike's original batch",
    )
    args = ap.parse_args()
    if args.mode == "kappa":
        if not args.decoder:
            raise SystemExit("kappa needs a decoder: kappa d1|d2")
        kappa(args.decoder, args.annotator, args.items, args.second)
    else:
        annotate(args.mode, args.annotator, args.pass2, args.subset, args.items, args.second)


if __name__ == "__main__":
    main()
