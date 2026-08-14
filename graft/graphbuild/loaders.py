"""P6.7 — D3/D4 external supervision, and what "interface" is allowed to mean (G6).

`GATE0_CONTRACT.md` item 1 sources D3 from **DialogRE** (ACL 2020, scored with
its progressive `F1_c`) and **Re-DocRED** (EMNLP 2022 — *not* DocRED, whose false
negatives punish recall-oriented models by ~13 F1), and D4 from **TORQUE**
(EMNLP 2020, MATRES' annotation scheme, ACL 2018).  None of their label sets maps
1:1 onto GRAFT's eleven relations, which is why every parent document says
"-style" and "supervision interface".

**What "interface" means here, pinned so it cannot drift** (decision 7):

* D3 and D4 **train and evaluate on the datasets' native label sets**, through
  the *same frozen decoder interface* the GRAFT decoders use.  What that
  establishes is the Gate-1 question actually worth asking — does the shared
  encoder + typed-decoder machinery carry real relational and temporal signal —
  rather than a comparison against a schema nobody annotated.
* Their **GRAFT-schema application** (writing `contradicts` / `valid_during`
  edges on conversation) uses the declared mapping below, and **every mapping
  loss is tabulated**: which native classes are dropped, which are merged.  The
  contract's own red line is that an adaptation is *never presented as native
  supervision*, and a mapping that silently discarded 30 of 36 relations while
  reporting "trained on DialogRE" would cross it.

**The loader is the only dataset-specific code.**  Above it the trainer sees
`(item, label)` streams and cannot tell DialogRE from TORQUE — the fix-F6 pattern
applied one stage up.

**Nothing here imports torch.**  Loading and mapping are pure Python over JSON,
which keeps the mapping losses testable on a bare interpreter — and those losses
are the number the contract cares about.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from graft.graphbuild.pins import DATASETS

__all__ = [
    "RAW_ROOT",
    "DIALOGRE_TO_GRAFT",
    "REDOCRED_TO_GRAFT",
    "TORQUE_TO_GRAFT",
    "load_split",
    "dialogre_items",
    "redocred_items",
    "torque_items",
    "mapping_report",
    "loader_artefact",
]

RAW_ROOT = Path("data") / "phase6" / "raw"


# --------------------------------------------------------------------------
# the declared mappings, and their honest scope
# --------------------------------------------------------------------------

#: DialogRE's 36 relation types → GRAFT's schema.
#:
#: **Almost all of them map to nothing, and that is the finding, not a defect.**
#: GRAFT's eleven relations are about *provenance, identity, time and conflict*;
#: DialogRE's are about *interpersonal and biographical facts* (``per:spouse``,
#: ``per:employee_or_member_of``).  The only honest targets are:
#:
#: * relations asserting an entity's attribute → ``has_value`` (an entity holds a
#:   value),
#: * identity/alias relations → ``same_as``,
#: * everything else → **dropped**, counted, and reported.
#:
#: This is what "supervision interface" means in practice: D3 *trains* on
#: DialogRE's native 36-way task and the GRAFT mapping is a separate, lossy
#: application whose loss is published.
DIALOGRE_TO_GRAFT: Mapping[str, str] = {
    "per:alternate_names": "same_as",
    "per:alias": "same_as",
    "gpe:alternate_names": "same_as",
    "org:alternate_names": "same_as",
    "per:age": "has_value",
    "per:date_of_birth": "has_value",
    "per:place_of_birth": "has_value",
    "per:place_of_residence": "has_value",
    "per:origin": "has_value",
    "per:title": "has_value",
}

#: Re-DocRED uses Wikidata property ids.  The same reasoning and the same
#: honesty: only identity-ish and attribute-ish properties have a GRAFT target.
#: ``P31`` (instance of) and ``P279`` (subclass of) are *not* ``same_as`` —
#: mapping them there would assert that a dog is identical to the concept "dog",
#: which is the kind of merge error the whole commit validator exists around.
REDOCRED_TO_GRAFT: Mapping[str, str] = {
    "P1449": "same_as",   # nickname
    "P742": "same_as",    # pseudonym
    "P1477": "same_as",   # birth name
    "P569": "has_value",  # date of birth
    "P570": "has_value",  # date of death
    "P19": "has_value",   # place of birth
    "P20": "has_value",   # place of death
    "P571": "has_value",  # inception
    "P576": "has_value",  # dissolved/abolished
}

#: TORQUE asks temporal-ordering questions over events.  Its answers are event
#: spans, not intervals, so the mapping is **structural rather than relational**:
#: an answered temporal question yields a ``valid_during`` obligation on the
#: events it names.  Recorded as one entry because there is one target, and
#: pretending to a richer mapping would be the misclaim the contract forbids.
TORQUE_TO_GRAFT: Mapping[str, str] = {"temporal_answer": "valid_during"}


# --------------------------------------------------------------------------
# loading, SHA-verified
# --------------------------------------------------------------------------


def load_split(dataset: str, split: str, root: Path | None = None, verify: bool = True) -> Any:
    """Read one pinned split, verifying its SHA-256.

    Same discipline as ``graft.ingest.corpus``: a run that reads a different file
    than the one pinned is a different experiment, and the loader says so rather
    than proceeding.  ``verify=False`` exists for fixtures and for nothing else.
    """
    spec = DATASETS[dataset]
    rel, expected = spec["files"][split]
    path = (root or RAW_ROOT) / rel
    if not path.is_file():
        raise FileNotFoundError(
            f"{dataset}/{split} not found at {path}. Fetch it from {spec['source']} "
            f"(licence: {spec['licence']}; raw files are gitignored and fetched per machine)."
        )
    blob = path.read_bytes()
    if verify:
        got = hashlib.sha256(blob).hexdigest()
        if got != expected:
            raise ValueError(
                f"{dataset}/{split} SHA mismatch: expected {expected}, got {got}. "
                "The pin describes a different file; re-fetch, or re-pin deliberately."
            )
    return json.loads(blob.decode("utf-8"))


# --------------------------------------------------------------------------
# native-label item streams
# --------------------------------------------------------------------------


def dialogre_items(data: Sequence[Any]) -> list[dict[str, Any]]:
    """``(dialogue, entity pair) -> native relation labels``.

    DialogRE is multi-label per pair (a pair can be both ``per:spouse`` and
    ``per:parents`` in different readings), and the stream keeps it that way:
    collapsing to a single label would change the task the decoder is being
    tested on, which is exactly what "native label set" forbids.
    """
    items: list[dict[str, Any]] = []
    for ix, entry in enumerate(data):
        turns, annotations = entry[0], entry[1]
        text = "\n".join(turns)
        for jx, ann in enumerate(annotations):
            items.append(
                {
                    "item_id": f"dialogre_{ix:05d}_{jx:02d}",
                    "dataset": "dialogre",
                    "text": text,
                    "head": ann.get("x", ""),
                    "tail": ann.get("y", ""),
                    "head_type": ann.get("x_type", ""),
                    "tail_type": ann.get("y_type", ""),
                    "labels": list(ann.get("r", ())),
                }
            )
    return items


def redocred_items(data: Sequence[Any], max_docs: int | None = None) -> list[dict[str, Any]]:
    """``(document, head entity, tail entity) -> Wikidata property`` positives.

    **Positives only, and the reason is Re-DocRED's whole point**: DocRED's
    unlabelled pairs are not reliably negatives, which is the false-negative
    problem Re-DocRED was built to fix.  Sampling negatives from unlabelled pairs
    here would reintroduce it one layer up.  Negative construction is the
    trainer's business and is done per the contract's item-6 rule.
    """
    items: list[dict[str, Any]] = []
    for ix, doc in enumerate(data[: max_docs or len(data)]):
        sents = doc.get("sents", [])
        text = " ".join(" ".join(s) for s in sents)
        vertices = doc.get("vertexSet", [])
        for jx, label in enumerate(doc.get("labels", ())):
            h, t = label.get("h"), label.get("t")
            if h is None or t is None or h >= len(vertices) or t >= len(vertices):
                continue
            items.append(
                {
                    "item_id": f"redocred_{ix:05d}_{jx:03d}",
                    "dataset": "redocred",
                    "text": text,
                    "head": vertices[h][0].get("name", ""),
                    "tail": vertices[t][0].get("name", ""),
                    "head_type": vertices[h][0].get("type", ""),
                    "tail_type": vertices[t][0].get("type", ""),
                    "labels": [label.get("r", "")],
                    "evidence": list(label.get("evidence", ())),
                }
            )
    return items


def _torque_passages(data: Any) -> Iterator[Mapping[str, Any]]:
    """Passages from either TORQUE split shape.

    **The two splits do not ship the same JSON**, which a train-only loader
    hides until the day dev is loaded: ``train.json`` is a *list* of annotator
    HITs each carrying a ``passages`` list, while ``dev.json`` is a *dict* keyed
    by passage id whose values are passage records directly.  Normalising here
    keeps the difference in one place — the fix-F6 pattern, applied to a file
    format.  (Found 15 Aug 2026 by the dataset survey; the old loader raised
    ``AttributeError: 'str' object has no attribute 'get'`` on dev, because it
    iterated the dict's keys.)
    """
    if isinstance(data, Mapping):
        for passage in data.values():
            if isinstance(passage, Mapping):
                yield passage
        return
    for hit in data or ():
        if not isinstance(hit, Mapping):
            continue
        for passage in hit.get("passages", ()) or ():
            if isinstance(passage, Mapping):
                yield passage


def _torque_qa_pairs(passage: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Question/answer groups, list-shaped (train) or question-keyed (dev)."""
    pairs = passage.get("question_answer_pairs", ())
    if isinstance(pairs, Mapping):
        for question, qa in pairs.items():
            if isinstance(qa, Mapping):
                # Dev keys the group by its question text and does not repeat it
                # inside the value; train carries a "question" field.
                yield {**qa, "question": qa.get("question", question)}
        return
    for qa in pairs or ():
        if isinstance(qa, Mapping):
            yield qa


def torque_items(data: Sequence[Any], max_passages: int | None = None) -> list[dict[str, Any]]:
    """``(passage, temporal question) -> answer spans``.

    TORQUE ships as annotator HITs, each holding several passages, each holding
    several question/answer groups.  Flattened to one item per question, which is
    the unit its exact-match/F1 metric is defined over.  **Both split shapes are
    accepted** — see :func:`_torque_passages`.
    """
    items: list[dict[str, Any]] = []
    seen = 0
    for passage in _torque_passages(data):
        if max_passages is not None and seen >= max_passages:
            return items
        seen += 1
        text = passage.get("passage", "")
        # ``events`` is the passage's event inventory — the answer to the
        # implicit question "which words are events".  Carried on every item
        # rather than emitted as one, because D4's temporal head needs to know
        # what the candidate events *are* before it can order them.
        events = _spans_of(passage.get("events", ()))
        for qa in _torque_qa_pairs(passage):
            answer = qa.get("answer", {})
            items.append(
                {
                    "item_id": f"torque_{len(items):06d}",
                    "dataset": "torque",
                    "text": text,
                    "question": qa.get("question", ""),
                    "events": events,
                    "answer_spans": list(answer.get("spans", []))
                    if isinstance(answer, Mapping)
                    else [],
                    # TORQUE marks questions whose correct answer is *no
                    # events*.  Keeping them is not optional: they are the
                    # only negative supervision the dataset gives, and
                    # dropping them would train a head that always answers.
                    "is_default_question": bool(qa.get("is_default_question", False)),
                }
            )
    return items


def _spans_of(events: Any) -> list[str]:
    """TORQUE's event inventory.

    Two shapes again: train gives a *list* of ``{"answer": {"spans": [...]}}``
    entries, dev gives that record directly as a single mapping.
    """
    if isinstance(events, Mapping):
        events = [events]
    out: list[str] = []
    for entry in events or ():
        if isinstance(entry, Mapping):
            answer = entry.get("answer", {})
            if isinstance(answer, Mapping):
                out.extend(answer.get("spans", []) or [])
    return out


# --------------------------------------------------------------------------
# the mapping loss — the number the contract actually cares about
# --------------------------------------------------------------------------


def mapping_report(
    items: Sequence[Mapping[str, Any]], mapping: Mapping[str, str], dataset: str
) -> dict[str, Any]:
    """Which native classes survive the GRAFT mapping, which are dropped, which merge.

    Printed with the loader (exit criterion 9) so that "trained on DialogRE"
    cannot be written next to a mapping that discarded most of it.  Three
    numbers, and the third is the one a reviewer will ask for:

    * **coverage** — the share of labelled items whose native class has a GRAFT
      target at all;
    * **dropped classes** — named, not counted, because "26 dropped" and "26
      dropped including every conflict-bearing relation" are different findings;
    * **merges** — GRAFT targets receiving more than one native class, which is
      where a downstream macro-F1 silently stops being comparable to the
      dataset's published numbers.
    """
    native: dict[str, int] = {}
    for item in items:
        for label in item.get("labels", ()) or ():
            native[label] = native.get(label, 0) + 1

    mapped = {k: v for k, v in native.items() if k in mapping}
    dropped = {k: v for k, v in native.items() if k not in mapping}
    merges: dict[str, list[str]] = {}
    for native_label, target in sorted(mapping.items()):
        if native_label in native:
            merges.setdefault(target, []).append(native_label)

    total = sum(native.values())
    return {
        "dataset": dataset,
        "items": len(items),
        "labelled_instances": total,
        "native_classes": len(native),
        "mapped_classes": len(mapped),
        "dropped_classes": sorted(dropped),
        "dropped_instances": sum(dropped.values()),
        "instance_coverage": (sum(mapped.values()) / total) if total else float("nan"),
        "merges": {k: v for k, v in sorted(merges.items()) if len(v) > 1},
        "reading": (
            "the GRAFT mapping is a declared, lossy adaptation and is never "
            "presented as native supervision (GATE0_CONTRACT.md item 1). D3/D4 "
            "train and are scored on the NATIVE label set; this table is what the "
            "GRAFT-schema application costs."
        ),
    }


def loader_artefact(root: Path | None = None, limit: int | None = 200) -> dict[str, Any]:
    """One artefact carrying every dataset's pin, licence, sizes and mapping loss.

    ``limit`` bounds the parse for a smoke run; the mapping percentages are then
    over that prefix and the artefact says so, rather than reporting a
    prefix-derived number as if it covered the corpus.
    """
    out: dict[str, Any] = {"limited_to_first_n_docs": limit, "datasets": {}}
    for name, spec in DATASETS.items():
        entry: dict[str, Any] = {
            "decoder": spec["decoder"],
            "source": spec["source"],
            "licence": spec["licence"],
            "metric": spec["metric"],
        }
        try:
            if name == "dialogre":
                data = load_split(name, "train", root)
                items = dialogre_items(data[:limit] if limit else data)
                entry["mapping"] = mapping_report(items, DIALOGRE_TO_GRAFT, name)
            elif name == "redocred":
                data = load_split(name, "train", root)
                items = redocred_items(data, max_docs=limit)
                entry["mapping"] = mapping_report(items, REDOCRED_TO_GRAFT, name)
            else:
                data = load_split(name, "train", root)
                items = torque_items(data, max_passages=limit)
                entry["mapping"] = {
                    "dataset": name,
                    "items": len(items),
                    "note": "TORQUE's answers are event spans, not typed relations; "
                    "the GRAFT target is structural (valid_during) and the "
                    "class-level mapping table does not apply",
                    "graft_target": TORQUE_TO_GRAFT["temporal_answer"],
                }
            entry["items_parsed"] = len(items)
        except (FileNotFoundError, ValueError) as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        out["datasets"][name] = entry
    return out
