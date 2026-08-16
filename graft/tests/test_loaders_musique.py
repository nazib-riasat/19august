"""MuSiQue-Full: the pin, the JSONL reader, and the contrast-pair grouping.

Phase 8's decision 1 sources the answerability gate from **MuSiQue-Full**, whose
defining property is that each question's unanswerable twin carries a
**byte-identical question string** — so the classifier cannot score answerability
from wording and is forced onto pool-side features (`GRAFT_PHASE8_BUILD.md` G2).
Everything here is about keeping that property intact between the file and the
trainer.

Runs on a bare interpreter. The raw corpus is gitignored and re-fetched per
machine, so the real-file checks skip when it is absent and the grouping logic is
tested on fixtures that always run.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from graft.graphbuild.loaders import MUSIQUE_ROOT, load_split, musique_pairs
from graft.graphbuild.pins import DATASETS, stage_b_fingerprint


def _row(pair_id: str, answerable: bool, question: str = "who did it?", supporting: int = 2):
    """One MuSiQue-Full row, in the shape the real file ships."""
    return {
        "id": pair_id,
        "question": question,
        "answerable": answerable,
        "answer": "someone" if answerable else "",
        "answer_aliases": [],
        "question_decomposition": [],
        "paragraphs": [
            {
                "idx": i,
                "title": f"t{i}",
                "paragraph_text": f"p{i}",
                "is_supporting": answerable and i < supporting,
            }
            for i in range(20)
        ],
    }


# -- the pin ----------------------------------------------------------------


def test_musique_is_pinned_as_full_not_ans():
    """Decision 1 picks **Full**; Ans is Phase 9's and is a different corpus.

    Ans is answerable-only, so a gate trained on it would have no negatives and
    the byte-identical contrast — the entire reason Full was chosen — would not
    exist.  Pinning the wrong one is a silent, plausible-looking mistake, which
    is why the name is asserted rather than assumed.
    """
    spec = DATASETS["musique_full"]
    assert spec["phase"] == 8
    assert "CC BY 4.0" in spec["licence"]
    assert set(spec["files"]) == {"train", "dev", "test"}
    for split, (rel, sha) in spec["files"].items():
        assert rel == f"musique/musique_full_v1.0_{split}.jsonl"
        assert len(sha) == 64 and int(sha, 16) >= 0  # a real hex digest
    assert "musique_ans" not in str(spec["files"])


def test_adding_a_phase_8_dataset_did_not_move_the_stage_b_fingerprint():
    """`DATASETS` is deliberately absent from `frozen_values()`.

    If it were included, registering a Phase-8 corpus would re-identify every
    Phase-6 run — a configuration change nobody made, invalidating comparisons
    for a reason unrelated to Stage B.  Measured before and after the 15 Aug 2026
    addition: unchanged.
    """
    assert stage_b_fingerprint().startswith("a2520eef45b9e084")


# -- the JSONL branch -------------------------------------------------------


def test_load_split_parses_jsonl_and_still_verifies_the_sha(tmp_path):
    """MuSiQue ships JSON Lines; the Phase-6 three ship JSON.

    Handled inside `load_split` so the SHA check stays on **one** path — a second
    reader in `graft/gate/` would be a second place for verify-then-parse to
    drift.  The SHA must still bite on the new branch, which is what the mismatch
    half of this test asserts.
    """
    root = tmp_path / "raw"
    (root / "musique").mkdir(parents=True)
    target = root / "musique" / "musique_full_v1.0_dev.jsonl"
    rows = [_row("2hop__a", True), _row("2hop__a", False)]
    blob = "\n".join(json.dumps(r) for r in rows).encode("utf-8")
    target.write_bytes(blob)

    loaded = load_split("musique_full", "dev", root=root, verify=False)
    assert isinstance(loaded, list) and len(loaded) == 2

    # and the pinned digest is not this file's, so verification must refuse
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_split("musique_full", "dev", root=root, verify=True)

    # a trailing newline must not become an empty record
    target.write_bytes(blob + b"\n\n")
    assert len(load_split("musique_full", "dev", root=root, verify=False)) == 2


# -- the grouping, and the trap it exists for -------------------------------


def test_id_is_the_pair_key_not_a_row_key():
    """**The defect this function exists to prevent.**

    Both members of a contrast pair carry the *same* ``id`` (verified on the real
    dev split, 15 Aug 2026).  `{row["id"]: row}` therefore keeps one twin and
    discards the other — halving the corpus and destroying the contrast that is
    the whole reason decision 1 chose Full.
    """
    rows = [_row("2hop__a", True), _row("2hop__a", False), _row("2hop__b", True), _row("2hop__b", False)]
    naive = {r["id"]: r for r in rows}
    assert len(naive) == 2 and len(rows) == 4  # the trap, demonstrated

    pairs, report = musique_pairs(rows)
    assert len(pairs) == 2
    assert report["rows"] == 4 and report["groups"] == 2 and report["pairs"] == 2
    assert report["malformed"] == {"wrong_size": 0, "wrong_labels": 0, "question_mismatch": 0}


def test_each_pair_carries_one_answerable_and_one_unanswerable():
    pairs, _ = musique_pairs([_row("2hop__a", True), _row("2hop__a", False)])
    assert pairs[0]["answerable"]["answerable"] is True
    assert pairs[0]["unanswerable"]["answerable"] is False
    assert pairs[0]["question"] == pairs[0]["unanswerable"]["question"]
    # the twin's supporting paragraphs are gone: that is what makes it unanswerable
    assert not any(p["is_supporting"] for p in pairs[0]["unanswerable"]["paragraphs"])
    assert any(p["is_supporting"] for p in pairs[0]["answerable"]["paragraphs"])


@pytest.mark.parametrize(
    "rows, cause",
    [
        ([_row("2hop__a", True)], "wrong_size"),
        ([_row("2hop__a", True), _row("2hop__a", True)], "wrong_labels"),
        ([_row("2hop__a", True), _row("2hop__a", False, question="different?")], "question_mismatch"),
    ],
    ids=["lone_row", "both_answerable", "questions_differ"],
)
def test_malformed_groups_are_counted_not_silently_dropped(rows, cause):
    """A future release that changes the shape must be *visible*.

    ``question_mismatch`` is the one that matters most: byte-identical questions
    are the property the leakage argument rests on, so it is checked rather than
    assumed — if it ever stopped holding, silently accepting the pair would let
    the gate learn wording while the plan still claimed it could not.
    """
    pairs, report = musique_pairs(rows)
    assert pairs == []
    assert report["malformed"][cause] == 1


# -- the real corpus, when it is on this machine ----------------------------


def _dev_present() -> bool:
    rel, _sha = DATASETS["musique_full"]["files"]["dev"]
    return (MUSIQUE_ROOT / rel).is_file()


@pytest.mark.skipif(not _dev_present(), reason="MuSiQue-Full is gitignored and fetched per machine")
def test_the_real_dev_split_is_all_exact_contrast_pairs():
    """Measured 15 Aug 2026 on the official ``musique_v1.0.zip``:
    4,834 rows → 2,417 pairs, zero malformed, perfectly balanced."""
    rows = load_split("musique_full", "dev", root=MUSIQUE_ROOT)
    assert len(rows) == 4834
    assert sum(1 for r in rows if r["answerable"]) == 2417

    pairs, report = musique_pairs(rows)
    assert report["pairs"] == 2417
    assert sum(report["malformed"].values()) == 0
    assert all(p["answerable"]["question"] == p["unanswerable"]["question"] for p in pairs)


@pytest.mark.skipif(not _dev_present(), reason="MuSiQue-Full is gitignored and fetched per machine")
def test_the_pinned_sha_matches_the_file_on_disk():
    rel, expected = DATASETS["musique_full"]["files"]["dev"]
    got = hashlib.sha256((MUSIQUE_ROOT / rel).read_bytes()).hexdigest()
    assert got == expected
