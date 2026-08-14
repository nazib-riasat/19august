"""The multi-seed trainer (P6.11) and the adjudication batch.

Exit criteria 6, 10 and 15.  Everything here runs on tiny synthetic graphs and a
two-epoch budget — the point is that the *loop* is correct (identical budget,
early stopping restores, unreachable gold counted, splits user-level), not that
a four-node graph learns anything.  Nothing in this file trains a real arm or
touches a real corpus.
"""

from __future__ import annotations

import math

import pytest

from graft.eventlog import EventLog
from graft.graphbuild.commit import Committer, entity_node
from graft.graphbuild.items import adjudication_items
from graft.schemas import Assertion, AssertionFlags, SourceSpan, Turn

torch = pytest.importorskip("torch", reason="the trainer needs the ML environment")

from graft.graphbuild.train import (  # noqa: E402
    ARM_VARIANTS,
    SPLIT_SEED,
    D1Arm,
    FeatureCache,
    SimilarityArm,
    build_arm,
    calibrate,
    parse_d1_gold,
    predict_d1,
    read_labels,
    split_questions,
    train_d1_arm,
)

T1 = "2023-01-01T00:00:00+00:00"
TINY = {"epochs": 2, "lr": 1e-3, "weight_decay": 1e-4, "early_stop_patience": 1}


# -- splits (GATE0 item 5) ---------------------------------------------------


def test_splits_are_user_level_no_question_in_two_splits():
    """A question_id IS one simulated user's whole haystack, so splitting on
    question ids means no conversation straddles a boundary — item 5's
    requirement, met structurally."""
    qids = [f"q{i}" for i in range(20)]
    splits = split_questions(qids)
    seen = [q for part in ("train", "dev", "test") for q in splits[part]]
    assert sorted(seen) == sorted(qids)
    assert len(seen) == len(set(seen)), "a question appears in two splits"


def test_splits_are_stratified_by_question_type():
    """Knowledge-update questions are the rare-class carrier; an unstratified
    draw moves D2's rare classes between splits (item 5's own reasoning)."""
    qids = [f"k{i}" for i in range(10)] + [f"m{i}" for i in range(10)]
    types = {q: ("knowledge-update" if q.startswith("k") else "multi-session") for q in qids}
    splits = split_questions(qids, types)
    for part in ("train", "dev", "test"):
        assert any(q.startswith("k") for q in splits[part]), f"{part} lost the rare type"


def test_splits_are_a_function_of_the_seed_not_of_input_order():
    a = split_questions(["q3", "q1", "q2", "q4", "q5"])
    b = split_questions(["q5", "q4", "q1", "q3", "q2"])
    assert a == b
    assert split_questions(["q1", "q2", "q3", "q4", "q5"], seed=SPLIT_SEED + 1) != a


# -- gold parsing ------------------------------------------------------------


def test_link_gold_keeps_the_entity_id_apart_from_the_action():
    """The primary metric scores action AND id as a conjunction; a parser that
    dropped the id would silently weaken it to action-only."""
    assert parse_d1_gold("LINK_EXISTING(e7)") == ("LINK_EXISTING", "e7")
    assert parse_d1_gold("CREATE_NEW_ENTITY") == ("CREATE_NEW_ENTITY", None)
    with pytest.raises(ValueError, match="unknown D1 label"):
        parse_d1_gold("MERGE_ALL")


def test_only_pass_one_labels_are_read_for_training():
    """Pass 2 is the kappa re-annotation of a subset; mixing it in would put two
    labels of one item in the map with the later write silently winning."""
    rows = [
        {"item_id": "d1_0", "label": "NON_ENTITY", "pass": 1},
        {"item_id": "d1_0", "label": "DEFER", "pass": 2},
    ]
    assert read_labels(rows) == {"d1_0": "NON_ENTITY"}


# -- the graph the arms train on ---------------------------------------------


@pytest.fixture()
def built(tmp_path):
    """A small committed graph plus two D1 items, one linkable, one cold start."""
    log = EventLog.open(tmp_path / "e.jsonl")
    log.append("span.add", SourceSpan("sp1", "t1", 0, 6).to_dict())
    log.append(
        "assertion.add",
        Assertion("a1", "claim", "the user has a widget", ("sp1",),
                  AssertionFlags(asserted_by="user", entailed_score=0.9), T1,
                  eligibility="eligible").to_dict(),
    )
    log.append("turn.add", Turn("t1", "q1", "s1", "user", T1, "Widget every day").to_dict())

    committer = Committer(log)
    committer.create_entity("a1", "claim", "Widget", "q1", T1, ["sp1"])
    entity = entity_node("Widget", "q1").node_id
    seq_after = committer.snapshot().snapshot_id

    items = [
        {
            "item_id": "d1_0000", "question_id": "q1", "mention": "Widget",
            "candidates": [], "stage_b_seq": 0, "turn_text": "Widget every day",
        },
        {
            "item_id": "d1_0001", "question_id": "q1", "mention": "Widget",
            "candidates": [{"entity_id": entity, "name": "widget", "score": 1.0,
                            "how": "normalised_exact"}],
            "stage_b_seq": seq_after, "turn_text": "Widget every day",
        },
    ]
    gold = {"d1_0000": "CREATE_NEW_ENTITY", "d1_0001": f"LINK_EXISTING({entity})"}
    yield log, items, gold, entity
    log.close()


def _vectors(items, dim=384):
    torch.manual_seed(0)
    return {i["item_id"]: torch.randn(dim) for i in items}


# -- the identical-budget loop (decision 10) ---------------------------------


@pytest.mark.parametrize("arm_name", ["E1", "E2", "E3"])
def test_every_arm_trains_under_the_shared_budget(built, arm_name):
    """Exit criterion 6 with a loop behind it: forward, backward, early-stop,
    and the same budget dict for every arm."""
    log, items, gold, _ = built
    arm = build_arm(arm_name, hidden=8)
    cache = FeatureCache(log, ARM_VARIANTS[arm_name], embed=_stub_embed)
    report = train_d1_arm(
        arm, cache, {"train": items, "dev": items}, gold, _vectors(items),
        seed=13, budget=TINY,
    )
    assert report["epochs_run"] >= 1
    assert report["parameters"] > 0
    assert math.isfinite(report["best_dev_loss"])
    assert report["budget"]["epochs"] == 2  # the override is reported, not hidden


def _stub_embed(texts):
    import numpy as np

    from graft.canonical import digest_of

    out = np.zeros((len(texts), 384), dtype="float32")
    for i, t in enumerate(texts):
        seed = int(digest_of(t, 8), 16) % (2**32)
        v = np.random.default_rng(seed).standard_normal(384).astype("float32")
        out[i] = v / (np.linalg.norm(v) + 1e-12)
    return out


def test_early_stopping_restores_the_best_state_rather_than_merely_stopping(built):
    """A loop that stops without restoring leaves the arm at the *worst* epoch
    it saw after the best — the arm that gets scored is then not the arm the dev
    loss selected.  Recomputed with the trainer's own loss and its own weights,
    because a dev loss under different class weights is a different quantity."""
    from graft.graphbuild.train import d1_action_weights, d1_loss

    log, items, gold, _ = built
    vectors = _vectors(items)
    arm = build_arm("E1", hidden=8)
    cache = FeatureCache(log, "base")
    report = train_d1_arm(
        arm, cache, {"train": items, "dev": items}, gold, vectors,
        seed=13, budget={**TINY, "epochs": 6, "early_stop_patience": 1},
    )

    weights = d1_action_weights(items, gold)
    with torch.no_grad():
        restored, _, _ = d1_loss(arm, cache, items, gold, vectors, weights)
    assert float(restored.item()) == pytest.approx(report["best_dev_loss"], abs=1e-5), (
        "the arm left in hand is not the one that achieved best_dev_loss"
    )


def test_two_runs_at_one_seed_produce_identical_parameters(built):
    """Seeds {13, 42, 7} only mean something if a seed reproduces a run — and
    that requires seeding **initialisation**, not just the loop."""
    log, items, gold, _ = built
    vectors = _vectors(items)
    finals = []
    for _ in range(2):
        arm = build_arm("E1", hidden=8, seed=13)
        cache = FeatureCache(log, "base")
        train_d1_arm(arm, cache, {"train": items, "dev": items}, gold, vectors,
                     seed=13, budget=TINY)
        finals.append(torch.cat([p.flatten() for p in arm.parameters()]))
    assert torch.equal(finals[0], finals[1])


def test_different_seeds_give_different_initialisations(built):
    """The other half: if the seed did not reach construction, three seeds would
    estimate variance over dropout alone."""
    a = build_arm("E1", hidden=8, seed=13)
    b = build_arm("E1", hidden=8, seed=42)
    flat = lambda m: torch.cat([p.flatten() for p in m.parameters()])
    assert not torch.equal(flat(a), flat(b))


def test_class_weights_come_from_train_only(built):
    """Dev-derived weights would leak the dev balance into the objective."""
    from graft.graphbuild.train import d1_action_weights

    log, items, gold, _ = built
    arm = build_arm("E1", hidden=8)
    cache = FeatureCache(log, "base")
    report = train_d1_arm(
        arm, cache, {"train": items[:1], "dev": items}, gold, _vectors(items),
        seed=13, budget=TINY,
    )
    expected = d1_action_weights(items[:1], gold)
    assert report["class_weights"] == [float(w) for w in expected]


def test_unreachable_gold_is_counted_and_kept_out_of_the_loss(built):
    """Gold links to an entity the generator never proposed: no logit
    corresponds to the right answer, so the item cannot train — but dropping it
    from the *test* set would inflate every arm on the generator's misses."""
    log, items, gold, _ = built
    unreachable = dict(items[0])
    unreachable["item_id"] = "d1_0002"
    gold = {**gold, "d1_0002": "LINK_EXISTING(never-proposed)"}
    batch = items + [unreachable]

    arm = build_arm("E1", hidden=8)
    cache = FeatureCache(log, "base")
    report = train_d1_arm(
        arm, cache, {"train": batch, "dev": batch}, gold, _vectors(batch),
        seed=13, budget=TINY,
    )
    assert report["train_items_unreachable"] == 1

    # ...and it still receives a prediction, which will be wrong by construction.
    preds = predict_d1(arm, cache, batch, _vectors(batch))
    assert len(preds) == len(batch)
    assert preds[-1]["entity_id"] != "never-proposed"


def test_predictions_come_back_one_per_item_in_order(built):
    """`gate1` scores predictions against gold positionally and now refuses
    mismatched lengths; the trainer must not be the thing that mismatches."""
    log, items, gold, _ = built
    arm = build_arm("E2", hidden=8)
    cache = FeatureCache(log, "base")
    preds = predict_d1(arm, cache, items, _vectors(items))
    assert len(preds) == len(items)
    assert all(p["action"] in ("LINK_EXISTING", "CREATE_NEW_ENTITY", "NON_ENTITY", "DEFER")
               for p in preds)
    assert preds[0]["entity_id"] is None  # no candidates -> cannot link


def test_the_feature_cache_builds_one_graph_per_distinct_sequence(built):
    """G12 per example, without replaying the log per example per epoch: the
    cache key IS the correctness condition."""
    log, items, _, _ = built
    cache = FeatureCache(log, "base")
    cache.at(items[0]["stage_b_seq"])
    cache.at(items[1]["stage_b_seq"])
    cache.at(items[0]["stage_b_seq"])  # repeat
    assert len(cache) == 2


def test_the_two_snapshots_differ_so_the_pin_is_doing_work(built):
    """If at(stage_b_seq) returned the same graph for both items, the leak guard
    would be decorative."""
    log, items, _, _ = built
    cache = FeatureCache(log, "base")
    before = cache.at(items[0]["stage_b_seq"])
    after = cache.at(items[1]["stage_b_seq"])
    assert before.counts().get("Entity", 0) == 0
    assert after.counts().get("Entity", 0) == 1


# -- the no-learning control -------------------------------------------------


def test_the_similarity_arm_has_no_parameters_and_still_tunes_on_dev():
    """"Tuned on dev, scored on test" holds for every arm; the simple method
    gets one knob instead of a gradient, not an exemption."""
    items = [
        {"item_id": "i0", "candidates": [{"entity_id": "e1", "score": 0.9}]},
        {"item_id": "i1", "candidates": [{"entity_id": "e2", "score": 0.1}]},
    ]
    gold = {"i0": "LINK_EXISTING(e1)", "i1": "CREATE_NEW_ENTITY"}
    arm = SimilarityArm()
    chosen = arm.fit(items, gold)
    assert 0.1 < chosen <= 0.9
    assert arm.parameter_count() == 0
    preds = arm.predict(items)
    assert preds[0] == {"action": "LINK_EXISTING", "entity_id": "e1", "confidence": 0.9}
    assert preds[1]["action"] == "CREATE_NEW_ENTITY"


def test_the_similarity_arm_cold_starts_to_create(built):
    """An empty candidate list is the legal cold start (G4), not an error."""
    assert SimilarityArm().predict_one({"item_id": "x", "candidates": []})["action"] == (
        "CREATE_NEW_ENTITY"
    )


# -- calibration (decision 8) ------------------------------------------------


def test_calibration_reports_both_metrics_before_and_after_and_fits_on_dev():
    """Decision 9's prerequisite: an uncalibrated confidence floor is a number
    with no meaning.  Brier and ECE both, because they fail differently."""
    torch.manual_seed(0)
    logits = torch.randn(200, 4) * 4
    targets = torch.randint(0, 4, (200,))
    report = calibrate(logits, targets)
    assert report["fitted_on"] == "dev"
    assert report["ece_after"] < report["ece_before"]
    assert all(math.isfinite(report[k]) for k in
               ("brier_before", "brier_after", "ece_before", "ece_after"))


# -- the adjudication batch (exit criterion 15's last third) -----------------


def test_the_adjudication_batch_is_only_the_disagreements():
    """Item 7: self-adjudication of the disagreeing items after the kappa pass.
    Agreements are not work."""
    items = [{"item_id": "d1_0", "mention": "Fitbit"}, {"item_id": "d1_1", "mention": "Priya"}]
    a = {"d1_0": "CREATE_NEW_ENTITY", "d1_1": "NON_ENTITY"}
    b = {"d1_0": "CREATE_NEW_ENTITY", "d1_1": "DEFER"}
    rows = adjudication_items(items, a, b)
    assert [r["item_id"] for r in rows] == ["d1_1"]
    assert rows[0]["pass_1"] == "NON_ENTITY" and rows[0]["pass_2"] == "DEFER"
    assert rows[0]["adjudicated"] == ""  # the human column


def test_an_id_only_disagreement_is_surfaced_but_marked_as_not_action_level():
    """The kappa convention collapses LINK_EXISTING(<id>) to its action; an
    id-only disagreement is still a real disagreement to resolve, it is simply
    not what kappa counts."""
    collapse = lambda label: "LINK_EXISTING" if label.startswith("LINK_EXISTING") else label
    rows = adjudication_items(
        [{"item_id": "d1_0", "mention": "Fitbit"}],
        {"d1_0": "LINK_EXISTING(e1)"},
        {"d1_0": "LINK_EXISTING(e2)"},
        collapse=collapse,
    )
    assert len(rows) == 1
    assert rows[0]["action_level_disagreement"] is False


def test_items_labelled_in_only_one_pass_are_not_adjudicated():
    """Pass 2 is a 20-item subset; the un-re-annotated remainder has nothing to
    disagree with."""
    rows = adjudication_items(
        [{"item_id": "d1_0"}, {"item_id": "d1_1"}],
        {"d1_0": "DEFER", "d1_1": "DEFER"},
        {"d1_0": "NON_ENTITY"},
    )
    assert [r["item_id"] for r in rows] == ["d1_0"]


# -- gold-label durability (the blocker caught before this path ever ran) -----


def _driver():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "phase6_gate1.py"
    spec = importlib.util.spec_from_file_location("phase6_gate1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_positional_item_ids_do_not_survive_a_re_derivation():
    """The measurement that made the fix necessary: the spike's ``d1_0000`` and
    the Phase-6 log's ``d1_0000`` name **different mentions**, so a gold join on
    ``item_id`` attaches every label to the wrong item.  Pinned as a fact about
    the data, so nobody re-introduces the join."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "data" / "phase2_5"
    spike = [json.loads(l) for l in (root / "d1_items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert spike[0]["item_id"] == "d1_0000"
    # Two different derivations, same id, different mention — that is the trap.
    assert spike[0]["mention"] == "Gitzo"


def test_gold_is_keyed_by_the_span_not_the_item_id():
    """The durable key is a fact about the corpus (turn + offsets), not about
    the order items happened to be emitted in."""
    from graft.ids import span_id

    driver = _driver()
    gold, report = driver.load_d1_gold()
    assert report["labels_read"] == len(gold), "one label per span"
    assert "span_id" in report["keyed_by"]
    assert report["labels_without_a_source_item"] == 0
    # Every key is a real span id, not an item id.
    assert all(not k.startswith("d1_") for k in gold)
    # And it is derivable from the source item's three components — for whichever
    # item file the surviving labels actually came from.
    #
    # **Not hardcoded to the spike batch any more** (14 Aug 2026).  It was, and
    # the assertion broke the moment the spike-batch labels were moved to
    # `labels/superseded/`: their candidate lists carry synthetic entity ids
    # (`e_08e075c7_000`) and **0 of 19 exist as real graph nodes**, so every
    # LINK_EXISTING label in that batch pointed at a dead namespace.  Pinning the
    # test to that file asserted the durable-key property *via* a batch the
    # project had just retired — so it failed for a reason that had nothing to do
    # with the property under test.
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "data" / "phase2_5"
    derivable = False
    for path in sorted(root.glob("d1_items*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if span_id(item["turn_id"], item["start"], item["end"]) in gold:
                derivable = True
                break
        if derivable:
            break
    assert derivable, (
        "no gold key was derivable from any d1_items*.jsonl item, so the labels "
        "are keyed by something other than the span"
    )


def test_bootstrap_labels_never_reach_the_decisive_path():
    """They carry ``answers_gate0_item8: false`` and are machine assisted;
    training a decisive arm on them is the bootstrap-labels mistake two phases
    later."""
    driver = _driver()
    _, report = driver.load_d1_gold()
    assert all("bootstrap" not in name for name in report["label_files"])


# -- the 14 Aug review's nine confirmed findings, as regressions ---------------


def test_a_stale_link_label_is_excluded_not_scored_as_a_failure():
    """A LINK_EXISTING label naming an entity outside this graph's namespace is
    a stale *label*, not a model error and not a candidate-recall miss.  The
    spike minted ids positionally (``e_41698283_000``); Phase 6 derives them by
    content hash, so 0 of 7 spike link labels name a node that exists here."""
    driver = _driver()
    gold_all, report_all = driver.load_d1_gold(None)
    gold_scoped, report_scoped = driver.load_d1_gold({"only-this-entity"})
    assert report_all["stale_link_labels_excluded"] == 0, "no filter without a namespace"
    assert report_scoped["stale_link_labels_excluded"] > 0
    assert len(gold_scoped) < len(gold_all)
    # ...and every surviving label is one the graph could satisfy.
    from graft.graphbuild.train import parse_d1_gold

    for label in gold_scoped.values():
        action, entity = parse_d1_gold(label)
        assert action != "LINK_EXISTING" or entity == "only-this-entity"


def test_duplicate_item_ids_across_item_files_are_refused(tmp_path, monkeypatch):
    """Positional ids restart at zero in every derivation; two item files
    sharing an id space would re-key labels to the wrong spans."""
    import json as _json

    driver = _driver()
    root = tmp_path
    (root / "data" / "phase2_5" / "labels").mkdir(parents=True)
    row = {"item_id": "d1_0000", "turn_id": "t1", "start": 0, "end": 3}
    for name in ("d1_items.jsonl", "d1_items_v2.jsonl"):
        (root / "data" / "phase2_5" / name).write_text(
            _json.dumps(row) + "\n", encoding="utf-8"
        )
    monkeypatch.setattr(driver, "REPO", root)
    with pytest.raises(SystemExit, match="appears in both"):
        driver.load_d1_gold()


def test_the_cli_refuses_smoke_and_decisive_together():
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [sys.executable, str(repo / "scripts" / "phase6_gate1.py"), "--smoke", "--decisive"],
        capture_output=True, text=True, cwd=repo,
    )
    assert out.returncode != 0
    assert "not allowed with argument" in (out.stderr + out.stdout)


def test_training_refuses_when_no_dev_item_is_scorable(built):
    """A model that never saw a scorable dev item is a random model; reporting
    it as "early stopped" would have every arm return its initialisation and the
    McNemar table read as four arms agreeing."""
    log, items, gold, _ = built
    arm = build_arm("E1", hidden=8, seed=13)
    cache = FeatureCache(log, "base")
    with pytest.raises(ValueError, match="no scorable dev item"):
        train_d1_arm(arm, cache, {"train": items, "dev": []}, gold, _vectors(items),
                     seed=13, budget=TINY)


def test_a_zero_epoch_budget_does_not_crash(built):
    """``epochs=0`` is a legal budget; the counters must be bound before the
    loop rather than only inside it."""
    log, items, gold, _ = built
    arm = build_arm("E1", hidden=8, seed=13)
    cache = FeatureCache(log, "base")
    with pytest.raises(ValueError, match="no scorable dev item"):
        # Zero epochs means dev was never scored — the guard fires, and the
        # point is that it is a *guard*, not an UnboundLocalError.
        train_d1_arm(arm, cache, {"train": items, "dev": items}, gold,
                     _vectors(items), seed=13, budget={**TINY, "epochs": 0})


def test_features_are_timed_against_the_item_s_own_turn(built):
    """Left to default, the relative temporal encoding anchors at the latest
    edge in the WHOLE construction log — another conversation's calendar.  The
    cache keys on the reference so two references cannot collide."""
    log, items, _, _ = built
    cache = FeatureCache(log, "base")
    early = cache.at(items[1]["stage_b_seq"], "2023-01-01T00:00:00+00:00")
    late = cache.at(items[1]["stage_b_seq"], "2024-01-01T00:00:00+00:00")
    assert len(cache) == 2, "the reference time is part of the cache key"
    assert not torch.equal(early.x["Entity"], late.x["Entity"]), (
        "a different 'now' must produce a different temporal encoding"
    )


def test_a_decisive_run_refuses_an_empty_test_split():
    """With n = 0 every McNemar test reports p = 1.0 and 'not significant',
    which reads as a measured null and is the absence of a measurement."""
    from graft.graphbuild.gate1 import run_gate1

    with pytest.raises(ValueError, match="non-empty test split"):
        run_gate1({"proposed": [], "E1": []}, [], smoke=False)


def test_a_decisive_artefact_names_the_arms_that_did_not_run():
    """The rule block promises five arms; a run with four is legitimate, but the
    artefact must say which are missing rather than leaving a reader to infer
    it from a shorter comparisons table."""
    from graft.graphbuild.gate1 import run_gate1

    gold = [{"action": "NON_ENTITY", "entity_id": None}]
    artefact = run_gate1({"proposed": gold, "E1": gold}, gold, smoke=False)
    assert set(artefact["arms_declared"]) >= {"similarity", "E1", "E2", "llm_prompted", "proposed"}
    assert "llm_prompted" in artefact["arms_omitted"]
    assert "E2" in artefact["arms_omitted"]
    assert "similarity" in artefact["arms_omitted"]
    assert "E1" not in artefact["arms_omitted"]


# -- the 15 Aug audit: multi-annotator labels, and D2's parity with D1 ---------


def _two_annotator_tree(tmp_path, second_label):
    import json as _json

    root = tmp_path
    (root / "data" / "phase2_5" / "labels").mkdir(parents=True)
    item = {"item_id": "d1p_0000", "turn_id": "t1", "start": 0, "end": 3}
    (root / "data" / "phase2_5" / "d1_items_pilot.jsonl").write_text(
        _json.dumps(item) + "\n", encoding="utf-8"
    )
    for who, label in (("Alice", "NON_ENTITY"), ("Bob", second_label)):
        (root / "data" / "phase2_5" / "labels" / f"d1_labels_{who}_pilot.jsonl").write_text(
            _json.dumps({"item_id": "d1p_0000", "label": label, "pass": 1}) + "\n",
            encoding="utf-8",
        )
    return root


def test_two_annotators_disagreeing_on_an_item_refuse_rather_than_last_file_wins(
    tmp_path, monkeypatch
):
    """The annotate CLI writes a second annotator's labels under their own name
    (pass 1 — provenance is the point), so two files can label the same items.
    Concatenating rows would let the alphabetically-later file win silently:
    gold decided by filename sort order.  A disagreement is the adjudication
    batch's decision (contract item 7), never a loader's accident."""
    driver = _driver()
    monkeypatch.setattr(driver, "REPO", _two_annotator_tree(tmp_path, "DEFER"))
    with pytest.raises(SystemExit, match="adjudication"):
        driver.load_d1_gold()


def test_two_annotators_agreeing_on_an_item_pass_through(tmp_path, monkeypatch):
    driver = _driver()
    monkeypatch.setattr(driver, "REPO", _two_annotator_tree(tmp_path, "NON_ENTITY"))
    gold, report = driver.load_d1_gold()
    assert len(gold) == 1
    assert sorted(report["label_files"]) == [
        "d1_labels_Alice_pilot.jsonl",
        "d1_labels_Bob_pilot.jsonl",
    ]


def test_d2_training_refuses_when_no_dev_item_is_scorable(built):
    """Same guard as D1, same reason: a decoder that never saw a scorable dev
    item is its random initialisation, and returning it as 'trained' would
    poison the D2 secondary silently."""
    from graft.graphbuild.decoders import D2Decoder
    from graft.graphbuild.train import train_d2

    log, _, _, _ = built
    decoder = D2Decoder(8)
    cache = FeatureCache(log, "base")
    items = [{"item_id": "d2_0000", "stage_b_seq": 0}]
    gold = {"d2_0000": "INDEPENDENT"}

    def vectors(item, features):
        torch.manual_seed(0)
        return torch.randn(8), torch.randn(8)

    with pytest.raises(ValueError, match="no scorable dev item"):
        train_d2(decoder, cache, {"train": items, "dev": []}, gold, vectors,
                 seed=13, budget=TINY)


def test_d2_items_carry_the_linking_turn_as_their_own_now():
    """D1 items got ``turn_ts`` when the global temporal anchor was found; D2
    items are decided at link time (G5), so the linking turn is their 'now'.
    Leaving it off re-opens the same cross-conversation anchor for D2."""
    from graft.graphbuild.items import build_d2_item

    a = {"assertion_id": "a1", "text": "x", "turn_id": "t1", "session_id": "s1",
         "question_id": "q1", "session_date": "2023-01-01T00:00:00+00:00"}
    b = {"assertion_id": "a2", "text": "y", "turn_id": "t2", "session_id": "s2",
         "question_id": "q1", "session_date": "2023-02-01T00:00:00+00:00"}
    item = build_d2_item(0, a, b, turn_ts="2023-02-01T00:00:00+00:00")
    assert item["turn_ts"] == "2023-02-01T00:00:00+00:00"


# -- the 15 Aug audit, round 2: cross-batch spans, adjudication, provenance ----


def _cross_batch_tree(tmp_path, second_label, adjudicate=None):
    """Two item GENERATIONS (different id prefixes) naming the same span, each
    labelled by a different file."""
    import json as _json

    root = tmp_path
    (root / "data" / "phase2_5" / "labels").mkdir(parents=True)
    for prefix, batch in (("d1p", "pilot"), ("d1q", "next")):
        item = {"item_id": f"{prefix}_0000", "turn_id": "t1", "start": 0, "end": 3}
        (root / "data" / "phase2_5" / f"d1_items_{batch}.jsonl").write_text(
            _json.dumps(item) + "\n", encoding="utf-8"
        )
    labels = root / "data" / "phase2_5" / "labels"
    (labels / "d1_labels_Sabbir_pilot.jsonl").write_text(
        _json.dumps({"item_id": "d1p_0000", "label": "NON_ENTITY", "pass": 1}) + "\n",
        encoding="utf-8",
    )
    (labels / "d1_labels_Sabbir_next.jsonl").write_text(
        _json.dumps({"item_id": "d1q_0000", "label": second_label, "pass": 1}) + "\n",
        encoding="utf-8",
    )
    if adjudicate:
        (labels / "d1_labels_adjudicated.jsonl").write_text(
            "\n".join(
                _json.dumps({"item_id": iid, "label": lab, "pass": 1})
                for iid, lab in adjudicate
            )
            + "\n",
            encoding="utf-8",
        )
    return root


def test_two_batches_disagreeing_on_one_span_refuse_rather_than_filename_sort(
    tmp_path, monkeypatch
):
    """Two item generations assign different item_ids to the same span, so the
    item-level refusal can never see the conflict — only the span-keyed gold
    write can, and without a check the later file won silently (measured: the
    winner flipped with a filename rename)."""
    driver = _driver()
    monkeypatch.setattr(driver, "REPO", _cross_batch_tree(tmp_path, "DEFER"))
    with pytest.raises(SystemExit, match="one mention"):
        driver.load_d1_gold()


def test_two_batches_agreeing_on_one_span_pass_through(tmp_path, monkeypatch):
    driver = _driver()
    monkeypatch.setattr(driver, "REPO", _cross_batch_tree(tmp_path, "NON_ENTITY"))
    gold, _ = driver.load_d1_gold()
    assert len(gold) == 1  # one span, one label


def test_an_adjudicated_label_resolves_a_disagreement_instead_of_deadlocking(
    tmp_path, monkeypatch
):
    """The refusal names adjudication as the resolution, so adjudication must be
    a path code can consume: d1_labels_adjudicated*.jsonl overrides, per item.
    Covering both generations' item ids settles the span conflict too."""
    driver = _driver()
    root = _cross_batch_tree(
        tmp_path, "DEFER",
        adjudicate=[("d1p_0000", "DEFER"), ("d1q_0000", "DEFER")],
    )
    monkeypatch.setattr(driver, "REPO", root)
    gold, report = driver.load_d1_gold()
    assert report["adjudicated_overrides"] == 2
    assert list(gold.values()) == ["DEFER"], "the adjudicated decision stands"


def _annotate_module():
    import importlib.util
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[2] / "scripts" / "phase2_5"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("annotate_cli", scripts / "annotate.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))
    return module


def test_a_second_annotators_rows_carry_their_own_name_and_pass_one(
    tmp_path, monkeypatch
):
    """The row's provenance is the person who produced it, and a different
    person's labels are their own pass 1 — stamping them pass 2 would silently
    drop them from gold and hand every measured disagreement to the first
    annotator (the exact overwrite the loader refuses)."""
    import json as _json

    cli = _annotate_module()
    data = tmp_path / "data"
    labels = tmp_path / "labels"
    labels.mkdir()
    data.mkdir()
    (data / "d1_items.jsonl").write_text(
        _json.dumps({"item_id": "d1_0000", "mention": "X", "turn_text": "X here",
                     "start": 0, "end": 1, "candidates": [],
                     "actions": ["CREATE_NEW_ENTITY"]}) + "\n",
        encoding="utf-8",
    )
    (labels / "d1_labels_Sabbir.jsonl").write_text(
        _json.dumps({"item_id": "d1_0000", "label": "NON_ENTITY", "pass": 1,
                     "ts": "2026-08-14T00:00:00+00:00"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "DATA", data)
    monkeypatch.setattr(cli, "LABELS", labels)
    monkeypatch.setattr(cli, "_read_answer", lambda d, i: ("DEFER", None, 1.0))
    monkeypatch.setattr(cli, "_show_d1", lambda i: None)

    # The documented IAA flow: the κ subset (--pass-2) done by a DIFFERENT person.
    cli.annotate("d1", "Sabbir", pass2=True, subset=1, items_tag="", second="Alice")

    out = labels / "d1_labels_Alice.jsonl"
    assert out.exists(), "a second annotator writes under their own name"
    row = _json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["annotator"] == "Alice", "provenance is the person who produced the row"
    assert row["pass"] == 1, "a different person's labels are their own pass 1"
    assert row["machine_assisted"] is False


def test_a_decisive_artefact_names_the_secondaries_that_were_not_computed():
    """The rule block promises six secondaries; the arms fix alone left a reader
    unable to distinguish 'never measured' from 'measured and lost'."""
    from graft.graphbuild.gate1 import run_gate1

    gold = [{"action": "NON_ENTITY", "entity_id": None}]
    artefact = run_gate1({"proposed": gold, "E1": gold}, gold, smoke=False)
    omitted = artefact["secondaries_omitted"]
    assert {"proposer_recall_g5", "d2_macro_f1", "calibration_brier_ece"} <= set(omitted)
    assert all(isinstance(reason, str) and reason for reason in omitted.values())
