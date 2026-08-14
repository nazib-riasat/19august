"""Embedder, encoders, decoders, loaders, the LLM baseline and the Gate-1 rule.

Exit criteria 6, 9, 10, 11, 12 and 16.  Encoder forward/backward needs torch and
is skipped without it (the Phase-3 pattern); everything else — the mapping
losses, the budget refusal, the McNemar arithmetic, the smoke suppression — runs
on a bare interpreter, because those are the parts whose *correctness* the
write-up depends on.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from graft.graphbuild import pins
from graft.graphbuild.gate1 import (
    GATE1_RULE,
    d1_report,
    end_to_end_correct,
    macro_f1,
    mcnemar,
    run_gate1,
)
from graft.graphbuild.llmlink import BudgetRefused, LlmLinker, PromptCache
from graft.graphbuild.loaders import (
    DIALOGRE_TO_GRAFT,
    REDOCRED_TO_GRAFT,
    dialogre_items,
    torque_items,
    mapping_report,
)

torch = pytest.importorskip("torch", reason="encoder tests need the ML environment")


# -- the Stage-B fingerprint (criterion 16) --------------------------------


def test_the_fingerprint_covers_the_endpoint_table():
    """It decides *which graphs are constructible*, so two machines agreeing on
    the embedder and disagreeing here would build different graphs from the same
    log and never notice."""
    base = pins.stage_b_fingerprint()
    assert pins.endpoint_table_hash(None) in json.dumps(pins.frozen_values())
    assert pins.stage_b_fingerprint() == base


def test_every_dataset_pins_a_full_digest_and_records_its_licence():
    """A 16-character prefix is not a pin, and an unrecorded licence is how a
    non-redistributable corpus ends up in a repo."""
    for name, spec in pins.DATASETS.items():
        assert spec["licence"], name
        for split, (rel, sha) in spec["files"].items():
            assert len(sha) == 64, f"{name}/{split} has a truncated digest"
            assert rel


def test_the_llm_budget_starts_unsigned():
    """Fail-closed, like an unfrozen extractor: decision 12 requires a dollar cap
    declared at sign-off, and until then nothing may be spent."""
    assert pins.LLM_BASELINE["max_usd"] is None


# -- G6: the mapping losses ------------------------------------------------


def test_the_mapping_report_names_dropped_classes_rather_than_counting_them():
    """"26 dropped" and "26 dropped including every conflict-bearing relation"
    are different findings, and only one of them is checkable."""
    items = [
        {"labels": ["per:alias"]},
        {"labels": ["per:spouse"]},
        {"labels": ["per:children"]},
    ]
    report = mapping_report(items, DIALOGRE_TO_GRAFT, "dialogre")
    assert report["mapped_classes"] == 1
    assert sorted(report["dropped_classes"]) == ["per:children", "per:spouse"]
    assert report["instance_coverage"] == pytest.approx(1 / 3)


def test_the_mapping_report_flags_merges():
    """A GRAFT target receiving several native classes is where a downstream
    macro-F1 silently stops being comparable to the dataset's published number."""
    items = [{"labels": ["per:age"]}, {"labels": ["per:title"]}]
    report = mapping_report(items, DIALOGRE_TO_GRAFT, "dialogre")
    assert "has_value" in report["merges"]
    assert set(report["merges"]["has_value"]) >= {"per:age", "per:title"}


def test_instance_type_relations_are_not_mapped_to_same_as():
    """``P31``/``P279`` are *not* identity.  Mapping them to ``same_as`` would
    assert a dog is identical to the concept "dog" — the merge error the whole
    commit validator exists around."""
    assert "P31" not in REDOCRED_TO_GRAFT
    assert "P279" not in REDOCRED_TO_GRAFT


def test_the_mapping_report_carries_the_contract_s_red_line():
    report = mapping_report([], DIALOGRE_TO_GRAFT, "dialogre")
    assert "never presented as native supervision" in report["reading"]


def test_dialogre_items_keep_the_native_multi_label_shape():
    """Collapsing to a single label would change the task the decoder is tested
    on, which is what "native label set" forbids."""
    raw = [[["speaker1: hi"], [{"x": "A", "y": "B", "r": ["per:alias", "per:friends"]}]]]
    items = dialogre_items(raw)
    assert items[0]["labels"] == ["per:alias", "per:friends"]


def test_torque_items_read_both_split_shapes():
    """**The two TORQUE splits do not ship the same JSON** and a train-only
    loader hides that until the day dev is loaded: train is a list of annotator
    HITs carrying a ``passages`` list, dev is a dict keyed by passage id whose
    values are passage records, whose ``question_answer_pairs`` are keyed by
    question text and whose ``events`` is a single mapping rather than a list.
    The old loader raised AttributeError on dev.  (Found 15 Aug 2026.)"""
    train_shaped = [
        {
            "passages": [
                {
                    "passage": "The market rallied.",
                    "events": [{"answer": {"spans": ["rallied"]}}],
                    "question_answer_pairs": [
                        {
                            "question": "What has already happened?",
                            "answer": {"spans": ["rallied"]},
                            "is_default_question": False,
                        }
                    ],
                }
            ]
        }
    ]
    dev_shaped = {
        "docid_X_sentid_1": {
            "passage": "The market rallied.",
            "events": {"answer": {"spans": ["rallied"]}},
            "question_answer_pairs": {
                "What has already happened?": {
                    "answer": {"spans": ["rallied"]},
                    "is_default_question": False,
                }
            },
        }
    }
    for name, raw in (("train", train_shaped), ("dev", dev_shaped)):
        items = torque_items(raw)
        assert len(items) == 1, name
        assert items[0]["question"] == "What has already happened?", name
        assert items[0]["answer_spans"] == ["rallied"], name
        assert items[0]["events"] == ["rallied"], name


# -- the LLM baseline (G11) -------------------------------------------------


def test_the_baseline_refuses_to_spend_without_a_signed_budget(tmp_path):
    linker = LlmLinker(
        {**pins.LLM_BASELINE, "max_usd": None},
        cache_path=tmp_path / "cache.json",
        replay_only=False,
    )
    with pytest.raises(BudgetRefused, match="no LLM budget"):
        linker._authorise()


def test_a_cache_miss_in_replay_mode_raises_rather_than_calling(tmp_path):
    """A test that accidentally exercised an unrecorded prompt would quietly cost
    money; this makes it fail loudly instead."""
    linker = LlmLinker(cache_path=tmp_path / "cache.json", replay_only=True)
    with pytest.raises(BudgetRefused, match="cache miss"):
        linker.link("Fitbit", "I use a Fitbit", [])


def test_a_recorded_cache_replays_without_spending(tmp_path):
    """The cache is the run's record: after the budget is spent the numbers stay
    reproducible by replay."""
    path = tmp_path / "cache.json"
    cache = PromptCache(path)
    linker = LlmLinker(cache_path=path, replay_only=True)
    from graft.graphbuild.llmlink import LINK_SYSTEM, LINK_USER

    user = LINK_USER.format(
        context="(none)", turn="I use a Fitbit", mention="Fitbit", candidates="- e1: Fitbit"
    )
    key = cache.key(LINK_SYSTEM, user, pins.LLM_BASELINE["model"])
    cache.put(key, {"reply": '{"action": "LINK_EXISTING", "entity_id": "e1"}'})
    cache.flush()

    linker = LlmLinker(cache_path=path, replay_only=True)
    got = linker.link("Fitbit", "I use a Fitbit", [{"entity_id": "e1", "name": "Fitbit"}])
    assert got == {"action": "LINK_EXISTING", "entity_id": "e1", "parse_ok": True, "cached": True}
    assert linker.spent_usd == 0.0


def test_an_unparseable_reply_scores_as_defer_not_as_a_crash():
    """It is a *baseline behaviour* — the failure mode that makes prompting
    fragile — and DEFER neither credits nor punishes it for the parse failure."""
    got = LlmLinker._parse("I think it's the Fitbit one")
    assert got["action"] == "DEFER" and got["parse_ok"] is False


def test_an_out_of_vocabulary_action_falls_back_to_defer():
    got = LlmLinker._parse('{"action": "MERGE_ALL", "entity_id": "e1"}')
    assert got["action"] == "DEFER"


# -- Gate 1's rule (G10) ----------------------------------------------------


def test_the_rule_is_predeclared_in_the_artefact():
    """Fix F12: a decision rule chosen after seeing results is not a decision
    rule, so it ships as data rather than as prose someone paraphrases later."""
    assert GATE1_RULE["predeclared"].startswith("2026-08-14")
    assert "McNemar" in GATE1_RULE["test"]
    assert GATE1_RULE["seeds"] == [13, 42, 7]
    assert "consolidation, not a weaker evaluation" in GATE1_RULE["stop_or_redesign"]


def test_the_primary_metric_punishes_a_right_action_with_a_wrong_entity():
    """A wrong merge corrupts every proof built on it afterwards, so it cannot
    score as correct."""
    gold = {"action": "LINK_EXISTING", "entity_id": "e1"}
    assert end_to_end_correct({"action": "LINK_EXISTING", "entity_id": "e1"}, gold)
    assert not end_to_end_correct({"action": "LINK_EXISTING", "entity_id": "e9"}, gold)


def test_linking_accuracy_is_conditional_on_the_model_choosing_to_link():
    """Conditioning on gold instead would hide the over-linking failure this
    number exists to expose."""
    gold = [
        {"action": "LINK_EXISTING", "entity_id": "e1"},
        {"action": "CREATE_NEW_ENTITY", "entity_id": None},
    ]
    over_linker = [
        {"action": "LINK_EXISTING", "entity_id": "e1"},
        {"action": "LINK_EXISTING", "entity_id": "e2"},
    ]
    report = d1_report(over_linker, gold)
    assert report["model_link_rate"] == 1.0
    assert report["linking_accuracy_at_1"] == 0.5
    assert report["end_to_end_score"] == 0.5


def test_macro_f1_gives_nan_to_an_unexercised_class():
    """An unexercised class must not flatter the average, and must be visibly
    absent rather than silently perfect."""
    got = macro_f1(["A", "A"], ["A", "A"], ["A", "B"])
    assert got["A.f1"] == 1.0
    assert math.isnan(got["B.f1"])
    assert got["macro_f1"] == 1.0


def test_mcnemar_uses_the_exact_binomial_not_chi_square():
    """Gate-1 discordant counts can sit below ~25, where chi-square is a p-value
    that looks like evidence and is not."""
    got = mcnemar([False] * 5, [True] * 5)
    assert got["discordant"] == 5
    assert got["p_value"] == pytest.approx(2 * (1 / 2**5))
    assert got["direction"] == "b better"


def test_mcnemar_with_no_discordant_pairs_is_p_one():
    got = mcnemar([True, False], [True, False])
    assert got["p_value"] == 1.0 and not got["significant"]


def test_a_smoke_run_has_no_comparison_to_quote():
    """G1's discipline made structural: the comparison is *suppressed*, not
    labelled, because a comparison under a stamp is quotable by anyone who reads
    past the stamp."""
    gold = [{"action": "NON_ENTITY", "entity_id": None}]
    artefact = run_gate1({"proposed": gold, "e1": gold}, gold, smoke=True)
    assert artefact["comparisons"] is None
    assert artefact["smoke"] is True
    assert "may be quoted as a result" in artefact["smoke_notice"]
    assert artefact["per_arm"]  # the plumbing IS checked


def test_a_decisive_run_requires_the_proposed_arm():
    gold = [{"action": "NON_ENTITY", "entity_id": None}]
    with pytest.raises(KeyError, match="proposed"):
        run_gate1({"e1": gold}, gold, smoke=False)


# -- encoders and decoders (need torch) ------------------------------------

META = (
    ["Claim", "Entity"],
    [("Claim", "about_entity", "Entity"), ("Entity", "rev_about_entity", "Claim")],
)


def _features(n_claim: int = 4, n_entity: int = 3, dim: int = 32):
    from graft.graphbuild.encoders import GraphFeatures

    return GraphFeatures(
        x={"Claim": torch.randn(n_claim, dim), "Entity": torch.randn(n_entity, dim)},
        edge_index={
            ("Claim", "about_entity", "Entity"): torch.tensor([[0, 1], [0, 1]]),
            ("Entity", "rev_about_entity", "Claim"): torch.tensor([[0, 1], [0, 1]]),
        },
    )


@pytest.mark.parametrize("arm", ["E1", "E2", "E3"])
def test_each_encoder_forwards_and_backwards(arm):
    """Exit criterion 6.  Backward as well as forward, because PyG's failure mode
    on Windows is a missing kernel at the first scatter, not an import error."""
    from graft.graphbuild.encoders import build_encoder, parameter_count

    encoder = build_encoder(arm, 32, META, hidden=16)
    out = encoder(_features())
    assert set(out) == {"Claim", "Entity"}
    sum(v.sum() for v in out.values()).backward()
    assert parameter_count(encoder) > 0


def test_an_encoder_handles_a_graph_with_no_edges():
    """The first turn of a conversation has no edges, and that is a real state
    rather than an error."""
    from graft.graphbuild.encoders import GraphFeatures, build_encoder

    empty = GraphFeatures(x={"Claim": torch.randn(1, 32), "Entity": torch.zeros(0, 32)}, edge_index={})
    out = build_encoder("E3", 32, META, hidden=16)(empty)
    assert out["Claim"].shape[0] == 1


def test_e2_refuses_rather_than_silently_becoming_e1(monkeypatch):
    """A comparison in which the HGT arm is quietly an MLP would report a number
    for an encoder that never ran."""
    from graft.graphbuild import encoders

    monkeypatch.setattr(encoders, "hgt_available", lambda: (False, "simulated failure"))
    with pytest.raises(RuntimeError, match="CompGCN"):
        encoders.build_encoder("E2", 32, META)


def test_d1_handles_the_empty_candidate_cold_start():
    """G4: the head must be trained with empty-candidate examples present, or it
    learns that linking is always possible."""
    from graft.graphbuild.decoders import D1Decoder

    d1 = D1Decoder(16, hidden=8)
    logits = d1(torch.randn(16), torch.zeros(0, 16))
    assert logits.shape == (3,)
    action, entity, _ = D1Decoder.decode(logits, [])
    assert action in ("CREATE_NEW_ENTITY", "NON_ENTITY", "DEFER") and entity is None


def test_d1_width_tracks_the_candidate_count():
    from graft.graphbuild.decoders import D1Decoder

    d1 = D1Decoder(16, hidden=8)
    assert d1(torch.randn(16), torch.randn(5, 16)).shape == (8,)


def test_temperature_scaling_improves_calibration_and_fits_on_one_split():
    """Exit criterion 10, and decision 9's prerequisite: an uncalibrated
    confidence floor is a number with no meaning."""
    from graft.graphbuild.decoders import (
        TemperatureScaler,
        expected_calibration_error,
    )

    torch.manual_seed(0)
    logits = torch.randn(300, 4) * 4
    targets = torch.randint(0, 4, (300,))
    before = expected_calibration_error(torch.softmax(logits, -1), targets)
    scaler = TemperatureScaler()
    scaler.fit(logits, targets)
    after = expected_calibration_error(torch.softmax(scaler(logits), -1), targets)
    assert after < before


def test_class_weights_never_resample():
    """Decision 6 / contract item 6: resampling would make the quoted class
    balance unreproducible, and D2's rare classes are the contribution."""
    from graft.graphbuild.decoders import class_weights

    weights = class_weights([0] * 90 + [1] * 10, 2)
    assert weights[1] > weights[0]
    assert float(weights[1]) == pytest.approx(5.0)


def test_an_absent_class_gets_weight_one_not_infinity():
    from graft.graphbuild.decoders import class_weights

    weights = class_weights([0, 0], 3)
    assert all(math.isfinite(float(w)) for w in weights)


# -- the feature builder (decision 15, added 14 Aug) --------------------------


def _committed_graph(tmp_path):
    """A small real graph: entity, two linked claims, one valid_during."""
    from graft.eventlog import EventLog
    from graft.graphbuild.commit import Committer, claim_node, entity_node
    from graft.schemas import Assertion, AssertionFlags, SourceSpan, Turn

    handle = EventLog.open(tmp_path / "g.jsonl")
    handle.append("span.add", SourceSpan("sp1", "t1", 0, 5).to_dict())
    for aid, score in (("a1", 0.95), ("a2", 0.85)):
        handle.append(
            "assertion.add",
            Assertion(
                aid, "claim", f"fact {aid}", ("sp1",),
                AssertionFlags(asserted_by="user", entailed_by_span=True,
                               entailed_score=score),
                "2023-01-01T00:00:00+00:00", eligibility="eligible",
            ).to_dict(),
        )
    handle.append(
        "turn.add",
        Turn("t1", "c1", "s1", "user", "2023-01-01T00:00:00+00:00", "hello").to_dict(),
    )
    committer = Committer(handle)
    committer.create_entity("a1", "claim", "Fitbit", "c1", "2023-01-01T00:00:00+00:00", ["sp1"])
    entity = entity_node("Fitbit", "c1").node_id
    committer.link_existing("a2", "claim", entity, "2023-06-01T00:00:00+00:00", ["sp1"])
    committer.valid_during(
        claim_node("a1").node_id, 100.0, 200.0, "2023-06-01T00:00:00+00:00", ["sp1"]
    )
    return handle


def test_the_two_feature_variants_have_the_declared_dimensions(tmp_path):
    """base = bias + log-degree + RTE; graft = base + provenance flags + the
    pinned embedding.  The dims are the contract E1/E2 vs E3 build against."""
    from graft.graphbuild.embed import StubEmbedder
    from graft.graphbuild.features import BASE_DIM, GRAFT_DIM, build_features
    from graft.graphstore import ReplayGraphStore

    handle = _committed_graph(tmp_path)
    snap = ReplayGraphStore(handle).at()

    base = build_features(snap, "base")
    assert base.dim == BASE_DIM
    graft_f = build_features(snap, "graft", embed=StubEmbedder(384).embed)
    assert graft_f.dim == GRAFT_DIM
    assert set(base.x) == set(graft_f.x)

    # Provenance flags land only in the graft variant, on assertion-backed rows.
    claims = graft_f.x["Claim"]
    assert claims[:, BASE_DIM].max() == 1.0  # eligibility flag
    assert base.x["Claim"].shape[1] == BASE_DIM

    # Live edges appear forward and reversed (the encoder convention).
    assert any(k[1] == "about_entity" for k in graft_f.edge_index)
    assert any(k[1] == "rev_about_entity" for k in graft_f.edge_index)
    handle.close()


def test_e3_is_no_longer_parameter_identical_to_e2(tmp_path):
    """The audit measured E2 and E3 byte-identical (same class, same in_dim, no
    feature builder anywhere): the proposed encoder WAS the baseline.  With the
    declared variants they differ exactly where the hypothesis lives — the input
    features."""
    from graft.graphbuild.encoders import build_encoder, parameter_count
    from graft.graphbuild.features import BASE_DIM, GRAFT_DIM, encoder_metadata

    meta = encoder_metadata()
    e2 = build_encoder("E2", BASE_DIM, meta, hidden=16)
    e3 = build_encoder("E3", GRAFT_DIM, meta, hidden=16)
    assert parameter_count(e3) > parameter_count(e2)


def test_encoder_metadata_covers_the_endpoint_table_with_reverses():
    """Every schema relation appears as concrete (src, rel, dst) triples plus a
    rev_ counterpart — the convention the tests previously invented ad hoc."""
    from graft.graphbuild.features import encoder_metadata
    from graft.schemas import ENDPOINT_TABLE, NODE_TYPES

    node_types, relations = encoder_metadata()
    assert list(node_types) == list(NODE_TYPES)
    rels = {r[1] for r in relations}
    for etype in ENDPOINT_TABLE:
        assert etype in rels
        assert f"rev_{etype}" in rels
    # The widened supersedes rows are concrete triples now.
    assert ("Value", "supersedes", "Value") in set(relations)


def test_encoders_run_on_real_built_features(tmp_path):
    """End to end: committed graph -> features -> all three encoders forward and
    backward, each on its declared variant."""
    from graft.graphbuild.embed import StubEmbedder
    from graft.graphbuild.encoders import build_encoder
    from graft.graphbuild.features import (
        BASE_DIM,
        GRAFT_DIM,
        build_features,
        encoder_metadata,
    )
    from graft.graphstore import ReplayGraphStore

    handle = _committed_graph(tmp_path)
    snap = ReplayGraphStore(handle).at()
    meta = encoder_metadata()

    base = build_features(snap, "base")
    graft_f = build_features(snap, "graft", embed=StubEmbedder(384).embed)

    for arm, features, dim in (
        ("E1", base, BASE_DIM), ("E2", base, BASE_DIM), ("E3", graft_f, GRAFT_DIM),
    ):
        encoder = build_encoder(arm, dim, meta, hidden=16)
        out = encoder(features)
        assert "Claim" in out and "Entity" in out
        sum(v.sum() for v in out.values()).backward()
    handle.close()


def test_invalidated_edges_are_not_in_the_message_passing_structure(tmp_path):
    """An encoder aggregating over retired facts would reintroduce exactly what
    supersession removed."""
    from graft.graphbuild.commit import Committer, claim_node
    from graft.graphbuild.features import build_features
    from graft.graphstore import ReplayGraphStore

    handle = _committed_graph(tmp_path)
    committer = Committer(handle)
    old, new = claim_node("a1").node_id, claim_node("a2").node_id
    committer.supersede(old, new, "2023-07-01T00:00:00+00:00", ["sp1"])

    snap = ReplayGraphStore(handle).at()
    features = build_features(snap, "base")
    # a1's about_entity and valid_during edges were invalidated; only a2's link
    # and the supersedes history edge remain live.
    live_edges = sum(t.shape[1] for k, t in features.edge_index.items()
                     if not k[1].startswith("rev_"))
    assert live_edges == 2, f"expected link(a2) + supersedes, got {live_edges}"
    handle.close()


def test_the_stage_b_fingerprint_covers_the_prompt_registry():
    """A prompt edit must move a recorded hash (the llmlink docstring claimed
    registry membership that existed nowhere)."""
    from graft.graphbuild import pins
    from graft.graphbuild.prompts import REGISTRY_SHA

    assert pins.frozen_values()["prompt_registry_sha"] == REGISTRY_SHA


def test_the_budget_gate_refuses_the_call_that_would_overshoot(tmp_path):
    """Phase 4's enforcement lesson in dollars: checking only money already
    spent authorises the final call that lands past the cap."""
    from graft.graphbuild.llmlink import BudgetRefused, LlmLinker

    linker = LlmLinker(
        {"model": "m", "provider": "p", "max_usd": 1.0, "max_calls": 10,
         "est_max_usd_per_call": 0.002, "cache_dir": str(tmp_path)},
        cache_path=tmp_path / "cache.json",
        replay_only=False,
    )
    linker.spent_usd = 0.999
    with pytest.raises(BudgetRefused, match="would be exceeded"):
        linker._authorise()
    linker.spent_usd = 0.5
    linker._authorise()  # comfortably inside the cap: authorised
