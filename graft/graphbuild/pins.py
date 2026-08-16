"""Everything Phase 6 freezes, importable without an ML library.

Same shape and same reason as ``graft.ingest.pins``: this module is §6's decision
table made executable, and it carries the **Stage-B fingerprint** (exit criterion
16) that ``scripts/verify_handoff.py`` prints — so it must stay importable on a
bare interpreter, and every model wrapper below it imports lazily.

Values that belong to the *config tree* are absent here on purpose. Giving a
frozen value two homes is the failure mode `CLAUDE.md` §5 catalogues, and a run's
full identity is the triple ``(config_hash, ingestion_fingerprint,
stage_b_fingerprint)``.
"""

from __future__ import annotations

from typing import Any

from graft.canonical import digest_of
from graft.schemas import ENDPOINT_TABLE, NODE_TYPES

__all__ = [
    "EMBEDDER",
    "K_CANDIDATES",
    "S_PAIRS",
    "TRAINING",
    "COMMIT_FLOOR",
    "DATASETS",
    "LLM_BASELINE",
    "endpoint_table_hash",
    "stage_b_fingerprint",
    "frozen_values",
]

# --------------------------------------------------------------------------
# decision 2 — the embedder (G7)
# --------------------------------------------------------------------------

#: **[ANALYSIS]** the pin.  The architecture says "bge-small text embeddings" and
#: Stage C will reuse the same vectors — two embedders would make channel-fusion
#: scores incomparable — so it is pinned here once, by id *and* revision.
#: Loaded through ``transformers`` directly (the Phase-5 precedent: a second
#: library for the same weights is a dependency bought for a wrapper).
#:
#: bge-small's published retrieval numbers are the model card's and are **not**
#: independently verified here; nothing in this project rests on them.
EMBEDDER: dict[str, Any] = {
    "model_id": "BAAI/bge-small-en-v1.5",
    "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "dim": 384,
    "pooling": "cls",
    "normalize": True,
    "max_length": 512,
    "batch_size": 32,
}

# --------------------------------------------------------------------------
# decisions 4 and 5 — candidates and pairs
# --------------------------------------------------------------------------

#: One constant with one precedent, used for both jobs: fix F8 takes top-*s* = 10
#: from Mem0's retrieval-before-update pattern (**[EVIDENCE-adjacent]**, ECAI
#: 2025).  Two separate knobs would be two tunables with one justification.
K_CANDIDATES = 10
S_PAIRS = 10

# --------------------------------------------------------------------------
# decision 10 — the training budget, identical across every arm
# --------------------------------------------------------------------------

#: **The Phase-3 lesson, applied one stage later.**  Comparisons at unmatched
#: budget are uninterpretable, and Phase 3 found three control defects that all
#: ran in the direction flattering the proposed method.  So the budget is one
#: dict, shared by E1/E2/E3 and every Gate-1 arm, and parameter counts are
#: reported per arm beside the scores.
TRAINING: dict[str, Any] = {
    "epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "hidden": 128,
    "heads": 2,
    "dropout": 0.1,
    "early_stop_patience": 5,
    "early_stop_metric": "dev_loss",
    "seeds": (13, 42, 7),
}

#: Decision 9.  D3/D4 edges commit only above a dev-chosen confidence floor;
#: D1/D2 always commit their argmax **because they are decisions, not
#: annotations** — a D1 that declines to act leaves a mention unresolved forever,
#: which is not the same kind of abstention as declining to assert a relation.
COMMIT_FLOOR: dict[str, float] = {"d3_relation": 0.5, "d4_temporal": 0.5}

# --------------------------------------------------------------------------
# decision 7 — the external datasets (G6)
# --------------------------------------------------------------------------

#: Pinned by SHA at download, licences recorded, raw files gitignored — the
#: ``ingest.corpus`` pattern.  Licences were read from each repository on
#: 14 Aug 2026 and are reproduced as *facts about the source*, not as legal
#: advice: **DialogRE is non-commercial research only**, which is a real
#: constraint on what may be redistributed and is why the raw files stay out of
#: git.
DATASETS: dict[str, dict[str, Any]] = {
    "dialogre": {
        "decoder": "D3",
        "source": "github.com/nlpdata/dialogre (ACL 2020)",
        "licence": "non-commercial research use only (license.txt)",
        "metric": "F1_c (progressive, the dataset's own)",
        "files": {
            "train": ("dialogre/train.json", "4ddee624c9f241451c455a4c1eb1d7a97f916c8bec9cd10444c97c09f5902236"),
            "dev": ("dialogre/dev.json", "12f783d1cecd3646a542709898eb8b7af4d14bbfe676cd402dadebb870718f5c"),
            "test": ("dialogre/test.json", "198402e8a6d2f4c7d05e41a41231a97ec0381aa7f7fceb1b3a42e786336eab89"),
        },
    },
    "redocred": {
        "decoder": "D3",
        "source": "github.com/tonytan48/Re-DocRED (EMNLP 2022)",
        "licence": "MIT",
        "metric": "micro-F1",
        "note": "**use this, not DocRED** — DocRED's false negatives punish "
        "recall-oriented models by ~13 F1 (GATE0_CONTRACT.md item 1)",
        "files": {
            "train": ("redocred/train_revised.json", "c40a137a1c57f3e2daaf5ebf511eaa1518f9305ffaa3052d46271921a91826f2"),
            "dev": ("redocred/dev_revised.json", "051ee1d057204a5d08ef5502beacdadf191245b5eaf0e29ec2c607cf002c016f"),
            "test": ("redocred/test_revised.json", "ea673b7ef91e16510d7eb88863d3903e47318dc5653ef361266f27ae3b84723c"),
        },
    },
    "torque": {
        "decoder": "D4",
        "source": "github.com/qiangning/TORQUE-dataset (EMNLP 2020)",
        "licence": "Apache-2.0",
        "metric": "exact match + F1",
        "note": "annotation scheme is MATRES (ACL 2018)",
        "files": {
            "train": ("torque/train.json", "ad939c59ea81ad1efdc0d0d4b66f78737293bcf15d77ae00191247a33ca2923d"),
            "dev": ("torque/dev.json", "7a8dd84c984f28a5284bdfda57b447218e1269cd2eaf05b5e173394fc1522434"),
        },
    },
    # ----------------------------------------------------------------------
    # **Phase 8's dataset, registered here because `load_split` is the one
    # SHA-pinned reader** (Phase-8 decision 1; G9 names `loaders.load_split`
    # explicitly).  A second copy of that verify-then-parse logic in
    # `graft/gate/` would be the duplication the pin discipline exists to
    # prevent, so the entry lives here and the *files* live under
    # `data/phase8/raw/` — which is why every path below is read with
    # `root=loaders.MUSIQUE_ROOT` rather than the Phase-6 default.
    #
    # It is **not** in `frozen_values()` and therefore does not move
    # `stage_b_fingerprint`: `DATASETS` has never been part of Stage B's
    # configuration identity, and adding a Phase-8 corpus must not silently
    # re-identify Phase 6's runs.
    #
    # **MuSiQue-Full, not MuSiQue-Ans**, and the distinction is the decision:
    # Full ships the *unanswerable twin* of each question with a
    # **byte-identical question string**, so the gate cannot score
    # answerability from wording and is forced onto pool-side features
    # (`GRAFT_PHASE8_BUILD.md` decision 1, G2's leakage argument).  Ans is
    # answerable-only and is **Phase 9's** Stage-D training source; it is in
    # the same archive and is deliberately left unextracted here.
    #
    # Verified on download, 15 Aug 2026: 39,876 + 4,834 + 4,918 = **49,628**
    # rows, matching `DATASET_DECISION.md` §2's figure exactly; dev is
    # 2,417 answerable + 2,417 unanswerable; 2,407 of its 2,412 question-text
    # groups are exact contrast pairs (the other 5 are duplicated pairs), and
    # **every** exact pair is one answerable and one unanswerable.
    "musique_full": {
        "phase": 8,
        "source": "github.com/StonyBrookNLP/musique — MuSiQue (TACL 2022), musique_v1.0.zip",
        "licence": "CC BY 4.0 (repository LICENSE, read 15 Aug 2026)",
        "metric": "answerability AUROC/AURC (Phase 8); the dataset's own is answer + support F1",
        "note": "contrast pairs; `id` is the PAIR key and repeats across the twin — see loaders.musique_pairs",
        "files": {
            "train": ("musique/musique_full_v1.0_train.jsonl", "b1cd998f7e0e2838d6fda024e4ad1eb0e7fc3edefdadb0bd9b5b10b0907f2034"),
            "dev": ("musique/musique_full_v1.0_dev.jsonl", "8cab31d56a3a1c4ef491b205a8dab3f1ac9c66e472098c6cf1de4e20294f7a4a"),
            "test": ("musique/musique_full_v1.0_test.jsonl", "01d7b0f752bdc1fc98856d605560e021a89be903baeb9daff11adeb7fac61328"),
        },
    },
    # -- Phase 9, Stage D's two Tier-1 training corpora (decision 1) --------
    #
    # **[EVIDENCE]** 2WikiMultiHopQA, *Constructing A Multi-hop QA Dataset for
    # Comprehensive Evaluation of Reasoning Steps* (COLING 2020) — the only
    # adopted source whose annotation is simultaneously an evidence **set** and a
    # typed relational path, with evidence-F1 an official metric (human 78.81
    # against baseline 14.94, which is the headroom Stage D's claim needs).
    #
    # Licence **re-verified at download** against the primary source, as
    # `GRAFT_PHASE9_BUILD.md` G10 requires and `DATASET_DECISION.md` §8 warns is
    # necessary: that document's verification pass did not complete, so every
    # adoption in it is single-sourced.
    "2wiki": {
        "phase": 9,
        "source": "github.com/Alab-NII/2wikimultihop — data.zip (dropbox), fetched 16 Aug 2026",
        "licence": "Apache-2.0 (repository, re-verified at download 16 Aug 2026)",
        "metric": "answer F1 + evidence F1 (the dataset's own)",
        "note": (
            "context is ALWAYS 10 paragraphs, each pre-split into sentences; "
            "supporting_facts are [title, sentence_idx] pairs (sentence-level); "
            "evidences are (subject, property, object) triples. Measured on dev: "
            "10/10 paragraphs on 12,576/12,576 rows, every supporting-fact title "
            "resolves into context, and the first evidence subject is a context "
            "title on 10,346/12,576 (82.3%)."
        ),
        "files": {
            "train": ("2wiki/train.json", "b3fddb4d5bb42cd797919cad67616545be51b24740e0a7dabdae7bf76b8f7bfa"),
            "dev": ("2wiki/dev.json", "48b9bdc69654dc580fda5f935a48b88cb89f11887587310af60d406c8d0111a6"),
        },
    },
    # **[EVIDENCE]** MuSiQue (TACL 2022) — the *answerable* split, and the only
    # adopted corpus shipping the **obligations** element of Stage D's abstract
    # triple: its `question_decomposition` is a per-hop DAG, so 2Wiki's
    # obligations must be synthesised while MuSiQue's are read.
    #
    # A different split from Phase 8's `musique_full`, and deliberately so: the
    # gate needs `full`'s answerable/unanswerable contrast pairs, Stage D needs
    # `ans`'s gold supporting sets. Same archive, same licence, separate pins —
    # two experiments must not share one file entry.
    "musique_ans": {
        "phase": 9,
        "source": "github.com/StonyBrookNLP/musique — musique_v1.0.zip (already on disk)",
        "licence": "CC BY 4.0 (repository LICENSE, read 15 Aug 2026)",
        "metric": "answer F1 + support F1 (the dataset's own)",
        "note": (
            "20 paragraphs with `is_supporting` flags; `question_decomposition` "
            "gives per-hop (question, answer, paragraph_support_idx). Hop questions "
            "are '<subject> >> <relation>' and later hops reference '#k', which is "
            "what OBLIGATION_SYNTHESIS's entity_anchor rule parses."
        ),
        "files": {
            "train": ("musique_ans/musique_ans_v1.0_train.jsonl", "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a"),
            "dev": ("musique_ans/musique_ans_v1.0_dev.jsonl", "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"),
        },
    },
}

# --------------------------------------------------------------------------
# decision 12 — the LLM baseline's budget (G11)
# --------------------------------------------------------------------------

#: **[EVIDENCE]** the baseline is faithful to the deployed pattern rather than a
#: strawman: Mem0's update pipeline is LLM-prompted (ECAI 2025), which is exactly
#: what this competes with.
#:
#: ``max_usd`` is ``None`` until signed off, and ``llmlink`` refuses to spend
#: anything while it is — the same fail-closed shape as an unfrozen extractor.
#: A local-model stand-in is **not** an acceptable substitute (it would change
#: the comparison's meaning); the smoke path replays a recorded cache instead, so
#: no test spends money.
LLM_BASELINE: dict[str, Any] = {
    "model": "gpt-4o-mini",
    "provider": "openai",
    "max_usd": None,
    "max_calls": 2000,
    # Worst-case cost of ONE call, used by the budget gate BEFORE spending
    # (Phase 4's enforcement lesson, applied to dollars): a gate that only
    # checks money already spent authorises the final call that overshoots the
    # cap.  ~4k prompt + ~1k completion tokens at the recorded list price is
    # ~$0.0012; 0.002 is that with headroom, and refusing a hair early is the
    # right side to miss on.
    "est_max_usd_per_call": 0.002,
    "cache_dir": "artefacts/phase6/llmlink_cache",
}

# --------------------------------------------------------------------------
# the fingerprint (exit criterion 16)
# --------------------------------------------------------------------------


def endpoint_table_hash(length: int | None = 16) -> str:
    """Hash of the frozen endpoint table and node vocabulary (G3).

    In the fingerprint because a change to either changes *which graphs are
    constructible*, which is configuration identity in the strongest sense: two
    machines agreeing on the embedder and disagreeing here would build different
    graphs from the same log and never notice.
    """
    payload = {
        "node_types": list(NODE_TYPES),
        "endpoints": {k: [list(v[0]), list(v[1])] for k, v in sorted(ENDPOINT_TABLE.items())},
    }
    return digest_of(payload, length)


def frozen_values() -> dict[str, Any]:
    from graft.graphbuild.prompts import REGISTRY_SHA

    return {
        "embedder": EMBEDDER,
        "prompt_registry_sha": REGISTRY_SHA,
        "k_candidates": K_CANDIDATES,
        "s_pairs": S_PAIRS,
        "training": {k: (list(v) if isinstance(v, tuple) else v) for k, v in TRAINING.items()},
        "commit_floor": COMMIT_FLOOR,
        "endpoint_table": endpoint_table_hash(None),
    }


def stage_b_fingerprint(length: int | None = None) -> str:
    """Configuration identity for Stage B, printed by ``verify_handoff.py``.

    Binds the config, not the output — the same G11 distinction Phase 5 drew, and
    for the same reason: encoder training on two machines will not produce
    bit-identical weights, but it must produce them from an identical setup.
    """
    return digest_of(frozen_values(), length)
