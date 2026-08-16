"""P9.0 — everything Phase 9 freezes, importable without an ML library.

Same shape and same reason as ``graft.ingest.pins``, ``graft.graphbuild.pins``,
``graft.retrieve.pins`` and ``graft.gate.pins``: this module is §6's **signed**
decision table made executable, and it carries the **stage-D fingerprint** (exit
criterion 16) that ``scripts/verify_handoff.py`` prints — so it must stay
importable on a bare interpreter.  ``graft/setgen/__init__.py`` is empty by
design, so importing this does not drag in ``features.py``'s torch.

**§6 was signed 16 August 2026 with no Phase-9 code in existence.**  That is the
cleanest `GRAFT_PHASE2_BUILD.md` §6b position the project has had, and it is
load-bearing for exactly one value here: :data:`GATE3_RULE`, the Gate-3 (real)
decision rule that Phase 4 measured itself unable to set.  A rule chosen after
seeing a learner result is not a rule; this file is the evidence of the order.

**Values that belong elsewhere are absent here on purpose.**  ``K``,
``checker_budget``, ``pool_cap``, ``max_atoms``, ``beta``, ``r_fail``,
``u_weights`` and the seeds are the **config tree's** (`CLAUDE.md` §6) and reach
a run's identity through ``config_hash``.  Giving a frozen value two homes is the
failure `CLAUDE.md` §5 catalogues, and Phase 8's own audit found a fingerprint
that bound block *labels* while the features underneath them changed.  What this
module freezes is what Phase 9 *adds*, and it binds feature **names in vector
order** rather than block names for that reason.

Two values are deliberately **derived, not written here**, each with a named
source and both recorded in §6's signature block:

``beta`` / the ``r_fail`` gap
    Phase-3 step 6's calibration gate.  Eligible candidates are ``{4, 8}``; the
    gate has not run, so :func:`training_blocked_reason` refuses a training run
    rather than letting one proceed at the placeholder default.

``N_real``
    Derived at build step 4 from the **slowest arm's measured** trajectories per
    second, against :data:`BUDGET`'s ceiling and ladder.  A rate is a property of
    the machine and the architecture, not a learner result, so measuring it first
    is §6b-clean — the Phase-3 decision-4 argument, reused deliberately.
"""

from __future__ import annotations

from typing import Any

from graft.canonical import digest_of
from graft.graphbuild.pins import EMBEDDER

__all__ = [
    "EMBEDDER",
    "CORPORA",
    "SUBSET",
    "OBLIGATION_SYNTHESIS",
    "FEATURE_BLOCKS",
    "BLOCK_PRESENCE_FLAG",
    "VACUOUS_ON_WIKIPEDIA",
    "OBLIGATION_SIGNAL_ON_WIKIPEDIA",
    "CREDIT_CONVENTION",
    "COLLISION_INSTRUMENT",
    "DISTILL",
    "PORTFOLIO",
    "BUDGET",
    "GATE3_RULE",
    "STAGE_SPLIT",
    "ADAPTATION_LOSSES",
    "beta_frozen",
    "training_blocked_reason",
    "frozen_values",
    "stage_d_fingerprint",
]

# --------------------------------------------------------------------------
# decision 1 — the training corpora (G3, G10)
# --------------------------------------------------------------------------

#: Stage-A corpora, in the order the build wires them.  Dataset **files and
#: SHAs** live in ``graphbuild.pins.DATASETS`` — one SHA-verified reader
#: project-wide — so this table carries only what Phase 9 decides *about* them.
#:
#: ``obligations`` is the G3 distinction that decided the pair: MuSiQue-Ans is
#: the only corpus shipping the *obligations* element of the abstract triple (its
#: decomposition DAG), and 2Wiki's must be synthesised.  ``distractors`` records
#: whether the corpus ships a pool or one has to be built (G10's HoVer rule).
#:
#: **HotpotQA is absent and must stay absent.**  ``DATASET_DECISION.md`` §2
#: rejects it for training because it is the surface submodular and PCST were
#: tuned on; training Stage D there would compromise the independence of the
#: very baselines Gate 3 scores against.
CORPORA: dict[str, dict[str, Any]] = {
    "wiki2": {
        "dataset": "2wiki",
        "tier": 1,
        "obligations": "synthesised",
        "distractors": "shipped",
        # **AMENDED 16 Aug 2026, by measurement.** This read "supporting
        # sentences over the shipped distractor context" — G3's phrasing, and
        # `supporting_facts` really are `[title, sentence_idx]` pairs. Measured
        # over 40 dev rows at `pool_cap = 64`, sentence-level documents put the
        # closed pool *on* the cap (34–64) and dropped gold on 30% of questions,
        # against paragraph-level's 29–30 and 40/40. `PHASE9_DECISIONS.md` §2.1
        # carries the table. The plan's sentence-level intent is honoured as the
        # *iff* clause below, and this is what the implementation does.
        "gold": (
            "paragraph documents; a paragraph is gold iff it contains a supporting "
            "sentence (granularity decided by measurement, PHASE9_DECISIONS.md §2.1)"
        ),
    },
    "musique_ans": {
        "dataset": "musique_ans",
        "tier": 1,
        "obligations": "native (decomposition DAG)",
        "distractors": "shipped",
        "gold": "is_supporting paragraph flags",
    },
    "hover": {
        "dataset": "hover",
        "tier": 1,
        "gated": True,
        "obligations": "synthesised",
        "distractors": "constructed — see COLLISION_INSTRUMENT's sibling note below",
        "gold": "claim evidence set",
    },
}

#: HoVer ships evidence but **no per-claim distractor context**, so its pools
#: need a rule.  Declared here with its loss stated rather than improvised in the
#: adapter: distractors are drawn from *other claims'* gold evidence within HoVer,
#: seeded, up to ``pool_cap``.  In-corpus, so no new retrieval stack enters (the
#: architecture's boring-stack rule).
#:
#: **[ANALYSIS] the loss:** these distractors are topically looser than a
#: retriever's would be, which makes HoVer pools *easier* to discriminate than
#: 2Wiki's or MuSiQue's.  A HoVer row in the Gate-3 table therefore may not be
#: pooled with the other two without saying so.
HOVER_DISTRACTOR_RULE = "in_corpus_gold_of_other_claims_seeded"

# --------------------------------------------------------------------------
# decision 2 — subset and stratification (SIGNED 16 Aug 2026)
# --------------------------------------------------------------------------

#: **[ANALYSIS]** sized from a measured anchor, not from a guess: Phase 8
#: embedded MuSiQue-Full **cold** in 1,146 s (``artefacts/phase8_gate.json``; its
#: ``embed_cache`` was empty, so nothing was reused), giving ~300 texts/s on the
#: dev GPU.  At ~75 atoms per question this subset's one-time pool prep is
#: ~17 min against ~13 h for the full corpora.
#:
#: **Cheap prep is the decision, not a side effect.**  Phases 7 and 8 both had to
#: regenerate artefacts after audit; a subset that costs a coffee break to
#: rebuild keeps that affordable, and an expensive one silently discourages the
#: regeneration that honesty requires.
#:
#: ``ladder`` is declared **now** so a later increase is a rule rather than a
#: reaction.  It is stepped only on a *learning-curve* reading — arms still
#: improving at the budget ceiling — and **never** on a Gate-3 result.
SUBSET: dict[str, Any] = {
    "train_per_corpus": 2000,
    "dev_per_corpus": 500,
    "ladder": (2000, 5000, 10000),
    "seed": 20260816,
    "balance": "1:1 across corpora, so neither dominates the Gate-3 table",
    "split_level": "question",
    "stratify": {
        # 2Wiki's four published question types, and MuSiQue's hop count.  Drawn
        # proportionally: the first N rows of either file are not a sample, and a
        # subset that is accidentally all-2-hop would make minimality untestable
        # — the closure rule and `max_atoms` only bite on large proof sets.
        "wiki2": ("comparison", "inference", "compositional", "bridge_comparison"),
        "musique_ans": ("2hop", "3hop", "4hop"),
        "hover": ("2hop", "3hop", "4hop"),
    },
}

# --------------------------------------------------------------------------
# decision 5 — obligation synthesis (G5)
# --------------------------------------------------------------------------

#: The frozen obligation parser is conversational GPU work (architecture fix F2);
#: a Wikipedia question never passes through it.  These rules are what makes
#: ``U``'s ``coverage`` term computable on these corpora at all.
#:
#: **The loss is declared, not hidden:** synthesised obligations are *weaker*
#: than parsed ones.  They exist so ``coverage`` has something deterministic to
#: measure and so G2's entity-anchor flags are computable.  **No claim is made
#: that they match the frozen parser's distribution**, and per-slot fill rates
#: ship in every artefact (exit criterion 8) so a slot that is empty in practice
#: cannot masquerade as a satisfied one.
OBLIGATION_SYNTHESIS: dict[str, dict[str, str]] = {
    "musique_ans": {
        "entity_anchor": "subject entity string of decomposition hop 1",
        "aggregate": "from question form",
        "time_constraint": "absent — not expressible on this corpus",
        "scope": "(question_id,)",
    },
    "wiki2": {
        # **AMENDED 16 Aug 2026, by measurement.** The rule read "subject of the
        # first evidence triple" until an adversarial audit identified it as a
        # gold leak: `evidences` is 2Wiki's own annotation, scored by its
        # evidence-F1 metric, absent at inference. Measured on the full dev
        # split, that subject matched a GOLD title on 92.5% of rows and a
        # distractor on 9.3% — a near-exact gold-paragraph indicator reaching
        # every arm's `state_repr` through d(s), L7/L7b's Δd block, and the
        # distilled head. The replacement is question-derived and measured at
        # 100% coverage. `PHASE9_DECISIONS.md` §1.3 carries the measurement.
        "entity_anchor": "longest context title appearing in the question (question-derived, inference-computable)",
        "aggregate": "absent",
        "time_constraint": "absent",
        "scope": "(question_id,)",
    },
    "hover": {
        "entity_anchor": "first named entity of the claim",
        "aggregate": "absent",
        "time_constraint": "absent",
        "scope": "(question_id,)",
    },
}

# --------------------------------------------------------------------------
# decision 4 — the featurizer blocks (G2's table, verbatim and ordered)
# --------------------------------------------------------------------------

#: Block names **in vector order**.  Frozen because an arm is a column mask over
#: this list, not a re-featurisation: if the order moved, a mask computed against
#: one run would silently select different columns in the next.
#:
#: The *actual* contract is :func:`frozen_values`'s ``feature_names`` — every
#: feature name in vector order, per block.  Phase 8's audit found a fingerprint
#: that bound five block strings while the features underneath them changed from
#: normalised to raw, producing **the same fingerprint for two different
#: experiments**.  That failure is not repeated here.
FEATURE_BLOCKS: tuple[str, ...] = (
    "text_embedding",
    "channel_scores",
    "stage_b_embedding",
    "obligation_match",
    "deficit",
)

#: Every block ships a presence flag (G2).  Stage-B encoder embeddings **do not
#: exist** on a Wikipedia pool — a 2Wiki paragraph never passed through D1–D4
#: commit — and zeroing them silently would let the policy learn "block absent ⇒
#: corpus X", a dataset classifier wearing a policy's name.  With the flag,
#: absence is an input.
BLOCK_PRESENCE_FLAG = "{block}_present"

#: ``H`` binds only **five of its nine** sub-checks on a Wikipedia pool (G4).
#: These three are vacuous *by construction* and are declared rather than
#: discovered: there are no ``valid_during`` intervals, no binding atoms, and
#: nothing retired.  The binding five — size, closure, identity/duplicates,
#: support-eligibility, scope — are real on the snapshots P9.1 builds.
#:
#: **Consequence, stated once so a reader of the Gate-3 table has it:** with
#: coverage and sufficiency living in ``U`` rather than ``H``, most non-empty
#: subsets of an eligible Wikipedia pool are ``H``-valid and ``STOP`` is rarely
#: masked — so reward discrimination there comes almost entirely from ``U``.
#: That is the intended design, not a defect, but "H-valid" on this track is a
#: weaker statement than "H-valid" on the conversational one and must never be
#: written up as the same claim.
VACUOUS_ON_WIKIPEDIA: tuple[str, ...] = ("temporal", "binding", "retired")

#: **The same story one level down, and it lands on Contribution 3** — added
#: 16 Aug 2026 after an adversarial audit measured it.  ``VACUOUS_ON_WIKIPEDIA``
#: above declares which of ``H``'s sub-checks cannot bind here; this declares how
#: much of ``d(s)`` can *move* here, which is what L7 and L7b actually learn from.
#:
#: Measured on real dev pools (8 examples/corpus, uniform-policy walks, after the
#: MuSiQue anchor fix): **exactly one of the six deficit components ever varies**
#: — ``anchor``.  ``value``, ``time``, ``source``, ``binding`` and ``closure`` are
#: constant on every reachable state, because a Wikipedia question synthesises no
#: value type, no time constraint and no source requirement (see
#: :data:`OBLIGATION_SYNTHESIS`), and Wikipedia pools carry no binding atoms.
#:
#: ==============  =====================  ==============================
#: corpus          legal ADDs with Δd≠0   states with no informative ADD
#: ==============  =====================  ==============================
#: 2Wiki           5.04%                  7.2%
#: MuSiQue-Ans     1.73%                  39.5%
#: ==============  =====================  ==============================
#:
#: **Why this is declared and not repaired.**  L7's mechanism is "prefer the ADD
#: that discharges more obligation"; where nothing is discharged, L7 and L6 see
#: the same state and C3's mechanism has nothing to act on.  Enriching the
#: synthesised obligations until ``Δd`` moved would be **reward engineering in
#: the proposed method's favour** — `CLAUDE.md` §5's catalogued failure — so the
#: number is reported instead and travels with every Stage-A C3 row.
#:
#: **The reading a null result licenses, written down before the run.**  A
#: Stage-A finding of "L7 ≈ L6" is consistent with *both* "checker-conditioning
#: does not help" and "this corpus offered almost nothing to condition on", and
#: this table cannot separate them.  It is the same shape as
#: `PHASE3_DECISIONS.md` §2.3's warning about an ``N`` that under-serves the
#: slowest arm: a negative C3 verdict that is really about the setup.  C3's
#: confirmatory test remains Gate 2 (:data:`STAGE_SPLIT`), where the synthetic
#: instances exercise all six components by construction — which is now a second,
#: independent reason that split exists.
OBLIGATION_SIGNAL_ON_WIKIPEDIA: dict[str, Any] = {
    "components_that_vary": ("anchor",),
    "components_constant": ("value", "time", "source", "binding", "closure"),
    "measured": {
        "2wiki": {"legal_adds_with_nonzero_delta_d": 0.0504, "states_with_no_informative_add": 0.072},
        "musique_ans": {"legal_adds_with_nonzero_delta_d": 0.0173, "states_with_no_informative_add": 0.395},
    },
    "reading": (
        "a Stage-A L7-vs-L6 null cannot distinguish 'checker-conditioning does not "
        "help' from 'this corpus offered almost nothing to condition on'. C3's "
        "confirmatory test is Gate 2, not this phase."
    ),
}

# --------------------------------------------------------------------------
# decision 1 / G10 — the multi-proof credit convention
# --------------------------------------------------------------------------

#: Adopted **unconditionally**, before any corpus that needs it is wired
#: (``DATASET_DECISION.md`` §2's FEVER/FEVEROUS row).
#:
#: Wherever multiple complete gold sets exist, a prediction counts as sufficient
#: iff **at least one complete gold set is a subset of the predicted set**.  On
#: 2Wiki and MuSiQue-Ans — single gold set apiece — this reduces to the ordinary
#: subset-of-gold-covered rule, so adopting it now costs nothing.  It is declared
#: now so the conversational track, where Tier-B gold is *a* minimal proof rather
#: than *the* one (`PHASE7_DECISIONS.md` §3.2c), inherits a convention instead of
#: improvising one under result pressure.
CREDIT_CONVENTION = "at_least_one_complete_gold_set_is_a_subset"

# --------------------------------------------------------------------------
# decision 12 — the equivalent-action collision instrument (G11)
# --------------------------------------------------------------------------

#: **AMENDED 16 Aug 2026 by an adversarial audit — see `PHASE9_DECISIONS.md`
#: §1.4, which amends signed decision 12.**  Two things were wrong here.
#:
#: *The measured quantity was structurally zero.*  ``content_key`` is
#: ``(kind, target, label, *refs)`` and ``target`` is a node id minted from the
#: assertion id, itself minted from text and spans — so two atoms sharing a
#: content key already share an ``atom_id`` and ``AtomPool`` collapsed them.
#: "Content-key collisions" is therefore a **proof, not a measurement**, and G11
#: is discharged by reporting it as one (measured: 0, as predicted).
#:
#: *The paper does not license the analogue.*  The quantity that *can* vary is
#: **distinct atoms carrying identical normalised text** — measured 3.86% of
#: 2Wiki dev rows.  It was cited as what Cor 5.1 corrects.  It is not:
#: **[EVIDENCE]** Symmetry-Aware GFlowNets (ICML 2025) Cor 5.1 scales a terminal
#: by ``|Aut(G)|``, the **graph automorphism group** under a node-by-node
#: generation scheme — orbit-equivalent *construction actions* on one graph.
#: Here the state is a set of atom **ids**: two identical-text paragraphs have
#: different ids, different spans and different provenance, so they are genuinely
#: different terminals of an id-defined MDP, not one object reached twice.
#: Applying a scalar would be an **ad-hoc correction wearing a citation**, which
#: is `CLAUDE.md` §5's catalogued failure.
#:
#: *And the bias it would "fix" is real but not ours to fix here.*  Measured by
#: enumerating a real pool's 20 valid singleton terminals: a duplicated content
#: class carries exactly **2.0×** the reward mass of an equivalent de-duplicated
#: one.  That is arithmetic of the corpus, not of the sampler — recorded so a
#: later reader cannot conclude "apply it to be safe" from the absence of a note.
#: Provenance-preservation is Contribution 1's claim; collapsing two provenances
#: would contradict it.
COLLISION_INSTRUMENT: dict[str, Any] = {
    "measure": "content_key collisions (structurally zero by id derivation -- reported as the proof G11 asks for, never as a measurement)",
    "diagnostic": "equivalent_evidence_atoms — distinct atoms with identical normalised text; a DATASET property, reported per corpus",
    "correction": "none — Cor 5.1 corrects graph automorphisms, not corpus duplicate text (PHASE9_DECISIONS.md §1.4)",
    "apply_when": "never on this MDP; the id-defined state space has no automorphism group to quotient",
}

# --------------------------------------------------------------------------
# decision 8 — the distilled utility head (G8, fix F13)
# --------------------------------------------------------------------------

#: Fix F13's inversion made concrete.  On the synthetic lattice the scorer was
#: **exact** ``U``, which is why Phase 4 found greedy globally optimal on 30/30
#: instances and could not ask Gate 3's question at all.  On real data gold is
#: absent at inference, so ranking runs on a regression head — and Robust
#: Scheduling with GFlowNets (ICLR 2023), the single best argument for using a
#: flow method here, requires precisely that: a **cheap, imperfect** proxy with an
#: expensive true evaluator.
#:
#: ``pooling`` is mean ⊕ max because a set has no order (plan §3.4's canonical
#: state rule); a sequence model here would impose one.
#:
#: **Two structural guarantees, both tested rather than intended:** the head never
#: appears in ``H``'s import graph (v1.2 §4.4 — the multiplicative gate property
#: dies otherwise, `CLAUDE.md` §4.2), and it never reads a gold field.  Exact
#: ``U`` values reach it as training **targets**, never as input features.
#:
#: ``report_rho`` is not optional decoration: the head's noise **is** the
#: experimental condition Gate 3 is asked under, so a Gate-3 row without its
#: held-out Spearman ρ beside it cannot be read.
DISTILL: dict[str, Any] = {
    "pooling": ("mean", "max"),
    "hidden": 64,
    "layers": 2,
    "dropout": 0.1,
    "max_params": 200_000,
    "target": "exact train-time U(X) at visited terminals",
    "report_rho": "held-out Spearman vs exact U, beside every Gate-3 row",
    "early_stop": True,  # this is not a compared arm; the arms may not early-stop
}

# --------------------------------------------------------------------------
# decision 9 — portfolio inference (G9, fix F4)
# --------------------------------------------------------------------------

#: ``K`` and ``checker_budget`` are **referenced, never duplicated** — they are
#: ``Config``'s and reach identity through ``config_hash`` (`CLAUDE.md` §6 makes
#: ``K`` = 8 simultaneously the portfolio size and the search comparison's
#: "returned sets" count: change both or neither).
#:
#: ``1 greedy + K−1 sampled`` is fix F4's shape.  Ranking is by the distilled
#: head; ties break to the **smaller** set, which is where minimality enters the
#: inference path rather than the reward.
#:
#: ``fallback`` closes the loop Phase 8 reserved: budget exhaustion or a dead end
#: on every attempt reaches ``FAIL`` and increments ``abstain_fallback``, so plan
#: §4.2's *two* abstention routes are countable from the first end-to-end run
#: instead of being retrofitted.
#: **AMENDED 16 Aug 2026, at the build** (`PHASE9_DECISIONS.md` §1.1). ``filter``
#: named an ``h_filter`` pass after deduplication — decision 9 as signed, and
#: wrong. ``search.base.h_filter`` is the direct-builder path only: this
#: portfolio constructs through the masks, where ``stop_allowed`` **is** ``H``,
#: so routing it through the filter would charge 8 terminal checks it does not
#: owe against a budget of 32 and collapse the 0-vs-1 check-family split Gate 3's
#: budget row reports. The error ran *against* the proposed method. The pin now
#: says what the code does, so the fingerprint stops certifying a design that was
#: never built.
PORTFOLIO: dict[str, Any] = {
    "greedy": 1,
    "sampled": "K - 1",
    "filter": "none — valid by construction through the masks (PHASE9_DECISIONS.md §1.1)",
    "rank": "distilled utility head",
    "tie_break": "smaller set",
    "fallback_counter": "abstain_fallback",
    "budget_enforcement": "would_exceed before spending, never observed after",
}

#: fix F4's ``contested`` flag, split into what is computable where (the Phase-8
#: G7 transfer pattern).  On a gold-bearing split the binding of a set *is*
#: computable — the atoms whose text contains the gold answer or an alias — so
#: Phase 9 reports a contested **rate** as an evaluation diagnostic.
#:
#: At deployment there is no gold.  The architecture's own words are "costs one
#: comparison", a reader-level check, and that belongs to the orchestrator.
#: **Transferred to Phase 10 by name.**  Inventing a gold-free proxy here would
#: be unstated machinery, which is the one thing G9 refuses to do.
CONTESTED: dict[str, str] = {
    "here": "gold-alias binding diagnostic, evaluation splits only",
    "phase_10": "inference-time reader comparison over top valid sets",
}

# --------------------------------------------------------------------------
# decision 11 — the training budget (SIGNED 16 Aug 2026)
# --------------------------------------------------------------------------

#: **The ceiling and the ladder are frozen; ``N_real`` is derived.**
#:
#: ``N_real`` = the largest rung the **slowest arm** completes within ``ceiling_s``
#: at its *measured* rate, taken at build step 4 before any scored run.  It is
#: then **identical across all 27 runs** (9 arms × 3 seeds) — fix F12's
#: fixed-budget primary, and exit criterion 10a asserts it.
#:
#: **Why the slowest arm and not the mean.**  Phase 3 sized ``N`` on L5
#: (~2,700 traj/s measured) and spent it on the LED arms (~900 traj/s), buying L7
#: 2.9 h against a 1 h ceiling — and the rung's own guard could not see it,
#: because that guard read only ``beta_sweep`` and ``sanity_check``, which run L4
#: and L5 exclusively.  The same ~3× gap will exist here.
#:
#: **[ANALYSIS] the 50,000 midpoint, projected not measured.**  Phase 9's profile
#: (``pool_cap`` 64, ``max_atoms`` 16, ~400-dim atoms) is ~50–150× costlier per
#: trajectory than the synthetic one (32, 8, ~16-dim), putting the slowest arm at
#: ~6–18 traj/s → 50,000 ≈ 1.4 h/run, ~37 h serial.  ``DATASET_DECISION.md`` §4
#: independently estimated "~33 h, worst case ~7 d" for this row, which brackets
#: the same ladder.  The projection is **not** what gets written into a run: the
#: measurement at step 4 is.
#:
#: ``checkpoints`` are monitoring only.  **No arm may early-stop** — selection at
#: a fixed budget is fix F12's discipline, and early-stopping one arm would break
#: the identical-budget comparison that Gate 3 rests on.  Exact TV does not exist
#: here (the state space is not enumerable), so checkpoint monitoring is training
#: loss plus a small dev best-of-K probe, recorded as *monitoring, never
#: selection*.
#: **``n_real`` was DERIVED on 16 August 2026 and the projection above was wrong
#: — in the safe direction, which is the only reason it cost nothing.**
#:
#: Measured on real 2Wiki + MuSiQue-Ans pools (30–60 atoms, the true distribution)
#: by ``scripts/phase9_measure.py``, artefact ``artefacts/phase9_measure.json``:
#:
#: ===================  ==========  ==================================
#: slowest arm          30.06/s     ``l7b_aux`` — a proposed-method arm
#: fastest arm          513.32/s    ``l2_imitation`` (no rollouts)
#: spread               17.08x      which is *why* the rule reads "slowest"
#: ===================  ==========  ==================================
#:
#: **Re-measured 16 Aug 2026 after the audit fixed two things.** The sample is
#: now stratified (the first reading head-sliced each dev split, which on
#: MuSiQue-Ans is 100% 2-hop against a true 51.8/31.4/16.7 mix), and each arm is
#: timed three times with the **slowest** taken, because single-run readings of
#: one arm ranged 29.67–36.16. Both corrections land within noise of the original
#: 29.67, so `N_real` never moved — which is the useful result: the derivation is
#: robust to the defect that was found in it.
#:
#: **Re-derived after the 16 Aug adversarial audit's Δd fix**, which made the
#: gradient-time ``Δd`` block dense over legal actions (it had been sparse and
#: sign-inverted).  That is a real ~28% throughput cost on L7/L7b and it is the
#: correct cost: a sparse ``Δd`` is not the feature Contribution 3 is about.
#:
#: The **[ANALYSIS]** projection written above at signing was 6–18 traj/s, from a
#: 50–150x cost ratio against Phase 3's synthetic profile.  The measurement is
#: **29.67**, so the real ratio is ~30x rather than 50–150x.  The projection is
#: left in place rather than quietly corrected, because the *point* of decision 11
#: is that ``N_real`` is derived and not guessed — and this is the evidence that
#: the guess would have been wrong by 2.3–7x had anyone frozen it.
#:
#: **All three rungs fit inside the 2 h ceiling** (0.46 h, 0.92 h, 1.85 h), so the
#: ladder never had to escalate and the largest rung is adopted outright.  The top
#: rung uses **92% of the ceiling** (6,653 s of 7,200) at the slowest of three
#: timed repeats.  The headroom is thin: any further per-trajectory cost drops the
#: adopted rung to 100,000, and that is a *derivation* changing rather than a
#: decision being revisited.  Two consequences worth
#: stating rather than leaving to be discovered:
#:
#: 1. The ceiling is **not** the binding constraint here, so it certifies nothing
#:    about convergence.  If 200,000 turns out to leave L7b under-converged, there
#:    is no rung above it and raising one is a §6b decision-rule amendment, not a
#:    tuning step.  `PHASE3_DECISIONS.md` §2.3 is the standing warning: an ``N``
#:    that under-serves the slowest arm hands Contribution 3 a negative verdict
#:    that is about budget.
#: 2. 200,000 x 27 runs at 30.06/s is **~49.9 h serial** (~6.2 h wall clock across
#:    this machine's 16 logical cores).  ``DATASET_DECISION.md`` §4's independent
#:    estimate for this row is "~33 h, worst case ~7 d", which still brackets it.
BUDGET: dict[str, Any] = {
    "ceiling_s": 7200,  # 2 h per run
    "ladder": (50_000, 100_000, 200_000),
    "n_real": 200_000,
    "measured_rate_slowest": 30.06,
    "measured_slowest_arm": "l7b_aux",
    # Kept a real arm name because a test asserts it is one, and that invariant is
    # worth more than the nuance -- which lives here instead: `l7b_aux` and
    # `l7_checker_led` are within noise of each other and alternate between runs.
    # `l7b_aux` is the arm that produced the pinned floor. Both are
    # proposed-method arms, which is the point the "slowest arm" rule turns on.
    "measured_slowest_arm_note": "l7b_aux and l7_checker_led alternate between runs; both are proposed-method arms",
    "ceiling_utilisation": 0.92,
    "measured_on": "artefacts/phase9_measure.json (16 Aug 2026, 24 stratified real pools, 29-60 atoms, post-audit). SLOWEST OF THREE RUNS - see rate_variance.",
    "rate_variance": "29.67-36.16 traj/s across runs. The script times each arm `reps` times and takes the slowest, because a single wall-clock sample is noise; the PINNED figure is the slowest across all observed runs, so it is a conservative FLOOR and not a copy of any one artefact row (the artefact records its own run). N_real = 200,000 at every point in the observed range - 1.54 h to 1.85 h against a 2 h ceiling - which is what makes the derivation robust to the noise rather than dependent on one draw. Chasing the last digit is what produced the pins-vs-artefact drift this note replaces.",
    "sized_from": "slowest arm's measured trajectories/second",
    "identical_across_arms": True,
    "early_stop_arms": False,
    "checkpoints": "monitoring only — training loss + dev best-of-K probe, never selection",
}

# --------------------------------------------------------------------------
# decision 7 — the Gate-3 (real) decision rule (G7) — VERBATIM
# --------------------------------------------------------------------------

#: **The rule, frozen before a single Phase-9 rollout existed.**
#:
#: `GRAFT_PHASE2_BUILD.md` §6b's second procedure requires that **no learner
#: results be inspected** before a decision rule is set.  When §6 was signed on
#: 16 August 2026 there was no Phase-9 code at all — not a rollout, not a
#: fixture, not a throughput number — so this rule was fixed with nothing to have
#: peeked at.  Phase 4 could not say that of its own §6: its G9 measurement
#: predated its ruling, which is *why* Gate 3's decision moved here.
#:
#: **[EVIDENCE]** the framing is Robust Scheduling with GFlowNets (ICLR 2023):
#: diverse candidates sampled under a cheap proxy beat proxy-optimisation when
#: the true evaluator is expensive.  **[EVIDENCE]** the test-selection protocol is
#: Dror et al. (ACL 2018).  **[ANALYSIS]** the seed count and the predeclaration
#: are this project's own discipline, not prescribed by either paper.
#:
#: The ``consolidation`` clause is plan §7's Gate-3 text and `CLAUDE.md` §8's:
#: a training-free method matching a *learned* one at equal budget narrows Stage
#: D's claim, and the project consolidates on Contribution 1.  Written down
#: because a gate with no losing branch is not a gate.
#:
#: ``diversity_secondary`` adopts the **size-controlled** form Phase 4
#: recommended (`PHASE4_DECISIONS.md` §1.3): it measured the raw metric
#: size-confounded — S4-informed scored 0.483 against ``p*``'s own 0.4506 — and
#: deferred the rule change to "Phase 9's re-ask".  This is that re-ask, taken at
#: the §6b-clean moment rather than after a result made it interesting.
GATE3_RULE: dict[str, Any] = {
    "primary": "best-of-K valid-set utility at the fixed budget",
    "k": "Config.K",
    "budget": "Config.checker_budget terminal H checks per query, enforced via would_exceed",
    "scored_with": (
        "exact train-time U (gold is available on held-out dev); the distilled "
        "head SELECTS, the exact value SCORES — the gap between them is fix F13's "
        "caveat made measurable rather than argued"
    ),
    "arms": "every trained arm and S1-S5, same held-out pools, same distilled head",
    "seeds": "Config.seeds",
    "test": "paired bootstrap (gate2.paired_bootstrap), higher-is-better negation",
    "consolidation": (
        "if the best training-free arm's interval overlaps or beats the best "
        "learned arm's, Stage D's learning claim narrows and the project "
        "consolidates on Contribution 1"
    ),
    "diversity_secondary": (
        "size-controlled: observed - E[random portfolio at the same set sizes "
        "from the same pool], DIVERSITY_CONTROL_SEED as frozen in Phase 4"
    ),
}

# --------------------------------------------------------------------------
# decision 13 — the stage split (G12)
# --------------------------------------------------------------------------

#: What each stage may honestly claim.  Stage A's numbers are **real** — the
#: corpora are the declared training interface and the decision rule is
#: :data:`GATE3_RULE` — but they are *not* conversational-memory numbers, and the
#: artefact says so in its own body (exit criterion 17).
#:
#: **C3's verdict is not Phase 9's.**  The L7/L7b vs capacity-matched L6 and
#: GAFlowNet comparison is reported here as *supporting evidence*; its
#: confirmatory test remains Gate 2 on the enumerable environment, where exact TV
#: exists.  Writing Phase 9 up as the C3 verdict would be claiming a
#: distributional result from a phase that cannot compute one.
STAGE_SPLIT: dict[str, str] = {
    "A": "Wikipedia track (2Wiki + MuSiQue-Ans) — runnable once beta freezes",
    "B": "conversational track — deferred by name, needs scope-c ingestion",
    "c3_verdict": "Gate 2 on the enumerable environment, NOT this phase",
}

#: Every declared departure from a corpus's native semantics, reported in the
#: artefact (the Gate-0 item-1 adaptation discipline, and Phase 8's own pattern).
ADAPTATION_LOSSES: dict[str, str] = {
    "obligations": (
        "synthesised on 2Wiki/HoVer by OBLIGATION_SYNTHESIS; weaker than the "
        "frozen parser's and not claimed to match its distribution"
    ),
    "h_subset": (
        "three of H's nine sub-checks are vacuous on Wikipedia pools "
        "(VACUOUS_ON_WIKIPEDIA); 'H-valid' here is a weaker statement than on "
        "the conversational track"
    ),
    "obligation_signal": (
        "five of d(s)'s six components are constant on Wikipedia pools, so Delta-d "
        "is nonzero on 1.7-5.0% of legal ADDs (OBLIGATION_SIGNAL_ON_WIKIPEDIA). A "
        "Stage-A L7-vs-L6 null is therefore not a C3 verdict; Gate 2 is."
    ),
    "stage_b_features": (
        "no Stage-B encoder embedding exists on a Wikipedia pool; the block is "
        "zeroed with its presence flag cleared, never silently zeroed"
    ),
    "transfer": (
        "Wikipedia->conversation transfer is a DECLARED, UNTESTED claim "
        "(CLAUDE.md §7). Stage B is what turns it into a measurement."
    ),
}


# --------------------------------------------------------------------------
# the two derived values, and the refusals that keep them honest
# --------------------------------------------------------------------------

#: β candidates that survived Phase-2's target-mass bands.  ``{1, 2}`` are not
#: eligible; the argmin runs over the eligible ones only (`CLAUDE.md` §6).
BETA_ELIGIBLE: tuple[float, ...] = (4.0, 8.0)


def beta_frozen(config: Any = None, record: Any = None) -> tuple[bool, str | None]:
    """``(frozen, problem)`` — whether Phase-3 step 6 has actually frozen β.

    **Reads the record's contents, not merely its name** (found by adversarial
    audit, 16 Aug 2026). The first version returned ``record.exists()``, so an
    empty file, a ``--quick`` calibration, or a record whose adopted β disagreed
    with the config all read as "frozen" and would have let a scored run start at
    the placeholder 4.0.

    Three things are checked, and each corresponds to a way Phase 3 can produce a
    record that does not license a Stage-D run: the gate must have *adopted* a
    rung rather than exhausted its ladder; the adopted β must be one of the
    eligible candidates Phase 2 left standing; and it must equal the β this
    config would actually train at, or the run is at a different reward than the
    freeze certified.
    """
    import json
    from pathlib import Path

    # `record` is a parameter so this is testable without mocking `Path`, which
    # is how the first version of its test recursed into itself. A function that
    # can only be tested by monkey-patching the standard library is a function
    # with a missing parameter.
    if record is None:
        record = Path(__file__).resolve().parents[2] / "artefacts" / "phase3_calibration.json"
    record = Path(record)
    if not record.exists():
        return False, "artefacts/phase3_calibration.json does not exist"
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        return False, f"the calibration record is unreadable ({exc})"
    if not data.get("adopted"):
        return False, "the calibration record adopted no rung (Gate 2 inconclusive)"
    beta = data.get("adopted", {}).get("beta")
    if beta is None:
        return False, "the calibration record carries no adopted beta"
    if float(beta) not in {float(b) for b in BETA_ELIGIBLE}:
        return False, f"adopted beta {beta} is not one of the eligible {list(BETA_ELIGIBLE)}"
    if config is not None and float(config.beta) != float(beta):
        return False, (
            f"config.beta is {config.beta} but the calibration froze {beta}; a run "
            "would train at a different reward than the freeze certified"
        )
    return True, None


def training_blocked_reason(config: Any = None) -> str | None:
    """``None`` when a Phase-9 training run may start, else why it may not.

    The reward ``R = 1[H]·exp(β·U)`` is identical across arms **by construction**
    (`CLAUDE.md` §6), so training before β freezes would either waste every run
    or contaminate the freeze with results taken at a different β.  Decision 6
    makes that a refusal rather than a note, for the reason Phase 7 learned the
    hard way: a smoke run quoted as a measurement is a defect the artefact cannot
    self-report.

    **Building and fixture-testing Phase 9 is not blocked** — only scored runs
    are.  Steps 0-5 of the build order need no β at all.
    """
    frozen, problem = beta_frozen(config)
    if not frozen:
        return (
            f"beta is not frozen: {problem}. Config.beta's 4.0 is a placeholder and "
            f"the eligible candidates are {list(BETA_ELIGIBLE)}. Every arm shares one "
            "reward by construction, so a run started now is either wasted or "
            "contaminates the freeze. Fixture tests and build steps 0-5 are "
            "unaffected."
        )
    return None


def frozen_values() -> dict[str, Any]:
    """Everything that must agree across machines for two Stage-D numbers to compare.

    **Binds feature names in vector order**, not block labels — the Phase-8
    correction, adopted here from the start rather than after an audit finds two
    different experiments sharing one fingerprint.

    ``n_real`` is bound too: it is derived rather than written, but a run at
    50,000 trajectories and a run at 200,000 are **different experiments**, and a
    fingerprint that could not tell them apart would be exactly the failure this
    function exists to prevent.  ``None`` before step 4 is itself informative — it
    marks a fingerprint taken before the budget was known.

    ``beta``, ``K``, ``checker_budget``, ``pool_cap``, ``max_atoms``,
    ``u_weights``, ``r_fail`` and the seeds stay **absent**: they are the config
    tree's and reach identity through ``config_hash``.
    """
    # **Imported from `featurenames`, not from `atomfeat`** (found by adversarial
    # audit, 16 Aug 2026). `atomfeat` imports torch, so calling this function --
    # which `scripts/verify_handoff.py` does -- pulled torch into what this
    # module's own docstring promises is a bare-interpreter path. The module
    # imported clean and the *call* did not, which is the harder version of the
    # bug to notice. The names now live in a torch-free module that `atomfeat`
    # also imports, so there is still exactly one definition.
    from graft.setgen.featurenames import BLOCK_FEATURES

    return {
        "embedder": EMBEDDER,
        "corpora": {k: v for k, v in sorted(CORPORA.items())},
        "hover_distractor_rule": HOVER_DISTRACTOR_RULE,
        "subset": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in sorted(SUBSET.items())
            if k != "stratify"
        },
        "stratify": {k: list(v) for k, v in sorted(SUBSET["stratify"].items())},
        "obligation_synthesis": {k: dict(v) for k, v in sorted(OBLIGATION_SYNTHESIS.items())},
        "feature_blocks": list(FEATURE_BLOCKS),
        # the actual contract: every feature name, in vector order, per block
        "feature_names": {k: list(v) for k, v in sorted(BLOCK_FEATURES.items())},
        "block_presence_flag": BLOCK_PRESENCE_FLAG,
        "vacuous_on_wikipedia": list(VACUOUS_ON_WIKIPEDIA),
        "obligation_signal_on_wikipedia": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in OBLIGATION_SIGNAL_ON_WIKIPEDIA.items()
        },
        "credit_convention": CREDIT_CONVENTION,
        "collision_instrument": COLLISION_INSTRUMENT,
        "distill": {k: (list(v) if isinstance(v, tuple) else v) for k, v in DISTILL.items()},
        "portfolio": dict(PORTFOLIO),
        "contested": dict(CONTESTED),
        # **Only the budget's *identity* half** (16 Aug 2026 audit). Binding the
        # whole dict put `measured_rate_slowest`, `ceiling_utilisation` and
        # `measured_on` — all outputs of a run — inside a digest this function's
        # own docstring defines as "the config, not the output". The effect was
        # circular: step 4 measures a rate, the rate is written into pins, the
        # fingerprint moves, and the artefact step 4 just wrote now disagrees
        # with the fingerprint of the code that wrote it. That is exactly the
        # observed d6f7fe84/2f40c3f drift, and re-running could never converge.
        # `n_real` stays bound: a run at 50k and a run at 200k are different
        # experiments and the identity must separate them.
        "budget": {
            k: (list(BUDGET[k]) if isinstance(BUDGET[k], tuple) else BUDGET[k])
            for k in (
                "ceiling_s", "ladder", "n_real",
                "identical_across_arms", "early_stop_arms", "sized_from",
            )
        },
        "gate3_rule": dict(GATE3_RULE),
        "stage_split": dict(STAGE_SPLIT),
        "adaptation_losses": dict(ADAPTATION_LOSSES),
        "beta_eligible": list(BETA_ELIGIBLE),
    }


def stage_d_fingerprint(length: int | None = None) -> str:
    """Configuration identity for Stage D, printed by ``verify_handoff.py``.

    Binds the **config, not the output** — the same G11 distinction Phases 5, 6,
    7 and 8 drew.  Two machines will not produce bit-identical policy weights;
    they must produce them from an identical setup, and they must score with
    identical arithmetic, or two Gate-3 tables are not comparable.
    """
    return digest_of(frozen_values(), length)
