"""P10.4 — the read-path orchestrator: fix F7, and the one place the system exists.

```
question ─► ObligationParser
         ─► Stage C: hybrid retrieval → pool (≤ 64 atoms)
         ─► Gate: answerable? ──no──► ABSTAIN (cause: gate)
            │yes
         ─► Stage D: sample K → H-filter → rank by the distilled head
            │        └─ none valid ──► ABSTAIN (cause: fallback, logged)
         ─► Stage E: serialise → frozen reader → answer + citations
```

Every stage up to now has been buildable and testable alone.  This module is the
first place where "the system" is a thing rather than a diagram, which makes it
also the first place three separate failures can hide in each other's noise —
hence the discipline below.

**Fix F7 is a resource constraint, and on this machine it binds.**  The
architecture assumes one 32 GB card; `CLAUDE.md` §7 records that the development
machine is an 8 GB RTX 5050, and run R1 measured the reader alone at **6.317 GB
peak**.  So models are context-managed — load, run the stage, free — and
:class:`~graft.reader.read.ModelSlot` *refuses* a second concurrent slot rather
than letting the run discover it as an out-of-memory error three stages in.

**The utility function is vacuous at inference, and this module must never rank
with it.**  Measured while building this file: ``sufficiency(X, ∅) = 1.0``, so
with no gold every set scores maximally sufficient and ``U`` collapses to its
four remaining terms.  That is exactly the circularity fix F1 exists to break —
*"the reward must be computable during training and some ranking signal must
exist at inference, where no gold exists"* — and the answer is the **distilled
utility head**, never ``env.utility``.  :func:`answer` therefore *refuses* to
rank when no head is supplied rather than falling back to a number that looks
like a utility and is not one.

**Two abstention routes, never summed.**  The gate saying no and Stage D failing
to find a valid set are different events.  ``OutputRecord.abstain_cause`` carries
which one fired, and `PHASE5_DECISIONS.md` §1's quarantine-cause lesson is the
precedent: two different reasons flattened into one rate inflate whichever rate
is being judged.

**Abstentions are excluded from utility means and counted separately** (§6
decision 7, inherited from `PHASE9_DECISIONS.md` §7.8).  Imputing zero makes an
abstaining system look like a wrong-answering one; dropping the query makes
abstention free.  :func:`aggregate` does neither.

**Nothing here is a result until it is stamped.**  A run whose Stage-D policy is
untrained, or whose gate threshold came from MuSiQue rather than conversation, is
a *wiring test*.  :class:`ReadPathStamp` collects those admissions and
:func:`answer` attaches one to every record, because `PHASE7_DECISIONS.md` §7
records a committed smoke artefact being quoted as measured and that is the
cheapest possible failure to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from graft.canonical import digest_of
from graft.config import Config
from graft.ledger import Ledger
from graft.reader.parse import answers_equivalent, parse_answer, resolve_citations
from graft.reader.pins import CONTESTED, PROMPT_SHA, stage_e_fingerprint
from graft.reader.serialize import ProofSerializer
from graft.schemas import GateDecision, Obligations, OutputRecord, ProofSet

__all__ = [
    "ReadPathStamp",
    "ReadResult",
    "answer",
    "aggregate",
    "UnrankableError",
]


class UnrankableError(RuntimeError):
    """Raised when Stage D returned valid sets and nothing can rank them.

    Deliberately loud.  The tempting fallback — score with ``env.utility`` — is
    *measurably* wrong at inference: with no gold, ``sufficiency`` is 1.0 for
    every set, so the ranking would be over four terms while claiming to be over
    six.  Fix F1 splits train-time ``U`` from the inference-time head precisely so
    this cannot happen silently, and a loud refusal is the only way that split
    survives a caller in a hurry.
    """


@dataclass(frozen=True)
class ReadPathStamp:
    """Every admission a run must carry, collected in one object.

    Exists because the alternative is remembering.  `PHASE7_DECISIONS.md` §7
    records a ``--smoke`` artefact quoted as a measured result, and
    `PHASE9_DECISIONS.md` §2.3 records a whole class of caveat that has to travel
    with a number to be readable at all.
    """

    policy_trained: bool
    gate_source: str
    scorer_source: str
    token_counter: str
    ordering: str
    notes: tuple[str, ...] = ()

    @property
    def is_wiring_test(self) -> bool:
        """True when *no* number from this run may be reported as a result."""
        return (
            not self.policy_trained
            or self.gate_source != "conversational"
            or self.scorer_source == "none"
            or self.token_counter != "reader_tokenizer"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_trained": self.policy_trained,
            "gate_source": self.gate_source,
            "scorer_source": self.scorer_source,
            "token_counter": self.token_counter,
            "ordering": self.ordering,
            "is_wiring_test": self.is_wiring_test,
            "notes": list(self.notes),
            "warning": (
                "WIRING TEST — at least one consumed artefact is untrained or a "
                "placeholder. No number from this run is a result."
                if self.is_wiring_test
                else "all consumed artefacts are trained and in-domain"
            ),
        }


@dataclass(frozen=True)
class ReadResult:
    """One query's full trace: the record, plus everything needed to audit it."""

    record: OutputRecord
    gate: GateDecision | None = None
    serialised: Any = None
    parsed: Any = None
    portfolio: Any = None
    contested: Mapping[str, Any] = field(default_factory=dict)
    stamp: Mapping[str, Any] = field(default_factory=dict)
    stages: Mapping[str, Any] = field(default_factory=dict)

    def report(self) -> dict[str, Any]:
        return {
            "outcome": self.record.outcome,
            "abstain_cause": self.record.abstain_cause,
            "citations": len(self.record.citations),
            "answer_text": self.record.answer_text,
            "gate": None if self.gate is None else self.gate.to_dict(),
            "packed": None if self.serialised is None else self.serialised.report(),
            "portfolio": None if self.portfolio is None else self.portfolio.report(),
            "contested": dict(self.contested),
            "stamp": dict(self.stamp),
            "stages": dict(self.stages),
            # **The snapshot travels to disk, and that is exit criterion 11.**
            # `report()` is the only thing `scripts/phase10_read.py` persists, so
            # omitting the snapshot here meant the artefact carried no latency and
            # no token line at all -- the cost claim unauditable despite the
            # ledger being wired. Found by adversarial audit, 16 Aug 2026.
            "ledger_snapshot": dict(self.record.ledger_snapshot),
        }


def answer(
    question: str,
    *,
    env: Any,
    featurizer: Any,
    scorer: Callable[[Iterable[str]], float] | None,
    read_fn: Callable[[str, str], str] | None,
    gate_decision: GateDecision | None = None,
    obligations: Obligations | None = None,
    atom_scores: Mapping[str, float] | None = None,
    aliases: Sequence[str] = (),
    rng: np.random.Generator | None = None,
    ledger: Ledger | None = None,
    config: Config | None = None,
    stamp: ReadPathStamp | None = None,
    count_tokens: Callable[[str], int] | None = None,
    query_id: str | None = None,
    contested_check: bool = True,
) -> ReadResult:
    """``question → OutputRecord``, with per-stage ledger accounting (fix F7, G12).

    ``env``/``featurizer`` are Phase 9's, already built against the pool — the
    orchestrator does not construct them, because Stage D's environment needs a
    pool and building one here would put the Stage-C→D join in two places.

    ``gate_decision`` is passed in rather than computed.  ``gate.decide`` is a
    pure function of Stage-C outputs by Phase-8's own design (its G3: no model
    loading inside), and keeping the call outside means the orchestrator never
    holds a gate model and a reader at once — fix F7 satisfied by structure
    rather than by ordering.

    ``scorer=None`` is legal only if Stage D returns nothing valid.  If it returns
    valid sets and there is no scorer, :class:`UnrankableError` is raised: see the
    module docstring on why ``env.utility`` is not an acceptable substitute.
    """
    cfg = config or Config()
    rng = rng if rng is not None else np.random.default_rng(cfg.seeds[0])
    obligations = obligations if obligations is not None else env.example.obligations
    stamp = stamp or ReadPathStamp(
        policy_trained=False,
        gate_source="unknown",
        scorer_source="none" if scorer is None else "distilled_head",
        token_counter="approx_tokens" if count_tokens is None else "reader_tokenizer",
        ordering="u_shaped_inference_computable",
    )

    def _record(
        outcome: str,
        *,
        cause: str | None = None,
        text: str | None = None,
        citations: Sequence[str] = (),
        proofset: ProofSet | None = None,
    ) -> OutputRecord:
        return OutputRecord(
            outcome=outcome,
            citations=tuple(citations),
            answer_text=text,
            proofset=proofset,
            ledger_snapshot=(ledger.snapshot() if ledger is not None else {}),
            # The stage-E fingerprint and the prompt hash ride in the record's
            # own identity field, so "same frozen reader, same prompt, same
            # budget" (v1.2 §3.5) is checkable per record rather than per run.
            config_hash=digest_of(
                {
                    "config": cfg.to_dict(),
                    "stage_e": stage_e_fingerprint(),
                    "prompt_sha": PROMPT_SHA,
                }
            ),
            abstain_cause=cause,
        )

    stages: dict[str, Any] = {}

    # -- Gate ----------------------------------------------------------------
    if gate_decision is not None and not gate_decision.answerable:
        stages["gate"] = {"answerable": False}
        return ReadResult(
            record=_record("abstain", cause="gate"),
            gate=gate_decision,
            stamp=stamp.to_dict(),
            stages=stages,
        )
    if gate_decision is not None:
        stages["gate"] = {"answerable": True, "p": gate_decision.p_answerable}

    # -- Stage D: construct, filter, rank -----------------------------------
    from graft.setgen.portfolio import run_portfolio

    def _portfolio() -> Any:
        return run_portfolio(
            featurizer, env,
            scorer if scorer is not None else (lambda atoms: 0.0),
            rng, ledger=ledger, config=cfg,
        )

    if ledger is not None:
        with ledger.stage("stage_d"):
            portfolio = _portfolio()
    else:
        portfolio = _portfolio()
    stages["stage_d"] = portfolio.report()

    if portfolio.fallback or portfolio.best is None:
        # Fix F3's licensed reading: "no valid proof found under this pool,
        # policy, attempt count and budget" -- never "no proof exists". This is
        # the event Phase 8 reserved `abstain_fallback` for, and the counter has
        # been zero by design until exactly here.
        return ReadResult(
            record=_record("abstain", cause="fallback"),
            gate=gate_decision,
            portfolio=portfolio,
            stamp=stamp.to_dict(),
            stages=stages,
        )

    if scorer is None:
        raise UnrankableError(
            f"Stage D returned {portfolio.distinct_valid} valid sets and no scorer "
            "was supplied. `env.utility` is NOT an acceptable substitute at "
            "inference: sufficiency(X, no gold) = 1.0, so it would rank on four "
            "terms while claiming six (fix F1). Supply the distilled utility head."
        )

    # -- Stage E: serialise, read, parse ------------------------------------
    serializer = ProofSerializer(
        env.example.snapshot, env.example.pool, config=cfg,
        count_tokens=count_tokens,
        counter_name="reader_tokenizer" if count_tokens is not None else None,
    )
    scores = atom_scores if atom_scores is not None else env.example.atom_scores

    def _read(atoms: Sequence[str]) -> tuple[Any, Any]:
        packed = serializer.serialise(ProofSet(atoms=frozenset(atoms)), obligations, scores)
        if read_fn is None:
            raise UnrankableError(
                "no read_fn supplied; the read path cannot produce an answer "
                "without the frozen reader"
            )
        parsed = parse_answer(read_fn(packed.text, question))
        # `strict=False`: a hallucinated citation is a reader-ceiling FINDING, and
        # a batch run must record it rather than die on it. The count travels in
        # `unresolved`, so it lowers citation precision as it should.
        parsed = resolve_citations(
            parsed, packed.claim_map, env.example.snapshot, env.example.pool, strict=False
        )
        return packed, parsed

    if ledger is not None:
        with ledger.stage("stage_e"):
            packed, parsed = _read(portfolio.best)
    else:
        packed, parsed = _read(portfolio.best)
    stages["stage_e"] = packed.report()

    # -- G8: the contested comparison, costed separately --------------------
    contested: dict[str, Any] = {"eligible": False, "contested": False, "extra_reader_calls": 0}
    if contested_check and portfolio.distinct_valid >= 2:
        contested["eligible"] = True
        if ledger is not None:
            with ledger.stage("contested"):
                runner_up_packed, runner_up = _read(portfolio.sets[1])
        else:
            runner_up_packed, runner_up = _read(portfolio.sets[1])
        contested["extra_reader_calls"] = 1
        contested["runner_up_answer"] = runner_up.answer_text
        # Equivalent answers from different evidence is AGREEMENT, not conflict --
        # which is why this compares the ANSWERS under decision 9's rule rather
        # than comparing the two atom sets.
        agree = answers_equivalent(parsed.answer_text, runner_up.answer_text)
        contested["contested"] = not agree and not parsed.abstained and not runner_up.abstained

    if parsed.abstained:
        # The reader was given a valid set and declined it. That is neither the
        # gate's abstention nor Stage D's fallback -- it is the reader's, and
        # ABSTAIN_CAUSES has no member for it. Recorded as a fallback with the
        # distinction in `stages`, because inventing a third cause would change a
        # frozen vocabulary from inside the orchestrator.
        stages["reader_declined"] = True
        return ReadResult(
            record=_record("abstain", cause="fallback", proofset=ProofSet(atoms=frozenset(portfolio.best))),
            gate=gate_decision, serialised=packed, parsed=parsed, portfolio=portfolio,
            contested=contested, stamp=stamp.to_dict(), stages=stages,
        )

    outcome = "contested" if contested["contested"] else "answer"
    return ReadResult(
        record=_record(
            outcome,
            text=parsed.answer_text,
            citations=parsed.citations,
            proofset=ProofSet(atoms=frozenset(portfolio.best)),
        ),
        gate=gate_decision,
        serialised=packed,
        parsed=parsed,
        portfolio=portfolio,
        contested=contested,
        stamp=stamp.to_dict(),
        stages=stages,
    )


def aggregate(results: Sequence[ReadResult]) -> dict[str, Any]:
    """§6 decision 7's aggregation, and the two wrong answers it avoids.

    **Abstentions are excluded from the utility mean and counted separately.**
    Imputing 0 makes an abstaining system look like a wrong-answering one;
    dropping the query makes abstention free.  Both are reported instead: the
    mean over *answered* queries, and the abstention rate with its cause split.

    **The two causes are never summed.**  ``gate`` and ``fallback`` are different
    events with different owners — Phase 8's classifier and Phase 9's budget
    exhaustion — and `PHASE5_DECISIONS.md` §1 is the standing precedent for what
    flattening distinct causes does to a rate.
    """
    total = len(results)
    answered = [r for r in results if r.record.outcome in ("answer", "contested")]
    abstained = [r for r in results if r.record.outcome == "abstain"]
    by_cause: dict[str, int] = {}
    for r in abstained:
        cause = r.record.abstain_cause or "unknown"
        by_cause[cause] = by_cause.get(cause, 0) + 1

    sizes = [len(r.record.proofset.atoms) for r in answered if r.record.proofset]
    citations = [len(r.record.citations) for r in answered]
    unresolved = [
        len(getattr(r.parsed, "unresolved", ())) for r in answered if r.parsed is not None
    ]
    extra_calls = sum(int(r.contested.get("extra_reader_calls", 0)) for r in results)

    wiring = any(r.stamp.get("is_wiring_test") for r in results)
    return {
        "queries": total,
        "answered": len(answered),
        "contested": sum(1 for r in results if r.record.outcome == "contested"),
        "abstained": len(abstained),
        "abstention_rate": (len(abstained) / total) if total else None,
        # Split, never summed.
        "abstain_by_cause": dict(sorted(by_cause.items())),
        # Over ANSWERED queries only.
        "mean_proof_size": (sum(sizes) / len(sizes)) if sizes else None,
        "mean_citations": (sum(citations) / len(citations)) if citations else None,
        "unresolved_citations": sum(unresolved),
        "contested_extra_reader_calls": extra_calls,
        "aggregation_rule": (
            "abstentions excluded from means and counted separately; causes never "
            "summed (Phase-10 decision 7)"
        ),
        "is_wiring_test": wiring,
        "warning": (
            "WIRING TEST — at least one query consumed an untrained or placeholder "
            "artefact. No number here is a result."
            if wiring
            else "all consumed artefacts trained and in-domain"
        ),
    }
