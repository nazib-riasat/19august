"""Formal validity ``H`` — nine deterministic sub-checks.

``H`` is called **formal validity**, never "valid proof". A set can satisfy every
schema, id, interval and scope constraint and still be semantically insufficient
to answer the question. Sufficiency is a separate, graded quantity and it lives
in ``U`` (v1.2 §4.1). The write-up must keep the two words apart.

**Prohibition, enforced by construction (v1.2 §4.4).** This module imports no
model of any kind, and there is no code path by which a learned score reaches
``H``. Entailment, sufficiency, authority and answerability are routed to ``U``,
the answerability gate and Stage B respectively — the routing table is
implemented as module boundaries and asserted by an import test. Note the
distinction that makes sub-check 7 legal: *computing* entailment is learned;
*reading the flag it already produced* is a stored-field lookup, structurally
identical to sub-check 4.

**Structure, and why it is this shape.** The sub-checks split into two families:

*per-atom* — type, retired, support, scope, and the supersession half of
temporal — depend on one atom and the snapshot, never on what else is selected.
These do the graph traversal, so they are the expensive ones, and they are
exactly what :class:`~graft.core.incremental.IncrementalChecker` memoises.

*set-level* — size, closure, identity, binding, and the binding half of temporal
— depend on the set as a whole and are pure bookkeeping over at most
``max_atoms`` entries.

Both the batch and the incremental path assemble their result through
:func:`assemble`, in ``CHECKS`` order. Agreement between them is therefore a
property of the code rather than of a test — the test confirms it rather than
carrying it.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from graft.config import Config
from graft.core import resolve
from graft.graphstore import GraphSnapshot
from graft.ledger import Ledger
from graft.schemas import AtomPool, CandidateAtom, CheckResult, Obligations, ProofSet, Violation

__all__ = [
    "H",
    "CHECKS",
    "PER_ATOM_CHECKS",
    "SET_LEVEL_CHECKS",
    "per_atom_violations",
    "set_level_violations",
    "assemble",
    "atom_ids",
]

CHECK_SIZE = "size"
CHECK_TYPE = "type"
CHECK_CLOSURE = "closure"
CHECK_IDENTITY = "identity"
CHECK_BINDING = "binding"
CHECK_RETIRED = "retired"
CHECK_SUPPORT = "support"
CHECK_TEMPORAL = "temporal"
CHECK_SCOPE = "scope"

#: The nine sub-checks, in evaluation order (cheapest first).  Named constants so
#: that Phase 2's audits and Phase 4's H-filter can tally failure categories
#: without string matching.  The order is fixed rather than conditional, so a
#: full audit always reports violations in the same sequence.
CHECKS: tuple[str, ...] = (
    CHECK_SIZE,
    CHECK_TYPE,
    CHECK_CLOSURE,
    CHECK_IDENTITY,
    CHECK_BINDING,
    CHECK_RETIRED,
    CHECK_SUPPORT,
    CHECK_TEMPORAL,
    CHECK_SCOPE,
)

PER_ATOM_CHECKS: tuple[str, ...] = (
    CHECK_TYPE,
    CHECK_RETIRED,
    CHECK_SUPPORT,
    CHECK_TEMPORAL,
    CHECK_SCOPE,
)

SET_LEVEL_CHECKS: tuple[str, ...] = (
    CHECK_SIZE,
    CHECK_CLOSURE,
    CHECK_IDENTITY,
    CHECK_BINDING,
    CHECK_TEMPORAL,
)


def atom_ids(X: ProofSet | Iterable[str]) -> tuple[str, ...]:
    """Sorted, **deduplicated** ids, so violations come out in a stable order.

    Dedup matters for bare iterables: the state is a *set* (plan v1.2 §3.4), so
    ``H([a, a])`` must judge the same object as ``H([a])``.  Before 13 Aug 2026
    a duplicated id in a raw list produced a spurious *identity* violation
    ("atoms X and X carry identical content") — fail-closed, so never unsound,
    but a false failure category that would pollute a Phase-2/4 tally.
    ``ProofSet`` (frozenset) and ``IncrementalChecker`` (double-add raises)
    were never exposed.
    """
    atoms = X.atoms if isinstance(X, ProofSet) else X
    return tuple(sorted(set(atoms)))


# --------------------------------------------------------------------------
# per-atom sub-checks
# --------------------------------------------------------------------------


def per_atom_violations(
    atom_id: str,
    pool: AtomPool,
    q: Obligations,
    G: GraphSnapshot,
) -> dict[str, tuple[Violation, ...]]:
    """Every violation that depends on this atom alone.

    Memoisable: nothing here reads the rest of the selected set, so the answer is
    the same at every state the atom appears in.
    """
    out: dict[str, list[Violation]] = {name: [] for name in PER_ATOM_CHECKS}
    atom = pool.get(atom_id)

    # 1 — type legality and target resolution.
    if atom is None:
        out[CHECK_TYPE].append(
            Violation(CHECK_TYPE, f"atom {atom_id} is not in this pool", (atom_id,))
        )
        return {k: tuple(v) for k, v in out.items()}
    if not resolve.target_resolves(atom, G):
        out[CHECK_TYPE].append(
            Violation(
                CHECK_TYPE,
                f"atom {atom_id} ({atom.kind}) denotes {atom.target!r}, which this "
                "snapshot does not hold as that kind of object",
                (atom_id,),
            )
        )

    # 4 — retired evidence.  Nothing is deleted (Zep's precedent), so an
    # invalidated edge is still addressable, which is exactly why a proof has to
    # be stopped from using it.
    edge = resolve.target_edge(atom, G)
    if edge is not None and not G.is_live(edge.edge_id):
        out[CHECK_RETIRED].append(
            Violation(
                CHECK_RETIRED,
                f"atom {atom_id} denotes edge {edge.edge_id}, invalidated at "
                f"{edge.t_invalid}",
                (atom_id,),
            )
        )

    # 7 — support eligibility (architecture fix F9).  Reads a flag Phase 5 wrote.
    # A node of an assertion-backed type with no traceable assertion is a
    # violation rather than a skip: "no evidence against it" must not read as
    # "supported".
    aids, problems = resolve.assertion_dependencies(atom, G)
    for problem in problems:
        out[CHECK_SUPPORT].append(
            Violation(CHECK_SUPPORT, f"atom {atom_id}: {problem}", (atom_id,))
        )
    for assertion_id in aids:
        if not G.is_eligible(assertion_id):
            out[CHECK_SUPPORT].append(
                Violation(
                    CHECK_SUPPORT,
                    f"atom {atom_id} rests on assertion {assertion_id}, which the "
                    "support gate quarantined",
                    (atom_id,),
                )
            )

    # 3a — a supersession running backwards in transaction time is formally
    # wrong, and so is one pointing at an edge the snapshot does not hold.
    if edge is not None and edge.superseded_by is not None:
        successor = G.edge(edge.superseded_by)
        if successor is None:
            out[CHECK_TEMPORAL].append(
                Violation(
                    CHECK_TEMPORAL,
                    f"atom {atom_id} denotes edge {edge.edge_id}, superseded by "
                    f"{edge.superseded_by}, which this snapshot does not hold",
                    (atom_id,),
                )
            )
        elif successor.t_created < edge.t_created:
            out[CHECK_TEMPORAL].append(
                Violation(
                    CHECK_TEMPORAL,
                    f"edge {edge.edge_id} (created {edge.t_created}) is superseded by "
                    f"{successor.edge_id} (created {successor.t_created}); a "
                    "supersession cannot run backwards in transaction time",
                    (atom_id,),
                )
            )

    # 5 — conversational scope (Phase-1 gap G6).  Empty scope is unrestricted.
    # Three states, kept apart on purpose: an atom with no provenance at all (an
    # Entity, a Source, a TimeInterval) is scope-neutral; provenance resolving
    # outside the scope is a violation; and a *broken* provenance chain is also a
    # violation, because the atom was not shown to be in scope and the check
    # fails closed.
    if q.scope:
        allowed = set(q.scope)
        convs, issues = resolve.conv_ids(atom, G, pool)
        for issue in issues:
            out[CHECK_SCOPE].append(
                Violation(CHECK_SCOPE, f"atom {atom_id}: {issue}", (atom_id,))
            )
        outside = tuple(c for c in convs if c not in allowed)
        if outside:
            out[CHECK_SCOPE].append(
                Violation(
                    CHECK_SCOPE,
                    f"atom {atom_id} rests on evidence from {list(outside)}, outside "
                    f"the question's scope {sorted(allowed)}",
                    (atom_id,),
                )
            )

    return {k: tuple(v) for k, v in out.items()}


# --------------------------------------------------------------------------
# set-level sub-checks
# --------------------------------------------------------------------------


def set_level_violations(
    ids: tuple[str, ...],
    pool: AtomPool,
    q: Obligations,
    G: GraphSnapshot,
    cfg: Config,
) -> dict[str, tuple[Violation, ...]]:
    """Every violation that depends on the set as a whole."""
    out: dict[str, list[Violation]] = {name: [] for name in SET_LEVEL_CHECKS}
    selected = set(ids)

    # 6 — size.  The lower bound is Phase-1 gap G1: the empty set passes every
    # other check vacuously and would score U = 0.5 on a question with no time
    # constraint, beating the bottom third of the valid utility range with a
    # proof that cites nothing.  That is the exp(beta*0) = 1 error in a new
    # place — a null object landing mid-range because "nothing is wrong with it"
    # was confused with "it is good".
    if not ids:
        out[CHECK_SIZE].append(
            Violation(
                CHECK_SIZE,
                "the empty set is not a proof: it cites nothing, cannot be "
                "serialised, and would outscore genuinely weak evidence",
            )
        )
    elif len(ids) > cfg.max_atoms:
        out[CHECK_SIZE].append(
            Violation(
                CHECK_SIZE,
                f"{len(ids)} atoms exceeds max_atoms {cfg.max_atoms}",
                ids,
            )
        )

    # 8 — structural closure (architecture fix F10).  Enforced by the ADD masks
    # during policy construction, and checked here because S3 and S4 build sets
    # directly and bypass those masks entirely.
    for aid in ids:
        atom = pool.get(aid)
        if atom is None:
            continue
        missing = tuple(r for r in atom.refs if r not in selected)
        if missing:
            out[CHECK_CLOSURE].append(
                Violation(
                    CHECK_CLOSURE,
                    f"atom {aid} references {list(missing)}, which are not selected",
                    (aid, *missing),
                )
            )

    # 2 — identity.  frozenset already excludes literal duplicates, so the real
    # risk is a malformed pool in which two ids carry identical content, letting
    # a proof count one piece of evidence twice and look larger than it is.
    seen: dict[tuple[str, ...], str] = {}
    for aid in ids:
        atom = pool.get(aid)
        if atom is None:
            continue
        key = atom.content_key()
        if key in seen:
            out[CHECK_IDENTITY].append(
                Violation(
                    CHECK_IDENTITY,
                    f"atoms {seen[key]} and {aid} carry identical content",
                    (seen[key], aid),
                )
            )
        else:
            seen[key] = aid

    # 9 — binding consistency (Phase-1 gap G3).  Two atoms both binding `answer`
    # is an internally contradictory proof, and it is formally detectable, so it
    # belongs in H rather than being penalised in U.
    for slot, claimants in sorted(pool.binding_slots(ids).items()):
        if len(claimants) > 1:
            out[CHECK_BINDING].append(
                Violation(
                    CHECK_BINDING,
                    f"slot {slot!r} is claimed by {len(claimants)} bindings; a proof "
                    "cannot assert two different fillers for one slot",
                    claimants,
                )
            )

    # 3b — hard temporal contradiction only (Phase-1 gap G5).  A binding whose
    # evidence carries validity intervals *all disjoint* from the constraint is
    # formally wrong.  Evidence that is merely vague is not contradictory, and
    # grading vagueness is U's temporal_correctness.
    if q.time_constraint is not None:
        for aid in ids:
            atom = pool.get(aid)
            if atom is None or atom.kind != "binding":
                continue
            intervals = [
                iv
                for ref in atom.refs
                if ref in selected and pool.get(ref) is not None
                for iv in resolve.validity_intervals(pool[ref], G)
            ]
            if intervals and not any(iv.overlaps(q.time_constraint) for iv in intervals):
                out[CHECK_TEMPORAL].append(
                    Violation(
                        CHECK_TEMPORAL,
                        f"binding {aid} rests on evidence whose validity is disjoint "
                        f"from the requested interval "
                        f"[{q.time_constraint.start}, {q.time_constraint.end})",
                        (aid,),
                    )
                )

    return {k: tuple(v) for k, v in out.items()}


# --------------------------------------------------------------------------
# assembly — the single ordering both paths use
# --------------------------------------------------------------------------


def assemble(
    ids: tuple[str, ...],
    per_atom: Mapping[str, Mapping[str, tuple[Violation, ...]]],
    set_level: Mapping[str, tuple[Violation, ...]],
    first_failure_only: bool = False,
) -> CheckResult:
    """Compose one :class:`CheckResult` in ``CHECKS`` order.

    Both ``H`` and the incremental checker route through here, which is what
    makes them agree by construction rather than by coincidence.  Within the
    temporal check, set-level (binding) violations precede per-atom
    (supersession) ones; within every other check, atoms are visited in sorted-id
    order.
    """
    violations: list[Violation] = []
    for name in CHECKS:
        found: list[Violation] = []
        if name in SET_LEVEL_CHECKS:
            found.extend(set_level.get(name, ()))
        if name in PER_ATOM_CHECKS:
            for aid in ids:
                found.extend(per_atom.get(aid, {}).get(name, ()))
        if found:
            violations.extend(found)
            if first_failure_only:
                break
    if violations:
        return CheckResult(ok=False, violations=tuple(violations))
    return CheckResult(ok=True)


# --------------------------------------------------------------------------
# H
# --------------------------------------------------------------------------


def H(
    X: ProofSet | Iterable[str],
    q: Obligations,
    G: GraphSnapshot,
    pool: AtomPool,
    cfg: Config,
    *,
    ledger: Ledger | None = None,
    first_failure_only: bool = False,
) -> CheckResult:
    """Formal validity of a completed candidate set.

    **Spends exactly one ``terminal_check``** when a ledger is given (Phase-1 gap
    G9).  Metering happens here rather than in callers, because caller-side
    counting always drifts.

    ``ledger=None`` means "do not meter", which Phase 2's exhaustive enumeration
    needs — enumerating every terminal of a lattice would exhaust any per-query
    budget, and it is an offline audit rather than a query.  It is also a way for
    a search module to cheat, so the Phase-3/4 harness always passes one and a
    test asserts that every registered ``SearchModule`` does.
    """
    if ledger is not None:
        ledger.count("terminal_checks")

    ids = atom_ids(X)
    per_atom = {aid: per_atom_violations(aid, pool, q, G) for aid in ids}
    set_level = set_level_violations(ids, pool, q, G, cfg)
    return assemble(ids, per_atom, set_level, first_failure_only)
