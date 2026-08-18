"""P10.1 — the proof serializer: a ``ProofSet`` becomes reader context.

```
ProofSet + snapshot ──► ProofSerializer ──► (text, claim_map, report)
                                              │        │        │
                                    the reader's    citation   ceiling 4
                                       prompt      resolution
```

**Three jobs, and the third is a measurement.**  Turn a set of atoms into
numbered, quoted, ordered text; hand back the id → atom map that makes a
citation resolvable; and report exactly what the token budget dropped, because
that is what ceiling 4 consumes (v1.2 §6.3: *"does a sufficient proof survive
the evidence/token budget and serialization?"*).

**Ordering is U-shaped, and its signals are inference-computable.**
**[EVIDENCE]** Lost in the Middle (TACL 2024) measured a U-shaped position
curve: models use evidence at the **edges** of a context far better than in the
middle.  So the strongest evidence goes first and last.

The architecture's own phrasing was *"anchor and **answer-binding** evidence at
the beginning and end"*, and answer-binding requires the answer.  This module
therefore ranks on ``pins.ORDERING["signals"]`` — obligation-anchor match (the
parser reads the question), fused retrieval score (Stage C's own output) and
atom kind (structural) — and on nothing else.  ``pins.ORDERING`` lists the
forbidden signals by name and ``test_reader.py`` asserts none of them is read.

**This is the third appearance of that class of error in this project**, which
is why it is guarded structurally rather than remembered:
`PHASE9_DECISIONS.md` §1.3 caught a gold annotation reaching every arm's
``state_repr`` through the 2Wiki anchor rule, and Phase 9's G9 caught it in fix
F4's contested flag.  Each time it read as innocuous prose in a plan.

**The gold ordering is implemented too, and only as a diagnostic.**  The gap
between the honest ordering and the gold one *is* the measurement of what the
honest ordering costs; reporting either alone is the failure decision 3 exists
to prevent.

**Token counting is a parameter, and the default is not good enough for a
reported number.**  Counting real tokens needs the reader's tokenizer, and
importing it here would drag transformers into a module that must stay cheap and
testable.  So the counter is injected: :func:`approx_tokens` is a declared
heuristic for fixtures, and :meth:`ProofSerializer.serialise` **records which
counter it used**.  A ceiling-4 number computed under the heuristic is marked as
such — the same discipline Phase 6 and 7 apply to the stub embedder, where the
decisive path refuses one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from graft.config import Config
from graft.core import resolve
from graft.reader.pins import CLAIM_ID_FORMAT, ORDERING
from graft.schemas import AtomPool, Obligations, ProofSet

__all__ = [
    "SerialisedProof",
    "ProofSerializer",
    "approx_tokens",
    "atom_quote",
    "normalise",
    "ordering_gap",
]

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Casefold and collapse whitespace — the anchor-match rule, one place.

    Deliberately the *same* rule ``graft.setgen.atomfeat`` uses for its
    obligation-match features, and it inherits the same known limitation, which
    `PHASE7_DECISIONS.md` §3.2 measured: a normalised-exact rule missed an anchor
    demonstrably present in the graph, and six further misses were a
    category-vs-instance distinction that no normalisation fixes.  One rule, one
    known weakness, one place to change it — rather than a second matcher here
    with a second unknown one.
    """
    return _WS.sub(" ", text.casefold()).strip()


def approx_tokens(text: str) -> int:
    """A declared heuristic: whitespace tokens × 1.3, rounded up.

    **[ANALYSIS]** and deliberately crude.  English subword tokenizers run
    roughly 1.3 tokens per whitespace token on prose; this is a stand-in for
    fixtures and for the shape of the budget logic, **not** a substitute for the
    reader's own tokenizer.  Anything reported as a ceiling-4 number must pass
    the real counter, and :class:`SerialisedProof` records which was used so the
    distinction survives into the artefact rather than living in a caller's
    memory.
    """
    words = len(text.split())
    return -(-words * 13 // 10)


def atom_quote(atom: Any, snapshot: Any, max_chars: int = 400) -> tuple[str, tuple[str, ...]]:
    """``(rendered text, span ids)`` for one atom, through the provenance chain.

    Uses :func:`graft.core.resolve.provenance_spans` — the walk ``H``'s support
    and scope sub-checks already make — rather than a second traversal.  A
    citation must resolve to the *same* span the checker would have validated,
    or citation precision measures a different object than formal validity does.

    **An edge atom renders as its relation, not as a re-quoted span** (found by
    the first smoke run, 16 Aug 2026).  Edge atoms carry their endpoint's
    provenance by construction — ``proofs.build_snapshot`` sources an
    ``about_entity`` edge from the very span that sources its claim, because *the
    paragraph is the warrant for both* — so quoting them the same way emitted the
    identical sentence under two different claim ids.  The reader would then see
    duplicated evidence, and citation precision would score a citation to the
    relation as though it were a citation to the claim.

    So a node atom is quoted from its span, and an edge atom is rendered as
    ``(etype → target)``: compact, truthful about what an edge asserts, and
    distinguishable from the claim it connects.  The span ids come back either
    way, so the citation still resolves.

    Falls back to the atom's own text when a span cannot be resolved, and returns
    the empty span tuple so the caller can see that it did.  Silently emitting a
    quote with no provenance would make an unciteable atom look citeable.
    """
    span_ids, _problems = resolve.provenance_spans(atom, snapshot)

    if atom.kind == "edge":
        edge = resolve.target_edge(atom, snapshot)
        if edge is not None:
            from graft.schemas import PAYLOAD_NAME

            dst = snapshot.node(edge.dst)
            name = ""
            if dst is not None:
                name = str(dst.payload.get(PAYLOAD_NAME, "") or "")
            if not name:
                from graft.setgen.proofs import atom_text

                name = atom_text(snapshot, atom) or edge.dst
            return f"({edge.etype} → {name})", tuple(span_ids)

    quoted = ""
    for span_id in span_ids:
        span = snapshot.span(span_id)
        if span is None:
            continue
        turn = snapshot.turn(span.turn_id)
        if turn is None:
            continue
        quoted = turn.text[span.start : span.end].strip()
        if quoted:
            break
    if not quoted:
        from graft.setgen.proofs import atom_text

        quoted = atom_text(snapshot, atom)
    if len(quoted) > max_chars:
        quoted = quoted[: max_chars - 1].rstrip() + "…"
    return quoted, tuple(span_ids)


@dataclass(frozen=True)
class SerialisedProof:
    """What the reader is shown, and what ceiling 4 measures.

    ``claim_map`` is ``claim_id -> atom_id``: the object a citation resolves
    through.  ``dropped`` is what the budget refused, which is the *whole* point
    of ceiling 4 — a proof that was sufficient before serialization and is not
    after has failed at packing, not at retrieval.
    """

    text: str
    claim_map: Mapping[str, str]
    included: tuple[str, ...]
    dropped: tuple[str, ...]
    tokens: int
    budget: int
    ordering: str
    counter: str
    positions: Mapping[str, int] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """Whether every atom of the set survived the budget — ceiling 4's bit."""
        return not self.dropped

    def report(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "budget": self.budget,
            "included": len(self.included),
            "dropped": len(self.dropped),
            "complete": self.complete,
            "ordering": self.ordering,
            "token_counter": self.counter,
            "approximate_tokens": self.counter != "reader_tokenizer",
        }


class ProofSerializer:
    """``ProofSet`` → reader context, at a declared budget and ordering."""

    def __init__(
        self,
        snapshot: Any,
        pool: AtomPool,
        *,
        config: Config | None = None,
        count_tokens: Callable[[str], int] | None = None,
        counter_name: str | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.pool = pool
        self.cfg = config or Config()
        self.count_tokens = count_tokens or approx_tokens
        self.counter_name = counter_name or (
            "reader_tokenizer" if count_tokens is not None else "approx_tokens"
        )

    # -- ordering ----------------------------------------------------------

    def _rank_key(
        self, atom_id: str, obligations: Obligations, scores: Mapping[str, float]
    ) -> tuple[float, float, str]:
        """The declared, inference-computable ranking key.

        Returns ``(anchor_hit, fused_score, atom_id)`` descending on the first
        two.  ``atom_id`` breaks ties so the order is **total** — two runs must
        serialise the same set identically, and a dict-ordered tie would make the
        claim ids unstable across launches.
        """
        atom = self.pool[atom_id]
        anchor = normalise(obligations.entity_anchor or "")
        hit = 0.0
        if anchor:
            from graft.setgen.proofs import atom_text

            hit = 1.0 if anchor in normalise(atom_text(self.snapshot, atom)) else 0.0
        return (hit, float(scores.get(atom_id, 0.0)), atom_id)

    def order(
        self,
        atoms: Sequence[str],
        obligations: Obligations,
        scores: Mapping[str, float],
        *,
        gold: frozenset[str] | None = None,
    ) -> tuple[list[str], str]:
        """U-shaped order, and the name of the ordering used.

        ``gold`` engages the **diagnostic** ordering and is the only path that
        may see it.  It exists so the honest ordering's cost is a measured
        number; a caller that passes it into a *reported* run is doing the thing
        decision 3 forbids, which is why the ordering name travels with the
        result and into the artefact.
        """
        if gold is not None:
            strong = sorted(a for a in atoms if a in gold)
            rest = sorted(a for a in atoms if a not in gold)
            name = "gold_diagnostic"
        else:
            ranked = sorted(
                atoms, key=lambda a: self._rank_key(a, obligations, scores), reverse=True
            )
            # Connective evidence is the middle: an edge atom *describes* a
            # relation between two node atoms, so it is exactly the "connective"
            # the architecture's ordering row names, and it is identifiable from
            # the atom's kind rather than from anything semantic.
            strong = [a for a in ranked if self.pool[a].kind != "edge"]
            rest = [a for a in ranked if self.pool[a].kind == "edge"]
            name = "u_shaped_inference_computable"

        # U-shape: strongest first, then the connective middle, then the next
        # strongest last. **[EVIDENCE]** Lost in the Middle (TACL 2024).
        head = strong[: max(1, (len(strong) + 1) // 2)] if strong else []
        tail = list(reversed(strong[len(head):]))
        return head + rest + tail, name

    # -- serialisation -----------------------------------------------------

    def serialise(
        self,
        proof: ProofSet | Sequence[str],
        obligations: Obligations,
        scores: Mapping[str, float] | None = None,
        *,
        budget: int | None = None,
        gold: frozenset[str] | None = None,
    ) -> SerialisedProof:
        """Serialise ``proof`` into numbered, quoted, budget-capped evidence.

        Atoms are dropped from the **end** of the ordered list when the budget
        binds, never from the middle — the U-shape's whole premise is that the
        edges matter most, so truncating the middle would discard the position
        the ordering just paid to fill.  What was dropped is reported rather than
        inferred from a length.
        """
        atom_ids = sorted(proof.atoms) if isinstance(proof, ProofSet) else sorted(proof)
        cap = int(self.cfg.serialization_budget_tokens if budget is None else budget)
        ordered, ordering_name = self.order(atom_ids, obligations, scores or {}, gold=gold)

        # -- ids first, over the WHOLE ordered list -------------------------
        # **Stable across budgets, and deliberately so.** Ceiling 4 is reported
        # at every rung of the ladder (decision 1), and if ids were renumbered
        # per budget then the same atom would be `[c3]` at 512 tokens and `[c2]`
        # at 160 — making the three tables incomparable atom-by-atom, which is
        # the one thing a ladder is for. Ids may therefore be non-contiguous
        # after truncation; a gap means "the budget dropped that one", which is
        # information rather than a defect.
        ids = {a: CLAIM_ID_FORMAT.format(index=i + 1) for i, a in enumerate(ordered)}

        # -- render, with edges naming their endpoints ----------------------
        # An edge atom asserts a *relation between two atoms*, so it renders as
        # one. Without the endpoint reference, two edges from different claims to
        # the same entity render identically — measured on the first smoke run,
        # where `[c3]` and `[c4]` were both `(about_entity → London)` and a
        # citation to either was indistinguishable to the reader.
        #
        # An edge whose source sits in the U's *tail* produces a **forward
        # reference** — `[c3] [c6] (about_entity → London)` appears before `[c6]`
        # does. That is accepted rather than fixed: removing it means ordering
        # every edge after every node, which discards the U-shape that Lost in
        # the Middle (TACL 2024) is the evidence for. The reference still
        # resolves, and the id map is what a citation is scored through.
        rendered: dict[str, str] = {}
        for atom_id in ordered:
            atom = self.pool[atom_id]
            quoted, _spans = atom_quote(atom, self.snapshot)
            if atom.kind == "edge" and atom.refs:
                src = ids.get(atom.refs[0])
                if src:
                    quoted = f"[{src}] {quoted}"
            rendered[atom_id] = f"[{ids[atom_id]}] {quoted}"

        # -- budget, then close under references ----------------------------
        kept: list[str] = []
        dropped_set: set[str] = set()
        used = 0
        for atom_id in ordered:
            cost = self.count_tokens(rendered[atom_id] + "\n")
            if used + cost > cap and kept:
                # **`break`, not `continue`** — corrected 16 Aug 2026 after an
                # adversarial audit measured this as *first-fit* rather than the
                # tail truncation this method has always documented. Continuing
                # let a cheaper atom further down the order slip in after a more
                # important one had been dropped: at budget 30 the kept indices
                # were [0,1,2,3,5], skipping 4.
                #
                # That is not a cosmetic ordering difference. The U-shape's entire
                # premise (Lost in the Middle, TACL 2024) is that the head and
                # tail positions carry the most weight, so first-fit can discard a
                # tail atom the ordering just paid to place and keep a weaker
                # middle one — spending the budget on exactly the positions the
                # evidence says matter least.
                #
                # `and kept`: a single atom larger than the whole budget is still
                # emitted, because an empty evidence block would make the reader
                # abstain for a reason that has nothing to do with the evidence.
                # The overflow is visible as `tokens` > `budget`.
                dropped_set.update(ordered[ordered.index(atom_id):])
                break
            kept.append(atom_id)
            used += cost

        # **Truncation must not break closure.** The budget is a token count and
        # knows nothing about references, so it can drop a node while keeping an
        # edge that points at it — leaving the reader a structurally broken proof
        # whose citation resolves to something not shown. Closure is what makes a
        # partial set sound (fix F10), and it has to survive serialization too.
        #
        # **The test is membership in `kept`, not in `dropped_set`** — corrected
        # 16 Aug 2026. Those differ whenever a reference points at an atom that
        # was never a candidate: `dropped_set` holds only what the budget refused,
        # so an edge whose endpoint was absent from the *input* set passed the
        # check and shipped a dangling citation. Stage D emits closed sets, so this
        # cannot arise on the live path — but the serializer is also handed
        # hand-built and gold sets, and a guard that holds only when its input is
        # already correct is not a guard.
        changed = True
        while changed:
            changed = False
            for atom_id in list(kept):
                refs = self.pool[atom_id].refs
                if any(r not in kept for r in refs):
                    kept.remove(atom_id)
                    dropped_set.add(atom_id)
                    used -= self.count_tokens(rendered[atom_id] + "\n")
                    changed = True

        # -- the repair gives tokens back; spend them ------------------------
        changed = True
        while changed:
            changed = False
            for atom_id in ordered:
                if atom_id not in dropped_set:
                    continue
                # Same correction as the repair above: a reference that was never
                # a candidate is not in `dropped_set`, so testing that alone
                # re-admitted atoms with dangling references. `ids` holds every
                # ordered atom, so "not in ids" is exactly "never a candidate".
                if any(
                    r in dropped_set or r not in ids
                    for r in self.pool[atom_id].refs
                ):
                    continue
                cost = self.count_tokens(rendered[atom_id] + "\n")
                if used + cost <= cap:
                    dropped_set.discard(atom_id)
                    used += cost
                    changed = True
                else:
                    # **Stop at the first atom that still does not fit**, rather
                    # than skipping it for a cheaper one further down. Continuing
                    # would reintroduce first-fit through the reclaim pass — the
                    # same defect the `break` above removed from the main loop,
                    # arriving by another route. Reclaiming *in order* keeps the
                    # U-shape's premise intact: tokens freed by closure repair go
                    # to the strongest atom that can use them, not the cheapest.
                    break
        kept = [a for a in ordered if a not in dropped_set]

        blocks = [rendered[a] for a in kept]
        claim_map = {ids[a]: a for a in kept}
        positions = {ids[a]: i for i, a in enumerate(kept)}
        included, dropped = kept, sorted(dropped_set)

        return SerialisedProof(
            text="\n".join(blocks),
            claim_map=dict(sorted(claim_map.items())),
            included=tuple(included),
            dropped=tuple(sorted(dropped)),
            tokens=used,
            budget=cap,
            ordering=ordering_name,
            counter=self.counter_name,
            positions=dict(sorted(positions.items())),
        )


def ordering_gap(
    serializer: "ProofSerializer",
    proof: Any,
    obligations: Obligations,
    scores: Mapping[str, float],
    gold: frozenset[str],
    *,
    budget: int | None = None,
) -> dict[str, Any]:
    """The honest ordering measured against the gold one — §6 decision 3.

    **Added 16 Aug 2026 after an adversarial audit observed that no code path
    could compute it.**  Decision 3 says the gold ordering is *"implemented as a
    DIAGNOSTIC only; the head-to-head gap against the honest ordering is the
    measurement of what honesty costs, and reporting either alone is the failure
    decision 3 exists to prevent."*  Both orderings existed; the gap did not, so
    the decision was half-kept — the harder half to notice, because the code
    looked complete.

    Returns both serialisations' shape and the positional agreement between them.
    ``rank_correlation`` is Spearman over the shared atoms' positions: 1.0 means
    the honest ordering reproduced the gold one, and lower values are the cost of
    ranking on signals available at inference.
    """
    honest = serializer.serialise(proof, obligations, scores, budget=budget)
    diagnostic = serializer.serialise(proof, obligations, scores, budget=budget, gold=gold)

    shared = [a for a in honest.included if a in diagnostic.included]
    correlation: float | None = None
    if len(shared) >= 2:
        h_rank = {a: i for i, a in enumerate(honest.included)}
        g_rank = {a: i for i, a in enumerate(diagnostic.included)}
        n = len(shared)
        d2 = sum((h_rank[a] - g_rank[a]) ** 2 for a in shared)
        correlation = 1.0 - (6.0 * d2) / (n * (n * n - 1))

    return {
        "honest": honest.report(),
        "gold_diagnostic": diagnostic.report(),
        "shared_atoms": len(shared),
        "rank_correlation": correlation,
        "gold_first_under_honest": [
            honest.included.index(a) for a in honest.included if a in gold
        ],
        "note": (
            "the gold ordering is a DIAGNOSTIC and is never used on a reported "
            "run; this gap is what the honest ordering costs (decision 3)"
        ),
    }
