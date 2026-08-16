"""P9.1 — the source-agnostic boundary: ``(pool, obligations, gold_proof)``.

```
2Wiki ───┐
MuSiQue ─┼─► corpus adapter ─► ProofExample ─► featurizer ─► environment ─► learner
HoVer ───┤                     (this module)
LongMemEval (Stage B, later) ──┘
```

**The architecture states both the requirement and its enforcement**: the loader
consumes the abstract ``(pool, obligations, gold_proof)`` triple and *no
corpus-specific parser may appear above it* — **verified by an import-graph
test** — so a conversational-source variant is a drop-in if the
Wikipedia→conversation transfer claim fails.  That claim is declared and
untested (`CLAUDE.md` §7); this module is what keeps the escape hatch structural
rather than aspirational.

**Why a real snapshot rather than a shortcut.**  It would be quicker to hand the
environment a bare ``AtomPool`` and skip Turn/SourceSpan/Assertion entirely.  It
would also make five of ``H``'s nine sub-checks unevaluable, and "H-valid" would
degenerate into "the size check passed".  So each question gets a genuine
per-question snapshot and the checks that *can* bind, do:

===================  =====================================================
**binds** size       ``max_atoms`` over the selected set
**binds** closure    ``about_entity`` edge atoms reference both endpoints,
                     so an edge is unaddable until its Claim *and* its
                     Entity are selected (fix F10)
**binds** identity   duplicate ``content_key`` within one set
**binds** support    ``Assertion.eligibility``, set explicitly here
**binds** scope      ``conv_id`` = question id, walked assertion → span →
                     turn, exactly as ``H`` does it
*vacuous* temporal   no ``valid_during`` intervals exist on these corpora
*vacuous* binding    no binding atoms
*vacuous* retired    nothing is retired
===================  =====================================================

The three vacuous ones are declared in ``pins.VACUOUS_ON_WIKIPEDIA`` and are
**not** silently counted as passes.  Consequence, stated once here because the
Gate-3 table's reader needs it: with sufficiency and coverage living in ``U``
rather than ``H`` (v1.2 §4.4), most non-empty subsets of an eligible Wikipedia
pool are ``H``-valid and ``STOP`` is rarely masked — so reward discrimination on
this track comes almost entirely from ``U``.  That is the intended design, not a
defect, but it makes "H-valid" here a weaker statement than on the conversational
track, and the write-up must not conflate the two.

**Gold lives on the example, and that is deliberate.**  Fix F1 routes train-time
``sufficiency`` against the gold proof, so the *reward* legitimately reads it.
What must never read it is the **feature path**: ``atomfeat.py`` builds vectors
without it and ``distill.py`` takes exact ``U`` as a training *target* rather
than an input.  Both are asserted structurally in ``test_structure.py`` rather
than left to discipline — the Phase-7/8 gold-quarantine pattern, one phase on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from graft import ids
from graft.config import Config
from graft.graphstore import DictGraphSnapshot
from graft.retrieve.pool import build_pool, node_atom_id, node_text
from graft.schemas import (
    PAYLOAD_ASSERTION_ID,
    PAYLOAD_CONV_ID,
    PAYLOAD_NAME,
    Assertion,
    AssertionFlags,
    AtomPool,
    CandidateAtom,
    Edge,
    Node,
    Obligations,
    SourceSpan,
    Turn,
)

__all__ = [
    "SourceDoc",
    "ProofExample",
    "build_snapshot",
    "build_example",
    "atom_text",
    "register_adapter",
    "get_adapter",
    "adapter_names",
    "EPOCH",
]

#: Wikipedia corpora carry no timestamps, so transaction time is **synthetic and
#: constant-based**: document *i* is created at ``EPOCH + i`` seconds.  A
#: declared adaptation, not a discovery.  It is well-formed ISO-8601 (which the
#: expansion walk's recency sort and the supersession check both assume) and it
#: is *order-preserving*, so nothing reads a meaningless ordering as a meaningful
#: one.  Nothing on this track supersedes anything, so no check actually consumes
#: it — but a malformed timestamp would fail the parse rather than the semantics,
#: and that is a worse way to find out.
EPOCH = "2020-01-01T00:00:00+00:00"


def _ts(index: int) -> str:
    """``EPOCH`` advanced by ``index`` seconds, ISO-8601 UTC."""
    from datetime import datetime, timedelta

    base = datetime.fromisoformat(EPOCH)
    return (base + timedelta(seconds=int(index))).isoformat()


@dataclass(frozen=True)
class SourceDoc:
    """One retrievable unit of text, as an adapter hands it over.

    This is the *whole* corpus-facing vocabulary.  An adapter's job is to turn
    2Wiki's supporting-fact indices or MuSiQue's ``is_supporting`` flags into a
    list of these; everything above this module sees only the result, which is
    what makes the conversational track a drop-in.

    ``entities`` are surface names this document is *about*.  They become
    ``Entity`` nodes joined by ``about_entity`` edges, and they are the reason
    closure is a binding check rather than a vacuous one on this track: the edge
    atom references both its Claim and its Entity, so it cannot be added until
    both are selected (fix F10).

    ``is_gold`` marks membership of the corpus's supporting set.  Under
    ``pins.CREDIT_CONVENTION`` a corpus with several complete alternative proofs
    supplies ``gold_group`` to keep them apart; 2Wiki and MuSiQue-Ans have one
    apiece, so it defaults to group 0 and the convention reduces to
    subset-of-gold-covered.
    """

    doc_id: str
    text: str
    entities: tuple[str, ...] = ()
    is_gold: bool = False
    gold_group: int = 0
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entities", tuple(self.entities))
        if not self.doc_id:
            raise ValueError("a SourceDoc needs a doc_id; it keys provenance")


@dataclass(frozen=True)
class ProofExample:
    """One training instance, with no trace of where it came from.

    ``atom_scores`` is ``build_pool``'s own map — **every** atom in the pool,
    structural companions included at 0.0 — so the featurizer never has to ask
    which atoms were retrieval hits.  ``build_pool``'s docstring names this file
    as its consumer.

    ``channel_scores`` keeps the **two views** Phase 8 had to learn the hard way
    (`PHASE8_DECISIONS.md` §3.3): min–max normalisation is scale-invariant per
    question, so it made 10 of 13 channel features constant and put a real run at
    chance.  Raw and normalised are both carried; fusion's arithmetic is
    untouched and the *consumers* choose.

    ``atom_feat`` holds text embeddings keyed by atom id rather than being packed
    into ``CandidateAtom.feat``.  Three reasons: ``build_pool`` materialises
    atoms and re-materialising them here would give the pool mapping a second
    home (the Phase-7 rule is one mapping project-wide); an embedding is derived
    data, not identity, and ``atom_id``/``content_key`` must not move when an
    embedder does; and the cache round-trips a plain array map far more simply
    than a frozen dataclass graph.

    ``gold_atom_ids`` is the gold field.  See the module docstring for who may
    read it.
    """

    example_id: str
    snapshot: Any
    pool: AtomPool
    obligations: Obligations
    atom_scores: Mapping[str, float]
    channel_scores: Mapping[str, Mapping[str, float]]
    atom_feat: Mapping[str, np.ndarray]
    gold_atom_ids: frozenset[str]
    gold_groups: Mapping[int, frozenset[str]]
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def gold_complete(self) -> bool:
        """Whether at least one complete gold group survived pool construction.

        **A capped pool can drop a gold atom**, and an example whose gold is
        partial is not a harder training instance — it is one whose ``sufficiency``
        can never reach 1.0, so its maximum achievable reward is silently lower
        than every other example's.  Training on it without noticing would bias
        the budget toward examples the pool happened to fit.  The runner filters
        on this and **reports the count it dropped** rather than quietly shrinking
        the subset decision 2 pinned.
        """
        return any(group and group <= set(self.pool.ids()) for group in self.gold_groups.values())

    def report(self) -> dict[str, Any]:
        """Per-example counts for the artefact's per-question listing."""
        kinds: dict[str, int] = {}
        for atom in self.pool:
            kinds[atom.kind] = kinds.get(atom.kind, 0) + 1
        return {
            "example_id": self.example_id,
            "pool_size": len(self.pool),
            "atoms_by_kind": dict(sorted(kinds.items())),
            "gold_atoms": len(self.gold_atom_ids),
            "gold_groups": len(self.gold_groups),
            "gold_complete": self.gold_complete,
            "active_slots": list(self.obligations.active_slots()),
        }


def build_snapshot(example_id: str, docs: Sequence[SourceDoc]) -> DictGraphSnapshot:
    """One question's documents → a snapshot ``H`` can actually evaluate.

    Every document becomes a four-record provenance chain — Turn → SourceSpan →
    Assertion → ``Claim`` node — because that chain *is* what the support and
    scope sub-checks walk.  Skipping it would leave those checks unevaluable
    while still letting the code report "H-valid".

    ``eligibility`` is set **explicitly** to ``"eligible"``.  The schema's default
    is ``"quarantined"`` and fails closed by design (Phase-5 fix F9), so an
    adapter that forgot would produce an empty pool rather than an unsupported
    one — a loud failure, which is the point of that default.

    ``conv_id`` is the question id.  That is what makes ``H``'s scope sub-check
    bind: ``Obligations.scope = (example_id,)`` and the walk assertion → span →
    turn → ``conv_id`` has to agree, so an atom from another question's snapshot
    is refused rather than merely unlikely.
    """
    if not example_id:
        raise ValueError("example_id is the conv_id and the scope; it cannot be empty")
    seen: set[str] = set()
    for doc in docs:
        if doc.doc_id in seen:
            raise ValueError(
                f"duplicate doc_id {doc.doc_id!r} in example {example_id}: doc ids key "
                "provenance, so two documents sharing one would collapse into a "
                "single assertion and silently shrink the pool"
            )
        seen.add(doc.doc_id)

    turns: list[Turn] = []
    spans: list[SourceSpan] = []
    assertions: list[Assertion] = []
    nodes: list[Node] = []
    edges: list[Edge] = []
    entity_node: dict[str, str] = {}

    for index, doc in enumerate(docs):
        turn_id = f"{example_id}::t::{doc.doc_id}"
        ts = _ts(index)
        turns.append(
            Turn(
                turn_id=turn_id,
                conv_id=example_id,
                session_id=example_id,
                # Not "user"/"assistant": a Wikipedia paragraph is neither, and
                # borrowing a conversational speaker label would make the corpus
                # look like something it is not to anything that reads it later.
                speaker="corpus",
                ts=ts,
                text=doc.text,
            )
        )
        span = SourceSpan(
            span_id=ids.span_id(turn_id, 0, len(doc.text)),
            turn_id=turn_id,
            start=0,
            end=len(doc.text),
        )
        spans.append(span)

        assertion = Assertion(
            assertion_id=ids.assertion_id("claim", doc.text, (span.span_id,)),
            kind="claim",
            text_norm=doc.text,
            spans=(span.span_id,),
            flags=AssertionFlags(
                asserted_by="corpus",
                # The corpus *is* the span here: a paragraph is its own evidence,
                # so grounding is exact by construction rather than by an NLI
                # score. Recorded as 1.0 with `asserted_by="corpus"` so the
                # difference from an extractor-derived assertion stays visible.
                entailed_by_span=True,
                entailed_score=1.0,
            ),
            t_created=ts,
            eligibility="eligible",
        )
        assertions.append(assertion)

        claim_id = ids.node_id("Claim", assertion.assertion_id)
        nodes.append(
            Node(
                node_id=claim_id,
                ntype="Claim",
                payload={PAYLOAD_ASSERTION_ID: assertion.assertion_id},
            )
        )

        for name in sorted(set(doc.entities)):
            key = f"{example_id}::{name}"
            ent_id = entity_node.get(key)
            if ent_id is None:
                ent_id = ids.node_id("Entity", key)
                entity_node[key] = ent_id
                nodes.append(
                    Node(
                        node_id=ent_id,
                        ntype="Entity",
                        payload={PAYLOAD_NAME: name, PAYLOAD_CONV_ID: example_id},
                    )
                )
            edges.append(
                Edge(
                    edge_id=ids.edge_id("about_entity", claim_id, ent_id),
                    etype="about_entity",
                    src=claim_id,
                    dst=ent_id,
                    t_created=ts,
                    # The document's own span, and the schema is right to demand
                    # it: "a schema that permits an unsourced edge permits an
                    # unsourced proof". The evidence that this paragraph is about
                    # this entity *is* the paragraph, so the span is not a
                    # formality here — it is the actual warrant.
                    provenance=(span.span_id,),
                )
            )

    return DictGraphSnapshot(
        snapshot_id=len(docs),
        nodes=nodes,
        edges=edges,
        assertions=assertions,
        turns=turns,
        spans=spans,
    )


def atom_text(snapshot: Any, atom: CandidateAtom) -> str:
    """The string an atom is embedded from.

    One function so that the pool-prep cache and the featurizer embed **the same
    text** — the Phase-7 lesson from ``node_text``, where two channels indexing
    different strings would have made a per-channel difference partly an artefact
    of what each was shown.

    A ``Claim`` atom is its assertion's ``text_norm``; an ``Entity`` atom is its
    name; an edge atom is its relation label.  Nothing returns the raw document
    when a normalised form exists.
    """
    if atom.kind == "node":
        text = node_text(snapshot, atom.target)
        if text:
            return text
        node = snapshot.node(atom.target)
        if node is not None:
            return str(node.payload.get(PAYLOAD_NAME, "") or "")
        return ""
    return atom.label


def build_example(
    example_id: str,
    docs: Sequence[SourceDoc],
    obligations: Obligations,
    doc_scores: Mapping[str, float],
    *,
    channel_scores: Mapping[str, Mapping[str, float]] | None = None,
    embed: Callable[[Sequence[str]], np.ndarray] | None = None,
    config: Config | None = None,
    meta: Mapping[str, Any] | None = None,
) -> ProofExample:
    """Documents + scores → a closed, capped :class:`ProofExample`.

    ``doc_scores`` is keyed by ``doc_id`` — the adapter's vocabulary — and is
    translated to node ids here, so no adapter has to know how a ``Claim`` node
    is named.

    **The pool comes from Phase 7's** :func:`~graft.retrieve.pool.build_pool`,
    never a re-implementation (exit criterion 5).  That single mapping owns
    closure, the cap, the eligibility boundary and edge-ref validation; a second
    one here would be the first crack in the Phase-7 rule and would drift the
    moment either side was fixed.
    """
    cfg = config or Config()
    snapshot = build_snapshot(example_id, docs)

    # doc_id -> Claim node id, rebuilt by the same construction build_snapshot
    # used.  Recomputed rather than returned alongside the snapshot: the id
    # functions are pure, so re-deriving is cheap and it keeps `build_snapshot`'s
    # signature a snapshot rather than a snapshot-plus-bookkeeping tuple.
    claim_of: dict[str, str] = {}
    for index, doc in enumerate(docs):
        turn_id = f"{example_id}::t::{doc.doc_id}"
        span = ids.span_id(turn_id, 0, len(doc.text))
        claim_of[doc.doc_id] = ids.node_id(
            "Claim", ids.assertion_id("claim", doc.text, (span,))
        )

    unknown = sorted(set(doc_scores) - set(claim_of))
    if unknown:
        raise ValueError(
            f"example {example_id}: doc_scores names {unknown[:3]} which are not "
            "documents of this example; a score keyed to nothing would silently "
            "drop out of the pool ranking"
        )

    scored_nodes = {claim_of[doc_id]: float(score) for doc_id, score in doc_scores.items()}
    pool, atom_scores, pool_report = build_pool(
        snapshot, scored_nodes, config=cfg, conv_id=example_id
    )

    # -- gold, resolved to atom ids and grouped ----------------------------
    groups: dict[int, set[str]] = {}
    for doc in docs:
        if not doc.is_gold:
            continue
        groups.setdefault(int(doc.gold_group), set()).add(node_atom_id(claim_of[doc.doc_id]))
    gold_groups = {k: frozenset(v) for k, v in sorted(groups.items())}
    gold_atom_ids = frozenset().union(*gold_groups.values()) if gold_groups else frozenset()

    # -- per-channel scores, translated into atom space --------------------
    channels: dict[str, dict[str, float]] = {}
    for channel, per_doc in (channel_scores or {}).items():
        translated: dict[str, float] = {}
        for doc_id, score in per_doc.items():
            node = claim_of.get(doc_id)
            if node is None:
                continue
            translated[node_atom_id(node)] = float(score)
        channels[channel] = dict(sorted(translated.items()))

    # -- embeddings, one call for the whole pool ---------------------------
    atom_feat: dict[str, np.ndarray] = {}
    if embed is not None:
        atom_ids = list(pool.ids())
        texts = [atom_text(snapshot, pool[aid]) for aid in atom_ids]
        vectors = np.asarray(embed(texts), dtype=np.float64)
        if vectors.shape[0] != len(atom_ids):
            raise ValueError(
                f"embedder returned {vectors.shape[0]} vectors for {len(atom_ids)} "
                "atoms; the featurizer indexes them positionally"
            )
        atom_feat = {aid: vectors[i] for i, aid in enumerate(atom_ids)}

    return ProofExample(
        example_id=example_id,
        snapshot=snapshot,
        pool=pool,
        obligations=obligations,
        atom_scores=dict(sorted(atom_scores.items())),
        channel_scores=dict(sorted(channels.items())),
        atom_feat=atom_feat,
        gold_atom_ids=gold_atom_ids,
        gold_groups=gold_groups,
        meta={
            "pool": pool_report,
            "docs": len(docs),
            "gold_docs": sum(1 for d in docs if d.is_gold),
            **dict(meta or {}),
        },
    )


# --------------------------------------------------------------------------
# the adapter registry — the only place a corpus name may appear above P9.2
# --------------------------------------------------------------------------

#: ``name -> callable(**kwargs) -> Iterable[ProofExample]``.
#:
#: Registration is by **explicit call from the adapter module**, not by import
#: scanning: an adapter that is never imported is never registered, so the
#: import-graph test's "nothing above this module names a corpus" stays true of
#: the runtime and not merely of the source.
_ADAPTERS: dict[str, Callable[..., Iterable[ProofExample]]] = {}


def register_adapter(name: str, loader: Callable[..., Iterable[ProofExample]]) -> None:
    """Register a corpus adapter under ``name``.

    Refuses a silent re-registration.  Two adapters under one name is how a run
    ends up training on a corpus its artefact does not name, and the failure is
    invisible in the output.
    """
    if name in _ADAPTERS and _ADAPTERS[name] is not loader:
        raise ValueError(
            f"adapter {name!r} is already registered to a different loader; "
            "a name silently rebound is a run whose artefact names the wrong corpus"
        )
    _ADAPTERS[name] = loader


def get_adapter(name: str) -> Callable[..., Iterable[ProofExample]]:
    """The adapter registered under ``name``.

    The error names the import that would fix it, because "unknown adapter" with
    a bare list is the least useful form of this message: adapters register on
    import, so the fix is always an import.
    """
    try:
        return _ADAPTERS[name]
    except KeyError:
        raise KeyError(
            f"no adapter registered as {name!r} (have: {sorted(_ADAPTERS)}). "
            f"Adapters register on import — try `import graft.setgen.corpora.{name}`."
        ) from None


def adapter_names() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))
