"""P4.2 — G1's declared relevance stand-in, in both ruled variants.

**Why this file exists at all.** S3's objective opens with ``w_rel·Rel`` and
S4's PCST needs prizes; in the finished system both come from Stage C's hybrid
retrieval, which is **Phase 7 and does not exist**. A pool atom carries a
12-dimensional feature vector and no relevance channel. `CLAUDE.md` §8 names
these two baselines as the ones most likely to embarrass the project, so
**defining ``rel`` badly is how they get accidentally weakened**, and a Gate-3
pass bought that way would be worthless.

**Decision 1, ruled:** obligation-match is primary, the ``U``-marginal is a
declared *informed* variant, and **both are reported in every table**. That
closes the escape hatch in either direction — if S5 wins it wins against a
relevance-informed S3/S4, and if S5 loses the loss is not explained away as
"the proxy was weak".

Both are **[ANALYSIS]**: neither is a published relevance model, and the
write-up must not present either as one.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from graft.config import Config
from graft.core.obligations import SLOT_COMPONENTS, slot_status
from graft.core.utility import U
from graft.graphstore import GraphSnapshot
from graft.schemas import AtomPool, Obligations, ProofSet

__all__ = [
    "RELEVANCE_VARIANTS",
    "obligation_relevance",
    "informed_relevance",
    "relevance_vector",
]

#: The two variants of decision 1.  Every Phase-4 table is labelled with one of
#: these (exit criterion 8) — a number without its variant is unreadable.
RELEVANCE_VARIANTS = ("obligation", "informed")


def obligation_relevance(
    atom_id: str,
    pool: AtomPool,
    q: Obligations,
    G: GraphSnapshot,
) -> float:
    """Does this atom, alone, address any obligation slot? In [0, 1].

    ``1 − mean(slot deficit of {atom})`` over the **active** slots, computed with
    the project's own ``slot_status`` — the single implementation shared by
    ``d(s)`` and ``U``'s ``coverage`` (Phase-1 gap G4). Reusing it is deliberate:
    a second, private notion of "addresses an obligation" is exactly the drift
    that gap exists to prevent, and it would make S3/S4 optimise a target the
    rest of the system does not recognise.

    This is the **realistic** variant of decision 1 — the signal a Stage-C
    retriever would actually produce (an anchor hit, a value-type match, a
    temporal overlap) rather than the evaluation metric itself.

    A question with no active slots scores every atom 1.0 by the same convention
    ``coverage`` uses, which keeps ``rel`` uniform rather than zero and leaves
    the rest of S3's objective doing the discriminating.
    """
    active = q.active_slots()
    if not active:
        return 1.0
    status = slot_status([atom_id], pool, q, G)
    # ``active_slots()`` names the *fields*; ``slot_status`` keys by deficit
    # *component*.  ``SLOT_COMPONENTS`` is the project's own map between them —
    # writing the correspondence out again here is how the two drift apart.
    deficit = float(np.mean([status[SLOT_COMPONENTS[s]] for s in active]))
    return float(np.clip(1.0 - deficit, 0.0, 1.0))


def informed_relevance(
    atom_id: str,
    pool: AtomPool,
    q: Obligations,
    G: GraphSnapshot,
    gold: ProofSet | Iterable[str],
    cfg: Config,
) -> float:
    """The atom's singleton utility ``U({a})``, clamped to [0, 1].

    Decision 1's **informed** variant, and it must be labelled wherever it
    appears: it hands S3 and S4 the *scorer* inside their own objective, so they
    are optimising the thing they are scored on — which no deployed retriever
    does. It exists to remove the "S5 only won because ``rel`` was weak"
    objection, not because it is realistic.

    Singleton rather than a context-dependent marginal because a PCST prize is a
    property of a node, fixed before the solve. ``redundancy`` is 0 for a
    singleton by construction (Phase-1 exit criterion 10), so the only negative
    term is ``size = 1/max_atoms``; the clamp therefore moves at most
    ``w_size/max_atoms`` and only for an atom of zero positive utility.
    """
    value = U([atom_id], q, G, pool, gold, cfg)
    return float(np.clip(value, 0.0, 1.0))


def relevance_vector(env: Any, variant: str) -> dict[str, float]:
    """``{atom_id: rel}`` over the whole pool, for one instance and one variant.

    Computed once per (instance, variant) and handed to S3 and S4, so the two
    baselines are demonstrably reading the same numbers.
    """
    if variant not in RELEVANCE_VARIANTS:
        raise ValueError(
            f"unknown relevance variant {variant!r}; decision 1 declares "
            f"{list(RELEVANCE_VARIANTS)} and a third would be an undeclared "
            "choice about how strong the baselines are"
        )
    inst = env.instance
    pool, q, G, cfg = inst.pool, inst.obligations, inst.graph, inst.cfg
    if variant == "obligation":
        return {aid: obligation_relevance(aid, pool, q, G) for aid in pool.ids()}
    return {
        aid: informed_relevance(aid, pool, q, G, inst.gold, cfg) for aid in pool.ids()
    }
