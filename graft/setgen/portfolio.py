"""P9.6 — Stage-D inference: fix F4's portfolio, on real pools.

``1 greedy + K−1 sampled → rank by the distilled head → tie-break smaller set``,
with ``FAIL`` mapped to the abstain fallback Phase 8 reserved.

**A correction to this phase's own §6 decision 9, made at the build and recorded
rather than quietly applied.**  Decision 9 as signed reads "``h_filter`` after
dedup by ``canon_set_hash``".  That is wrong here, and Phase 4 already said so
in a docstring: ``search.base.h_filter`` is *"the direct-builder path only.  S1,
S2 and S5 construct through the masks, where ``stop_allowed`` **is** ``H``, so
they are valid by construction and must not come through here — routing them
through it would charge them a check each and erase the family difference G6
exists to expose."*  ``s5_portfolio`` reports ``terminal_checks=0`` for exactly
that reason.

This portfolio constructs through the same masks, so routing it through
``h_filter`` would have charged it **8 terminal checks it does not owe**,
against a budget of 32 — and the error runs *against* the proposed method, which
is the direction this project's §5 pattern says to look for last and find
anyway.  The decision-9 text was inherited from fix F4's prose without being
checked against Phase 4's measured design.  **Spend here is 0 terminal checks**,
and the 0-vs-1 split Gate 3's budget row reports is preserved.

**So what does the budget bind on, and what is the fallback?**  Not exhaustion by
validation — there is nothing to validate.  ``would_exceed`` is still consulted
before any check this module makes (the optional audit path below), so the
budget is *enforced rather than observed* even when the spend is zero, which is
what exit criterion 13 asks.  The fallback fires when **every** rollout reaches
``FAIL``: no legal ``ADD`` and ``STOP`` masked, on all ``K`` attempts.  That is
fix F3's licensed reading — "no valid proof found under this pool, policy,
attempt count and budget" — and never "no proof exists".

**Fix F4's ``contested`` flag is Phase 10's, and this module no longer claims
otherwise** (corrected 16 Aug 2026).  F4 flags an output ``contested`` when the
top valid sets imply **different answer bindings**.  That is not computable from
gold aliases, for a structural reason rather than a missing feature: every alias
is an alias *of the one gold answer*, so every match resolves to the same answer
by construction.  What this module reports instead is
:func:`answer_agreement` — whether the top sets agree on *carrying* the answer,
and how many distinct evidence groups carry it.  Both are real, neither is F4's
flag, and the names now say which is which.  The deployment-time comparison is
the architecture's "costs one comparison" reader check, **transferred to
Phase 10 by name**; inventing a gold-free proxy here would be the unstated
machinery G9 refuses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import numpy as np

from graft.config import Config
from graft.ids import canon_set_hash
from graft.ledger import Ledger
from graft.setgen.pins import PORTFOLIO
from graft.setgen.realenv import RealEnvironment, sample_real

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.setgen.atomfeat import RealFeaturizer

__all__ = ["PortfolioResult", "run_portfolio", "binding_of", "answer_agreement"]


class PortfolioResult:
    """One query's Stage-D output.

    ``distinct_valid`` is **reported, never assumed to be ``K``** — Phase-4
    decision 3, which measured greedy returning ~2.45 distinct valid sets on the
    lattice while a table saying 8 would have been the label the ruling was
    about.
    """

    __slots__ = (
        "sets", "scores", "portfolio", "attempted", "distinct_valid",
        "terminal_checks", "fallback", "budget_exhausted", "extra",
    )

    def __init__(
        self,
        sets: Sequence[tuple[str, ...]],
        scores: Sequence[float],
        *,
        portfolio: Sequence[tuple[str, ...]],
        attempted: int,
        terminal_checks: int = 0,
        fallback: bool = False,
        budget_exhausted: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if len(sets) != len(scores):
            raise ValueError(f"{len(sets)} sets against {len(scores)} scores")
        self.sets = tuple(sets)
        self.scores = tuple(float(s) for s in scores)
        self.portfolio = tuple(portfolio)
        self.attempted = int(attempted)
        self.distinct_valid = len(self.sets)
        self.terminal_checks = int(terminal_checks)
        self.fallback = bool(fallback)
        self.budget_exhausted = bool(budget_exhausted)
        self.extra = dict(extra or {})

    @property
    def best(self) -> tuple[str, ...] | None:
        """The delivered set, or ``None`` when the fallback fired."""
        return self.sets[0] if self.sets else None

    def report(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "distinct_valid": self.distinct_valid,
            "terminal_checks": self.terminal_checks,
            "fallback": self.fallback,
            "budget_exhausted": self.budget_exhausted,
            "best_size": len(self.best) if self.best else 0,
            "best_score": self.scores[0] if self.scores else float("nan"),
            **self.extra,
        }


def run_portfolio(
    feat: "RealFeaturizer",
    env: RealEnvironment,
    scorer: Callable[[Iterable[str]], float],
    rng: np.random.Generator,
    *,
    k: int | None = None,
    ledger: Ledger | None = None,
    config: Config | None = None,
) -> PortfolioResult:
    """Sample ``K``, rank by ``scorer``, break ties toward the smaller set.

    ``K`` is ``Config.K`` — the same constant that is the search comparison's
    "returned sets" count (`CLAUDE.md` §6: change both or neither), referenced
    here rather than restated.

    **The tie-break is where minimality enters the inference path.**  ``U``
    already carries a ``size`` penalty at weight 0.1, which is deliberately small
    so that size does not dominate sufficiency; at equal score, preferring the
    smaller set is the free half of the same preference and costs no reward
    engineering.  ``canon_set_hash`` breaks the remaining ties so the ordering is
    total and two runs deliver the same set rather than whichever the dict
    happened to yield.
    """
    cfg = config or Config()
    size = int(cfg.K if k is None else k)
    if size < 1:
        raise ValueError(f"K must be at least 1, got {size}")

    traj = sample_real(feat, env, size, rng, epsilon=0.0, greedy=int(PORTFOLIO["greedy"]))
    terminals = traj.terminals()

    if ledger is not None:
        # **Stage D's model work, metered where it happens.** `realenv.py`'s
        # docstring already designates this module as the metering site
        # ("portfolio.py meters; nothing here does"), and the read path's cost
        # claim (`CLAUDE.md` §9) needs a numerator on the policy side too, not
        # only the reader's. One unit per policy forward pass, matching
        # `ingest/nli.py`'s convention that the meter counts *forward passes*,
        # not items. `terminal_checks` stays at zero and must: the portfolio
        # constructs through the masks where `stop_allowed` IS `H`, so it owes
        # none (`PHASE9_DECISIONS.md` §1.1).
        ledger.count("model_forwards", int(traj.lengths.sum()) + len(traj))

    # Valid by construction: a trajectory that took STOP did so through
    # `stop_allowed`, which *is* H. A dead-ended one is recognisable from the
    # walk without a checker call. Hence zero terminal checks -- see the module
    # docstring for why this departs from decision 9 as written.
    seen: dict[str, tuple[str, ...]] = {}
    portfolio: list[tuple[str, ...]] = []
    for r in range(len(traj)):
        if bool(traj.is_fail[r]):
            continue
        atoms = terminals[r]
        portfolio.append(atoms)
        seen.setdefault(canon_set_hash(atoms), atoms)

    if not seen:
        # Every attempt reached FAIL. Fix F3's licensed reading, and the event
        # Phase 8's reserved counter (its G5) counts -- zero until now by design.
        #
        # **Not a ledger meter, and the distinction is real.** `ledger.METERS` is
        # a closed vocabulary of *budget-enforced resources* -- things that are
        # spent, capped, and refused at the cap. An abstention has no cap and is
        # not spent; it is an outcome. Phase 8 reserved `abstain_fallback` as an
        # artefact field for exactly that reason and wrote it as one. Counting it
        # on the ledger would have needed a new meter with no cap, which is a
        # meter the enforcement path cannot mean anything for.
        return PortfolioResult(
            (), (), portfolio=(), attempted=size, terminal_checks=0, fallback=True,
            extra={
                "fail_rollouts": int(traj.is_fail.sum()),
                str(PORTFOLIO["fallback_counter"]): 1,
            },
        )

    candidates = list(seen.values())
    scored = sorted(
        ((float(scorer(a)), len(a), canon_set_hash(a), a) for a in candidates),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    return PortfolioResult(
        [t[3] for t in scored],
        [t[0] for t in scored],
        portfolio=portfolio,
        attempted=size,
        terminal_checks=0,
        fallback=False,
        extra={
            "fail_rollouts": int(traj.is_fail.sum()),
            str(PORTFOLIO["fallback_counter"]): 0,
        },
    )


def relevance_select(
    env: RealEnvironment,
    atom_scores: Mapping[str, float],
    *,
    ledger: Ledger | None = None,
    config: Config | None = None,
) -> PortfolioResult:
    """Stage D by **Stage-C relevance**, training-free: top-``max_atoms``, closed, `H`-checked.

    **What this is, stated plainly so no reader has to infer it.**  This is not
    the GFlowNet sampler and makes no learned-construction claim.  It selects the
    highest-scoring atoms under the *question-conditioned* fused score Stage C
    already computed, completes them under the closure rule, and asks `H`.  It is
    the training-free relevance baseline the architecture always named -- run on
    the live path because the Stage-D policy is untrained.

    **Why it exists** *(20 Aug 2026)*.  With a randomly-initialised policy, a
    rollout picks ~``max_atoms`` of ``pool_cap`` atoms with no view of the
    question, so the evidence handed to the reader is close to a random subset of
    the conversation.  The 1,986-question run shows the consequence: 977
    abstentions, most of them the reader correctly reporting that off-topic
    evidence does not answer the question.  Ranking cannot repair that -- the
    scorer chooses among candidates, and every candidate was drawn blind.

    **What it costs, and what it does not.**  `pool_cap` and `max_atoms` are
    untouched, `H` still decides validity, the budget is still enforced through
    ``would_exceed``, and the reward, the checker and the metrics are all
    unchanged.  What is given up is that a run using this path says nothing about
    Contribution 2 or 3 -- which was already true of a run using an untrained
    sampler, and is now *visible* instead of implicit.  Gate 3 is the learned
    sampler's exam and is deferred by name (`CLAUDE.md` §8).

    Ties break toward the smaller set, the same rule
    :func:`run_portfolio` applies, so minimality still enters the inference path.
    """
    cfg = config or Config()
    pool = env.example.pool
    ranked = sorted(
        pool.ids(),
        key=lambda a: (-float(atom_scores.get(a, 0.0)), len(pool[a].refs), a),
    )

    # Closure first, then the cap: an atom is legal only once everything it
    # references is selected (fix F10), and a set that broke that would be
    # refused by `H` for a structural reason having nothing to do with relevance.
    chosen: list[str] = []
    seen: set[str] = set()

    def _admit(atom_id: str) -> bool:
        """Add ``atom_id`` and its reference closure, if they fit ``max_atoms``."""
        need: list[str] = []
        stack = [atom_id]
        while stack:
            current = stack.pop()
            if current in seen or current in need:
                continue
            need.append(current)
            stack.extend(pool[current].refs)
        if len(seen) + len(need) > cfg.max_atoms:
            return False
        for atom in need:
            seen.add(atom)
            chosen.append(atom)
        return True

    for atom_id in ranked:
        if len(seen) >= cfg.max_atoms:
            break
        _admit(atom_id)

    atoms = tuple(sorted(seen))
    if not atoms:
        return PortfolioResult(
            (), (), portfolio=(), attempted=1, terminal_checks=0, fallback=True,
            extra={"selection": "relevance", str(PORTFOLIO["fallback_counter"]): 1},
        )

    # One terminal check, enforced before it is spent. Unlike the sampled
    # portfolio this set is *not* valid by construction -- nothing walked the
    # masks -- so it owes `H` exactly one call and the ledger must permit it.
    spent = 0
    if ledger is not None and ledger.would_exceed("terminal_checks", 1):
        return PortfolioResult(
            (), (), portfolio=(atoms,), attempted=1, terminal_checks=0,
            fallback=True, budget_exhausted=True,
            extra={"selection": "relevance", str(PORTFOLIO["fallback_counter"]): 1},
        )
    state = env.checker(atoms)
    ok = bool(state.ok())
    spent = 1
    if ledger is not None:
        ledger.count("terminal_checks", 1)

    if not ok:
        return PortfolioResult(
            (), (), portfolio=(atoms,), attempted=1, terminal_checks=spent,
            fallback=True,
            extra={"selection": "relevance", str(PORTFOLIO["fallback_counter"]): 1},
        )
    return PortfolioResult(
        [atoms], [0.0], portfolio=[atoms], attempted=1, terminal_checks=spent,
        fallback=False,
        extra={"selection": "relevance", str(PORTFOLIO["fallback_counter"]): 0},
    )


def audit_validity(
    result: PortfolioResult, env: RealEnvironment, ledger: Ledger | None = None
) -> dict[str, Any]:
    """Re-check delivered sets against batch ``H`` — **off the measured path**.

    "Valid by construction" is a claim about the masks, and this is how it gets
    tested rather than trusted.  It is deliberately **not** called by
    :func:`run_portfolio`: the spend it would incur is exactly the spend the
    module docstring explains the portfolio does not owe, so it belongs in a test
    and an audit, never in a Gate-3 row.

    The budget is consulted with ``would_exceed`` **before** each check, so even
    this path is enforced rather than observed (exit criterion 13).  A ``ledger``
    must therefore be inside an open ``query_scope()``: ``terminal_checks`` is
    capped *per query*, and the ledger refuses to answer ``would_exceed`` outside
    one rather than silently measuring against a total — which is the right
    refusal, since a per-query cap checked against a lifetime count would let
    early queries spend the whole budget.

    **It calls the batch ``checker.H``, not ``IncrementalChecker.ok()``, and the
    difference is the whole point** (corrected 16 Aug 2026 by adversarial audit).
    The first version re-invoked ``env.checker(atoms).ok()`` — but
    ``masks.stop_allowed`` *is* ``state.ok()``, so it re-ran the exact predicate
    that admitted the set and could not have disagreed with it. An audit that
    re-asks the question the claim is made of verifies nothing; the batch entry
    point is an independent implementation over the same specification, which is
    what makes a disagreement meaningful.
    """
    from graft.core.checker import H

    checked = spent = 0
    invalid: list[tuple[str, ...]] = []
    for atoms in result.sets:
        if ledger is not None and ledger.would_exceed("terminal_checks"):
            return {
                "checked": checked, "spent": spent, "invalid": invalid,
                "budget_exhausted": True,
            }
        verdict = H(
            atoms,
            env.example.obligations,
            env.example.snapshot,
            env.example.pool,
            env.cfg,
            ledger=ledger,
        )
        if ledger is not None:
            spent += 1
        checked += 1
        if not verdict.ok:
            invalid.append(atoms)
    return {"checked": checked, "spent": spent, "invalid": invalid, "budget_exhausted": False}


def binding_of(
    atoms: Sequence[str], env: RealEnvironment, aliases: Sequence[str]
) -> frozenset[str]:
    """Which selected atoms carry the answer, by alias match.

    **Gold-bearing splits only.**  Fix F4's ``contested`` flag asks whether the
    top valid sets imply *different answer bindings*; on train and dev the answer
    string is known, so a set's binding is the subset of its atoms whose text
    contains it.  This is a diagnostic and is labelled one everywhere it is
    reported.

    Matching is the same normalised-substring rule ``atomfeat`` uses for
    obligation flags, and it inherits the same limitation, which
    `PHASE7_DECISIONS.md` §3.2 measured: a normalised-exact rule missed an anchor
    demonstrably present, and six further misses were a category-vs-instance
    distinction that no normalisation fixes.  One rule, one place, one known
    weakness — rather than a second matcher with a second unknown one.
    """
    from graft.setgen.atomfeat import _normalise
    from graft.setgen.proofs import atom_text

    wanted = [_normalise(a) for a in aliases if a]
    if not wanted:
        return frozenset()
    out = set()
    for aid in atoms:
        if aid not in env.example.pool:
            continue
        text = _normalise(atom_text(env.example.snapshot, env.example.pool[aid]))
        if any(w and w in text for w in wanted):
            out.add(aid)
    return frozenset(out)


def answer_agreement(
    result: PortfolioResult, env: RealEnvironment, aliases: Sequence[str], top: int = 2
) -> dict[str, Any]:
    """Do the top valid sets agree on **whether** they carry the gold answer?

    **Renamed from ``contested_rate`` on 16 Aug 2026, because the old name
    claimed a measurement this track cannot make.**  Fix F4's ``contested`` flag
    fires when the top sets imply *different answer bindings*.  With gold aliases
    alone that is **not computable**, and the reason is structural rather than an
    implementation gap: every alias in ``aliases`` is an alias *of the one gold
    answer*, so every match resolves to the same answer by construction.  Two
    sets can differ in the *evidence* they carry and in *whether* they carry the
    answer at all — never in which answer they assert.

    The old implementation compared **atom-id sets** while its own docstring
    claimed to compare "the *answer* they resolve to", and it was wrong in both
    directions: two paragraphs supporting the same answer read as contested, and
    a set that bound nothing at all read as agreeing.

    So this reports the two things that *are* measurable here:

    ``all_bound`` / ``none_bound``
        whether every top set carries the answer, or none does.  The mixed case
        — some bind, some do not — is a **real disagreement** and is what
        ``answer_presence_disagreement`` flags.

    ``distinct_evidence``
        how many distinct atom groups carry it.  This is *evidence* diversity,
        not answer conflict, and >1 is agreement about the answer reached by
        different routes — which is what a portfolio is *supposed* to produce.

    **Fix F4's true contested flag is Phase 10's**, by name (G9): at inference
    there is no gold, and the architecture's own reading is a reader-level
    comparison over the top valid sets.  Inventing a gold-free proxy here would
    be the unstated machinery G9 refuses.
    """
    heads = list(result.sets[: max(1, int(top))])
    bindings = [binding_of(a, env, aliases) for a in heads]
    bound = [b for b in bindings if b]
    return {
        "top_considered": len(heads),
        "bound": len(bound),
        "unbound": len(heads) - len(bound),
        "all_bound": len(heads) > 0 and len(bound) == len(heads),
        "none_bound": len(bound) == 0,
        # A genuine disagreement this track CAN see: the sets differ on whether
        # the answer is present at all.
        "answer_presence_disagreement": 0 < len(bound) < len(heads),
        # Evidence diversity, NOT answer conflict. Named so it cannot be read as
        # fix F4's contested flag.
        "distinct_evidence": len({frozenset(b) for b in bound}),
        "note": (
            "gold-alias diagnostic on an evaluation split. Competing answer "
            "bindings are NOT measurable here — every alias is an alias of the "
            "one gold answer — so fix F4's contested flag is Phase 10's (G9)."
        ),
    }
