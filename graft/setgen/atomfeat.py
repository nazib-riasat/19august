"""P9.3 — the real featurizer: the F6 boundary, one stage later.

```
ProofExample ──► RealFeaturizer ──► policy(state_repr, action_reprs) → logits
                   (adapter)              (the learner, UNCHANGED)
```

**This module is the payoff of fix F6, and the whole reason Phase 3 was built
against an adapter.**  ``SyntheticFeaturizer`` is the only module in
``graft.setgen`` permitted to touch a ``StateGraph``, a ``LatticeInstance`` or a
lattice atom id; this is its real-data sibling, and between them every learner
file stays byte-identical (exit criterion 1).  If a learner ever had to know
which of the two it was talking to, Phase 9 would have been a rewrite.

**The one interface change G1 forces, and why it is not a leak.**  The synthetic
featurizer takes ``state_ix`` — an index into an *enumerated* state graph.  Real
pools are ~10¹³ subsets (plan §3.4), so there is no index to pass and this one
takes an explicit ``selected`` mask instead.  That changes what the **trainer**
hands the featurizer; it does not change what the featurizer hands the
**policy**, which is still ``(state_repr, action_reprs) → logits`` with the same
two tensors of the same rank.  The frozen surface is the second one.

**Every block declares its source, and absence is an input** (G2).  Stage-B
encoder embeddings do not exist on a Wikipedia pool — a 2Wiki paragraph never
passed through D1–D4 commit — so the block is zeroed *with its presence flag
cleared* rather than silently zeroed.  Without the flag the policy can learn
"stage-B block is zero ⇒ this is the Wikipedia corpus", which is a dataset
classifier wearing a policy's name.  The width is identical on both tracks so
that Stage B's transfer measurement compares two runs that differ by their data
and nothing else.

**Channel scores are carried in both views, raw and normalised**, and that is
not redundancy.  `PHASE8_DECISIONS.md` §3.3 is the record: min–max normalisation
is scale-invariant per question, so it made 10 of 13 channel features constant
and put a real gate run at chance while AURC still read healthy.  A
normalisation correct for its own purpose destroyed exactly the information a
different consumer needed.  Fusion keeps its arithmetic; this reads both.

**``Δd`` belongs to a transition, not to a state**, and the ``delta_d`` flag is
the entire L6/L7 difference (decision 19a) — ``True`` for L7 and L7b, ``False``
for L1–L6 *and* GAFlowNet, whose ``Δd`` reaches its loss as an intrinsic reward
and never its policy.  A leak in either direction makes Gate 2 compare two arms
that differ by zero, which is why exit criterion 4 asserts it in both directions
rather than one.

**No gold reaches a feature vector.**  ``ProofExample.gold_atom_ids`` is read
here in exactly one place — :meth:`RealFeaturizer.gold_path`, L1 and L2's
supervision target, which is a *label* and not an input — and
``test_structure.py`` asserts the feature builders never touch it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch

from graft.config import Config
from graft.core import obligations as ob
from graft.setgen.featurenames import (
    ACTION_EXTRA_DIMS,
    ATOM_BLOCKS,
    ATOM_WIDTH,
    BLOCK_FEATURES,
    CHANNELS,
    EMBED_DIM,
    STAGE_B_DIM,
    STATE_EXTRA_DIMS,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.setgen.proofs import ProofExample

__all__ = [
    "RealFeaturizer",
    "BLOCK_FEATURES",
    "ATOM_BLOCKS",
    "ATOM_WIDTH",
    "STATE_EXTRA_DIMS",
    "ACTION_EXTRA_DIMS",
    "EMBED_DIM",
    "STAGE_B_DIM",
    "CHANNELS",
]

#: Names, widths and the channel vocabulary live in ``featurenames`` — a
#: torch-free module — and are re-exported here so callers keep one import.  The
#: split exists because ``pins.frozen_values()`` hashes these names to build the
#: stage-D fingerprint and ``verify_handoff.py`` calls it on a bare interpreter.

NEG_INF = float("-inf")


def _in_scope(snapshot: Any, atom: Any, scope: set[str]) -> bool:
    """Whether this atom's provenance resolves into the question's scope.

    The same walk ``H``'s scope sub-check makes — assertion → span → turn →
    ``conv_id`` — reached through ``core.resolve`` rather than reimplemented, so
    the feature and the checker cannot disagree about what "in scope" means.
    An atom with no provenance at all (an Entity, a TimeInterval) is
    scope-neutral and counts as in scope, exactly as the checker treats it.
    """
    from graft.core import resolve

    try:
        convs, issues = resolve.conv_ids(atom, snapshot, None)
    except TypeError:  # pragma: no cover - signature guard
        return True
    if issues:
        return False
    return all(c in scope for c in convs)


def _normalise(text: str) -> str:
    """Casefold and collapse whitespace — the cheapest defensible match rule.

    **[ANALYSIS]** and deliberately crude.  `PHASE7_DECISIONS.md` §3.2 recorded
    that decision 7's normalised-exact entity rule missed an anchor demonstrably
    in the graph, and that six further misses were a category-vs-instance
    distinction rather than a normalisation one — so a more elaborate rule here
    would not fix the real problem and would make this feature incomparable with
    the entity channel's.  Same rule, same limitation, one place to change it.
    """
    return " ".join(text.casefold().split())


class RealFeaturizer:
    """Featurises one :class:`~graft.setgen.proofs.ProofExample`.

    One example per featurizer, mirroring ``SyntheticFeaturizer``'s one instance
    per featurizer.  Everything policy-independent — the atom matrix, the
    obligation-match flags, the channel views — is built once in ``__init__`` and
    reused at every step of every trajectory, which is the same observation that
    made Phase 2's state graph reusable across policies.
    """

    def __init__(
        self,
        example: "ProofExample",
        policy: torch.nn.Module | None = None,
        cfg: Config | None = None,
        *,
        delta_d: bool,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.example = example
        self.policy = policy
        self.cfg = cfg if cfg is not None else Config()
        self.delta_d = bool(delta_d)
        self.device = torch.device(device)
        self.dtype = dtype

        self.atom_ids: tuple[str, ...] = tuple(example.pool.ids())
        self.n_atoms = len(self.atom_ids)
        self.index = {aid: i for i, aid in enumerate(self.atom_ids)}

        self.atom_feat = torch.as_tensor(
            self._atom_matrix(), dtype=self.dtype, device=self.device
        )

    # -- the atom matrix ---------------------------------------------------

    def _atom_matrix(self) -> np.ndarray:
        """``[n_atoms, ATOM_WIDTH]`` — every per-atom block, in ``ATOM_BLOCKS`` order."""
        ex = self.example
        pool = ex.pool
        rows = np.zeros((self.n_atoms, ATOM_WIDTH), dtype=np.float64)

        # -- block 1: text embedding ---------------------------------------
        col = 0
        has_text = bool(ex.atom_feat)
        for i, aid in enumerate(self.atom_ids):
            vector = ex.atom_feat.get(aid)
            if vector is None:
                continue
            vector = np.asarray(vector, dtype=np.float64).ravel()
            if vector.shape[0] != EMBED_DIM:
                raise ValueError(
                    f"atom {aid} carries a {vector.shape[0]}-dim embedding; the "
                    f"pinned embedder is {EMBED_DIM}-dim. A width mismatch here "
                    "would shift every later block by the difference."
                )
            rows[i, col : col + EMBED_DIM] = vector
        rows[:, col + EMBED_DIM] = 1.0 if has_text else 0.0
        col += EMBED_DIM + 1

        # -- block 2: channel scores, raw AND normalised (Phase-8 §3.3) -----
        #
        # **The raw column reads ``{channel}_raw``, and getting this wrong is how
        # Phase 8's defect repeated one phase later** (found by adversarial
        # audit, 16 Aug 2026). The adapters supply the *normalised* view under
        # the bare channel name and the raw magnitudes under ``{channel}_raw``.
        # Reading the bare name into the column *called* ``_raw`` and min-maxing
        # it again produced two bit-identical normalised columns — min-max is
        # idempotent — while the genuine BM25 magnitude was computed, carried on
        # the example, and never read by anything.
        #
        # `PHASE8_DECISIONS.md` §3.3 is the record of what that costs: min-max is
        # scale-invariant per question, so it made 10 of 13 channel features
        # constant and put a real run at chance (0.52) while AURC still read
        # healthy. The stated fix was to expose **both** views. One view exposed
        # twice under two names is not that fix, and the fingerprint could not
        # tell a corrected run from this one because the *names* never moved.
        supplied = ex.channel_scores
        for channel in CHANNELS:
            raw_map = supplied.get(f"{channel}_raw")
            norm_map = supplied.get(channel)
            # A channel is present when the adapter supplied *either* view. The
            # raw view is preferred for the raw column and falls back to the
            # normalised one only when no raw view was supplied at all, which is
            # a legitimate adapter (an entity channel has no natural magnitude).
            present = raw_map is not None or norm_map is not None
            source = raw_map if raw_map is not None else (norm_map or {})
            values = np.array(
                [float(source.get(aid, 0.0)) for aid in self.atom_ids], dtype=np.float64
            )
            rows[:, col] = values

            if norm_map is not None:
                normed = np.array(
                    [float(norm_map.get(aid, 0.0)) for aid in self.atom_ids],
                    dtype=np.float64,
                )
            else:
                lo = float(values.min()) if values.size else 0.0
                hi = float(values.max()) if values.size else 0.0
                span = hi - lo
                normed = (values - lo) / span if span > 0 else np.zeros_like(values)
            rows[:, col + 1] = normed
            rows[:, col + 2] = 1.0 if present else 0.0
            col += 3
        rows[:, col] = 1.0 if supplied else 0.0
        col += 1

        # -- block 3: Stage-B encoder embedding ----------------------------
        # Absent on every Wikipedia pool, present on the conversational track.
        # The width is occupied either way so the two tracks are comparable.
        stage_b = getattr(self.example, "stage_b_feat", None) or {}
        for i, aid in enumerate(self.atom_ids):
            vector = stage_b.get(aid)
            if vector is None:
                continue
            vector = np.asarray(vector, dtype=np.float64).ravel()
            if vector.shape[0] != STAGE_B_DIM:
                raise ValueError(
                    f"atom {aid} carries a {vector.shape[0]}-dim Stage-B embedding; "
                    f"the pinned encoder width is {STAGE_B_DIM}"
                )
            rows[i, col : col + STAGE_B_DIM] = vector
        rows[:, col + STAGE_B_DIM] = 1.0 if stage_b else 0.0
        col += STAGE_B_DIM + 1

        # -- block 4: obligation match -------------------------------------
        from graft.setgen.proofs import atom_text

        q = ex.obligations
        anchor = _normalise(q.entity_anchor) if q.entity_anchor else ""
        value_type = _normalise(q.value_type) if q.value_type else ""
        scope = set(q.scope)
        for i, aid in enumerate(self.atom_ids):
            atom = pool[aid]
            text = _normalise(atom_text(ex.snapshot, atom))
            rows[i, col + 0] = 1.0 if anchor and anchor in text else 0.0
            rows[i, col + 1] = 1.0 if value_type and value_type in text else 0.0
            # `needs_source` is satisfied by an atom that resolves to an
            # assertion at all, which is what H's support sub-check walks.
            rows[i, col + 2] = 1.0 if q.needs_source and atom.kind == "node" else 0.0
            # **`scope_ok` was inverted and therefore identically zero** (found
            # by adversarial audit, 16 Aug 2026): it read `1.0 if not scope`,
            # and `scope` is always set — `proofs.build_snapshot` makes the
            # question id the conv_id precisely so H's scope sub-check binds. So
            # the column was a constant, and a constant feature is a wasted
            # column that also makes the block look richer than it is.
            #
            # The honest quantity is what H's scope sub-check actually decides:
            # does this atom's provenance resolve INTO the question's scope.
            rows[i, col + 3] = 1.0 if (not scope or _in_scope(ex.snapshot, atom, scope)) else 0.0
            rows[i, col + 4] = 1.0 if atom.kind == "node" else 0.0
            rows[i, col + 5] = 1.0 if atom.kind == "edge" else 0.0
        # Scope is resolved per atom by H itself; this flag says the *block* was
        # computed, which it always is -- the obligations object exists even when
        # every slot is empty.
        rows[:, col + 6] = 1.0
        col += 7

        if col != ATOM_WIDTH:  # pragma: no cover - guarded by a test
            raise AssertionError(
                f"atom matrix filled {col} columns against ATOM_WIDTH {ATOM_WIDTH}; "
                "BLOCK_FEATURES and _atom_matrix have drifted apart, which would "
                "make the stage-D fingerprint describe a vector that does not exist"
            )
        if not np.all(np.isfinite(rows)):
            raise ValueError("atom features contain a non-finite value")
        return rows

    # -- shapes ------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        return ATOM_WIDTH + STATE_EXTRA_DIMS

    @property
    def action_dim(self) -> int:
        return ATOM_WIDTH + ACTION_EXTRA_DIMS

    @staticmethod
    def dims() -> tuple[int, int]:
        """``(state_dim, action_dim)`` without building anything.

        Constant here, unlike the synthetic featurizer's, because the atom width
        is fixed by the pins rather than by the instance.  The trainer uses it to
        size every arm identically before any of them exists.
        """
        return ATOM_WIDTH + STATE_EXTRA_DIMS, ATOM_WIDTH + ACTION_EXTRA_DIMS

    # -- featurisation -----------------------------------------------------

    def state_repr(
        self, selected: torch.Tensor, deficits: torch.Tensor, sizes: torch.Tensor
    ) -> torch.Tensor:
        """``[n, state_dim]`` — mean-pooled selected atoms, ``d(s)``, budget.

        Mean-pooled rather than summed so the representation does not scale with
        ``|s|``; the size is supplied explicitly instead, where the network can
        use it without it contaminating every other feature.  Same choice as the
        synthetic featurizer, for the same reason.
        """
        counts = selected.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (selected @ self.atom_feat) / counts
        size_frac = (sizes / float(self.cfg.max_atoms)).unsqueeze(1)
        return torch.cat([pooled, deficits, size_frac, 1.0 - size_frac], dim=1)

    def action_reprs(self, n: int, delta: torch.Tensor | None = None) -> torch.Tensor:
        """``[n, n_atoms + 1, action_dim]`` — per-action features including ``Δd``.

        The final slot is ``STOP``.  Its ``Δd`` is **zero by construction** —
        ``STOP`` does not change the selected set — and it is flagged so the
        network can tell "no change" from "not an ADD".

        ``delta`` is ignored unless :attr:`delta_d` is set.  The gate is here
        rather than in the caller so that an arm cannot obtain ``Δd`` by passing
        it: L6 and GAFlowNet may compute ``Δd`` for their *losses* and still must
        not see it in their *policy* (decision 19a).
        """
        out = torch.zeros(
            (n, self.n_atoms + 1, self.action_dim), dtype=self.dtype, device=self.device
        )
        out[:, : self.n_atoms, :ATOM_WIDTH] = self.atom_feat.unsqueeze(0).expand(n, -1, -1)
        out[:, self.n_atoms, ATOM_WIDTH + len(ob.DEFICIT_COMPONENTS)] = 1.0  # STOP flag
        if self.delta_d and delta is not None:
            out[:, : self.n_atoms, ATOM_WIDTH : ATOM_WIDTH + len(ob.DEFICIT_COMPONENTS)] = delta
        return out

    def instance_repr(self) -> torch.Tensor:
        """``[1, state_dim]`` — a pooled description of the *example*, for ``logZ_θ``.

        The mean over the **whole pool**, padded to ``state_dim``.  The root
        state's own representation would not serve: an empty selected set pools
        to zero on every example alike, so ``logZ_θ`` could not tell them apart.
        """
        pooled = self.atom_feat.mean(dim=0, keepdim=True)
        pad = torch.zeros((1, STATE_EXTRA_DIMS), dtype=self.dtype, device=self.device)
        return torch.cat([pooled, pad], dim=1)

    def logits(
        self,
        selected: torch.Tensor,
        deficits: torch.Tensor,
        sizes: torch.Tensor,
        legal: torch.Tensor,
        delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[n, n_atoms + 1]`` masked log-probabilities, differentiable.

        Masked **before** the softmax, so an illegal action receives exactly zero
        probability rather than a small positive one that would leak into the
        sampler and produce trajectories the checker refuses.
        """
        if self.policy is None:
            raise ValueError(
                "no policy attached: build one with RealFeaturizer.dims() and "
                "assign it to `.policy` before asking for logits"
            )
        n = int(selected.shape[0])
        raw = self.policy.action_logits(
            self.state_repr(selected, deficits, sizes), self.action_reprs(n, delta)
        )
        return torch.log_softmax(raw.masked_fill(~legal, NEG_INF), dim=1)

    # -- supervision (L1, L2) — the one place gold is read -----------------

    def gold_path(
        self, rng: np.random.Generator | None = None
    ) -> tuple[list[tuple[str, ...]], list[int]]:
        """``(states, actions)`` from the empty set to the gold terminal.

        L1 and L2's supervision (plan §5.1: *supervised stepwise* and *canonical
        set imitation*).  ``rng = None`` walks the **canonical** order — ascending
        pool position — which is what makes L2 an imitation of *one* sequence;
        with an ``rng`` the order is uniform over the closure-legal ones, which is
        what makes L1 stepwise supervision of the *set*.

        **This is the only method here that reads gold**, and it reads it as a
        target rather than as an input feature.  It walks the *smallest complete*
        gold group under ``pins.CREDIT_CONVENTION``: on 2Wiki and MuSiQue-Ans
        there is exactly one, so the choice is vacuous there and defined for the
        corpora where it will not be.

        Raises rather than truncating.  Silently training L1 and L2 on a partial
        path would give them a target that is not the gold proof, and nothing
        downstream would notice — the failure would read as "imitation does not
        work here".
        """
        ex = self.example
        complete = [
            group
            for group in ex.gold_groups.values()
            if group and all(aid in self.index for aid in group)
        ]
        if not complete:
            raise ValueError(
                f"example {ex.example_id}: no complete gold group survives in the "
                "pool, so there is no gold path to imitate. `ProofExample."
                "gold_complete` is the filter that should have excluded it."
            )
        remaining = set(min(complete, key=lambda g: (len(g), sorted(g))))

        from graft.core.incremental import IncrementalChecker
        from graft.core.masks import legal_add_ids, stop_allowed

        state = IncrementalChecker(ex.pool, ex.obligations, ex.snapshot, self.cfg)
        states: list[tuple[str, ...]] = [state.selected()]
        actions: list[int] = []
        while remaining:
            allowed = [aid for aid in legal_add_ids(state) if aid in remaining]
            if not allowed:
                raise ValueError(
                    f"example {ex.example_id}: the gold proof is not constructible "
                    f"under the closure rule — no legal gold ADD with {len(remaining)} "
                    "gold atoms outstanding. Fix F10 makes every valid terminal "
                    "constructible, so this means this gold set is not one."
                )
            pick = allowed[0] if rng is None else allowed[int(rng.integers(len(allowed)))]
            state.add(pick)
            remaining.discard(pick)
            actions.append(self.index[pick])
            states.append(state.selected())
        if not stop_allowed(state):
            raise ValueError(
                f"example {ex.example_id}: the gold set reaches a state that cannot "
                "STOP — the supervised target would be a set H rejects"
            )
        return states, actions

    # -- diagnostics -------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """Which blocks are actually present, for the artefact.

        Reported rather than assumed: G2's whole mechanism is that absence is an
        input, and a run whose artefact does not say which blocks were present
        cannot be compared with one whose blocks differed.
        """
        present: dict[str, bool] = {}
        col = 0
        for block in ATOM_BLOCKS:
            width = len(BLOCK_FEATURES[block])
            present[block] = bool(self.atom_feat[:, col + width - 1].max().item() > 0.0)
            col += width
        return {
            "n_atoms": self.n_atoms,
            "atom_width": ATOM_WIDTH,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "delta_d": self.delta_d,
            "blocks_present": present,
        }
