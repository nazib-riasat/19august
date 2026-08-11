"""The F6 boundary: the one adapter that joins Phase 2's evaluator to a learner.

```
StateGraph ──► SyntheticFeaturizer ──► policy(state_repr, action_reprs) → logits
                    (adapter)                  (the learner, F6 interface)
           └────────── implements ActionPolicy ──────────┘
```

**This module is the entire coupling risk of Phase 3** (G5). Fix F6 exists so a
learner never sees where its features came from: an MLP over synthetic atoms
here, the Stage-B graph encoder at Phase 9, and the learner cannot tell. If that
leaks, Phase 9 is a rewrite. So this is the **only** module in ``graft.setgen``
permitted to touch a ``StateGraph``, a ``LatticeInstance`` or an atom id, and a
test asserts the rest do not.

**``Δd`` belongs to a transition, not to a state** (G5). ``d(s)`` is a property of
the selected set and goes in ``state_repr``; ``Δd = d(s) − d(s′)`` is indexed by
*(state, action)* and goes in ``action_reprs``. Fix F11 defines L7 as "``Δd`` as
input features to ``φ_θ`` and the policy" — put it anywhere else and either the
per-action resolution is lost or it leaks to every arm.

**The ``delta_d`` flag is the whole L6/L7 difference** (decision 19a):

======================  =========================================================
``delta_d = True``      L7, L7b — the proposed mechanism
``delta_d = False``     L1-L6 **and GAFlowNet**, whose ``Δd`` reaches its *loss*
                        as an intrinsic reward and never its policy (G11)
======================  =========================================================

A leak in either direction makes Gate 2 compare two arms that differ by zero,
which is why exit criterion 5 asserts it in both directions rather than one.

**Everything policy-independent is computed once.** The deficits, the per-edge
``Δd`` and the atom features depend on the instance and the masks, not on θ — the
same observation that made Phase 2's state graph reusable across policies (G2).
They are built in ``__init__`` and reused at every checkpoint of every run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from graft.config import Config
from graft.core import obligations as ob
from graft.schemas import FEAT_DTYPE
from graft.synth.enumerate import StateGraph

if TYPE_CHECKING:  # pragma: no cover - typing only
    from graft.synth.lattice import LatticeInstance

__all__ = ["SyntheticFeaturizer", "STATE_EXTRA_DIMS", "ACTION_EXTRA_DIMS"]

#: ``d(s)`` (6) + ``|s|/max_atoms`` + ``budget_left``.
STATE_EXTRA_DIMS = len(ob.DEFICIT_COMPONENTS) + 2

#: ``Δd`` (6) + a flag marking the ``STOP`` slot.
ACTION_EXTRA_DIMS = len(ob.DEFICIT_COMPONENTS) + 1

NEG_INF = float("-inf")


class SyntheticFeaturizer:
    """Implements :class:`~graft.synth.policies.ActionPolicy` over a learner.

    Batch-first by construction (decision 8): it is handed an array of state
    indices and returns dense ``[n_states, n_atoms + 1]`` log-probabilities. It
    **chunks internally** — "batched" means "not one call per state", not one
    call per instance, because 1e5 states of activations is a memory constraint
    Phase 3 should not be forced into by an interface.
    """

    def __init__(
        self,
        instance: "LatticeInstance",
        graph: StateGraph,
        policy: torch.nn.Module | None = None,
        cfg: Config | None = None,
        *,
        delta_d: bool,
        chunk: int = 4096,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.graph = graph
        self.policy = policy
        self.cfg = cfg if cfg is not None else instance.cfg
        self.delta_d = bool(delta_d)
        self.chunk = int(chunk)
        self.device = torch.device(device)
        # float32 for training, float64 when a test needs to separate "the
        # adapter distorts the distribution" from "float32 has ~1e-7 of
        # precision".  Exit criterion 1 is a claim about the adapter, so it is
        # checked in float64; the float32 agreement is reported separately as a
        # property of the dtype, not of this module.
        self.dtype = dtype

        pool, q, G = instance.pool, instance.obligations, instance.graph
        n_states, n_atoms = graph.n_states, graph.n_atoms
        max_atoms = self.cfg.max_atoms

        # -- atom features: the only place atom ids are read ---------------
        width = max((pool[aid].feat.shape[0] for aid in graph.atom_ids), default=0)
        atom_feat = np.zeros((n_atoms, max(width, 1)), dtype=np.float64)
        for i, aid in enumerate(graph.atom_ids):
            feat = pool[aid].feat
            atom_feat[i, : feat.shape[0]] = feat
        self.atom_feat = torch.as_tensor(atom_feat, dtype=self.dtype, device=self.device)

        # -- d(s) per state, and Δd per edge -------------------------------
        deficits = np.empty((n_states, len(ob.DEFICIT_COMPONENTS)), dtype=np.float64)
        for i in range(n_states):
            deficits[i] = ob.deficit(graph.atoms_of(i), pool, q, G)
        self._deficits = torch.as_tensor(deficits, dtype=self.dtype, device=self.device)

        # Δd is a difference of two rows already materialised — a lookup, not a
        # recomputation (G5).
        delta = deficits[graph.edge_parent] - deficits[graph.edge_child]
        self._edge_delta = torch.as_tensor(delta, dtype=self.dtype, device=self.device)

        # **Dense `[n_states, n_atoms, 6]`, and the density is the point.**
        #
        # The first version scattered per-edge Δd into the queried rows via a
        # state→row map. That is wrong whenever ``state_ix`` repeats a state:
        # the map holds one row per state, so the last occurrence wins and every
        # earlier one keeps a zeroed Δd block. Unique queries (the exact DP, the
        # sampler) never hit it; **training batches hit it constantly**, because
        # a batch of trajectories revisits states by design. The failure is
        # silent and it points the wrong way — L7 would train on mostly-zeroed
        # features and look like "Δd does not help", which is a *false negative*
        # for Contribution 3.
        #
        # Indexing a dense table is duplicate-safe by construction. Cost at this
        # scale is ~1 MB; at G1's 1e5-state ceiling it is ~150 MB in float32,
        # which is the reason it is built only when this arm actually uses it.
        self._delta_table = None
        if self.delta_d:
            table = np.zeros((n_states, n_atoms, len(ob.DEFICIT_COMPONENTS)))
            table[graph.edge_parent, graph.edge_action] = delta
            self._delta_table = torch.as_tensor(table, dtype=self.dtype, device=self.device)

        # -- selected-set indicator, for pooled atom features --------------
        masks = graph.mask.astype(np.uint64)
        bits = ((masks[:, None] >> np.arange(n_atoms, dtype=np.uint64)[None, :]) & 1)
        self._selected = torch.as_tensor(
            bits.astype(np.float64), dtype=self.dtype, device=self.device
        )

        # Which pool positions are the gold proof — the only thing L1 and L2 need
        # to know about atom identity, reduced to action indices here so no
        # learner ever sees an atom id (fix F6, exit criterion 6).
        gold_atoms = set(instance.gold.atoms)
        self._is_gold = np.array(
            [aid in gold_atoms for aid in graph.atom_ids], dtype=bool
        )

        sizes = torch.as_tensor(
            graph.size.astype(np.float64), dtype=self.dtype, device=self.device
        )
        self._size_frac = (sizes / max_atoms).unsqueeze(1)
        self._budget_left = 1.0 - self._size_frac

        # -- dense per-state action layout, built once ---------------------
        self._edge_row = torch.as_tensor(
            graph.edge_parent.astype(np.int64), dtype=torch.long, device=self.device
        )
        self._edge_col = torch.as_tensor(
            graph.edge_action.astype(np.int64), dtype=torch.long, device=self.device
        )
        legal = torch.zeros((n_states, n_atoms + 1), dtype=torch.bool, device=self.device)
        legal[self._edge_row, self._edge_col] = True
        legal[:, n_atoms] = torch.as_tensor(
            graph.stop_allowed.copy(), dtype=torch.bool, device=self.device
        )
        self._legal = legal

    # -- shapes ------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        return int(self.atom_feat.shape[1]) + STATE_EXTRA_DIMS

    @property
    def action_dim(self) -> int:
        return int(self.atom_feat.shape[1]) + ACTION_EXTRA_DIMS

    @staticmethod
    def dims(instance: "LatticeInstance", graph: StateGraph) -> tuple[int, int]:
        """``(state_dim, action_dim)`` **without** building anything.

        The policy's input widths are a property of the environment, not of the
        policy, but a policy has to be constructed with them — so asking the
        featurizer for its dimensions would need a policy that does not exist
        yet. This breaks that cycle, and the trainer uses it to size every arm
        identically before any of them is built.
        """
        width = max((instance.pool[aid].feat.shape[0] for aid in graph.atom_ids), default=0)
        width = max(width, 1)
        return width + STATE_EXTRA_DIMS, width + ACTION_EXTRA_DIMS

    # -- featurisation -----------------------------------------------------

    def state_repr(self, state_ix: torch.Tensor) -> torch.Tensor:
        """``[n, state_dim]`` — pooled selected-atom features, ``d(s)``, budget.

        Mean-pooled rather than summed so the representation does not scale with
        ``|s|``; the size is supplied explicitly instead, where the network can
        use it without it contaminating every feature.
        """
        selected = self._selected[state_ix]
        counts = selected.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (selected @ self.atom_feat) / counts
        return torch.cat(
            [pooled, self._deficits[state_ix], self._size_frac[state_ix],
             self._budget_left[state_ix]],
            dim=1,
        )

    def action_reprs(self, state_ix: torch.Tensor) -> torch.Tensor:
        """``[n, n_atoms + 1, action_dim]`` — per-action features including ``Δd``.

        The final slot is ``STOP``. Its ``Δd`` is **zero by construction** —
        ``STOP`` does not change the selected set — which is the same exclusion
        Phase-2 G5 applies when measuring ``Δd`` density, and it is flagged so the
        network can tell "no change" from "not an ADD".
        """
        n, n_atoms = state_ix.shape[0], self.graph.n_atoms
        out = torch.zeros((n, n_atoms + 1, self.action_dim), dtype=self.dtype,
                          device=self.device)
        width = self.atom_feat.shape[1]
        out[:, :n_atoms, :width] = self.atom_feat.unsqueeze(0).expand(n, -1, -1)
        out[:, n_atoms, width + len(ob.DEFICIT_COMPONENTS)] = 1.0  # the STOP flag

        if self.delta_d:
            # A gather, not a scatter: repeated entries in `state_ix` each get
            # their own copy, which a state-keyed scatter cannot do.
            out[:, :n_atoms, width : width + len(ob.DEFICIT_COMPONENTS)] = (
                self._delta_table[state_ix]
            )
        return out

    def deficits(self, state_ix: torch.Tensor) -> torch.Tensor:
        """``[n, 6]`` — ``d(s)``, precomputed and indexed.

        Public because three consumers outside this module need it and none of
        them should be slicing it back out of ``state_repr`` by offset: the
        realised ``Δd`` of a transition (GAFlowNet's intrinsic reward, G11),
        L7b's forward-looking target ``d(X(τ))`` (decision 26), and the audits.
        """
        return self._deficits[state_ix]

    def gold_path(
        self, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(states[k+1], actions[k])`` from the empty set to the gold terminal.

        L1 and L2's supervision (plan §5.1: *supervised stepwise* and *canonical
        set imitation*), reduced to indices here because atom identity stops at
        this module.

        ``rng = None`` walks the **canonical** order — ascending pool position —
        which is what makes L2 an imitation of *one* sequence. With an ``rng`` the
        order is uniform over the closure-legal ones, which is what makes L1
        stepwise supervision of the *set* rather than of an arbitrary sequence
        the generator happened to emit.

        Raises if the gold set is not reachable or is not a valid terminal:
        silently training on a truncated path would give L1 and L2 a target that
        is not the gold proof, and nothing downstream would notice.
        """
        graph = self.graph
        remaining = set(np.flatnonzero(self._is_gold).tolist())
        states = [0]
        actions: list[int] = []
        state = 0
        while remaining:
            acts, kids = graph.children_of(state)
            legal = [(int(a), int(c)) for a, c in zip(acts, kids) if int(a) in remaining]
            if not legal:
                raise ValueError(
                    "the gold proof is not constructible under the closure rule: "
                    f"no legal gold ADD from state {state} with "
                    f"{len(remaining)} gold atoms outstanding. Fix F10 makes every "
                    "valid terminal constructible, so this means the instance's "
                    "gold set is not one."
                )
            pick = legal[0] if rng is None else legal[int(rng.integers(len(legal)))]
            action, child = pick
            remaining.discard(action)
            actions.append(action)
            states.append(child)
            state = child
        if not bool(graph.stop_allowed[state]):
            raise ValueError(
                f"the gold set reaches state {state}, which cannot STOP — the "
                "supervised target would be a set H rejects"
            )
        return np.asarray(states, dtype=np.int64), np.asarray(actions, dtype=np.int64)

    def instance_repr(self) -> torch.Tensor:
        """``[1, state_dim]`` — a pooled description of the *instance*, for ``logZ_θ``.

        G6 trains one model per (arm, seed) across all 20 main-suite instances
        with ``log Z_θ`` conditioned on which one it is, so the head needs
        something that identifies the instance and does not change as a
        trajectory progresses. The mean over the **whole pool** does that; the
        root state's own representation would not, since an empty selected set
        pools to zero on every instance alike.

        Same width as ``state_repr`` so one head shape serves both.
        """
        pooled = self.atom_feat.mean(dim=0, keepdim=True)
        pad = torch.zeros((1, STATE_EXTRA_DIMS), dtype=self.dtype, device=self.device)
        return torch.cat([pooled, pad], dim=1)

    # -- the ActionPolicy protocol ----------------------------------------

    def action_log_probs(
        self, state_ix: np.ndarray, graph: StateGraph
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(log P_F(ADD a_j | s_i), log P_F(STOP | s_i))`` as numpy arrays.

        Masked **before** the softmax, so an illegal action receives exactly zero
        probability rather than a small positive one that would leak into the DP.
        """
        if graph is not self.graph:
            raise ValueError(
                "this featurizer was built against a different StateGraph; its "
                "precomputed deficits and Δd are indexed by that graph's numbering"
            )
        idx = torch.as_tensor(np.asarray(state_ix, dtype=np.int64), dtype=torch.long,
                              device=self.device)
        if bool(torch.as_tensor(graph.dead_end.copy())[idx.cpu()].any()):
            raise ValueError(
                "a dead-end state was queried: it has no legal ADD and a masked "
                "STOP, so no distribution over its actions exists (G6)"
            )
        logits = self.logits(idx)
        n_atoms = graph.n_atoms
        return (
            logits[:, :n_atoms].detach().cpu().numpy().astype(np.float64),
            logits[:, n_atoms].detach().cpu().numpy().astype(np.float64),
        )

    def logits(self, state_ix: torch.Tensor) -> torch.Tensor:
        """``[n, n_atoms + 1]`` masked log-probabilities, differentiable.

        The training path uses this directly; ``action_log_probs`` is the numpy
        face Phase 2's evaluator consumes.
        """
        if self.policy is None:
            raise ValueError(
                "no policy attached: build one with SyntheticFeaturizer.dims(...) "
                "and assign it to `.policy` before asking for logits"
            )
        chunks = []
        for lo in range(0, state_ix.shape[0], self.chunk):
            block = state_ix[lo : lo + self.chunk]
            raw = self.policy.action_logits(self.state_repr(block), self.action_reprs(block))
            masked = raw.masked_fill(~self._legal[block], NEG_INF)
            chunks.append(torch.log_softmax(masked, dim=1))
        return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
