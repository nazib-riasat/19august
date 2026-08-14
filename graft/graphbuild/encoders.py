"""P6.4 — the three encoders, behind one frozen interface (G2, decision 1).

``encode(features, edges) -> node_reprs``.  Decoders never learn which encoder
produced their inputs — fix F6's pattern, applied at Stage B — so swapping E1 for
E3 is a constructor change and nothing else.  That is what makes the Gate-1
comparison a comparison of *encoders* rather than of three separately-tuned
systems.

**E1 — GraphMixer-style, plain torch, and deliberately dependency-free.**
**[EVIDENCE]** GraphMixer (ICLR 2023): an MLP link-encoder plus mean-pooling plus
an MLP head matched or beat RNN/attention temporal GNNs.  The caveat travels with
the citation, as the plan requires: that result is on *temporal link prediction*,
not on a heterogeneous assertion graph, so E1 here is a faithful port of the
*architecture*, not a claim that its published result transfers.  It imports no
PyG, so the mandatory baseline survives any PyG breakage.

**E2 — 2-layer HGTConv, the architecture's own pick.**  **[EVIDENCE]** HGT
(WWW 2020) reports 9–21% over prior GNN baselines on the 179M-node OAG, and its
node- and edge-type-dependent attention is the closest published encoder to this
graph's shape.  It is the *point of comparison* E3 has to beat, and
re-implementing typed attention by hand is a correctness risk exactly where a
subtle bug reads as "the baseline is weak" — the failure Phase 3's capacity-match
defect already demonstrated here.  :func:`hgt_available` re-runs G2's platform
check at import time; if it ever fails, E2 falls back to CompGCN and the swap is
**recorded as a departure**, never silent.

**E3 — the proposed encoder: HGT backbone plus the GRAFT feature set.**
**[HYPOTHESIS]**, and the plan is explicit about what a negative result means: if
E3 does not beat E1 *and* E2, the encoder story is "HGT suffices" and the
write-up says so.  Its only difference from E2 is the features — bge-small text
embeddings, provenance flags, time deltas, degree — which is the same
one-mechanism discipline L7 was held to in Phase 3.

**Capacity is matched, not hoped for.**  All three take their width from
``pins.TRAINING["hidden"]`` and report parameter counts, because Phase 3's
lesson was that an unmatched comparison is uninterpretable in whichever
direction it lands.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from graft.graphbuild.pins import TRAINING

__all__ = [
    "hgt_available",
    "GraphFeatures",
    "MlpEncoder",
    "HgtEncoder",
    "GraftEncoder",
    "build_encoder",
    "parameter_count",
]


def hgt_available() -> tuple[bool, str]:
    """G2's platform check, re-run rather than assumed.

    Returns ``(ok, note)``.  PyG's compiled extensions have no reliable Windows
    wheels, so the condition is that ``HGTConv`` runs **forward and backward** on
    plain torch; importing it is not enough, because the failure mode is a
    missing kernel at the first scatter.
    """
    try:
        import torch_geometric.typing as pyg_typing
        from torch_geometric.nn import HGTConv
    except Exception as exc:  # noqa: BLE001 - any import failure is the same answer
        return False, f"torch_geometric unavailable: {type(exc).__name__}: {exc}"
    try:
        meta = (["A", "B"], [("A", "r", "B"), ("B", "rev_r", "A")])
        conv = HGTConv(4, 4, meta, heads=1)
        x = {"A": torch.randn(2, 4), "B": torch.randn(2, 4)}
        index = {
            ("A", "r", "B"): torch.tensor([[0], [0]]),
            ("B", "rev_r", "A"): torch.tensor([[0], [0]]),
        }
        out = conv(x, index)
        sum(v.sum() for v in out.values()).backward()
    except Exception as exc:  # noqa: BLE001
        return False, f"HGTConv forward/backward failed: {type(exc).__name__}: {exc}"
    return True, (
        f"HGTConv forward+backward OK; pyg_lib={pyg_typing.WITH_PYG_LIB}, "
        f"torch_scatter={pyg_typing.WITH_TORCH_SCATTER}"
    )


class GraphFeatures:
    """A typed graph as tensors: per-type node features and per-relation edges.

    Not a dataclass (Phase-0 criterion 12).  Holding the three encoders' input in
    one shape is what lets the interface be frozen: E1 flattens it, E2 and E3 use
    the type structure, and none of them needs a different builder.
    """

    __slots__ = ("x", "edge_index", "node_ids", "meta")

    def __init__(
        self,
        x: Mapping[str, "torch.Tensor"],
        edge_index: Mapping[tuple[str, str, str], "torch.Tensor"],
        node_ids: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.x = dict(x)
        self.edge_index = dict(edge_index)
        self.node_ids = {k: list(v) for k, v in (node_ids or {}).items()}
        self.meta = (sorted(self.x), sorted(self.edge_index))

    @property
    def dim(self) -> int:
        for tensor in self.x.values():
            return int(tensor.shape[-1])
        return 0

    def counts(self) -> dict[str, int]:
        return {k: int(v.shape[0]) for k, v in sorted(self.x.items())}


# --------------------------------------------------------------------------
# E1 — GraphMixer-style
# --------------------------------------------------------------------------


class MlpEncoder(nn.Module):
    """E1.  Per-node MLP, neighbour mean-pool, MLP head.  No PyG.

    The mean-pool is over *all* incoming relations without distinguishing them —
    that is the point of the baseline.  If E2 or E3 cannot beat a
    relation-agnostic pool, typed attention is not earning its complexity on this
    graph, which is a result worth having rather than an embarrassment to avoid.
    """

    def __init__(self, in_dim: int, node_types: Sequence[str], hidden: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden or TRAINING["hidden"])
        self.node_types = list(node_types)
        self.hidden = hidden
        self.encode_node = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(TRAINING["dropout"])
        )
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, features: GraphFeatures) -> dict[str, "torch.Tensor"]:
        encoded = {k: self.encode_node(v) for k, v in features.x.items()}
        pooled = {k: torch.zeros_like(v) for k, v in encoded.items()}
        counts = {k: torch.zeros(v.shape[0], 1) for k, v in encoded.items()}

        for (src_t, _, dst_t), index in features.edge_index.items():
            if index.numel() == 0 or src_t not in encoded or dst_t not in encoded:
                continue
            src, dst = index[0], index[1]
            pooled[dst_t] = pooled[dst_t].index_add(0, dst, encoded[src_t][src])
            counts[dst_t] = counts[dst_t].index_add(
                0, dst, torch.ones(dst.shape[0], 1, device=counts[dst_t].device)
            )

        out: dict[str, torch.Tensor] = {}
        for key, value in encoded.items():
            mean = pooled[key] / counts[key].clamp(min=1.0).to(value.device)
            out[key] = self.head(torch.cat([value, mean], dim=-1))
        return out


# --------------------------------------------------------------------------
# E2 / E3 — HGT backbone
# --------------------------------------------------------------------------


class HgtEncoder(nn.Module):
    """E2.  Two ``HGTConv`` layers over the typed graph.

    ``metadata`` is fixed at construction because HGT allocates per-type and
    per-relation parameters: a graph that gains an unseen relation at inference
    has no weights for it.  Passing the *schema's* metadata rather than the
    current graph's is therefore deliberate — the endpoint table (G3) enumerates
    every relation that can ever appear, so the parameterisation is complete by
    construction rather than by whatever the pilot happened to contain.
    """

    def __init__(
        self,
        in_dim: int,
        metadata: tuple[Sequence[str], Sequence[tuple[str, str, str]]],
        hidden: int | None = None,
        heads: int | None = None,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import HGTConv

        hidden = int(hidden or TRAINING["hidden"])
        heads = int(heads or TRAINING["heads"])
        meta = (list(metadata[0]), [tuple(r) for r in metadata[1]])
        self.project = nn.ModuleDict({t: nn.Linear(in_dim, hidden) for t in meta[0]})
        self.conv1 = HGTConv(hidden, hidden, meta, heads=heads)
        self.conv2 = HGTConv(hidden, hidden, meta, heads=heads)
        self.dropout = nn.Dropout(TRAINING["dropout"])
        self.hidden = hidden

    def forward(self, features: GraphFeatures) -> dict[str, "torch.Tensor"]:
        x = {k: self.project[k](v) for k, v in features.x.items() if k in self.project}
        edges = {k: v for k, v in features.edge_index.items() if v.numel() > 0}
        if not edges:
            # An isolated graph is a real state — the first turn of a
            # conversation has no edges yet — and HGTConv has nothing to
            # aggregate.  Returning the projection is correct and is what lets
            # the cold-start case flow through the same code path.
            return {k: self.dropout(v) for k, v in x.items()}
        h = self.conv1(x, edges)
        h = {k: self.dropout(torch.relu(v)) for k, v in h.items()}
        return self.conv2(h, edges)


class GraftEncoder(HgtEncoder):
    """E3.  **[HYPOTHESIS]** — the HGT backbone with the GRAFT feature set.

    Structurally identical to E2 on purpose: the *only* difference is what
    ``GraphFeatures.x`` carries, which is what makes "E3 beats E2" a statement
    about the features rather than about depth or width.  Phase 3 held L7 to
    exactly this discipline (one mechanism, ``Δd`` as input features and nothing
    else) after fix F11 found the original design was satisfiable by copying.

    If E3 does not beat both E1 and E2 at matched budget, the encoder story is
    "HGT suffices" and the plan says to write that down.
    """


# --------------------------------------------------------------------------
# construction and reporting
# --------------------------------------------------------------------------


def build_encoder(
    arm: str,
    in_dim: int,
    metadata: tuple[Sequence[str], Sequence[tuple[str, str, str]]],
    hidden: int | None = None,
) -> nn.Module:
    """``'E1' | 'E2' | 'E3'`` → an encoder, or a refusal that names the reason.

    E2/E3 refuse rather than silently degrading to E1 when the platform check
    fails: a comparison in which the HGT arm is quietly an MLP would report a
    number for an encoder that never ran.  The CompGCN fallback is a *declared*
    substitution, made by the caller who then records the departure.
    """
    if arm == "E1":
        return MlpEncoder(in_dim, metadata[0], hidden)
    if arm in ("E2", "E3"):
        ok, note = hgt_available()
        if not ok:
            raise RuntimeError(
                f"{arm} needs HGTConv and the platform check failed: {note}. "
                "Falling back silently would report a number for an encoder that "
                "never ran; the declared alternate is CompGCN, adopted "
                "explicitly and recorded as a departure (G2)."
            )
        cls = HgtEncoder if arm == "E2" else GraftEncoder
        return cls(in_dim, metadata, hidden)
    raise KeyError(f"unknown encoder arm {arm!r}; the Tier-1 ladder is E1, E2, E3")


def parameter_count(module: nn.Module) -> int:
    """Trainable parameters — reported per arm beside every Gate-1 score.

    Phase 3's capacity lesson, which is why this is a function and not a comment:
    a comparison at unmatched capacity is uninterpretable, and the only way that
    stays visible is if the number is in the table.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
