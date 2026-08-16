"""P7.8 — the query-conditioned GNN scorer (G6, decision 9).

**The sixth channel, and the only one that learns.**  Five channels are
training-free by design; this one is additive, last in the build order, and its
absence is a legal configuration of the fusion rather than a crash.  That
separation is G6's and it is what let steps 0–5 be built, run and measured while
Gate 0 was still unsigned.

**[EVIDENCE] Why a small query-conditioned GNN rather than an LLM reranker or an
iterative loop.**

* **GFM-RAG** (NeurIPS 2025) is the scale point this row copies: an **8M-parameter**
  query-conditioned GNN reaching Recall@5 of 87.1 / 58.2 / 95.6 on
  HotpotQA / MuSiQue / 2Wiki in **one pass at 0.107 s**, against 3.162 s for
  iterative IRCoT+HippoRAG.  Hence ``max_params`` 8M and ``passes`` 1 in
  ``pins.SCORER`` — both asserted by tests rather than left as intent.
* **GNN-RAG** (Findings ACL 2025): GNN retrieval beat LLM-based retrieval by
  **8.9–15.5 F1** on multi-hop and multi-entity questions at **9× fewer KG
  tokens**.  *(Findings venue — qualified tier.)*
* GFM-RAG also names **noisy, incomplete KG-index quality as its principal
  dependency**, which is exactly why ceilings 1–2 are reported beside ceiling 3
  and why `PHASE7_DECISIONS.md` §3.1's saturation flag exists.

**[ANALYSIS] What is this project's judgement, not GFM-RAG's.**  The architecture
below — relation-typed message passing over the *pool* with a query gate — is a
faithful small implementation of "query-conditioned, one pass", not a
reimplementation of GFM-RAG's model.  No published number transfers to it, and
none is claimed.  The pool is ≤ ``pool_cap`` atoms, so plain ``torch`` suffices
and `torch_geometric` is not imported: PyG buys typed attention over large
heterogeneous graphs, and at 64 nodes it would be a dependency bought for
nothing (the boring-stack rule, and the same reasoning that kept an ANN index
out of the dense channel).

**Training signal: the item-2 *distant* one, and it arrives pre-labelled.**
``GATE0_CONTRACT.md`` item 2 — written against real records, so it beats the
architecture's 2Wiki row — trains Stage C on "an atom is relevant iff it derives
from a turn in the question's evidence sessions", and evaluates on the fine
signal (required-atom recall).  The distant signal is conversation-native, so
the Wikipedia→conversation transfer declaration (`CLAUDE.md` §7) is **not spent
here**.

**This module never reads a gold field.**  Deriving the distant labels needs
``answer_session_ids``, which is a gold field, so that derivation lives in
``graft.retrieve.recall`` — the one sanctioned gold boundary (G1) — and this
module receives labels as plain numbers.  A structural test asserts the
quarantine over every ``graft.retrieve`` module except ``recall.py``, and moving
the derivation here would break it, correctly.

**The three trainer guards are inherited from P6.11, not reinvented**
(`GRAFT_PHASE7_BUILD.md` P7.8's 15 Aug amendment): the seed reaches
**initialisation** (``build_scorer``, the ``build_arm`` pattern), early stopping
**restores** the argmin-dev state rather than merely stopping, and a loop with no
scorable dev item **refuses** rather than returning its initialisation.  Each of
those is a defect Phase 6 actually shipped and had to have found for it.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from graft.graphbuild.encoders import parameter_count
from graft.graphbuild.pins import EMBEDDER, TRAINING
from graft.retrieve.pins import SCORER
from graft.retrieve.pool import eligible_nodes, node_text, uncapped_pool
from graft.schemas import ATOM_KINDS, EDGE_TYPES, NODE_TYPES

__all__ = [
    "RELATIONS",
    "ATOM_FEATURE_DIM",
    "atom_features",
    "pool_adjacency",
    "PoolScorer",
    "build_scorer",
    "score_pool",
    "channel_scores",
    "train_scorer",
]

#: The relation vocabulary for message passing: every edge type, its reverse, and
#: ``incident`` for the edge atom↔endpoint links.  Derived from the **frozen**
#: :data:`graft.schemas.EDGE_TYPES` rather than enumerated here, so a schema
#: amendment reaches the scorer instead of silently leaving it parameterised for
#: a vocabulary the graph no longer uses.
#:
#: ``features.encoder_metadata()``'s full ``(src, rel, dst)`` triples are the
#: Stage-B parameterisation and are **deliberately not** used: they type a
#: relation by its endpoints, which matters for an encoder over the whole typed
#: graph and is redundant here, where an edge atom already carries its own
#: ``label``.  Recorded as a departure in `PHASE7_DECISIONS.md`.
RELATIONS: tuple[str, ...] = (
    tuple(EDGE_TYPES) + tuple(f"rev_{e}" for e in EDGE_TYPES) + ("incident",)
)

_REL_IX = {name: ix for ix, name in enumerate(RELATIONS)}
_KIND_IX = {kind: ix for ix, kind in enumerate(ATOM_KINDS)}
_NTYPE_IX = {name: ix for ix, name in enumerate(NODE_TYPES)}
_ETYPE_IX = {name: ix for ix, name in enumerate(EDGE_TYPES)}

#: kind one-hot + node-type one-hot + edge-type one-hot + the pinned text vector.
ATOM_FEATURE_DIM = len(ATOM_KINDS) + len(NODE_TYPES) + len(EDGE_TYPES) + int(EMBEDDER["dim"])


def atom_features(
    pool: Any,
    snapshot: Any,
    embedder: Any,
    *,
    order: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """``(n, ATOM_FEATURE_DIM)`` features and the atom-id order they are in.

    The order is ``pool.ids()`` — sorted — unless one is supplied, because a row
    index is a tensor position and set iteration order is randomised per process.
    Everything downstream (labels, scores, the adjacency) indexes against the
    returned tuple rather than re-deriving it.

    Text comes through :func:`graft.retrieve.pool.node_text`, the same accessor
    BM25 and the dense channel use, so all three see one string per atom.
    Structural atoms have no assertion and get a zero text block — they are
    support, and the type one-hots are what the model has to work with for them.
    """
    ids = tuple(order) if order is not None else pool.ids()
    texts = [node_text(snapshot, pool[aid].target) if pool[aid].kind == "node" else "" for aid in ids]
    nonempty = [i for i, t in enumerate(texts) if t.strip()]
    dim = int(EMBEDDER["dim"])
    vectors = np.zeros((len(ids), dim), dtype=np.float32)
    if nonempty:
        encoded = np.asarray(embedder.embed([texts[i] for i in nonempty]), dtype=np.float32)
        for slot, i in enumerate(nonempty):
            vectors[i] = encoded[slot]

    out = np.zeros((len(ids), ATOM_FEATURE_DIM), dtype=np.float32)
    off_n = len(ATOM_KINDS)
    off_e = off_n + len(NODE_TYPES)
    off_v = off_e + len(EDGE_TYPES)
    for i, aid in enumerate(ids):
        atom = pool[aid]
        out[i, _KIND_IX[atom.kind]] = 1.0
        if atom.kind == "node" and atom.label in _NTYPE_IX:
            out[i, off_n + _NTYPE_IX[atom.label]] = 1.0
        elif atom.kind == "edge" and atom.label in _ETYPE_IX:
            out[i, off_e + _ETYPE_IX[atom.label]] = 1.0
        out[i, off_v:] = vectors[i]
    return out, ids


def pool_adjacency(pool: Any, ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """``(src_ix, dst_ix)`` pairs and their relation ids, from the pool's own refs.

    The pool is a *reference* structure, not a copy of the graph, so the
    adjacency is read off it directly: an edge atom's ``refs`` are its endpoint
    node atoms (enforced by ``pool.validate_edge_refs``, so this is safe to
    trust here), giving three message paths per edge atom —

    * ``src → dst`` under the edge's own type,
    * ``dst → src`` under ``rev_<type>``, so information flows both ways without
      the model having to learn that edges are symmetric when they are not,
    * ``edge atom ↔ each endpoint`` under ``incident``, which is what gives the
      edge atom a representation at all.

    Returns empty arrays for a pool with no edge atoms — a legal pool (nodes
    reference nothing, which is what makes nodes-first construction always valid)
    and one the forward pass must handle without a special case.
    """
    position = {aid: ix for ix, aid in enumerate(ids)}
    src: list[int] = []
    dst: list[int] = []
    rel: list[int] = []

    def link(a: str, b: str, relation: str) -> None:
        if a in position and b in position:
            src.append(position[a])
            dst.append(position[b])
            rel.append(_REL_IX[relation])

    for aid in ids:
        atom = pool[aid]
        if atom.kind != "edge" or len(atom.refs) != 2:
            continue
        a, b = atom.refs
        etype = atom.label if atom.label in _ETYPE_IX else None
        if etype is not None:
            link(a, b, etype)
            link(b, a, f"rev_{etype}")
        link(aid, a, "incident")
        link(a, aid, "incident")
        link(aid, b, "incident")
        link(b, aid, "incident")

    if not src:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.asarray([src, dst], dtype=np.int64), np.asarray(rel, dtype=np.int64)


class PoolScorer(nn.Module):
    """Query-conditioned relational message passing over one pool.

    One forward pass per question — ``pins.SCORER['passes'] == 1`` and a test
    asserts it — because the evidence for iteration is weak and points the other
    way: HippoRAG (NeurIPS 2024) matched iterative IRCoT at 10–20× lower cost,
    and Beyond Static Retrieval (arXiv 2509.25530, **provisional**) found
    iteration *hurting* simple questions through over-retrieval.

    **Where the query enters, and why there.**  The question vector gates every
    message (``sigmoid`` over a bilinear-ish score of message and query) and is
    concatenated at the head.  Gating rather than mere concatenation is what
    makes it *query-conditioned* rather than query-augmented: with concatenation
    alone the propagation is identical for every question and only the readout
    moves, which is the thing GFM-RAG's design is explicitly not.
    """

    def __init__(
        self,
        in_dim: int | None = None,
        hidden: int | None = None,
        layers: int | None = None,
        query_dim: int | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        # Depth comes from the frozen config, not a constructor literal (15 Aug
        # 2026 audit): a hard-coded default sat outside the Stage-C fingerprint,
        # so two differently-deep scorers would have shared one identity.
        if layers is None:
            layers = int(SCORER["layers"])
        in_dim = int(in_dim or ATOM_FEATURE_DIM)
        hidden = int(hidden or TRAINING["hidden"])
        query_dim = int(query_dim or EMBEDDER["dim"])
        self.hidden = hidden
        self.n_layers = int(layers)
        self.atom_proj = nn.Linear(in_dim, hidden)
        self.query_proj = nn.Linear(query_dim, hidden)
        self.relation = nn.Embedding(len(RELATIONS), hidden)
        self.message = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(self.n_layers))
        self.gate = nn.ModuleList(nn.Linear(hidden * 2, 1) for _ in range(self.n_layers))
        self.update = nn.ModuleList(nn.Linear(hidden * 2, hidden) for _ in range(self.n_layers))
        self.norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(self.n_layers))
        self.drop = nn.Dropout(float(TRAINING["dropout"] if dropout is None else dropout))
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rel: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """``(n,)`` logits, one per atom, in ``features``' row order."""
        h = torch.relu(self.atom_proj(features))
        q = torch.relu(self.query_proj(query)).reshape(1, -1)
        n = h.shape[0]

        for layer in range(self.n_layers):
            if edge_index.shape[1] > 0:
                src, dst = edge_index[0], edge_index[1]
                msg = self.message[layer](h[src]) + self.relation(edge_rel)
                gate = torch.sigmoid(self.gate[layer](torch.cat([msg, q.expand_as(msg)], dim=-1)))
                msg = msg * gate
                agg = torch.zeros((n, self.hidden), dtype=h.dtype, device=h.device)
                agg.index_add_(0, dst, msg)
                # Mean rather than sum: a hub atom would otherwise receive a
                # message whose magnitude scales with its degree, which the
                # fan-out cap bounds but does not equalise (G10).
                degree = torch.zeros((n, 1), dtype=h.dtype, device=h.device)
                degree.index_add_(0, dst, torch.ones((msg.shape[0], 1), dtype=h.dtype, device=h.device))
                agg = agg / degree.clamp(min=1.0)
            else:
                agg = torch.zeros((n, self.hidden), dtype=h.dtype, device=h.device)
            h = self.norm[layer](h + self.drop(torch.relu(self.update[layer](torch.cat([h, agg], dim=-1)))))

        return self.head(torch.cat([h, q.expand(n, -1)], dim=-1)).reshape(-1)


def build_scorer(seed: int | None = None, **kwargs: Any) -> PoolScorer:
    """Construct the scorer, **seeding initialisation** when a seed is given.

    The ``build_arm`` pattern, inherited deliberately (P7.8's amendment): seeding
    after construction leaves every seed sharing one initialisation, so three
    seeds would estimate variance over dropout and batch order alone — a far
    narrower interval than a reader assumes, and irreproducible besides. Phase 6
    shipped that defect and its determinism test caught it on 14 Aug 2026.

    Refuses a configuration over ``pins.SCORER['max_params']`` rather than
    letting it through: 8M is decision 9's number and the GFM-RAG scale point,
    and a scorer that quietly grew past it would make the "same scale as the
    published one-pass result" claim false.
    """
    if seed is not None:
        torch.manual_seed(int(seed))
    scorer = PoolScorer(**kwargs)
    count = parameter_count(scorer)
    if count > int(SCORER["max_params"]):
        raise ValueError(
            f"scorer has {count:,} trainable parameters, over decision 9's "
            f"{int(SCORER['max_params']):,}; that cap is the GFM-RAG scale point "
            "and the claim of comparable scale rests on it"
        )
    return scorer


def score_pool(
    scorer: PoolScorer,
    pool: Any,
    snapshot: Any,
    embedder: Any,
    question: str,
) -> dict[str, float]:
    """**The frozen fix-F6 interface**: ``(question, pool) -> per-atom scores``.

    One call, one forward pass, no iteration.  Returns ``atom_id -> score`` —
    **atom** ids, which is the fix-F6 shape Stage D's featurizer consumes without
    knowing the scorer exists.  It is *not* what fusion consumes: channels emit
    ``node_id -> score``, and the conversion (plus the assertion-backed filter)
    lives in :func:`channel_scores`, the sixth-channel entry point.  An earlier
    docstring here claimed this output "drops straight into the declared
    arithmetic", which was false — every atom id would have been refused as an
    unknown node at assembly (15 Aug 2026 audit).
    """
    if len(pool) == 0:
        return {}
    features, ids = atom_features(pool, snapshot, embedder)
    edge_index, edge_rel = pool_adjacency(pool, ids)
    query = np.asarray(embedder.embed([question]), dtype=np.float32)[0]
    scorer.eval()
    with torch.no_grad():
        logits = scorer(
            torch.from_numpy(features),
            torch.from_numpy(edge_index),
            torch.from_numpy(edge_rel),
            torch.from_numpy(query),
        )
    return {aid: float(v) for aid, v in zip(ids, logits.tolist())}


def channel_scores(
    scorer: PoolScorer,
    snapshot: Any,
    embedder: Any,
    question: str,
    conv_id: str | None = None,
) -> dict[str, float]:
    """The scorer **as the sixth retrieval channel**: node scores over the whole scope.

    **What it scores, and why that is the fix and not a preference** (15 Aug 2026
    audit).  The first wiring scored the already-capped five-channel pool — so
    the GNN could never surface an atom the cheap channels' cap had dropped, and
    under ``max`` fusion it could only re-rank what was already admitted.  That
    is not a sixth member of plan §3.3's *union*, and it inverts the order the
    cited evidence supports: GFM-RAG's GNN scores **before** retrieval ranking,
    over the whole indexed graph, not after a cap.  So this function scores the
    **uncapped closed pool of every eligible node in the question's scope** — the
    same one-pass constraint, now over the candidate space rather than a
    truncation of it.  The cap is applied where it belongs, once, at fusion
    assembly, *after* every channel has spoken.

    Emits ``node_id -> score`` for **assertion-backed** nodes only — the channel
    protocol every other channel follows.  Structural atoms are scored by the
    forward pass (they shape the message passing) but are not emitted: they are
    support, assembly re-derives them by closure, and emitting them would inflate
    ``hits_refused_ineligible`` — the count that exists to flag quarantine
    leakage — with atoms that were never hits (the first wiring did exactly
    that).

    One forward pass per question, unchanged: ``pins.SCORER['passes'] == 1``
    counts *model invocations*, and the scope pool is one graph.
    """
    nodes = eligible_nodes(snapshot, conv_id)
    if not nodes:
        return {}
    pool, _, _ = uncapped_pool(snapshot, {n: 1.0 for n in nodes}, conv_id=conv_id)
    raw = score_pool(scorer, pool, snapshot, embedder, question)
    wanted = set(nodes)
    return {
        pool[aid].target: value
        for aid, value in sorted(raw.items())
        if pool[aid].kind == "node" and pool[aid].target in wanted
    }


def _example_loss(
    scorer: PoolScorer, example: Mapping[str, Any]
) -> tuple[torch.Tensor, int]:
    """Binary cross-entropy over one question's pool; ``(loss, n_labelled)``.

    Per-atom binary relevance rather than a softmax over the pool, because the
    distant signal is not one-of-n: a question's evidence sessions can make many
    atoms relevant at once, and a softmax would force them to compete for one
    unit of probability mass they do not share.
    """
    labels = torch.as_tensor(example["labels"], dtype=torch.float32)
    if labels.numel() == 0:
        return torch.zeros((), dtype=torch.float32), 0
    logits = scorer(
        torch.as_tensor(example["features"], dtype=torch.float32),
        torch.as_tensor(example["edge_index"], dtype=torch.long),
        torch.as_tensor(example["edge_rel"], dtype=torch.long),
        torch.as_tensor(example["query"], dtype=torch.float32),
    )
    # `pos_weight` from this example's own balance: the distant signal is sparse
    # (a handful of relevant atoms in a pool of up to 64), and unweighted BCE
    # reaches a low loss by predicting "irrelevant" everywhere -- which would
    # score well and retrieve nothing.
    positives = float(labels.sum().item())
    negatives = float(labels.numel() - positives)
    weight = torch.tensor(max(negatives, 1.0) / max(positives, 1.0), dtype=torch.float32)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)
    return loss, int(labels.numel())


def train_scorer(
    scorer: PoolScorer,
    examples_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    seed: int,
    budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train on the item-2 distant signal under the shared budget.

    Each example is a prepared dict — ``features``, ``edge_index``, ``edge_rel``,
    ``query``, ``labels`` — so this function never touches a snapshot, a channel
    or a gold field.  ``recall.distant_labels`` produces the labels; see this
    module's docstring for why that boundary is where it is.

    **The budget is read from ``pins.TRAINING``, never from an argument default**,
    so a caller cannot quietly give the scorer more epochs than Stage B's arms
    got; ``budget`` exists so tests can run a two-epoch version, and it is
    reported when it differs.

    Returns the history and the restored-best dev loss, in the shape
    ``train_d1_arm`` returns, so both are readable by one reporting path.
    """
    budget = dict(budget or TRAINING)
    torch.manual_seed(int(seed))
    optimiser = torch.optim.Adam(
        scorer.parameters(),
        lr=float(budget["lr"]),
        weight_decay=float(budget["weight_decay"]),
    )
    train_examples = list(examples_by_split.get("train", ()))
    dev_examples = list(examples_by_split.get("dev", ()))

    def _split_loss(examples: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, int]:
        total = torch.zeros((), dtype=torch.float32)
        scored = 0
        for example in examples:
            loss, n = _example_loss(scorer, example)
            if n:
                total = total + loss
                scored += 1
        return (total / scored if scored else total), scored

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(scorer.state_dict())
    best_dev = math.inf
    patience = int(budget["early_stop_patience"])
    since_best = 0
    n_train = 0

    for epoch in range(int(budget["epochs"])):
        scorer.train()
        optimiser.zero_grad()
        train_loss, n_train = _split_loss(train_examples)
        if n_train:
            train_loss.backward()
            optimiser.step()

        scorer.eval()
        with torch.no_grad():
            dev_loss, n_dev = _split_loss(dev_examples)
        dev_value = float(dev_loss.item()) if n_dev else math.inf
        history.append(
            {"epoch": epoch, "train_loss": float(train_loss.item()), "dev_loss": dev_value}
        )

        if dev_value < best_dev - 1e-6:
            best_dev, since_best = dev_value, 0
            best_state = copy.deepcopy(scorer.state_dict())
        else:
            since_best += 1
            if since_best >= patience:
                break

    if not math.isfinite(best_dev):
        # Phase 6's guard, and its reasoning transfers exactly: a scorer that
        # never saw a scorable dev item is a *random* scorer, and returning it as
        # "early stopped" would put noise into the fusion as a sixth channel
        # while every report read as a completed training run.
        raise ValueError(
            f"the scorer has no scorable dev example ({len(dev_examples)} dev "
            f"examples, {n_train} train examples scored): early stopping cannot "
            "select, so training would return the random initialisation. Enlarge "
            "the split or the labelled set."
        )
    scorer.load_state_dict(best_state)  # restores, never merely stops
    scorer.eval()
    return {
        "seed": int(seed),
        "epochs_run": len(history),
        "best_dev_loss": best_dev,
        "history": history,
        "parameters": parameter_count(scorer),
        "train_examples_scored": n_train,
        "budget": {k: v for k, v in budget.items() if k != "seeds"},
    }
