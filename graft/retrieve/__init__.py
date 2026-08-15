"""Phase 7 — Stage C: hybrid candidate retrieval and the bounded pool builder.

A question in, a **closed, capped pool of candidate atoms** out, with measured
recall.  The plan says twice (§3.3) that this stage's purpose is *recall, not
final proof selection*: if the proof is not in the pool, nothing downstream can
recover it, and every Stage-D result in Phases 8–10 is bounded by ceiling 3,
which is this stage's number.

**The channel protocol, and one deliberate departure from G7's wording.**
G7 says every channel emits ``(atom_id, score)``.  The channels here emit
``(node_id, score)`` instead, and the mapping to atom space happens once, at pool
assembly, through :func:`graft.retrieve.pool.node_atom_id`.

The reason is that closure runs on the *graph*: turning a set of hits into a
closed pool means asking which entities and intervals their edges reference, and
that question is only answerable in node space.  Channels emitting atom ids would
force the pool builder to invert the mapping — a reverse lookup needing the
snapshot — to do the same work.  Since ``node_atom_id`` is a pure, total function
of a node id, the two spaces carry identical information and G7's protocol is
honoured in substance: one shape, ``Mapping[str, float]``, for every channel.
Recorded as a departure in ``PHASE7_DECISIONS.md`` rather than left as a silent
reading of the spec.

**Module map, in build order**

===================  ==========================================================
``pool``             P7.0 — the graph→pool mapping, closure enforced at assembly
``entity``           P7.3 — anchor → entities → their assertion-backed endpoints
``temporal``         P7.4 — the fail-open filter; it removes, never adds
``expand``           P7.5 — the depth- *and* width-bounded walk
``bm25``             P7.1 — the lexical channel, on ``bm25s``
``dense``            P7.2 — exact cosine on the shared pinned embedder
``fuse``             P7.6 — the declared arithmetic, and full assembly
``recall``           P7.7 — the two-tier instrument; the only gold-reading module
``pins``             what Phase 7 freezes, plus the Stage-C fingerprint
===================  ==========================================================

``scorer`` (P7.8) is **not built**: it trains, and ``GATE0_CONTRACT.md``'s first
line is that nothing is trained before it signs.  Its absence is a legal
configuration of the fusion, not a missing piece — the five training-free
channels stand alone by design (G6).
"""
