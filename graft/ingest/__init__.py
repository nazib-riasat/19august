"""Phase 5 — Stage A: ingestion, extraction, span grounding, NLI verification
and the support gate that decides eligible vs quarantined.

**This is where extraction error enters the system.**  Phases 1–4 ran on a
synthetic lattice where every atom was born grounded; from here on the records
are produced by a model and carry its mistakes.  The architecture's two-layer
design (fix F9) exists for exactly this moment: the **audit layer** stores
everything — which is what makes extraction quality measurable at all — and the
**active layer** admits only assertions the support gate marks eligible, so an
invented claim can never become retrievable evidence.  `H`'s seventh sub-check
has read that eligibility flag since Phase 1; this package is what finally
*writes* it from real data.

**Stage A is hybrid, not deterministic** (plan §3.1).  Storage, hashing and
offsets are deterministic; extraction, coreference and entailment are learned and
carry error.  The naming discipline of plan §4.4 applies throughout: nothing here
verifies *truth*, and no field in this package means "this is true".

**This is the only package besides ``graft.setgen`` allowed to import an ML
library** (decision 13), and importing ``graft.ingest`` itself still pulls in
neither torch nor transformers — the model wrappers import lazily, so
``scripts/verify_handoff.py`` keeps running on a bare interpreter.

Module map, in the order the write path runs them:

===================  ==========================================================
``corpus``           LongMemEval-S → a deterministic ``Turn`` stream (P5.1)
``summary``          the synchronous rolling summary, Mem0's recipe (P5.2, G3)
``extractor``        one interface, three model backends and a replay backend (P5.3)
``grounding``        the four-rung ladder with boundary snapping (P5.4, G5)
``nli``              the pinned entailment cross-encoder (P5.5, G6)
``support``          eligible vs quarantined under ``support_policy`` (P5.6, F9)
``oblparse``         the learned obligation parser (P5.7, F2, G7)
``pipeline``         the two stage-sequential passes and the log writes (P5.8, G4)
``bakeoff``          the G2 extractor bakeoff and its predeclared rule
``pins``             every frozen value, plus the ingestion fingerprint (G11)
``prompts``          every prompt, and one SHA over all of them
``records``          the transient shapes between the model and the log
``timeexpr``         relative dates → half-open intervals, in code not prompts
===================  ==========================================================
"""

from graft.ingest.pins import ingestion_fingerprint

__all__ = ["ingestion_fingerprint"]
