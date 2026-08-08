"""GRAFT — provenance-preserving temporal graph memory with a checker-conditioned
evidence-set learner.

The names re-exported here are the **Phase-1 handoff contract**.  Phase 1 writes
``H``, ``U``, ``R``, the action masks and the obligation parser against exactly
these and nothing else.  If Phase 1 needs a sixth import, Phase 0 is incomplete
and should be fixed here rather than by widening Phase 1.
"""

from graft.config import Config
from graft.graphstore import GraphSnapshot
from graft.ids import canon_set_hash
from graft.ledger import Ledger
from graft.schemas import CandidateAtom, Interval, Obligations, ProofSet

__version__ = "0.1.0"

__all__ = [
    "Config",
    "GraphSnapshot",
    "canon_set_hash",
    "Ledger",
    "CandidateAtom",
    "Interval",
    "Obligations",
    "ProofSet",
    "__version__",
]
