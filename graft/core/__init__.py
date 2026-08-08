"""Phase 1 — the deterministic core: ``H``, ``U``, ``R``, masks, obligations.

Depends on the ``GraphSnapshot`` *protocol* only.  Importing a concrete store
here would make Phase 1 block on Phase 6 and is checked against by
``graft/tests/test_structure.py``.
"""
