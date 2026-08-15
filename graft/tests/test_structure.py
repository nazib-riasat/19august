"""Structural rules that keep later phases buildable.

Discharges Phase-0 exit criteria 11 (Phase 1 depends on the GraphSnapshot
protocol, never on a concrete store), 12 (only ``schemas.py`` defines a
dataclass that crosses a module boundary) and 13 (Phase 0 imports no ML
library).

These are the tests that fail *later*, when someone takes a shortcut, which is
exactly why they are worth having now.
"""

from __future__ import annotations

import ast
import importlib
import json
import pkgutil
import sys
from pathlib import Path

import pytest

import graft

PACKAGE_ROOT = Path(graft.__file__).parent
REPO_ROOT = PACKAGE_ROOT.parent


#: Packages that may import an ML library, and the phase that opened each.
#: Everything else in ``graft`` must stay importable on a bare interpreter.
#:
#: **This list is a narrowing, not a weakening** (Phase-3 P3.0).  Phase 3 needs
#: torch for its learners, so a blanket "``graft`` imports no ML library" test
#: would have had to be deleted the moment the first learner was written.
#: Deleting it would also have removed the guard on ``graft.core`` and
#: ``graft.synth``, where the rule still matters: Phase-2 exit criterion 21
#: requires the exact evaluator to run without a GPU stack, and Phase 1's whole
#: prohibition is that nothing learned may reach ``H``.
#:
#: **Phase 5 widened this by one package, deliberately** (`GRAFT_PHASE5_BUILD.md`
#: decision 13): ``graft.ingest`` runs an extractor LLM and an NLI cross-encoder.
#: Two guards go with the widening rather than after it —
#: ``test_the_ingest_package_does_not_import_ml_eagerly`` keeps the *import* of
#: ``graft.ingest`` free of torch and transformers, so ``verify_handoff.py``
#: still runs on a bare interpreter, and the per-package rules on ``graft.core``
#: and ``graft.synth`` are untouched.
#:
#: **Phase 6 widened it again by one** (`GRAFT_PHASE6_BUILD.md` decision 1):
#: ``graft.graphbuild`` runs the encoders, and E2/E3 need ``torch_geometric`` —
#: which is *already* on ``ML_LIBRARIES`` below, so admitting the package was a
#: deliberate one-line edit against a failing test, exactly the intended shape.
#: The same containment applies: importing ``graft.graphbuild`` pulls in no ML
#: library, so the Stage-B fingerprint and the commit validator stay usable on a
#: bare interpreter.
#:
#: **Phase 7 widened it a fourth time** (`GRAFT_PHASE7_BUILD.md` decision 1, gap
#: G2): ``graft.retrieve`` runs ``bm25s`` for the lexical channel, the shared
#: pinned embedder for the dense one, and will run torch for the GNN scorer when
#: Gate 0 signs.  ``bm25s`` was already on ``ML_LIBRARIES`` below — listed there
#: by Phase 0 *before* anything needed it — so admitting the package is again the
#: deliberate one-line edit against a failing test.  Containment is stricter here
#: than for the other three, because Stage C's five training-free channels are
#: the part that has to run today: importing any non-scorer ``graft.retrieve``
#: module must pull in **no torch at all**, which
#: ``test_the_retrieve_package_does_not_import_torch`` asserts module by module.
ML_ALLOWED_PACKAGES = ("graft.setgen", "graft.ingest", "graft.graphbuild", "graft.retrieve")


def _source_files(subdir: str | None = None, *, ml_allowed: bool = False) -> list[Path]:
    root = PACKAGE_ROOT if subdir is None else PACKAGE_ROOT / subdir
    files = sorted(p for p in root.rglob("*.py") if "tests" not in p.parts)
    if ml_allowed:
        return files
    allowed = tuple(pkg.split(".", 1)[1] for pkg in ML_ALLOWED_PACKAGES)
    return [p for p in files if not any(part in allowed for part in p.parts)]


def _module_names() -> list[str]:
    names = ["graft"]
    for info in pkgutil.walk_packages([str(PACKAGE_ROOT)], prefix="graft."):
        if ".tests" in info.name:
            continue
        names.append(info.name)
    return names


# -- criterion 13 -----------------------------------------------------------

ML_LIBRARIES = (
    "torch",
    "torch_geometric",
    "transformers",
    "sentence_transformers",
    "bitsandbytes",
    "pcst_fast",
    "bm25s",
)


def test_the_phase_0_to_2_surface_imports_no_ml_library():
    """Phase-0 criterion 13, narrowed by Phase-3 P3.0 rather than deleted.

    Everything outside :data:`ML_ALLOWED_PACKAGES` must stay importable on a bare
    interpreter — Kaggle before any heavy install, and Phase-2 exit criterion 21,
    which keeps the exact evaluator usable without a GPU stack.

    Run in a **subprocess** because the check is on ``sys.modules``: importing
    ``graft.setgen`` in this process to test something else would put torch there
    and make the assertion unfalsifiable. A clean interpreter is the only honest
    way to ask "does *this* subset pull in an ML library".
    """
    import subprocess

    names = [n for n in _module_names() if not n.startswith(ML_ALLOWED_PACKAGES)]
    program = (
        "import importlib, sys, json\n"
        f"for n in {names!r}:\n"
        "    importlib.import_module(n)\n"
        f"print(json.dumps(sorted(l for l in {list(ML_LIBRARIES)!r} if l in sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    leaked = json.loads(result.stdout.strip().splitlines()[-1])
    assert not leaked, f"the Phase 0-2 surface imported ML libraries: {leaked}"


def test_the_ml_boundary_is_a_narrowing_not_a_hole():
    """The allowance is four packages, and each was opened by a named decision.

    A future phase widening this list is making a decision; a future phase
    widening it *silently* is removing the guard that keeps ``H`` free of
    anything learned (v1.2 §4.4).  Phase 3 opened ``graft.setgen`` (P3.0),
    Phase 5 opened ``graft.ingest`` (decision 13), Phase 6 opened
    ``graft.graphbuild`` (decision 1) and Phase 7 opened ``graft.retrieve``
    (decision 1, gap G2); a fifth entry needs a fifth written decision.
    """
    assert ML_ALLOWED_PACKAGES == (
        "graft.setgen",
        "graft.ingest",
        "graft.graphbuild",
        "graft.retrieve",
    )
    for pkg in ML_ALLOWED_PACKAGES:
        importlib.import_module(pkg)


def test_the_ingest_package_does_not_import_ml_eagerly():
    """Phase-5 decision 13's other half.

    ``graft.ingest`` *may* import torch and transformers; importing the package
    must not *do* it.  ``scripts/verify_handoff.py`` prints the ingestion
    fingerprint, ``graft.ingest.pins`` carries every frozen value, and the
    grounding ladder and the time resolver are pure Python — all of which have to
    stay usable on a bare interpreter and inside a Kaggle notebook before any
    heavy install.  The model wrappers therefore import lazily, inside function
    bodies, and this is the test that says so.

    A subprocess again, for the same reason as the Phase-0-to-2 check: asking
    ``sys.modules`` inside a process that has already imported torch for some
    other test proves nothing.
    """
    import subprocess

    modules = [
        "graft.ingest",
        "graft.ingest.pins",
        "graft.ingest.prompts",
        "graft.ingest.records",
        "graft.ingest.corpus",
        "graft.ingest.grounding",
        "graft.ingest.timeexpr",
        "graft.ingest.summary",
        "graft.ingest.support",
        "graft.ingest.nli",
        "graft.ingest.extractor",
        "graft.ingest.oblparse",
        "graft.ingest.pipeline",
        "graft.ingest.bakeoff",
    ]
    program = (
        "import importlib, sys, json\n"
        f"for n in {modules!r}:\n"
        "    importlib.import_module(n)\n"
        "import graft.ingest as I\n"
        "I.ingestion_fingerprint()\n"
        f"print(json.dumps(sorted(l for l in {list(ML_LIBRARIES)!r} if l in sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    leaked = json.loads(result.stdout.strip().splitlines()[-1])
    assert not leaked, f"importing graft.ingest pulled in ML libraries: {leaked}"


def test_ingest_defines_no_cross_module_dataclass():
    """Criterion 12 held over the newly ML-allowed package.

    ``_source_files()`` skips ML-allowed packages, so widening the allowance
    would otherwise have quietly exempted ``graft.ingest`` from the rule that
    ``schemas.py`` is the single home of the data model.  Phase 5's transient
    shapes are ``__slots__`` classes for exactly this reason (see
    ``graft/ingest/records.py``), and Tier B is frozen, so there is no reason for
    a dataclass to appear here.
    """
    offenders: list[str] = []
    for path in sorted((PACKAGE_ROOT / "ingest").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = getattr(target, "id", None) or getattr(target, "attr", None)
                if name == "dataclass":
                    offenders.append(f"{path.relative_to(REPO_ROOT)}::{node.name}")
    assert not offenders, "dataclasses defined in graft.ingest: " + ", ".join(offenders)


@pytest.mark.parametrize(
    "path",
    sorted((PACKAGE_ROOT / "ingest").rglob("*.py")),
    ids=lambda p: p.name,
)
def test_every_ingest_module_has_a_docstring(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert ast.get_docstring(tree), f"{path.relative_to(REPO_ROOT)} has no module docstring"


def test_torch_is_only_referenced_lazily():
    """`runtime.set_seed` may use torch when it is present, but only from inside
    a function body — never at module scope.

    Scoped to everything outside :data:`ML_ALLOWED_PACKAGES`: Phase 3's learners
    import torch at module scope deliberately, which is what the narrowing in
    P3.0 permits."""
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module scope only
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            offending = [n for n in names if n in ML_LIBRARIES]
            assert not offending, f"{path.name} imports {offending} at module scope"


# -- criterion 11 -----------------------------------------------------------

CONCRETE_STORES = {"ReplayGraphStore", "DictGraphSnapshot"}


def test_phase1_namespace_depends_only_on_the_protocol():
    """Criterion 11.  If ``graft.core`` imported a concrete store, Phase 1 would
    block on Phase 6 and the 'build Phases 1-4 with zero real data' strategy
    would collapse.  Trivially true while ``core`` is empty; it becomes a real
    guard the moment the checker is written."""
    for path in _source_files("core"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("graft"):
                imported = {a.name for a in node.names}
                forbidden = imported & CONCRETE_STORES
                assert not forbidden, (
                    f"{path.relative_to(REPO_ROOT)} imports a concrete graph store "
                    f"{sorted(forbidden)}; Phase 1 must depend on the GraphSnapshot "
                    "protocol alone"
                )


#: The full import surface ``graft.core`` is allowed.  Phase 1's handoff contract
#: plus ``AtomPool``, and nothing that could carry a learned score.
CORE_ALLOWED_GRAFT_MODULES = {
    "graft",
    "graft.canonical",
    "graft.config",
    "graft.core",
    "graft.core.checker",
    "graft.core.incremental",
    "graft.core.masks",
    "graft.core.obligations",
    "graft.core.resolve",
    "graft.core.reward",
    "graft.core.utility",
    "graft.graphstore",
    "graft.ids",
    "graft.ledger",
    "graft.schemas",
}


def test_the_deterministic_core_imports_nothing_it_should_not():
    """Phase-1 exit criterion 15.

    v1.2 §4.4's prohibition — *nothing learned may enter ``H``* — as a property
    of the import graph rather than a promise in a docstring.  If a learned
    scorer could reach ``H``, ``H`` would stop being a predicate, ``1[H]`` would
    stop being a hard gate, and the multiplicative safety argument of §4.1 would
    degrade into a soft threshold with a tunable cutoff.
    """
    for path in _source_files("core"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                root = module.split(".")[0]
                assert root not in ML_LIBRARIES, (
                    f"{path.relative_to(REPO_ROOT)} imports {module}; the checker "
                    "may not reach a model of any kind"
                )
                if root == "graft":
                    assert module in CORE_ALLOWED_GRAFT_MODULES, (
                        f"{path.relative_to(REPO_ROOT)} imports {module}, which is "
                        "outside the Phase-1 handoff contract"
                    )


def test_the_core_never_touches_a_concrete_graph_store():
    """Criterion 11 again, now that ``graft.core`` has contents.

    ``DictGraphSnapshot`` is permitted nowhere in the core: Phase 1 depends on
    the protocol, and importing the replay-backed store would make it block on
    Phase 6.
    """
    for path in _source_files("core"):
        source = path.read_text(encoding="utf-8")
        for forbidden in CONCRETE_STORES:
            assert forbidden not in source, (
                f"{path.relative_to(REPO_ROOT)} mentions {forbidden}; the core is "
                "written against the GraphSnapshot protocol alone"
            )


def test_every_declared_phase_package_exists():
    """The build order depends on these namespaces being reserved up front."""
    expected = {
        "core", "synth", "setgen", "ingest", "graphbuild",
        "retrieve", "gate", "reader", "baselines", "diagnostics",
    }
    for name in expected:
        module = importlib.import_module(f"graft.{name}")
        assert module.__doc__, f"graft.{name} has no docstring naming its phase"


# -- criterion 12 -----------------------------------------------------------

#: ``schemas.py`` is the single home for the data model.  Three exceptions are
#: allowed and each is deliberate:
#:   config/schema.py — the config tree, which is a different kind of object and
#:     is hashed rather than logged;
#:   eventlog.py — ``Event``, which is the log's own line format;
#:   ledger.py / graphstore.py — no dataclasses at all, listed so a future one
#:     has to be argued for.
DATACLASS_ALLOWED = {"schemas.py", "schema.py", "eventlog.py"}


def test_only_schemas_defines_cross_module_dataclasses():
    """Criterion 12."""
    offenders: list[str] = []
    for path in _source_files():
        if path.name in DATACLASS_ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = getattr(target, "id", None) or getattr(target, "attr", None)
                if name == "dataclass":
                    offenders.append(f"{path.relative_to(REPO_ROOT)}::{node.name}")
    assert not offenders, (
        "dataclasses defined outside schemas.py: " + ", ".join(offenders)
    )


# -- the Phase-1 handoff contract ------------------------------------------


def test_the_handoff_contract_is_exactly_what_phase1_was_promised():
    """Phase 1 writes H, U, R, masks and the obligation parser against these five
    imports and nothing else.  A sixth means Phase 0 is incomplete."""
    promised = {
        "ProofSet", "CandidateAtom", "Obligations", "Interval",  # graft.schemas
        "GraphSnapshot",                                          # graft.graphstore
        "canon_set_hash",                                         # graft.ids
        "Ledger",                                                 # graft.ledger
        "Config",                                                 # graft.config
    }
    assert promised <= set(graft.__all__)
    for name in promised:
        assert getattr(graft, name) is not None


# -- naming discipline ------------------------------------------------------


FORBIDDEN_FIELD_NAMES = {"true", "verified", "is_true", "correct", "is_correct", "factual"}


def test_no_schema_field_claims_truth():
    """Grounding is not truth.  ``externally_verified`` names a source relation —
    corroborated outside the conversation — and is allowed; a field meaning
    'this statement is true' is not, because the schema must be able to hold a
    span-entailed falsehood."""
    from graft import schemas

    tree = ast.parse(Path(schemas.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                assert stmt.target.id not in FORBIDDEN_FIELD_NAMES, (
                    f"{node.name}.{stmt.target.id} asserts truth; grounding is not truth"
                )


def test_the_live_plans_do_not_contradict_themselves():
    """Run ``scripts/check_plan_consistency.py`` as part of the suite.

    The plan documents restate each decision in four or five places, and five
    review rounds showed that a fix reliably lands in three of them.  Seven of
    the eleven defects found in the fifth round were introduced by the fourth
    round's own corrections.  The code has needed one round for the same volume
    of work, because tests execute it and nothing executes prose.

    This is the nearest substitute, and it belongs in the suite rather than in a
    habit: a retired wording that reappears fails the build.
    """
    import subprocess

    script = REPO_ROOT / "scripts" / "check_plan_consistency.py"
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_every_module_has_a_docstring(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert ast.get_docstring(tree), f"{path.relative_to(REPO_ROOT)} has no module docstring"


def test_the_graphbuild_package_does_not_import_ml_eagerly():
    """Phase-6 decision 1's containment half, mirroring the Phase-5 guard.

    ``graft.graphbuild`` *may* import torch and torch_geometric; importing the
    package must not *do* it.  ``verify_handoff.py`` prints the Stage-B
    fingerprint, ``graphbuild.pins`` carries every frozen value, and the commit
    validator decides what may enter the active graph — all of which have to stay
    usable on a bare interpreter, and the last of which sits on the same side of
    the line as ``H``.
    """
    import subprocess

    modules = [
        "graft.graphbuild",
        "graft.graphbuild.pins",
        "graft.graphbuild.validate",
        "graft.graphbuild.items",
        "graft.graphbuild.candidates",
        "graft.graphbuild.commit",
        "graft.graphbuild.loaders",
        "graft.graphbuild.gate1",
        "graft.graphbuild.llmlink",
    ]
    program = (
        "import importlib, sys, json\n"
        f"for n in {modules!r}:\n"
        "    importlib.import_module(n)\n"
        "from graft.graphbuild.pins import stage_b_fingerprint\n"
        "stage_b_fingerprint()\n"
        f"print(json.dumps(sorted(l for l in {list(ML_LIBRARIES)!r} if l in sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    leaked = json.loads(result.stdout.strip().splitlines()[-1])
    assert not leaked, f"importing graft.graphbuild pulled in ML libraries: {leaked}"


def test_graphbuild_defines_no_cross_module_dataclass():
    """Criterion 12 held over the third ML-allowed package, for the same reason it
    was held over the second: widening the allowance must not quietly exempt a
    package from the rule that ``schemas.py`` is the single home of the data
    model."""
    offenders: list[str] = []
    for path in sorted((PACKAGE_ROOT / "graphbuild").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = getattr(target, "id", None) or getattr(target, "attr", None)
                if name == "dataclass":
                    offenders.append(f"{path.relative_to(REPO_ROOT)}::{node.name}")
    assert not offenders, "dataclasses defined in graft.graphbuild: " + ", ".join(offenders)


@pytest.mark.parametrize(
    "path",
    sorted((PACKAGE_ROOT / "graphbuild").rglob("*.py")),
    ids=lambda p: p.name,
)
def test_every_graphbuild_module_has_a_docstring(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert ast.get_docstring(tree), f"{path.relative_to(REPO_ROOT)} has no module docstring"


#: Every ``graft.retrieve`` module that is **not** the GNN scorer.  Phase 7's
#: five channels are training-free by design (G6: "the channel stack must run and
#: be measured *without* the scorer"), which is only true if none of them drags
#: torch in.  Listed explicitly rather than globbed so that adding a module is a
#: decision about which side of the line it falls on.
RETRIEVE_NON_SCORER_MODULES = (
    "graft.retrieve",
    "graft.retrieve.pins",
    "graft.retrieve.pool",
    "graft.retrieve.bm25",
    "graft.retrieve.dense",
    "graft.retrieve.entity",
    "graft.retrieve.temporal",
    "graft.retrieve.expand",
    "graft.retrieve.fuse",
    "graft.retrieve.recall",
)


def test_the_retrieve_package_does_not_import_torch():
    """Phase-7 decision 1's containment half — stricter than the other three.

    ``graft.ingest`` and ``graft.graphbuild`` promise only that *importing the
    package* is torch-free.  Stage C promises more, because G6 makes the
    five-channel stack the thing that runs and is measured **today**, while the
    scorer waits on the Gate-0 signature: every non-scorer module must be
    importable with no torch anywhere in ``sys.modules``.

    ``bm25s`` and the shared embedder are *permitted* here — that is what
    decision 1 bought — so this checks torch and torch_geometric specifically
    rather than all of :data:`ML_LIBRARIES`.  ``graft.retrieve.dense`` reaches
    the pinned encoder through ``graphbuild.embed``, whose model loading is
    already lazy; if that ever regresses, the dense channel stops being runnable
    on a bare interpreter and this is the test that says so.
    """
    import subprocess

    program = (
        "import importlib, sys, json\n"
        f"for n in {list(RETRIEVE_NON_SCORER_MODULES)!r}:\n"
        "    importlib.import_module(n)\n"
        "from graft.retrieve.pins import stage_c_fingerprint\n"
        "stage_c_fingerprint()\n"
        "print(json.dumps(sorted(l for l in ['torch', 'torch_geometric'] if l in sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    leaked = json.loads(result.stdout.strip().splitlines()[-1])
    assert not leaked, f"a non-scorer graft.retrieve module pulled in {leaked}"


def test_retrieve_defines_no_cross_module_dataclass():
    """Criterion 12 held over the fourth ML-allowed package.

    Same reason as the third: ``_source_files()`` skips ML-allowed packages, so
    widening the allowance would quietly exempt ``graft.retrieve`` from the rule
    that ``schemas.py`` is the single home of the data model.  Stage C's transient
    shapes are plain tuples and dicts, and the one type that crosses a module
    boundary — ``CandidateAtom`` — is Phase 0's, unchanged.
    """
    offenders: list[str] = []
    for path in sorted((PACKAGE_ROOT / "retrieve").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = getattr(target, "id", None) or getattr(target, "attr", None)
                if name == "dataclass":
                    offenders.append(f"{path.relative_to(REPO_ROOT)}::{node.name}")
    assert not offenders, "dataclasses defined in graft.retrieve: " + ", ".join(offenders)


@pytest.mark.parametrize(
    "path",
    sorted((PACKAGE_ROOT / "retrieve").rglob("*.py")),
    ids=lambda p: p.name,
)
def test_every_retrieve_module_has_a_docstring(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert ast.get_docstring(tree), f"{path.relative_to(REPO_ROOT)} has no module docstring"
