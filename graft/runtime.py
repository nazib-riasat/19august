"""Determinism, run identity, and artefact layout.

The manifest is split into two blocks, and the split is the point.

``reproducibility``
    config hash, git commit and dirty flag, seed, interpreter and package
    versions.  Two runs of the same condition must produce an identical block —
    **including on two different machines**.  That is what makes a teammate's
    result comparable to yours rather than merely similar, and it is the
    checkable form of the project's predeclared protocol ([ANALYSIS] — the
    predeclaration discipline is the project's own; Dror et al., ACL 2018, is
    its test-selection authority and does not prescribe it).

``environment``
    hostname, OS, CPU, GPU, start time.  Expected to differ, recorded so that a
    result which turns out to be machine-dependent can be traced.

A dirty tree is recorded explicitly.  A run made against uncommitted changes is
not reproducible and should be visibly marked rather than quietly logged.
"""

from __future__ import annotations

import math
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from graft.config import Config, config_hash
from graft.schemas import SCHEMA_VERSION

__all__ = [
    "set_seed",
    "git_info",
    "run_manifest",
    "new_run_dir",
    "json_sanitize",
    "REPRO_KEY",
    "ENV_KEY",
]

REPRO_KEY = "reproducibility"
ENV_KEY = "environment"

#: Recorded in every manifest.  Deliberately the declared dependency set rather
#: than a full freeze, so the block stays stable across machines that happen to
#: have different unrelated packages installed.
TRACKED_PACKAGES = ("PyYAML", "numpy", "pytest", "torch")


def json_sanitize(obj: Any) -> Any:
    """``NaN``/``±inf`` → ``None``, recursively, before an artefact is written.

    Python's ``json.dumps`` default (``allow_nan=True``) emits bare ``NaN``
    tokens, which are **not JSON** (RFC 8259) — ``jq``, JavaScript's
    ``JSON.parse`` and strict parsers all reject the file.  The empty-denominator
    rates this project deliberately reports as ``nan`` rather than a flattering
    ``0`` (a bakeoff candidate that errored, an empty pilot re-run) therefore
    made whole artefacts unreadable (found by the 13 Aug 2026 audit).  ``null``
    is the honest JSON spelling of "no measurement".  Scripts should pass the
    sanitized object to ``json.dumps(..., allow_nan=False)`` so any unsanitized
    path fails loudly instead of shipping an invalid artefact.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    return obj


#: The keys :func:`deterministic_view` strips — every place a wall clock or the
#: host environment leaks into an artefact.  ``latency``/``latency_ms``/
#: ``timings_ms`` are ``time.perf_counter`` readings; ``environment`` is
#: :func:`run_manifest`'s ``ENV_KEY`` section (``utc_started``, hostname, GPU).
VOLATILE_KEYS: tuple[str, ...] = ("latency", "latency_ms", "timings_ms", ENV_KEY)


def deterministic_view(obj: Any) -> Any:
    """The artefact minus its volatile keys — the surface two runs must agree on.

    Exit criterion 3 of Phase 7 asks that two runs over the same graph and
    questions produce "identical artefacts", and taken byte-literally that is
    impossible: the artefact embeds wall-clock latencies, which is itself an exit
    criterion (14).  This function is the resolution (15 Aug 2026 audit): the
    determinism claim is made — and digested, and tested — over everything
    *except* the declared volatile keys, so a timing cannot hide a real
    nondeterminism and a real nondeterminism cannot hide behind a timing.
    """
    if isinstance(obj, dict):
        return {
            k: deterministic_view(v) for k, v in obj.items() if k not in VOLATILE_KEYS
        }
    if isinstance(obj, (list, tuple)):
        return [deterministic_view(v) for v in obj]
    return obj


def set_seed(seed: int) -> None:
    """Seed every RNG this project may touch.

    ``torch`` is imported lazily and only if already installed, so that Phase 0
    keeps its "no ML dependency" property and its test suite runs on a bare
    interpreter.

    Note on hash randomisation: ``PYTHONHASHSEED`` cannot be set usefully after
    interpreter start, and this project does not rely on it — every place where
    set or dict order could reach bytes sorts first (see ``ids.canon_set_hash``
    and ``ProofSet.to_dict``).
    """
    seed = int(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a declared dependency
        pass
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - older torch builds
        pass


def _git(args: list[str], root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_info(root: str | Path | None = None) -> dict[str, Any]:
    """Commit, branch and dirty flag, or ``available: False`` outside a repo."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    sha = _git(["rev-parse", "HEAD"], root)
    if sha is None:
        return {"available": False, "sha": None, "branch": None, "dirty": None}
    status = _git(["status", "--porcelain"], root)
    return {
        "available": True,
        "sha": sha,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root),
        "dirty": bool(status),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _gpu_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch": False, "cuda": False, "devices": []}
    if not torch.cuda.is_available():
        return {"torch": True, "cuda": False, "devices": []}
    return {
        "torch": True,
        "cuda": True,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


def run_manifest(cfg: Config, seed: int, root: str | Path | None = None) -> dict[str, Any]:
    """Build the manifest recorded alongside every run."""
    return {
        REPRO_KEY: {
            "config_hash": config_hash(cfg),
            "config": cfg.to_dict(),
            "seed": int(seed),
            "git": git_info(root),
            "python": ".".join(str(p) for p in sys.version_info[:3]),
            "packages": _package_versions(),
            "schema_version": SCHEMA_VERSION,
        },
        ENV_KEY: {
            "utc_started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "gpu": _gpu_info(),
        },
    }


def new_run_dir(
    cfg: Config,
    seed: int,
    root: str | Path = "runs",
    write_manifest: bool = True,
) -> Path:
    """Create ``runs/{utc}_{config_hash[:8]}_{seed}/`` and write its manifest."""
    import json

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(root) / f"{stamp}_{config_hash(cfg, 8)}_{int(seed)}"
    directory.mkdir(parents=True, exist_ok=True)
    if write_manifest:
        manifest = run_manifest(cfg, seed)
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return directory
