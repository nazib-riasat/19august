"""Determinism, run identity and the manifest.

Discharges Phase-0 exit criteria 9 (two runs with the same config and seed
produce identical manifests apart from timestamp) and 10 (a dirty tree is
flagged).
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess

import pytest

from graft.config import config_hash, load_config
from graft.runtime import ENV_KEY, REPRO_KEY, git_info, new_run_dir, run_manifest, set_seed

HAS_GIT = shutil.which("git") is not None


def test_manifests_agree_on_the_reproducibility_block():
    """Criterion 9.  The block is deliberately free of hostname, OS and clock, so
    two *different machines* running the same condition also agree — which is
    what makes a teammate's result comparable rather than merely similar."""
    cfg = load_config()
    first = run_manifest(cfg, seed=13)
    second = run_manifest(cfg, seed=13)
    assert first[REPRO_KEY] == second[REPRO_KEY]


def test_the_environment_block_carries_what_is_allowed_to_differ():
    manifest = run_manifest(load_config(), seed=13)
    env = manifest[ENV_KEY]
    assert set(env) == {
        "utc_started", "hostname", "platform", "processor", "cpu_count", "gpu"
    }
    assert REPRO_KEY in manifest and "config_hash" in manifest[REPRO_KEY]


def test_a_different_seed_changes_the_reproducibility_block():
    cfg = load_config()
    assert run_manifest(cfg, 13)[REPRO_KEY] != run_manifest(cfg, 42)[REPRO_KEY]


def test_a_different_config_changes_the_reproducibility_block():
    a = run_manifest(load_config(), 13)[REPRO_KEY]
    b = run_manifest(load_config(overrides={"beta": 5.0}), 13)[REPRO_KEY]
    assert a["config_hash"] != b["config_hash"]


def test_the_manifest_is_json_serialisable():
    """It is written to disk on every run; a non-serialisable field would only
    surface at the end of a long job."""
    text = json.dumps(run_manifest(load_config(), 13), sort_keys=True)
    assert json.loads(text)[REPRO_KEY]["seed"] == 13


@pytest.mark.skipif(not HAS_GIT, reason="git not installed")
def test_a_dirty_tree_is_flagged(tmp_path):
    """Criterion 10.  A run made against uncommitted changes is not reproducible
    and must be visibly marked rather than quietly logged."""
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(a, cwd=repo, capture_output=True, check=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "t")
    (repo / "f.txt").write_text("one", encoding="utf-8")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "initial")

    clean = git_info(repo)
    assert clean["available"] is True
    assert clean["dirty"] is False
    assert len(clean["sha"]) == 40

    (repo / "f.txt").write_text("two", encoding="utf-8")
    assert git_info(repo)["dirty"] is True


def test_git_info_outside_a_repository_is_reported_not_crashed(tmp_path):
    info = git_info(tmp_path)
    assert info["available"] is False
    assert info["sha"] is None


def test_run_dir_encodes_the_condition_and_writes_the_manifest(tmp_path):
    cfg = load_config()
    directory = new_run_dir(cfg, seed=42, root=tmp_path)
    assert directory.is_dir()
    assert directory.name.endswith(f"_{config_hash(cfg, 8)}_42")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest[REPRO_KEY]["config_hash"] == config_hash(cfg)


def test_run_dirs_for_different_conditions_do_not_collide(tmp_path):
    a = new_run_dir(load_config(), 13, root=tmp_path)
    b = new_run_dir(load_config(overrides={"beta": 5.0}), 13, root=tmp_path)
    assert a != b


def test_set_seed_makes_python_and_numpy_reproducible():
    import numpy as np

    set_seed(13)
    first = (random.random(), np.random.rand())
    set_seed(13)
    assert (random.random(), np.random.rand()) == first


def test_set_seed_works_without_torch_installed():
    """Phase 0 has no ML dependency; seeding must not require one."""
    set_seed(7)
