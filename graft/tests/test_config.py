"""Config loading, validation and hashing.

Discharges Phase-0 exit criterion 8 (comment-only YAML edits leave the hash
unchanged; any value change alters it).
"""

from __future__ import annotations

import math

import pytest

from graft.config import (
    PRESETS,
    Config,
    ConfigError,
    UWeights,
    config_hash,
    load_config,
    preset_path,
)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


BASE = """
beta: 4.0
K: 8
checker_budget: 32
"""


def test_comment_only_edits_leave_the_hash_unchanged(tmp_path):
    """Criterion 8, first half.  The hash covers the resolved config, never the
    YAML text, so comments, whitespace and a CRLF checkout cannot change the
    identity of an experimental condition."""
    plain = _write(tmp_path, "a.yaml", BASE)
    commented = _write(
        tmp_path,
        "b.yaml",
        "# a leading comment\n\nbeta:   4.0    # trailing\n\n\nK: 8\nchecker_budget: 32\n",
    )
    assert config_hash(load_config(plain)) == config_hash(load_config(commented))


def test_crlf_line_endings_leave_the_hash_unchanged(tmp_path):
    """The team edits on Windows and runs on Kaggle; line endings must not matter."""
    lf = tmp_path / "lf.yaml"
    crlf = tmp_path / "crlf.yaml"
    lf.write_bytes(BASE.strip().encode("utf-8") + b"\n")
    crlf.write_bytes(BASE.strip().replace("\n", "\r\n").encode("utf-8") + b"\r\n")
    assert config_hash(load_config(lf)) == config_hash(load_config(crlf))


@pytest.mark.parametrize(
    "override", [{"beta": 4.5}, {"K": 4}, {"max_atoms": 8}, {"seeds": [1, 2, 3]}]
)
def test_any_value_change_alters_the_hash(override):
    """Criterion 8, second half."""
    base = load_config()
    assert config_hash(load_config(overrides=override)) != config_hash(base)


def test_int_and_float_spellings_of_a_float_agree(tmp_path):
    """`beta: 4` and `beta: 4.0` are the same condition; normalisation happens at
    the dataclass boundary, before hashing."""
    as_int = _write(tmp_path, "i.yaml", "beta: 4\n")
    as_float = _write(tmp_path, "f.yaml", "beta: 4.0\n")
    assert config_hash(load_config(as_int)) == config_hash(load_config(as_float))


def test_defaults_match_the_frozen_gate0_values():
    cfg = load_config()
    assert cfg.beta == 4.0
    assert cfg.r_fail == 1e-6
    assert cfg.K == 8
    assert cfg.checker_budget == 32
    assert cfg.pool_cap == 64
    assert cfg.max_atoms == 16
    assert cfg.tau_nli == 0.8
    assert cfg.support_policy == "strict"
    assert cfg.seeds == (13, 42, 7)
    assert cfg.u_weights.to_dict() == {
        "suff": 1.0, "cov": 0.5, "src": 0.25, "temp": 0.5, "red": 0.25, "size": 0.1
    }


def test_config_is_frozen():
    cfg = load_config()
    with pytest.raises(Exception):
        cfg.beta = 9.0  # type: ignore[misc]


def test_config_is_frozen_all_the_way_down():
    """``frozen=True`` blocks rebinding ``source_tiers``; it does nothing about
    mutating the dict behind it.  That dict feeds ``config_hash``, which is the
    identity of an experimental condition — a config able to change its own hash
    mid-run makes two runs reporting the same hash potentially different
    experiments."""
    cfg = load_config()
    before = config_hash(cfg)
    with pytest.raises(TypeError):
        cfg.source_tiers["first_party"] = 0.1
    assert config_hash(cfg) == before


def test_the_default_tier_table_cannot_be_mutated_globally():
    """It is the default every Config starts from, so a mutation here would
    change the reward of every condition built afterwards."""
    from graft.config import DEFAULT_SOURCE_TIERS

    with pytest.raises(TypeError):
        DEFAULT_SOURCE_TIERS["first_party"] = 0.1


def test_source_tiers_defaults_and_validation():
    cfg = load_config()
    assert dict(cfg.source_tiers) == {
        "first_party": 1.0, "corroborated": 0.75, "reported": 0.5, "unknown": 0.25
    }
    assert cfg.default_tier == "unknown"
    with pytest.raises(ConfigError, match="outside \\[0, 1\\]"):
        load_config(overrides={"source_tiers": {"first_party": 2.0}})
    with pytest.raises(ConfigError, match="not in source_tiers"):
        load_config(overrides={"default_tier": "made_up"})


def test_unknown_keys_raise(tmp_path):
    """A typo must not silently leave the default in place."""
    path = _write(tmp_path, "typo.yaml", "betta: 4.0\n")
    with pytest.raises(ConfigError, match="unknown config keys: betta"):
        load_config(path)


def test_unknown_weight_keys_raise(tmp_path):
    path = _write(tmp_path, "w.yaml", "u_weights:\n  sufficiency: 1.0\n")
    with pytest.raises(ConfigError, match="unknown u_weights keys"):
        load_config(path)


def test_partial_weight_override_keeps_the_other_five(tmp_path):
    path = _write(tmp_path, "w.yaml", "u_weights:\n  size: 0.2\n")
    weights = load_config(path).u_weights
    assert weights.size == 0.2
    assert weights.suff == 1.0 and weights.cov == 0.5


@pytest.mark.parametrize(
    "override,message",
    [
        ({"beta": 0.0}, "beta must be > 0"),
        ({"beta": -1.0}, "beta must be > 0"),
        ({"r_fail": 0.0}, "r_fail must be > 0"),
        ({"K": 64}, "K <= checker_budget"),
        ({"K": 0}, "1 <= K"),
        ({"max_atoms": 128}, "max_atoms <= pool_cap"),
        ({"tau_nli": 1.5}, "tau_nli must be in"),
        ({"support_policy": "loose"}, "support_policy must be 'strict'"),
        ({"seeds": [13, 42]}, "at least 3 seeds"),
        ({"seeds": [13, 13, 13]}, "seeds must be distinct"),
        ({"u_weights": {"size": -0.1}}, "u_weights.size must be >= 0"),
    ],
)
def test_validation_rejects(override, message):
    with pytest.raises(ConfigError, match=message):
        load_config(overrides=override)


def test_r_fail_must_stay_negligible_against_the_worst_valid_terminal():
    """The automated form of CLAUDE.md §7's open item.

    p*(FAIL) is bounded by r_fail / exp(beta*U_min).  A beta sweep that pushes
    the worst valid reward down towards r_fail silently promotes FAIL into a
    competitive terminal; this is the check that catches it at load time rather
    than in a Gate-2 result.
    """
    ok = load_config(overrides={"beta": 4.0})
    assert ok.r_fail < ok.r_fail_margin * ok.r_valid_min

    with pytest.raises(ConfigError, match="not negligible"):
        load_config(overrides={"beta": 20.0})


def test_r_valid_min_matches_the_reward_definition():
    cfg = load_config()
    expected = math.exp(cfg.beta * (-(cfg.u_weights.red + cfg.u_weights.size)))
    assert cfg.r_valid_min == pytest.approx(expected)


def test_u_is_bounded_so_beta_scales_a_bounded_quantity():
    """Phase-0 gap G4: every term in [0, 1], so U has a finite declared range."""
    w = UWeights()
    assert w.u_min == pytest.approx(-0.35)
    assert w.u_max == pytest.approx(2.25)


# -- environment profiles ---------------------------------------------------


@pytest.mark.parametrize("name", PRESETS)
def test_shipped_presets_load_and_validate(name):
    cfg = load_config(preset=name)
    assert cfg.profile in {"real", "synthetic"}
    assert preset_path(name).is_file()


def test_synthetic_profile_keeps_the_size_term_full_range():
    """max_atoms is both the H size limit and the size-term denominator, and it
    follows the environment.  Under a global max_atoms the lattice's size term
    would span only [0, 0.5] and U would have a different range there than on
    real data — exactly what G4 exists to prevent."""
    synthetic = load_config(preset="synthetic")
    real = load_config(preset="default")
    assert synthetic.max_atoms == 8 and synthetic.pool_cap == 32
    assert real.max_atoms == 16 and real.pool_cap == 64
    assert synthetic.u_weights == real.u_weights
    assert synthetic.beta == real.beta
    assert config_hash(synthetic) != config_hash(real)


def test_profiles_agree_on_the_shared_comparison_constants():
    """K and checker_budget are one constant used everywhere, not per-profile."""
    synthetic = load_config(preset="synthetic")
    real = load_config(preset="default")
    assert synthetic.K == real.K
    assert synthetic.checker_budget == real.checker_budget
    assert synthetic.seeds == real.seeds
    assert synthetic.r_fail == real.r_fail


def test_preset_and_path_are_mutually_exclusive(tmp_path):
    with pytest.raises(ConfigError, match="not both"):
        load_config(tmp_path / "x.yaml", preset="default")


def test_missing_preset_names_the_alternatives():
    with pytest.raises(FileNotFoundError, match="available"):
        preset_path("nope")


def test_empty_yaml_is_all_defaults(tmp_path):
    path = _write(tmp_path, "empty.yaml", "# nothing here\n")
    assert config_hash(load_config(path)) == config_hash(Config())
