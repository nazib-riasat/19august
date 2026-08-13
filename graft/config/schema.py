"""The frozen config tree.

Defaults live here, in code.  YAML files may only *override* them.  If defaults
lived in YAML, a missing key would silently change an experiment instead of
raising.

Every value in this file is frozen at Gate 0.  Changing one after Phase 3 means
re-running experiments — see ``PHASE0_DECISIONS.md`` for what each change
costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "UWeights",
    "Config",
    "ConfigError",
    "SCHEMA_VERSION",
    "DEFAULT_SOURCE_TIERS",
]

# Distinct from ``graft.schemas.SCHEMA_VERSION``, which stamps event-log lines.
# This one is validated for equality at load.  Both moved to 0.2.0 in Phase 1:
# that one because Tier A gained types, this one because the config tree gained
# ``source_tiers`` and ``default_tier``.
SCHEMA_VERSION = "0.2.0"

#: Source-type reliability scores, in [0, 1].  Part of the reward, so frozen at
#: Gate 0 and identical across all seven learners for the same reason
#: ``u_weights`` must be (v1.2 §5.1) — otherwise the comparison measures reward
#: engineering.  Metadata-derived, never learned (v1.2 §4.4 routing table).
#:
#: Read-only: this is the default every ``Config`` starts from, so a mutation
#: here would silently change the reward of every condition built afterwards.
DEFAULT_SOURCE_TIERS: Mapping[str, float] = MappingProxyType(
    {
        "first_party": 1.0,
        "corroborated": 0.75,
        "reported": 0.5,
        "unknown": 0.25,
    }
)


class ConfigError(ValueError):
    """Raised at load time for any config that could not produce a valid run."""


@dataclass(frozen=True)
class UWeights:
    """Weights of the six terms of the utility ``U``.

    ``U = suff*sufficiency + cov*coverage + src*source_quality
          + temp*temporal_correctness - red*redundancy - size*size``

    Every term is normalised to [0, 1] (Phase-0 gap G4), so ``U`` is bounded by
    ``[-(red + size), suff + cov + src + temp]`` and ``beta`` scales a bounded
    quantity.  Without that, the Phase-3 beta sweep would be uninterpretable.

    These must be **identical across all seven learners**.  If different
    learners see different rewards, the comparison measures reward engineering
    rather than learning (research plan v1.2 §5.1).
    """

    suff: float = 1.0
    cov: float = 0.5
    src: float = 0.25
    temp: float = 0.5
    red: float = 0.25
    size: float = 0.1

    def __post_init__(self) -> None:
        for name in ("suff", "cov", "src", "temp", "red", "size"):
            object.__setattr__(self, name, float(getattr(self, name)))

    @property
    def u_min(self) -> float:
        """Lowest utility a formally valid set can reach (all penalties maxed)."""
        return -(self.red + self.size)

    @property
    def u_max(self) -> float:
        """Highest utility any set can reach (all rewards maxed, no penalty)."""
        return self.suff + self.cov + self.src + self.temp

    def to_dict(self) -> dict[str, float]:
        return {
            "suff": self.suff,
            "cov": self.cov,
            "src": self.src,
            "temp": self.temp,
            "red": self.red,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UWeights":
        return cls(**{k: float(v) for k, v in data.items()})


@dataclass(frozen=True)
class Config:
    """One resolved experimental condition.  Its hash is its identity."""

    schema_version: str = SCHEMA_VERSION

    # Names the environment profile this config describes.  It exists so that
    # the synthetic lattice and real data are visibly different conditions
    # rather than the same config with two different atom caps.
    profile: str = "real"

    # --- reward -----------------------------------------------------------
    # beta is a placeholder until the Phase-3 sweep on the synthetic lattice;
    # it is frozen immediately afterwards and before any comparison run.
    beta: float = 4.0
    u_weights: UWeights = field(default_factory=UWeights)

    # Reward of the single absorbing FAIL terminal (architecture fix F3).  FAIL
    # is a *member of the target support*, so changing this re-derives p* and
    # every Gate-2 result.
    r_fail: float = 1.0e-6

    # How far below the worst valid terminal's reward r_fail must sit.  This
    # turns "r_fail is small enough that p*(FAIL) is negligible" from a claim
    # into a load-time assertion, and it is the automated form of the open item
    # in CLAUDE.md §7 ("re-check r_fail after the beta sweep").
    r_fail_margin: float = 1.0e-3

    # --- budget -----------------------------------------------------------
    # K is the returned portfolio size AND the "returned sets" count in the
    # search comparison.  One constant, used in both places (fix F5).
    K: int = 8
    # Terminal H checks per query.  This is the primary budget axis and the
    # unit the Stage-D primary metric is defined in.
    checker_budget: int = 32

    # --- environment shape ------------------------------------------------
    # max_atoms is simultaneously the H size limit and the denominator of the
    # size term (`size = |X| / max_atoms`).  One field for both, so the two can
    # never drift apart.  It is per-profile, not global — see PHASE0_DECISIONS.
    pool_cap: int = 64
    max_atoms: int = 16

    # --- ingestion --------------------------------------------------------
    tau_nli: float = 0.8
    support_policy: str = "strict"

    # --- source quality (Phase-1 gap G7) ----------------------------------
    # `U`'s source_quality term reads these.  An atom whose Source node cannot be
    # resolved scores `default_tier` rather than 0, so a missing edge is a weak
    # source rather than a disqualifying one — disqualification is `H`'s job.
    # A factory rather than a bare default: ``dataclasses`` refuses any default
    # whose type is unhashable, and a mappingproxy is.  Handing back the module
    # constant directly is safe precisely because it is already read-only.
    source_tiers: Mapping[str, float] = field(default_factory=lambda: DEFAULT_SOURCE_TIERS)
    default_tier: str = "unknown"

    # --- protocol ---------------------------------------------------------
    # Three seeds for every trained method.  [ANALYSIS] — the multi-seed,
    # fixed-in-advance rule is this project's own protocol discipline; its
    # significance-testing authority (Dror et al., ACL 2018) covers *test
    # selection* and does not itself prescribe seed counts or predeclaration.
    # The config refuses fewer.
    seeds: tuple[int, ...] = (13, 42, 7)

    # --- runtime ----------------------------------------------------------
    # Off during Phases 1-4 (no real data at risk), on for ingestion runs.
    fsync: bool = False
    frozen_at_gate0: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "beta", float(self.beta))
        object.__setattr__(self, "r_fail", float(self.r_fail))
        object.__setattr__(self, "r_fail_margin", float(self.r_fail_margin))
        object.__setattr__(self, "tau_nli", float(self.tau_nli))
        object.__setattr__(self, "K", int(self.K))
        object.__setattr__(self, "checker_budget", int(self.checker_budget))
        object.__setattr__(self, "pool_cap", int(self.pool_cap))
        object.__setattr__(self, "max_atoms", int(self.max_atoms))
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))
        # Read-only, because ``frozen=True`` blocks rebinding the field but not
        # mutating the dict behind it — and this dict feeds ``config_hash``,
        # which is the identity of an experimental condition.  A config that can
        # change its own hash mid-run makes two runs reporting the same hash
        # potentially different experiments.
        object.__setattr__(
            self,
            "source_tiers",
            MappingProxyType({str(k): float(v) for k, v in self.source_tiers.items()}),
        )

    # -- derived quantities ------------------------------------------------

    @property
    def r_valid_min(self) -> float:
        """Reward of the worst formally valid terminal, ``exp(beta * U_min)``.

        Every valid terminal earns at least this, so ``p*(FAIL)`` is bounded
        above by ``r_fail / r_valid_min`` whenever at least one valid terminal
        exists.
        """
        return math.exp(self.beta * self.u_weights.u_min)

    def with_overrides(self, **kwargs: Any) -> "Config":
        return replace(self, **kwargs)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "beta": self.beta,
            "u_weights": self.u_weights.to_dict(),
            "r_fail": self.r_fail,
            "r_fail_margin": self.r_fail_margin,
            "K": self.K,
            "checker_budget": self.checker_budget,
            "pool_cap": self.pool_cap,
            "max_atoms": self.max_atoms,
            "tau_nli": self.tau_nli,
            "support_policy": self.support_policy,
            "source_tiers": dict(sorted(self.source_tiers.items())),
            "default_tier": self.default_tier,
            "seeds": list(self.seeds),
            "fsync": self.fsync,
            "frozen_at_gate0": self.frozen_at_gate0,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        data = dict(data)
        weights = data.pop("u_weights", None)
        cfg = cls(
            u_weights=UWeights.from_dict(weights) if weights else UWeights(),
            **data,
        )
        return cfg


def validate(cfg: Config) -> None:
    """Reject any config that could not produce a valid, comparable run.

    Every check here corresponds to something that would otherwise fail late,
    quietly, or in a way that invalidates a comparison rather than crashing.
    """
    errors: list[str] = []

    if cfg.schema_version != SCHEMA_VERSION:
        errors.append(
            f"schema_version {cfg.schema_version!r} != {SCHEMA_VERSION!r}; "
            "the config predates or postdates this code"
        )

    if not cfg.beta > 0:
        errors.append(f"beta must be > 0, got {cfg.beta}")
    if not cfg.r_fail > 0:
        errors.append(f"r_fail must be > 0, got {cfg.r_fail}")

    for name, value in cfg.u_weights.to_dict().items():
        if value < 0:
            errors.append(f"u_weights.{name} must be >= 0, got {value}")

    if not 1 <= cfg.K <= cfg.checker_budget:
        errors.append(
            f"need 1 <= K <= checker_budget, got K={cfg.K} "
            f"checker_budget={cfg.checker_budget}"
        )
    if cfg.max_atoms < 1:
        errors.append(f"max_atoms must be >= 1, got {cfg.max_atoms}")
    if cfg.max_atoms > cfg.pool_cap:
        errors.append(
            f"need max_atoms <= pool_cap, got max_atoms={cfg.max_atoms} "
            f"pool_cap={cfg.pool_cap}"
        )

    if not 0 < cfg.tau_nli <= 1:
        errors.append(f"tau_nli must be in (0, 1], got {cfg.tau_nli}")

    if not cfg.source_tiers:
        errors.append("source_tiers must not be empty; U's source_quality reads it")
    for tier, score in sorted(cfg.source_tiers.items()):
        if not 0.0 <= score <= 1.0:
            errors.append(
                f"source_tiers[{tier!r}] = {score} is outside [0, 1]; every U term "
                "must be normalised or beta scales an unbounded quantity (gap G4)"
            )
    if cfg.default_tier not in cfg.source_tiers:
        errors.append(
            f"default_tier {cfg.default_tier!r} is not in source_tiers "
            f"({sorted(cfg.source_tiers)}); an unresolvable source would have no score"
        )
    if cfg.support_policy not in {"strict"}:
        errors.append(
            f"support_policy must be 'strict', got {cfg.support_policy!r}; "
            "a looser policy lets unsupported assertions into the active graph"
        )

    if len(cfg.seeds) < 3:
        errors.append(
            f"need at least 3 seeds for the significance protocol, got {len(cfg.seeds)}"
        )
    if len(set(cfg.seeds)) != len(cfg.seeds):
        errors.append(f"seeds must be distinct, got {cfg.seeds}")

    # FAIL must stay a negligible share of the target mass.  p*(FAIL) is
    # bounded by r_fail / r_valid_min, so requiring that ratio below
    # r_fail_margin bounds p*(FAIL) by the same figure.  This is the check that
    # catches a beta sweep quietly promoting FAIL into a competitive terminal.
    ceiling = cfg.r_fail_margin * cfg.r_valid_min
    if cfg.r_fail >= ceiling:
        errors.append(
            f"r_fail={cfg.r_fail:g} is not negligible against the worst valid "
            f"terminal reward exp(beta*U_min)={cfg.r_valid_min:g}: it must be "
            f"below {ceiling:g} (r_fail_margin={cfg.r_fail_margin:g}). "
            "Lower r_fail or lower beta, and re-derive p* either way."
        )

    if errors:
        raise ConfigError("invalid config:\n  - " + "\n  - ".join(errors))
