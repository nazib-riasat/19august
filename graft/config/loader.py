"""Load a YAML override file onto the code defaults, validate, and hash.

The hash is computed over the **resolved config**, never over the YAML text.
That is what lets a comment, an indentation change, or a Windows CRLF checkout
leave an experiment's identity untouched — which matters because this repo is
edited on several machines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from graft.canonical import canonical_bytes, sha256_hex
from graft.config.schema import Config, ConfigError, UWeights, validate

__all__ = ["load_config", "config_hash", "PRESETS", "preset_path"]

_PRESET_DIR = Path(__file__).parent / "presets"

PRESETS = ("default", "synthetic")

_TOP_LEVEL_KEYS = set(Config().to_dict())
_WEIGHT_KEYS = set(UWeights().to_dict())


def preset_path(name: str) -> Path:
    """Absolute path of a shipped preset.

    Presets live inside the package rather than beside it so that they travel
    with a ``pip install -e .`` and are importable from a Kaggle notebook
    without knowing where the checkout landed.
    """
    path = _PRESET_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"no preset {name!r}; available: {', '.join(sorted(PRESETS))}"
        )
    return path


def _reject_unknown(data: dict[str, Any]) -> None:
    """A typo in a YAML key must raise, not silently do nothing."""
    unknown = sorted(set(data) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(
            f"unknown config keys: {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(_TOP_LEVEL_KEYS))}"
        )
    weights = data.get("u_weights")
    if weights is not None:
        if not isinstance(weights, dict):
            raise ConfigError("u_weights must be a mapping")
        unknown_w = sorted(set(weights) - _WEIGHT_KEYS)
        if unknown_w:
            raise ConfigError(f"unknown u_weights keys: {', '.join(unknown_w)}")


def load_config(
    path: str | Path | None = None,
    *,
    preset: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Build a validated :class:`Config`.

    ``path`` and ``preset`` are alternative ways to name a YAML override file;
    with neither, the code defaults are used unchanged.  ``overrides`` is
    applied last and is intended for tests and sweeps, not for experiments.
    """
    if path is not None and preset is not None:
        raise ConfigError("pass either path or preset, not both")
    if preset is not None:
        path = preset_path(preset)

    data: dict[str, Any] = {}
    if path is not None:
        text = Path(path).read_text(encoding="utf-8")
        loaded = yaml.safe_load(text)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
        data = loaded

    if overrides:
        data = {**data, **overrides}

    _reject_unknown(data)

    # u_weights overrides merge onto the default weights rather than replacing
    # them, so a YAML that tunes one weight does not silently zero the other five.
    weights = UWeights()
    if "u_weights" in data:
        merged = {**weights.to_dict(), **data.pop("u_weights")}
        weights = UWeights.from_dict(merged)

    cfg = Config(u_weights=weights, **data)
    validate(cfg)
    return cfg


def config_hash(cfg: Config, length: int | None = None) -> str:
    """SHA-256 of the resolved config.  This is the identity of a condition."""
    full = sha256_hex(canonical_bytes(cfg.to_dict()))
    return full if length is None else full[:length]
