"""Configuration: code defaults, YAML overrides, validation, and hashing."""

from graft.config.loader import PRESETS, config_hash, load_config, preset_path
from graft.config.schema import (
    DEFAULT_SOURCE_TIERS,
    SCHEMA_VERSION,
    Config,
    ConfigError,
    UWeights,
    validate,
)

__all__ = [
    "Config",
    "ConfigError",
    "UWeights",
    "SCHEMA_VERSION",
    "DEFAULT_SOURCE_TIERS",
    "PRESETS",
    "load_config",
    "config_hash",
    "preset_path",
    "validate",
]
