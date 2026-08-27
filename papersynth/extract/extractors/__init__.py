"""Bundled extractors. Importing this package self-registers each one."""

from __future__ import annotations

from papersynth.extract.extractors.hyperparameter import HyperparameterExtractor

__all__ = ["HyperparameterExtractor"]
