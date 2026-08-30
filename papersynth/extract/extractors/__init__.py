"""Bundled extractors. Importing this package self-registers each one."""

from __future__ import annotations

from papersynth.extract.extractors.algorithm import AlgorithmExtractor
from papersynth.extract.extractors.equation import EquationExtractor
from papersynth.extract.extractors.hyperparameter import HyperparameterExtractor
from papersynth.extract.extractors.method import MethodExtractor
from papersynth.extract.extractors.result import ResultExtractor

__all__ = [
    "AlgorithmExtractor",
    "EquationExtractor",
    "HyperparameterExtractor",
    "MethodExtractor",
    "ResultExtractor",
]
