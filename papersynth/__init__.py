"""PaperSynth - multi-paper implementation spec synthesizer.

N papers in, one verified, citation-traced, contradiction-annotated
implementation spec out. The system deliberately stops before code generation:
a human approves the spec, then hands it to a coding agent (DD-01).
"""

from __future__ import annotations

__version__ = "0.1.0"

#: Version of spec.schema.json this build emits. Semver; additive-only within
#: a minor. Downstream agents pin against this (DD-06).
SPEC_VERSION = "0.1.0"

__all__ = ["SPEC_VERSION", "__version__"]
