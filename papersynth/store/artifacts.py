"""Reading a run back off disk.

Every stage artifact is human-readable YAML or JSON (constraint 3.3), which
means a finished run can be reloaded, amended, and re-emitted without rerunning
anything upstream. That is what makes `resolve` cheap: a human decision on one
conflict re-synthesizes the spec without re-extracting a single paper.

Runs are immutable and content-addressed. Amending a run means writing new
artifacts alongside the old ones, never rewriting history - so `diff` between
two runs stays meaningful.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from papersynth.core.document import StructuredDocument
from papersynth.core.errors import PaperSynthError
from papersynth.core.models import (
    Claim,
    ClaimSet,
    Contradiction,
    Gap,
    ReconciliationResult,
)


@dataclass
class LoadedRun:
    """Everything needed to re-emit a spec without rerunning the pipeline."""

    run_id: str
    root: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    documents: list[StructuredDocument] = field(default_factory=list)
    claims: dict[str, Claim] = field(default_factory=dict)
    contradictions: list[Contradiction] = field(default_factory=list)
    reconciliation: ReconciliationResult | None = None
    gaps: list[Gap] = field(default_factory=list)
    spec: dict[str, Any] | None = None

    @property
    def objective(self) -> str:
        return str(self.manifest.get("objective", "")) or "(objective not recorded)"

    def contradiction(self, contradiction_id: str) -> Contradiction | None:
        return next(
            (c for c in self.contradictions if c.contradiction_id == contradiction_id),
            None,
        )

    def open_contradictions(self, severity: str | None = None) -> list[Contradiction]:
        """Conflicts still needing a human, worst first."""
        out = []
        for contradiction in self.contradictions:
            resolution = (
                self.reconciliation.for_contradiction(contradiction.contradiction_id)
                if self.reconciliation
                else None
            )
            if resolution is not None and not resolution.is_open:
                continue
            if severity and contradiction.severity != severity:
                continue
            out.append(contradiction)
        return out

    @property
    def blocking(self) -> list[Contradiction]:
        return [c for c in self.open_contradictions() if c.severity == "BLOCKING"]


class RunStore:
    """Loads and amends a run workspace."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def exists(self) -> bool:
        return (self.root / "manifest.yaml").exists()

    def load(self) -> LoadedRun:
        if not self.exists():
            raise PaperSynthError(
                f"{self.root} is not a PaperSynth run directory (no manifest.yaml). "
                "Pass the path printed by `papersynth run --out`."
            )

        manifest = _read_yaml(self.root / "manifest.yaml") or {}
        run = LoadedRun(
            run_id=str(manifest.get("run_id", self.root.name)),
            root=self.root,
            manifest=manifest,
        )

        for path in sorted((self.root / "00_documents").glob("*.json")):
            payload = _read_json(path)
            if payload:
                run.documents.append(StructuredDocument.model_validate(payload))

        # Verified claims are the ones downstream stages actually used; the
        # pre-verification set in 01_claims is kept for inspection only.
        for path in sorted((self.root / "02_verified").glob("*.yaml")):
            payload = _read_yaml(path)
            if not payload:
                continue
            for claim in ClaimSet.model_validate(payload).claims:
                run.claims[claim.claim_id] = claim

        run.contradictions = [
            Contradiction.model_validate(entry)
            for entry in (_read_yaml(self.root / "04_contradictions.yaml") or [])
        ]

        reconciliation = _read_yaml(self.root / "05_reconciliation.yaml")
        if reconciliation:
            run.reconciliation = ReconciliationResult.model_validate(reconciliation)

        run.gaps = [
            Gap.model_validate(entry) for entry in (_read_yaml(self.root / "06_gaps.yaml") or [])
        ]

        for name in ("implementation_spec.yaml", "implementation_spec.draft.yaml"):
            spec = _read_yaml(self.root / name)
            if spec:
                run.spec = spec
                break

        return run

    def save_reconciliation(self, reconciliation: ReconciliationResult) -> Path:
        path = self.root / "05_reconciliation.yaml"
        path.write_text(
            yaml.safe_dump(reconciliation.model_dump(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return path

    def save_spec(self, spec: dict[str, Any], *, draft: bool = False) -> Path:
        name = "implementation_spec.draft.yaml" if draft else "implementation_spec.yaml"
        path = self.root / name
        path.write_text(
            yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )
        if not draft:
            # A previously blocked draft must not linger next to a real spec;
            # two files claiming to be the deliverable is how the wrong one
            # gets handed to a coding agent.
            (self.root / "implementation_spec.draft.yaml").unlink(missing_ok=True)
        return path

    def save_review(self, text: str) -> Path:
        path = self.root / "SPEC_REVIEW.md"
        path.write_text(text, encoding="utf-8")
        return path


def _read_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PaperSynthError(f"{path} is not readable YAML: {exc}") from exc


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PaperSynthError(f"{path} is not readable JSON: {exc}") from exc
