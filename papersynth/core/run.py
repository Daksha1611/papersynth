"""Run orchestration and the artifact workspace (sections 4.2, 4.3).

Every stage writes its artifact before the next begins. That is what makes a
run inspectable after the fact and resumable after a crash - and it is why the
intermediate formats are YAML and JSON rather than pickles (constraint 3.3).

Stages 0-2 are per paper. One paper failing does not abort the run (NFR-09): it
degrades to a partial spec with a recorded warning, because a spec covering two
of three papers with that fact stated is more useful than no spec at all.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

import papersynth
from papersynth.align import Aligner
from papersynth.contradict import ContradictionScan, attach_paper_support
from papersynth.core.config import Settings, get_settings
from papersynth.core.document import StructuredDocument
from papersynth.core.errors import PaperSynthError
from papersynth.core.ledger import Ledger
from papersynth.core.models import (
    Claim,
    ClaimSet,
    Contradiction,
    Gap,
    ReconciliationResult,
    utcnow,
)
from papersynth.extract import registry
from papersynth.gapcheck import AdversarialGapAgent, Checklist
from papersynth.llm.base import LLMProvider
from papersynth.reconcile import Policy, PolicyEngine
from papersynth.synth import SpecBuilder, SpecValidator, render_review
from papersynth.verify import RangeRules, VerificationReport, Verifier

STAGES = (
    "ingest",
    "extract",
    "verify",
    "align",
    "contradict",
    "reconcile",
    "gapcheck",
    "synth",
)


@dataclass
class RunResult:
    run_id: str
    documents: list[StructuredDocument] = field(default_factory=list)
    claims: dict[str, Claim] = field(default_factory=dict)
    contradictions: list[Contradiction] = field(default_factory=list)
    reconciliation: ReconciliationResult | None = None
    gaps: list[Gap] = field(default_factory=list)
    reports: list[VerificationReport] = field(default_factory=list)
    #: Canonical names carrying an unresolved conflict. A disputed value is not
    #: a missing one, and reporting it as both doubles the review list.
    disputed_fields: set[str] = field(default_factory=set)
    spec: dict[str, Any] | None = None
    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    provenance_completeness: float = 1.0
    #: Papers the caller asked for, which exceeds len(documents) when one
    #: could not be fetched or parsed.
    papers_requested: int = 0

    @property
    def open_conflicts(self) -> list[dict[str, Any]]:
        return list(self.spec.get("open_conflicts", [])) if self.spec else []

    @property
    def status(self) -> str:
        if self.blocking:
            return "awaiting_review"
        return "ready" if self.spec else "failed"


class Workspace:
    """runs/{run_id}/ - one directory per stage, everything inspectable."""

    def __init__(self, root: Path | str, run_id: str) -> None:
        self.root = Path(root) / run_id
        self.run_id = run_id

    def ensure(self) -> Workspace:
        for sub in ("00_documents", "01_claims", "02_verified"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        return self

    @property
    def ledger_path(self) -> Path:
        return self.root / "ledger.jsonl"

    def write_yaml(self, name: str, payload: Any) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )
        return path

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path


class Pipeline:
    """Runs stages 1-7 over already-ingested documents."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        settings: Settings | None = None,
        workspace: Workspace | None = None,
        extractors: list[str] | None = None,
        entailment: bool = True,
        split_gate: bool = False,
        embedding_merges: bool = False,
        adversarial_gaps: bool = False,
        resume: bool = False,
        ledger: Ledger | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider
        self.workspace = workspace
        self.extractors = extractors or ["hyperparameter"]
        self.entailment = entailment
        #: Off by default, on the measurement rather than on principle. On
        #: BERT/RoBERTa/ALBERT the gate split batch_size into three concepts
        #: and lost the genuine BERT-256 against RoBERTa-8000 disagreement,
        #: which is the corpus's headline finding, while the splits it got
        #: right (hidden_dim into three model variants) removed no false
        #: contradiction because the condition grouping had already kept those
        #: apart. One real finding lost, none gained.
        #:
        #: It remains worth enabling where its designed job actually arises:
        #: alongside embedding merges, where it correctly rejected all five
        #: proposals including num_steps merged with warmup_steps.
        self.split_gate = split_gate
        self.embedding_merges = embedding_merges
        #: Pass B. One call, run against the assembled spec rather than the
        #: papers, so the questions it raises are ones a real implementer would
        #: actually face.
        self.adversarial_gaps = adversarial_gaps
        #: Reuse per-paper artifacts already on disk (FR-15, NFR-05). Extract
        #: and verify are where a run's calls actually go, so skipping papers
        #: already done is what makes a resumed run affordable rather than a
        #: second full run.
        self.resume = resume
        self.ledger = ledger or Ledger()

    def run(
        self,
        documents: list[StructuredDocument],
        *,
        objective: str,
        run_id: str | None = None,
        reviewer: str | None = None,
        papers_requested: int | None = None,
    ) -> RunResult:
        run_id = run_id or self.workspace.run_id if self.workspace else "run_local"
        result = RunResult(run_id=run_id, documents=list(documents))
        result.papers_requested = papers_requested or len(documents)

        if self.workspace:
            self.workspace.ensure()
            self._write_manifest(documents, objective)
            for doc in documents:
                self.workspace.write_json(
                    f"00_documents/{_safe(doc.paper_id)}.json", doc.model_dump()
                )

        verified_sets = self._per_paper(documents, result)
        result.claims = {c.claim_id: c for cs in verified_sets for c in cs.claims}

        graph, alignment = Aligner(
            threshold=self.settings.align_threshold,
            provider=self.provider if self.split_gate else None,
            embedding_merges=self.embedding_merges,
        ).align(verified_sets)
        if self.workspace:
            self.workspace.write_json("03_concept_graph.json", graph.model_dump())

        result.contradictions = self._detect(graph, documents)
        result.disputed_fields = {
            cluster.canonical_name
            for cluster in graph.clusters
            if any(c.cluster_id == cluster.cluster_id for c in result.contradictions)
        }
        if self.workspace:
            self.workspace.write_yaml(
                "04_contradictions.yaml",
                [c.model_dump() for c in result.contradictions],
            )

        result.reconciliation = self._reconcile(result.contradictions)
        result.gaps = self._gapcheck(result)
        if self.workspace:
            self.workspace.write_yaml("05_reconciliation.yaml", result.reconciliation.model_dump())
            self.workspace.write_yaml("06_gaps.yaml", [g.model_dump() for g in result.gaps])

        self._synthesize(result, objective=objective, reviewer=reviewer)
        result.warnings.extend(alignment.notes)
        return result

    # -- stages ------------------------------------------------------------

    def _per_paper(self, documents: list[StructuredDocument], result: RunResult) -> list[ClaimSet]:
        extractors = registry.build(
            self.extractors, self.provider, temperature=self.settings.temperature
        )
        rules = RangeRules.load(self.settings.range_rules)
        verifier = Verifier(
            provider=self.provider,
            range_rules=rules,
            settings=self.settings,
            entailment=self.entailment,
        )

        verified_sets: list[ClaimSet] = []
        for doc in documents:
            done = self._resume_paper(doc)
            if done is not None:
                verified_sets.append(done)
                result.reports.append(
                    VerificationReport(
                        paper_id=doc.paper_id,
                        total=len(done.claims),
                        verified=len(done.verified),
                        rejected=len(done.rejected),
                    )
                )
                result.warnings.append(f"{doc.paper_id}: reused verified claims from disk")
                continue

            try:
                extraction = registry.run_all(doc, extractors)
            except PaperSynthError as exc:
                # A partial spec with a visible warning beats no spec (NFR-09).
                result.warnings.append(f"{doc.paper_id}: extraction failed ({exc}); skipped")
                continue

            result.warnings.extend(extraction.warnings)
            claim_set = ClaimSet(paper_id=doc.paper_id, claims=extraction.claims)
            if self.workspace:
                self.workspace.write_yaml(
                    f"01_claims/{_safe(doc.paper_id)}.yaml", claim_set.model_dump()
                )

            verified, report = verifier.verify(claim_set, doc)
            result.reports.append(report)
            verified_sets.append(verified)

            if self.workspace:
                self.workspace.write_yaml(
                    f"02_verified/{_safe(doc.paper_id)}.yaml", verified.model_dump()
                )

        if self.workspace:
            # Named in section 4.3 and previously never written. Without it a
            # rebuilt spec reported zero claims examined while simultaneously
            # listing fifty-two contributed - two contradictory counts in the
            # deliverable, from the artifact that was missing rather than from
            # any disagreement in the data.
            self.workspace.write_json(
                "02_verified/verification_report.json",
                [asdict(r) for r in result.reports],
            )
        return verified_sets

    def _resume_paper(self, doc: StructuredDocument) -> ClaimSet | None:
        """Verified claims for this paper from a previous run, if any.

        Resumption is per paper rather than per stage because that is where
        the granularity actually helps: a run interrupted by exhausted quotas
        has some papers fully extracted and verified, and others untouched.
        The corpus stages downstream are rerun regardless, since they are
        cheap next to extraction and depend on the full claim set.

        A partially written or unreadable artifact is treated as absent. Doing
        the work again costs calls; trusting a truncated file would put
        half a paper's claims into the spec and call it complete.
        """
        if not self.resume or self.workspace is None:
            return None

        path = self.workspace.root / "02_verified" / f"{_safe(doc.paper_id)}.yaml"
        if not path.exists():
            return None

        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            claim_set = ClaimSet.model_validate(payload)
        except (yaml.YAMLError, ValueError, OSError):
            return None

        return claim_set if claim_set.claims else None

    def _detect(self, graph: Any, documents: list[StructuredDocument]) -> list[Contradiction]:
        metadata = {d.paper_id: (d.venue, d.year) for d in documents}
        found = ContradictionScan().run(graph)
        # Primacy is deliberately left unset. See _primary_sources below.
        return [attach_paper_support(c, metadata, {}) for c in found]

    def _reconcile(self, contradictions: list[Contradiction]) -> ReconciliationResult:
        policy = Policy.load(self.settings.policy)
        from papersynth.contradict import DETECTORS

        auto_resolvable = {
            conflict_type: getattr(detector, "auto_resolvable", True)
            for conflict_type, detector in DETECTORS.items()
        }
        return PolicyEngine(policy, auto_resolvable=auto_resolvable).resolve(contradictions)

    def _gapcheck(self, result: RunResult) -> list[Gap]:
        """Stage 6, Pass A. What the corpus does not supply.

        Runs against the whole corpus rather than per paper: a value stated in
        any one paper satisfies the requirement, and reporting it as missing
        because a different paper omitted it would be false.
        """
        checklist = Checklist.load(self.settings.checklist)
        return checklist.audit(
            list(result.claims.values()),
            paper_ids=[d.paper_id for d in result.documents],
            component_id=None,
        )

    def _synthesize(self, result: RunResult, *, objective: str, reviewer: str | None) -> None:
        builder = SpecBuilder(
            run_id=result.run_id,
            objective=objective,
            documents=result.documents,
            claims=result.claims,
            papers_requested=result.papers_requested,
        )

        def assemble() -> dict[str, Any]:
            return builder.build(
                contradictions=result.contradictions,
                reconciliation=result.reconciliation,
                gaps=result.gaps,
                reports=result.reports,
                reviewer=reviewer,
            )

        spec = assemble()

        if self.adversarial_gaps:
            # Pass B needs the assembled spec, which only exists at stage 7, so
            # the spec is built once for the audit and rebuilt with whatever it
            # finds. Assembly costs no model calls, so the second build is free.
            found = self._adversarial_pass(spec, result)
            if found:
                result.gaps.extend(found)
                if self.workspace:
                    self.workspace.write_yaml("06_gaps.yaml", [g.model_dump() for g in result.gaps])
                spec = assemble()

        validator = SpecValidator(result.claims)
        report = validator.validate(
            spec,
            contradictions=result.contradictions,
            reconciliation=result.reconciliation,
        )
        spec["verification_summary"]["provenance_completeness"] = report.provenance_completeness
        result.provenance_completeness = report.provenance_completeness
        result.blocking = report.blocking_conflicts
        result.spec = spec

        review = render_review(
            spec,
            contradictions=result.contradictions,
            reconciliation=result.reconciliation,
            gaps=result.gaps,
            blocking=report.blocking_conflicts,
        )
        if self.workspace:
            self.workspace.write_text("SPEC_REVIEW.md", review)
            if report.blocking_conflicts:
                # Emission is gated, but the reviewer still needs the evidence.
                # The draft goes to a distinct filename so nothing downstream
                # can mistake it for an emittable spec.
                self.workspace.write_yaml("implementation_spec.draft.yaml", spec)
            else:
                report.raise_first()
                self.workspace.write_yaml("implementation_spec.yaml", spec)

    def _adversarial_pass(self, spec: dict[str, Any], result: RunResult) -> list[Gap]:
        """Ask what an implementer would have to guess. Never fatal.

        A failed audit costs the gaps this pass would have found; aborting the
        run would cost the whole spec, including the gaps Pass A already found
        and every claim behind them.
        """
        try:
            return AdversarialGapAgent(self.provider).audit(
                spec,
                claims=list(result.claims.values()),
                existing=result.gaps,
                disputed=result.disputed_fields,
                paper_ids=[d.paper_id for d in result.documents],
            )
        except PaperSynthError as exc:
            result.warnings.append(f"adversarial gap pass failed: {exc}")
            return []

    def _write_manifest(self, documents: list[StructuredDocument], objective: str) -> None:
        if not self.workspace:
            return
        self.workspace.write_yaml(
            "manifest.yaml",
            {
                "run_id": self.workspace.run_id,
                "created_at": utcnow(),
                "papersynth_version": papersynth.__version__,
                "spec_version": papersynth.SPEC_VERSION,
                "objective": objective,
                "papers": [
                    {"paper_id": d.paper_id, "sha256": d.sha256, "ingest": d.ingest_method}
                    for d in documents
                ],
                "extractors": self.extractors,
                "stages": list(STAGES),
                # Everything that can change model output, so a spec differing
                # from a prior run is attributable to config rather than drift.
                "config": self.settings.reproducibility_fingerprint(),
            },
        )


def _primary_sources(graph: Any, documents: list[StructuredDocument]) -> dict[str, str]:
    """Which paper originally introduced each concept. Not inferable yet.

    The obvious heuristic - earliest paper in the cluster wins - was tried and
    removed, because it is wrong in the common case and wrong invisibly. Paper
    A predating Paper B does not make A's learning rate authoritative for B's
    model; a hyperparameter is not a concept anyone introduces. Yet primacy
    feeds `prefer_primary_source`, so that heuristic silently auto-resolved a
    genuine disagreement in favour of whichever paper happened to be older.

    That is R-10 exactly: a policy rule encoding a subtle systematic bias. The
    rule stays in the policy, because for DEFINITION_CONFLICT primacy is
    meaningful and a real signal may exist later - a citation graph, or an
    explicit "we adopt the formulation of [12]". Until then this returns
    nothing, the rule never fires, and such conflicts escalate to a human.
    Absent beats guessed.
    """
    return {}


def _safe(paper_id: str) -> str:
    return paper_id.replace("/", "_").replace(":", "_")
