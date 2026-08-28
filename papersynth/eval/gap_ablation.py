"""Ablation harness for gap detection (sections 13.1, 13.3).

Two numbers, and the second matters as much as the first.

RECALL is the self-labelling half. Start from a spec that states everything an
implementer needs, delete one field, and check the gap appears. Ground truth is
exactly what was deleted, so no annotation is needed and the measure cannot
drift. Target is 0.80 (section 13.1).

FALSE POSITIVE RATE is measured on the untouched complete spec, where the
correct answer is zero gaps. Any gap raised there is invented. This is tracked
because a noisy gap list gets skimmed and then ignored, at which point the real
entries are invisible too - the same failure R-02 describes for contradictions.
A harness that only measured recall would reward a detector that reports
everything.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from papersynth.core import ids
from papersynth.core.models import Claim, Gap, Provenance
from papersynth.gapcheck import AdversarialGapAgent, Checklist
from papersynth.llm.base import LLMProvider

#: A spec stating everything the checklist requires and everything an
#: implementer commonly has to guess. The reference point for both measures:
#: gaps raised against it are false positives by construction.
COMPLETE_SPEC: dict[str, Any] = {
    "objective": "Implement a masked language model with a transformer encoder.",
    "components": [
        {
            "component_id": "cmp_global",
            "name": "Global configuration",
            "role": "Configuration applying across the implementation.",
            "hyperparameters": [
                {
                    "canonical_name": "learning_rate",
                    "value": 0.0001,
                    "unit": None,
                    "condition": "base model",
                },
                {
                    "canonical_name": "batch_size",
                    "value": 256,
                    "unit": "sequences",
                    "condition": "base model",
                },
                {
                    "canonical_name": "num_steps",
                    "value": 1000000,
                    "unit": "steps",
                    "condition": None,
                },
                {
                    "canonical_name": "warmup_steps",
                    "value": 10000,
                    "unit": "steps",
                    "condition": None,
                },
                {"canonical_name": "dropout", "value": 0.1, "unit": None, "condition": None},
                {"canonical_name": "weight_decay", "value": 0.01, "unit": None, "condition": None},
                {"canonical_name": "optimizer", "value": "Adam", "unit": None, "condition": None},
                {
                    "canonical_name": "weight_initialization",
                    "value": "N(0, 0.02)",
                    "unit": None,
                    "condition": None,
                },
                {"canonical_name": "num_layers", "value": 12, "unit": None, "condition": None},
                {"canonical_name": "hidden_dim", "value": 768, "unit": None, "condition": None},
                {"canonical_name": "num_heads", "value": 12, "unit": None, "condition": None},
                {
                    "canonical_name": "sequence_length",
                    "value": 512,
                    "unit": "tokens",
                    "condition": None,
                },
                {"canonical_name": "vocab_size", "value": 30522, "unit": None, "condition": None},
                {"canonical_name": "random_seed", "value": 42, "unit": None, "condition": None},
                {"canonical_name": "gradient_clip", "value": 1.0, "unit": None, "condition": None},
                {"canonical_name": "num_epochs", "value": 40, "unit": "epochs", "condition": None},
                # Design decisions, not just numbers. The first run of this
                # harness reported eleven "false positives" against a spec
                # holding only hyperparameters - activation_function,
                # layernorm_epsilon, the special token IDs, the tokenizer.
                # None were invented: a masked language model cannot be
                # implemented from numbers alone, and the reference point
                # was simply not complete. A false-positive measure is only
                # as honest as the spec it declares sufficient.
                {
                    "canonical_name": "activation_function",
                    "value": "GeLU",
                    "unit": None,
                    "condition": None,
                },
                {
                    "canonical_name": "layernorm_epsilon",
                    "value": 1e-12,
                    "unit": None,
                    "condition": None,
                },
                {
                    "canonical_name": "layernorm_placement",
                    "value": "post-residual",
                    "unit": None,
                    "condition": None,
                },
                {
                    "canonical_name": "tokenizer_type",
                    "value": "WordPiece",
                    "unit": None,
                    "condition": None,
                },
                {
                    "canonical_name": "masking_strategy",
                    "value": "15% of tokens: 80% [MASK], 10% random, 10% unchanged",
                    "unit": None,
                    "condition": None,
                },
                {
                    "canonical_name": "position_embedding_type",
                    "value": "learned absolute",
                    "unit": None,
                    "condition": None,
                },
                {
                    "canonical_name": "embedding_tying",
                    "value": "input and output embeddings tied",
                    "unit": None,
                    "condition": None,
                },
                {
                    "canonical_name": "attention_mask_convention",
                    "value": "1 for real tokens, 0 for padding",
                    "unit": None,
                    "condition": None,
                },
                {"canonical_name": "cls_token_id", "value": 101, "unit": None, "condition": None},
                {"canonical_name": "sep_token_id", "value": 102, "unit": None, "condition": None},
                {"canonical_name": "mask_token_id", "value": 103, "unit": None, "condition": None},
                {"canonical_name": "pad_token_id", "value": 0, "unit": None, "condition": None},
                {
                    "canonical_name": "sentence_level_objective",
                    "value": "next sentence prediction",
                    "unit": None,
                    "condition": None,
                },
                {
                    "canonical_name": "loss_function",
                    "value": "cross-entropy over masked positions",
                    "unit": None,
                    "condition": None,
                },
                {
                    "canonical_name": "data_shuffling",
                    "value": "shuffled each epoch",
                    "unit": None,
                    "condition": None,
                },
                {
                    "canonical_name": "checkpoint_selection",
                    "value": "final training step",
                    "unit": None,
                    "condition": None,
                },
                # Round two of the same exercise: the audit next asked for the
                # feed-forward width, the decay schedule after warmup, and how
                # the two pretraining losses are weighted. All three were
                # genuinely absent again.
                {
                    "canonical_name": "intermediate_dim",
                    "value": 3072,
                    "unit": None,
                    "condition": None,
                },
                {
                    "canonical_name": "lr_scheduler_decay",
                    "value": "linear decay to zero after warmup",
                    "unit": None,
                    "condition": None,
                },
                {
                    "canonical_name": "total_loss_weighting",
                    "value": "MLM and NSP losses summed with equal weight",
                    "unit": None,
                    "condition": None,
                },
            ],
            "equations": [],
        }
    ],
    "open_conflicts": [],
    "missing_but_critical": [],
}

#: Fields the harness deletes one at a time, each named by what should then be
#: reported missing and the values that must go for that to be true.
#:
#: The tuple form exists for one_of requirements. Deleting num_steps alone
#: proves nothing while num_epochs remains, because the requirement is
#: genuinely still satisfied - scoring that as a miss would have marked correct
#: behaviour as a failure and pushed the checklist in the wrong direction.
ABLATABLE_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("learning_rate", ("learning_rate",)),
    ("batch_size", ("batch_size",)),
    ("optimizer", ("optimizer",)),
    ("weight_decay", ("weight_decay",)),
    ("dropout", ("dropout",)),
    ("weight_initialization", ("weight_initialization",)),
    ("warmup_steps", ("warmup_steps",)),
    ("random_seed", ("random_seed",)),
    ("num_steps_or_epochs", ("num_steps", "num_epochs")),
)


@dataclass
class GapEvalReport:
    """Both halves of the measure, kept together so neither is read alone."""

    detected: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        total = len(self.detected) + len(self.missed)
        return len(self.detected) / total if total else 0.0

    @property
    def false_positive_count(self) -> int:
        return len(self.false_positives)

    @property
    def false_positive_rate(self) -> float:
        """Invented gaps per ablation trial.

        Expressed per trial rather than as a share of reported gaps, because on
        a complete spec the denominator would be the false positives
        themselves, which always gives 1.0 and says nothing.
        """
        trials = len(self.detected) + len(self.missed)
        return self.false_positive_count / trials if trials else 0.0

    def render(self) -> str:
        lines = [
            f"gap recall            {self.recall:.2f}  "
            f"({len(self.detected)}/{len(self.detected) + len(self.missed)} ablated "
            "fields recovered)",
            f"false positives       {self.false_positive_count} on a complete spec "
            "(correct answer is 0)",
        ]
        if self.missed:
            lines.append(f"  missed: {', '.join(sorted(self.missed))}")
        if self.false_positives:
            lines.append(f"  invented: {', '.join(sorted(self.false_positives))}")
        return "\n".join(lines)


def ablate(spec: dict[str, Any], *field_names: str) -> dict[str, Any]:
    """Return a copy of the spec with the named hyperparameters removed."""
    removing = set(field_names)
    out = deepcopy(spec)
    for component in out.get("components", []):
        component["hyperparameters"] = [
            h for h in component.get("hyperparameters", []) if h["canonical_name"] not in removing
        ]
    return out


def claims_for(spec: dict[str, Any], paper_id: str = "eval_paper") -> list[Claim]:
    """Verified claims matching a spec, so both gap passes see the same corpus.

    Without these the checklist would report every field missing regardless of
    what the spec says, and the ablation would measure nothing.
    """
    out = []
    for component in spec.get("components", []):
        for hyperparameter in component.get("hyperparameters", []):
            payload = {
                "canonical_name": hyperparameter["canonical_name"],
                "paper_symbol": None,
                "value": hyperparameter["value"],
                "value_type": "float" if isinstance(hyperparameter["value"], float) else "int",
                "unit": hyperparameter.get("unit"),
                "applies_to": "global",
                "condition": hyperparameter.get("condition"),
                "stated_explicitly": True,
            }
            claim = Claim.build(
                paper_id=paper_id,
                claim_type="hyperparameter",
                provenance=Provenance(
                    paper_id=paper_id,
                    span_id=f"{paper_id}#s1.p0.0",
                    section="Training Setup",
                    page=1,
                    char_start=0,
                    char_end=40,
                    quote_hash=ids.quote_hash(str(hyperparameter["value"])),
                    extraction_method="llm",
                    extractor_version="hyperparameter@1.0.0",
                    confidence=0.95,
                ),
                payload=payload,
            )
            claim.status = "verified"
            out.append(claim)
    return out


def evaluate_gaps(
    provider: LLMProvider,
    *,
    checklist: Checklist | None = None,
    fields: tuple[tuple[str, tuple[str, ...]], ...] = ABLATABLE_FIELDS,
    spec: dict[str, Any] | None = None,
    adversarial: bool = True,
) -> GapEvalReport:
    """Measure gap recall by ablation and false positives on a complete spec."""
    spec = spec or COMPLETE_SPEC
    checklist = checklist or Checklist.load("config/implementability_checklist.yaml")
    agent = AdversarialGapAgent(provider) if adversarial else None
    report = GapEvalReport()

    # False positives first, on the untouched spec, where zero is correct.
    for gap in _run_passes(spec, checklist, agent):
        report.false_positives.append(gap.field)

    for expected, removed in fields:
        ablated = ablate(spec, *removed)
        found = {g.field for g in _run_passes(ablated, checklist, agent)}
        if _recovered(expected, found) or any(_recovered(r, found) for r in removed):
            report.detected.append(expected)
        else:
            report.missed.append(expected)
            report.notes.append(f"{expected}: reported {sorted(found) or 'nothing'}")

    return report


def _run_passes(
    spec: dict[str, Any], checklist: Checklist, agent: AdversarialGapAgent | None
) -> list[Gap]:
    claims = claims_for(spec)
    paper_ids = ["eval_paper"]
    gaps = checklist.audit(claims, paper_ids=paper_ids)

    if agent is not None:
        gaps = gaps + agent.audit(
            spec, claims=claims, existing=gaps, disputed=set(), paper_ids=paper_ids
        )
    return gaps


def _recovered(field_name: str, found: set[str]) -> bool:
    """Whether the deleted field was reported, under any reasonable name."""
    from rapidfuzz import fuzz

    target = field_name.replace("_", " ")
    return any(fuzz.token_set_ratio(target, reported.replace("_", " ")) >= 85 for reported in found)
