"""Reported-result extraction.

Feeds two things: the `expected_results` block of the spec, which tells an
implementer what their reimplementation should produce, and RESULT_CONFLICT
detection.

Every scoping field matters more here than anywhere else in the system. A
benchmark number means nothing without the dataset, split, variant and protocol
that produced it, and ER-06 makes that a rule: results measured differently are
distinct claims, never conflicting ones. An omitted split turns a dev score and
a test score into a contradiction that does not exist.
"""

from __future__ import annotations

from typing import Any, ClassVar

from papersynth.core.document import Section, StructuredDocument
from papersynth.extract.base import LLMExtractor, render_sections
from papersynth.extract.prompts import render
from papersynth.extract.registry import register

#: Metric names normalized so two papers naming the same measurement agree.
#: Conservative on purpose: only unambiguous spellings of one metric.
CANONICAL_METRICS: dict[str, str] = {
    "bleu score": "bleu",
    "bleu-4": "bleu",
    "sacrebleu": "bleu",
    "acc": "accuracy",
    "accuracy (%)": "accuracy",
    "top-1": "top_1_accuracy",
    "top-1 accuracy": "top_1_accuracy",
    "top-5": "top_5_accuracy",
    "top-5 accuracy": "top_5_accuracy",
    "f1 score": "f1",
    "f-1": "f1",
    "macro f1": "macro_f1",
    "micro f1": "micro_f1",
    "em": "exact_match",
    "exact match": "exact_match",
    "ppl": "perplexity",
    "wer": "word_error_rate",
    "rouge-l": "rouge_l",
    "mcc": "matthews_correlation",
}

_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "metric": {"type": "string"},
        "value": {"type": "number"},
        "dataset": {"type": ["string", "null"]},
        "split": {"type": ["string", "null"]},
        "model_variant": {"type": ["string", "null"]},
        "conditions": {"type": "object"},
        "reported_variance": {"type": ["number", "null"]},
        "stated_explicitly": {"type": "boolean"},
        "quote": {"type": "string"},
    },
    "required": ["metric", "value", "quote"],
}


@register
class ResultExtractor(LLMExtractor):
    claim_type: ClassVar[str] = "result"
    version: ClassVar[str] = "1.0.0"
    payload_schema_name: ClassVar[str] = "payload.result.json"
    output_schema: ClassVar[dict[str, Any]] = {"type": "array", "items": _ITEM_SCHEMA}
    looks_for: ClassVar[str] = (
        "reported measurements: benchmark scores, accuracies, error rates, costs, "
        "with the dataset and conditions they were measured under"
    )
    section_pattern: ClassVar[str] = (
        r"result|experiment|evaluation|benchmark|analysis|ablation|comparison|appendix"
    )
    system_prompt: ClassVar[str] = (
        "You extract reported measurements from research papers. You always "
        "record the dataset, split and variant a number was measured on, "
        "because a score without them cannot be compared to anything. You "
        "never attribute a baseline row to the paper reporting it."
    )

    def build_prompt(self, doc: StructuredDocument, sections: list[Section]) -> str:
        return render("extract_result.md", sections=render_sections(doc, sections))

    def normalize_payload(self, payload: dict[str, Any], doc: StructuredDocument) -> dict[str, Any]:
        raw = str(payload.get("metric", "")).strip().lower()
        payload["metric"] = CANONICAL_METRICS.get(raw, raw.replace(" ", "_").replace("-", "_"))

        for field_name in ("dataset", "split", "model_variant"):
            value = payload.get(field_name)
            payload[field_name] = (
                str(value).strip() if isinstance(value, str) and value.strip() else None
            )

        conditions = payload.get("conditions")
        payload["conditions"] = conditions if isinstance(conditions, dict) else {}

        variance = payload.get("reported_variance")
        payload["reported_variance"] = (
            abs(float(variance))
            if isinstance(variance, int | float) and not isinstance(variance, bool)
            else None
        )
        payload.setdefault("stated_explicitly", True)
        return payload
