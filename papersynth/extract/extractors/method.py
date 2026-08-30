"""Design-decision extraction.

The claim type that makes METHOD_CONFLICT detectable. A hyperparameter carries
a value two papers can disagree about numerically; a design decision carries
none, so no value detector can ever see the disagreement between BERT using
next-sentence prediction and RoBERTa removing it - which is one of the sharpest
disagreements in that literature.

The load-bearing field is `sub_problem`: the QUESTION being answered rather
than the answer. Both "next sentence prediction" and "sentence order
prediction" answer `sentence_level_objective`. Named after the answer instead,
two papers that genuinely disagree would land in different clusters and never
be compared.
"""

from __future__ import annotations

from typing import Any, ClassVar

from papersynth.core.document import Section, StructuredDocument
from papersynth.extract.base import LLMExtractor, render_sections
from papersynth.extract.prompts import render
from papersynth.extract.registry import register

#: Sub-problem names seen in the wild, mapped to a canonical form. Alignment
#: happens on this field, so a paper calling it "sentence-level task" and one
#: calling it "inter-sentence objective" must arrive at the same key or their
#: disagreement is invisible.
CANONICAL_SUB_PROBLEMS: dict[str, str] = {
    "sentence_level_task": "sentence_level_objective",
    "inter_sentence_objective": "sentence_level_objective",
    "sentence_prediction_objective": "sentence_level_objective",
    "nsp": "sentence_level_objective",
    "pretraining_objective": "pretraining_objective",
    "pre_training_objective": "pretraining_objective",
    "training_objective": "pretraining_objective",
    "masking_strategy": "masking_strategy",
    "mask_strategy": "masking_strategy",
    "masking": "masking_strategy",
    "position_encoding": "positional_encoding",
    "positional_embedding": "positional_encoding",
    "position_embedding": "positional_encoding",
    "normalization": "normalization_placement",
    "layer_norm_placement": "normalization_placement",
    "layernorm_placement": "normalization_placement",
    "tokenization": "tokenizer",
    "tokenizer_choice": "tokenizer",
    "subword_tokenization": "tokenizer",
    "optimizer_choice": "optimizer_choice",
    "attention_mechanism": "attention_mechanism",
    "attention_type": "attention_mechanism",
    "parameter_sharing": "parameter_sharing",
    "weight_sharing": "parameter_sharing",
    "embedding_factorization": "embedding_factorization",
    "activation": "activation_function",
    "activation_fn": "activation_function",
    "lr_schedule": "learning_rate_schedule",
    "learning_rate_scheduling": "learning_rate_schedule",
    "schedule": "learning_rate_schedule",
}

_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sub_problem": {"type": "string"},
        "approach": {"type": "string"},
        "adopted": {"type": "boolean"},
        "alternatives_rejected": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": ["string", "null"]},
        "applies_to": {"type": "string"},
        "condition": {"type": ["string", "null"]},
        "stated_explicitly": {"type": "boolean"},
        "quote": {"type": "string"},
    },
    "required": ["sub_problem", "approach", "adopted", "quote"],
}


@register
class MethodExtractor(LLMExtractor):
    claim_type: ClassVar[str] = "method"
    version: ClassVar[str] = "1.0.0"
    payload_schema_name: ClassVar[str] = "payload.method.json"
    output_schema: ClassVar[dict[str, Any]] = {"type": "array", "items": _ITEM_SCHEMA}
    looks_for: ClassVar[str] = (
        "design decisions - which approach was taken to a sub-problem: objective, "
        "architecture choice, tokenizer, enforcement mechanism, evaluation protocol"
    )
    section_pattern: ClassVar[str] = (
        r"method|model|architecture|approach|objective|pre-?training|training|"
        r"setup|design|ablation|analysis|implementation"
    )
    system_prompt: ClassVar[str] = (
        "You extract design decisions from research papers: which approach was "
        "taken to which sub-problem. You record removals and rejections as "
        "decisions in their own right, and you never infer a decision from a "
        "paper's silence."
    )

    def build_prompt(self, doc: StructuredDocument, sections: list[Section]) -> str:
        return render("extract_method.md", sections=render_sections(doc, sections))

    def normalize_payload(self, payload: dict[str, Any], doc: StructuredDocument) -> dict[str, Any]:
        raw = str(payload.get("sub_problem", "")).strip().lower()
        raw = raw.replace(" ", "_").replace("-", "_")
        payload["sub_problem"] = CANONICAL_SUB_PROBLEMS.get(raw, raw)

        payload["approach"] = str(payload.get("approach", "")).strip()
        payload.setdefault("applies_to", "global")
        payload.setdefault("condition", None)
        payload.setdefault("rationale", None)
        payload.setdefault("alternatives_rejected", [])
        payload.setdefault("stated_explicitly", True)

        # "own" is the safe default: a decision wrongly kept is visible in the
        # conflict list and can be dismissed, while one wrongly discarded as
        # background is simply absent and nobody knows to look for it.
        attribution = str(payload.get("attribution", "own")).strip().lower()
        payload["attribution"] = attribution if attribution == "prior_work" else "own"

        # A model that omits `adopted` is describing something the paper uses;
        # a rejection is never the silent default.
        payload["adopted"] = bool(payload.get("adopted", True))
        return payload
