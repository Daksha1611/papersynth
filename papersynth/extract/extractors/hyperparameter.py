"""Hyperparameter extraction.

Built first (section 17.2 build order) because the payload is the simplest to
verify: a value either appears in the cited span or it does not, and the range
rules catch the OCR failure mode - a lost decimal point - without any model in
the loop.
"""

from __future__ import annotations

from typing import Any, ClassVar

from papersynth.core.document import Section, StructuredDocument
from papersynth.extract.base import LLMExtractor, render_sections
from papersynth.extract.prompts import render
from papersynth.extract.registry import register
from papersynth.schemas import load_schema

#: Names seen in the wild mapped to the canonical form. This is a within-paper
#: spelling normalization, deliberately not a cross-paper symbol alignment -
#: ER-05 reserves that for the symbol map, which has both papers' context.
#: Everything here is an unambiguous synonym, never a judgement call.
CANONICAL_NAMES: dict[str, str] = {
    "lr": "learning_rate",
    "learningrate": "learning_rate",
    "learning_rate": "learning_rate",
    "base_learning_rate": "learning_rate",
    "step_size": "learning_rate",
    "bs": "batch_size",
    "batchsize": "batch_size",
    "minibatch_size": "batch_size",
    "mini_batch_size": "batch_size",
    "dropout_rate": "dropout",
    "dropout_prob": "dropout",
    "dropout_probability": "dropout",
    "p_drop": "dropout",
    "n_layers": "num_layers",
    "num_layer": "num_layers",
    "layers": "num_layers",
    "depth": "num_layers",
    "n_heads": "num_heads",
    "num_head": "num_heads",
    "heads": "num_heads",
    "attention_heads": "num_heads",
    "d_model": "hidden_dim",
    "hidden_size": "hidden_dim",
    "hidden_dimension": "hidden_dim",
    "embedding_dim": "embed_dim",
    "embedding_size": "embed_dim",
    "wd": "weight_decay",
    "l2": "weight_decay",
    "l2_regularization": "weight_decay",
    "warmup": "warmup_steps",
    "warmup_step": "warmup_steps",
    "n_epochs": "num_epochs",
    "epochs": "num_epochs",
    "training_steps": "num_steps",
    "train_steps": "num_steps",
    "max_steps": "num_steps",
    "temp": "temperature",
    "tau": "temperature",
    "beam": "beam_size",
    "beam_width": "beam_size",
    "seq_len": "sequence_length",
    "max_seq_len": "sequence_length",
    "context_length": "sequence_length",
    "vocab": "vocab_size",
    "vocabulary_size": "vocab_size",
    "clip": "gradient_clip",
    "grad_clip": "gradient_clip",
    "label_smoothing_eps": "label_smoothing",
}

_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "canonical_name": {"type": "string"},
        "paper_symbol": {"type": ["string", "null"]},
        "value": {"type": ["number", "string", "boolean"]},
        "value_type": {"type": "string", "enum": ["float", "int", "bool", "categorical"]},
        "unit": {"type": ["string", "null"]},
        "applies_to": {"type": "string"},
        "condition": {"type": ["string", "null"]},
        "stated_explicitly": {"type": "boolean"},
        "quote": {"type": "string"},
    },
    "required": ["canonical_name", "value", "value_type", "quote"],
}


@register
class HyperparameterExtractor(LLMExtractor):
    claim_type: ClassVar[str] = "hyperparameter"
    version: ClassVar[str] = "1.0.0"
    payload_schema_name: ClassVar[str] = "payload.hyperparameter.json"
    output_schema: ClassVar[dict[str, Any]] = {"type": "array", "items": _ITEM_SCHEMA}
    section_pattern: ClassVar[str] = (
        r"experiment|training|setup|implementation|method|model|detail|"
        r"configuration|hyperparam|appendix"
    )
    system_prompt: ClassVar[str] = (
        "You extract implementation details from research papers with total "
        "fidelity to the source. You never invent a value the paper does not "
        "state. You quote verbatim."
    )

    def build_prompt(self, doc: StructuredDocument, sections: list[Section]) -> str:
        return render("extract_hyperparameter.md", sections=render_sections(doc, sections))

    def normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Canonicalize names and shapes. Never fills in a missing value."""
        name = str(payload.get("canonical_name", "")).strip().lower()
        name = name.replace(" ", "_").replace("-", "_")
        payload["canonical_name"] = CANONICAL_NAMES.get(name, name)

        payload.setdefault("applies_to", "global")
        payload.setdefault("stated_explicitly", True)
        payload.setdefault("paper_symbol", None)
        payload.setdefault("unit", None)
        payload.setdefault("condition", None)

        value = payload.get("value")
        declared = payload.get("value_type")

        # A model routinely returns "0.0001" as a string. Recovering the number
        # is safe: it is the same value the paper stated, just typed correctly.
        if isinstance(value, str) and declared in ("float", "int"):
            recovered = _to_number(value)
            if recovered is not None:
                value = recovered
                payload["value"] = recovered

        if declared not in ("float", "int", "bool", "categorical"):
            payload["value_type"] = _infer_value_type(value)
        elif isinstance(value, bool):
            payload["value_type"] = "bool"
        elif isinstance(value, float) and declared == "int":
            payload["value_type"] = "float"

        return payload

    @staticmethod
    def payload_schema() -> dict[str, Any]:
        return load_schema("payload.hyperparameter.json")


def _to_number(text: str) -> float | int | None:
    cleaned = text.strip().replace(",", "").replace("−", "-")
    try:
        if cleaned.lstrip("-").isdigit():
            return int(cleaned)
        return float(cleaned)
    except ValueError:
        return None


def _infer_value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "categorical"
