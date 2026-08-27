"""Schema contract tests (section 14.2): valid fixtures pass, and each
required-field omission fails."""

from __future__ import annotations

import pytest

from papersynth.core.errors import SchemaValidationError
from papersynth.schemas import assert_valid, load_schema, validate

VALID_PROVENANCE = {
    "paper_id": "1706.03762",
    "span_id": "1706.03762#s3.p2.114",
    "section": "3.2 Attention",
    "page": 4,
    "char_start": 114,
    "char_end": 287,
    "quote_hash": "sha256:" + "9" * 64,
    "extraction_method": "llm",
    "extractor_version": "hyperparameter@1.2.0",
    "confidence": 0.94,
}

VALID_CLAIM = {
    "claim_id": "clm_7f3a2b",
    "paper_id": "1706.03762",
    "type": "hyperparameter",
    "status": "verified",
    "provenance": VALID_PROVENANCE,
    "verification": {
        "citation_trace": "pass",
        "symbol_check": "n/a",
        "range_check": "pass",
        "self_consistency": "3/3",
        "notes": [],
    },
    "payload": {"canonical_name": "learning_rate", "value": 0.0001},
    "confidence": 0.94,
}


def test_valid_claim_passes():
    assert validate(VALID_CLAIM, "claim.schema.json") == []


@pytest.mark.parametrize(
    "field",
    [
        "claim_id",
        "paper_id",
        "type",
        "status",
        "provenance",
        "verification",
        "payload",
        "confidence",
    ],
)
def test_each_required_claim_field_is_required(field):
    instance = {k: v for k, v in VALID_CLAIM.items() if k != field}
    assert validate(instance, "claim.schema.json"), f"omitting {field} should fail"


@pytest.mark.parametrize(
    "field",
    [
        "paper_id",
        "span_id",
        "section",
        "char_start",
        "char_end",
        "quote_hash",
        "extraction_method",
        "extractor_version",
        "confidence",
    ],
)
def test_each_required_provenance_field_is_required(field):
    prov = {k: v for k, v in VALID_PROVENANCE.items() if k != field}
    instance = {**VALID_CLAIM, "provenance": prov}
    assert validate(instance, "claim.schema.json"), f"omitting provenance.{field} should fail"


def test_claim_id_pattern_is_enforced():
    assert validate({**VALID_CLAIM, "claim_id": "clm_XYZ"}, "claim.schema.json")
    assert validate({**VALID_CLAIM, "claim_id": "7f3a2b"}, "claim.schema.json")


def test_span_id_pattern_is_enforced():
    bad = {**VALID_PROVENANCE, "span_id": "1706.03762#section3"}
    assert validate({**VALID_CLAIM, "provenance": bad}, "claim.schema.json")


def test_quote_hash_must_be_sha256_prefixed():
    """R-12: artifacts carry a hash, never the verbatim source text."""
    bad = {**VALID_PROVENANCE, "quote_hash": "the model outputs are computed as"}
    assert validate({**VALID_CLAIM, "provenance": bad}, "claim.schema.json")


def test_cross_file_refs_resolve():
    """common.schema.json#/$defs/provenance must actually be reachable."""
    errors = validate({**VALID_CLAIM, "provenance": {"paper_id": "x"}}, "claim.schema.json")
    assert errors, "a $ref that silently failed to resolve would validate anything"


def test_unknown_top_level_field_is_rejected():
    assert validate({**VALID_CLAIM, "surprise": 1}, "claim.schema.json")


def test_assert_valid_raises_typed_error():
    with pytest.raises(SchemaValidationError) as exc:
        assert_valid({"claim_id": "nope"}, "claim.schema.json")
    assert exc.value.schema_name == "claim.schema.json"
    assert exc.value.errors


def test_all_schemas_load_and_are_wellformed():
    import json
    from pathlib import Path

    from papersynth.schemas import SCHEMA_DIR

    names = sorted(p.name for p in Path(SCHEMA_DIR).glob("*.json"))
    assert names, "no schemas found"
    for name in names:
        schema = load_schema(name)
        assert "$schema" in schema, f"{name} declares no $schema"
        assert "$id" in schema, f"{name} declares no $id"
        json.dumps(schema)


class TestHyperparameterPayload:
    valid = {
        "canonical_name": "learning_rate",
        "paper_symbol": "\\eta",
        "value": 0.0001,
        "value_type": "float",
        "unit": None,
        "applies_to": "global",
        "condition": "base model, WMT14 EN-DE",
        "stated_explicitly": True,
    }

    def test_valid(self):
        assert validate(self.valid, "payload.hyperparameter.json") == []

    def test_canonical_name_must_be_snake_case(self):
        assert validate(
            {**self.valid, "canonical_name": "LearningRate"}, "payload.hyperparameter.json"
        )

    def test_categorical_value_may_be_a_string(self):
        instance = {**self.valid, "value": "cosine", "value_type": "categorical"}
        assert validate(instance, "payload.hyperparameter.json") == []

    def test_condition_is_nullable_but_present_in_schema(self):
        """ER-04 hangs off this field; it must exist even when null."""
        assert "condition" in load_schema("payload.hyperparameter.json")["properties"]


class TestSpecSchema:
    def test_open_conflicts_cannot_carry_blocking_severity(self):
        """Section 20.3: BLOCKING can never appear in an emitted spec."""
        enum = load_schema("spec.schema.json")["properties"]["open_conflicts"]["items"][
            "properties"
        ]["severity"]["enum"]
        assert "BLOCKING" not in enum
        assert enum == ["MATERIAL"]

    def test_provenance_refs_require_at_least_one_entry(self):
        """NFR-01: no spec field exists without a traceable source."""
        refs = load_schema("spec.schema.json")["$defs"]["provenance_refs"]
        assert refs["minItems"] == 1
