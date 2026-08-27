"""Identifier determinism (NFR-02). Every ID is a pure function of content."""

from __future__ import annotations

from papersynth.core import ids


def test_claim_id_is_deterministic():
    payload = {"canonical_name": "learning_rate", "value": 0.0001}
    a = ids.claim_id("1706.03762", "hyperparameter", "1706.03762#s1.p0.0", payload)
    b = ids.claim_id("1706.03762", "hyperparameter", "1706.03762#s1.p0.0", dict(payload))
    assert a == b
    assert a.startswith("clm_") and len(a) == 10


def test_claim_id_is_key_order_independent():
    a = ids.claim_id("p", "hyperparameter", "p#s0.p0.0", {"x": 1, "y": 2})
    b = ids.claim_id("p", "hyperparameter", "p#s0.p0.0", {"y": 2, "x": 1})
    assert a == b


def test_claim_id_changes_with_value():
    base = {"canonical_name": "learning_rate", "value": 0.0001}
    other = {"canonical_name": "learning_rate", "value": 0.0003}
    assert ids.claim_id("p", "hyperparameter", "p#s0.p0.0", base) != ids.claim_id(
        "p", "hyperparameter", "p#s0.p0.0", other
    )


def test_claim_id_changes_with_span():
    payload = {"canonical_name": "learning_rate", "value": 0.0001}
    assert ids.claim_id("p", "hyperparameter", "p#s0.p0.0", payload) != ids.claim_id(
        "p", "hyperparameter", "p#s0.p0.5", payload
    )


def test_span_id_format():
    assert ids.span_id("1706.03762", 3, 2, 114) == "1706.03762#s3.p2.114"


def test_slugify():
    assert ids.slugify("Attention Temperature") == "attention_temperature"
    assert ids.slugify("  Multi--Head  Attention! ") == "multi_head_attention"
    assert ids.slugify("") == "unnamed"
    assert len(ids.slugify("x" * 200)) <= 40


def test_cluster_id_shape():
    cid = ids.cluster_id("hyperparameter", "attention temperature")
    assert cid == "cnc_hype_attention_temperature"


def test_contradiction_id_is_position_order_independent():
    a = ids.contradiction_id("cnc_x", "VALUE_CONFLICT", ["clm_a", "clm_b"])
    b = ids.contradiction_id("cnc_x", "VALUE_CONFLICT", ["clm_b", "clm_a"])
    assert a == b


def test_resolution_id_tracks_its_contradiction():
    assert ids.resolution_id("ctr_0031") == "res_0031"


def test_prompt_hash_changes_with_template():
    """ER-10: a prompt change must invalidate cached claims."""
    a = ids.prompt_hash("v1 template", "rendered", "llama-3.3-70b")
    b = ids.prompt_hash("v2 template", "rendered", "llama-3.3-70b")
    assert a != b


def test_prompt_hash_changes_with_model():
    a = ids.prompt_hash("t", "r", "llama-3.3-70b")
    b = ids.prompt_hash("t", "r", "gemini-2.5-flash")
    assert a != b
