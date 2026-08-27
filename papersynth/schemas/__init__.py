"""JSON Schema loading and validation.

Schemas are the system's public contract (DD-06). They live as .json files next
to this module so that non-Python consumers can read them directly, and are
loaded here into a `referencing` registry so cross-file ``$ref``s resolve.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

SCHEMA_DIR = Path(__file__).parent

# Every schema file is registered under BOTH its absolute $id and its bare
# filename, so `{"$ref": "common.schema.json#/$defs/provenance"}` resolves the
# same way as the fully-qualified URI.
_BASE_URI = "https://papersynth.dev/schemas/"


@lru_cache(maxsize=1)
def _registry() -> Registry:
    registry: Registry = Registry()
    for path in sorted(SCHEMA_DIR.glob("*.json")):
        contents = json.loads(path.read_text(encoding="utf-8"))
        resource = Resource.from_contents(contents, default_specification=DRAFT202012)
        registry = registry.with_resource(uri=_BASE_URI + path.name, resource=resource)
        registry = registry.with_resource(uri=path.name, resource=resource)
    return registry


@lru_cache(maxsize=32)
def load_schema(name: str) -> dict[str, Any]:
    """Load a schema by filename, e.g. ``load_schema("claim.schema.json")``."""
    path = SCHEMA_DIR / name
    if not path.exists():
        available = ", ".join(sorted(p.name for p in SCHEMA_DIR.glob("*.json")))
        raise FileNotFoundError(f"No schema named {name!r}. Available: {available}")
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@lru_cache(maxsize=32)
def validator_for(name: str) -> Draft202012Validator:
    """Return a cached validator for a named schema, with $refs wired up."""
    # Each schema declares an absolute $id under _BASE_URI, so a relative ref
    # like "common.schema.json#/$defs/provenance" resolves against it without
    # any explicit resolver wiring.
    return Draft202012Validator(load_schema(name), registry=_registry())


def validate(instance: Any, schema_name: str) -> list[str]:
    """Validate ``instance``; return a list of human-readable error strings.

    Returns an empty list when valid. Errors are returned rather than raised so
    callers can aggregate across many objects (one bad claim must not abort a
    whole run - NFR-09).
    """
    errors = []
    for err in sorted(validator_for(schema_name).iter_errors(instance), key=lambda e: list(e.path)):
        location = "/".join(str(p) for p in err.path) or "<root>"
        errors.append(f"{location}: {err.message}")
    return errors


def assert_valid(instance: Any, schema_name: str) -> None:
    """Validate and raise on failure. For internal invariants only."""
    from papersynth.core.errors import SchemaValidationError

    errors = validate(instance, schema_name)
    if errors:
        raise SchemaValidationError(schema_name, errors)


__all__ = ["SCHEMA_DIR", "assert_valid", "load_schema", "validate", "validator_for"]
