"""Exception hierarchy.

The distinction that matters most here is between errors that should cause the
LLM router to fall through to the next provider (rate limits, capacity) and
errors that must surface immediately because they indicate a real bug
(schema violations, content policy). See section 6.4.3.
"""

from __future__ import annotations


class PaperSynthError(Exception):
    """Base for everything raised by this package."""

    #: RFC 7807 problem type suffix, used by the HTTP layer.
    problem_type = "internal-error"
    http_status = 500


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


class IngestError(PaperSynthError):
    problem_type = "ingest-failed"
    http_status = 424


class InvalidPaperRef(PaperSynthError):
    problem_type = "invalid-paper-ref"
    http_status = 400


class SpanResolutionError(PaperSynthError):
    """A span_id does not resolve against its document."""

    problem_type = "span-unresolvable"
    http_status = 500


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


class SchemaValidationError(PaperSynthError):
    problem_type = "schema-validation-failed"
    http_status = 422

    def __init__(self, schema_name: str, errors: list[str]) -> None:
        self.schema_name = schema_name
        self.errors = errors
        detail = "; ".join(errors[:5])
        if len(errors) > 5:
            detail += f" (+{len(errors) - 5} more)"
        super().__init__(f"{schema_name}: {detail}")


class ProvenanceIncompleteError(PaperSynthError):
    """A spec field was populated without a backing claim. Always a builder bug."""

    problem_type = "provenance-incomplete"
    http_status = 409

    def __init__(self, offending_fields: list[str]) -> None:
        self.offending_fields = offending_fields
        super().__init__(
            f"{len(offending_fields)} spec field(s) lack traceable provenance: "
            + ", ".join(offending_fields[:5])
        )


class BlockingConflictsError(PaperSynthError):
    """Spec emission blocked by unresolved BLOCKING contradictions (expected, not a fault)."""

    problem_type = "blocking-conflicts"
    http_status = 409

    def __init__(self, contradiction_ids: list[str]) -> None:
        self.contradiction_ids = contradiction_ids
        super().__init__(
            f"{len(contradiction_ids)} BLOCKING contradiction(s) unresolved: "
            + ", ".join(contradiction_ids)
        )


class CyclicDependencyError(PaperSynthError):
    """components[].depends_on is not a DAG."""

    problem_type = "cyclic-dependency"
    http_status = 422

    def __init__(self, cycle: list[str]) -> None:
        self.cycle = cycle
        super().__init__("Cyclic component dependency: " + " -> ".join(cycle))


# --------------------------------------------------------------------------
# LLM providers
# --------------------------------------------------------------------------


class ProviderError(PaperSynthError):
    """Base for provider failures."""


class RateLimitError(ProviderError):
    """Fall-through error: the router should try the next provider (6.4.3)."""

    problem_type = "llm-rate-limited"
    http_status = 429

    def __init__(self, provider_id: str, retry_after: float | None = None) -> None:
        self.provider_id = provider_id
        self.retry_after = retry_after
        super().__init__(
            f"{provider_id} rate limited"
            + (f"; retry after {retry_after}s" if retry_after is not None else "")
        )


class CapacityError(ProviderError):
    """Fall-through error: provider is up but cannot serve right now."""

    problem_type = "llm-capacity"
    http_status = 503


class ContentPolicyError(ProviderError):
    """Terminal error: never falls through, because a different provider
    'coincidentally' succeeding would hide the real problem."""

    problem_type = "llm-content-policy"
    http_status = 422


class ModelNotFoundError(ProviderError):
    """Configured model ID 404s - free-tier lineups rotate (6.4.4)."""

    problem_type = "llm-model-not-found"
    http_status = 424


class AllProvidersExhausted(ProviderError):
    """Every provider in the chain is tapped out. The run pauses, it does not fail."""

    problem_type = "all-providers-exhausted"
    http_status = 429

    def __init__(self, last_error: Exception | None = None) -> None:
        self.last_error = last_error
        super().__init__(
            "All providers in the chain are exhausted. "
            "Re-run the same command with --resume once quotas reset; papers "
            "already extracted are reused rather than re-fetched."
            + (f" Last error: {last_error}" if last_error else "")
        )


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


class StageFailure(PaperSynthError):
    problem_type = "stage-failure"
    http_status = 500

    def __init__(self, stage: str, cause: Exception) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(f"Stage {stage!r} failed: {cause}")
