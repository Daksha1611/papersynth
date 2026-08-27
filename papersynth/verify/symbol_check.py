"""Symbol closure (section 8.3.2).

An equation with undefined symbols is not implementable: you cannot write code
for a quantity nobody defined. That alone justifies the check, but its more
valuable side effect is catching corrupted math. Garbled OCR reliably produces
phantom symbols that nothing in the paper defines, so an equation arriving with
a symbol table full of undefined entries is usually a parsing failure rather
than an unusually terse author (R-01).
"""

from __future__ import annotations

from papersynth.core.models import Claim
from papersynth.verify.range_check import CheckOutcome

#: Above this share of undefined symbols, the equation is more likely mangled
#: than merely under-explained.
CORRUPTION_RATIO = 0.5


def symbol_check(claim: Claim) -> CheckOutcome:
    """Fail an equation whose symbols are not all defined."""
    if claim.type != "equation":
        return CheckOutcome("n/a")

    symbols = claim.payload.get("symbols") or []
    undefined = list(claim.payload.get("undefined_symbols") or [])

    if not symbols:
        return CheckOutcome(
            "fail",
            "equation has no symbol table; nothing about it can be verified",
        )

    if not undefined:
        return CheckOutcome("pass")

    ratio = len(undefined) / len(symbols)
    listed = ", ".join(undefined[:6])

    if ratio >= CORRUPTION_RATIO or claim.payload.get("source_fidelity") == "ocr_recovered":
        return CheckOutcome(
            "fail",
            f"{len(undefined)} of {len(symbols)} symbols are undefined ({listed}); "
            "this pattern usually means the equation was mangled during "
            "extraction rather than left unexplained by the authors",
        )

    return CheckOutcome(
        "fail",
        f"undefined symbols: {listed}. An equation with undefined symbols cannot be implemented.",
    )
