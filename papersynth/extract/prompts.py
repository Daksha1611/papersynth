"""Prompt template loading.

Templates live as .md files next to the provider code so they can be reviewed
and diffed as prose rather than buried in string literals. The template hash
feeds the cache key, so editing a prompt invalidates the claims it produced
(ER-10).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    if not path.exists():
        available = ", ".join(sorted(p.name for p in PROMPT_DIR.glob("*.md"))) or "(none)"
        raise FileNotFoundError(f"No prompt template {name!r}. Available: {available}")
    return path.read_text(encoding="utf-8")


def render(name: str, **fields: str) -> str:
    """Fill a template's ``{placeholder}`` fields.

    Uses explicit replacement rather than str.format because prompts contain
    literal JSON braces, which format() would try to interpret as fields.
    """
    text = load_prompt(name)
    for key, value in fields.items():
        text = text.replace("{" + key + "}", value)
    return text
