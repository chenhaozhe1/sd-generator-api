import re
from typing import Tuple

_BLOCKLIST = [
    r"(?i)gore|beheading|child\s*sexual|csam|rape|bestiality",
]

def moderate_prompt(prompt: str, negative: str | None = None) -> Tuple[bool, str]:
    """Return (allowed, message)."""
    for pat in _BLOCKLIST:
        if re.search(pat, prompt or "") or re.search(pat, negative or ""):
            return False, "Prompt blocked by safety policy."
    return True, "ok"
