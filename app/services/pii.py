import re

# This is intentionally "basic" and conservative.
# It will not catch everything and may over-mask in some cases.

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
UK_PHONE_RE = re.compile(r"\b(?:\+44\s?7\d{3}|\(?07\d{3}\)?)\s?\d{3}\s?\d{3}\b")
E164_RE = re.compile(r"\+\d{8,15}\b")


def redact_basic(text: str) -> str:
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = UK_PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = E164_RE.sub("[REDACTED_PHONE]", text)
    return text
