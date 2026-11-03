"""Text cleaning shared by training and the web app."""
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_ALPHA_RE = re.compile(r"[^a-z0-9'!? ]")
REPEAT_RE = re.compile(r"(.)\1{2,}")  # "soooo" -> "soo"
SPACES_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" url ", text)
    text = MENTION_RE.sub(" user ", text)
    text = text.replace("#", " ")
    text = text.replace("&amp;", " and ").replace("&quot;", " ").replace("&lt;", " ").replace("&gt;", " ")
    text = REPEAT_RE.sub(r"\1\1", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = text.replace("!", " ! ").replace("?", " ? ")
    return SPACES_RE.sub(" ", text).strip()
