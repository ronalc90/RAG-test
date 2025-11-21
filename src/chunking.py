# src/chunking.py
import re
from typing import List

def split_text(text: str, max_chars: int = 1000, overlap: int = 150) -> List[str]:
    if not text:
        return []
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    parts: List[str] = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunk = text[i:j].strip()
        if chunk:
            parts.append(chunk)
        if j >= len(text):
            break
        i = max(0, j - overlap)

    return [p for p in parts if p]

