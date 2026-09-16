"""
Splits a document into comparable chunks.

Starting point: paragraph-level chunking, merging short fragments so a chunk
carries enough context to embed meaningfully. Swap this out later for
sentence-level or heading-aware chunking if you want finer-grained diffs.
"""

import re


def chunk_document(text: str, min_length: int = 40) -> list[str]:
    """Split text into paragraph chunks. Short paragraphs get merged forward
    so every chunk has enough content for a stable embedding."""
    raw_paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks: list[str] = []
    buffer = ""

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        buffer = f"{buffer}\n\n{para}".strip() if buffer else para
        if len(buffer) >= min_length:
            chunks.append(buffer)
            buffer = ""

    if buffer:
        if chunks:
            chunks[-1] += "\n\n" + buffer
        else:
            chunks.append(buffer)

    return chunks
