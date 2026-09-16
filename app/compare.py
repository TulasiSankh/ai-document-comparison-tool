"""
Core comparison logic: chunk both documents, embed every chunk, align chunks
across documents by similarity, and classify each as unchanged / modified /
added / removed.

This is a greedy best-match aligner, not a true edit-distance algorithm — it's
simple, fast, and good enough for prose documents where chunks don't get
wildly reordered. If you outgrow it, look at the Needleman-Wunsch /
sequence-alignment family of algorithms for order-aware matching.
"""

from dataclasses import dataclass

import numpy as np

from .chunking import chunk_document
from .embeddings import cosine_similarity_matrix, get_embeddings_batch
from .llm import explain_change

# Tune these thresholds against your own documents.
SIM_UNCHANGED = 0.995   # was 0.97   # above this: treat as identical in meaning
SIM_MODIFIED = 0.75    # below this: not a match at all (treat as added/removed)


@dataclass
class ChunkDiff:
    status: str  # "unchanged" | "modified" | "added" | "removed"
    doc_a_text: str | None = None
    doc_b_text: str | None = None
    similarity: float | None = None
    explanation: str | None = None


def compare_documents(text_a: str, text_b: str, explain: bool = True) -> list[ChunkDiff]:
    chunks_a = chunk_document(text_a)
    chunks_b = chunk_document(text_b)

    if not chunks_a and not chunks_b:
        return []
    if not chunks_a:
        return [ChunkDiff(status="added", doc_b_text=c) for c in chunks_b]
    if not chunks_b:
        return [ChunkDiff(status="removed", doc_a_text=c) for c in chunks_a]

    emb_a = get_embeddings_batch(chunks_a)
    emb_b = get_embeddings_batch(chunks_b)
    sim = cosine_similarity_matrix(emb_a, emb_b)

    matched_b: set[int] = set()
    results: list[ChunkDiff] = []

    # Walk doc A in order, greedily grab each chunk's best unclaimed match in doc B.
    for i, chunk_a in enumerate(chunks_a):
        j = int(np.argmax(sim[i]))
        score = float(sim[i, j])

        if j in matched_b or score < SIM_MODIFIED:
            results.append(ChunkDiff(status="removed", doc_a_text=chunk_a))
            continue

        matched_b.add(j)
        chunk_b = chunks_b[j]

        if score >= SIM_UNCHANGED:
            results.append(
                ChunkDiff(status="unchanged", doc_a_text=chunk_a, doc_b_text=chunk_b, similarity=score)
            )
        else:
            explanation = explain_change(chunk_a, chunk_b) if explain else None
            results.append(
                ChunkDiff(
                    status="modified",
                    doc_a_text=chunk_a,
                    doc_b_text=chunk_b,
                    similarity=score,
                    explanation=explanation,
                )
            )

    # Anything in doc B nobody claimed is new content.
    for j, chunk_b in enumerate(chunks_b):
        if j not in matched_b:
            results.append(ChunkDiff(status="added", doc_b_text=chunk_b))

    return results
