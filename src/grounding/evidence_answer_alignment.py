"""
Evidence–answer alignment (post-generation).

This is not CA-RAG (Collini et al.): we keep standard retrieve-then-read with top-k
FAISS, then apply a lightweight check *after* the LLM responds.

Motivation (vs. common RAG failure modes):
- The retriever can be right while the generator still drifts from the passages
  (parametric knowledge, hedging, or fabrication).
- Comparing the *answer embedding* to the *same evidence* used in the prompt gives
  a second, independent signal: "does the reply still live in the neighborhood of
  what we actually retrieved?"

One batched encode over evidence chunks + one over the answer; uses the same
normalized dot product as elsewhere in the project (cosine for L2-normalized vectors).
"""
from __future__ import annotations

from typing import Any, List

import numpy as np

# If the model already refused, skip alignment (no need to score abstention text).
_ABSTAIN_MARKERS = (
    "insufficient evidence",
    "not supported by the evidence",
    "cannot answer",
)


def _is_abstention_phrase(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(m in t for m in _ABSTAIN_MARKERS) or len(t) < 8


def score_answer_evidence_max_sim(
    answer: str,
    evidence_chunks: List[dict[str, Any]],
    embedder,
    max_chars_per_chunk: int = 800,
) -> float:
    """
    Return max cosine similarity between the answer and each evidence chunk
    (embedding space must match retrieval / answerability).
    """
    if not evidence_chunks or _is_abstention_phrase(answer):
        return 1.0

    a_vec = embedder.encode(
        [answer.strip()],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    texts = []
    for chunk in evidence_chunks:
        t = (chunk.get("text") or "").strip()
        if len(t) > max_chars_per_chunk:
            t = t[:max_chars_per_chunk]
        texts.append(t if t else " ")

    c_vecs = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=min(8, len(texts)),
    )
    sims = (c_vecs @ a_vec.T).flatten()
    return float(np.max(sims))


def alignment_acceptable(score: float, threshold: float) -> bool:
    return score >= threshold
