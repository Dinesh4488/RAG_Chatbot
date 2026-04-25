import os
import sys
import warnings
from pathlib import Path

# Running `python retrieval\rag_pipeline.py` sets sys.path to `retrieval/`, not `src/`.
_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import faiss
import numpy as np
import orjson

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import SentenceTransformer

from generation.answer_generator import generate_answer
from grounding.evidence_answer_alignment import (
    alignment_acceptable,
    score_answer_evidence_max_sim,
)
from runtime_settings import (
    ANSWERABILITY_THRESHOLD,
    EMBEDDING_MODEL,
    ENABLE_POST_ANSWER_ALIGNMENT,
    MAX_ANSWERABILITY_CHARS,
    POST_ANSWER_ALIGNMENT_MODE,
    POST_ANSWER_ALIGNMENT_THRESHOLD,
    SIM_THRESHOLD,
    TOP_K,
)

WITHHELD_MESSAGE = (
    "This response was withheld because it did not align closely enough with the "
    "retrieved passages, to reduce unsupported or off-context statements."
)

INDEX_PATH = os.environ.get(
    "RAG_FAISS_INDEX",
    r"C:\rag_project\data\faiss_index\test_index.faiss",
)
META_PATH = os.environ.get(
    "RAG_META_JSONL",
    r"C:\rag_project\data\wikipedia_processed\meta_test.jsonl",
)


def _load_index_and_meta():
    if not os.path.isfile(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index missing: {INDEX_PATH}")
    if not os.path.isfile(META_PATH):
        raise FileNotFoundError(f"Metadata missing: {META_PATH}")
    idx = faiss.read_index(INDEX_PATH)
    meta = []
    with open(META_PATH, "rb") as f:
        for line in f:
            meta.append(orjson.loads(line))
    if len(meta) < idx.ntotal:
        raise ValueError(
            f"Metadata rows ({len(meta)}) < FAISS vectors ({idx.ntotal}). Rebuild meta/embeddings."
        )
    if len(meta) > idx.ntotal:
        warnings.warn(
            f"Metadata has {len(meta)} rows but index has {idx.ntotal} vectors; "
            "only the first ntotal rows align with vector IDs.",
            stacklevel=2,
        )
    return idx, meta


# One embedder + index + metadata (loaded at import; LLM loads only on first answer)
embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
index, metadata = _load_index_and_meta()



def compute_confidence(scores):
    top_score = scores[0]

    if top_score >= 0.75:
        return "High"
    elif top_score >= 0.60:
        return "Moderate"
    elif top_score >= 0.55:
        return "Low"
    else:
        return "Very Low"



def is_answerable(question, evidence_chunks, embedder):
    return semantic_answerable(
        question,
        evidence_chunks,
        embedder,
        threshold=ANSWERABILITY_THRESHOLD,
        max_chars_per_chunk=MAX_ANSWERABILITY_CHARS,
    )


def semantic_answerable(
    question,
    evidence_chunks,
    embedder,
    threshold=0.62,
    max_chars_per_chunk=800,
):
    """
    One batched encode over truncated chunks (not per-sentence) — much faster on CPU
    and similar signal for 'is this passage relevant to the question?'.
    """
    if not evidence_chunks:
        return False
    q_vec = embedder.encode(
        [question],
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
    scores = (c_vecs @ q_vec.T).flatten()
    return float(scores.max()) >= threshold

def run_rag(question):

    query_vec = embedder.encode([question], normalize_embeddings=True)
    scores, ids = index.search(np.array(query_vec), TOP_K)

    scores = scores[0]
    ids = ids[0]

    # Validation
    if scores[0] < SIM_THRESHOLD:
        return {
            "answer": "Insufficient evidence to answer this question.",
            "confidence": "Very Low",
            "evidence": [],
            "alignment_score": None,
            "alignment_passed": None,
        }

    # Collect evidence safely (FAISS may return -1 for missing results)
    evidence = []
    for raw_id in ids:
        try:
            id_ = int(raw_id)
        except Exception:
            continue
        if id_ < 0 or id_ >= len(metadata):
            continue
        evidence.append(metadata[id_])

    if not evidence:
        return {
            "answer": "Insufficient evidence to answer this question.",
            "confidence": "Very Low",
            "evidence": [],
            "alignment_score": None,
            "alignment_passed": None,
        }

    answer = generate_answer(question, evidence)
    confidence = compute_confidence(scores)

    alignment_score = None
    alignment_passed = None
    alignment_note = None

    if ENABLE_POST_ANSWER_ALIGNMENT:
        alignment_score = score_answer_evidence_max_sim(
            answer,
            evidence,
            embedder,
            max_chars_per_chunk=MAX_ANSWERABILITY_CHARS,
        )
        alignment_passed = alignment_acceptable(
            alignment_score, POST_ANSWER_ALIGNMENT_THRESHOLD
        )
        if not alignment_passed:
            if POST_ANSWER_ALIGNMENT_MODE == "flag":
                alignment_note = (
                    f"Low answer–evidence alignment (score={alignment_score:.3f}; "
                    f"threshold={POST_ANSWER_ALIGNMENT_THRESHOLD})."
                )
                confidence = f"{confidence} (verify against evidence)"
            else:
                answer = WITHHELD_MESSAGE
                confidence = "Very Low (post-check)"

    out = {
        "answer": answer,
        "confidence": confidence,
        "evidence": evidence,
        "alignment_score": alignment_score,
        "alignment_passed": alignment_passed,
    }
    if alignment_note:
        out["alignment_note"] = alignment_note
    return out


if __name__ == "__main__":

    while True:
        q = input("\nAsk a question (or type exit): ")
        if q.lower() == "exit":
            break

        result = run_rag(q)

        print("\n📌 Answer:")
        print(result["answer"])

        print(f"\n🔐 Confidence: {result['confidence']}")
        if result.get("alignment_score") is not None:
            print(
                f"📐 Answer–evidence alignment: {result['alignment_score']:.3f} "
                f"(passed={result['alignment_passed']})"
            )
        if result.get("alignment_note"):
            print(f"⚠️  {result['alignment_note']}")

        print("\n📚 Evidence:")
        for i, e in enumerate(result["evidence"], 1):
            print(f"\nEvidence {i}:")
            print(f"Title: {e['title']}")
            print(f"Section: {e['section']}")
            print(e["text"][:400])
