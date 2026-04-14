"""
Resource limits for laptops (limited RAM / CPU). Override via environment variables.

RAG_GPU_LAYERS   0 = CPU only (default, safest). -1 = all layers on GPU if CUDA works.
RAG_N_CTX        LLM context length (lower = less RAM, faster prefill).
RAG_N_THREADS    llama.cpp thread count (default: leave 1 core free).
RAG_MAX_TOKENS   Max new tokens per answer (lower = faster).
RAG_POST_ANSWER_ALIGNMENT   1 = after LLM, score answer vs evidence (default 1).
RAG_POST_ANSWER_ALIGNMENT_THRESHOLD   e.g. 0.32–0.42 (short answers score lower).
RAG_POST_ANSWER_ALIGNMENT_MODE   abstain | flag (flag keeps text, adds warning).
"""
import os

# Embedding / retrieval
TOP_K = int(os.environ.get("RAG_TOP_K", "3"))
SIM_THRESHOLD = float(os.environ.get("RAG_SIM_THRESHOLD", "0.55"))
ANSWERABILITY_THRESHOLD = float(os.environ.get("RAG_ANSWERABILITY_THRESHOLD", "0.62"))
# Match embedder used when building the FAISS index (change both if you switch models).
EMBEDDING_MODEL = os.environ.get(
    "RAG_EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5",
)

# Answerability: compare question to first N chars of each chunk (one encode per chunk).
MAX_ANSWERABILITY_CHARS = int(os.environ.get("RAG_ANSWERABILITY_CHARS", "800"))

# Post-generation: answer must align with retrieved evidence (embeddings).
# Differs from CA-RAG (IEEE Access 2025): we do not drop retrieval or feed whole documents;
# we add a second gate after the LLM to catch generator drift from the passages.
ENABLE_POST_ANSWER_ALIGNMENT = os.environ.get("RAG_POST_ANSWER_ALIGNMENT", "1") == "1"
POST_ANSWER_ALIGNMENT_THRESHOLD = float(
    os.environ.get("RAG_POST_ANSWER_ALIGNMENT_THRESHOLD", "0.36")
)
# abstain = replace answer when alignment fails; flag = keep answer, expose score + warning
POST_ANSWER_ALIGNMENT_MODE = os.environ.get(
    "RAG_POST_ANSWER_ALIGNMENT_MODE", "abstain"
).lower()

# Prompt / LLM — keep context small for speed and RAM
MAX_EVIDENCE_CHARS_PER_CHUNK = int(os.environ.get("RAG_EVIDENCE_CHARS", "900"))
LLM_N_CTX = int(os.environ.get("RAG_N_CTX", "1536"))
LLM_MAX_TOKENS = int(os.environ.get("RAG_MAX_TOKENS", "128"))
LLM_TEMPERATURE = float(os.environ.get("RAG_TEMPERATURE", "0.15"))


def llama_n_threads() -> int:
    cpu = os.cpu_count() or 4
    default = max(1, min(cpu - 1, 8))
    return int(os.environ.get("RAG_N_THREADS", str(default)))


def llama_n_gpu_layers() -> int:
    return int(os.environ.get("RAG_GPU_LAYERS", "0"))
