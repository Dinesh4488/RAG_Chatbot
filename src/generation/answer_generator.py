import os
from typing import Optional

from llama_cpp import Llama

from runtime_settings import (
    LLM_MAX_TOKENS,
    LLM_N_CTX,
    LLM_TEMPERATURE,
    MAX_EVIDENCE_CHARS_PER_CHUNK,
    llama_n_gpu_layers,
    llama_n_threads,
)

MODEL_PATH = os.environ.get(
    "RAG_LLM_PATH",
    r"C:\rag_project\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf",
)

_llm: Optional[Llama] = None


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def get_llm() -> Llama:
    """Load LLM on first use to avoid RAM spike on import and allow import without GGUF."""
    global _llm
    if _llm is not None:
        return _llm
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"LLM not found: {MODEL_PATH}\n"
            "Set RAG_LLM_PATH to your .gguf file or place the model at the default path."
        )
    _llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=LLM_N_CTX,
        n_threads=llama_n_threads(),
        n_gpu_layers=llama_n_gpu_layers(),
        temperature=LLM_TEMPERATURE,
        top_p=0.9,
        verbose=False,
        use_mmap=True,
        use_mlock=False,
    )
    return _llm


def build_validation_prompt(question, evidence_chunks):

    evidence_text = ""

    for i, chunk in enumerate(evidence_chunks, 1):
        body = _truncate(chunk["text"], MAX_EVIDENCE_CHARS_PER_CHUNK)
        evidence_text += f"\nEvidence {i}:\n"
        evidence_text += f"Title: {chunk['title']}\n"
        evidence_text += f"Section: {chunk['section']}\n"
        evidence_text += f"Text: {body}\n"

    prompt = f"""
You are an academic question answering system.

Your task is to analyze whether the provided question can be answered using ONLY the given evidence.

Question:
{question}

Evidence:
{evidence_text}

Analysis:
- Determine if the evidence contains sufficient information to answer the question.
- Consider if the question requires specific facts, dates, names, or concepts that are present in the evidence.
- Do not use any external knowledge or assumptions.

Respond with ONLY one of these two options:
- "ANSWERABLE" if the evidence supports answering the question.
- "NOT_ANSWERABLE" if the evidence is insufficient.
"""

    return prompt


def validate_answerability(question, evidence_chunks):
    """Use LLM to determine if the question can be answered with the provided evidence."""
    if not evidence_chunks:
        return False
    
    prompt = build_validation_prompt(question, evidence_chunks)
    llm = get_llm()

    response = llm(
        prompt,
        max_tokens=50,  # Short response
        stop=["</s>"],
        temperature=0.1,  # Low temperature for consistency
    )

    result = response["choices"][0]["text"].strip().upper()
    return "ANSWERABLE" in result


def build_prompt(question, evidence_chunks):

    evidence_text = ""

    for i, chunk in enumerate(evidence_chunks, 1):
        body = _truncate(chunk["text"], MAX_EVIDENCE_CHARS_PER_CHUNK)
        evidence_text += f"\nEvidence {i}:\n"
        evidence_text += f"Title: {chunk['title']}\n"
        evidence_text += f"Section: {chunk['section']}\n"
        evidence_text += f"Text: {body}\n"

    prompt = f"""
You are an academic question answering system.

RULES (must follow strictly):
- Use ONLY the provided evidence.
- Do NOT use prior knowledge.
- Do NOT guess or assume.
- Provide a concise and accurate answer based solely on the evidence.

Question:
{question}

Evidence:
{evidence_text}

Task:
Based ONLY on the evidence above, provide a concise and accurate answer.
"""

    return prompt


def generate_answer(question, evidence_chunks):

    # First, validate if the question can be answered
    if not validate_answerability(question, evidence_chunks):
        return "Insufficient evidence to answer this question."
    
    prompt = build_prompt(question, evidence_chunks)
    llm = get_llm()

    response = llm(
        prompt,
        max_tokens=LLM_MAX_TOKENS,
        stop=["</s>"],
    )

    return response["choices"][0]["text"].strip()
