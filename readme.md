# RAG Wikipedia QA System — Complete Project Documentation

**Purpose:** Grounded question answering over a Wikipedia-derived corpus with multiple semantic gates and optional post-generation alignment.  
**Typical environment:** Conda env `ragx`, run from `c:\rag_project\src`.  
**Corpus scope:** Indexed subset (see §7.4)—not full English Wikipedia in the FAISS index.

---

## 1. Introduction

Large language models (LLMs) can produce fluent but **factually unsupported** text (**hallucination**). **Retrieval-Augmented Generation (RAG)** reduces reliance on parametric memory by conditioning answers on **retrieved passages** from a trusted corpus. This project uses **English Wikipedia–derived chunks**, **dense retrieval** (FAISS + sentence embeddings), and a **local** quantized **Mistral 7B Instruct** model (`llama.cpp`).

The system goes beyond naive “retrieve and prompt” by adding:

1. A **retrieval confidence gate** (minimum similarity of the top hit).  
2. A **question–evidence relevance gate** (semantic answerability before calling the LLM).  
3. A **strict evidence-only** generation prompt with an explicit abstention instruction.  
4. An optional **post-LLM** check: **answer–evidence embedding alignment** to catch **generator drift** when retrieval looked acceptable but the model’s reply no longer matches the retrieved text.

---

## 2. Problem Statement

- **Input:** A natural-language **question** from the user.  
- **Output:** A **short answer** (or abstention), a **confidence label** derived from retrieval scores, **evidence** (title, section, text), and optionally **alignment score / pass flag** for research logging.  
- **Constraint:** Answers should be **grounded** in retrieved Wikipedia chunks; the design prioritizes **refusal** when evidence is weak or the model’s output diverges from evidence (configurable).

---

## 2.5 Design rationale: why this approach (and not others)

This section states **why** major decisions were taken, **what was not chosen**, and **how far** you can claim the result is “better” than alternatives. It is written for a **B.Tech report**: examiners expect justification, not hype.

### 2.5.1 Goals that drove the choices

| Goal | Implication |
|------|-------------|
| **Run locally** (privacy, cost, no API key during demos) | **GGUF + `llama.cpp`**, not hosted GPT-class APIs as the default generator. |
| **Work on a typical laptop** (RAM/CPU) | **Smaller** embedding model, **quantized** LLM, **truncated** context, **lazy** LLM load, **CPU** embeddings by default. |
| **Reduce hallucination risk** (research focus) | **RAG** + **explicit abstention** + **similarity gates** before and after generation—not “model-only” QA. |
| **Reproducible student project** | Open components (FAISS, SentenceTransformers, open-weight Mistral), fixed paths configurable via env vars. |

If a goal were **maximum accuracy at any cost**, the design would differ (larger models, GPU, full corpus, cloud APIs, heavier verifiers). Those were **not** the primary constraints here.

### 2.5.2 Why this, not that — component by component

| Choice | Why this | Alternatives not used (or deferred) | “Better” in what sense |
|--------|----------|----------------------------------------|-------------------------|
| **Task: single-turn QA** | Matches **clear I/O** for evaluation and demo (one question → one answer). | **Chatbot** with history (e.g. WikiChat-style dialogue): more engineering and eval complexity. | **Simpler** to build and test; not “better chat.” |
| **RAG vs fine-tuning only** | Updates knowledge by **rebuilding the index**, without **retraining** the LLM; fits **time/budget** of a major project. | **Full fine-tune** on Wikipedia: heavy GPU, data prep, risk of overfitting; **model-only** QA: stronger hallucination risk on facts. | **Better fit** for updatable corpus + limited training infra; not proven higher F1 without your own eval. |
| **Dense retrieval (BGE + FAISS)** | **Semantic** match between question and passages; standard in modern RAG; good tradeoff of quality vs size with **bge-small**. | **BM25 / sparse only**: cheaper but weaker paraphrase; **larger** embedders (e.g. bge-large): better quality but **slower** and more RAM on CPU. | **Better semantic recall** than keyword-only *in general* (literature); **your** index must still be evaluated on *your* questions. |
| **FAISS `IndexFlatIP`** | **Exact** maximum inner product over all vectors; simple, correct for small–medium N; normalized vectors ⇒ cosine. | **IVF / HNSW**: faster at **very large** N but needs tuning and can miss neighbors; overkill for ~10k vectors. | **Better simplicity and exactness** at current scale; **worse** scaling if N → millions without switching index type. |
| **Mistral 7B Instruct, Q4 GGUF** | Strong **instruction-following** for “use only evidence”; **4-bit** fits **consumer RAM**; **Instruct** models are already **alignment-trained**. | **13B+** or FP16: higher quality possible, **often infeasible** on the same laptop; **smaller** 3B: faster but may follow instructions worse. | **Better practicality** for local deployment; **not** automatically better than the largest models you cannot run. |
| **`llama.cpp` via `llama-cpp-python`** | **Inference-focused**, mmap, CPU-friendly; widely used for GGUF. | **PyTorch `transformers` FP16**: higher RAM; **vLLM** etc.: server-class. | **Better** for **low-resource local** inference in this setup. |
| **Gate A (retrieval threshold)** | If the **best** chunk is still a **weak** match, answering often **hurts** more than abstaining. | **No gate**: always call LLM → more **confident hallucinations** on off-corpus queries. | **Better calibrated refusal** when retrieval is uncertain (motivated by IR+RAG literature); threshold values should be **tuned on data**. |
| **Gate B (question–evidence)** | Top-K can include **tangential** passages; second check asks whether the **question** aligns with **at least one** chunk. | **Skip**: saves latency but increases **wrong-context** answers. | **Better** screening of relevance before generation; same caveat—**embedding ≠ logical entailment**. |
| **Strict prompt + low temperature** | Reduces **creative** additions not in evidence. | **High temperature / loose prompt**: more fluent, **higher hallucination** risk in RAG. | **Better faithfulness *intent***; model can still violate instructions—hence Gate C. |
| **Gate C (post answer–evidence alignment)** | Catches cases where retrieval passed but the **answer text** **drifts** from passages (parametric knowledge). | **NLI / cross-encoder**: can be **more accurate** but **heavier**; **no post-check**: faster, riskier. | **Better** cost/speed vs a second large model; **weaker** logically than entailment—see limitations. |
| **Wikipedia chunks** | **Curated**, broad **open-domain** knowledge; aligns with cited papers (e.g. WikiChat). | **Domain PDFs**: better for one company but **narrower** demo; **proprietary** data: permission issues. | **Better** for a **general-knowledge** student demo without private data. |
| **`MAX_CHUNKS = 10_000`** | **Bounds** embedding time, RAM, and index size for development machines. | **Full corpus**: **better coverage** but **long** builds and **high** RAM. | **Better for iteration** on a laptop; **worse coverage** of rare topics. |

### 2.5.3 How you know whether this is “much better than the others” — what the project can and cannot claim

**Important:** The repository, as shipped, **does not include** a full comparative study (e.g. same 500 questions × multiple systems with significance tests). Therefore:

- You **must not** honestly write: *“Our system is much better than all other approaches”* **without** your own **tables, metrics, and baselines**.
- You **can** write: *“We chose X over Y because of constraints Z (local run, RAM, time), consistent with surveys on RAG and hallucination mitigation.”*
- You **can** write: *“Compared to **vanilla RAG** (retrieve + prompt, no gates), our design **adds** explicit refusal and post-generation alignment, which **in principle** address failure modes described in the literature (weak retrieval, generator drift).”* That is a **design** argument, not a **measured** superiority claim.

**How you would know empirically (for your report):**

1. Fix a **question set** (answerable vs should-abstain).  
2. Run **baselines**: e.g. (i) vanilla RAG, gates off or threshold 0; (ii) your full pipeline.  
3. Report **metrics** (accuracy, abstention precision/recall, optional human rubric on 20–30 samples).  
4. Then you may say: *“On our benchmark, configuration A scored … vs B …”* — that is **evidence**; everything else is **motivation**.

**Sources of “knowing” besides your own numbers:** peer-reviewed **surveys** (e.g. hallucination mitigation, RAG limitations) motivate **why** retrieval + grounding + abstention are **reasonable** mitigations—they do **not** prove **your** implementation beats **every** alternative on **your** hardware without measurement.

---

## 3. Methodology (Conceptual)

### 3.1 Retrieval-Augmented Generation (RAG)

1. **Offline:** Documents are split into **chunks** with metadata (`title`, `section`, `text`). Each chunk is embedded with a **sentence transformer**; vectors are **L2-normalized** so **inner product** equals **cosine similarity**. Vectors are stored in a **FAISS** `IndexFlatIP` index.  
2. **Online:** The question is embedded with the **same** model. **Top-K** nearest neighbors are retrieved. The generator receives **only** those passages (truncated) plus instructions to use **only** that evidence.

### 3.2 Multi-Stage Gating (Mitigation Strategy)

*Rationale for stacking gates vs a single prompt-only baseline is in **§2.5**.*

| Stage | Mechanism | Intent |
|--------|-----------|--------|
| **Gate A** | Top-1 retrieval score ≥ `SIM_THRESHOLD` | Reject queries whose best match is still a weak semantic match (no LLM call if failed early with empty evidence path). |
| **Gate B** | Max similarity **question ↔ each retrieved chunk** ≥ `ANSWERABILITY_THRESHOLD` | Ensure at least one passage is **semantically related** to the question beyond raw retrieval ranking quirks. |
| **Generation** | Low temperature, short `max_tokens`, evidence-only prompt | Reduce creative drift; force concise answers. |
| **Gate C (optional)** | Max similarity **answer ↔ each evidence chunk** ≥ `POST_ANSWER_ALIGNMENT_THRESHOLD` | Detect **post-hoc** mismatch: model output not “near” the evidence embeddings (abstain or flag). |

### 3.3 Resource-Oriented Design

- **CPU** embedding model (`device="cpu"` in `rag_pipeline`).  
- **Lazy LLM load** (`get_llm()`): GGUF loads on **first** `generate_answer`, not at import.  
- **`use_mmap=True`, `use_mlock=False`** for `llama.cpp` to limit RAM pressure.  
- **Truncation** of evidence in prompt (`MAX_EVIDENCE_CHARS_PER_CHUNK`) and in gates (`MAX_ANSWERABILITY_CHARS`).  
- **Batched** encoding in `semantic_answerable` and in `evidence_answer_alignment` (batch size capped at 8 for chunk lists).

---

## 4. Technologies and Dependencies

| Component | Role |
|-----------|------|
| **Python 3** | Application language |
| **Conda env `ragx`** | Isolated dependencies (user-stated) |
| **sentence-transformers** + **PyTorch** | `SentenceTransformer` for BGE embeddings |
| **FAISS** (`faiss` / `faiss-cpu`) | Approximate/exact similarity search; here **exact** `IndexFlatIP` |
| **NumPy** | Vector math, stacking embeddings |
| **orjson** | Fast JSONL read/write |
| **llama-cpp-python** | Loads **GGUF** and runs **Mistral** inference |
| **tqdm** | Progress bars in offline scripts |
| **HuggingFace `datasets`** (optional) | `wikipedia_download.py` — alternate ingest path |
| **unidecode** (optional) | `wikipedia_ingest.py` — alternate ingest path |

**Note:** There is no `requirements.txt` in the repository; reproduce the env via Conda/pip from the list above.

**Why these stacks vs others:** See **§2.5** (e.g. dense vs sparse retrieval, FAISS flat vs approximate, GGUF vs API, small embedder vs large).

---

## 5. End-to-End Pipeline

### 5.1 Data Flow Overview

```
[Raw Wikipedia JSONL shards]  →  wiki_chunker  →  chunks_0.jsonl
       →  wiki_cleaner  →  clean_chunks_0.jsonl
       →  embedder  →  embeddings_test.npy + meta_test.jsonl
       →  faiss_index  →  test_index.faiss

[User question]  →  rag_pipeline.run_rag  →  answer + metadata
```

### 5.2 Offline Pipeline (Build the Index)

1. **`wiki_chunker.py`**  
   - Reads **one** unpacked file: `enwiki_namespace_0_0.jsonl` (structured articles: `name`, `sections`, `has_parts`, paragraph `value`).  
   - Emits one JSONL line per paragraph ≥ 120 characters: `{title, section, text}`.

2. **`wiki_cleaner.py`**  
   - Reads `chunks_0.jsonl`, strips citation markers `[digits]`, normalizes whitespace, drops lines with `text` length &lt; 120 after cleaning.  
   - Writes `clean_chunks_0.jsonl`.

3. **`embedder.py`**  
   - Streams `clean_chunks_0.jsonl`, stops after **`MAX_CHUNKS` (10,000)** lines.  
   - Batches encoding (64), **normalized** embeddings.  
   - Saves `embeddings_test.npy` (2D float32 array) and **`meta_test.jsonl`** (same order as rows in `.npy`).  
   - Uses **hard-coded** local HuggingFace snapshot path for `BAAI/bge-small-en-v1.5` (must stay consistent with query-time model).

4. **`faiss_index.py`**  
   - Loads `embeddings_test.npy`, builds **`IndexFlatIP`**, `add`s all vectors, writes `test_index.faiss`.  
   - **Invariant:** Row `i` of embeddings ↔ line `i` of `meta_test.jsonl` ↔ FAISS id `i`.

### 5.3 Online Pipeline (`run_rag(question)`)

Detailed flow (see `retrieval/rag_pipeline.py`):

1. **Embed question** — `normalize_embeddings=True`.  
2. **FAISS search** — `index.search(query_vec, TOP_K)` → `scores[0..K-1]`, `ids[0..K-1]`.  
3. **Gate A** — If `scores[0] < SIM_THRESHOLD` → return abstention, **empty** `evidence`, `alignment_*` = `None`.  
4. **Resolve evidence** — For each id, skip invalid/`negative`/out-of-range; append `metadata[id]` to `evidence`.  
5. **Empty evidence guard** → same abstention shape.  
6. **Gate B** — `is_answerable` → `semantic_answerable`: batch-encode truncated chunk texts; **max** `(chunk_emb · q_emb)` ≥ `ANSWERABILITY_THRESHOLD`. If false → abstention but **return** `evidence` for transparency.  
7. **`generate_answer`** — `answer_generator.build_prompt` + `get_llm()` + single completion, `stop=["</s>"]`.  
8. **`compute_confidence(scores)`** — Maps **retrieval** top score to labels High / Moderate / Low / Very Low (thresholds 0.75 / 0.60 / 0.55).  
9. **Gate C (if `ENABLE_POST_ANSWER_ALIGNMENT`)** — `score_answer_evidence_max_sim`; if below threshold: **`abstain`** replaces answer with `WITHHELD_MESSAGE` or **`flag`** keeps answer and sets `alignment_note`.  
10. **Return** dict: `answer`, `confidence`, `evidence`, `alignment_score`, `alignment_passed`, optional `alignment_note`.

### 5.4 CLI Loop (`if __name__ == "__main__"`)

Interactive prompts until `exit`; prints answer, confidence, alignment lines if present, and first 400 characters of each evidence item.

---

## 6. File-by-File Reference

### 6.1 Core runtime (query path)

| File | Contents / responsibility |
|------|---------------------------|
| **`src/runtime_settings.py`** | Central **constants** from environment variables: `TOP_K`, `SIM_THRESHOLD`, `ANSWERABILITY_THRESHOLD`, `EMBEDDING_MODEL`, `MAX_ANSWERABILITY_CHARS`, post-alignment flags/threshold/mode, LLM `n_ctx`, `max_tokens`, temperature, evidence truncation, `llama_n_threads()`, `llama_n_gpu_layers()`. |
| **`src/retrieval/rag_pipeline.py`** | **Main orchestration:** loads `SentenceTransformer` on CPU, loads FAISS + metadata via `_load_index_and_meta()`, defines `compute_confidence`, `semantic_answerable`, `is_answerable`, **`run_rag`**, CLI. Enforces index/meta consistency: **error** if `len(meta) < ntotal`, **warning** if `len(meta) > ntotal`. Sets `HF_HUB_OFFLINE` and `TOKENIZERS_PARALLELISM=false` by default. |
| **`src/generation/answer_generator.py`** | **`get_llm()`** singleton `Llama` (lazy). **`_truncate`** for chunk bodies. **`build_prompt`**: academic QA rules + question + numbered evidence. **`generate_answer`**: runs model with `LLM_MAX_TOKENS`, `stop=["</s>"]`. |
| **`src/grounding/__init__.py`** | Package marker; short docstring. |
| **`src/grounding/evidence_answer_alignment.py`** | **`_is_abstention_phrase`**: skips harsh scoring for abstention-like or very short outputs (returns alignment score `1.0`). **`score_answer_evidence_max_sim`**: encode answer + batch chunks, return **max** cosine/IP. **`alignment_acceptable`**: score ≥ threshold. |

### 6.2 Offline indexing

| File | Contents / responsibility |
|------|---------------------------|
| **`src/embeddings/embedder.py`** | Streaming reader with **`MAX_CHUNKS=10000`** cap; writes `embeddings_test.npy` and `meta_test.jsonl`; **local** `MODEL_NAME` path. |
| **`src/retrieval/faiss_index.py`** | Loads `.npy`, builds `IndexFlatIP`, writes `test_index.faiss`. |
| **`src/retrieval/test_search.py`** | **Diagnostic only:** embed query with hub model name `BAAI/bge-small-en-v1.5`, top-3 search, print snippets—**no** LLM, **no** gates. Paths hard-coded. |

### 6.3 Data preparation (corpus build)

| File | Contents / responsibility |
|------|---------------------------|
| **`src/processing/wiki_chunker.py`** | JSONL article → paragraph chunks; **input hard-coded** to `enwiki_namespace_0_0.jsonl` only; min paragraph length 120. |
| **`src/processing/wiki_cleaner.py`** | Citation + whitespace cleanup; min length 120 after clean. |
| **`src/ingest/wikipedia_download.py`** | `load_dataset("wikipedia", "20230601.en")` → `save_to_disk` under `data/wikipedia_raw/` — **alternate** corpus source (not the same schema as unpacked namespace JSONL used by `wiki_chunker`). |
| **`src/ingest/wikipedia_ingest.py`** | Reads **disk** dataset from `wikipedia_raw`, writes **per-paragraph JSON files** — **alternate** pipeline; not wired to `embedder.py` paths as-is. |
| **`src/ingest/test_parser.py`** | Prints first 3 articles’ paragraph previews from one JSONL — **debug/inspection** utility. |

### 6.4 Tests / scratch

| File | Contents / responsibility |
|------|---------------------------|
| **`tests/test.py`** | Almost entirely **commented-out** exploratory code (JSON structure, SentenceTransformer load). **Inactive.** |
| **`tests/LLM_test.py`** | Standalone **smoke test** for `llama_cpp`: loads GGUF with `n_ctx=512`, `n_gpu_layers=-1`, runs one prompt. **Independent** of `rag_pipeline` settings. |

### 6.5 Data directories (expected layout)

| Path | Role |
|------|------|
| **`data/wikipedia_raw/unpacked/`** | Source JSONL shards (`enwiki_namespace_0_*.jsonl`) for chunker-style ingest. |
| **`data/wikipedia_processed/chunks_0.jsonl`** | Output of `wiki_chunker`. |
| **`data/wikipedia_processed/clean_chunks_0.jsonl`** | Output of `wiki_cleaner`; **input** to `embedder`. |
| **`data/wikipedia_processed/embeddings_test.npy`** | Matrix of normalized embeddings. |
| **`data/wikipedia_processed/meta_test.jsonl`** | Parallel metadata lines. |
| **`data/faiss_index/test_index.faiss`** | Serialized FAISS index. |
| **`models/*.gguf`** | Local LLM weights (default filename in `answer_generator`). |
| **`src/papers/*.pdf`** | Reference PDFs (WikiChat, CA-RAG, etc.) — **not** executed by code. |

### 6.6 Editor

| File | Role |
|------|------|
| **`.vscode/settings.json`** | Workspace editor settings only. |

---

## 7. Key Functions (Behavioral Detail)

### 7.1 `_load_index_and_meta()` (`rag_pipeline.py`)

- Validates files exist.  
- Reads entire `meta_test.jsonl` into a **Python list** in memory (RAM scales with indexed corpus size).  
- Ensures every FAISS id `0..ntotal-1` has a metadata row: **`len(meta) >= ntotal`**.

### 7.2 `semantic_answerable(...)`

- Returns **False** if `evidence_chunks` empty.  
- Encodes question once; encodes each chunk text truncated to `max_chars_per_chunk`; computes matrix–vector product `c_vecs @ q_vec.T`; passes if **maximum** scalar ≥ `threshold`.  
- **Design choice:** whole-chunk vs first sentences — favors **CPU efficiency** over fine-grained sentence matching.

### 7.3 `compute_confidence(scores)`

- Uses **only** the **retrieval** score vector’s **first element** (top hit), **not** the post-alignment score.  
- Label bands: ≥0.75 High; ≥0.60 Moderate; ≥0.55 Low; else Very Low.

### 7.4 `score_answer_evidence_max_sim(...)`

- If answer matches abstention heuristics or length &lt; 8, returns **`1.0`** (treat as pass—avoids double-penalizing refusals).  
- Otherwise encodes full answer string and truncated chunks; returns **max** similarity.

### 7.5 `get_llm()` (`answer_generator.py`)

- Raises **`FileNotFoundError`** with instructions if GGUF missing.  
- **`n_gpu_layers`** default **0** (CPU); override with `RAG_GPU_LAYERS` for GPU builds.

---

## 8. Configuration (Environment Variables)

| Variable | Default (via `runtime_settings`) | Meaning |
|----------|----------------------------------|---------|
| `RAG_TOP_K` | 3 | Neighbors retrieved from FAISS |
| `RAG_SIM_THRESHOLD` | 0.55 | Minimum top-1 retrieval score to proceed |
| `RAG_ANSWERABILITY_THRESHOLD` | 0.62 | Min max question–chunk similarity |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Query-time embedder (must match index) |
| `RAG_ANSWERABILITY_CHARS` | 800 | Truncate chunk for answerability encoding |
| `RAG_POST_ANSWER_ALIGNMENT` | `1` | Enable post-LLM alignment |
| `RAG_POST_ANSWER_ALIGNMENT_THRESHOLD` | 0.36 | Min answer–evidence max similarity |
| `RAG_POST_ANSWER_ALIGNMENT_MODE` | `abstain` | `abstain` or `flag` |
| `RAG_EVIDENCE_CHARS` | 900 | Max chars per chunk **in LLM prompt** |
| `RAG_N_CTX` | 1536 | LLM context length |
| `RAG_MAX_TOKENS` | 128 | Max new tokens |
| `RAG_TEMPERATURE` | 0.15 | Sampling temperature |
| `RAG_N_THREADS` | auto | `llama.cpp` threads |
| `RAG_GPU_LAYERS` | 0 | GPU offload layers |
| `RAG_LLM_PATH` | default path under `models/` | GGUF location |
| `RAG_FAISS_INDEX` | default under `data/faiss_index/` | Index file |
| `RAG_META_JSONL` | default `meta_test.jsonl` | Metadata JSONL |

---

## 9. Relation to Referenced Papers (Honest Comparison)

**Why not copy those systems wholesale, and whether that makes this “better”:** The papers target **different scopes** (multi-turn chat, different pipelines, different compute). This project **prioritizes** a **minimal local stack** (§2.5). **Better** here means **better aligned with those constraints**, not **higher reported F1** than WikiChat on their benchmarks unless you replicate their eval—which this repo does not do by default.

### 9.1 WikiChat (e.g. arXiv:2305.14292)

- **Their approach:** Multi-stage **dialogue** system—generate, **filter grounded claims**, retrieve again, merge; distillation; strong **human/LLM** evaluation.  
- **This project:** **Single-turn** QA, **no** claim-level decomposition, **no** second retrieval pass, **no** distillation pipeline.  
- **Overlap:** Wikipedia as corpus, emphasis on **grounding** and **reducing hallucination**.  
- **Why not WikiChat for this project:** Their pipeline (claim filtering, re-retrieval, distillation, dialogue eval) is **heavier** than a single **B.Tech** timeline and laptop budget typically allow.  
- **Improvement claim (defensible):** Simpler **deployable** stack for **low-resource** machines (local GGUF, CPU embeddings, explicit thresholds). **Not** a replication of WikiChat; **not** “better factuality than WikiChat” **without** running their evaluation protocol.

### 9.2 CA-RAG / similarity validation (e.g. Collini et al., IEEE Access 2025)

- **Their framing:** Context-aware RAG with **similarity** in the loop and emphasis on **context inconsistency**; their full method includes **document-oriented** chunking and **post-processing** in their paper’s design.  
- **This project:** **Standard top-K dense retrieval** first; **post-LLM** similarity is **only** between **final answer** and **retrieved** chunks—**not** “give all document chunks to the LLM” as in their described CA-RAG variant.  
- **Why not implement full CA-RAG:** Their method description includes **different** retrieval/context use (e.g. document-scale choices in the paper); this project keeps **standard top-K FAISS** and adds a **lightweight** post-hoc **answer–evidence** similarity—**fewer moving parts**, easier to run locally.  
- **Defensible differentiation:** **Dual semantic gates** (pre-LLM question–evidence + post-LLM answer–evidence) on a **retrieve-then-read** architecture, with **resource** tuning—explicitly documented in code comments as **not** CA-RAG. **Not** “better than CA-RAG on TriviaQA” unless you **reimplement their setup** and **match** their benchmarks.

### 9.3 What this project adds relative to “vanilla RAG”

1. **Two pre-LLM embedding gates** (retrieval score + answerability).  
2. **Post-LLM embedding alignment** with **abstain/flag** modes.  
3. **Centralized runtime configuration** via env vars.  
4. **Lazy LLM loading** and **truncation** aimed at **laptop-class** hardware.

These points are **architectural differences** from naive RAG; **measurable** “better” requires the evaluation procedure in **§2.5.3**.

---

## 10. Limitations of This Project

| Limitation | Explanation |
|------------|-------------|
| **Indexed subset** | `embedder.py` caps **`MAX_CHUNKS = 10_000`**; full Wikipedia raw shards are **not** all in the index. |
| **Single shard chunker** | `wiki_chunker.py` reads only **`enwiki_namespace_0_0.jsonl`** unless manually edited—other shards ignored for that script. |
| **Embedding model path split** | `embedder.py` uses a **fixed local path**; `rag_pipeline` uses **`EMBEDDING_MODEL`** string. **Mismatch** breaks retrieval semantics without crashing. |
| **Similarity ≠ truth** | High embedding similarity does **not** prove factual correctness or entailment. |
| **Post-alignment false positives/negatives** | Correct short paraphrases may score low; fluent off-topic text may score medium-high. |
| **Corpus quality** | Wikipedia can be wrong or outdated; the system assumes retrieved text is the authority. |
| **`test_search.py` diverges** | Uses hub model name directly—should stay aligned with `EMBEDDING_MODEL` / index build. |
| **No automated evaluation suite** | No bundled benchmark JSONL, metrics, or statistical tests in `tests/`. |
| **Metadata RAM** | Full `meta_test.jsonl` loaded into a list—large indices increase RAM. |
| **Alternate ingest scripts unused** | `wikipedia_download.py` / `wikipedia_ingest.py` produce **different** layouts than `embedder` expects unless reconciled. |

---

## 11. How Limitations Can Be Addressed (Roadmap)

| Limitation | Mitigation |
|------------|------------|
| Small index | Raise `MAX_CHUNKS` (or remove cap), re-run **embedder → faiss_index**; monitor RAM. |
| Single shard | Loop over all `enwiki_namespace_0_*.jsonl` in `wiki_chunker` or merge JSONL first. |
| Model path drift | Read **`MODEL_NAME` from same env as `RAG_EMBEDDING_MODEL`** in `embedder.py`; document in one place. |
| Weak faithfulness signal | Add **NLI** or **token-overlap** checks; require **extractive spans**; human eval on a fixed question set. |
| Post-alignment tuning | Grid-search threshold on a **labeled** dev set; use **`flag`** mode first to log scores. |
| Evaluation | Add `data/benchmark/questions.jsonl` + script to batch `run_rag` and compute accuracy / abstention precision. |
| Scale | For huge N, consider **IVF** or **PQ** FAISS (trade speed/recall); or shard indexes by topic. |

---

## 12. How to Run (Summary)

```bash
conda activate ragx
cd c:\rag_project\src
python retrieval\rag_pipeline.py
```

**Rebuild pipeline from scratch (after changing chunks or model):**

1. `python processing\wiki_chunker.py`  
2. `python processing\wiki_cleaner.py`  
3. `python embeddings\embedder.py`  
4. `python retrieval\faiss_index.py`  
5. `python retrieval\rag_pipeline.py`

---

## 13. `run_rag` Return Schema

| Key | Type | When |
|-----|------|------|
| `answer` | `str` | Always |
| `confidence` | `str` | Always |
| `evidence` | `list[dict]` | Each dict: `title`, `section`, `text` — may be empty on early abstain |
| `alignment_score` | `float` or `None` | Set when post-alignment runs; `None` if disabled or early exit |
| `alignment_passed` | `bool` or `None` | Same |
| `alignment_note` | `str` (optional) | Present in **`flag`** mode when alignment fails |

---

## 14. Document History

- Generated from a **line-by-line review** of the repository source files listed in §6.  
- **§2.5** added: **design rationale** (why this vs alternatives) and **epistemic limits** on claiming superiority without benchmark results.  
- Align with code under `c:\rag_project\src` and `c:\rag_project\tests` as of documentation date; if code changes, update §5–§8 accordingly.

---

*End of documentation.*
