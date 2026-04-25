# RAG Evaluation Guide

## Critical Issues with Standard Metrics

### ⚠️ Issue 1: LLM Paraphrase vs. Exact Match

**Problem:** The model will NOT produce answers identical to ground truth because it:
- Paraphrases based on retrieved evidence
- May use different phrasing or order same concepts differently
- Is trained to be concise, while ground truth may be verbose

**Example:**
```
Ground Truth: "Albert Einstein was a theoretical physicist who developed the theory of relativity."
Model Output: "The theory of relativity was formulated by Einstein, a theoretical physicist."
→ Exact Match: 0 (WRONG! —答案是正确的)
→ Semantic Similarity: 0.87 (CORRECT!)
```

**Solution:** Use **semantic_similarity** instead of exact_match.

### ⚠️ Issue 2: Test Set Size

**Problem:** 2–3 test samples are statistically meaningless.

| Test Size | Confidence Interval | Reliability |
|-----------|-------------------|------------|
| 3 samples | ±60% | 🔴 Useless |
| 10 samples | ±35% | 🟡 Poor |
| 50 samples | ±14% | 🟢 Acceptable |
| 100 samples | ±10% | 🟢 Good |
| 200+ samples | ±7% | 🟢🟢 Excellent |

**Minimum for your project:** **50 test questions** (ideally 100+)


---

## Overview

The `metrics.py` module provides comprehensive evaluation for your RAG system covering:

- **Generation metrics**: 
  - ✨ **Semantic Similarity** (RECOMMENDED for RAG)
  - Exact Match (not recommended — too strict for paraphrases)
  - F1 score, BLEU
- **Retrieval metrics**: Precision@K, Recall@K, Mean Reciprocal Rank (MRR), NDCG@K
- **Alignment metrics**: Answer–evidence alignment score and pass rate
- **Calibration**: Abstention rate and confidence analysis

---

## Quick Start

### 1. Basic Evaluation Loop (with semantic similarity)

```python
from src.retrieval.rag_pipeline import run_rag, embedder
from src.evaluation.metrics import RAGEvaluator

# Create evaluator WITH embedder for semantic similarity
eval = RAGEvaluator(embedder=embedder)

# For each test question
question = "What is photosynthesis?"
ground_truth = "Photosynthesis is the process by which plants convert light into chemical energy stored in glucose."

# Run RAG
result = run_rag(question)

# Evaluate
eval.evaluate_single(
    question=question,
    predicted_answer=result["answer"],
    ground_truth_answer=ground_truth,
    retrieved_chunk_ids=[0, 1, 2],
    relevant_chunk_ids=[1, 5],  # Ground truth: which docs are relevant
    confidence=result["confidence"],
    alignment_score=result.get("alignment_score"),
    alignment_passed=result.get("alignment_passed"),
)

# Summary
eval.print_summary()
```

### 2. Evaluate Full Dataset (50-100+ samples)

```python
from src.evaluation.metrics import RAGEvaluator

eval = RAGEvaluator(embedder=embedder)

for test_case in TEST_DATASET:  # Load or define your test set
    result = run_rag(test_case["question"])
    
    eval.evaluate_single(
        question=test_case["question"],
        predicted_answer=result["answer"],
        ground_truth_answer=test_case["ground_truth"],
        retrieved_chunk_ids=[...],  # Extract from result["evidence"]
        relevant_chunk_ids=test_case["relevant_chunk_ids"],  # You manually label
        confidence=result["confidence"],
        alignment_score=result.get("alignment_score"),
        alignment_passed=result.get("alignment_passed"),
    )

eval.print_summary()
```

---

## Metrics Explained

### Generation Metrics

| Metric | Range | When to Use | Example |
|--------|-------|------------|---------|
| **Semantic Similarity** ✨ | [0, 1] | **PRIMARY for RAG** — LLM paraphrases evidence | Ground: "Paris is the capital of France" / Pred: "The French capital is Paris" → 0.92 |
| **Exact Match (EM)** | [0, 1] | **NOT recommended for RAG** — too strict on paraphrases | Same example above → 0 (WRONG!) |
| **F1 Score** | [0, 1] | Token overlap if EM doesn't apply | Same example → 0.80 (better than EM) |
| **BLEU** | [0, 1] | N-gram precision (use with caution) | Same example → 0.65 |

**Key insight:** Semantic similarity is 0–1 where:
- 0.9–1.0: Nearly identical meaning
- 0.7–0.9: Same idea, different wording ✅ GOOD for RAG
- 0.5–0.7: Related but not same
- <0.5: Different topics

### Retrieval Metrics

| Metric | Meaning |
|--------|---------|
| **Precision@K** | Of top-K retrieved, how many are actually relevant? |
| **Recall@K** | Of all relevant documents, how many are in top-K? |
| **MRR** | Reciprocal rank of first relevant document (1/rank) |
| **NDCG@K** | Normalized ranking quality (accounts for position) |

**When to use:**
- **High Precision@1**: Top result is almost always relevant (safe, confident)
- **High Recall@3**: Most relevant docs found in top 3 (comprehensive)
- **High MRR**: Relevant info is found quickly
- **High NDCG**: Good ranking quality overall

### Alignment Metrics

| Metric | Meaning |
|--------|---------|
| **alignment_score** | Cosine similarity between answer and evidence (max over all chunks) |
| **alignment_passed** | True if score ≥ threshold (default 0.36) |
| **alignment_pass_rate** | % of answers that stay aligned with evidence |

**When to use:**
- Catch **hallucinations**: answers that drift from retrieved passages
- Monitor **generator drift**: when retrieval is good but output is off-topic

### Abstention Rate

| Metric | Meaning |
|--------|---------|
| **abstention_rate** | % of questions answered with "Insufficient evidence" |

**When to use:**
- Too high (~50%+): Thresholds may be too strict
- Too low (~0%): System may be over-confident, not filtering weak questions

---

## How to Prepare Test Data

### Step 1: Sample Size Planning

For dataset with 100k+ chunks, you need **at least 50 test questions**, better 100+.

| Coverage Goal | Test Size | Confidence |
|--------------|-----------|-----------|
| Quick validation | 20–30 | ±17% (rough) |
| Baseline comparison | 50 | **±14%** ✅ START HERE |
| Published results | 100–200 | ±10–7% ✅ IDEAL FOR B.TECH REPORT |

### Step 2: Preparation Workflow

1. **Create your test set** (Google Sheets, JSON, CSV):
   ```json
   [
     {
       "question": "What is photosynthesis?",
       "ground_truth_answer": "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose.",
       "qid": 1
     },
     {
       "question": "Who was Marie Curie?",
       "ground_truth_answer": "A physicist and chemist who conducted pioneering research on radioactivity.",
       "qid": 2
     }
     ... 50+ more ...
   ]
   ```

2. **Run through RAG** and record retrieved chunks:
   ```python
   from src.retrieval.rag_pipeline import run_rag
   
   result = run_rag(question)
   # result["evidence"] = [
   #   {"title": "...", "section": "...", "text": "..."},
   #   ...
   # ]
   ```

3. **Annotate relevant chunks** — for each test case:
   - Read the retrieved chunks
   - Mark which ones are relevant to the ground truth (by their ID/index)
   - Example: `relevant_chunk_ids: [0, 2]` (chunks 0 and 2 are relevant)

4. **Build final evaluation dataset:**
   ```json
   [
     {
       "question": "What is photosynthesis?",
       "ground_truth_answer": "Photosynthesis is...",
       "predicted_answer": "Photosynthesis is the process...",  # From run_rag()
       "retrieved_chunk_ids": [0, 1, 2],  # Returned by FAISS
       "relevant_chunk_ids": [0, 2],      # YOU LABEL (chunks with good info)
       "confidence": "High",
       "alignment_score": 0.68,
       "alignment_passed": true
     },
     ...
   ]
   ```

### Option 1: Manually Annotate (Recommended)

1. Ask 1–2 domain experts to read 50 questions + chunks
2. For each, mark: "Which chunks are relevant?"
3. If disagreement, take intersection or average
4. Record in the JSON format above

### Option 2: Use Existing Benchmark

**SQuAD** (Simple Question Answering over Wikipedia) is a good fit:
- Extractive QA format (answer span from passage)
- 100k+ question–passage pairs
- Can filter to Wikipedia articles in your corpus

Download: https://rajpurkar.github.io/SQuAD-explorer/

---

## How to Build Your 50-100 Test Set

### Quick Workflow (1–2 hours for 50 questions)

1. Pick 50 diverse topics from Wikipedia in your corpus
2. For each topic, write 1 question (e.g., "What is X?")
3. Write reference answer (2–3 sentences)
4. Run `run_rag(q)` and record result
5. Manually check result["evidence"] and mark which chunks are relevant
6. Save to JSON and run evaluation

---

## Example: Calculate Metrics on Your System

```python
from src.evaluation.metrics import exact_match, f1_score, precision_at_k

# Single metrics
print(exact_match("Paris", "paris"))  # True → 1.0
print(f1_score("The capital of France is Paris", "Paris"))  # ~0.5
print(precision_at_k([0, 1, 2, 5], [1, 2], k=3))  # 2/3 = 0.67
```

---

## Running Full Evaluation

```bash
cd C:\rag_project
python src/evaluation/eval_example.py
```

This will:
1. Run RAG on sample test questions
2. Calculate all metrics **including semantic similarity**
3. Print aggregated summary with sample size warning
4. Show per-question breakdown

---

## Example Output with Semantic Similarity

```
======================================================================
RAG EVALUATION SUMMARY
Total samples: 50
======================================================================

📊 GENERATION METRICS (✨ RECOMMENDED for RAG):
  (Use semantic_similarity, not exact_match, since LLM paraphrases)
  semantic_similarity_mean    0.7850 ± 0.1240
  f1_mean                      0.6850 ± 0.2150
  bleu_mean                    0.4230 ± 0.1890

📊 GENERATION METRICS (reference only):
  (exact_match too strict for LLM paraphrases)
  exact_match_mean             0.3200 ± 0.4700  ← ⚠️ TOO LOW!

🎯 RETRIEVAL METRICS:
  precision@1_mean             0.7400 ± 0.4400
  precision@3_mean             0.6200 ± 0.2100
  recall@3_mean                0.5100 ± 0.3200
  mrr_mean                      0.6450 ± 0.2340
  ndcg@5_mean                  0.6120 ± 0.1560

🔗 ALIGNMENT & ABSTENTION:
  alignment_pass_rate          0.8500
  abstention_rate              0.1200

======================================================================
```

**Key insight:**
- Semantic similarity (0.785) shows the model generates correct answers
- Exact match (0.32) is meaningless because of paraphrasing
- Retrieval metrics show you're finding relevant chunks 74% of the time at rank 1

---

## Interpreting Semantic Similarity Thresholds

| Semantic Sim | Interpretation | Example |
|--------------|----------------|---------|
| 0.90–1.0 | Near-perfect match | Ground: "Paris is the capital of France" / Pred: "The capital of France is Paris" |
| 0.75–0.90 | Good match, minor paraphrase | Ground: "Einstein developed relativity" / Pred: "Relativity was formulated by Einstein" |
| 0.60–0.75 | Acceptable, different focus | Ground: "Photosynthesis converts light to energy" / Pred: "Plants use photosynthesis to make glucose" |
| 0.45–0.60 | Weak or partial match | Ground: "The process requires CO2" / Pred: "Light is needed for photosynthesis" |
| <0.45 | Wrong answer or off-topic | Ground: "What is X?" / Pred: "Something about Y" |

**For your report:** Semantic similarity >0.7 across 100 samples indicates your RAG is **generating correct answers**.

---

## Tips for Improvement

### If semantic_similarity is low (<0.60):

**Possible causes:**
1. **Weak retrieval**: Top chunks don't contain relevant info
   - Check: Is `precision@3` low?
   - Fix: Lower `SIM_THRESHOLD` or `ANSWERABILITY_THRESHOLD` in `runtime_settings.py`

2. **Bad ground truth answers**: Your reference answers don't match evidence
   - Check: Are retrieved chunks actually relevant?
   - Fix: Re-write ground truth closer to Wikipedia text

3. **LLM not following evidence**: Model ignores retrieved chunks
   - Check: Are `alignment_score`s low?
   - Fix: Lower `LLM_TEMPERATURE` (currently 0.15), or lower `POST_ANSWER_ALIGNMENT_THRESHOLD`

### If semantic_similarity is high (>0.75) but exact_match is low:

**This is EXPECTED and GOOD!** LLM is paraphrasing correctly. Report semantic_similarity, not EM.

### If precision@1 is low (<0.60):

**Problem:** Top retrieval is often wrong.
- Check if FAISS index is built correctly with `test_index.py`
- Try lowering `SIM_THRESHOLD` (allow weaker matches)
- Consider using the full dataset (via Colab) instead of 10k sample

### If recall@3 is low (<0.50):

**Problem:** Relevant documents not in top-3.
- Increase `TOP_K` in `runtime_settings.py` (but slower)
- Use better embedding model (if not already using `bge-small-en-v1.5`)
- Your corpus may be too small — use full Wikipedia via Colab pipeline

### If alignment_pass_rate is low (<0.70):

**Problem:** Model answers drift from evidence.
- Lower `POST_ANSWER_ALIGNMENT_THRESHOLD` (currently 0.36)
- Check if `LLM_TEMPERATURE` is too high (encourage randomness)

### If abstention_rate is too high (>0.30):

**Problem:** System refuses too many questions.
- Raise `SIM_THRESHOLD` (currently 0.55)
- Raise `ANSWERABILITY_THRESHOLD` (currently 0.62)
- Verify your `relevant_chunk_ids` annotations are correct

### To publish / write report:

1. **Report semantic_similarity, not EM:**
   ```
   "Our system achieves 0.78 semantic similarity on 100 test questions,
    indicating correct and faithful answer generation."
   ```

2. **Show before/after gates:**
   ```
   Vanilla RAG (no gates):      semantic_sim = 0.62, hallucination = 15%
   Full pipeline (with gates):  semantic_sim = 0.78, hallucination = 3%
   ```

3. **Justify test set size:**
   ```
   "We evaluated on 100 questions (±10% confidence interval) to ensure
    statistical significance for a corpus of 100k+ chunks."
   ```

4. **Document annotation process:**
   ```
   "Relevant chunks were annotated by [X experts] with [Y% inter-annotator
    agreement], using the [method: manual reading / NLI model / etc.]"
   ```
