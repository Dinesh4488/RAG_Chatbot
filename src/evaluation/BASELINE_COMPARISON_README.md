# Baseline Comparison Guide

## What This Does

Compares two approaches on the same questions:

1. **LLM-only (Baseline)** — Just asking the model without retrieval
   - Shows hallucinations and parametric knowledge
   - Fast but potentially inaccurate on domain facts

2. **RAG Pipeline (Your System)** — Retrieval + generation + gates
   - Evidence-based answers
   - Demonstrates grounding and factuality

## Why Comparison Matters for Your Paper

**Before (LLM-only):**
```
Q: "What did Marie Curie win the Nobel Prize for?"
A: "Marie Curie won the Nobel Prize for her work on 
   atomic structure and quantum mechanics..."
✗ WRONG: She won for radioactivity research
```

**After (RAG):**
```
Q: "What did Marie Curie win the Nobel Prize for?"
Evidence: [Retrieved Wikipedia chunks on Curie]
A: "Marie Curie won the Nobel Prize in Physics (1903) 
   for her research on radioactivity..."
✓ CORRECT: Grounded in retrieved evidence
```

This comparison **proves** your RAG gates reduce hallucination.

---

## Quick Start

### Step 1: Prepare test set
```powershell
cd C:\rag_project
python src/evaluation/build_testset.py
```

(Creates `testset_final.json`)

### Step 2: Run comparison
```powershell
python src/evaluation/compare_baseline.py
```

This will:
1. For each test question:
   - Get LLM-only answer (baseline)
   - Get RAG answer (your system)
   - Compare both against ground truth using semantic similarity
2. Generate two files:
   - `comparison_results.json` — Raw data (for further analysis)
   - `comparison_report.txt` — Formatted report (for your paper)

### Step 3: Use in Paper

```
📄 comparison_report.txt contains:
   - Summary metrics
   - Best/worst examples
   - Results table
   - Recommended text for your paper
```

Copy the suggested text directly into your paper!

---

## Output Example

### Summary Metrics
```
📊 AGGREGATE METRICS:
   Total questions: 50
   LLM-only semantic similarity: 0.6234
   RAG pipeline semantic similarity: 0.7812
   Average improvement: +0.1578 (+25.3%)
   Queries where RAG better: 82.0%

🤐 ABSTENTION RATE:
   LLM-only: 2/50 (4.0%)
   RAG: 6/50 (12.0%)
```

**Interpretation:**
- RAG achieves **+25.3% improvement** in answer quality
- RAG abstains more (12% vs 4%), showing it **correctly rejects uncertain queries**
- This is GOOD — shows the gates are working

### Best Example
```
🟢 BEST IMPROVEMENT:
   Q: "Who invented the telephone?"
   Ground Truth: "Alexander Graham Bell patented the telephone in 1876"
   
   LLM-only (sim=0.521):
   "The telephone was invented by Thomas Edison who developed 
    the carbon transmitter for improved sound quality..."
   
   RAG (sim=0.923):
   "Alexander Graham Bell is credited with inventing the telephone, 
    patenting the device in 1876."
   [Improvement: +0.402]
```

### Table
```
ID  Question                                   LLM-only  RAG       Improvement
1   What is photosynthesis?                   0.642     0.890     +0.248
2   Who was Marie Curie?                      0.745     0.821     +0.076
3   What is DNA?                              0.523     0.876     +0.353
...
```

---

## For Your Paper

The script generates recommended text:

```markdown
## Evaluation: Comparison with Baseline

To demonstrate the effectiveness of the RAG approach with multi-stage gating,
we compared outputs on 50 test questions:

**Key Findings:**

1. **Semantic Similarity Improvement**: Our RAG pipeline achieves 0.781 average
   semantic similarity vs. 0.623 for LLM-only baseline, representing a
   +25.3% improvement.

2. **Consistency**: In 82.0% of queries, RAG produces more accurate
   answers than the LLM baseline.

3. **Grounding & Abstention**: The RAG system abstains on 6 queries
   (12.0%) where evidence is insufficient, compared to 2 for the baseline (4.0%).
   This demonstrates the effectiveness of the similarity gates in reducing hallucination.

4. **Evidence-based Answers**: All RAG answers are grounded in retrieved Wikipedia
   passages, while LLM-only answers rely on parametric knowledge and are prone to
   factual errors on domain-specific questions.
```

Just copy this into your paper with your actual numbers!

---

## Interpretation Tips

### Good Signs (RAG is better):
- ✅ RAG semantic similarity >> LLM-only (>0.10 improvement)
- ✅ RAG abstains more (correctly refuses uncertain queries)
- ✅ RAG answers cite evidence, LLM-only makes up facts
- ✅ Majority of queries show RAG > LLM-only (>70%)

### Signs to Investigate:
- ⚠️ LLM-only sometimes better? Likely retrieval failure
- ⚠️ Small improvement (<5%)? May need better embedding model
- ⚠️ High abstention rate (>30%)? Consider lowering gate thresholds

### Analyzing Failures
Use `comparison_results.json` to find problematic questions:

```python
import json

with open("src/evaluation/comparison_results.json") as f:
    results = json.load(f)

# Find where LLM-only was better
failures = [r for r in results if not r["rag_better"]]
print(f"RAG worse on: {len(failures)} queries")

for r in failures[:3]:
    print(f"\nQ: {r['question']}")
    print(f"LLM-only: {r['llm_only_answer'][:100]}")
    print(f"RAG: {r['rag_answer'][:100]}")
    print(f"LLM better by: {r['improvement']:.3f}")
```

---

## File Locations

| File | Purpose |
|------|---------|
| `compare_baseline.py` | Main comparison script |
| `comparison_results.json` | Raw comparison data (JSON) |
| `comparison_report.txt` | Formatted report (for paper) |
| `testset_final.json` | Your annotated test set (input) |

---

## Runtime Notes

⏱️ **This takes time!**
- 50 questions: ~5–10 minutes (LLM load + generation)
- 100 questions: ~10–20 minutes

The script:
- Loads the LLM once, reuses for all queries
- All computing happens locally (no API calls)

**Pro tip:** Run during lunch or before you go to bed. It's worth the wait!

---

## Customization Options

If you want to modify the comparison, edit `src/evaluation/compare_baseline.py`:

### Change LLM-only prompt
Around line 50, edit `get_llm_only_answer()`:
```python
def get_llm_only_answer(question: str) -> str:
    prompt = f"""
... modify prompt here ...
Question: {question}
"""
```

### Add more metrics
In `compare_single()` function, add:
```python
result["custom_metric"] = custom_value
```

### Change output format
Edit `format_report()` to customize report layout

---

## Recommended Sections for Your Paper

### 1. Introduction / Problem Statement
```
LLMs are powerful but prone to hallucination. Show LLM-only baseline 
and its errors as motivation for RAG.
```

### 2. Methodology
```
In §X, we describe our baseline and comparison methodology.
We evaluate on 50 questions from Wikipedia...
```

### 3. Evaluation / Results
```
[Use the generated report text here with your actual numbers]
Semantic similarity improved by X.X%, demonstrating that 
retrieval-augmented generation with multi-stage gates 
reduces hallucination and improves factuality.
```

### 4. Discussion
```
- Why did some queries still prefer LLM-only? 
- When does retrieval help most?
- What are the limitations?
```

---

## Next Steps

1. **After running comparison:**
   - Review the report
   - Check best/worst examples
   - Look at `comparison_results.json` for patterns

2. **For your paper:**
   - Copy recommended text
   - Add your numbers and findings
   - Reference the comparison in figures/tables

3. **Optional enhancements:**
   - Add manual quality judgment ("hallucination?", "grounded?")
   - Compute additional metrics (readability, length, etc.)
   - Create visualizations (bar charts, score distributions)

---

Questions? Check the code comments in `compare_baseline.py` or ask!
