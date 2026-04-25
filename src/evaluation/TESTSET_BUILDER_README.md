# Test Set Builder — Quick Start

## Overview

The test set builder helps you interactively:
1. Create a list of 50+ questions
2. Run each through the RAG pipeline
3. Mark which retrieved chunks are relevant
4. Evaluate metrics

## Workflow

### Step 1: Prepare Your Questions

Copy the sample test set to start:
```bash
cd C:\rag_project\src\evaluation
cp sample_testset.json template_testset.json
```

Or create fresh:
```powershell
python src/evaluation/build_testset.py --template
```

This creates `template_testset.json` with this structure:
```json
[
  {
    "qid": 1,
    "question": "Your question here?",
    "ground_truth_answer": "Expected answer",
    "predicted_answer": null,  # Filled by script
    "retrieved_chunk_ids": [],  # Filled by script
    "relevant_chunk_ids": [],  # YOU FILL THIS
    "confidence": null,  # Filled by script
    "alignment_score": null,  # Filled by script
    "alignment_passed": null,  # Filled by script
    "notes": ""
  }
]
```

### Step 2: Fill In Questions

1. Open `template_testset.json` in VSCode or any editor
2. Edit the example entry OR add more entries
3. For each question:
   - `qid`: Unique ID (1, 2, 3, ...)
   - `question`: What to ask the RAG system
   - `ground_truth_answer`: Expected/reference answer
   - Leave all other fields empty (script fills them)
4. Aim for **50+ questions** for statistical validity
5. Save the file

Example:
```json
[
  {
    "qid": 1,
    "question": "What is photosynthesis?",
    "ground_truth_answer": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
    "predicted_answer": null,
    ...
  },
  {
    "qid": 2,
    "question": "Who was Albert Einstein?",
    "ground_truth_answer": "A theoretical physicist known for the theory of relativity.",
    "predicted_answer": null,
    ...
  }
]
```

### Step 3: Run the Builder

```powershell
cd C:\rag_project
python src/evaluation/build_testset.py
```

The script will:

1. **Load your template**
   ```
   ✅ Loaded 50 question(s)
   ```

2. **For each question, show:**
   ```
   ❓ Question:
     Your question here?

   📖 Ground Truth:
     Expected answer

   ⏳ Running RAG...

   🤖 RAG Answer:
     Generated answer here...

   🔐 Confidence: High

   📚 Retrieved Evidence (3 chunks):

     [0] Title: Topic Name
         Section: Section Name
         Text: Chunk text preview...

     [1] Title: Another Topic
         Section: Another Section
         Text: More text...
   ```

3. **Ask you to mark relevant chunks:**
   ```
   Enter relevant chunk IDs (comma-separated, or 'none'): 0,2
   ✅ Marked 2 chunk(s) as relevant: [0, 2]
   ```

   **How to decide:**
   - Read each chunk
   - If it contains info relevant to answering the question → mark it
   - Example: Question "What is photosynthesis?" and chunk talks about plants converting light → mark it (0,2)

4. **Optional notes:**
   ```
   Any notes for this question? (or press Enter to skip):
   ```

### Step 4: Resume If Interrupted

If the script stops (e.g., Ctrl+C), your progress is saved in `testset_work.json`.

Resume later:
```powershell
python src/evaluation/build_testset.py
```

You'll be prompted:
```
💾 Found in-progress work. Continue? (y/n): y
   Resumed from: testset_work.json
```

### Step 5: View Your Progress

Anytime, check status:
```powershell
python src/evaluation/build_testset.py --show
```

Shows:
```
📋 In-progress (45/50 done):
  ✅ Q1: What is photosynthesis?...
  ✅ Q2: Who was Albert Einstein?...
  ⏳ Q3: What is DNA?...
  ...
```

### Step 6: Evaluate

Once all questions are annotated, run:

```powershell
python src/evaluation/evaluate_testset.py
```

This computes:
- **Semantic Similarity** (main metric for RAG)
- Retrieval metrics (Precision@K, Recall@K, MRR)
- Alignment scores
- Abstention rate

Outputs:
```
======================================================================
RAG EVALUATION SUMMARY
Total samples: 50
======================================================================

📊 GENERATION METRICS (✨ RECOMMENDED for RAG):
  semantic_similarity_mean    0.7850 ± 0.1240

🎯 RETRIEVAL METRICS:
  precision@1_mean            0.7400 ± 0.4400
  recall@3_mean               0.5100 ± 0.3200
  mrr_mean                      0.6450 ± 0.2340

🔗 ALIGNMENT & ABSTENTION:
  alignment_pass_rate         0.8500
  abstention_rate             0.1200
```

---

## Tips

### Mark Relevant Chunks

When deciding which chunks are relevant:

✅ **Mark it** if:
- It directly answers the question
- It contains key facts needed for the answer
- It's cited/mentioned in the ground truth

❌ **Don't mark** if:
- It tangentially mentions the topic but doesn't help answer
- It's about a different topic with the same name
- It contradicts other retrieved chunks

Example:
```
Question: "What is photosynthesis?"
Ground Truth: "Process by which plants convert light to chemical energy"

Chunk 0: "Photosynthesis involves converting solar energy into glucose."
         → MARK (answers the question)

Chunk 1: "Flowers are part of many plants used in photosynthesis."
         → DON'T MARK (tangential, doesn't explain photosynthesis)

Chunk 2: "Chlorophyll absorbs light during photosynthesis."
         → MARK (key mechanism, relevant to answer)
```

### If No Chunks Are Relevant

If the RAG retriever failed to find relevant info:
```
Enter relevant chunk IDs: none
```

This will mark `relevant_chunk_ids: []`, which is useful for analyzing **retrieval failures**.

### Speed Up the Process

- 50 questions: ~30–45 minutes (if you read carefully)
- 100 questions: ~1–1.5 hours
- Run in multiple sessions and use `--resume`

---

## Example Run

```
$ python src/evaluation/build_testset.py

======================================================================
RAG TEST SET BUILDER — Interactive Annotation
======================================================================

📋 Loaded 50 question(s)

📊 Running evaluation...

======================================================================
[1/50] Question 1
======================================================================

❓ Question:
  What is photosynthesis?

📖 Ground Truth:
  Photosynthesis is the process by which plants convert light energy into chemical energy.

⏳ Running RAG...

🤖 RAG Answer:
  Photosynthesis is the process by which plants and algae use sunlight to produce chemical energy...

🔐 Confidence: High

📚 Retrieved Evidence (3 chunks):

  [0] Title: Photosynthesis
      Section: Overview
      Text: Photosynthesis is the metabolic process used by plants, algae, and certain bacteria...

  [1] Title: Plant Biology
      Section: Energy Production
      Text: In plants, light reactions occur in the thylakoid membrane...

  [2] Title: Chlorophyll
      Section: Function in Photosynthesis
      Text: Chlorophyll is the pigment responsible for absorbing light...

  Chunks retrieved: 3
  ✏️  Review the chunks above and enter relevant IDs.
  Enter relevant chunk IDs (comma-separated): 0,2
  ✅ Marked 2 chunk(s) as relevant: [0, 2]

  Any notes for this question? (or press Enter to skip): Good retrieval, all relevant chunks found
  
💾 Progress saved

[2/50] Question 2
...
```

---

## File Locations

| File | Purpose |
|------|---------|
| `template_testset.json` | Your questions (you edit this) |
| `testset_work.json` | In-progress work (auto-saved) |
| `testset_final.json` | Completed, ready for evaluation |

---

## Troubleshooting

**Script crashes on a question:**
- Progress is auto-saved
- Resume with `python src/evaluation/build_testset.py`
- Check that question for issues (bad ground truth, etc.)

**Can't decide if chunk is relevant:**
- Ask: "Does this chunk help answer the question?"
- If unsure, mark it. Better to over-include than miss relevant info.

**Want to skip a question:**
- Not directly supported, but you can edit `testset_work.json` to remove it

**RAG system takes too long:**
- This is normal for first run (LLM loads)
- Subsequent questions are faster

---

## Next Steps After Evaluation

Once you have metrics:

1. **Document in your report:**
   ```
   "We evaluated on 50 Wikipedia-based questions with semantic similarity 
    as the primary metric (mean 0.78 ± 0.12), accounting for LLM paraphrasing."
   ```

2. **Compare baselines (optional):**
   - Turn off gates and re-evaluate to show improvement
   - Document which gates matter most

3. **Iterate:**
   - If semantic_similarity is low (<0.65), investigate:
     - Are questions hard?
     - Is retrieval weak?
     - Is LLM not following evidence?
   - Adjust thresholds and re-test

---

Happy testing! 🚀
