"""
Example: Run evaluation on your RAG pipeline.

⚠️  IMPORTANT NOTES FOR RAG EVALUATION:

1. LLM PARAPHRASING: The model will NOT give exact answers like ground truth.
   It paraphrases based on retrieved evidence. Use semantic_similarity, not exact_match.

2. TEST SET SIZE: This example has only 3 samples. For proper evaluation:
   - Minimum: 50 test questions (provides ~±14% confidence interval)
   - Better: 100+ test questions (provides ~±10% confidence interval)
   - Best: 200+ for statistical significance

3. GROUND TRUTH PREPARATION: For each test question, you must:
   a) Write a reference answer
   b) Mark which retrieved chunks are relevant (by chunk ID)
   c) Record model's output (from RAG)

Usage:
    python src/evaluation/eval_example.py
"""

import sys
from pathlib import Path

# Add src to path
_SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_ROOT))

from retrieval.rag_pipeline import run_rag, embedder
from evaluation.metrics import RAGEvaluator, evaluate_dataset


# ============================================================================
# SAMPLE QA DATASET — EXPAND THIS TO 50-100+ QUESTIONS FOR REAL EVALUATION
# ============================================================================
# NOTE: You should build your test set by:
# 1. Asking domain experts to write reference answers
# 2. Running these queries through your RAG system (using run_rag())
# 3. Manually marking which retrieved chunks are "relevant" 
#    (chunk_id, can find in meta_test.jsonl)
#
# For now, this is just a 3-sample placeholder to show the format.

TEST_DATASET = [
    {
        "question": "What is photosynthesis?",
        "ground_truth_answer": "Photosynthesis is the process by which plants convert light energy from the sun into chemical energy stored in glucose.",
        # Below will be filled by run_rag() automatically
        "predicted_answer": None,
        "retrieved_chunk_ids": [],
        "relevant_chunk_ids": [1, 5, 12],  # MANUALLY LABEL: which chunks contain relevant info
        "confidence": None,
        "alignment_score": None,
        "alignment_passed": None,
    },
    {
        "question": "Who was Albert Einstein?",
        "ground_truth_answer": "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
        "predicted_answer": None,
        "retrieved_chunk_ids": [],
        "relevant_chunk_ids": [25, 30, 45],
        "confidence": None,
        "alignment_score": None,
        "alignment_passed": None,
    },
    {
        "question": "What does DNA stand for?",
        "ground_truth_answer": "DNA stands for deoxyribonucleic acid, a molecule that carries genetic instructions for life.",
        "predicted_answer": None,
        "retrieved_chunk_ids": [],
        "relevant_chunk_ids": [100, 105, 110],
        "confidence": None,
        "alignment_score": None,
        "alignment_passed": None,
    },
]

print("\n⚠️  SAMPLE TEST SET (3 items)")
print("For real evaluation, expand to 50-100+ questions and manually annotate relevant_chunk_ids.")
print("See EVALUATION_GUIDE.md for detailed instructions.\n")


def run_eval_on_dataset(dataset):
    """Run RAG pipeline on each question and collect results."""
    
    print("\n" + "="*70)
    print("Running RAG Pipeline on Test Dataset...")
    print("="*70)
    
    for i, item in enumerate(dataset, 1):
        question = item["question"]
        print(f"\n[{i}/{len(dataset)}] Question: {question}")
        
        # Run RAG
        result = run_rag(question)
        
        # Extract results
        item["predicted_answer"] = result["answer"]
        item["confidence"] = result["confidence"]
        item["alignment_score"] = result.get("alignment_score")
        item["alignment_passed"] = result.get("alignment_passed")
        
        # Extract chunk IDs from evidence
        # (NOTE: In rag_pipeline, we don't return IDs directly,
        #  so you may need to modify rag_pipeline.py to return them for evaluation)
        item["retrieved_chunk_ids"] = list(range(len(result.get("evidence", []))))
        
        print(f"  Predicted: {item['predicted_answer'][:100]}...")
        print(f"  Confidence: {item['confidence']}")
        if item['alignment_score'] is not None:
            print(f"  Alignment: {item['alignment_score']:.3f} (passed={item['alignment_passed']})")


def main():
    # Step 1: Run RAG on all questions
    run_eval_on_dataset(TEST_DATASET)
    
    # Step 2: Evaluate with semantic similarity
    print("\n" + "="*70)
    print("Evaluating Results...")
    print("="*70)
    
    # Create evaluator with embedder for semantic similarity
    evaluator = RAGEvaluator(embedder=embedder)
    
    for item in TEST_DATASET:
        evaluator.evaluate_single(
            question=item["question"],
            predicted_answer=item["predicted_answer"],
            ground_truth_answer=item["ground_truth_answer"],
            retrieved_chunk_ids=item["retrieved_chunk_ids"],
            relevant_chunk_ids=item["relevant_chunk_ids"],
            confidence=item["confidence"],
            alignment_score=item.get("alignment_score"),
            alignment_passed=item.get("alignment_passed"),
        )
    
    evaluator.print_summary()
    
    # Step 3: Print detailed results
    print("\n" + "="*70)
    print("DETAILED RESULTS PER QUESTION")
    print("="*70)
    
    for i, r in enumerate(evaluator.results, 1):
        print(f"\n[{i}] {r['question']}")
        print(f"    Ground truth: {r['ground_truth_answer']}")
        print(f"    Prediction:   {r['predicted_answer'][:120]}")
        print(f"    ✨ Semantic sim: {r['semantic_similarity']:.3f} (MAIN METRIC FOR RAG)")
        print(f"    (EM: {r['exact_match']:.1f} | F1: {r['f1']:.3f} | BLEU: {r['bleu']:.3f})")
        print(f"    Precision@3: {r['precision@3']:.3f} | Recall@3: {r['recall@3']:.3f} | MRR: {r['mrr']:.3f}")
        if r['alignment_score'] is not None:
            print(f"    Alignment: {r['alignment_score']:.3f} (passed={r['alignment_passed']})")


if __name__ == "__main__":
    main()
