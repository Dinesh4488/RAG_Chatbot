"""
Run evaluation on your annotated test set.

Loads the finished test set (testset_final.json) and computes all metrics.

Usage:
    python src/evaluation/evaluate_testset.py

Requires:
    - Completed test set saved by build_testset.py
"""

import sys
import json
from pathlib import Path

# Add src to path
_SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_ROOT))

from retrieval.rag_pipeline import embedder
from evaluation.metrics import RAGEvaluator

# Path to annotated test set (created by build_testset.py)
FINAL_FILE = Path(__file__).parent / "testset_final.json"


def load_final_testset() -> list:
    """Load the annotated test set."""
    if not FINAL_FILE.exists():
        print(f"❌ Test set not found: {FINAL_FILE}")
        print("\n📝 First, build your test set:")
        print("   python src/evaluation/build_testset.py")
        sys.exit(1)
    
    with open(FINAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("\n" + "="*70)
    print("RAG EVALUATION — Running on Annotated Test Set")
    print("="*70)
    
    # Load test set
    testset = load_final_testset()
    print(f"\n✅ Loaded {len(testset)} test questions")
    
    # Create evaluator with embedder
    evaluator = RAGEvaluator(embedder=embedder)
    
    # Evaluate each item
    print("\n📊 Running evaluation...")
    
    for item in testset:
        try:
            evaluator.evaluate_single(
                question=item.get("question", ""),
                predicted_answer=item.get("predicted_answer", ""),
                ground_truth_answer=item.get("ground_truth_answer", ""),
                retrieved_chunk_ids=item.get("retrieved_chunk_ids", []),
                relevant_chunk_ids=item.get("relevant_chunk_ids", []),
                confidence=item.get("confidence", "Unknown"),
                alignment_score=item.get("alignment_score"),
                alignment_passed=item.get("alignment_passed"),
            )
        except Exception as e:
            print(f"⚠️  Error evaluating Q{item.get('qid')}: {e}")
            continue
    
    # Print summary
    evaluator.print_summary()
    
    # Detailed breakdown
    print("\n" + "="*70)
    print("DETAILED RESULTS PER QUESTION")
    print("="*70)
    
    for i, result in enumerate(evaluator.results, 1):
        print(f"\n[{i}] {result['question'][:80]}")
        print(f"    Ground Truth: {result['ground_truth_answer'][:100]}")
        print(f"    Prediction:   {result['predicted_answer'][:100]}")
        
        if result['semantic_similarity'] is not None:
            print(f"    ✨ Semantic Sim: {result['semantic_similarity']:.3f}")
        
        print(f"    Precision@1: {result['precision@1']:.3f} | Recall@3: {result['recall@3']:.3f} | MRR: {result['mrr']:.3f}")
        
        if result['alignment_score'] is not None:
            print(f"    Alignment: {result['alignment_score']:.3f} / {result['alignment_passed']}")


if __name__ == "__main__":
    main()
