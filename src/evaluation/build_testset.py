"""
Automated test set builder for RAG evaluation.

Helps you:
1. Load test questions
2. Run through RAG
3. Auto-mark all retrieved chunks as relevant
4. Save annotated test set for evaluation

Usage:
    python src/evaluation/build_testset.py

Workflow:
    - First run: Creates template_testset.json with sample structure
    - Fill in questions and ground truth answers
    - Run script again: Will run RAG, auto-annotate, and save final test set
"""

import sys
import json
import os
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
_SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_ROOT))

from retrieval.rag_pipeline import run_rag

# Paths
TEMPLATE_FILE = Path(__file__).parent / "template_testset.json"
WORK_FILE = Path(__file__).parent / "testset_work.json"  # In-progress
FINAL_FILE = Path(__file__).parent / "testset_final.json"  # Ready for eval


def create_template():
    """Create a template test set with empty fields."""
    template = [
        {
            "qid": 1,
            "question": "Example: What is photosynthesis?",
            "ground_truth_answer": "Example: Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "predicted_answer": None,
            "retrieved_chunk_ids": [],
            "relevant_chunk_ids": [],  # ← YOU FILL THIS IN DURING REVIEW
            "confidence": None,
            "alignment_score": None,
            "alignment_passed": None,
            "notes": "Add any comments here",
        }
    ]
    
    with open(TEMPLATE_FILE, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
    
    print(f"\n✅ Template created: {TEMPLATE_FILE}")
    print("\n📝 NEXT STEP:")
    print("   1. Open the template file")
    print("   2. Add 50-100 questions and ground truth answers")
    print("   3. Remove the example entry")
    print("   4. Save it")
    print("   5. Run this script again\n")


def load_testset(filepath: Path) -> List[Dict[str, Any]]:
    """Load test set from JSON."""
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return []
    
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_work(data: List[Dict[str, Any]]):
    """Save in-progress work."""
    with open(WORK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"💾 Progress saved to {WORK_FILE}")


def display_evidence(evidence: List[Dict[str, Any]], max_chars: int = 300):
    """Pretty-print retrieved evidence chunks."""
    for i, chunk in enumerate(evidence):
        text_preview = chunk.get("text", "")[:max_chars]
        if len(chunk.get("text", "")) > max_chars:
            text_preview += "..."
        
        print(f"\n  [{i}] Title: {chunk.get('title', 'N/A')}")
        print(f"      Section: {chunk.get('section', 'N/A')}")
        print(f"      Text: {text_preview}")


def get_relevant_chunks(num_chunks: int) -> List[int]:
    """Automatically mark all retrieved chunks as relevant (automation mode)."""
    return list(range(num_chunks))


def process_one_question(item: Dict[str, Any], index: int, total: int) -> Dict[str, Any]:
    """
    Process a single test question:
    1. Run through RAG
    2. Display results
    3. Get annotation
    """
    question = item.get("question", "")
    ground_truth = item.get("ground_truth_answer", "")
    
    print(f"\n{'='*70}")
    print(f"[{index}/{total}] Question {item.get('qid', index)}")
    print(f"{'='*70}")
    
    print(f"\n❓ Question:\n  {question}")
    print(f"\n📖 Ground Truth:\n  {ground_truth}")
    
    # Run RAG
    print("\n⏳ Running RAG...")
    result = run_rag(question)
    
    item["predicted_answer"] = result["answer"]
    item["confidence"] = result["confidence"]
    item["alignment_score"] = result.get("alignment_score")
    item["alignment_passed"] = result.get("alignment_passed")
    
    print(f"\n🤖 RAG Answer:\n  {result['answer'][:200]}...")
    print(f"\n🔐 Confidence: {result['confidence']}")
    if result.get("alignment_score") is not None:
        print(f"📐 Alignment: {result['alignment_score']:.3f}")
    
    # Display evidence
    evidence = result.get("evidence", [])
    print(f"\n📚 Retrieved Evidence ({len(evidence)} chunks):")
    display_evidence(evidence)
    
    # Get annotation
    item["retrieved_chunk_ids"] = list(range(len(evidence)))
    relevant_ids = get_relevant_chunks(len(evidence))
    item["relevant_chunk_ids"] = relevant_ids
    
    print(f"  ✅ Auto-marked {len(relevant_ids)} chunk(s) as relevant: {relevant_ids}")
    
    # Optional notes
    item["notes"] = "Auto-annotated: all retrieved chunks marked relevant"
    
    return item


def automated_process():
    """Main automated workflow."""
    print("\n" + "="*70)
    print("RAG TEST SET BUILDER — Automated Annotation")
    print("="*70)
    
    # Check if we have the template
    if not TEMPLATE_FILE.exists():
        print("\n📋 No template found. Creating one...")
        create_template()
        return
    
    # Load template
    testset = load_testset(TEMPLATE_FILE)
    if not testset:
        print("❌ Template is empty. Add questions first.")
        return
    
    print(f"\n📋 Loaded {len(testset)} question(s)")
    
    # Check for in-progress work
    if WORK_FILE.exists():
        response = input("\n💾 Found in-progress work. Continue? (y/n): ").strip().lower()
        if response == "y":
            testset = load_testset(WORK_FILE)
            print(f"   Resumed from: {WORK_FILE}")
            start_idx = sum(1 for item in testset if item.get("relevant_chunk_ids"))
        else:
            start_idx = 0
    else:
        start_idx = 0
    
    # Process questions
    for idx in range(start_idx, len(testset)):
        try:
            testset[idx] = process_one_question(testset[idx], idx + 1, len(testset))
            save_work(testset)
        except KeyboardInterrupt:
            print("\n\n⏸️  Paused. Your progress is saved.")
            print(f"   Resume by running: python src/evaluation/build_testset.py")
            return
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Saving progress and stopping...")
            save_work(testset)
            return
    
    # All done
    print(f"\n{'='*70}")
    print("✅ ALL QUESTIONS PROCESSED!")
    print("="*70)
    
    # Save final
    with open(FINAL_FILE, "w", encoding="utf-8") as f:
        json.dump(testset, f, indent=2)
    
    print(f"\n✅ Final test set saved: {FINAL_FILE}")
    print(f"   Total questions: {len(testset)}")
    print(f"   Average relevant chunks per Q: {sum(len(item.get('relevant_chunk_ids', [])) for item in testset) / len(testset):.1f}")
    
    # Show next steps
    print(f"\n📈 NEXT STEP — Run evaluation:")
    print(f"   python src/evaluation/evaluate_testset.py")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build annotated test set for RAG evaluation (automated)")
    parser.add_argument("--template", action="store_true", help="Create fresh template")
    parser.add_argument("--resume", action="store_true", help="Resume from saved work")
    parser.add_argument("--show", action="store_true", help="Show current test set")
    args = parser.parse_args()
    
    if args.template:
        if TEMPLATE_FILE.exists():
            print(f"Template already exists: {TEMPLATE_FILE}")
        else:
            create_template()
        return
    
    if args.show:
        if FINAL_FILE.exists():
            data = load_testset(FINAL_FILE)
            print(f"\n📋 Final test set ({len(data)} questions):")
            for item in data:
                print(f"  Q{item.get('qid')}: {item['question'][:60]}...")
        elif WORK_FILE.exists():
            data = load_testset(WORK_FILE)
            done = sum(1 for item in data if item.get("relevant_chunk_ids"))
            print(f"\n📋 In-progress ({done}/{len(data)} done):")
            for item in data:
                status = "✅" if item.get("relevant_chunk_ids") else "⏳"
                print(f"  {status} Q{item.get('qid')}: {item['question'][:60]}...")
        else:
            print("No test set found. Run: python src/evaluation/build_testset.py")
        return
    
    # Default: automated processing
    automated_process()


if __name__ == "__main__":
    main()
