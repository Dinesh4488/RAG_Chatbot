"""
Evaluation metrics for RAG system.

Covers:
- Retrieval metrics (Precision@K, Recall@K, MRR, NDCG)
- Generation metrics (Semantic Similarity, F1, BLEU)
- Alignment metrics (alignment acceptance rate)
- Confidence calibration
- LLM-based evaluation (answer relevance to evidence)

NOTE: EM (Exact Match) is NOT recommended for RAG because LLM paraphrases
based on evidence. Use semantic similarity instead.
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison (lowercase, remove articles, punctuation)."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def exact_match(prediction: str, ground_truth: str) -> bool:
    """
    ⚠️  NOT RECOMMENDED for RAG evaluation.
    1 if normalized prediction matches ground truth, else 0.
    Use semantic_similarity() instead for LLM paraphrases.
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def semantic_similarity(
    prediction: str,
    ground_truth: str,
    embedder=None,
) -> float:
    """
    Cosine similarity between prediction and ground truth embeddings.
    
    Args:
        prediction: Model's answer
        ground_truth: Expected answer
        embedder: SentenceTransformer model (e.g., from rag_pipeline)
    
    Returns:
        Similarity score in [0, 1]. Higher = more similar.
    
    Example:
        >>> from sentence_transformers import SentenceTransformer
        >>> model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        >>> score = semantic_similarity("Paris is the capital", "The capital of France is Paris", model)
        >>> print(f"Similarity: {score:.3f}")
        Similarity: 0.892
    """
    if embedder is None:
        raise ValueError("embedder (SentenceTransformer) is required for semantic_similarity")
    
    pred_vec = embedder.encode(
        [prediction.strip()],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    gt_vec = embedder.encode(
        [ground_truth.strip()],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    
    # Cosine similarity between normalized vectors = dot product
    similarity = float((pred_vec @ gt_vec.T).flatten()[0])
    return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground truth."""
    pred_tokens = set(normalize_answer(prediction).split())
    gt_tokens = set(normalize_answer(ground_truth).split())
    
    if not pred_tokens or not gt_tokens:
        return 0.0 if pred_tokens != gt_tokens else 1.0
    
    common = pred_tokens & gt_tokens
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * (precision * recall) / (precision + recall)


def bleu_score(prediction: str, ground_truth: str, n_gram: int = 4) -> float:
    """Simplified BLEU (1-4 gram precision, geometric mean)."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    precisions = []
    for n in range(1, n_gram + 1):
        if len(pred_tokens) < n or len(gt_tokens) < n:
            precisions.append(0.0)
            continue
        
        pred_ngrams = [" ".join(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)]
        gt_ngrams = [" ".join(gt_tokens[i:i+n]) for i in range(len(gt_tokens) - n + 1)]
        
        matches = sum(1 for ng in pred_ngrams if ng in gt_ngrams)
        precisions.append(matches / len(pred_ngrams) if pred_ngrams else 0.0)
    
    if any(p == 0.0 for p in precisions):
        return 0.0
    return (np.prod(precisions)) ** (1 / n_gram)


def precision_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """Precision@K: fraction of top-K retrieved docs relevant."""
    if k == 0 or not retrieved_ids:
        return 0.0
    
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    matches = len(top_k & relevant)
    return matches / k


def recall_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """Recall@K: fraction of relevant docs in top-K."""
    if k == 0 or not relevant_ids:
        return 0.0
    
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    matches = len(top_k & relevant)
    return matches / len(relevant)


def mean_reciprocal_rank(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    """MRR: reciprocal of rank of first relevant document."""
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """NDCG@K: normalized discounted cumulative gain."""
    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]
    
    dcg = sum(
        (1.0 if doc_id in relevant else 0.0) / np.log2(idx + 2)
        for idx, doc_id in enumerate(top_k)
    )
    
    ideal_k = min(k, len(relevant_ids))
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_k))
    
    return dcg / idcg if idcg > 0 else 0.0


class RAGEvaluator:
    """Full evaluation suite for RAG pipeline."""
    
    def __init__(self, embedder=None):
        """
        Args:
            embedder: SentenceTransformer model for semantic similarity.
                     If None, semantic metrics are skipped.
        """
        self.results = []
        self.embedder = embedder
    
    def evaluate_single(
        self,
        question: str,
        predicted_answer: str,
        ground_truth_answer: str,
        retrieved_chunk_ids: List[int],
        relevant_chunk_ids: List[int],
        confidence: str,
        alignment_score: float = None,
        alignment_passed: bool = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single QA instance.
        
        Args:
            question: User's question
            predicted_answer: Model's answer (from RAG)
            ground_truth_answer: Expected/reference answer
            retrieved_chunk_ids: IDs returned by FAISS (in order)
            relevant_chunk_ids: Ground truth: which chunks are relevant
            confidence: Confidence label from RAG
            alignment_score: Answer-evidence alignment score
            alignment_passed: Whether alignment threshold passed
        """
        
        result = {
            "question": question,
            "predicted_answer": predicted_answer,
            "ground_truth_answer": ground_truth_answer,
            
            # Generation metrics
            # NOTE: EM is included but NOT recommended for RAG
            "exact_match": exact_match(predicted_answer, ground_truth_answer),
            "semantic_similarity": (
                semantic_similarity(predicted_answer, ground_truth_answer, self.embedder)
                if self.embedder
                else None
            ),
            "f1": f1_score(predicted_answer, ground_truth_answer),
            "bleu": bleu_score(predicted_answer, ground_truth_answer),
            
            # Retrieval metrics
            "precision@1": precision_at_k(retrieved_chunk_ids, relevant_chunk_ids, 1),
            "precision@3": precision_at_k(retrieved_chunk_ids, relevant_chunk_ids, 3),
            "precision@5": precision_at_k(retrieved_chunk_ids, relevant_chunk_ids, 5),
            "recall@1": recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, 1),
            "recall@3": recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, 3),
            "recall@5": recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, 5),
            "mrr": mean_reciprocal_rank(retrieved_chunk_ids, relevant_chunk_ids),
            "ndcg@5": ndcg_at_k(retrieved_chunk_ids, relevant_chunk_ids, 5),
            
            # Confidence and alignment
            "confidence": confidence,
            "alignment_score": alignment_score,
            "alignment_passed": alignment_passed,
            "is_abstention": "insufficient evidence" in predicted_answer.lower(),
        }
        
        self.results.append(result)
        return result
    
    def aggregate_metrics(self) -> Dict[str, float]:
        """Compute aggregate metrics over all evaluations."""
        if not self.results:
            return {}
        
        metrics = defaultdict(list)
        for r in self.results:
            metrics["semantic_similarity"].append(r["semantic_similarity"] or 0.0)
            metrics["exact_match"].append(r["exact_match"])  # Keep for reference only
            metrics["f1"].append(r["f1"])
            metrics["bleu"].append(r["bleu"])
            metrics["precision@1"].append(r["precision@1"])
            metrics["precision@3"].append(r["precision@3"])
            metrics["precision@5"].append(r["precision@5"])
            metrics["recall@1"].append(r["recall@1"])
            metrics["recall@3"].append(r["recall@3"])
            metrics["recall@5"].append(r["recall@5"])
            metrics["mrr"].append(r["mrr"])
            metrics["ndcg@5"].append(r["ndcg@5"])
            
            if r["alignment_score"] is not None:
                metrics["alignment_score"].append(r["alignment_score"])
            if r["alignment_passed"] is not None:
                metrics["alignment_passed"].append(r["alignment_passed"])
        
        # Compute means
        summary = {}
        for key, values in metrics.items():
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
        
        # Abstention analysis
        n_abstain = sum(1 for r in self.results if r["is_abstention"])
        summary["abstention_rate"] = n_abstain / len(self.results) if self.results else 0.0
        summary["n_samples"] = len(self.results)
        
        # Alignment analysis
        if metrics["alignment_passed"]:
            n_aligned = sum(metrics["alignment_passed"])
            summary["alignment_pass_rate"] = n_aligned / len(metrics["alignment_passed"])
        
        return summary
    
    def print_summary(self):
        """Print human-readable summary."""
        agg = self.aggregate_metrics()
        
        print("\n" + "="*70)
        print("RAG EVALUATION SUMMARY")
        print(f"Total samples: {agg.get('n_samples', 0)}")
        if agg.get('n_samples', 0) < 50:
            print("⚠️  WARNING: <50 samples may not be statistically significant")
        print("="*70)
        
        print("\n📊 GENERATION METRICS (✨ RECOMMENDED for RAG):")
        print("  (Use semantic_similarity, not exact_match, since LLM paraphrases)")
        for key in ["semantic_similarity_mean", "f1_mean", "bleu_mean"]:
            if key in agg:
                print(f"  {key:30s} {agg[key]:.4f} ± {agg.get(key.replace('_mean', '_std'), 0):.4f}")
        
        print("\n📊 GENERATION METRICS (reference only):")
        print("  (exact_match too strict for LLM paraphrases)")
        if "exact_match_mean" in agg:
            print(f"  {'exact_match_mean':30s} {agg['exact_match_mean']:.4f} ± {agg.get('exact_match_std', 0):.4f}")
        
        print("\n🎯 RETRIEVAL METRICS:")
        for key in ["precision@1_mean", "precision@3_mean", "recall@3_mean", "mrr_mean", "ndcg@5_mean"]:
            if key in agg:
                print(f"  {key:30s} {agg[key]:.4f} ± {agg.get(key.replace('_mean', '_std'), 0):.4f}")
        
        print("\n🔗 ALIGNMENT & ABSTENTION:")
        for key in ["alignment_pass_rate", "abstention_rate"]:
            if key in agg:
                print(f"  {key:30s} {agg[key]:.4f}")
        
        print("\n" + "="*70 + "\n")
        
        return agg


def evaluate_dataset(
    qa_dataset: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Evaluate a complete QA dataset.
    
    Expected format for each item:
    {
        "question": "...",
        "predicted_answer": "...",
        "ground_truth_answer": "...",
        "retrieved_chunk_ids": [0, 1, 2],  # IDs returned by FAISS
        "relevant_chunk_ids": [1, 5, 12],  # IDs marked as relevant (ground truth)
        "confidence": "High",
        "alignment_score": 0.65,
        "alignment_passed": True,
    }
    """
    evaluator = RAGEvaluator()
    
    for item in qa_dataset:
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
    
    evaluator.print_summary()
    return evaluator.results, evaluator.aggregate_metrics()
