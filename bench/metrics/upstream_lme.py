# ruff: noqa
"""Vendored from upstream LongMemEval `src/retrieval/eval_utils.py` (A6).

Upstream repo xiaowu0162/LongMemEval, last commit touching the file: 6a92d1ab7a7f8815525e43048f752f59e1d711a9
(2024-10-13, as reported by the GitHub API on 2026-10-23; `bench fetch` re-checks it). Only change:
`np.asfarray` (removed in NumPy 2) -> `np.asarray(..., dtype=float)`. The function bodies are
otherwise byte-for-byte upstream's, kept so `ndcg_any@10 (upstream)` can be compared with published
numbers. `bench.metrics.ir.ndcg_any_upstream` is the dependency-free equivalent; the test suite
asserts the two agree, and that upstream gives 1.0 where nDCG@10 (std) gives 0.63093 (rank-2 item).
"""

import numpy as np


def dcg(relevances, k):
    """Discounted Cumulative Gain at k."""
    relevances = np.asarray(relevances, dtype=float)[:k]
    if relevances.size:
        return relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, relevances.size + 1)))
    return 0.0


def ndcg(rankings, correct_docs, corpus_ids, k=10):
    """Normalized Discounted Cumulative Gain at k."""
    relevances = [1 if doc_id in correct_docs else 0 for doc_id in corpus_ids]
    sorted_relevances = [relevances[idx] for idx in rankings[:k]]
    ideal_relevance = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal_relevance, k)
    actual_dcg = dcg(sorted_relevances, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def evaluate_retrieval(rankings, correct_docs, corpus_ids, k=10):
    recalled_docs = set(corpus_ids[idx] for idx in rankings[:k])
    recall_any = float(any(doc in recalled_docs for doc in correct_docs))
    recall_all = float(all(doc in recalled_docs for doc in correct_docs))
    ndcg_score = ndcg(rankings, correct_docs, corpus_ids, k)
    return recall_any, recall_all, ndcg_score
