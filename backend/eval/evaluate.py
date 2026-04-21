"""RAGAS evaluation pipeline for retrieval quality assessment.

Usage:
    python eval/evaluate.py

Metrics:
    - Context Precision: Are retrieved docs relevant?
    - Context Recall: Is important info missing?
    - Faithfulness: Does the answer hallucinate?
    - Answer Relevancy: Does the answer address the question?

Comparisons:
    - With reranker vs without reranker
    - With query rewrite vs without query rewrite
"""

# TODO Day 6: Implement RAGAS evaluation
# - Load eval/test_set.json (50 QA pairs)
# - Run RAG pipeline with reranker ON → collect contexts + answers
# - Run RAG pipeline with reranker OFF → collect contexts + answers
# - Evaluate both with RAGAS metrics
# - Output comparison table to eval/results/
