# Reproduction protocol

## Scope

This document describes the current frozen paper protocol. It is not a promise that an unconfigured checkout can reproduce the numerical tables without the private benchmark pools, model weights, retrieval caches, and service endpoints used in the A40 runs.

## Main execution paths

- `normal_predict_separate.py`: local and cloud comparison paths.
- `hybrid_predict_separate.py`: CALIPER collaborative path.
- `hybrid_predict_separate_clf.py`: optional execution-classifier path.
- `build_retrieval_cache.py` and `build_retrieval_cache_ft.py`: retrieval cache construction.
- `scripts/eval_table2_final.py`, `eval_table3_full.py`, and `eval_ablation_final.py`: table-level aggregation.
- `scripts/measure_table4_final.py` and `scripts/bootstrap_table5_current_20260930.py`: latency estimate and interval processing.

## Frozen model and threshold binding

The paper uses local `Qwen3.5-9B`, cloud `Qwen3.7-plus`, `bge-base-en-v1.5` initialization, and `bge-reranker-large` initialization. Fine-tuned checkpoints are dataset/configuration-specific and are not included here. Execution thresholds are 0.974, 0.978, and 0.977 for HotpotQA, MuSiQue, and 2Wiki, respectively.

## Evidence boundary

The cloud guidance request receives the question. Retrieved evidence remains in the local execution path. The query and returned guidance can still expose sensitive information; this protocol must not be described as formal privacy protection.

## Latency boundary

Table 5 combines per-question retrieval and generation timings under the adopted protocol. Model loading is excluded. The result is an estimated end-to-end latency, not an integrated live measurement of one uninterrupted request path.

## Evaluation and statistics

Question-level paired comparisons use 10,000 bootstrap resamples and percentile 95% intervals. Threshold selection uses dataset-specific overlap-free validation pools. Current sample IDs, hashes, and detailed provenance remain in the private companion archive. The dated `legacy/work/repair/` scripts are historical provenance, not a clean current entry point.

## Manuscript clarifications synchronized on 2026-10-16

Known benchmark-pool identity selects the execution threshold before source routing; the router predicts retrieval weights. Unknown-source threshold selection is not evaluated. Retrieval, reranking, and generation operate on sentence-level units, without parent-paragraph expansion. MuSiQue adds BASE top-three units and deduplicates normalized text. Table 7 maps selected MuSiQue units to supporting paragraphs for recall scoring only. Table 6 uses HE/n over all audit questions; validation hard-class false-negative rate uses HE/(HE+HH). Historical segmentation rules are retained in Supplementary Configuration Notes Section 12.4.
