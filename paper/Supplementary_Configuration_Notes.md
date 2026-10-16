# CALIPER Supplementary Configuration Notes

Version: 2026-10-15, method clarifications only. The retained companion archive is `2026-10-07-v14` and includes its manuscript snapshot, the October 5–7 review clarifications, Section 17, and the current Table 7 numbering. These notes document retained experiments, the Table 2 Hit scoring correction in Section 13, the classifier-case repair in Section 14, the author's query-only provenance confirmation in Section 15, current Table 5 confidence intervals in Section 16, and the four-question 2Wiki annotation and answer repair in Section 17. The September 30 classifier repair reused existing answers; the October 4 repair generates new answers only for four annotation-corrected questions. The October 5–7 revision documents existing samples and quality-audit evidence without changing observations, statistics, or protocols. The October 15 clarification distinguishes classifier-metric denominators, states the known-benchmark evaluation setting for threshold selection, and documents sentence-level retrieval and generation, including historical segmentation differences. No table value, threshold, prediction, or experimental protocol is changed, and this update does not record a new companion-archive release. Paths below are artifact identifiers relative to the original experiment repository, not portable installation requirements.


Version lineage: the October 1 packaged version superseded `2026-09-30-v8`, bound to paper commit `0dc00b8b7632c8316fb3558baa31969680e4f98c`, and added retained-timing Table 5 confidence intervals, the AI-use declaration, revised Highlights, Figure 2 spacing, and standalone figure exports. The October 4 local revision changed only the 2Wiki cells in Tables 3 and 7, their associated statistics, and annotation-repair documentation. The October 5–6 review-consistency revision was packaged in v13. The October 7 follow-up packaged in v14 clarifies the method execution order, query-only provenance wording, and Table 7 mean-recall wording; it changes no experiment, statistic, table value, Figure 3 point, threshold, sample, or evaluation protocol. Artifact paths containing `candidate` retain their original names and do not by themselves determine adoption. Current Table 3 repair statistics are identified in Section 17; the external current-pointer receipt remains authoritative for completed packaging and synchronization.

## 1. Retrieval checkpoints used for Table 7

All initialization families are bge-base-en-v1.5 (embedder) and bge-reranker-large (reranker). Fine-tuned weights, training data construction, and training budgets are not identical.

Checkpoint names below are relative to `retrieve_fine_tuning/models/` unless stated otherwise.

| Dataset / condition | Embedder checkpoint | Reranker checkpoint |
|---|---|---|
| All / BASE | pretrained bge-base-en-v1.5 | pretrained bge-reranker-large |
| HotpotQA / FT-SEP | hotpotqa_bge-embedder-ft | unified_large_ftsep_hotpotqa |
| 2Wiki / FT-SEP | 2wikimultihopqa_bge-embedder-ft | unified_large_ftsep_2wiki |
| MuSiQue / FT-SEP | unified_bgebase_musique_fulltrain_embedder | unified_large_ftsep_musique |
| HotpotQA and 2Wiki / FT-MER | merged_bge-embedder-ft | merged_bge-reranker-ft/checkpoint-5000 |
| MuSiQue / FT-MER | unified_bgebase_musique_firstpos_embedder | merged_bge-reranker-ft_v2 |
| Routed / predicted HotpotQA | hotpotqa_bge-embedder-ft | unified_large_ftsep_hotpotqa |
| Routed / predicted 2Wiki | 2wikimultihopqa_bge-embedder-ft | unified_large_ftsep_2wiki |
| Routed / predicted MuSiQue | unified_bgebase_musique_firstpos_embedder | unified_large_ftsep_musique |

The routed row selects the pair from the predicted source, not the gold source. FT-MER is a dataset-specific system comparator: MuSiQue's embedder is source-specific, while its reranker is trained on merged data. It is not one shared, identically trained embedder/reranker pair deployed unchanged on all three datasets. Table 7 is therefore not a single-factor ablation.

Evidence: `work/rebuild_hp_tw_large_20260921.py`, `work/rebuild_routed_large_20260921.py`, `work/rebuild_unified_musique_cache.py`; retained `merged_cache_build_ckpt5000.py` and HotpotQA/2Wiki checkpoint-5000 cache logs. The current Table 7 receipt retains the earlier FT-MER values and explicitly identifies that scope.

## 2. MuSiQue training-data construction

Both constructions read `musique_full_v1.0_train.jsonl` and split paragraph text using the literal delimiter `. ` in the retained preprocessing scripts. The names refer to positive-pool construction, not a change to the evaluation split.

- First-position: first sentence of each supporting paragraph is a positive. Remaining sentences in supporting paragraphs are placed before sentences from non-supporting paragraphs in the negative pool; the combined pool is truncated to 24 candidates. These are position-based negatives, which may still contain relevant evidence.
- Full-train: all sentences from supporting paragraphs enter the positive pool; all sentences from non-supporting paragraphs enter the negative pool, without the first-position 24-candidate truncation.
- Training samples one positive and seven negatives (group size 8) from these pools. No claim is made that first-position negatives are semantically irrelevant.

Evidence: retained `build_ms_first.py`, `build_fulltrain_jsonl.py`, and the training log `logs/unified_bgebase_musique_embed_20260920.log`.

## 3. Recorded training parameters

| Artifact family | Training JSONL under retrieve_fine_tuning/ | Learning rate | Epoch setting | Per-device batch | Accumulation |
|---|---|---:|---:|---:|---:|
| HotpotQA embedder | hotpotqa_bge-embedder-ft.jsonl | 6e-5 | 2 | 4 | 8 |
| 2Wiki embedder | 2wikimultihopqa_bge-embedder-ft.jsonl | 6e-5 | 2 | 4 | 8 |
| Merged embedder | merged_bge-embedder-ft.jsonl | 6e-5 | 2 | 4 | 8 |
| MuSiQue first-position embedder | musique_firstpos_bge-embedder-ft.jsonl | 6e-5 | 2 | 2 | 8 |
| MuSiQue full-train embedder | musique_fulltrain_bge-embedder-ft.jsonl | 6e-5 | 2 | 2 | 8 |
| Unified HotpotQA reranker | hotpotqa_bge-reranker-ft.jsonl | 6e-5 | 3 | 2 | 8 |
| Unified 2Wiki reranker | 2wikimultihopqa_bge-reranker-ft.jsonl | 6e-5 | 3 | 2 | 8 |
| Unified MuSiQue reranker | musique_fulltrain_bge-reranker-ft.jsonl | 6e-5 | 3 | 2 | 8 |
| Merged reranker, checkpoint-5000 | merged_bge-reranker-ft.jsonl | 6e-5 | 3 configured; selected checkpoint earlier | 4 | 8 |
| Merged reranker v2 | merged_bge-reranker-ft.jsonl | 2e-5 | 2 | 4 | 8 |

The checkpoint-5000 trainer state records global step 5000 and epoch 0.575895878025253; it must not be described as a completed three-epoch checkpoint. All listed runs record seed 42, group size 8, and query/passage limits 512; reranker combined maximum length is 512. Embedder logs record InfoNCE temperature 0.02. The unified September 20 training runs use four GPUs; no identical effective batch size is asserted across all historical runs.

Evidence logs: `ft_embed_hotpotqa.log`, `ft_embed_2wiki_retry.log`, `ft_mer_embedder.log`, `ft_mer_reranker.log`, `train_merged_v2.log`, `unified_bgebase_musique_embed_20260920.log`, `unified_large_ftsep_20260920_final.log`; selected checkpoint trainer state. The archive includes source records and hashes. Logged settings and checkpoint identity are distinguished from claims of exact numerical retraining reproducibility.

## 4. Table 4 V2 and generation controls

V2 uses the frozen dataset-specific `data/<dataset>/test_retrieved.json` BASE top-five context. It retains query-only cloud instructions, uses the same two local generation prompt forms as V1, and runs temperature 0, seed 42, thinking disabled, and 256/64 token limits. It does not invoke the source router or append an additional MuSiQue BASE top-three support stream. The retained V2 run predates the MuSiQue retrieval update; its historical cache is not silently relabeled as that newer cache.

Consequently, V1 versus V2 compares retrieval configurations, not only source-router classification. V2's output values remain 69.53/52.31/72.63 F1. V4 is the separately frozen generation-aligned BASE baseline, not the old V4 output of the joint V2/V4 script.

Evidence: `work/router_e2e_v2v4_rerun_20260919.py`, its imported `work/router_e2e_rerun_20260918.py`, finalization scripts, and retained metrics. No new inference was run for these notes.

## 5. Operational classifier and latency scope

The overlap-free operational-difficulty classifier binary, split records, and run manifest support reproduction of the current frozen-model inference and threshold selection. The original historical training command and hyperparameter log are unavailable; the adopted classifier is instead documented by its retained training/validation inputs, model hash, and validation manifest.

For Table 2 and Figure 3, the classifier chooses between retained local two-stage baseline predictions (`data/<dataset>/<dataset>_baseline_2stage_soft.json`) and current core predictions. Easy predictions are reused, not regenerated through the current source router. MuSiQue's finalizer reuses the same composition code with the updated core run.

Table 5 reports estimated end-to-end latency on one fixed 200-question sample per dataset, shared by all methods within that dataset. Retrieval and generation are timed separately and combined per question according to each execution branch. The local baseline uses one call with a 128-token limit, while CALIPER uses 96/32-token local reasoning/extraction limits; classifier time is not counted twice. This protocol is distinct from the Table 2 quality-generation configuration. The adopted latency result has 200 valid rows for every method and dataset. The original HotpotQA index 177 is retained as a failed content-inspection record and replaced by source index 3716 selected by the deterministic seed-42 extension rule; the replacement was measured for all four methods, with no timing imputation. Model loading is excluded.

MuSiQue sample lineage (clarified October 5): the adopted latency sample is `musique_resample_seed20260922_v1`, replacing the earlier sample in `full_generation_table5_20260922_v1`. After inspecting the earlier results, including a slow successful request, the authors requested a fresh sample. Python `random.Random(20260922).sample(eligible_indices, 200)` selected 200 questions without replacement from the common 2,403-question intersection, and `SEALED_SAMPLE.json` records sealing at 2026-09-22 15:17:50 +08:00, before the replacement sample's measurements. All four methods used those same questions. The previously observed slow question was not drawn; no individual question in the replacement sample was deleted or replaced based on its measured latency, and successful slow requests were retained. Sealing before the new measurements does not make the decision to replace the earlier sample outcome-blind; the reported estimates and intervals are conditional on the adopted sample, and sample-selection uncertainty is not quantified. Differences between the two runs also involve different questions and measurement periods, not a controlled model-change comparison.

The sample receipt and its SHA-256 (`71364e60885e3ac3242c082ba845f1f94dc97beba98e23d2908ea7b13652179e`) are retained under `latency_runs/musique_resample_seed20260922_v1/SEALED_SAMPLE.json`, with `REPORT.md`, `VALIDATION_RECEIPT.json`, and `analysis/paired_composition.json` identifying the rerun. Subsequent classifier recomposition retains this sample and yields the currently displayed CALIPER+clf values; the original rerun report's 0.997 operating point is not the current 0.978 operating point. Earlier samples and timing records remain historical evidence, not additional observations in the current 200-question estimate.

Evidence: `classifier_analysis/overlap_free_retrain_20260927/figure3_table5_candidate_20260929_v2/`, `latency_runs/table5_repair_20260929_v1/`, `work/finalize_router_e2e_20260918.py`, and `work/finalize_router_e2e_musique_unified_bgebase_20260920.py`.

## 6. Rounding convention

Narrative differences based on printed table cells use those displayed values. Bootstrap estimates, confidence intervals and p-values retain unrounded per-question scores: MuSiQue V0 minus V1 is 2.0555 points (2.06 rounded), while 53.61 minus 51.56 is 2.05. Neither underlying predictions nor bootstrap results are changed.

## 7. Evaluation input provenance and archive locator

The 2Wiki study pool was loaded from the remote experiment repository's `data/2wikimultihopqa/test.json`: 12,576 records with the fields `id`, `question`, `answers`, `ctxs`, `supporting_facts`, and `type`. The source file's SHA-256 is `5d4aa10f016ee02994ebfaf321dcc309c2a23894e20902fe61fb9ebf5`. The checksum is recorded in the retrieval-run metadata at `retrieval_table8/routed_large_20260921/manifest.json`; the adopted Table 7 result is identified by `retrieval_table8/routed_large_20260921/FREEZE_RECEIPT.json`.

The local filename does not establish the official dataset split. The file contains answer and supporting-fact annotations, whereas the official 2Wiki dataset README states that these labels are absent from its test data (https://github.com/Alab-NII/2wikimultihop/blob/main/README.md). The retained records do not establish an item-ID match to a specific official development release. We therefore describe it in the manuscript as a labeled 2Wiki study pool and do not call it the official test or development split. No new data or evaluation was run for this clarification.


## 8. Detailed result comparisons retained during prose revision

Added 2026-09-23. The following detailed result records were moved from the main results narrative during its concision revision. They retain the numerical comparisons, timing assumptions, and deterministic example-selection rule; no experiment, sample, metric, or result was changed. LaTeX citation and table labels refer to the accompanying manuscript. At that time, this document update did not itself update or refreeze the companion archive; the current v13 archive contains the synchronized record.

### 8.1 Performance Comparison

```tex
Table~\ref{main} shows that CALIPER core reaches F1 scores of 71.37, 51.56, and 75.62 on HotpotQA, MuSiQue, and 2WikiMultiHopQA, respectively. It exceeds strict Naive RAG by 5.91, 11.79, and 19.92 points, respectively (paired $p=0.0002$ for each comparison). The numerical gaps over adapted Adaptive-RAG are 3.88, 8.13, and 15.07 points, also with paired $p=0.0002$; however, the adapted router's 44.44\% validation accuracy limits this comparison, so it is not interpreted as evidence of superiority over a strong reproduction of Adaptive-RAG. The more informative workflow control is Always-Multi: CALIPER shows no statistically significant difference from Always-Multi on HotpotQA (+0.59, $p=0.166$), but is significantly better on MuSiQue (+4.53) and 2Wiki (+13.42), both $p=0.0002$. Thus, the benefit beyond always selecting the multi-step branch is dataset-dependent rather than universal. One plausible interpretation is that HotpotQA's multi-hop structure and supporting-fact supervision already allow the Always-Multi branch to cover much of the required cross-document reasoning, leaving less room for dynamic routing to improve it; because all three benchmarks are multi-hop, we treat this as an explanatory hypothesis rather than a demonstrated causal mechanism. These CALIPER contrasts use the current Scheme-A source-router rerun; the retained baseline rows follow the definitions in Table~\ref{main}.

CALIPER+clf trades quality for fewer collaborative calls. At the dataset-specific thresholds 0.974, 0.978, and 0.977, its F1 scores are 70.61, 49.11, and 75.29 on HotpotQA, MuSiQue, and 2Wiki, respectively. Each threshold-selection pool is disjoint from its corresponding dataset's evaluation questions and classifier training questions; this is not a global exclusion across all three benchmarks. The reported classifier remains specific to the evaluated local model and prompts. Qwen3.7-plus remains stronger than CALIPER core by 5.63, 3.66, and 1.57 F1 points, respectively, in Table~\ref{main}; thus, the result supports a quality--locality trade-off rather than universal cloud-quality parity.

With fully relevant evidence (Table~\ref{inf}), CALIPER improves over the local model by 6.02, 6.34, and 10.82 F1 points on HotpotQA, MuSiQue, and 2Wiki, respectively (all $p=0.0002$), but Qwen3.7-plus still leads by 7.97, 13.42, and 7.97 points, respectively, using displayed cells. All methods use 7,405/2,410/12,576 questions after the Table 3 exclusion in Section 14 and the four-question annotation repair in Section 17. The controlled setting therefore identifies both the benefit of collaborative guidance and the remaining local-model capacity ceiling.
```

### 8.2 Ablation Study

```tex
Table~\ref{ablation} summarizes the component ablations under the same question sets as the main comparison. V0, V1, and V3 use the current Scheme-A text-grouped source-router rerun; V2 uses the retained BASE-plus-cloud-guidance rerun; V4 uses the BASE top-5 cache with the same local generation prompts and settings as V3. The retrieval-only source-router audit is reported separately in Table~\ref{retrieval}.


V2 replaces the entire routed retrieval context with the frozen BASE top-five cache, retains query-only cached cloud guidance, and uses the same two local generation prompts as V1 (temperature 0, seed 42, thinking disabled, token limits 256/64). It does not add MuSiQue's separate BASE top-three support stream. Consequently, V1--V2 is a retrieval-configuration comparison, not an isolated source-router-classifier effect.

V1--V3 assesses collaborative guidance, with descriptive F1 differences of $-0.38$, $+10.54$, and $+1.23$ on HotpotQA, MuSiQue, and 2Wiki. These gains are dataset-dependent; no significance claim is made for this contrast.

V1 exceeds V2 on HotpotQA (+1.84 F1, $p=0.0002$) and 2Wiki (+2.99, $p=0.0002$), but not on MuSiQue (-0.75, $p=0.304$). With local generation fixed, V3 exceeds V4 by 4.75 points on HotpotQA (95\% CI $[3.96,5.55]$, $p=0.0002$) and 11.40 on 2Wiki ($[10.74,12.08]$, $p=0.0002$); the MuSiQue difference is 0.46 ($[-1.17,2.14]$, $p=0.589$). These exploratory p-values are unadjusted. V3--V4 compares retrieval configurations, not collaborative guidance or isolated routing/dual-stream effects. Neither retrieval comparison establishes a MuSiQue gain. V1 exceeds V4 by 4.37, 11.00, and 12.63 points, respectively. MuSiQue V0--V2 (53.61 vs. 52.31) changes both retrieval and Reflection, unlike the paired V0--V1 contrast reported above. The latter remains an output-level association: absent complete reflection traces, it cannot decompose evaluator/corrector effects. A separate frozen HotpotQA audit of 200 questions with gold supporting evidence finds 96\% immediate acceptance, with mean/median/maximum revision iterations of 0.08, 0, and 3 and overheads of 0.92, 0.17, and 27.67 seconds; F1 rises from 72.49 to 73.49.
```

### 8.3 Qualitative Error Analysis

```tex
To avoid selecting favorable examples manually, we define three MuSiQue candidate groups and choose the smallest SHA-256 value of \texttt{42|question} in each group. A success requires CALIPER F1$=1$ and strict Naive RAG F1$=0$; a failure reverses these conditions; and a classifier error requires a true-hard query routed locally with collaborative F1$=1$ and local F1$=0$.

\textbf{Collaborative success.} For ``In 2002, what company merged with the top PC manufacturer from 1994?'', the gold answer is \textit{Hewlett Packard}. Both retrieval streams place the merger sentence first. Naive RAG answers \textit{Compaq}, confusing the acquired company with the resulting company, whereas CALIPER resolves the relation and returns the gold answer. This example is consistent with a possible reasoning-direction benefit, but does not isolate a causal effect of guidance or rule out other context differences.

\textbf{Benchmark-sensitive failure.} For ``When did the state where Copake Lake is located reinstate the death penalty?'', strict Naive RAG returns the gold label \textit{1984}, while CALIPER refuses to infer a date. The retrieved evidence identifies Copake Lake as being in New York, but the sentence containing 1984 describes Oregon. Thus, CALIPER is scored as wrong under the frozen label even though the evidence chain is internally inconsistent. This example cautions against treating every token-level error as a model failure.

\textbf{Classifier misrouting.} For the MuSiQue question ``Who did the performer of Night Rocker play on Young and Restless?'', the gold answer is \textit{Snapper Foster}. The query is labeled hard under the frozen three-run operational rule because the local labeling outputs are \textit{Snapper Foster}, \textit{Snapper Foster}, and \textit{David Hasselhoff}; not all three runs recover the gold answer. The overlap-free classifier nevertheless assigns easy probability 0.997155, above the frozen MuSiQue threshold of 0.978, and selects the local-only path, whose retained output is ``Not possible to answer based on the provided text.'' CALIPER's collaborative output returns \textit{Snapper Foster}. This is a hard-to-local routing error, although the paths' different available contexts mean the case does not isolate cloud guidance. The example is the smallest SHA-256 of \texttt{42|question} among the five eligible current-classifier cases, not a manually selected favorable example. Its model, input hashes, and selection record are retained in Section 14.
```

### 8.4 Efficiency and Classifier Analysis

```tex
Table~\ref{time} uses one fixed 200-question sample per dataset, shared by all methods within that dataset. Retrieval and generation are timed separately and combined per question according to each method's execution branch. All three datasets have 200 valid rows for every method in the adopted latency result; the deterministic HotpotQA replacement is documented above. The timing protocol differs from the main quality evaluation: the local baseline uses one 128-token call, while CALIPER uses 96/32-token local reasoning/extraction limits.
Table~\ref{time} reports estimated end-to-end latency as mean (p50/p95), covering retrieval and generation. The dataset-specific samples are fixed before measurement; sample identifiers and random seeds are retained in the companion archive. Every condition uses 20 warm-up questions and concurrency four. The generation stage uses local temperature 0.1 and cloud temperature 0.2; the two-stage cloud reference uses a 256-token first-stage thinking budget. CALIPER has 32.21\%, 48.39\%, and 37.50\% lower estimated end-to-end latency than the two-stage cloud deep-reasoning protocol on HotpotQA, MuSiQue, and 2Wiki, respectively. CALIPER+clf has 37.89\%, 49.63\%, and 38.41\% lower estimated latency than that cloud reference, and remains lower than CALIPER on all three datasets. These are composed estimates rather than integrated live measurements; per-question rows and manifests are retained in the companion archive.



Table~\ref{clf} evaluates the operational-difficulty classifier at dataset-specific thresholds 0.974/0.978/0.977 and reports hard-class recall because hard-to-local errors are the risk-bearing failure mode. The underlying four-way counts are given in Table~\ref{clfconf}; all metrics and counts use these same dataset-specific thresholds. The validation subsets for threshold selection are independent and overlap-free (2,320/3,080/1,371 questions), while the displayed diagnostics use fixed audit subsets of 414/332/2,000 questions. The hard-to-local group accounts for 3.86\%, 3.61\%, and 1.95\% of the HotpotQA, MuSiQue, and 2Wiki audit sets. Figure~\ref{clf_sensitivity} reports a descriptive threshold sweep under the current Scheme-A prediction cache: the source router and both branch outputs are fixed, and the colored dashed lines mark the selected dataset-specific operating points. The sweep is descriptive and is not used to retune the frozen operating points.
```

### 8.4.1 Separate retained generation-quality audit

Evidence locator added October 5; no generation or statistical results are changed. The quality statement in manuscript Section 4.4 refers to `table-result/table4_paired_quality.json`, not the latency intervals in Section 16. The `table4` filename is an original artifact identifier, not the current manuscript's ablation-table number. Its SHA-256 is `d9168020afb45022ab247fa075017bdfbb133286d8a906212488a17bfb515787`; the generating script is `scripts/analyze_table4_paired_quality.py` (SHA-256 `df49914b21b007df0dc709b244706955950a356420c4348398afc4c3fd303247`). Local retained copies match these remote hashes.

The script joins the cloud-reference outputs in `table-result/table4_cloud_deep_<dataset>_full_b256_seed42.json` to frozen full-pool CALIPER predictions by exact question text, with the corresponding `table-result/table4_timing_<dataset>_seed42.json` records determining timing-call eligibility. Rows with a cloud error, a CALIPER timing error, or missing method predictions are omitted by the retained script. Its matched samples contain 199 HotpotQA, 200 MuSiQue, and 200 2Wiki questions. The core prediction sources are `data/hotpotqa/hotpotqa_caliper_planA_top5.json`, `data/musique/musique_v1_ubh_2stream.json`, and `data/2wikimultihopqa/2wikimultihopqa_caliper_planA_top5.json`.

| Dataset | Matched n | CALIPER F1 | Cloud-reference F1 | CALIPER minus cloud F1 | Paired 95% interval | Retained two-sided p |
|---|---:|---:|---:|---:|---|---:|
| HotpotQA | 199 | 66.26 | 73.58 | -7.32 | [-11.80, -3.05] | 0.0004 |
| MuSiQue | 200 | 54.99 | 50.86 | +4.13 | [-2.18, 10.79] | 0.2082 |
| 2Wiki | 200 | 77.36 | 79.63 | -2.27 | [-6.15, 1.48] | 0.2438 |

These are the retained script's 10,000 question-paired bootstrap draws using Python `random.Random`, with seeds 42/43/44 for the core-versus-cloud comparisons. The script uses sorted bootstrap entries 249 and 9749 for the interval and an empirical two-tail proportion without the finite-sample +1 correction used by the main-result analyses; this revision preserves the original audit statistics rather than silently recomputing them. A read-only replay on October 5 reproduces all reported sample counts and mean F1 values from the retained source predictions; confidence intervals and p-values above are taken from the hash-verified audit file.

This is a separate retained-output quality audit, not the quality of the current Table 5 timing run or an evaluation of the newly composed classifier outputs. In particular, only 14 questions overlap between its MuSiQue sample and the adopted seed-20260922 latency sample. Its HotpotQA sample predates the replacement used to obtain 200 valid current latency observations. Failure to detect a significant difference on MuSiQue or 2Wiki does not establish equivalence, and this audit does not form a jointly measured quality--latency comparison with Table 5.

### 8.5 Retrieval-Only Comparison

```tex
Table~\ref{retrieval} evaluates BASE, separately fine-tuned retrievers (FT-SEP), a merged-data retriever (FT-MER), and CALIPER with the source router applied per question. The text-grouped source router reaches 92.87\% accuracy over the 24,274 held-out evaluation-pool questions, corresponding to a 7.13\% misrouting rate. CALIPER's routed values are below the source-known FT-SEP reference on HotpotQA and 2Wiki. On MuSiQue, they are lower at R@3 and R@5 but higher at R@10 (87.19 versus 86.94); the different first-position and full-train checkpoints preclude interpreting this as a gain caused by routing. CALIPER has higher mean recall than FT-MER at all three cutoffs on HotpotQA and 2Wiki, but lower mean recall than FT-MER at all three cutoffs on MuSiQue. On MuSiQue, CALIPER has higher mean recall than BASE at all three cutoffs. Consequently, the retrieval-only table does not establish uniform superiority, does not provide a downstream F1 estimate for the current router, and does not directly evaluate the separate MuSiQue dual-stream generation context.
```

## 9. Experimental setup details retained during condensation

Updated 2026-09-30. The following source text preserves sample exclusions, split-provenance caveats, baseline training settings, routing diagnostics, statistical details, and branch-specific latency assignments moved out of the main setup. The execution classifier, dataset-specific thresholds, Table~\ref{main}, Figure~\ref{clf_sensitivity}, Table~\ref{time}, and classifier diagnostics were adopted in the September 29 freeze. The September 30 Table 3 exclusion and current-classifier recomputation are recorded in Section 14; unaffected ablation and retrieval results remain unchanged. References below use the manuscript's LaTeX labels. Detailed V0--V1 intervals are retained here rather than printed in the main text.

```tex
\subsection{Experiments Setup}
The evaluated workflow comparisons include strict Naive RAG, adapted Adaptive-RAG, and its Always-Multi control. HippoRAG, Self-RAG, CRAG, Reflexion, and RouteLLM are discussed as related approaches, not evaluated baselines.

\begin{itemize}
\item \textbf{Datasets}: To evaluate multi-hop reasoning across distinct benchmark construction processes, we use HotpotQA \cite{yang2018hotpotqa}, MuSiQue \cite{trivedi2022musique}, and 2WikiMultiHopQA \cite{ho2020constructing}. Training and evaluation split sizes are summarized in Table~\ref{size}. All three are Wikipedia-derived; they do not constitute a broad real-world domain-generalization test.
\begin{itemize}
    \item \textbf{HotpotQA} is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.
    \item \textbf{MuSiQue} is a new multi-hop question answering dataset, containing 2-4-hop questions with different combinatorial structures. MuSiQue is more challenging and harder to cheat than other multi-hop datasets.
    \item \textbf{2WikiMultiHopQA} introduces evidence in every sample that contains comprehensive and concise information to explain the prediction, interpreting the predictions and testing the reasoning skills of the model.
\end{itemize}
\end{itemize}



The evaluation pools in Table~\ref{size} are the study inputs used for the reported benchmark metrics. The HotpotQA pool has 7,405 questions, matching the distractor-validation count; MuSiQue-Ans dev has 2,417 source rows. The 2Wiki input is the labeled local study file \texttt{data/2wikimultihopqa/test.json} (12,576 questions). Its name is a local artifact name, not an official split designation; the exact upstream release has not been confirmed by item-ID matching, so we do not call this pool official dev or test. The file checksum and archive manifest locator are provided in the Supplementary Configuration Notes. MuSiQue uses table-specific samples: Tables~\ref{main} and~\ref{ablation} use the common 2,403-question paired intersection, excluding all 14 rows in seven canonical-question collision pairs to ensure unambiguous question matching. Table~\ref{inf} starts from 2,412 exact-question-deduplicated records and excludes two records matching one normalized classifier-training question, leaving the same 2,410 records for all four methods; Section 14 gives the exclusion and recomputation records. Table~\ref{retrieval} instead uses MuSiQue-Full dev (4,834 source rows): 4,293 questions have both supporting-paragraph annotations and at least one parsed sentence for recall scoring; the other 541 are skipped. This is a retrieval-metric eligibility filter: rows without a supporting annotation or parsed candidate sentence cannot contribute a support-recall score.







\begin{itemize}
\item \textbf{MuSiQue sample alignment}: The retrieval-only audit uses 4,293 eligible MuSiQue-Full dev questions, whereas Tables~\ref{main} and~\ref{ablation} use the common 2,403-question MuSiQue-Ans dev intersection so every end-to-end workflow is evaluated on exactly the same questions. These are distinct dataset variants and evaluation protocols; the retrieval-only eligibility rule is given below Table~\ref{size}.

\item \textbf{Evaluation metrics}: We report SBERT cosine similarity \cite{reimers2019sentence}, normalized answer-substring Hit, and SQuAD-style token precision, recall, and macro F1. SBERT captures semantic similarity but does not guarantee factual correctness; the token and hit metrics remain the primary factoid-QA measures. Pairwise claims use identical questions and 10,000 paired-bootstrap resamples; each comparison's fixed random seed is recorded in the accompanying experimental archive.


\item \textbf{Implementation details}: The source router and operational-difficulty classifier are separate fastText models. Source-routing accuracy on the 24,274-question held-out evaluation-pool audit is 92.87\%: 96.48\% on HotpotQA (7,144/7,405), 66.67\% on eligible MuSiQue-Full dev questions (2,862/4,293), and 99.69\% on the labeled 2Wiki evaluation pool (12,537/12,576). The pooled value masks weaker MuSiQue source identification. Retrieval checkpoints and training configurations are specified in Section~3.2 and the Supplementary Configuration Notes. Generation uses Qwen3.5-9B locally and Qwen3.7-plus in the cloud, with temperature 0 and seed 42 unless stated otherwise. CALIPER core uses the Scheme-A current source-router rerun with cached cloud instructions; CALIPER+clf combines its answers with frozen local-baseline answers. Controls retain the definitions in Tables~\ref{main} and~\ref{ablation}. At the selected dataset-specific thresholds 0.974/0.978/0.977, collaborative-call rates are 80.74\%, 90.39\%, and 91.32\%, respectively. Classifier fitting and threshold selection use the retained overlap-free split records; frozen outputs and evaluation scripts are retained with SHA-256 manifests.

\item \textbf{Adaptive-RAG implementation}: We use a fixed revision of the official Adaptive-RAG repository as the routing reference. We execute its three-way policy on the unified CALIPER retrieval/generation backend rather than claiming a parameter-for-parameter reproduction of the original Elasticsearch/FLAN-T5 QA pipeline. The T5-Large router is trained with the accompanying \texttt{flan\_t5\_xl} binary and silver labels (3,692 training and 1,350 validation examples, 15 epochs, seed 42; best validation accuracy 44.44\%). Branch A performs no retrieval, Branch B reuses strict Naive RAG, and Branch C keeps BASE top-5 passages, generates two focused subqueries, adds BM25 evidence from the same candidate corpus, and makes at most three retrieval rounds before short-answer extraction. Test answers are not exposed to routing or generation. Because the adapted router has limited validation accuracy, we treat this row as a bounded workflow comparison rather than evidence that CALIPER dominates a strong reproduction of Adaptive-RAG. Always-Multi is the more direct control for the capability of the adapted multi-step branch independently of router errors.

\item \textbf{Statistical procedure}: For each paired comparison, question indices are resampled with replacement and the mean per-question F1 difference is recomputed 10,000 times. We report the percentile 95\% interval and a two-sided bootstrap p-value computed from the two tails with the finite-sample $+1$ correction. For the primary Table~\ref{main} and Table~\ref{inf} contrasts, the paired-bootstrap intervals, seeds, and per-question sample counts are retained in the accompanying analysis files; Section 14 identifies the recomputed Table 3 statistics. Current Table 5 latency intervals are given in Section 16; detailed V0--V1 intervals are retained below rather than printed in the main text. No family-wise multiplicity correction is applied to the remaining exploratory pairwise comparisons. For transparency, the three V0--V1 dataset-level tests are additionally summarized with a Holm-adjusted sensitivity calculation below; secondary metrics and exploratory threshold curves remain descriptive.

For the optional Reflection component, we additionally perform the same paired bootstrap on the exact V0--V1 question matches used in Table~\ref{ablation}. In the current rerun, the V0-minus-V1 F1 differences are +0.68 points on HotpotQA ($n=7{,}405$, 95\% CI $[0.32,1.05]$, $p=0.0008$), +2.05 by the displayed Table~\ref{ablation} values on MuSiQue ($n=2{,}403$, 95\% CI $[1.13,3.00]$, $p=0.0002$), and $-0.16$ on 2WikiMultiHopQA ($n=12{,}576$, 95\% CI $[-0.39,0.06]$, $p=0.1536$). Bootstrap intervals and p-values use unrounded per-question scores (the MuSiQue mean difference is 2.0555 points). Treating these three dataset-level comparisons as one family gives approximate Holm-adjusted p-values of 0.0016, 0.0006, and 0.1536, respectively. These are output-level paired comparisons; because the full V0 artifacts retain final predictions but not complete evaluator/corrector traces, they do not by themselves establish a causal effect of the internal Reflection procedure.

\item \begin{minipage}[t]{\linewidth}
\textbf{Latency boundary}: Table~\ref{time} reports end-to-end latency estimates on fixed samples of 200 questions per dataset. Retrieval and generation are timed separately and combined per question according to the corresponding execution branch, rather than measured as one integrated execution. CALIPER uses the source-routed fine-tuned retrieval time, including MuSiQue's additional BASE stream and context merging. CALIPER+clf uses the fixed dataset-specific retrieval reference for its local-only branch and the routed retrieval time for its collaborative branch; classifier time already present in the generation record is not added twice. The local and cloud baselines use the fixed-retriever measurement. Retrieval includes embedding, FAISS construction, and reranking; model loading is excluded. Client-observed cloud request/response time is included only for branches that call the cloud. CALIPER sends only the question for cloud guidance, whereas the two-stage cloud baseline also sends five retrieved passages.
\end{minipage}
\end{itemize}
```

## 10. Figure and table caption details

Updated 2026-09-30. Caption details are retained below after condensation and reflect the adopted results plus the local Table 3 and presentation revision. This date is not a new archive-freeze receipt.

Current numbering note: the manuscript contains seven numbered tables. Section 11 contains supplementary classifier confusion counts, not an additional numbered manuscript table. The retrieval-only comparison is Table 7.

### Figure 1

```tex
\caption{Data flow in cloud RAG (left) and CALIPER (right). CALIPER keeps the knowledge base and retrieved evidence local and transmits the query to the cloud for instruction generation. This narrows the cloud-visible evidence surface but does not make the query private or constitute a formal privacy guarantee.}
```

### Figure 2

```tex
\caption{CALIPER architecture. Cloud guidance receives only the question; retrieval and answering remain local. Source routing and optional execution classification serve separate decisions. Dashed arrows denote optional Reflection.}
```

The September 30 vector redraw replaces illustrative query/answer paragraphs with concise process labels. Source labels are benchmark training targets, not inference inputs. The local-only branch uses its retained baseline context rather than rerunning the source router; MuSiQue's additional BASE support is shown only in the collaborative context. No model, branch output, threshold, or experimental result changes.

### Table 1

```tex
\caption{Training-set and evaluation-pool sizes. Table-specific evaluation samples are described in the text.}
```

### Table 2

```tex
\caption{Answer quality with retrieved evidence; Reflection is disabled. Bold values denote column maxima across all methods, including the cloud reference. MuSiQue uses the common 2,403-question intersection; HotpotQA and 2Wiki use the evaluation pools in Table~\ref{size}. Baseline definitions and generation settings are given in Section~4.1.}
```

### Table 3

```tex
\caption{Generation results with fully relevant evidence (7,405/2,410/12,576 unique questions). Two MuSiQue records overlapping classifier training are excluded from every method. The cloud row is the strongest reference in this controlled setting; because the generation pipeline and sample construction differ from Table~\ref{main}, this table is not a strict upper-bound comparison.}
```

### Table 4

```tex
\caption{V0--V4 ablation F1 on the same questions as Table~\ref{main}: HotpotQA (7,405), MuSiQue (2,403), and 2Wiki (12,576). Reflection appears only in V0. V0/V1/V3 use the current rerun; V2 uses fixed BASE top-5 plus cached cloud guidance, without source routing or the extra MuSiQue support stream. V4 uses BASE top-5 with V3's local prompts, temperature 0, seed 42, thinking disabled, and token limits 256/64.}
```

### Figure 3

```tex
\caption{Descriptive threshold sensitivity of Scheme-A CALIPER+clf: F1 (left) and cloud-call rate (right). Each marker is one measured threshold setting; lines connect adjacent measured settings for visual guidance only and do not imply measurements at intermediate thresholds. Thresholds are equally spaced categorical settings, not a linear numerical axis. Source-router assignments and cached branch predictions remain fixed. Colored dashed lines mark the frozen thresholds 0.974/0.978/0.977 for HotpotQA/MuSiQue/2Wiki, each selected from its dataset-specific validation pool disjoint from its own evaluation questions and classifier training, with hard-to-local false-negative rate at most 10\%. This is not a global three-benchmark exclusion. The points are descriptive and are not used to retune the operating points.}
```

### Table 5

```tex
\caption{End-to-end latency estimates in seconds per question: mean (p50/p95), with one fixed sample of 200 valid observations per dataset shared across methods. Retrieval and generation times are combined per question according to each execution branch. These are composed estimates, not integrated live measurements. The HotpotQA replacement record and all per-question rows are retained in the archive. Model loading is excluded.}
```

### Table 6

```tex
\caption{Operational-difficulty classifier diagnostics (\%), computed at dataset-specific frozen evaluation thresholds 0.974/0.978/0.977 for HotpotQA/MuSiQue/2Wiki. Hard$\rightarrow$local is the fraction of all audit questions that are true-hard but routed locally (HE/$n$).}
```

### Supplementary classifier confusion counts (Section 11)

```tex
\caption{Four-way confusion counts for the operational-difficulty classifier at the dataset-specific frozen evaluation thresholds 0.974/0.978/0.977. The two-letter codes denote true difficulty followed by predicted difficulty, which maps to local for easy and collaborative for hard: EE (easy/easy), EH (easy/hard), HE (hard/easy), and HH (hard/hard).}
```

### Table 7

```tex
\caption{Retrieval-only evidence recall. All \(R@k\) values are means of per-question recall, and \(k\) counts retrieved sentences. For HotpotQA and 2Wiki, recall is the fraction of gold supporting sentences found in the top \(k\); for MuSiQue, the selected sentences are mapped to source paragraphs and recall is the fraction of gold supporting paragraphs covered. MuSiQue uses 4,293 eligible MuSiQue-Full dev questions, versus the 2,403-question MuSiQue-Ans intersection in Tables~\ref{main} and~\ref{ablation}. This is a per-dataset system comparison, not a component ablation or cross-dataset aggregate. The routed row uses the current text-grouped source router under this separate retrieval-only protocol.}
```

## 11. Execution-classifier confusion counts

Updated 2026-09-29 with the adopted overlap-free classifier. Counts use the dataset-specific frozen thresholds 0.974/0.978/0.977. Codes give true/predicted operational difficulty: E = easy (local), H = hard (collaborative).

| Dataset | n | EE | EH | HE | HH |
| --- | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | 414 | 56 | 225 | 16 | 117 |
| MuSiQue | 332 | 24 | 111 | 12 | 185 |
| 2Wiki | 2,000 | 119 | 1,318 | 39 | 524 |

The manuscript retains the derived diagnostics in Table 6: accuracy = (EE + HH)/n, hard recall = HH/(HE + HH), and hard-to-local fraction = HE/n, each expressed as a percentage. The displayed audit subsets are fixed for comparison; threshold selection itself uses separate overlap-free validation pools documented below.


## 12. Method configuration and audit details

Added 2026-09-24 during condensation of manuscript Sections 3.2--3.3. These paragraphs preserve the original detailed records; this is not a new experiment, provenance verification, or archive freeze. LaTeX labels refer to the accompanying manuscript. Checkpoint mappings and retrieval-training settings remain in Sections 1--3.

### 12.1 Source-router fitting and evaluation pools

```tex
where \(s_i\) is the dataset-source label. The three source training corpora contain 277,839 rows before exclusion. We assign each training question deterministically using SHA-256 over the normalized model-input text; two rows whose normalized text matches a held-out evaluation question are excluded, leaving 277,837 rows for the reported 80/10/10 train--validation--holdout split (222,235/27,959/27,643 questions across the three sources). The held-out evaluation pools (7,405 HotpotQA distractor-validation-sized questions, 4,293 eligible MuSiQue-Full dev questions, and 12,576 labeled 2Wiki questions; 24,274 total) have zero normalized model-input text overlap with the source-router fitting pool and are used for the final routing audit. The local 2Wiki input is labeled, but we do not infer its official dev/test identity from its row count alone. The reported experiments evaluate the three known benchmark sources; BASE is retained as the non-specialized reference retriever.
```

### 12.2 Execution-threshold selection

```tex
The execution thresholds are frozen at \(t_{\mathrm{HP}}=0.974\), \(t_{\mathrm{Mu}}=0.978\), and \(t_{\mathrm{2Wiki}}=0.977\), each the smallest 0.001-grid value with a false-negative rate among truly hard questions at most 10\% on its independent overlap-free validation pool. This conditional rate is HE/(HE + HH), whereas Table~\ref{clf} reports HE/$n$, the fraction of all audited questions that are hard but sent to the local-only branch. The corresponding validation sizes are 2,320, 3,080, and 1,371 questions, with hard-to-local false-negative rates 9.75\%, 9.99\%, and 9.50\%, respectively. Table~\ref{clf} and the confusion counts in Section~11 use these dataset-specific operating points.
```

Each benchmark evaluation pool is processed separately. Its known benchmark identity selects the execution threshold before the optional execution gate; this identity is not obtained from the source router. The source router predicts which retrieval weights to use when that branch is taken. No threshold-selection policy is established for requests of unknown source; a separately calibrated deployment policy would be required.

Evidence: `work/repair/finalize_table3_case_20260930.py` iterates over benchmark-specific thresholds, loads each corresponding evaluation pool, and checks the threshold recorded for every recomposed prediction. This is an offline benchmark composition, not evidence of an integrated unknown-source deployment.

### 12.3 Overlap, classifier training records, and reproducibility

```tex
The frozen source and execution-classifier binaries are retained in the accompanying archive, with their full SHA-256 values recorded in the archive manifest. The current classifier run manifest retains the overlap-free training/validation split records, selected thresholds, prediction rows, and model hash. Classifier diagnostics use fixed audit subsets of 414 HotpotQA, 332 MuSiQue, and 2,000 2Wiki questions, while threshold selection uses the larger independent pools described above. The current text-grouped source-router training script is retained, including its split construction and hyperparameters (37 epochs, learning rate 0.05, embedding dimension 160, and word bigrams). The historical classifier recipe remains unavailable, but the adopted frozen classifier supports inference reproduction from its retained artifacts and manifest.
```

The execution-classifier validation separation is distinct from the source-router fitting/evaluation separation. The adopted classifier excludes normalized-question overlap with the frozen Table~\ref{main} pools; retained binaries, split records, and manifests support frozen-model inference and threshold-selection reproduction.

### 12.4 Retrieval units, generation context, and retained segmentation

Candidate contexts or paragraphs define the question-specific search space; the independently embedded and reranked items are sentence-level text units. HotpotQA and 2Wiki use the sentence entries in the retained `ctxs[].sentences[]` inputs. Generation consumes the frozen ranked `retrieved_text` strings directly, rather than expanding selected units to full parent paragraphs. It uses the routed top five units; for MuSiQue evaluation questions it additionally merges BASE top-three units, removing duplicates by lowercased, whitespace-normalized text. This extra stream is keyed by the known evaluation dataset, not by the source-router prediction.

For Table 7, k counts the retrieved sentence-level units. HotpotQA and 2Wiki score coverage of gold supporting sentences. MuSiQue maps the retrieved units back to their source paragraphs and scores coverage of gold supporting paragraphs; that mapping is a recall-scoring operation, not a paragraph expansion of the generator input. The retrieval-only comparison does not score the combined MuSiQue generation context.

Retained MuSiQue segmentation is path-specific. The Table 7 BASE, FT-SEP, and FT-MER evaluator uses `nltk.sent_tokenize` on paragraph text. In the routed row, MuSiQue questions predicted as MuSiQue retain NLTK sentence tokenization, whereas MuSiQue questions routed to HotpotQA or 2Wiki retain the helper's literal `. ` splitting. The parent-paragraph recall lookup uses NLTK sentence tokenization and normalized text matching. These historical differences are documented rather than harmonized: this clarification changes neither cached candidates nor reported results. Generation reads its existing retrieval caches without applying a new tokenizer.

Evidence: `build_retrieval_cache.py` (sentence flattening and retrieval), `work/router_e2e_rerun_20260918.py` (`context_for` and generation prompts), `work/run_queryonly_repair_20260924.py` (reuse of that context composition), `scripts/eval_table6_sentence_recall.py`, `scripts/eval_table6_musique_paragraph_recall.py`, `work/rebuild_routed_large_20260921.py`, and the original `work/repair/true_routed_recall_heldout.py` helper (retained under `legacy/work/repair/` in the public repository).

## 13. Table 2 Hit scoring correction (2026-09-25)

Hit equals one only when the normalized gold answer is nonempty and occurs as a substring of the normalized prediction; otherwise it equals zero. Questions whose gold answers normalize to an empty string remain in the evaluation denominator and receive Hit zero for every method.

Two HotpotQA gold answers, `The The` and `!!!`, become empty after answer normalization. The historical fixed-baseline evaluator counted an empty string as a substring match, while the retained routed evaluator already required a nonempty normalized gold answer. Applying that nonempty requirement consistently to the retained predictions corrects five HotpotQA Hit entries in Table 2:

| Method | Previously displayed Hit (%) | Corrected Hit (%) |
|---|---:|---:|
| Local two-stage | 66.47 | 66.44 |
| Naive RAG (strict) | 58.41 | 58.38 |
| Adaptive-RAG (adapted) | 60.26 | 60.23 |
| Always-Multi (adapted C) | 62.85 | 62.82 |
| Cloud | 71.71 | 71.68 |

This correction uses existing predictions, retains all evaluation questions, and changes neither generation nor the gold answers. The retained audit reproduces the other Table 2 metrics; precision, recall, F1, SBERT similarity, and the F1-based comparisons are unaffected. The CALIPER Hit entries already follow the nonempty rule.

Evidence: `router_e2e_runs/queryonly_repair_20260924_v1/analysis/statistics_summary.json`, specifically `conflicts`, `conflict_explanation`, and `empty_normalized_gold`, records the historical-baseline recomputation. Only those baseline scoring corrections are adopted here; the separate new-generation results in that analysis are not adopted. Historical evaluator outputs remain available as historical records. At the time of that recomputation, the manuscript correction did not itself update the companion archive; the current v13 archive contains the synchronized manuscript and evidence record.

## 14. Table 3 and current-classifier case repair (2026-09-30)

The September 30 Table 3 repair retained fully relevant-evidence branch predictions and replaced only the execution classifier with the overlap-free model whose SHA-256 is `44d7ea0de0d214bce1e44f92b9452419838daec1e71ba38a9bef42e32806fd6d`. Its operating thresholds remain 0.974/0.978/0.977. No answers were regenerated at that stage. The October 4 four-question answer repair is recorded separately in Section 17; no model training or threshold tuning is performed in either repair.

### 14.1 Shared evaluation samples

The original MuSiQue Table 3 pool contains 2,412 exact-question-deduplicated records. The following two records both match one normalized question in the current classifier training set:

- `Who did the original voice of the character Mila Kunis plays on Family Guy?`
- `Who did the original voice of the character Mila Kunis plays on family Guy?`

Both are excluded from all four methods before scoring, leaving 2,410 MuSiQue questions. HotpotQA and 2Wiki remain at 7,405 and 12,576. Exclusion is based solely on training-set membership, not answer quality. Normalization lowercases text, replaces non-alphanumeric characters with spaces, and collapses whitespace. The original prediction files and historical statistics are preserved.

The retained Table 3 pools have zero overlap with classifier training, classifier fitting-validation data, or their own dataset-specific threshold-selection pools. The latter are reconstructed from `query_clf/train_new/l1o_<dataset>_valid.txt` after removing classifier-training overlap; their sizes and label counts reproduce the retained 2,320/3,080/1,371-question threshold report. This is dataset-specific separation, not a globally sealed three-benchmark test: evaluation questions can occur in the threshold-selection pools of other benchmarks. In particular, HotpotQA has 717/731 matches in the MuSiQue/2Wiki pools, MuSiQue has 213/223 matches in the HotpotQA/2Wiki pools, and 2Wiki has 1,118/1,097 matches in the HotpotQA/MuSiQue pools. Each reported threshold is applied only to its corresponding benchmark.

### 14.2 Updated F1 and paired statistics

| Method | HotpotQA (7,405) | MuSiQue (2,410) | 2Wiki (12,576) |
|---|---:|---:|---:|
| Local (Qwen3.5-9B) | 69.39 | 49.94 | 63.48 |
| CALIPER+clf | 74.36 | 55.12 | 73.14 |
| CALIPER | 75.41 | 56.28 | 74.30 |
| Cloud (Qwen3.7-plus) | 83.38 | 69.70 | 82.27 |

All Table 3 paired contrasts are recomputed on the shared retained questions with 10,000 bootstrap resamples and the original seeds: 42 for CALIPER versus local, 43 for CALIPER versus CALIPER+clf, and 44 for CALIPER versus cloud. For CALIPER minus CALIPER+clf, the unrounded-score mean differences, rounded to two decimals, are 1.06/1.16/1.16 F1 points, with 95% intervals [0.71, 1.39]/[0.61, 1.72]/[0.93, 1.41]; all three two-sided p-values are 0.0002. Narrative differences based on displayed cells follow Section 6 instead. This repair computes F1 only; it introduces no newly evaluated SBERT, Hit, precision, or recall claims for Table 3.

### 14.3 Replacement classifier case

The third case in Section 8.3 uses the current classifier and the unchanged selection rule. Five MuSiQue cases satisfy true-hard labeling, a local route, collaborative F1=1, and local F1=0; the smallest SHA-256 of `42|question` selects `Who did the performer of Night Rocker play on Young and Restless?`. Its selection hash is `1ffe275615a63e36b395b619f1d243f7ca3827c28697d32421129fcd0c841ee8`. The easy probability is 0.9971545338630676, above 0.978. The local branch declines to answer, while the collaborative branch returns the gold answer, `Snapper Foster`. The three labeling outputs recover the gold in two runs, but not the third; therefore the frozen all-three-correct rule labels the question hard.

September 30 evidence: `scripts/audit_table3_case_20260930.py`, `scripts/finalize_table3_case_20260930.py`, and `table-result/table3_current_classifier_20260930/`, containing source predictions, paired statistics, and exact exclusion records. The October 4 overlay and current 2Wiki statistics are in Section 17. Tables 2, 4, 5, and 6, Figure 3, and the classifier case remain unchanged; Table 7 is rescored only for the four corrected annotations.

## 15. Author-confirmed query-only provenance (2026-09-30)

The author confirms that the adopted main results use the query-only guidance cache generated on August 3, 2026, replacing the older cache contents before the September 18/20 reruns. The adopted results comprise the HotpotQA/2Wiki September 18 reruns and the MuSiQue September 20 rerun; the September 24 exploratory rerun is not adopted. A retained old cache filename is not a claim about its historical runtime contents. The author renewed this confirmation on September 30, 2026.

Evidence category: **AUTHOR_ATTESTED / AUTHOR_CONFIRMED_CORRECT**. This records the provenance of the adopted cache and the author's confirmation of its query-only input boundary. Previous audit observations remain retained, and no experimental values, cache files, prediction files, or evaluation protocol are altered by the confirmation. The method's boundary concerns cloud input: guidance requests contain the question, not retrieved passages; cloud-generated reasoning or candidate answers do not themselves constitute evidence transmission.

The archive retains `documentation/CALIPER_cache_provenance_author_confirmation_20260924.md`, including the September 30 confirmation, and identifies its evidence category in `CURRENT_EVIDENCE_INDEX.json` and `FREEZE_RECEIPT.json`.

## 16. Current Table 5 latency confidence intervals (2026-09-30)

This supplemental analysis uses the adopted Table 5 observations, not the superseded latency statistics. Each dataset retains 200 unique questions shared across all four methods. The already-adopted HotpotQA replacement at index 177 is applied to every method; no new replacement, exclusion, inference, or timing measurement is introduced. All recomputed mean, p50, and p95 values reproduce the current displayed Table 5 cells. The mean and quantiles in Table 5 are unchanged.

MuSiQue uses the replacement seed-20260922 sample documented in Section 5. These intervals do not include uncertainty introduced by the outcome-informed decision to replace the preceding sample. The separate generation-quality audit in Section 8.4.1 is not the source of these latency intervals.

Question indices are resampled with replacement 10,000 times using NumPy `default_rng(42)`. All four methods use the same sampled indices within each dataset. The 2.5th and 97.5th percentiles give pointwise 95% confidence intervals. These intervals describe question-sampling uncertainty conditional on the retained stage timings and adopted samples; they do not quantify cross-run timing variability, across-day network fluctuations, uncertainty from stage alignment, or an integrated online execution. No simultaneous-coverage or multiplicity-adjusted inference is claimed. Resamples and deterministic analysis replay are not additional independent observations.

### 16.1 Mean estimated latency

Cells show the mean and its 95% confidence interval in seconds, not p50/p95.

| Method | HotpotQA | MuSiQue | 2Wiki |
|---|---:|---:|---:|
| Local (Qwen3.5-9B) | 3.90 [3.78, 4.01] | 4.95 [4.84, 5.05] | 4.11 [4.02, 4.19] |
| CALIPER+clf | 5.53 [5.25, 5.85] | 6.21 [6.11, 6.29] | 5.84 [5.58, 6.15] |
| CALIPER | 6.03 [5.75, 6.37] | 6.36 [6.31, 6.41] | 5.93 [5.67, 6.23] |
| Two-stage cloud reasoning | 8.90 [8.74, 9.06] | 12.32 [12.09, 12.56] | 9.49 [9.33, 9.63] |

### 16.2 Paired estimated-latency reductions

Cells show the reduction in seconds and its paired 95% interval. A positive value means the candidate has lower estimated latency than the stated reference.

| Candidate versus reference | HotpotQA | MuSiQue | 2Wiki |
|---|---:|---:|---:|
| Local versus cloud | 5.00 [4.85, 5.16] | 7.37 [7.16, 7.58] | 5.38 [5.24, 5.51] |
| CALIPER versus cloud | 2.87 [2.49, 3.20] | 5.96 [5.73, 6.20] | 3.56 [3.21, 3.86] |
| CALIPER+clf versus cloud | 3.37 [3.03, 3.67] | 6.12 [5.89, 6.35] | 3.64 [3.30, 3.95] |
| CALIPER+clf versus CALIPER | 0.50 [0.31, 0.74] | 0.15 [0.08, 0.24] | 0.09 [0.04, 0.14] |

Relative reductions use the ratio of unrounded means, `(reference mean - candidate mean) / reference mean`, not the mean of per-question percentages. CALIPER versus cloud has reductions of 32.21% [28.25%, 35.62%], 48.39% [47.36%, 49.40%], and 37.50% [34.12%, 40.36%]. CALIPER+clf versus cloud has reductions of 37.89% [34.20%, 41.08%], 49.63% [48.57%, 50.69%], and 38.41% [35.00%, 41.35%]. These comparisons do not establish a jointly measured quality--latency frontier with the main quality evaluation.

Reproducibility files: `latency_runs/table5_ci_current_20260930/table5_current_bootstrap_ci.json`, `table5_current_paired_rows.jsonl`, and `VALIDATION_RECEIPT.json`; script: `scripts/bootstrap_table5_current_20260930.py`. The JSON records source paths, SHA-256 values, sample checks, exact results, environment versions, bootstrap count, and seed. The HotpotQA replacement's retained retrieval total already includes classifier time; it is not added twice. The v9 package retains this analysis under `results/table5_ci_current_20260930/` and its script under `scripts/`; the external current-pointer verification receipt determines whether packaging is complete. Earlier latency intervals are not evidence for the current Table 5.

## 17. Four-question 2Wiki annotation and answer repair (2026-10-04)

An item-level comparison against the retained third-party `framolfese/2WikiMultihopQA` validation parquet (revision `fe713bfbd1afbca1a65246741a75890405d56a3a`, SHA-256 `408e2dbb28edc6c8b9ca3ba0c94d4fc7bf17ffb923766593a3a7f546ab4cba59`) matches all 12,576 IDs, questions, answers, types, and contexts. Four supporting-fact titles in the experiment copy instead select disambiguation or non-film articles. This mirror match does not independently verify the original official release identity. The original dataset is preserved; a separate corrected copy changes only these four titles, retaining sentence indices, question order, answers, and sample size.

| Question ID | Old supporting-fact title | Corrected title |
|---|---|---|
| b728e70c094411ebbdaeac1f6bf848b6 | Dead man | Dead Man |
| d1aa7f57087e11ebbd6bac1f6bf848b6 | Two wrongs make a right | Two Wrongs Make a Right |
| 43b55fee087311ebbd66ac1f6bf848b6 | Honeymoon suite | Honeymoon Suite |
| 59acfe98096c11ebbdafac1f6bf848b6 | Little black book | Little Black Book |

### 17.1 Retrieval-only rescoring

No retriever, model weights, or ranked sentences are changed. Exact sentence-match recall is recomputed over the same 12,576 questions. The 2Wiki R@3/5/10 cells become BASE 62.62/72.01/85.42, FT-SEP 89.37/98.05/99.47, FT-MER 84.50/95.10/98.84, and routed CALIPER 89.32/98.02/99.43. The maximum unrounded change is 0.0159034 percentage points. Other datasets are unchanged.

### 17.2 Targeted answer repair and configuration boundary

Exactly these four questions are regenerated for Local, CALIPER, and the cloud reference, using their corrected supporting sentences. The new repair uses Qwen3.5-9B locally and Qwen3.7-plus remotely, temperature 0, seed 42, thinking disabled, and 256/64-token reasoning/extraction limits. It uses the retained two-stage fully relevant-evidence prompt forms, with a 256-token question-only guidance call for CALIPER. Reflection is disabled. The cloud reference receives corrected evidence; CALIPER's cloud guidance receives only the question. Gold answers are not included in either generation prompt.

This is an authorized new repair run, not a verified reproduction of the historical generation recipe: the original source files do not establish a complete runtime model/prompt/decoding binding. Only four answer records are overlaid on the retained predictions. Classifier probabilities, routes, and threshold 0.977 are unchanged; all four questions route to CALIPER, so CALIPER+clf reuses the repaired CALIPER predictions. No questions are excluded or replaced.

Each stage has one retained output. Eight valid stages produced before a result-filename collision are reused only after exact request-body matching; the twenty remaining stages are executed once, without automatic API retries or best-output selection. A preceding localhost-proxy readiness failure produced no generation requests. Both interruptions and the continuation are retained. The owned local inference server is stopped after completion.

The four corrected questions have F1=1 for each compared method. The resulting full-pool 2Wiki F1 values are Local 63.48, CALIPER+clf 73.14, CALIPER 74.30, and cloud 82.27. The local and collaborative results on all other questions, both other datasets, classifier settings, main retrieved-evidence results, and latency evaluation are unchanged.

### 17.3 Updated paired statistics and evidence locators

The same 10,000 question-paired bootstrap resamples and seeds 42/43/44 are used for CALIPER versus Local / CALIPER+clf / cloud. On 2Wiki, the unrounded-score mean F1 differences rounded to two decimals are +10.82 (95% CI [10.12, 11.51]), +1.16 ([0.93, 1.41]), and -7.97 ([-8.49, -7.47]); each two-sided p-value is 0.0002. These intervals are conditional on the retained outputs and four new repairs, not API rerun variability or proof of historical configuration equivalence. The other datasets' paired statistics remain unchanged.

Remote experiment-relative locators for these repairs are retained in the experiment repository; the current archive pointer and manifest determine their packaging status:

- `annotation_repairs/2wiki_sf_20261002/`: corrected data, exact title/sentence patch, ranking-only recall results, and input/output hash manifest.
- `annotation_repairs/2wiki_sf_generation_20261004_v3/`: explicit run recipe, four-question replacement predictions, full request/response records, exact reused-stage bindings, generation receipt, server log, and cleanup receipt. Preflight and interrupted-stage evidence remain under `2wiki_sf_generation_20261004/` and `2wiki_sf_generation_20261004_v2/`.
- `annotation_repairs/table3_2wiki_sf_final_20261004/`: current F1, updated per-question scores, paired bootstrap statistics, old/new four-question predictions, and validation receipt.
- Scripts: `scripts/repair_2wiki_supporting_facts_20261002.py`, `scripts/repair_table3_2wiki_generate_20261004_v3.py`, and `scripts/finalize_table3_2wiki_sf_20261004.py`.

Original predictions, original annotations, and historical statistics are not overwritten. This revision does not modify the previously retained author confirmation of the main query-only cache binding.
