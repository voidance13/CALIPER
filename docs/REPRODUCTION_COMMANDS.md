# Reproduction command map

This is a command template, not a claim that the public checkout contains private data, weights, caches, or services. Set the paths and endpoints in `configs/current_protocol.json` and `.env` before running.

## 1. Static checkout check

```bash
python tests/smoke_test.py
```

This checks the public configuration and parses the current entry-point files. It does not call a model or an API.

## 1.1 Compile the included manuscript source

```bash
cd paper
pdflatex cas-dc-template.tex
bibtex cas-dc-template
pdflatex cas-dc-template.tex
pdflatex cas-dc-template.tex
```

The CAS class/style files, bibliography, Figure 1 asset, and Figure 2 source
are included in `paper/`. This is a source-packaging check; it does not
reproduce the reported experiments.

## 2. Prepare private resources

Provide the benchmark files as `data/<dataset>/test.json`, the fine-tuned retrieval checkpoints under `retrieve_fine_tuning/models/`, and the private router/classifier checkpoints expected by the selected script. The public repository intentionally does not provide these files.

## 3. Build a BASE retrieval cache

```bash
python build_retrieval_cache.py \
  --dataset hotpotqa \
  --embed_model /path/to/bge-base-en-v1.5 \
  --rerank_model /path/to/bge-reranker-large \
  --device cuda:0 --workers 4
```

The command writes `data/<dataset>/test_retrieved.json` in the private working tree.

## 4. Run the collaborative path on a smoke subset

```bash
python hybrid_predict_separate.py \
  --dataset hotpotqa \
  --small_model Qwen3.5-9B \
  --large_model Qwen3.7-plus \
  --embed_model BAAI/bge-base-en-v1.5 \
  --rerank_model BAAI/bge-reranker-large \
  --device cuda:0 --limit 1
```

The local OpenAI-compatible endpoint is selected by `CALIPER_LOCAL_BASE_URL`; the cloud endpoint and key are selected by the corresponding environment variables. The output is written under the private `data/<dataset>/` directory.

For the optional execution classifier, provide `CALIPER_CLASSIFIER_PATH` and run the parallel entry point:

```bash
python hybrid_predict_separate_clf.py \
  --dataset hotpotqa \
  --small_model Qwen3.5-9B \
  --large_model Qwen3.7-plus \
  --embed_model BAAI/bge-base-en-v1.5 \
  --rerank_model BAAI/bge-reranker-large \
  --device cuda:0 --limit 1 --random_seed 42
```

The classifier checkpoint is deliberately private; its expected location is the value of `CALIPER_CLASSIFIER_PATH`, not a hidden repository default.

## 5. Paper-level aggregation

The following commands aggregate frozen private outputs; they are not fresh experiments by themselves:

```bash
python scripts/eval_table2_final.py
python scripts/eval_table3_full.py
python scripts/eval_ablation_final.py
python scripts/bootstrap_table5_current_20260930.py
```

Table 6 and Figure 3 use the classifier analysis scripts listed in the root README and the private classifier audit splits. Table 7 uses the retrieval-only scripts and private benchmark pools. Exact frozen numbers remain in the manuscript, supplementary notes, and companion archive.

## 6. Reproduction boundary

The public checkout supports code inspection and protocol alignment. Exact table reproduction additionally requires the private inputs listed in `configs/current_protocol.json`; a clean checkout is not intended to download benchmark data, checkpoints, caches, or API credentials automatically.
