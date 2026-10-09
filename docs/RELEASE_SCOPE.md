# Public release scope

## Included

- Python source and shell entry points needed to understand the method;
- retrieval, router, classifier, baseline, ablation, and statistical-analysis helpers;
- paper figures, table source snippets, manuscript source, and lightweight notes;
- environment and path templates without credentials;
- external comparison code only with its original license files.

## Excluded

- benchmark data, supporting evidence, and raw text corpora;
- model weights, fine-tuning checkpoints, classifier binaries, FAISS indexes, and caches;
- per-question predictions, cloud responses, logs, and multi-gigabyte archives;
- `.env`, API keys, authentication files, and machine-specific secrets.

## History policy

Scripts used only for repairs, migration, diagnostics, or superseded protocols live under `legacy/`. They preserve provenance but are not current paper entry points. Current numerical claims must be tied to the manuscript source, a frozen manifest, and the corresponding private artifact.
