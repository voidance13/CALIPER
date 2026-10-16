# Public release status: 2026-10-16-submission-v1

Manuscript sources, reference PDF, Supplementary Configuration Notes, figures,
and standalone tables match Overleaf `f72ec11e7e690fb9c55af2e6d08172ca3aace237`. The PDF has 10 pages and
31 references. Figure 2 uses `paper/workflow.pdf`; the PDF exports match it
byte-for-byte. All table values and experimental outputs are unchanged.

This synchronization includes metric denominators, known-benchmark threshold
selection, sentence-level generation context, and historical segmentation
details. The release commit is recorded in Git history; source hashes are in
`paper/MANUSCRIPT_RELEASE.json`.

The public tree excludes private datasets, weights, caches, per-question
predictions, logs, and credentials. Research scripts may still require private
paths and services; this is not one-command experimental reproduction.
Project license selection remains an author decision. No experiments or
new numerical analyses were run for this manuscript synchronization.
