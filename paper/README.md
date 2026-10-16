# Manuscript source

Release: `2026-10-16-submission-v1`. Synchronized with Overleaf `f72ec11e7e690fb9c55af2e6d08172ca3aace237`.

`cas-dc-template.tex`, the bibliography/style, Supplementary Configuration Notes,
Highlights, template assets, and figure exports match the adopted Overleaf files.
The reference PDF is `CALIPER_current_manuscript.pdf` (10 pages, 31 references).
Figure 2 is included directly from `workflow.pdf`; `figure_exports/Figure2.pdf`
is a byte-identical export. Its Reflections connector lanes are offset.
`figure2_architecture.tex` is retained historical material, not the current Figure 2.

Compile from this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error cas-dc-template.tex
bibtex cas-dc-template
pdflatex -interaction=nonstopmode -halt-on-error cas-dc-template.tex
pdflatex -interaction=nonstopmode -halt-on-error cas-dc-template.tex
```

The supplementary notes distinguish HE/n from HE/(HE+HH), explain known-pool
threshold selection, and document sentence-level retrieval/generation and
historical segmentation differences. Table values and experimental outputs
are unchanged. Experimental reproduction still requires the private inputs
described in the repository README and `docs/REPRODUCTION_COMMANDS.md`.
