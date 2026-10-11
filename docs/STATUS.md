# Public release status: 2026-10-11-submission-v1

This release packages the current frozen manuscript protocol and the public
code/documentation tree. The release commit is recorded in Git history.

## Completed

- Created local public-repository staging tree from the A40 CALIPER code directory.
- Added current paper figures, table source snippets, manuscript source, template assets, and supplementary notes.
- Removed copied datasets, classifier training pools, nested Git metadata, third-party sample prompts, sample-ID configs, generated screenshots, and Python cache directories.
- Added public-scope, protocol, license, data, model, result, and log documentation.
- Added `.env.example` with placeholders only; no real API key was copied.

## Not yet complete

- Remaining research scripts still contain A40-specific absolute paths and service ports.
- The current public tree is not yet a clean-clone one-command reproduction.
- Project license selection is pending author decision.
- GitHub `main` is the public release branch.

## Evidence boundary

This release operation copied code and documentation only. It did not rerun experiments, validate new metrics, or change the frozen paper protocol.
