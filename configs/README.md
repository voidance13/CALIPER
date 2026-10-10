# Configuration templates

- `current_protocol.json` is the machine-readable record of the frozen paper configuration.
- `.env.example` contains secret and endpoint placeholders only.

The public code does not download private datasets, checkpoints, caches, or API credentials. Set private paths outside the repository and pass them explicitly to the cache builders. The smoke test is dependency-free and does not contact a model or provider.
