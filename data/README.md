# Data directory policy

- `raw/` contains downloaded/original data. It is Git-ignored.
- `processed/` contains derived datasets. It is Git-ignored except deliberately tiny test fixtures kept under `tests/fixtures/`.
- `splits/` contains split manifests. Lightweight deterministic question-ID manifests may be version controlled when dataset licensing permits; include dataset revision and generation details. Large or licensed data must not be committed.

Do not place model weights, credentials, or experimental output here. Keep paths relative and machine independent in shared configuration.
