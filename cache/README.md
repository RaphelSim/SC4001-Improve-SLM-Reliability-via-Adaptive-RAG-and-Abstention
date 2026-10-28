# Cache directory

Use `cache/` for regenerable embeddings, retrieval outputs, and development model outputs. It is Git-ignored. Cache keys should include the model identifier/revision, preprocessing or prompt version, relevant configuration, and dataset revision so stale artifacts are not reused silently. Never store model weights or credentials here.
