# Configuration files

`base.yaml` is the shared provisional baseline. `development.yaml` selects only the synthetic contract fixture and a mock runtime for local wiring. Both use the typed schema in `reliable_rag.config.settings`.

Values such as Top-k, embedding model, model ID, quantization, token limits, and evidence formatting are development placeholders. Select experimental values using validation data, then freeze a config snapshot before evaluating test data. See [configuration policy](../documentation/05_configuration.md).
