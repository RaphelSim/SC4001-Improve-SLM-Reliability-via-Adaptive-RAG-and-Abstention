# Prompt templates

The Member 2 baseline prompts are versioned text templates. Their current wording is a development baseline and must be tuned only with validation data, then frozen before test evaluation.

Reserved categories and example filenames:

- `sufficiency_assessment` — `sufficiency_v1.txt`
- `followup_query_generation` — `followup_query_v1.txt`
- `final_answer` — `final_answer_v1.txt`
- `structured_output_retry` — `structured_retry_v1.txt` (format repair only; one retry maximum)

See [prompt conventions](../documentation/09_prompt_conventions.md) before adding or changing a prompt. Do not request hidden chain-of-thought; parse and validate every model response against the shared Pydantic contract.
