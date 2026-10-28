# Prompt conventions

Research wording is intentionally not finalized in the foundation. Prompts are version-controlled text/template files under `prompts/`, each with a stable category, ID, and version.

Future categories and reserved names:

| Category | Reserved file pattern |
|---|---|
| `sufficiency_assessment` | `prompts/sufficiency_v1.txt` |
| `followup_query_generation` | `prompts/followup_query_v1.txt` |
| `final_answer` | `prompts/final_answer_v1.txt` |

Conventions:

- Record a prompt version and change note whenever wording changes during validation.
- Freeze prompt versions before test evaluation; do not modify after inspecting test results.
- Ask only for concise task outputs and documented JSON-compatible fields.
- Do not ask for hidden chain-of-thought or expose it in the stored records.
- Parse and validate model outputs before they enter the pipeline. Invalid output should follow an explicit error/abstention policy.
- Keep model and prompt IDs in run metadata/config snapshots so cached outputs cannot be mistaken for another version.

This page defines handling rules, not final research prompt content.
