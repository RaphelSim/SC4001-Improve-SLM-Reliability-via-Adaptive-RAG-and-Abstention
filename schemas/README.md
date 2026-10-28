# JSON Schemas

These files are generated from the Pydantic v2 models in `src/reliable_rag/contracts/records.py`. They are checked-in snapshots of the shared serialized contract.

Regenerate after an agreed schema change:

```bash
python scripts/generate_json_schemas.py
```

Commit generated schema changes with the corresponding model and documentation updates. Runtime-only cross-field validation is enforced by Pydantic and may not be fully expressible in JSON Schema.
