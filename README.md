# SC4001: Reliable Small Language Models with Adaptive RAG and Abstention

**Research question:** Can adaptive retrieval and abstention improve the reliability of a small language model when retrieved evidence is incomplete?

This project studies how a small language model behaves when retrieval returns only part of the evidence needed for a multi-hop answer. The planned system checks evidence sufficiency, makes one targeted follow-up retrieval when needed, and abstains if the available evidence still cannot support an answer. Reliability and evidence support are the primary outcomes; inference cost is secondary.

## Architecture

```text
Question → initial retrieval → sufficiency assessment
                              ├─ sufficient → answer from evidence
                              └─ insufficient → one follow-up retrieval
                                                 ├─ sufficient → answer
                                                 └─ insufficient → abstain
```

Implementations plug into shared Pydantic records and Protocol interfaces under `src/reliable_rag/contracts/`. Dataset handling, retrieval, model inference, adaptive control, and evaluation are separate modules.

## Repository map

- `src/reliable_rag/` — shared records, configuration, and JSONL utilities
- `configs/` — development configuration defaults
- `documentation/` — definitions, contracts, protocol, and team workflow
- `schemas/` — generated JSON Schema contract files
- `scripts/` — environment, config, JSONL, and schema validation tools
- `tests/` — foundation tests and synthetic handoff fixtures
- `data/`, `cache/`, `outputs/` — local datasets, caches, and run artifacts

## Quick setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Common commands

```bash
pytest
ruff check .
ruff format --check .
python scripts/validate_config.py configs/base.yaml
python scripts/check_environment.py
python scripts/generate_json_schemas.py
python scripts/validate_jsonl.py --schema experiment-record tests/fixtures/example_experiment.jsonl
```

Read the [documentation index](documentation/README.md) for setup details and team contracts. Research modules are intentionally still stubs at this foundation stage; no real dataset, retrieval, inference, adaptive algorithm, or final evaluation has been implemented.
