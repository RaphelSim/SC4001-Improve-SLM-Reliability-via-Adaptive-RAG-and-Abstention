# Team workflow

Recommended branches:

```text
main
├── feature/data-retrieval
├── feature/sufficiency-abstention
└── feature/adaptive-evaluation
```

Working rules:

- Do not do experimental feature work directly on `main`.
- Pull/rebase the latest `main` before integration.
- Changes to shared contracts, configs, or docs that affect other members require team agreement.
- Treat Pydantic schema changes as API changes. Do not casually rename fields or enum values.
- Update docs, tests, and generated schemas whenever a shared contract changes.
- Keep commits small and focused.
- Pull requests state whether they change contracts, configs, data formats, prompts, or metrics.
- Keep datasets, credentials, weights, caches, and run outputs out of version control unless the team explicitly selects a small, licensed final artifact.

Use the repository PR template as a lightweight integration check. Members can develop against the fixture and Protocol mocks while other modules are in progress.
