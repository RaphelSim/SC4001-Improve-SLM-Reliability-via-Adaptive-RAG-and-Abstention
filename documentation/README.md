# Documentation index

Start with the scope, then use the contract and protocol pages as the shared reference. A change affecting a shared interface, field, metric denominator, or test-data rule should update the relevant page in the same pull request.

| File | Purpose | Intended reader | Update when |
|---|---|---|---|
| [00_project_scope.md](00_project_scope.md) | Research question, scope, and boundaries | Everyone | The research question or project scope is agreed to change |
| [01_architecture.md](01_architecture.md) | Package roles, data flow, dependency direction | All module owners | Modules or pipeline boundaries change |
| [02_definitions.md](02_definitions.md) | Operational evidence and reliability terminology | Everyone | A term or experimental label is clarified |
| [03_data_contracts.md](03_data_contracts.md) | Canonical records, enums, IDs, and validation rules | All implementers | A shared schema or serialized value changes |
| [04_module_interfaces.md](04_module_interfaces.md) | Protocol signatures and ownership handoffs | Module owners | A module boundary or method contract changes |
| [05_configuration.md](05_configuration.md) | YAML structure, defaults, and environment overrides | Everyone | A config key/default or override changes |
| [06_environment_setup.md](06_environment_setup.md) | Python, Mac, and Colab setup instructions | Everyone | Dependency or runtime setup changes |
| [07_experiment_protocol.md](07_experiment_protocol.md) | Data-use rules and final-run controls | Experiment operators | Split policy, freeze rules, or run metadata changes |
| [08_metrics.md](08_metrics.md) | Metric formulas and exact denominators | Evaluator and reviewers | A metric formula, label, or denominator changes |
| [09_prompt_conventions.md](09_prompt_conventions.md) | Prompt versioning and output conventions | Model/prompt owner | Prompt structure or frozen version changes |
| [10_reproducibility.md](10_reproducibility.md) | Seeds, snapshots, caches, and provenance | Everyone | Reproduction procedure or environment capture changes |
| [11_team_workflow.md](11_team_workflow.md) | Branch and review workflow | Everyone | Team integration practices change |
| [12_integration_and_handoffs.md](12_integration_and_handoffs.md) | Example records and member-to-member handoffs | All three members | Handoffs or examples change with the contracts |
| [13_decision_log.md](13_decision_log.md) | Initial project decisions and rationale | Everyone | A recorded decision changes or a new one is made |

Quick lookup: terminology → `02`; record field definitions → `03`; module contracts → `04`; dependencies and installation → `06`; experiment rules → `07`; metric denominators → `08`; team coordination → `11` and `12`.
