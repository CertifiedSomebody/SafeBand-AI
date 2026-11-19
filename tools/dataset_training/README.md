# Training dataset lineage gate (2.12)

This tool audits provenance for the accepted dataset inventory and assigns
each generated dataset to the SHA-256 identity of its real source session.
It does **not** create train/validation/test partitions.

Run from the repository root:

```powershell
.\venv\Scripts\python.exe tools\dataset_training\build_training_dataset.py
```

For the complete workstream (fresh 2.11.1-2.11.9 acceptance, lineage gate,
feature tests, matrix generation, offline replay, .NET state-machine tests,
and consolidated report), run:

```powershell
.\venv\Scripts\python.exe tools\run_development_readiness.py
```

The runner treats the expected 2.12 `BLOCKED` exit as a successful integrity
gate only when the complete lineage audit confirms the reason and no split was
created. The final status report is
`reports/development_readiness_2.28.json`.

The source acceptance report and inventory must pass before lineage resolution.
Dataset lineage is resolved from the provenance manifest, native provenance,
validation reports, or preserved sample-index/time/activity signatures against
the recorded real source files. Ambiguous or unresolvable lineage is reported,
not guessed. IDs are derived from source-file SHA-256 so a synthetic derivative
cannot become an independent session by changing its samples.

Outputs:

- `datasets/training/lineage_manifest.json`
- `reports/dataset_training/training_dataset_acceptance_2.12.json`

Expected current result: `BLOCKED — INSUFFICIENT INDEPENDENT SOURCE SESSIONS`.
No split CSVs are produced. The fixed seed and 70/15/15 ratios in the config
are reserved for a future group-level split after enough independent
recordings are available.

The current accepted corpus remains appropriate for development and
augmentation, but not for independent model validation or final test metrics.
