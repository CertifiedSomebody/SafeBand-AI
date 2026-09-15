# Stage 2 fixes

This revision fixes two issues found before any benchmark result is accepted:

1. `python tools\benchmark_ppgdalia_hr.py` now explicitly adds the repository root to
   `sys.path`, so the `ai` package resolves when the script is launched from the repo root.
2. `--max-windows` no longer takes the first N rows only. The smoke subset is distributed
   across subjects so subject-independent train/validation/test splitting remains valid.

The full benchmark is still subject-independent and uses validation-only model selection.
No smoke-test metric should be reported as a research result.
