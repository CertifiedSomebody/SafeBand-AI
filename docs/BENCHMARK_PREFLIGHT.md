# Benchmark preflight

Before accepting a result:

1. The exact script invocation from repository root must work.
2. All required imports must resolve.
3. The input schema must be checked before feature generation.
4. Smoke sampling must not create subject imbalance that invalidates the split.
5. Train/validation/test must have disjoint subjects.
6. Validation must contain enough windows to compare models.
7. Test data must not influence model selection.
8. Smoke metrics are runtime checks only, not research results.
9. PPG-DaLiA E4 BVP must not be represented as MAX30102 data.
10. The full benchmark is run only after the smoke test passes.
