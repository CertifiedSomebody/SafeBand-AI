# Development replay (2.22)

Run all accepted CSV datasets through the deterministic feature extractor:

```powershell
.\venv\Scripts\python.exe tools\runtime_replay\replay_development_datasets.py
```

The consolidated workflow also runs this replay:
`.\venv\Scripts\python.exe tools\run_development_readiness.py`.

The replay validates feature schema, finite output, source-session tracking,
sensor masks, and timing-stall indicators. It reports feature-extraction
throughput and Python traced memory on the current host only.

No model is loaded and no fall probabilities or event metrics are produced.
Synthetic labels are not used as predictions. This is not an end-to-end model
evaluation, a wearable performance measurement, or evidence of real-world
accuracy.
