# SafeBand AI — Nonspeech7k Audit & Preparation V1

First controlled dataset stage for the INMP441 audio intelligence branch.

## Contents

```text
SafeBand_Nonspeech7k_Audit_Prep_V1/
├── README.md
├── tools/
│   ├── audit_nonspeech7k.py
│   └── prepare_nonspeech7k.py
└── docs/
    └── NONSPEECH7K_DATASET_AUDIT_AND_PREPARATION_V1.md
```

## Run from SafeBand-AI repository root

```powershell
python tools\audit_nonspeech7k.py
python tools\prepare_nonspeech7k.py
```

Copy the two scripts into the repository `tools/` directory and the documentation into `docs/` when integrating this package.

The scripts are deliberately independent of a fixed extraction depth and preserve the raw dataset.
