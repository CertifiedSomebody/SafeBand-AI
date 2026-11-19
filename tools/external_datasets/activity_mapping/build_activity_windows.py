"""Compatibility CLI for the unified raw-to-window external activity builder."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.external_datasets.activity_mapping.build_activity_dataset import main


if __name__ == "__main__":
    main()
