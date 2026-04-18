import sys
from pathlib import Path

# Ensure repo root + src are on sys.path for tests.
# - repo root: import run_pipeline.py, scripts/* (namespace package)
# - src: import tema package
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
