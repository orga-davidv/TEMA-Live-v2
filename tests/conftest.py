import sys
from pathlib import Path

# ensure src is on sys.path for tests to import tema package
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
