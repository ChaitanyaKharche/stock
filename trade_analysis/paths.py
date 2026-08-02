"""Shared filesystem locations, anchored to this package's location rather
than the caller's CWD, so logs/data land in the same place regardless of
which directory a script gets launched from."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

LOGS_DIR = PACKAGE_ROOT / "logs"
DATA_DIR = PACKAGE_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"       # matches the existing cache dir, don't orphan it
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"
LOCAL_DATA_DIR = PROJECT_ROOT / "local_data"

for _dir in (LOGS_DIR, DATA_DIR, CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
