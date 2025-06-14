from pathlib import Path

"""Utility helpers to manage output paths for runs.

All artefacts (
    - temporary parquet/checkpoints,
    - cached pickles or scalers,
    - trained model weights (joblib, pth, …),
) should be written **outside** of the source tree so that the repository
remains clean.  The convention adopted across the project is:

    RUNS/
        data/   # any intermediate data artefacts (parquet, csv, pkl, …)
        models/ # serialised models (joblib, pth, …)

The helpers below expose `data_path()` and `model_path()` helpers that take a
filename (or relative sub-path) and return the full `Path` instance inside the
appropriate directory, creating it on the fly so that callers can write to the
location directly without worrying about its existence.
"""

# --------------------------------------------------------------------------------------
# Locate repository root (…/EcoBici-AI) starting from current file (…/src/utils/path_utils.py)
# --------------------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Base directories
RUNS_DIR: Path = PROJECT_ROOT / "RUNS"
DATA_DIR: Path = RUNS_DIR / "data"
MODELS_DIR: Path = RUNS_DIR / "models"
TEMP_DIR: Path = DATA_DIR / "temp"

# Ensure they exist so the rest of the code can assume the directories are present.
for _dir in (DATA_DIR, MODELS_DIR, TEMP_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def data_path(filename: str | Path) -> Path:
    """Return a path inside RUNS/data for the given *filename* (string or Path)."""
    return (DATA_DIR / filename).resolve()


def model_path(filename: str | Path) -> Path:
    """Return a path inside RUNS/models for the given *filename* (string or Path)."""
    return (MODELS_DIR / filename).resolve()


def temp_path(filename: str | Path | None = None) -> Path:
    """Return a path inside RUNS/data/temp.

    If *filename* is *None*, simply returns the *temp* directory itself.
    """
    if filename is None:
        return TEMP_DIR
    return (TEMP_DIR / filename).resolve()


__all__ = [
    "PROJECT_ROOT",
    "RUNS_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "TEMP_DIR",
    "data_path",
    "model_path",
    "temp_path",
] 