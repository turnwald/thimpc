"""Backward-compatible wrapper for the study-wide student generator."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.create_student_material import *  # noqa: F401,F403
from tools.create_student_material import main


if __name__ == "__main__":
    main()
