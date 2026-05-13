"""Check that the current tree is safe for a student-facing release."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.create_student_material import FORBIDDEN_STUDENT_TEXT, PROJECT_DIRS, REQUIRED_PLANE_NOTEBOOKS, student_notebooks


REQUIRED_DIRECTORIES = [
    Path("plane_code"),
    Path("projects"),
    Path("tools"),
    Path("tests"),
]

GENERATED_OUTPUT_PARTS = {
    "outputs",
    "figures",
    "__pycache__",
}


def _source_to_text(source: list[str] | str) -> str:
    return "".join(source) if isinstance(source, list) else source


def fail(message: str) -> None:
    raise SystemExit(f"student release check failed: {message}")


def current_branch() -> str:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def strict_public_projection_mode() -> bool:
    mode = os.environ.get("THIMPC_PUBLIC_PROJECTION", "").lower()
    return mode in {"1", "true", "yes"} or current_branch() == "main"


def compact_solution_notebooks() -> list[Path]:
    leaked = sorted(Path("projects").rglob("*solution*.ipynb"))
    leaked += sorted(Path("plane_code").rglob("*solution*.ipynb"))
    return leaked


def check_required_structure() -> None:
    for path in REQUIRED_DIRECTORIES:
        if not path.is_dir():
            fail(f"missing required directory: {path}")
    for path in REQUIRED_PLANE_NOTEBOOKS:
        if not path.exists():
            fail(f"missing required plane-code notebook: {path}")
    for project_dir in PROJECT_DIRS:
        if not project_dir.is_dir():
            fail(f"missing required project directory: {project_dir}")
        walkthrough = project_dir / "walkthrough.ipynb"
        if not walkthrough.exists():
            fail(f"missing required project walkthrough: {walkthrough}")


def check_no_solution_notebooks_for_public_projection() -> None:
    if not strict_public_projection_mode():
        return
    leaked = compact_solution_notebooks()
    if leaked:
        fail("solution notebooks are present in compact public material: " + ", ".join(str(path) for path in leaked))


def check_no_tracked_outputs() -> None:
    try:
        tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        fail("could not inspect tracked files with git ls-files")

    leaked = []
    for name in tracked:
        path = Path(name)
        parts = set(path.parts)
        if GENERATED_OUTPUT_PARTS & parts:
            leaked.append(name)
        elif path.name.endswith(("_executed.ipynb", ".pyc", ".pyo")):
            leaked.append(name)
    if leaked:
        fail("generated outputs are tracked: " + ", ".join(sorted(leaked)))


def check_student_notebooks() -> None:
    notebooks = student_notebooks()
    if not notebooks:
        fail("no compact student notebooks found")
    for path in notebooks:
        if not path.exists():
            fail(f"missing student notebook: {path}")
        if "solution" in path.name.lower():
            fail(f"student notebook path looks instructor-only: {path}")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        parts: list[str] = []
        for cell in notebook.get("cells", []):
            parts.append(_source_to_text(cell.get("source", "")))
            if cell.get("cell_type") == "code":
                if cell.get("execution_count") is not None:
                    fail(f"{path}: code cell still has execution_count")
                if cell.get("outputs"):
                    fail(f"{path}: code cell still has outputs")
                for output in cell.get("outputs", []):
                    parts.append(json.dumps(output, sort_keys=True))
        text = "\n".join(parts).lower()
        leaked = [marker for marker in FORBIDDEN_STUDENT_TEXT if marker.lower() in text]
        if leaked:
            fail(f"{path}: instructor-only content leaked: {', '.join(leaked)}")


def main() -> None:
    check_required_structure()
    check_no_tracked_outputs()
    check_no_solution_notebooks_for_public_projection()
    check_student_notebooks()
    print("student release check passed")


if __name__ == "__main__":
    main()
