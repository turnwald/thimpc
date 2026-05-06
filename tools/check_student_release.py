"""Check that the current tree is safe for a student-facing release."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.create_student_material import FORBIDDEN_STUDENT_TEXT, student_notebooks


def _source_to_text(source: list[str] | str) -> str:
    return "".join(source) if isinstance(source, list) else source


def fail(message: str) -> None:
    raise SystemExit(f"student release check failed: {message}")


def check_no_solution_notebooks() -> None:
    leaked = sorted(Path("studies").rglob("*solution*.ipynb"))
    leaked += sorted(Path("projects").rglob("*solution*.ipynb"))
    leaked += sorted(Path("plane_code").rglob("*solution*.ipynb"))
    leaked += sorted(Path("notebooks_solution").glob("*.ipynb"))
    leaked += sorted(Path("notebooks").rglob("*_solution.ipynb"))
    if leaked:
        fail("solution notebooks are present: " + ", ".join(str(path) for path in leaked))


def check_student_notebooks() -> None:
    notebooks = student_notebooks()
    if not notebooks:
        fail("no student study notebooks found")
    for path in notebooks:
        if not path.exists():
            fail(f"missing student notebook: {path}")
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
            fail(f"{path}: restricted content leaked: {', '.join(leaked)}")


def main() -> None:
    check_no_solution_notebooks()
    check_student_notebooks()
    print("student release check passed")


if __name__ == "__main__":
    main()
