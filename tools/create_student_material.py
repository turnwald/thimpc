"""Create student-facing notebooks from instructor solution notebooks."""

from __future__ import annotations

import json
from pathlib import Path


STUDIES_DIR = Path("studies")
PROJECTS_DIR = Path("projects")
PLANE_CODE_DIR = Path("plane_code")
SOLUTION_NOTEBOOK_NAME = "walkthrough_solution.ipynb"
STUDENT_NOTEBOOK_NAME = "walkthrough.ipynb"
PLANE_SOLUTION_SUFFIX = "_solution.ipynb"
PLANE_STUDENT_SUFFIX = ".ipynb"


def solution_notebooks() -> dict[Path, Path]:
    notebooks = {
        path: path.with_name(STUDENT_NOTEBOOK_NAME)
        for path in sorted(STUDIES_DIR.glob(f"study_*/{SOLUTION_NOTEBOOK_NAME}"))
    }
    notebooks.update(
        {
            path: path.with_name(STUDENT_NOTEBOOK_NAME)
            for path in sorted(PROJECTS_DIR.glob(f"project_*/{SOLUTION_NOTEBOOK_NAME}"))
        }
    )
    notebooks.update(
        {
            path: path.with_name(path.name.removesuffix(PLANE_SOLUTION_SUFFIX) + PLANE_STUDENT_SUFFIX)
            for path in sorted(PLANE_CODE_DIR.glob(f"*{PLANE_SOLUTION_SUFFIX}"))
        }
    )
    return notebooks


def study_notebooks() -> dict[Path, Path]:
    """Return legacy study notebook pairs for older tests and callers."""
    return {
        path: path.with_name(STUDENT_NOTEBOOK_NAME)
        for path in sorted(STUDIES_DIR.glob(f"study_*/{SOLUTION_NOTEBOOK_NAME}"))
    }


def student_notebooks() -> list[Path]:
    notebooks = sorted(STUDIES_DIR.glob(f"study_*/{STUDENT_NOTEBOOK_NAME}"))
    notebooks += sorted(PROJECTS_DIR.glob(f"project_*/{STUDENT_NOTEBOOK_NAME}"))
    notebooks += [
        path
        for path in sorted(PLANE_CODE_DIR.glob(f"*{PLANE_STUDENT_SUFFIX}"))
        if not path.name.endswith(PLANE_SOLUTION_SUFFIX)
    ]
    return notebooks


NOTEBOOKS = solution_notebooks()


TODO_BLOCK = [
    "# TODO: implement or tune this design choice.\n",
    "# The surrounding setup is provided so you can focus on the control idea.\n",
]

START_MARKER = "SOLUTION_" + "START"
END_MARKER = "SOLUTION_" + "END"

FORBIDDEN_STUDENT_TEXT = [
    START_MARKER,
    END_MARKER,
    "Instructor " + "solution",
    "private " + "note",
    "instructor-" + "only",
]


def _source_to_text(source: list[str] | str) -> str:
    return "".join(source) if isinstance(source, list) else source


def _text_to_source(text: str) -> list[str]:
    return [line for line in text.splitlines(keepends=True)]


def strip_solution_regions(source: list[str] | str) -> list[str]:
    """Replace restricted code regions with TODO text."""
    lines = _text_to_source(_source_to_text(source))
    stripped: list[str] = []
    inside_solution = False

    for line in lines:
        if START_MARKER in line:
            if inside_solution:
                raise ValueError("Nested restricted start marker")
            inside_solution = True
            stripped.extend(TODO_BLOCK)
            continue
        if END_MARKER in line:
            if not inside_solution:
                raise ValueError("restricted end marker without start marker")
            inside_solution = False
            continue
        if inside_solution:
            continue
        stripped.append(line)
    if inside_solution:
        raise ValueError("restricted start marker without end marker")
    return stripped


def is_instructor_markdown(cell: dict) -> bool:
    if cell.get("cell_type") != "markdown":
        return False
    text = _source_to_text(cell.get("source", ""))
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return first_line.lower().startswith("## instructor solution")


def convert_notebook(source_path: Path, target_path: Path) -> None:
    nb = json.loads(source_path.read_text(encoding="utf-8"))
    cells = []
    for cell in nb.get("cells", []):
        if is_instructor_markdown(cell):
            continue
        new_cell = dict(cell)
        new_cell.setdefault("id", f"student-{len(cells):02d}")
        if new_cell.get("cell_type") == "code":
            new_cell["source"] = strip_solution_regions(new_cell.get("source", []))
            new_cell["execution_count"] = None
            new_cell["outputs"] = []
        cells.append(new_cell)
    nb["cells"] = cells
    validate_student_notebook(nb, target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")


def validate_student_notebook(nb: dict, target_path: Path) -> None:
    """Fail fast if restricted content survived stripping."""
    text_parts: list[str] = []
    for cell in nb.get("cells", []):
        text_parts.append(_source_to_text(cell.get("source", "")))
        if cell.get("cell_type") == "code":
            if cell.get("execution_count") is not None:
                raise ValueError(f"{target_path}: code cell still has execution_count")
            if cell.get("outputs"):
                raise ValueError(f"{target_path}: code cell still has outputs")
            for output in cell.get("outputs", []):
                text_parts.append(json.dumps(output, sort_keys=True))
    text = "\n".join(text_parts).lower()
    leaked = [marker for marker in FORBIDDEN_STUDENT_TEXT if marker.lower() in text]
    if leaked:
        raise ValueError(f"{target_path}: restricted content leaked: {', '.join(leaked)}")


def main() -> None:
    notebooks = solution_notebooks()
    if not notebooks:
        print("no solution notebooks found; nothing to generate")
        return
    for source_name, target_name in notebooks.items():
        convert_notebook(source_name, target_name)


if __name__ == "__main__":
    main()
