"""Create student-facing notebooks from compact instructor materials."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


PROJECTS_DIR = Path("projects")
PLANE_CODE_DIR = Path("plane_code")
SOLUTION_NOTEBOOK_NAME = "walkthrough_solution.ipynb"
STUDENT_NOTEBOOK_NAME = "walkthrough.ipynb"
PLANE_SOLUTION_SUFFIX = "_solution.ipynb"
PLANE_STUDENT_SUFFIX = ".ipynb"

REQUIRED_PLANE_NOTEBOOKS = [
    PLANE_CODE_DIR / "01_lqr.ipynb",
    PLANE_CODE_DIR / "02_mpc.ipynb",
    PLANE_CODE_DIR / "03_mpc_geometry.ipynb",
    PLANE_CODE_DIR / "04_casadi.ipynb",
]

PROJECT_DIRS = [
    PROJECTS_DIR / "project_1_attitude_constraints",
    PROJECTS_DIR / "project_2_mobile_robot_corridor",
    PROJECTS_DIR / "project_3_learning_enhanced_prediction",
]

REQUIRED_PROJECT_WALKTHROUGHS = [
    project_dir / STUDENT_NOTEBOOK_NAME
    for project_dir in PROJECT_DIRS
]


def solution_notebooks() -> dict[Path, Path]:
    notebooks = {
        path: path.with_name(STUDENT_NOTEBOOK_NAME)
        for path in sorted(PROJECTS_DIR.glob(f"project_*/{SOLUTION_NOTEBOOK_NAME}"))
    }
    notebooks.update(
        {
            path: path.with_name(path.name.removesuffix(PLANE_SOLUTION_SUFFIX) + PLANE_STUDENT_SUFFIX)
            for path in sorted(PLANE_CODE_DIR.glob(f"*{PLANE_SOLUTION_SUFFIX}"))
        }
    )
    return notebooks


def student_notebooks() -> list[Path]:
    return [*REQUIRED_PLANE_NOTEBOOKS, *REQUIRED_PROJECT_WALKTHROUGHS]


NOTEBOOKS = solution_notebooks()


TODO_BLOCK = [
    "# TODO: implement or tune this design choice.\n",
    "# The surrounding setup is provided so you can focus on the control idea.\n",
]

FORBIDDEN_STUDENT_TEXT = [
    "SOLUTION_START",
    "SOLUTION_END",
    "Instructor solution",
    "private note",
    "instructor-only",
]


def _source_to_text(source: list[str] | str) -> str:
    return "".join(source) if isinstance(source, list) else source


def _text_to_source(text: str) -> list[str]:
    return [line for line in text.splitlines(keepends=True)]


def strip_solution_regions(source: list[str] | str) -> list[str]:
    """Replace code between SOLUTION_START and SOLUTION_END with TODO text."""
    lines = _text_to_source(_source_to_text(source))
    stripped: list[str] = []
    inside_solution = False

    for line in lines:
        if "SOLUTION_START" in line:
            if inside_solution:
                raise ValueError("Nested SOLUTION_START marker")
            inside_solution = True
            stripped.extend(TODO_BLOCK)
            continue
        if "SOLUTION_END" in line:
            if not inside_solution:
                raise ValueError("SOLUTION_END marker without SOLUTION_START")
            inside_solution = False
            continue
        if inside_solution:
            continue
        stripped.append(line)
    if inside_solution:
        raise ValueError("SOLUTION_START marker without SOLUTION_END")
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
    """Fail fast if instructor-only content survived stripping."""
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
        raise ValueError(f"{target_path}: instructor-only content leaked: {', '.join(leaked)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate generation in a temporary directory without changing tracked notebooks",
    )
    args = parser.parse_args()

    notebooks = solution_notebooks()
    if not notebooks:
        print("no solution notebooks found; nothing to generate")
        return
    if args.check:
        with tempfile.TemporaryDirectory(prefix="thimpc_student_material_") as tmp:
            tmp_root = Path(tmp)
            for source_name, target_name in notebooks.items():
                convert_notebook(source_name, tmp_root / target_name)
        return
    for source_name, target_name in notebooks.items():
        convert_notebook(source_name, target_name)


if __name__ == "__main__":
    main()
