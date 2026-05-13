import json
import os
import subprocess
from pathlib import Path

import pytest

from tools.check_student_release import compact_solution_notebooks, strict_public_projection_mode
from tools.create_student_material import FORBIDDEN_STUDENT_TEXT, PROJECT_DIRS, REQUIRED_PLANE_NOTEBOOKS, student_notebooks


NOTEBOOKS = student_notebooks()


def public_release_mode() -> bool:
    mode = os.environ.get("THIMPC_RELEASE_MODE", "").lower()
    if mode == "public":
        return True
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return False
    return branch == "main"


def notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", [])
        parts.append("".join(source) if isinstance(source, list) else source)
        for output in cell.get("outputs", []):
            parts.append(json.dumps(output, sort_keys=True))
    return "\n".join(parts)


def test_required_compact_student_material_exists():
    for path in REQUIRED_PLANE_NOTEBOOKS:
        assert path.exists(), path
    for project_dir in PROJECT_DIRS:
        assert project_dir.is_dir(), project_dir
        assert (project_dir / "walkthrough.ipynb").exists()


@pytest.mark.skipif(not strict_public_projection_mode(), reason="strict no-solution tree check runs on main/public projection")
def test_public_projection_has_no_compact_solution_notebooks():
    leaked = compact_solution_notebooks()
    assert leaked == []


@pytest.mark.skipif(not public_release_mode(), reason="public release checks run on main or THIMPC_RELEASE_MODE=public")
@pytest.mark.parametrize("student_notebook", NOTEBOOKS)
def test_student_notebooks_have_no_instructor_text_or_outputs(student_notebook):
    text = notebook_text(student_notebook).lower()
    for marker in FORBIDDEN_STUDENT_TEXT:
        assert marker.lower() not in text

    notebook = json.loads(student_notebook.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


def test_no_tracked_generated_outputs():
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    leaked = []
    for name in tracked:
        path = Path(name)
        parts = set(path.parts)
        if {"outputs", "figures", "__pycache__"} & parts:
            leaked.append(name)
        elif path.name.endswith(("_executed.ipynb", ".pyc", ".pyo")):
            leaked.append(name)
    assert leaked == []
