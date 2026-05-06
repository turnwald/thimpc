import json
import os
import subprocess
from pathlib import Path

import pytest

from tools.create_student_material import FORBIDDEN_STUDENT_TEXT, student_notebooks


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


@pytest.mark.skipif(not public_release_mode(), reason="public release check runs on main or THIMPC_RELEASE_MODE=public")
def test_public_release_has_no_solution_notebooks():
    leaked = sorted(Path("studies").rglob("*solution*.ipynb"))
    leaked += sorted(Path("notebooks_solution").glob("*.ipynb"))
    leaked += sorted(Path("notebooks").rglob("*_solution.ipynb"))
    assert leaked == []


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
