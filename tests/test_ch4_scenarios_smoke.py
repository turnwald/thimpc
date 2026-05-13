import json
from pathlib import Path

from tools.create_student_material import solution_notebooks, student_notebooks


def notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", [])
        parts.append("".join(source) if isinstance(source, list) else source)
    return "\n".join(parts)


def test_compact_solution_mapping_ignores_retired_material():
    pairs = solution_notebooks()
    solution_sources = sorted(Path("plane_code").glob("*_solution.ipynb"))
    solution_sources += sorted(Path("projects").glob("project_*/walkthrough_solution.ipynb"))
    if not solution_sources:
        assert pairs == {}
        return
    assert pairs
    assert all(path.parts[0] in {"plane_code", "projects"} for path in pairs)
    assert all(target.parts[0] in {"plane_code", "projects"} for target in pairs.values())


def test_compact_student_notebook_list_ignores_retired_material():
    notebooks = student_notebooks()
    assert notebooks
    assert all(path.parts[0] in {"plane_code", "projects"} for path in notebooks)
    assert all("solution" not in path.name.lower() for path in notebooks)


def test_compact_student_notebooks_exist_and_are_clean():
    for path in student_notebooks():
        assert path.exists(), path
        text = notebook_text(path).lower()
        assert "solution_start" not in text
        assert "solution_end" not in text
