import json
import os
from pathlib import Path

import pytest

from tools.create_student_material import FORBIDDEN_STUDENT_TEXT, convert_notebook, strip_solution_regions, study_notebooks


INSTRUCTOR_RELEASE_MODES = {"instructor", "solutions", "private"}
NOTEBOOKS = study_notebooks()


def require_solution_notebook(path: Path) -> None:
    if path.exists():
        return
    mode = os.environ.get("THIMPC_RELEASE_MODE", "").lower()
    if mode in INSTRUCTOR_RELEASE_MODES:
        pytest.fail(f"missing instructor solution notebook: {path}")
    pytest.skip("solution notebooks are intentionally absent on public/student branches")


def notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for cell in notebook["cells"]:
        source = cell.get("source", [])
        parts.append("".join(source) if isinstance(source, list) else source)
        for output in cell.get("outputs", []):
            parts.append(json.dumps(output, sort_keys=True))
    return "\n".join(parts)


@pytest.mark.parametrize(("source_name", "target_name"), NOTEBOOKS.items())
def test_student_strip_removes_instructor_solution(tmp_path, source_name, target_name):
    require_solution_notebook(source_name)
    target = tmp_path / Path(target_name).name

    convert_notebook(source_name, target)
    text = notebook_text(target)

    for marker in FORBIDDEN_STUDENT_TEXT:
        assert marker.lower() not in text.lower()
    assert "TODO: implement or tune this design choice" in text


@pytest.mark.parametrize("target_name", NOTEBOOKS.values())
def test_committed_student_notebooks_have_no_solution_leakage(target_name):
    path = Path(target_name)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = notebook_text(path)

    for marker in FORBIDDEN_STUDENT_TEXT:
        assert marker.lower() not in text.lower()
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


@pytest.mark.parametrize(("source_name", "target_name"), NOTEBOOKS.items())
def test_student_generation_matches_committed_notebooks(tmp_path, source_name, target_name):
    require_solution_notebook(source_name)
    generated = tmp_path / Path(target_name).name
    convert_notebook(source_name, generated)

    expected = json.loads(target_name.read_text(encoding="utf-8"))
    actual = json.loads(generated.read_text(encoding="utf-8"))
    assert actual == expected


def test_strip_solution_regions_rejects_unclosed_marker():
    with pytest.raises(ValueError, match="without SOLUTION_END"):
        strip_solution_regions(["keep\n", "# SOLUTION_START\n", "secret\n"])


def test_strip_solution_regions_rejects_unmatched_end_marker():
    with pytest.raises(ValueError, match="without SOLUTION_START"):
        strip_solution_regions(["# SOLUTION_END\n"])


def test_strip_solution_regions_rejects_nested_marker():
    with pytest.raises(ValueError, match="Nested"):
        strip_solution_regions(["# SOLUTION_START\n", "# SOLUTION_START\n", "# SOLUTION_END\n"])
