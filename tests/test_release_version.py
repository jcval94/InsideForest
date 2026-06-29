import json
import re
from importlib import metadata
from pathlib import Path

import InsideForest


RELEASE_VERSION = "0.4.1"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _project_version():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_public_and_distribution_versions_match_release():
    assert _project_version() == RELEASE_VERSION
    assert InsideForest.__version__ == RELEASE_VERSION
    assert metadata.version("InsideForest") == RELEASE_VERSION


def test_release_version_is_used_by_docs_and_notebook():
    notebook = json.loads(
        (REPO_ROOT / "InsideForest/examples/InsideForest_Caso_de_Uso.ipynb")
        .read_text(encoding="utf-8")
    )
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "InsideForest>=0.4.1" in notebook_source

    for relative_path in (
        "README.md",
        "README.es.md",
        "docs/index.html",
        "docs/index_es.html",
        "docs/installation.html",
        "docs/installation_es.html",
        "docs/changelog.html",
        "docs/changelog_es.html",
    ):
        content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert RELEASE_VERSION in content, relative_path
