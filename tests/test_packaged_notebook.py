import json
from importlib.resources import as_file
from pathlib import Path

from InsideForest.examples import use_case_notebook


NOTEBOOK_RELATIVE_PATH = "InsideForest/examples/InsideForest_Caso_de_Uso.ipynb"
COLAB_NOTEBOOK_URL = (
    "https://colab.research.google.com/github/jcval94/InsideForest/blob/"
    f"master/{NOTEBOOK_RELATIVE_PATH}"
)


def test_use_case_notebook_is_packaged_and_contains_multiclass_example():
    resource = use_case_notebook()

    with as_file(resource) as notebook_path:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    assert notebook["nbformat"] == 4
    assert "InsideForestMulticlassClassifier" in source
    assert "load_wine" in source
    assert "prototype_regions" in source
    assert "confusion_regions" in source
    assert "model_fallback" in source


def test_all_notebook_code_cells_compile():
    resource = use_case_notebook()

    with as_file(resource) as notebook_path:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"notebook-cell-{index}", "exec")


def test_readme_colab_links_point_to_default_branch_notebook():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / NOTEBOOK_RELATIVE_PATH

    assert notebook_path.is_file()

    for readme_name in ("README.md", "README.es.md"):
        readme_text = (repo_root / readme_name).read_text(encoding="utf-8")

        assert COLAB_NOTEBOOK_URL in readme_text
        assert "/blob/main/" not in readme_text
