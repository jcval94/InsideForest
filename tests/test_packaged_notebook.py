import json
from importlib.resources import as_file

from InsideForest.examples import use_case_notebook


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
