import json
import os
from importlib.resources import as_file
from pathlib import Path

import nbformat
from nbclient import NotebookClient

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
    assert "InsideForestClassRegionClusterer" in source
    assert "InsideForestContinuousRegionClusterer" in source
    assert "generate_model_hypothesis" in source
    assert "RUN_OPENAI_DEMO = False" in source
    assert "traditional, \"species\"" in source
    assert "multiclass_model, \"wine_class\"" in source
    assert "continuous_model, \"disease_progression\"" in source
    assert "load_wine" in source
    assert "regions_for_class" in source
    assert "ambiguous_regions" in source
    assert "assign_regions" in source
    assert "region_quality_report" in source
    assert "branch_aggregation=\"none\"" in source
    assert "transform(X_wine_test)" in source
    assert "rf_accuracy" not in source
    assert "df_clusters_description_" not in source
    assert "frontiers_" not in source
    assert "model_fallback" not in source
    assert "InsideForestRegressor" not in source
    assert "InsideForest==0.4.3" in source
    assert "--upgrade" in source
    assert "--no-cache-dir" in source
    assert "find_spec" not in source
    assert "percentil=" not in source
    assert "low_frac=" not in source
    assert "max_rules_per_class=" not in source
    assert '"eta_squared"' in source
    assert '"forest_r2"' in source
    assert '"forest_rmse"' in source
    assert "cluster `-1`" in source
    assert "regression_region_validation/summary.csv" not in source


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
        assert f"]({NOTEBOOK_RELATIVE_PATH})" not in readme_text
        assert "/blob/main/" not in readme_text


def test_notebook_executes_from_empty_working_directory(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / NOTEBOOK_RELATIVE_PATH
    notebook = nbformat.read(notebook_path, as_version=4)

    # The release installation is verified separately from the wheel. During
    # source tests, replace only the networked bootstrap cell.
    notebook.cells[1].source = """
from importlib.metadata import version
import InsideForest as _insideforest

assert version("InsideForest") == "0.4.3"
for _name in (
    "InsideForestRegionClusterer",
    "InsideForestClassRegionClusterer",
    "InsideForestContinuousRegionClusterer",
    "generate_model_hypothesis",
):
    assert hasattr(_insideforest, _name)
"""

    pythonpath = os.environ.get("PYTHONPATH")
    paths = [str(repo_root)]
    if pythonpath:
        paths.append(pythonpath)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(paths))
    for name in (
        "IPYTHONDIR",
        "JUPYTER_CONFIG_DIR",
        "JUPYTER_DATA_DIR",
        "JUPYTER_RUNTIME_DIR",
    ):
        directory = tmp_path / name.lower()
        directory.mkdir()
        monkeypatch.setenv(name, str(directory))

    client = NotebookClient(
        notebook,
        timeout=240,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    executed = client.execute()

    assert all(
        output.get("output_type") != "error"
        for cell in executed.cells
        for output in cell.get("outputs", [])
    )
