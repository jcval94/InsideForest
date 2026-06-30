import os
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest


RELEASE_VERSION = "0.4.3"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _release_wheel():
    configured = os.environ.get("INSIDEFOREST_RELEASE_WHEEL")
    if configured:
        wheel = Path(configured).resolve()
        assert wheel.is_file(), wheel
        return wheel
    matches = sorted((REPO_ROOT / "dist").glob(f"insideforest-{RELEASE_VERSION}-*.whl"))
    if not matches:
        pytest.skip("Build the 0.4.3 wheel before running artifact checks")
    assert len(matches) == 1
    return matches[0]


def _venv_python(environment):
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _clean_subprocess_environment():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    return environment


def test_release_wheel_contains_notebook_exports_and_metadata():
    wheel = _release_wheel()

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")
        package_init = archive.read("InsideForest/__init__.py").decode("utf-8")

    assert "InsideForest/examples/InsideForest_Caso_de_Uso.ipynb" in names
    assert f"Version: {RELEASE_VERSION}" in metadata
    assert f'__version__ = "{RELEASE_VERSION}"' in package_init
    for public_name in (
        "InsideForestRegionClusterer",
        "InsideForestClassRegionClusterer",
        "InsideForestContinuousRegionClusterer",
    ):
        assert public_name in package_init


@pytest.mark.skipif(
    os.environ.get("INSIDEFOREST_RUN_INSTALL_REGRESSION") != "1",
    reason="Set INSIDEFOREST_RUN_INSTALL_REGRESSION=1 for the networked release smoke",
)
def test_release_wheel_upgrades_the_stale_pypi_install(tmp_path):
    wheel = _release_wheel()
    environment = tmp_path / "venv"
    venv.create(environment, with_pip=True, system_site_packages=True)
    python = _venv_python(environment)
    subprocess_environment = _clean_subprocess_environment()

    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-cache-dir",
            "InsideForest==0.4.0",
        ],
        cwd=tmp_path,
        env=subprocess_environment,
        check=True,
    )
    stale_import = subprocess.run(
        [
            str(python),
            "-c",
            "from InsideForest import InsideForestClassRegionClusterer",
        ],
        cwd=tmp_path,
        env=subprocess_environment,
        capture_output=True,
        text=True,
    )
    assert stale_import.returncode != 0
    assert "ImportError" in stale_import.stderr

    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            str(wheel),
        ],
        cwd=tmp_path,
        env=subprocess_environment,
        check=True,
    )
    verified = subprocess.run(
        [
            str(python),
            "-c",
            (
                "from importlib.metadata import version; "
                "from InsideForest import (InsideForestRegionClusterer, "
                "InsideForestClassRegionClusterer, "
                "InsideForestContinuousRegionClusterer); "
                f"assert version('InsideForest') == '{RELEASE_VERSION}'"
            ),
        ],
        cwd=tmp_path,
        env=subprocess_environment,
        capture_output=True,
        text=True,
    )
    assert verified.returncode == 0, verified.stderr
