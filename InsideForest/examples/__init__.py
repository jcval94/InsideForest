"""Packaged, executable examples for InsideForest."""

from importlib.resources import files


def use_case_notebook():
    """Return the packaged InsideForest use-case notebook resource."""

    return files(__package__).joinpath("InsideForest_Caso_de_Uso.ipynb")


__all__ = ["use_case_notebook"]
