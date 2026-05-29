"""Public facade for the opt-in multiclass InsideForest API."""

from .multiclass_interpreter import InsideForestMulticlassClassifier
from .multiclass_labels import get_multiclass_labels
from .multiclass_metrics import score_multiclass_rules
from .multiclass_rules import extract_multiclass_leaf_rules

__all__ = [
    "InsideForestMulticlassClassifier",
    "extract_multiclass_leaf_rules",
    "score_multiclass_rules",
    "get_multiclass_labels",
]
