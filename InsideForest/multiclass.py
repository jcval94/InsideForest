"""Public facade for class-aware supervised region clustering."""

from .multiclass_interpreter import (
    InsideForestClassRegionClusterer,
    InsideForestMulticlassClassifier,
)
from .multiclass_labels import get_multiclass_labels
from .multiclass_metrics import score_multiclass_rules
from .multiclass_rules import extract_multiclass_leaf_rules

__all__ = [
    "InsideForestClassRegionClusterer",
    "InsideForestMulticlassClassifier",
    "extract_multiclass_leaf_rules",
    "score_multiclass_rules",
    "get_multiclass_labels",
]
