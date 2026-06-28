"""Compatibility entry point for supervised region-cluster validation.

The former classifier/fallback benchmark was removed because the class-aware
InsideForest contract now returns region cluster IDs rather than class labels.
"""

from validate_class_region_clusters import main


if __name__ == "__main__":
    main()
