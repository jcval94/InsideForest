from .labels import Labels
from .regions import Regions
from .trees import Trees
from .models import Models
from .inside_forest import InsideForestClassifier, InsideForestRegressor
from .metadata import (
    MetaExtractor,
    Profile,
    parse_rule_string,
    token_from_condition,
    conditions_to_tokens,
    experiments_from_df2,
    run_experiments,
)
from .cluster_selector import (
    MenuClusterSelector,
    balance_lists_n_clusters,
    max_prob_clusters,
    match_class_distribution,
    ChimeraValuesSelector,
)
from .group_priority import (
    RegionDescriptor,
    feature_weights_from_model,
    from_multiclass_rules,
    from_traditional_regions,
    rank_region_pairs,
)
from .region_quality import (
    build_region_rule_table,
    score_region_rules,
    summarize_region_quality,
    cluster_label_quality,
)

# Backward compatibility
InsideForest = InsideForestClassifier
