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

# Backward compatibility
InsideForest = InsideForestClassifier
