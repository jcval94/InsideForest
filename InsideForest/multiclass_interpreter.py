"""High-level opt-in multiclass interpreter for InsideForest."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .multiclass_labels import get_multiclass_labels
from .multiclass_metrics import build_class_priors
from .multiclass_rules import extract_multiclass_leaf_rules


class InsideForestMulticlassClassifier:
    """Opt-in multiclass interpretation layer for InsideForest.

    The legacy binary-oriented pipeline remains untouched.  This class trains
    or accepts a scikit-learn ``RandomForestClassifier`` and extracts rule rows
    with full class probability vectors per leaf.
    """

    def __init__(
        self,
        rf_params=None,
        percentil=95,
        low_frac=0.05,
        min_support=1,
        max_rules_per_class=None,
        score="purity_lift_coverage",
        random_state=42,
        n_jobs=1,
        conflict_margin=0.15,
    ):
        self.rf_params = dict(rf_params or {})
        self.percentil = percentil
        self.low_frac = low_frac
        self.min_support = min_support
        self.max_rules_per_class = max_rules_per_class
        self.score = score
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.conflict_margin = conflict_margin

        self.rf_ = None
        self.rules_ = None
        self.labels_ = None
        self.class_priors_ = None
        self.classes_ = None
        self.feature_names_ = None

    def fit(self, X, y, rf: Optional[RandomForestClassifier] = None):
        """Fit the forest if needed and extract multiclass rule vectors."""

        X_df = self._coerce_X_fit(X)
        y_fit = np.asarray(y)
        if y_fit.ndim != 1:
            y_fit = np.ravel(y_fit)
        y_array = y_fit.astype(object, copy=False)
        if len(y_array) != len(X_df):
            raise ValueError("X and y must contain the same number of rows")
        if pd.unique(y_array).size < 2:
            raise ValueError("At least two classes are required")

        if rf is None:
            params = dict(self.rf_params)
            params.setdefault("random_state", self.random_state)
            self.rf_ = RandomForestClassifier(**params)
        else:
            if not isinstance(rf, RandomForestClassifier):
                raise TypeError("rf must be a sklearn.ensemble.RandomForestClassifier")
            self.rf_ = rf

        try:
            check_is_fitted(self.rf_)
        except NotFittedError:
            self.rf_.fit(X_df, y_fit)

        self.classes_ = np.asarray(self.rf_.classes_, dtype=object)
        self.class_priors_ = build_class_priors(y_array, self.classes_)
        self.rules_ = extract_multiclass_leaf_rules(
            self.rf_,
            X_df,
            y_array,
            feature_names=self.feature_names_,
            percentil=self.percentil,
            low_frac=self.low_frac,
            min_support=self.min_support,
            max_rules_per_class=self.max_rules_per_class,
            class_priors=self.class_priors_,
            score=self.score,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.labels_ = get_multiclass_labels(
            self.rules_,
            X_df,
            y_array,
            class_labels=self.classes_,
        )
        return self

    def explain(self, class_label=None, top_n=None) -> pd.DataFrame:
        """Return scored multiclass rule rows."""

        self._require_fitted()
        out = self.rules_.copy()
        if class_label is not None:
            self._validate_class_label(class_label)
            out = out[out["target_class"] == class_label]
        out = out.sort_values(["score", "support"], ascending=False)
        if top_n is not None:
            out = out.head(int(top_n))
        return out.reset_index(drop=True)

    def assign_regions(self, X) -> pd.DataFrame:
        """Assign each row to its best matching region or model fallback."""

        self._require_fitted()
        X_df = self._coerce_X_predict(X)
        fallback_proba = self.rf_.predict_proba(X_df)
        fallback_pred = self.rf_.predict(X_df)
        match_matrix = self._rule_match_matrix(X_df)
        class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        rows = []

        for row_position in range(len(X_df)):
            matched_positions = np.flatnonzero(match_matrix[row_position])
            matches = (
                self.rules_.iloc[matched_positions]
                if matched_positions.size
                else self.rules_.iloc[0:0]
            )
            if matches.empty:
                rows.append(self._fallback_assignment(row_position, fallback_pred, fallback_proba))
                continue

            class_scores = (
                matches.groupby("target_class", sort=False)["score"]
                .sum()
                .reindex(self.classes_, fill_value=0.0)
            )
            best_class = class_scores.idxmax()
            best_matches = matches[matches["target_class"] == best_class]
            best = best_matches.sort_values(["score", "support"], ascending=False).iloc[0]
            distribution = np.asarray(best["class_distribution"], dtype=float)
            top = _top_two(self.classes_, distribution)

            rows.append(
                {
                    "region_id": int(best["region_id"]),
                    "predicted_class": best_class,
                    "confidence": float(distribution[class_to_index[best_class]]),
                    "score": float(class_scores.loc[best_class]),
                    "matched_region_count": int(matches["leaf_region_id"].nunique()),
                    "second_class": top["second_class"],
                    "margin": float(top["margin"]),
                    "is_conflict": bool(top["margin"] <= self.conflict_margin),
                    "source": "region",
                }
            )

        return pd.DataFrame(rows)

    def prototype_regions(self, class_label=None, top_n=10) -> pd.DataFrame:
        """Return top prototype regions per class or for one class."""

        self._require_fitted()
        if class_label is not None:
            return self.explain(class_label=class_label, top_n=top_n)

        parts = []
        for label in self.classes_:
            part = self.explain(class_label=label, top_n=top_n)
            parts.append(part)
        return pd.concat(parts, axis=0, ignore_index=True) if parts else self.rules_.iloc[0:0].copy()

    def confusion_regions(self, top_n=20) -> pd.DataFrame:
        """Return regions whose top two class probabilities are close."""

        self._require_fitted()
        unique = (
            self.rules_.sort_values(["leaf_region_id", "score"], ascending=[True, False])
            .drop_duplicates("leaf_region_id")
            .copy()
        )
        metrics = [_top_two(self.classes_, dist) for dist in unique["class_distribution"]]
        for key in ("top_class", "top_probability", "second_class", "second_probability", "margin"):
            unique[key] = [item[key] for item in metrics]
        unique["is_conflict"] = unique["margin"] <= self.conflict_margin
        unique = unique[unique["is_conflict"]].sort_values(
            ["margin", "support", "js_divergence"],
            ascending=[True, False, False],
        )
        return unique.head(int(top_n)).reset_index(drop=True)

    def _matching_rules(self, sample: pd.Series) -> pd.DataFrame:
        if self.rules_.empty:
            return self.rules_

        matched = []
        for idx, rule in self.rules_.iterrows():
            ok = True
            for feature, lower in rule["lower_bounds"].items():
                if float(sample[feature]) <= float(lower):
                    ok = False
                    break
            if not ok:
                continue
            for feature, upper in rule["upper_bounds"].items():
                if float(sample[feature]) > float(upper):
                    ok = False
                    break
            if ok:
                matched.append(idx)
        return self.rules_.loc[matched] if matched else self.rules_.iloc[0:0]

    def _rule_match_matrix(self, X_df: pd.DataFrame) -> np.ndarray:
        n_samples = len(X_df)
        n_rules = len(self.rules_)
        matches = np.ones((n_samples, n_rules), dtype=bool)
        if n_rules == 0:
            return matches

        column_cache = {
            column: X_df[column].to_numpy(dtype=float, copy=False)
            for column in X_df.columns
        }
        for rule_position, rule in enumerate(self.rules_.itertuples(index=False)):
            for feature, lower in getattr(rule, "lower_bounds").items():
                matches[:, rule_position] &= column_cache[feature] > float(lower)
            for feature, upper in getattr(rule, "upper_bounds").items():
                matches[:, rule_position] &= column_cache[feature] <= float(upper)
        return matches

    def _fallback_assignment(self, row_position: int, fallback_pred, fallback_proba) -> dict:
        probabilities = np.asarray(fallback_proba[row_position], dtype=float)
        top = _top_two(self.classes_, probabilities)
        return {
            "region_id": -1,
            "predicted_class": fallback_pred[row_position],
            "confidence": float(top["top_probability"]),
            "score": float(top["top_probability"]),
            "matched_region_count": 0,
            "second_class": top["second_class"],
            "margin": float(top["margin"]),
            "is_conflict": bool(top["margin"] <= self.conflict_margin),
            "source": "model_fallback",
        }

    def _coerce_X_fit(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            self.feature_names_ = list(X_df.columns)
            return X_df
        arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
        if arr.ndim != 2:
            raise ValueError("X must be a 2-dimensional array or DataFrame")
        self.feature_names_ = [f"feature_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=self.feature_names_)

    def _coerce_X_predict(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            missing = [col for col in self.feature_names_ if col not in X_df.columns]
            if missing:
                raise ValueError(f"Input data is missing required feature columns: {missing}")
            return X_df[self.feature_names_]
        arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
        if arr.ndim != 2:
            raise ValueError("X must be a 2-dimensional array or DataFrame")
        return pd.DataFrame(arr, columns=self.feature_names_)

    def _validate_class_label(self, class_label) -> None:
        if class_label not in set(self.classes_):
            raise ValueError(f"Unknown class_label {class_label!r}")

    def _require_fitted(self) -> None:
        if self.rf_ is None or self.rules_ is None:
            raise RuntimeError("InsideForestMulticlassClassifier is not fitted yet")


def _top_two(classes: np.ndarray, probabilities: np.ndarray) -> dict:
    order = np.argsort(-probabilities)
    top_idx = int(order[0])
    second_idx = int(order[1]) if len(order) > 1 else top_idx
    return {
        "top_class": classes[top_idx],
        "top_probability": float(probabilities[top_idx]),
        "second_class": classes[second_idx],
        "second_probability": float(probabilities[second_idx]),
        "margin": float(probabilities[top_idx] - probabilities[second_idx]),
    }
