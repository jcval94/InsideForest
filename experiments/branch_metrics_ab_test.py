"""A/B test branch support and effectiveness scoring strategies.

The experiment compares the historical rule-by-rule masks, sklearn's native
``apply`` assignment, and a compatibility traversal that reuses shared tree
prefixes while honoring InsideForest's exported six-decimal thresholds.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from InsideForest.regions import Regions
from InsideForest.trees import Trees, _SUMMARY_ASSIGNMENT_MAX_BYTES
from experiments.get_summary_ab_test import (
    SUMMARY_ATOL,
    assert_summary_equivalent,
    canonicalize_summary,
)


STRATEGIES = ("rule_masks", "native_apply", "shared_prefix")
ADOPTION_MIN_MEDIAN_SPEEDUP = 1.20
ADOPTION_MAX_SLOWDOWN = 1.10


@dataclass
class Scenario:
    name: str
    task: str
    X: pd.DataFrame
    y: np.ndarray
    model: object
    percentile: float = 96
    low_fraction: float = 0.03


def _make_scenarios(random_state: int = 42) -> list[Scenario]:
    scenarios = []

    X_reg, y_reg = make_regression(
        n_samples=750,
        n_features=12,
        n_informative=8,
        noise=2.0,
        random_state=random_state,
    )
    X_reg = pd.DataFrame(X_reg, columns=[f"f{i}" for i in range(X_reg.shape[1])])
    for name, target, bootstrap in (
        ("regression_mixed_bootstrap", y_reg, True),
        ("regression_negative_no_bootstrap", -np.abs(y_reg) - 1.0, False),
    ):
        model = RandomForestRegressor(
            n_estimators=30,
            max_depth=8,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=1,
        ).fit(X_reg, target)
        scenarios.append(
            Scenario(name, "regression", X_reg.copy(), np.asarray(target), model)
        )

    X_cls, y_cls = make_classification(
        n_samples=750,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        weights=[0.65, 0.35],
        random_state=random_state,
    )
    X_cls = pd.DataFrame(X_cls, columns=[f"f{i}" for i in range(X_cls.shape[1])])
    for name, bootstrap in (
        ("binary_bootstrap", True),
        ("binary_no_bootstrap", False),
    ):
        model = RandomForestClassifier(
            n_estimators=30,
            max_depth=8,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=1,
        ).fit(X_cls, y_cls)
        scenarios.append(
            Scenario(name, "classification", X_cls.copy(), y_cls.copy(), model)
        )

    scenarios.append(_threshold_boundary_scenario(random_state))
    return scenarios


def _threshold_boundary_scenario(random_state: int) -> Scenario:
    X, y = make_regression(
        n_samples=180,
        n_features=5,
        n_informative=4,
        noise=0.5,
        random_state=random_state + 17,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    model = RandomForestRegressor(
        n_estimators=8,
        max_depth=5,
        bootstrap=True,
        random_state=random_state + 17,
        n_jobs=1,
    ).fit(X, y)

    estimator = model.estimators_[0]
    tree = estimator.tree_
    path = estimator.decision_path(X.to_numpy()).tocsc()
    injected = False
    for node in range(tree.node_count):
        if tree.children_left[node] == tree.children_right[node]:
            continue
        raw = float(tree.threshold[node])
        rounded = float(f"{raw:.6f}")
        if raw == rounded:
            continue
        reaching = path.indices[path.indptr[node] : path.indptr[node + 1]]
        if reaching.size == 0:
            continue
        row_index = int(reaching[0])
        probe = X.iloc[row_index].copy()
        probe.iloc[int(tree.feature[node])] = (raw + rounded) / 2.0
        X = pd.concat([X, probe.to_frame().T], ignore_index=True)
        y = np.concatenate([np.asarray(y), [float(y[row_index])]])
        injected = True
        break
    if not injected:
        raise RuntimeError("Could not construct a threshold-boundary scenario")

    return Scenario(
        "threshold_boundary",
        "regression",
        X,
        np.asarray(y),
        model,
        percentile=0,
        low_fraction=0,
    )


def _score_functions(
    trees: Trees,
    df: pd.DataFrame,
    parsed_rules: pd.DataFrame,
    target: str,
    model,
) -> dict[str, Callable[[], pd.DataFrame]]:
    return {
        "rule_masks": lambda: trees._get_summary_optimizado_impl(
            df, parsed_rules, target, n_jobs=1
        ),
        "native_apply": lambda: trees._get_summary_from_fitted_forest(
            df,
            parsed_rules,
            target,
            model,
            assignment_strategy="native_apply",
        ),
        "shared_prefix": lambda: trees._get_summary_from_fitted_forest(
            df,
            parsed_rules,
            target,
            model,
            assignment_strategy="shared_prefix",
            max_work_bytes=_SUMMARY_ASSIGNMENT_MAX_BYTES,
        ),
    }


def _same_branch_identity(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    identity = ["N_arbol", "N_regla", "feature", "operador", "rangos"]
    return canonicalize_summary(left)[identity].equals(
        canonicalize_summary(right)[identity]
    )


def _frames_equivalent(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=SUMMARY_ATOL,
        )
        return True
    except AssertionError:
        return False


def _downstream_outputs(
    trees: Trees,
    summary: pd.DataFrame,
    df: pd.DataFrame,
    target: str,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], np.ndarray]:
    rectangles = trees.extract_rectangles(summary)
    if not rectangles:
        return [], [], np.array([], dtype=int)
    regions = Regions()
    prioritized = regions.prio_ranges(rectangles, df)
    labels = regions.labels(
        df=df,
        df_reres=prioritized,
        n_clusters=None,
        include_summary_cluster=False,
        method="select_clusters",
        return_dfs=False,
        var_obj=target,
        seed=42,
    )
    return rectangles, prioritized, np.asarray(labels)


def _same_frame_lists(left: list[pd.DataFrame], right: list[pd.DataFrame]) -> bool:
    return len(left) == len(right) and all(
        _frames_equivalent(left_frame, right_frame)
        for left_frame, right_frame in zip(left, right)
    )


def _native_node_divergence(
    trees: Trees,
    model,
    X: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, float]:
    support_matches = []
    support_differences = []
    effectiveness_differences = []
    classes = np.asarray(getattr(model, "classes_", []))

    for tree_index, estimator in enumerate(model.estimators_):
        assignments = estimator.apply(X.to_numpy())
        selected = trees._selected_leaf_ids(estimator, tree_index)
        for leaf_id in selected.values():
            mask = assignments == leaf_id
            full_support = int(mask.sum())
            node_support = int(estimator.tree_.n_node_samples[leaf_id])
            support_matches.append(full_support == node_support)
            support_differences.append(abs(full_support - node_support))
            if full_support == 0:
                continue
            full_effectiveness = float(np.asarray(y)[mask].mean())
            node_value = np.asarray(estimator.tree_.value[leaf_id], dtype=float).ravel()
            if classes.size:
                probabilities = node_value / node_value.sum() if node_value.sum() else node_value
                node_effectiveness = float(probabilities @ classes.astype(float))
            else:
                node_effectiveness = float(node_value[0])
            effectiveness_differences.append(
                abs(full_effectiveness - node_effectiveness)
            )

    return {
        "node_support_match_rate": float(np.mean(support_matches)) if support_matches else np.nan,
        "node_support_max_abs_diff": float(max(support_differences, default=0)),
        "node_effectiveness_max_abs_diff": float(
            max(effectiveness_differences, default=0.0)
        ),
    }


def _within_shared_prefix_budget(model, n_samples: int) -> bool:
    item_bytes = np.dtype(np.intp).itemsize + np.dtype(bool).itemsize
    return all(
        n_samples * (int(estimator.tree_.max_depth) + 2) * item_bytes
        <= _SUMMARY_ASSIGNMENT_MAX_BYTES
        for estimator in model.estimators_
    )


def _run_scenario(scenario: Scenario, n_runs: int) -> list[dict]:
    target = "target"
    df = scenario.X.copy()
    df[target] = scenario.y
    trees = Trees(
        percentil=scenario.percentile,
        low_frac=scenario.low_fraction,
    )
    parsed_rules = trees.get_fro(
        trees.get_rangos(scenario.model, scenario.X, percentil=scenario.percentile)
    )
    score_functions = _score_functions(
        trees, df, parsed_rules, target, scenario.model
    )

    outputs = {strategy: function() for strategy, function in score_functions.items()}
    baseline = outputs["rule_masks"]
    baseline_downstream = _downstream_outputs(trees, baseline, df, target)
    node_metrics = _native_node_divergence(
        trees, scenario.model, scenario.X, scenario.y
    )
    shared_prefix_within_budget = _within_shared_prefix_budget(
        scenario.model, len(scenario.X)
    )
    compatibility = {}
    for strategy, output in outputs.items():
        try:
            max_abs_diff = assert_summary_equivalent(output, baseline)
            summary_equal = True
        except AssertionError:
            max_abs_diff = np.nan
            summary_equal = False
        downstream = _downstream_outputs(trees, output, df, target)
        compatibility[strategy] = {
            "summary_equal": summary_equal,
            "max_abs_diff": max_abs_diff,
            "branch_identity_equal": _same_branch_identity(output, baseline),
            "regions_equal": _same_frame_lists(
                downstream[0], baseline_downstream[0]
            )
            and _same_frame_lists(downstream[1], baseline_downstream[1]),
            "labels_equal": np.array_equal(
                downstream[2], baseline_downstream[2]
            ),
        }

    rows = []
    for run_id in range(n_runs):
        order = STRATEGIES[run_id % len(STRATEGIES) :] + STRATEGIES[: run_id % len(STRATEGIES)]
        durations = {}
        for strategy in order:
            start = perf_counter()
            score_functions[strategy]()
            durations[strategy] = perf_counter() - start
        baseline_duration = durations["rule_masks"]
        for strategy in STRATEGIES:
            row = {
                "scenario": scenario.name,
                "task": scenario.task,
                "run_id": run_id,
                "strategy": strategy,
                "duration_seconds": durations[strategy],
                "speedup_vs_rule_masks": baseline_duration / durations[strategy],
                "n_summary_rows": len(outputs[strategy]),
                "within_memory_budget": (
                    shared_prefix_within_budget
                    if strategy == "shared_prefix"
                    else True
                ),
                **compatibility[strategy],
                **node_metrics,
            }
            rows.append(row)
    return rows


def _aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "summary_equal",
        "branch_identity_equal",
        "regions_equal",
        "labels_equal",
        "within_memory_budget",
    ]
    grouped = raw.groupby(["scenario", "task", "strategy"], as_index=False)
    aggregate = grouped.agg(
        duration_seconds=("duration_seconds", "median"),
        speedup_vs_rule_masks=("speedup_vs_rule_masks", "median"),
        min_speedup_vs_rule_masks=("speedup_vs_rule_masks", "min"),
        max_abs_diff=("max_abs_diff", "max"),
        n_summary_rows=("n_summary_rows", "median"),
        node_support_match_rate=("node_support_match_rate", "first"),
        node_support_max_abs_diff=("node_support_max_abs_diff", "first"),
        node_effectiveness_max_abs_diff=("node_effectiveness_max_abs_diff", "first"),
        **{column: (column, "all") for column in columns},
    )
    aggregate.insert(2, "run_id", "aggregate")
    return aggregate


def adoption_decision(raw: pd.DataFrame, strategy: str = "shared_prefix") -> dict:
    candidate = raw[raw["strategy"] == strategy]
    compatibility_columns = [
        "summary_equal",
        "branch_identity_equal",
        "regions_equal",
        "labels_equal",
        "within_memory_budget",
    ]
    all_equivalent = bool(candidate[compatibility_columns].to_numpy(dtype=bool).all())
    median_speedup = float(candidate["speedup_vs_rule_masks"].median())
    minimum_speedup = float(candidate["speedup_vs_rule_masks"].min())
    adopted = (
        all_equivalent
        and median_speedup >= ADOPTION_MIN_MEDIAN_SPEEDUP
        and minimum_speedup >= 1.0 / ADOPTION_MAX_SLOWDOWN
    )
    return {
        "strategy": strategy,
        "adopted": adopted,
        "all_equivalent": all_equivalent,
        "median_speedup": median_speedup,
        "minimum_speedup": minimum_speedup,
    }


def _write_report(path: Path, aggregate: pd.DataFrame, decision: dict) -> None:
    native = aggregate[aggregate["strategy"] == "native_apply"]
    shared = aggregate[aggregate["strategy"] == "shared_prefix"]
    bootstrap_rows = aggregate[aggregate["scenario"].str.contains("bootstrap")]
    node_match = float(bootstrap_rows["node_support_match_rate"].min())
    lines = [
        "# Branch metric strategy A/B test",
        "",
        "## Metric audit",
        "",
        "- `n_sample` is empirical support on the DataFrame passed to InsideForest; it is not geometric hyperrectangle volume.",
        "- `ef_sample` is `mean(y)` for covered rows. For binary labels encoded as 0/1 this is the positive-class rate; it is not a label-invariant multiclass purity metric.",
        "- sklearn node statistics summarize the tree training sample. With bootstrap they are not equivalent to rescoring the full analysis DataFrame.",
        f"- Lowest observed native-node support match rate in bootstrap scenarios: {node_match:.3f}.",
        "",
        "## Decision",
        "",
        f"- Candidate: `{decision['strategy']}`.",
        f"- Public-output equivalence: `{decision['all_equivalent']}`.",
        f"- Median speedup: {decision['median_speedup']:.2f}x.",
        f"- Worst observed speedup: {decision['minimum_speedup']:.2f}x.",
        f"- Adopted in production: `{decision['adopted']}`.",
        "",
        "The native `apply(X)` path is retained only as experimental evidence because full-precision sklearn thresholds can disagree with InsideForest's historical six-decimal exported rules at boundary values.",
        "",
        "## Scenario medians",
        "",
        "| Scenario | Strategy | Seconds | Speedup | Summary equal | Regions equal | Labels equal |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in pd.concat([native, shared]).sort_values(["scenario", "strategy"]).itertuples():
        lines.append(
            f"| {row.scenario} | {row.strategy} | {row.duration_seconds:.4f} | "
            f"{row.speedup_vs_rule_masks:.2f}x | {row.summary_equal} | "
            f"{row.regions_equal} | {row.labels_equal} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_strategy_ab_test(
    output_dir: Path | None = None,
    *,
    random_state: int = 42,
    n_runs: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    rows = [
        row
        for scenario in _make_scenarios(random_state)
        for row in _run_scenario(scenario, n_runs)
    ]
    raw = pd.DataFrame(rows)
    aggregate = _aggregate(raw)
    decision = adoption_decision(raw)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.concat([raw, aggregate], ignore_index=True).to_csv(
            output_dir / "results.csv", index=False
        )
        _write_report(output_dir / "summary.md", aggregate, decision)
    return raw, aggregate, decision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/branch_metrics_strategy"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _, aggregate, decision = run_strategy_ab_test(
        args.output_dir,
        random_state=args.random_state,
        n_runs=args.n_runs,
    )
    print(aggregate.to_string(index=False))
    print(decision)


if __name__ == "__main__":
    main()
