import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from InsideForest.trees import Trees
from experiments.get_summary_ab_test import (
    run_ab_test,
    _baseline_get_summary_optimizado,
    assert_summary_equivalent,
)
from experiments.branch_metrics_ab_test import (
    _downstream_outputs,
    _threshold_boundary_scenario,
    adoption_decision,
)


def _build_regression_data():
    X = np.zeros((4, 11))
    X[:, 1] = [0, 1, 0, 1]
    X[:, 10] = [0, 0, 1, 1]
    y = [0, 1, 1, 2]
    return pd.DataFrame(X, columns=[f"col{i}" for i in range(11)]), y


@pytest.fixture(scope="module")
def randomized_summary_inputs():
    rng = np.random.default_rng(123)
    n_samples = 180
    n_features = 7

    X = rng.normal(size=(n_samples, n_features))
    feature_names = [f"f{i}" for i in range(n_features)]
    df_features = pd.DataFrame(X, columns=feature_names)

    weights = rng.normal(size=n_features)
    y = (X @ weights + rng.normal(scale=0.5, size=n_samples)).astype(float)

    reg = RandomForestRegressor(
        n_estimators=6,
        max_depth=6,
        random_state=321,
    ).fit(df_features, y)

    trees = Trees()
    df_full_arboles = trees.get_rangos(reg, df_features, percentil=0, n_jobs=1)
    df_full_arboles = trees.get_fro(df_full_arboles)

    df_with_target = df_features.copy()
    df_with_target["target"] = y

    return trees, df_with_target, df_full_arboles


def test_feature_replacement_no_overlap():
    df, y = _build_regression_data()
    reg = RandomForestRegressor(n_estimators=1, random_state=0).fit(df, y)
    t = Trees()
    res = t.get_rangos(reg, df, percentil=0, n_jobs=1)
    assert 'feature_' not in ''.join(res['Regla'])
    assert any(r.strip().startswith('col10') for r in res['Regla'])


def test_class_leaf_no_crash():
    X = pd.DataFrame({'col0': [0, 1, 2, 3]})
    y = [0, 1, 0, 1]
    clf = RandomForestClassifier(n_estimators=1, random_state=0).fit(X, y)
    t = Trees()
    res = t.get_rangos(clf, X, n_jobs=1)
    assert isinstance(res, pd.DataFrame)


def test_percentil_parameter_affects_output():
    df, y = _build_regression_data()
    reg = RandomForestRegressor(n_estimators=1, random_state=0).fit(df, y)
    t = Trees()
    res0 = t.get_rangos(reg, df, percentil=0, n_jobs=1)
    res100 = t.get_rangos(reg, df, percentil=100, n_jobs=1)
    assert len(res0) >= len(res100)


def test_n_jobs_consistency():
    df, y = _build_regression_data()
    reg = RandomForestRegressor(n_estimators=1, random_state=0).fit(df, y)
    t = Trees()
    res_seq = t.get_rangos(reg, df, percentil=0, n_jobs=1)
    res_par = t.get_rangos(reg, df, percentil=0, n_jobs=-1)
    pd.testing.assert_frame_equal(res_seq.sort_index(axis=1), res_par.sort_index(axis=1))


def test_get_summary_optimizado_matches_get_summary():
    df, y = _build_regression_data()
    reg = RandomForestRegressor(n_estimators=1, random_state=0).fit(df, y)
    t = Trees()
    df_full_arboles = t.get_rangos(reg, df, percentil=0, n_jobs=1)
    df_full_arboles = t.get_fro(df_full_arboles)
    df_with_target = df.copy()
    df_with_target['y'] = y
    res_opt = t.get_summary_optimizado(df_with_target, df_full_arboles, 'y', n_jobs=1)
    res_orig = t.get_summary(df_with_target, df_full_arboles, 'y')
    pd.testing.assert_frame_equal(
        res_opt.sort_index(axis=1).fillna(0).reset_index(drop=True),
        res_orig.sort_index(axis=1).fillna(0).reset_index(drop=True),
        check_dtype=False,
    )


def test_run_ab_test_creates_consistent_csv(tmp_path):
    csv_path = tmp_path / "ab_results.csv"
    n_runs = 3
    results = run_ab_test(
        csv_path,
        random_state=1,
        n_runs=n_runs,
        n_samples=120,
        n_features=5,
        feature_cardinality=8,
        n_estimators=8,
        max_depth=5,
        noise_scale=1.5,
    )

    assert csv_path.exists()
    csv_results = pd.read_csv(csv_path)

    assert len(results) == n_runs + 1

    non_summary = results[results['run_id'] != 'aggregate']
    assert len(non_summary) == n_runs
    assert non_summary['matches_baseline'].all()
    assert non_summary['n_rows_equal'].all()
    assert csv_results['matches_baseline'].all()
    assert csv_results['all_runs_match'].all()
    aggregate_row = csv_results.loc[csv_results['run_id'] == 'aggregate'].iloc[0]
    assert aggregate_row['speedup_vs_baseline'] > 0
    assert 'optimized_faster_share' in csv_results.columns
    assert aggregate_row['optimized_faster_share'] > 0

    pd.testing.assert_frame_equal(
        results.sort_index(axis=1).reset_index(drop=True),
        csv_results.sort_index(axis=1).reset_index(drop=True),
        check_dtype=False,
    )


def test_get_summary_optimizado_matches_baseline_randomized(randomized_summary_inputs):
    trees, df_with_target, df_full_arboles = randomized_summary_inputs

    optimized = trees.get_summary_optimizado(
        df_with_target,
        df_full_arboles,
        "target",
        n_jobs=1,
    )
    baseline = _baseline_get_summary_optimizado(
        df_with_target,
        df_full_arboles,
        "target",
    )

    pd.testing.assert_frame_equal(
        optimized.sort_index(axis=1).reset_index(drop=True),
        baseline.sort_index(axis=1).reset_index(drop=True),
        check_dtype=False,
    )


def test_get_summary_optimizado_parallel_matches_serial(randomized_summary_inputs):
    trees, df_with_target, df_full_arboles = randomized_summary_inputs

    serial = trees.get_summary_optimizado(
        df_with_target,
        df_full_arboles,
        "target",
        n_jobs=1,
    )
    parallel = trees.get_summary_optimizado(
        df_with_target,
        df_full_arboles,
        "target",
        n_jobs=2,
    )

    pd.testing.assert_frame_equal(
        serial.sort_index(axis=1).reset_index(drop=True),
        parallel.sort_index(axis=1).reset_index(drop=True),
        check_dtype=False,
    )


def test_get_summary_optimizado_respects_branch_limit(randomized_summary_inputs):
    trees, df_with_target, df_full_arboles = randomized_summary_inputs

    limited = trees.get_summary_optimizado(
        df_with_target,
        df_full_arboles,
        "target",
        no_branch_lim=1,
        n_jobs=1,
    )
    baseline_limited = _baseline_get_summary_optimizado(
        df_with_target,
        df_full_arboles,
        "target",
        no_branch_lim=1,
    )

    assert limited["N_arbol"].nunique() <= 1

    pd.testing.assert_frame_equal(
        limited.sort_index(axis=1).reset_index(drop=True),
        baseline_limited.sort_index(axis=1).reset_index(drop=True),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "estimator_cls,target",
    [
        (RandomForestRegressor, np.array([0.0, 1.0, 1.0, 2.0])),
        (RandomForestClassifier, np.array([0, 0, 1, 1])),
    ],
)
@pytest.mark.parametrize("bootstrap", [True, False])
def test_shared_prefix_summary_matches_rule_masks(estimator_cls, target, bootstrap):
    X, _ = _build_regression_data()
    model = estimator_cls(
        n_estimators=3,
        max_depth=3,
        bootstrap=bootstrap,
        random_state=17,
    ).fit(X, target)
    trees = Trees(percentil=0, low_frac=0)
    parsed = trees.get_fro(trees.get_rangos(model, X, percentil=0))
    df = X.copy()
    df["target"] = target

    baseline = trees.get_summary_optimizado(df, parsed, "target")
    shared = trees._get_summary_from_fitted_forest(
        df,
        parsed,
        "target",
        model,
        assignment_strategy="shared_prefix",
    )

    assert assert_summary_equivalent(shared, baseline) == 0.0


def test_shared_prefix_falls_back_when_memory_budget_is_exceeded(
    randomized_summary_inputs,
):
    trees, df_with_target, parsed = randomized_summary_inputs
    X = df_with_target.drop(columns=["target"])
    y = df_with_target["target"]
    model = RandomForestRegressor(
        n_estimators=6,
        max_depth=6,
        random_state=321,
    ).fit(X, y)

    baseline = trees.get_summary_optimizado(df_with_target, parsed, "target")
    fallback = trees._get_summary_from_fitted_forest(
        df_with_target,
        parsed,
        "target",
        model,
        assignment_strategy="shared_prefix",
        max_work_bytes=0,
    )

    assert_summary_equivalent(fallback, baseline)


def test_native_apply_boundary_difference_is_detected_but_shared_prefix_matches():
    scenario = _threshold_boundary_scenario(9)
    trees = Trees(percentil=0, low_frac=0)
    parsed = trees.get_fro(
        trees.get_rangos(scenario.model, scenario.X, percentil=0)
    )
    df = scenario.X.copy()
    df["target"] = scenario.y
    baseline = trees.get_summary_optimizado(df, parsed, "target")
    native = trees._get_summary_from_fitted_forest(
        df,
        parsed,
        "target",
        scenario.model,
        assignment_strategy="native_apply",
    )
    shared = trees._get_summary_from_fitted_forest(
        df,
        parsed,
        "target",
        scenario.model,
        assignment_strategy="shared_prefix",
    )

    with pytest.raises(AssertionError):
        assert_summary_equivalent(native, baseline)
    assert_summary_equivalent(shared, baseline)


def test_shared_prefix_preserves_regions_and_labels(randomized_summary_inputs):
    trees, df, parsed = randomized_summary_inputs
    X = df.drop(columns=["target"])
    model = RandomForestRegressor(
        n_estimators=6,
        max_depth=6,
        random_state=321,
    ).fit(X, df["target"])
    baseline = trees.get_summary_optimizado(df, parsed, "target")
    shared = trees._get_summary_from_fitted_forest(
        df,
        parsed,
        "target",
        model,
        assignment_strategy="shared_prefix",
    )

    base_rectangles, base_regions, base_labels = _downstream_outputs(
        trees, baseline, df, "target"
    )
    new_rectangles, new_regions, new_labels = _downstream_outputs(
        trees, shared, df, "target"
    )
    assert len(new_rectangles) == len(base_rectangles)
    assert len(new_regions) == len(base_regions)
    for actual, expected in zip(new_rectangles + new_regions, base_rectangles + base_regions):
        pd.testing.assert_frame_equal(actual, expected)
    np.testing.assert_array_equal(new_labels, base_labels)


def test_adoption_gate_requires_equivalence_and_speed_thresholds():
    rows = []
    for speedup in [1.3, 1.4, 1.5]:
        rows.append(
            {
                "strategy": "shared_prefix",
                "speedup_vs_rule_masks": speedup,
                "summary_equal": True,
                "branch_identity_equal": True,
                "regions_equal": True,
                "labels_equal": True,
                "within_memory_budget": True,
            }
        )
    assert adoption_decision(pd.DataFrame(rows))["adopted"]
    rows[0]["labels_equal"] = False
    assert not adoption_decision(pd.DataFrame(rows))["adopted"]
