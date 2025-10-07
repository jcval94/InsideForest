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
