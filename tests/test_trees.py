import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from InsideForest.trees import Trees


def _build_regression_data():
    X = np.zeros((4, 11))
    X[:, 1] = [0, 1, 0, 1]
    X[:, 10] = [0, 0, 1, 1]
    y = [0, 1, 1, 2]
    return pd.DataFrame(X, columns=[f"col{i}" for i in range(11)]), y


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


def test_categorical_class_labels():
    X = pd.DataFrame({'col0': [0, 1, 2, 3]})
    y = ['a', 'b', 'a', 'b']
    clf = RandomForestClassifier(n_estimators=1, random_state=0).fit(X, y)
    t = Trees()
    res = t.get_rangos(clf, X, percentil=0, n_jobs=1)
    assert not res.empty


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
