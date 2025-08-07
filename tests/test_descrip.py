import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest

from InsideForest.descrip import (
    categorize_conditions,
    categorize_conditions_generalized,
    build_conditions_table,
    _gpt_hypothesis,
    generar_hypotesis,
    get_frontiers,
)


def test_categorize_conditions_basic():
    df = pd.DataFrame({'Var1': range(100), 'Var2': range(100)})
    conds = ['0 <= Var1 <= 10 and 0 <= Var2 <= 10']
    result = categorize_conditions(conds, df, n_groups=2)
    assert result == {'responses': ['Var1 = Percentile 50, Var2 = Percentile 50.']}


def test_categorize_conditions_three_groups():
    df = pd.DataFrame({'Var1': range(100), 'Var2': range(100)})
    conds = ['70 <= Var1 <= 90 and 10 <= Var2 <= 40']
    result = categorize_conditions(conds, df, n_groups=3)
    assert result == {'responses': ['Var1 = Percentile 100, Var2 = Percentile 33.33.']}


def test_categorize_conditions_invalid_inputs():
    err_df = categorize_conditions(['0<=Var1<=10'], pd.DataFrame(), n_groups=2)
    assert 'error' in err_df

    err_ngroups = categorize_conditions(['0<=Var1<=10'], pd.DataFrame({'Var1':[1]}), n_groups=1)
    assert 'error' in err_ngroups


def test_categorize_conditions_generalized_boolean():
    df = pd.DataFrame({'Var1': [True, False] * 50, 'Var2': range(100)})
    conds = ['Var1 == True and 10 <= Var2 <= 30']
    result = categorize_conditions_generalized(conds, df, n_groups=2)
    assert result == {'responses': ['Var1 = TRUE, Var2 = Percentile 50.']}


def test_categorize_conditions_generalized_boolean_false():
    df = pd.DataFrame({'Var1': [True, False] * 50})
    conds = ['Var1 == False']
    result = categorize_conditions_generalized(conds, df, n_groups=2)
    assert result == {'responses': ['Var1 = FALSE.']}


def test_build_conditions_table_basic():
    df = pd.DataFrame({'Var1': range(100), 'Var2': range(100)})
    conds = [
        '0 <= Var1 <= 10 and 0 <= Var2 <= 10',
        '70 <= Var1 <= 90 and 10 <= Var2 <= 40',
    ]
    efectiv = [0.5, 0.2]
    ponder = [1.0, 2.0]
    table = build_conditions_table(conds, df, efectiv, ponder, n_groups=3)
    expected = pd.DataFrame(
        {
            'Grupo': [1, 2],
            'Efectividad': [0.5, 0.2],
            'Ponderador': [1.0, 2.0],
            'Var1': ['Percentile 33.33', 'Percentile 100'],
            'Var2': ['Percentile 33.33', 'Percentile 33.33'],
        }
    )
    pd.testing.assert_frame_equal(table, expected)


def test_categorize_conditions_equal_percentiles():
    df = pd.DataFrame({'Var1': [5] * 100})
    conds = ['5 <= Var1 <= 5']
    result = categorize_conditions(conds, df, n_groups=3)
    assert result == {'responses': ['Var1 = Percentile 100.']}


def test_gpt_hypothesis_returns_none_without_client(monkeypatch):
    from InsideForest import descrip
    monkeypatch.setattr(descrip, "_client", None)
    assert _gpt_hypothesis({}, model="noop", temperature=0.0) is None


def test_gpt_hypothesis_handles_empty_choices(monkeypatch):
    class DummyClientNoChoices:
        class Chat:
            class Completions:
                def create(self, **kwargs):
                    return type("Rsp", (), {"choices": []})()

            completions = Completions()

        chat = Chat()

    from InsideForest import descrip
    monkeypatch.setattr(descrip, "_client", DummyClientNoChoices())
    assert _gpt_hypothesis({}, model="noop", temperature=0.0) is None


def test_generar_hypotesis_fallback_without_gpt(monkeypatch):
    from InsideForest import descrip
    monkeypatch.setattr(descrip, "_client", None)
    meta_df = pd.DataFrame({"rule_token": ["tok"], "identity.label_i18n.es": ["Etiqueta"]})
    exp_df = pd.DataFrame({
        "intersection": ["tok"],
        "only_cluster_a": [""],
        "only_cluster_b": [""],
        "cluster_ef_a": [0.1],
        "cluster_ef_b": [0.2],
        "delta_ef": [0.1],
    })
    result = generar_hypotesis(meta_df, exp_df, target="tok", use_gpt=True, lang="es")
    assert "Etiqueta" in result


def test_generar_hypotesis_handles_missing_subgroups(monkeypatch):
    """Ensure missing subgroup columns don't raise KeyError."""
    from InsideForest import descrip

    monkeypatch.setattr(descrip, "_client", None)
    meta_df = pd.DataFrame({"rule_token": ["tok"], "identity.label_i18n.es": ["Etiqueta"]})
    exp_df = pd.DataFrame({
        "intersection": ["tok"],
        # omit only_cluster_a and only_cluster_b
        "cluster_ef_a": [0.1],
        "cluster_ef_b": [0.2],
        "delta_ef": [0.1],
    })
    result = generar_hypotesis(meta_df, exp_df, target="tok", use_gpt=False, lang="es")
    assert "Etiqueta" in result


def test_generar_hypotesis_handles_lowercase_columns(monkeypatch):
    """Subgroup columns provided in lowercase are handled correctly."""
    from InsideForest import descrip

    monkeypatch.setattr(descrip, "_client", None)
    meta_df = pd.DataFrame({"rule_token": ["tok"], "identity.label_i18n.es": ["Etiqueta"]})
    exp_df = pd.DataFrame({
        "intersection": ["tok"],
        "only_cluster_a": [""],
        "only_cluster_b": [""],
        "cluster_ef_a": [0.1],
        "cluster_ef_b": [0.2],
        "delta_ef": [0.1],
    })
    result = generar_hypotesis(meta_df, exp_df, target="tok", use_gpt=False, lang="es")
    assert "Etiqueta" in result
    assert "10.00%" in result


def test_get_frontiers_basic():
    df = pd.DataFrame({'x': range(10), 'y': range(10)})
    desc_df = pd.DataFrame({
        'cluster': [0, 1],
        'cluster_descripcion': [
            '0 <= x <= 4 and 0 <= y <= 4',
            '5 <= x <= 9 and 0 <= y <= 4',
        ],
        'cluster_n_sample': [100, 100],
        'cluster_ef_sample': [0.1, 0.2],
    })

    df_explain, frontiers = get_frontiers(desc_df.copy(), df)

    assert 'cluster_desc_relative' in df_explain.columns
    assert df_explain.loc[df_explain['cluster'] == 0, 'x'].iat[0] == 'PERCENTILE 40'
    assert frontiers.iloc[0]['cluster_1'] == 0
    assert frontiers.iloc[0]['cluster_2'] == 1
    assert frontiers.iloc[0]['similitud'] == pytest.approx(0.5)
