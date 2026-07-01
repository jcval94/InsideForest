import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest

from InsideForest.descrip import (
    categorize_conditions,
    categorize_conditions_generalized,
    build_conditions_table,
    encode_features,
    _gpt_hypothesis,
    generate_hypothesis,
    generate_model_hypothesis,
    get_frontiers,
)


def test_encode_features_supports_pandas_string_dtype():
    frame = pd.DataFrame(
        {
            "cluster_desc_relative": ["a", "b"],
            "segment": pd.Series(
                ["PERCENTILE 1", "PERCENTILE 2"], dtype="string"
            ),
        }
    )

    encoded = encode_features(
        frame, {"PERCENTILE 1": 1, "PERCENTILE 2": 2}
    )

    assert encoded["segment"].tolist() == [0.0, 1.0]


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
            'Group': [1, 2],
            'Effectiveness': [0.5, 0.2],
            'Weight': [1.0, 2.0],
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
    monkeypatch.setattr(descrip, "get_openai_client", lambda api_key=None: None)
    meta_df = pd.DataFrame({"rule_token": ["tok"], "identity.label_i18n.es": ["Etiqueta"]})
    exp_df = pd.DataFrame({
        "intersection": ["tok"],
        "only_cluster_a": [""],
        "only_cluster_b": [""],
        "cluster_ef_a": [0.1],
        "cluster_ef_b": [0.2],
        "delta_ef": [0.1],
    })
    result = generate_hypothesis(meta_df, exp_df, target="tok", use_gpt=True, lang="es")
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
    result = generate_hypothesis(meta_df, exp_df, target="tok", use_gpt=False, lang="es")
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
    result = generate_hypothesis(meta_df, exp_df, target="tok", use_gpt=False, lang="es")
    assert "Etiqueta" in result
    assert "10.00%" in result


class FakeRegionEstimator:
    def __init__(self, regions, classes=None):
        self._regions = regions
        if classes is not None:
            self.classes_ = classes

    def explain_regions(self, top_n=None):
        result = self._regions.copy()
        return result if top_n is None else result.head(top_n)


def _traditional_regions():
    return pd.DataFrame(
        {
            "region_id": [10, 20, 30],
            "description": ["x <= 1", "x > 1", "x > 2"],
            "target_distribution": [
                {"no": 0.8, "yes": 0.2},
                {"no": 0.3, "yes": 0.7},
                {"no": 0.2, "yes": 0.8},
            ],
            "dominant_target": ["no", "yes", "yes"],
            "dominant_probability": [0.8, 0.7, 0.8],
            "region_score": [0.9, 0.8, 0.7],
            "support": [30, 25, 20],
        }
    )


def _multiclass_regions():
    return pd.DataFrame(
        {
            "cluster_id": [1, 2, 3],
            "description": ["a <= 1", "a > 1", "a > 2"],
            "class_distribution": [
                [0.8, 0.1, 0.1],
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7],
            ],
            "region_target_class": ["setosa", "versicolor", "virginica"],
            "target_probability": [0.8, 0.7, 0.7],
            "lift": [2.0, 1.8, 1.7],
            "entropy": [0.4, 0.6, 0.6],
            "class_margin": [0.7, 0.5, 0.5],
            "region_score": [0.95, 0.85, 0.75],
            "support": [20, 18, 16],
        }
    )


def _regression_regions():
    return pd.DataFrame(
        {
            "cluster_id": [4, 5, 6],
            "description": ["b <= 1", "b > 1", "b > 2"],
            "target_mean": [105.5, 155.0, 205.25],
            "target_median": [100.0, 150.0, 200.0],
            "target_std": [15.0, 12.0, 10.0],
            "target_iqr": [20.0, 18.0, 15.0],
            "mean_shift": [-45.0, 4.5, 54.75],
            "standardized_mean_shift": [-1.0, 0.1, 1.2],
            "dispersion_reduction": [0.1, 0.2, 0.3],
            "region_score": [0.7, 0.8, 0.9],
            "support": [30, 25, 20],
        }
    )


def test_generate_model_hypothesis_detects_traditional_model():
    result = generate_model_hypothesis(FakeRegionEstimator(_traditional_regions()))
    assert "clasificación" in result
    assert "Región 10" in result
    assert "Región 20" in result
    assert "Probabilidad dominante" in result


def test_generate_model_hypothesis_detects_multiclass_string_labels():
    estimator = FakeRegionEstimator(
        _multiclass_regions(), classes=["setosa", "versicolor", "virginica"]
    )
    result = generate_model_hypothesis(estimator)
    assert "multiclase" in result
    assert "setosa" in result
    assert "versicolor" in result
    assert "Distribución" in result


def test_generate_model_hypothesis_regression_uses_numeric_language():
    result = generate_model_hypothesis(FakeRegionEstimator(_regression_regions()))
    assert "regresión" in result
    assert "Región 4" in result
    assert "Región 6" in result
    assert "105.500" in result
    assert "probabilidad" not in result.lower()
    assert "%" not in result


def test_generate_model_hypothesis_honors_explicit_region_ids():
    result = generate_model_hypothesis(
        FakeRegionEstimator(_traditional_regions()), region_ids=[30, 20]
    )
    assert result.index("Región 30") < result.index("Región 20")
    assert "Región 10" not in result


def test_generate_model_hypothesis_requires_fitted_estimator():
    class UnfittedEstimator:
        def explain_regions(self, top_n=None):
            raise RuntimeError("not fitted")

    with pytest.raises(ValueError, match="must be fitted"):
        generate_model_hypothesis(UnfittedEstimator())


def test_generate_model_hypothesis_requires_two_regions():
    one_region = _traditional_regions().head(1)
    with pytest.raises(ValueError, match="At least two"):
        generate_model_hypothesis(FakeRegionEstimator(one_region))


class FakeOpenAIClient:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []
        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, owner):
            self.completions = FakeOpenAIClient.Completions(owner)

    class Completions:
        def __init__(self, owner):
            self.owner = owner

        def create(self, **kwargs):
            self.owner.calls.append(kwargs)
            if self.owner.error:
                raise self.owner.error
            message = type("Message", (), {"content": self.owner.content})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()


def test_generate_model_hypothesis_uses_injected_openai_client():
    client = FakeOpenAIClient("## Informe generado")
    result = generate_model_hypothesis(
        FakeRegionEstimator(_traditional_regions()), use_gpt=True, client=client
    )
    assert result == "## Informe generado"
    assert client.calls[0]["model"] == "gpt-4o-mini"


def test_generate_model_hypothesis_falls_back_when_openai_fails(caplog):
    client = FakeOpenAIClient(error=RuntimeError("offline"))
    result = generate_model_hypothesis(
        FakeRegionEstimator(_regression_regions()), use_gpt=True, client=client
    )
    assert "regresión" in result
    assert "using the local report" in caplog.text


def test_legacy_generate_hypothesis_initializes_openai_client(monkeypatch):
    from InsideForest import descrip

    client = FakeOpenAIClient("## Hipótesis API")
    calls = []

    def fake_get_client(api_key=None):
        calls.append(api_key)
        return client

    monkeypatch.setattr(descrip, "get_openai_client", fake_get_client)
    meta_df = pd.DataFrame(
        {"rule_token": ["tok"], "identity.label_i18n.es": ["Etiqueta"]}
    )
    exp_df = pd.DataFrame(
        {
            "intersection": ["tok"],
            "only_cluster_a": [""],
            "only_cluster_b": [""],
            "cluster_ef_a": [0.1],
            "cluster_ef_b": [0.2],
            "delta_ef": [0.1],
        }
    )
    result = generate_hypothesis(
        meta_df, exp_df, target="tok", use_gpt=True, api_key="test-key"
    )
    assert result == "## Hipótesis API"
    assert calls == ["test-key"]


def test_get_frontiers_basic():
    df = pd.DataFrame({'x': range(10), 'y': range(10)})
    desc_df = pd.DataFrame({
        'cluster': [0, 1],
        'cluster_description': [
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
    assert frontiers.iloc[0]['similarity'] == pytest.approx(0.5)
