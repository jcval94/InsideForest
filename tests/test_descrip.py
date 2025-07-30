import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from InsideForest.descrip import (
    categorize_conditions,
    categorize_conditions_generalized,
    build_conditions_table,
)


def test_categorize_conditions_basic():
    df = pd.DataFrame({'Var1': range(100), 'Var2': range(100)})
    conds = ['0 <= Var1 <= 10 and 0 <= Var2 <= 10']
    result = categorize_conditions(conds, df, n_groups=2)
    assert result == {'respuestas': ['Var1 es Percentile 50, Var2 es Percentile 50.']}


def test_categorize_conditions_three_groups():
    df = pd.DataFrame({'Var1': range(100), 'Var2': range(100)})
    conds = ['70 <= Var1 <= 90 and 10 <= Var2 <= 40']
    result = categorize_conditions(conds, df, n_groups=3)
    assert result == {'respuestas': ['Var1 es Percentile 100, Var2 es Percentile 33.33.']}


def test_categorize_conditions_invalid_inputs():
    err_df = categorize_conditions(['0<=Var1<=10'], pd.DataFrame(), n_groups=2)
    assert 'error' in err_df

    err_ngroups = categorize_conditions(['0<=Var1<=10'], pd.DataFrame({'Var1':[1]}), n_groups=1)
    assert 'error' in err_ngroups


def test_categorize_conditions_generalized_boolean():
    df = pd.DataFrame({'Var1': [True, False] * 50, 'Var2': range(100)})
    conds = ['Var1 == True and 10 <= Var2 <= 30']
    result = categorize_conditions_generalized(conds, df, n_groups=2)
    assert result == {'respuestas': ['Var1 es TRUE, Var2 es Percentile 50.']}


def test_categorize_conditions_generalized_boolean_false():
    df = pd.DataFrame({'Var1': [True, False] * 50})
    conds = ['Var1 == False']
    result = categorize_conditions_generalized(conds, df, n_groups=2)
    assert result == {'respuestas': ['Var1 es FALSE.']}


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
    assert result == {'respuestas': ['Var1 es Percentile 100.']}
