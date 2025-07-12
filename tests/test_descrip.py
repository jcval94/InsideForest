import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from InsideForest.descrip import categorize_conditions


def test_categorize_conditions_basic():
    df = pd.DataFrame({'Var1': range(100), 'Var2': range(100)})
    conds = ['0 <= Var1 <= 10 and 0 <= Var2 <= 10']
    result = categorize_conditions(conds, df, n_groups=2)
    assert result == {'respuestas': ['Var1 es BAJO, Var2 es BAJO.']}


def test_categorize_conditions_three_groups():
    df = pd.DataFrame({'Var1': range(100), 'Var2': range(100)})
    conds = ['70 <= Var1 <= 90 and 10 <= Var2 <= 40']
    result = categorize_conditions(conds, df, n_groups=3)
    assert result == {'respuestas': ['Var1 es ALTO, Var2 es BAJO.']}


def test_categorize_conditions_invalid_inputs():
    err_df = categorize_conditions(['0<=Var1<=10'], pd.DataFrame(), n_groups=2)
    assert 'error' in err_df

    err_ngroups = categorize_conditions(['0<=Var1<=10'], pd.DataFrame({'Var1':[1]}), n_groups=1)
    assert 'error' in err_ngroups
