import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from InsideForest.descrip import primer_punto_inflexion_decreciente, replace_with_dict


def test_primer_punto_inflexion_decreciente_returns_value():
    data = np.array([1]*20 + [5]*60 + [10]*20)
    result = primer_punto_inflexion_decreciente(data)
    assert result is not None


def test_replace_with_dict_basic():
    df = pd.DataFrame({'A': ['foo', 'foobar', 'bar', 'cat']})
    var_map = {'foo': 'FOO', 'bar': 'BAR'}
    replaced, info = replace_with_dict(df, ['A'], var_map)
    assert replaced['A'].tolist() == ['FOO', 'FOOBAR', 'BAR', 'cat']
    assert info['A']['var_rename'] == var_map

