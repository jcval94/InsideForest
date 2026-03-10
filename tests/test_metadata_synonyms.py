import pandas as pd

from InsideForest.metadata import MetaExtractor


def test_multiword_labels_create_unambiguous_word_synonyms():
    metadata_df = pd.DataFrame(
        {
            "identity.label_i18n.es": [
                "Ingreso Mensual Neto",
                "Gasto Anual Total",
            ],
        },
        index=["ingreso_neto", "gasto_total"],
    )

    extractor = MetaExtractor(metadata_df, var_obj="ingreso_neto")

    # Full normalized string without spaces remains supported.
    assert extractor._map_to_var("ingresomensualneto") == "ingreso_neto"
    # Individual words from multi-word labels now map when unambiguous.
    assert extractor._map_to_var("mensual") == "ingreso_neto"
    assert extractor._map_to_var("anual") == "gasto_total"


def test_accent_and_punctuation_normalization_for_full_and_word_keys():
    metadata_df = pd.DataFrame(
        {
            "identity.label_i18n.es": ["Índice, de Mora!"],
        },
        index=["indice_mora"],
    )

    extractor = MetaExtractor(metadata_df, var_obj="indice_mora")

    assert extractor._map_to_var("indicedemora") == "indice_mora"
    assert extractor._map_to_var("indice") == "indice_mora"
    assert extractor._map_to_var("mora") == "indice_mora"
