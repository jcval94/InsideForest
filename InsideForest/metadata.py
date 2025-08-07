# meta_extractor.py  – FINAL
# ─────────────────────────────────────────────────────────────
from __future__ import annotations
from enum import Enum, auto
from functools import lru_cache
from typing import Dict, List, Set, Union
import pandas as pd, re, unicodedata, string

import re
import itertools
import pandas as pd

# ─────────────── fuzzy util (RapidFuzz → difflib fallback) ───────────────
try:
    from rapidfuzz import process, fuzz
    def _fuzzy_extract(q, choices):
        r = process.extractOne(q, choices, scorer=fuzz.QRatio)
        return (r[0], r[1]) if r else (None, 0)
except ImportError:
    from difflib import get_close_matches
    def _fuzzy_extract(q, choices):
        hit = (get_close_matches(q, choices, n=1, cutoff=0.65) + [None])[0]
        return (hit, 90) if hit else (None, 0)

class Profile(Enum):
    ESSENTIAL     = auto()
    INVESTIGATION = auto()
    BUSINESS      = auto()
    FULL          = auto()


# ─────────────────────────────────────────────────────────────
class MetaExtractor:
    """
    Extract a minimal, investigation or full slice of metadata for the
    variables referenced in df_QW['cluster_descripcion'].

    *Synonyms and suffix stripping are fully inferred from the metadata.*
    """

    # ───────────────────────── constructor ─────────────────────────
    def __init__(
        self, metadata_df: pd.DataFrame, var_obj: str, *,
        user_synonyms: Dict[str, str] | None = None
    ):
        self.meta = metadata_df

        self.var_obj = var_obj

        # ---------- 1) Build categorical level set ----------
        cat_label_cols = [c for c in metadata_df.columns
                          if c.startswith("domain.categorical.labels.")]
        suffixes_from_cols = {
            c.split("domain.categorical.labels.", 1)[1].lower()
            for c in cat_label_cols
        }
        suffixes_from_values: Set[str] = {
            str(v).lower()
            for c in cat_label_cols
            for v in metadata_df[c].dropna().unique()
        }
        self._cat_levels: Set[str] = suffixes_from_cols | suffixes_from_values

        # ---------- 2) Build data‑driven synonyms ----------
        self.synonyms: Dict[str, str] = {}

        def _slug(text: str) -> str:
            """lower + strip accents + remove punctuation/whitespace."""
            txt = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
            txt = txt.lower()
            return txt.translate(str.maketrans("", "", string.punctuation)).replace(" ", "")

        # (a) variable_id itself
        for var in metadata_df.index:
            self.synonyms[var.lower()] = var

        # (b) label strings & each word that uniquely maps to a single var
        word_to_vars: Dict[str, Set[str]] = {}
        label_cols = [c for c in metadata_df.columns if c.startswith("identity.label")]
        for var, row in metadata_df[label_cols].iterrows():
            for col in label_cols:
                label = row[col]
                if pd.isna(label):
                    continue
                slug = _slug(str(label))
                self.synonyms.setdefault(slug, var)  # full slug
                # individual words
                for w in slug.split():
                    word_to_vars.setdefault(w, set()).add(var)

        # keep only unambiguous single‑var words
        for w, vars_set in word_to_vars.items():
            if len(vars_set) == 1:
                self.synonyms.setdefault(w, vars_set.pop())

        # (c) allow user overrides / additions (highest priority)
        if user_synonyms:
            self.synonyms.update({k.lower(): v for k, v in user_synonyms.items()})

    # ─────────────────── columnas por perfil (igual) ───────────────────
    def _cols(self, profile: Profile) -> List[str]:
        m = self.meta
        c      = lambda *n: [x for x in n if x in m.columns]
        starts = lambda p: [x for x in m.columns if x.startswith(p)]

        id_cols   = c("identity.variable_id",
                      "identity.label_i18n.es",  "identity.description_i18n.es",
                      "identity.label_i18n.en",  "identity.description_i18n.en")
        type_cols = c("type.logical_type", "type.measurement_scale")
        dom_num   = c("domain.numeric.min", "domain.numeric.max")
        dom_cat   = c("domain.categorical.codes") or starts("domain.categorical.labels.")
        stats_b   = c("statistics.n_total", "statistics.n_non_null",
                      "statistics.missing_ratio", "statistics.n_distinct")
        stats_add = c("statistics.numeric_summary.mean",
                      "statistics.numeric_summary.std",
                      "statistics.numeric_summary.p50")
        action    = starts("actionability.")

        if profile is Profile.ESSENTIAL:
            return id_cols + type_cols + dom_num + dom_cat + stats_b + action
        if profile is Profile.INVESTIGATION:
            return self._cols(Profile.ESSENTIAL) + stats_add
        if profile is Profile.BUSINESS:
            extras = c("identity.tags", "domain.allowed_nulls_pct", "domain.unique",
                       "actionability.side_effects")
            return self._cols(Profile.INVESTIGATION) + extras
        return list(m.columns)        # FULL

    # ───────────────────────── helpers ─────────────────────────
    @staticmethod
    def _tokens(series: pd.Series) -> Set[str]:
        return set(re.findall(r"[a-z]+__[A-Za-z0-9_]+",
                              " ".join(series.dropna().astype(str))))
    
    @staticmethod
    def _extract_tokens(series: pd.Series) -> Set[str]:
        """
        Extrae de una Serie de pandas todos los nombres de variable que aparezcan:
        • En comparaciones con operadores (<=, >=, <, >, ==, !=), incluyendo
            números en notación científica y con guiones bajos como separadores
            (p. ej. 1_000_000, 3.2e-4).
        • En tokens con formato prefijo__nombre (p. ej. cat__score).

        Devuelve un conjunto único de nombres de variable.
        """
        # 1) convertimos la serie a un solo string
        text = " ".join(series.dropna().astype(str))

        # 2) patrones inline
        var_re = r"[A-Za-z_][A-Za-z0-9_\.\[\]]*"       # variables Python-like
        num_re = r"""
            [-+]?                                     # signo opcional
            \d[\d_]*                                  # parte entera (1 ó 1_000_000)
            (?:\.\d[\d_]*)?                           # parte decimal opcional
            (?:[eE][-+]?\d+)?                         # notación científica opcional
        """
        op_re  = r"(?:<=|>=|<|>|==|!=)"                # operadores permitidos

        # 3) regex para comparaciones “izquierda OP derecha”
        compare_re = re.compile(
            fr"(?P<left>{var_re}|{num_re})\s*{op_re}\s*(?P<right>{var_re}|{num_re})",
            re.VERBOSE
        )

        # 4) regex para tokens prefijo__nombre
        dbl_und_re = re.compile(r"[a-z]+__[A-Za-z0-9_]+")

        # 5) recogemos tokens __ y variables en comparaciones
        tokens: Set[str] = set(dbl_und_re.findall(text))
        for m in compare_re.finditer(text):
            for side in ("left", "right"):
                lex = m.group(side)
                # si lex es variable (no numérico), la añadimos
                if re.fullmatch(var_re, lex) and not re.fullmatch(num_re, lex):
                    tokens.add(lex)

        return tokens
    
    def _canon(self, token: str) -> str:
        """Remove logical prefix + categorical suffix (if present)."""
        stem = re.sub(r"^(?:cat|num|bool)__", "", token)
        parts = stem.split("_")
        if len(parts) > 1 and parts[-1].lower() in self._cat_levels:
            stem = "_".join(parts[:-1])
        return stem

    @lru_cache(maxsize=None)
    def _map_to_var(self, canon: str) -> str | None:
        canon_lc = canon.lower()
        # 1) direct synonym
        if canon_lc in self.synonyms:
            return self.synonyms[canon_lc]
        # 2) variable_id exact
        if canon_lc in {v.lower() for v in self.meta.index}:
            return canon
        # 3) fuzzy against variable ids
        hit, score = _fuzzy_extract(canon_lc, [v.lower() for v in self.meta.index])
        return hit if hit and score >= 65 else None

    # ───────────────────────── API pública ─────────────────────────
    def extract(
        self,
        df_QW: pd.DataFrame,
        *,
        profile: Profile = Profile.ESSENTIAL,
        cols: List[str] | None = None
    ) -> pd.DataFrame:

        if "cluster_descripcion" not in df_QW.columns:
            raise KeyError("df_QW must contain 'cluster_descripcion' column")

        # tokens = self._tokens(df_QW["cluster_descripcion"])
        tokens = self._extract_tokens(df_QW["cluster_descripcion"])
        mapping = {t: self._map_to_var(self._canon(t)) for t in tokens}
        valid   = {tok: var for tok, var in mapping.items() if var in list(self.meta.index)}

        if not valid:
            return pd.DataFrame()

        want_cols = cols or self._cols(profile)
        keep_cols = [c for c in want_cols if c in self.meta.columns]

        mini = self.meta.loc[list(valid.values()), keep_cols].copy()
        mini.insert(0, "rule_token", list(valid.keys()))
        mini.reset_index(names="metadata_row", inplace=True)
        try:
            metadata_OBJ = self.meta.loc[[self.var_obj]]
        except:
            metadata_OBJ = self.meta.loc[[self.var_obj.upper()]]

        metadata_OBJ['metadata_row'] = self.var_obj
        metadata_OBJ['rule_token'] = self.var_obj

        return pd.concat([mini, metadata_OBJ[mini.columns]], axis=0)

# ------------------------------------------------------------------ #
# 1. UTILITARIOS DE PARSEO
# ------------------------------------------------------------------ #
def parse_rule_string(rule_str: str) -> list[str]:
    """Devuelve lista de condiciones limpias quitando AND y espacios extra."""
    if pd.isna(rule_str):
        return []
    # Normaliza espacios y quita paréntesis (si hubiera)
    parts = [re.sub(r'\s+', ' ', p.strip('() ')) for p in rule_str.split('AND')]
    return [p for p in parts if p]                 # sin strings vacíos

def token_from_condition(cond: str) -> str | None:
    """
    Extrae el token de la variable dentro de la condición.
    Ej.: '-3.2 <= num__age <= 1.5' → 'num__age'
    """
    # busca la palabra con '__' (primera preferencia)
    for word in cond.split():
        if '__' in word:
            return word
    # respaldo: primera palabra alfabética
    for word in cond.split():
        if re.search(r'[A-Za-z]', word):
            return word
    return None

def conditions_to_tokens(conds: list[str]) -> set[str]:
    return {token_from_condition(c) for c in conds if token_from_condition(c)}

# ------------------------------------------------------------------ #
# 2. GENERADOR DE EXPERIMENTOS PARA UN SOLO Df2
# ------------------------------------------------------------------ #

def experiments_from_df2(df2: pd.DataFrame,
                         meta: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve UNA fila por par de clústeres con:
      · variables_a : lista de tokens exclusivos del clúster menos efectivo
      · variables_b : lista de tokens exclusivos del clúster más efectivo
      · difficulty_a : máx(actionability.increase_difficulty) en variables_a
      · difficulty_b : máx(actionability.decrease_difficulty) en variables_b
      · n_intersection, n_only_a, n_only_b (conteos)
      · score (penaliza difficulty_a, exclusivas y premia intersección)
    """
    # --- tabla de acción -----------------------------
    meta_idx = meta.set_index('rule_token')

    recs = []
    for (_, row_a), (_, row_b) in itertools.combinations(df2.iterrows(), 2):

        conds_a = set(parse_rule_string(row_a['cluster_descripcion']))
        conds_b = set(parse_rule_string(row_b['cluster_descripcion']))

        inters     = sorted(conds_a & conds_b)
        only_a     = sorted(conds_a - conds_b)
        only_b     = sorted(conds_b - conds_a)

        # Determinar qué clúster es el de menor efectividad
        delta_ef   = row_a['cluster_ef_sample'] - row_b['cluster_ef_sample']
        row_a_subset, row_b_subset = (row_a, row_b) if delta_ef < 0 else (row_b, row_a)
        only_subset_a, only_subset_b = (only_a, only_b) if delta_ef < 0 else (only_b, only_a)

        # -------------------- listas de tokens exclusivas ------------------
        def to_tokens(cond_list):
            return sorted({token_from_condition(c) for c in cond_list
                           if token_from_condition(c) is not None})

        vars_a = to_tokens(only_subset_a)
        vars_b = to_tokens(only_subset_b)

        # -------------------- máximos de dificultad ------------------------
        def max_difficulty(tokens, col):
            vals = [meta_idx.at[t, col] for t in tokens
                    if t in meta_idx.index and pd.notna(meta_idx.at[t, col])]
            return max(vals) if vals else 10        # castigo alto si faltan datos

        difficulty_a = max_difficulty(vars_a, 'actionability.increase_difficulty')
        difficulty_b = max_difficulty(vars_b, 'actionability.decrease_difficulty')

        # -------------------- score --------------------
        n_tot   = (row_a_subset.get('cluster_n_sample', 0) or 0) + \
                  (row_b_subset.get('cluster_n_sample', 0) or 0)
        weight  = max(n_tot ** 0.5, 1)

        n_inter = len(inters)
        n_only_a = len(vars_a)
        n_only_b = len(vars_b)

        score = (abs(delta_ef) * weight * (1 + n_inter) /
                 (difficulty_a + 0.5) /
                 (1 + n_only_a + n_only_b))

        recs.append({
            'cluster_a'          : int(row_a_subset['cluster']),
            'cluster_b'          : int(row_b_subset['cluster']),
            'cluster_ef_a'       : row_a['cluster_ef_sample'],
            'cluster_ef_b'       : row_b['cluster_ef_sample'],
            'delta_ef'           : abs(delta_ef),
            'avg_n'              : n_tot / 2,
            'variables_a'        : vars_a,
            'variables_b'        : vars_b,
            'difficulty_a'       : difficulty_a,
            'difficulty_b'       : difficulty_b,
            'n_intersection'     : n_inter,
            'n_only_a'           : n_only_a,
            'n_only_b'           : n_only_b,
            'intersection'       : inters,
            'only_cluster_a'     : only_subset_a,
            'only_cluster_b'     : only_subset_b,
            'score'              : score,
        })

    return pd.DataFrame.from_records(recs)



# ------------------------------------------------------------------ #
# 3. PIPELINE GENERAL PARA «n» Df2
# ------------------------------------------------------------------ #
def run_experiments(mx, df2_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Genera y consolida hipótesis para un diccionario de Df2.
    """
    all_hypotheses = []

    for name, df2 in df2_dict.items():
        df1  = mx.extract(df2)
        hypo = experiments_from_df2(df2, df1)

        if not hypo.empty:
            hypo['dataset'] = name
            hypo['variables_intersection'] = [
                sorted({token_from_condition(c) for c in conds
                        if token_from_condition(c) is not None})
                for conds in hypo['intersection']
            ]
            all_hypotheses.append(hypo)

    if not all_hypotheses:
        cols = ['dataset', 'cluster_a', 'cluster_b',
                'cluster_ef_a', 'cluster_ef_b', 'delta_ef', 'avg_n',
                'variables_a', 'variables_b', 'variables_intersection',
                'difficulty_a', 'difficulty_b', 'n_intersection',
                'n_only_a', 'n_only_b', 'intersection',
                'only_cluster_a', 'only_cluster_b', 'score']
        return pd.DataFrame(columns=cols)

    return (pd.concat(all_hypotheses, ignore_index=True)
              .sort_values('score', ascending=False)
              .reset_index(drop=True))
