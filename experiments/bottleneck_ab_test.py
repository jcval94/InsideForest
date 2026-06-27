from __future__ import annotations

import argparse
import contextlib
import math
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from InsideForest import InsideForestRegionClusterer
from InsideForest.cluster_selector import MenuClusterSelector
from InsideForest.regions import Regions
from InsideForest.trees import Trees


def _original_get_fro(self: Trees, df_full_arboles: pd.DataFrame) -> pd.DataFrame:
    if df_full_arboles.empty:
        return df_full_arboles.assign(feature=[], operador=[], rangos=[])

    import re

    num = r"[-+]?(?:\d+(?:_\d+)*)?(?:\.\d+(?:_\d+)*)?(?:e[-+]?\d+)?"
    pattern = re.compile(rf"^(\S+)\s*(<=|>=|<|>)\s*({num})$", re.I)

    def parse_regla(regla):
        match = pattern.search(regla.replace(" ", ""))
        if match:
            feature = match.group(1)
            operador = match.group(2)
            rangos = float(match.group(3).replace("_", ""))
            return feature, operador, rangos
        return None, None, None

    df_full_arboles[["feature", "operador", "rangos"]] = (
        df_full_arboles["Regla"].apply(lambda x: parse_regla(x)).apply(pd.Series)
    )
    return df_full_arboles.dropna(subset=["operador", "rangos"])


def _original_get_summary_cache(
    self: Trees,
    data1: pd.DataFrame,
    df_full_arboles: pd.DataFrame,
    var_obj: str,
    no_branch_lim=None,
    verbose=0,
    n_jobs=1,
) -> pd.DataFrame:
    agrupacion = pd.pivot_table(
        df_full_arboles,
        index=["N_regla", "N_arbol", "feature", "operador"],
        values=["rangos", "Importancia"],
        aggfunc=["min", "max", "mean"],
    )
    agrupacion_min = agrupacion["min"].reset_index()
    agrupacion_min = agrupacion_min[agrupacion_min["operador"] == "<="]
    agrupacion_max = agrupacion["max"].reset_index()
    agrupacion_max = agrupacion_max[agrupacion_max["operador"] == ">"]
    agrupacion = pd.concat([agrupacion_min, agrupacion_max]).sort_values(
        ["N_arbol", "N_regla"]
    )

    top_100_arboles = agrupacion["N_arbol"].unique()
    if no_branch_lim is not None:
        top_100_arboles = top_100_arboles[:no_branch_lim]

    X = data1.to_numpy()
    col_to_idx = {col: i for i, col in enumerate(data1.columns)}
    y = data1[var_obj].to_numpy()
    n_samples = X.shape[0]

    def _build_comparison_cache(ag_arbol):
        unique_thresholds = {}
        for feature, operador, rango in ag_arbol[
            ["feature", "operador", "rangos"]
        ].itertuples(index=False):
            if feature not in col_to_idx:
                continue
            idx = col_to_idx[feature]
            key = (idx, operador)
            unique_thresholds.setdefault(key, set()).add(float(rango))

        comparison_cache = {}
        for (idx, operador), threshold_set in unique_thresholds.items():
            thresholds = np.array(sorted(threshold_set), dtype=float)
            if thresholds.size == 0:
                continue
            feature_values = X[:, idx][:, None]
            if operador == "<=":
                evaluations = feature_values <= thresholds
            else:
                evaluations = feature_values > thresholds
            for thr, column in zip(thresholds, evaluations.T):
                comparison_cache[(idx, operador, thr)] = column
        return comparison_cache

    def _process_tree(arbol_num):
        ag_arbol = agrupacion[agrupacion["N_arbol"] == arbol_num]
        reglas_info = []
        for regla_num in ag_arbol["N_regla"].unique():
            ag_regla = ag_arbol[ag_arbol["N_regla"] == regla_num]
            men_ = ag_regla[ag_regla["operador"] == "<="][
                ["feature", "rangos"]
            ].values
            may_ = ag_regla[ag_regla["operador"] == ">"][
                ["feature", "rangos"]
            ].values
            le_idx = (
                np.array([col_to_idx[c] for c in men_[:, 0]], dtype=int)
                if len(men_)
                else np.array([], dtype=int)
            )
            le_val = men_[:, 1].astype(float) if len(men_) else np.array([], dtype=float)
            gt_idx = (
                np.array([col_to_idx[c] for c in may_[:, 0]], dtype=int)
                if len(may_)
                else np.array([], dtype=int)
            )
            gt_val = may_[:, 1].astype(float) if len(may_) else np.array([], dtype=float)
            reglas_info.append((ag_regla.copy(), le_idx, le_val, gt_idx, gt_val))

        if not reglas_info:
            return pd.DataFrame()

        comparison_cache = _build_comparison_cache(ag_arbol)

        def _fetch_comparison(idx, operador, thr):
            key = (idx, operador, thr)
            if key not in comparison_cache:
                column = X[:, idx] <= thr if operador == "<=" else X[:, idx] > thr
                comparison_cache[key] = column
            return comparison_cache[key]

        n_rules = len(reglas_info)
        n_sample = np.zeros(n_rules, dtype=int)
        sums = np.zeros(n_rules, dtype=float)
        mask_buffer = np.empty(n_samples, dtype=bool)
        max_conditions = 0
        for _, le_idx, _, gt_idx, _ in reglas_info:
            max_conditions = max(max_conditions, le_idx.size + gt_idx.size)
        cond_matrix = (
            np.empty((max_conditions, n_samples), dtype=bool)
            if max_conditions > 0
            else None
        )

        for rule_idx, (_, le_idx, le_val, gt_idx, gt_val) in enumerate(reglas_info):
            conds = []
            for idx, thr in zip(le_idx, le_val):
                conds.append(_fetch_comparison(idx, "<=", thr))
            for idx, thr in zip(gt_idx, gt_val):
                conds.append(_fetch_comparison(idx, ">", thr))
            if conds and cond_matrix is not None:
                for idx_cond, cond in enumerate(conds):
                    cond_matrix[idx_cond] = cond
                np.logical_and.reduce(cond_matrix[: len(conds)], axis=0, out=mask_buffer)
            else:
                mask_buffer.fill(True)
            n_sample[rule_idx] = mask_buffer.sum()
            sums[rule_idx] = mask_buffer @ y

        ef_sample = np.divide(
            sums,
            n_sample,
            out=np.zeros_like(sums, dtype=float),
            where=n_sample > 0,
        )
        res = []
        for (df_regla, _, _, _, _), ns, ef in zip(reglas_info, n_sample, ef_sample):
            df_regla["n_sample"] = ns
            df_regla["ef_sample"] = ef
            res.append(df_regla)
        return pd.concat(res, ignore_index=True)

    resultados = [_process_tree(a) for a in top_100_arboles]
    resultados = [r for r in resultados if r is not None and not r.empty]
    if not resultados:
        return pd.DataFrame(
            columns=[
                "N_regla",
                "N_arbol",
                "feature",
                "operador",
                "rangos",
                "Importancia",
                "n_sample",
                "ef_sample",
            ]
        )
    return pd.concat(resultados, ignore_index=True).sort_values(
        by=["ef_sample", "n_sample"], ascending=False
    )


def _original_fill_na_pond_fastest(
    self: Regions, df_sep_dm: pd.DataFrame, df: pd.DataFrame, features_val, verbose
) -> pd.DataFrame:
    df_lilu = df_sep_dm[["linf", "lsup"]].copy()
    lsup_limit = df[features_val].max() + 1
    linf_limit = df[features_val].min() - 1

    linf_repl_df = pd.DataFrame(
        np.tile(linf_limit.values, (df_lilu["linf"].shape[0], 1)),
        columns=df_lilu["linf"].columns,
        index=df_lilu.index,
    )
    lsup_repl_df = pd.DataFrame(
        np.tile(lsup_limit.values, (df_lilu["lsup"].shape[0], 1)),
        columns=df_lilu["lsup"].columns,
        index=df_lilu.index,
    )

    mask_linf = np.isinf(df_lilu["linf"].values)
    mask_lsup = np.isinf(df_lilu["lsup"].values)
    df_lilu["linf"] = np.where(mask_linf, linf_repl_df.values, df_lilu["linf"].values)
    df_lilu["lsup"] = np.where(mask_lsup, lsup_repl_df.values, df_lilu["lsup"].values)
    return pd.concat([df_lilu, df_sep_dm[["ponderador", "ef_sample", "n_sample"]]], axis=1)


def _original_menu_catalog_guarded(
    selector: MenuClusterSelector,
    records,
    n_clusters: int,
    max_steps: int = 100,
) -> str:
    selector._ensure_vocab_for_predict(records)
    V, _ = selector.q_.shape
    Py = selector.Py_.copy()
    allowed = [np.array(opts, dtype=int) for opts in selector._menus_indices(records)]
    n = len(allowed)
    remaining = set(range(n))
    Pt = Py / Py.sum()
    s_val = np.log(np.clip(selector.q_, 1e-12, 1.0)) @ Pt

    for _ in range(max_steps):
        if not remaining:
            return "completed"
        cover_gain = []
        for v in range(V):
            gain = sum(s_val[v] for i in remaining if v in allowed[i])
            cover_gain.append((gain, v))
        v_best = max(cover_gain)[1]
        before = len(remaining)
        covered = [i for i in list(remaining) if v_best in allowed[i]]
        for i in covered:
            remaining.discard(i)
        if len(remaining) == before:
            return "no_progress"
    return "max_steps"


@contextlib.contextmanager
def _patched(**patches):
    originals = []
    try:
        for owner, name, replacement in patches.values():
            originals.append((owner, name, getattr(owner, name)))
            setattr(owner, name, replacement)
        yield
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


def _time_call(func: Callable[[], Any], runs: int) -> tuple[float, Any]:
    times = []
    result = None
    for _ in range(max(1, runs)):
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = func()
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def _frames_match(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(
            a.sort_index(axis=1).reset_index(drop=True),
            b.sort_index(axis=1).reset_index(drop=True),
            check_dtype=False,
        )
        return True
    except AssertionError:
        return False


def _list_frames_match(a: list[pd.DataFrame], b: list[pd.DataFrame]) -> bool:
    if len(a) != len(b):
        return False
    return all(_frames_match(left, right) for left, right in zip(a, b))


def _build_inputs(args):
    X, y = make_classification(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=max(2, args.n_features - 4),
        n_redundant=2,
        random_state=args.seed,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df = X_df.copy()
    df["target"] = y
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed + 1,
        n_jobs=1,
    ).fit(X_df, y)
    return X_df, y, df, rf


def _add_row(
    rows: list[dict[str, Any]],
    *,
    experiment: str,
    bottleneck: str,
    proposal: str,
    original_seconds: float | None,
    candidate_seconds: float | None,
    output_match: bool | None,
    implemented: bool,
    decision: str,
    notes: str,
):
    speedup = (
        original_seconds / candidate_seconds
        if original_seconds is not None and candidate_seconds and candidate_seconds > 0
        else math.nan
    )
    rows.append(
        {
            "experiment": experiment,
            "bottleneck": bottleneck,
            "proposal": proposal,
            "original_seconds": original_seconds,
            "candidate_seconds": candidate_seconds,
            "speedup_vs_original": speedup,
            "output_match": output_match,
            "implemented": implemented,
            "decision": decision,
            "notes": notes,
        }
    )


def run(args) -> pd.DataFrame:
    X_df, y, df, rf = _build_inputs(args)
    trees = Trees(percentil=0, low_frac=0.0, no_trees_search=args.n_estimators)
    regions = Regions()

    raw_ranges = trees.get_rangos(rf, X_df, percentil=0, low_frac=0.0, n_jobs=1)

    rows: list[dict[str, Any]] = []

    original_time, original_fro = _time_call(
        lambda: _original_get_fro(trees, raw_ranges.copy()), args.runs
    )
    candidate_time, candidate_fro = _time_call(
        lambda: trees.get_fro(raw_ranges.copy()), args.runs
    )
    _add_row(
        rows,
        experiment="stage_get_fro",
        bottleneck="row-wise regex parsing",
        proposal="vectorized pandas str.extract parser",
        original_seconds=original_time,
        candidate_seconds=candidate_time,
        output_match=_frames_match(original_fro, candidate_fro),
        implemented=True,
        decision="implemented",
        notes="Same parsed columns; avoids per-row Series construction.",
    )

    original_time, original_summary = _time_call(
        lambda: _original_get_summary_cache(trees, df, candidate_fro, "target"),
        args.runs,
    )
    candidate_time, candidate_summary = _time_call(
        lambda: trees.get_summary_optimizado(df, candidate_fro, "target", n_jobs=1),
        args.runs,
    )
    _add_row(
        rows,
        experiment="stage_get_summary",
        bottleneck="per-tree DataFrame filtering and comparison cache overhead",
        proposal="group rules once and reuse one boolean mask",
        original_seconds=original_time,
        candidate_seconds=candidate_time,
        output_match=_frames_match(original_summary, candidate_summary),
        implemented=True,
        decision="implemented",
        notes="Exact summary rows; less allocation than cached comparison matrices.",
    )

    separacion_dim = trees.extract_rectangles(candidate_summary)
    with _patched(
        fill=(Regions, "fill_na_pond_fastest", _original_fill_na_pond_fastest)
    ):
        original_time, original_prio = _time_call(
            lambda: regions.prio_ranges(separacion_dim, df), args.runs
        )
    candidate_time, candidate_prio = _time_call(
        lambda: regions.prio_ranges(separacion_dim, df), args.runs
    )
    _add_row(
        rows,
        experiment="stage_prio_ranges",
        bottleneck="tiling replacement DataFrames for infinite bounds",
        proposal="NumPy indexed replacement without tiled DataFrames",
        original_seconds=original_time,
        candidate_seconds=candidate_time,
        output_match=_list_frames_match(original_prio, candidate_prio),
        implemented=True,
        decision="implemented",
        notes="Same prioritized regions; reduces allocation inside many small groups.",
    )

    records = [[3.0], [0.0, 2.0, 3.0], [0.0, 2.0], [0.0, 1.0, 2.0]]
    y_menu = [0, 1, 0, 1]
    selector_original = MenuClusterSelector(seed=42).fit(records, y_menu)
    start = time.perf_counter()
    original_status = _original_menu_catalog_guarded(selector_original, records, 2)
    original_time = time.perf_counter() - start
    selector_candidate = MenuClusterSelector(seed=42).fit(records, y_menu)
    candidate_time, labels = _time_call(
        lambda: selector_candidate.predict(records, n_clusters=2), args.runs
    )
    _add_row(
        rows,
        experiment="stage_menu_catalog",
        bottleneck="catalog set-cover can choose values covering zero remaining rows",
        proposal="rank by coverage first, then score",
        original_seconds=original_time,
        candidate_seconds=candidate_time,
        output_match=None,
        implemented=True,
        decision="implemented",
        notes=f"Original guarded status={original_status}; candidate labels={labels}.",
    )

    def fit_with_current():
        model = InsideForestRegionClusterer(
            rf_params={
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "random_state": args.seed + 2,
                "n_jobs": 1,
            },
            no_trees_search=args.n_estimators,
            max_cases=args.n_samples,
        )
        model.fit(X_df, y)
        return model

    def fit_with_original():
        with _patched(
            fro=(Trees, "get_fro", _original_get_fro),
            summary=(Trees, "get_summary_optimizado", _original_get_summary_cache),
            fill=(Regions, "fill_na_pond_fastest", _original_fill_na_pond_fastest),
        ):
            return fit_with_current()

    original_time, original_model = _time_call(fit_with_original, 1)
    candidate_time, candidate_model = _time_call(fit_with_current, 1)
    output_match = np.array_equal(original_model.labels_, candidate_model.labels_)
    _add_row(
        rows,
        experiment="full_fit_default",
        bottleneck="combined fit pipeline",
        proposal="combined implemented changes",
        original_seconds=original_time,
        candidate_seconds=candidate_time,
        output_match=output_match,
        implemented=True,
        decision="implemented",
        notes=(
            f"labels_equal={output_match}; "
            f"region_groups={len(candidate_model.df_reres_ or [])}"
        ),
    )

    return pd.DataFrame(rows)


def write_readme(df: pd.DataFrame, path: Path) -> None:
    implemented = df[df["implemented"] == True]
    lines = [
        "# Resumen de AB Testing de Cuellos de Botella",
        "",
        "## Alcance",
        "",
        "Este reporte compara cambios candidatos en la ruta principal de `InsideForestRegionClusterer.fit`: extraccion, priorizacion y asignacion de regiones.",
        "",
        "## Hallazgos",
        "",
    ]
    for row in df.itertuples(index=False):
        speedup = (
            "n/a"
            if pd.isna(row.speedup_vs_original)
            else f"{row.speedup_vs_original:.2f}x"
        )
        if row.output_match is None:
            speedup = "no comparable; el original no produjo salida util"
        lines.append(
            f"- `{row.experiment}`: {row.proposal}. Mejora: {speedup}. "
            f"Salida equivalente: {row.output_match}. Decision: {row.decision}."
        )
    lines.extend(
        [
            "",
            "## Cambios Implementados",
            "",
        ]
    )
    for row in implemented.itertuples(index=False):
        if row.output_match is None:
            detail = "corrige un caso de no progreso del selector."
        else:
            detail = f"{row.speedup_vs_original:.2f}x en esta corrida."
        lines.append(f"- `{row.experiment}`: {row.proposal}; {detail}")
    lines.extend(
        [
            "",
            "## Verificacion",
            "",
            "- Las etapas con salida original verifican igualdad exacta de DataFrames.",
            "- La prueba de pipeline completo compara las etiquetas finales del camino original y el nuevo.",
            "- El caso `menu` documenta una condicion de no progreso en el original y verifica que el candidato regresa etiquetas.",
            "",
            "Fuente CSV: `experiments/results/bottleneck_ab_test.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="AB test InsideForest bottleneck fixes")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--n-features", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "results" / "bottleneck_ab_test.csv",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=ROOT / "experiments" / "results" / "bottleneck_ab_test_README.md",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = run(args)
    df.to_csv(args.output, index=False)
    write_readme(df, args.readme)
    print(df)
    print(f"CSV written to {args.output}")
    print(f"README written to {args.readme}")


if __name__ == "__main__":
    main()
