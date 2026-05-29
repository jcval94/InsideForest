"""Run the downloaded InsideForest use case with traditional and multiclass APIs.

The notebook exported at ``C:/Users/ASUS/Downloads/insideforest_caso_de_uso (1).py``
contains Colab magics and commented cells.  This script is the local,
reproducible version: it executes the Iris multiclass case and the Titanic
case, compares outputs from the traditional API against the new multiclass
API, and saves tables plus figures.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from InsideForest import InsideForestClassifier
from InsideForest.descrip import get_frontiers
from InsideForest.metadata import MetaExtractor, run_experiments
from InsideForest.multiclass import InsideForestMulticlassClassifier


OUT_DIR = ROOT / "experiments" / "results" / "insideforest_use_case_multiclass"
FIG_DIR = OUT_DIR / "figures"


def timed(label, func):
    start = perf_counter()
    result = func()
    elapsed = perf_counter() - start
    print(f"{label}: {elapsed:.3f}s")
    return result, elapsed


def aligned_cluster_accuracy(y_true, cluster_labels):
    y_true = np.asarray(y_true)
    cluster_labels = np.asarray(cluster_labels)
    valid = cluster_labels != -1
    if not np.any(valid):
        return 0.0, {}

    true_values = np.unique(y_true)
    cluster_values = np.unique(cluster_labels[valid])
    matrix = np.zeros((len(true_values), len(cluster_values)), dtype=int)
    for i, cls in enumerate(true_values):
        for j, cluster in enumerate(cluster_values):
            matrix[i, j] = np.sum((y_true == cls) & (cluster_labels == cluster))

    row_ind, col_ind = linear_sum_assignment(-matrix)
    mapping = {cluster_values[col]: true_values[row] for row, col in zip(row_ind, col_ind)}
    aligned = np.array([mapping.get(label, -1) for label in cluster_labels])
    return float(np.mean(aligned == y_true)), mapping


def stacked_plot(y_true, labels, title, xlabel, legend_title, path):
    data = pd.DataFrame({"target": y_true, "label": labels})
    stacked = data.groupby(["target", "label"]).size().unstack(fill_value=0)
    colors = cm.viridis(np.linspace(0, 1, max(len(stacked.columns), 1)))
    ax = stacked.plot(kind="bar", stacked=True, figsize=(10, 6), color=colors)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def confusion_plot(y_true, y_pred, labels, title, path):
    cmatrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmatrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cmatrix.shape[0]):
        for j in range(cmatrix.shape[1]):
            ax.text(j, i, str(cmatrix[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def frontiers_plot(frontiers, title, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    if frontiers is None or frontiers.empty:
        ax.text(0.5, 0.5, "No frontiers available", ha="center", va="center")
    else:
        scatter = ax.scatter(
            frontiers["delta_cluster_ef_sample"],
            frontiers["similarity"],
            c=frontiers["score"],
            cmap="viridis",
            alpha=0.75,
        )
        fig.colorbar(scatter, ax=ax, label="Score")
    ax.set_xlabel("Delta cluster effectiveness")
    ax.set_ylabel("Similarity")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def multiclass_margin_plot(assignments, title, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = assignments["source"].map({"region": "#2E86AB", "model_fallback": "#F18F01"}).fillna("#999999")
    ax.scatter(assignments["confidence"], assignments["margin"], c=colors, alpha=0.75)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Top-2 margin")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def prototype_plot(rules, title, path, top_n=8):
    if rules.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No rules available", ha="center", va="center")
    else:
        plot_df = (
            rules.sort_values("score", ascending=False)
            .groupby("target_class")
            .head(top_n)
            .copy()
        )
        plot_df["label"] = plot_df["target_class"].astype(str) + " #" + plot_df.groupby("target_class").cumcount().astype(str)
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(plot_df["label"], plot_df["score"], color="#5B8E7D")
        ax.set_xlabel("Score")
        ax.set_title(title)
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def build_simple_metadata(feature_names, target_name, lang="es"):
    rows = []
    for name in feature_names:
        rows.append(
            {
                "metadata_row": name,
                "identity.label_i18n.es": name.replace("_", " ").title(),
                "identity.description_i18n.es": f"Variable {name}",
                "type.logical_type": "numeric",
                "domain.categorical.codes": "",
                "actionability.increase_difficulty": 5,
                "actionability.decrease_difficulty": 5,
                "actionability.side_effects": "",
            }
        )
    rows.append(
        {
            "metadata_row": target_name,
            "identity.label_i18n.es": target_name.replace("_", " ").title(),
            "identity.description_i18n.es": f"Objetivo {target_name}",
            "type.logical_type": "categorical",
            "domain.categorical.codes": "",
            "actionability.increase_difficulty": 5,
            "actionability.decrease_difficulty": 5,
            "actionability.side_effects": "",
        }
    )
    return pd.DataFrame(rows).set_index("metadata_row")


def traditional_fit_predict(X_df, y, *, var_obj, rf_params, tree_params, fit_kwargs):
    df = X_df.copy()
    df[var_obj] = y
    model = InsideForestClassifier(
        rf_params=rf_params,
        tree_params=tree_params,
        var_obj=var_obj,
        get_detail=True,
        **fit_kwargs,
    )
    model.fit(df)
    labels = model.predict(X_df)
    return model, labels


def multiclass_fit_assign(X_df, y, *, rf_params, percentil, low_frac, min_support, random_state):
    model = InsideForestMulticlassClassifier(
        rf_params=rf_params,
        percentil=percentil,
        low_frac=low_frac,
        min_support=min_support,
        random_state=random_state,
        conflict_margin=0.15,
    )
    model.fit(X_df, y)
    assignments = model.assign_regions(X_df)
    return model, assignments


def run_metadata_pipeline(name, df_raw, var_obj, df_datos_explain, frontiers):
    if frontiers is None or frontiers.empty or df_datos_explain is None or df_datos_explain.empty:
        return pd.DataFrame()

    percentile = frontiers["score"].quantile(0.65)
    selected = frontiers[frontiers["score"] > percentile]
    if selected.empty:
        selected = frontiers.head(1)

    mis_df2s = {}
    for idx, pair in enumerate(selected[["cluster_1", "cluster_2"]].head(5).values):
        mis_df2s[f"{name}_experiment_{idx}"] = df_datos_explain[df_datos_explain["cluster"].isin(pair)]

    if not mis_df2s:
        return pd.DataFrame()

    meta = build_simple_metadata([c for c in df_raw.columns if c != var_obj], var_obj)
    mx = MetaExtractor(meta, var_obj)
    return run_experiments(mx, mis_df2s, data_dict={key: df_raw for key in mis_df2s})


def run_iris():
    print("\n=== Iris multiclass use case ===")
    iris = load_iris()
    X_df = pd.DataFrame(iris.data, columns=["petal_length", "petal_width", "sepal_length", "sepal_width"])
    y = iris.target.astype(float)
    var_obj = "species"
    rf_params = {"random_state": 15, "n_estimators": 60, "max_depth": 6, "n_jobs": 1}
    tree_params = {"lang": "py", "n_sample_multiplier": 0.05, "ef_sample_multiplier": 10}

    (traditional, traditional_labels), trad_fit = timed(
        "iris traditional fit+predict",
        lambda: traditional_fit_predict(
            X_df,
            y,
            var_obj=var_obj,
            rf_params=rf_params,
            tree_params=tree_params,
            fit_kwargs={"leaf_percentile": 95, "low_leaf_fraction": 0.25, "max_cases": len(X_df)},
        ),
    )
    (multiclass, assignments), mc_fit = timed(
        "iris multiclass fit+assign",
        lambda: multiclass_fit_assign(
            X_df,
            y,
            rf_params=rf_params,
            percentil=95,
            low_frac=0.25,
            min_support=2,
            random_state=15,
        ),
    )

    return build_outputs("iris", X_df, y, var_obj, traditional, traditional_labels, trad_fit, multiclass, assignments, mc_fit)


def run_titanic():
    print("\n=== Titanic binary case through multiclass API ===")
    df = sns.load_dataset("titanic")
    var_obj = "survived"
    y = df[var_obj].to_numpy()
    features = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
        "class",
        "who",
        "adult_male",
        "deck",
        "embark_town",
        "alone",
    ]
    raw_X = df[features].copy()
    for col in ["adult_male", "alone"]:
        raw_X[col] = raw_X[col].astype("int64")

    numerical_features = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical_features = ["sex", "embarked", "class", "who", "deck", "embark_town"]
    boolean_features = ["adult_male", "alone"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "bool",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                boolean_features,
            ),
        ]
    )
    X_transformed = preprocessor.fit_transform(raw_X)
    X_dense = X_transformed.toarray() if hasattr(X_transformed, "toarray") else np.asarray(X_transformed)
    feature_names = [str(name).replace(" ", "_") for name in preprocessor.get_feature_names_out()]
    X_df = pd.DataFrame(X_dense, columns=feature_names)

    rf_params = {"random_state": 15, "n_estimators": 40, "max_depth": 6, "n_jobs": 1}
    tree_params = {
        "lang": "py",
        "n_sample_multiplier": 0.05,
        "ef_sample_multiplier": 10,
        "percentil": 99,
        "low_frac": 0.02,
    }

    (traditional, traditional_labels), trad_fit = timed(
        "titanic traditional fit+predict",
        lambda: traditional_fit_predict(
            X_df,
            y,
            var_obj=var_obj,
            rf_params=rf_params,
            tree_params=tree_params,
            fit_kwargs={
                "max_cases": 500,
                "no_trees_search": 80,
                "auto_fast": True,
                "auto_feature_reduce": True,
            },
        ),
    )
    X_pred = X_df[traditional.feature_names_] if traditional.feature_names_ else X_df
    traditional_labels = traditional.predict(X_pred)

    (multiclass, assignments), mc_fit = timed(
        "titanic multiclass fit+assign",
        lambda: multiclass_fit_assign(
            X_df,
            y,
            rf_params=rf_params,
            percentil=99,
            low_frac=0.02,
            min_support=5,
            random_state=15,
        ),
    )

    return build_outputs("titanic", X_df, y, var_obj, traditional, traditional_labels, trad_fit, multiclass, assignments, mc_fit)


def build_outputs(name, X_df, y, var_obj, traditional, traditional_labels, trad_time, multiclass, assignments, mc_time):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    traditional_acc, cluster_mapping = aligned_cluster_accuracy(y, traditional_labels)
    multiclass_pred = assignments["predicted_class"].to_numpy()
    multiclass_acc = accuracy_score(y, multiclass_pred)

    df_for_frontiers = X_df.copy()
    df_for_frontiers[var_obj] = y
    try:
        df_datos_explain, frontiers = get_frontiers(traditional.df_clusters_description_, df_for_frontiers, divide=5)
    except Exception as exc:
        print(f"{name} frontiers failed: {exc}")
        df_datos_explain, frontiers = pd.DataFrame(), pd.DataFrame()

    summary = {
        "dataset": name,
        "traditional_elapsed_seconds": trad_time,
        "multiclass_elapsed_seconds": mc_time,
        "traditional_rf_accuracy": float(traditional.rf.score(X_df[traditional.feature_names_], y) if traditional.feature_names_ else traditional.rf.score(X_df, y)),
        "multiclass_rf_accuracy": float(multiclass.rf_.score(X_df, y)),
        "traditional_assignment_accuracy_aligned": traditional_acc,
        "multiclass_assignment_accuracy": float(multiclass_acc),
        "traditional_cluster_count": int(len(set(traditional_labels))),
        "traditional_unmatched_rate": float(np.mean(np.asarray(traditional_labels) == -1)),
        "multiclass_rule_count": int(len(multiclass.rules_)),
        "multiclass_fallback_rate": float(np.mean(assignments["source"] == "model_fallback")),
        "multiclass_conflict_rate": float(np.mean(assignments["is_conflict"])),
    }

    pd.DataFrame([summary]).to_csv(OUT_DIR / f"{name}_summary.csv", index=False)
    assignments.to_csv(OUT_DIR / f"{name}_multiclass_assignments.csv", index=False)
    multiclass.explain(top_n=50).to_csv(OUT_DIR / f"{name}_multiclass_top_rules.csv", index=False)
    if traditional.df_clusters_description_ is not None:
        traditional.df_clusters_description_.to_csv(OUT_DIR / f"{name}_traditional_clusters_description.csv", index=False)
    if frontiers is not None and not frontiers.empty:
        frontiers.to_csv(OUT_DIR / f"{name}_traditional_frontiers.csv", index=False)

    metadata_experiments = run_metadata_pipeline(
        name,
        df_for_frontiers,
        var_obj,
        df_datos_explain,
        frontiers,
    )
    if not metadata_experiments.empty:
        metadata_experiments.to_csv(OUT_DIR / f"{name}_metadata_experiments.csv", index=False)

    stacked_plot(
        y,
        traditional_labels,
        f"{name}: target vs traditional cluster labels",
        var_obj,
        "Traditional label",
        FIG_DIR / f"{name}_traditional_stacked_target_vs_labels.png",
    )
    stacked_plot(
        y,
        multiclass_pred,
        f"{name}: target vs multiclass predicted class",
        var_obj,
        "Predicted class",
        FIG_DIR / f"{name}_multiclass_stacked_target_vs_prediction.png",
    )
    confusion_plot(
        y,
        multiclass_pred,
        sorted(np.unique(y)),
        f"{name}: multiclass assignment confusion matrix",
        FIG_DIR / f"{name}_multiclass_confusion_matrix.png",
    )
    frontiers_plot(frontiers, f"{name}: traditional frontier comparison", FIG_DIR / f"{name}_traditional_frontiers.png")
    multiclass_margin_plot(assignments, f"{name}: multiclass confidence vs margin", FIG_DIR / f"{name}_multiclass_confidence_margin.png")
    prototype_plot(multiclass.prototype_regions(top_n=5), f"{name}: multiclass prototype rule scores", FIG_DIR / f"{name}_multiclass_prototype_scores.png")

    print(json.dumps(summary, indent=2))
    print(f"cluster mapping for {name}: {cluster_mapping}")
    return summary


def write_report(summaries):
    df = pd.DataFrame(summaries)
    df.to_csv(OUT_DIR / "use_case_comparison_summary.csv", index=False)

    lines = [
        "# InsideForest Use Case: Traditional vs Multiclass",
        "",
        "This report was generated by `experiments/insideforest_use_case_multiclass_comparison.py`.",
        "",
        "| Dataset | Traditional time s | Multiclass time s | Traditional RF acc | Multiclass RF acc | Traditional aligned assignment acc | Multiclass assignment acc | Traditional unmatched | Multiclass fallback | Multiclass conflicts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {dataset} | {traditional_elapsed_seconds:.3f} | {multiclass_elapsed_seconds:.3f} | "
            "{traditional_rf_accuracy:.4f} | {multiclass_rf_accuracy:.4f} | "
            "{traditional_assignment_accuracy_aligned:.4f} | {multiclass_assignment_accuracy:.4f} | "
            "{traditional_unmatched_rate:.4f} | {multiclass_fallback_rate:.4f} | "
            "{multiclass_conflict_rate:.4f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "Figures are saved under `experiments/results/insideforest_use_case_multiclass/figures`.",
            "",
        ]
    )
    for path in sorted(FIG_DIR.glob("*.png")):
        rel = path.relative_to(ROOT).as_posix()
        lines.append(f"- `{rel}`")

    (OUT_DIR / "use_case_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [run_iris(), run_titanic()]
    write_report(summaries)
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
