import time
from sklearn.tree import DecisionTreeClassifier, export_text, _tree
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_classification
import pandas as pd
import numpy as np


def extract_rules_export_text(tree, feature_names):
    text = export_text(
        tree,
        feature_names=list(feature_names),
        decimals=6,
        max_depth=tree.tree_.max_depth,
    )
    lines = text.splitlines()
    rules = []
    stack: list[str] = []
    for line in lines:
        depth = line.count("|   ")
        content = line.strip()
        if "class:" in content:
            class_idx = int(content.split("class:")[1].strip())
            rules.append((stack[:depth], class_idx))
        else:
            condition = " ".join(line.split("|---")[-1].split())
            stack = stack[:depth]
            stack.append(condition)
    return rules


def extract_rules_direct(tree, feature_names):
    t = tree.tree_
    rules = []
    def recurse(node, path):
        if t.feature[node] != _tree.TREE_UNDEFINED:
            feature = feature_names[t.feature[node]]
            threshold = t.threshold[node]
            recurse(t.children_left[node], path + [f"{feature} <= {threshold:.6f}"])
            recurse(t.children_right[node], path + [f"{feature} > {threshold:.6f}"])
        else:
            value = t.value[node]
            class_idx = int(np.argmax(value))
            rules.append((path, class_idx))
    recurse(0, [])
    return rules


def benchmark(func, *args, runs=5):
    times = []
    result = None
    for _ in range(runs):
        start = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - start)
    return result, min(times)


def compare(tree, feature_names):
    rules_txt, time_txt = benchmark(extract_rules_export_text, tree, feature_names)
    rules_dir, time_dir = benchmark(extract_rules_direct, tree, feature_names)
    normalize = lambda r: (tuple(r[0]), r[1])
    same = set(map(normalize, rules_txt)) == set(map(normalize, rules_dir))
    return time_txt, time_dir, same


def main():
    datasets = {
        "iris": load_iris(),
        "wine": load_wine(),
        "breast_cancer": load_breast_cancer(),
        "synthetic": make_classification(n_samples=10000, n_features=20, random_state=0)
    }
    for name, data in datasets.items():
        if isinstance(data, tuple):
            X, y = data
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        else:
            X, y = data.data, data.target
            feature_names = data.feature_names
        X_df = pd.DataFrame(X, columns=feature_names)
        clf = DecisionTreeClassifier(random_state=0).fit(X_df, y)
        t_txt, t_dir, same = compare(clf, feature_names)
        print(f"{name}: export_text {t_txt:.6f}s vs direct {t_dir:.6f}s -> same: {same}")


if __name__ == "__main__":
    main()
