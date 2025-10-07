from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from InsideForest.regions import pairwise_iou_blocked, choose_block, pairwise_iou_sparse


def pairwise_iou_sparse_legacy(lows: np.ndarray, highs: np.ndarray) -> sparse.coo_matrix:
    """Reference implementation of ``pairwise_iou_sparse`` from the previous version."""

    n, _ = lows.shape
    vol = np.prod(highs - lows, axis=1)
    centers = (lows + highs) / 2.0
    semi = 0.5 * np.linalg.norm(highs - lows, axis=1)
    max_semi = semi.max()
    tree = cKDTree(centers)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for i in range(n):
        idxs = tree.query_ball_point(centers[i], r=semi[i] + max_semi)
        for j in idxs:
            if j <= i:
                continue
            inter_low = np.maximum(lows[i], lows[j])
            inter_high = np.minimum(highs[i], highs[j])
            inter_dims = np.clip(inter_high - inter_low, 0, None)
            inter_vol = inter_dims.prod()
            union = vol[i] + vol[j] - inter_vol
            if union == 0:
                if np.all(lows[i] == lows[j]) and np.all(highs[i] == highs[j]):
                    iou = 1.0
                else:
                    iou = 0.0
            else:
                iou = inter_vol / union
            if iou > 0.0:
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([iou, iou])

    return sparse.coo_matrix((data, (rows, cols)), shape=(n, n))


def pairwise_iou_naive(lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    n = len(lows)
    vol = np.prod(highs - lows, axis=1)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            inter_low = np.maximum(lows[i], lows[j])
            inter_high = np.minimum(highs[i], highs[j])
            inter_dims = np.clip(inter_high - inter_low, 0, None)
            inter_vol = inter_dims.prod()
            union = vol[i] + vol[j] - inter_vol
            if union == 0:
                if np.all(lows[i] == lows[j]) and np.all(highs[i] == highs[j]):
                    iou = 1.0
                else:
                    iou = 0.0
            else:
                iou = inter_vol / union
            dist_val = 1.0 - iou
            dist[i, j] = dist_val
            dist[j, i] = dist_val
    return dist


def run_pairwise_iou_sparse_ab_test(
    *,
    sample_sizes: tuple[int, ...] = (25, 50, 100, 200),
    dims: int = 3,
    repeats: int = 3,
    seed: int = 123,
    csv_path: str | Path | None = None,
    consistency_csv_path: str | Path | None = None,
    equivalence_tol: float = 1e-12,
) -> pd.DataFrame:
    """Benchmark the legacy and optimized sparse IoU implementations."""

    rng = np.random.default_rng(seed)
    records: list[dict[str, float]] = []

    for n in sample_sizes:
        lows = rng.random((n, dims))
        highs = lows + rng.random((n, dims))

        times_old = []
        times_new = []

        out_old = None
        out_new = None

        for _ in range(repeats):
            start = time.perf_counter()
            out_old = pairwise_iou_sparse_legacy(lows, highs)
            times_old.append(time.perf_counter() - start)

            start = time.perf_counter()
            out_new = pairwise_iou_sparse(lows, highs)
            times_new.append(time.perf_counter() - start)

        assert out_old is not None and out_new is not None
        dense_old = out_old.toarray()
        dense_new = out_new.toarray()
        diff = np.abs(dense_old - dense_new)

        mean_old_ms = float(np.mean(times_old) * 1000)
        mean_new_ms = float(np.mean(times_new) * 1000)
        speedup = mean_old_ms / mean_new_ms if mean_new_ms else np.inf

        records.append(
            {
                "n_regions": n,
                "dims": dims,
                "legacy_mean_ms": mean_old_ms,
                "optimized_mean_ms": mean_new_ms,
                "speedup": speedup,
                "max_abs_diff": float(diff.max()),
            }
        )

    results = pd.DataFrame.from_records(records)
    if csv_path is not None:
        results.to_csv(csv_path, index=False)

    if consistency_csv_path is not None:
        consistency = results[["n_regions", "dims", "max_abs_diff"]].copy()
        consistency["identical_within_tol"] = (
            consistency["max_abs_diff"] <= equivalence_tol
        )
        consistency.to_csv(consistency_csv_path, index=False)

    return results


def test_pairwise_iou_blocked_equivalence():
    rng = np.random.default_rng(0)
    n, d = 20, 3
    lows = rng.random((n, d))
    highs = lows + rng.random((n, d))
    dist_old = pairwise_iou_naive(lows, highs)
    block = choose_block(n, d)
    dist_new = pairwise_iou_blocked(lows, highs, block=block)
    mxdiff = np.nanmax(np.abs(dist_old - dist_new))
    assert mxdiff < 1e-8


def test_pairwise_iou_sparse_equivalence():
    rng = np.random.default_rng(1)
    n, d = 15, 3
    lows = rng.random((n, d))
    highs = lows + rng.random((n, d))
    dense = pairwise_iou_blocked(lows, highs)
    sparse_matrix = pairwise_iou_sparse(lows, highs).toarray()
    np.fill_diagonal(sparse_matrix, 1.0)
    dense_from_sparse = 1.0 - sparse_matrix
    mxdiff = np.nanmax(np.abs(dense - dense_from_sparse))
    assert mxdiff < 1e-8


def test_blocked_delegates_to_sparse():
    rng = np.random.default_rng(2)
    n, d = 10, 3
    lows = rng.random((n, d))
    highs = lows + rng.random((n, d))
    block = choose_block(n, d)
    dist_sparse = pairwise_iou_blocked(
        lows, highs, block=block, sparse_threshold=0
    )
    dist_dense = pairwise_iou_blocked(lows, highs, block=block)
    mxdiff = np.nanmax(np.abs(dist_sparse - dist_dense))
    assert mxdiff < 1e-8


def test_pairwise_iou_sparse_ab(tmp_path):
    csv_file = tmp_path / "pairwise_iou_sparse_ab.csv"
    consistency_csv = tmp_path / "pairwise_iou_sparse_consistency.csv"
    results = run_pairwise_iou_sparse_ab_test(
        csv_path=csv_file, consistency_csv_path=consistency_csv
    )

    assert (results["max_abs_diff"] < 1e-8).all()
    assert (results["speedup"] > 0).all()
    assert (results[["legacy_mean_ms", "optimized_mean_ms"]] > 0).all().all()
    assert csv_file.exists()
    assert consistency_csv.exists()

    reloaded = pd.read_csv(csv_file)
    pd.testing.assert_frame_equal(reloaded, results)

    consistency_loaded = pd.read_csv(consistency_csv)
    expected_consistency = results[["n_regions", "dims", "max_abs_diff"]].copy()
    expected_consistency["identical_within_tol"] = True
    pd.testing.assert_frame_equal(consistency_loaded, expected_consistency)
