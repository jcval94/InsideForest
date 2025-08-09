import numpy as np

from InsideForest.regions import _pairwise_iou_blocked


def _pairwise_iou_naive(lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    n = lows.shape[0]
    dist = np.zeros((n, n), dtype=np.float64)
    volumes = np.prod(highs - lows, axis=1)
    for i in range(n):
        for j in range(i, n):
            inter_low = np.maximum(lows[i], lows[j])
            inter_high = np.minimum(highs[i], highs[j])
            inter_dims = np.clip(inter_high - inter_low, 0, None)
            inter_vol = inter_dims.prod()
            union = volumes[i] + volumes[j] - inter_vol
            if union == 0:
                same = np.all(lows[i] == lows[j]) and np.all(highs[i] == highs[j])
                iou = 1.0 if same else 0.0
            else:
                iou = inter_vol / union
            dist[i, j] = 1.0 - iou
            dist[j, i] = dist[i, j]
    np.fill_diagonal(dist, 0.0)
    return dist


def test_blocked_matches_naive():
    rng = np.random.default_rng(0)
    n, d = 10, 4
    lows = rng.random((n, d))
    highs = lows + rng.random((n, d))
    # insert identical degenerate boxes to exercise union==0 path
    lows[0] = highs[0] = 0.5
    lows[1] = highs[1] = 0.5
    dist_block = _pairwise_iou_blocked(lows, highs, block=5)
    dist_naive = _pairwise_iou_naive(lows, highs)
    mxdiff = np.nanmax(np.abs(dist_block - dist_naive))
    assert mxdiff < 1e-8

