import numpy as np
from InsideForest.regions import pairwise_iou_blocked, choose_block

def pairwise_iou_naive(lows, highs):
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
