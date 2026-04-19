
import numpy as np
from acs.search.backend import search_channel_seeds_numpy, search_channel_seeds_numba

def test_numba_matches_numpy():
    rng = np.random.default_rng(0)
    spec = rng.normal(1.0, 0.1, size=(16,128)).astype(np.float32)
    seed_bins = np.array([20,40,90], dtype=np.int64)
    anchors = np.array([8,8,8], dtype=np.int64)
    drifts = np.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np.float32)
    widths = np.array([1,3,5], dtype=np.int64)
    a = search_channel_seeds_numpy(spec, seed_bins, anchors, drifts, widths)
    b = search_channel_seeds_numba(spec, seed_bins, anchors, drifts, widths)
    for xa, xb in zip(a, b):
        assert np.allclose(xa, xb, atol=1e-5, rtol=1e-5)
