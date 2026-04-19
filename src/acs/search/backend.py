
from __future__ import annotations
import os
import numpy as np

_DISABLE_NUMBA = os.environ.get("ACS_DISABLE_NUMBA", "").lower() in {"1", "true", "yes", "on"}
if _DISABLE_NUMBA:  # pragma: no cover - operational fallback
    HAVE_NUMBA = False

    def njit(fn=None, **_kwargs):
        if fn is None:
            return lambda f: f
        return fn
else:
    try:  # pragma: no cover
        from numba import njit
        HAVE_NUMBA = True
    except Exception:  # pragma: no cover
        HAVE_NUMBA = False

        def njit(fn=None, **_kwargs):
            if fn is None:
                return lambda f: f
            return fn

try:  # pragma: no cover
    import cupy as cp
    HAVE_CUPY = True
except Exception:  # pragma: no cover
    HAVE_CUPY = False

def resolve_backend(requested: str = "auto") -> str:
    requested = (requested or "auto").lower()
    if requested == "cupy":
        if HAVE_CUPY:
            return "cupy"
        return "numba" if HAVE_NUMBA else "numpy"
    if requested == "numpy":
        return "numpy"
    if requested == "numba":
        return "numba" if HAVE_NUMBA else "numpy"
    if requested == "auto":
        if HAVE_CUPY:
            return "cupy"
        return "numba" if HAVE_NUMBA else "numpy"
    return "numba" if HAVE_NUMBA else "numpy"

@njit
def _build_prefix(spec: np.ndarray) -> np.ndarray:
    n_frames, nfft = spec.shape
    pref = np.empty((n_frames, nfft + 1), dtype=np.float32)
    pref[:, 0] = 0.0
    for t in range(n_frames):
        running = 0.0
        for b in range(nfft):
            running += spec[t, b]
            pref[t, b + 1] = running
    return pref

@njit
def _search_channel_seeds_numba(spec: np.ndarray, seed_bins: np.ndarray, anchor_frames: np.ndarray, drift_bins_per_frame: np.ndarray, widths: np.ndarray):
    pref = _build_prefix(spec)
    n_frames, nfft = spec.shape
    n_seeds = len(seed_bins)
    best_scores = np.empty(n_seeds, dtype=np.float32)
    best_trials = np.empty(n_seeds, dtype=np.int64)
    best_widths = np.empty(n_seeds, dtype=np.int64)
    for si in range(n_seeds):
        seed_bin = int(seed_bins[si]); anchor = int(anchor_frames[si])
        best_score = -1e30; best_trial = 0; best_width = 1
        for wi in range(len(widths)):
            width = int(widths[wi]); half = width // 2
            for ti in range(len(drift_bins_per_frame)):
                dbpf = drift_bins_per_frame[ti]
                s_all = 0.0; valid = 0
                peak_vals = np.empty(n_frames, dtype=np.float32)
                peak_n = 0
                for t in range(n_frames):
                    center = int(round(seed_bin + dbpf * (t - anchor)))
                    lo = center - half; hi = center + half + 1
                    if lo < 0 or hi > nfft:
                        continue
                    acc = (pref[t, hi] - pref[t, lo]) / width
                    s_all += acc; valid += 1
                    peak_vals[peak_n] = acc; peak_n += 1
                if valid > 0:
                    mean_lin = s_all / max(valid, 1)
                    # transient-sensitive top-30%-of-valid statistic
                    take = max(1, peak_n // 3)
                    # simple partial selection by insertion-like scan for small n
                    for a in range(peak_n):
                        for b in range(a + 1, peak_n):
                            if peak_vals[b] > peak_vals[a]:
                                tmp = peak_vals[a]; peak_vals[a] = peak_vals[b]; peak_vals[b] = tmp
                    top_sum = 0.0
                    for k in range(take):
                        top_sum += peak_vals[k]
                    top_lin = top_sum / max(take, 1)
                    score = 10.0 * np.log10(0.65 * mean_lin + 0.35 * top_lin + 1e-12)
                    if score > best_score:
                        best_score = score; best_trial = ti; best_width = width
        best_scores[si] = best_score
        best_trials[si] = best_trial
        best_widths[si] = best_width
    return best_scores, best_trials, best_widths

def search_channel_seeds_numba(spec_ch: np.ndarray, seed_bins: np.ndarray, anchor_frames: np.ndarray, drift_bins_per_frame: np.ndarray, widths: np.ndarray):
    if not HAVE_NUMBA:
        return search_channel_seeds_numpy(spec_ch, seed_bins, anchor_frames, drift_bins_per_frame, widths)
    return _search_channel_seeds_numba(
        spec_ch.astype(np.float32),
        seed_bins.astype(np.int64),
        anchor_frames.astype(np.int64),
        drift_bins_per_frame.astype(np.float32),
        widths.astype(np.int64),
    )

def search_channel_seeds_numpy(spec_ch: np.ndarray, seed_bins: np.ndarray, anchor_frames: np.ndarray, drift_bins_per_frame: np.ndarray, widths: np.ndarray):
    spec = spec_ch.astype(np.float32, copy=False)
    pref = np.concatenate([np.zeros((spec.shape[0], 1), dtype=np.float32), np.cumsum(spec, axis=1, dtype=np.float32)], axis=1)
    n_frames, nfft = spec.shape
    best_scores = np.full(len(seed_bins), -1e30, dtype=np.float32)
    best_trials = np.zeros(len(seed_bins), dtype=np.int64)
    best_widths = np.ones(len(seed_bins), dtype=np.int64)
    frame_idx = np.arange(n_frames, dtype=np.float32)
    for si, seed_bin in enumerate(seed_bins.astype(np.int64)):
        anchor = int(anchor_frames[si])
        for width in widths.astype(np.int64):
            half = int(width // 2)
            for ti, dbpf in enumerate(drift_bins_per_frame.astype(np.float32)):
                centers = np.rint(seed_bin + dbpf * (frame_idx - anchor)).astype(np.int32)
                lo = centers - half; hi = centers + half + 1
                valid = (lo >= 0) & (hi <= nfft)
                if not np.any(valid):
                    continue
                idx = np.flatnonzero(valid)
                vals = (pref[idx, hi[idx]] - pref[idx, lo[idx]]) / float(width)
                mean_lin = float(vals.mean())
                n_top = max(1, len(vals)//3)
                top = np.sort(vals)[-n_top:]
                top_lin = float(top.mean())
                score = float(10.0*np.log10(0.65*mean_lin + 0.35*top_lin + 1e-12))
                if score > best_scores[si]:
                    best_scores[si] = score; best_trials[si] = ti; best_widths[si] = width
    return best_scores, best_trials, best_widths

def search_channel_seeds_cupy(spec_ch, seed_bins: np.ndarray, anchor_frames: np.ndarray, drift_bins_per_frame: np.ndarray, widths: np.ndarray):  # pragma: no cover
    # conservative fallback: use numpy if cupy absent
    return search_channel_seeds_numpy(np.asarray(spec_ch), seed_bins, anchor_frames, drift_bins_per_frame, widths)
