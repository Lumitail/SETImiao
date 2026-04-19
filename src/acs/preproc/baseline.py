from __future__ import annotations
import numpy as np
from scipy.ndimage import median_filter
from ..types import BaselineConfig, SpectrogramTile

EPS = 1e-12


def apply_baseline_and_masks(tile: SpectrogramTile, cfg: BaselineConfig) -> SpectrogramTile:
    frames, channels, nfft = tile.power.shape
    flat = tile.power.reshape(frames, channels * nfft)
    med_db = 10.0 * np.log10(np.median(flat, axis=0) + EPS)
    smooth_bins = int(cfg.smooth_bins)
    if smooth_bins % 2 == 0:
        smooth_bins += 1
    smooth_bins = max(3, min(smooth_bins, channels * nfft - 1 if (channels * nfft) % 2 == 0 else channels * nfft))
    baseline_db = median_filter(med_db, size=smooth_bins, mode="nearest")
    baseline_lin = np.power(10.0, baseline_db / 10.0).astype(np.float32) + EPS
    norm = flat / baseline_lin[None, :]
    mean_excess_db = 10.0 * np.log10(np.mean(norm, axis=0) + EPS)
    max_excess_db = 10.0 * np.log10(np.max(norm, axis=0) + EPS)
    p90_excess_db = 10.0 * np.log10(np.quantile(norm, 0.90, axis=0) + EPS)

    # Search mask: deterministic edge blanking only. Strong spectral outliers are not masked
    # out of the detector seed stage, because that was suppressing real injected signals.
    mask = np.zeros((channels, nfft), dtype=bool)
    e = int(cfg.edge_bins_per_coarse)
    if e > 0:
        mask[:, :e] = True
        mask[:, -e:] = True

    tile.norm_power[...] = norm.reshape(frames, channels, nfft)
    tile.mean_excess_db[...] = mean_excess_db.reshape(channels, nfft).astype(np.float32)
    tile.max_excess_db[...] = max_excess_db.reshape(channels, nfft).astype(np.float32)
    tile.p90_excess_db[...] = p90_excess_db.reshape(channels, nfft).astype(np.float32)
    tile.mask[...] = mask
    return tile


def qc_display_matrix(tile: SpectrogramTile, cfg: BaselineConfig) -> tuple[np.ndarray, float, float]:
    """Return a QC display matrix that preserves macro texture.

    The search path uses fully flattened power. The QC display path should not.
    We therefore show log-power after subtracting only a broad spectral baseline in dB,
    not the fully normalized search matrix.
    """
    frames, channels, nfft = tile.power.shape
    power_db = 10.0 * np.log10(tile.power.reshape(frames, channels * nfft) + EPS)
    bg = np.median(power_db, axis=0)
    smooth = int(cfg.qc_display_smooth_bins)
    if smooth % 2 == 0:
        smooth += 1
    smooth = max(3, min(smooth, power_db.shape[1] - 1 if power_db.shape[1] % 2 == 0 else power_db.shape[1]))
    bg_smooth = median_filter(bg, size=smooth, mode="nearest")
    disp = power_db - bg_smooth[None, :]
    disp = disp - np.median(disp, axis=1, keepdims=True)
    vmin = float(np.percentile(disp, cfg.qc_low_percentile))
    vmax = float(np.percentile(disp, cfg.qc_high_percentile))
    return disp.astype(np.float32), vmin, vmax
