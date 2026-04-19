
from __future__ import annotations
import numpy as np
from ..types import StitchedObservation, Hit, RefineConfig
from ..io.dat_reader import decode_words_to_complex64
from ..io.stitcher import read_channel_rows

def refine_hits(obs: StitchedObservation, hits: list[Hit], cfg: RefineConfig) -> list[Hit]:
    if not cfg.enabled or not hits:
        return hits
    hits = list(hits)
    order = np.argsort([h.incoherent_snr for h in hits])[::-1][:cfg.top_n]
    coarse_sr = obs.contract.coarse_df_hz
    # Cache only the channel needed by each hit.  Earlier versions decoded all
    # 256 channels for every refined tile, which dominated runtime on large .dat
    # files.  Refinement only operates on one coarse channel at a time.
    cache: dict[tuple[int, int, int], np.ndarray] = {}
    for idx in order:
        hit = hits[int(idx)]
        key = (int(hit.tile_row0), int(hit.tile_row1), int(hit.coarse_channel))
        if key not in cache:
            words = read_channel_rows(obs, hit.tile_row0, hit.tile_row1, hit.coarse_channel)
            if len(words) == 0:
                cache[key] = np.empty((0,), dtype=np.complex64)
            else:
                words2d = np.asarray(words, dtype=np.uint16).reshape(-1, 1)
                cache[key] = decode_words_to_complex64(words2d)[:, 0]
        x = cache[key].astype(np.complex64, copy=False)
        if x.size == 0:
            continue
        x = x - x.mean()
        n = len(x)
        if n == 0:
            continue
        t = np.arange(n, dtype=np.float64) / coarse_sr
        win = np.hanning(n).astype(np.float32)
        coarse_center = float(obs.freq_map.coarse_centers_hz[hit.coarse_channel])
        local_center_bin = int(hit.fine_bin)
        half_window = max(float(cfg.drift_half_window_hz_per_s), min(16.0, 0.5 * abs(float(hit.drift_hz_per_s))))
        drift_grid = np.arange(float(hit.drift_hz_per_s) - half_window, float(hit.drift_hz_per_s) + half_window + 0.5 * float(cfg.drift_step_hz_per_s), float(cfg.drift_step_hz_per_s), dtype=np.float64)

        # local ridge estimate from strongest frames in a neighborhood around the seeded bin
        neighborhood = slice(max(0, local_center_bin - cfg.freq_half_window_bins), min(n, local_center_bin + cfg.freq_half_window_bins + 1))
        stft_peak_bins = []
        stft_peak_strength = []
        frame_hop = max(1, n // max(8, min(64, n // 32 if n >= 32 else 1)))
        for s in range(0, n - frame_hop + 1, frame_hop):
            seg = x[s:s + frame_hop]
            spec = np.fft.fftshift(np.fft.fft(seg * np.hanning(len(seg)).astype(np.float32)))
            amp = np.abs(spec)[neighborhood]
            if amp.size == 0:
                continue
            pk = int(np.argmax(amp)) + neighborhood.start
            stft_peak_bins.append(pk)
            stft_peak_strength.append(float(np.max(amp)))
        if stft_peak_bins:
            order_pk = np.argsort(stft_peak_strength)[::-1]
            keep = max(1, len(order_pk) // 3)
            kept_bins = np.asarray([stft_peak_bins[i] for i in order_pk[:keep]], dtype=np.float64)
            kept_times = np.linspace(0.0, float(n - 1), num=len(kept_bins), dtype=np.float64)
            if len(kept_bins) >= 2:
                # map bin slope to drift estimate
                p = np.polyfit(kept_times, kept_bins, 1)
                fine_df = obs.contract.coarse_df_hz / max(hit.width_bins, 1)  # rough local scale, only for centering
                ridge_drift = float(p[0] * (obs.contract.coarse_df_hz / n))
                if np.isfinite(ridge_drift):
                    center_drift = 0.5 * float(hit.drift_hz_per_s) + 0.5 * ridge_drift
                    drift_grid = np.arange(center_drift - half_window, center_drift + half_window + 0.5 * float(cfg.drift_step_hz_per_s), float(cfg.drift_step_hz_per_s), dtype=np.float64)

        best_amp = -1.0
        best_freq = float(hit.freq_hz if hit.freq_hz is not None else coarse_center)
        best_drift = float(hit.drift_hz_per_s)
        raw_amp = np.abs(np.fft.fftshift(np.fft.fft(x * win)))
        base = float(np.median(raw_amp) + 1e-6)
        fine_offsets = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / coarse_sr))
        for drift in drift_grid:
            dechirped = x * np.exp(-1j * np.pi * drift * (t ** 2))
            spec = np.fft.fftshift(np.fft.fft(dechirped * win))
            amp = np.abs(spec)
            peak = int(np.argmax(amp))
            peak_amp = float(amp[peak])
            if peak_amp > best_amp:
                # parabolic interp in amplitude domain around peak when available
                frac = 0.0
                if 1 <= peak < len(amp) - 1:
                    y1, y2, y3 = float(amp[peak - 1]), float(amp[peak]), float(amp[peak + 1])
                    denom = (y1 - 2.0 * y2 + y3)
                    if abs(denom) > 1e-9:
                        frac = 0.5 * (y1 - y3) / denom
                        frac = float(np.clip(frac, -0.5, 0.5))
                best_amp = peak_amp
                best_freq = float((fine_offsets[peak] + frac * (coarse_sr / n)) + coarse_center)
                best_drift = float(drift)
        refined_snr = 20.0 * np.log10(best_amp / base)
        gain = refined_snr - hit.incoherent_snr
        hits[int(idx)] = Hit(**{
            **hit.to_dict(),
            "refined_snr": float(refined_snr),
            "refined_freq_hz": float(best_freq),
            "refined_drift_hz_per_s": float(best_drift),
            "coherent_gain_db": float(gain),
        })
    return hits
