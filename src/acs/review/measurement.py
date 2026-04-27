from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any

import numpy as np

from ..frontend.stft import frame_times_s, get_window
from ..io.dat_reader import decode_words_to_complex64, fine_offsets_hz
from ..io.stitcher import read_channel_rows
from ..types import Event, MeasurementConfig, STFTConfig, StitchedObservation


def _finite_or_none(value: Any) -> float | int | str | None:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _db10(value: float | None) -> float | None:
    if value is None or not math.isfinite(value) or value <= 0.0:
        return None
    return 10.0 * math.log10(float(value))


def _snap_noise_bias_factor(cfg: MeasurementConfig) -> float:
    """Approximate exponential-noise bias from snapping to the brightest bin.

    STFT power of complex Gaussian noise is approximately exponential. If the
    ridge center is snapped to the maximum of M neighboring bins, the expected
    noise contribution is H_M times the mean noise power. When the configured
    noise statistic is the median, the local median estimates ln(2) times the
    mean, so the correction is H_M / ln(2).
    """
    snap = max(0, int(getattr(cfg, "ridge_snap_half_width_bins", 0)))
    m = 2 * snap + 1
    if m <= 1:
        return 1.0
    h_m = sum(1.0 / k for k in range(1, m + 1))
    if str(cfg.noise_statistic).lower() == "median":
        return float(h_m / math.log(2.0))
    return float(h_m)


def _blank_measurement(cfg: MeasurementConfig, warning: str) -> dict[str, Any]:
    return {
        "snr_measurement_method": cfg.method if cfg.enabled else "disabled",
        "recovered_ridge_pixel_snr_db": None,
        "recovered_band_excess_snr_db": None,
        "recovered_local_noise_floor_power": None,
        "recovered_signal_excess_power": None,
        "recovered_background_pixel_count": 0,
        "recovered_ridge_pixel_count": 0,
        "recovered_ridge_width_bins": None,
        "recovered_ridge_snap_half_width_bins": int(getattr(cfg, "ridge_snap_half_width_bins", 0)),
        "recovered_reference_bandwidth_hz": None,
        "recovered_measurement_enbw_bins": None,
        "recovered_snap_noise_bias_factor": _snap_noise_bias_factor(cfg),
        "recovered_mean_ridge_power": None,
        "recovered_mean_ridge_excess_power": None,
        "recovered_same_band_noise_power": None,
        "recovered_measurement_warning": warning,
    }


def _choose_channel(obs: StitchedObservation, event: Event) -> int:
    if event.representative_coarse_channel is not None:
        return int(np.clip(event.representative_coarse_channel, 0, obs.contract.channels - 1))
    if event.freq_hz is None:
        return obs.contract.channels // 2
    return int(np.argmin(np.abs(obs.freq_map.coarse_centers_hz - float(event.freq_hz))))


def _ridge_width_bins(
    event: Event,
    cfg: MeasurementConfig,
    stft_cfg: STFTConfig,
    obs: StitchedObservation,
    reference_width_hz: float | None = None,
) -> int:
    df = float(obs.contract.coarse_df_hz) / float(stft_cfg.nfft)
    min_bins = max(1, int(cfg.min_ridge_bins))
    max_bins = max(min_bins, int(cfg.max_ridge_bins))
    source = str(cfg.ridge_width_source or "auto").lower()

    bins: int
    if reference_width_hz is not None and math.isfinite(float(reference_width_hz)) and float(reference_width_hz) > 0:
        bins = int(math.ceil(float(reference_width_hz) / df))
    elif source in {"event", "auto"} and event.best_width_bins is not None:
        bins = int(event.best_width_bins)
    else:
        bins = min_bins

    bins = max(min_bins, min(max_bins, bins))
    return bins


def measure_event_snr(
    obs: StitchedObservation,
    event: Event,
    measurement_cfg: MeasurementConfig,
    stft_cfg: STFTConfig,
    *,
    reference_width_hz: float | None = None,
) -> dict[str, Any]:
    """Measure post-detection average signal excess relative to local STFT noise.

    This is deliberately not a detector ranking metric. It reads the relevant
    coarse channel, forms the same STFT grid used by the search frontend, follows
    the event ridge, estimates nearby off-ridge background power, and reports an
    average ridge/band excess SNR.

    The main SETI-comparable field is ``recovered_band_excess_snr_db``. It applies
    a window ENBW correction so a bin-centered narrowband tone with
    ``snr_reference: local_psd`` is reported on approximately the same SNR scale
    as the injected local-PSD SNR.
    """
    cfg = measurement_cfg
    if not cfg.enabled:
        return _blank_measurement(cfg, "measurement disabled")
    if str(cfg.method) != "stft_local_background":
        return _blank_measurement(cfg, f"unsupported measurement method: {cfg.method}")
    if event.freq_hz is None:
        return _blank_measurement(cfg, "event has no frequency")

    row0 = max(0, int(event.row0))
    row1 = min(obs.total_rows, int(event.row1))
    if row1 <= row0:
        return _blank_measurement(cfg, "event row range is empty")
    if row1 - row0 < stft_cfg.nfft:
        row1 = min(obs.total_rows, row0 + stft_cfg.nfft)
        row0 = max(0, row1 - stft_cfg.nfft)
    if row1 - row0 < stft_cfg.nfft:
        return _blank_measurement(cfg, "not enough rows for STFT measurement")

    channel = _choose_channel(obs, event)
    words = read_channel_rows(obs, row0, row1, channel)
    x = decode_words_to_complex64(np.asarray(words, dtype=np.uint16).reshape(-1, 1))[:, 0].astype(np.complex64, copy=False)
    if x.size < stft_cfg.nfft:
        return _blank_measurement(cfg, "not enough samples after reading event channel")
    x = x - x.mean()

    hop = int(stft_cfg.hop)
    nfft = int(stft_cfg.nfft)
    n_frames = 1 + (len(x) - nfft) // hop
    if n_frames <= 0:
        return _blank_measurement(cfg, "no STFT frames available for measurement")

    win = get_window(stft_cfg.window, nfft).astype(np.float64)
    win_sum = float(np.sum(win))
    win_sq_sum = float(np.sum(win * win))
    enbw_bins = float(nfft * win_sq_sum / (win_sum * win_sum)) if abs(win_sum) > 1e-12 else 1.0

    power = np.empty((n_frames, nfft), dtype=np.float64)
    for fi in range(n_frames):
        s = fi * hop
        seg = x[s:s + nfft].astype(np.complex128) * win
        spec = np.fft.fftshift(np.fft.fft(seg))
        power[fi, :] = (spec.real * spec.real + spec.imag * spec.imag)

    freq = obs.freq_map.coarse_centers_hz[channel] + fine_offsets_hz(obs.contract, nfft)
    df = float(obs.contract.coarse_df_hz) / float(nfft)
    frame_time_abs = row0 * obs.contract.native_dt_s + frame_times_s(n_frames, hop, nfft, obs.contract.coarse_df_hz)
    ref_row = int(event.peak_row) if event.peak_row is not None else int(round(0.5 * (event.row0 + event.row1)))
    ref_time = ref_row * obs.contract.native_dt_s

    ridge_bins = _ridge_width_bins(event, cfg, stft_cfg, obs, reference_width_hz)
    half_low = (ridge_bins - 1) // 2
    half_high = ridge_bins - half_low

    ridge_values: list[float] = []
    background_values: list[float] = []
    frames_used = 0
    bg_half_width = max(int(cfg.bg_half_width_bins), int(cfg.guard_bins) + ridge_bins + 4)
    expanded = False

    def collect_with_half_width(half_width: int) -> tuple[list[float], list[float], int]:
        ridge: list[float] = []
        bg: list[float] = []
        used = 0
        guard = max(0, int(cfg.guard_bins))
        for fi, t in enumerate(frame_time_abs):
            track_freq = float(event.freq_hz) + float(event.drift_hz_per_s) * (float(t) - ref_time)
            center = int(np.argmin(np.abs(freq - track_freq)))
            snap = max(0, int(getattr(cfg, "ridge_snap_half_width_bins", 0)))
            if snap > 0:
                s0 = max(0, center - snap)
                s1 = min(nfft, center + snap + 1)
                if s1 > s0:
                    center = s0 + int(np.argmax(power[fi, s0:s1]))
            if center < 0 or center >= nfft:
                continue
            r0 = max(0, center - half_low)
            r1 = min(nfft, center + half_high)
            if r1 <= r0:
                continue
            b0 = max(0, center - half_width)
            b1 = min(nfft, center + half_width + 1)
            g0 = max(0, r0 - guard)
            g1 = min(nfft, r1 + guard)
            if b1 <= b0:
                continue
            row = power[fi]
            ridge.extend(float(v) for v in row[r0:r1])
            if b0 < g0:
                bg.extend(float(v) for v in row[b0:g0])
            if g1 < b1:
                bg.extend(float(v) for v in row[g1:b1])
            used += 1
        return ridge, bg, used

    ridge_values, background_values, frames_used = collect_with_half_width(bg_half_width)
    if len(background_values) < int(cfg.min_background_pixels):
        expanded = True
        # Recompute with nearly the full coarse channel but keep ridge/guard excluded.
        ridge_values, background_values, frames_used = collect_with_half_width(nfft // 2)

    if len(ridge_values) == 0:
        return _blank_measurement(cfg, "ridge mask contained no pixels")
    if len(background_values) < int(cfg.min_background_pixels):
        out = _blank_measurement(
            cfg,
            f"insufficient background pixels: {len(background_values)} < {int(cfg.min_background_pixels)}",
        )
        out["recovered_ridge_pixel_count"] = len(ridge_values)
        out["recovered_background_pixel_count"] = len(background_values)
        out["recovered_ridge_width_bins"] = ridge_bins
        out["recovered_ridge_snap_half_width_bins"] = int(getattr(cfg, "ridge_snap_half_width_bins", 0))
        out["recovered_reference_bandwidth_hz"] = float(ridge_bins * df)
        out["recovered_measurement_enbw_bins"] = enbw_bins
        return out

    ridge_arr = np.asarray(ridge_values, dtype=np.float64)
    bg_arr = np.asarray(background_values, dtype=np.float64)
    if str(cfg.noise_statistic).lower() == "mean":
        noise_floor = float(np.mean(bg_arr))
    else:
        noise_floor = float(np.median(bg_arr))
    if not math.isfinite(noise_floor) or noise_floor <= 0:
        return _blank_measurement(cfg, "invalid non-positive noise floor")

    ridge_sum = float(np.sum(ridge_arr))
    ridge_count = int(ridge_arr.size)
    ridge_mean = float(ridge_sum / max(ridge_count, 1))
    snap_bias_factor = _snap_noise_bias_factor(cfg)
    # If we snap to the brightest nearby bin, subtract the expected selected-bin
    # noise contribution rather than the ordinary background median. This avoids
    # turning a pure-noise maximum into a false positive recovered SNR.
    selected_noise_floor = noise_floor * snap_bias_factor
    ridge_excess_mean = ridge_mean - selected_noise_floor
    band_excess_total = ridge_sum - selected_noise_floor * ridge_count
    same_band_noise_total = noise_floor * ridge_count
    band_ratio = band_excess_total / same_band_noise_total if same_band_noise_total > 0 else None
    pixel_ratio = ridge_excess_mean / noise_floor if noise_floor > 0 else None

    warning = ""
    if expanded:
        warning = "background window expanded to satisfy min_background_pixels"
    if ridge_excess_mean <= 0 or band_excess_total <= 0:
        warning = (warning + "; " if warning else "") + "non-positive signal excess"

    # ``ridge_pixel`` is literal average waterfall pixel excess. ``band_excess``
    # applies the window ENBW correction and is the field intended for comparing
    # against injected local_psd SNR.
    ridge_pixel_db = _db10(pixel_ratio)
    band_db = _db10(None if band_ratio is None else band_ratio * enbw_bins)

    out = {
        "snr_measurement_method": str(cfg.method),
        "recovered_ridge_pixel_snr_db": ridge_pixel_db,
        "recovered_band_excess_snr_db": band_db,
        "recovered_local_noise_floor_power": noise_floor,
        "recovered_signal_excess_power": band_excess_total / max(frames_used, 1),
        "recovered_background_pixel_count": int(bg_arr.size),
        "recovered_ridge_pixel_count": ridge_count,
        "recovered_ridge_width_bins": int(ridge_bins),
        "recovered_ridge_snap_half_width_bins": int(getattr(cfg, "ridge_snap_half_width_bins", 0)),
        "recovered_reference_bandwidth_hz": float(ridge_bins * df),
        "recovered_measurement_enbw_bins": enbw_bins,
        "recovered_snap_noise_bias_factor": snap_bias_factor,
        "recovered_mean_ridge_power": ridge_mean,
        "recovered_mean_ridge_excess_power": ridge_excess_mean,
        "recovered_same_band_noise_power": same_band_noise_total / max(frames_used, 1),
        "recovered_measurement_warning": warning,
    }
    return {k: _finite_or_none(v) for k, v in out.items()}


def measurement_field_names() -> list[str]:
    """Stable field order for review/index and injection-comparison outputs."""
    return [
        "snr_measurement_method",
        "recovered_ridge_pixel_snr_db",
        "recovered_band_excess_snr_db",
        "recovered_local_noise_floor_power",
        "recovered_signal_excess_power",
        "recovered_background_pixel_count",
        "recovered_ridge_pixel_count",
        "recovered_ridge_width_bins",
        "recovered_reference_bandwidth_hz",
        "recovered_ridge_snap_half_width_bins",
        "recovered_measurement_enbw_bins",
        "recovered_snap_noise_bias_factor",
        "recovered_mean_ridge_power",
        "recovered_mean_ridge_excess_power",
        "recovered_same_band_noise_power",
        "recovered_measurement_warning",
    ]
