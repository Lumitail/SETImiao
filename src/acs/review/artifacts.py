from __future__ import annotations
import csv
import json
import math
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..frontend.stft import build_spectrogram_tile, frame_times_s, get_window
from ..io.dat_reader import decode_words_to_complex64, fine_offsets_hz
from ..io.stitcher import read_channel_rows
from ..preproc.baseline import apply_baseline_and_masks
from ..types import BaselineConfig, Event, MeasurementConfig, ReviewConfig, SpectrogramTile, STFTConfig, StitchedObservation
from .measurement import measure_event_snr


def _cutout(tile: SpectrogramTile, event: Event, cfg: ReviewConfig):
    flat_freq = tile.fine_freq_hz.reshape(-1)
    if event.freq_hz is None:
        center_idx = len(flat_freq) // 2
    else:
        center_idx = int(np.argmin(np.abs(flat_freq - event.freq_hz)))
    # Use scalar local normalization for review. Per-frequency time-median
    # normalization can self-subtract continuous, near-zero-drift narrowband
    # signals because the signal occupies the same bin for most of the event.
    raw_power = np.asarray(tile.power, dtype=np.float32).reshape(tile.power.shape[0], -1)
    med = float(np.nanmedian(raw_power[np.isfinite(raw_power)])) if raw_power.size else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    power2d = 10.0 * np.log10(raw_power / med + 1e-12)
    frame_center = len(tile.frame_times_s) // 2
    fs = max(0, frame_center - cfg.cutout_frames // 2)
    fe = min(len(tile.frame_times_s), fs + cfg.cutout_frames)
    bs = max(0, center_idx - cfg.cutout_bins // 2)
    be = min(power2d.shape[1], bs + cfg.cutout_bins)
    cut = power2d[fs:fe, bs:be]
    vmin = float(np.percentile(cut, 3.0))
    vmax = float(np.percentile(cut, 99.0))
    return cut, flat_freq[bs:be], tile.frame_times_s[fs:fe], vmin, vmax


def _read_single_channel_complex(obs: StitchedObservation, row0: int, row1: int, coarse_channel: int) -> np.ndarray:
    words = read_channel_rows(obs, row0, row1, coarse_channel)
    words2d = np.asarray(words, dtype=np.uint16).reshape(-1, 1)
    return decode_words_to_complex64(words2d)[:, 0]



def _choose_event_channel(obs: StitchedObservation, event: Event) -> int:
    coarse_centers = obs.freq_map.coarse_centers_hz
    if event.representative_coarse_channel is not None:
        return int(np.clip(event.representative_coarse_channel, 0, len(coarse_centers) - 1))
    if event.freq_hz is None:
        return len(coarse_centers) // 2
    return int(np.argmin(np.abs(coarse_centers - float(event.freq_hz))))


def _overview_cutout(obs: StitchedObservation, event: Event, cfg: ReviewConfig, stft_cfg: STFTConfig):
    row0 = int(event.row0)
    row1 = int(event.row1)
    if row1 - row0 < stft_cfg.nfft:
        row1 = min(obs.total_rows, row0 + stft_cfg.nfft)
        row0 = max(0, row1 - stft_cfg.nfft)
    coarse_centers = obs.freq_map.coarse_centers_hz
    if event.representative_coarse_channel is not None:
        coarse_channel = int(np.clip(event.representative_coarse_channel, 0, len(coarse_centers) - 1))
    elif event.freq_hz is None:
        coarse_channel = len(coarse_centers) // 2
    else:
        coarse_channel = int(np.argmin(np.abs(coarse_centers - event.freq_hz)))
    x = _read_single_channel_complex(obs, row0, row1, coarse_channel).astype(np.complex64, copy=False)
    if x.size < stft_cfg.nfft:
        raise ValueError("Event overview shorter than NFFT")
    x = x - x.mean()
    n_rows = len(x)
    hop = int(stft_cfg.hop)
    max_frames = max(int(cfg.overview_max_frames), 1)
    if n_rows > stft_cfg.nfft and max_frames > 1:
        est_hop = math.ceil((n_rows - stft_cfg.nfft) / max(max_frames - 1, 1))
        hop = max(hop, est_hop)
    n_frames = 1 + (n_rows - stft_cfg.nfft) // hop
    win = get_window(stft_cfg.window, stft_cfg.nfft)
    power = np.empty((n_frames, stft_cfg.nfft), dtype=np.float32)
    for fi in range(n_frames):
        s = fi * hop
        seg = x[s:s + stft_cfg.nfft] * win
        spec = np.fft.fftshift(np.fft.fft(seg))
        power[fi] = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)
    # Use scalar local normalization for the review overview.  A per-bin
    # time-median flattening erases persistent/low-drift continuous tones by
    # treating them as part of the baseline.
    med_power = float(np.nanmedian(power[np.isfinite(power)])) if power.size else 1.0
    if not np.isfinite(med_power) or med_power <= 0:
        med_power = 1.0
    flat = 10.0 * np.log10(power / med_power + 1e-12)
    freq = coarse_centers[coarse_channel] + fine_offsets_hz(obs.contract, stft_cfg.nfft)
    time = row0 * obs.contract.native_dt_s + frame_times_s(n_frames, hop, stft_cfg.nfft, obs.contract.coarse_df_hz)

    # v1.1.3X: do not blindly center only on event.freq_hz. For long or
    # drifting events, the track can span multiple bins. Select a frequency
    # window that covers the predicted track over the whole overview, with
    # enough guard bins for manual inspection.
    if event.freq_hz is None:
        center_idx = len(freq) // 2
        bs = max(0, center_idx - int(cfg.cutout_bins) // 2)
        be = min(len(freq), bs + int(cfg.cutout_bins))
    else:
        track = _predicted_track_freq_hz(event, obs, time)
        if track is None or len(track) == 0:
            center_idx = int(np.argmin(np.abs(freq - event.freq_hz)))
            track_lo = track_hi = center_idx
        else:
            finite = np.asarray(track)[np.isfinite(track)]
            if finite.size == 0:
                center_idx = int(np.argmin(np.abs(freq - event.freq_hz)))
                track_lo = track_hi = center_idx
            else:
                track_lo = int(np.argmin(np.abs(freq - float(np.min(finite)))))
                track_hi = int(np.argmin(np.abs(freq - float(np.max(finite)))))
                if track_hi < track_lo:
                    track_lo, track_hi = track_hi, track_lo
                center_idx = int(round(0.5 * (track_lo + track_hi)))
        min_bins = min(max(int(cfg.cutout_bins), 128), len(freq))
        track_span = max(1, track_hi - track_lo + 1)
        margin = max(min_bins // 2, int(getattr(cfg, "aligned_half_width_bins", 48)) + 8)
        bins = min(len(freq), max(min_bins, track_span + 2 * margin))
        bs = max(0, min(center_idx - bins // 2, len(freq) - bins))
        be = min(len(freq), bs + bins)
    cut = flat[:, bs:be]
    finite_cut = cut[np.isfinite(cut)]
    if finite_cut.size:
        vmin = float(np.percentile(finite_cut, 2.0))
        vmax = float(np.percentile(finite_cut, 99.5))
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = -1.0, 1.0
    return cut, freq[bs:be], time, vmin, vmax, coarse_channel

def _local_preview(obs: StitchedObservation, event: Event, cfg: ReviewConfig, stft_cfg: STFTConfig, baseline_cfg: BaselineConfig):
    preview_rows = max(stft_cfg.nfft, (cfg.cutout_frames - 1) * stft_cfg.hop + stft_cfg.nfft)
    center_row = int(event.peak_row) if event.peak_row is not None else int(0.5 * (event.row0 + event.row1))
    start = max(0, center_row - preview_rows // 2)
    end = min(obs.total_rows, start + preview_rows)
    start = max(0, end - preview_rows)
    # Review only needs the representative coarse channel. Building a 256-channel
    # local preview for every candidate dominated review-build time on validation
    # sweeps, and it is unnecessary for a single-candidate cutout.
    coarse_channel = _choose_event_channel(obs, event)
    tile = build_spectrogram_tile(obs, start, end, stft_cfg, channel_indices=(coarse_channel,))
    tile = apply_baseline_and_masks(tile, baseline_cfg)
    cut, freq, time_local, vmin, vmax = _cutout(tile, event, cfg)
    time_abs = time_local + start * obs.contract.native_dt_s
    return cut, freq, time_abs, vmin, vmax


def _predicted_track_freq_hz(event: Event, obs: StitchedObservation, time_s: np.ndarray) -> np.ndarray | None:
    if event.freq_hz is None:
        return None
    ref_row = int(event.peak_row) if event.peak_row is not None else int(round(0.5 * (event.row0 + event.row1)))
    ref_t = ref_row * obs.contract.native_dt_s
    return np.asarray(event.freq_hz, dtype=np.float64) + float(event.drift_hz_per_s) * (np.asarray(time_s, dtype=np.float64) - ref_t)


def _truth_track_freq_hz(truth: dict | None, time_s: np.ndarray) -> np.ndarray | None:
    """Return injected truth track frequencies for the times where it is active.

    This is used only in injection-validation review plots. Real survey review
    plots remain driven by the recovered event track.
    """
    if not truth:
        return None
    try:
        start_s = float(truth["injected_start_s"])
        duration_s = float(truth["injected_duration_s"])
        f0 = float(truth["injected_start_freq_hz"])
        drift = float(truth["injected_drift_hz_per_s"])
    except Exception:
        return None
    t = np.asarray(time_s, dtype=np.float64)
    out = f0 + drift * (t - start_s)
    active = (t >= start_s) & (t <= start_s + duration_s)
    out = out.astype(np.float64, copy=True)
    out[~active] = np.nan
    return out


def _track_aligned_cutout(
    cut: np.ndarray,
    freq_hz: np.ndarray,
    time_s: np.ndarray,
    event: Event,
    obs: StitchedObservation,
    cfg: ReviewConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, dict]:
    """De-drift a waterfall cutout so the recovered event ridge is vertical.

    The image itself is still the ordinary flattened dB waterfall, but the mean
    aligned profile is computed in *linear power-ratio space* before conversion
    back to dB. Averaging dB pixels can hide weak continuous tracks; averaging
    linear power is the correct visual analogue of an incoherent track stack.
    """
    half = max(4, int(getattr(cfg, "aligned_half_width_bins", 48)))
    width = 2 * half + 1
    aligned = np.full((len(time_s), width), np.nan, dtype=np.float32)
    blank_diag = {
        "aligned_profile_peak_excess_db": None,
        "aligned_profile_center_excess_db": None,
        "aligned_profile_peak_total_db": None,
        "aligned_profile_center_total_db": None,
        "aligned_profile_background_median_db": None,
        "aligned_profile_peak_offset_hz": None,
        "aligned_half_width_bins": half,
    }
    if len(freq_hz) < 2 or len(time_s) == 0 or event.freq_hz is None:
        offsets = np.arange(-half, half + 1, dtype=np.float64)
        return aligned, offsets, np.full(width, np.nan), -1.0, 1.0, blank_diag

    df = float(np.nanmedian(np.diff(freq_hz)))
    offsets_hz = (np.arange(width, dtype=np.float64) - half) * df
    track = _predicted_track_freq_hz(event, obs, time_s)
    if track is not None:
        for i, tf in enumerate(track):
            if not np.isfinite(tf):
                continue
            center = int(np.argmin(np.abs(freq_hz - float(tf))))
            src0 = center - half
            src1 = center + half + 1
            dst0 = max(0, -src0)
            dst1 = width - max(0, src1 - len(freq_hz))
            s0 = max(0, src0)
            s1 = min(len(freq_hz), src1)
            if s1 > s0 and dst1 > dst0:
                aligned[i, dst0:dst1] = cut[i, s0:s1]

    finite = aligned[np.isfinite(aligned)]
    if finite.size:
        vmin = float(np.percentile(finite, 2.0))
        vmax = float(np.percentile(finite, 99.5))
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = -1.0, 1.0

    # Convert flattened dB pixels to linear ratios before stacking. This makes
    # the profile sensitive to persistent weak excess along the whole event.
    with np.errstate(invalid="ignore"):
        aligned_ratio = np.power(10.0, aligned.astype(np.float64) / 10.0)
        profile_ratio = np.nanmean(aligned_ratio, axis=0)

    guard = max(1, int(getattr(cfg, "aligned_profile_guard_bins", 4)))
    bg_mask = np.ones(width, dtype=bool)
    bg_mask[max(0, half - guard):min(width, half + guard + 1)] = False
    bg_vals = profile_ratio[bg_mask & np.isfinite(profile_ratio)]
    if bg_vals.size == 0 or not np.any(np.isfinite(bg_vals)):
        return aligned, offsets_hz, np.full(width, np.nan), vmin, vmax, blank_diag
    bg_med = float(np.nanmedian(bg_vals))
    if not np.isfinite(bg_med) or bg_med <= 0:
        return aligned, offsets_hz, np.full(width, np.nan), vmin, vmax, blank_diag

    profile_total_db = 10.0 * np.log10(np.maximum(profile_ratio / bg_med, 1e-12))
    center_region = slice(max(0, half - 1), min(width, half + 2))
    near_slice = slice(max(0, half - guard), min(width, half + guard + 1))
    near = profile_ratio[near_slice]
    peak_excess = None
    peak_total = None
    center_excess = None
    center_total = None
    peak_offset = None
    if np.any(np.isfinite(near)):
        local_peak_idx = int(np.nanargmax(near)) + max(0, half - guard)
        peak_ratio = float(profile_ratio[local_peak_idx])
        peak_total = float(10.0 * np.log10(max(peak_ratio / bg_med, 1e-12)))
        excess = peak_ratio - bg_med
        if excess > 0:
            peak_excess = float(10.0 * np.log10(excess / bg_med))
        peak_offset = float(offsets_hz[local_peak_idx])
    center_vals = profile_ratio[center_region]
    if np.any(np.isfinite(center_vals)):
        center_ratio = float(np.nanmean(center_vals))
        center_total = float(10.0 * np.log10(max(center_ratio / bg_med, 1e-12)))
        excess = center_ratio - bg_med
        if excess > 0:
            center_excess = float(10.0 * np.log10(excess / bg_med))
    diagnostics = {
        "aligned_profile_peak_excess_db": peak_excess,
        "aligned_profile_center_excess_db": center_excess,
        "aligned_profile_peak_total_db": peak_total,
        "aligned_profile_center_total_db": center_total,
        "aligned_profile_background_median_db": float(10.0 * np.log10(bg_med)),
        "aligned_profile_peak_offset_hz": peak_offset,
        "aligned_half_width_bins": half,
    }
    return aligned, offsets_hz, profile_total_db, vmin, vmax, diagnostics



def _render_from_observation(obs: StitchedObservation, event: Event, review_dir: str | Path, cfg: ReviewConfig, stft_cfg: STFTConfig, baseline_cfg: BaselineConfig, measurement_cfg: MeasurementConfig | None = None, reference_width_hz: float | None = None, reference_truth: dict | None = None) -> dict:
    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    overview_cut, overview_freq, overview_time, overview_vmin, overview_vmax, coarse_channel = _overview_cutout(obs, event, cfg, stft_cfg)
    local_cut, local_freq, local_time, local_vmin, local_vmax = _local_preview(obs, event, cfg, stft_cfg, baseline_cfg)
    measurement_payload = {}
    if measurement_cfg is not None:
        measurement_payload = measure_event_snr(
            obs,
            event,
            measurement_cfg,
            stft_cfg,
            reference_width_hz=reference_width_hz,
        )

    show_aligned = bool(getattr(cfg, "show_track_aligned", True))
    aligned_payload = {}
    if show_aligned:
        aligned_cut, aligned_offsets_hz, aligned_profile, aligned_vmin, aligned_vmax, aligned_payload = _track_aligned_cutout(
            overview_cut,
            overview_freq,
            overview_time,
            event,
            obs,
            cfg,
        )
        fig, axes = plt.subplots(
            4,
            1,
            figsize=(11.5, 13.0),
            constrained_layout=True,
            height_ratios=[1.35, 1.10, 0.95, 0.65],
        )
    else:
        aligned_cut = aligned_offsets_hz = aligned_profile = None
        aligned_vmin = aligned_vmax = None
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True, height_ratios=[1.4, 1.0, 0.8])

    overview_extent = [overview_freq[0] / 1e3, overview_freq[-1] / 1e3, overview_time[0], overview_time[-1]]
    im0 = axes[0].imshow(
        overview_cut,
        aspect="auto",
        origin="lower",
        extent=overview_extent,
        cmap="viridis",
        vmin=overview_vmin,
        vmax=overview_vmax,
    )
    fig.colorbar(im0, ax=axes[0], label="Overview local-normalized power (dB)")
    track_overview = _predicted_track_freq_hz(event, obs, overview_time)
    if track_overview is not None:
        axes[0].plot(track_overview / 1e3, overview_time, ls='--', lw=1.25, color='white', alpha=0.95, label='recovered track')
    truth_overview = _truth_track_freq_hz(reference_truth, overview_time)
    if truth_overview is not None and np.any(np.isfinite(truth_overview)):
        axes[0].plot(truth_overview / 1e3, overview_time, ls=':', lw=1.15, color='red', alpha=0.9, label='injected truth')
        axes[0].legend(loc='upper left', fontsize=8, framealpha=0.7)
    axes[0].set_xlim(overview_extent[0], overview_extent[1])
    axes[0].set_ylim(overview_extent[2], overview_extent[3])
    axes[0].set_title(f"{event.event_id} | score={event.score:.2f} | candidate={event.candidate_passed}")
    axes[0].set_ylabel("Observation time (s)")

    if show_aligned and aligned_cut is not None and aligned_offsets_hz is not None:
        aligned_extent = [aligned_offsets_hz[0], aligned_offsets_hz[-1], overview_time[0], overview_time[-1]]
        im_al = axes[1].imshow(
            aligned_cut,
            aspect="auto",
            origin="lower",
            extent=aligned_extent,
            cmap="viridis",
            vmin=aligned_vmin,
            vmax=aligned_vmax,
        )
        fig.colorbar(im_al, ax=axes[1], label="Aligned local-normalized power (dB)")
        axes[1].axvline(0.0, ls='--', lw=1.0, color='white', alpha=0.95, label='recovered track')
        if truth_overview is not None and track_overview is not None:
            truth_offset = truth_overview - track_overview
            if np.any(np.isfinite(truth_offset)):
                axes[1].plot(truth_offset, overview_time, ls=':', lw=1.15, color='red', alpha=0.9, label='truth offset')
                axes[1].legend(loc='upper left', fontsize=8, framealpha=0.7)
        axes[1].set_xlim(aligned_extent[0], aligned_extent[1])
        axes[1].set_ylim(aligned_extent[2], aligned_extent[3])
        axes[1].set_ylabel("Observation time (s)")
        axes[1].set_title("Track-aligned overview: recovered track is fixed at 0 Hz")

        local_ax = axes[2]
        profile_ax = axes[3]
    else:
        local_ax = axes[1]
        profile_ax = axes[2]

    local_extent = [local_freq[0] / 1e3, local_freq[-1] / 1e3, local_time[0], local_time[-1]]
    im1 = local_ax.imshow(
        local_cut,
        aspect="auto",
        origin="lower",
        extent=local_extent,
        cmap="viridis",
        vmin=local_vmin,
        vmax=local_vmax,
    )
    fig.colorbar(im1, ax=local_ax, label="Local local-normalized power (dB)")
    track_local = _predicted_track_freq_hz(event, obs, local_time)
    if track_local is not None:
        local_ax.plot(track_local / 1e3, local_time, ls='--', lw=1.0, color='white', alpha=0.9)
    truth_local = _truth_track_freq_hz(reference_truth, local_time)
    if truth_local is not None and np.any(np.isfinite(truth_local)):
        local_ax.plot(truth_local / 1e3, local_time, ls=':', lw=1.1, color='red', alpha=0.9)
    local_ax.set_xlim(local_extent[0], local_extent[1])
    local_ax.set_ylim(local_extent[2], local_extent[3])
    local_ax.set_ylabel("Observation time (s)")
    local_ax.set_title("Local preview around event peak")

    if show_aligned and aligned_profile is not None and aligned_offsets_hz is not None:
        profile_ax.plot(aligned_offsets_hz, aligned_profile, lw=1.2)
        profile_ax.axhline(0.0, ls=':', lw=0.8)
        profile_ax.axvline(0.0, ls='--', lw=1.0)
        profile_ax.set_xlabel("Frequency offset from recovered track (Hz)")
        profile_ax.set_ylabel("Mean aligned total power / bg (dB)")
        profile_ax.set_title(
            "Linear-power track stack "
            f"(center_excess={aligned_payload.get('aligned_profile_center_excess_db')}, "
            f"peak_excess={aligned_payload.get('aligned_profile_peak_excess_db')})"
        )
    else:
        profile_ax.plot(local_freq / 1e3, np.nanmean(local_cut, axis=0), lw=1.2)
        if event.freq_hz is not None:
            profile_ax.axvline(event.freq_hz / 1e3, ls='--', lw=1.0)
        profile_ax.set_xlabel("Frequency (kHz)")
        profile_ax.set_ylabel("Mean local power (dB)")

    event_t0 = float(event.row0) * obs.contract.native_dt_s
    event_t1 = float(event.row1) * obs.contract.native_dt_s
    peak_time_s = None if event.peak_row is None else float(event.peak_row) * obs.contract.native_dt_s
    meta_text = (
        f"obs_id={event.obs_id}\nbeam={event.beam_id}\npol={event.pol_id}\n"
        f"scan={event.scan_id}\ntarget={event.target_id}\n"
        f"event_start_s={event_t0:.6f}\nevent_end_s={event_t1:.6f}\nevent_duration_s={event_t1 - event_t0:.6f}\n"
        f"peak_time_s={peak_time_s}\n"
        f"overview_coarse_channel={coarse_channel}\n"
        f"freq_hz={event.freq_hz}\ndrift_hz_per_s={event.drift_hz_per_s}\n"
        f"best_incoh_search_metric_db={event.best_incoherent_snr:.2f}\n"
        f"best_refined_search_metric_db={event.best_refined_snr}\n"
        f"recovered_band_excess_snr_db={measurement_payload.get('recovered_band_excess_snr_db')}\n"
        f"recovered_ridge_pixel_snr_db={measurement_payload.get('recovered_ridge_pixel_snr_db')}\n"
        f"aligned_peak_excess_db={aligned_payload.get('aligned_profile_peak_excess_db')}\n"
        f"aligned_center_excess_db={aligned_payload.get('aligned_profile_center_excess_db')}\n"
        f"aligned_center_total_db={aligned_payload.get('aligned_profile_center_total_db')}\n"
        f"review_visual_evidence_passed={aligned_payload.get('review_visual_evidence_passed')}\n"
        f"review_visual_evidence_warning={aligned_payload.get('review_visual_evidence_warning')}\n"
        f"best_width={event.best_width_bins}\nsupport={event.best_support_fraction:.2f}\n"
        f"gain_db={event.best_coherent_gain_db}\ncandidate={event.candidate_passed}\n"
        f"candidate_reasons={list(event.candidate_reasons)}"
    )
    axes[0].text(1.01, 0.98, meta_text, transform=axes[0].transAxes, va='top', ha='left', fontsize=9)

    png = review_dir / f"{event.event_id}.png"
    pdf = review_dir / f"{event.event_id}.pdf"
    js = review_dir / f"{event.event_id}.json"
    fig.savefig(png, dpi=140, bbox_inches='tight')
    if getattr(cfg, 'write_pdf', True):
        fig.savefig(pdf, dpi=140, bbox_inches='tight')
        artifact_pdf = pdf.name
    else:
        artifact_pdf = None
    plt.close(fig)
    plt.close('all')
    gc.collect()

    payload = event.to_dict()
    payload["event_start_s"] = event_t0
    payload["event_end_s"] = event_t1
    payload["event_duration_s"] = event_t1 - event_t0
    payload["overview_time0_s"] = float(overview_time[0])
    payload["overview_time1_s"] = float(overview_time[-1])
    payload["local_preview_time0_s"] = float(local_time[0])
    payload["local_preview_time1_s"] = float(local_time[-1])
    payload["peak_time_s"] = None if peak_time_s is None else float(peak_time_s)
    payload["display_coarse_channel"] = int(coarse_channel)
    payload["best_incoherent_search_metric_db"] = float(event.best_incoherent_snr)
    payload["best_refined_search_metric_db"] = None if event.best_refined_snr is None else float(event.best_refined_snr)
    payload["snr_metric_kind"] = "search_metric_db_not_time_domain_injection_snr"
    payload["snr_metric_note"] = (
        "best_incoherent_snr/best_refined_snr are detector ranking metrics after "
        "STFT path integration and coherent dechirp FFT. They are not the same "
        "quantity as YAML injection snr_db. recovered_*_snr_db fields are post-detection "
        "local-background measurements along the event ridge."
    )
    # v1.1.4x: annotate whether the candidate is visibly supported in the
    # aligned stack.  This is a review diagnostic, not a replacement for search
    # metrics.  It helps identify coherent-noise candidates whose waterfall and
    # aligned profile do not show a persistent narrowband ridge.
    peak_excess = aligned_payload.get("aligned_profile_peak_excess_db")
    center_excess = aligned_payload.get("aligned_profile_center_excess_db")
    min_peak = float(getattr(cfg, "visual_evidence_min_peak_excess_db", 0.0))
    min_center = float(getattr(cfg, "visual_evidence_min_center_excess_db", -3.0))
    visual_pass = False
    if peak_excess is not None and np.isfinite(float(peak_excess)) and float(peak_excess) >= min_peak:
        visual_pass = True
    if center_excess is not None and np.isfinite(float(center_excess)) and float(center_excess) >= min_center:
        visual_pass = True
    aligned_payload["review_visual_evidence_passed"] = bool(visual_pass)
    if event.candidate_passed and not visual_pass:
        aligned_payload["review_visual_evidence_warning"] = (
            "candidate has weak or negative aligned-profile evidence; inspect all-event review "
            "and search metrics before treating as a persistent narrowband signal"
        )
    else:
        aligned_payload["review_visual_evidence_warning"] = ""

    payload.update(measurement_payload)
    payload.update(aligned_payload)
    payload["artifact_png"] = png.name
    payload["artifact_pdf"] = artifact_pdf
    js.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload


def render_event_artifact(source, event: Event, review_dir: str | Path, cfg: ReviewConfig, stft_cfg: STFTConfig | None = None, baseline_cfg: BaselineConfig | None = None, measurement_cfg: MeasurementConfig | None = None, reference_width_hz: float | None = None, reference_truth: dict | None = None) -> dict:
    if isinstance(source, SpectrogramTile) or stft_cfg is None or baseline_cfg is None:
        tile = source
        review_dir = Path(review_dir)
        review_dir.mkdir(parents=True, exist_ok=True)
        cut, freq, time, vmin, vmax = _cutout(tile, event, cfg)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
        im = axes[0].imshow(cut, aspect="auto", origin="lower",
                            extent=[freq[0]/1e3, freq[-1]/1e3, time[0], time[-1]], cmap="viridis", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axes[0], label="Flattened power (dB)")
        title = f"{event.event_id} | score={event.score:.2f} | candidate={event.candidate_passed}"
        axes[0].set_title(title)
        axes[0].set_ylabel("Local preview time (s)")
        axes[1].plot(freq/1e3, np.nanmean(cut, axis=0), lw=1.2)
        if event.freq_hz is not None:
            axes[1].axvline(event.freq_hz/1e3, ls='--', lw=1.0)
        axes[1].set_xlabel("Frequency (kHz)")
        axes[1].set_ylabel("Mean cutout power (dB)")
        png = review_dir / f"{event.event_id}.png"
        pdf = review_dir / f"{event.event_id}.pdf"
        js = review_dir / f"{event.event_id}.json"
        fig.savefig(png, dpi=140, bbox_inches='tight')
        if getattr(cfg, 'write_pdf', True):
            fig.savefig(pdf, dpi=140, bbox_inches='tight')
            artifact_pdf = pdf.name
        else:
            artifact_pdf = None
        plt.close(fig)
        plt.close('all')
        gc.collect()
        payload = event.to_dict()
        payload["best_incoherent_search_metric_db"] = float(event.best_incoherent_snr)
        payload["best_refined_search_metric_db"] = None if event.best_refined_snr is None else float(event.best_refined_snr)
        payload["snr_metric_kind"] = "search_metric_db_not_time_domain_injection_snr"
        payload["snr_metric_note"] = (
            "best_incoherent_snr/best_refined_snr are detector ranking metrics after "
            "STFT path integration and coherent dechirp FFT. They are not the same "
            "quantity as YAML injection snr_db. Observation-backed review artifacts "
            "also include recovered_*_snr_db local-background measurements."
        )
        payload["artifact_png"] = png.name
        payload["artifact_pdf"] = artifact_pdf
        js.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        return payload
    return _render_from_observation(source, event, review_dir, cfg, stft_cfg, baseline_cfg, measurement_cfg, reference_width_hz, reference_truth)


def build_review_index(review_dir: str | Path, artifacts: list[dict]) -> None:
    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / 'index.json').write_text(json.dumps(artifacts, indent=2), encoding='utf-8')
    if artifacts:
        fieldnames: list[str] = []
        seen: set[str] = set()
        for item in artifacts:
            for key in item.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
        with open(review_dir / 'index.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader()
            w.writerows(artifacts)
    lines = [
        "<html><head><meta charset='utf-8'><title>ACS Review Index</title></head><body>",
        "<h1>ACS Review Index</h1>",
        "<table border='1' cellspacing='0' cellpadding='6'>",
        "<tr><th>event_id</th><th>score</th><th>candidate_passed</th><th>freq_hz</th><th>drift_hz_per_s</th><th>event_duration_s</th><th>preview</th></tr>",
    ]
    for item in artifacts:
        lines.append(
            f"<tr><td>{item.get('event_id')}</td><td>{item.get('score')}</td><td>{item.get('candidate_passed')}</td><td>{item.get('freq_hz')}</td><td>{item.get('drift_hz_per_s')}</td><td>{item.get('event_duration_s')}</td><td><a href='{item.get('artifact_png')}'><img src='{item.get('artifact_png')}' width='320'></a></td></tr>"
        )
    lines += ["</table>", "</body></html>"]
    (review_dir / 'index.html').write_text("\n".join(lines), encoding='utf-8')
