from __future__ import annotations
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..frontend.stft import build_spectrogram_tile, frame_times_s, get_window
from ..io.dat_reader import decode_words_to_complex64, fine_offsets_hz
from ..io.stitcher import read_channel_rows
from ..preproc.baseline import apply_baseline_and_masks
from ..types import BaselineConfig, Event, ReviewConfig, SpectrogramTile, STFTConfig, StitchedObservation


def _cutout(tile: SpectrogramTile, event: Event, cfg: ReviewConfig):
    flat_freq = tile.fine_freq_hz.reshape(-1)
    if event.freq_hz is None:
        center_idx = len(flat_freq) // 2
    else:
        center_idx = int(np.argmin(np.abs(flat_freq - event.freq_hz)))
    power2d = 10.0 * np.log10(tile.norm_power.reshape(tile.norm_power.shape[0], -1) + 1e-12)
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
    power_db = 10.0 * np.log10(power + 1e-12)
    flat = power_db - np.median(power_db, axis=0, keepdims=True)
    freq = coarse_centers[coarse_channel] + fine_offsets_hz(obs.contract, stft_cfg.nfft)
    center_idx = len(freq) // 2 if event.freq_hz is None else int(np.argmin(np.abs(freq - event.freq_hz)))
    bins = min(max(int(cfg.cutout_bins), 128), len(freq))
    bs = max(0, center_idx - bins // 2)
    be = min(len(freq), bs + bins)
    cut = flat[:, bs:be]
    time = row0 * obs.contract.native_dt_s + frame_times_s(n_frames, hop, stft_cfg.nfft, obs.contract.coarse_df_hz)
    vmin = float(np.percentile(cut, 3.0))
    vmax = float(np.percentile(cut, 99.0))
    return cut, freq[bs:be], time, vmin, vmax, coarse_channel


def _local_preview(obs: StitchedObservation, event: Event, cfg: ReviewConfig, stft_cfg: STFTConfig, baseline_cfg: BaselineConfig):
    preview_rows = max(stft_cfg.nfft, (cfg.cutout_frames - 1) * stft_cfg.hop + stft_cfg.nfft)
    center_row = int(event.peak_row) if event.peak_row is not None else int(0.5 * (event.row0 + event.row1))
    start = max(0, center_row - preview_rows // 2)
    end = min(obs.total_rows, start + preview_rows)
    start = max(0, end - preview_rows)
    tile = build_spectrogram_tile(obs, start, end, stft_cfg)
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


def _render_from_observation(obs: StitchedObservation, event: Event, review_dir: str | Path, cfg: ReviewConfig, stft_cfg: STFTConfig, baseline_cfg: BaselineConfig) -> dict:
    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    overview_cut, overview_freq, overview_time, overview_vmin, overview_vmax, coarse_channel = _overview_cutout(obs, event, cfg, stft_cfg)
    local_cut, local_freq, local_time, local_vmin, local_vmax = _local_preview(obs, event, cfg, stft_cfg, baseline_cfg)

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
    fig.colorbar(im0, ax=axes[0], label="Overview flattened power (dB)")
    track_overview = _predicted_track_freq_hz(event, obs, overview_time)
    if track_overview is not None:
        axes[0].plot(track_overview / 1e3, overview_time, ls='--', lw=1.0, color='white', alpha=0.8)
    axes[0].set_xlim(overview_extent[0], overview_extent[1])
    axes[0].set_ylim(overview_extent[2], overview_extent[3])
    axes[0].set_title(f"{event.event_id} | score={event.score:.2f} | candidate={event.candidate_passed}")
    axes[0].set_ylabel("Observation time (s)")

    local_extent = [local_freq[0] / 1e3, local_freq[-1] / 1e3, local_time[0], local_time[-1]]
    im1 = axes[1].imshow(
        local_cut,
        aspect="auto",
        origin="lower",
        extent=local_extent,
        cmap="viridis",
        vmin=local_vmin,
        vmax=local_vmax,
    )
    fig.colorbar(im1, ax=axes[1], label="Local flattened power (dB)")
    track_local = _predicted_track_freq_hz(event, obs, local_time)
    if track_local is not None:
        axes[1].plot(track_local / 1e3, local_time, ls='--', lw=1.0, color='white', alpha=0.8)
    axes[1].set_xlim(local_extent[0], local_extent[1])
    axes[1].set_ylim(local_extent[2], local_extent[3])
    axes[1].set_ylabel("Observation time (s)")
    axes[1].set_title("Local preview around event peak")

    axes[2].plot(local_freq / 1e3, np.nanmean(local_cut, axis=0), lw=1.2)
    if event.freq_hz is not None:
        axes[2].axvline(event.freq_hz / 1e3, ls='--', lw=1.0)
    axes[2].set_xlabel("Frequency (kHz)")
    axes[2].set_ylabel("Mean local power (dB)")

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
        f"best_width={event.best_width_bins}\nsupport={event.best_support_fraction:.2f}\n"
        f"gain_db={event.best_coherent_gain_db}\ncandidate={event.candidate_passed}\n"
        f"candidate_reasons={list(event.candidate_reasons)}"
    )
    axes[0].text(1.01, 0.98, meta_text, transform=axes[0].transAxes, va='top', ha='left', fontsize=9)

    png = review_dir / f"{event.event_id}.png"
    pdf = review_dir / f"{event.event_id}.pdf"
    js = review_dir / f"{event.event_id}.json"
    fig.savefig(png, dpi=140, bbox_inches='tight')
    fig.savefig(pdf, dpi=140, bbox_inches='tight')
    plt.close(fig)

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
        "quantity as YAML injection snr_db, which is a raw active-sample RMS ratio."
    )
    payload["artifact_png"] = png.name
    payload["artifact_pdf"] = pdf.name
    js.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload


def render_event_artifact(source, event: Event, review_dir: str | Path, cfg: ReviewConfig, stft_cfg: STFTConfig | None = None, baseline_cfg: BaselineConfig | None = None) -> dict:
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
        fig.savefig(pdf, dpi=140, bbox_inches='tight')
        plt.close(fig)
        payload = event.to_dict()
        payload["best_incoherent_search_metric_db"] = float(event.best_incoherent_snr)
        payload["best_refined_search_metric_db"] = None if event.best_refined_snr is None else float(event.best_refined_snr)
        payload["snr_metric_kind"] = "search_metric_db_not_time_domain_injection_snr"
        payload["snr_metric_note"] = (
            "best_incoherent_snr/best_refined_snr are detector ranking metrics after "
            "STFT path integration and coherent dechirp FFT. They are not the same "
            "quantity as YAML injection snr_db, which is a raw active-sample RMS ratio."
        )
        payload["artifact_png"] = png.name
        payload["artifact_pdf"] = pdf.name
        js.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        return payload
    return _render_from_observation(source, event, review_dir, cfg, stft_cfg, baseline_cfg)


def build_review_index(review_dir: str | Path, artifacts: list[dict]) -> None:
    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / 'index.json').write_text(json.dumps(artifacts, indent=2), encoding='utf-8')
    if artifacts:
        with open(review_dir / 'index.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(artifacts[0].keys()))
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
