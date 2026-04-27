from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from acs.bench.inject import encode_complex_to_words
from acs.config import load_runtime_config
from acs.io.stitcher import build_observation_from_paths
from acs.review.artifacts import build_review_index, render_event_artifact, _overview_cutout
from acs.review.injection_compare import write_injection_comparison
from acs.review.measurement import measure_event_snr
from acs.types import DatContract, Event, MeasurementConfig, ReviewConfig, STFTConfig


def _write_synthetic_dat(path: Path, *, rows: int = 32768, channels: int = 256, signal_amp: float = 0.0, channel: int = 20, seed: int = 1, tone_bin: int = 37):
    rng = np.random.default_rng(seed)
    noise_sigma = 20.0
    x = (
        rng.normal(0.0, noise_sigma / math.sqrt(2), size=(rows, channels))
        + 1j * rng.normal(0.0, noise_sigma / math.sqrt(2), size=(rows, channels))
    ).astype(np.complex64)
    if signal_amp:
        coarse_sr_hz = DatContract().coarse_df_hz
        n = np.arange(rows, dtype=np.float64)
        freq_hz = tone_bin * (coarse_sr_hz / 2048.0)
        tone = signal_amp * np.exp(2j * np.pi * freq_hz * n / coarse_sr_hz)
        x[:, channel] += tone.astype(np.complex64)
    words = encode_complex_to_words(x)
    path.write_bytes(words.astype("<u2").tobytes())
    return noise_sigma


def _obs(path: Path, *, rows: int = 32768):
    contract = DatContract(lo_hz=1_000_000_000.0, start_coarse_channel=27392)
    return build_observation_from_paths([path], contract, "synthetic", beam_id="00", pol_id="00", scan_id="s", target_id="t")


def _event(obs, channel: int = 20, rows: int = 32768, tone_bin: int = 37):
    return Event(
        event_id="synthetic_event_00000",
        obs_id=obs.meta.obs_id,
        beam_id=obs.meta.beam_id,
        pol_id=obs.meta.pol_id,
        scan_id=obs.meta.scan_id,
        target_id=obs.meta.target_id,
        row0=0,
        row1=rows,
        freq_hz=float(obs.freq_map.coarse_centers_hz[channel] + tone_bin * (obs.contract.coarse_df_hz / 2048.0)),
        drift_hz_per_s=0.0,
        score=25.0,
        n_hits=10,
        best_incoherent_snr=20.0,
        best_refined_snr=40.0,
        best_width_bins=1,
        representative_coarse_channel=channel,
        peak_row=rows // 2,
        candidate_passed=True,
    )


def _amp_for_local_psd_snr(noise_sigma: float, nfft: int, snr_db: float) -> float:
    # width_hz == 0 means one spectral-resolution element. Local PSD SNR is
    # signal_power / (sigma^2 / nfft).
    return math.sqrt((noise_sigma * noise_sigma / nfft) * 10.0 ** (snr_db / 10.0))


def test_recovered_snr_measurement_known_synthetic_ridge(tmp_path):
    stft = STFTConfig("search", nfft=2048, hop=1024, window="hann")
    target_snr_db = 16.0
    noise_sigma = 20.0
    amp = _amp_for_local_psd_snr(noise_sigma, stft.nfft, target_snr_db)
    dat = tmp_path / "synthetic.dat"
    _write_synthetic_dat(dat, signal_amp=amp, seed=3)
    obs = _obs(dat)
    event = _event(obs)

    meas = measure_event_snr(obs, event, MeasurementConfig(), stft)
    assert meas["snr_measurement_method"] == "stft_local_background"
    assert meas["recovered_band_excess_snr_db"] is not None
    assert abs(meas["recovered_band_excess_snr_db"] - target_snr_db) < 4.0
    assert meas["recovered_background_pixel_count"] >= 256
    assert meas["recovered_ridge_pixel_count"] > 0


def test_recovered_snr_measurement_monotonic(tmp_path):
    stft = STFTConfig("search", nfft=2048, hop=1024, window="hann")
    noise_sigma = 20.0
    dat_lo = tmp_path / "lo.dat"
    dat_hi = tmp_path / "hi.dat"
    _write_synthetic_dat(dat_lo, signal_amp=_amp_for_local_psd_snr(noise_sigma, stft.nfft, 8.0), seed=4)
    _write_synthetic_dat(dat_hi, signal_amp=_amp_for_local_psd_snr(noise_sigma, stft.nfft, 18.0), seed=4)
    obs_lo = _obs(dat_lo)
    obs_hi = _obs(dat_hi)
    mlo = measure_event_snr(obs_lo, _event(obs_lo), MeasurementConfig(), stft)
    mhi = measure_event_snr(obs_hi, _event(obs_hi), MeasurementConfig(), stft)
    assert mlo["recovered_band_excess_snr_db"] is not None
    assert mhi["recovered_band_excess_snr_db"] is not None
    assert mhi["recovered_band_excess_snr_db"] > mlo["recovered_band_excess_snr_db"] + 5.0


def test_measurement_fields_appear_in_artifacts_and_index_csv(tmp_path):
    stft = STFTConfig("search", nfft=2048, hop=1024, window="hann")
    noise_sigma = 20.0
    dat = tmp_path / "artifact.dat"
    _write_synthetic_dat(dat, signal_amp=_amp_for_local_psd_snr(noise_sigma, stft.nfft, 14.0), seed=6)
    obs = _obs(dat)
    event = _event(obs)
    review_dir = tmp_path / "review"
    payload = render_event_artifact(
        obs,
        event,
        review_dir,
        ReviewConfig(write_pdf=False),
        stft,
        load_runtime_config(Path(__file__).parents[1] / "configs" / "h1_search.yaml").baseline,
        MeasurementConfig(),
    )
    build_review_index(review_dir, [payload])
    assert "recovered_band_excess_snr_db" in payload
    assert "aligned_profile_peak_excess_db" in payload
    assert payload["aligned_half_width_bins"] >= 4
    rows = list(csv.DictReader((review_dir / "index.csv").open()))
    assert rows and "recovered_band_excess_snr_db" in rows[0]
    assert "aligned_profile_peak_excess_db" in rows[0]


def test_injection_comparison_includes_recovered_snr_fields(tmp_path):
    cfg = load_runtime_config(Path(__file__).parents[1] / "configs" / "h1_search.yaml")
    report = {
        "contract": {"frontend_fs_hz": cfg.contract.frontend_fs_hz, "channelizer_fft": cfg.contract.channelizer_fft},
        "signals": [{
            "name": "sig",
            "requested": {"name": "sig", "start_s": 0.0, "duration_s": 4.0, "start_freq_hz": 1_000_000_000.0, "drift_hz_per_s": 0.0, "snr_db": 5.0},
            "resolved": {"start_s": 0.0, "duration_s": 4.0, "start_freq_hz": 1_000_000_000.0, "drift_hz_per_s": 0.0, "snr_db": 5.0, "width_hz": 0.0},
            "snr_reference": "local_psd",
            "noise_estimator": "coarse_psd",
            "requested_local_psd_snr_db": 5.0,
            "realized_local_psd_snr_db": 5.1,
            "target_channels": [{"channel_index": 0, "baseband_hz": 0.0}],
        }],
    }
    report_path = tmp_path / "injected.inject.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    artifacts = [{
        "event_id": "e0",
        "row0": 0,
        "row1": int(4.0 / cfg.contract.native_dt_s),
        "event_start_s": 0.0,
        "event_end_s": 4.0,
        "peak_time_s": 2.0,
        "freq_hz": 1_000_000_000.0,
        "drift_hz_per_s": 0.0,
        "best_incoherent_search_metric_db": 20.0,
        "best_refined_search_metric_db": 40.0,
        "recovered_band_excess_snr_db": 4.8,
        "recovered_ridge_pixel_snr_db": 3.2,
        "score": 20.0,
        "n_hits": 5,
        "artifact_png": "e0.png",
    }]
    summary = write_injection_comparison(tmp_path, report_path, artifacts, cfg)
    assert summary["n_matched"] == 1
    rows = list(csv.DictReader((tmp_path / "injection_comparison.csv").open()))
    assert "recovered_band_excess_snr_db" in rows[0]
    assert "recovered_ridge_pixel_snr_db" in rows[0]
    assert "recovered_snr_error_db" in rows[0]
    assert abs(float(rows[0]["recovered_snr_error_db"]) - (4.8 - 5.1)) < 1e-9


def test_existing_search_metric_fields_are_preserved(tmp_path):
    cfg = load_runtime_config(Path(__file__).parents[1] / "configs" / "h1_search.yaml")
    artifacts = [{
        "event_id": "e0",
        "row0": 0,
        "row1": int(4.0 / cfg.contract.native_dt_s),
        "event_start_s": 0.0,
        "event_end_s": 4.0,
        "peak_time_s": 2.0,
        "freq_hz": 1_000_000_000.0,
        "drift_hz_per_s": 0.0,
        "best_incoherent_search_metric_db": 21.0,
        "best_refined_search_metric_db": 42.0,
        "recovered_band_excess_snr_db": 5.0,
    }]
    report = {
        "contract": {"frontend_fs_hz": cfg.contract.frontend_fs_hz, "channelizer_fft": cfg.contract.channelizer_fft},
        "signals": [{
            "name": "sig",
            "requested": {"name": "sig", "start_s": 0.0, "duration_s": 4.0, "start_freq_hz": 1_000_000_000.0, "drift_hz_per_s": 0.0, "snr_db": 5.0},
            "resolved": {"start_s": 0.0, "duration_s": 4.0, "start_freq_hz": 1_000_000_000.0, "drift_hz_per_s": 0.0, "snr_db": 5.0},
            "target_channels": [{"channel_index": 0}],
        }],
    }
    report_path = tmp_path / "injected.inject.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    write_injection_comparison(tmp_path, report_path, artifacts, cfg)
    rows = list(csv.DictReader((tmp_path / "injection_comparison.csv").open()))
    assert rows[0]["best_refined_search_metric_db"] == "42.0"
    assert rows[0]["best_incoherent_search_metric_db"] == "21.0"


def test_review_overview_does_not_self_subtract_persistent_tone(tmp_path):
    """A continuous zero-drift tone should remain visible in review display data.

    Earlier review plots used per-frequency time-median flattening. That is
    appropriate for some waterfall displays but it can erase a persistent
    SETI-like tone because the tone is present in the same frequency bin for
    most frames. v1.1.3X uses scalar local normalization for review cutouts.
    """
    stft = STFTConfig("search", nfft=2048, hop=1024, window="hann")
    noise_sigma = 20.0
    dat = tmp_path / "persistent.dat"
    _write_synthetic_dat(dat, signal_amp=_amp_for_local_psd_snr(noise_sigma, stft.nfft, 18.0), seed=11, tone_bin=41)
    obs = _obs(dat)
    event = _event(obs, tone_bin=41)
    cut, freq, time, vmin, vmax, channel = _overview_cutout(obs, event, ReviewConfig(write_pdf=False), stft)
    center = int(np.argmin(np.abs(freq - float(event.freq_hz))))
    power_ratio_profile = np.nanmean(np.power(10.0, cut / 10.0), axis=0)
    bg = np.concatenate([power_ratio_profile[:max(0, center-8)], power_ratio_profile[min(len(power_ratio_profile), center+9):]])
    assert bg.size > 0
    assert power_ratio_profile[center] > np.nanmedian(bg) * 2.0
