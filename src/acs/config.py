from __future__ import annotations
from pathlib import Path
import yaml
from .types import DatContract, STFTConfig, BaselineConfig, DriftConfig, RefineConfig, CoincidenceConfig, CandidateConfig, ReviewConfig, RuntimeConfig


def _as_tuple_ints(v, default):
    if v is None:
        return tuple(default)
    return tuple(int(x) for x in v)


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    obj = yaml.safe_load(Path(path).read_text())
    c = obj.get("contract", {})
    contract = DatContract(
        channels=int(c.get("channels", 256)),
        bytes_per_cell=int(c.get("bytes_per_cell", 2)),
        frontend_fs_hz=float(c.get("frontend_fs_hz", 1_000_000_000.0)),
        channelizer_fft=int(c.get("channelizer_fft", 65536)),
        lo_hz=float(c["lo_hz"]) if c.get("lo_hz") is not None else None,
        start_coarse_channel=int(c["start_coarse_channel"]) if c.get("start_coarse_channel") is not None else None,
    )
    s = obj.get("search_stft", {})
    b = obj.get("baseline", {})
    d = obj.get("drift", {})
    r = obj.get("refine", {})
    cc = obj.get("coincidence", {})
    cand = obj.get("candidate", {})
    rv = obj.get("review", {})
    return RuntimeConfig(
        contract=contract,
        search_tile_rows=int(obj.get("search_tile_rows", 31250)),
        search_overlap_rows=int(obj.get("search_overlap_rows", 4096)),
        search_stft=STFTConfig(
            name=str(s.get("name", "search")),
            nfft=int(s.get("nfft", 2048)),
            hop=int(s.get("hop", 1024)),
            window=str(s.get("window", "hann")),
        ),
        baseline=BaselineConfig(
            smooth_bins=int(b.get("smooth_bins", 1001)),
            edge_bins_per_coarse=int(b.get("edge_bins_per_coarse", 16)),
            mask_sigma_mad=float(b.get("mask_sigma_mad", 6.0)),
            qc_display_smooth_bins=int(b.get("qc_display_smooth_bins", 257)),
            qc_low_percentile=float(b.get("qc_low_percentile", 2.0)),
            qc_high_percentile=float(b.get("qc_high_percentile", 98.0)),
        ),
        drift=DriftConfig(
            min_hz_per_s=float(d.get("min_hz_per_s", -48.0)),
            max_hz_per_s=float(d.get("max_hz_per_s", 48.0)),
            step_hz_per_s=float(d["step_hz_per_s"]) if d.get("step_hz_per_s") is not None else None,
            seed_top_k_mean_per_channel=int(d.get("seed_top_k_mean_per_channel", 6)),
            seed_top_k_max_per_channel=int(d.get("seed_top_k_max_per_channel", 6)),
            seed_min_separation_bins=int(d.get("seed_min_separation_bins", 8)),
            snr_threshold=float(d.get("snr_threshold", 6.0)),
            widths=_as_tuple_ints(d.get("widths"), (1, 3, 5, 7)),
            trial_chunk_size=int(d.get("trial_chunk_size", 64)),
            backend=str(d.get("backend", "auto")),
            support_threshold=float(d.get("support_threshold", 1.5)),
            channel_min=int(d["channel_min"]) if d.get("channel_min") is not None else None,
            channel_max=int(d["channel_max"]) if d.get("channel_max") is not None else None,
        ),
        refine=RefineConfig(
            enabled=bool(r.get("enabled", True)),
            top_n=int(r.get("top_n", 12)),
            drift_half_window_hz_per_s=float(r.get("drift_half_window_hz_per_s", 8.0)),
            drift_step_hz_per_s=float(r.get("drift_step_hz_per_s", 0.25)),
            freq_half_window_bins=int(r.get("freq_half_window_bins", 4)),
            min_refined_snr=float(r.get("min_refined_snr", 14.5)),
        ),
        coincidence=CoincidenceConfig(
            freq_tol_hz=float(cc.get("freq_tol_hz", 40.0)),
            drift_tol_hz_per_s=float(cc.get("drift_tol_hz_per_s", 4.0)),
            time_tol_s=float(cc.get("time_tol_s", 0.35)),
            multibeam_penalty=float(cc.get("multibeam_penalty", 2.5)),
            single_beam_bonus=float(cc.get("single_beam_bonus", 0.0)),
            max_row_gap_s=float(cc.get("max_row_gap_s", 2.5)),
        ),
        candidate=CandidateConfig(
            min_event_score=float(cand.get("min_event_score", 14.0)),
            min_refined_snr=float(cand.get("min_refined_snr", 15.5)),
            min_hits_if_no_strong_refine=int(cand.get("min_hits_if_no_strong_refine", 2)),
            max_width_bins=int(cand.get("max_width_bins", 3)),
            min_support_fraction=float(cand.get("min_support_fraction", 0.18)),
            allow_singleton_if_refined_snr_ge=float(cand.get("allow_singleton_if_refined_snr_ge", 18.0)),
            allow_singleton_candidates=bool(cand.get("allow_singleton_candidates", False)),
            min_candidate_hits=int(cand.get("min_candidate_hits", 3)),
            min_candidate_duration_s=float(cand.get("min_candidate_duration_s", 0.0)),
            merge_max_gap_s=float(cand.get("merge_max_gap_s", 35.0)),
            merge_freq_tol_hz=float(cand.get("merge_freq_tol_hz", 96.0)),
            merge_singleton_freq_tol_hz=float(cand.get("merge_singleton_freq_tol_hz", 192.0)),
            merge_singleton_drift_tol_hz_per_s=float(cand.get("merge_singleton_drift_tol_hz_per_s", 16.0)),
            merge_short_max_hits=int(cand.get("merge_short_max_hits", 3)),
            merge_short_max_duration_s=float(cand.get("merge_short_max_duration_s", 3.0)),
            merge_short_max_overlap_fraction=float(cand.get("merge_short_max_overlap_fraction", 0.5)),
            merge_short_freq_tol_hz=float(cand.get("merge_short_freq_tol_hz", 256.0)),
            merge_short_drift_tol_hz_per_s=float(cand.get("merge_short_drift_tol_hz_per_s", 24.0)),
            dedup_freq_tol_hz=float(cand.get("dedup_freq_tol_hz", 128.0)),
            dedup_time_overlap_fraction=float(cand.get("dedup_time_overlap_fraction", 0.5)),
            dedup_max_gap_s=float(cand.get("dedup_max_gap_s", 1.0)),
            dedup_short_freq_tol_hz=float(cand.get("dedup_short_freq_tol_hz", 256.0)),
            dedup_short_drift_tol_hz_per_s=float(cand.get("dedup_short_drift_tol_hz_per_s", 24.0)),
        ),
        review=ReviewConfig(
            top_k=int(rv.get("top_k", 100)),
            cutout_frames=int(rv.get("cutout_frames", 21)),
            cutout_bins=int(rv.get("cutout_bins", 128)),
            overview_max_frames=int(rv.get("overview_max_frames", 320)),
        ),
    )
