from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Any
import numpy as np
import numpy.typing as npt

ArrayU16 = npt.NDArray[np.uint16]
ArrayF32 = npt.NDArray[np.float32]
ArrayF64 = npt.NDArray[np.float64]
ArrayC64 = npt.NDArray[np.complex64]
ArrayBool = npt.NDArray[np.bool_]

@dataclass(frozen=True)
class DatContract:
    channels: int = 256
    bytes_per_cell: int = 2
    frontend_fs_hz: float = 1_000_000_000.0
    channelizer_fft: int = 65536
    lo_hz: Optional[float] = None
    start_coarse_channel: Optional[int] = None
    hi_rest_hz: float = 1_420_405_751.77

    @property
    def row_bytes(self) -> int:
        return self.channels * self.bytes_per_cell

    @property
    def coarse_df_hz(self) -> float:
        return self.frontend_fs_hz / self.channelizer_fft

    @property
    def native_dt_s(self) -> float:
        return 1.0 / self.coarse_df_hz

    @property
    def raw_bandwidth_hz(self) -> float:
        return self.channels * self.coarse_df_hz

@dataclass(frozen=True)
class ObservationMeta:
    obs_id: str
    dat_paths: tuple[Path, ...]
    source_name: Optional[str] = None
    beam_id: Optional[str] = None
    pol_id: Optional[str] = None
    scan_id: Optional[str] = None
    target_id: Optional[str] = None
    timestamp_start: Optional[str] = None
    lo_hz: Optional[float] = None
    start_coarse_channel: Optional[int] = None
    metadata_source_path: Optional[Path] = None

@dataclass(frozen=True)
class ObservationSpec:
    obs_id: str
    dat_paths: tuple[Path, ...]
    source_name: Optional[str] = None
    beam_id: Optional[str] = None
    pol_id: Optional[str] = None
    scan_id: Optional[str] = None
    target_id: Optional[str] = None
    timestamp_start: Optional[str] = None
    lo_hz: Optional[float] = None
    start_coarse_channel: Optional[int] = None

@dataclass(frozen=True)
class FileSegment:
    path: Path
    rows: int
    row_start: int
    row_stop: int

@dataclass(frozen=True)
class FrequencyMap:
    mode: str
    coarse_centers_hz: ArrayF64
    fine_offsets_hz: ArrayF64

@dataclass(frozen=True)
class StitchedObservation:
    meta: ObservationMeta
    contract: DatContract
    segments: tuple[FileSegment, ...]
    total_rows: int
    freq_map: FrequencyMap

@dataclass(frozen=True)
class STFTConfig:
    name: str
    nfft: int
    hop: int
    window: str = "hann"

@dataclass(frozen=True)
class BaselineConfig:
    smooth_bins: int = 1001
    edge_bins_per_coarse: int = 16
    mask_sigma_mad: float = 6.0
    qc_display_smooth_bins: int = 257
    qc_low_percentile: float = 2.0
    qc_high_percentile: float = 98.0

@dataclass(frozen=True)
class DriftConfig:
    min_hz_per_s: float = -48.0
    max_hz_per_s: float = 48.0
    step_hz_per_s: Optional[float] = None
    seed_top_k_mean_per_channel: int = 6
    seed_top_k_max_per_channel: int = 6
    seed_min_separation_bins: int = 8
    snr_threshold: float = 6.0
    widths: tuple[int, ...] = (1, 3, 5, 7)
    trial_chunk_size: int = 64
    backend: str = "auto"
    support_threshold: float = 1.5
    channel_min: Optional[int] = None
    channel_max: Optional[int] = None

@dataclass(frozen=True)
class RefineConfig:
    enabled: bool = True
    top_n: int = 12
    drift_half_window_hz_per_s: float = 8.0
    drift_step_hz_per_s: float = 0.25
    freq_half_window_bins: int = 4
    min_refined_snr: float = 14.5

@dataclass(frozen=True)
class CoincidenceConfig:
    freq_tol_hz: float = 40.0
    drift_tol_hz_per_s: float = 4.0
    time_tol_s: float = 0.35
    multibeam_penalty: float = 2.5
    single_beam_bonus: float = 0.0
    max_row_gap_s: float = 2.5

@dataclass(frozen=True)
class CandidateConfig:
    min_event_score: float = 14.0
    min_refined_snr: float = 15.5
    min_hits_if_no_strong_refine: int = 2
    max_width_bins: int = 3
    min_support_fraction: float = 0.18
    allow_singleton_if_refined_snr_ge: float = 18.0
    # v0.2.5X: final candidates can be required to be real tracks rather than
    # one-tile coherent peaks.  This is important for continuous narrowband
    # surveys because a single 2.048 s tile can have a large coherent metric even
    # when the long-duration signal is not reliably recovered.
    allow_singleton_candidates: bool = False
    min_candidate_hits: int = 3
    min_candidate_duration_s: float = 0.0
    merge_max_gap_s: float = 35.0
    merge_freq_tol_hz: float = 96.0
    merge_singleton_freq_tol_hz: float = 192.0
    merge_singleton_drift_tol_hz_per_s: float = 16.0
    # v0.2.4X: short fragments (for example one/two-tile edge fragments)
    # have intrinsically less reliable drift estimates, so they get their own
    # merge/dedup tolerances instead of being treated as full tracks.
    merge_short_max_hits: int = 3
    merge_short_max_duration_s: float = 3.0
    merge_short_max_overlap_fraction: float = 0.5
    merge_short_freq_tol_hz: float = 256.0
    merge_short_drift_tol_hz_per_s: float = 24.0
    dedup_freq_tol_hz: float = 128.0
    dedup_time_overlap_fraction: float = 0.5
    dedup_max_gap_s: float = 1.0
    dedup_short_freq_tol_hz: float = 256.0
    dedup_short_drift_tol_hz_per_s: float = 24.0

@dataclass(frozen=True)
class ReviewConfig:
    top_k: int = 100
    cutout_frames: int = 21
    cutout_bins: int = 128
    overview_max_frames: int = 320

@dataclass(frozen=True)
class RuntimeConfig:
    contract: DatContract
    search_tile_rows: int
    search_overlap_rows: int
    search_stft: STFTConfig
    baseline: BaselineConfig
    drift: DriftConfig
    refine: RefineConfig
    coincidence: CoincidenceConfig
    candidate: CandidateConfig
    review: ReviewConfig

@dataclass
class SpectrogramTile:
    obs_id: str
    beam_id: Optional[str]
    pol_id: Optional[str]
    scan_id: Optional[str]
    target_id: Optional[str]
    row0: int
    row1: int
    power: ArrayF32
    norm_power: ArrayF32
    mask: ArrayBool
    frame_times_s: ArrayF64
    fine_freq_hz: ArrayF64
    mean_excess_db: ArrayF32
    max_excess_db: ArrayF32
    p90_excess_db: ArrayF32

@dataclass
class Seed:
    coarse_channel: int
    fine_bin: int
    anchor_frame: int
    score: float
    seed_kind: str

@dataclass
class Hit:
    obs_id: str
    beam_id: Optional[str]
    pol_id: Optional[str]
    scan_id: Optional[str]
    target_id: Optional[str]
    row0: int
    row1: int
    tile_row0: int
    tile_row1: int
    coarse_channel: int
    fine_bin: int
    anchor_frame: int
    freq_hz: Optional[float]
    drift_hz_per_s: float
    incoherent_snr: float
    width_bins: int
    backend: str
    seed_kind: str
    support_fraction: float = 0.0
    edge_distance_bins: int = 0
    refined_snr: Optional[float] = None
    refined_freq_hz: Optional[float] = None
    refined_drift_hz_per_s: Optional[float] = None
    coherent_gain_db: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class Event:
    event_id: str
    obs_id: str
    beam_id: Optional[str]
    pol_id: Optional[str]
    scan_id: Optional[str]
    target_id: Optional[str]
    row0: int
    row1: int
    freq_hz: Optional[float]
    drift_hz_per_s: float
    score: float
    n_hits: int
    best_incoherent_snr: float
    best_refined_snr: Optional[float]
    best_width_bins: int = 1
    best_support_fraction: float = 0.0
    best_coherent_gain_db: Optional[float] = None
    representative_coarse_channel: Optional[int] = None
    peak_row: Optional[int] = None
    beam_multiplicity: int = 1
    coincident_beams: tuple[str, ...] = ()
    is_multibeam_coincident: bool = False
    coincidence_group_id: Optional[str] = None
    candidate_passed: bool = False
    candidate_reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["coincident_beams"] = list(self.coincident_beams)
        d["candidate_reasons"] = list(self.candidate_reasons)
        d["notes"] = list(self.notes)
        return d
