
from acs.types import Hit, CandidateConfig, CoincidenceConfig
from acs.post.eventize import cluster_hits_to_events


def _hit(row0: int, freq: float, drift: float = 0.0, refined_snr: float = 20.0) -> Hit:
    return Hit(
        obs_id="obs",
        beam_id="00",
        pol_id="00",
        scan_id="scan",
        target_id="target",
        row0=row0,
        row1=row0 + 1000,
        tile_row0=row0,
        tile_row1=row0 + 1000,
        coarse_channel=10,
        fine_bin=100,
        anchor_frame=4,
        freq_hz=freq,
        drift_hz_per_s=drift,
        incoherent_snr=10.0,
        width_bins=1,
        backend="test",
        seed_kind="test",
        support_fraction=0.8,
        edge_distance_bins=64,
        refined_snr=refined_snr,
        refined_freq_hz=freq,
        refined_drift_hz_per_s=drift,
        coherent_gain_db=3.0,
    )


def test_track_inlier_filter_trims_off_track_fragments_from_event_duration():
    hits = [
        _hit(0, 1000.0, refined_snr=21.0),
        _hit(1000, 1001.0, refined_snr=20.0),
        _hit(2000, 999.5, refined_snr=19.5),
        _hit(-2000, 1160.0, refined_snr=18.0),
        _hit(5000, 850.0, refined_snr=18.0),
    ]
    coincidence = CoincidenceConfig(freq_tol_hz=300.0, drift_tol_hz_per_s=10.0, max_row_gap_s=100.0)
    candidate = CandidateConfig(
        min_event_score=0.0,
        min_refined_snr=0.0,
        min_candidate_hits=1,
        min_candidate_duration_s=0.0,
        max_width_bins=10,
        min_support_fraction=0.0,
        merge_max_gap_s=100.0,
        track_inlier_filter_enabled=True,
        track_inlier_freq_tol_hz=50.0,
        track_inlier_drift_tol_hz_per_s=2.0,
    )
    events, event_hits, _, candidates = cluster_hits_to_events(
        hits,
        coincidence,
        candidate,
        native_dt_s=0.01,
        frame_hop_rows=100,
        frame_window_rows=200,
        tile_step_rows=1000,
    )
    assert candidates
    ev = candidates[0]
    assert ev.n_hits == 3
    assert "track_inlier_hits=3/5" in ev.notes
    assert (ev.row1 - ev.row0) < 4000


def test_track_inlier_filter_can_be_disabled_for_diagnostics():
    hits = [
        _hit(0, 1000.0, refined_snr=21.0),
        _hit(1000, 1001.0, refined_snr=20.0),
        _hit(2000, 999.5, refined_snr=19.5),
        _hit(-2000, 1160.0, refined_snr=18.0),
        _hit(5000, 850.0, refined_snr=18.0),
    ]
    coincidence = CoincidenceConfig(freq_tol_hz=300.0, drift_tol_hz_per_s=10.0, max_row_gap_s=100.0)
    candidate = CandidateConfig(
        min_event_score=0.0,
        min_refined_snr=0.0,
        min_candidate_hits=1,
        min_candidate_duration_s=0.0,
        max_width_bins=10,
        min_support_fraction=0.0,
        merge_max_gap_s=100.0,
        track_inlier_filter_enabled=False,
    )
    events, event_hits, _, candidates = cluster_hits_to_events(
        hits,
        coincidence,
        candidate,
        native_dt_s=0.01,
        frame_hop_rows=100,
        frame_window_rows=200,
        tile_step_rows=1000,
    )
    assert candidates
    assert candidates[0].n_hits == 5
