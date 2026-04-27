from acs.types import Hit, CandidateConfig, CoincidenceConfig
from acs.post.eventize import cluster_hits_to_events


def _hit(row0, refined_snr, incoherent_snr=6.0):
    return Hit(
        obs_id="obs",
        beam_id="00",
        pol_id="00",
        scan_id="scan",
        target_id="target",
        row0=row0,
        row1=row0 + 31250,
        tile_row0=row0,
        tile_row1=row0 + 31250,
        coarse_channel=5,
        fine_bin=100,
        anchor_frame=10,
        freq_hz=1420.0e6,
        drift_hz_per_s=0.5,
        incoherent_snr=incoherent_snr,
        width_bins=1,
        backend="test",
        seed_kind="test",
        support_fraction=0.6,
        edge_distance_bins=64,
        refined_snr=refined_snr,
        refined_freq_hz=1420.0e6,
        refined_drift_hz_per_s=0.5,
        coherent_gain_db=refined_snr - incoherent_snr,
    )


def test_weak_two_hit_short_track_stays_event_not_candidate():
    native_dt_s = 65.536e-6
    hits = [_hit(0, 12.0), _hit(27000, 12.2)]
    events, _, _, candidates = cluster_hits_to_events(
        hits,
        CoincidenceConfig(freq_tol_hz=40.0, drift_tol_hz_per_s=1.0, max_row_gap_s=2.5),
        CandidateConfig(
            min_event_score=8.0,
            min_refined_snr=8.0,
            min_candidate_hits=2,
            min_candidate_duration_s=1.0,
            short_track_max_hits=2,
            short_track_max_duration_s=4.0,
            short_track_min_refined_snr=14.5,
            short_track_min_event_score=16.0,
        ),
        native_dt_s=native_dt_s,
        frame_hop_rows=1024,
        frame_window_rows=2048,
        tile_step_rows=27000,
    )
    assert len(events) == 1
    assert not candidates
    assert events[0].candidate_reasons == ("short_track_refined_snr_below_threshold",)


def test_strong_two_hit_short_track_can_remain_candidate():
    native_dt_s = 65.536e-6
    hits = [_hit(0, 18.0, incoherent_snr=8.0), _hit(27000, 18.5, incoherent_snr=8.0)]
    events, _, _, candidates = cluster_hits_to_events(
        hits,
        CoincidenceConfig(freq_tol_hz=40.0, drift_tol_hz_per_s=1.0, max_row_gap_s=2.5),
        CandidateConfig(
            min_event_score=8.0,
            min_refined_snr=8.0,
            min_candidate_hits=2,
            min_candidate_duration_s=1.0,
            short_track_max_hits=2,
            short_track_max_duration_s=4.0,
            short_track_min_refined_snr=14.5,
            short_track_min_event_score=16.0,
        ),
        native_dt_s=native_dt_s,
        frame_hop_rows=1024,
        frame_window_rows=2048,
        tile_step_rows=27000,
    )
    assert len(events) == 1
    assert len(candidates) == 1
    assert "short_track_strong_refine" in candidates[0].candidate_reasons
