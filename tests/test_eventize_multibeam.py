from acs.types import Hit, CoincidenceConfig, CandidateConfig
from acs.post.eventize import cluster_hits_to_events

def _hit(obs_id, beam, freq, drift, row0=0, row1=100, scan='scan0', coarse_channel=0, anchor_frame=0, incoherent_snr=10.0, refined_snr=20.0):
    return Hit(obs_id=obs_id, beam_id=beam, pol_id='00', scan_id=scan, target_id='t0',
               row0=row0, row1=row1, tile_row0=row0, tile_row1=row1, coarse_channel=coarse_channel,
               fine_bin=0, anchor_frame=anchor_frame, freq_hz=freq, drift_hz_per_s=drift,
               incoherent_snr=incoherent_snr, width_bins=1, backend='numba', seed_kind='mean',
               support_fraction=0.5, edge_distance_bins=20, refined_snr=refined_snr, coherent_gain_db=2.0)

def test_multibeam_coincidence_grouping():
    hits = [_hit('obs_a','00',1420.4e6,12.0), _hit('obs_b','01',1420.4e6+5.0,11.5), _hit('obs_c','02',1420.8e6,12.0)]
    events, _, coincidences, candidates = cluster_hits_to_events(hits, CoincidenceConfig(freq_tol_hz=20.0, drift_tol_hz_per_s=2.0), CandidateConfig(), native_dt_s=65.536e-6)
    assert coincidences
    assert any(e.is_multibeam_coincident for e in events)


def test_cross_scan_not_coincident():
    hits = [_hit('obs_a','00',1420.4e6,12.0, scan='scan0'), _hit('obs_b','01',1420.4e6+5.0,12.0, scan='scan1')]
    events, _, coincidences, candidates = cluster_hits_to_events(hits, CoincidenceConfig(freq_tol_hz=20.0, drift_tol_hz_per_s=2.0), CandidateConfig(), native_dt_s=65.536e-6)
    assert not coincidences


def test_singleton_requires_stronger_candidate_logic():
    h = Hit(obs_id='obs', beam_id='00', pol_id='00', scan_id='s', target_id='t', row0=0, row1=100, tile_row0=0, tile_row1=100, coarse_channel=0, fine_bin=0, anchor_frame=0, freq_hz=1420.4e6, drift_hz_per_s=3.0, incoherent_snr=20.0, width_bins=1, backend='numba', seed_kind='mean', support_fraction=0.1, edge_distance_bins=20, refined_snr=14.0, coherent_gain_db=-4.0)
    events, _, _, candidates = cluster_hits_to_events([h], CoincidenceConfig(), CandidateConfig(), native_dt_s=65.536e-6)
    assert len(events) == 1
    assert len(candidates) == 0


def test_long_drift_chain_merges_into_single_event():
    native_dt_s = 65.536e-6
    step_rows = int(round(1.779565 * (1.0 / native_dt_s)))
    span_rows = int(round(2.048 * (1.0 / native_dt_s)))
    hits = []
    base_freq = 1420.400000e6
    drift = -5.0
    for i in range(12):
        row0 = i * step_rows
        row1 = row0 + span_rows
        mid_t = 0.5 * (row0 + row1) * native_dt_s
        hits.append(_hit('obs_long', '00', base_freq + drift * mid_t, drift, row0=row0, row1=row1, scan='scan0'))
    events, _, _, candidates = cluster_hits_to_events(
        hits,
        CoincidenceConfig(freq_tol_hz=40.0, drift_tol_hz_per_s=1.0, max_row_gap_s=2.5),
        CandidateConfig(),
        native_dt_s=native_dt_s,
    )
    assert len(events) == 1
    assert events[0].n_hits == len(hits)
    assert len(candidates) == 1


def test_support_rows_distinguish_signals_with_same_tile_bounds_but_different_anchor_extent():
    native_dt_s = 65.536e-6
    hop = 1024
    nfft = 2048
    step_rows = int(round(1.779565 * (1.0 / native_dt_s)))
    span_rows = int(round(2.048 * (1.0 / native_dt_s)))
    hits = []
    for i in range(4):
        row0 = i * step_rows
        row1 = row0 + span_rows
        hits.append(_hit('obs_a', '00', 1420.4e6, 2.0, row0=row0, row1=row1, coarse_channel=10, anchor_frame=3))
        hits.append(_hit('obs_b', '00', 1420.8e6, 2.0, row0=row0, row1=row1, coarse_channel=20, anchor_frame=9))
    events, _, _, _ = cluster_hits_to_events(
        hits,
        CoincidenceConfig(freq_tol_hz=40.0, drift_tol_hz_per_s=1.0, max_row_gap_s=2.5),
        CandidateConfig(),
        native_dt_s=native_dt_s,
        frame_hop_rows=hop,
        frame_window_rows=nfft,
        tile_step_rows=step_rows,
    )
    by_obs = {e.obs_id: e for e in events}
    assert len(by_obs) == 2
    dur_a = by_obs['obs_a'].row1 - by_obs['obs_a'].row0
    dur_b = by_obs['obs_b'].row1 - by_obs['obs_b'].row0
    assert dur_a != dur_b


def test_overlapping_same_channel_shadow_candidates_are_deduplicated():
    native_dt_s = 65.536e-6
    hits = []
    for i in range(3):
        row0 = i * 1000
        row1 = row0 + 2000
        hits.append(_hit('obs', '00', 1420.400000e6, 5.0, row0=row0, row1=row1, coarse_channel=11, incoherent_snr=18.0, refined_snr=24.0))
        hits.append(_hit('obs', '00', 1420.400110e6, 5.0, row0=row0, row1=row1, coarse_channel=11, incoherent_snr=11.0, refined_snr=19.0))
    events, _, _, candidates = cluster_hits_to_events(
        hits,
        CoincidenceConfig(freq_tol_hz=40.0, drift_tol_hz_per_s=1.0, max_row_gap_s=2.5),
        CandidateConfig(dedup_freq_tol_hz=128.0, dedup_time_overlap_fraction=0.5),
        native_dt_s=native_dt_s,
    )
    assert len(events) == 2
    assert len(candidates) == 1
    assert abs(candidates[0].freq_hz - 1420.400000e6) < 30.0


def test_adjacent_singleton_fragment_merges_and_deduplicates():
    native_dt_s = 65.536e-6
    freq0 = 1420.5e6
    long_hits = []
    for i in range(4):
        row0 = i * 27000
        row1 = row0 + 31250
        tmid = 0.5 * (row0 + row1) * native_dt_s
        long_hits.append(_hit('obs', '00', freq0 + 34.0 * tmid, 34.0, row0=row0, row1=row1, coarse_channel=35, anchor_frame=10, incoherent_snr=16.0, refined_snr=23.0))
    frag_row0 = long_hits[-1].row1 + int(round(0.25 / native_dt_s))
    frag_row1 = frag_row0 + 31250
    frag_tmid = 0.5 * (frag_row0 + frag_row1) * native_dt_s
    frag = _hit('obs', '00', freq0 + 34.0 * frag_tmid, 45.5, row0=frag_row0, row1=frag_row1, coarse_channel=35, anchor_frame=10, incoherent_snr=9.0, refined_snr=18.5)
    events, _, _, candidates = cluster_hits_to_events(
        long_hits + [frag],
        CoincidenceConfig(freq_tol_hz=40.0, drift_tol_hz_per_s=4.0, max_row_gap_s=2.5),
        CandidateConfig(merge_singleton_freq_tol_hz=192.0, merge_singleton_drift_tol_hz_per_s=16.0, dedup_max_gap_s=1.0),
        native_dt_s=native_dt_s,
        frame_hop_rows=1024,
        frame_window_rows=2048,
        tile_step_rows=27000,
    )
    assert len(events) == 1
    # v1.1.1x applies a post-clustering track-inlier filter. The adjacent
    # fragment is still absorbed/deduplicated into one event, but a biased local
    # drift estimate may be excluded from the support count.
    assert events[0].n_hits >= 4
    assert len(candidates) == 1


def test_strong_single_tile_event_is_not_final_candidate_by_default():
    native_dt_s = 65.536e-6
    # One full 2.048 s search tile with a very strong coherent metric.  It remains
    # an Event for diagnostics, but v0.2.5X does not promote it to Candidate by
    # default because continuous narrowband surveys need multi-tile support.
    span_rows = int(round(2.048 / native_dt_s))
    h = _hit(
        'obs_single', '00', 1420.123e6, 13.0,
        row0=0, row1=span_rows, coarse_channel=7,
        incoherent_snr=35.0, refined_snr=50.0,
    )
    events, _, _, candidates = cluster_hits_to_events(
        [h],
        CoincidenceConfig(freq_tol_hz=40.0, drift_tol_hz_per_s=4.0, max_row_gap_s=2.5),
        CandidateConfig(min_candidate_hits=3, min_candidate_duration_s=4.0, allow_singleton_candidates=False),
        native_dt_s=native_dt_s,
        frame_hop_rows=1024,
        frame_window_rows=2048,
    )
    assert len(events) == 1
    assert events[0].n_hits == 1
    assert len(candidates) == 0
    assert events[0].candidate_reasons == ('insufficient_track_hits',)


def test_low_snr_track_fragments_bridge_long_gaps_with_multi_hit_support():
    native_dt_s = 65.536e-6
    span_rows = int(round(2.048 / native_dt_s))
    freq0 = 1420.777e6
    drift = 13.0
    # Three groups separated by gaps larger than coincidence.max_row_gap_s but
    # smaller than candidate.merge_max_gap_s.  This reproduces the low-SNR
    # intermittent-hit pattern seen in the 4-minute run.
    group_starts_s = [50.0, 65.0, 88.0]
    hits = []
    for gi, gs in enumerate(group_starts_s):
        for j in range(2):
            row0 = int(round((gs + 1.8 * j) / native_dt_s))
            row1 = row0 + span_rows
            tmid = 0.5 * (row0 + row1) * native_dt_s
            hits.append(_hit(
                'obs_frag', '00', freq0 + drift * tmid, drift,
                row0=row0, row1=row1, coarse_channel=42, anchor_frame=10,
                incoherent_snr=22.0, refined_snr=32.0,
            ))
    events, _, _, candidates = cluster_hits_to_events(
        hits,
        CoincidenceConfig(freq_tol_hz=40.0, drift_tol_hz_per_s=4.0, max_row_gap_s=2.5),
        CandidateConfig(
            merge_max_gap_s=35.0,
            min_candidate_hits=3,
            min_candidate_duration_s=4.0,
            allow_singleton_candidates=False,
        ),
        native_dt_s=native_dt_s,
        frame_hop_rows=1024,
        frame_window_rows=2048,
        tile_step_rows=int(round(1.78 / native_dt_s)),
    )
    assert len(events) == 1
    assert events[0].n_hits == len(hits)
    assert len(candidates) == 1
    assert (candidates[0].row1 - candidates[0].row0) * native_dt_s > 35.0
