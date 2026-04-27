from __future__ import annotations

import json
from pathlib import Path

from acs.cli.main import _select_review_records
from acs.post.eventize import cluster_hits_to_events
from acs.types import CandidateConfig, CoincidenceConfig, Hit


def _hit(row0: int, freq: float, drift: float = 2.0, refined_snr: float = 18.0) -> Hit:
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
        coarse_channel=12,
        fine_bin=100,
        anchor_frame=4,
        freq_hz=freq,
        drift_hz_per_s=drift,
        incoherent_snr=9.0,
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


def _track_hit(row0: int, *, native_dt_s: float = 0.001, drift: float = 2.0) -> Hit:
    # Anchor row in eventize is row0 + 4*100 + 0.5*200 = row0 + 500.
    t = (row0 + 500) * native_dt_s
    return _hit(row0, 1000.0 + drift * t, drift=drift)


def test_long_gap_merge_recovers_weak_continuous_track_fragments():
    hits = [_track_hit(r) for r in (0, 1000, 2000, 12000, 13000, 14000)]
    coincidence = CoincidenceConfig(freq_tol_hz=40.0, drift_tol_hz_per_s=1.0, max_row_gap_s=2.0)
    base_candidate = dict(
        min_event_score=0.0,
        min_refined_snr=0.0,
        min_candidate_hits=1,
        min_candidate_duration_s=0.0,
        max_width_bins=10,
        min_support_fraction=0.0,
        merge_max_gap_s=3.0,
        track_inlier_filter_enabled=True,
        track_inlier_freq_tol_hz=30.0,
        track_inlier_drift_tol_hz_per_s=1.0,
    )
    disabled = CandidateConfig(**base_candidate, merge_long_gap_s=0.0)
    events0, _, _, cands0 = cluster_hits_to_events(
        hits, coincidence, disabled, native_dt_s=0.001, frame_hop_rows=100, frame_window_rows=200, tile_step_rows=1000
    )
    assert len(cands0) == 2

    enabled = CandidateConfig(
        **base_candidate,
        merge_long_gap_s=12.0,
        merge_long_gap_min_hits=3,
        merge_long_gap_freq_tol_hz=60.0,
        merge_long_gap_drift_tol_hz_per_s=2.0,
    )
    events1, _, _, cands1 = cluster_hits_to_events(
        hits, coincidence, enabled, native_dt_s=0.001, frame_hop_rows=100, frame_window_rows=200, tile_step_rows=1000
    )
    assert len(cands1) == 1
    assert cands1[0].n_hits == 6
    assert (cands1[0].row1 - cands1[0].row0) * 0.001 > 13.0


def test_review_scope_selects_candidates_or_all_events(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    events = [
        {"event_id": "e1", "score": 1.0},
        {"event_id": "e2", "score": 2.0},
    ]
    candidates = [{"event_id": "e2", "score": 2.0}]
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(x) for x in events) + "\n")
    (run_dir / "candidates.jsonl").write_text("\n".join(json.dumps(x) for x in candidates) + "\n")

    records, scope = _select_review_records(run_dir, "candidates")
    assert scope == "candidates"
    assert [r["event_id"] for r in records] == ["e2"]

    records, scope = _select_review_records(run_dir, "all-events")
    assert scope == "all-events"
    assert [r["event_id"] for r in records] == ["e1", "e2"]
