from acs.review.injection_compare import _track_freq_errors


def test_short_track_path_match_tolerates_unresolved_drift():
    signal = {
        "injected_start_freq_hz": 1_420_000_000.0,
        "injected_drift_hz_per_s": 2.0,
        "injected_start_s": 10.0,
    }
    # Event drift is biased by more than 1 Hz/s, but over the 4-second overlap
    # the recovered path remains within a few Hz of the injected path.  This is
    # the scientifically relevant criterion for short narrowband tracks.
    max_err, rms_err = _track_freq_errors(
        signal,
        event_freq_hz=1_420_000_004.0,
        event_drift_hz_per_s=0.4,
        event_peak_s=12.0,
        overlap_start_s=10.0,
        overlap_end_s=14.0,
    )
    assert max_err < 10.0
    assert rms_err < 6.0
