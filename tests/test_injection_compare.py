import csv
import json
from pathlib import Path

from acs.config import load_runtime_config
from acs.review.injection_compare import compare_injections_to_artifacts, find_injection_report, write_injection_comparison


def _report():
    return {
        "contract": {"frontend_fs_hz": 1_000_000_000.0, "channelizer_fft": 65536},
        "signals": [
            {
                "name": "sig_a",
                "requested": {"name": "sig_a", "freq_tolerance_hz": 120.0, "drift_tolerance_hz_per_s": 6.0, "snr_db": 2.0},
                "resolved": {"start_s": 10.0, "duration_s": 20.0, "start_freq_hz": 1_418_400_000.0, "drift_hz_per_s": 13.0, "snr_db": 2.0, "width_hz": 0.0, "effective_tone_count": 1},
                "target_channels": [{"realized_snr_db": 2.0, "realized_snr_db_vs_original": 1.8, "channel_index": 28}],
            },
            {
                "name": "sig_b",
                "requested": {"name": "sig_b", "freq_tolerance_hz": 120.0, "drift_tolerance_hz_per_s": 6.0, "snr_db": 7.0},
                "resolved": {"start_s": 10.0, "duration_s": 20.0, "start_freq_hz": 1_419_400_000.0, "drift_hz_per_s": 13.0, "snr_db": 7.0, "width_hz": 0.0, "effective_tone_count": 1},
                "target_channels": [{"realized_snr_db": 7.0, "realized_snr_db_vs_original": 6.8, "channel_index": 42}],
            },
        ],
    }


def test_compare_injections_matches_and_marks_unmatched(tmp_path: Path):
    cfg = load_runtime_config(Path(__file__).resolve().parents[1] / "configs" / "h1_search.yaml")
    # Event A is exactly on sig_a at peak time 15 s. Event X is an extra unmatched candidate.
    artifacts = [
        {
            "event_id": "ev_a",
            "row0": int(10.0 / cfg.contract.native_dt_s),
            "row1": int(30.0 / cfg.contract.native_dt_s),
            "peak_row": int(15.0 / cfg.contract.native_dt_s),
            "freq_hz": 1_418_400_000.0 + 13.0 * 5.0,
            "drift_hz_per_s": 13.0,
            "score": 50.0,
            "n_hits": 12,
            "best_incoherent_search_metric_db": 2.0 + 10.0 * __import__('math').log10(cfg.search_stft.nfft),
            "best_refined_search_metric_db": 2.0 + 10.0 * __import__('math').log10(cfg.search_tile_rows),
            "artifact_png": "ev_a.png",
            "artifact_pdf": "ev_a.pdf",
        },
        {
            "event_id": "ev_x",
            "row0": int(10.0 / cfg.contract.native_dt_s),
            "row1": int(30.0 / cfg.contract.native_dt_s),
            "peak_row": int(15.0 / cfg.contract.native_dt_s),
            "freq_hz": 1_420_900_000.0,
            "drift_hz_per_s": 13.0,
            "score": 30.0,
            "n_hits": 8,
            "best_refined_search_metric_db": 44.0,
        },
    ]
    rows, summary = compare_injections_to_artifacts(_report(), artifacts, cfg)
    assert summary["n_injected_signals"] == 2
    assert summary["n_review_candidates"] == 2
    assert summary["n_matched"] == 1
    assert summary["n_unmatched_injected"] == 1
    assert summary["n_unmatched_candidates"] == 1
    by_status = {r["match_status"]: r for r in rows}
    assert by_status["matched"]["signal_name"] == "sig_a"
    assert abs(by_status["matched"]["estimated_raw_snr_db_from_refined"] - 2.0) < 1e-6
    assert by_status["unmatched_injected"]["signal_name"] == "sig_b"
    assert by_status["unmatched_candidate"]["event_id"] == "ev_x"


def test_write_injection_comparison_and_auto_find(tmp_path: Path):
    cfg = load_runtime_config(Path(__file__).resolve().parents[1] / "configs" / "h1_search.yaml")
    run_dir = tmp_path / "outputs" / "run"
    review_dir = run_dir / "review"
    review_dir.mkdir(parents=True)
    report_path = tmp_path / "outputs" / "injected.inject.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    assert find_injection_report(run_dir) == report_path
    artifacts = [{
        "event_id": "ev_a",
        "row0": int(10.0 / cfg.contract.native_dt_s),
        "row1": int(30.0 / cfg.contract.native_dt_s),
        "peak_row": int(15.0 / cfg.contract.native_dt_s),
        "freq_hz": 1_418_400_065.0,
        "drift_hz_per_s": 13.0,
        "score": 50.0,
        "n_hits": 12,
        "best_refined_search_metric_db": 2.0 + 10.0 * __import__('math').log10(cfg.search_tile_rows),
        "artifact_png": "ev_a.png",
        "artifact_pdf": "ev_a.pdf",
    }]
    summary = write_injection_comparison(review_dir, report_path, artifacts, cfg)
    assert summary["n_matched"] == 1
    assert (review_dir / "injection_comparison.csv").exists()
    assert (review_dir / "injection_comparison.html").exists()
    with (review_dir / "injection_comparison.csv").open(newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    assert any(r["signal_name"] == "sig_a" and r["match_status"] == "matched" for r in rows)
    assert any(r["signal_name"] == "sig_b" and r["match_status"] == "unmatched_injected" for r in rows)
