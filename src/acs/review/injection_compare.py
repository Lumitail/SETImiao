from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from ..types import RuntimeConfig


def find_injection_report(run_dir: str | Path, explicit_path: str | Path | None = None) -> Path | None:
    """Find the injection report associated with a run directory.

    The independent injection workflow normally writes ``outputs/injected.inject.json``
    while ``search-run`` writes into ``outputs/run``.  Review therefore searches the
    run directory and its parent for ``*.inject.json`` reports unless an explicit
    path is provided.
    """
    if explicit_path:
        p = Path(explicit_path)
        return p if p.exists() else None
    run_dir = Path(run_dir)
    candidates: list[Path] = []
    for base in (run_dir, run_dir.parent):
        if not base.exists():
            continue
        preferred = base / "injected.inject.json"
        if preferred.exists():
            candidates.append(preferred)
        candidates.extend(sorted(base.glob("*.inject.json")))
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique[0] if unique else None


def _native_dt_from_report(report: dict[str, Any], cfg: RuntimeConfig) -> float:
    c = report.get("contract", {}) if isinstance(report, dict) else {}
    fs = float(c.get("frontend_fs_hz", cfg.contract.frontend_fs_hz))
    fft = int(c.get("channelizer_fft", cfg.contract.channelizer_fft))
    return float(fft / fs)


def _signal_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, sig in enumerate(report.get("signals", []), start=1):
        resolved = sig.get("resolved", {}) or {}
        requested = sig.get("requested", {}) or {}
        chans = sig.get("target_channels", []) or []
        ch0 = chans[0] if chans else {}
        start_s = float(resolved.get("start_s", requested.get("start_s", 0.0)))
        duration_s = float(resolved.get("duration_s", requested.get("duration_s", 0.0)))
        start_freq_hz = float(resolved.get("start_freq_hz", requested.get("start_freq_hz", requested.get("abs_freq_hz", 0.0))))
        drift = float(resolved.get("drift_hz_per_s", requested.get("drift_hz_per_s", 0.0)))
        injected_snr = float(resolved.get("snr_db", requested.get("snr_db", 0.0)))
        realized_snr = ch0.get("realized_snr_db")
        realized_original = ch0.get("realized_snr_db_vs_original")
        rows.append({
            "signal_index": i,
            "signal_name": sig.get("name", requested.get("name", f"signal_{i:03d}")),
            "injected_start_s": start_s,
            "injected_duration_s": duration_s,
            "injected_end_s": start_s + duration_s,
            "injected_start_freq_hz": start_freq_hz,
            "injected_drift_hz_per_s": drift,
            "injected_snr_db": injected_snr,
            "injected_realized_snr_db": None if realized_snr is None else float(realized_snr),
            "injected_realized_snr_db_vs_original": None if realized_original is None else float(realized_original),
            "freq_tolerance_hz": float(requested.get("freq_tolerance_hz", 120.0)),
            "drift_tolerance_hz_per_s": float(requested.get("drift_tolerance_hz_per_s", 6.0)),
            "width_hz": float(resolved.get("width_hz", requested.get("width_hz", 0.0))),
            "effective_tone_count": int(resolved.get("effective_tone_count", requested.get("n_tones", 1))),
            "target_channel_index": ch0.get("channel_index"),
            "target_baseband_hz": ch0.get("baseband_hz"),
        })
    return rows


def _event_timing(event: dict[str, Any], native_dt_s: float) -> tuple[float, float, float]:
    start = event.get("event_start_s")
    end = event.get("event_end_s")
    peak = event.get("peak_time_s")
    if start is None:
        start = float(event.get("row0", 0)) * native_dt_s
    if end is None:
        end = float(event.get("row1", event.get("row0", 0))) * native_dt_s
    if peak is None:
        pr = event.get("peak_row")
        if pr is not None:
            peak = float(pr) * native_dt_s
        else:
            peak = 0.5 * (float(start) + float(end))
    return float(start), float(end), float(peak)


def _expected_freq_at(signal: dict[str, Any], time_s: float) -> float:
    return float(signal["injected_start_freq_hz"]) + float(signal["injected_drift_hz_per_s"]) * (float(time_s) - float(signal["injected_start_s"]))


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _metric_value(event: dict[str, Any], new_key: str, legacy_key: str) -> float | None:
    value = event.get(new_key, event.get(legacy_key))
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _blank_row(signal: dict[str, Any]) -> dict[str, Any]:
    return {
        "match_status": "unmatched_injected",
        "signal_index": signal["signal_index"],
        "signal_name": signal["signal_name"],
        "event_id": "",
        "candidate_rank": "",
        "injected_start_s": signal["injected_start_s"],
        "injected_duration_s": signal["injected_duration_s"],
        "injected_end_s": signal["injected_end_s"],
        "injected_start_freq_hz": signal["injected_start_freq_hz"],
        "injected_end_freq_hz": _expected_freq_at(signal, signal["injected_end_s"]),
        "injected_drift_hz_per_s": signal["injected_drift_hz_per_s"],
        "injected_snr_db": signal["injected_snr_db"],
        "injected_realized_snr_db": signal["injected_realized_snr_db"],
        "injected_realized_snr_db_vs_original": signal["injected_realized_snr_db_vs_original"],
        "recovered_start_s": "",
        "recovered_end_s": "",
        "recovered_duration_s": "",
        "duration_error_s": "",
        "recovered_peak_time_s": "",
        "recovered_freq_hz": "",
        "expected_freq_hz_at_recovered_peak": "",
        "freq_error_hz_at_recovered_peak": "",
        "recovered_drift_hz_per_s": "",
        "drift_error_hz_per_s": "",
        "best_incoherent_search_metric_db": "",
        "best_refined_search_metric_db": "",
        "approx_incoherent_gain_db": "",
        "approx_refined_gain_db": "",
        "estimated_raw_snr_db_from_incoherent": "",
        "estimated_raw_snr_db_from_refined": "",
        "estimated_raw_snr_refined_error_db": "",
        "estimated_raw_snr_incoherent_error_db": "",
        "n_hits": "",
        "score": "",
        "artifact_png": "",
        "artifact_pdf": "",
        "match_score": "",
        "notes": "no candidate matched this injected signal within tolerance",
    }


def compare_injections_to_artifacts(
    report: dict[str, Any],
    artifacts: list[dict[str, Any]],
    cfg: RuntimeConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One-to-one match injected signals to candidate/review artifacts."""
    native_dt_s = _native_dt_from_report(report, cfg)
    signals = _signal_rows(report)
    approx_refined_gain_db = float(10.0 * math.log10(max(int(cfg.search_tile_rows), 1)))
    approx_incoherent_gain_db = float(10.0 * math.log10(max(int(cfg.search_stft.nfft), 1)))

    event_meta: list[dict[str, Any]] = []
    for rank, ev in enumerate(artifacts):
        ev_start, ev_end, ev_peak = _event_timing(ev, native_dt_s)
        event_meta.append({"rank": rank, "event": ev, "start": ev_start, "end": ev_end, "peak": ev_peak})

    candidate_pairs: list[tuple[float, int, int, dict[str, Any]]] = []
    all_pair_diagnostics: list[dict[str, Any]] = []
    for si, sig in enumerate(signals):
        for ei, meta in enumerate(event_meta):
            ev = meta["event"]
            ev_freq = ev.get("freq_hz")
            if ev_freq is None:
                continue
            ev_drift = float(ev.get("drift_hz_per_s", 0.0))
            overlap_s = _overlap(sig["injected_start_s"], sig["injected_end_s"], meta["start"], meta["end"])
            if overlap_s <= 0.0:
                # Do not match events outside the injected active interval.
                continue
            expected_freq = _expected_freq_at(sig, meta["peak"])
            df = abs(float(ev_freq) - expected_freq)
            dd = abs(ev_drift - sig["injected_drift_hz_per_s"])
            freq_tol = max(float(sig["freq_tolerance_hz"]), 1e-9)
            drift_tol = max(float(sig["drift_tolerance_hz_per_s"]), 1e-9)
            duration_error = abs((meta["end"] - meta["start"]) - sig["injected_duration_s"])
            duration_norm = duration_error / max(sig["injected_duration_s"], 1.0)
            match_score = (df / freq_tol) + (dd / drift_tol) + 0.05 * duration_norm
            diag = {
                "signal_name": sig["signal_name"],
                "event_id": ev.get("event_id"),
                "freq_error_hz": df,
                "drift_error_hz_per_s": dd,
                "time_overlap_s": overlap_s,
                "match_score": match_score,
                "within_tolerance": df <= freq_tol and dd <= drift_tol,
            }
            all_pair_diagnostics.append(diag)
            if diag["within_tolerance"]:
                candidate_pairs.append((match_score, si, ei, diag))

    assigned_signals: set[int] = set()
    assigned_events: set[int] = set()
    assignments: dict[int, tuple[int, dict[str, Any]]] = {}
    for score, si, ei, diag in sorted(candidate_pairs, key=lambda x: x[0]):
        if si in assigned_signals or ei in assigned_events:
            continue
        assigned_signals.add(si)
        assigned_events.add(ei)
        assignments[si] = (ei, diag)

    rows: list[dict[str, Any]] = []
    for si, sig in enumerate(signals):
        if si not in assignments:
            rows.append(_blank_row(sig))
            continue
        ei, diag = assignments[si]
        meta = event_meta[ei]
        ev = meta["event"]
        expected_freq = _expected_freq_at(sig, meta["peak"])
        incoh = _metric_value(ev, "best_incoherent_search_metric_db", "best_incoherent_snr")
        refined = _metric_value(ev, "best_refined_search_metric_db", "best_refined_snr")
        est_raw_incoh = None if incoh is None else incoh - approx_incoherent_gain_db
        est_raw_refined = None if refined is None else refined - approx_refined_gain_db
        injected_snr_for_error = sig["injected_realized_snr_db"] if sig["injected_realized_snr_db"] is not None else sig["injected_snr_db"]
        rows.append({
            "match_status": "matched",
            "signal_index": sig["signal_index"],
            "signal_name": sig["signal_name"],
            "event_id": ev.get("event_id", ""),
            "candidate_rank": meta["rank"],
            "injected_start_s": sig["injected_start_s"],
            "injected_duration_s": sig["injected_duration_s"],
            "injected_end_s": sig["injected_end_s"],
            "injected_start_freq_hz": sig["injected_start_freq_hz"],
            "injected_end_freq_hz": _expected_freq_at(sig, sig["injected_end_s"]),
            "injected_drift_hz_per_s": sig["injected_drift_hz_per_s"],
            "injected_snr_db": sig["injected_snr_db"],
            "injected_realized_snr_db": sig["injected_realized_snr_db"],
            "injected_realized_snr_db_vs_original": sig["injected_realized_snr_db_vs_original"],
            "recovered_start_s": meta["start"],
            "recovered_end_s": meta["end"],
            "recovered_duration_s": meta["end"] - meta["start"],
            "duration_error_s": (meta["end"] - meta["start"]) - sig["injected_duration_s"],
            "recovered_peak_time_s": meta["peak"],
            "recovered_freq_hz": ev.get("freq_hz", ""),
            "expected_freq_hz_at_recovered_peak": expected_freq,
            "freq_error_hz_at_recovered_peak": abs(float(ev.get("freq_hz")) - expected_freq) if ev.get("freq_hz") is not None else "",
            "recovered_drift_hz_per_s": ev.get("drift_hz_per_s", ""),
            "drift_error_hz_per_s": abs(float(ev.get("drift_hz_per_s", 0.0)) - sig["injected_drift_hz_per_s"]),
            "best_incoherent_search_metric_db": incoh,
            "best_refined_search_metric_db": refined,
            "approx_incoherent_gain_db": approx_incoherent_gain_db,
            "approx_refined_gain_db": approx_refined_gain_db,
            "estimated_raw_snr_db_from_incoherent": est_raw_incoh,
            "estimated_raw_snr_db_from_refined": est_raw_refined,
            "estimated_raw_snr_refined_error_db": None if est_raw_refined is None else est_raw_refined - injected_snr_for_error,
            "estimated_raw_snr_incoherent_error_db": None if est_raw_incoh is None else est_raw_incoh - injected_snr_for_error,
            "n_hits": ev.get("n_hits", ""),
            "score": ev.get("score", ""),
            "artifact_png": ev.get("artifact_png", ""),
            "artifact_pdf": ev.get("artifact_pdf", ""),
            "match_score": diag.get("match_score", ""),
            "notes": "best_*_search_metric_db are detector metrics; estimated_raw_snr_db subtracts the nominal coherent/STFT integration gain and is approximate",
        })

    # Include extra candidates that do not correspond to an injected signal.  This
    # keeps false positives visible in the comparison outputs.
    for ei, meta in enumerate(event_meta):
        if ei in assigned_events:
            continue
        ev = meta["event"]
        incoh = _metric_value(ev, "best_incoherent_search_metric_db", "best_incoherent_snr")
        refined = _metric_value(ev, "best_refined_search_metric_db", "best_refined_snr")
        rows.append({
            "match_status": "unmatched_candidate",
            "signal_index": "",
            "signal_name": "",
            "event_id": ev.get("event_id", ""),
            "candidate_rank": meta["rank"],
            "injected_start_s": "",
            "injected_duration_s": "",
            "injected_end_s": "",
            "injected_start_freq_hz": "",
            "injected_end_freq_hz": "",
            "injected_drift_hz_per_s": "",
            "injected_snr_db": "",
            "injected_realized_snr_db": "",
            "injected_realized_snr_db_vs_original": "",
            "recovered_start_s": meta["start"],
            "recovered_end_s": meta["end"],
            "recovered_duration_s": meta["end"] - meta["start"],
            "duration_error_s": "",
            "recovered_peak_time_s": meta["peak"],
            "recovered_freq_hz": ev.get("freq_hz", ""),
            "expected_freq_hz_at_recovered_peak": "",
            "freq_error_hz_at_recovered_peak": "",
            "recovered_drift_hz_per_s": ev.get("drift_hz_per_s", ""),
            "drift_error_hz_per_s": "",
            "best_incoherent_search_metric_db": incoh,
            "best_refined_search_metric_db": refined,
            "approx_incoherent_gain_db": approx_incoherent_gain_db,
            "approx_refined_gain_db": approx_refined_gain_db,
            "estimated_raw_snr_db_from_incoherent": None if incoh is None else incoh - approx_incoherent_gain_db,
            "estimated_raw_snr_db_from_refined": None if refined is None else refined - approx_refined_gain_db,
            "estimated_raw_snr_refined_error_db": "",
            "estimated_raw_snr_incoherent_error_db": "",
            "n_hits": ev.get("n_hits", ""),
            "score": ev.get("score", ""),
            "artifact_png": ev.get("artifact_png", ""),
            "artifact_pdf": ev.get("artifact_pdf", ""),
            "match_score": "",
            "notes": "candidate did not match any injected signal within time/frequency/drift tolerance",
        })

    # Calibrate the detector metric to injected raw SNR whenever this review is
    # itself an injection sweep.  This is deliberately reported as empirical
    # calibration, not as a universal physical conversion.
    matched_for_fit = []
    for r in rows:
        if r.get("match_status") != "matched":
            continue
        x = r.get("injected_realized_snr_db")
        if x in (None, ""):
            x = r.get("injected_snr_db")
        y = r.get("best_refined_search_metric_db")
        if x in (None, "") or y in (None, ""):
            continue
        try:
            matched_for_fit.append((float(x), float(y)))
        except Exception:
            pass
    calibration = None
    if len(matched_for_fit) >= 2:
        xs = [p[0] for p in matched_for_fit]
        ys = [p[1] for p in matched_for_fit]
        xbar = sum(xs) / len(xs)
        ybar = sum(ys) / len(ys)
        var = sum((x - xbar) ** 2 for x in xs)
        if var > 1e-12:
            slope = sum((x - xbar) * (y - ybar) for x, y in matched_for_fit) / var
            intercept = ybar - slope * xbar
        else:
            slope = 1.0
            intercept = ybar - xbar
        residuals = [y - (slope * x + intercept) for x, y in matched_for_fit]
        rms = math.sqrt(sum(r * r for r in residuals) / max(len(residuals), 1))
        calibration = {
            "metric": "best_refined_search_metric_db",
            "model": "metric_db = slope * injected_realized_snr_db + intercept_db",
            "slope": float(slope),
            "intercept_db": float(intercept),
            "rms_residual_db": float(rms),
            "n_fit": len(matched_for_fit),
        }
        for r in rows:
            metric = r.get("best_refined_search_metric_db")
            if metric in (None, "") or abs(slope) < 1e-12:
                r["calibrated_raw_snr_db_from_refined"] = ""
                r["calibrated_raw_snr_refined_error_db"] = ""
                continue
            est = (float(metric) - intercept) / slope
            r["calibrated_raw_snr_db_from_refined"] = est
            inj_snr = r.get("injected_realized_snr_db")
            if inj_snr in (None, ""):
                inj_snr = r.get("injected_snr_db")
            if inj_snr in (None, ""):
                r["calibrated_raw_snr_refined_error_db"] = ""
            else:
                r["calibrated_raw_snr_refined_error_db"] = est - float(inj_snr)
    else:
        for r in rows:
            r["calibrated_raw_snr_db_from_refined"] = ""
            r["calibrated_raw_snr_refined_error_db"] = ""

    n_matched = sum(1 for r in rows if r["match_status"] == "matched")
    detected_snr = []
    for r in rows:
        if r.get("match_status") == "matched":
            val = r.get("injected_realized_snr_db")
            if val in (None, ""):
                val = r.get("injected_snr_db")
            if val not in (None, ""):
                detected_snr.append(float(val))
    summary = {
        "n_injected_signals": len(signals),
        "n_review_candidates": len(artifacts),
        "n_matched": n_matched,
        "n_unmatched_injected": sum(1 for r in rows if r["match_status"] == "unmatched_injected"),
        "n_unmatched_candidates": sum(1 for r in rows if r["match_status"] == "unmatched_candidate"),
        "approx_refined_gain_db": approx_refined_gain_db,
        "approx_incoherent_gain_db": approx_incoherent_gain_db,
        "lowest_matched_injected_snr_db": min(detected_snr) if detected_snr else None,
        "highest_unmatched_injected_snr_db": max([
            float(r.get("injected_realized_snr_db") if r.get("injected_realized_snr_db") not in (None, "") else r.get("injected_snr_db"))
            for r in rows if r.get("match_status") == "unmatched_injected" and r.get("injected_snr_db") not in (None, "")
        ], default=None),
        "refined_metric_empirical_calibration": calibration,
        "snr_definitions": {
            "injected_snr_db": "raw active-sample RMS ratio relative to the channel RMS used by the injector",
            "best_incoherent_search_metric_db": "STFT drift-path detector metric; it includes an approximate 10*log10(nfft) processing gain for a bin-centered tone",
            "best_refined_search_metric_db": "coherent dechirp FFT detector metric; it includes an approximate 10*log10(search_tile_rows) coherent processing gain",
            "estimated_raw_snr_db_from_refined": "best_refined_search_metric_db minus 10*log10(search_tile_rows); useful as a nominal estimate only",
            "calibrated_raw_snr_db_from_refined": "empirical conversion from refined metric to raw injected SNR fitted from the matched injected signals in this run",
        },
        "all_pair_diagnostics": all_pair_diagnostics,
    }
    return rows, summary


def write_injection_comparison(
    review_dir: str | Path,
    report_path: str | Path,
    artifacts: list[dict[str, Any]],
    cfg: RuntimeConfig,
) -> dict[str, Any]:
    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    rows, summary = compare_injections_to_artifacts(report, artifacts, cfg)
    summary = {**summary, "injection_report": str(Path(report_path))}

    csv_path = review_dir / "injection_comparison.csv"
    json_path = review_dir / "injection_comparison.json"
    summary_path = review_dir / "injection_comparison_summary.json"
    html_path = review_dir / "injection_comparison.html"

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "<html><head><meta charset='utf-8'><title>ACS Injection Comparison</title></head><body>",
        "<h1>ACS Injection Comparison</h1>",
        f"<p>Matched {summary['n_matched']} / {summary['n_injected_signals']} injected signals. "
        f"Unmatched candidates: {summary['n_unmatched_candidates']}.</p>",
        "<p><b>SNR note:</b> injected SNR is a raw active-sample RMS ratio. Recovered SNR columns are search metrics with processing gain. Use the estimated raw-SNR columns and injection calibration for sensitivity claims.</p>",
        "<table border='1' cellspacing='0' cellpadding='4'>",
        "<tr><th>status</th><th>signal</th><th>injected SNR dB</th><th>event</th><th>duration error s</th><th>freq error Hz</th><th>drift error Hz/s</th><th>refined metric dB</th><th>nominal raw SNR dB</th><th>calibrated raw SNR dB</th><th>preview</th></tr>",
    ]
    for r in rows:
        png = r.get("artifact_png", "")
        preview = f"<a href='{png}'>{png}</a>" if png else ""
        lines.append(
            "<tr>"
            f"<td>{r.get('match_status','')}</td>"
            f"<td>{r.get('signal_name','')}</td>"
            f"<td>{r.get('injected_snr_db','')}</td>"
            f"<td>{r.get('event_id','')}</td>"
            f"<td>{r.get('duration_error_s','')}</td>"
            f"<td>{r.get('freq_error_hz_at_recovered_peak','')}</td>"
            f"<td>{r.get('drift_error_hz_per_s','')}</td>"
            f"<td>{r.get('best_refined_search_metric_db','')}</td>"
            f"<td>{r.get('estimated_raw_snr_db_from_refined','')}</td>"
            f"<td>{r.get('calibrated_raw_snr_db_from_refined','')}</td>"
            f"<td>{preview}</td>"
            "</tr>"
        )
    lines += ["</table>", "</body></html>"]
    html_path.write_text("\n".join(lines), encoding="utf-8")
    return summary
