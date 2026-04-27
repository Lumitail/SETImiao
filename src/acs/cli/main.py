from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import yaml

from .. import __version__
from dataclasses import replace
from ..config import load_runtime_config
from ..types import Event
from ..io.dat_reader import validate_geometry
from ..io.stitcher import build_observation_from_paths, build_observations_from_specs
from ..io.manifest import load_manifest
from ..frontend.stft import build_spectrogram_tile
from ..preproc.baseline import apply_baseline_and_masks, qc_display_matrix
from ..search.drift import search_tile, iter_search_channels
from ..refine.coherent import refine_hits
from ..post.eventize import cluster_hits_to_events
from ..bench.inject import (
    build_case_suite,
    build_suite_plan,
    load_injection_plan,
    save_injection_plan,
    write_injected_dataset,
    write_injected_observation,
)


def _obs_from_args_or_manifest(args, cfg):
    if getattr(args, "manifest", None):
        specs = load_manifest(args.manifest)
        return build_observations_from_specs(specs, cfg.contract, nfft=cfg.search_stft.nfft)
    dat_paths = [Path(p) for p in getattr(args, "dat_paths", [])]
    obs = build_observation_from_paths(
        dat_paths,
        cfg.contract,
        obs_id=args.obs_id or "obs0",
        source_name=args.source_name,
        beam_id=args.beam_id,
        pol_id=args.pol_id,
        scan_id=args.scan_id,
        target_id=args.target_id or args.source_name,
        lo_hz=args.lo_hz if args.lo_hz is not None else cfg.contract.lo_hz,
        start_coarse_channel=args.start_coarse_channel if args.start_coarse_channel is not None else cfg.contract.start_coarse_channel,
    )
    return [obs]


def _iter_tiles(total_rows: int, tile_rows: int, overlap_rows: int):
    if total_rows <= tile_rows:
        yield 0, total_rows
        return
    step = max(1, tile_rows - overlap_rows)
    s = 0
    while s < total_rows:
        e = min(total_rows, s + tile_rows)
        yield s, e
        if e >= total_rows:
            break
        s += step


def _override_runtime_thresholds(cfg, args):
    drift_updates = {}
    refine_updates = {}
    candidate_updates = {}
    if getattr(args, 'drift_snr_threshold', None) is not None:
        drift_updates['snr_threshold'] = float(args.drift_snr_threshold)
    if getattr(args, 'refine_min_snr', None) is not None:
        refine_updates['min_refined_snr'] = float(args.refine_min_snr)
    if getattr(args, 'candidate_min_refined_snr', None) is not None:
        candidate_updates['min_refined_snr'] = float(args.candidate_min_refined_snr)
    if getattr(args, 'candidate_min_score', None) is not None:
        candidate_updates['min_event_score'] = float(args.candidate_min_score)
    if drift_updates:
        cfg = replace(cfg, drift=replace(cfg.drift, **drift_updates))
    if refine_updates:
        cfg = replace(cfg, refine=replace(cfg.refine, **refine_updates))
    if candidate_updates:
        cfg = replace(cfg, candidate=replace(cfg.candidate, **candidate_updates))
    return cfg


def _run_observation_search(obs, cfg):
    hits = []
    search_channels = iter_search_channels(obs.contract.channels, cfg.drift)
    use_sparse_tile = len(search_channels) < obs.contract.channels
    for row0, row1 in _iter_tiles(obs.total_rows, cfg.search_tile_rows, cfg.search_overlap_rows):
        tile = build_spectrogram_tile(obs, row0, row1, cfg.search_stft, channel_indices=search_channels if use_sparse_tile else None)
        tile = apply_baseline_and_masks(tile, cfg.baseline)
        tile_hits = search_tile(tile, cfg.drift)
        tile_hits = refine_hits(obs, tile_hits, cfg.refine)
        if cfg.refine.enabled:
            tile_hits = [h for h in tile_hits if h.refined_snr is not None and h.refined_snr >= cfg.refine.min_refined_snr]
        hits.extend(tile_hits)
    return hits


def _run_pipeline(obs_list, cfg, run_dir: Path | None = None):
    all_hits = []
    for obs in obs_list:
        all_hits.extend(_run_observation_search(obs, cfg))
    native_dt_s = obs_list[0].contract.native_dt_s if obs_list else 1.0
    events, event_hits, coincidences, candidates = cluster_hits_to_events(
        all_hits,
        cfg.coincidence,
        cfg.candidate,
        native_dt_s=native_dt_s,
        frame_hop_rows=cfg.search_stft.hop,
        frame_window_rows=cfg.search_stft.nfft,
        tile_step_rows=max(1, cfg.search_tile_rows - cfg.search_overlap_rows),
    )
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "hits.jsonl").write_text("\n".join(json.dumps(h.to_dict()) for h in all_hits), encoding="utf-8")
        (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e.to_dict()) for e in events), encoding="utf-8")
        (run_dir / "candidates.jsonl").write_text("\n".join(json.dumps(c.to_dict()) for c in candidates), encoding="utf-8")
        (run_dir / "coincidences.jsonl").write_text("\n".join(json.dumps(c) for c in coincidences), encoding="utf-8")
        run_meta = {
            "version": __version__,
            "observations": [o.meta.obs_id for o in obs_list],
            "n_hits": len(all_hits),
            "n_events": len(events),
            "n_candidates": len(candidates),
            "n_coincidences": len(coincidences),
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps({"n_hits": len(all_hits), "n_events": len(events), "n_candidates": len(candidates), "n_coincidences": len(coincidences)}, indent=2), encoding="utf-8")
    return all_hits, events, candidates, coincidences


def cmd_inspect_format(args):
    from ..types import DatContract
    contract = DatContract(
        channels=args.channels,
        bytes_per_cell=args.bytes_per_cell,
        frontend_fs_hz=args.frontend_fs_hz,
        channelizer_fft=args.channelizer_fft,
        lo_hz=args.lo_hz,
        start_coarse_channel=args.start_coarse_channel,
    )
    total_bytes, rows = validate_geometry(args.dat_path, contract)
    out = {
        "dat_path": str(args.dat_path),
        "total_bytes": total_bytes,
        "rows": rows,
        "channels": contract.channels,
        "row_bytes": contract.row_bytes,
        "native_dt_s": contract.native_dt_s,
        "coarse_df_hz": contract.coarse_df_hz,
        "raw_bandwidth_hz": contract.raw_bandwidth_hz,
        "version": __version__,
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_qc_waterfall(args):
    cfg = load_runtime_config(args.config)
    cfg = _override_runtime_thresholds(cfg, args)
    obs_list = _obs_from_args_or_manifest(args, cfg)
    obs = obs_list[0] if args.obs_id is None else next(o for o in obs_list if o.meta.obs_id == args.obs_id)
    row1 = min(obs.total_rows, cfg.search_tile_rows)
    tile = build_spectrogram_tile(obs, 0, row1, cfg.search_stft)
    tile = apply_baseline_and_masks(tile, cfg.baseline)
    disp, vmin, vmax = qc_display_matrix(tile, cfg.baseline)
    import matplotlib.pyplot as plt
    mean_spec = tile.mean_excess_db.reshape(-1)
    freq = tile.fine_freq_hz.reshape(-1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    im = axes[0].imshow(disp, aspect='auto', origin='lower', extent=[freq[0]/1e3, freq[-1]/1e3, tile.frame_times_s[0], tile.frame_times_s[-1]], cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axes[0], label='QC display (dB, broad-baseline removed)')
    axes[0].set_title(f"QC Waterfall | {obs.meta.obs_id}")
    axes[0].set_ylabel("Time (s)")
    axes[1].plot(freq/1e3, mean_spec, lw=1.0)
    axes[1].set_xlabel("Frequency (kHz)")
    axes[1].set_ylabel("Mean excess (dB)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(str(out))
    return 0


def cmd_search_run(args):
    cfg = load_runtime_config(args.config)
    cfg = _override_runtime_thresholds(cfg, args)
    obs_list = _obs_from_args_or_manifest(args, cfg)
    run_dir = Path(args.run_dir)
    hits, events, candidates, coincidences = _run_pipeline(obs_list, cfg, run_dir=run_dir)
    summary = {"n_hits": len(hits), "n_events": len(events), "n_candidates": len(candidates), "n_coincidences": len(coincidences)}
    print(json.dumps(summary, indent=2))
    return 0


def _load_records(path: Path):
    items = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                items.append(json.loads(line))
    return items


def _select_review_records(run_dir: Path, scope: str) -> tuple[list[dict], str]:
    """Load records for review-build.

    ``candidates`` preserves the historical behavior: review final candidates,
    falling back to events only when no candidates exist. ``all-events`` reviews
    every event from events.jsonl so SETI analysts can manually inspect objects
    that did not pass the automatic candidate gate.
    """
    scope = (scope or "candidates").replace("_", "-")
    if scope in {"all", "events", "all-events"}:
        return _load_records(run_dir / 'events.jsonl'), "all-events"
    if scope != "candidates":
        raise ValueError(f"unsupported review scope: {scope}")
    records = _load_records(run_dir / 'candidates.jsonl')
    actual = "candidates"
    if not records:
        records = _load_records(run_dir / 'events.jsonl')
        actual = "all-events-fallback"
    return records, actual


def cmd_review_build(args):
    from ..review.artifacts import render_event_artifact, build_review_index
    from ..review.injection_compare import find_injection_report, select_injection_truth_for_event, write_injection_comparison
    cfg = load_runtime_config(args.config)
    run_dir = Path(args.run_dir)
    obs_list = _obs_from_args_or_manifest(args, cfg)
    obs_by_id = {o.meta.obs_id: o for o in obs_list}
    review_scope = getattr(args, 'review_scope', 'candidates')
    records, actual_scope = _select_review_records(run_dir, review_scope)
    records = sorted(records, key=lambda e: e['score'], reverse=True)
    top_k_arg = getattr(args, 'top_k', None)
    if top_k_arg is not None and int(top_k_arg) > 0:
        records = records[:int(top_k_arg)]
    elif actual_scope == "candidates" and getattr(cfg.review, 'top_k', 0) and int(cfg.review.top_k) > 0:
        records = records[:int(cfg.review.top_k)]
    review_dir_arg = getattr(args, 'review_dir', None)
    if review_dir_arg:
        review_dir = Path(review_dir_arg)
    else:
        review_dir = run_dir / ('review_all_events' if actual_scope.startswith('all-events') else 'review')
    review_dir.mkdir(parents=True, exist_ok=True)
    injection_report_path = None
    injection_report_obj = None
    if not getattr(args, 'no_injection_compare', False):
        injection_report_path = find_injection_report(run_dir, getattr(args, 'injection_report', None))
        if injection_report_path is not None:
            injection_report_obj = json.loads(Path(injection_report_path).read_text(encoding="utf-8"))
    artifacts = []
    for e in records:
        obs = obs_by_id[e['obs_id']]
        event = Event(
            event_id=e['event_id'], obs_id=e['obs_id'], beam_id=e.get('beam_id'), pol_id=e.get('pol_id'), scan_id=e.get('scan_id'), target_id=e.get('target_id'),
            row0=e['row0'], row1=e['row1'], freq_hz=e.get('freq_hz'), drift_hz_per_s=e.get('drift_hz_per_s', 0.0), score=e['score'], n_hits=e.get('n_hits',1),
            best_incoherent_snr=e.get('best_incoherent_snr',0.0), best_refined_snr=e.get('best_refined_snr'), best_width_bins=e.get('best_width_bins',1),
            best_support_fraction=e.get('best_support_fraction',0.0), best_coherent_gain_db=e.get('best_coherent_gain_db'),
            representative_coarse_channel=e.get('representative_coarse_channel'), peak_row=e.get('peak_row'), beam_multiplicity=e.get('beam_multiplicity',1),
            coincident_beams=tuple(e.get('coincident_beams', [])), is_multibeam_coincident=bool(e.get('is_multibeam_coincident', False)), coincidence_group_id=e.get('coincidence_group_id'),
            candidate_passed=bool(e.get('candidate_passed', False)), candidate_reasons=tuple(e.get('candidate_reasons', [])), notes=tuple(e.get('notes', [])),
        )
        truth = select_injection_truth_for_event(injection_report_obj, e, cfg) if injection_report_obj is not None else None
        truth_width_hz = None if truth is None else float(truth.get("width_hz", 0.0))
        artifact = render_event_artifact(
            obs,
            event,
            review_dir,
            cfg.review,
            cfg.search_stft,
            cfg.baseline,
            cfg.measurement,
            truth_width_hz,
            truth,
        )
        artifact["review_scope"] = actual_scope
        artifacts.append(artifact)
    build_review_index(review_dir, artifacts)
    if not getattr(args, 'no_injection_compare', False):
        if injection_report_path is not None:
            summary = write_injection_comparison(review_dir, injection_report_path, artifacts, cfg)
            print(
                f"Wrote injection comparison for {summary['n_matched']}/{summary['n_injected_signals']} "
                f"matched signals to {review_dir / 'injection_comparison.csv'}"
            )
    print(f"Wrote {len(artifacts)} review artifacts to {review_dir}")
    return 0


def _split_csv_arg(value):
    if value is None:
        return None
    items = [x.strip() for x in str(value).split(',') if x.strip()]
    return items or None


def _override_injection_plan_metadata(plan, args):
    updates = {}
    for attr in ('obs_id', 'source_name', 'beam_id', 'pol_id', 'scan_id', 'target_id'):
        value = getattr(args, attr, None)
        if value is not None:
            updates[attr] = value
    return replace(plan, **updates) if updates else plan


def cmd_dump_injection_suite(args):
    cfg = load_runtime_config(args.config)
    plan = build_suite_plan(
        cfg.contract,
        args.suite,
        plan_name=args.plan_name,
        case_names=_split_csv_arg(args.case_names),
    )
    plan = _override_injection_plan_metadata(plan, args)
    out = save_injection_plan(args.out, plan)
    payload = {
        'suite': args.suite,
        'out': str(out),
        'plan_name': plan.plan_name,
        'n_signals': len(plan.signals),
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_inject_signals(args):
    cfg = load_runtime_config(args.config)
    plan = load_injection_plan(args.plan, cfg.contract)
    plan = _override_injection_plan_metadata(plan, args)
    outputs = write_injected_dataset(
        args.base_dat_path,
        args.out_dat,
        cfg.contract,
        plan,
        report_path=args.report_out,
        manifest_path=args.manifest_out,
        stft_nfft=cfg.search_stft.nfft,
    )
    report_payload = {}
    if outputs.get('report') is not None:
        try:
            report_payload = json.loads(Path(outputs['report']).read_text(encoding='utf-8'))
        except Exception:
            report_payload = {}
    payload = {
        'base_dat': str(args.base_dat_path),
        'out_dat': str(outputs['out_dat']),
        'report': None if outputs['report'] is None else str(outputs['report']),
        'manifest': None if outputs['manifest'] is None else str(outputs['manifest']),
        'plan_name': plan.plan_name,
        'n_signals': len(plan.signals),
        'n_signals_limited_by_clip': int(report_payload.get('n_signals_limited_by_clip', 0)),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _best_matching_event(events, case):
    best = None
    best_err = 1e99
    for e in events:
        if e.freq_hz is None:
            continue
        ferr = abs((e.freq_hz or 0.0) - case.abs_freq_hz)
        derr = abs(float(e.drift_hz_per_s) - float(case.drift_hz_per_s))
        err = ferr + 10.0 * derr
        if err < best_err:
            best = e
            best_err = err
    return best


def cmd_benchmark_inject(args):
    cfg = load_runtime_config(args.config)
    cfg = _override_runtime_thresholds(cfg, args)
    base_dat = Path(args.dat_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_names = _split_csv_arg(getattr(args, 'case_names', None))
    cases = build_case_suite(cfg.contract, args.suite)
    # widen the in-memory drift plan for benchmark suites if the configured search span is narrower
    if cases:
        max_abs_drift = max(abs(float(c.drift_hz_per_s)) for c in cases)
        if max_abs_drift > max(abs(cfg.drift.min_hz_per_s), abs(cfg.drift.max_hz_per_s)):
            pad = 4.0
            cfg = replace(cfg, drift=replace(cfg.drift, min_hz_per_s=-(max_abs_drift + pad), max_hz_per_s=(max_abs_drift + pad)))
    if case_names:
        wanted = set(case_names)
        cases = [c for c in cases if c.name in wanted]
    if args.max_cases is not None:
        cases = cases[:int(args.max_cases)]
    results = []
    success = 0
    for case_idx, case in enumerate(cases, start=1):
        print(f"[benchmark] case {case_idx}/{len(cases)}: {case.name}", flush=True)
        case_dir = out_dir / case.name
        paths = write_injected_observation(base_dat, case_dir, cfg.contract, case, stft_nfft=cfg.search_stft.nfft)
        manifest = {'observations': [{
            'obs_id': case.name + '_beam00',
            'dat_paths': [str(paths[0])],
            'source_name': 'inject',
            'beam_id': '00',
            'pol_id': '00',
            'scan_id': case.name,
            'target_id': case.name,
            'lo_hz': cfg.contract.lo_hz,
            'start_coarse_channel': cfg.contract.start_coarse_channel,
        }]}
        mpath = case_dir / 'manifest.yaml'
        mpath.write_text(yaml.safe_dump(manifest), encoding='utf-8')
        obs_list = _obs_from_args_or_manifest(argparse.Namespace(manifest=str(mpath), dat_paths=[], obs_id=None, source_name=None, beam_id=None, pol_id=None, scan_id=None, target_id=None, lo_hz=None, start_coarse_channel=None), cfg)
        hits, events, candidates, coincidences = _run_pipeline(obs_list, cfg, run_dir=case_dir / 'run')
        target_pool = candidates if candidates else events
        hit = _best_matching_event(target_pool, case)
        detected = bool(hit is not None and abs((hit.freq_hz or 0.0) - case.abs_freq_hz) <= case.freq_tolerance_hz and abs(float(hit.drift_hz_per_s) - float(case.drift_hz_per_s)) <= case.drift_tolerance_hz_per_s)
        from ..review.artifacts import render_event_artifact, build_review_index
        review_dir = case_dir / 'review'
        artifacts = []
        if hit is not None:
            obs = obs_list[0]
            artifacts.append(render_event_artifact(obs, hit, review_dir, cfg.review, cfg.search_stft, cfg.baseline))
            build_review_index(review_dir, artifacts)
        result = {
            'case': case.name,
            'morphology': case.morphology,
            'snr_db': case.snr_db,
            'expected_freq_hz': case.abs_freq_hz,
            'expected_drift_hz_per_s': case.drift_hz_per_s,
            'freq_tolerance_hz': case.freq_tolerance_hz,
            'drift_tolerance_hz_per_s': case.drift_tolerance_hz_per_s,
            'detected': bool(detected),
            'recovered_freq_hz': None if hit is None else hit.freq_hz,
            'recovered_drift_hz_per_s': None if hit is None else hit.drift_hz_per_s,
            'recovered_score': None if hit is None else hit.score,
            'used_candidates': bool(bool(candidates)),
            'n_hits': len(hits),
            'n_events': len(events),
            'n_candidates': len(candidates),
        }
        results.append(result)
        success += int(bool(detected))
        (out_dir / 'benchmark_progress.json').write_text(json.dumps({'n_cases_done': len(results), 'n_detected': success, 'pass_fraction_so_far': success / max(len(results), 1), 'results': results}, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
    summary = {'suite': args.suite, 'n_cases': len(cases), 'n_detected': success, 'pass_fraction': success / max(len(cases), 1), 'results': results}
    (out_dir / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0


def cmd_smoke_sample(args):
    cfg = load_runtime_config(args.config)
    cfg = _override_runtime_thresholds(cfg, args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'observations': [{
        'obs_id': args.obs_id,
        'dat_paths': [str(Path(args.dat_path))],
        'source_name': args.source_name,
        'beam_id': args.beam_id,
        'pol_id': args.pol_id,
        'scan_id': args.scan_id,
        'target_id': args.target_id,
        'lo_hz': args.lo_hz if args.lo_hz is not None else cfg.contract.lo_hz,
        'start_coarse_channel': args.start_coarse_channel if args.start_coarse_channel is not None else cfg.contract.start_coarse_channel,
    }]}
    manifest_path = out_dir / 'smoke_manifest.yaml'
    manifest_path.write_text(yaml.safe_dump(manifest), encoding='utf-8')
    total_bytes, rows = validate_geometry(args.dat_path, cfg.contract)
    inspect_payload = {'dat_path': str(args.dat_path), 'rows': rows, 'total_bytes': total_bytes, 'coarse_df_hz': cfg.contract.coarse_df_hz, 'native_dt_s': cfg.contract.native_dt_s}
    (out_dir / 'inspect.json').write_text(json.dumps(inspect_payload, indent=2), encoding='utf-8')
    obs_list = _obs_from_args_or_manifest(argparse.Namespace(manifest=str(manifest_path), dat_paths=[], obs_id=None, source_name=None, beam_id=None, pol_id=None, scan_id=None, target_id=None, lo_hz=None, start_coarse_channel=None), cfg)
    obs = obs_list[0]
    row1 = min(obs.total_rows, cfg.search_tile_rows)
    tile = build_spectrogram_tile(obs, 0, row1, cfg.search_stft)
    tile = apply_baseline_and_masks(tile, cfg.baseline)
    disp, vmin, vmax = qc_display_matrix(tile, cfg.baseline)
    import matplotlib.pyplot as plt
    mean_spec = tile.mean_excess_db.reshape(-1)
    freq = tile.fine_freq_hz.reshape(-1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    im = axes[0].imshow(disp, aspect='auto', origin='lower', extent=[freq[0]/1e3, freq[-1]/1e3, tile.frame_times_s[0], tile.frame_times_s[-1]], cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axes[0], label='QC display (dB, broad-baseline removed)')
    axes[0].set_title(f"QC Waterfall | {obs.meta.obs_id}")
    axes[0].set_ylabel('Time (s)')
    axes[1].plot(freq/1e3, mean_spec, lw=1.0)
    axes[1].set_xlabel('Frequency (kHz)')
    axes[1].set_ylabel('Mean excess (dB)')
    fig.savefig(out_dir / 'qc.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    hits, events, candidates, coincidences = _run_pipeline(obs_list, cfg, run_dir=out_dir / 'run')
    from ..review.artifacts import render_event_artifact, build_review_index
    review_dir = out_dir / 'run' / 'review'
    artifacts = []
    pool = candidates if candidates else events
    for e in pool[:1]:
        artifacts.append(render_event_artifact(obs, e, review_dir, cfg.review, cfg.search_stft, cfg.baseline, cfg.measurement))
    build_review_index(review_dir, artifacts)
    summary = {'n_hits': len(hits), 'n_events': len(events), 'n_candidates': len(candidates), 'n_coincidences': len(coincidences)}
    (out_dir / 'smoke_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog='acs')
    sub = p.add_subparsers(dest='cmd', required=True)

    pi = sub.add_parser('inspect-format')
    pi.add_argument('dat_path')
    pi.add_argument('--channels', type=int, default=256)
    pi.add_argument('--bytes-per-cell', type=int, default=2)
    pi.add_argument('--frontend-fs-hz', type=float, default=1_000_000_000.0)
    pi.add_argument('--channelizer-fft', type=int, default=65536)
    pi.add_argument('--lo-hz', type=float, default=None)
    pi.add_argument('--start-coarse-channel', type=int, default=None)
    pi.set_defaults(func=cmd_inspect_format)

    for name, func in [('qc-waterfall', cmd_qc_waterfall), ('search-run', cmd_search_run), ('review-build', cmd_review_build)]:
        sp = sub.add_parser(name)
        sp.add_argument('--config', required=True)
        sp.add_argument('--manifest', default=None)
        sp.add_argument('dat_paths', nargs='*')
        sp.add_argument('--obs-id', default=None)
        sp.add_argument('--source-name', default=None)
        sp.add_argument('--beam-id', default=None)
        sp.add_argument('--pol-id', default=None)
        sp.add_argument('--scan-id', default=None)
        sp.add_argument('--target-id', default=None)
        sp.add_argument('--lo-hz', type=float, default=None)
        sp.add_argument('--start-coarse-channel', type=int, default=None)
        if name == 'qc-waterfall':
            sp.add_argument('--out', required=True)
        elif name == 'search-run':
            sp.add_argument('--run-dir', required=True)
            sp.add_argument('--drift-snr-threshold', type=float, default=None)
            sp.add_argument('--refine-min-snr', type=float, default=None)
            sp.add_argument('--candidate-min-refined-snr', type=float, default=None)
            sp.add_argument('--candidate-min-score', type=float, default=None)
        else:
            sp.add_argument('--run-dir', required=True)
            sp.add_argument('--top-k', type=int, default=None, help='Maximum review records. For --review-scope all-events, omit or set 0 to render every event.')
            sp.add_argument('--review-scope', choices=['candidates', 'all-events', 'events', 'all'], default='candidates', help='Review final candidates only, or render all events for manual SETI inspection.')
            sp.add_argument('--review-dir', default=None, help='Optional output directory for review artifacts. Defaults to run/review or run/review_all_events.')
            sp.add_argument('--injection-report', default=None, help='Optional path to an inject-signals *.inject.json report. If omitted, review-build auto-detects one next to the run directory.')
            sp.add_argument('--no-injection-compare', action='store_true', help='Disable automatic injection-vs-candidate comparison output.')
        sp.set_defaults(func=func)


    sd = sub.add_parser('dump-injection-suite')
    sd.add_argument('--config', required=True)
    sd.add_argument('--suite', default='v018_full30')
    sd.add_argument('--out', required=True)
    sd.add_argument('--plan-name', default=None)
    sd.add_argument('--case-names', default=None)
    sd.add_argument('--obs-id', default=None)
    sd.add_argument('--source-name', default=None)
    sd.add_argument('--beam-id', default=None)
    sd.add_argument('--pol-id', default=None)
    sd.add_argument('--scan-id', default=None)
    sd.add_argument('--target-id', default=None)
    sd.set_defaults(func=cmd_dump_injection_suite)

    si = sub.add_parser('inject-signals')
    si.add_argument('base_dat_path')
    si.add_argument('--config', required=True)
    si.add_argument('--plan', required=True)
    si.add_argument('--out-dat', required=True)
    si.add_argument('--report-out', default=None)
    si.add_argument('--manifest-out', default=None)
    si.add_argument('--obs-id', default=None)
    si.add_argument('--source-name', default=None)
    si.add_argument('--beam-id', default=None)
    si.add_argument('--pol-id', default=None)
    si.add_argument('--scan-id', default=None)
    si.add_argument('--target-id', default=None)
    si.set_defaults(func=cmd_inject_signals)

    sb = sub.add_parser('benchmark-inject')
    sb.add_argument('dat_path')
    sb.add_argument('--config', required=True)
    sb.add_argument('--out-dir', required=True)
    sb.add_argument('--suite', default='v018_phase1_5')
    sb.add_argument('--max-cases', type=int, default=None)
    sb.add_argument('--case-names', default=None)
    sb.add_argument('--drift-snr-threshold', type=float, default=None)
    sb.add_argument('--refine-min-snr', type=float, default=None)
    sb.add_argument('--candidate-min-refined-snr', type=float, default=None)
    sb.add_argument('--candidate-min-score', type=float, default=None)
    sb.set_defaults(func=cmd_benchmark_inject)

    sm = sub.add_parser('smoke-sample')
    sm.add_argument('dat_path')
    sm.add_argument('--config', required=True)
    sm.add_argument('--out-dir', required=True)
    sm.add_argument('--obs-id', default='smoke_obs')
    sm.add_argument('--source-name', default='sample')
    sm.add_argument('--beam-id', default='00')
    sm.add_argument('--pol-id', default='00')
    sm.add_argument('--scan-id', default='smoke_scan')
    sm.add_argument('--target-id', default='sample')
    sm.add_argument('--lo-hz', type=float, default=None)
    sm.add_argument('--start-coarse-channel', type=int, default=None)
    sm.add_argument('--drift-snr-threshold', type=float, default=None)
    sm.add_argument('--refine-min-snr', type=float, default=None)
    sm.add_argument('--candidate-min-refined-snr', type=float, default=None)
    sm.add_argument('--candidate-min-score', type=float, default=None)
    sm.set_defaults(func=cmd_smoke_sample)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == '__main__':
    raise SystemExit(main())
