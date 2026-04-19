from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import yaml
from acs.config import load_runtime_config
from acs.bench.inject import build_case_suite, write_injected_observation
from acs.cli.main import _obs_from_args_or_manifest, _run_pipeline
from acs.frontend.stft import build_spectrogram_tile
from acs.preproc.baseline import apply_baseline_and_masks
from acs.review.artifacts import render_event_artifact, build_review_index


def best_matching(pool, case):
    best=None; best_err=1e99
    for e in pool:
        if e.freq_hz is None:
            continue
        ferr=abs((e.freq_hz or 0.0)-case.abs_freq_hz)
        derr=abs(float(e.drift_hz_per_s)-float(case.drift_hz_per_s))
        err=ferr+10.0*derr
        if err<best_err:
            best=e; best_err=err
    return best


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--base-dat', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--suite', default='v017_10')
    ap.add_argument('--case-names', required=True, help='comma separated')
    args=ap.parse_args()
    cfg=load_runtime_config(args.config)
    out_dir=Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_names=[x.strip() for x in args.case_names.split(',') if x.strip()]
    all_cases={c.name:c for c in build_case_suite(cfg.contract, args.suite)}
    results=[]
    for case_name in case_names:
        case=all_cases[case_name]
        case_dir=out_dir/case.name
        shutil.rmtree(case_dir, ignore_errors=True)
        case_dir.mkdir(parents=True, exist_ok=True)
        paths=write_injected_observation(args.base_dat, case_dir, cfg.contract, case)
        manifest={'observations':[{'obs_id':case.name+'_beam00','dat_paths':[str(paths[0])],'source_name':'inject','beam_id':'00','pol_id':'00','scan_id':case.name,'target_id':case.name,'lo_hz':cfg.contract.lo_hz,'start_coarse_channel':cfg.contract.start_coarse_channel}]}
        mpath=case_dir/'manifest.yaml'
        mpath.write_text(yaml.safe_dump(manifest), encoding='utf-8')
        ns=argparse.Namespace(manifest=str(mpath), dat_paths=[], obs_id=None, source_name=None, beam_id=None, pol_id=None, scan_id=None, target_id=None, lo_hz=None, start_coarse_channel=None)
        obs_list=_obs_from_args_or_manifest(ns, cfg)
        hits, events, candidates, coincidences = _run_pipeline(obs_list, cfg, run_dir=case_dir/'run')
        pool=candidates if candidates else events
        best=best_matching(pool, case)
        detected=bool(best is not None and abs((best.freq_hz or 0.0)-case.abs_freq_hz)<=case.freq_tolerance_hz and abs(float(best.drift_hz_per_s)-float(case.drift_hz_per_s))<=case.drift_tolerance_hz_per_s)
        if best is not None:
            obs=obs_list[0]
            tile=build_spectrogram_tile(obs, best.row0, best.row1, cfg.search_stft)
            tile=apply_baseline_and_masks(tile, cfg.baseline)
            art=render_event_artifact(obs, best, case_dir/'review', cfg.review, cfg.search_stft, cfg.baseline)
            build_review_index(case_dir/'review', [art])
        result={
            'case':case.name,'morphology':case.morphology,'snr_db':case.snr_db,
            'expected_freq_hz':case.abs_freq_hz,'expected_drift_hz_per_s':case.drift_hz_per_s,
            'freq_tolerance_hz':case.freq_tolerance_hz,'drift_tolerance_hz_per_s':case.drift_tolerance_hz_per_s,
            'detected':detected,
            'recovered_freq_hz':None if best is None else best.freq_hz,
            'recovered_drift_hz_per_s':None if best is None else best.drift_hz_per_s,
            'recovered_score':None if best is None else best.score,
            'used_candidates': bool(candidates),
            'n_hits':len(hits),'n_events':len(events),'n_candidates':len(candidates),
            'candidate_passed': None if best is None else best.candidate_passed,
            'candidate_reasons': None if best is None else list(best.candidate_reasons),
        }
        print(json.dumps(result), flush=True)
        results.append(result)
    summary_path=out_dir/'batch_results.json'
    summary_path.write_text(json.dumps(results, indent=2), encoding='utf-8')

if __name__=='__main__':
    main()
