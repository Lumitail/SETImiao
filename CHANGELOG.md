# Changelog

## v1.1.4x

- Hardened candidate promotion for continuous narrowband SETI review by adding stronger short-track thresholds and per-hit coherent-strength inlier filtering.
- Added review visual-evidence diagnostics so candidate artifacts explicitly flag weak or negative aligned-profile evidence instead of silently presenting invisible candidates.
- Added robust event-level track reporting from the consensus of event hits, reducing biased single-tile drift/frequency assignments.
- Improved injection-truth matching to use time-frequency path consistency across the overlapping track, which is scientifically better for short low-drift events whose drift is not independently well constrained.
- Added a 20-signal `test1.dat` visibility/sensitivity benchmark covering 4 s and 28 s signals at local-PSD SNR 1--5 dB and drifts 0.3/2.0 Hz/s.


## v1.1.3X

- Added track-aligned review diagnostics for weak continuous narrowband candidates.
- Review PNGs now include a de-drifted event overview and a mean track-aligned profile so low per-pixel SNR tracks can be manually vetted even when the raw waterfall overview is visually ambiguous.
- Review artifacts and `index.csv` now include `aligned_profile_peak_excess_db`, `aligned_profile_center_excess_db`, `aligned_profile_background_median_db`, `aligned_profile_peak_offset_hz`, and `aligned_half_width_bins` diagnostic fields.
- Injection-validation review plots now overlay the injected truth track when an injection report is available.
- Overview cutout selection now covers the predicted event track over the full event duration rather than centering only on one frequency.


## v1.1.0x

- Added post-detection recovered SNR measurement during `review-build`.
- Added `measurement:` runtime config and local STFT-background ridge measurement.
- Added recovered SNR fields to review index CSV/JSON, per-event JSON, and injection comparison outputs.
- Preserved existing refined/incoherent SNR values as search metrics.
- Optimized review local preview generation to read only the representative coarse channel.
- Added 15-signal `test1.dat` local-PSD validation plan and config.
- Added regression tests for recovered SNR measurement and comparison outputs.

## v1.0.2x

- Added sparse coarse-channel search support through `drift.channel_list`.
- Optimized targeted validation by reading and transforming only selected coarse channels when a sparse channel list is configured.
- Added `configs/h1_search_continuous_local_psd.yaml`, a continuous-narrowband oriented configuration using longer tiles and a coarser coherent-refine drift step for local-PSD sensitivity studies.
- Added `examples/test1_20sig_local_psd_snr1to5_v102x.yaml`, a 20-signal local-PSD sensitivity plan for the provided 35.97 s `test1.dat`.
- Added regression tests for sparse channel I/O, sparse STFT channel mapping, and `channel_list` filtering.

## v1.0.1x

- Added injection SNR reference mode `snr_reference: local_psd` with `noise_estimator: coarse_psd`.
- Preserved legacy behavior as `snr_reference: coarse_channel` and as the default when `snr_reference` is omitted.
- For `local_psd`, `snr_db` is now interpreted as signal power relative to `local_noise_psd * reference_bandwidth_hz`, where `local_noise_psd = coarse_channel_rms^2 / coarse_channel_bandwidth_hz`.
- `width_hz > 0` uses the signal width as reference bandwidth; `width_hz == 0` uses one STFT spectral-resolution element, `coarse_df_hz / search_stft.nfft`.
- Injection reports and automatic review-stage `injection_comparison.csv` now include local-PSD SNR bookkeeping fields.
- Added local-PSD regression tests and a short `test1.dat` validation example/config.
- Added optional `review.write_pdf` config to disable PDF generation in resource-constrained validation runs.


## 0.2.5X

- Hardened continuous narrowband candidate gating: single-tile coherent peaks are retained in `events.jsonl` for diagnostics but are no longer promoted to final candidates by default.
- Increased default sensitivity headroom for low-SNR multi-signal injections by raising search seed counts and coherent-refine `top_n` in `configs/h1_search.yaml`.
- Extended same-track merge span (`candidate.merge_max_gap_s`) so weak continuous signals with intermittent tile-level detections can merge into one event instead of producing multiple short candidates.
- Added explicit candidate-quality controls: `allow_singleton_candidates`, `min_candidate_hits`, and `min_candidate_duration_s`.
- Added 35-second low-SNR validation examples and fast test configs for `test1.dat`-style data.
- Added regression tests for single-tile candidate suppression and long-gap weak-track merging.



## 0.2.4X

- Added short-fragment-aware event merging and candidate deduplication to suppress two-hit edge fragments that belong to a longer same-channel narrowband track.
- Added explicit search-metric SNR fields and notes to review JSON/CSV so detector ranking metrics are not confused with YAML injection `snr_db`.
- Added optional `ACS_DISABLE_NUMBA=1` fallback support and lazy review-artifact import to make non-review CLI commands independent of Matplotlib import side effects.
- Updated default candidate merge/dedup configuration with short-fragment tolerances.


## 0.2.3
- Fixed review overview/local plot autoscaling so track overlays no longer truncate waterfall images along frequency.
- Improved same-channel event merge and candidate dedup for short adjacent fragments of one continuous track.
- Reworked injection headroom handling to use per-channel uniform rescaling instead of crushing individual signals.
- Added optional CLI search threshold overrides on search-run, benchmark-inject, and smoke-sample.

## 0.2.2

- fixed long-event support estimation so long injections no longer collapse to identical raw tile durations
- added second-stage event merging and candidate deduplication for overlapping same-channel tracks
- added clip-aware amplitude limiting with requested vs realized SNR reporting in injection JSON
- centered local review previews on the event peak and overlaid the predicted track
- exposed new candidate merge/dedup knobs in `configs/h1_search.yaml`
- corrected `examples/multisignal_10x40k.example.yaml` so all 10 signals are actually 40 kHz spaced
- `inject-signals` now reports how many signals were clip-limited

## v0.1.9a0
- added independent `inject-signals` CLI for writing one new `.dat` with multiple synthetic signals
- added editable YAML injection plans with top-level `defaults` and examples for multi-signal workflows
- added `dump-injection-suite` CLI to export built-in benchmark cases into YAML for further editing
- stabilized multi-signal amplitude calibration by anchoring each signal to the original channel RMS before any injections
- added injection JSON reports with resolved target channels and clipping summaries
- bumped package version metadata to 0.1.9a0

## v0.1.5a0
- rebuilt baseline from the frozen ACS design
- improved README and docs
- fixed STFT fine-frequency axis contract
- redesigned seed extraction using mean-driven and max-driven seed families
- added anchor-frame aware drift search
- improved coherent refinement with ridge-centered drift estimation
- expanded benchmark campaign with 15 new single-beam cases (35 total available, 15-case cycle suite)
- added bundled cycle-05 benchmark summary and review artifacts
- preserved real 2-second smoke workflow with no runtime errors


## v0.1.7a0
- separated QC display normalization from search normalization
- defined strict hit/event/candidate boundaries
- added single-beam morphology-aware scoring
- added v017_10 injection benchmark suite


## v0.1.8a0
- added built-in injection suites: v018_phase1_5 and v018_full30
- expanded injected morphology support for broadband-like and checkerboard-like RFI cases
- updated benchmark CLI and README with Linux run instructions and signal customization guidance
- added score/threshold handbook material for hits, events, and candidates

## v1.0.0x

- Added automatic injection-vs-detection comparison during `review-build`.
- `review/injection_comparison.csv`, `.json`, `.html`, and `_summary.json` are generated automatically when a `*.inject.json` report is found next to the run directory.
- The comparison table explicitly separates raw injected SNR from recovered detector metrics and adds approximate raw-SNR estimates from coherent and incoherent metrics.
- Added one-to-one injection/candidate matching with unmatched injected signals and unmatched extra candidates reported explicitly.

## v1.1.1x

- Added post-clustering track-inlier filtering so permissive low-SNR fragment merging cannot inflate event duration and recovered-SNR apertures with off-track raw/noise fragments.
- Added a fixed 15-signal `test1.dat` local-PSD benchmark and a matching targeted config.
- Tuned the fixed benchmark candidate gate to suppress unmatched candidates while preserving short two-hit detections.
- Increased the local-background SNR measurement ridge snap window in the fixed benchmark to make recovered SNR less sensitive to few-bin weak-track frequency errors.
- Added regression tests for track-inlier trimming and diagnostic disable behavior.

## v1.1.2x

- Added `review-build --review-scope all-events` for manual SETI inspection of every event, while keeping candidate-only review as the default.
- Added optional output override `review-build --review-dir` and `--top-k 0` semantics for unbounded all-event review.
- Added geometry-constrained long-gap event merging for weak continuous tracks (`candidate.merge_long_gap_*`). This is disabled in the historical `h1_search.yaml` defaults unless explicitly enabled, and enabled in the continuous local-PSD/test benchmark configs.
- Added injection-comparison `match_quality` and `match_quality_note` fields so geometry matches with poor recovered local-SNR evidence are auditable.
- Made review index and injection comparison CSV writers robust to heterogeneous artifact fields.
- Added the fixed 16-signal `test1.dat` benchmark config and injection plan.
