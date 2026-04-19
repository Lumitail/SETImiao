# Changelog

## v1.0.0x

- Added automatic injection-vs-detection comparison during `review-build`.
- Added `review/injection_comparison.csv`, `review/injection_comparison.json`, `review/injection_comparison_summary.json`, and `review/injection_comparison.html`.
- Added one-to-one injection/candidate matching with explicit `matched`, `unmatched_injected`, and `unmatched_candidate` rows.
- Added nominal raw-SNR estimates by subtracting approximate STFT/coherent processing gain from search metrics.
- Added empirical refined-metric calibration when a matched injection sweep is present.
- Clarified that `best_incoherent_snr` and `best_refined_snr` are detector metrics, not raw injected SNR.
- Added `--injection-report` and `--no-injection-compare` options to `review-build`.
- Added regression tests for injection comparison and report discovery.
- Rewrote README for GitHub release and fixed the MIT license file.

## v0.2.5X

- Hardened continuous narrowband candidate gating: single-tile coherent peaks are retained in `events.jsonl` for diagnostics but are no longer promoted to final candidates by default.
- Increased default sensitivity headroom for low-SNR multi-signal injections by raising search seed counts and coherent-refine `top_n` in `configs/h1_search.yaml`.
- Extended same-track merge span (`candidate.merge_max_gap_s`) so weak continuous signals with intermittent tile-level detections can merge into one event instead of producing multiple short candidates.
- Added explicit candidate-quality controls: `allow_singleton_candidates`, `min_candidate_hits`, and `min_candidate_duration_s`.
- Added 35-second low-SNR validation examples and fast test configs for `test1.dat`-style data.
- Added regression tests for single-tile candidate suppression and long-gap weak-track merging.

## v0.2.4X

- Added short-fragment-aware event merging and candidate deduplication to suppress two-hit edge fragments that belong to a longer same-channel narrowband track.
- Added explicit search-metric SNR fields and notes to review JSON/CSV so detector ranking metrics are not confused with YAML injection `snr_db`.
- Added optional `ACS_DISABLE_NUMBA=1` fallback support and lazy review-artifact import to make non-review CLI commands independent of Matplotlib import side effects.
- Updated default candidate merge/dedup configuration with short-fragment tolerances.

## v0.2.3

- Fixed review overview/local plot autoscaling so track overlays no longer truncate waterfall images along frequency.
- Improved same-channel event merge and candidate dedup for short adjacent fragments of one continuous track.
- Reworked injection headroom handling to use per-channel uniform rescaling instead of crushing individual signals.
- Added optional CLI search threshold overrides on `search-run`, `benchmark-inject`, and `smoke-sample`.

## v0.2.2

- Fixed long-event support estimation so long injections no longer collapse to identical raw tile durations.
- Added second-stage event merging and candidate deduplication for overlapping same-channel tracks.
- Added clip-aware amplitude limiting with requested vs realized SNR reporting in injection JSON.
- Centered local review previews on the event peak and overlaid the predicted track.
- Exposed new candidate merge/dedup knobs in `configs/h1_search.yaml`.
- Corrected `examples/multisignal_10x40k.example.yaml` so all 10 signals are actually 40 kHz spaced in that historical version.
- `inject-signals` now reports how many signals were clip-limited.

## v0.1.9a0

- Added independent `inject-signals` CLI for writing one new `.dat` with multiple synthetic signals.
- Added editable YAML injection plans with top-level `defaults` and examples for multi-signal workflows.
- Added `dump-injection-suite` CLI to export built-in benchmark cases into YAML for further editing.
- Stabilized multi-signal amplitude calibration by anchoring each signal to the original channel RMS before any injections.
- Added injection JSON reports with resolved target channels and clipping summaries.
- Bumped package version metadata to 0.1.9a0.

## v0.1.8a0

- Added built-in injection suites: `v018_phase1_5` and `v018_full30`.
- Expanded injected morphology support for broadband-like and checkerboard-like RFI cases.
- Updated benchmark CLI and README with Linux run instructions and signal customization guidance.
- Added score/threshold handbook material for hits, events, and candidates.

## v0.1.7a0

- Separated QC display normalization from search normalization.
- Defined strict hit/event/candidate boundaries.
- Added single-beam morphology-aware scoring.
- Added `v017_10` injection benchmark suite.

## v0.1.5a0

- Rebuilt baseline from the frozen ACS design.
- Improved README and docs.
- Fixed STFT fine-frequency axis contract.
- Redesigned seed extraction using mean-driven and max-driven seed families.
- Added anchor-frame-aware drift search.
- Improved coherent refinement with ridge-centered drift estimation.
- Expanded benchmark campaign with 15 new single-beam cases.
- Preserved real 2-second smoke workflow with no runtime errors.
