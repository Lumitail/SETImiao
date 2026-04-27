# Adaptive Chirp Search (ACS) v0.2.3

ACS is a research-grade narrowband search tool for channelized raw `.dat` radio data. It assumes the exact contract validated from the development brief and QC script:

- headerless binary payload
- row-major `(rows, 256)` layout
- `2` bytes per cell
- packed signed-int8 I/Q inside each 16-bit word
- metadata supplied externally
- FFTs performed **along time inside each coarse channel**

---

## What ACS does

ACS currently provides:

1. `.dat` contract validation and parsing
2. filename + manifest metadata handling
3. stitched observation construction from multiple `.dat` files
4. QC waterfall generation
5. single-beam seeded drift search
6. raw-complex coherent refinement
7. hit / event / candidate outputs
8. review artifact generation (PNG / PDF / JSON / HTML)
9. independent multi-signal injection plans and built-in benchmark suites

---

## Hits, events, and candidates

### Hit
A **hit** is a tile-local detection produced from one seed, one drift hypothesis, and one local width. Hits are deliberately over-inclusive.

### Event
An **event** is a merged group of compatible hits in the same observation/beam. Event formation is structural; it is not the same thing as a final scientific decision.

### Candidate
A **candidate** is an event that passes the configured candidate gate. In the default single-beam configuration, candidate gating uses:

- event score
- refined SNR
- maximum width
- support fraction
- stricter handling of singleton events

See `docs/score_handbook.md` for the full interpretation guide.

---

## Linux setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Run the test suite:

```bash
pytest -q
```

---

## Standard workflow

### 1. Inspect one `.dat`

```bash
acs inspect-format /path/to/file.dat --lo-hz 1000000000 --start-coarse-channel 27392
```

### 2. QC waterfall

```bash
acs qc-waterfall \
  --config configs/h1_search.yaml \
  /path/to/file.dat \
  --obs-id sample --source-name sample --beam-id 00 --pol-id 00 \
  --scan-id scan0 --target-id sample \
  --lo-hz 1000000000 --start-coarse-channel 27392 \
  --out outputs/qc.png
```

### 3. Search run

```bash
acs search-run \
  --config configs/h1_search.yaml \
  /path/to/file.dat \
  --obs-id sample --source-name sample --beam-id 00 --pol-id 00 \
  --scan-id scan0 --target-id sample \
  --lo-hz 1000000000 --start-coarse-channel 27392 \
  --run-dir outputs/run
```

You can optionally override the main detection thresholds at runtime without editing YAML:

```bash
acs search-run \
  --config configs/h1_search.yaml \
  /path/to/file.dat \
  --obs-id sample --source-name sample --beam-id 00 --pol-id 00 \
  --scan-id scan0 --target-id sample \
  --lo-hz 1000000000 --start-coarse-channel 27392 \
  --drift-snr-threshold 1.5 \
  --refine-min-snr 14.5 \
  --candidate-min-refined-snr 15.5 \
  --candidate-min-score 14.0 \
  --run-dir outputs/run
```

### 4. Build review artifacts

```bash
acs review-build \
  --config configs/h1_search.yaml \
  /path/to/file.dat \
  --obs-id sample --source-name sample --beam-id 00 --pol-id 00 \
  --scan-id scan0 --target-id sample \
  --lo-hz 1000000000 --start-coarse-channel 27392 \
  --run-dir outputs/run --top-k 20
```

### 5. Manifest-based run (recommended for long observations)

```bash
acs search-run --config configs/h1_search.yaml --manifest path/to/manifest.yaml --run-dir outputs/run
acs review-build --config configs/h1_search.yaml --manifest path/to/manifest.yaml --run-dir outputs/run --top-k 20
```

---

## Independent multi-signal injection workflow

The new injection path is separate from search. You first write a new `.dat`, then you analyze that file with the normal inspect/QC/search/review commands.

### 1. Start from an editable YAML plan

You can begin from a built-in suite:

```bash
acs dump-injection-suite \
  --config configs/h1_search.yaml \
  --suite v018_phase1_5 \
  --out outputs/phase1_plan.yaml
```

Or start from one of the examples:

- `examples/injection_plan.example.yaml`
- `examples/multisignal_10x40k.example.yaml`
- `examples/test1_multisignal_35s.example.yaml`

The YAML supports:

- `defaults:` for global parameters shared by many signals
- one `signals:` list entry per injected signal
- independent per-signal control of `start_s`, `duration_s`, `start_freq_hz`, `drift_hz_per_s`, `snr_db`, and `width_hz`
- friendly aliases such as `start_freq_hz`, `width_hz`, `start_time_s`, `length_s`, `drift_rate_hz_per_s`, and `snr`
- `snr_db` is normalized against the active-sample RMS of the synthesized signal, so width / multi-tone signals now preserve the requested RMS target
- single-channel narrowband signals are validated against the coarse-channel band; oversized `width_hz` or drift excursions now raise a clear error instead of aliasing silently
- frequency specified as either `start_freq_hz`, or `channel_index + offset_hz`, or `coarse_channel + offset_hz`

### 2. Inject one or many signals into a new `.dat`

```bash
acs inject-signals /path/to/base.dat \
  --config configs/h1_search.yaml \
  --plan examples/injection_plan.example.yaml \
  --out-dat outputs/injected.dat \
  --report-out outputs/injected.dat.inject.json \
  --manifest-out outputs/injected_manifest.yaml
```

This command only writes files. It does **not** run QC, search, or review.

The JSON report records:

- every requested signal
- the resolved per-signal start time, duration, start frequency, drift, SNR, and width
- the resolved target channel(s)
- per-channel headroom scaling (`channel_scale`) when the requested composite would exceed int8 range
- realized SNR relative to both the final written channel RMS and the original base channel RMS
- clipping statistics after all signals are combined

### 3. Run the normal ACS workflow on the injected `.dat`

Direct-file style:

```bash
acs inspect-format outputs/injected.dat --lo-hz 1000000000 --start-coarse-channel 27392
acs qc-waterfall --config configs/h1_search.yaml outputs/injected.dat --obs-id inject_obs --source-name inject --beam-id 00 --pol-id 00 --scan-id inject --target-id inject --lo-hz 1000000000 --start-coarse-channel 27392 --out outputs/qc.png
acs search-run --config configs/h1_search.yaml outputs/injected.dat --obs-id inject_obs --source-name inject --beam-id 00 --pol-id 00 --scan-id inject --target-id inject --lo-hz 1000000000 --start-coarse-channel 27392 --run-dir outputs/run
acs review-build --config configs/h1_search.yaml outputs/injected.dat --obs-id inject_obs --source-name inject --beam-id 00 --pol-id 00 --scan-id inject --target-id inject --lo-hz 1000000000 --start-coarse-channel 27392 --run-dir outputs/run --top-k 20
```

Manifest style:

```bash
acs search-run --config configs/h1_search.yaml --manifest outputs/injected_manifest.yaml --run-dir outputs/run
acs review-build --config configs/h1_search.yaml --manifest outputs/injected_manifest.yaml --run-dir outputs/run --top-k 20
```

---

## Built-in benchmark workflow

`benchmark-inject` is still available when you want injection plus immediate recovery scoring in one command.

### Fast 5-case verification on the 2-second probe

```bash
acs benchmark-inject /path/to/probe_2s.dat \
  --config configs/h1_search.yaml \
  --out-dir outputs/bench_phase1 \
  --suite v018_phase1_5
```

This runs exactly 5 representative signals:
- 2 SETI-like narrowband cases
- 3 RFI-like cases

### Full 30-case packaged suite

```bash
acs benchmark-inject /path/to/real_observation.dat \
  --config configs/h1_search.yaml \
  --out-dir outputs/bench_full30 \
  --suite v018_full30
```

The full suite contains:
- **10 SETI-like narrowband signals**
- **20 RFI-like signals**

### Restrict to specific cases

```bash
acs benchmark-inject /path/to/file.dat \
  --config configs/h1_search.yaml \
  --out-dir outputs/bench_subset \
  --suite v018_full30 \
  --case-names seti_01_cw_w5_snr20,rfi_01_checkerboard_low
```

### Where built-in morphologies still live

If you need to add a new morphology or change the packaged benchmark suites themselves, edit:

```text
src/acs/bench/inject.py
```

See `docs/injection_handbook.md` for the YAML schema and field reference.

---

## Output files

A search run writes:
- `hits.jsonl`
- `events.jsonl`
- `candidates.jsonl`
- `coincidences.jsonl`
- `summary.json`
- `run_metadata.json`

A review build writes:
- `review/index.html`
- `review/index.csv`
- `review/index.json`
- one PNG / PDF / JSON set per reviewed object

Review artifacts now include a full-event overview panel with observation-time axes, plus a local preview panel around the event center.

An independent injection run writes:
- the new injected `.dat`
- an optional manifest YAML
- an injection JSON report with resolved channels and clipping statistics

A benchmark run writes:
- one folder per case
- a review folder for the best recovered object (when present)
- `benchmark_progress.json`
- `benchmark_summary.json`

---

## Documentation

- `docs/data_contract.md`
- `docs/manifest_schema.md`
- `docs/review_artifacts.md`
- `docs/injection_handbook.md`
- `docs/score_handbook.md`

---

## Current limitations

- scoring is still single-beam focused
- true multi-beam veto logic is deferred
- GPU acceleration is optional and only used if CuPy is installed
- the 2-second `.dat` is a format/QC probe, not a full science dwell

ACS v0.2.2 is a research alpha, not a final production release.


## v0.2.4X note

This version adds short-fragment-aware event merging/deduplication and clarifies that review `best_*_snr` columns are search ranking metrics, not the raw injection `snr_db` scale.


### v0.2.5X continuous-narrowband candidate policy

For long-duration narrowband surveys, a `Hit` is a local tile-level detection, not a whole signal. A continuous injected or astrophysical signal should normally produce many hits that merge into one event/candidate. Starting in v0.2.5X, the default `configs/h1_search.yaml` therefore does **not** promote one-tile events directly to final candidates. They still remain in `events.jsonl` for diagnostics, but final candidates require multi-hit, multi-second track support by default:

```yaml
candidate:
  allow_singleton_candidates: false
  min_candidate_hits: 3
  min_candidate_duration_s: 4.0
  merge_max_gap_s: 35.0
```

This prevents the common low-SNR failure mode where isolated 2.048 s tiles appear as separate final candidates. To intentionally search for very short bursts, lower `min_candidate_hits`/`min_candidate_duration_s` or set `allow_singleton_candidates: true` in a separate burst-search config.

The search/refine defaults are also more sensitive to weak multi-signal runs:

```yaml
drift:
  seed_top_k_mean_per_channel: 8
  seed_top_k_max_per_channel: 8
refine:
  top_n: 64
```

Those settings reduce the risk that weak signals are excluded before coherent refinement when many stronger injected signals are present in the same run.

### v1.0.0x automatic injection comparison

When `review-build` is run after an independent `inject-signals` step, ACS now looks for a `*.inject.json` report next to the run directory. If found, it automatically writes:

- `review/injection_comparison.csv`
- `review/injection_comparison.json`
- `review/injection_comparison_summary.json`
- `review/injection_comparison.html`

These files match injected signals to final candidates and list injected and recovered duration, frequency, drift, and SNR-related quantities. Unmatched injected signals and extra unmatched candidates are listed explicitly.

The injected `snr_db` in a YAML plan is a raw active-sample RMS ratio. The recovered `best_incoherent_snr` and `best_refined_snr` columns in `index.csv` are detector/search metrics after STFT and coherent processing. They include processing gain and should not be interpreted as raw signal SNR. The new comparison CSV therefore includes approximate raw-SNR estimates such as `estimated_raw_snr_db_from_refined` for injection-calibrated sensitivity analysis.

No command changes are required for normal runs. To force a specific report path:

```bash
acs review-build --config configs/h1_search.yaml --manifest outputs/injected_manifest.yaml \
  --run-dir outputs/run --top-k 100 --injection-report outputs/injected.inject.json
```

To disable the automatic comparison:

```bash
acs review-build ... --no-injection-compare
```


### Local-PSD injection SNR reference

By default, `inject-signals` keeps the historical behavior: `snr_db` is referenced to the full coarse-channel time-domain RMS. For narrowband SETI sensitivity studies, use the local-PSD reference instead:

```yaml
defaults:
  snr_reference: local_psd
  noise_estimator: coarse_psd

signals:
  - name: example_local_psd
    start_freq_hz: 1418500000.0
    start_s: 1.0
    duration_s: 20.0
    drift_hz_per_s: 2.0
    snr_db: 5.0
    width_hz: 0.0
```

With `snr_reference: local_psd`, the requested injection power is

```text
P_signal = local_noise_psd * reference_bandwidth_hz * 10^(snr_db/10)
local_noise_psd = coarse_channel_rms^2 / coarse_channel_bandwidth_hz
```

If `width_hz > 0`, the reference bandwidth is `width_hz`. If `width_hz == 0`, the reference bandwidth is one search spectral-resolution element, `coarse_df_hz / search_stft.nfft`. The injection report records requested and realized local-PSD SNR values, and `review-build` propagates those fields into `review/injection_comparison.csv`.


## Recovered SNR measurement

`review-build` now writes post-detection recovered SNR measurements in addition
to the existing detector metrics. The existing `best_refined_snr` and
`best_incoherent_snr` columns are still search/ranking metrics with processing
gain. New fields such as `recovered_band_excess_snr_db` and
`recovered_ridge_pixel_snr_db` estimate the average candidate-track excess
relative to a local waterfall background. See
`docs/recovered_snr_measurement.md`.

### v1.1.2x review scopes

For normal operation, `review-build` still reviews final candidates:

```bash
acs review-build --config configs/h1_search.yaml --manifest outputs/injected_manifest.yaml --run-dir outputs/run
```

For manual SETI inspection of all events, use:

```bash
acs review-build --config configs/h1_search.yaml --manifest outputs/injected_manifest.yaml --run-dir outputs/run --review-scope all-events --top-k 0
```

This writes to `outputs/run/review_all_events` by default. Candidate review remains in `outputs/run/review`.


## Track-aligned review diagnostics

From v1.1.3X, review plots include a de-drifted, track-aligned view for weak continuous narrowband candidates. The ordinary waterfall panel is still shown, but low per-pixel-SNR signals may not be obvious there. The aligned panel shifts every STFT frame so the recovered event track is fixed at 0 Hz offset, and the final profile panel averages along that recovered track. These diagnostics are written automatically during `review-build` and do not change the normal command sequence.

## v1.1.4x candidate visibility and truth-matching notes

Version v1.1.4x adds safeguards for continuous narrowband review workflows. Final-candidate promotion now rejects weak short fragments unless they have sufficiently strong coherent support, while all such events remain available through `review-build --review-scope all-events`. Review artifacts include visual-evidence diagnostics based on the track-aligned profile; candidates with weak aligned evidence are flagged in the JSON/CSV outputs.

The injection comparison module now matches truth to detections using time-frequency path consistency across the overlapping interval. This is important for short signals: a 4 second, 2 Hz/s track sweeps only about one STFT bin, so its event-level drift may not be precisely constrained even when the recovered path is clearly at the injected frequency.

The normal command sequence is unchanged.
