# Score Handbook

ACS uses three object levels:

- **Hit**: one tile-local detection from one seed and one drift hypothesis.
- **Event**: a merged group of compatible hits in the same beam/observation.
- **Candidate**: an event that passes the candidate gate.

## Important clarification

An event does **not** need to exceed a single global score threshold to exist.
An event is formed structurally by merging compatible hits.
A **candidate** is the first object that must satisfy explicit thresholds.

## Candidate thresholds (from config)

Defaults are in `configs/h1_search.yaml` and the runtime config loader.
Typical gates include:
- minimum event score
- minimum refined SNR
- maximum width
- minimum support fraction
- stronger requirements for singleton events

## How to interpret scores

The score is a ranking statistic, not a formal probability.
Higher means the event is more consistent with a narrowband-like signal under the current single-beam rules.
Low double-digit scores can still be normal, especially on noisy real observations.
The score should always be interpreted together with:
- `best_refined_snr`
- `n_hits`
- `best_support_fraction`
- `candidate_passed`
- `candidate_reasons`


## SNR terminology

`snr_db` in an injection YAML file is a raw active-sample RMS ratio relative to the selected coarse-channel RMS before the synthetic tone is added.  It is useful for controlling injection amplitude, but it is **not** the same quantity as the SNR-like columns in `review/index.csv`.

`best_incoherent_snr` is a drift-path score computed from baseline-normalized STFT power.  `best_refined_snr` is a coherent dechirp/FFT peak-to-median amplitude ratio.  Continuous tones integrate over many samples, so these search metrics are expected to be tens of dB higher than the raw YAML injection SNR.  For a 2.048 s tile with 31,250 complex samples, the coherent integration scale alone is roughly `10 log10(31250) ≈ 45 dB` before window and noise-statistic constants.

Use `drift.snr_threshold` as a coarse-hit threshold, `refine.min_refined_snr` as the minimum refined hit threshold, and `candidate.min_refined_snr` plus `candidate.min_event_score` as final candidate gates.  These thresholds are search-metric thresholds and should be calibrated with injection/recovery tests when the science requirement is stated as a raw time-domain SNR such as 5 dB or 10 dB.


## v1.0.0x injection comparison

When a review build is associated with an injection report, ACS writes `review/injection_comparison.csv` and related JSON/HTML files. These outputs match injected signals to recovered candidates and explicitly report both raw injected SNR and recovered search metrics. Use these comparison outputs for injection-recovery curves and SETI sensitivity/EIRP calibration.
