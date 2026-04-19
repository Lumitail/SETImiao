# Injection comparison outputs

Starting in v1.0.0x, `review-build` automatically compares injected signals with recovered candidates whenever it can find an `*.inject.json` report. The usual independent workflow writes `outputs/injected.inject.json` and searches into `outputs/run`, so the report is discovered without additional command-line arguments.

Outputs are written under `outputs/run/review/`:

- `injection_comparison.csv`
- `injection_comparison.json`
- `injection_comparison_summary.json`
- `injection_comparison.html`

The matcher is one-to-one. It compares each injected signal to candidates using active-time overlap, predicted frequency at the candidate peak time, and drift-rate tolerance from the injection report. Each injected signal appears in the CSV either as `matched` or `unmatched_injected`. Any final candidate that is not assigned to an injected signal appears as `unmatched_candidate`.

## SNR definitions

The YAML `snr_db` is a raw active-sample RMS ratio relative to the channel RMS used by the injector.

The `index.csv` columns `best_incoherent_snr` and `best_refined_snr` are search metrics. They are expected to be much larger than the raw injected SNR for a long coherent narrowband tone. In the default config, the approximate processing gains are:

- incoherent/STFT metric: `10*log10(search_stft.nfft)`
- refined coherent metric: `10*log10(search_tile_rows)`

The comparison CSV includes:

- `best_incoherent_search_metric_db`
- `best_refined_search_metric_db`
- `estimated_raw_snr_db_from_incoherent`
- `estimated_raw_snr_db_from_refined`

The estimated raw-SNR fields are useful diagnostics for injection sweeps. Publication-grade sensitivity thresholds should still be calibrated by a grid of injections over SNR, drift, duration, bandwidth, and RFI backgrounds.
