# Adaptive Chirp Search (ACS) v1.0.0x

Adaptive Chirp Search (ACS) is a Python research pipeline for detecting and validating continuous narrowband signals in channelized raw radio astronomy `.dat` observations. It was built for SETI-style Doppler-drift searches on packed complex-voltage data, with an independent multi-signal injection workflow and automatic injection-recovery comparison reports.

ACS is designed for a specific raw data contract and is most useful when you need to:

- inspect and validate headerless channelized `.dat` files;
- search for narrowband drifting tones inside each coarse channel;
- refine candidate signals coherently on the raw complex samples;
- inject controlled synthetic signals into real observations;
- measure injection recovery completeness and raw-SNR sensitivity;
- generate review artifacts for human inspection and downstream analysis.

> **Status:** ACS v1.0.0x is a research release. It is suitable for controlled experiments and pipeline development. For publication-quality sensitivity limits, use the injection-comparison outputs to build calibrated recovery curves on data representative of the survey.

---

## Data contract

ACS assumes the following `.dat` layout:

| Field | Assumption |
|---|---|
| Header | none |
| Shape | row-major `(rows, 256)` |
| Cell size | 2 bytes |
| I/Q format | packed signed-int8 I and signed-int8 Q inside one 16-bit word |
| Metadata | supplied externally by CLI flags or manifest YAML |
| Search transform | FFTs are performed along time inside each coarse channel |

The default frequency contract in `configs/h1_search.yaml` is:

```yaml
contract:
  channels: 256
  bytes_per_cell: 2
  frontend_fs_hz: 1000000000.0
  channelizer_fft: 65536
  lo_hz: 1000000000.0
  start_coarse_channel: 27392
```

This gives a coarse-channel width of `1e9 / 65536 = 15258.7890625 Hz` and a native row cadence of `65.536 microseconds`.

---

## Search model

ACS uses three object levels:

| Object | Meaning |
|---|---|
| **Hit** | One tile-local detection from one seed, one drift hypothesis, and one local width. A continuous signal usually produces many hits. |
| **Event** | A merged group of compatible hits in the same observation/beam/channel. Events are structural; they are not necessarily final science candidates. |
| **Candidate** | An event that passes the configured candidate gates. In the continuous-narrowband default, candidates require multi-hit, multi-second support. |

The default continuous-narrowband candidate policy intentionally avoids promoting isolated one-tile coherent peaks into final candidates:

```yaml
candidate:
  allow_singleton_candidates: false
  min_candidate_hits: 3
  min_candidate_duration_s: 4.0
  merge_max_gap_s: 35.0
```

One-tile events remain in `events.jsonl` for diagnostics, but final candidates should represent a track with repeated support.

---

## SNR terminology

This is critical for SETI sensitivity and EIRP work.

The injection YAML field:

```yaml
snr_db: 5.0
```

means **raw active-sample RMS signal amplitude relative to the channel RMS used by the injector**.

The recovered review fields:

```text
best_incoherent_snr
best_refined_snr
```

are **search metrics**, not raw injected SNR. They include processing gain from STFT/bin integration and coherent dechirp/FFT refinement. For the default config:

```text
10*log10(search_stft.nfft=2048)      ≈ 33.11 dB
10*log10(search_tile_rows=31250)     ≈ 44.95 dB
```

Therefore, a raw injected 1 dB continuous tone can legitimately produce a refined search metric near 46 dB. Use `review/injection_comparison.csv` to relate recovered search metrics back to injected raw-SNR values.

For science thresholds, do **not** quote `best_refined_snr` directly as physical SNR. Instead:

1. run injection sweeps over raw `snr_db`, drift, duration, width, frequency position, and representative RFI/noise backgrounds;
2. use `injection_comparison.csv` to compute completeness versus raw injected SNR;
3. choose the survey SNR threshold at the desired recovery completeness and false-positive rate;
4. use that calibrated raw SNR threshold for EIRP calculations.

See `docs/score_handbook.md` and `docs/injection_comparison.md`.

