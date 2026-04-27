# Injection Handbook

ACS v0.2.0 supports two related injection modes:

- **independent multi-signal injection** via `acs inject-signals`
- **built-in benchmark suites** via `acs benchmark-inject`

The independent path is the one to use when you want to create a new injected `.dat` and then run the standard ACS workflow yourself.

## CLI commands

### Dump a built-in suite to an editable YAML plan

```bash
acs dump-injection-suite \
  --config configs/h1_search.yaml \
  --suite v018_phase1_5 \
  --out outputs/phase1_plan.yaml
```

### Inject one or many signals into a new `.dat`

```bash
acs inject-signals /path/to/base.dat \
  --config configs/h1_search.yaml \
  --plan outputs/phase1_plan.yaml \
  --out-dat outputs/injected.dat \
  --report-out outputs/injected.dat.inject.json \
  --manifest-out outputs/injected_manifest.yaml
```

### Run a benchmark suite with automatic recovery scoring

```bash
acs benchmark-inject /path/to/base.dat \
  --config configs/h1_search.yaml \
  --out-dir outputs/bench_phase1 \
  --suite v018_phase1_5
```

## YAML plan schema

A plan is a top-level mapping with these main sections:

- `plan_name`: human-readable name for the plan
- `observation`: optional metadata for a manifest sidecar
- `notes`: optional free-text notes
- `defaults`: optional shared signal parameters applied to every signal
- `signals`: list of one or more signal entries

### Example

```yaml
plan_name: custom_multisignal_example
observation:
  obs_id: custom_multisignal_example
  source_name: inject
  beam_id: '00'
  pol_id: '00'
  scan_id: custom_multisignal_example
  target_id: custom_multisignal_example
defaults:
  morphology: linear
  snr_db: 20.0
  start_s: 0.20
  duration_s: 1.40
  width_hz: 0.0
signals:
  - name: sig_a
    start_freq_hz: 1418103606.25
    start_s: 0.10
    duration_s: 1.55
    drift_hz_per_s: 0.0
    snr_db: 22.0
    width_hz: 0.0
  - name: sig_b
    channel_index: 40
    offset_hz: 900.0
    start_s: 0.25
    duration_s: 1.25
    drift_hz_per_s: 6.0
    snr_db: 19.0
    width_hz: 6.0
```

## How frequencies are specified

Each signal can choose one of three frequency styles:

### 1. Absolute RF / start frequency

```yaml
start_freq_hz: 1418336160.9375
```

### 2. Channel index inside the current `.dat`

```yaml
channel_index: 24
offset_hz: 1200.0
```

Here `channel_index` runs from `0` to `channels-1` inside the current file. `offset_hz` is measured from that coarse-channel center and can be positive or negative.

### 3. Absolute coarse-channel number plus offset

```yaml
coarse_channel: 27416
offset_hz: 1200.0
```

This uses the coarse-channel number in the larger telescope/channelizer system.

When you use `channel_index` or `coarse_channel`, ACS needs `contract.lo_hz` and `contract.start_coarse_channel` in the config.

## Useful signal fields

These are the main `InjectionCase` fields supported by YAML plans:

- `name`
- `morphology`
- `start_freq_hz` (alias: `abs_freq_hz`)
- `snr_db`
- `drift_hz_per_s`
- `start_s`
- `duration_s`
- `duty_cycle`
- `period_s`
- `width_hz` (alias: `bandwidth_hz`)
- `n_tones` / `tone_count`
- `piecewise_second_drift_hz_per_s`
- `quadratic_hz_per_s2`
- `wobble_hz`
- `wobble_rate_hz`
- `amp_mod_depth`
- `amp_mod_rate_hz`
- `secondary_start_s`
- `secondary_duration_s`
- `freq_tolerance_hz`
- `drift_tolerance_hz_per_s`
- `coarse_channel_count`
- `coarse_channel_step`
- `amplitude_taper`
- `checkerboard_period`


## Per-signal control pattern

Every entry under `signals:` can override the shared defaults. In practice, that means you can make one YAML file where each signal has its own:

- start time: `start_s` or `start_time_s`
- duration: `duration_s`, `length_s`, or `signal_duration_s`
- start frequency: `start_freq_hz` or `abs_freq_hz`
- drift rate: `drift_hz_per_s` or `drift_rate_hz_per_s`
- SNR: `snr_db` or `snr`
- narrowband width: `width_hz` or `bandwidth_hz`

If `width_hz > 0` and you do not explicitly set `n_tones`, ACS now auto-builds a small tone cluster so width control works for ordinary narrowband morphologies like `cw` and `linear`, not just `cluster`.

## Supported morphologies

Single-channel or cluster-like:

- `cw`
- `linear`
- `cluster`
- `burst`
- `intermittent`
- `piecewise`
- `quadratic`
- `wobble`
- `amp_mod`
- `dualburst`

Multi-channel RFI-like:

- `broadband_contig`
- `checkerboard`

## Important behavior in v0.2.0

### Multiple signals are injected into one output `.dat`

`acs inject-signals` reads the base file once, applies every requested signal to the same working array, then writes one new `.dat`.

### Signal amplitude is anchored to the original base data

Channel RMS is measured from the **original** base observation before any synthetic signal is added. That means if you add or remove another injected signal elsewhere, the amplitude calibration of existing signals stays stable.

### A clipping summary is always reported

The JSON report includes counts and fractions for any real/imag components that exceeded the int8 storage range before the final pack back to `.dat`.

## Output files from `inject-signals`

Typical outputs are:

- `injected.dat`
- `injected.dat.inject.json`
- optional `injected_manifest.yaml`

The JSON report includes:

- requested signal parameters
- resolved target coarse channels and baseband offsets
- per-channel applied amplitude
- clipping statistics

## Built-in suites

ACS still ships these benchmark suites:

- `v018_phase1_5`: 5 representative cases
- `v018_full30`: 30 packaged cases
- `v017_10`: retained for backward compatibility

Use `acs dump-injection-suite` when you want to start from one of these suites but then edit the signals in YAML instead of editing Python code.

## Example files

- `examples/injection_plan.example.yaml`
- `examples/multisignal_10x40k.example.yaml`


## Notes on SNR and width

- `snr_db` is applied to the active-sample RMS of the synthesized signal relative to the base channel RMS.
- For `width_hz > 0`, ACS may synthesize multiple tones, but it now renormalizes the composite so the requested `snr_db` is preserved.
- Single-channel narrowband morphologies must remain inside one coarse channel. If width plus drift would leave the coarse channel, ACS now raises a validation error and you should either reduce the width/drift or use `broadband_contig` / `checkerboard`.
