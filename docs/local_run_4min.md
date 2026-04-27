# Running ACS on a local ~4 minute `.dat` observation

## 1. Prepare Python
Create a clean environment and install ACS in editable mode.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## 2. Prepare metadata
Create a YAML manifest that points to all `.dat` files in time order.

Example:

```yaml
observations:
  - obs_id: m13_4min
    dat_paths:
      - /data/run/file_0001.dat
      - /data/run/file_0002.dat
      - /data/run/file_0003.dat
    source_name: m13
    beam_id: "00"
    pol_id: "00"
    scan_id: "m13_scan_001"
    target_id: "m13"
    lo_hz: 1000000000.0
    start_coarse_channel: 27392
```

## 3. First inspect one file
```bash
acs inspect-format /data/run/file_0001.dat --lo-hz 1000000000 --start-coarse-channel 27392
```

Confirm:
- rows are reasonable
- row width is 512 bytes
- the file parses cleanly

## 4. Generate a QC waterfall
```bash
acs qc-waterfall --config configs/h1_search.yaml --manifest /data/run/manifest.yaml --obs-id m13_4min --out outputs/m13_4min_qc.png
```

Inspect the QC waterfall before searching.

## 5. Run the search
```bash
acs search-run --config configs/h1_search.yaml --manifest /data/run/manifest.yaml --run-dir outputs/m13_4min_run
```

This will create:
- `hits.jsonl`
- `events.jsonl`
- `coincidences.jsonl`
- `summary.json`

## 6. Build review artifacts
```bash
acs review-build --config configs/h1_search.yaml --manifest /data/run/manifest.yaml --run-dir outputs/m13_4min_run --top-k 100
```

This creates:
- `review/index.html`
- per-event PNG/PDF/JSON

## 7. Manual inspection
Open:
```text
outputs/m13_4min_run/review/index.html
```

Start by reviewing:
- highest score
- strongest refined SNR
- events near the hydrogen-line region
- events with no obvious edge artifact

## 8. Optional: tune for long runs
For a 4 minute run you may want to adjust in `configs/h1_search.yaml`:
- `search_tile_rows`
- `search_overlap_rows`
- `seed_top_k_mean_per_channel`
- `seed_top_k_max_per_channel`
- `top_n`
- `min_refined_snr`

Recommended tuning order:
1. keep contract fixed
2. inspect QC output
3. increase/decrease seed counts
4. adjust refinement gate
5. rerun review build

## 9. Sanity check after run
Check:
```bash
cat outputs/m13_4min_run/summary.json
```

Then inspect:
- number of hits
- number of events
- number of coincidences

If the event count is too large:
- raise `min_refined_snr`
- reduce seed counts slightly
- inspect edge-masking behavior in QC

If the event count is too small:
- lower `min_refined_snr`
- increase seed counts slightly
- inspect whether the baseline flattening is over-suppressing structure
