# Track-aligned review diagnostics (v1.1.3X)

For weak continuous narrowband signals, the ordinary waterfall can look noise-like even when the detector has accumulated enough evidence across time. The review artifact now separates ordinary visual inspection from an integrated diagnostic view:

1. **Raw event overview**: the original local waterfall with recovered-track overlay.
2. **Track-aligned overview**: each STFT frame is shifted so the recovered frequency-drift track is fixed at 0 Hz offset.
3. **Local preview**: a short window around the peak hit.
4. **Mean track-aligned profile**: the average aligned spectrum across the event.

The new aligned view is a manual-veto diagnostic only. It does not change the search metrics, event scoring, candidate gates, or the recovered local-background SNR measurement.

New artifact/index fields:

- `aligned_profile_peak_excess_db`
- `aligned_profile_center_excess_db`
- `aligned_profile_background_median_db`
- `aligned_profile_peak_offset_hz`
- `aligned_half_width_bins`

When an injection report is available, review plots overlay the injected truth track as a red dotted line. This is used only for validation runs.
