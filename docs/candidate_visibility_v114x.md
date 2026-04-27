# v1.1.4x candidate visibility safeguards

The v1.1.3X failure mode was not primarily a plotting failure: several final candidates were short coherent-noise fragments with no positive track-aligned visual evidence. v1.1.4x adds:

- stronger short-track candidate gates;
- per-hit coherent-strength track inlier filtering;
- event-level robust track reporting from supported hits;
- review visual-evidence warning fields;
- time-frequency path-based injection matching.

For blind SETI runs, candidate review remains conservative, and `review-build --review-scope all-events` can still be used for manual inspection of sub-threshold events.
