# Recovered SNR measurement

ACS keeps three SNR concepts separate:

1. **Injected SNR**: the amplitude requested by an injection plan. With
   `snr_reference: local_psd`, this is signal power divided by the local PSD
   noise power in the configured signal reference bandwidth.
2. **Search metrics**: `best_incoherent_snr`, `best_refined_snr`,
   `best_incoherent_search_metric_db`, and `best_refined_search_metric_db`.
   These are detection/ranking metrics and include STFT/coherent processing gain.
3. **Recovered SNR measurement**: a post-detection measurement of the average
   signal-track brightness relative to a local off-track background in the
   waterfall. These fields are written during `review-build`.

The main recovered SNR fields are:

- `recovered_band_excess_snr_db`: average same-band excess power along the
  recovered track, divided by local same-band background power and corrected by
  the window ENBW. This is the main field for comparison with local-PSD
  injection SNR.
- `recovered_ridge_pixel_snr_db`: literal average ridge-pixel excess relative
  to the local background pixel power. This is useful for visual/waterfall
  brightness diagnostics but is not ENBW-corrected.
- `recovered_local_noise_floor_power`, `recovered_signal_excess_power`,
  `recovered_background_pixel_count`, `recovered_ridge_pixel_count`,
  `recovered_ridge_width_bins`, and `recovered_reference_bandwidth_hz`: diagnostic
  quantities used to understand each measurement.

The measurement algorithm reads the candidate's representative coarse channel,
computes an STFT with the configured search STFT, predicts the event ridge from
frequency and drift, estimates the off-ridge local background with a robust
median, and measures excess power on the ridge. If injection truth is available
during review, the injected `width_hz` is used to choose the ridge width;
otherwise the event width is used.

The measurement is controlled by the `measurement:` config block. It runs by
default during `review-build`.
