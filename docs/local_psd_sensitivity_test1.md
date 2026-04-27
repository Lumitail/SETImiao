# Local-PSD 20-signal sensitivity validation

This version adds a focused 20-signal validation plan:

- file: `examples/test1_20sig_local_psd_snr1to5_v102x.yaml`
- reference mode: `snr_reference: local_psd`
- noise estimator: `noise_estimator: coarse_psd`
- SNR levels: 1, 2, 3, 4, 5 dB, with four injected signals at each SNR
- drift: +12 Hz/s for every signal
- duration: 60% of the measured `test1.dat` length
- frequency spacing: 7 coarse channels, about 106.812 kHz

The companion validation config is:

- `configs/test1_20sig_local_psd_sparse_continuous_v102x.yaml`

It uses `drift.channel_list` to search only the injected coarse channels. This is a targeted validation control, not a blind-survey setting.
For blind surveys, leave `channel_list: null`.

For continuous narrowband work, `configs/h1_search_continuous_local_psd.yaml` uses longer search tiles than the historical default. This improves track duration recovery and weak local-PSD sensitivity, but it is less optimized for very short transient signals.
