import json
from pathlib import Path

import numpy as np
import yaml

from acs.bench.inject import (
    InjectionCase,
    decode_words,
    encode_complex_to_words,
    inject_cases_into_words,
    load_injection_plan,
)
from acs.cli.main import main
from acs.types import DatContract


def _random_words(rows: int, channels: int = 256, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = (rng.normal(0.0, 3.0, size=(rows, channels)) + 1j * rng.normal(0.0, 3.0, size=(rows, channels))).astype(np.complex64)
    return encode_complex_to_words(x)


def test_load_injection_plan_with_defaults_and_channel_index(tmp_path: Path):
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    plan_path = tmp_path / 'plan.yaml'
    payload = {
        'plan_name': 'multi_test',
        'defaults': {
            'morphology': 'linear',
            'snr': 22.0,
            'start_time_s': 0.1,
            'length_s': 0.5,
            'width_hz': 5.0,
        },
        'signals': [
            {
                'name': 'sig_a',
                'channel_index': 24,
                'offset_hz': 1200.0,
                'start_time_s': 0.12,
                'length_s': 0.42,
                'drift_rate_hz_per_s': 3.5,
                'snr': 24.0,
                'width_hz': 7.0,
                'tone_count': 5,
            },
            {
                'name': 'sig_b',
                'start_freq_hz': contract.lo_hz + (contract.start_coarse_channel + 40) * contract.coarse_df_hz + 800.0,
                'drift_hz_per_s': -6.0,
                'start_s': 0.18,
                'duration_s': 0.36,
                'snr_db': 19.0,
                'width_hz': 0.0,
            },
        ],
    }
    plan_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    plan = load_injection_plan(plan_path, contract)
    assert plan.plan_name == 'multi_test'
    assert len(plan.signals) == 2
    assert plan.signals[0].name == 'sig_a'
    assert plan.signals[0].snr_db == 24.0
    assert plan.signals[0].start_s == 0.12
    assert plan.signals[0].duration_s == 0.42
    assert plan.signals[0].drift_hz_per_s == 3.5
    assert plan.signals[0].bandwidth_hz == 7.0
    assert plan.signals[0].n_tones == 5
    expected_freq = contract.lo_hz + (contract.start_coarse_channel + 24) * contract.coarse_df_hz + 1200.0
    assert abs(plan.signals[0].abs_freq_hz - expected_freq) < 1e-6
    assert plan.signals[1].start_s == 0.18
    assert plan.signals[1].duration_s == 0.36
    assert plan.signals[1].drift_hz_per_s == -6.0


def test_width_hz_applies_to_linear_signal_without_explicit_tone_count():
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    words = _random_words(4096)
    band_start = contract.lo_hz + contract.start_coarse_channel * contract.coarse_df_hz
    case = InjectionCase(
        'wide_linear',
        'linear',
        band_start + 30 * contract.coarse_df_hz + 1100.0,
        18.0,
        drift_hz_per_s=4.0,
        start_s=0.1,
        duration_s=0.5,
        bandwidth_hz=18.0,
    )
    out_words, report = inject_cases_into_words(words.reshape(4096, 256), contract, [case])
    signal_report = report['signals'][0]
    assert signal_report['resolved']['width_hz'] == 18.0
    assert signal_report['resolved']['effective_tone_count'] >= 3
    before = decode_words(words.reshape(4096, 256))
    after = decode_words(out_words)
    touched = signal_report['target_channels'][0]['channel_index']
    assert float(np.mean(np.abs(after[:, touched]) ** 2)) > float(np.mean(np.abs(before[:, touched]) ** 2))


def test_inject_cases_into_words_modifies_multiple_channels():
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    words = _random_words(4096)
    band_start = contract.lo_hz + contract.start_coarse_channel * contract.coarse_df_hz
    cases = [
        InjectionCase('sig1', 'linear', band_start + 24 * contract.coarse_df_hz + 1200.0, 20.0),
        InjectionCase('sig2', 'linear', band_start + 44 * contract.coarse_df_hz + 1500.0, 20.0),
    ]
    out_words, report = inject_cases_into_words(words.reshape(4096, 256), contract, cases)
    before = decode_words(words.reshape(4096, 256))
    after = decode_words(out_words)
    assert report['n_signals'] == 2
    touched = []
    for signal in report['signals']:
        touched.extend(tc['channel_index'] for tc in signal['target_channels'])
    assert len(set(touched)) >= 2
    for ch in set(touched):
        before_power = float(np.mean(np.abs(before[:, ch]) ** 2))
        after_power = float(np.mean(np.abs(after[:, ch]) ** 2))
        assert after_power > before_power


def test_cli_inject_signals_and_dump_suite(tmp_path: Path):
    rows = 4096
    base_words = _random_words(rows)
    dat = tmp_path / 'base.dat'
    base_words.tofile(dat)
    cfg = Path(__file__).resolve().parents[1] / 'configs' / 'h1_search.yaml'

    suite_plan = tmp_path / 'suite.yaml'
    rc = main([
        'dump-injection-suite',
        '--config', str(cfg),
        '--suite', 'v018_phase1_5',
        '--out', str(suite_plan),
        '--plan-name', 'phase1_dump',
    ])
    assert rc == 0
    dumped = yaml.safe_load(suite_plan.read_text(encoding='utf-8'))
    assert dumped['plan_name'] == 'phase1_dump'
    assert len(dumped['signals']) == 5

    custom_plan = tmp_path / 'custom.yaml'
    payload = {
        'plan_name': 'custom_multi',
        'observation': {
            'obs_id': 'inj_obs',
            'source_name': 'inj_src',
            'beam_id': '00',
            'pol_id': '00',
            'scan_id': 'inj_scan',
            'target_id': 'inj_target',
        },
        'defaults': {
            'morphology': 'linear',
            'snr': 24.0,
            'start_time_s': 0.1,
            'length_s': 0.6,
        },
        'signals': [
            {'name': 'sig_a', 'channel_index': 24, 'offset_hz': 1200.0, 'width_hz': 6.0, 'drift_rate_hz_per_s': 4.0},
            {'name': 'sig_b', 'channel_index': 33, 'offset_hz': 900.0, 'start_time_s': 0.18, 'length_s': 0.4, 'snr': 21.0},
        ],
    }
    custom_plan.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')

    out_dat = tmp_path / 'out.dat'
    report = tmp_path / 'out.inject.json'
    manifest = tmp_path / 'manifest.yaml'
    rc = main([
        'inject-signals', str(dat),
        '--config', str(cfg),
        '--plan', str(custom_plan),
        '--out-dat', str(out_dat),
        '--report-out', str(report),
        '--manifest-out', str(manifest),
        '--obs-id', 'override_obs',
    ])
    assert rc == 0
    assert out_dat.exists()
    assert out_dat.stat().st_size == dat.stat().st_size
    assert report.exists()
    assert manifest.exists()
    report_payload = json.loads(report.read_text(encoding='utf-8'))
    assert report_payload['n_signals'] == 2
    assert report_payload['observation']['obs_id'] == 'override_obs'
    assert report_payload['signals'][0]['resolved']['width_hz'] == 6.0
    manifest_payload = yaml.safe_load(manifest.read_text(encoding='utf-8'))
    assert manifest_payload['observations'][0]['obs_id'] == 'override_obs'
    assert dat.read_bytes() != out_dat.read_bytes()


def test_requested_snr_db_is_preserved_for_multi_tone_active_samples():
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    words = _random_words(8192)
    before = decode_words(words.reshape(8192, 256))
    band_start = contract.lo_hz + contract.start_coarse_channel * contract.coarse_df_hz
    case = InjectionCase(
        'snr_preserved',
        'linear',
        band_start + 20 * contract.coarse_df_hz + 800.0,
        12.0,
        start_s=0.1,
        duration_s=0.25,
        bandwidth_hz=18.0,
    )
    out_words, report = inject_cases_into_words(words.reshape(8192, 256), contract, [case])
    after = decode_words(out_words)
    ch = report['signals'][0]['target_channels'][0]['channel_index']
    fs = contract.coarse_df_hz
    start = int(case.start_s * fs)
    end = start + int(case.duration_s * fs)
    delta = after[start:end, ch] - before[start:end, ch]
    sigma = float(np.sqrt(np.mean(np.abs(before[:, ch]) ** 2)))
    measured = float(np.sqrt(np.mean(np.abs(delta) ** 2)))
    measured_snr_db = 20.0 * np.log10(measured / max(sigma, 1e-12))
    assert abs(measured_snr_db - case.snr_db) < 0.75


def test_invalid_single_channel_width_raises_clean_error():
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    words = _random_words(4096)
    band_start = contract.lo_hz + contract.start_coarse_channel * contract.coarse_df_hz
    case = InjectionCase(
        'too_wide',
        'linear',
        band_start + 30 * contract.coarse_df_hz + 1100.0,
        18.0,
        start_s=0.1,
        duration_s=0.5,
        bandwidth_hz=320000.0,
    )
    try:
        inject_cases_into_words(words.reshape(4096, 256), contract, [case])
    except ValueError as exc:
        assert 'does not fit inside one coarse channel' in str(exc)
    else:
        raise AssertionError('Expected ValueError for invalid single-channel width')


def test_multisignal_10x40k_example_exposes_per_signal_controls():
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    plan = load_injection_plan(Path(__file__).resolve().parents[1] / 'examples' / 'multisignal_10x40k.example.yaml', contract)
    freqs = [sig.abs_freq_hz for sig in plan.signals]
    starts = [sig.start_s for sig in plan.signals]
    durations = [sig.duration_s for sig in plan.signals]
    drifts = [sig.drift_hz_per_s for sig in plan.signals]
    snrs = [sig.snr_db for sig in plan.signals]
    widths = [sig.bandwidth_hz for sig in plan.signals]
    assert len(freqs) == 10
    assert freqs == sorted(freqs)
    assert len(set(freqs)) == 10
    assert len(set(starts)) > 1
    assert len(set(durations)) > 1
    assert len(set(drifts)) > 1
    assert len(set(snrs)) > 1
    assert len(set(widths)) > 1


def test_headroom_rescaling_is_reported_cleanly():
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    rows = 8192
    rng = np.random.default_rng(11)
    x = (rng.normal(0.0, 32.0, size=(rows, 256)) + 1j * rng.normal(0.0, 32.0, size=(rows, 256))).astype(np.complex64)
    words = encode_complex_to_words(x)
    band_start = contract.lo_hz + contract.start_coarse_channel * contract.coarse_df_hz
    case = InjectionCase(
        'headroom_limited',
        'linear',
        band_start + 24 * contract.coarse_df_hz + 900.0,
        26.0,
        start_s=0.05,
        duration_s=0.25,
        bandwidth_hz=6.0,
    )
    _, report = inject_cases_into_words(words.reshape(rows, 256), contract, [case])
    sig_report = report['signals'][0]
    target = sig_report['target_channels'][0]
    assert sig_report['channel_rescaled_for_headroom'] is True
    assert target['channel_rescaled_for_headroom'] is True
    assert report['n_signals_rescaled_for_headroom'] == 1
    assert report['n_channels_rescaled_for_headroom'] == 1
    assert target['channel_scale'] < 1.0
    assert target['realized_snr_db'] is not None
    assert abs(target['realized_snr_db'] - target['requested_snr_db']) < 0.75


def test_legacy_yaml_without_snr_reference_defaults_to_coarse_channel(tmp_path: Path):
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    plan_path = tmp_path / 'legacy.yaml'
    plan_path.write_text(yaml.safe_dump({
        'plan_name': 'legacy_snr_mode',
        'defaults': {'morphology': 'linear', 'snr_db': 5.0, 'start_s': 0.1, 'duration_s': 0.2},
        'signals': [{'name': 'legacy_sig', 'channel_index': 20, 'offset_hz': 1000.0}],
    }), encoding='utf-8')
    plan = load_injection_plan(plan_path, contract)
    assert plan.signals[0].snr_reference == 'coarse_channel'
    assert plan.signals[0].noise_estimator == 'coarse_psd'


def test_local_psd_report_fields_and_realized_snr_close():
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    rows = 8192
    words = _random_words(rows, seed=123)
    band_start = contract.lo_hz + contract.start_coarse_channel * contract.coarse_df_hz
    case = InjectionCase(
        'local_psd_sig',
        'linear',
        band_start + 25 * contract.coarse_df_hz + 900.0,
        3.0,
        start_s=0.1,
        duration_s=0.25,
        bandwidth_hz=0.0,
        snr_reference='local_psd',
        noise_estimator='coarse_psd',
    )
    _, report = inject_cases_into_words(words.reshape(rows, 256), contract, [case], stft_nfft=2048)
    sig = report['signals'][0]
    assert sig['snr_reference'] == 'local_psd'
    assert sig['noise_estimator'] == 'coarse_psd'
    for key in [
        'requested_snr_db',
        'requested_local_psd_snr_db',
        'realized_local_psd_snr_db',
        'local_noise_psd',
        'reference_bandwidth_hz',
        'reference_noise_power',
        'requested_signal_power',
        'realized_signal_power',
        'applied_amplitude',
    ]:
        assert key in sig
    assert abs(sig['requested_local_psd_snr_db'] - 3.0) < 1e-9
    assert abs(sig['realized_local_psd_snr_db'] - 3.0) < 0.05


def test_local_psd_width_zero_uses_one_stft_resolution_element():
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    rows = 4096
    words = _random_words(rows, seed=124)
    band_start = contract.lo_hz + contract.start_coarse_channel * contract.coarse_df_hz
    case = InjectionCase(
        'zero_width_local_psd',
        'linear',
        band_start + 26 * contract.coarse_df_hz + 900.0,
        0.0,
        start_s=0.1,
        duration_s=0.15,
        bandwidth_hz=0.0,
        snr_reference='local_psd',
    )
    _, report = inject_cases_into_words(words.reshape(rows, 256), contract, [case], stft_nfft=4096)
    sig = report['signals'][0]
    expected = contract.coarse_df_hz / 4096
    assert abs(sig['reference_bandwidth_hz'] - expected) < 1e-12
    assert sig['reference_bandwidth_source'] == 'one_stft_resolution_element'


def test_local_psd_positive_width_uses_signal_width():
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    rows = 4096
    words = _random_words(rows, seed=125)
    band_start = contract.lo_hz + contract.start_coarse_channel * contract.coarse_df_hz
    case = InjectionCase(
        'finite_width_local_psd',
        'linear',
        band_start + 27 * contract.coarse_df_hz + 900.0,
        0.0,
        start_s=0.1,
        duration_s=0.15,
        bandwidth_hz=25.0,
        n_tones=5,
        snr_reference='local_psd',
    )
    _, report = inject_cases_into_words(words.reshape(rows, 256), contract, [case], stft_nfft=2048)
    sig = report['signals'][0]
    assert abs(sig['reference_bandwidth_hz'] - 25.0) < 1e-12
    assert sig['reference_bandwidth_source'] == 'width_hz'
    assert abs(sig['realized_local_psd_snr_db'] - 0.0) < 0.05
