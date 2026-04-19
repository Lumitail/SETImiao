from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable
import json
import shutil

import numpy as np
import yaml

from ..types import DatContract
from ..io.dat_reader import decode_words_to_complex64


@dataclass(frozen=True)
class InjectionCase:
    name: str
    morphology: str
    abs_freq_hz: float
    snr_db: float
    drift_hz_per_s: float = 0.0
    start_s: float = 0.2
    duration_s: float = 1.4
    duty_cycle: float = 1.0
    period_s: float = 0.2
    bandwidth_hz: float = 0.0
    n_tones: int = 1
    piecewise_second_drift_hz_per_s: float | None = None
    quadratic_hz_per_s2: float = 0.0
    wobble_hz: float = 0.0
    wobble_rate_hz: float = 0.0
    amp_mod_depth: float = 0.0
    amp_mod_rate_hz: float = 0.0
    secondary_start_s: float | None = None
    secondary_duration_s: float | None = None
    beams: tuple[str, ...] = ("00",)
    freq_tolerance_hz: float = 120.0
    drift_tolerance_hz_per_s: float = 6.0
    # multi-channel RFI-like controls
    coarse_channel_count: int = 1
    coarse_channel_step: int = 1
    amplitude_taper: str = "flat"   # flat | alternating | triangular
    checkerboard_period: int = 4

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["beams"] = list(self.beams)
        return d

    def to_plan_dict(self) -> dict[str, Any]:
        d = self.to_dict()
        d["start_freq_hz"] = d.pop("abs_freq_hz")
        d["width_hz"] = d.pop("bandwidth_hz")
        return d


@dataclass(frozen=True)
class InjectionPlan:
    plan_name: str
    signals: tuple[InjectionCase, ...]
    obs_id: str | None = None
    source_name: str | None = "inject"
    beam_id: str | None = "00"
    pol_id: str | None = "00"
    scan_id: str | None = None
    target_id: str | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_name": self.plan_name,
            "observation": {
                "obs_id": self.obs_id or self.plan_name,
                "source_name": self.source_name,
                "beam_id": self.beam_id,
                "pol_id": self.pol_id,
                "scan_id": self.scan_id or self.plan_name,
                "target_id": self.target_id or self.plan_name,
            },
            "notes": list(self.notes),
            "signals": [sig.to_plan_dict() for sig in self.signals],
        }


_CASE_FIELD_NAMES = {f.name for f in fields(InjectionCase)}
_ALLOWED_CASE_ALIASES = {
    "channel_index",
    "coarse_channel",
    "offset_hz",
    "signal_name",
    "kind",
    "signal_type",
    "start_freq_hz",
    "start_frequency_hz",
    "freq_hz",
    "width_hz",
    "signal_width_hz",
    "tone_count",
    "start_time_s",
    "length_s",
    "signal_duration_s",
    "drift_rate_hz_per_s",
    "snr",
}
_REQUIRED_CASE_KEYS = {"name", "morphology", "snr_db"}


def decode_words(words: np.ndarray) -> np.ndarray:
    i = (words & 0x00FF).astype(np.uint8).view(np.int8).astype(np.float32)
    q = ((words >> 8) & 0x00FF).astype(np.uint8).view(np.int8).astype(np.float32)
    return (i + 1j * q).astype(np.complex64)



def encode_complex_to_words(x: np.ndarray) -> np.ndarray:
    i = np.clip(np.rint(x.real), -128, 127).astype(np.int16)
    q = np.clip(np.rint(x.imag), -128, 127).astype(np.int16)
    ui = i.astype(np.int8).view(np.uint8).astype(np.uint16)
    uq = q.astype(np.int8).view(np.uint8).astype(np.uint16)
    return (ui | (uq << 8)).astype(np.uint16)



def _band_bases(contract: DatContract) -> list[float]:
    lo = contract.lo_hz if contract.lo_hz is not None else 1_000_000_000.0
    start_ch = contract.start_coarse_channel if contract.start_coarse_channel is not None else 27392
    coarse_df = contract.coarse_df_hz
    band_start = lo + start_ch * coarse_df
    # avoid edges heavily so injections survive edge masks
    return [band_start + k * coarse_df for k in [24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216]]



def _seti_cases(contract: DatContract) -> list[InjectionCase]:
    b = _band_bases(contract)
    # 10 realistic SETI-like narrowband signals spanning the requested drifts and widths
    return [
        InjectionCase('seti_01_cw_w5_snr20', 'cluster', b[0] + 1200.0, 20.0, drift_hz_per_s=0.0, bandwidth_hz=5.0, n_tones=3),
        InjectionCase('seti_02_cw_w20_snr12', 'cluster', b[1] + 2200.0, 12.0, drift_hz_per_s=0.0, bandwidth_hz=20.0, n_tones=5),
        InjectionCase('seti_03_drift10_w5_snr15', 'cluster', b[2] + 1400.0, 15.0, drift_hz_per_s=10.0, bandwidth_hz=5.0, n_tones=3),
        InjectionCase('seti_04_drift20_w5_snr10', 'cluster', b[3] + 1700.0, 10.0, drift_hz_per_s=20.0, bandwidth_hz=5.0, n_tones=3, drift_tolerance_hz_per_s=8.0),
        InjectionCase('seti_05_drift50_w5_snr18', 'cluster', b[4] + 900.0, 18.0, drift_hz_per_s=50.0, bandwidth_hz=5.0, n_tones=3, drift_tolerance_hz_per_s=12.0),
        InjectionCase('seti_06_cw_w20_snr6', 'cluster', b[5] + 3000.0, 6.0, drift_hz_per_s=0.0, bandwidth_hz=20.0, n_tones=5),
        InjectionCase('seti_07_drift10_w20_snr8', 'cluster', b[6] + 1800.0, 8.0, drift_hz_per_s=10.0, bandwidth_hz=20.0, n_tones=5),
        InjectionCase('seti_08_drift20_w20_snr14', 'cluster', b[7] + 2600.0, 14.0, drift_hz_per_s=20.0, bandwidth_hz=20.0, n_tones=5, drift_tolerance_hz_per_s=8.0),
        InjectionCase('seti_09_drift50_w20_snr9', 'cluster', b[8] + 1000.0, 9.0, drift_hz_per_s=50.0, bandwidth_hz=20.0, n_tones=5, drift_tolerance_hz_per_s=12.0),
        InjectionCase('seti_10_intermittent_drift10_w5_snr11', 'intermittent', b[9] + 1500.0, 11.0, drift_hz_per_s=10.0, duty_cycle=0.45, period_s=0.11, freq_tolerance_hz=150.0, drift_tolerance_hz_per_s=8.0),
    ]



def _rfi_cases(contract: DatContract) -> list[InjectionCase]:
    b = _band_bases(contract)
    # 20 RFI-like signals: broadband-ish, distorted narrowband, checkerboard, instrument-like
    return [
        InjectionCase('rfi_01_checkerboard_low', 'checkerboard', b[0] + 800.0, 6.0, coarse_channel_count=24, checkerboard_period=4),
        InjectionCase('rfi_02_checkerboard_high', 'checkerboard', b[1] + 800.0, 20.0, coarse_channel_count=32, checkerboard_period=3),
        InjectionCase('rfi_03_broadband_khz_low', 'broadband_contig', b[2], 8.0, coarse_channel_count=4, bandwidth_hz=4000.0, n_tones=9, freq_tolerance_hz=6000.0),
        InjectionCase('rfi_04_broadband_khz_high', 'broadband_contig', b[3], 20.0, coarse_channel_count=4, bandwidth_hz=8000.0, n_tones=13, freq_tolerance_hz=10000.0),
        InjectionCase('rfi_05_broadband_mhz_mid', 'broadband_contig', b[4], 14.0, coarse_channel_count=40, bandwidth_hz=2000.0, n_tones=5, amplitude_taper='triangular', freq_tolerance_hz=300000.0),
        InjectionCase('rfi_06_broadband_mhz_high', 'broadband_contig', b[5], 24.0, coarse_channel_count=64, bandwidth_hz=2500.0, n_tones=7, amplitude_taper='alternating', freq_tolerance_hz=500000.0),
        InjectionCase('rfi_07_cluster_distorted_low', 'cluster', b[6] + 1200.0, 7.0, bandwidth_hz=80.0, n_tones=11),
        InjectionCase('rfi_08_cluster_distorted_high', 'cluster', b[7] + 1800.0, 22.0, bandwidth_hz=150.0, n_tones=15),
        InjectionCase('rfi_09_piecewise_flip', 'piecewise', b[8] + 900.0, 16.0, drift_hz_per_s=18.0, piecewise_second_drift_hz_per_s=-22.0, drift_tolerance_hz_per_s=12.0),
        InjectionCase('rfi_10_quadratic_strong', 'quadratic', b[9] + 1200.0, 18.0, drift_hz_per_s=6.0, quadratic_hz_per_s2=14.0, drift_tolerance_hz_per_s=12.0),
        InjectionCase('rfi_11_wobble_low', 'wobble', b[10] + 1100.0, 7.0, drift_hz_per_s=2.0, wobble_hz=18.0, wobble_rate_hz=2.5, freq_tolerance_hz=180.0, drift_tolerance_hz_per_s=12.0),
        InjectionCase('rfi_12_wobble_high', 'wobble', b[11] + 1200.0, 20.0, drift_hz_per_s=3.0, wobble_hz=40.0, wobble_rate_hz=5.0, freq_tolerance_hz=250.0, drift_tolerance_hz_per_s=18.0),
        InjectionCase('rfi_13_ampmod_low', 'amp_mod', b[12] + 1300.0, 6.0, drift_hz_per_s=4.0, amp_mod_depth=0.8, amp_mod_rate_hz=5.0),
        InjectionCase('rfi_14_ampmod_high', 'amp_mod', b[13] + 1400.0, 20.0, drift_hz_per_s=0.0, amp_mod_depth=1.0, amp_mod_rate_hz=7.0),
        InjectionCase('rfi_15_dualburst', 'dualburst', b[14] + 1500.0, 18.0, start_s=0.25, duration_s=0.15, secondary_start_s=1.10, secondary_duration_s=0.16),
        InjectionCase('rfi_16_intermittent_sparse', 'intermittent', b[15] + 1600.0, 14.0, duty_cycle=0.2, period_s=0.08, drift_hz_per_s=12.0, drift_tolerance_hz_per_s=10.0),
        InjectionCase('rfi_17_linear_fast_pos', 'linear', b[16] + 1700.0, 20.0, drift_hz_per_s=48.0, drift_tolerance_hz_per_s=12.0),
        InjectionCase('rfi_18_linear_fast_neg', 'linear', b[1] + 1800.0, 20.0, drift_hz_per_s=-48.0, drift_tolerance_hz_per_s=12.0),
        InjectionCase('rfi_19_checkerboard_with_drift', 'checkerboard', b[2] + 700.0, 16.0, drift_hz_per_s=10.0, coarse_channel_count=20, checkerboard_period=5, freq_tolerance_hz=6000.0, drift_tolerance_hz_per_s=10.0),
        InjectionCase('rfi_20_contig_alternating', 'broadband_contig', b[3], 16.0, coarse_channel_count=12, bandwidth_hz=1500.0, n_tones=5, amplitude_taper='alternating', freq_tolerance_hz=15000.0),
    ]



def build_v018_phase1_5_cases(contract: DatContract) -> list[InjectionCase]:
    # exactly 5 representative signals for the required end-to-end verification: 2 SETI + 3 RFI
    seti = _seti_cases(contract)
    rfi = _rfi_cases(contract)
    return [
        seti[0],   # strong realistic SETI narrowband
        seti[3],   # drifted realistic SETI narrowband
        rfi[0],    # checkerboard-like RFI
        rfi[3],    # broadband-ish kHz RFI
        rfi[11],   # strong wobble/instrument-like RFI
    ]



def build_v018_full30_cases(contract: DatContract) -> list[InjectionCase]:
    return _seti_cases(contract) + _rfi_cases(contract)



def build_case_suite(contract: DatContract, suite: str) -> list[InjectionCase]:
    suite = suite.lower()
    if suite == 'v017_10':
        # backward compatibility with the older suite
        b = _band_bases(contract)
        return [
            InjectionCase('v017_01_strong_cw', 'cw', b[0] + 1200.0, 18.0),
            InjectionCase('v017_02_weak_cw', 'cw', b[1] + 2100.0, 9.5),
            InjectionCase('v017_03_pos_drift', 'linear', b[2] + 900.0, 16.0, drift_hz_per_s=12.0),
            InjectionCase('v017_04_neg_drift', 'linear', b[3] + 1400.0, 16.0, drift_hz_per_s=-14.0),
            InjectionCase('v017_05_short_burst', 'burst', b[4] + 1800.0, 18.0, start_s=0.45, duration_s=0.18),
            InjectionCase('v017_06_intermittent', 'intermittent', b[5] + 2600.0, 18.0, drift_hz_per_s=7.0, duty_cycle=0.35, period_s=0.09, drift_tolerance_hz_per_s=8.0),
            InjectionCase('v017_07_ampmod', 'amp_mod', b[6] + 3200.0, 16.0, drift_hz_per_s=4.0, amp_mod_depth=0.65, amp_mod_rate_hz=3.5),
            InjectionCase('v017_08_cluster_rfi_like', 'cluster', b[7] + 4000.0, 14.0, bandwidth_hz=32.0, n_tones=7),
            InjectionCase('v017_09_piecewise', 'piecewise', b[8] + 5200.0, 17.0, drift_hz_per_s=10.0, piecewise_second_drift_hz_per_s=-8.0, drift_tolerance_hz_per_s=12.0),
            InjectionCase('v017_10_wobble_noise_like', 'wobble', b[9] + 6200.0, 15.0, drift_hz_per_s=3.0, wobble_hz=10.0, wobble_rate_hz=2.0, freq_tolerance_hz=180.0, drift_tolerance_hz_per_s=16.0),
        ]
    if suite == 'v018_phase1_5':
        return build_v018_phase1_5_cases(contract)
    if suite == 'v018_full30':
        return build_v018_full30_cases(contract)
    return build_v018_full30_cases(contract)



def build_suite_plan(contract: DatContract, suite: str, *, plan_name: str | None = None, case_names: Iterable[str] | None = None) -> InjectionPlan:
    cases = build_case_suite(contract, suite)
    if case_names is not None:
        wanted = {str(name).strip() for name in case_names if str(name).strip()}
        cases = [case for case in cases if case.name in wanted]
    if not cases:
        raise ValueError(f"No injection cases selected for suite {suite!r}")
    resolved_plan_name = plan_name or f"suite_{suite}"
    return InjectionPlan(
        plan_name=resolved_plan_name,
        signals=tuple(cases),
        obs_id=resolved_plan_name,
        source_name="inject",
        beam_id="00",
        pol_id="00",
        scan_id=resolved_plan_name,
        target_id=resolved_plan_name,
        notes=(f"Generated from built-in suite {suite}",),
    )



def _require_contract_frequency_metadata(contract: DatContract) -> tuple[float, int]:
    if contract.lo_hz is None or contract.start_coarse_channel is None:
        raise ValueError(
            "Injection plan frequency helpers require contract.lo_hz and contract.start_coarse_channel. "
            "Either set them in the config or specify abs_freq_hz explicitly for every signal."
        )
    return float(contract.lo_hz), int(contract.start_coarse_channel)



def _band_limits(contract: DatContract) -> tuple[float, float] | None:
    if contract.lo_hz is None or contract.start_coarse_channel is None:
        return None
    lo, start_ch = _require_contract_frequency_metadata(contract)
    coarse_df = contract.coarse_df_hz
    band_min = lo + start_ch * coarse_df - 0.5 * coarse_df
    band_max = lo + (start_ch + contract.channels - 1) * coarse_df + 0.5 * coarse_df
    return band_min, band_max



def _resolve_abs_freq_hz(spec: dict[str, Any], contract: DatContract) -> float:
    if spec.get("abs_freq_hz") is not None:
        return float(spec["abs_freq_hz"])
    lo, start_ch = _require_contract_frequency_metadata(contract)
    coarse_df = contract.coarse_df_hz
    offset_hz = float(spec.get("offset_hz", 0.0))
    if spec.get("channel_index") is not None:
        channel_index = int(spec["channel_index"])
        if not 0 <= channel_index < contract.channels:
            raise ValueError(f"channel_index must be in [0, {contract.channels - 1}], got {channel_index}")
        return float(lo + (start_ch + channel_index) * coarse_df + offset_hz)
    if spec.get("coarse_channel") is not None:
        coarse_channel = int(spec["coarse_channel"])
        return float(lo + coarse_channel * coarse_df + offset_hz)
    raise ValueError(
        "Each signal must specify abs_freq_hz, or use channel_index+offset_hz, or use coarse_channel+offset_hz."
    )



def _validate_abs_freq_hz(abs_freq_hz: float, contract: DatContract, name: str) -> None:
    limits = _band_limits(contract)
    if limits is None:
        return
    band_min, band_max = limits
    if not (band_min <= abs_freq_hz <= band_max):
        raise ValueError(
            f"Signal {name!r} resolved to abs_freq_hz={abs_freq_hz:.6f}, outside the current .dat band [{band_min:.6f}, {band_max:.6f}]"
        )



def injection_case_from_mapping(payload: dict[str, Any], contract: DatContract) -> InjectionCase:
    if not isinstance(payload, dict):
        raise TypeError(f"Injection signal entries must be mappings, got {type(payload)!r}")
    raw = _normalize_case_aliases(dict(payload))
    if raw.get("morphology") is not None:
        raw["morphology"] = str(raw["morphology"]).lower()
    raw["abs_freq_hz"] = _resolve_abs_freq_hz(raw, contract)
    _validate_abs_freq_hz(float(raw["abs_freq_hz"]), contract, str(raw.get("name", "<unnamed>")))
    if raw.get("beams") is not None:
        raw["beams"] = tuple(str(b) for b in raw["beams"])
    unknown = set(raw) - _CASE_FIELD_NAMES - _ALLOWED_CASE_ALIASES
    if unknown:
        raise ValueError(f"Unknown injection signal keys: {sorted(unknown)}")
    missing = _REQUIRED_CASE_KEYS - raw.keys()
    if missing:
        raise ValueError(f"Missing required injection signal keys: {sorted(missing)}")
    kwargs = {k: raw[k] for k in _CASE_FIELD_NAMES if k in raw}
    return InjectionCase(**kwargs)


def _normalize_case_aliases(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    _alias_copy(out, "name", "signal_name")
    _alias_copy(out, "morphology", "kind")
    _alias_copy(out, "morphology", "signal_type")
    _alias_copy(out, "abs_freq_hz", "start_freq_hz")
    _alias_copy(out, "abs_freq_hz", "start_frequency_hz")
    _alias_copy(out, "abs_freq_hz", "freq_hz")
    _alias_copy(out, "bandwidth_hz", "width_hz")
    _alias_copy(out, "bandwidth_hz", "signal_width_hz")
    _alias_copy(out, "n_tones", "tone_count")
    _alias_copy(out, "start_s", "start_time_s")
    _alias_copy(out, "duration_s", "length_s")
    _alias_copy(out, "duration_s", "signal_duration_s")
    _alias_copy(out, "drift_hz_per_s", "drift_rate_hz_per_s")
    _alias_copy(out, "snr_db", "snr")
    return out


def _alias_copy(mapping: dict[str, Any], canonical: str, alias: str) -> None:
    if alias not in mapping:
        return
    alias_value = mapping.pop(alias)
    if canonical not in mapping:
        mapping[canonical] = alias_value
        return
    canonical_value = mapping[canonical]
    if canonical_value != alias_value:
        raise ValueError(
            f"Conflicting injection keys {canonical!r} and {alias!r}: {canonical_value!r} != {alias_value!r}"
        )



def load_injection_plan(path: str | Path, contract: DatContract) -> InjectionPlan:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"Injection plan {path} is empty")
    if not isinstance(payload, dict):
        raise TypeError(f"Injection plan must be a mapping at top level, got {type(payload)!r}")

    defaults = payload.get("defaults") or {}
    if defaults and not isinstance(defaults, dict):
        raise TypeError("Injection plan 'defaults' must be a mapping when present")
    normalized_defaults = _normalize_case_aliases(dict(defaults)) if defaults else {}

    suite_cases: list[InjectionCase] = []
    suite_name = payload.get("suite")
    if suite_name:
        suite_cases = build_case_suite(contract, str(suite_name))
        case_names = payload.get("case_names")
        if case_names is not None:
            wanted = {str(name).strip() for name in case_names if str(name).strip()}
            suite_cases = [case for case in suite_cases if case.name in wanted]

    signals_raw = payload.get("signals") or []
    if not isinstance(signals_raw, list):
        raise TypeError("Injection plan 'signals' must be a list when present")

    signals: list[InjectionCase] = list(suite_cases)
    for item in signals_raw:
        merged = dict(normalized_defaults)
        if isinstance(item, dict):
            merged.update(_normalize_case_aliases(dict(item)))
        else:
            raise TypeError(f"Injection plan signal entries must be mappings, got {type(item)!r}")
        signals.append(injection_case_from_mapping(merged, contract))

    if not signals:
        raise ValueError(f"Injection plan {path} did not produce any signals")

    observation = payload.get("observation") or {}
    if observation and not isinstance(observation, dict):
        raise TypeError("Injection plan 'observation' must be a mapping when present")
    notes = payload.get("notes") or []
    if notes and not isinstance(notes, list):
        raise TypeError("Injection plan 'notes' must be a list when present")

    plan_name = str(payload.get("plan_name") or path.stem)
    return InjectionPlan(
        plan_name=plan_name,
        signals=tuple(signals),
        obs_id=observation.get("obs_id") or plan_name,
        source_name=observation.get("source_name", "inject"),
        beam_id=observation.get("beam_id", "00"),
        pol_id=observation.get("pol_id", "00"),
        scan_id=observation.get("scan_id") or plan_name,
        target_id=observation.get("target_id") or plan_name,
        notes=tuple(str(note) for note in notes),
    )



def save_injection_plan(path: str | Path, plan: InjectionPlan) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = plan.to_dict()
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path



def _channel_and_baseband(contract: DatContract, abs_freq_hz: float) -> tuple[int, float]:
    lo = contract.lo_hz if contract.lo_hz is not None else 1_000_000_000.0
    start_ch = contract.start_coarse_channel if contract.start_coarse_channel is not None else 27392
    ccent = lo + (start_ch + np.arange(contract.channels)) * contract.coarse_df_hz
    ch = int(np.argmin(np.abs(ccent - abs_freq_hz)))
    return ch, float(abs_freq_hz - ccent[ch])



def _amplitude_for_channel(case: InjectionCase, channel_index_in_pattern: int) -> float:
    if case.amplitude_taper == 'alternating':
        return 1.0 if channel_index_in_pattern % 2 == 0 else 0.55
    if case.amplitude_taper == 'triangular':
        center = max(case.coarse_channel_count - 1, 1) / 2.0
        return max(0.25, 1.0 - abs(channel_index_in_pattern - center) / max(center, 1.0))
    return 1.0



def _target_channels(contract: DatContract, case: InjectionCase) -> list[tuple[int, float, float]]:
    base_ch, baseband = _channel_and_baseband(contract, case.abs_freq_hz)
    if case.morphology == 'checkerboard':
        start = max(0, base_ch - case.coarse_channel_count // 2)
        chans = list(range(start, min(contract.channels, start + case.coarse_channel_count)))
        selected = [ch for ch in chans if (ch - start) % max(case.checkerboard_period, 1) == 0]
        out = []
        for i, ch in enumerate(selected):
            coarse_center_shift = (base_ch - ch) * contract.coarse_df_hz
            out.append((ch, baseband + coarse_center_shift, _amplitude_for_channel(case, i)))
        return out
    if case.morphology == 'broadband_contig':
        start = max(0, base_ch - case.coarse_channel_count // 2)
        chans = list(range(start, min(contract.channels, start + case.coarse_channel_count)))
        out = []
        for i, ch in enumerate(chans):
            coarse_center_shift = (base_ch - ch) * contract.coarse_df_hz
            out.append((ch, baseband + coarse_center_shift, _amplitude_for_channel(case, i)))
        return out
    return [(base_ch, baseband, 1.0)]



def _active_envelope(rows: int, fs: float, case: InjectionCase) -> np.ndarray:
    t = np.arange(rows, dtype=np.float64) / fs
    active = np.zeros(rows, dtype=np.float32)
    start_idx = max(0, int(case.start_s * fs))
    end_idx = min(rows, start_idx + int(case.duration_s * fs))
    active[start_idx:end_idx] = 1.0
    if case.morphology == 'intermittent':
        tt = t[start_idx:end_idx] - t[start_idx]
        phase = np.mod(tt, case.period_s)
        active[start_idx:end_idx] = (phase <= case.period_s * case.duty_cycle).astype(np.float32)
    if case.morphology == 'dualburst':
        s2 = max(0, int((case.secondary_start_s or 0.8) * fs))
        e2 = min(rows, s2 + int((case.secondary_duration_s or 0.2) * fs))
        active[s2:e2] = 1.0
    return active



def _complex_signal(rows: int, fs: float, case: InjectionCase, baseband: float) -> np.ndarray:
    inst_freq = _instantaneous_frequency(rows, fs, case, baseband)
    tone_count = _effective_tone_count(case)
    if tone_count <= 1:
        phi = 2 * np.pi * np.cumsum(inst_freq) / fs
        sig = np.exp(1j * phi).astype(np.complex64)
        return sig

    acc = np.zeros(rows, dtype=np.complex64)
    offsets = np.linspace(-case.bandwidth_hz / 2.0, case.bandwidth_hz / 2.0, tone_count)
    for off in offsets:
        phi = 2 * np.pi * np.cumsum(inst_freq + off) / fs
        acc += np.exp(1j * phi).astype(np.complex64)
    return acc.astype(np.complex64)


def _instantaneous_frequency(rows: int, fs: float, case: InjectionCase, baseband: float) -> np.ndarray:
    t = np.arange(rows, dtype=np.float64) / fs
    local_t = t - case.start_s
    inst_freq = np.full(rows, baseband, dtype=np.float64)
    if case.morphology == 'piecewise':
        tmid = case.start_s + case.duration_s / 2.0
        m1 = t < tmid
        m2 = ~m1
        inst_freq[m1] = baseband + case.drift_hz_per_s * local_t[m1]
        dt2 = t[m2] - tmid
        fmid = baseband + case.drift_hz_per_s * (tmid - case.start_s)
        d2 = float(case.piecewise_second_drift_hz_per_s or case.drift_hz_per_s)
        inst_freq[m2] = fmid + d2 * dt2
        return inst_freq
    if case.morphology in {'linear','burst','intermittent','amp_mod','cw','cluster','broadband_contig','checkerboard','dualburst'}:
        inst_freq = baseband + case.drift_hz_per_s * local_t
    if case.morphology == 'wobble':
        inst_freq = baseband + case.drift_hz_per_s * local_t + case.wobble_hz*np.sin(2*np.pi*case.wobble_rate_hz*local_t)
    if case.morphology == 'quadratic':
        inst_freq = baseband + case.drift_hz_per_s * local_t + 0.5*case.quadratic_hz_per_s2*(local_t**2)
    return inst_freq


def _effective_tone_count(case: InjectionCase) -> int:
    if case.bandwidth_hz <= 0.0:
        return 1
    if case.n_tones > 1:
        return int(case.n_tones)
    approx = int(np.ceil(case.bandwidth_hz / 6.0)) + 1
    approx = max(3, min(17, approx))
    if approx % 2 == 0:
        approx += 1
    return approx



def _amp_envelope(rows: int, fs: float, case: InjectionCase) -> np.ndarray:
    if case.morphology != 'amp_mod':
        return np.ones(rows, dtype=np.float32)
    local_t = np.arange(rows, dtype=np.float64) / fs - case.start_s
    amp_env = (1.0 + case.amp_mod_depth * np.sin(2*np.pi*case.amp_mod_rate_hz*local_t)).astype(np.float32)
    return np.clip(amp_env, 0.0, None)


def _active_window_times_s(case: InjectionCase) -> np.ndarray:
    edges = [float(case.start_s), float(case.start_s + case.duration_s)]
    if case.secondary_start_s is not None:
        edges.append(float(case.secondary_start_s))
    if case.secondary_start_s is not None and case.secondary_duration_s is not None:
        edges.append(float(case.secondary_start_s + case.secondary_duration_s))
    t0 = max(0.0, min(edges))
    t1 = max(edges)
    if t1 <= t0:
        t1 = t0 + 1e-6
    return np.linspace(t0, t1, 128, dtype=np.float64)


def _instantaneous_frequency_at_times(times_s: np.ndarray, case: InjectionCase, baseband: float) -> np.ndarray:
    local_t = times_s - float(case.start_s)
    inst_freq = np.full_like(times_s, baseband, dtype=np.float64)
    if case.morphology == 'piecewise':
        tmid = case.start_s + case.duration_s / 2.0
        m1 = times_s < tmid
        m2 = ~m1
        inst_freq[m1] = baseband + case.drift_hz_per_s * local_t[m1]
        dt2 = times_s[m2] - tmid
        fmid = baseband + case.drift_hz_per_s * (tmid - case.start_s)
        d2 = float(case.piecewise_second_drift_hz_per_s or case.drift_hz_per_s)
        inst_freq[m2] = fmid + d2 * dt2
        return inst_freq
    if case.morphology in {'linear','burst','intermittent','amp_mod','cw','cluster','broadband_contig','checkerboard','dualburst'}:
        inst_freq = baseband + case.drift_hz_per_s * local_t
    if case.morphology == 'wobble':
        inst_freq = baseband + case.drift_hz_per_s * local_t + case.wobble_hz * np.sin(2 * np.pi * case.wobble_rate_hz * local_t)
    if case.morphology == 'quadratic':
        inst_freq = baseband + case.drift_hz_per_s * local_t + 0.5 * case.quadratic_hz_per_s2 * (local_t ** 2)
    return inst_freq


def _validate_single_channel_envelope(contract: DatContract, case: InjectionCase, baseband: float) -> None:
    if case.morphology in {'broadband_contig', 'checkerboard'}:
        return
    if case.coarse_channel_count > 1:
        return
    nyquist = 0.5 * contract.coarse_df_hz
    times = _active_window_times_s(case)
    inst = _instantaneous_frequency_at_times(times, case, baseband)
    half_width = 0.5 * float(case.bandwidth_hz)
    excursion = float(np.max(np.abs(inst))) + half_width
    margin = max(8.0, 2.0 * (contract.coarse_df_hz / 2048.0))
    limit = nyquist - margin
    if excursion > limit:
        raise ValueError(
            f"Signal {case.name!r} does not fit inside one coarse channel: max |baseband|+width/2={excursion:.3f} Hz exceeds {limit:.3f} Hz. "
            "Use a smaller width/drift, move the start frequency closer to coarse-channel center, or use broadband_contig/checkerboard for multi-channel injections."
        )


def _signal_rms_on_active_samples(sig: np.ndarray, active: np.ndarray, amp_env: np.ndarray) -> float:
    mod = active.astype(np.float32) * amp_env.astype(np.float32)
    active_mask = mod > 0.0
    if not np.any(active_mask):
        return 0.0
    shaped = mod[active_mask].astype(np.float32) * sig[active_mask]
    return float(np.sqrt(np.mean(np.abs(shaped) ** 2)))


_CLIP_COMPONENT_LIMIT = 127.0
_CLIP_COMPONENT_MARGIN = 1.0
_ALLOWED_ACTIVE_CLIP_FRACTION = 1e-6


def _amplitude_caps_for_component(base: np.ndarray, delta: np.ndarray, *, limit: float, margin: float) -> np.ndarray:
    upper = float(limit - margin)
    lower = float(-limit + margin)
    caps = []
    pos = delta > 1e-12
    if np.any(pos):
        caps.append((upper - base[pos]) / delta[pos])
    neg = delta < -1e-12
    if np.any(neg):
        caps.append((lower - base[neg]) / delta[neg])
    if not caps:
        return np.empty(0, dtype=np.float32)
    out = np.concatenate([c.astype(np.float32, copy=False) for c in caps])
    out = out[np.isfinite(out)]
    return out[out >= 0.0]


def _max_safe_amplitude(work: np.ndarray, shaped: np.ndarray, *, limit: float = _CLIP_COMPONENT_LIMIT, margin: float = _CLIP_COMPONENT_MARGIN, allowed_fraction: float = _ALLOWED_ACTIVE_CLIP_FRACTION) -> float:
    active_mask = (np.abs(shaped.real) + np.abs(shaped.imag)) > 1e-12
    if not np.any(active_mask):
        return float('inf')
    wr = work.real[active_mask].astype(np.float32, copy=False)
    wi = work.imag[active_mask].astype(np.float32, copy=False)
    sr = shaped.real[active_mask].astype(np.float32, copy=False)
    si = shaped.imag[active_mask].astype(np.float32, copy=False)
    caps = np.concatenate([
        _amplitude_caps_for_component(wr, sr, limit=limit, margin=margin),
        _amplitude_caps_for_component(wi, si, limit=limit, margin=margin),
    ])
    if caps.size == 0:
        return float('inf')
    q = float(np.clip(allowed_fraction, 0.0, 0.25))
    if q <= 0.0:
        return float(np.min(caps))
    return float(np.quantile(caps, q))


def _active_component_clip_fraction(work: np.ndarray, shaped: np.ndarray, amp: float) -> float:
    active_mask = (np.abs(shaped.real) + np.abs(shaped.imag)) > 1e-12
    if not np.any(active_mask):
        return 0.0
    z = work[active_mask] + float(amp) * shaped[active_mask]
    clip_real = np.count_nonzero((np.rint(z.real) < -128.0) | (np.rint(z.real) > 127.0))
    clip_imag = np.count_nonzero((np.rint(z.imag) < -128.0) | (np.rint(z.imag) > 127.0))
    total = max(2 * int(np.count_nonzero(active_mask)), 1)
    return float((clip_real + clip_imag) / total)


def _amplitude_plan(work: np.ndarray, shaped: np.ndarray, sigma: float, case: InjectionCase, amp_scale: float, raw_sig_rms: float) -> dict[str, float | bool | None]:
    requested_target_rms = float(sigma * (10.0 ** (case.snr_db / 20.0)) * amp_scale)
    if raw_sig_rms <= 0.0:
        return {
            'requested_amplitude': 0.0,
            'applied_amplitude': 0.0,
            'safe_amplitude_cap': 0.0,
            'requested_active_signal_rms': requested_target_rms,
            'realized_active_signal_rms': 0.0,
            'requested_snr_db': float(case.snr_db),
            'realized_snr_db': None,
            'amplitude_limited_by_clip': False,
            'active_component_clip_fraction': 0.0,
        }
    requested_amp = requested_target_rms / raw_sig_rms
    safe_cap = _max_safe_amplitude(work, shaped)
    applied_amp = min(float(requested_amp), float(safe_cap))
    applied_amp = max(0.0, float(applied_amp))
    realized_rms = float(raw_sig_rms * applied_amp)
    realized_snr_db = None
    if sigma > 0.0 and realized_rms > 0.0:
        realized_snr_db = float(20.0 * np.log10(realized_rms / sigma))
    return {
        'requested_amplitude': float(requested_amp),
        'applied_amplitude': float(applied_amp),
        'safe_amplitude_cap': float(safe_cap),
        'requested_active_signal_rms': float(requested_target_rms),
        'realized_active_signal_rms': float(realized_rms),
        'requested_snr_db': float(case.snr_db),
        'realized_snr_db': realized_snr_db,
        'amplitude_limited_by_clip': bool(applied_amp + 1e-12 < requested_amp),
        'active_component_clip_fraction': _active_component_clip_fraction(work, shaped, applied_amp),
    }



def _fraction_components_clipped(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    rounded_real = np.rint(np.asarray(x.real, dtype=np.float32))
    rounded_imag = np.rint(np.asarray(x.imag, dtype=np.float32))
    clip_real = np.count_nonzero((rounded_real < -128.0) | (rounded_real > 127.0))
    clip_imag = np.count_nonzero((rounded_imag < -128.0) | (rounded_imag > 127.0))
    total = max(2 * x.size, 1)
    return float((clip_real + clip_imag) / total)


def _best_uniform_channel_scale(x: np.ndarray, *, allowed_fraction: float = _ALLOWED_ACTIVE_CLIP_FRACTION) -> float:
    frac = _fraction_components_clipped(x)
    if frac <= allowed_fraction:
        return 1.0
    lo = 0.0
    hi = 1.0
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        mid_frac = _fraction_components_clipped(mid * x)
        if mid_frac <= allowed_fraction:
            lo = mid
        else:
            hi = mid
    return float(lo)


def _initial_signal_report(case: InjectionCase, rows: int, fs: float) -> dict[str, Any]:
    active = _active_envelope(rows, fs, case)
    effective_tone_count = _effective_tone_count(case)
    start_row = max(0, int(case.start_s * fs))
    end_row = min(rows, start_row + int(case.duration_s * fs))
    return {
        "name": case.name,
        "morphology": case.morphology,
        "requested": case.to_plan_dict(),
        "resolved": {
            "start_s": float(case.start_s),
            "duration_s": float(case.duration_s),
            "start_freq_hz": float(case.abs_freq_hz),
            "drift_hz_per_s": float(case.drift_hz_per_s),
            "snr_db": float(case.snr_db),
            "width_hz": float(case.bandwidth_hz),
            "effective_tone_count": int(effective_tone_count),
            "start_row": int(start_row),
            "end_row": int(end_row),
        },
        "active_samples": int(np.count_nonzero(active)),
        "active_fraction": float(np.mean(active > 0.0)),
        "amplitude_limited_by_clip": False,
        "channel_rescaled_for_headroom": False,
        "target_channels": [],
    }


def _build_signal_reports_and_channel_plan(case_list: list[InjectionCase], rows: int, contract: DatContract) -> tuple[list[dict[str, Any]], dict[int, list[tuple[int, InjectionCase, float, float]]]]:
    fs = contract.coarse_df_hz
    signal_reports: list[dict[str, Any]] = []
    channel_plan: dict[int, list[tuple[int, InjectionCase, float, float]]] = {}
    for signal_idx, case in enumerate(case_list):
        signal_reports.append(_initial_signal_report(case, rows, fs))
        for ch, baseband, amp_scale in _target_channels(contract, case):
            channel_plan.setdefault(int(ch), []).append((signal_idx, case, float(baseband), float(amp_scale)))
    return signal_reports, channel_plan


def _apply_channel_entries(channel_index: int, base_channel: np.ndarray, contract: DatContract, entries: list[tuple[int, InjectionCase, float, float]]) -> tuple[np.ndarray, list[tuple[int, dict[str, Any]]], dict[str, Any]]:
    rows = int(base_channel.shape[0])
    fs = contract.coarse_df_hz
    sigma = float(np.sqrt(np.mean(np.abs(base_channel) ** 2)))
    requested_total = np.zeros(rows, dtype=np.complex64)
    components: list[dict[str, Any]] = []
    for signal_idx, case, baseband, amp_scale in entries:
        _validate_single_channel_envelope(contract, case, baseband)
        active = _active_envelope(rows, fs, case)
        amp_env = _amp_envelope(rows, fs, case)
        sig = _complex_signal(rows, fs, case, baseband)
        shaped = (active * amp_env).astype(np.float32) * sig
        raw_sig_rms = _signal_rms_on_active_samples(sig, active, amp_env)
        requested_target_rms = float(sigma * (10.0 ** (case.snr_db / 20.0)) * amp_scale)
        requested_amp = 0.0 if raw_sig_rms <= 0.0 else float(requested_target_rms / raw_sig_rms)
        requested_delta = requested_amp * shaped
        requested_total += requested_delta
        components.append({
            'signal_idx': int(signal_idx),
            'case': case,
            'baseband_hz': float(baseband),
            'amp_scale': float(amp_scale),
            'effective_tone_count': int(_effective_tone_count(case)),
            'width_hz': float(case.bandwidth_hz),
            'raw_signal_rms_before_scaling': float(raw_sig_rms),
            'requested_active_signal_rms_before_channel_scale': float(requested_target_rms),
            'requested_amplitude_before_channel_scale': float(requested_amp),
            'active_mask': (np.abs(shaped.real) + np.abs(shaped.imag)) > 1e-12,
        })
    combined_unscaled = np.asarray(base_channel, dtype=np.complex64) + requested_total
    channel_scale = _best_uniform_channel_scale(combined_unscaled)
    final_channel = (channel_scale * combined_unscaled).astype(np.complex64, copy=False)
    output_channel_rms = float(channel_scale * sigma)
    channel_meta = {
        'channel_scale': float(channel_scale),
        'channel_rms_before_scale': float(sigma),
        'channel_rms_after_scale': float(output_channel_rms),
        'channel_rescaled_for_headroom': bool(channel_scale + 1e-12 < 1.0),
    }
    packed_records: list[tuple[int, dict[str, Any]]] = []
    for comp in components:
        requested_amp = float(comp['requested_amplitude_before_channel_scale'])
        applied_amp = float(channel_scale * requested_amp)
        realized_rms = float(comp['raw_signal_rms_before_scaling'] * applied_amp)
        realized_snr_db = None
        if output_channel_rms > 0.0 and realized_rms > 0.0:
            realized_snr_db = float(20.0 * np.log10(realized_rms / output_channel_rms))
        realized_snr_db_vs_original = None
        if sigma > 0.0 and realized_rms > 0.0:
            realized_snr_db_vs_original = float(20.0 * np.log10(realized_rms / sigma))
        active_vals = final_channel[comp['active_mask']]
        packed_records.append((
            int(comp['signal_idx']),
            {
                'channel_index': int(channel_index),
                'baseband_hz': float(comp['baseband_hz']),
                'amp_scale': float(comp['amp_scale']),
                'channel_rms': float(sigma),
                'channel_rms_after_scale': float(output_channel_rms),
                'channel_scale': float(channel_scale),
                'channel_rescaled_for_headroom': bool(channel_scale + 1e-12 < 1.0),
                'requested_amplitude': float(requested_amp),
                'applied_amplitude': float(applied_amp),
                'safe_amplitude_cap': float(channel_scale * requested_amp),
                'effective_tone_count': int(comp['effective_tone_count']),
                'width_hz': float(comp['width_hz']),
                'raw_signal_rms_before_scaling': float(comp['raw_signal_rms_before_scaling']),
                'requested_active_signal_rms_before_channel_scale': float(comp['requested_active_signal_rms_before_channel_scale']),
                'requested_active_signal_rms': float(comp['requested_active_signal_rms_before_channel_scale'] * channel_scale),
                'realized_active_signal_rms': float(realized_rms),
                'requested_snr_db': float(comp['case'].snr_db),
                'realized_snr_db': realized_snr_db,
                'realized_snr_db_vs_original': realized_snr_db_vs_original,
                'amplitude_limited_by_clip': False,
                'active_component_clip_fraction': float(_fraction_components_clipped(active_vals)),
            },
        ))
    return final_channel, packed_records, channel_meta


def _inject_case_inplace(
    x: np.ndarray,
    contract: DatContract,
    case: InjectionCase,
    channel_sigma: np.ndarray,
) -> dict[str, Any]:
    rows = x.shape[0]
    fs = contract.coarse_df_hz
    active = _active_envelope(rows, fs, case)
    amp_env = _amp_envelope(rows, fs, case)
    target_records = []
    effective_tone_count = _effective_tone_count(case)
    for idx, (ch, baseband, amp_scale) in enumerate(_target_channels(contract, case)):
        _validate_single_channel_envelope(contract, case, baseband)
        sigma = float(channel_sigma[ch])
        sig = _complex_signal(rows, fs, case, baseband)
        shaped = (active * amp_env).astype(np.float32) * sig
        raw_sig_rms = _signal_rms_on_active_samples(sig, active, amp_env)
        amp_plan = _amplitude_plan(x[:, ch], shaped, sigma, case, amp_scale, raw_sig_rms)
        amp = float(amp_plan['applied_amplitude'])
        x[:, ch] += amp * shaped
        target_records.append({
            "channel_index": int(ch),
            "baseband_hz": float(baseband),
            "amp_scale": float(amp_scale),
            "channel_rms": sigma,
            "requested_amplitude": float(amp_plan['requested_amplitude']),
            "applied_amplitude": float(amp_plan['applied_amplitude']),
            "safe_amplitude_cap": float(amp_plan['safe_amplitude_cap']),
            "effective_tone_count": int(effective_tone_count),
            "width_hz": float(case.bandwidth_hz),
            "raw_signal_rms_before_scaling": float(raw_sig_rms),
            "requested_active_signal_rms": float(amp_plan['requested_active_signal_rms']),
            "realized_active_signal_rms": float(amp_plan['realized_active_signal_rms']),
            "requested_snr_db": float(amp_plan['requested_snr_db']),
            "realized_snr_db": amp_plan['realized_snr_db'],
            "amplitude_limited_by_clip": bool(amp_plan['amplitude_limited_by_clip']),
            "active_component_clip_fraction": float(amp_plan['active_component_clip_fraction']),
        })
    start_row = max(0, int(case.start_s * fs))
    end_row = min(rows, start_row + int(case.duration_s * fs))
    return {
        "name": case.name,
        "morphology": case.morphology,
        "requested": case.to_plan_dict(),
        "resolved": {
            "start_s": float(case.start_s),
            "duration_s": float(case.duration_s),
            "start_freq_hz": float(case.abs_freq_hz),
            "drift_hz_per_s": float(case.drift_hz_per_s),
            "snr_db": float(case.snr_db),
            "width_hz": float(case.bandwidth_hz),
            "effective_tone_count": int(effective_tone_count),
            "start_row": int(start_row),
            "end_row": int(end_row),
        },
        "active_samples": int(np.count_nonzero(active)),
        "active_fraction": float(np.mean(active > 0.0)),
        "amplitude_limited_by_clip": bool(any(bool(tc.get("amplitude_limited_by_clip", False)) for tc in target_records)),
        "target_channels": target_records,
    }



def _clipping_summary(x: np.ndarray) -> dict[str, Any]:
    rounded_real = np.rint(x.real)
    rounded_imag = np.rint(x.imag)
    clip_real = int(np.count_nonzero((rounded_real < -128.0) | (rounded_real > 127.0)))
    clip_imag = int(np.count_nonzero((rounded_imag < -128.0) | (rounded_imag > 127.0)))
    total_components = int(2 * x.size)
    return {
        "real_components_clipped": clip_real,
        "imag_components_clipped": clip_imag,
        "total_components": total_components,
        "clip_fraction": float((clip_real + clip_imag) / max(total_components, 1)),
        "max_abs_real_before_clip": float(np.max(np.abs(x.real))),
        "max_abs_imag_before_clip": float(np.max(np.abs(x.imag))),
    }



def inject_cases_into_words(words: np.ndarray, contract: DatContract, cases: Iterable[InjectionCase]) -> tuple[np.ndarray, dict[str, Any]]:
    case_list = list(cases)
    if not case_list:
        raise ValueError("At least one InjectionCase is required")
    base_x = decode_words(words)
    rows = int(base_x.shape[0])
    signal_reports, channel_plan = _build_signal_reports_and_channel_plan(case_list, rows, contract)
    work_x = np.array(base_x, copy=True)
    rescaled_channels = 0
    for ch, entries in sorted(channel_plan.items()):
        final_channel, packed_records, channel_meta = _apply_channel_entries(int(ch), base_x[:, ch], contract, entries)
        work_x[:, ch] = final_channel
        if channel_meta.get("channel_rescaled_for_headroom", False):
            rescaled_channels += 1
        for signal_idx, target in packed_records:
            signal_reports[signal_idx]["target_channels"].append(target)
            signal_reports[signal_idx]["channel_rescaled_for_headroom"] = bool(signal_reports[signal_idx].get("channel_rescaled_for_headroom", False) or target.get("channel_rescaled_for_headroom", False))
            signal_reports[signal_idx]["amplitude_limited_by_clip"] = bool(signal_reports[signal_idx].get("amplitude_limited_by_clip", False) or target.get("amplitude_limited_by_clip", False))
    report = {
        "n_signals": len(case_list),
        "n_signals_limited_by_clip": int(sum(1 for sig in signal_reports if sig.get("amplitude_limited_by_clip", False))),
        "n_signals_rescaled_for_headroom": int(sum(1 for sig in signal_reports if sig.get("channel_rescaled_for_headroom", False))),
        "n_channels_rescaled_for_headroom": int(rescaled_channels),
        "signals": signal_reports,
        "clipping": _clipping_summary(work_x),
    }
    return encode_complex_to_words(work_x), report



def inject_case_into_words(words: np.ndarray, contract: DatContract, case: InjectionCase) -> np.ndarray:
    out_words, _ = inject_cases_into_words(words, contract, [case])
    return out_words



def _observation_manifest_record(plan: InjectionPlan, dat_path: str | Path, contract: DatContract) -> dict[str, Any]:
    return {
        "obs_id": plan.obs_id or plan.plan_name,
        "dat_paths": [str(Path(dat_path))],
        "source_name": plan.source_name,
        "beam_id": plan.beam_id,
        "pol_id": plan.pol_id,
        "scan_id": plan.scan_id or plan.plan_name,
        "target_id": plan.target_id or plan.plan_name,
        "lo_hz": contract.lo_hz,
        "start_coarse_channel": contract.start_coarse_channel,
    }



def write_injected_dataset(
    base_dat: str | Path,
    out_dat: str | Path,
    contract: DatContract,
    plan: InjectionPlan,
    *,
    report_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Path | None]:
    if not plan.signals:
        raise ValueError("Injection plan must contain at least one signal")

    base_dat = Path(base_dat)
    out_dat = Path(out_dat)
    out_dat.parent.mkdir(parents=True, exist_ok=True)
    words = np.memmap(base_dat, dtype='<u2', mode='r')
    if len(words) % contract.channels != 0:
        raise ValueError(
            f"File {base_dat} has {len(words)} uint16 cells, which is not divisible by channels={contract.channels}"
        )
    rows = len(words) // contract.channels
    base_mm = np.memmap(base_dat, dtype='<u2', mode='r', shape=(rows, contract.channels))

    signal_reports, channel_plan = _build_signal_reports_and_channel_plan(list(plan.signals), rows, contract)

    shutil.copyfile(base_dat, out_dat)
    out_mm = np.memmap(out_dat, dtype='<u2', mode='r+', shape=(rows, contract.channels))

    clip_real = 0
    clip_imag = 0
    max_abs_real_before_clip = 127.0
    max_abs_imag_before_clip = 127.0
    total_components = int(2 * rows * contract.channels)
    rescaled_channels = 0

    for ch, entries in sorted(channel_plan.items()):
        x = decode_words_to_complex64(np.asarray(base_mm[:, ch], dtype=np.uint16).reshape(-1, 1))[:, 0]
        final_channel, packed_records, channel_meta = _apply_channel_entries(int(ch), x, contract, entries)
        if channel_meta.get("channel_rescaled_for_headroom", False):
            rescaled_channels += 1
        for signal_idx, target in packed_records:
            signal_reports[signal_idx]["target_channels"].append(target)
            signal_reports[signal_idx]["channel_rescaled_for_headroom"] = bool(signal_reports[signal_idx].get("channel_rescaled_for_headroom", False) or target.get("channel_rescaled_for_headroom", False))
            signal_reports[signal_idx]["amplitude_limited_by_clip"] = bool(signal_reports[signal_idx].get("amplitude_limited_by_clip", False) or target.get("amplitude_limited_by_clip", False))
        rounded_real = np.rint(final_channel.real)
        rounded_imag = np.rint(final_channel.imag)
        clip_real += int(np.count_nonzero((rounded_real < -128.0) | (rounded_real > 127.0)))
        clip_imag += int(np.count_nonzero((rounded_imag < -128.0) | (rounded_imag > 127.0)))
        max_abs_real_before_clip = max(max_abs_real_before_clip, float(np.max(np.abs(final_channel.real))))
        max_abs_imag_before_clip = max(max_abs_imag_before_clip, float(np.max(np.abs(final_channel.imag))))
        out_mm[:, ch] = encode_complex_to_words(final_channel.reshape(-1, 1)).reshape(-1)

    inject_report = {
        "n_signals": len(plan.signals),
        "n_signals_limited_by_clip": int(sum(1 for sig in signal_reports if sig.get("amplitude_limited_by_clip", False))),
        "n_signals_rescaled_for_headroom": int(sum(1 for sig in signal_reports if sig.get("channel_rescaled_for_headroom", False))),
        "n_channels_rescaled_for_headroom": int(rescaled_channels),
        "signals": signal_reports,
        "clipping": {
            "real_components_clipped": int(clip_real),
            "imag_components_clipped": int(clip_imag),
            "total_components": total_components,
            "clip_fraction": float((clip_real + clip_imag) / max(total_components, 1)),
            "max_abs_real_before_clip": float(max_abs_real_before_clip),
            "max_abs_imag_before_clip": float(max_abs_imag_before_clip),
        },
    }

    resolved_report_path = Path(report_path) if report_path is not None else out_dat.with_name(out_dat.name + '.inject.json')
    resolved_manifest_path = Path(manifest_path) if manifest_path is not None else None
    if resolved_report_path is not None:
        resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    if resolved_manifest_path is not None:
        resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_record = _observation_manifest_record(plan, out_dat, contract)
    payload = {
        "plan_name": plan.plan_name,
        "base_dat_path": str(base_dat),
        "out_dat_path": str(out_dat),
        "rows": int(rows),
        "channels": int(contract.channels),
        "contract": asdict(contract),
        "observation": manifest_record,
        "notes": list(plan.notes),
        **inject_report,
    }
    resolved_report_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    if resolved_manifest_path is not None:
        manifest_payload = {"observations": [manifest_record]}
        resolved_manifest_path.write_text(yaml.safe_dump(manifest_payload, sort_keys=False), encoding='utf-8')

    return {
        "out_dat": out_dat,
        "report": resolved_report_path,
        "manifest": resolved_manifest_path,
    }


def write_injected_observation(base_dat: str | Path, out_dir: str | Path, contract: DatContract, case: InjectionCase) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for beam in case.beams:
        out_path = out_dir / f"{case.name}_beam{beam}.dat"
        plan = InjectionPlan(
            plan_name=case.name,
            signals=(case,),
            obs_id=case.name + f'_beam{beam}',
            source_name='inject',
            beam_id=beam,
            pol_id='00',
            scan_id=case.name,
            target_id=case.name,
        )
        write_injected_dataset(base_dat, out_path, contract, plan)
        meta = {
            'obs_id': case.name + f'_beam{beam}',
            'beam_id': beam,
            'pol_id': '00',
            'scan_id': case.name,
            'target_id': case.name,
            'source_name': 'inject',
            'lo_hz': contract.lo_hz,
            'start_coarse_channel': contract.start_coarse_channel,
        }
        (out_dir / f"{case.name}_beam{beam}.json").write_text(json.dumps(meta, indent=2), encoding='utf-8')
        out_paths.append(out_path)
    return out_paths
