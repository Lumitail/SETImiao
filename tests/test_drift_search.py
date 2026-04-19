
import numpy as np
from pathlib import Path
from acs.types import DatContract, STFTConfig, DriftConfig, BaselineConfig
from acs.io.stitcher import build_observation_from_paths
from acs.frontend.stft import build_spectrogram_tile
from acs.preproc.baseline import apply_baseline_and_masks
from acs.search.drift import search_tile

def _encode_complex(z):
    i = np.clip(np.rint(z.real), -128, 127).astype(np.int16)
    q = np.clip(np.rint(z.imag), -128, 127).astype(np.int16)
    return (i.astype(np.int8).view(np.uint8).astype(np.uint16) | (q.astype(np.int8).view(np.uint8).astype(np.uint16) << 8)).astype(np.uint16)

def test_search_recovers_near_true_frequency(tmp_path: Path):
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    rows = 8192
    fs = contract.coarse_df_hz
    rng = np.random.default_rng(0)
    x = (rng.normal(0,12,size=(rows,256)) + 1j*rng.normal(0,12,size=(rows,256))).astype(np.complex64)
    ch = 120
    t = np.arange(rows)/fs
    freq0 = 1500.0
    drift = 12.0
    x[:, ch] += (40*np.exp(1j*2*np.pi*(freq0*t + 0.5*drift*t*t))).astype(np.complex64)
    p = tmp_path/'inj.dat'
    _encode_complex(x).tofile(p)
    obs = build_observation_from_paths([p], contract, obs_id='inj', beam_id='00', target_id='inj', scan_id='s0')
    tile = build_spectrogram_tile(obs, 0, rows, STFTConfig(name='search', nfft=1024, hop=512))
    tile = apply_baseline_and_masks(tile, BaselineConfig())
    hits = search_tile(tile, DriftConfig(min_hz_per_s=-32, max_hz_per_s=32, step_hz_per_s=1.0, snr_threshold=5.0, widths=(1,3,5)))
    assert hits
    true_abs = obs.freq_map.coarse_centers_hz[ch] + freq0
    best = min(hits, key=lambda h: abs((h.freq_hz or 0.0) - true_abs))
    assert abs((best.freq_hz or 0.0) - true_abs) <= 2 * (fs / 1024)
