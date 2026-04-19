
import numpy as np
from pathlib import Path
from acs.types import DatContract, STFTConfig, DriftConfig, BaselineConfig, RefineConfig
from acs.io.stitcher import build_observation_from_paths
from acs.frontend.stft import build_spectrogram_tile
from acs.preproc.baseline import apply_baseline_and_masks
from acs.search.drift import search_tile
from acs.refine.coherent import refine_hits

def _encode_complex(z):
    i = np.clip(np.rint(z.real), -128, 127).astype(np.int16)
    q = np.clip(np.rint(z.imag), -128, 127).astype(np.int16)
    return (i.astype(np.int8).view(np.uint8).astype(np.uint16) | (q.astype(np.int8).view(np.uint8).astype(np.uint16) << 8)).astype(np.uint16)

def test_refine_outputs_values(tmp_path: Path):
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    rows = 8192
    fs = contract.coarse_df_hz
    rng = np.random.default_rng(0)
    x = (rng.normal(0,12,size=(rows,256)) + 1j*rng.normal(0,12,size=(rows,256))).astype(np.complex64)
    ch = 100
    t = np.arange(rows)/fs
    x[:, ch] += (35*np.exp(1j*2*np.pi*(2500*t + 0.5*10*t*t))).astype(np.complex64)
    p = tmp_path/'inj.dat'
    _encode_complex(x).tofile(p)
    obs = build_observation_from_paths([p], contract, obs_id='inj', beam_id='00', target_id='inj', scan_id='s0')
    tile = build_spectrogram_tile(obs, 0, rows, STFTConfig(name='search', nfft=1024, hop=512))
    tile = apply_baseline_and_masks(tile, BaselineConfig())
    hits = search_tile(tile, DriftConfig(min_hz_per_s=-32, max_hz_per_s=32, step_hz_per_s=1.0, snr_threshold=5.0, widths=(1,3,5)))
    refined = refine_hits(obs, hits, RefineConfig(enabled=True, top_n=8, drift_half_window_hz_per_s=2.0, drift_step_hz_per_s=0.25))
    assert refined
    best = max(refined, key=lambda h: (h.refined_snr or -1))
    assert best.refined_snr is not None
    assert best.refined_freq_hz is not None
