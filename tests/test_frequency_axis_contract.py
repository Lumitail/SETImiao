
from pathlib import Path
import numpy as np
from acs.types import DatContract, STFTConfig
from acs.io.stitcher import build_observation_from_paths
from acs.frontend.stft import build_spectrogram_tile

def test_frequency_axis_matches_stft_nfft(tmp_path: Path):
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    words = np.zeros((4096,256), dtype=np.uint16)
    p = tmp_path/'x.dat'
    words.tofile(p)
    obs = build_observation_from_paths([p], contract, obs_id='obs')
    tile = build_spectrogram_tile(obs, 0, 4096, STFTConfig(name='s', nfft=512, hop=256))
    assert tile.fine_freq_hz.shape == (256,512)
