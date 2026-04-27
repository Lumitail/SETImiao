from pathlib import Path
import numpy as np

from acs.config import load_runtime_config
from acs.frontend.stft import build_spectrogram_tile
from acs.io.stitcher import build_observation_from_paths, read_rows
from acs.search.drift import iter_search_channels
from acs.types import DatContract, STFTConfig


def test_sparse_read_rows_matches_full_columns(tmp_path: Path):
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    data = np.arange(4096 * 256, dtype=np.uint16).reshape(4096, 256)
    path = tmp_path / "x.dat"
    data.tofile(path)
    obs = build_observation_from_paths([path], contract, obs_id="obs")
    full = read_rows(obs, 10, 30)
    sparse = read_rows(obs, 10, 30, channels=(2, 7, 19))
    assert sparse.shape == (20, 3)
    np.testing.assert_array_equal(sparse, full[:, [2, 7, 19]])


def test_sparse_spectrogram_preserves_true_channel_indices(tmp_path: Path):
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    rng = np.random.default_rng(123)
    data = rng.integers(0, 65535, size=(4096, 256), dtype=np.uint16)
    path = tmp_path / "x.dat"
    data.tofile(path)
    obs = build_observation_from_paths([path], contract, obs_id="obs")
    tile = build_spectrogram_tile(obs, 0, 4096, STFTConfig(name="test", nfft=2048, hop=1024), channel_indices=(5, 11))
    assert tile.channel_indices == (5, 11)
    assert tile.power.shape[1] == 2
    assert tile.fine_freq_hz.shape[0] == 2
    assert abs(tile.fine_freq_hz[0, 1024] - obs.freq_map.coarse_centers_hz[5]) < 1e-6
    assert abs(tile.fine_freq_hz[1, 1024] - obs.freq_map.coarse_centers_hz[11]) < 1e-6


def test_channel_list_filtering_from_config(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("""
contract:
  lo_hz: 1000000000
  start_coarse_channel: 27392
search_tile_rows: 31250
search_overlap_rows: 4096
search_stft:
  name: search
  nfft: 2048
  hop: 1024
drift:
  channel_min: 5
  channel_max: 20
  channel_list: [1, 5, 7, 20, 21]
""")
    cfg = load_runtime_config(cfg_path)
    assert iter_search_channels(256, cfg.drift) == [5, 7, 20]
