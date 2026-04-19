
from pathlib import Path
import numpy as np
from acs.types import DatContract
from acs.io.stitcher import build_observation_from_paths, read_rows

def test_cross_file_read(tmp_path: Path):
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    a = np.ones((10,256), dtype=np.uint16)
    b = np.full((10,256), 2, dtype=np.uint16)
    pa, pb = tmp_path/'a.dat', tmp_path/'b.dat'
    a.tofile(pa); b.tofile(pb)
    obs = build_observation_from_paths([pa,pb], contract, obs_id='obs')
    block = read_rows(obs, 8, 12)
    assert block.shape == (4,256)
    assert block[:2,0].tolist() == [1,1]
    assert block[2:,0].tolist() == [2,2]
