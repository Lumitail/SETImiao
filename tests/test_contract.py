
from pathlib import Path
import numpy as np
from acs.types import DatContract
from acs.io.dat_reader import validate_geometry, decode_words_to_complex64

def test_validate_geometry_and_decode(tmp_path: Path):
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    arr = np.zeros((100, 256), dtype=np.uint16)
    path = tmp_path / "x.dat"
    arr.tofile(path)
    total_bytes, rows = validate_geometry(path, contract)
    assert rows == 100
    assert total_bytes == 100 * 256 * 2
    decoded = decode_words_to_complex64(arr[:2])
    assert decoded.shape == (2, 256)
