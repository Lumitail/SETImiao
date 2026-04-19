
from pathlib import Path
from acs.io.manifest import load_manifest

def test_load_manifest(tmp_path: Path):
    p = tmp_path / "m.yaml"
    p.write_text("""
observations:
  - obs_id: obs0
    dat_paths: [/tmp/a.dat]
    beam_id: "00"
    lo_hz: 1000000000.0
    start_coarse_channel: 27392
""")
    specs = load_manifest(p)
    assert len(specs) == 1
    assert specs[0].obs_id == "obs0"
