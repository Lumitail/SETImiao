import json
from pathlib import Path
import numpy as np
from acs.cli.main import main

def test_smoke_cli_runs(tmp_path: Path):
    rows = 4096
    data = np.zeros((rows,256), dtype=np.uint16)
    dat = tmp_path/'x.dat'
    data.tofile(dat)
    out = tmp_path/'smoke'
    cfg = Path(__file__).resolve().parents[1] / 'configs' / 'h1_search.yaml'
    rc = main([
        'smoke-sample', str(dat),
        '--config', str(cfg),
        '--out-dir', str(out),
        '--obs-id', 'smoke',
        '--source-name', 'test',
        '--beam-id', '00',
        '--pol-id', '00',
        '--scan-id', 'scan',
        '--target-id', 'test',
        '--lo-hz', '1000000000',
        '--start-coarse-channel', '27392',
    ])
    assert rc == 0
    summary = json.loads((out/'smoke_summary.json').read_text())
    assert 'n_hits' in summary and 'n_events' in summary and 'n_candidates' in summary
