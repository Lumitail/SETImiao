from acs.types import DatContract
from acs.bench.inject import build_case_suite

def test_suite_counts():
    contract = DatContract(lo_hz=1e9, start_coarse_channel=27392)
    assert len(build_case_suite(contract, "v017_10")) == 10
    assert len(build_case_suite(contract, "v018_phase1_5")) == 5
    full30 = build_case_suite(contract, "v018_full30")
    assert len(full30) == 30
    assert len({c.name for c in full30}) == 30
