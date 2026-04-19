
from __future__ import annotations
from pathlib import Path
import yaml
from ..types import ObservationSpec

def load_manifest(path: str | Path) -> list[ObservationSpec]:
    obj = yaml.safe_load(Path(path).read_text())
    obs = obj.get("observations", [])
    specs: list[ObservationSpec] = []
    for entry in obs:
        specs.append(
            ObservationSpec(
                obs_id=str(entry["obs_id"]),
                dat_paths=tuple(Path(p) for p in entry["dat_paths"]),
                source_name=entry.get("source_name"),
                beam_id=entry.get("beam_id"),
                pol_id=entry.get("pol_id"),
                scan_id=entry.get("scan_id"),
                target_id=entry.get("target_id"),
                timestamp_start=entry.get("timestamp_start"),
                lo_hz=float(entry["lo_hz"]) if entry.get("lo_hz") is not None else None,
                start_coarse_channel=int(entry["start_coarse_channel"]) if entry.get("start_coarse_channel") is not None else None,
            )
        )
    return specs
