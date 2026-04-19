
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..types import DatContract, ObservationMeta, ObservationSpec, FileSegment, FrequencyMap, StitchedObservation
from .dat_reader import validate_geometry, open_raw_u16_memmap, build_frequency_map

def build_observation_from_paths(
    dat_paths: list[Path],
    contract: DatContract,
    obs_id: str,
    source_name: str | None = None,
    beam_id: str | None = None,
    pol_id: str | None = None,
    scan_id: str | None = None,
    target_id: str | None = None,
    timestamp_start: str | None = None,
    lo_hz: float | None = None,
    start_coarse_channel: int | None = None,
) -> StitchedObservation:
    # clone contract with per-observation metadata overrides
    contract = DatContract(
        channels=contract.channels,
        bytes_per_cell=contract.bytes_per_cell,
        frontend_fs_hz=contract.frontend_fs_hz,
        channelizer_fft=contract.channelizer_fft,
        lo_hz=lo_hz if lo_hz is not None else contract.lo_hz,
        start_coarse_channel=start_coarse_channel if start_coarse_channel is not None else contract.start_coarse_channel,
        hi_rest_hz=contract.hi_rest_hz,
    )
    segments = []
    cursor = 0
    for path in dat_paths:
        _, rows = validate_geometry(path, contract)
        segments.append(FileSegment(path=Path(path), rows=rows, row_start=cursor, row_stop=cursor + rows))
        cursor += rows
    meta = ObservationMeta(
        obs_id=obs_id,
        dat_paths=tuple(Path(p) for p in dat_paths),
        source_name=source_name,
        beam_id=beam_id,
        pol_id=pol_id,
        scan_id=scan_id,
        target_id=target_id,
        timestamp_start=timestamp_start,
        lo_hz=contract.lo_hz,
        start_coarse_channel=contract.start_coarse_channel,
        metadata_source_path=None,
    )
    freq_map = build_frequency_map(contract, nfft=2048, relative=(contract.lo_hz is None or contract.start_coarse_channel is None))
    return StitchedObservation(meta=meta, contract=contract, segments=tuple(segments), total_rows=cursor, freq_map=freq_map)

def build_observations_from_specs(specs: list[ObservationSpec], contract: DatContract, nfft: int = 2048) -> list[StitchedObservation]:
    obs_list: list[StitchedObservation] = []
    for s in specs:
        c = DatContract(
            channels=contract.channels,
            bytes_per_cell=contract.bytes_per_cell,
            frontend_fs_hz=contract.frontend_fs_hz,
            channelizer_fft=contract.channelizer_fft,
            lo_hz=s.lo_hz if s.lo_hz is not None else contract.lo_hz,
            start_coarse_channel=s.start_coarse_channel if s.start_coarse_channel is not None else contract.start_coarse_channel,
            hi_rest_hz=contract.hi_rest_hz,
        )
        obs = build_observation_from_paths(
            list(s.dat_paths), c, s.obs_id, s.source_name, s.beam_id, s.pol_id, s.scan_id, s.target_id, s.timestamp_start, c.lo_hz, c.start_coarse_channel
        )
        obs_list.append(obs)
    return obs_list

def read_rows(obs: StitchedObservation, row0: int, row1: int) -> np.ndarray:
    if row0 < 0 or row1 > obs.total_rows or row1 < row0:
        raise ValueError("Bad row slice")
    out = np.empty((row1 - row0, obs.contract.channels), dtype=np.uint16)
    write_ptr = 0
    for seg in obs.segments:
        if row1 <= seg.row_start or row0 >= seg.row_stop:
            continue
        loc0 = max(row0, seg.row_start)
        loc1 = min(row1, seg.row_stop)
        src0 = loc0 - seg.row_start
        src1 = loc1 - seg.row_start
        mm = open_raw_u16_memmap(seg.path, seg.rows, obs.contract.channels)
        block = np.asarray(mm[src0:src1], dtype=np.uint16)
        out[write_ptr:write_ptr + len(block)] = block
        write_ptr += len(block)
    return out


def read_channel_rows(obs: StitchedObservation, row0: int, row1: int, channel: int) -> np.ndarray:
    if row0 < 0 or row1 > obs.total_rows or row1 < row0:
        raise ValueError("Bad row slice")
    if channel < 0 or channel >= obs.contract.channels:
        raise ValueError(f"Bad channel index {channel}")
    out = np.empty((row1 - row0,), dtype=np.uint16)
    write_ptr = 0
    for seg in obs.segments:
        if row1 <= seg.row_start or row0 >= seg.row_stop:
            continue
        loc0 = max(row0, seg.row_start)
        loc1 = min(row1, seg.row_stop)
        src0 = loc0 - seg.row_start
        src1 = loc1 - seg.row_start
        mm = open_raw_u16_memmap(seg.path, seg.rows, obs.contract.channels)
        block = np.asarray(mm[src0:src1, channel], dtype=np.uint16)
        out[write_ptr:write_ptr + len(block)] = block
        write_ptr += len(block)
    return out
