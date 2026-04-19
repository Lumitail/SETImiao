
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from ..types import DatContract, FrequencyMap

def validate_geometry(dat_path: str | Path, contract: DatContract) -> tuple[int, int]:
    path = Path(dat_path)
    total_bytes = path.stat().st_size
    if total_bytes % contract.row_bytes != 0:
        raise ValueError(f"Illegal file size: {total_bytes} not divisible by {contract.row_bytes}")
    rows = total_bytes // contract.row_bytes
    return total_bytes, rows

def parse_filename_metadata(dat_path: str | Path) -> dict:
    stem = Path(dat_path).stem
    parts = stem.split("_")
    out = {"filename": Path(dat_path).name}
    if len(parts) >= 5:
        out["software"] = parts[0]
        out["source"] = parts[1]
        out["band_label"] = parts[2]
        out["receiver"] = parts[3]
        out["beam_1based"] = parts[4] if parts[4].isdigit() else None
    return out

def open_raw_u16_memmap(dat_path: str | Path, rows: int, channels: int) -> np.memmap:
    return np.memmap(Path(dat_path), dtype="<u2", mode="r", shape=(rows, channels))

def decode_words_to_complex64(words_u16: np.ndarray) -> np.ndarray:
    i = (words_u16 & 0x00FF).astype(np.uint8).view(np.int8).astype(np.float32)
    q = ((words_u16 >> 8) & 0x00FF).astype(np.uint8).view(np.int8).astype(np.float32)
    return (i + 1j * q).astype(np.complex64)

def coarse_centers_hz(contract: DatContract) -> np.ndarray:
    if contract.lo_hz is None or contract.start_coarse_channel is None:
        raise ValueError("Absolute frequency mapping requires lo_hz and start_coarse_channel")
    coarse_df = contract.coarse_df_hz
    return contract.lo_hz + (contract.start_coarse_channel + np.arange(contract.channels)) * coarse_df

def fine_offsets_hz(contract: DatContract, nfft: int) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / contract.coarse_df_hz)).astype(np.float64)

def build_frequency_map(contract: DatContract, nfft: int, relative: bool = False) -> FrequencyMap:
    if relative or contract.lo_hz is None or contract.start_coarse_channel is None:
        coarse = np.arange(contract.channels, dtype=np.float64) * contract.coarse_df_hz
        mode = "relative"
    else:
        coarse = coarse_centers_hz(contract)
        mode = "absolute"
    return FrequencyMap(mode=mode, coarse_centers_hz=coarse.astype(np.float64), fine_offsets_hz=fine_offsets_hz(contract, nfft))

def load_sidecar(path: str | Path | None) -> dict:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    raise ValueError(f"Unsupported sidecar format: {p.suffix}")
