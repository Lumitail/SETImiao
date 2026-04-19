
from __future__ import annotations
import numpy as np
from ..types import STFTConfig, StitchedObservation, SpectrogramTile
from ..io.dat_reader import decode_words_to_complex64, fine_offsets_hz
from ..io.stitcher import read_rows

def get_window(name: str, n: int) -> np.ndarray:
    name = name.lower()
    if name == "hann":
        return np.hanning(n).astype(np.float32)
    if name == "hamming":
        return np.hamming(n).astype(np.float32)
    if name == "blackman":
        return np.blackman(n).astype(np.float32)
    return np.ones(n, dtype=np.float32)

def frame_times_s(n_frames: int, hop: int, nfft: int, coarse_sr_hz: float) -> np.ndarray:
    return ((np.arange(n_frames) * hop) + (nfft / 2.0)) / coarse_sr_hz

def build_spectrogram_tile(obs: StitchedObservation, row0: int, row1: int, stft: STFTConfig) -> SpectrogramTile:
    words = read_rows(obs, row0, row1)
    if words.shape[0] < stft.nfft:
        raise ValueError("Tile shorter than NFFT")
    x = decode_words_to_complex64(words)
    x = x - x.mean(axis=0, keepdims=True)
    n_rows, channels = x.shape
    n_frames = 1 + (n_rows - stft.nfft) // stft.hop
    win = get_window(stft.window, stft.nfft)[:, None]
    power = np.empty((n_frames, channels, stft.nfft), dtype=np.float32)
    for fi in range(n_frames):
        s = fi * stft.hop
        seg = x[s:s + stft.nfft, :] * win
        spec = np.fft.fftshift(np.fft.fft(seg, axis=0), axes=0)
        p = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)
        power[fi] = np.transpose(p, (1, 0))
    fine = obs.freq_map.coarse_centers_hz[:, None] + fine_offsets_hz(obs.contract, stft.nfft)[None, :]
    return SpectrogramTile(
        obs_id=obs.meta.obs_id,
        beam_id=obs.meta.beam_id,
        pol_id=obs.meta.pol_id,
        scan_id=obs.meta.scan_id,
        target_id=obs.meta.target_id,
        row0=row0,
        row1=row1,
        power=power,
        norm_power=np.empty_like(power),
        mask=np.zeros((channels, stft.nfft), dtype=bool),
        frame_times_s=frame_times_s(n_frames, stft.hop, stft.nfft, obs.contract.coarse_df_hz),
        fine_freq_hz=fine,
        mean_excess_db=np.empty((channels, stft.nfft), dtype=np.float32),
        max_excess_db=np.empty((channels, stft.nfft), dtype=np.float32),
        p90_excess_db=np.empty((channels, stft.nfft), dtype=np.float32),
    )
