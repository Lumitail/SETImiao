from __future__ import annotations
import numpy as np
from ..types import SpectrogramTile, DriftConfig, Seed, Hit
from .backend import resolve_backend, search_channel_seeds_numba, search_channel_seeds_numpy, search_channel_seeds_cupy


def make_drift_trials(cfg: DriftConfig) -> np.ndarray:
    step = cfg.step_hz_per_s if cfg.step_hz_per_s is not None else 1.0
    return np.arange(cfg.min_hz_per_s, cfg.max_hz_per_s + 0.5 * step, step, dtype=np.float32)



def iter_search_channels(n_channels: int, cfg: DriftConfig) -> list[int]:
    """Return the coarse-channel indices searched by drift extraction.

    channel_list is a sparse channel-selection control for injection validation
    and targeted follow-up.  If channel_min/channel_max are also set, the list is
    filtered by that window.  When channel_list is omitted, behavior is unchanged.
    """
    ch0 = 0 if cfg.channel_min is None else max(0, int(cfg.channel_min))
    ch1 = n_channels if cfg.channel_max is None else min(n_channels, int(cfg.channel_max) + 1)
    if cfg.channel_list is None:
        return list(range(ch0, ch1))
    out: list[int] = []
    seen: set[int] = set()
    for raw in cfg.channel_list:
        ch = int(raw)
        if 0 <= ch < n_channels and ch0 <= ch < ch1 and ch not in seen:
            out.append(ch)
            seen.add(ch)
    return out


def _pick_top_bins(score: np.ndarray, valid_mask: np.ndarray, top_k: int, min_sep: int) -> list[int]:
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size == 0:
        return []
    order = valid_idx[np.argsort(score[valid_idx])[::-1]]
    chosen = []
    for idx in order:
        if all(abs(idx - c) >= min_sep for c in chosen):
            chosen.append(int(idx))
        if len(chosen) >= top_k:
            break
    return chosen


def extract_seeds(tile: SpectrogramTile, cfg: DriftConfig) -> list[Seed]:
    seeds: list[Seed] = []
    channels, _ = tile.mean_excess_db.shape
    if tile.channel_indices is not None and len(tile.channel_indices) == channels:
        search_channels = list(range(channels))
    else:
        search_channels = iter_search_channels(channels, cfg)
    for ch in search_channels:
        valid = ~tile.mask[ch]
        mean_bins = _pick_top_bins(tile.mean_excess_db[ch], valid, cfg.seed_top_k_mean_per_channel, cfg.seed_min_separation_bins)
        max_bins = _pick_top_bins(tile.max_excess_db[ch], valid, cfg.seed_top_k_max_per_channel, cfg.seed_min_separation_bins)
        for b in mean_bins:
            anchor = int(np.argmax(tile.norm_power[:, ch, b]))
            score = float(0.70 * tile.mean_excess_db[ch, b] + 0.20 * tile.max_excess_db[ch, b] + 0.10 * tile.p90_excess_db[ch, b])
            seeds.append(Seed(coarse_channel=ch, fine_bin=b, anchor_frame=anchor, score=score, seed_kind="mean"))
        for b in max_bins:
            anchor = int(np.argmax(tile.norm_power[:, ch, b]))
            score = float(0.25 * tile.mean_excess_db[ch, b] + 0.55 * tile.max_excess_db[ch, b] + 0.20 * tile.p90_excess_db[ch, b])
            seeds.append(Seed(coarse_channel=ch, fine_bin=b, anchor_frame=anchor, score=score, seed_kind="max"))
    uniq = {}
    for s in seeds:
        key = (s.coarse_channel, s.fine_bin)
        if key not in uniq or s.score > uniq[key].score:
            uniq[key] = s
    return list(uniq.values())


def _path_support_fraction(spec_ch: np.ndarray, seed_bin: int, anchor_frame: int, dbpf: float, width: int, threshold: float) -> float:
    n_frames, nfft = spec_ch.shape
    half = width // 2
    vals = []
    for t in range(n_frames):
        center = int(round(seed_bin + dbpf * (t - anchor_frame)))
        lo = center - half
        hi = center + half + 1
        if lo < 0 or hi > nfft:
            continue
        vals.append(float(np.mean(spec_ch[t, lo:hi])))
    if not vals:
        return 0.0
    vals = np.asarray(vals, dtype=np.float32)
    return float(np.mean(vals > threshold))


def search_tile(tile: SpectrogramTile, cfg: DriftConfig) -> list[Hit]:
    backend = resolve_backend(cfg.backend)
    drift_trials = make_drift_trials(cfg)
    frame_dt = tile.frame_times_s[1] - tile.frame_times_s[0] if len(tile.frame_times_s) > 1 else 1.0
    fine_df = abs(tile.fine_freq_hz[0, 1] - tile.fine_freq_hz[0, 0])
    drift_bins_per_frame = drift_trials * (frame_dt / fine_df)
    seeds = extract_seeds(tile, cfg)
    hits: list[Hit] = []
    seeds_by_channel: dict[int, list[Seed]] = {}
    for s in seeds:
        seeds_by_channel.setdefault(s.coarse_channel, []).append(s)
    widths = np.asarray(cfg.widths, dtype=np.int64)
    for ch, ch_seeds in seeds_by_channel.items():
        spec_ch = tile.norm_power[:, ch, :]
        seed_bins = np.asarray([s.fine_bin for s in ch_seeds], dtype=np.int64)
        anchor_frames = np.asarray([s.anchor_frame for s in ch_seeds], dtype=np.int64)
        best_scores = np.full(len(seed_bins), -1e30, dtype=np.float32)
        best_trials = np.zeros(len(seed_bins), dtype=np.int64)
        best_widths = np.ones(len(seed_bins), dtype=np.int64)
        for s0 in range(0, len(drift_bins_per_frame), cfg.trial_chunk_size):
            sub = drift_bins_per_frame[s0:s0 + cfg.trial_chunk_size]
            if backend == "cupy":
                scores, trials, widths_out = search_channel_seeds_cupy(spec_ch, seed_bins, anchor_frames, sub, widths)
            elif backend == "numpy":
                scores, trials, widths_out = search_channel_seeds_numpy(spec_ch, seed_bins, anchor_frames, sub, widths)
            else:
                scores, trials, widths_out = search_channel_seeds_numba(spec_ch, seed_bins, anchor_frames, sub, widths)
            improved = scores > best_scores
            best_scores[improved] = scores[improved]
            best_trials[improved] = s0 + trials[improved]
            best_widths[improved] = widths_out[improved]
        for i, seed in enumerate(ch_seeds):
            score = float(best_scores[i])
            if score < cfg.snr_threshold:
                continue
            trial_idx = int(best_trials[i])
            drift = float(drift_trials[trial_idx])
            support_fraction = _path_support_fraction(spec_ch, seed.fine_bin, seed.anchor_frame, float(drift_bins_per_frame[trial_idx]), int(best_widths[i]), cfg.support_threshold)
            edge_distance = int(min(seed.fine_bin, spec_ch.shape[1] - 1 - seed.fine_bin))
            hits.append(Hit(
                obs_id=tile.obs_id,
                beam_id=tile.beam_id,
                pol_id=tile.pol_id,
                scan_id=tile.scan_id,
                target_id=tile.target_id,
                row0=tile.row0,
                row1=tile.row1,
                tile_row0=tile.row0,
                tile_row1=tile.row1,
                coarse_channel=int(tile.channel_indices[ch]) if tile.channel_indices is not None else ch,
                fine_bin=seed.fine_bin,
                anchor_frame=seed.anchor_frame,
                freq_hz=float(tile.fine_freq_hz[ch, seed.fine_bin]),
                drift_hz_per_s=drift,
                incoherent_snr=score,
                width_bins=int(best_widths[i]),
                backend=backend,
                seed_kind=seed.seed_kind,
                support_fraction=support_fraction,
                edge_distance_bins=edge_distance,
            ))
    return hits
