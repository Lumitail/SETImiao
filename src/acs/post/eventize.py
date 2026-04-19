from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Optional

from ..types import Hit, Event, CoincidenceConfig, CandidateConfig


def _interval_gap(row0_a: int, row1_a: int, row0_b: int, row1_b: int) -> int:
    if row1_a < row0_b:
        return row0_b - row1_a
    if row1_b < row0_a:
        return row0_a - row1_b
    return 0


def _overlap_rows(row0_a: int, row1_a: int, row0_b: int, row1_b: int) -> int:
    return max(0, min(row1_a, row1_b) - max(row0_a, row0_b))


def _mid_time_s(row0: int, row1: int, native_dt_s: float) -> float:
    return 0.5 * (row0 + row1) * native_dt_s


def _cluster_bounds(cl: "_ClusterState") -> tuple[int, int]:
    return min(h.row0 for h in cl.hits), max(h.row1 for h in cl.hits)


def _cluster_duration_s(cl: "_ClusterState", native_dt_s: float) -> float:
    row0, row1 = _cluster_bounds(cl)
    return max(0.0, float(row1 - row0) * native_dt_s)


def _is_short_cluster(cl: "_ClusterState", cfg: CandidateConfig, native_dt_s: float) -> bool:
    if len(cl.hits) <= int(cfg.merge_short_max_hits):
        return True
    return _cluster_duration_s(cl, native_dt_s) <= float(cfg.merge_short_max_duration_s)


def _event_duration_s(ev: Event, native_dt_s: float) -> float:
    return max(0.0, float(ev.row1 - ev.row0) * native_dt_s)


def _is_short_event(ev: Event, cfg: CandidateConfig, native_dt_s: float) -> bool:
    if int(ev.n_hits) <= int(cfg.merge_short_max_hits):
        return True
    return _event_duration_s(ev, native_dt_s) <= float(cfg.merge_short_max_duration_s)


def _reference_drift(hit: Hit) -> float:
    return float(hit.refined_drift_hz_per_s if hit.refined_drift_hz_per_s is not None else hit.drift_hz_per_s)


def _reference_freq(hit: Hit) -> Optional[float]:
    if hit.refined_freq_hz is not None:
        return float(hit.refined_freq_hz)
    if hit.freq_hz is not None:
        return float(hit.freq_hz)
    return None


def _anchor_row(hit: Hit, frame_hop_rows: int | None, frame_window_rows: int | None) -> int:
    if frame_hop_rows is None or frame_window_rows is None:
        return int(round(0.5 * (hit.row0 + hit.row1)))
    return int(round(hit.row0 + hit.anchor_frame * frame_hop_rows + 0.5 * frame_window_rows))


def _anchor_time_s(hit: Hit, native_dt_s: float, frame_hop_rows: int | None, frame_window_rows: int | None) -> float:
    return _anchor_row(hit, frame_hop_rows, frame_window_rows) * native_dt_s


@dataclass
class _ClusterState:
    hits: list[Hit] = field(default_factory=list)
    drift_history: list[float] = field(default_factory=list)
    last_hit: Hit | None = None
    last_mid_s: float = 0.0

    def representative_drift(self) -> float:
        if not self.drift_history:
            return 0.0
        return float(median(self.drift_history))

    def representative_coarse_channel(self) -> int:
        if not self.hits:
            return -1
        counts: dict[int, int] = {}
        for h in self.hits:
            counts[h.coarse_channel] = counts.get(h.coarse_channel, 0) + 1
        return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    def best_hit(self) -> Hit:
        return max(
            self.hits,
            key=lambda h: (
                h.refined_snr if h.refined_snr is not None else -1e9,
                h.incoherent_snr,
                -h.row0,
            ),
        )

    def compatible(
        self,
        hit: Hit,
        coincidence_cfg: CoincidenceConfig,
        native_dt_s: float,
        max_row_gap: int,
        frame_hop_rows: int | None,
        frame_window_rows: int | None,
    ) -> tuple[bool, float]:
        if self.last_hit is None:
            return True, 0.0
        ref = self.last_hit
        if ref.obs_id != hit.obs_id or ref.beam_id != hit.beam_id:
            return False, float("inf")
        if ref.coarse_channel != hit.coarse_channel:
            return False, float("inf")
        ref_freq = _reference_freq(ref)
        hit_freq = _reference_freq(hit)
        if ref_freq is None or hit_freq is None:
            return False, float("inf")
        if _interval_gap(ref.row0, ref.row1, hit.row0, hit.row1) > max_row_gap:
            return False, float("inf")
        rep_drift = self.representative_drift()
        drift_err = abs(rep_drift - _reference_drift(hit))
        if drift_err > coincidence_cfg.drift_tol_hz_per_s:
            return False, float("inf")
        hit_mid = _anchor_time_s(hit, native_dt_s, frame_hop_rows, frame_window_rows)
        pred_freq = ref_freq + rep_drift * (hit_mid - self.last_mid_s)
        freq_err = abs(hit_freq - pred_freq)
        if freq_err > coincidence_cfg.freq_tol_hz:
            return False, float("inf")
        cost = (freq_err / max(coincidence_cfg.freq_tol_hz, 1e-9)) + (
            drift_err / max(coincidence_cfg.drift_tol_hz_per_s, 1e-9)
        )
        return True, float(cost)

    def add(
        self,
        hit: Hit,
        native_dt_s: float,
        frame_hop_rows: int | None,
        frame_window_rows: int | None,
    ) -> None:
        self.hits.append(hit)
        self.drift_history.append(_reference_drift(hit))
        hit_mid = _anchor_time_s(hit, native_dt_s, frame_hop_rows, frame_window_rows)
        if self.last_hit is None or hit_mid >= self.last_mid_s:
            self.last_hit = hit
            self.last_mid_s = hit_mid

    def absorb(
        self,
        other: "_ClusterState",
        native_dt_s: float,
        frame_hop_rows: int | None,
        frame_window_rows: int | None,
    ) -> None:
        for h in sorted(other.hits, key=lambda hh: _anchor_time_s(hh, native_dt_s, frame_hop_rows, frame_window_rows)):
            self.add(h, native_dt_s, frame_hop_rows, frame_window_rows)


def _event_score(best: Hit, n_hits: int) -> tuple[float, list[str]]:
    refined = float(best.refined_snr if best.refined_snr is not None else 0.0)
    gain = float(best.coherent_gain_db if best.coherent_gain_db is not None else -10.0)
    support = float(best.support_fraction)
    width = int(best.width_bins)
    edge = int(best.edge_distance_bins)
    reasons = []
    score = 0.0
    score += 0.75 * refined
    score += 0.25 * float(best.incoherent_snr)
    score += 1.2 * math.log2(1.0 + n_hits)
    score += 2.5 * min(1.0, support)
    if n_hits >= 2:
        score += 1.0
        reasons.append("multi_hit_support")
    else:
        score -= 2.0
        reasons.append("singleton_penalty")
    if refined >= 15.0:
        reasons.append("strong_refine")
    else:
        score -= 2.5
        reasons.append("weak_refine")
    if gain < -1.0:
        score -= 1.5
        reasons.append("negative_gain_penalty")
    if width > 3:
        score -= 0.75 * float(width - 3)
        reasons.append("broad_width_penalty")
    if edge < 8:
        score -= 0.15 * float(8 - edge)
        reasons.append("edge_penalty")
    return float(score), reasons


def _candidate_pass(event: Event, best: Hit, cfg: CandidateConfig, native_dt_s: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    refined = float(event.best_refined_snr if event.best_refined_snr is not None else 0.0)
    gain = float(event.best_coherent_gain_db if event.best_coherent_gain_db is not None else -10.0)
    duration_s = max(0.0, float(event.row1 - event.row0) * native_dt_s)
    if event.score < cfg.min_event_score:
        return False, ["score_below_threshold"]
    if refined < cfg.min_refined_snr:
        return False, ["refined_snr_below_threshold"]
    if event.best_width_bins > cfg.max_width_bins:
        return False, ["width_above_threshold"]
    if event.best_support_fraction < cfg.min_support_fraction:
        return False, ["support_below_threshold"]

    # v0.2.5X: for a continuous-narrowband survey, a single tile is not a
    # sufficiently reliable final candidate, even when coherent refinement makes
    # its local search metric large.  Such objects remain in events.jsonl for
    # diagnostics, but they are withheld from candidates.jsonl/review unless the
    # config explicitly re-enables singleton candidates.
    if event.n_hits < cfg.min_candidate_hits:
        if cfg.allow_singleton_candidates and event.n_hits == 1 and refined >= cfg.allow_singleton_if_refined_snr_ge and gain >= -1.0:
            reasons.append("strong_singleton_refine")
            return True, reasons
        return False, ["insufficient_track_hits"]

    if duration_s < cfg.min_candidate_duration_s:
        return False, ["track_duration_below_threshold"]

    if event.n_hits >= cfg.min_hits_if_no_strong_refine:
        reasons.append("multi_hit_event")
        return True, reasons
    return False, ["singleton_without_strong_refine"]


def _cluster_reference_track(
    cl: _ClusterState,
    native_dt_s: float,
    frame_hop_rows: int | None,
    frame_window_rows: int | None,
) -> tuple[Hit, float, float, float]:
    best = cl.best_hit()
    freq = float(_reference_freq(best) or 0.0)
    drift = float(_reference_drift(best))
    tref = _anchor_time_s(best, native_dt_s, frame_hop_rows, frame_window_rows)
    return best, freq, drift, tref


def _clusters_compatible_for_merge(
    a: _ClusterState,
    b: _ClusterState,
    coincidence_cfg: CoincidenceConfig,
    candidate_cfg: CandidateConfig,
    native_dt_s: float,
    frame_hop_rows: int | None,
    frame_window_rows: int | None,
    merge_max_gap_rows: int,
) -> bool:
    if not a.hits or not b.hits:
        return False
    a_best = a.best_hit()
    b_best = b.best_hit()
    if a_best.obs_id != b_best.obs_id or a_best.beam_id != b_best.beam_id:
        return False
    if a.representative_coarse_channel() != b.representative_coarse_channel():
        return False
    a_row0, a_row1 = _cluster_bounds(a)
    b_row0, b_row1 = _cluster_bounds(b)
    if _interval_gap(a_row0, a_row1, b_row0, b_row1) > merge_max_gap_rows:
        return False
    _, a_freq, a_drift, a_tref = _cluster_reference_track(a, native_dt_s, frame_hop_rows, frame_window_rows)
    _, b_freq, b_drift, b_tref = _cluster_reference_track(b, native_dt_s, frame_hop_rows, frame_window_rows)

    a_singleton = len(a.hits) <= 1
    b_singleton = len(b.hits) <= 1
    relaxed_singleton = a_singleton or b_singleton
    short_fragment_raw = _is_short_cluster(a, candidate_cfg, native_dt_s) or _is_short_cluster(b, candidate_cfg, native_dt_s)
    overlap_rows = _overlap_rows(a_row0, a_row1, b_row0, b_row1)
    shorter_rows = max(1, min(a_row1 - a_row0, b_row1 - b_row0))
    overlap_fraction_of_short = overlap_rows / shorter_rows
    # Do not merge two co-temporal tracks merely because one is short; those are
    # handled by candidate deduplication.  Short-fragment merge is reserved for
    # edge/adjacent fragments that only touch or weakly overlap the long track.
    short_fragment = short_fragment_raw and overlap_fraction_of_short < candidate_cfg.merge_short_max_overlap_fraction

    drift_tol = coincidence_cfg.drift_tol_hz_per_s
    freq_tol = candidate_cfg.merge_freq_tol_hz
    if relaxed_singleton:
        drift_tol = max(drift_tol, candidate_cfg.merge_singleton_drift_tol_hz_per_s)
        freq_tol = max(freq_tol, candidate_cfg.merge_singleton_freq_tol_hz)
    if short_fragment:
        drift_tol = max(drift_tol, candidate_cfg.merge_short_drift_tol_hz_per_s)
        freq_tol = max(freq_tol, candidate_cfg.merge_short_freq_tol_hz)
    if abs(a_drift - b_drift) > drift_tol:
        return False

    eval_row0 = max(a_row0, b_row0)
    eval_row1 = min(a_row1, b_row1)
    if eval_row1 <= eval_row0:
        eval_row0 = min(a_row1, b_row1)
        eval_row1 = max(a_row0, b_row0)
    eval_t = _mid_time_s(eval_row0, eval_row1, native_dt_s)
    a_pred = a_freq + a_drift * (eval_t - a_tref)
    b_pred = b_freq + b_drift * (eval_t - b_tref)
    if abs(a_pred - b_pred) <= freq_tol:
        return True

    # Edge fragments with only a few hits can have a biased drift estimate.  When a
    # short fragment touches or overlaps a long track, compare every short-fragment
    # anchor to the long-track prediction rather than trusting the short drift.
    if short_fragment:
        a_is_short = _is_short_cluster(a, candidate_cfg, native_dt_s)
        b_is_short = _is_short_cluster(b, candidate_cfg, native_dt_s)
        if a_is_short != b_is_short:
            short = a if a_is_short else b
            long = b if a_is_short else a
        else:
            short = a if len(a.hits) <= len(b.hits) else b
            long = b if short is a else a
        _, long_freq, long_drift, long_tref = _cluster_reference_track(long, native_dt_s, frame_hop_rows, frame_window_rows)
        errs: list[float] = []
        for short_hit in short.hits:
            short_freq = _reference_freq(short_hit)
            if short_freq is None:
                continue
            short_t = _anchor_time_s(short_hit, native_dt_s, frame_hop_rows, frame_window_rows)
            long_pred = long_freq + long_drift * (short_t - long_tref)
            errs.append(abs(float(short_freq) - long_pred))
        if errs and min(errs) <= freq_tol:
            return True

    if a_singleton != b_singleton:
        short = a if a_singleton else b
        long = b if a_singleton else a
        short_best = short.best_hit()
        _long_best, long_freq, long_drift, long_tref = _cluster_reference_track(long, native_dt_s, frame_hop_rows, frame_window_rows)
        short_freq = _reference_freq(short_best)
        if short_freq is None:
            return False
        short_t = _anchor_time_s(short_best, native_dt_s, frame_hop_rows, frame_window_rows)
        long_pred = long_freq + long_drift * (short_t - long_tref)
        return abs(short_freq - long_pred) <= freq_tol
    return False

def _merge_clusters(
    clusters: list[_ClusterState],
    coincidence_cfg: CoincidenceConfig,
    candidate_cfg: CandidateConfig,
    native_dt_s: float,
    frame_hop_rows: int | None,
    frame_window_rows: int | None,
) -> list[_ClusterState]:
    if len(clusters) <= 1:
        return clusters
    merge_max_gap_rows = int(round(candidate_cfg.merge_max_gap_s / max(native_dt_s, 1e-12)))
    changed = True
    clusters = list(clusters)
    while changed:
        changed = False
        clusters.sort(key=lambda cl: (cl.hits[0].obs_id if cl.hits else "", cl.hits[0].beam_id if cl.hits else "", min(h.row0 for h in cl.hits) if cl.hits else 0))
        for i in range(len(clusters)):
            if changed:
                break
            for j in range(i + 1, len(clusters)):
                if _clusters_compatible_for_merge(
                    clusters[i],
                    clusters[j],
                    coincidence_cfg,
                    candidate_cfg,
                    native_dt_s,
                    frame_hop_rows,
                    frame_window_rows,
                    merge_max_gap_rows,
                ):
                    clusters[i].absorb(clusters[j], native_dt_s, frame_hop_rows, frame_window_rows)
                    del clusters[j]
                    changed = True
                    break
    return clusters


def _estimate_support_rows(
    hit_list: list[Hit],
    native_dt_s: float,
    frame_hop_rows: int | None,
    frame_window_rows: int | None,
    tile_step_rows: int | None,
) -> tuple[int, int]:
    tile_row0 = min(h.row0 for h in hit_list)
    tile_row1 = max(h.row1 for h in hit_list)
    if len(hit_list) == 1 or frame_hop_rows is None or frame_window_rows is None:
        return tile_row0, tile_row1
    anchors = sorted(_anchor_row(h, frame_hop_rows, frame_window_rows) for h in hit_list)
    uniq_starts = sorted(set(h.row0 for h in hit_list))
    inferred_step = tile_step_rows if tile_step_rows is not None and tile_step_rows > 0 else 0
    if inferred_step <= 0 and len(uniq_starts) >= 2:
        diffs = [b - a for a, b in zip(uniq_starts[:-1], uniq_starts[1:]) if b > a]
        if diffs:
            inferred_step = int(round(median(diffs)))
    if inferred_step <= 0:
        inferred_step = max(frame_hop_rows, int(round(0.5 * (tile_row1 - tile_row0))))
    pad = max(frame_window_rows // 2, inferred_step // 2)
    start = max(tile_row0, anchors[0] - pad)
    end = min(tile_row1, anchors[-1] + pad)
    if end <= start:
        return tile_row0, tile_row1
    return int(start), int(end)


def _event_pred_freq_at_time(
    event: Event,
    best_hit: Hit,
    t_s: float,
    native_dt_s: float,
    frame_hop_rows: int | None,
    frame_window_rows: int | None,
) -> float:
    ref_freq = float(_reference_freq(best_hit) or 0.0)
    ref_t = _anchor_time_s(best_hit, native_dt_s, frame_hop_rows, frame_window_rows)
    return ref_freq + float(event.drift_hz_per_s) * (t_s - ref_t)


def _deduplicate_candidates(
    candidates: list[Event],
    best_hits: dict[str, Hit],
    cfg: CandidateConfig,
    native_dt_s: float,
    frame_hop_rows: int | None,
    frame_window_rows: int | None,
) -> list[Event]:
    kept: list[Event] = []
    base_dedup_gap_rows = int(round(cfg.dedup_max_gap_s / max(native_dt_s, 1e-12)))
    short_gap_rows = max(base_dedup_gap_rows, int(round(cfg.merge_max_gap_s / max(native_dt_s, 1e-12))))
    for ev in sorted(candidates, key=lambda e: e.score, reverse=True):
        suppress = False
        ev_best = best_hits[ev.event_id]
        for keep in kept:
            keep_best = best_hits[keep.event_id]
            if ev.obs_id != keep.obs_id or ev.beam_id != keep.beam_id:
                continue
            if ev_best.coarse_channel != keep_best.coarse_channel:
                continue

            short_pair = _is_short_event(ev, cfg, native_dt_s) or _is_short_event(keep, cfg, native_dt_s)
            freq_tol = max(cfg.dedup_freq_tol_hz, cfg.dedup_short_freq_tol_hz) if short_pair else cfg.dedup_freq_tol_hz
            drift_tol = cfg.dedup_short_drift_tol_hz_per_s if short_pair else cfg.merge_singleton_drift_tol_hz_per_s
            gap_limit = short_gap_rows if short_pair else base_dedup_gap_rows
            # Do not require the shadow fragment's drift to match.  A short or
            # low-support shadow often has a biased drift estimate; frequency-time
            # proximity is the safer duplicate test.
            _ = drift_tol

            overlap = _overlap_rows(ev.row0, ev.row1, keep.row0, keep.row1)
            gap = _interval_gap(ev.row0, ev.row1, keep.row0, keep.row1)
            if overlap > 0:
                shorter = min(ev.row1 - ev.row0, keep.row1 - keep.row0)
                if shorter <= 0:
                    continue
                # For short fragments, any temporal contact is enough; for full
                # tracks retain the fractional-overlap guard.
                if not short_pair and (overlap / max(shorter, 1)) < cfg.dedup_time_overlap_fraction:
                    continue
                tmid = 0.5 * (max(ev.row0, keep.row0) + min(ev.row1, keep.row1)) * native_dt_s
                ferr = abs(
                    _event_pred_freq_at_time(ev, ev_best, tmid, native_dt_s, frame_hop_rows, frame_window_rows)
                    - _event_pred_freq_at_time(keep, keep_best, tmid, native_dt_s, frame_hop_rows, frame_window_rows)
                )
                if ferr <= freq_tol:
                    suppress = True
                    break
                continue
            if gap <= gap_limit:
                if ev.row1 <= keep.row0:
                    bridge_row = 0.5 * (ev.row1 + keep.row0)
                else:
                    bridge_row = 0.5 * (keep.row1 + ev.row0)
                bridge_t = bridge_row * native_dt_s
                ferr = abs(
                    _event_pred_freq_at_time(ev, ev_best, bridge_t, native_dt_s, frame_hop_rows, frame_window_rows)
                    - _event_pred_freq_at_time(keep, keep_best, bridge_t, native_dt_s, frame_hop_rows, frame_window_rows)
                )
                if ferr <= freq_tol:
                    suppress = True
                    break
        if not suppress:
            kept.append(ev)
    return kept

def cluster_hits_to_events(
    hits: list[Hit],
    coincidence_cfg: CoincidenceConfig,
    candidate_cfg: CandidateConfig,
    native_dt_s: float = 1.0,
    frame_hop_rows: int | None = None,
    frame_window_rows: int | None = None,
    tile_step_rows: int | None = None,
) -> tuple[list[Event], dict[str, list[Hit]], list[dict], list[Event]]:
    if not hits:
        return [], {}, [], []
    max_row_gap = int(round(coincidence_cfg.max_row_gap_s / max(native_dt_s, 1e-12)))
    hits_sorted = sorted(hits, key=lambda h: (h.obs_id, h.beam_id or "", h.row0, h.row1, h.freq_hz or 0.0))
    clusters: list[_ClusterState] = []
    for hit in hits_sorted:
        best_idx = None
        best_cost = float("inf")
        for idx, cl in enumerate(clusters):
            ok, cost = cl.compatible(
                hit,
                coincidence_cfg,
                native_dt_s,
                max_row_gap,
                frame_hop_rows,
                frame_window_rows,
            )
            if ok and cost < best_cost:
                best_idx = idx
                best_cost = cost
        if best_idx is None:
            cl = _ClusterState()
            cl.add(hit, native_dt_s, frame_hop_rows, frame_window_rows)
            clusters.append(cl)
        else:
            clusters[best_idx].add(hit, native_dt_s, frame_hop_rows, frame_window_rows)

    clusters = _merge_clusters(
        clusters,
        coincidence_cfg,
        candidate_cfg,
        native_dt_s,
        frame_hop_rows,
        frame_window_rows,
    )

    events: list[Event] = []
    event_hits: dict[str, list[Hit]] = {}
    best_hits: dict[str, Hit] = {}
    for i, cl in enumerate(clusters):
        hit_list = list(cl.hits)
        best = cl.best_hit()
        best_refined = max((h.refined_snr for h in hit_list if h.refined_snr is not None), default=None)
        score, notes = _event_score(best, len(hit_list))
        row0, row1 = _estimate_support_rows(
            hit_list,
            native_dt_s,
            frame_hop_rows,
            frame_window_rows,
            tile_step_rows,
        )
        eid = f"{best.obs_id}_event_{i:05d}"
        event = Event(
            event_id=eid,
            obs_id=best.obs_id,
            beam_id=best.beam_id,
            pol_id=best.pol_id,
            scan_id=best.scan_id,
            target_id=best.target_id,
            row0=row0,
            row1=row1,
            freq_hz=_reference_freq(best),
            drift_hz_per_s=_reference_drift(best),
            score=float(score),
            n_hits=len(hit_list),
            best_incoherent_snr=float(best.incoherent_snr),
            best_refined_snr=float(best_refined) if best_refined is not None else None,
            best_width_bins=int(best.width_bins),
            best_support_fraction=float(best.support_fraction),
            best_coherent_gain_db=float(best.coherent_gain_db) if best.coherent_gain_db is not None else None,
            representative_coarse_channel=int(cl.representative_coarse_channel()),
            peak_row=int(_anchor_row(best, frame_hop_rows, frame_window_rows)),
            notes=tuple(notes),
        )
        events.append(event)
        event_hits[eid] = hit_list
        best_hits[eid] = best

    coincidences: list[dict] = []
    used = set()
    for i, e in enumerate(events):
        if i in used:
            continue
        group = [i]
        beams = {e.beam_id}
        e_mid = _mid_time_s(e.row0, e.row1, native_dt_s)
        for j in range(i + 1, len(events)):
            o = events[j]
            if e.scan_id != o.scan_id or e.target_id != o.target_id:
                continue
            if e.beam_id == o.beam_id:
                continue
            if e.freq_hz is None or o.freq_hz is None:
                continue
            o_mid = _mid_time_s(o.row0, o.row1, native_dt_s)
            if abs(e_mid - o_mid) > coincidence_cfg.time_tol_s:
                continue
            if abs(e.freq_hz - o.freq_hz) <= coincidence_cfg.freq_tol_hz and abs(e.drift_hz_per_s - o.drift_hz_per_s) <= coincidence_cfg.drift_tol_hz_per_s:
                group.append(j)
                beams.add(o.beam_id)
        if len(group) > 1 and len(beams) > 1:
            gid = f"coinc_{i:05d}"
            coinc_beams = tuple(sorted(b for b in beams if b is not None))
            coincidences.append({"coincidence_id": gid, "event_ids": [events[g].event_id for g in group], "beam_ids": list(coinc_beams)})
            for g in group:
                ev = events[g]
                events[g] = Event(**{
                    **ev.to_dict(),
                    "beam_multiplicity": len(coinc_beams),
                    "coincident_beams": coinc_beams,
                    "is_multibeam_coincident": True,
                    "coincidence_group_id": gid,
                    "score": float(ev.score - coincidence_cfg.multibeam_penalty),
                    "notes": tuple(list(ev.notes) + ["multibeam_coincident"]),
                })
                used.add(g)

    candidates: list[Event] = []
    updated_events: list[Event] = []
    for ev in events:
        best = best_hits[ev.event_id]
        ok, reasons = _candidate_pass(ev, best, candidate_cfg, native_dt_s)
        ev2 = Event(**{
            **ev.to_dict(),
            "candidate_passed": bool(ok),
            "candidate_reasons": tuple(reasons),
        })
        updated_events.append(ev2)
        if ok:
            candidates.append(ev2)

    updated_events.sort(key=lambda e: e.score, reverse=True)
    candidates = _deduplicate_candidates(
        candidates,
        best_hits,
        candidate_cfg,
        native_dt_s,
        frame_hop_rows,
        frame_window_rows,
    )
    candidates.sort(key=lambda e: e.score, reverse=True)
    return updated_events, event_hits, coincidences, candidates
