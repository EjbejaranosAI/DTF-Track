"""Appearance identity gallery: restores the original track_id after an ID switch.

Cycle (call `step()` while holding the tracks lock, after YOLO confirmation):
  1) Person tracks that disappear -> their embedding bank moves to the gallery (key = old id).
  2) New person tracks (few camera samples) are compared with the gallery: with one
     sample >= `strong_threshold` they inherit its id at once; between `match_threshold` and
     that value, after `confirm_hits` consecutive matches with the same candidate.
     While deciding, the track is `pending` (consumers wait for its verdict).
  3) Names (`enroll`) travel with the id: a restored track also gets its name back.

Constraints so identities are not stolen:
  - only dead ids (never two live tracks with the same id);
  - the candidate cannot have been seen by the camera at the same time as the new track (co-visibility);
  - similarity >= threshold and a margin over the second-best candidate from ANOTHER person.

Duplicates: if a person is lost several times without being recovered, the gallery ends up
with several entries for them (e.g. ids 4 and 11) that fight over the margin, and the person is
never recovered again. Entries whose centroids are more similar than `duplicate_threshold`
are the same person: the margin is measured against the others, the canonical entry is
restored (the named one, or the oldest id) and the rest are merged
into it.
Numpy only: tested without torch or ROS.
"""

import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from .types import Track


@dataclass
class GalleryEntry:
    track_id: Optional[int]  # None for names loaded from disk (ids from another run)
    bank: List[np.ndarray]
    lost_sec: float
    last_rgb_sec: float
    name: Optional[str] = None


@dataclass
class _PendingMatch:
    key: int  # gallery key
    hits: int = 0
    best_sim: float = 0.0


@dataclass
class ReIdConfig:
    match_threshold: float = 0.75
    match_margin: float = 0.05
    # Two gallery entries are the same person if their centroids exceed this.
    duplicate_threshold: float = 0.8
    confirm_hits: int = 2
    # A single sample is enough if the similarity reaches this (less waiting on reappearance).
    strong_threshold: float = 0.85
    # The track stops searching the gallery after this many samples (it is "its own" by then).
    max_samples_to_match: int = 20
    min_samples_to_store: int = 3
    bank_size: int = 30
    # Minimum spacing between bank samples (s): avoids filling it with near-identical frames.
    sample_min_interval_sec: float = 0.4
    appearance_ema: float = 0.9
    lost_ttl_sec: float = 300.0
    gallery_max_entries: int = 64


def _l2n(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def bank_similarity(query: np.ndarray, bank: List[np.ndarray]) -> float:
    """Mean of the top-3 cosine similarities (robust to one bad sample)."""
    if not bank:
        return -1.0
    sims = np.asarray(bank) @ query
    k = min(3, sims.shape[0])
    return float(np.sort(sims)[-k:].mean())


def centroid(bank: List[np.ndarray]) -> np.ndarray:
    return _l2n(np.mean(np.asarray(bank), axis=0))


def add_track_appearance(tr: Track, emb: np.ndarray, stamp_sec: float, cfg: ReIdConfig) -> None:
    """Called when a YOLO person detection (with embedding) is associated to a track. The same
    YOLO box is associated over several consecutive scans with the same embedding (same cache
    object): it only counts the first time (otherwise it used up max_samples_to_match samples
    evaluating the same thing)."""
    if emb is tr.reid_last_raw:
        return
    tr.reid_last_raw = emb
    emb = _l2n(np.asarray(emb, dtype=np.float32))
    if tr.reid_appearance is None:
        tr.reid_appearance = emb
    else:
        a = float(cfg.appearance_ema)
        tr.reid_appearance = _l2n(a * tr.reid_appearance + (1.0 - a) * emb)
    tr.reid_samples += 1
    tr.reid_last_emb = emb
    tr.reid_last_emb_sec = float(stamp_sec)
    if tr.reid_first_rgb_sec <= 0.0:
        tr.reid_first_rgb_sec = float(stamp_sec)
    if stamp_sec - tr.reid_last_bank_sec >= cfg.sample_min_interval_sec:
        if tr.reid_bank is None:
            tr.reid_bank = deque(maxlen=int(cfg.bank_size))
        tr.reid_bank.append(emb)
        tr.reid_last_bank_sec = float(stamp_sec)


class IdentityManager:
    def __init__(self, cfg: Optional[ReIdConfig] = None, log: Optional[Callable[[str], None]] = None) -> None:
        self.cfg = cfg or ReIdConfig()
        self._log = log or (lambda _msg: None)
        self.gallery: Dict[int, GalleryEntry] = {}
        self._next_key = -1  # negative keys = entries without an id (loaded from disk)
        self._live: Dict[int, Track] = {}
        self._pending: Dict[int, _PendingMatch] = {}  # by id(Track)
        self.names: Dict[int, str] = {}  # live track_id -> name
        self.events: Deque[dict] = deque(maxlen=50)

    # ─── Public API ──────────────────────────────────────────────────

    def enroll(self, tracks: List[Track], track_id: int, name: str) -> bool:
        name = str(name).strip()
        if not name:
            return False
        tr = next((t for t in tracks if t.track_id == int(track_id)), None)
        if tr is None:
            return False
        # One name = one person: remove it from other tracks/entries.
        for tid, n in list(self.names.items()):
            if n == name and tid != tr.track_id:
                del self.names[tid]
        for key, e in list(self.gallery.items()):
            if e.name == name:
                if e.track_id is None:
                    del self.gallery[key]
                else:
                    e.name = None
        self.names[tr.track_id] = name
        tr.reid_name = name
        self._log(f"ReID: track {tr.track_id} enrolled as '{name}'")
        return True

    def forget(self, name: str) -> bool:
        name = str(name).strip()
        found = False
        for tid, n in list(self.names.items()):
            if n == name:
                del self.names[tid]
                found = True
        for t in self._live.values():
            if t.reid_name == name:
                t.reid_name = None
        for key, e in list(self.gallery.items()):
            if e.name == name:
                del self.gallery[key]
                found = True
        return found

    def step(self, tracks: List[Track], now_sec: float) -> List[Tuple[int, int, float]]:
        """Updates the gallery and restores ids. Returns [(previous_new_id, restored_id, sim)]."""
        self._collect_lost(tracks, now_sec)
        self._expire(now_sec)
        restored = self._match_new_tracks(tracks, now_sec)
        self._live = {t.track_id: t for t in tracks}
        for t in tracks:
            name = self.names.get(t.track_id)
            t.reid_name = name
        return restored

    def pending(self, tr: Track) -> bool:
        """The track may still inherit an id from the gallery (person without a verdict, with
        candidates): whoever is about to use its id should wait a little."""
        return (str(tr.class_name).lower() == "person" and not tr.reid_matched
                and tr.reid_samples <= self.cfg.max_samples_to_match
                and bool(self._candidates(tr)))

    def any_pending(self, tracks: List[Track]) -> bool:
        return any(self.pending(t) for t in tracks)

    def identities(self, tracks: List[Track]) -> List[dict]:
        out = []
        for t in tracks:
            pending = self.pending(t)
            if t.reid_appearance is None and t.track_id not in self.names and not pending:
                continue
            out.append({
                "track_id": int(t.track_id),
                "name": self.names.get(t.track_id),
                "samples": int(t.reid_samples),
                "restored_from": t.reid_restored_from,
                "pending": pending,
            })
        return out

    # ─── Persistence (named entries only) ────────────────────────────

    def save_named(self, path: str) -> int:
        data = {}
        for t in self._live.values():
            name = self.names.get(t.track_id)
            if name and t.reid_bank:
                data[name] = [e.tolist() for e in t.reid_bank]
        for e in self.gallery.values():
            if e.name and e.name not in data and e.bank:
                data[e.name] = [b.tolist() for b in e.bank]
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"version": 1, "identities": data}, f)
        os.replace(tmp, path)
        return len(data)

    def load_named(self, path: str, now_sec: float) -> int:
        if not path or not os.path.isfile(path):
            return 0
        with open(path) as f:
            data = json.load(f).get("identities", {})
        n = 0
        for name, bank in data.items():
            arr = [_l2n(np.asarray(b, dtype=np.float32)) for b in bank]
            if not arr:
                continue
            self.gallery[self._next_key] = GalleryEntry(
                track_id=None, bank=arr, lost_sec=now_sec, last_rgb_sec=0.0, name=str(name),
            )
            self._next_key -= 1
            n += 1
        return n

    # ─── Internos ────────────────────────────────────────────────────

    def _collect_lost(self, tracks: List[Track], now_sec: float) -> None:
        alive = {t.track_id for t in tracks}
        for tid, t in self._live.items():
            if tid in alive:
                continue
            name = self.names.pop(tid, None)
            if t.reid_bank is None or len(t.reid_bank) < self.cfg.min_samples_to_store:
                if name:
                    self._log(f"ReID: '{name}' (id {tid}) lost without enough samples")
                continue
            self.gallery[tid] = GalleryEntry(
                track_id=tid,
                bank=list(t.reid_bank),
                lost_sec=float(now_sec),
                last_rgb_sec=float(t.last_rgb_person_sec),
                name=name,
            )
        if len(self.gallery) > self.cfg.gallery_max_entries:
            unnamed = sorted(
                (k for k, e in self.gallery.items() if not e.name),
                key=lambda k: self.gallery[k].lost_sec,
            )
            for k in unnamed[: len(self.gallery) - self.cfg.gallery_max_entries]:
                del self.gallery[k]

    def _expire(self, now_sec: float) -> None:
        ttl = float(self.cfg.lost_ttl_sec)
        if ttl <= 0.0:
            return
        for k, e in list(self.gallery.items()):
            if not e.name and now_sec - e.lost_sec > ttl:
                del self.gallery[k]

    def _candidates(self, tr: Track) -> List[Tuple[int, GalleryEntry]]:
        out = []
        for k, e in self.gallery.items():
            # Co-visibility: if the old one was seen by the camera after the new one appeared
            # in the camera, they are two different people.
            if e.last_rgb_sec > 0.0 and tr.reid_first_rgb_sec > 0.0 and e.last_rgb_sec >= tr.reid_first_rgb_sec:
                continue
            out.append((k, e))
        return out

    def _same_person(self, a: int, b: int) -> bool:
        ea, eb = self.gallery[a], self.gallery[b]
        if ea.name and eb.name and ea.name != eb.name:
            return False
        return float(centroid(ea.bank) @ centroid(eb.bank)) >= self.cfg.duplicate_threshold

    def _canonical(self, keys: List[int]) -> int:
        """Among several entries of the same person: the named one or, if none, the oldest
        id (the one the other nodes know it by: conversation, history)."""
        def rank(k: int):
            e = self.gallery[k]
            return (e.name is None, e.track_id if e.track_id is not None else float("inf"))
        return min(keys, key=rank)

    def _match_new_tracks(self, tracks: List[Track], now_sec: float) -> List[Tuple[int, int, float]]:
        cfg = self.cfg
        restored: List[Tuple[int, int, float]] = []
        live_ids = {t.track_id for t in tracks}
        claimed: set = set()
        alive_objs = {id(t) for t in tracks}
        for k in list(self._pending):
            if k not in alive_objs:
                del self._pending[k]

        for tr in tracks:
            if str(tr.class_name).lower() != "person" or tr.reid_last_emb is None:
                continue
            if tr.reid_matched or tr.reid_samples > cfg.max_samples_to_match:
                continue
            # Only one evaluation per new embedding.
            if tr.reid_last_eval_sec >= tr.reid_last_emb_sec:
                continue
            tr.reid_last_eval_sec = tr.reid_last_emb_sec
            cands = self._candidates(tr)
            if not cands:
                continue
            q = tr.reid_last_emb
            scored = sorted(
                ((bank_similarity(q, e.bank), k) for k, e in cands), reverse=True,
            )
            best_sim, best_key = scored[0]
            # entries of the same person as the best one: they do not compete for the margin
            group = [best_key] + [k for _, k in scored[1:] if self._same_person(best_key, k)]
            second = next((s for s, k in scored[1:] if k not in group), -1.0)
            target = self._canonical(group)
            pend = self._pending.get(id(tr))
            dup = (f" (same person: ids {[self.gallery[k].track_id for k in group]})"
                   if len(group) > 1 else "")
            self._log(
                f"ReID eval: track {tr.track_id} (sample {tr.reid_samples}) best=id "
                f"{self.gallery[best_key].track_id} sim {best_sim:.2f}, 2nd {second:.2f}{dup}"
            )
            if best_sim < cfg.match_threshold or best_sim - second < cfg.match_margin:
                if pend is not None:
                    pend.hits = 0
                continue
            if pend is None or pend.key != target:
                pend = _PendingMatch(key=target)
                self._pending[id(tr)] = pend
            pend.hits += 1
            pend.best_sim = max(pend.best_sim, best_sim)
            hits_needed = 1 if best_sim >= cfg.strong_threshold else cfg.confirm_hits
            if pend.hits < hits_needed or any(k in claimed for k in group):
                continue

            entry = self.gallery.pop(target)
            claimed.update(group)
            for k in group:  # duplicates are merged into the restored entry
                if k == target:
                    continue
                dup_entry = self.gallery.pop(k)
                entry.bank = (entry.bank + dup_entry.bank)[-self.cfg.bank_size:]
                entry.name = entry.name or dup_entry.name
                self._log(f"ReID: entry id {dup_entry.track_id} merged into id {entry.track_id}")
            del self._pending[id(tr)]
            old_id = tr.track_id
            tr.reid_matched = True
            if entry.track_id is not None and entry.track_id not in live_ids:
                live_ids.discard(old_id)
                tr.track_id = entry.track_id
                live_ids.add(tr.track_id)
                tr.reid_restored_from = old_id
            if tr.reid_bank is not None:
                merged = deque(entry.bank, maxlen=tr.reid_bank.maxlen)
                merged.extend(tr.reid_bank)
                tr.reid_bank = merged
            name = entry.name or self.names.get(old_id)
            self.names.pop(old_id, None)
            if name:
                self.names[tr.track_id] = name
            restored.append((old_id, tr.track_id, best_sim))
            name_s = f" '{name}'" if name else ""
            self.events.append({
                "stamp": float(now_sec), "from": int(old_id), "to": int(tr.track_id),
                "sim": round(float(best_sim), 3), "name": name,
            })
            self._log(
                f"ReID: track {old_id} -> ID {tr.track_id}{name_s} restored "
                f"(sim {best_sim:.2f}, 2nd {second:.2f}, lost {now_sec - entry.lost_sec:.1f}s ago)"
            )
        return restored
