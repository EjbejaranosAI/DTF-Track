"""Re-ID gallery: restore the original id after an ID switch without stealing identities."""

import numpy as np

from dynamic_tracking.identity import IdentityManager, ReIdConfig, add_track_appearance
from dynamic_tracking.types import Track

# strong_threshold > 1: always by consecutive hits (what most tests check)
CFG = ReIdConfig(confirm_hits=3, min_samples_to_store=3, sample_min_interval_sec=0.0,
                 strong_threshold=1.01)


def _unit(seed: int, dim: int = 512) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _noisy(base: np.ndarray, seed: int, noise: float = 0.15) -> np.ndarray:
    v = base + noise * _unit(1000 + seed)
    return v / np.linalg.norm(v)


def _person(tid: int, t: float) -> Track:
    tr = Track(tid, "person", 0.0, 0.0, 0.9, t)
    return tr


def _see(tr: Track, emb: np.ndarray, t: float) -> None:
    tr.last_rgb_person_sec = t
    add_track_appearance(tr, emb, t, CFG)


def _run_id_switch(mgr, appearance_old, appearance_new, t0=0.0):
    """Track 1 seen 5 times, dies, track 2 is born and is seen 5 times."""
    tracks = [_person(1, t0)]
    t = t0
    for k in range(5):
        t += 0.5
        _see(tracks[0], _noisy(appearance_old, k), t)
        mgr.step(tracks, t)
    tracks = []
    t += 0.5
    mgr.step(tracks, t)
    new = _person(2, t)
    tracks = [new]
    for k in range(5):
        t += 0.5
        _see(new, _noisy(appearance_new, 50 + k), t)
        mgr.step(tracks, t)
    return new, t


def test_same_person_recovers_original_id():
    mgr = IdentityManager(CFG)
    a = _unit(1)
    new, _ = _run_id_switch(mgr, a, a)
    assert new.track_id == 1
    assert new.reid_restored_from == 2
    assert 1 not in mgr.gallery


def test_different_person_keeps_new_id():
    mgr = IdentityManager(CFG)
    new, _ = _run_id_switch(mgr, _unit(1), _unit(2))
    assert new.track_id == 2
    assert 1 in mgr.gallery


def test_needs_consecutive_hits_before_restoring():
    mgr = IdentityManager(CFG)
    a = _unit(1)
    tracks = [_person(1, 0.0)]
    for k in range(5):
        _see(tracks[0], _noisy(a, k), 0.5 * (k + 1))
        mgr.step(tracks, 0.5 * (k + 1))
    mgr.step([], 3.0)
    new = _person(2, 3.0)
    for k in range(CFG.confirm_hits - 1):
        _see(new, _noisy(a, 60 + k), 3.5 + k)
        mgr.step([new], 3.5 + k)
    assert new.track_id == 2
    _see(new, _noisy(a, 99), 10.0)
    mgr.step([new], 10.0)
    assert new.track_id == 1


def test_never_reuses_id_of_live_track():
    """Track 1 is still alive (coasting): track 2 with the same clothes cannot take its id."""
    mgr = IdentityManager(CFG)
    a = _unit(1)
    old = _person(1, 0.0)
    for k in range(5):
        _see(old, _noisy(a, k), 0.5 * (k + 1))
        mgr.step([old], 0.5 * (k + 1))
    new = _person(2, 3.0)
    for k in range(5):
        _see(new, _noisy(a, 40 + k), 3.5 + 0.5 * k)
        mgr.step([old, new], 3.5 + 0.5 * k)
    assert new.track_id == 2 and old.track_id == 1


def test_covisible_tracks_are_not_merged():
    """If the old one was seen by the camera after the new one appeared, they are two people."""
    mgr = IdentityManager(CFG)
    a = _unit(1)
    old, new = _person(1, 0.0), _person(2, 0.0)
    for k in range(5):
        t = 0.5 * (k + 1)
        _see(old, _noisy(a, k), t)
        _see(new, _noisy(a, 30 + k), t)
        mgr.step([old, new], t)
    mgr.step([new], 3.0)  # track 1 dies
    for k in range(5):
        _see(new, _noisy(a, 70 + k), 3.5 + 0.5 * k)
        mgr.step([new], 3.5 + 0.5 * k)
    assert new.track_id == 2


def test_name_follows_restored_id():
    mgr = IdentityManager(CFG)
    a = _unit(1)
    first = _person(1, 0.0)
    tracks = [first]
    mgr.step(tracks, 0.0)
    assert mgr.enroll(tracks, 1, "Edison")
    for k in range(5):
        _see(first, _noisy(a, k), 0.5 * (k + 1))
        mgr.step(tracks, 0.5 * (k + 1))
    mgr.step([], 3.0)
    assert mgr.gallery[1].name == "Edison"
    new = _person(7, 3.0)
    for k in range(5):
        _see(new, _noisy(a, 80 + k), 3.5 + 0.5 * k)
        mgr.step([new], 3.5 + 0.5 * k)
    assert new.track_id == 1
    assert new.reid_name == "Edison"
    assert mgr.identities([new])[0]["name"] == "Edison"


def test_unnamed_lost_entries_expire():
    cfg = ReIdConfig(confirm_hits=3, min_samples_to_store=3, sample_min_interval_sec=0.0, lost_ttl_sec=10.0)
    mgr = IdentityManager(cfg)
    tr = _person(1, 0.0)
    for k in range(5):
        tr.last_rgb_person_sec = float(k)
        add_track_appearance(tr, _unit(1), float(k), cfg)
        mgr.step([tr], float(k))
    mgr.step([], 5.0)
    assert 1 in mgr.gallery
    mgr.step([], 16.0)
    assert 1 not in mgr.gallery


def test_named_gallery_roundtrip(tmp_path):
    mgr = IdentityManager(CFG)
    a = _unit(1)
    tr = _person(1, 0.0)
    mgr.step([tr], 0.0)
    mgr.enroll([tr], 1, "Edison")
    for k in range(5):
        _see(tr, _noisy(a, k), 0.5 * (k + 1))
        mgr.step([tr], 0.5 * (k + 1))
    path = str(tmp_path / "gallery.json")
    assert mgr.save_named(path) == 1

    # New run: the ids mean nothing, but the name is restored.
    mgr2 = IdentityManager(CFG)
    assert mgr2.load_named(path, 0.0) == 1
    new = _person(1, 0.0)
    for k in range(5):
        _see(new, _noisy(a, 90 + k), 0.5 * (k + 1))
        mgr2.step([new], 0.5 * (k + 1))
    assert new.track_id == 1
    assert new.reid_name == "Edison"


def _entry(tid, base, seeds, lost=0.0):
    from dynamic_tracking.identity import GalleryEntry
    return GalleryEntry(track_id=tid, bank=[_noisy(base, s) for s in seeds], lost_sec=lost,
                        last_rgb_sec=lost)


def test_duplicate_entries_of_one_person_do_not_block_recovery():
    """Live test 33: the gallery had two entries of the same person (ids 4 and 11, sim
    0.81 / 0.80): the margin always failed. Now the oldest id is restored and the
    duplicate is merged into it."""
    mgr = IdentityManager(CFG)
    a = _unit(1)
    mgr.gallery[4] = _entry(4, a, range(10, 20))
    mgr.gallery[11] = _entry(11, a, range(20, 25))
    new = _person(17, 5.0)
    for k in range(CFG.confirm_hits):
        _see(new, _noisy(a, 70 + k), 5.5 + k)
        mgr.step([new], 5.5 + k)
    assert new.track_id == 4 and new.reid_restored_from == 17
    assert not mgr.gallery                       # 11 was merged into 4


def test_two_similar_but_different_people_still_need_margin():
    """Two different people (centroids below duplicate_threshold) at the same
    distance from the new track: still not restored (an identity is not stolen)."""
    mgr = IdentityManager(CFG)
    a, b = _unit(1), _unit(2)
    mid = (a + b) / np.linalg.norm(a + b)
    mgr.gallery[4] = _entry(4, a, range(10, 15))
    mgr.gallery[9] = _entry(9, b, range(20, 25))
    new = _person(17, 5.0)
    for k in range(CFG.confirm_hits + 2):
        _see(new, mid, 5.5 + k)
        mgr.step([new], 5.5 + k)
    assert new.track_id == 17 and set(mgr.gallery) == {4, 9}


def test_strong_match_restores_with_one_sample_and_pending_until_then():
    cfg = ReIdConfig(confirm_hits=2, min_samples_to_store=3, sample_min_interval_sec=0.0)
    mgr = IdentityManager(cfg)
    a = _unit(1)
    mgr.gallery[5] = _entry(5, a, range(10, 20))
    new = _person(10, 5.0)
    new.class_name = "person"
    assert mgr.pending(new)                         # no samples yet: waiting for a verdict
    _see(new, _noisy(a, 80, noise=0.05), 5.5)       # very similar (>= 0.85): one is enough
    mgr.step([new], 5.5)
    assert new.track_id == 5 and not mgr.pending(new)
    stranger = _person(11, 6.0)
    _see(stranger, _unit(3), 6.5)
    mgr.step([new, stranger], 6.5)
    assert stranger.track_id == 11
    assert [i["pending"] for i in mgr.identities([new])] == [False]
