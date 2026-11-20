"""Static-map and costmap cluster filters (extracted from dynamic_tracking_node)."""

import math
from typing import List, Optional, Tuple

import numpy as np
from nav_msgs.msg import OccupancyGrid

from .tf_utils import TFHelper
from .geometry import quat_to_rotmat
from .types import LidarMeasurement


class CostmapFilter:
    """Nav2 costmap filter: rejects LiDAR clusters that land on occupied cells."""

    def __init__(
        self,
        occupancy_threshold: int,
        inflation_threshold: int,
        unknown_as_free: bool,
        neighbor_radius_cells: int,
    ) -> None:
        self.occupancy_threshold = occupancy_threshold
        self.inflation_threshold = inflation_threshold
        self.unknown_as_free = unknown_as_free
        self.neighbor_radius = max(0, neighbor_radius_cells)

    def filter(
        self,
        clusters: List[LidarMeasurement],
        costmap: Optional[OccupancyGrid],
        tracking_frame: str,
        tf_helper: TFHelper,
    ) -> List[LidarMeasurement]:
        if costmap is None or not clusters:
            return clusters
        rt = self._costmap_rt(costmap, tracking_frame, tf_helper)
        return [c for c in clusters if self._keep(c, costmap, rt)]

    # ------------------------------------------------------------------
    def _costmap_rt(self, cm: OccupancyGrid, tracking_frame: str, tf: TFHelper):
        fid = cm.header.frame_id
        if fid == tracking_frame:
            return None
        return tf.lookup_R_t(fid, tracking_frame, None)

    def _to_costmap_xy(self, mx, my, cm, rt) -> Optional[Tuple[float, float]]:
        if rt is None:
            if cm.header.frame_id:
                return mx, my
            return None
        R, t = rt
        p = R @ np.array([mx, my, 0.0], dtype=np.float64) + t
        return float(p[0]), float(p[1])

    def _to_cell(self, mx, my, cm, rt) -> Optional[Tuple[int, int]]:
        p = self._to_costmap_xy(mx, my, cm, rt)
        if p is None:
            return None
        o = cm.info.origin
        res = float(cm.info.resolution)
        q = o.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        dx, dy = p[0] - o.position.x, p[1] - o.position.y
        c, s = math.cos(-yaw), math.sin(-yaw)
        ix = int(math.floor((dx * c - dy * s) / res))
        iy = int(math.floor((dx * s + dy * c) / res))
        w, h = int(cm.info.width), int(cm.info.height)
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return None
        return ix, iy

    def _cell_blocks(self, val: Optional[int]) -> bool:
        if val is None:
            return False
        if val < 0:
            return not self.unknown_as_free
        if val >= self.occupancy_threshold:
            return True
        if self.inflation_threshold > 0 and val >= self.inflation_threshold:
            return True
        return False

    def _occ_at(self, ix, iy, cm) -> Optional[int]:
        w, h = int(cm.info.width), int(cm.info.height)
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return None
        return int(cm.data[iy * w + ix])

    def _keep(self, lm: LidarMeasurement, cm: OccupancyGrid, rt) -> bool:
        cell = self._to_cell(lm.map_x, lm.map_y, cm, rt)
        if cell is None:
            return True
        ix, iy = cell
        r = self.neighbor_radius
        if r == 0:
            return not self._cell_blocks(self._occ_at(ix, iy, cm))
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if self._cell_blocks(self._occ_at(ix + di, iy + dj, cm)):
                    return False
        return True


class StaticMapFilter:
    """Filters clusters against /map OccupancyGrid (walls only, no inflation)."""

    def __init__(
        self,
        occupied_threshold: int,
        neighbor_radius_cells: int,
        keep_if_tf_fails: bool,
        corner_probe_enable: bool,
        probe_radius_m: float,
        corner_probe_radius_m: float,
        disk_filter_enable: bool,
        disk_radius_m: float,
        disk_min_free_ratio: float,
        disk_permit_outside_map: bool,
    ) -> None:
        self.occ_thr = occupied_threshold
        self.neighbor_r = neighbor_radius_cells
        self.keep_if_tf_fails = keep_if_tf_fails
        self.corner_probe = corner_probe_enable
        self.probe_r = probe_radius_m
        self.corner_probe_r = corner_probe_radius_m if corner_probe_radius_m > 1e-9 else probe_radius_m
        self.disk_enable = disk_filter_enable
        self.disk_r = disk_radius_m
        self.disk_min_free = disk_min_free_ratio
        self.disk_outside_ok = disk_permit_outside_map

    def filter(
        self,
        clusters: List[LidarMeasurement],
        static_map: Optional[OccupancyGrid],
        tracking_frame: str,
        tf_helper: TFHelper,
    ) -> List[LidarMeasurement]:
        if static_map is None or not clusters:
            return clusters
        return [c for c in clusters if not self._rejects(c.map_x, c.map_y, static_map, tracking_frame, tf_helper)]

    def point_on_wall(
        self, mx: float, my: float, static_map: Optional[OccupancyGrid],
        tracking_frame: str, tf_helper: TFHelper,
    ) -> bool:
        if static_map is None:
            return False
        return self._rejects(mx, my, static_map, tracking_frame, tf_helper)

    # ------------------------------------------------------------------
    def _cell_occupied(self, mx, my, sm) -> bool:
        res = float(sm.info.resolution)
        o = sm.info.origin
        ix = int(math.floor((mx - o.position.x) / res))
        iy = int(math.floor((my - o.position.y) / res))
        w, h = int(sm.info.width), int(sm.info.height)
        r = self.neighbor_r
        if r == 0:
            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                return False
            return int(sm.data[iy * w + ix]) >= self.occ_thr
        for di in range(-r, r + 1):
            ni = ix + di
            if ni < 0 or ni >= w:
                continue
            for dj in range(-r, r + 1):
                nj = iy + dj
                if nj < 0 or nj >= h:
                    continue
                if int(sm.data[nj * w + ni]) >= self.occ_thr:
                    return True
        return False

    def _disk_rejects(self, mx, my, sm) -> bool:
        if not self.disk_enable:
            return False
        res = float(sm.info.resolution)
        if res <= 1e-9:
            return False
        o = sm.info.origin
        cix = int(math.floor((mx - o.position.x) / res))
        ciy = int(math.floor((my - o.position.y) / res))
        radius = int(math.ceil(max(0.0, self.disk_r) / res))
        w, h = int(sm.info.width), int(sm.info.height)
        visited = free = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                cx, cy = cix + dx, ciy + dy
                if cx < 0 or cx >= w or cy < 0 or cy >= h:
                    continue
                visited += 1
                if int(sm.data[cy * w + cx]) < self.occ_thr:
                    free += 1
        if visited == 0:
            return not self.disk_outside_ok
        return (free / float(visited)) <= self.disk_min_free

    def _rejects(self, tx, ty, sm, tracking_frame, tf) -> bool:
        map_frame = sm.header.frame_id or "map"
        center = tf.tracking_to_map(tx, ty, tracking_frame, map_frame)
        if center is None:
            return not self.keep_if_tf_fails
        mx, my = center
        cr = self.corner_probe_r

        if not self.corner_probe or cr <= 1e-9:
            if self._cell_occupied(mx, my, sm):
                return True
            return self._disk_rejects(mx, my, sm)

        probes = [(mx, my), (mx + cr, my), (mx - cr, my), (mx, my + cr), (mx, my - cr)]
        occ = [self._cell_occupied(px, py, sm) for px, py in probes]
        if occ[0]:
            return True
        if sum(1 for o in occ[1:] if o) >= 2:
            return True
        return self._disk_rejects(mx, my, sm)


class AreaFilter:
    """ROI include/exclude zones (Spencer IRI style).

    track_margin_m adds hysteresis: new tracks are only born inside the zones,
    but an existing track is kept (and keeps receiving measurements) until it
    is more than track_margin_m outside, so walking along the border does not
    break its ID.
    """

    def __init__(self, zones: List[dict], area_frame: str, track_margin_m: float = 0.0) -> None:
        self.zones = zones
        self.frame = area_frame
        self.track_margin_m = max(0.0, float(track_margin_m))

    def filter_tracks(self, tracks, tracking_frame: str, tf_helper: TFHelper) -> None:
        """Remove tracks outside allowed zones (+ track_margin_m) (in-place)."""
        if not self.zones:
            return
        to_remove = []
        for i, tr in enumerate(tracks):
            tx, ty = tr.state()
            if tracking_frame == self.frame:
                pt = (tx, ty)
            else:
                pt = tf_helper.tracking_to_map(tx, ty, tracking_frame, self.frame)
            if pt is not None and not self._passes(pt[0], pt[1], self.track_margin_m):
                to_remove.append(i)
        for i in reversed(to_remove):
            tracks.pop(i)

    def filter_measurements(
        self,
        clusters: List[LidarMeasurement],
        tracking_frame: str,
        tf_helper: TFHelper,
        margin_m: float = 0.0,
    ) -> List[LidarMeasurement]:
        """Drop LiDAR clusters whose centre lies outside allowed zones (+ margin_m)."""
        if not self.zones or not clusters:
            return clusters
        return [
            c for c in clusters
            if self.passes_xy(float(c.map_x), float(c.map_y), tracking_frame, tf_helper, margin_m)
        ]

    def passes_xy(
        self, x: float, y: float, tracking_frame: str, tf_helper: TFHelper,
        margin_m: float = 0.0,
    ) -> bool:
        """True if (x,y) in tracking_frame is inside the configured ROI."""
        if not self.zones:
            return True
        if tracking_frame == self.frame:
            pt = (x, y)
        else:
            pt = tf_helper.tracking_to_map(x, y, tracking_frame, self.frame)
        if pt is None:
            return False
        return self._passes(pt[0], pt[1], margin_m)

    def _passes(self, mx, my, margin_m: float = 0.0) -> bool:
        """Inside an include zone grown by margin_m and not inside an exclude zone."""
        add = exclude = False
        for z in self.zones:
            m = margin_m if z["add"] else 0.0
            if _in_zone(z, mx, my, m):
                if z["add"]:
                    add = True
                else:
                    exclude = True
        return add and not exclude


def _in_zone(z: dict, x: float, y: float, margin_m: float) -> bool:
    if "polygon" not in z:
        return (z["x_min"] - margin_m < x < z["x_max"] + margin_m
                and z["y_min"] - margin_m < y < z["y_max"] + margin_m)
    poly = z["polygon"]
    if point_in_polygon(poly, x, y):
        return True
    return margin_m > 0.0 and _dist_to_polygon_edges(poly, x, y) < margin_m


def point_in_polygon(poly: List[Tuple[float, float]], x: float, y: float) -> bool:
    """Even-odd ray casting; poly is open (last vertex connects to the first)."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _dist_to_polygon_edges(poly: List[Tuple[float, float]], x: float, y: float) -> float:
    best = math.inf
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        t = 0.0 if l2 == 0.0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / l2))
        best = min(best, math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)))
    return best


def parse_zone(z: dict) -> dict:
    """Normalise one zone dict: rectangle (x_min/x_max/y_min/y_max) or polygon [[x,y],...]."""
    out = {"add": bool(z.get("add", True)), "name": str(z.get("name", ""))}
    if "polygon" in z:
        poly = [(float(p[0]), float(p[1])) for p in z["polygon"]]
        if len(poly) > 3 and poly[0] == poly[-1]:
            poly = poly[:-1]
        if len(poly) < 3:
            raise ValueError(f"zone '{out['name']}': polygon needs >= 3 points")
        out["polygon"] = poly
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        out.update(x_min=min(xs), x_max=max(xs), y_min=min(ys), y_max=max(ys))
    else:
        out.update(
            x_min=float(z["x_min"]), x_max=float(z["x_max"]),
            y_min=float(z["y_min"]), y_max=float(z["y_max"]),
        )
    return out


def load_zones_file(path: str) -> Tuple[Optional[str], List[dict]]:
    """Read a zones YAML (see config/zones/). Returns (frame or None, zones)."""
    import yaml

    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    frame = data.get("frame")
    zones = [parse_zone(z) for z in data.get("zones", [])]
    return (str(frame).strip() if frame else None), zones
