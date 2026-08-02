#!/usr/bin/env python3

import argparse
import glob
import math
import os
import tempfile
import json
import re
import urllib.error
import urllib.request
from datetime import timedelta

import cv2
import numpy as np

from common import discover_input_videos, extract_gps, fit_frame, get_screen_resolution, load_gps

TILE_SIZE = 256
EARTH_M_PER_DEG_LAT = 110540.0
EARTH_M_PER_DEG_LON_AT_EQ = 111320.0
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "DashcamViewMapRenderer/1.0"
MIN_ZOOM = 8
MAX_ZOOM = 19
DEFAULT_ZOOM = 16
OUTPUT_NAME = "map.m4v"
ROAD_DATA_ZOOM = 15
MIN_ROAD_FETCH_ZOOM = 9
KMH_TO_MPH = 0.621371
TYPICAL_LANE_WIDTH_M = 3.65


def discover_pbf_files(base_dir, pbf_arg=None):
    """Find one or more .pbf files from arg path or default ./pbf folder."""
    if pbf_arg:
        p = os.path.expanduser(pbf_arg)
        if os.path.isdir(p):
            return sorted(glob.glob(os.path.join(p, "*.pbf")))
        if os.path.isfile(p) and p.lower().endswith(".pbf"):
            return [p]
        return []

    return sorted(glob.glob(os.path.join(base_dir, "pbf", "*.pbf")))


class OSMRoadCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.mem = {}
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, z, x, y):
        return os.path.join(self.cache_dir, str(z), str(x), f"{y}.json")

    def get_roads_near(self, lat, lon, zoom, heading_deg):
        fetch_zoom = max(MIN_ROAD_FETCH_ZOOM, min(ROAD_DATA_ZOOM, int(zoom)))
        tx, ty = latlon_to_tile_xy(lat, lon, fetch_zoom)
        if fetch_zoom <= 10:
            ring = 3
        elif fetch_zoom <= 12:
            ring = 2
        else:
            ring = 1

        tile_coords = set()
        for oy in range(-ring, ring + 1):
            for ox in range(-ring, ring + 1):
                tile_coords.add((tx + ox, ty + oy))

        h = math.radians(float(heading_deg))
        fwd_x = math.sin(h)
        fwd_y = -math.cos(h)
        right_x = math.cos(h)
        right_y = math.sin(h)

        ahead_steps = 6 if fetch_zoom <= 10 else 4
        side_steps = 2 if fetch_zoom <= 11 else 1
        for step in range(1, ahead_steps + 1):
            for side in range(-side_steps, side_steps + 1):
                fx = tx + int(round(fwd_x * step + right_x * side))
                fy = ty + int(round(fwd_y * step + right_y * side))
                tile_coords.add((fx, fy))

        ways = []
        for cx, cy in tile_coords:
            ways.extend(self._get_tile_roads(fetch_zoom, cx, cy))
        return ways

    def _normalize_cached_roads(self, rows):
        roads = []
        if not isinstance(rows, list):
            return roads
        for item in rows:
            if isinstance(item, dict):
                pts = item.get("points", [])
                if isinstance(pts, list) and len(pts) >= 2:
                    roads.append({
                        "points": pts,
                        "name": item.get("name"),
                        "ref": item.get("ref"),
                        "highway": item.get("highway"),
                        "maxspeed": item.get("maxspeed"),
                        "lanes": item.get("lanes"),
                        "lanes_forward": item.get("lanes_forward") or item.get("lanes:forward"),
                        "lanes_backward": item.get("lanes_backward") or item.get("lanes:backward"),
                    })
                continue
            if isinstance(item, list) and len(item) >= 2:
                roads.append({"points": item, "name": None, "ref": None,
                               "highway": None, "maxspeed": None, "lanes": None,
                               "lanes_forward": None, "lanes_backward": None})
        return roads

    def _get_tile_roads(self, z, x, y):
        n_tiles = 1 << z
        x_wrapped = x % n_tiles
        if y < 0 or y >= n_tiles:
            return []
        key = (z, x_wrapped, y)
        if key in self.mem:
            return self.mem[key]
        path = self._cache_path(z, x_wrapped, y)
        ways = None
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    ways = self._normalize_cached_roads(json.load(handle))
            except (OSError, json.JSONDecodeError):
                ways = None
        if ways is None:
            ways = self._download_tile_roads(z, x_wrapped, y)
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(ways, handle, separators=(",", ":"))
            except OSError:
                pass
        self.mem[key] = ways
        return ways

    def _download_tile_roads(self, z, x, y):
        south, west, north, east = tile_xy_to_bbox(x, y, z)
        query = (
            "[out:json][timeout:40];"
            f"(way[\"highway\"][\"area\"!~\"yes\"]({south},{west},{north},{east}););"
            "out tags geom;"
        )
        body = query.encode("utf-8")
        req = urllib.request.Request(
            OVERPASS_URL,
            data=body,
            headers={"User-Agent": USER_AGENT,
                     "Content-Type": "text/plain; charset=utf-8"},
            method="POST",
        )
        doc = None
        for _ in range(2):
            try:
                with urllib.request.urlopen(req, timeout=20.0) as resp:
                    payload = resp.read().decode("utf-8", errors="replace")
                doc = json.loads(payload)
                break
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
                doc = None
        if doc is None:
            return []
        ways = []
        for elem in doc.get("elements", []):
            if elem.get("type") != "way":
                continue
            geom = elem.get("geometry", [])
            if len(geom) < 2:
                continue
            pts = []
            for p in geom:
                lat = p.get("lat")
                lon = p.get("lon")
                if lat is None or lon is None:
                    continue
                pts.append([float(lat), float(lon)])
            if len(pts) >= 2:
                tags = elem.get("tags", {})
                ways.append({
                    "points": pts,
                    "name": tags.get("name"),
                    "ref": tags.get("ref"),
                    "highway": tags.get("highway"),
                    "maxspeed": tags.get("maxspeed"),
                    "lanes": tags.get("lanes"),
                    "lanes_forward": tags.get("lanes:forward"),
                    "lanes_backward": tags.get("lanes:backward"),
                })
        return ways


class PBFRoadCache(OSMRoadCache):
    """Loads road data from a local .osm.pbf file (e.g. from Geofabrik).
    Indexes once into the tile cache on first run; subsequent runs are instant.
    Install dependency: pip install osmium
    Download data: https://download.geofabrik.de/europe/great-britain/england/
    """

    def __init__(self, cache_dir, pbf_paths):
        super().__init__(cache_dir)
        self.pbf_paths = [os.path.abspath(p) for p in pbf_paths]
        self._index_pbf_if_needed()

    def _sentinel_path(self):
        return os.path.join(self.cache_dir, "pbf_indexed.flag")

    def _index_pbf_if_needed(self):
        sentinel = self._sentinel_path()
        expected = "\n".join(self.pbf_paths)
        if os.path.exists(sentinel):
            try:
                with open(sentinel, "r", encoding="utf-8") as fh:
                    indexed_path = fh.read().strip()
                if indexed_path == expected:
                    return
            except OSError:
                pass
        print("Indexing PBF file(s):")
        for p in self.pbf_paths:
            print(f"  - {p}")
        print("This runs once and may take a few minutes depending on file count/size...")
        count = self._run_pbf_index()
        print(f"Indexed {count} road ways into tile cache.")
        try:
            with open(sentinel, "w", encoding="utf-8") as fh:
                fh.write(expected + "\n")
        except OSError:
            pass

    def _run_pbf_index(self):
        try:
            import osmium
        except ImportError:
            print("ERROR: pyosmium is not installed.")
            print("  Install with:  pip install osmium")
            return 0

        tile_buckets = {}

        class HighwayHandler(osmium.SimpleHandler):
            def way(self2, w):  # noqa: N805
                if "highway" not in w.tags:
                    return
                if w.tags.get("area") == "yes":
                    return
                pts = []
                for n in w.nodes:
                    if n.location.valid():
                        pts.append([n.location.lat, n.location.lon])
                if len(pts) < 2:
                    return
                way_data = {
                    "points": pts,
                    "name": w.tags.get("name"),
                    "ref": w.tags.get("ref"),
                    "highway": w.tags.get("highway"),
                    "maxspeed": w.tags.get("maxspeed"),
                    "lanes": w.tags.get("lanes"),
                    "lanes_forward": w.tags.get("lanes:forward"),
                    "lanes_backward": w.tags.get("lanes:backward"),
                }
                seen_tiles = set()
                for lat, lon in pts:
                    tx, ty = latlon_to_tile_xy(lat, lon, ROAD_DATA_ZOOM)
                    seen_tiles.add((tx, ty))
                for tx, ty in seen_tiles:
                    key = (ROAD_DATA_ZOOM, tx, ty)
                    if key not in tile_buckets:
                        tile_buckets[key] = []
                    tile_buckets[key].append(way_data)

        handler = HighwayHandler()
        for pbf_path in self.pbf_paths:
            handler.apply_file(pbf_path, locations=True)

        for (z, x, y), ways in tile_buckets.items():
            path = self._cache_path(z, x, y)
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(ways, fh, separators=(",", ":"))
            except OSError:
                pass
            self.mem[(z, x, y)] = ways

        return sum(len(v) for v in tile_buckets.values())

    def get_roads_near(self, lat, lon, zoom, heading_deg):
        # PBF index is stored at ROAD_DATA_ZOOM; always query that zoom.
        fetch_zoom = ROAD_DATA_ZOOM
        tx, ty = latlon_to_tile_xy(lat, lon, fetch_zoom)

        ring = 2
        tile_coords = set()
        for oy in range(-ring, ring + 1):
            for ox in range(-ring, ring + 1):
                tile_coords.add((tx + ox, ty + oy))

        h = math.radians(float(heading_deg))
        fwd_x = math.sin(h)
        fwd_y = -math.cos(h)
        right_x = math.cos(h)
        right_y = math.sin(h)

        ahead_steps = 8
        side_steps = 2
        for step in range(1, ahead_steps + 1):
            for side in range(-side_steps, side_steps + 1):
                fx = tx + int(round(fwd_x * step + right_x * side))
                fy = ty + int(round(fwd_y * step + right_y * side))
                tile_coords.add((fx, fy))

        ways = []
        for cx, cy in tile_coords:
            ways.extend(self._get_tile_roads(fetch_zoom, cx, cy))
        return ways

    def _download_tile_roads(self, z, x, y):
        # PBF is the sole source; never fall back to network.
        return []


class HybridRoadCache:
    """Use primary cache first, then fallback cache if no roads are found."""

    def __init__(self, primary_cache, fallback_cache):
        self.primary_cache = primary_cache
        self.fallback_cache = fallback_cache

    def get_roads_near(self, lat, lon, zoom, heading_deg):
        roads = self.primary_cache.get_roads_near(lat, lon, zoom, heading_deg)
        if roads:
            return roads
        return self.fallback_cache.get_roads_near(lat, lon, zoom, heading_deg)


def clamp_lat(lat):
    return max(-85.05112878, min(85.05112878, lat))


def latlon_to_world_px(lat, lon, zoom):
    lat = clamp_lat(float(lat))
    lon = float(lon)
    n = float(1 << zoom) * TILE_SIZE
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) * 0.5 * n
    return x, y


def latlon_to_tile_xy(lat, lon, zoom):
    world_x, world_y = latlon_to_world_px(lat, lon, zoom)
    tx = int(world_x // TILE_SIZE)
    ty = int(world_y // TILE_SIZE)
    return tx, ty


def tile_xy_to_bbox(x, y, z):
    n = 1 << z
    west = (x / n) * 360.0 - 180.0
    east = ((x + 1) / n) * 360.0 - 180.0

    north_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    south_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n)))
    north = math.degrees(north_rad)
    south = math.degrees(south_rad)
    return south, west, north, east


def meters_per_pixel(lat, zoom):
    return 156543.03392 * math.cos(math.radians(clamp_lat(lat))) / float(1 << zoom)


def project_point_to_screen(lat, lon, anchor_lat, anchor_lon, heading_deg, px_per_meter, car_px):
    lat_mid = math.radians((anchor_lat + lat) * 0.5)
    dx_east = (lon - anchor_lon) * EARTH_M_PER_DEG_LON_AT_EQ * math.cos(lat_mid)
    dy_north = (lat - anchor_lat) * EARTH_M_PER_DEG_LAT

    h = math.radians(heading_deg)
    right_x = math.cos(h)
    right_y = -math.sin(h)
    fwd_x = math.sin(h)
    fwd_y = math.cos(h)

    rel_right = dx_east * right_x + dy_north * right_y
    rel_fwd = dx_east * fwd_x + dy_north * fwd_y

    sx = int(round(car_px[0] + rel_right * px_per_meter))
    sy = int(round(car_px[1] - rel_fwd * px_per_meter))
    return sx, sy


def latlon_to_local_xy_m(lat, lon, anchor_lat, anchor_lon):
    lat_mid = math.radians((anchor_lat + lat) * 0.5)
    dx_east = (lon - anchor_lon) * EARTH_M_PER_DEG_LON_AT_EQ * math.cos(lat_mid)
    dy_north = (lat - anchor_lat) * EARTH_M_PER_DEG_LAT
    return dx_east, dy_north


def point_to_segment_distance_m(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = (abx * abx) + (aby * aby)
    if denom <= 1e-9:
        return math.hypot(apx, apy)
    t = max(0.0, min(1.0, ((apx * abx) + (apy * aby)) / denom))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def nearest_road_info(lat, lon, roads):
    best = None
    best_dist = float("inf")

    for road in roads:
        pts = road.get("points", [])
        if len(pts) < 2:
            continue

        local = [latlon_to_local_xy_m(p[0], p[1], lat, lon) for p in pts]
        for i in range(len(local) - 1):
            ax, ay = local[i]
            bx, by = local[i + 1]
            d = point_to_segment_distance_m(0.0, 0.0, ax, ay, bx, by)
            if d < best_dist:
                best_dist = d
                best = road

    if best is None:
        return None
    return {
        "distance_m": best_dist,
        "name": best.get("name"),
        "ref": best.get("ref"),
        "highway": best.get("highway"),
        "maxspeed": best.get("maxspeed"),
        "lanes": best.get("lanes"),
        "lanes_forward": best.get("lanes_forward"),
        "lanes_backward": best.get("lanes_backward"),
    }


def format_road_info(road_info):
    if road_info is None:
        return "ROAD: unknown", "LANES: unknown"

    dist = road_info["distance_m"]
    if dist > 80.0:
        return "ROAD: unknown", "LANES: unknown"

    name = road_info.get("name")
    ref = road_info.get("ref")
    hwy = road_info.get("highway")
    speed = road_info.get("maxspeed")
    lanes_total, lanes_text = lane_info_for_display(
        road_info.get("lanes"),
        road_info.get("lanes_forward"),
        road_info.get("lanes_backward"),
    )

    road_label = name or ref or (hwy.replace("_", " ") if hwy else "unknown")
    if ref and name:
        road_label = f"{name} ({ref})"

    lanes_label = lanes_text if lanes_total is not None else "unknown"
    return f"ROAD: {road_label}", f"LANES: {lanes_label}"


def lane_info_for_display(lanes_value, lanes_forward, lanes_backward):
    total = parse_lane_count(lanes_value)
    fwd = parse_lane_count(lanes_forward)
    bwd = parse_lane_count(lanes_backward)

    if total is None and (fwd is not None or bwd is not None):
        total = (fwd or 0) + (bwd or 0)

    if total is None:
        return None, "unknown"

    if fwd is not None or bwd is not None:
        parts = []
        if fwd is not None:
            parts.append(f"F{fwd}")
        if bwd is not None:
            parts.append(f"B{bwd}")
        if parts:
            return total, f"{total} ({'/'.join(parts)})"

    return total, str(total)


def parse_lane_count(lanes_value):
    if lanes_value is None:
        return None
    s = str(lanes_value).strip()
    if not s:
        return None
    m = re.search(r"\d+", s)
    if m is None:
        return None
    try:
        return max(1, int(m.group(0)))
    except ValueError:
        return None


def road_line_thickness(road, px_per_meter, use_typical_lane_width=False):
    base_by_highway = {
        "motorway": 10,
        "trunk": 9,
        "primary": 8,
        "secondary": 7,
        "tertiary": 6,
        "residential": 5,
        "service": 4,
        "living_street": 4,
        "unclassified": 5,
    }
    hwy = str(road.get("highway") or "").strip().lower()
    base = base_by_highway.get(hwy, 3)

    lanes, _ = lane_info_for_display(
        road.get("lanes"),
        road.get("lanes_forward"),
        road.get("lanes_backward"),
    )
    if lanes is not None:
        if use_typical_lane_width:
            # Width from typical lane width projected into pixels.
            lane_based = int(round(lanes * TYPICAL_LANE_WIDTH_M * px_per_meter))
        else:
            # Use 4 pixels per lane when lane count is known.
            lane_based = int(lanes) * 4
        base = max(base, lane_based)
    return max(3, min(20, base))


def parse_speed_limit_mph(maxspeed_value):
    if maxspeed_value is None:
        return None

    s = str(maxspeed_value).strip().lower()
    if not s:
        return None

    first = s.split(";")[0].strip()
    m = re.search(r"\d+(?:\.\d+)?", first)
    if m is None:
        return None

    val = float(m.group(0))
    if "km" in first or "kph" in first:
        return val * KMH_TO_MPH
    if "mph" in first:
        return val
    # In UK OSM data, plain numeric maxspeed is typically mph.
    return val


def draw_speedlimit_widget(frame, speed_limit_mph, current_mph, visible):
    if not visible:
        return

    h, w = frame.shape[:2]
    center = (w - 95, h - 120)
    radius = 58

    cv2.circle(frame, center, radius, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, center, radius, (30, 30, 220), 9, cv2.LINE_AA)

    limit_text = "--" if speed_limit_mph is None else str(int(round(speed_limit_mph)))
    (tw, th), _ = cv2.getTextSize(limit_text, cv2.FONT_HERSHEY_DUPLEX, 1.25, 3)
    cv2.putText(
        frame,
        limit_text,
        (center[0] - tw // 2, center[1] + th // 3),
        cv2.FONT_HERSHEY_DUPLEX,
        1.25,
        (30, 30, 30),
        3,
        cv2.LINE_AA,
    )

    current_txt = f"{current_mph:4.1f} mph"
    (sw, _), _ = cv2.getTextSize(current_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)
    y = center[1] + radius + 30
    cv2.rectangle(frame, (center[0] - sw // 2 - 8, y - 24), (center[0] + sw // 2 + 8, y + 8), (0, 0, 0), -1)
    cv2.putText(frame, current_txt, (center[0] - sw // 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (235, 235, 235), 2, cv2.LINE_AA)


def road_line_color(road):
    """Return BGR center color by OSM highway type."""
    hwy = str(road.get("highway") or "").strip().lower()
    color_by_highway = {
        "motorway": (255, 80, 40),      # blue
        "trunk": (245, 190, 70),
        "primary": (90, 220, 250),
        "secondary": (120, 200, 140),
        "tertiary": (170, 170, 220),
        "residential": (205, 205, 205),
        "service": (165, 165, 165),
        "living_street": (165, 165, 165),
        "unclassified": (190, 190, 190),
    }
    return color_by_highway.get(hwy, (200, 200, 200))


def draw_km_scale_bar(frame, mpp):
    h, w = frame.shape[:2]
    max_bar_px = min(220, int(w * 0.24))
    if max_bar_px < 80:
        return

    max_km = (max_bar_px * mpp) / 1000.0
    if max_km <= 0:
        return

    candidates_km = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    chosen_km = candidates_km[0]
    for v in candidates_km:
        if v <= max_km:
            chosen_km = v
        else:
            break

    bar_px = int(round((chosen_km * 1000.0) / mpp))
    if bar_px < 40:
        return

    x0 = 24
    y0 = h - 34
    x1 = x0 + bar_px
    if x1 >= w - 12:
        x1 = w - 12

    cv2.rectangle(frame, (x0 - 8, y0 - 28), (x1 + 8, y0 + 10), (0, 0, 0), -1)
    cv2.line(frame, (x0, y0), (x1, y0), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(frame, (x0, y0 - 6), (x0, y0 + 6), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (x1, y0 - 6), (x1, y0 + 6), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{chosen_km:g} km", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)


def interp_angle_deg(a0, a1, frac):
    d = (a1 - a0 + 180.0) % 360.0 - 180.0
    return (a0 + frac * d) % 360.0


def safe_float(v, default=0.0):
    try:
        f = float(v)
        if math.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def parse_coordinate(v, default=0.0):
    """Parse decimal or DMS coordinates like: 51 deg 47' 49.92" N."""
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default

    try:
        f = float(s)
        if math.isnan(f):
            return default
        return f
    except ValueError:
        pass

    m = re.match(
        r"^\s*(\d+(?:\.\d+)?)\s*deg\s*(\d+(?:\.\d+)?)'\s*(\d+(?:\.\d+)?)\"\s*([NSEW])\s*$",
        s,
        flags=re.IGNORECASE,
    )
    if not m:
        return default

    deg = float(m.group(1))
    minutes = float(m.group(2))
    seconds = float(m.group(3))
    hemi = m.group(4).upper()

    val = deg + (minutes / 60.0) + (seconds / 3600.0)
    if hemi in ("S", "W"):
        val = -val
    return val


def gps_state_at_time(df, t_s, last_heading_deg):
    if df is None or len(df) == 0:
        return None

    i0 = min(int(t_s), len(df) - 1)
    i1 = min(i0 + 1, len(df) - 1)
    frac = t_s - int(t_s)

    r0 = df.iloc[i0]
    r1 = df.iloc[i1]

    lat0 = parse_coordinate(r0["Lat"])
    lat1 = parse_coordinate(r1["Lat"], lat0)
    lon0 = parse_coordinate(r0["Lon"])
    lon1 = parse_coordinate(r1["Lon"], lon0)
    spd0 = safe_float(r0["Speed"])
    spd1 = safe_float(r1["Speed"], spd0)

    trk0 = safe_float(r0["Track"], last_heading_deg)
    trk1 = safe_float(r1["Track"], trk0)

    lat = lat0 + frac * (lat1 - lat0)
    lon = lon0 + frac * (lon1 - lon0)
    spd = spd0 + frac * (spd1 - spd0)
    heading = interp_angle_deg(trk0, trk1, frac)

    return {
        "lat": lat,
        "lon": lon,
        "speed_kmh": spd,
        "heading_deg": heading,
    }


def render_heading_up_map(state, zoom, out_w, out_h, trail_latlon, road_cache, use_typical_lane_width=False):
    lat = state["lat"]
    lon = state["lon"]
    heading = state["heading_deg"]

    car_px = (out_w // 2, int(out_h * 0.68))

    frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    frame[:] = (18, 18, 18)

    mpp = meters_per_pixel(lat, zoom)
    px_per_meter = 1.0 / max(mpp, 1e-6)

    # Local oriented grid helps read scale when roads are sparse.
    grid_step_m = 20.0
    max_extent_px = int(max(out_w, out_h) * 0.8)
    n_steps = int((max_extent_px / px_per_meter) / grid_step_m) + 1
    for i in range(-n_steps, n_steps + 1):
        offset_m = i * grid_step_m
        p1 = project_point_to_screen(
            lat + (1000.0 / EARTH_M_PER_DEG_LAT),
            lon + (offset_m / (EARTH_M_PER_DEG_LON_AT_EQ * max(math.cos(math.radians(lat)), 1e-6))),
            lat,
            lon,
            heading,
            px_per_meter,
            car_px,
        )
        p2 = project_point_to_screen(
            lat - (1000.0 / EARTH_M_PER_DEG_LAT),
            lon + (offset_m / (EARTH_M_PER_DEG_LON_AT_EQ * max(math.cos(math.radians(lat)), 1e-6))),
            lat,
            lon,
            heading,
            px_per_meter,
            car_px,
        )
        cv2.line(frame, p1, p2, (30, 30, 30), 1, cv2.LINE_AA)

    roads = road_cache.get_roads_near(lat, lon, zoom, heading)
    for way in roads:
        points = way.get("points", [])
        pts = []
        for wlat, wlon in points:
            pts.append(project_point_to_screen(wlat, wlon, lat, lon, heading, px_per_meter, car_px))
        if len(pts) >= 2:
            poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            thickness = road_line_thickness(way, px_per_meter, use_typical_lane_width=use_typical_lane_width)
            center_color = road_line_color(way)
            # Two-pass stroke makes width differences easier to see.
            cv2.polylines(frame, [poly], False, (45, 45, 45), thickness + 2, cv2.LINE_AA)
            cv2.polylines(frame, [poly], False, center_color, thickness, cv2.LINE_AA)

    if len(trail_latlon) >= 2:
        pts = []
        for tlat, tlon in trail_latlon:
            pts.append(project_point_to_screen(tlat, tlon, lat, lon, heading, px_per_meter, car_px))

        if len(pts) >= 2:
            poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [poly], False, (0, 160, 255), 2, cv2.LINE_AA)

    draw_km_scale_bar(frame, mpp)
    draw_vehicle_icon(frame, car_px)
    return frame, nearest_road_info(lat, lon, roads), len(roads)


def draw_vehicle_icon(frame, center):
    cx, cy = center
    # Arrow points upward in screen-space; map is rotated to make travel direction up.
    arrow = np.array(
        [
            [cx, cy - 16],
            [cx - 10, cy + 12],
            [cx + 10, cy + 12],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, arrow, (50, 60, 255), cv2.LINE_AA)
    cv2.circle(frame, (cx, cy + 16), 6, (220, 220, 220), -1, cv2.LINE_AA)


def draw_hud(frame, video_label, gps_time_text, elapsed_text, speed_kmh, zoom, road_text, lane_text, road_count, use_typical_lane_width, show_speed_sign):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 190), (0, 0, 0), -1)
    cv2.putText(frame, f"{video_label}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(frame, f"GPS: {gps_time_text}  ELAPSED: {elapsed_text}", (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (190, 190, 190), 2, cv2.LINE_AA)
    cv2.putText(frame, f"SPEED: {speed_kmh:5.1f} km/h  ZOOM: {zoom}  ROADS: {road_count}", (16, 104), cv2.FONT_HERSHEY_DUPLEX, 0.9, (100, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, road_text, (16, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (220, 220, 130), 2, cv2.LINE_AA)
    cv2.putText(frame, lane_text, (w - 420, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (220, 220, 130), 2, cv2.LINE_AA)
    mode_txt = f"T lane-width: {'3.65m' if use_typical_lane_width else '4px/lane'}   O speed-sign: {'ON' if show_speed_sign else 'OFF'}"
    cv2.putText(frame, mode_txt, (16, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (200, 200, 200), 2, cv2.LINE_AA)
    # cv2.putText(frame, "OSM raw highways", (w - 300, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 220, 220), 2, cv2.LINE_AA)


def play_video_map(video_path, total_files, idx, writer, out_w, out_h, screen_w, screen_h, road_cache, zoom_start):
    print(f"[{idx}/{total_files}] Loading: {os.path.basename(video_path)}")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        csv_path = tmp.name
    try:
        extract_gps(video_path, csv_path)
        df, start_dt = load_gps(csv_path)
    finally:
        try:
            os.unlink(csv_path)
        except OSError:
            pass

    if df is None or len(df) == 0:
        print(f"No GPS data in: {os.path.basename(video_path)}")
        return True, zoom_start

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = max(1, int(1000 / fps))

    if out_w <= 0 or out_h <= 0:
        print("Invalid output dimensions; skipping video.")
        cap.release()
        return True, zoom_start

    win = "GPS Map (Heading Up)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    zoom = int(zoom_start)
    trail = []
    last_heading = 0.0
    warned_no_roads = False
    use_typical_lane_width = False
    show_speed_sign = True

    while cap.isOpened():
        ret, _ = cap.read()
        if not ret:
            break

        frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        t_s = frame_idx / fps

        gps_state = gps_state_at_time(df, t_s, last_heading)
        if gps_state is None:
            continue
        last_heading = gps_state["heading_deg"]

        trail.append((gps_state["lat"], gps_state["lon"]))
        if len(trail) > 600:
            trail = trail[-600:]

        frame, road_info, road_count = render_heading_up_map(
            gps_state,
            zoom,
            out_w,
            out_h,
            trail,
            road_cache,
            use_typical_lane_width=use_typical_lane_width,
        )
        road_text, lane_text = format_road_info(road_info)
        speed_limit_mph = parse_speed_limit_mph(road_info.get("maxspeed") if road_info is not None else None)
        current_mph = gps_state["speed_kmh"] * KMH_TO_MPH

        if road_count == 0 and not warned_no_roads:
            print(
                "Warning: no road data found near "
                f"lat={gps_state['lat']:.6f}, lon={gps_state['lon']:.6f}."
            )
            warned_no_roads = True

        total_ms = t_s * 1000.0
        hh = int(total_ms / 3600000)
        mm = int((total_ms / 60000) % 60)
        ss = int((total_ms / 1000) % 60)
        ms = int(total_ms % 1000)
        elapsed = f"{hh:02}:{mm:02}:{ss:02}.{ms:03}"

        if start_dt is not None:
            gps_dt = start_dt + timedelta(seconds=t_s)
            gps_time_text = gps_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        else:
            gps_time_text = "N/A"

        label = f"{idx}/{total_files}: {os.path.basename(video_path)}"
        draw_hud(
            frame,
            label,
            gps_time_text,
            elapsed,
            gps_state["speed_kmh"],
            zoom,
            road_text,
            lane_text,
            road_count,
            use_typical_lane_width,
            show_speed_sign,
        )
        draw_speedlimit_widget(frame, speed_limit_mph, current_mph, show_speed_sign)

        writer.write(frame)

        disp = fit_frame(frame, screen_w, screen_h)
        cv2.imshow(win, disp)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            cap.release()
            return False, zoom
        if key == ord("n"):
            break
        if key in (ord("+"), ord("=")):
            zoom = min(MAX_ZOOM, zoom + 1)
        if key in (ord("-"), ord("_")):
            zoom = max(MIN_ZOOM, zoom - 1)
        if key == ord("t"):
            use_typical_lane_width = not use_typical_lane_width
            print(f"Lane width mode: {'3.65m per lane' if use_typical_lane_width else '4px per lane'}")
        if key == ord("o"):
            show_speed_sign = not show_speed_sign
            print(f"Speed-limit sign: {'ON' if show_speed_sign else 'OFF'}")

    cap.release()
    return True, zoom


def main():
    parser = argparse.ArgumentParser(description="GPS map renderer for dashcam footage.")
    parser.add_argument(
        "--pbf",
        metavar="FILE_OR_DIR",
        default=None,
        help=(
            "Path to a local .osm.pbf file OR folder containing .pbf files. "
            "If omitted, auto-loads all .pbf files from ./pbf/. Download England: "
            "https://download.geofabrik.de/europe/great-britain/england-latest.osm.pbf"
        ),
    )
    args = parser.parse_args()

    cwd = os.getcwd()
    videos = discover_input_videos(cwd)
    if not videos:
        print("No source MP4 files found. Place videos in current directory or camera/.")
        return

    probe = cv2.VideoCapture(videos[0])
    out_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    probe.release()

    out_path = os.path.join(cwd, OUTPUT_NAME)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"Failed to open output writer: {out_path}")
        return

    pbf_files = discover_pbf_files(cwd, args.pbf)
    if pbf_files:
        pbf_cache = PBFRoadCache(os.path.join(cwd, ".tile_cache", "pbf_raw"), pbf_files)
        overpass_cache = OSMRoadCache(os.path.join(cwd, ".tile_cache", "osm_raw"))
        cache = HybridRoadCache(pbf_cache, overpass_cache)
        data_source = f"PBF files: {len(pbf_files)} + Overpass fallback"
        for p in pbf_files:
            print(f"Using PBF: {p}")
    elif args.pbf:
        print(f"ERROR: No .pbf files found from: {args.pbf}")
        print("Place files in ./pbf/ or pass --pbf <file_or_folder>")
        return
    else:
        cache = OSMRoadCache(os.path.join(cwd, ".tile_cache", "osm_raw"))
        data_source = "Overpass API (live)"
    screen_w, screen_h = get_screen_resolution()

    print(f"Found {len(videos)} MP4 file(s).")
    print(f"Road data source: {data_source}")
    print("Controls: [Q] Quit  [N] Next video  [+] Zoom In  [-] Zoom Out  [T] Lane Width Mode  [O] Speed Sign")
    print("Rendering heading-up OpenStreetMap roads from raw OSM data...")

    zoom = DEFAULT_ZOOM
    try:
        for i, vp in enumerate(videos, start=1):
            keep_running, zoom = play_video_map(vp, len(videos), i, writer, out_w, out_h, screen_w, screen_h, cache, zoom)
            if not keep_running:
                break
    finally:
        writer.release()
        print(f"Saved: {out_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
