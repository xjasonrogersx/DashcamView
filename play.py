import cv2
import pandas as pd
import subprocess
import os
import glob
import tempfile
import argparse
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from scipy.special import softmax as scipy_softmax
from ensemble_boxes import weighted_boxes_fusion
import onnxruntime as ort
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# COCO class IDs for vehicles
VEHICLE_CLASSES = {2: "CAR", 3: "MOTO", 5: "BUS", 7: "TRUCK"}

# Assumed real-world heights in metres per class (for distance estimation)
VEHICLE_HEIGHT_M = {2: 1.5, 3: 1.2, 5: 3.0, 7: 2.5}

# Bounding-box colours are now dynamic — see _box_color()

# Speed history window (frames) for smoothing
SPEED_HISTORY = 10

KMH_TO_MPH = 0.621371

def get_screen_resolution():
    """Get screen resolution using xrandr or fallback."""
    try:
        result = subprocess.run(["xrandr"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if " connected primary" in line or (" connected" in line and "*" in result.stdout):
                parts = line.split()
                for part in parts:
                    if "x" in part and part[0].isdigit():
                        w, h = part.split("x")
                        return int(w), int(h)
    except Exception:
        pass
    return 1920, 1080  # fallback

class VehicleTracker:
    """IoU-based centroid tracker that maintains persistent vehicle IDs."""

    def __init__(self, max_age=10):
        self.next_id = 0
        self.max_age = max_age   # frames before a lost track is dropped
        # id → {"bbox": (x1,y1,x2,y2), "class_id": int, "age": int,
        #        "history": deque([(bbox_h_px, timestamp_s), ...])}
        self.tracks = {}

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    def update(self, detections, timestamp_s):
        """
        detections: list of (x1, y1, x2, y2, class_id)
        Returns list of (x1, y1, x2, y2, class_id, track_id)
        """
        # Age all tracks
        for tid in list(self.tracks):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]

        matched_ids = set()
        results = []

        for det in detections:
            x1, y1, x2, y2, cls_id = det
            best_tid, best_iou = None, 0.3  # IoU threshold

            for tid, track in self.tracks.items():
                if tid in matched_ids:
                    continue
                if track["class_id"] != cls_id:
                    continue
                iou = self._iou((x1, y1, x2, y2), track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                tid = best_tid
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "bbox": (x1, y1, x2, y2),
                    "class_id": cls_id,
                    "age": 0,
                    "history": deque(maxlen=SPEED_HISTORY + 2),
                }

            h_px = y2 - y1
            self.tracks[tid]["bbox"] = (x1, y1, x2, y2)
            self.tracks[tid]["age"] = 0
            self.tracks[tid]["history"].append((h_px, timestamp_s, cls_id))
            matched_ids.add(tid)
            results.append((x1, y1, x2, y2, cls_id, tid))

        return results


def _yolo_detections(result, x_offset=0, y_offset=0, conf_thresh=0.4):
    """Extract vehicle detections from a YOLO result, offsetting coords into full-frame space."""
    dets = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue
        if float(box.conf[0]) < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        dets.append((x1 + x_offset, y1 + y_offset,
                     x2 + x_offset, y2 + y_offset,
                     cls_id, float(box.conf[0])))
    return dets


def _dual_pass_raw(frame, model):
    """Run full-frame + center-crop YOLO passes, return combined raw detections."""
    fh, fw = frame.shape[:2]
    dets_full = _yolo_detections(model(frame, verbose=False)[0])
    sq   = fh
    x0   = (fw - sq) // 2
    crop = frame[0:sq, x0:x0 + sq]
    dets_crop = _yolo_detections(model(crop, verbose=False)[0], x_offset=x0)
    return dets_full + dets_crop, fw, fh


def _apply_nms(all_dets, fw, fh):
    """Standard Non-Maximum Suppression via OpenCV."""
    if not all_dets:
        return []
    boxes   = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, _, _ in all_dets]
    confs   = [c for *_, c in all_dets]
    cls_ids = [cls for _, _, _, _, cls, _ in all_dets]
    indices = cv2.dnn.NMSBoxes(boxes, confs, score_threshold=0.4, nms_threshold=0.5)
    if len(indices) == 0:
        return []
    kept = [int(i) for i in (indices.flatten() if hasattr(indices, 'flatten') else indices)]
    return [(all_dets[i][0], all_dets[i][1], all_dets[i][2], all_dets[i][3], cls_ids[i])
            for i in kept]


def _apply_nmw(all_dets, fw, fh):
    """Non-Maximum Weighted (Weighted Box Fusion) — blends overlapping boxes."""
    if not all_dets:
        return []
    # WBF needs normalised [0,1] coordinates
    boxes_norm = [[x1/fw, y1/fh, x2/fw, y2/fh] for x1, y1, x2, y2, _, _ in all_dets]
    scores     = [c for *_, c in all_dets]
    labels     = [float(cls) for _, _, _, _, cls, _ in all_dets]
    boxes_out, scores_out, labels_out = weighted_boxes_fusion(
        [boxes_norm], [scores], [labels],
        weights=[1], iou_thr=0.5, skip_box_thr=0.4
    )
    return [(int(b[0]*fw), int(b[1]*fh), int(b[2]*fw), int(b[3]*fh), int(l))
            for b, l in zip(boxes_out, labels_out)
            if int(l) in VEHICLE_CLASSES]


def init_sahi_model(model_path, conf_thresh=0.4):
    """Create a SAHI AutoDetectionModel wrapping YOLOv8."""
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf_thresh,
        device="cpu",
    )


def _detect_sahi(frame, sahi_model):
    """SAHI sliced inference — best for small/distant objects."""
    # SAHI expects RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = get_sliced_prediction(
        frame_rgb,
        sahi_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        perform_standard_pred=True,
        verbose=0,
    )
    dets = []
    for pred in result.object_prediction_list:
        cls_id = pred.category.id
        if cls_id not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = (int(v) for v in pred.bbox.to_xyxy())
        dets.append((x1, y1, x2, y2, cls_id))
    return dets


# Set by argparse in main() — controls which merge strategy is used
MERGE_MODE = "SAHI"


def detect_vehicles(frame, model, sahi_model=None):
    """
    Detect vehicles using the active MERGE_MODE:
      SAHI — sliced inference (best recall for small/distant objects, default)
      NMW  — dual-pass (full + center crop) merged with Weighted Box Fusion
      NMS  — dual-pass (full + center crop) merged with standard NMS
    Returns list of (x1, y1, x2, y2, class_id).
    """
    if MERGE_MODE == "SAHI" and sahi_model is not None:
        return _detect_sahi(frame, sahi_model)

    all_dets, fw, fh = _dual_pass_raw(frame, model)

    if MERGE_MODE == "NMW":
        return _apply_nmw(all_dets, fw, fh)
    return _apply_nms(all_dets, fw, fh)  # NMS (default fallback)


def estimate_speed(track_history, frame_w, ego_kmh):
    """
    Estimate vehicle speeds using perspective geometry.

    Returns (clamped_speed_kmh, raw_speed_kmh, avg_rel_ms) or (None, None, None).
    raw_speed_kmh can be negative (oncoming / strongly approaching).
    avg_rel_ms > 0 → moving away; < 0 → approaching.
    """
    if len(track_history) < 2:
        return None, None

    focal_px = frame_w / 2.0

    entries = list(track_history)
    rel_speeds, abs_speeds, raw_speeds = [], [], []
    for i in range(1, len(entries)):
        h1, t1, cls1 = entries[i - 1]
        h2, t2, cls2 = entries[i]
        dt = t2 - t1
        if dt <= 0 or h1 <= 0 or h2 <= 0:
            continue
        real_h = VEHICLE_HEIGHT_M.get(cls2, 1.5)
        d1 = (focal_px * real_h) / h1
        d2 = (focal_px * real_h) / h2
        rel_ms = (d2 - d1) / dt
        raw_kmh = ego_kmh + rel_ms * 3.6
        rel_speeds.append(rel_ms)
        raw_speeds.append(raw_kmh)
        abs_speeds.append(max(0.0, raw_kmh))

    if not abs_speeds:
        return None, None, None
    return (sum(abs_speeds) / len(abs_speeds),
            sum(raw_speeds) / len(raw_speeds),
            sum(rel_speeds) / len(rel_speeds))


def _box_color(diff_mph, raw_speed_mph):
    """
    Green  — delta positive (other car faster than me)
    Orange — delta negative, speed positive (slower, same direction)
    Red    — delta negative AND speed negative (oncoming / strongly approaching)
    """
    if diff_mph > 0:
        return (0, 200, 0)       # green
    if raw_speed_mph < 0:
        return (0, 0, 220)       # red
    return (0, 140, 255)         # orange


def draw_vehicles(frame, tracked, frame_w, ego_kmh):
    """Draw bounding boxes with dynamic colours, speed in mph, and differential vs ego."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    ego_mph = ego_kmh * KMH_TO_MPH
    for (x1, y1, x2, y2, cls_id, tid) in tracked:
        label_name = VEHICLE_CLASSES.get(cls_id, "VEH")
        speed_str = "---"
        diff_str  = ""
        color     = (180, 180, 180)

        if tid in _tracker_ref and len(_tracker_ref[tid]["history"]) >= 2:
            spd_kmh, raw_kmh, _ = estimate_speed(_tracker_ref[tid]["history"], frame_w, ego_kmh)
            if spd_kmh is not None:
                spd_mph  = spd_kmh * KMH_TO_MPH
                raw_mph  = raw_kmh * KMH_TO_MPH
                diff_mph = spd_mph - ego_mph
                color    = _box_color(diff_mph, raw_mph)
                speed_str = f"~{spd_mph:.0f}mph"
                sign      = "+" if diff_mph >= 0 else ""
                diff_str  = f" ({sign}{diff_mph:.0f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{label_name} {speed_str}{diff_str}"
        (lw, lh), _ = cv2.getTextSize(label, font, 0.65, 2)
        ly = max(y1 - 6, lh + 4)
        cv2.rectangle(frame, (x1, ly - lh - 4), (x1 + lw + 4, ly + 2), color, -1)
        cv2.putText(frame, label, (x1 + 2, ly - 2), font, 0.65, (0, 0, 0), 2, cv2.LINE_AA)


def draw_bottom_banner(frame, tracked, frame_w, ego_kmh):
    """Draw a banner at the bottom with one coloured box per tracked vehicle."""
    if not tracked:
        return
    ego_mph   = ego_kmh * KMH_TO_MPH
    fh, fw    = frame.shape[:2]
    bar_h     = 52
    bar_y     = fh - bar_h
    cv2.rectangle(frame, (0, bar_y), (fw, fh), (0, 0, 0), -1)

    font   = cv2.FONT_HERSHEY_SIMPLEX
    fs     = 0.65
    pad    = 8
    x      = 8

    for (x1, y1, x2, y2, cls_id, tid) in tracked:
        label_name = VEHICLE_CLASSES.get(cls_id, "VEH")
        color      = (180, 180, 180)
        text       = f"{label_name} ---"

        if tid in _tracker_ref and len(_tracker_ref[tid]["history"]) >= 2:
            spd_kmh, raw_kmh, _ = estimate_speed(_tracker_ref[tid]["history"], frame_w, ego_kmh)
            if spd_kmh is not None:
                spd_mph  = spd_kmh * KMH_TO_MPH
                raw_mph  = raw_kmh * KMH_TO_MPH
                diff_mph = spd_mph - ego_mph
                color    = _box_color(diff_mph, raw_mph)
                sign     = "+" if diff_mph >= 0 else ""
                text     = f"{label_name} ~{spd_mph:.0f} ({sign}{diff_mph:.0f})"

        (tw, th), _ = cv2.getTextSize(text, font, fs, 2)
        box_w = tw + pad * 2
        box_x2 = x + box_w
        if box_x2 > fw - 4:
            break

        cv2.rectangle(frame, (x, bar_y + 6), (box_x2, fh - 6), color, -1)
        cv2.putText(frame, text, (x + pad, bar_y + 6 + th + 6),
                    font, fs, (0, 0, 0), 2, cv2.LINE_AA)
        x = box_x2 + 6


# Module-level reference so draw_vehicles can access tracker history by ID
_tracker_ref = {}


# ── Lane detection ────────────────────────────────────────────────────────────

LANE_W, LANE_H = 640, 320          # model input size
LANE_THRESHOLD  = 0.35             # softmax probability threshold
LANE_MIN_PTS    = 80               # minimum points to fit a polynomial
LANE_START_Y    = 0.50             # only use bottom 50% (road area, avoid horizon)

# Colors (BGR)
LANE_LEFT_COLOR  = (0, 255, 255)   # yellow
LANE_RIGHT_COLOR = (0, 200, 255)   # light orange
LANE_FILL_COLOR  = (0, 200, 0)     # green fill between lanes


def init_lane_model(model_path):
    """Load the EgoLanes ONNX model."""
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def _fit_lane_poly(mask, min_y_frac=LANE_START_Y):
    """
    Fit a 2nd-degree polynomial (x = f(y)) to bright pixels in a lane mask.
    Returns poly coefficients or None if insufficient data.
    """
    h, w = mask.shape
    y_start = int(h * min_y_frac)
    ys, xs = np.where(mask[y_start:] > 0)
    ys = ys + y_start
    if len(xs) < LANE_MIN_PTS:
        return None
    return np.polyfit(ys, xs, 2)


def detect_and_draw_lanes(frame, lane_sess):
    """
    Run lane detection and draw left/right ego lane lines + fill onto frame in-place.
    """
    fh, fw = frame.shape[:2]

    # ── preprocess ──────────────────────────────────────────────────────────
    resized = cv2.resize(frame, (LANE_W, LANE_H))
    inp = resized[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB, normalise
    inp = inp.transpose(2, 0, 1)[None]                      # HWC→NCHW

    # ── inference ───────────────────────────────────────────────────────────
    out = lane_sess.run(["output"], {"input": inp})[0]       # [1,3,320,640]
    probs = scipy_softmax(out[0], axis=0)                    # [3,320,640]

    left_mask  = (probs[0] > LANE_THRESHOLD).astype(np.uint8)
    right_mask = (probs[1] > LANE_THRESHOLD).astype(np.uint8)

    # ── fit polynomials ─────────────────────────────────────────────────────
    poly_l = _fit_lane_poly(left_mask)
    poly_r = _fit_lane_poly(right_mask)

    y_start_model = int(LANE_H * LANE_START_Y)
    ys_model = np.arange(y_start_model, LANE_H)

    # Clip drawing range at the vanishing point (where polys intersect)
    if poly_l is not None and poly_r is not None:
        diff = poly_l - poly_r
        roots = np.roots(diff)
        real_roots = roots[np.isreal(roots)].real
        valid = real_roots[(real_roots > y_start_model) & (real_roots < LANE_H)]
        if len(valid) > 0:
            y_start_model = int(np.max(valid)) + 1  # start just below crossing
            ys_model = np.arange(y_start_model, LANE_H)

    scale_x = fw / LANE_W
    scale_y = fh / LANE_H

    def model_to_frame_pts(poly, ys):
        xs = np.polyval(poly, ys)
        pts_x = (xs  * scale_x).astype(int)
        pts_y = (ys  * scale_y).astype(int)
        return np.stack([pts_x, pts_y], axis=1)

    # ── draw fill between lanes ─────────────────────────────────────────────
    if poly_l is not None and poly_r is not None:
        pts_l = model_to_frame_pts(poly_l, ys_model)
        pts_r = model_to_frame_pts(poly_r, ys_model)
        # Only fill if left stays left of right (sanity check)
        if pts_l[:, 0].mean() < pts_r[:, 0].mean():
            fill_pts = np.vstack([pts_l, pts_r[::-1]])
            overlay = frame.copy()
            cv2.fillPoly(overlay, [fill_pts], LANE_FILL_COLOR)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    # ── draw lane lines ──────────────────────────────────────────────────────
    for poly, color in [(poly_l, LANE_LEFT_COLOR), (poly_r, LANE_RIGHT_COLOR)]:
        if poly is None:
            continue
        pts = model_to_frame_pts(poly, ys_model).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=False, color=color,
                      thickness=4, lineType=cv2.LINE_AA)


def extract_gps(video_path, csv_path):
    """Extract GPS data from video using exiftool."""
    cmd = [
        "exiftool", "-ee3",
        "-p", "$QuickTime:GPSDateTime,$QuickTime:GPSLatitude,$QuickTime:GPSLongitude,$QuickTime:GPSSpeed,$QuickTime:GPSTrack",
        video_path
    ]
    with open(csv_path, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)

def load_gps(csv_path):
    """Load GPS CSV and parse start datetime. Returns (df, start_dt) or (None, None)."""
    try:
        df = pd.read_csv(csv_path, names=['Time', 'Lat', 'Lon', 'Speed', 'Track'])
        if df.empty or df['Time'].isna().all():
            return None, None
        raw_time = df['Time'].iloc[0]
        clean_time = str(raw_time).replace('Z', '').replace(':', '-', 2)
        start_dt = pd.to_datetime(clean_time, utc=True)
        return df, start_dt
    except Exception:
        return None, None

def fit_frame(frame, screen_w, screen_h):
    """Resize frame to fill screen while maintaining aspect ratio (letterbox/pillarbox)."""
    fh, fw = frame.shape[:2]
    scale = min(screen_w / fw, screen_h / fh)
    new_w = int(fw * scale)
    new_h = int(fh * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Pad to fill screen
    canvas = __import__('numpy').zeros((screen_h, screen_w, 3), dtype=resized.dtype)
    x_off = (screen_w - new_w) // 2
    y_off = (screen_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas

def overlay_info(frame, gps_time_display, time_display, lat, lon, speed_kmh):
    """Draw a single info bar across the top of the frame."""
    speed_mph = speed_kmh * KMH_TO_MPH
    fh, fw = frame.shape[:2]
    bar_h = 50
    cv2.rectangle(frame, (0, 0), (fw, bar_h), (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_DUPLEX
    fs = 0.9
    th = 2
    y = 36

    # Build fields: SPD | DATE/TIME | LAT | LON
    spd_str  = f"SPD: {speed_mph:.1f} mph"
    date_str = f"{gps_time_display}" if gps_time_display != "N/A" else f"TIME: {time_display}"
    lat_str  = f"LAT: {lat}"
    lon_str  = f"LON: {lon}"

    # Measure widths and space evenly
    items = [spd_str, date_str, lat_str, lon_str]
    widths = [cv2.getTextSize(t, font, fs, th)[0][0] for t in items]
    total_w = sum(widths)
    gap = (fw - total_w - 20) // (len(items) - 1)
    x = 10
    for text, w in zip(items, widths):
        cv2.putText(frame, text, (x, y), font, fs, (255, 255, 255), th, cv2.LINE_AA)
        x += w + gap

def play_video(video_path, screen_w, screen_h, total_files, file_idx, model, sahi_model, lane_sess, writer):
    """Play a single video with GPS overlay and vehicle detection."""
    global _tracker_ref
    print(f"[{file_idx}/{total_files}] Loading: {os.path.basename(video_path)}")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        csv_path = tmp.name

    try:
        extract_gps(video_path, csv_path)
        df, start_dt = load_gps(csv_path)
    finally:
        os.unlink(csv_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = max(1, int(1000 / fps))

    # Per-video tracker
    tracker = VehicleTracker()
    _tracker_ref = tracker.tracks

    win = "Dashcam Player"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)

        # Elapsed time
        total_ms = (frame_idx / fps) * 1000
        hours   = int(total_ms / (1000 * 60 * 60))
        minutes = int((total_ms / (1000 * 60)) % 60)
        seconds = int((total_ms / 1000) % 60)
        millis  = int(total_ms % 1000)
        time_display = f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"

        if start_dt is not None and df is not None:
            current_gps_dt  = start_dt + timedelta(seconds=frame_idx / fps)
            gps_time_display = current_gps_dt.strftime("%D %H:%M:%S.%f")[:-3]

            precise_idx = frame_idx / fps
            idx_floor   = int(precise_idx)
            idx_ceil    = min(idx_floor + 1, len(df) - 1)
            fraction    = precise_idx - idx_floor
            row_curr    = df.iloc[idx_floor]
            row_next    = df.iloc[idx_ceil]
            interp_speed = row_curr['Speed'] + fraction * (row_next['Speed'] - row_curr['Speed'])
            lat = row_curr['Lat']
            lon = row_curr['Lon']
        else:
            gps_time_display = "N/A"
            interp_speed = 0.0
            lat = "N/A"
            lon = "N/A"

        overlay_info(frame, gps_time_display, time_display, lat, lon, interp_speed)

        # Lane detection (drawn first so vehicle boxes appear on top)
        fh, fw = frame.shape[:2]
        if lane_sess is not None:
            detect_and_draw_lanes(frame, lane_sess)

        # Vehicle detection + tracking + speed overlay
        timestamp_s = frame_idx / fps
        detections = detect_vehicles(frame, model, sahi_model)
        tracked = tracker.update(detections, timestamp_s)
        draw_vehicles(frame, tracked, fw, interp_speed)
        draw_bottom_banner(frame, tracked, fw, interp_speed)

        # File index indicator (top-right)
        fname = f"{file_idx}/{total_files}: {os.path.basename(video_path)}"
        cv2.putText(frame, fname, (fw - 700, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (200, 200, 0), 2, cv2.LINE_AA)

        display = fit_frame(frame, screen_w, screen_h)
        cv2.imshow(win, display)
        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q'):
            return False   # quit entirely
        if key == ord('n'):
            break          # skip to next video

    cap.release()
    return True  # continue to next

def main():
    global MERGE_MODE

    parser = argparse.ArgumentParser(description="Dashcam Player with vehicle detection")
    parser.add_argument(
        "--merge",
        choices=["sahi", "nmw", "nms"],
        default="sahi",
        help="Detection merge strategy: sahi (default), nmw, nms",
    )
    args = parser.parse_args()
    MERGE_MODE = args.merge.upper()
    print(f"Merge mode: {MERGE_MODE}")

    cwd = os.getcwd()
    mp4_files = sorted(glob.glob(os.path.join(cwd, "*.MP4")) +
                       glob.glob(os.path.join(cwd, "*.mp4")))

    if not mp4_files:
        print("No MP4 files found in current directory.")
        return

    print(f"Found {len(mp4_files)} MP4 file(s).")
    print("Controls: [Q] Quit  [N] Next video\n")

    screen_w, screen_h = get_screen_resolution()
    print(f"Screen resolution: {screen_w}x{screen_h}")

    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    print("Model ready.")

    sahi_model = None
    if MERGE_MODE == "SAHI":
        print("Loading SAHI detection model...")
        sahi_model = init_sahi_model("yolov8n.pt")
        print("SAHI model ready.")
    print()

    lane_model_path = os.path.join(os.getcwd(), "EgoLanes_Lite_FP32.onnx")
    if os.path.exists(lane_model_path):
        print("Loading lane detection model...")
        lane_sess = init_lane_model(lane_model_path)
        print("Lane model ready.\n")
    else:
        print("EgoLanes_Lite_FP32.onnx not found — lane detection disabled.\n")
        lane_sess = None

    # Open output writer using first video's dimensions
    probe = cv2.VideoCapture(mp4_files[0])
    out_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    probe.release()
    out_path = os.path.join(os.getcwd(), "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))
    print(f"Writing processed video to: {out_path}\n")

    for i, video_path in enumerate(mp4_files, start=1):
        if not play_video(video_path, screen_w, screen_h, len(mp4_files), i,
                          model, sahi_model, lane_sess, writer):
            break

    writer.release()
    print(f"\nSaved: {out_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
