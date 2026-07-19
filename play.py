import cv2
import os
import tempfile
import argparse
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from ensemble_boxes import weighted_boxes_fusion
import onnxruntime as ort
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from common import (
    discover_input_videos,
    draw_crop_box,
    extract_gps,
    fit_frame,
    get_lower_2to1_crop,
    get_screen_resolution,
    load_gps,
)

# COCO class IDs for vehicles
VEHICLE_CLASSES = {2: "CAR", 3: "MOTO", 5: "BUS", 7: "TRUCK"}

# Assumed real-world heights in metres per class (for distance estimation)
VEHICLE_HEIGHT_M = {2: 1.5, 3: 1.2, 5: 3.0, 7: 2.5}

# Approximate top-down footprint dimensions per class: (width_m, length_m)
VEHICLE_FOOTPRINT_M = {
    2: (1.9, 4.5),   # car
    3: (0.8, 2.2),   # moto
    5: (2.6, 12.0),  # bus
    7: (2.8, 8.0),   # truck
}

# Bounding-box colours are now dynamic — see _box_color()

# Speed history window (frames) for smoothing
SPEED_HISTORY = 10

KMH_TO_MPH = 0.621371

# Camera / bird's-eye assumptions (until a monocular depth model is provided)
CAMERA_FOV_DEG = 140.0
TOPDOWN_HORIZON_FRAC = 0.42
TOPDOWN_MAX_DEPTH_M = 30.0
TOPDOWN_WIDTH_M = 24.0
TOPDOWN_EGO_LANE_WIDTH_M = 3.7

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


def _detect_sahi(frame, sahi_model, x_offset=0, y_offset=0):
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
        dets.append((x1 + x_offset, y1 + y_offset,
                     x2 + x_offset, y2 + y_offset, cls_id))
    return dets


# Set by argparse in main() — controls which merge strategy is used
MERGE_MODE = "SAHI"


def detect_vehicles(frame, model, sahi_model=None, crop_rect=None):
    """
    Detect vehicles using the active MERGE_MODE:
      SAHI — sliced inference (best recall for small/distant objects, default)
      NMW  — dual-pass (full + center crop) merged with Weighted Box Fusion
      NMS  — dual-pass (full + center crop) merged with standard NMS
    Returns list of (x1, y1, x2, y2, class_id).
    """
    x0, y0 = 0, 0
    proc_frame = frame
    if crop_rect is not None:
        x0, y0, cw, ch = crop_rect
        proc_frame = frame[y0:y0 + ch, x0:x0 + cw]

    if MERGE_MODE == "SAHI" and sahi_model is not None:
        return _detect_sahi(proc_frame, sahi_model, x_offset=x0, y_offset=y0)

    all_dets, fw, fh = _dual_pass_raw(proc_frame, model)

    if MERGE_MODE == "NMW":
        local = _apply_nmw(all_dets, fw, fh)
    else:
        local = _apply_nms(all_dets, fw, fh)  # NMS (default fallback)
    return [(x1 + x0, y1 + y0, x2 + x0, y2 + y0, cls_id) for x1, y1, x2, y2, cls_id in local]


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
LANE_START_Y    = 0.50             # only use bottom 50% (road area, avoid horizon)

# Colors (BGR)
LANE_LEFT_COLOR  = (255, 220, 0)
LANE_RIGHT_COLOR = (0, 170, 255)
LANE_FILL_COLOR  = (0, 210, 80)
OTHER_LANE_COLOR = (180, 80, 255)
BG_CLASS = 255


def init_lane_model(model_path):
    """Load the EgoLanes ONNX model."""
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def logits_to_class_mask(logits_chw):
    """Convert EgoLanes logits into VisionPilot class priorities."""
    height, width = logits_chw.shape[1:]
    mask = np.full((height, width), BG_CLASS, dtype=np.uint8)

    c0 = logits_chw[0] > 0.0  # ego-left
    c1 = logits_chw[1] > 0.0  # ego-right
    c2 = logits_chw[2] > 0.0  # other lanes

    mask[c2] = 2
    mask[np.logical_and(~c2, c1)] = 1
    mask[np.logical_and(~c2, np.logical_and(~c1, c0))] = 0
    return mask


def mask_to_color(mask_hw):
    color = np.zeros((mask_hw.shape[0], mask_hw.shape[1], 3), dtype=np.uint8)
    color[mask_hw == 2] = OTHER_LANE_COLOR
    color[mask_hw == 1] = LANE_RIGHT_COLOR
    color[mask_hw == 0] = LANE_LEFT_COLOR
    return color


def _class_mask_to_lane_points(mask_hw, class_id, start_y_frac=LANE_START_Y,
                               row_step=2, min_row_pixels=2, min_points=20):
    """Create a lane centerline as per-row mean x over the class mask."""
    h, _ = mask_hw.shape
    y_start = int(h * start_y_frac)
    pts = []
    for y in range(y_start, h, row_step):
        xs = np.where(mask_hw[y] == class_id)[0]
        if len(xs) < min_row_pixels:
            continue
        pts.append((float(np.mean(xs)), float(y)))

    if len(pts) < min_points:
        return None
    return np.array(pts, dtype=np.float32)


def _class_mask_to_raw_points(mask_hw, class_id, start_y_frac=LANE_START_Y, step=8):
    """Downsampled raw lane-class points for debug plotting."""
    h, _ = mask_hw.shape
    y_start = int(h * start_y_frac)
    ys, xs = np.where(mask_hw[y_start:] == class_id)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)
    ys = ys + y_start
    if len(xs) > step:
        xs = xs[::step]
        ys = ys[::step]
    return np.stack([xs.astype(np.int32), ys.astype(np.int32)], axis=1)


def draw_mask_contours(roi, mask_hw, crop_w, crop_h):
    class_specs = [
        (0, LANE_LEFT_COLOR, 3),
        (1, LANE_RIGHT_COLOR, 3),
        (2, OTHER_LANE_COLOR, 2),
    ]
    for class_id, color, thickness in class_specs:
        class_mask = (mask_hw == class_id).astype(np.uint8) * 255
        if not np.any(class_mask):
            continue
        resized_mask = cv2.resize(class_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(roi, contours, -1, color, thickness, cv2.LINE_AA)


def fit_and_draw_lane_polynomials(frame, class_mask, crop_w, crop_h):
    """Fit x=f(y) polynomials per lane class and draw them on the ROI frame."""
    class_specs = [
        (0, LANE_LEFT_COLOR, 3),
        (1, LANE_RIGHT_COLOR, 3),
        (2, OTHER_LANE_COLOR, 2),
    ]

    for class_id, color, thickness in class_specs:
        class_bin = (class_mask == class_id).astype(np.uint8) * 255
        if not np.any(class_bin):
            continue

        contours, _ = cv2.findContours(class_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 40:
                continue

            comp_mask = np.zeros_like(class_bin)
            cv2.drawContours(comp_mask, [contour], -1, 255, thickness=-1)
            ys, xs = np.where(comp_mask > 0)
            if len(xs) < 24:
                continue

            coeffs = np.polyfit(ys.astype(np.float32), xs.astype(np.float32), 2)
            y_min = int(np.min(ys))
            y_max = int(np.max(ys))
            y_vals = np.arange(y_min, y_max + 1, 3, dtype=np.float32)
            x_vals = np.polyval(coeffs, y_vals)

            scale_x = crop_w / float(LANE_W)
            scale_y = crop_h / float(LANE_H)
            poly_pts = []
            for x_m, y_m in zip(x_vals, y_vals):
                if x_m < 0 or x_m >= LANE_W:
                    continue
                x_px = int(round(x_m * scale_x))
                y_px = int(round(y_m * scale_y))
                if 0 <= x_px < frame.shape[1] and 0 <= y_px < frame.shape[0]:
                    poly_pts.append((x_px, y_px))

            if len(poly_pts) >= 2:
                arr = np.array(poly_pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [arr], False, color, thickness, cv2.LINE_AA)
                for p in poly_pts[::12]:
                    cv2.circle(frame, p, 2, color, -1, cv2.LINE_AA)


def detect_and_draw_lanes(frame, lane_sess, draw_overlay=True, x_offset=0, y_offset=0, draw_mode="mask"):
    """
    Run lane detection and draw left/right ego lane lines + fill onto frame in-place.
    Returns dict with frame-space lane points and raw points:
    {"left": pts or None, "right": pts or None, "left_raw": Nx2, "right_raw": Nx2}
    """
    fh, fw = frame.shape[:2]

    # ── preprocess ──────────────────────────────────────────────────────────
    resized = cv2.resize(frame, (LANE_W, LANE_H))
    inp = resized[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB, normalise
    inp = inp.transpose(2, 0, 1)[None]                      # HWC→NCHW

    # ── inference ───────────────────────────────────────────────────────────
    input_name = lane_sess.get_inputs()[0].name
    output_name = lane_sess.get_outputs()[0].name
    out = lane_sess.run([output_name], {input_name: inp})[0]  # [1,3,320,640]
    class_mask = logits_to_class_mask(out[0])

    left_model = _class_mask_to_lane_points(class_mask, 0)
    right_model = _class_mask_to_lane_points(class_mask, 1)
    left_raw_model = _class_mask_to_raw_points(class_mask, 0)
    right_raw_model = _class_mask_to_raw_points(class_mask, 1)

    scale_x = fw / LANE_W
    scale_y = fh / LANE_H

    def model_pts_to_frame_pts(pts_model):
        if pts_model is None or len(pts_model) == 0:
            return None
        pts = pts_model.copy()
        pts[:, 0] = pts[:, 0] * scale_x
        pts[:, 1] = pts[:, 1] * scale_y
        return pts.astype(np.int32)

    def model_raw_to_frame_raw(pts_model):
        if pts_model is None or len(pts_model) == 0:
            return np.empty((0, 2), dtype=np.int32)
        pts = pts_model.astype(np.float32).copy()
        pts[:, 0] = pts[:, 0] * scale_x
        pts[:, 1] = pts[:, 1] * scale_y
        return pts.astype(np.int32)

    pts_l = None
    pts_r = None

    left_raw = model_raw_to_frame_raw(left_raw_model)
    right_raw = model_raw_to_frame_raw(right_raw_model)
    pts_l = model_pts_to_frame_pts(left_model)
    pts_r = model_pts_to_frame_pts(right_model)

    if draw_overlay and draw_mode == "mask":
        color_mask = mask_to_color(class_mask)
        color_overlay = cv2.resize(color_mask, (fw, fh), interpolation=cv2.INTER_NEAREST)

        left_mask = cv2.resize((class_mask == 0).astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST)
        right_mask = cv2.resize((class_mask == 1).astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST)
        other_mask = cv2.resize((class_mask == 2).astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST)

        overlay = frame.copy()
        overlay[left_mask > 0] = cv2.addWeighted(frame[left_mask > 0], 0.30, color_overlay[left_mask > 0], 0.70, 0)
        overlay[right_mask > 0] = cv2.addWeighted(frame[right_mask > 0], 0.30, color_overlay[right_mask > 0], 0.70, 0)
        overlay[other_mask > 0] = cv2.addWeighted(frame[other_mask > 0], 0.45, color_overlay[other_mask > 0], 0.55, 0)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)

        if np.any(left_mask) and np.any(right_mask):
            corridor = np.zeros((fh, fw), dtype=np.uint8)
            corridor[np.logical_or(left_mask > 0, right_mask > 0)] = 255
            kernel = np.ones((9, 9), dtype=np.uint8)
            corridor = cv2.morphologyEx(corridor, cv2.MORPH_CLOSE, kernel)
            corridor_bgr = np.zeros_like(frame)
            corridor_bgr[corridor > 0] = LANE_FILL_COLOR
            cv2.addWeighted(corridor_bgr, 0.16, frame, 0.84, 0, frame)

        draw_mask_contours(frame, class_mask, fw, fh)

    if draw_overlay and draw_mode == "poly":
        fit_and_draw_lane_polynomials(frame, class_mask, fw, fh)

    if pts_l is not None:
        pts_l = pts_l.copy()
        pts_l[:, 0] += x_offset
        pts_l[:, 1] += y_offset
    if pts_r is not None:
        pts_r = pts_r.copy()
        pts_r[:, 0] += x_offset
        pts_r[:, 1] += y_offset
    if left_raw is not None and len(left_raw) > 0:
        left_raw = left_raw.copy()
        left_raw[:, 0] += x_offset
        left_raw[:, 1] += y_offset
    if right_raw is not None and len(right_raw) > 0:
        right_raw = right_raw.copy()
        right_raw[:, 0] += x_offset
        right_raw[:, 1] += y_offset

    return {
        "left": pts_l,
        "right": pts_r,
        "left_raw": left_raw,
        "right_raw": right_raw,
        "class_mask": class_mask,
    }


def draw_raw_lane_debug(frame, lane_result):
    """Plot raw EgoLanes pixel hits to help alignment debugging."""
    if lane_result is None:
        return
    for key, color in (("left_raw", (0, 255, 255)), ("right_raw", (0, 128, 255))):
        pts = lane_result.get(key)
        if pts is None or len(pts) == 0:
            continue
        for x, y in pts:
            cv2.circle(frame, (int(x), int(y)), 1, color, -1, lineType=cv2.LINE_AA)


def _pixel_to_ground(u, v, frame_w, frame_h):
    """Project an image pixel to assumed ground-plane coordinates (x_m, z_m)."""
    cx = frame_w / 2.0
    focal_px = (frame_w / 2.0) / np.tan(np.deg2rad(CAMERA_FOV_DEG / 2.0))
    horizon_y = frame_h * TOPDOWN_HORIZON_FRAC
    dv = v - horizon_y
    if dv <= 1.0:
        return None

    k = (frame_h - horizon_y) * 5.0
    z_m = k / dv
    if z_m <= 0 or z_m > TOPDOWN_MAX_DEPTH_M:
        return None

    x_m = ((u - cx) / focal_px) * z_m
    return x_m, z_m


def _lane_pts_to_ground(lane_pts, frame_w, frame_h):
    """Convert frame-space lane points to ground-plane coordinates."""
    if lane_pts is None:
        return None
    world_pts = []
    for u, v in lane_pts:
        p = _pixel_to_ground(float(u), float(v), frame_w, frame_h)
        if p is not None:
            world_pts.append(p)
    if len(world_pts) < 2:
        return None
    world_pts = np.array(world_pts, dtype=np.float32)
    order = np.argsort(world_pts[:, 1])
    return world_pts[order]


def _world_to_canvas(points_xz, out_w, out_h):
    """Map ground-plane coordinates (x,z) in metres into top-down canvas pixels."""
    if points_xz is None or len(points_xz) == 0:
        return None

    half_fov = np.deg2rad(CAMERA_FOV_DEG * 0.5)
    half_span_fov = np.tan(half_fov) * TOPDOWN_MAX_DEPTH_M
    half_span_m = max(TOPDOWN_WIDTH_M * 0.5, half_span_fov)
    m_per_px_x = (2.0 * half_span_m) / max(out_w, 1)
    m_per_px_z = TOPDOWN_MAX_DEPTH_M / max(out_h - 20, 1)
    px = (out_w * 0.5 + points_xz[:, 0] / m_per_px_x).astype(np.int32)
    py = (out_h - 10 - points_xz[:, 1] / m_per_px_z).astype(np.int32)
    pts = np.stack([px, py], axis=1)
    in_bounds = (
        (pts[:, 0] >= 0) & (pts[:, 0] < out_w) &
        (pts[:, 1] >= 0) & (pts[:, 1] < out_h)
    )
    pts = pts[in_bounds]
    if len(pts) < 1:
        return None
    return pts.reshape(-1, 1, 2)


def _world_point_to_canvas(x_m, z_m, out_w, out_h):
    """Project a single world point to BEV canvas coordinates."""
    half_fov = np.deg2rad(CAMERA_FOV_DEG * 0.5)
    half_span_fov = np.tan(half_fov) * TOPDOWN_MAX_DEPTH_M
    half_span_m = max(TOPDOWN_WIDTH_M * 0.5, half_span_fov)
    m_per_px_x = (2.0 * half_span_m) / max(out_w, 1)
    m_per_px_z = TOPDOWN_MAX_DEPTH_M / max(out_h - 20, 1)
    px = int(round(out_w * 0.5 + x_m / m_per_px_x))
    py = int(round(out_h - 10 - z_m / m_per_px_z))
    return px, py


def _shift_lane_x(points_xz, shift_m):
    """Shift lane polyline laterally in world coordinates."""
    if points_xz is None:
        return None
    shifted = points_xz.copy()
    shifted[:, 0] += shift_m
    return shifted


def _draw_topdown_fov_funnel(bev, frame_w, frame_h):
    """Draw a 140-degree field-of-view funnel in grey."""
    half_fov = np.deg2rad(CAMERA_FOV_DEG * 0.5)
    max_z = TOPDOWN_MAX_DEPTH_M
    edge_x = np.tan(half_fov) * max_z

    a = _world_point_to_canvas(0.0, 0.0, frame_w, frame_h)
    l = _world_point_to_canvas(-edge_x, max_z, frame_w, frame_h)
    r = _world_point_to_canvas(edge_x, max_z, frame_w, frame_h)
    overlay = bev.copy()
    poly = np.array([a, l, r], dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(overlay, [poly], (58, 58, 58))
    cv2.addWeighted(overlay, 0.20, bev, 0.80, 0, bev)
    cv2.line(bev, a, l, (150, 150, 150), 2, cv2.LINE_AA)
    cv2.line(bev, a, r, (150, 150, 150), 2, cv2.LINE_AA)


def _draw_topdown_range_arcs(bev, out_w, out_h, step_m=10):
    """Draw radial distance arcs every step_m inside FOV, with labels."""
    half_fov = np.deg2rad(CAMERA_FOV_DEG * 0.5)
    angles = np.linspace(-half_fov, half_fov, 140, dtype=np.float32)

    for dist_m in range(step_m, int(TOPDOWN_MAX_DEPTH_M) + 1, step_m):
        x = dist_m * np.sin(angles)
        z = dist_m * np.cos(angles)
        pts = np.array([_world_point_to_canvas(float(xi), float(zi), out_w, out_h) for xi, zi in zip(x, z)], dtype=np.int32)
        pts = pts.reshape(-1, 1, 2)
        cv2.polylines(bev, [pts], False, (95, 95, 95), 1, cv2.LINE_AA)

        tx, ty = _world_point_to_canvas(0.0, float(dist_m), out_w, out_h)
        cv2.putText(bev, f"{dist_m}m", (tx + 6, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (165, 165, 165), 1, cv2.LINE_AA)


def _build_parallel_lane_world(left_world, right_world, assumed_lane_width_m=TOPDOWN_EGO_LANE_WIDTH_M):
    """Generate smoother near-parallel lane boundaries for BEV rendering."""
    if left_world is None and right_world is None:
        return None, None

    if left_world is not None:
        left_world = left_world[np.argsort(left_world[:, 1])]
    if right_world is not None:
        right_world = right_world[np.argsort(right_world[:, 1])]

    if left_world is not None and right_world is not None:
        z_min = max(2.0, min(float(left_world[:, 1].min()), float(right_world[:, 1].min())))
        z_max = TOPDOWN_MAX_DEPTH_M
        if z_max <= z_min + 2.0 or len(left_world) < 6 or len(right_world) < 6:
            return left_world, right_world

        z_grid = np.linspace(z_min, z_max, 60, dtype=np.float32)
        fit_l = np.polyfit(left_world[:, 1], left_world[:, 0], 1)
        fit_r = np.polyfit(right_world[:, 1], right_world[:, 0], 1)
        x_left = np.polyval(fit_l, z_grid)
        x_right = np.polyval(fit_r, z_grid)
        width = np.median(x_right - x_left)
        width = float(np.clip(width, 2.8, 5.0))
        center = 0.5 * (x_left + x_right)

        if len(z_grid) >= 8:
            line = np.polyfit(z_grid, center, 1)
            center = np.polyval(line, z_grid)

        left_fit = np.stack([center - width * 0.5, z_grid], axis=1)
        right_fit = np.stack([center + width * 0.5, z_grid], axis=1)
        return left_fit.astype(np.float32), right_fit.astype(np.float32)

    if left_world is not None:
        z_grid = np.linspace(max(2.0, float(left_world[:, 1].min())),
                             TOPDOWN_MAX_DEPTH_M,
                             60, dtype=np.float32)
        if len(left_world) >= 6:
            fit_l = np.polyfit(left_world[:, 1], left_world[:, 0], 1)
            x_left = np.polyval(fit_l, z_grid)
        else:
            x_left = np.interp(z_grid, left_world[:, 1], left_world[:, 0])
        right_fit = np.stack([x_left + assumed_lane_width_m, z_grid], axis=1)
        left_fit = np.stack([x_left, z_grid], axis=1)
        return left_fit.astype(np.float32), right_fit.astype(np.float32)

    z_grid = np.linspace(max(2.0, float(right_world[:, 1].min())),
                         TOPDOWN_MAX_DEPTH_M,
                         60, dtype=np.float32)
    if len(right_world) >= 6:
        fit_r = np.polyfit(right_world[:, 1], right_world[:, 0], 1)
        x_right = np.polyval(fit_r, z_grid)
    else:
        x_right = np.interp(z_grid, right_world[:, 1], right_world[:, 0])
    left_fit = np.stack([x_right - assumed_lane_width_m, z_grid], axis=1)
    right_fit = np.stack([x_right, z_grid], axis=1)
    return left_fit.astype(np.float32), right_fit.astype(np.float32)


def _draw_vehicle_bev_box(bev, x_m, z_m, cls_id, color):
    """Draw class-sized axis-aligned top-down box in world coordinates."""
    w_m, l_m = VEHICLE_FOOTPRINT_M.get(cls_id, (1.9, 4.5))
    z0 = max(0.2, z_m - 0.5 * l_m)
    z1 = min(TOPDOWN_MAX_DEPTH_M, z_m + 0.5 * l_m)
    x0 = x_m - 0.5 * w_m
    x1 = x_m + 0.5 * w_m

    p1 = _world_point_to_canvas(x0, z0, bev.shape[1], bev.shape[0])
    p2 = _world_point_to_canvas(x1, z0, bev.shape[1], bev.shape[0])
    p3 = _world_point_to_canvas(x1, z1, bev.shape[1], bev.shape[0])
    p4 = _world_point_to_canvas(x0, z1, bev.shape[1], bev.shape[0])
    poly = np.array([p1, p2, p3, p4], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(bev, [poly], True, color, 2, cv2.LINE_AA)


def _draw_topdown_footprint_legend(bev):
    """Show assumed class footprint sizes used for BEV boxes."""
    x0, y0 = 12, 54
    line_h = 18
    panel_w = 250
    panel_h = line_h * 6 + 12

    overlay = bev.copy()
    cv2.rectangle(overlay, (x0 - 8, y0 - 20), (x0 + panel_w, y0 + panel_h), (22, 22, 22), -1)
    cv2.addWeighted(overlay, 0.55, bev, 0.45, 0, bev)
    cv2.rectangle(bev, (x0 - 8, y0 - 20), (x0 + panel_w, y0 + panel_h), (90, 90, 90), 1)

    cv2.putText(bev, "BEV box assumptions (W x L m)", (x0, y0 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)

    rows = [
        (2, "CAR"),
        (3, "MOTO"),
        (5, "BUS"),
        (7, "TRUCK"),
    ]
    for i, (cls_id, label) in enumerate(rows, start=1):
        w_m, l_m = VEHICLE_FOOTPRINT_M[cls_id]
        txt = f"{label:5s}: {w_m:.1f} x {l_m:.1f}"
        cv2.putText(bev, txt, (x0, y0 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (185, 185, 185), 1, cv2.LINE_AA)


def draw_top_down_view(frame_w, frame_h, tracked, lane_result, ego_kmh):
    """Build a top-down plot with FOV wedge, range arcs, lanes, and class-sized boxes."""
    bev = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    # Road background gradient
    for y in range(frame_h):
        t = y / max(frame_h - 1, 1)
        shade = int(16 + 22 * (1.0 - t))
        bev[y, :] = (shade, shade, shade)

    _draw_topdown_fov_funnel(bev, frame_w, frame_h)
    _draw_topdown_range_arcs(bev, frame_w, frame_h, step_m=10)

    cv2.putText(bev, "Distance (m)", (frame_w // 2 - 58, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (195, 195, 195), 1, cv2.LINE_AA)
    cv2.putText(bev, f"TOP-DOWN PLOT  FOV {CAMERA_FOV_DEG:.0f}deg",
                (12, 36), cv2.FONT_HERSHEY_DUPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    _draw_topdown_footprint_legend(bev)

    left_world = _lane_pts_to_ground(lane_result.get("left"), frame_w, frame_h) if lane_result else None
    right_world = _lane_pts_to_ground(lane_result.get("right"), frame_w, frame_h) if lane_result else None
    left_world, right_world = _build_parallel_lane_world(left_world, right_world)

    # Draw ego lane boundaries from EgoLanes
    left_canvas = _world_to_canvas(left_world, frame_w, frame_h)
    right_canvas = _world_to_canvas(right_world, frame_w, frame_h)
    if left_canvas is not None:
        cv2.polylines(bev, [left_canvas], False, (0, 255, 255), 3, cv2.LINE_AA)
    if right_canvas is not None:
        cv2.polylines(bev, [right_canvas], False, (0, 200, 255), 3, cv2.LINE_AA)

    # Draw adjacent lanes by shifting ego lane assumptions by typical lane width.
    for n in (1, 2):
        if left_world is not None:
            shifted = _shift_lane_x(left_world, -TOPDOWN_EGO_LANE_WIDTH_M * n)
            shifted_canvas = _world_to_canvas(shifted, frame_w, frame_h)
            if shifted_canvas is not None:
                cv2.polylines(bev, [shifted_canvas], False, (100, 100, 100), 1, cv2.LINE_AA)
        if right_world is not None:
            shifted = _shift_lane_x(right_world, TOPDOWN_EGO_LANE_WIDTH_M * n)
            shifted_canvas = _world_to_canvas(shifted, frame_w, frame_h)
            if shifted_canvas is not None:
                cv2.polylines(bev, [shifted_canvas], False, (100, 100, 100), 1, cv2.LINE_AA)

    # Ego vehicle marker
    ex, ey = _world_point_to_canvas(0.0, 2.0, frame_w, frame_h)
    if 0 <= ex < frame_w and 0 <= ey < frame_h:
        cv2.rectangle(bev, (ex - 12, ey - 18), (ex + 12, ey + 18), (60, 220, 60), -1)
        cv2.putText(bev, f"EGO {ego_kmh * KMH_TO_MPH:.0f}mph", (ex - 48, ey + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 255, 220), 1, cv2.LINE_AA)

    # Other vehicles from bbox bottom-center projected to ground
    ego_mph = ego_kmh * KMH_TO_MPH
    for x1, y1, x2, y2, cls_id, tid in tracked:
        p = _pixel_to_ground((x1 + x2) * 0.5, y2, frame_w, frame_h)
        if p is None:
            continue
        vx, vy = _world_point_to_canvas(float(p[0]), float(p[1]), frame_w, frame_h)
        if not (0 <= vx < frame_w and 0 <= vy < frame_h):
            continue
        label_name = VEHICLE_CLASSES.get(cls_id, "VEH")
        color = (180, 180, 180)
        txt = label_name

        if tid in _tracker_ref and len(_tracker_ref[tid]["history"]) >= 2:
            spd_kmh, raw_kmh, _ = estimate_speed(_tracker_ref[tid]["history"], frame_w, ego_kmh)
            if spd_kmh is not None:
                spd_mph = spd_kmh * KMH_TO_MPH
                raw_mph = raw_kmh * KMH_TO_MPH
                diff_mph = spd_mph - ego_mph
                color = _box_color(diff_mph, raw_mph)
                sign = "+" if diff_mph >= 0 else ""
                txt = f"{label_name} {spd_mph:.0f} ({sign}{diff_mph:.0f})"

        _draw_vehicle_bev_box(bev, float(p[0]), float(p[1]), cls_id, color)
        cv2.putText(bev, txt, (vx + 10, vy - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (235, 235, 235), 1, cv2.LINE_AA)

    return bev


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

def play_video(video_path, screen_w, screen_h, total_files, file_idx, model, sahi_model, lane_sess, writer, writer2):
    """Play a single video with GPS overlay and vehicle detection."""
    global _tracker_ref, TOPDOWN_MAX_DEPTH_M
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
    show_topdown = False
    yolo_enabled = True
    ego_enabled = True
    raw_lane_debug = False
    lane_fit_polys = False

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
        crop_rect = get_lower_2to1_crop(fw, fh)
        draw_crop_box(frame, crop_rect)

        lane_result = {"left": None, "right": None, "left_raw": np.empty((0, 2), dtype=np.int32), "right_raw": np.empty((0, 2), dtype=np.int32)}
        lane_mode = "POLY" if lane_fit_polys else "MASK"
        if ego_enabled and lane_sess is not None:
            cx, cy, cw, ch = crop_rect
            lane_roi = frame[cy:cy + ch, cx:cx + cw]
            lane_result = detect_and_draw_lanes(
                lane_roi,
                lane_sess,
                draw_overlay=True,
                x_offset=cx,
                y_offset=cy,
                draw_mode=("poly" if lane_fit_polys else "mask"),
            )
            if raw_lane_debug:
                draw_raw_lane_debug(frame, lane_result)

        # Vehicle detection + tracking + speed overlay
        timestamp_s = frame_idx / fps
        tracked = []
        if yolo_enabled:
            detections = detect_vehicles(frame, model, sahi_model, crop_rect=crop_rect)
            tracked = tracker.update(detections, timestamp_s)
            draw_vehicles(frame, tracked, fw, interp_speed)
            draw_bottom_banner(frame, tracked, fw, interp_speed)

        topdown = draw_top_down_view(fw, fh, tracked, lane_result, interp_speed)

        # File index indicator (top-right)
        fname = f"{file_idx}/{total_files}: {os.path.basename(video_path)}"
        cv2.putText(frame, fname, (fw - 700, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (200, 200, 0), 2, cv2.LINE_AA)
        status = (
            f"D:Topdown {'ON' if show_topdown else 'OFF'}  "
            f"Y:YOLO {'ON' if yolo_enabled else 'OFF'}  "
            f"E:EgoLanes {'ON' if ego_enabled else 'OFF'}  "
            f"R:RawLane {'ON' if raw_lane_debug else 'OFF'}  "
            f"F:LaneMode {lane_mode}  "
            f"<>:Depth {TOPDOWN_MAX_DEPTH_M:.0f}m"
        )
        cv2.putText(frame, status, (16, fh - 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (240, 240, 240), 2, cv2.LINE_AA)

        display_frame = topdown if show_topdown else frame
        display = fit_frame(display_frame, screen_w, screen_h)
        cv2.imshow(win, display)
        if writer is not None:
            writer.write(frame)
        if writer2 is not None:
            writer2.write(topdown)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q'):
            return False   # quit entirely
        if key == ord('n'):
            break          # skip to next video
        if key == ord('d'):
            show_topdown = not show_topdown
        if key == ord('y'):
            yolo_enabled = not yolo_enabled
        if key == ord('e'):
            ego_enabled = not ego_enabled
        if key == ord('r'):
            raw_lane_debug = not raw_lane_debug
        if key == ord('f'):
            lane_fit_polys = not lane_fit_polys
        # Support both shifted and unshifted keys on common keyboard layouts.
        if key in (ord('>'), ord('.')):
            TOPDOWN_MAX_DEPTH_M = min(200.0, TOPDOWN_MAX_DEPTH_M + 5.0)
        if key in (ord('<'), ord(',')):
            TOPDOWN_MAX_DEPTH_M = max(10.0, TOPDOWN_MAX_DEPTH_M - 5.0)

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
    mp4_files = discover_input_videos(cwd)

    if not mp4_files:
        print("No source MP4 files found. Place videos in current directory or camera/.")
        return

    print(f"Found {len(mp4_files)} MP4 file(s).")
    print("Controls: [Q] Quit  [N] Next video  [D] Toggle camera/top-down display  [Y] Toggle YOLO  [E] Toggle EgoLanes  [R] Toggle raw-lane debug  [F] Toggle lane mask/poly-fit mode  [<]/[>] (or [,]/[.]) top-down depth -/+ 5m\n")

    screen_w, screen_h = get_screen_resolution()
    print(f"Screen resolution: {screen_w}x{screen_h}")

    print("Loading YOLOv8n model...")
    model = YOLO("models/yolov8n.pt")
    print("Model ready.")

    sahi_model = None
    if MERGE_MODE == "SAHI":
        print("Loading SAHI detection model...")
        sahi_model = init_sahi_model("models/yolov8n.pt")
        print("SAHI model ready.")
    print()

    lane_model_path = os.path.join(os.getcwd(), "models/EgoLanes_Lite_FP32.onnx")
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
    out2_path = os.path.join(os.getcwd(), "output2.m4v")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))
    writer2 = cv2.VideoWriter(out2_path, fourcc, out_fps, (out_w, out_h))
    print(f"Writing processed camera view to: {out_path}")
    print(f"Writing processed top-down view to: {out2_path}\n")

    for i, video_path in enumerate(mp4_files, start=1):
        if not play_video(video_path, screen_w, screen_h, len(mp4_files), i,
                          model, sahi_model, lane_sess, writer, writer2):
            break

    writer.release()
    writer2.release()
    print(f"\nSaved: {out_path}")
    print(f"Saved: {out2_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
