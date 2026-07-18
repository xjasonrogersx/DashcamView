import cv2
import glob
import os
import subprocess

import numpy as np
import pandas as pd


DEFAULT_GENERATED_OUTPUT_NAMES = frozenset({
    "output.mp4",
    "output2.m4v",
    "output_autospeed.mp4",
})


def get_lower_2to1_crop(frame_w, frame_h):
    """Return an inference crop with 2:1 aspect ratio, biased to lower frame."""
    target_h = int(round(frame_w / 2.0))
    if target_h <= frame_h:
        return 0, frame_h - target_h, frame_w, target_h

    crop_h = frame_h
    crop_w = min(frame_w, int(round(frame_h * 2.0)))
    x0 = max(0, (frame_w - crop_w) // 2)
    return x0, 0, crop_w, crop_h


def draw_crop_box(frame, crop_rect):
    """Visualize the active inference crop in grey."""
    x0, y0, cw, ch = crop_rect
    cv2.rectangle(frame, (x0, y0), (x0 + cw, y0 + ch), (140, 140, 140), 2, cv2.LINE_AA)
    cv2.putText(frame, "2:1 inference crop", (x0 + 8, max(18, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 2, cv2.LINE_AA)


def discover_input_videos(base_dir, generated_output_names=None):
    """Find source MP4 files while excluding generated output videos."""
    excluded = DEFAULT_GENERATED_OUTPUT_NAMES if generated_output_names is None else generated_output_names
    patterns = [
        os.path.join(base_dir, "*.MP4"),
        os.path.join(base_dir, "*.mp4"),
        os.path.join(base_dir, "camera", "*.MP4"),
        os.path.join(base_dir, "camera", "*.mp4"),
    ]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))

    unique = sorted(set(found))
    return [path for path in unique if os.path.basename(path).lower() not in excluded]


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
    return 1920, 1080


def fit_frame(frame, screen_w, screen_h):
    """Resize frame to fit screen while preserving aspect ratio."""
    fh, fw = frame.shape[:2]
    scale = min(screen_w / fw, screen_h / fh)
    new_w = int(fw * scale)
    new_h = int(fh * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((screen_h, screen_w, 3), dtype=resized.dtype)
    x_off = (screen_w - new_w) // 2
    y_off = (screen_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def extract_gps(video_path, csv_path):
    """Extract GPS data from video using exiftool."""
    cmd = [
        "exiftool",
        "-ee3",
        "-p",
        "$QuickTime:GPSDateTime,$QuickTime:GPSLatitude,$QuickTime:GPSLongitude,$QuickTime:GPSSpeed,$QuickTime:GPSTrack",
        video_path,
    ]
    with open(csv_path, "w") as handle:
        subprocess.run(cmd, stdout=handle, stderr=subprocess.DEVNULL)


def load_gps(csv_path):
    """Load GPS CSV and parse start datetime. Returns (df, start_dt) or (None, None)."""
    try:
        df = pd.read_csv(csv_path, names=["Time", "Lat", "Lon", "Speed", "Track"])
        if df.empty or df["Time"].isna().all():
            return None, None
        raw_time = df["Time"].iloc[0]
        clean_time = str(raw_time).replace("Z", "").replace(":", "-", 2)
        start_dt = pd.to_datetime(clean_time, utc=True)
        return df, start_dt
    except Exception:
        return None, None