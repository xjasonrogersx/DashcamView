
#!/usr/bin/env python3

import cv2
import os
import math
import tempfile
import numpy as np
from datetime import timedelta
from collections import deque
import onnxruntime as ort
from common import (
	discover_input_videos,
	draw_crop_box,
	extract_gps,
	fit_frame,
	get_lower_2to1_crop,
	get_screen_resolution,
	load_gps,
)


MODEL_PATH = "models/autospeed.onnx"
CONF_THRESH = 0.60
IOU_THRESH = 0.45
KMH_TO_MPH = 0.621371

# Assumed object heights for rough relative speed estimation.
CLASS_HEIGHT_M = {
	0: 1.5,
	1: 1.5,
	2: 1.5,
	3: 2.8,
}

SPEED_HISTORY = 10

def resize_letterbox(img, target_size):
	target_w, target_h = target_size
	h, w = img.shape[:2]

	scale = min(target_w / w, target_h / h)
	new_w = int(w * scale)
	new_h = int(h * scale)

	resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
	out = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

	pad_x = (target_w - new_w) // 2
	pad_y = (target_h - new_h) // 2
	out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
	return out, scale, pad_x, pad_y


def xywh2xyxy(boxes):
	out = boxes.copy()
	out[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
	out[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
	out[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
	out[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
	return out


def nms(boxes, scores, iou_thresh):
	if len(boxes) == 0:
		return []
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	areas = (x2 - x1) * (y2 - y1)
	order = scores.argsort()[::-1]
	keep = []

	while order.size > 0:
		i = order[0]
		keep.append(i)

		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])

		w = np.maximum(0.0, xx2 - xx1)
		h = np.maximum(0.0, yy2 - yy1)
		inter = w * h
		ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
		inds = np.where(ovr <= iou_thresh)[0]
		order = order[inds + 1]

	return keep


class AutoSpeedONNX:
	def __init__(self, onnx_path):
		providers = ["CPUExecutionProvider"]
		self.session = ort.InferenceSession(onnx_path, providers=providers)
		self.input_name = self.session.get_inputs()[0].name
		input_shape = self.session.get_inputs()[0].shape

		# Expect NCHW; fallback to canonical 1024x512 if dynamic/unknown.
		if len(input_shape) == 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
			self.net_h = int(input_shape[2])
			self.net_w = int(input_shape[3])
		else:
			self.net_h = 512
			self.net_w = 1024

		print(f"AutoSpeed ONNX ready: input={self.net_w}x{self.net_h} provider={self.session.get_providers()[0]}")

	def infer(self, frame_bgr, conf_thres=CONF_THRESH, iou_thres=IOU_THRESH, crop_rect=None):
		x_off, y_off = 0, 0
		proc = frame_bgr
		if crop_rect is not None:
			x_off, y_off, cw, ch = crop_rect
			proc = frame_bgr[y_off:y_off + ch, x_off:x_off + cw]

		h, w = proc.shape[:2]
		img_lb, scale, pad_x, pad_y = resize_letterbox(proc, (self.net_w, self.net_h))

		img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
		inp = np.transpose(img_rgb, (2, 0, 1))[None]

		raw = self.session.run(None, {self.input_name: inp})[0]
		if raw.ndim != 3 or raw.shape[0] != 1:
			return []

		pred = raw[0].T  # [N, C]
		if pred.shape[1] < 5:
			return []

		boxes_xywh = pred[:, :4]
		class_logits = pred[:, 4:]
		class_probs = 1.0 / (1.0 + np.exp(-class_logits))
		scores = np.max(class_probs, axis=1)
		class_ids = np.argmax(class_probs, axis=1)

		mask = scores > conf_thres
		if mask.sum() == 0:
			return []

		boxes_xyxy = xywh2xyxy(boxes_xywh[mask])
		scores_f = scores[mask]
		class_ids_f = class_ids[mask]

		keep = nms(boxes_xyxy, scores_f, iou_thres)
		out = []
		for k in keep:
			x1, y1, x2, y2 = boxes_xyxy[k]
			x1 = (x1 - pad_x) / scale
			y1 = (y1 - pad_y) / scale
			x2 = (x2 - pad_x) / scale
			y2 = (y2 - pad_y) / scale

			x1 = int(max(0, min(w - 1, x1)))
			y1 = int(max(0, min(h - 1, y1)))
			x2 = int(max(0, min(w - 1, x2)))
			y2 = int(max(0, min(h - 1, y2)))
			if x2 <= x1 or y2 <= y1:
				continue

			out.append((x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off,
						int(class_ids_f[k]), float(scores_f[k])))
		return out


class Tracker:
	def __init__(self, max_age=10):
		self.max_age = max_age
		self.next_id = 0
		self.tracks = {}

	@staticmethod
	def iou(a, b):
		ax1, ay1, ax2, ay2 = a
		bx1, by1, bx2, by2 = b
		ix1, iy1 = max(ax1, bx1), max(ay1, by1)
		ix2, iy2 = min(ax2, bx2), min(ay2, by2)
		inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
		if inter <= 0:
			return 0.0
		aa = (ax2 - ax1) * (ay2 - ay1)
		ab = (bx2 - bx1) * (by2 - by1)
		return inter / (aa + ab - inter + 1e-6)

	def update(self, detections, ts_s):
		for tid in list(self.tracks.keys()):
			self.tracks[tid]["age"] += 1
			if self.tracks[tid]["age"] > self.max_age:
				del self.tracks[tid]

		used = set()
		out = []
		for x1, y1, x2, y2, cls_id, conf in detections:
			best_tid = None
			best = 0.3
			for tid, tr in self.tracks.items():
				if tid in used or tr["class_id"] != cls_id:
					continue
				ov = self.iou((x1, y1, x2, y2), tr["bbox"])
				if ov > best:
					best = ov
					best_tid = tid

			if best_tid is None:
				best_tid = self.next_id
				self.next_id += 1
				self.tracks[best_tid] = {
					"bbox": (x1, y1, x2, y2),
					"class_id": cls_id,
					"age": 0,
					"history": deque(maxlen=SPEED_HISTORY + 2),
				}

			tr = self.tracks[best_tid]
			tr["bbox"] = (x1, y1, x2, y2)
			tr["age"] = 0
			tr["history"].append((y2 - y1, ts_s, cls_id))
			used.add(best_tid)
			out.append((x1, y1, x2, y2, cls_id, conf, best_tid))
		return out


def estimate_abs_speed_kmh(track_hist, frame_w, ego_kmh):
	if len(track_hist) < 2:
		return None
	focal_px = frame_w / 2.0

	rel_ms = []
	for i in range(1, len(track_hist)):
		h1, t1, c1 = track_hist[i - 1]
		h2, t2, c2 = track_hist[i]
		dt = t2 - t1
		if dt <= 0 or h1 <= 0 or h2 <= 0:
			continue
		real_h = CLASS_HEIGHT_M.get(c2, 1.5)
		d1 = (focal_px * real_h) / h1
		d2 = (focal_px * real_h) / h2
		rel_ms.append((d2 - d1) / dt)

	if not rel_ms:
		return None
	avg_rel = sum(rel_ms) / len(rel_ms)
	return max(0.0, ego_kmh + avg_rel * 3.6)


def class_name(cls_id):
	return f"L{cls_id}"


def draw_overlay(frame, tracked, tracks_ref, ego_kmh, gps_time_display, elapsed_display):
	fh, fw = frame.shape[:2]
	ego_mph = ego_kmh * KMH_TO_MPH

	cv2.rectangle(frame, (0, 0), (fw, 74), (0, 0, 0), -1)
	cv2.putText(frame, f"EGO SPEED: {ego_mph:5.1f} mph", (20, 34),
				cv2.FONT_HERSHEY_DUPLEX, 1.0, (80, 255, 80), 2, cv2.LINE_AA)
	cv2.putText(frame, f"TIME: {gps_time_display}  ELAPSED: {elapsed_display}", (20, 64),
				cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1, cv2.LINE_AA)

	for x1, y1, x2, y2, cls_id, conf, tid in tracked:
		clr = (0, 180, 255)
		spd_txt = "---"
		tr = tracks_ref.get(tid)
		if tr is not None and len(tr["history"]) >= 2:
			spd_kmh = estimate_abs_speed_kmh(tr["history"], fw, ego_kmh)
			if spd_kmh is not None:
				spd_txt = f"{spd_kmh * KMH_TO_MPH:.0f}mph"

		cv2.rectangle(frame, (x1, y1), (x2, y2), clr, 2)
		label = f"{class_name(cls_id)} {spd_txt}  {conf*100:.0f}%  ID{tid}"
		(tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
		ly = max(y1 - 6, th + 6)
		cv2.rectangle(frame, (x1, ly - th - 6), (x1 + tw + 6, ly + 2), clr, -1)
		cv2.putText(frame, label, (x1 + 3, ly - 3), cv2.FONT_HERSHEY_SIMPLEX,
					0.55, (0, 0, 0), 2, cv2.LINE_AA)


def play_video(video_path, total_files, idx, autospeed, screen_w, screen_h):
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

	cap = cv2.VideoCapture(video_path)
	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	delay_ms = max(1, int(1000 / fps))

	tracker = Tracker()

	win = "AutoSpeed Player"
	cv2.namedWindow(win, cv2.WINDOW_NORMAL)
	cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		fh, fw = frame.shape[:2]
		crop_rect = get_lower_2to1_crop(fw, fh)
		draw_crop_box(frame, crop_rect)

		frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
		t_s = frame_idx / fps
		total_ms = t_s * 1000
		hh = int(total_ms / 3600000)
		mm = int((total_ms / 60000) % 60)
		ss = int((total_ms / 1000) % 60)
		ms = int(total_ms % 1000)
		elapsed = f"{hh:02}:{mm:02}:{ss:02}.{ms:03}"

		if start_dt is not None and df is not None and len(df) > 0:
			gps_dt = start_dt + timedelta(seconds=t_s)
			gps_time_display = gps_dt.strftime("%D %H:%M:%S.%f")[:-3]

			idx_floor = min(int(t_s), len(df) - 1)
			idx_ceil = min(idx_floor + 1, len(df) - 1)
			frac = t_s - int(t_s)
			row_curr = df.iloc[idx_floor]
			row_next = df.iloc[idx_ceil]
			ego_kmh = row_curr["Speed"] + frac * (row_next["Speed"] - row_curr["Speed"])
		else:
			gps_time_display = "N/A"
			ego_kmh = 0.0

		dets = autospeed.infer(frame, crop_rect=crop_rect)
		tracked = tracker.update(dets, t_s)

		draw_overlay(frame, tracked, tracker.tracks, ego_kmh, gps_time_display, elapsed)

		disp = fit_frame(frame, screen_w, screen_h)
		cv2.imshow(win, disp)

		key = cv2.waitKey(delay_ms) & 0xFF
		if key == ord("q"):
			cap.release()
			return False
		if key == ord("n"):
			break

	cap.release()
	return True


def main():
	cwd = os.getcwd()
	videos = discover_input_videos(cwd)
	if not videos:
		print("No source MP4 files found. Place videos in current directory or camera/.")
		return

	model_path = os.path.join(cwd, MODEL_PATH)
	if not os.path.exists(model_path):
		print(f"Missing AutoSpeed model: {model_path}")
		return

	print(f"Found {len(videos)} MP4 file(s).")
	print("Controls: [Q] Quit  [N] Next video")
	print("Reference: https://github.com/autowarefoundation/auto_speed")

	autospeed = AutoSpeedONNX(model_path)
	screen_w, screen_h = get_screen_resolution()
	print(f"Screen resolution: {screen_w}x{screen_h}")

	for i, vp in enumerate(videos, start=1):
		if not play_video(vp, len(videos), i, autospeed, screen_w, screen_h):
			break

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
