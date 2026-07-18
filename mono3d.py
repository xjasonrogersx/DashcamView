#!/usr/bin/env python3

import os
import tempfile
from datetime import timedelta

import cv2
import numpy as np
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


MODEL_PATH = "models/Scene3D_Lite_FP32.onnx"
KMH_TO_MPH = 0.621371
DEFAULT_VIEW_MODE = "overlay"


def normalize_depth(depth_map):
	finite = np.isfinite(depth_map)
	if not np.any(finite):
		return np.zeros(depth_map.shape, dtype=np.uint8), {
			"min": 0.0,
			"max": 0.0,
			"mean": 0.0,
			"center": 0.0,
		}

	valid = depth_map[finite].astype(np.float32)
	lo = float(np.percentile(valid, 2.0))
	hi = float(np.percentile(valid, 98.0))
	if not np.isfinite(lo):
		lo = float(np.min(valid))
	if not np.isfinite(hi):
		hi = float(np.max(valid))
	if hi <= lo:
		hi = lo + 1e-6

	clipped = np.clip(depth_map, lo, hi)
	norm = ((clipped - lo) * (255.0 / (hi - lo))).astype(np.uint8)

	ch = depth_map.shape[0] // 2
	cw = depth_map.shape[1] // 2
	cy0 = max(0, ch - 8)
	cy1 = min(depth_map.shape[0], ch + 8)
	cx0 = max(0, cw - 8)
	cx1 = min(depth_map.shape[1], cw + 8)
	center_patch = depth_map[cy0:cy1, cx0:cx1]
	center_valid = center_patch[np.isfinite(center_patch)]
	center_val = float(np.median(center_valid)) if center_valid.size else 0.0

	stats = {
		"min": float(np.min(valid)),
		"max": float(np.max(valid)),
		"mean": float(np.mean(valid)),
		"center": center_val,
	}
	return norm, stats


def colorize_depth(depth_map):
	norm, stats = normalize_depth(depth_map)
	color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
	return color, norm, stats


class Scene3DONNX:
	def __init__(self, onnx_path):
		self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
		self.input_name = self.session.get_inputs()[0].name
		self.output_name = self.session.get_outputs()[0].name
		input_shape = self.session.get_inputs()[0].shape

		if len(input_shape) == 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
			self.net_h = int(input_shape[2])
			self.net_w = int(input_shape[3])
		else:
			self.net_h = 512
			self.net_w = 1024

		print(
			f"Scene3D ONNX ready: input={self.net_w}x{self.net_h} "
			f"provider={self.session.get_providers()[0]} output={self.output_name}"
		)

	def infer(self, frame_bgr, crop_rect=None):
		x_off, y_off = 0, 0
		proc = frame_bgr
		if crop_rect is not None:
			x_off, y_off, crop_w, crop_h = crop_rect
			proc = frame_bgr[y_off:y_off + crop_h, x_off:x_off + crop_w]
		else:
			crop_h, crop_w = proc.shape[:2]

		resized = cv2.resize(proc, (self.net_w, self.net_h), interpolation=cv2.INTER_LINEAR)
		img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
		inp = np.transpose(img_rgb, (2, 0, 1))[None]

		out = self.session.run([self.output_name], {self.input_name: inp})[0]
		depth = np.asarray(out).squeeze().astype(np.float32)
		if depth.ndim != 2:
			return None

		depth_resized = cv2.resize(depth, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
		depth_color, depth_norm, stats = colorize_depth(depth_resized)

		return {
			"depth_map": depth_resized,
			"depth_color": depth_color,
			"depth_norm": depth_norm,
			"stats": stats,
			"crop_offset": (x_off, y_off),
			"crop_size": (crop_w, crop_h),
		}


def render_depth_view(frame, depth_result, view_mode):
	if depth_result is None:
		return frame

	out = frame.copy()
	x_off, y_off = depth_result["crop_offset"]
	crop_w, crop_h = depth_result["crop_size"]
	depth_color = depth_result["depth_color"]

	roi = out[y_off:y_off + crop_h, x_off:x_off + crop_w]
	if view_mode == "depth":
		roi[:] = depth_color
	else:
		cv2.addWeighted(depth_color, 0.62, roi, 0.38, 0, roi)
		mini_w = min(320, max(160, out.shape[1] // 4))
		mini_h = max(90, int(round(mini_w * crop_h / max(crop_w, 1))))
		mini = cv2.resize(depth_color, (mini_w, mini_h), interpolation=cv2.INTER_LINEAR)
		x1 = max(12, out.shape[1] - mini_w - 20)
		y1 = max(112, out.shape[0] - mini_h - 20)
		cv2.rectangle(out, (x1 - 4, y1 - 26), (x1 + mini_w + 4, y1 + mini_h + 4), (20, 20, 20), -1)
		cv2.putText(out, "Depth", (x1 + 4, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
					0.62, (230, 230, 230), 2, cv2.LINE_AA)
		out[y1:y1 + mini_h, x1:x1 + mini_w] = mini

	return out


def draw_overlay(frame, depth_result, ego_kmh, gps_time_display, elapsed_display, file_label, view_mode):
	fh, fw = frame.shape[:2]
	ego_mph = ego_kmh * KMH_TO_MPH

	depth_txt = "DEPTH: N/A"
	if depth_result is not None:
		stats = depth_result["stats"]
		depth_txt = (
			f"DEPTH min {stats['min']:.2f}  mean {stats['mean']:.2f}  "
			f"max {stats['max']:.2f}  center {stats['center']:.2f}"
		)

	mode_txt = "OVERLAY" if view_mode == "overlay" else "DEPTH"
	bar_h = 104
	cv2.rectangle(frame, (0, 0), (fw, bar_h), (0, 0, 0), -1)
	cv2.putText(frame, f"EGO SPEED: {ego_mph:5.1f} mph", (20, 34),
				cv2.FONT_HERSHEY_DUPLEX, 1.0, (80, 255, 80), 2, cv2.LINE_AA)
	cv2.putText(frame, depth_txt, (20, 64),
				cv2.FONT_HERSHEY_SIMPLEX, 0.64, (220, 220, 220), 2, cv2.LINE_AA)
	cv2.putText(frame, f"TIME: {gps_time_display}  ELAPSED: {elapsed_display}", (20, 92),
				cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 190, 190), 1, cv2.LINE_AA)
	cv2.putText(frame, f"VIEW: {mode_txt}", (fw - 240, 34), cv2.FONT_HERSHEY_SIMPLEX,
				0.8, (255, 220, 0), 2, cv2.LINE_AA)
	cv2.putText(frame, file_label, (fw - 720, 66), cv2.FONT_HERSHEY_SIMPLEX,
				0.8, (200, 200, 0), 2, cv2.LINE_AA)


def play_video(video_path, total_files, idx, scene3d, screen_w, screen_h):
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
	view_mode = DEFAULT_VIEW_MODE

	win = "Scene3D Player"
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

		depth_result = scene3d.infer(frame, crop_rect=crop_rect)
		display_frame = render_depth_view(frame, depth_result, view_mode)

		file_label = f"{idx}/{total_files}: {os.path.basename(video_path)}"
		draw_overlay(display_frame, depth_result, ego_kmh, gps_time_display, elapsed, file_label, view_mode)

		disp = fit_frame(display_frame, screen_w, screen_h)
		cv2.imshow(win, disp)

		key = cv2.waitKey(delay_ms) & 0xFF
		if key == ord("q"):
			cap.release()
			return False
		if key == ord("n"):
			break
		if key == ord("v"):
			view_mode = "depth" if view_mode == "overlay" else "overlay"

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
		print(f"Missing Scene3D model: {model_path}")
		return

	print(f"Found {len(videos)} MP4 file(s).")
	print("Controls: [Q] Quit  [N] Next video  [V] Toggle overlay/depth")
	print("Reference: https://github.com/autowarefoundation/vision_pilot/tree/e45165837e847f2ca5e5df5247cb4167379ecfc7/Models/visualizations/Scene3D")

	scene3d = Scene3DONNX(model_path)
	screen_w, screen_h = get_screen_resolution()
	print(f"Screen resolution: {screen_w}x{screen_h}")

	for i, video_path in enumerate(videos, start=1):
		if not play_video(video_path, len(videos), i, scene3d, screen_w, screen_h):
			break

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
