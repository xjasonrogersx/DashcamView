

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


MODEL_PATH = "models/autosteer_2.onnx"
KMH_TO_MPH = 0.621371
PATH_POINT_COUNT = 64
VIS_CONF_THRESH = 0.5


def resize_for_model(img, target_size):
	target_w, target_h = target_size
	return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


class AutoSteerONNX:
	def __init__(self, onnx_path):
		providers = ["CPUExecutionProvider"]
		self.session = ort.InferenceSession(onnx_path, providers=providers)
		self.input_name = self.session.get_inputs()[0].name
		self.output_names = [out.name for out in self.session.get_outputs()]
		input_shape = self.session.get_inputs()[0].shape

		if len(input_shape) == 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
			self.net_h = int(input_shape[2])
			self.net_w = int(input_shape[3])
		else:
			self.net_h = 512
			self.net_w = 1024

		print(
			f"AutoSteer ONNX ready: input={self.net_w}x{self.net_h} "
			f"provider={self.session.get_providers()[0]} outputs={self.output_names}"
		)

	def infer(self, frame_bgr, crop_rect=None):
		x_off, y_off = 0, 0
		proc = frame_bgr
		if crop_rect is not None:
			x_off, y_off, crop_w, crop_h = crop_rect
			proc = frame_bgr[y_off:y_off + crop_h, x_off:x_off + crop_w]
		else:
			crop_h, crop_w = proc.shape[:2]

		resized = resize_for_model(proc, (self.net_w, self.net_h))
		img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
		inp = np.transpose(img_rgb, (2, 0, 1))[None]

		outputs = self.session.run(None, {self.input_name: inp})
		if len(outputs) < 2:
			return None

		xp = np.asarray(outputs[0]).squeeze().astype(np.float32)
		h_vector = np.asarray(outputs[1]).squeeze().astype(np.float32)

		xp = xp.reshape(-1)
		h_vector = h_vector.reshape(-1)
		if xp.size != PATH_POINT_COUNT or h_vector.size != PATH_POINT_COUNT:
			return None

		y_model = np.linspace(0, self.net_h - 1, PATH_POINT_COUNT, dtype=np.float32)
		scale_x = crop_w / float(self.net_w)
		scale_y = crop_h / float(self.net_h)

		points = []
		for x_norm, y_net, conf in zip(xp, y_model, h_vector):
			if conf < VIS_CONF_THRESH:
				continue
			x_crop = float(x_norm) * self.net_w * scale_x
			y_crop = float(y_net) * scale_y
			x_full = int(round(x_crop + x_off))
			y_full = int(round(y_crop + y_off))
			points.append((x_full, y_full, float(conf)))

		confidence = float(np.mean(h_vector)) if h_vector.size > 0 else 0.0
		return {
			"xp": xp,
			"h_vector": h_vector,
			"points": points,
			"confidence": confidence,
			"crop_rect": crop_rect,
		}


def estimate_path_offset_m(path_points, frame_w, lane_width_m=3.7):
	if not path_points:
		return None
	bottom_points = sorted(path_points, key=lambda item: item[1], reverse=True)[:8]
	if not bottom_points:
		return None
	avg_x = sum(p[0] for p in bottom_points) / len(bottom_points)
	px_offset = avg_x - frame_w * 0.5
	metres_per_half_frame = lane_width_m * 0.5
	return (px_offset / max(frame_w * 0.5, 1.0)) * metres_per_half_frame


def draw_path_overlay(frame, steer_result):
	if not steer_result or not steer_result["points"]:
		return

	poly = np.array([(x, y) for x, y, _ in steer_result["points"]], dtype=np.int32).reshape(-1, 1, 2)
	if len(poly) >= 2:
		cv2.polylines(frame, [poly], False, (0, 255, 0), 3, cv2.LINE_AA)

	for x, y, conf in steer_result["points"]:
		radius = 3 if conf < 0.75 else 4
		cv2.circle(frame, (x, y), radius, (0, 255, 0), -1, cv2.LINE_AA)


def draw_overlay(frame, steer_result, ego_kmh, gps_time_display, elapsed_display, file_label):
	fh, fw = frame.shape[:2]
	ego_mph = ego_kmh * KMH_TO_MPH
	cv2.rectangle(frame, (0, 0), (fw, 100), (0, 0, 0), -1)

	conf_pct = 0.0
	cte_m = None
	point_count = 0
	if steer_result is not None:
		conf_pct = steer_result["confidence"] * 100.0
		point_count = len(steer_result["points"])
		cte_m = estimate_path_offset_m(steer_result["points"], fw)

	cte_txt = "N/A" if cte_m is None else f"{cte_m:+.2f} m"
	cv2.putText(frame, f"EGO SPEED: {ego_mph:5.1f} mph", (20, 34),
				cv2.FONT_HERSHEY_DUPLEX, 1.0, (80, 255, 80), 2, cv2.LINE_AA)
	cv2.putText(frame, f"PATH CONF: {conf_pct:4.0f}%  POINTS: {point_count:02d}  OFFSET: {cte_txt}", (20, 64),
				cv2.FONT_HERSHEY_SIMPLEX, 0.72, (220, 220, 220), 2, cv2.LINE_AA)
	cv2.putText(frame, f"TIME: {gps_time_display}  ELAPSED: {elapsed_display}", (20, 90),
				cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 190, 190), 1, cv2.LINE_AA)

	cv2.putText(frame, file_label, (fw - 720, 38), cv2.FONT_HERSHEY_SIMPLEX,
				0.8, (200, 200, 0), 2, cv2.LINE_AA)


def play_video(video_path, total_files, idx, autosteer, screen_w, screen_h):
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

	win = "AutoSteer Player"
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

		steer_result = autosteer.infer(frame, crop_rect=crop_rect)
		draw_path_overlay(frame, steer_result)

		file_label = f"{idx}/{total_files}: {os.path.basename(video_path)}"
		draw_overlay(frame, steer_result, ego_kmh, gps_time_display, elapsed, file_label)

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
		print(f"Missing AutoSteer model: {model_path}")
		return

	print(f"Found {len(videos)} MP4 file(s).")
	print("Controls: [Q] Quit  [N] Next video")
	print("Reference: https://github.com/autowarefoundation/auto_steer")

	autosteer = AutoSteerONNX(model_path)
	screen_w, screen_h = get_screen_resolution()
	print(f"Screen resolution: {screen_w}x{screen_h}")

	for i, vp in enumerate(videos, start=1):
		if not play_video(vp, len(videos), i, autosteer, screen_w, screen_h):
			break

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()