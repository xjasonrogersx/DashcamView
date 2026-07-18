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


MODEL_PATH = "models/EgoLanes_Lite_FP32.onnx"
KMH_TO_MPH = 0.621371
LANE_W = 640
LANE_H = 320

LANE_LEFT_COLOR = (255, 220, 0)
LANE_RIGHT_COLOR = (0, 170, 255)
LANE_FILL_COLOR = (0, 210, 80)
OTHER_LANE_COLOR = (180, 80, 255)
BG_CLASS = 255


def logits_to_class_mask(logits_chw):
	"""Convert EgoLanes 3-channel logits into a class map matching VisionPilot priority.

	Class ids:
	0 = ego-left
	1 = ego-right
	2 = other lanes
	255 = background
	"""
	height, width = logits_chw.shape[1:]
	mask = np.full((height, width), BG_CLASS, dtype=np.uint8)

	c0 = logits_chw[0] > 0.0
	c1 = logits_chw[1] > 0.0
	c2 = logits_chw[2] > 0.0

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


def draw_mask_contours(roi, mask_hw, crop_w, crop_h):
	"""Draw lane class contours over the resized ROI for stronger visual separation."""
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


def mask_bottom_mean_x(mask_hw, class_id, tail_rows=48):
	if mask_hw is None or mask_hw.size == 0:
		return None
	start = max(0, mask_hw.shape[0] - tail_rows)
	ys, xs = np.where(mask_hw[start:] == class_id)
	if len(xs) == 0:
		return None
	return float(np.mean(xs))


class EgoLanesONNX:
	def __init__(self, onnx_path):
		self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
		self.input_name = self.session.get_inputs()[0].name
		self.output_name = self.session.get_outputs()[0].name
		print(f"EgoLanes ONNX ready: provider={self.session.get_providers()[0]}")

	def infer(self, frame_bgr, crop_rect=None):
		x_off, y_off = 0, 0
		proc = frame_bgr
		if crop_rect is not None:
			x_off, y_off, crop_w, crop_h = crop_rect
			proc = frame_bgr[y_off:y_off + crop_h, x_off:x_off + crop_w]
		else:
			crop_h, crop_w = proc.shape[:2]

		resized = cv2.resize(proc, (LANE_W, LANE_H), interpolation=cv2.INTER_LINEAR)
		inp = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
		inp = inp.transpose(2, 0, 1)[None]

		out = self.session.run([self.output_name], {self.input_name: inp})[0]
		logits = out[0]
		class_mask = logits_to_class_mask(logits)
		color_mask = mask_to_color(class_mask)
		overlay = cv2.resize(color_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

		left_bottom_x = mask_bottom_mean_x(class_mask, 0)
		right_bottom_x = mask_bottom_mean_x(class_mask, 1)
		return {
			"crop_rect": crop_rect,
			"class_mask": class_mask,
			"color_overlay": overlay,
			"left_bottom_x": left_bottom_x,
			"right_bottom_x": right_bottom_x,
			"crop_offset": (x_off, y_off),
			"crop_size": (crop_w, crop_h),
		}


def draw_lane_overlay(frame, lane_result):
	if not lane_result:
		return

	x_off, y_off = lane_result["crop_offset"]
	crop_w, crop_h = lane_result["crop_size"]
	roi = frame[y_off:y_off + crop_h, x_off:x_off + crop_w]

	class_mask = lane_result["class_mask"]
	color_overlay = lane_result["color_overlay"]

	left_mask = cv2.resize((class_mask == 0).astype(np.uint8), (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
	right_mask = cv2.resize((class_mask == 1).astype(np.uint8), (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
	other_mask = cv2.resize((class_mask == 2).astype(np.uint8), (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

	overlay = roi.copy()
	overlay[left_mask > 0] = cv2.addWeighted(roi[left_mask > 0], 0.30, color_overlay[left_mask > 0], 0.70, 0)
	overlay[right_mask > 0] = cv2.addWeighted(roi[right_mask > 0], 0.30, color_overlay[right_mask > 0], 0.70, 0)
	overlay[other_mask > 0] = cv2.addWeighted(roi[other_mask > 0], 0.45, color_overlay[other_mask > 0], 0.55, 0)
	cv2.addWeighted(overlay, 0.92, roi, 0.08, 0, roi)

	if np.any(left_mask) and np.any(right_mask):
		corridor = np.zeros((crop_h, crop_w), dtype=np.uint8)
		corridor[np.logical_or(left_mask > 0, right_mask > 0)] = 255
		kernel = np.ones((9, 9), dtype=np.uint8)
		corridor = cv2.morphologyEx(corridor, cv2.MORPH_CLOSE, kernel)
		corridor_bgr = np.zeros_like(roi)
		corridor_bgr[corridor > 0] = LANE_FILL_COLOR
		cv2.addWeighted(corridor_bgr, 0.16, roi, 0.84, 0, roi)

	draw_mask_contours(roi, class_mask, crop_w, crop_h)


def fit_and_draw_lane_polynomials(frame, lane_result):
	"""Fit x=f(y) polynomials for each lane blob and render them."""
	if not lane_result:
		return

	x_off, y_off = lane_result["crop_offset"]
	crop_w, crop_h = lane_result["crop_size"]
	mask_hw = lane_result["class_mask"]

	scale_x = crop_w / float(LANE_W)
	scale_y = crop_h / float(LANE_H)

	class_specs = [
		(0, LANE_LEFT_COLOR, 3),
		(1, LANE_RIGHT_COLOR, 3),
		(2, OTHER_LANE_COLOR, 2),
	]

	for class_id, color, thickness in class_specs:
		class_bin = (mask_hw == class_id).astype(np.uint8) * 255
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

			# Fit x as a function of y (second-order) in model space.
			coeffs = np.polyfit(ys.astype(np.float32), xs.astype(np.float32), 2)
			y_min = int(np.min(ys))
			y_max = int(np.max(ys))
			y_vals = np.arange(y_min, y_max + 1, 3, dtype=np.float32)
			x_vals = np.polyval(coeffs, y_vals)

			poly_pts = []
			for x_m, y_m in zip(x_vals, y_vals):
				if x_m < 0 or x_m >= LANE_W:
					continue
				x_px = int(round(x_off + x_m * scale_x))
				y_px = int(round(y_off + y_m * scale_y))
				if 0 <= x_px < frame.shape[1] and 0 <= y_px < frame.shape[0]:
					poly_pts.append((x_px, y_px))

			if len(poly_pts) >= 2:
				arr = np.array(poly_pts, dtype=np.int32).reshape(-1, 1, 2)
				cv2.polylines(frame, [arr], False, color, thickness, cv2.LINE_AA)
				for p in poly_pts[::12]:
					cv2.circle(frame, p, 2, color, -1, cv2.LINE_AA)


def estimate_lane_center_offset_m(lane_result, frame_w, assumed_lane_width_m=3.7):
	left_bottom_x = lane_result.get("left_bottom_x")
	right_bottom_x = lane_result.get("right_bottom_x")
	if left_bottom_x is None or right_bottom_x is None:
		return None

	x_off, _ = lane_result["crop_offset"]
	crop_w, _ = lane_result["crop_size"]
	scale_x = crop_w / float(LANE_W)
	lane_center_x = x_off + 0.5 * (left_bottom_x + right_bottom_x) * scale_x
	px_offset = lane_center_x - frame_w * 0.5
	return (px_offset / max(frame_w * 0.5, 1.0)) * (assumed_lane_width_m * 0.5)


def draw_overlay(frame, lane_result, ego_kmh, gps_time_display, elapsed_display, file_label, view_mode):
	fh, fw = frame.shape[:2]
	ego_mph = ego_kmh * KMH_TO_MPH
	lane_offset_m = estimate_lane_center_offset_m(lane_result, fw) if lane_result else None
	lane_offset_txt = "N/A" if lane_offset_m is None else f"{lane_offset_m:+.2f} m"

	point_count = 0
	if lane_result:
		valid_mask = lane_result["class_mask"] != BG_CLASS
		point_count = int(np.count_nonzero(valid_mask))

	cv2.rectangle(frame, (0, 0), (fw, 100), (0, 0, 0), -1)
	cv2.putText(frame, f"EGO SPEED: {ego_mph:5.1f} mph", (20, 34),
				cv2.FONT_HERSHEY_DUPLEX, 1.0, (80, 255, 80), 2, cv2.LINE_AA)
	cv2.putText(frame, f"LANE OFFSET: {lane_offset_txt}  PATH PTS: {point_count:03d}", (20, 64),
				cv2.FONT_HERSHEY_SIMPLEX, 0.72, (220, 220, 220), 2, cv2.LINE_AA)
	cv2.putText(frame, f"TIME: {gps_time_display}  ELAPSED: {elapsed_display}", (20, 90),
				cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 190, 190), 1, cv2.LINE_AA)
	cv2.putText(frame, f"VIEW: {view_mode}", (fw - 220, 90),
				cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2, cv2.LINE_AA)
	cv2.putText(frame, file_label, (fw - 720, 38), cv2.FONT_HERSHEY_SIMPLEX,
				0.8, (200, 200, 0), 2, cv2.LINE_AA)


def play_video(video_path, total_files, idx, egolanes, screen_w, screen_h):
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
	show_fit_polys = False

	win = "EgoLanes Player"
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

		lane_result = egolanes.infer(frame, crop_rect=crop_rect)
		if show_fit_polys:
			fit_and_draw_lane_polynomials(frame, lane_result)
			view_mode = "POLY FIT"
		else:
			draw_lane_overlay(frame, lane_result)
			view_mode = "MASK"

		file_label = f"{idx}/{total_files}: {os.path.basename(video_path)}"
		draw_overlay(frame, lane_result, ego_kmh, gps_time_display, elapsed, file_label, view_mode)

		disp = fit_frame(frame, screen_w, screen_h)
		cv2.imshow(win, disp)

		key = cv2.waitKey(delay_ms) & 0xFF
		if key == ord("q"):
			cap.release()
			return False
		if key == ord("n"):
			break
		if key == ord("f"):
			show_fit_polys = not show_fit_polys

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
		print(f"Missing EgoLanes model: {model_path}")
		return

	print(f"Found {len(videos)} MP4 file(s).")
	print("Controls: [Q] Quit  [N] Next video  [F] Toggle mask/poly-fit view")
	print("Reference: https://github.com/autowarefoundation/vision_pilot/tree/e45165837e847f2ca5e5df5247cb4167379ecfc7/Models/visualizations/EgoLanes")

	egolanes = EgoLanesONNX(model_path)
	screen_w, screen_h = get_screen_resolution()
	print(f"Screen resolution: {screen_w}x{screen_h}")

	for i, video_path in enumerate(videos, start=1):
		if not play_video(video_path, len(videos), i, egolanes, screen_w, screen_h):
			break

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
