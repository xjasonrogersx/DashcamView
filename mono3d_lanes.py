#!/usr/bin/env python3

import os
import tempfile
from datetime import timedelta

import cv2
import numpy as np

from common import (
	discover_input_videos,
	draw_crop_box,
	extract_gps,
	fit_frame,
	get_lower_2to1_crop,
	get_screen_resolution,
	load_gps,
)
from egolanes import (
	BG_CLASS,
	EgoLanesONNX,
	draw_lane_overlay,
	estimate_lane_center_offset_m,
	fit_and_draw_lane_polynomials,
)
from mono3d import Scene3DONNX, render_depth_view


SCENE3D_MODEL_PATH = "models/Scene3D_Lite_FP32.onnx"
EGOLANES_MODEL_PATH = "models/EgoLanes_Lite_FP32.onnx"
KMH_TO_MPH = 0.621371


def draw_combined_overlay(
	frame,
	depth_result,
	lane_result,
	ego_kmh,
	gps_time_display,
	elapsed_display,
	file_label,
	depth_view_mode,
	lane_view_mode,
):
	fh, fw = frame.shape[:2]
	ego_mph = ego_kmh * KMH_TO_MPH

	depth_txt = "DEPTH: N/A"
	if depth_result is not None:
		stats = depth_result["stats"]
		depth_txt = (
			f"DEPTH min {stats['min']:.2f}  mean {stats['mean']:.2f}  "
			f"max {stats['max']:.2f}  center {stats['center']:.2f}"
		)

	lane_offset_txt = "N/A"
	point_count = 0
	if lane_result is not None:
		lane_offset_m = estimate_lane_center_offset_m(lane_result, fw)
		if lane_offset_m is not None:
			lane_offset_txt = f"{lane_offset_m:+.2f} m"
		valid_mask = lane_result["class_mask"] != BG_CLASS
		point_count = int(np.count_nonzero(valid_mask))

	depth_mode_txt = "OVERLAY" if depth_view_mode == "overlay" else "DEPTH"
	lane_mode_txt = "POLY FIT" if lane_view_mode == "poly" else "MASK"

	bar_h = 130
	cv2.rectangle(frame, (0, 0), (fw, bar_h), (0, 0, 0), -1)
	cv2.putText(frame, f"EGO SPEED: {ego_mph:5.1f} mph", (20, 34),
				cv2.FONT_HERSHEY_DUPLEX, 1.0, (80, 255, 80), 2, cv2.LINE_AA)
	cv2.putText(frame, depth_txt, (20, 62),
				cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2, cv2.LINE_AA)
	cv2.putText(frame, f"LANE OFFSET: {lane_offset_txt}  PATH PTS: {point_count:03d}", (20, 90),
				cv2.FONT_HERSHEY_SIMPLEX, 0.68, (220, 220, 220), 2, cv2.LINE_AA)
	cv2.putText(frame, f"TIME: {gps_time_display}  ELAPSED: {elapsed_display}", (20, 118),
				cv2.FONT_HERSHEY_SIMPLEX, 0.56, (190, 190, 190), 1, cv2.LINE_AA)
	cv2.putText(frame, f"VIEW: D={depth_mode_txt}  L={lane_mode_txt}", (fw - 430, 36),
				cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 220, 0), 2, cv2.LINE_AA)
	cv2.putText(frame, file_label, (fw - 720, 68), cv2.FONT_HERSHEY_SIMPLEX,
				0.75, (200, 200, 0), 2, cv2.LINE_AA)


def play_video(video_path, total_files, idx, scene3d, egolanes, screen_w, screen_h, writer=None):
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
	depth_view_mode = "overlay"
	lane_view_mode = "mask"

	win = "Mono3D + EgoLanes Player"
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
		lane_result = egolanes.infer(frame, crop_rect=crop_rect)

		display_frame = render_depth_view(frame, depth_result, depth_view_mode)
		if lane_view_mode == "poly":
			fit_and_draw_lane_polynomials(display_frame, lane_result)
		else:
			draw_lane_overlay(display_frame, lane_result)

		file_label = f"{idx}/{total_files}: {os.path.basename(video_path)}"
		draw_combined_overlay(
			display_frame,
			depth_result,
			lane_result,
			ego_kmh,
			gps_time_display,
			elapsed,
			file_label,
			depth_view_mode,
			lane_view_mode,
		)

		if writer is not None:
			writer.write(display_frame)

		disp = fit_frame(display_frame, screen_w, screen_h)
		cv2.imshow(win, disp)

		key = cv2.waitKey(delay_ms) & 0xFF
		if key == ord("q"):
			cap.release()
			return False
		if key == ord("n"):
			break
		if key == ord("v"):
			depth_view_mode = "depth" if depth_view_mode == "overlay" else "overlay"
		if key == ord("f"):
			lane_view_mode = "poly" if lane_view_mode == "mask" else "mask"

	cap.release()
	return True


def main():
	cwd = os.getcwd()
	videos = discover_input_videos(cwd)
	if not videos:
		print("No source MP4 files found. Place videos in current directory or camera/.")
		return

	scene3d_model_path = os.path.join(cwd, SCENE3D_MODEL_PATH)
	egolanes_model_path = os.path.join(cwd, EGOLANES_MODEL_PATH)

	if not os.path.exists(scene3d_model_path):
		print(f"Missing Scene3D model: {scene3d_model_path}")
		return
	if not os.path.exists(egolanes_model_path):
		print(f"Missing EgoLanes model: {egolanes_model_path}")
		return

	print(f"Found {len(videos)} MP4 file(s).")
	print("Controls: [Q] Quit  [N] Next video  [V] Toggle depth overlay/full  [F] Toggle lane mask/poly-fit")

	probe = cv2.VideoCapture(videos[0])
	out_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
	out_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
	out_fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
	probe.release()

	out_path = os.path.join(cwd, "mono3d_lanes.m4v")
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))
	if not writer.isOpened():
		print(f"Failed to open output writer: {out_path}")
		return
	print(f"Writing processed video to: {out_path}")

	scene3d = Scene3DONNX(scene3d_model_path)
	egolanes = EgoLanesONNX(egolanes_model_path)
	screen_w, screen_h = get_screen_resolution()
	print(f"Screen resolution: {screen_w}x{screen_h}")

	try:
		for i, video_path in enumerate(videos, start=1):
			if not play_video(video_path, len(videos), i, scene3d, egolanes, screen_w, screen_h, writer=writer):
				break
	finally:
		writer.release()
		print(f"Saved: {out_path}")

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
