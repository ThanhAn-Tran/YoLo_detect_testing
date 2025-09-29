"""Minimal real-time YOLO NCNN demo using OpenCV windows."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
from ultralytics import YOLO


def parse_resolution(value: str) -> Tuple[int, int]:
    try:
        w_str, h_str = value.lower().split("x", 1)
        width, height = int(w_str), int(h_str)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError("Use format <width>x<height>, e.g. 640x480") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution values must be positive")
    return width, height


def as_camera_index(source: str) -> Optional[int]:
    try:
        return int(source)
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time YOLO NCNN demo")
    parser.add_argument("--model", required=True, help="Path to Ultralytics NCNN model directory")
    parser.add_argument("--source", default="0", help="Camera index or video file path")
    parser.add_argument("--resolution", default="640x480", help="Camera resolution, e.g. 640x480")
    parser.add_argument("--save", action="store_true", help="Save annotated output")
    return parser.parse_args()


def annotate(model: YOLO, frame, imgsz: int):
    results = model.predict(frame, imgsz=imgsz, device="cpu", verbose=False)
    return results[0].plot()


def draw_fps(frame, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def run_stream(model: YOLO, capture: cv2.VideoCapture, save: bool, label: str) -> None:
    if not capture.isOpened():
        raise RuntimeError("Unable to open video source")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps_in = capture.get(cv2.CAP_PROP_FPS) or 30
    imgsz = max(width, height)

    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = Path("runs") / f"demo_{label}_{timestamp}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps_in, (width, height))
        print(f"[INFO] Saving video to {output_path}")

    window = "YOLO NCNN"
    print("Press ESC to exit.")
    prev = time.time()
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        now = time.time()
        dt = now - prev
        prev = now
        fps = 1.0 / dt if dt > 0 else 0.0

        annotated = annotate(model, frame, imgsz)
        draw_fps(annotated, fps)

        if writer is not None:
            writer.write(annotated)

        cv2.imshow(window, annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if writer is not None:
        writer.release()
    capture.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    width, height = parse_resolution(args.resolution)

    print(f"[INFO] Loading NCNN model from: {args.model}")
    model = YOLO(args.model)

    camera_index = as_camera_index(args.source)
    if camera_index is not None and not Path(args.source).exists():
        capture = cv2.VideoCapture(camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        run_stream(model, capture, args.save, f"camera{camera_index}")
    else:
        capture = cv2.VideoCapture(args.source)
        run_stream(model, capture, args.save, Path(args.source).stem or "video")


if __name__ == "__main__":
    main()
