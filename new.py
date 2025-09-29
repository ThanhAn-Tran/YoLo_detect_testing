"""Lightweight YOLO NCNN runner with FPS overlay.

This script mirrors the command-line interface used by ``test.py`` but is
intended for quick experiments where you want to monitor detection FPS in
real time.  It keeps the codebase torch-free and uses the NCNN export of
Ultralytics models, making it friendly for Raspberry Pi deployments.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_resolution(value: str) -> Tuple[int, int]:
    try:
        w_str, h_str = value.lower().split("x", 1)
        width, height = int(w_str), int(h_str)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(
            "Resolution must be in the form <width>x<height>"
        ) from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution values must be positive")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO NCNN with FPS overlay")
    parser.add_argument("--model", required=True, help="Path to *_ncnn_model directory")
    parser.add_argument("--source", default="0", help="Camera index, video file, or image folder")
    parser.add_argument("--resolution", default="640x480", help="Resolution for webcams (e.g. 640x480)")
    parser.add_argument("--save", action="store_true", help="Save annotated output")
    return parser.parse_args()


def as_camera_index(source: str) -> Optional[int]:
    try:
        return int(source)
    except ValueError:
        return None


def iter_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def ensure_save_dir(label: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = Path("runs") / f"demo_{label}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def annotate(model: YOLO, frame, imgsz: int):
    results = model.predict(frame, imgsz=imgsz, device="cpu", verbose=False)
    annotated = results[0].plot()
    return annotated, results[0]


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


def run_camera(model: YOLO, index: int, resolution: Tuple[int, int], save: bool) -> None:
    width, height = resolution
    capture = cv2.VideoCapture(index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera index {index}")

    save_dir = ensure_save_dir("camera") if save else None
    writer = None
    imgsz = max(width, height)
    window = "YOLO NCNN Demo"

    print("Press ESC to exit.")
    prev = time.time()
    while True:
        ok, frame = capture.read()
        if not ok:
            print("[WARN] Failed to grab camera frame")
            break

        now = time.time()
        dt = now - prev
        prev = now
        fps = 1.0 / dt if dt > 0 else 0.0

        annotated, _ = annotate(model, frame, imgsz)
        draw_fps(annotated, fps)

        if save and save_dir is not None:
            if writer is None:
                fps_cap = capture.get(cv2.CAP_PROP_FPS) or 30
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                output_path = save_dir / "camera.mp4"
                writer = cv2.VideoWriter(str(output_path), fourcc, fps_cap, (annotated.shape[1], annotated.shape[0]))
                print(f"[INFO] Saving video to {output_path}")
            writer.write(annotated)

        cv2.imshow(window, annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def run_video(model: YOLO, path: Path | str, save: bool) -> None:
    path_str = str(path)
    capture = cv2.VideoCapture(path_str)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {path}")

    fps_in = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    imgsz = max(width, height)

    label = Path(path_str).stem or "stream"
    save_dir = ensure_save_dir(label) if save else None
    writer = None
    window = f"YOLO NCNN Demo - {Path(path_str).name or path_str}"

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

        annotated, _ = annotate(model, frame, imgsz)
        draw_fps(annotated, fps)

        if save and save_dir is not None:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                output_path = save_dir / f"{label}_annotated.mp4"
                writer = cv2.VideoWriter(str(output_path), fourcc, fps_in, (annotated.shape[1], annotated.shape[0]))
                print(f"[INFO] Saving video to {output_path}")
            writer.write(annotated)

        cv2.imshow(window, annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def run_directory(model: YOLO, folder: Path, save: bool) -> None:
    images = list(iter_images(folder))
    if not images:
        raise RuntimeError(f"No image files found in {folder}")

    save_dir = ensure_save_dir(folder.stem) if save else None
    if save_dir is not None:
        (save_dir / "images").mkdir(parents=True, exist_ok=True)

    window = f"YOLO NCNN Demo - {folder.name}"
    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[WARN] Unable to read {image_path}")
            continue
        imgsz = max(frame.shape[0], frame.shape[1])
        annotated, _ = annotate(model, frame, imgsz)
        draw_fps(annotated, 0.0)

        if save and save_dir is not None:
            output_path = save_dir / "images" / image_path.name
            cv2.imwrite(str(output_path), annotated)

        cv2.imshow(window, annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    resolution = parse_resolution(args.resolution)

    print(f"[INFO] Loading NCNN model from: {args.model}")
    model = YOLO(args.model)

    camera_index = as_camera_index(args.source)
    source_path = Path(args.source)

    if camera_index is not None and not source_path.exists():
        run_camera(model, camera_index, resolution, args.save)
    elif source_path.is_dir():
        run_directory(model, source_path, args.save)
    elif source_path.is_file():
        run_video(model, source_path, args.save)
    else:
        run_video(model, args.source, args.save)


if __name__ == "__main__":
    main()
