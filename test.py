"""Generic YOLO NCNN inference helper.

This script is designed for Raspberry Pi deployments where we only have
CPU access.  It loads Ultralytics models exported to the NCNN runtime and
supports running detection on a webcam, a single video file, or a folder
of images.  It intentionally avoids importing torch/CUDA so that the
runtime only depends on ultralytics, ncnn, OpenCV, and NumPy.
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
        width_str, height_str = value.lower().split("x", 1)
        width, height = int(width_str), int(height_str)
    except (ValueError, AttributeError) as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(
            "Resolution must be in the form <width>x<height>, e.g. 640x480"
        ) from exc

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution values must be positive")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO NCNN inference")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to an Ultralytics NCNN export directory (ends with *_ncnn_model)",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video file, or directory with images",
    )
    parser.add_argument(
        "--resolution",
        default="640x480",
        help="Camera resolution in <width>x<height> format (used for webcams)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated results (video file or images depending on source)",
    )
    return parser.parse_args()


def as_camera_index(source: str) -> Optional[int]:
    try:
        idx = int(source)
    except ValueError:
        return None
    return idx


def iter_image_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def ensure_save_dir(suffix: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = Path("runs") / f"predict_{suffix}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def annotate_frame(model: YOLO, frame, imgsz: int):
    results = model.predict(frame, imgsz=imgsz, device="cpu", verbose=False)
    return results[0].plot()


def handle_camera(model: YOLO, index: int, resolution: Tuple[int, int], save: bool) -> None:
    width, height = resolution
    capture = cv2.VideoCapture(index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera index {index}")

    save_dir = ensure_save_dir("camera") if save else None
    writer = None
    window_name = "YOLO NCNN - Camera"
    imgsz = max(width, height)

    print("Press ESC to exit.")
    while True:
        ok, frame = capture.read()
        if not ok:
            print("[WARN] Failed to grab frame from camera")
            break

        annotated = annotate_frame(model, frame, imgsz)

        if save and save_dir is not None:
            if writer is None:
                fps = capture.get(cv2.CAP_PROP_FPS) or 30
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                output_path = save_dir / "camera.mp4"
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (annotated.shape[1], annotated.shape[0]))
                print(f"[INFO] Saving video to {output_path}")
            writer.write(annotated)

        cv2.imshow(window_name, annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def handle_video(model: YOLO, video_source: Path | str, save: bool) -> None:
    source_str = str(video_source)
    capture = cv2.VideoCapture(source_str)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    imgsz = max(width, height)

    label = Path(source_str).stem or "stream"
    save_dir = ensure_save_dir(label) if save else None
    writer = None
    window_name = f"YOLO NCNN - {Path(source_str).name or source_str}"

    print("Press ESC to exit.")
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        annotated = annotate_frame(model, frame, imgsz)

        if save and save_dir is not None:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                output_path = save_dir / f"{label}_annotated.mp4"
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (annotated.shape[1], annotated.shape[0]))
                print(f"[INFO] Saving video to {output_path}")
            writer.write(annotated)

        cv2.imshow(window_name, annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def handle_directory(model: YOLO, directory: Path, save: bool) -> None:
    image_paths = list(iter_image_files(directory))
    if not image_paths:
        raise RuntimeError(f"No image files found in directory: {directory}")

    save_dir = ensure_save_dir(directory.stem) if save else None
    if save_dir is not None:
        (save_dir / "images").mkdir(parents=True, exist_ok=True)

    window_name = f"YOLO NCNN - {directory.name}"

    print("Press ESC to exit.")
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[WARN] Unable to read image: {image_path}")
            continue

        imgsz = max(frame.shape[0], frame.shape[1])
        annotated = annotate_frame(model, frame, imgsz)

        if save and save_dir is not None:
            output_path = save_dir / "images" / image_path.name
            cv2.imwrite(str(output_path), annotated)

        cv2.imshow(window_name, annotated)
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
        handle_camera(model, camera_index, resolution, args.save)
    elif source_path.is_dir():
        handle_directory(model, source_path, args.save)
    elif source_path.is_file():
        handle_video(model, source_path, args.save)
    else:
        # Fall back to attempting VideoCapture on the provided string
        handle_video(model, args.source, args.save)


if __name__ == "__main__":
    main()
