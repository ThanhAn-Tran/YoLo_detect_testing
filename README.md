# YOLO NCNN Raspberry Pi Demo

This repository contains a set of small utilities for running Ultralytics
YOLO models that were exported to the **NCNN** backend.  The scripts are
written to run well on a Raspberry Pi 4 (64-bit Raspberry Pi OS Bookworm)
without installing PyTorch or relying on CUDA.

## 1. Setup on Raspberry Pi

```bash
# Clone repository
git clone https://github.com/ThanhAn-Tran/YoLo_detect_testing.git
cd YoLo_detect_testing

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages (Ultralytics pulls in the NCNN runtime)
pip install --upgrade pip
pip install -r requirements.txt
```

The dependencies are intentionally small: `ultralytics`, `ncnn`,
`opencv-python`, and `numpy`.

## 2. Export a YOLO model to NCNN

Model export can be performed on a more powerful machine (recommended) or
directly on the Pi if the `.pt` file is small.  The export only needs to be
done once.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")          # or your own trained checkpoint
model.export(format="ncnn")         # creates yolov8n_ncnn_model/
```

The export command generates a folder ending with `_ncnn_model`.  Copy that
folder to the Raspberry Pi (for example via `scp` or a USB drive).

## 3. Run the demos

All scripts share the same command-line interface.  The `--model` argument
must point to the exported NCNN directory, `--source` defines what to run
on (webcam index, video file, or folder with images), and `--resolution`
sets the camera capture size when using a webcam.  Add `--save` if you want
to store annotated output.

```bash
# Real-time webcam demo with bounding boxes and FPS overlay
python yolo_detect.py --model=weights_ncnn_model --source=0 --resolution=640x480

# Run inference on a video file and save the annotated result
python test.py --model=weights_ncnn_model --source=video.mp4 --save

# Alternate entry-point with the same options plus FPS overlay
python new.py --model=weights_ncnn_model --source=images/ --save
```

When using a webcam (`--source=0`), press **ESC** to close the OpenCV
window.  For directories of images the scripts iterate over all supported
files (`jpg`, `png`, `bmp`, `tif`).  Saved outputs are written to the `runs/`
folder with timestamped names.

## 4. Notes

- These scripts never import or require PyTorch.  They operate entirely on
the Ultralytics NCNN runtime.
- If you plan to export models on the Pi, make sure you have enough swap
space; exporting large checkpoints can be memory intensive.
- For headless usage you can disable the OpenCV windows by running the
scripts inside a virtual framebuffer such as `xvfb` or by modifying the
code to skip `cv2.imshow` calls.

Enjoy running YOLO on your Raspberry Pi!
