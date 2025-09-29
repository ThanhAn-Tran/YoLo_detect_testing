# yolo_detect.py
import argparse, cv2, time
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to exported NCNN model dir (e.g., my_model_ncnn_model)")
    ap.add_argument("--source", default="0", help="0 for webcam, or path to video/image dir")
    ap.add_argument("--resolution", default="640x480")
    ap.add_argument("--save", action="store_true", help="save annotated video")
    return ap.parse_args()

def open_source(src, w, h):
    try:
        cam_index = int(src)
        cap = cv2.VideoCapture(cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        return cap, True
    except ValueError:
        return src, False  # file path

def main():
    args = parse_args()
    W, H = map(int, args.resolution.split("x"))

    # Load NCNN model
    model = YOLO(args.model)  # <- trỏ vào thư mục *_ncnn_model

    src, is_cam = open_source(args.source, W, H)
    writer = None

    if is_cam:
        while True:
            ok, frame = src.read()
            if not ok:
                print("No frame from camera")
                break
            results = model.predict(frame, imgsz=max(W, H), device="cpu", verbose=False)
            annotated = results[0].plot()
            cv2.imshow("YOLO NCNN", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        src.release()
        cv2.destroyAllWindows()
    else:
        # file/video/dir — dùng API predict trực tiếp
        save = args.save
        model.predict(args.source, imgsz=max(W, H), device="cpu", save=save, stream=False, verbose=True)

if __name__ == "__main__":
    main()
