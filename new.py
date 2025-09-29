# pip install ultralytics opencv-python numpy
# python referee_highlight.py

import cv2, time, collections
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
WEIGHTS   = "weights.pt"
CONF      = 0.35
CLASS_OK  = None
COOLDOWN_FRAMES = 8
FPS       = 20              # fps camera (nếu khác thì set lại)
HISTORY_S = 10              # số giây muốn giữ lại trước khi ghi điểm
CAMERA_INDEX = 1
# ----------------------------------------

def point_side_of_diag(x, y, w, h):
    x1, y1 = w - 1, 0
    x2, y2 = 0, h - 1
    return (x - x2) * (y1 - y2) - (y - y2) * (x1 - x2)

def side_label(x, y, w, h):
    ref = point_side_of_diag(w - 5, 5, w, h)
    val = point_side_of_diag(x, y, w, h)
    return "IN" if val * ref >= 0 else "OUT"

def draw_ui(frame, score, state):
    h, w = frame.shape[:2]
    cv2.line(frame, (w - 1, 0), (0, h - 1), (0, 255, 255), 2, cv2.LINE_AA)
    overlay = frame.copy()
    pts = np.array([[w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.int32)
    cv2.fillConvexPoly(overlay, pts, (0, 255, 0))
    frame[:] = cv2.addWeighted(overlay, 0.08, frame, 0.92, 0)
    cv2.rectangle(frame, (10, 10), (270, 82), (0, 0, 0), -1)
    cv2.putText(frame, f"SCORE: {score}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"STATE: {state}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255) if state == "IN" else (0, 140, 255) if state == "OUT" else (200, 200, 200),
                2, cv2.LINE_AA)
    return frame

def main():
    model = YOLO(WEIGHTS)
    names = model.names
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    score, prev_side = 0, None
    last_score_frame, frame_idx = -9999, 0

    print("[INFO] q: quit, r: reset score")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        results = model(frame, conf=CONF, verbose=False)
        r = results[0]
        boxes = r.boxes

        curr_state = "NO_BALL"
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls  = boxes.cls.cpu().numpy().astype(int)

            idxs = list(range(len(cls)))
            if CLASS_OK is not None and names:
                idxs = [i for i in idxs if names[cls[i]] == CLASS_OK]

            if idxs:
                best_i = max(idxs, key=lambda i: conf[i])
                x1, y1, x2, y2 = xyxy[best_i].astype(int)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 200, 255), 2, cv2.LINE_AA)
                label = f"{names[cls[best_i]] if names else cls[best_i]} {conf[best_i]:.2f}"
                cv2.putText(frame, label, (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)

                curr_state = side_label(cx, cy, w, h)

                if prev_side == "IN" and curr_state == "OUT":
                    if frame_idx - last_score_frame >= COOLDOWN_FRAMES:
                        score += 1
                        last_score_frame = frame_idx
                prev_side = curr_state

        frame = draw_ui(frame, score, curr_state)
        cv2.imshow("Realtime Referee", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('r'):
            score, prev_side, last_score_frame = 0, None, -9999
            print("[INFO] Score reset.")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
