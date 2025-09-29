# realtime_face_id.py
"""
Hệ thống nhận diện khuôn mặt realtime cho một người duy nhất

Cách sử dụng:
1. Tạo thư mục gallery/<tên_của_bạn>/ và thêm ảnh mẫu
2. Đặt MY_NAME = "<tên_của_bạn>" trong phần cấu hình
3. Chạy script - lần đầu sẽ tự động build gallery
4. Để rebuild gallery: đặt FORCE_REBUILD_GALLERY = True

Yêu cầu:
- YOLO face detection model (yolov12n-face.pt)
- Webcam
- Python packages: ultralytics, facenet-pytorch, opencv-python, torch

Chức năng:
- Tự động detect & crop mặt trước khi tính embedding
- Sử dụng FaceNet pretrained (VGGFace2) cho embedding 512-d
- Strict Unknown labeling khi similarity < threshold
- Debounce logic để tránh nhận diện sai
- Cache gallery embeddings để tăng tốc độ khởi động
"""

import os, glob, time, pickle, cv2, torch, numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torchvision import transforms

# ====== CẤU HÌNH ======
YOLO_FACE_WEIGHTS  = "yolov12n-face.pt"   # đổi nếu bạn dùng file khác
GALLERY_DIR        = "gallery"             # thư mục ảnh mẫu
GALLERY_CACHE      = "gallery_embeddings.pkl"
COSINE_THRESHOLD   = 0.70                  # ngưỡng nhận diện (giảm để dễ nhận diện hơn)
IMG_SIZE_EMB       = 160                   # Facenet input
DETECTION_CONF     = 0.5                   # ngưỡng detect
DEBOUNCE_FRAMES    = 3                     # số frame liên tiếp để xác nhận
FORCE_REBUILD_GALLERY = True               # True để buộc tính lại gallery (để test fix)
AUTO_REBUILD_ON_CHANGE = True              # Tự động rebuild khi phát hiện thay đổi
DEBUG_MODE = False                         # True để hiện debug output
DETAILED_DEBUG = False                     # True để hiện chi tiết debug embedding

# ====== CẤU HÌNH DUAL CAMERA ======
USE_DUAL_CAMERA = True                     # True để sử dụng 2 camera
CAMERA_2_INDEX = 1                         # Index của camera phụ
DISPLAY_MODE = "side_by_side"              # "side_by_side", "stacked", "single"
ACTIVE_CAMERA = "both"                     # "cam1", "cam2", "both"

# ====== MODEL ======
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Detector: YOLO face
detector = YOLO(YOLO_FACE_WEIGHTS)

# Embedding: Facenet pretrained (VGGFace2)
embed_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Sử dụng ImageNet normalization thay vì fixed_image_standardization
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE_EMB, IMG_SIZE_EMB)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Test model với random input để đảm bảo model hoạt động
print("Testing FaceNet model...")
test_input = torch.randn(1, 3, 160, 160).to(device)
with torch.no_grad():
    test_output = embed_model(test_input)
print(f"Model test - Input shape: {test_input.shape}, Output shape: {test_output.shape}")
print(f"Model test - Output range: {test_output.min():.3f} to {test_output.max():.3f}")
print(f"Model test - Output norm: {torch.norm(test_output).item():.3f}")

# Test với input khác để xem có khác nhau không
test_input2 = torch.randn(1, 3, 160, 160).to(device)
with torch.no_grad():
    test_output2 = embed_model(test_input2)
print(f"Model test 2 - Are outputs identical? {torch.allclose(test_output, test_output2)}")
print(f"Model test 2 - Max difference: {torch.max(torch.abs(test_output - test_output2)).item():.6f}")
print("Model test completed.\n")

def l2_norm(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def is_face_quality_good(face_crop: np.ndarray, min_size=64) -> bool:
    """Kiểm tra chất lượng khuôn mặt với tiêu chí nghiêm ngặt hơn."""
    if face_crop is None or face_crop.size == 0:
        return False

    h, w = face_crop.shape[:2]

    # Kiểm tra kích thước tối thiểu
    if h < min_size or w < min_size:
        return False

    # Kiểm tra aspect ratio (mặt không quá méo)
    aspect_ratio = w / h
    if aspect_ratio < 0.6 or aspect_ratio > 1.4:
        return False

    # Kiểm tra độ sáng (tránh ảnh quá tối hoặc quá sáng)
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 40 or mean_brightness > 200:
        return False

    # Kiểm tra độ contrast (tránh ảnh quá mờ)
    contrast = gray.std()
    if contrast < 25:
        return False

    # Kiểm tra độ nét (blur detection)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:  # ảnh quá mờ
        return False

    # Kiểm tra histogram để tránh ảnh quá đơn điệu
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    if np.max(hist) > gray.size * 0.7:  # quá nhiều pixel cùng giá trị
        return False

    return True

def detect_and_crop_face(image_bgr: np.ndarray, debug=False):
    """Detect face in image and return the largest face crop."""
    if image_bgr is None or image_bgr.size == 0:
        return None

    if debug:
        print(f"    [Face Detection] Input image shape: {image_bgr.shape}")

    # Use YOLO to detect faces
    results = detector.predict(source=image_bgr, imgsz=640, conf=DETECTION_CONF, verbose=False)
    if not results:
        if debug:
            print(f"    [Face Detection] No YOLO results, using full image")
        return image_bgr  # fallback to full image

    res = results[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []

    if debug:
        print(f"    [Face Detection] Found {len(boxes)} faces")

    if len(boxes) == 0:
        if debug:
            print(f"    [Face Detection] No faces detected, using full image")
        return image_bgr  # fallback to full image

    # Get the largest face (by area)
    largest_box = None
    largest_area = 0

    for (x1, y1, x2, y2) in boxes:
        area = (x2 - x1) * (y2 - y1)
        if debug:
            print(f"    [Face Detection] Face box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}), area: {area:.0f}")
        if area > largest_area:
            largest_area = area
            largest_box = (x1, y1, x2, y2)

    if largest_box is not None:
        x1, y1, x2, y2 = map(int, largest_box)
        # Add padding
        pad = 8
        h, w = image_bgr.shape[:2]
        xa = max(0, x1 - pad); ya = max(0, y1 - pad)
        xb = min(w, x2 + pad); yb = min(h, y2 + pad)
        face_crop = image_bgr[ya:yb, xa:xb]

        if debug:
            print(f"    [Face Detection] Selected box: ({x1}, {y1}, {x2}, {y2})")
            print(f"    [Face Detection] Face crop shape: {face_crop.shape}")
            print(f"    [Face Detection] Face crop mean pixel: {face_crop.mean():.1f}")

        # Kiểm tra chất lượng khuôn mặt
        quality_good = is_face_quality_good(face_crop)
        if debug:
            print(f"    [Face Detection] Face quality check: {'PASS' if quality_good else 'FAIL'}")

        # Tạm thời disable quality check để debug
        return face_crop

        # if quality_good:
        #     return face_crop
        # else:
        #     return None  # Không dùng ảnh chất lượng kém

    if debug:
        print(f"    [Face Detection] No valid face found, using full image")
    return image_bgr  # fallback to full image

def face_embedding(face_bgr: np.ndarray, debug=False) -> np.ndarray:
    """Trả về embedding 512-d đã chuẩn hoá L2."""
    if face_bgr is None or face_bgr.size == 0:
        return None

    # Ensure the input is properly sized and normalized
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    if debug:
        print(f"  [Debug] Face shape: {face_bgr.shape}")
        print(f"  [Debug] Face RGB range: {face_rgb.min()}-{face_rgb.max()}")

    # Apply transforms: resize to 160x160 and normalize for Facenet
    x = transform(face_rgb).unsqueeze(0).to(device)

    if debug:
        print(f"  [Debug] Tensor shape: {x.shape}")
        print(f"  [Debug] Tensor range: {x.min():.3f}-{x.max():.3f}")

    with torch.no_grad():
        emb = embed_model(x).cpu().numpy()[0]

    if debug:
        print(f"  [Debug] Raw embedding shape: {emb.shape}")
        print(f"  [Debug] Raw embedding norm: {np.linalg.norm(emb):.3f}")

    normalized_emb = l2_norm(emb)

    if debug:
        print(f"  [Debug] Normalized embedding norm: {np.linalg.norm(normalized_emb):.3f}")
        print(f"  [Debug] First 5 values: {normalized_emb[:5]}")
        print(f"  [Debug] Last 5 values: {normalized_emb[-5:]}")
        print(f"  [Debug] Mean: {normalized_emb.mean():.6f}, Std: {normalized_emb.std():.6f}")
        print(f"  [Debug] Min: {normalized_emb.min():.6f}, Max: {normalized_emb.max():.6f}")

    return normalized_emb

def cosine(a, b, debug=False):
    """Tính cosine similarity giữa 2 vector đã normalize."""
    if a is None or b is None:
        return 0.0

    # Đảm bảo cả 2 vector đều được normalize
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if debug:
        print(f"    [Cosine Debug] Vector A norm: {a_norm:.6f}")
        print(f"    [Cosine Debug] Vector B norm: {b_norm:.6f}")
        print(f"    [Cosine Debug] Vector A first 3: {a[:3]}")
        print(f"    [Cosine Debug] Vector B first 3: {b[:3]}")
        print(f"    [Cosine Debug] Are vectors identical? {np.allclose(a, b)}")
        print(f"    [Cosine Debug] Max difference: {np.max(np.abs(a - b)):.6f}")

    # Nếu vector chưa normalize thì normalize lại
    if abs(a_norm - 1.0) > 1e-6:
        a = a / a_norm if a_norm > 0 else a
    if abs(b_norm - 1.0) > 1e-6:
        b = b / b_norm if b_norm > 0 else b

    # Tính cosine similarity
    similarity = float(np.dot(a, b))

    # Clamp để đảm bảo trong khoảng [-1, 1]
    similarity = np.clip(similarity, -1.0, 1.0)

    if debug:
        print(f"    [Cosine Debug] Dot product: {np.dot(a, b):.6f}")
        print(f"    [Cosine Debug] Final similarity: {similarity:.6f}")

    return similarity

def get_gallery_info(root: str):
    """Lấy thông tin về gallery để kiểm tra thay đổi."""
    if not os.path.isdir(root):
        return {}

    info = {}
    for person_dir in os.listdir(root):
        p_path = os.path.join(root, person_dir)
        if not os.path.isdir(p_path):
            continue

        # Đếm số file và thời gian modify cuối
        files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']:
            files.extend(glob.glob(os.path.join(p_path, ext)))
            files.extend(glob.glob(os.path.join(p_path, ext.upper())))

        if files:
            latest_mtime = max(os.path.getmtime(f) for f in files)
            info[person_dir] = {
                'count': len(files),
                'latest_mtime': latest_mtime
            }

    return info

def load_gallery_from_dir(root: str, target_person: str = None):
    """Đọc gallery/<name>/*.* → tính mean embedding cho target_person hoặc tất cả."""
    people = {}

    # Load tất cả người trong gallery (không chỉ định target_person nữa)
    person_dirs = []
    try:
        person_dirs = sorted([d for d in os.listdir(root)
                            if os.path.isdir(os.path.join(root, d))])
    except Exception as e:
        print(f"[Gallery] Lỗi đọc thư mục {root}: {e}")
        return people

    for person_dir in person_dirs:
        p_path = os.path.join(root, person_dir)
        if not os.path.isdir(p_path):
            continue

        print(f"[Gallery] Đang xử lý {person_dir}...")
        embs = []
        processed_count = 0

        # Hỗ trợ nhiều định dạng ảnh
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(p_path, ext)))
            image_files.extend(glob.glob(os.path.join(p_path, ext.upper())))

        for img_path in image_files:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  ⚠️  Không thể đọc: {os.path.basename(img_path)}")
                    continue

                processed_count += 1
                print(f"  Đang xử lý: {os.path.basename(img_path)}")

                # Debug cho ảnh đầu tiên của mỗi người
                debug_mode = (processed_count == 1 and DETAILED_DEBUG)

                # Detect và crop mặt trước khi tính embedding
                face_crop = detect_and_crop_face(img, debug=debug_mode)
                if face_crop is None:
                    print(f"  ⚠️  Không detect được mặt: {os.path.basename(img_path)}")
                    continue

                emb = face_embedding(face_crop, debug=debug_mode)
                if emb is not None:
                    embs.append(emb)
                    print(f"  ✓ Embedding thành công: {os.path.basename(img_path)} (norm: {np.linalg.norm(emb):.3f})")
                else:
                    print(f"  ⚠️  Embedding thất bại: {os.path.basename(img_path)}")

            except Exception as e:
                print(f"  ❌ Lỗi xử lý {os.path.basename(img_path)}: {e}")
                continue

        if embs:
            # Tính mean embedding và normalize
            emb_stack = np.stack(embs, axis=0)
            mean_emb = l2_norm(np.mean(emb_stack, axis=0))
            people[person_dir] = mean_emb

            # Debug diversity của embeddings
            if len(embs) > 1:
                pairwise_similarities = []
                for i in range(len(embs)):
                    for j in range(i+1, len(embs)):
                        sim = np.dot(embs[i], embs[j])
                        pairwise_similarities.append(sim)
                avg_sim = np.mean(pairwise_similarities)
                print(f"[Gallery] ✓ {person_dir}: {len(embs)}/{processed_count} ảnh, avg internal similarity: {avg_sim:.3f}")
            else:
                print(f"[Gallery] ✓ {person_dir}: {len(embs)}/{processed_count} ảnh")

            if DETAILED_DEBUG:
                print(f"[Gallery Debug] {person_dir} mean embedding first 3: {mean_emb[:3]}")
        else:
            print(f"[Gallery] ❌ {person_dir}: Không có embedding nào thành công")

    return people

def load_or_build_gallery():
    """Load gallery from cache or rebuild if needed."""

    current_gallery_info = get_gallery_info(GALLERY_DIR)
    should_rebuild = FORCE_REBUILD_GALLERY

    # Check if auto rebuild is needed
    if AUTO_REBUILD_ON_CHANGE and os.path.exists(GALLERY_CACHE):
        try:
            # Load cached info nếu có
            cache_info_file = GALLERY_CACHE + ".info"
            cached_info = {}
            if os.path.exists(cache_info_file):
                with open(cache_info_file, "rb") as f:
                    cached_info = pickle.load(f)

            # So sánh với info hiện tại
            if cached_info != current_gallery_info:
                print("[Gallery] 🔄 Phát hiện thay đổi trong gallery, rebuilding...")
                should_rebuild = True

        except Exception as e:
            print(f"[Gallery] ⚠️  Lỗi kiểm tra cache info: {e}")
            should_rebuild = True

    # Load từ cache nếu không cần rebuild
    if not should_rebuild and os.path.exists(GALLERY_CACHE):
        try:
            with open(GALLERY_CACHE, "rb") as f:
                g = pickle.load(f)
            if isinstance(g, dict) and g:
                print(f"[Gallery] ✓ Loaded cache: {len(g)} người - {list(g.keys())}")
                return g
        except Exception as e:
            print(f"[Gallery] ❌ Lỗi đọc cache: {e}")

    if FORCE_REBUILD_GALLERY:
        print("[Gallery] 🔄 Force rebuild gallery...")

    if not os.path.isdir(GALLERY_DIR):
        print(f"[Gallery] ❌ Không thấy thư mục {GALLERY_DIR}/ — tạo thư mục và thêm ảnh vào các folder con")
        return {}

    # Load tất cả người trong gallery
    print(f"[Gallery] 🔄 Building gallery cho tất cả người...")
    g = load_gallery_from_dir(GALLERY_DIR)

    if g:
        # Check cross-person similarity
        people_list = list(g.keys())
        if len(people_list) > 1:
            print(f"[Gallery] Kiểm tra cross-similarity giữa các người:")
            for i in range(len(people_list)):
                for j in range(i+1, len(people_list)):
                    person1, person2 = people_list[i], people_list[j]
                    sim = cosine(g[person1], g[person2])
                    print(f"[Gallery] {person1} vs {person2}: {sim:.3f}")

        try:
            # Save gallery cache
            with open(GALLERY_CACHE, "wb") as f:
                pickle.dump(g, f)

            # Save gallery info cache
            cache_info_file = GALLERY_CACHE + ".info"
            with open(cache_info_file, "wb") as f:
                pickle.dump(current_gallery_info, f)

            print(f"[Gallery] ✓ Cached → {GALLERY_CACHE}")
        except Exception as e:
            print(f"[Gallery] ⚠️  Không thể ghi cache: {e}")
    else:
        print(f"[Gallery] ❌ Trống — thêm ảnh vào các folder trong {GALLERY_DIR}/")

    return g

gallery = load_or_build_gallery()
if not gallery:
    print("⚠️  Chưa có gallery; script vẫn chạy nhưng chỉ hiện Unknown.")

# Debounce: đếm số frame liên tiếp khớp theo tên
streak = {}  # name -> count

def init_cameras():
    """Khởi tạo camera(s) theo cấu hình."""
    cameras = {}

    if USE_DUAL_CAMERA:

        # Thử khởi tạo camera 2
        cap2 = cv2.VideoCapture(CAMERA_2_INDEX)
        if cap2.isOpened():
            cameras['cam2'] = cap2
            print(f"[Camera] ✓ Camera 2 (index {CAMERA_2_INDEX}) đã kết nối")
        else:
            print(f"[Camera] ❌ Không thể kết nối Camera 2 (index {CAMERA_2_INDEX})")

        if not cameras:
            # Fallback về camera đơn
            print(f"[Camera] ⚠️  Không camera nào hoạt động, fallback về camera đơn...")
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cameras['cam1'] = cap
                print(f"[Camera] ✓ Fallback camera (index 0) đã kết nối")


    return cameras

def process_frame(frame, camera_name=""):
    """Xử lý frame từ camera và trả về frame đã được vẽ annotations."""
    if frame is None:
        return frame

    # Detect faces
    results = detector.predict(source=frame, imgsz=640, conf=DETECTION_CONF, verbose=False)
    if results:
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
        for (x1, y1, x2, y2) in boxes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            # Nới nhẹ bbox cho chắc (clip biên)
            pad = 8
            h, w = frame.shape[:2]
            xa = max(0, x1 - pad); ya = max(0, y1 - pad)
            xb = min(w, x2 + pad); yb = min(h, y2 + pad)
            face = frame[ya:yb, xa:xb]

            # Kiểm tra chất lượng khuôn mặt trước khi tính embedding
            if not is_face_quality_good(face):
                # Vẽ bbox màu vàng cho face chất lượng kém
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "Low Quality", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                continue

            emb = face_embedding(face)
            name, score = None, -1.0
            best_name, best_score = None, -1.0  # Để hiển thị debug info

            if emb is not None and gallery:
                # So sánh với TẤT CẢ người trong gallery
                if DEBUG_MODE:
                    print(f"\n[Recognition Debug] Realtime embedding norm: {np.linalg.norm(emb):.6f}")

                for person_name, g_emb in gallery.items():
                    if DEBUG_MODE:
                        print(f"[Recognition Debug] Comparing with {person_name} (gallery norm: {np.linalg.norm(g_emb):.6f})")

                    # Debug cho similarity calculation
                    similarity_score = cosine(emb, g_emb, debug=DEBUG_MODE)

                    if DEBUG_MODE:
                        print(f"[Recognition Debug] {person_name} similarity: {similarity_score:.6f}")

                    # Tìm người có similarity cao nhất
                    if similarity_score > best_score:
                        best_name, best_score = person_name, similarity_score

                if DEBUG_MODE:
                    print(f"[Recognition Debug] Best match: {best_name} with score {best_score:.6f}")

                # Chỉ nhận diện nếu similarity >= threshold
                if best_score >= COSINE_THRESHOLD:
                    name, score = best_name, best_score
                    if DEBUG_MODE:
                        print(f"[Recognition] ✓ RECOGNIZED: {best_name} ({score:.3f})")
                else:
                    if DEBUG_MODE:
                        print(f"[Recognition] ❌ UNKNOWN: Best {best_name}({best_score:.3f}) < threshold({COSINE_THRESHOLD})")

            # Debounce logic - cho phép nhận diện bất kỳ ai
            label = "Unknown"
            color = (0, 0, 255)  # Red for Unknown

            if name and name in gallery:
                streak_key = f"{camera_name}_{name}" if camera_name else name
                streak[streak_key] = streak.get(streak_key, 0) + 1

                # Reset tất cả streak khác cho camera này
                for k in list(streak.keys()):
                    if k.startswith(camera_name) and k != streak_key:
                        streak[k] = 0

                if streak[streak_key] >= DEBOUNCE_FRAMES:
                    label = f"{name} ({score:.2f})"
                    color = (0, 255, 0)  # Green for recognized person

                    # TODO: Ghi chấm công ở đây (API call)
                    # requests.post("http://localhost:3000/attendance", json={
                    #     "user_id": name, "score": score, "ts": time.time(), "camera": camera_name
                    # })
            else:
                # Reset streak cho camera này nếu không nhận diện được
                for k in list(streak.keys()):
                    if k.startswith(camera_name):
                        streak[k] = 0

            # Vẽ
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Hiển thị thông tin debug nếu có
            if best_name and best_score > 0:
                debug_text = f"Best: {best_name} ({best_score:.3f})"
                cv2.putText(frame, debug_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    return frame

def combine_frames(frame1, frame2):
    """Kết hợp 2 frame theo DISPLAY_MODE."""
    if frame1 is None and frame2 is None:
        return None

    if frame1 is None:
        return frame2
    if frame2 is None:
        return frame1

    # Đảm bảo cả 2 frame có cùng kích thước
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]

    if DISPLAY_MODE == "side_by_side":
        # Resize để có cùng chiều cao
        target_height = min(h1, h2)
        if h1 != target_height:
            frame1 = cv2.resize(frame1, (int(w1 * target_height / h1), target_height))
        if h2 != target_height:
            frame2 = cv2.resize(frame2, (int(w2 * target_height / h2), target_height))

        combined = np.hstack([frame1, frame2])

    elif DISPLAY_MODE == "stacked":
        # Resize để có cùng chiều rộng
        target_width = min(w1, w2)
        if w1 != target_width:
            frame1 = cv2.resize(frame1, (target_width, int(h1 * target_width / w1)))
        if w2 != target_width:
            frame2 = cv2.resize(frame2, (target_width, int(h2 * target_width / w2)))

        combined = np.vstack([frame1, frame2])

    else:  # single mode
        combined = frame1  # Chỉ hiển thị camera 1

    return combined

# ====== DUAL CAMERA LOOP ======
cameras = init_cameras()
if not cameras:
    print("❌ Không có camera nào hoạt động!")
    exit(1)

prev = time.time()
while True:
    frames = {}

    # Đọc frame từ tất cả camera
    for cam_name, cap in cameras.items():
        ok, frame = cap.read()
        if ok:
            frames[cam_name] = frame

    if not frames:
        print("❌ Không đọc được frame từ camera nào!")
        break

    # FPS
    now = time.time()
    fps = 1.0 / (now - prev) if now > prev else 0.0
    prev = now

    # Xử lý từng frame
    processed_frames = {}
    for cam_name, frame in frames.items():
        if ACTIVE_CAMERA == "both" or ACTIVE_CAMERA == cam_name.replace("cam", "cam"):
            processed_frame = process_frame(frame, cam_name)

            # Thêm thông tin camera vào frame
            cv2.putText(processed_frame, f"Camera {cam_name[-1]}", (10, 24),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)

            processed_frames[cam_name] = processed_frame

    # Kết hợp frames nếu có nhiều hơn 1 camera
    if len(processed_frames) == 1:
        display_frame = list(processed_frames.values())[0]
    elif len(processed_frames) >= 2:
        cam_names = list(processed_frames.keys())
        frame1 = processed_frames[cam_names[0]]
        frame2 = processed_frames[cam_names[1]]
        display_frame = combine_frames(frame1, frame2)
    else:
        continue

    # Display info chung trên frame
    if display_frame is not None:
        info_y = display_frame.shape[0] - 120  # Đặt ở dưới cùng

        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,255,50), 2)
        cv2.putText(display_frame, f"Threshold: {COSINE_THRESHOLD}", (10, info_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Gallery status với tên các người
        if gallery:
            people_names = ", ".join(gallery.keys())
            gallery_status = f"Gallery: {len(gallery)} people - {people_names}"
        else:
            gallery_status = "Gallery: Not loaded"
        cv2.putText(display_frame, gallery_status, (10, info_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 2)

        # Controls
        controls = f"q: quit | r: rebuild | 1: cam1 only | 2: cam2 only | b: both | s: side-by-side | t: stacked"
        cv2.putText(display_frame, controls, (10, info_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)

        # Window title tùy theo mode
        if USE_DUAL_CAMERA and len(cameras) > 1:
            window_title = f"FaceID - Dual Camera ({DISPLAY_MODE}) - {ACTIVE_CAMERA}"
        else:
            window_title = "FaceID - Single Camera"

        cv2.imshow(window_title, display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        print("\n[System] Rebuilding gallery...")
        gallery = load_gallery_from_dir(GALLERY_DIR)
        if gallery:
            # Save gallery và info cache
            with open(GALLERY_CACHE, "wb") as f:
                pickle.dump(gallery, f)

            cache_info_file = GALLERY_CACHE + ".info"
            current_gallery_info = get_gallery_info(GALLERY_DIR)
            with open(cache_info_file, "wb") as f:
                pickle.dump(current_gallery_info, f)

            print(f"[Gallery] ✓ Rebuilt and cached → {GALLERY_CACHE}")
            print(f"[Gallery] ✓ Found {len(gallery)} people: {list(gallery.keys())}")
        else:
            print(f"[Gallery] ❌ No embeddings found in gallery")
        streak.clear()  # Reset streak after rebuild
    elif key == ord('1'):
        ACTIVE_CAMERA = "cam1"
        print(f"[Control] Switched to Camera 1 only")
    elif key == ord('2'):
        ACTIVE_CAMERA = "cam2"
        print(f"[Control] Switched to Camera 2 only")
    elif key == ord('b'):
        ACTIVE_CAMERA = "both"
        print(f"[Control] Switched to both cameras")
    elif key == ord('s'):
        DISPLAY_MODE = "side_by_side"
        print(f"[Control] Display mode: side-by-side")
    elif key == ord('t'):
        DISPLAY_MODE = "stacked"
        print(f"[Control] Display mode: stacked")

# Cleanup
for cap in cameras.values():
    cap.release()
cv2.destroyAllWindows()
